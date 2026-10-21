"""Unit tests for the eval harness: metric arithmetic and the derived targets.

The point of these is that the reported numbers are auditable. Dice/Jaccard get
tested on the degenerate inputs that decide whether an aggregate is honest — the
empty/empty case that must not be scored as a hit, and the one-sided case that
must be a zero. The target tests re-derive intent and entities from gazetteers and
assert they agree with the dataset's own ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pt_healthcare.eval_runner import (  # noqa: E402
    amounts,
    collapse,
    dice,
    jaccard,
    matches_any,
    normalize_entity,
    percentile,
    prf1,
    recall_at_k,
    reciprocal_rank,
)

DATASET = REPO / "eval"
DB = REPO / ".cache" / "mrf_eval.duckdb"


def test_dice_identity_and_disjoint():
    assert dice({"a", "b"}, {"a", "b"}) == 1.0
    assert dice({"a"}, {"b"}) == 0.0
    assert dice({"a", "b", "c"}, {"a", "b"}) == pytest.approx(4 / 5)


def test_dice_and_jaccard_agree_on_partial_overlap():
    predicted, truth = {"a", "b", "c"}, {"b", "c", "d"}
    assert dice(predicted, truth) == pytest.approx(4 / 6)
    assert jaccard(predicted, truth) == pytest.approx(2 / 4)


def test_empty_sets_are_not_scored_as_perfect():
    # 120 of the 600 cases have no row: returning nothing there is correct but
    # must not enter the Dice average as a 1.0.
    assert dice(set(), set()) is None
    assert jaccard([], []) is None


def test_one_sided_empty_is_zero():
    assert dice({"a"}, set()) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_prf1_counts():
    precision, recall, f1 = prf1({"a", "b"}, {"b", "c"})
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_precision_and_recall_differ_when_sets_are_lopsided():
    precision, recall, _ = prf1({"a", "b", "c", "d"}, {"a"})
    assert precision == pytest.approx(0.25)
    assert recall == pytest.approx(1.0)


def test_recall_at_k_and_reciprocal_rank():
    ranked = ["x", "y", "z"]
    assert recall_at_k(ranked, {"y", "z", "w"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, {"x"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"z"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"q"}) == 0.0
    assert reciprocal_rank(ranked, set()) is None


def test_amount_extraction_handles_commas_and_cents():
    assert amounts("the rate is $1,234.56, cash $900") == [1234.56, 900.0]
    assert amounts("no figure here") == []
    assert amounts("") == []


def test_amount_tolerance_is_one_cent():
    assert matches_any(1234.56, [1234.56])
    assert matches_any(1234.56 + 0.0001, [1234.56]), "float artifacts must still match"
    assert matches_any(1234.57, [1234.56]) is False, "a wrong cent is wrong"
    assert matches_any(1234.56 + 0.01, [1234.56]) is False


def test_entity_normalization_makes_articles_neutral():
    assert normalize_entity("a colonoscopy") == normalize_entity("colonoscopy")
    assert normalize_entity("The  MRI ") == "mri"


def test_collapse_reduces_rate_lines_to_one_identity():
    rows = [
        {"cpt": "27130", "hospital": "Upmc Magee", "payer": "Aetna"},
        {"cpt": "27130", "hospital": "Upmc Magee", "payer": "Aetna"},
        {"cpt": "27130", "hospital": "Upmc Magee", "payer": "Highmark BCBS of PA"},
    ]
    assert len(collapse(rows)) == 2


def test_percentile_is_defined_for_small_samples():
    assert percentile([], 0.95) is None
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == 3.0


@pytest.mark.skipif(not DB.is_file() or not (DATASET / "pt_eval_v2.jsonl").is_file(),
                    reason="needs the MRF cache and the dataset")
def test_targets_agree_with_the_dataset_ground_truth():
    sys.path.insert(0, str(REPO / "eval_code_gen"))
    from build_targets import build  # type: ignore

    targets, audit = build(DATASET, DB)

    assert len(targets) == 600
    assert audit["hospital_not_in_gt"] == 0, "a question names a hospital the key omits"
    assert audit["payer_not_in_gt"] == 0, "a question names a payer the key omits"
    assert audit["gt_hospital_unnamed"] == 0, "the key names a hospital the question omits"
    assert audit["gt_payer_unnamed"] == 0, "the key names a payer the question omits"
    assert audit["multi_hospital"] == 60, "only the C1 comparison cases name two hospitals"
    assert audit["medical_withheld"] == 12, "only the 12 B1 service-withheld cases have no service"

    # Every case in this set asks what something costs, so the router's expected
    # label is "price" throughout. That makes the intent number a false-positive
    # test (a cost question must never route to lookup or explain), not a
    # three-way F1 — proving the other two labels needs a routing set.
    intents: dict = {}
    for record in targets:
        intents[record["intent"]] = intents.get(record["intent"], 0) + 1
    assert intents == {"price": 600}

    medication = [record for record in targets if record["canonical"]["medication"]]
    assert len(medication) == 54, "the insulin cases are the only ones with a medication entity"
    assert all(record["entities"]["medication"] for record in medication)


# ---------------------------------------------------------------------------
# the eval function itself
# ---------------------------------------------------------------------------


class StubClient:
    """A ``complete()`` that returns canned text and counts calls."""

    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def complete(self, prompt, max_tokens=None):
        from pt_healthcare.eval_runner import Completion

        self.calls += 1
        self.tokens_in += 10
        self.tokens_out += 2
        return Completion(text=self.reply, prompt_tokens=10, completion_tokens=2, latency_ms=1.0)


def _tiny_db(path: Path) -> Path:
    """The three objects ``Pricer`` reads, with one priced row."""
    import duckdb

    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE rows_ext (row_id VARCHAR, hospital VARCHAR, payer VARCHAR,"
                " plan VARCHAR, setting VARCHAR, cpt VARCHAR, hcpcs VARCHAR,"
                " rate DOUBLE, cash DOUBLE)")
    con.execute("INSERT INTO rows_ext VALUES ('r1', 'UPMC Presbyterian', 'UPMC Health Plan',"
                " 'P', 'outpatient', '77065', NULL, 133.0, 150.0)")
    con.execute("CREATE TABLE code_hospital AS SELECT hospital FROM rows_ext")
    con.execute("CREATE TABLE combos AS SELECT payer FROM rows_ext")
    con.close()
    return path


def _one_case(tmp_path: Path) -> Path:
    """A dataset dir holding a single real case and its target."""
    dataset = tmp_path / "eval_dataset"
    dataset.mkdir()
    case = json.loads((DATASET / "pt_eval_v2.jsonl").read_text().splitlines()[0])
    target = next(record for record in
                  map(json.loads, (DATASET / "pt_eval_v2_targets.jsonl").read_text().splitlines())
                  if record["eval_id"] == case["eval_id"])
    (dataset / "pt_eval_v2.jsonl").write_text(json.dumps(case) + "\n")
    (dataset / "pt_eval_v2_targets.jsonl").write_text(json.dumps(target) + "\n")
    return dataset


def test_evaluate_returns_the_report_and_scores_a_case(tmp_path):
    from pt_healthcare.eval_runner import evaluate

    client = StubClient("")
    report = evaluate(_one_case(tmp_path), db=_tiny_db(tmp_path / "tiny.duckdb"),
                      client=client, out=tmp_path / "run.jsonl", report_path=None,
                      progress=None)

    assert report["n_cases"] == 1
    assert report["codes"]["dice_mrf"] == 0.0, "an empty answer scores zero, not perfect"
    assert client.calls == 4, "intent, extract, decide, answer"
    assert len((tmp_path / "run.jsonl").read_text().splitlines()) == 1


def test_evaluate_resumes_instead_of_regenerating(tmp_path):
    from pt_healthcare.eval_runner import evaluate

    dataset, out = _one_case(tmp_path), tmp_path / "run.jsonl"
    first = StubClient("")
    evaluate(dataset, db=_tiny_db(tmp_path / "tiny.duckdb"), client=first, out=out,
             report_path=None, progress=None)
    second = StubClient("")
    messages = []
    report = evaluate(dataset, db=tmp_path / "tiny.duckdb", client=second, out=out,
                      report_path=None, progress=messages.append)

    assert second.calls == 0, "a case already in --out must not be regenerated"
    assert report["n_cases"] == 1
    assert any("cached" in message for message in messages)


def test_evaluate_says_how_to_build_a_missing_price_db(tmp_path):
    from pt_healthcare.eval_runner import evaluate

    with pytest.raises(FileNotFoundError, match="prepare_eval_mrf.py"):
        evaluate(_one_case(tmp_path), db=tmp_path / "nope.duckdb", client=StubClient(),
                 out=tmp_path / "run.jsonl", report_path=None, progress=None)


def test_find_dataset_dir_prefers_the_named_dir_and_explains_itself(tmp_path):
    from pt_healthcare.eval_runner import find_dataset_dir

    assert find_dataset_dir(None, root=REPO).name in ("eval", "eval_dataset")
    with pytest.raises(FileNotFoundError, match="pt_eval_v2.jsonl"):
        find_dataset_dir(tmp_path)


def test_backend_names_pick_a_client_and_typos_raise():
    from pt_healthcare.eval_runner import ChatClient, build_client

    assert isinstance(build_client("remote", model="m", base_url="http://x/v1"), ChatClient)
    with pytest.raises(ValueError, match="unknown backend='vlllm'"):
        build_client("vlllm", model="m", base_url="http://x/v1")
    with pytest.raises(ValueError, match="gguf"):
        build_client("llamacpp", model="m", base_url="", gguf=None)


def test_vllm_in_process_client_never_touches_a_server(monkeypatch):
    """The point of the backend: ``vllm.LLM`` in this process, no base URL."""
    import sys
    import types

    from pt_healthcare.eval_runner import VLLMInProcessClient

    calls = {"init": None, "chat": None}

    class Output:
        text = "  ~$133 negotiated  "
        token_ids = [1, 2, 3]

    class Completion_:
        prompt_token_ids = [9, 8, 7, 6]
        outputs = [Output()]

    class LLM:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def chat(self, **kwargs):
            calls["chat"] = kwargs
            return [Completion_()]

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module = types.ModuleType("vllm")
    module.LLM, module.SamplingParams = LLM, SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)

    client = VLLMInProcessClient(model="google/medgemma-1.5-4b-it", max_model_len=4096,
                                 gpu_memory_utilization=0.85)
    completion = client.complete("how much is 77065?")

    assert completion.text == "  ~$133 negotiated  "
    assert (completion.prompt_tokens, completion.completion_tokens) == (4, 3)
    assert (client.tokens_in, client.tokens_out) == (4, 3)
    assert calls["init"]["model"] == "google/medgemma-1.5-4b-it"
    assert calls["init"]["gpu_memory_utilization"] == 0.85
    assert calls["chat"]["messages"] == [{"role": "user", "content": "how much is 77065?"}]
    assert calls["chat"]["sampling_params"].temperature == 0.0
    assert not hasattr(client, "base_url"), "in-process means there is no endpoint"


def test_evaluate_can_reuse_the_ui_model(monkeypatch):
    """``client=VLLMInProcessClient(llm=shared)`` must not load a second copy."""
    import sys
    import types

    from pt_healthcare.eval_runner import VLLMInProcessClient

    built = []

    class LLM:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def chat(self, **kwargs):
            class Output:
                text, token_ids = "x", [1]
            class Completion_:
                prompt_token_ids, outputs = [1, 2], [Output()]
            return [Completion_()]

    module = types.ModuleType("vllm")
    module.LLM = LLM
    module.SamplingParams = lambda **kwargs: type("Params", (), kwargs)()
    monkeypatch.setitem(sys.modules, "vllm", module)

    shared = LLM(model="already-loaded")
    built.clear()  # the fixture above has built exactly one, and only one is wanted
    client = VLLMInProcessClient(llm=shared)
    completion = client.complete("hi")

    assert client.llm is shared
    assert built == [], "the shared instance is the only one"
    assert completion.text == "x"


def test_targets_are_found_beside_the_cases_or_in_the_sibling(tmp_path):
    """A checkout with cases in eval_dataset/ and keys in eval/ must still score."""
    import json

    from pt_healthcare.eval_runner import CASES_FILE, TARGETS_FILE, find_targets

    (tmp_path / "eval_dataset").mkdir()
    (tmp_path / "eval").mkdir()
    case = {"eval_id": "x1", "eval_query": "q"}
    (tmp_path / "eval_dataset" / CASES_FILE).write_text(json.dumps(case) + "\n")
    (tmp_path / "eval" / TARGETS_FILE).write_text(json.dumps({"eval_id": "x1"}) + "\n")

    assert find_targets(tmp_path / "eval_dataset") == tmp_path / "eval" / TARGETS_FILE


def test_a_missing_answer_key_names_every_place_it_looked(tmp_path):
    import json

    import pytest

    from pt_healthcare.eval_runner import CASES_FILE, find_targets

    (tmp_path / "eval_dataset").mkdir()
    (tmp_path / "eval_dataset" / CASES_FILE).write_text(json.dumps({"eval_id": "x1"}) + "\n")

    with pytest.raises(FileNotFoundError) as excinfo:
        find_targets(tmp_path / "eval_dataset")

    message = str(excinfo.value)
    assert "pt_eval_v2_targets.jsonl" in message
    assert "1 cases" in message
    assert str(tmp_path / "eval") in message


def test_the_catalog_comes_from_qdrant_not_the_csv(monkeypatch, tmp_path):
    """No ~/Downloads on Colab: code -> text must come from the store already there."""
    import sys
    import types

    from pt_healthcare import eval_runner

    seen = {}

    class Point:
        def __init__(self, code, text):
            self.payload = {"code": code, "text": text}

    class Client:
        def __init__(self, path):
            seen["path"] = path

        def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
            if collection_name == "hcpcs_medte":
                return [Point("G0204", "Diagnostic mammography, bilateral")], None
            return [Point("77065", "Diagnostic mammography, unilateral, including CAD")], None

        def close(self):
            seen["closed"] = True

    module = types.ModuleType("qdrant_client")
    module.QdrantClient = Client
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    (tmp_path / "meta.json").write_text("{}")  # find_qdrant_dir only trusts a real store

    # The store reader itself: resolve_catalog() prefers the shipped artifact now, so
    # this exercises the Qdrant path directly rather than through the ordering.
    catalog = eval_runner.catalog_from_qdrant(tmp_path)

    assert catalog["CPT"]["77065"].startswith("Diagnostic mammography")
    assert catalog["HCPCS"]["G0204"].startswith("Diagnostic mammography")
    assert seen["path"] == str(tmp_path) and seen.get("closed") is True


def test_the_store_is_found_from_the_environment_or_content(monkeypatch, tmp_path):
    """colab_run.setup exports PT_QDRANT_PATH; evaluate() must honour it."""
    from pt_healthcare.eval_runner import find_qdrant_dir

    store = tmp_path / "New_PT_DB"
    store.mkdir()
    (store / "meta.json").write_text("{}")

    monkeypatch.setenv("PT_QDRANT_PATH", str(store))
    assert find_qdrant_dir() == store.resolve()

    monkeypatch.delenv("PT_QDRANT_PATH")
    monkeypatch.setattr("pt_healthcare.config.qdrant_dir", lambda: str(tmp_path / "nope"))


def test_a_missing_store_says_where_it_looked(monkeypatch, tmp_path):
    import pytest

    from pt_healthcare.eval_runner import find_qdrant_dir

    monkeypatch.setenv("PT_QDRANT_PATH", str(tmp_path / "absent"))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        find_qdrant_dir()

    message = str(excinfo.value)
    assert "PT_QDRANT_PATH" in message
    assert "/content/New_PT_DB" in message
    assert str(tmp_path / "absent") in message


def test_the_shipped_catalog_needs_no_store_and_no_downloads():
    """The eval must run on a bare Colab checkout: no lock, no ~/Downloads."""
    from pt_healthcare.eval_runner import resolve_catalog

    catalog = resolve_catalog(qdrant_path="/nonexistent/store")

    assert len(catalog["CPT"]) > 15000
    assert len(catalog["HCPCS"]) > 13000
    assert catalog["CPT"]["77065"].startswith("Diagnostic mammography")


def test_the_qdrant_retriever_loads_only_the_embedder(monkeypatch):
    """A GPU already holds the 4B generator in vLLM; retrieval must not load a second."""
    import pt_healthcare.data as data
    import pt_healthcare.models as models
    import pt_healthcare.retrieval as retrieval
    from pt_healthcare.eval_runner import EmbeddingRetriever

    seen = {}

    monkeypatch.setattr(models, "load_embedder", lambda: "embedder")

    def explode(*args, **kwargs):
        raise AssertionError("load_models() would load the full generator")

    monkeypatch.setattr(models, "load_models", explode)
    monkeypatch.setattr(data, "get_qdrant_client", lambda path=None: ("client", path))

    class Stub:
        def __init__(self, embed_model, client, top_k=10):
            seen["args"] = (embed_model, client, top_k)

    monkeypatch.setattr(retrieval, "CodeRetriever", Stub)

    EmbeddingRetriever(qdrant_path="/store", top_k=7)

    assert seen["args"] == ("embedder", ("client", "/store"), 7)
