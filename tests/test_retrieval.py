"""Tests for query building and the Qdrant retrieval plumbing (no GPU/store)."""

import numpy as np
import pandas as pd
import pytest

from pt_healthcare.config import CPT_COLLECTION, HCPCS_COLLECTION
from pt_healthcare.retrieval import CodeRetriever, build_medical_query


class FakeModel:
    def __init__(self, vector=(0.1, 0.2, 0.3)):
        self.vector = vector
        self.calls = []

    def encode(self, text, normalize_embeddings=True):
        self.calls.append((text, normalize_embeddings))
        return np.array(self.vector, dtype=float)


class FakePoint:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class FakeResponse:
    def __init__(self, points):
        self.points = points


class FakeClient:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def query_points(self, collection_name, query, limit, with_payload):
        self.calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "limit": limit,
                "with_payload": with_payload,
            }
        )
        return FakeResponse(self.mapping.get(collection_name, [])[:limit])


def make_retriever():
    mapping = {
        CPT_COLLECTION: [
            FakePoint(0.903, {"code": "76090", "text": "Mammography; unilateral"}),
            FakePoint(0.860501, {"code": "77065", "text": "Diagnostic mammography, unilateral"}),
        ],
        HCPCS_COLLECTION: [
            FakePoint(0.799653, {"code": "G0204", "text": "Diagnostic mammography"}),
        ],
    }
    model = FakeModel()
    client = FakeClient(mapping)
    return CodeRetriever(model, client, top_k=10), model, client


def test_build_medical_query_joins_terms():
    assert build_medical_query({"medical": ["diagnostic mammogram", "breast imaging"]}) == (
        "diagnostic mammogram breast imaging"
    )


def test_build_medical_query_empty_and_invalid():
    assert build_medical_query({"medical": []}) == ""

    with pytest.raises(TypeError):
        build_medical_query(["not", "a", "dict"])


def test_retrieve_codes_empty_query_returns_empty_frame():
    retriever, model, client = make_retriever()

    result = retriever.retrieve_codes("   ", CPT_COLLECTION)

    assert result.empty
    assert list(result.columns) == ["rank", "score", "code", "text"]
    assert model.calls == []
    assert client.calls == []


def test_retrieve_codes_builds_ranked_frame():
    retriever, model, client = make_retriever()

    result = retriever.retrieve_codes("diagnostic mammogram", CPT_COLLECTION)

    assert list(result["rank"]) == [1, 2]
    assert list(result["code"]) == ["76090", "77065"]
    assert result["score"].dtype == float
    assert model.calls[0] == ("diagnostic mammogram", True)
    assert client.calls[0]["collection_name"] == CPT_COLLECTION
    assert client.calls[0]["limit"] == 10
    assert client.calls[0]["with_payload"] is True


def test_retrieve_cpt_and_hcpcs_use_correct_collections():
    retriever, model, client = make_retriever()
    categorized = {"medical": ["diagnostic mammogram"]}

    cpt = retriever.retrieve_cpt(categorized)
    hcpcs = retriever.retrieve_hcpcs(categorized)

    assert list(cpt["code"]) == ["76090", "77065"]
    assert list(hcpcs["code"]) == ["G0204"]
    assert client.calls[0]["collection_name"] == CPT_COLLECTION
    assert client.calls[1]["collection_name"] == HCPCS_COLLECTION


def test_retrieve_cpt_without_medical_terms_is_empty():
    retriever, model, client = make_retriever()

    result = retriever.retrieve_cpt({"medical": []})

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert client.calls == []


def test_top_k_override():
    retriever, model, client = make_retriever()

    retriever.retrieve_cpt({"medical": ["mammogram"]}, top_k=1)

    assert client.calls[0]["limit"] == 1
