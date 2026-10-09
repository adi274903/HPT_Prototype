/** Theme: light/dark with system default and localStorage persistence. */

export type Theme = "light" | "dark";

const KEY = "pt.theme";

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

export function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function currentTheme(): Theme {
  return stored() ?? systemTheme();
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* private mode — theme just won't persist */
  }
  window.dispatchEvent(new CustomEvent("pt:theme", { detail: theme }));
}

export function toggleTheme(): Theme {
  const next: Theme = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  return next;
}

/** Follow the OS until the user makes an explicit choice. */
export function watchSystemTheme(): void {
  const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
  if (!mq) return;
  mq.addEventListener("change", () => {
    if (!stored()) applyTheme(systemTheme());
  });
}
