/** Inline SVG icons. Kept as markup so the build stays asset-free. */

const wrap = (body: string, extra = "") =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ` +
  `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ${extra}>${body}</svg>`;

export const sun = wrap(
  `<circle cx="12" cy="12" r="4"/>` +
    `<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>`,
);

export const moon = wrap(`<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>`);

export const play = wrap(`<path d="M6 4.5 19 12 6 19.5z" fill="currentColor" stroke-linejoin="round"/>`);

export const spinner = wrap(`<path d="M21 12a9 9 0 1 1-6.2-8.6"/>`);

export const download = wrap(`<path d="M12 3v12M7 11l5 5 5-5M4 20h16"/>`);

export const check = wrap(`<path d="M20 6 9 17l-5-5"/>`);

export const alert = wrap(`<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01"/>`);

export const refresh = wrap(`<path d="M20 11A8 8 0 0 0 6.3 5.7L3 9M4 13a8 8 0 0 0 13.7 5.3L21 15"/><path d="M3 5v4h4M21 19v-4h-4"/>`);

export const send = wrap(`<path d="M4.5 12 20 4.5 14.5 20 11.5 13z" fill="currentColor" stroke-linejoin="round"/>`);
