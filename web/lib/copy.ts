/**
 * Site copy. Every sentence, label and name the site shows lives in web/content as JSON, and reaches a page
 * through this module. `npm run copy:check` scans web/content and fails on any digit, so a number can only
 * reach the page from the data root; the lint rules keep literal text out of the components.
 *
 * A sentence that needs a number has a placeholder in braces, filled from data with `fill`.
 */
import { Fragment, createElement, type ReactNode } from "react";
import home from "@/content/home.json";
import methods from "@/content/methods.json";
import prep from "@/content/prep.json";
import results from "@/content/results.json";
import review from "@/content/review.json";
import site from "@/content/site.json";

export const copy = { site, home, review, prep, results, methods } as const;

const PLACEHOLDER = /\{([a-zA-Z]+)\}/g;

/**
 * Fill the placeholders of a copy template with values. A placeholder with no value is an error, so a page
 * can never show a raw "{name}".
 */
export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(PLACEHOLDER, (_match, name: string) => {
    const value = values[name];
    if (value === undefined) {
      throw new Error(`copy template ${JSON.stringify(template)} has no value for {${name}}`);
    }
    return String(value);
  });
}

/**
 * Fill the placeholders of a copy template with elements (a link, a code span), for a sentence whose value is not
 * plain text. A placeholder with no value is an error, as in `fill`.
 */
export function fillNodes(template: string, values: Record<string, ReactNode>): ReactNode {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const match of template.matchAll(PLACEHOLDER)) {
    const name = match[1] as string;
    if (!(name in values)) throw new Error(`copy template ${JSON.stringify(template)} has no value for {${name}}`);
    parts.push(template.slice(last, match.index));
    parts.push(createElement(Fragment, { key: `${name}-${match.index}` }, values[name]));
    last = (match.index ?? 0) + match[0].length;
  }
  parts.push(template.slice(last));
  return createElement(Fragment, null, ...parts);
}

/** The label of a data key (a result, a split, a verdict class), or the key itself when copy has no label. */
export function label(labels: Record<string, string>, key: string): string {
  return labels[key] ?? key;
}
