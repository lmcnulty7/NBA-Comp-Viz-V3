/**
 * Site copy. Every sentence, label and name the site shows lives in web/content as JSON, and reaches a page
 * through this module. `npm run copy:check` scans web/content and fails on any digit, so a number can only
 * reach the page from the data root; the lint rules keep literal text out of the components.
 *
 * A sentence that needs a number has a placeholder in braces, filled from data with `fill`.
 */
import home from "@/content/home.json";
import prep from "@/content/prep.json";
import results from "@/content/results.json";
import review from "@/content/review.json";
import site from "@/content/site.json";

export const copy = { site, home, review, prep, results } as const;

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

/** The label of a data key (a result, a split, a verdict class), or the key itself when copy has no label. */
export function label(labels: Record<string, string>, key: string): string {
  return labels[key] ?? key;
}
