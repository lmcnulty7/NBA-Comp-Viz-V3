import type { Page } from "@playwright/test";

/** Collects what goes wrong while a page loads: console errors, uncaught errors, failed and 4xx/5xx requests. */
export function watch(page: Page): { problems: string[]; hosts: Set<string> } {
  const problems: string[] = [];
  const hosts = new Set<string>();
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => problems.push(`error: ${error.message}`));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.protocol.startsWith("http")) hosts.add(url.hostname);
  });
  page.on("response", (response) => {
    if (response.status() >= 400) problems.push(`${response.status()}: ${response.url()}`);
  });
  page.on("requestfailed", (request) => {
    const reason = request.failure()?.errorText ?? "";
    // A prefetch dropped because the page moved on is not a failure of the site.
    if (/abort|cancel/i.test(reason)) return;
    problems.push(`failed (${reason}): ${request.url()}`);
  });
  return { problems, hosts };
}
