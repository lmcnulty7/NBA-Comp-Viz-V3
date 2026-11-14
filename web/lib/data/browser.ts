/**
 * The data source of the browser. It fetches the JSON files the static site serves under data/. A live source
 * would replace this module and nothing else (PLAN.md, Site structure).
 */
import { assetUrl } from "@/lib/site";
import { DataError, StaticSource, type DataSource } from "./source";

async function fetchJson(relativePath: string): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(assetUrl(`data/${relativePath}`));
  } catch (error) {
    throw new DataError(relativePath, "unreachable", (error as Error).message);
  }
  if (response.status === 404) return undefined;
  if (!response.ok) {
    throw new DataError(relativePath, "unreachable", `the server answered ${response.status}`);
  }
  try {
    return (await response.json()) as unknown;
  } catch (error) {
    throw new DataError(relativePath, "invalid", (error as Error).message);
  }
}

let source: DataSource | undefined;

export function getBrowserSource(): DataSource {
  source ??= new StaticSource(fetchJson);
  return source;
}
