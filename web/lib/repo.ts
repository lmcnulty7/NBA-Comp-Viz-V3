/**
 * The public repository the site's numbers come from. A source path relative to the repository (for example
 * reports/xfg.md) links to the file on GitHub; a data table under $DATA_ROOT is not in the repository and gets
 * no link.
 */
export const REPO_URL = "https://github.com/lmcnulty7/NBA-Comp-Viz-V3";
export const REPO_BRANCH = "main";

const REPO_PATH = /^(?![/$~.])[A-Za-z0-9_./-]+$/;

/** The GitHub page of a file in the repository, or null when the path is not a repository file. */
export function repoFileUrl(path: string): string | null {
  const clean = path.trim();
  if (!REPO_PATH.test(clean) || clean.includes("..") || clean.includes("<")) return null;
  return `${REPO_URL}/blob/${REPO_BRANCH}/${clean}`;
}
