/** Which nav item is the current one. Pure, so it can be tested without a browser. */

function withoutTrailingSlash(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

/** True when `pathname` is the section `href` or a page under it. Home matches only itself. */
export function isCurrentSection(pathname: string, href: string): boolean {
  const path = withoutTrailingSlash(pathname);
  const section = withoutTrailingSlash(href);
  if (section === "/") return path === "/";
  return path === section || path.startsWith(`${section}/`);
}
