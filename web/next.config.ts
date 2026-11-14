import type { NextConfig } from "next";

// VARIANT is public (the default) or private. Anything else is an error, so a typo cannot build the wrong site.
const rawVariant = process.env.VARIANT ?? "public";
if (rawVariant !== "public" && rawVariant !== "private") {
  throw new Error(`VARIANT must be "public" or "private", got "${rawVariant}"`);
}
const variant: "public" | "private" = rawVariant;

// The site can live under a sub-path (GitHub Pages serves it under the repository name). An empty value means
// the root. The value is inlined into the client bundle, so it must be set when the site is built.
function normaliseBasePath(value: string | undefined): string {
  const trimmed = (value ?? "").trim().replace(/\/+$/, "");
  if (trimmed === "") return "";
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}
const basePath = normaliseBasePath(process.env.NEXT_PUBLIC_BASE_PATH);

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  reactStrictMode: true,
  images: { unoptimized: true },
  // A route file named page.private.tsx exists only in the private build. The public build does not see it,
  // so a private-only route (the /talk slides of PLAN.md M10) cannot leak into the public site.
  pageExtensions: variant === "private" ? ["private.tsx", "private.ts", "tsx", "ts"] : ["tsx", "ts"],
  env: {
    NEXT_PUBLIC_VARIANT: variant,
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
  // `next dev` would otherwise write an AGENTS.md into this folder. web/README.md carries the same pointer to
  // the documentation that ships with the installed Next.js.
  agentRules: false,
};

export default nextConfig;
