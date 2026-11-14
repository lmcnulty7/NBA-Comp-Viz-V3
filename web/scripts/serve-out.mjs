// A small static server for the built site, the way a static host serves it under a sub-path.
//
// It serves <dir> under <base>/ and answers 404 for everything outside that path, so a link that forgets the
// base path fails here the way it would on GitHub Pages. A folder is served by its index.html, a folder asked
// for without its trailing slash is redirected to it, and an unknown path gets the site's 404.html.
// It listens on 127.0.0.1 only.
//
//   node scripts/serve-out.mjs [--dir out] [--base /NBA-Comp-Viz-V3] [--port 4173]
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { BUILD_INFO_FILE, OUT_DIR, WEB_DIR, normaliseBasePath, shown } from "./lib/paths.mjs";

const TYPES = {
  html: "text/html; charset=utf-8",
  txt: "text/plain; charset=utf-8",
  css: "text/css; charset=utf-8",
  js: "text/javascript; charset=utf-8",
  mjs: "text/javascript; charset=utf-8",
  json: "application/json; charset=utf-8",
  map: "application/json; charset=utf-8",
  svg: "image/svg+xml",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  avif: "image/avif",
  gif: "image/gif",
  ico: "image/x-icon",
  woff: "font/woff",
  woff2: "font/woff2",
  mp4: "video/mp4",
  webm: "video/webm",
  xml: "application/xml; charset=utf-8",
  webmanifest: "application/manifest+json",
};

function option(argv, name, fallback) {
  const index = argv.indexOf(name);
  return index >= 0 && argv[index + 1] !== undefined ? argv[index + 1] : fallback;
}

/** The base path a built folder was made for, from the build-info.json the build writes into it. */
function builtBasePath(dir) {
  const file = path.join(dir, BUILD_INFO_FILE);
  if (!existsSync(file)) return "";
  try {
    return normaliseBasePath(JSON.parse(readFileSync(file, "utf8")).basePath);
  } catch {
    return "";
  }
}

export function createStaticServer({ dir, base }) {
  const root = path.resolve(dir);
  const notFoundPage = path.join(root, "404.html");

  const send = (response, method, status, file) => {
    const extension = path.extname(file).slice(1).toLowerCase();
    response.writeHead(status, {
      "Content-Type": TYPES[extension] ?? "application/octet-stream",
      "Content-Length": statSync(file).size,
      "Cache-Control": "no-store",
    });
    if (method === "HEAD") response.end();
    else createReadStream(file).pipe(response);
  };

  const notFound = (response, method, inSite) => {
    if (inSite && existsSync(notFoundPage)) return send(response, method, 404, notFoundPage);
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" });
    response.end(method === "HEAD" ? undefined : "404: nothing is served at this path\n");
  };

  return createServer((request, response) => {
    const method = request.method ?? "GET";
    if (method !== "GET" && method !== "HEAD") {
      response.writeHead(405, { Allow: "GET, HEAD" });
      response.end();
      return;
    }
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(request.url ?? "/", "http://localhost").pathname);
    } catch {
      response.writeHead(400);
      response.end();
      return;
    }
    if (base && pathname === base) {
      response.writeHead(301, { Location: `${base}/` });
      response.end();
      return;
    }
    if (base && !pathname.startsWith(`${base}/`)) return notFound(response, method, false);
    const relative = pathname.slice(base.length);
    const target = path.join(root, relative);
    if (target !== root && !target.startsWith(root + path.sep)) return notFound(response, method, true);
    if (existsSync(target) && statSync(target).isDirectory()) {
      if (!pathname.endsWith("/")) {
        response.writeHead(301, { Location: `${pathname}/` });
        response.end();
        return;
      }
      const index = path.join(target, "index.html");
      return existsSync(index) ? send(response, method, 200, index) : notFound(response, method, true);
    }
    if (existsSync(target) && statSync(target).isFile()) return send(response, method, 200, target);
    return notFound(response, method, true);
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const argv = process.argv.slice(2);
  const dir = path.resolve(WEB_DIR, option(argv, "--dir", OUT_DIR));
  if (!existsSync(dir) || !statSync(dir).isDirectory()) {
    console.error(`serve: no built site at ${shown(dir)}. Run npm run build:public or npm run build:private first.`);
    process.exit(1);
  }
  const base = normaliseBasePath(option(argv, "--base", builtBasePath(dir)));
  const port = Number(option(argv, "--port", process.env.PORT ?? "4173"));
  const server = createStaticServer({ dir, base });
  server.on("error", (error) => {
    console.error(`serve: ${error.message}`);
    process.exit(1);
  });
  server.listen(port, "127.0.0.1", () => {
    console.log(`serve: ${shown(dir)} at http://127.0.0.1:${port}${base}/`);
  });
  for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => server.close(() => process.exit(0)));
}
