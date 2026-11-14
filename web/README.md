# web/: the site

A Next.js static export in TypeScript (PLAN.md D26). Python produces the data, the site only reads it, through
one JSON contract: `export/schemas.py`. Nothing here fits, scores or computes a result.

Next.js 16 differs from earlier versions in its APIs and conventions. Its documentation ships with the
package: read the relevant guide under `node_modules/next/dist/docs/` before changing routing, config or data
loading.

## Commands

Run these inside `web/`. Anything heavy (install, type check, build, browser tests) goes through the heavy-job
lock from the repository root, for example
`python3 -m export.heavy --name "web check" -- npm --prefix web run check`.

| command | what it does |
| --- | --- |
| `npm ci` | install the pinned dependencies |
| `npx playwright install chromium webkit` | Playwright's own headless browsers, once |
| `npm run check` | every check, in order (see below) |
| `npm run build:public` | the public variant into `out/` (fails if `out/` holds a clip) |
| `npm run build:private` | the private variant into `out/`, with the clips folder copied to `out/clips` |
| `npm run serve` | serve `out/` on 127.0.0.1 under the base path it was built for |
| `npm run shots` | screenshots of the built pages at 390 and 1280 px into `test-results/shots`, headless |
| `npm run types` | regenerate `lib/types` from `export/schemas.py` |
| `npm run dev` | the Next.js development server |

Variables: `SITE_DATA_DIR` (the data root, default `web/data`), `NEXT_PUBLIC_BASE_PATH` (default empty; GitHub
Pages needs `/NBA-Comp-Viz-V3`), `VARIANT` (`public` or `private`, when the build is not started through the two
build scripts), `MEDIA_PRIVATE_DIR` (the clips folder, default `web/media-private`), `SITE_DATA_ALLOW_MISSING`
(fixed data files that may be absent, for example `demo/swap.json demo/frames.json`), `PYTHON` (the interpreter
with pydantic, default `python3`).

## Data

The site reads the data root named by `SITE_DATA_DIR`, else `web/data`. Write a synthetic one with
`python3 -m export.sample --out web/data` from the repository root. `scripts/prepare-data.mjs` copies the data
root into `public/data` before every build and fails when there is none. When `manifest.synthetic` is true,
every page shows a banner: fixture data, not results.

Pages and components never read a file or fetch a URL. They call a `DataSource` (`lib/data/source.ts`):
`getServerSource()` while the site is built, `getBrowserSource()` in the browser. A live service would be a
third implementation of the same interface. The lint rules enforce this.

`lib/types` is generated and never edited by hand: `contract.schema.json` by `export/schemas.py`,
`contract.ts` from it with json-schema-to-typescript, and `contract-rules.json` (the private-string patterns
and the file-name lists of `export/schemas.py`, which the schema does not carry).

`npm run data:validate` checks the data root in two halves. `scripts/validate-data.mjs` checks every file with
ajv against the JSON Schema, scans for private strings, compares the manifest with the files on disk and checks
what the routes need. Then `python3 -m export.schemas --validate` runs on the same folder: the cross-field
rules of the contract live in its pydantic validators, which a JSON Schema cannot carry. The build scripts run
the first half only, so a build needs no Python.

## Copy

Every sentence and label lives in `content/*.json` and reaches a page through `lib/copy.ts`. A copy file holds
no digit: a number comes from the data root through a `{placeholder}` and `lib/format.ts`. Two checks hold the
line: `npm run copy:check` scans `content/`, and the lint rules refuse literal text in a page or a component.
Longer text (the slides of M10) can be Markdown under `content/`; the same scan covers it.

A number in the cold open and the slides is a token of `docs/SCRIPT.md`, such as `{swap.fill_share}`: it names
a field of the site data, and `lib/tokens.ts` fills it with the digits the file stores (a share moved two
decimal places to read as a percentage, never rounded). `TOKENS` there is the token table of the script, held
equal by a unit test; a token that is not in the table fails the copy check and the build. The cold open copy
in `content/home.json` is the approved copy of the script, line for line, and `npm run reading:check` holds it
there, with the reading grade (Flesch-Kincaid, at most 9, computed in `scripts/lib/readability.mjs`) and the
list of words kept out of the copy. The slides are `content/talk/*.md` in the format of `scripts/lib/slides.mjs`.

## The court

`components/court` is the one court renderer: `geometry.ts` (feet to canvas pixels and the court lines, pure),
`scene.ts` (contract data to a scene), `draw.ts` (a scene onto a 2D context), `Court.tsx` (the canvas, the
device pixel ratio, redraw on resize and theme change). The transform is stated a second time in
`export/court_reference.py`, and a browser test compares the two within half a pixel on a possession of the
data root in use.

## Variants

`VARIANT=public|private` is fixed when the site is built. A route file named `page.private.tsx` exists only in
the private build (`next.config.ts`), and the public build fails if such a route, a video file or a moving
image is found in `out/`. `lib/site.ts` gives components `VARIANT`, `BASE_PATH`, `assetUrl` and `clipUrl`.

`public/.nojekyll` ships with the site so that a Pages build that runs Jekyll does not drop the `_next` folder
and the files whose names start with an underscore.

## npm run check

`types` (generated files are current), `typecheck` (`next typegen`, `tsc --noEmit`), `lint`, `data` (the data
root against the contract, both halves, no private string), `copy`, `reading` (grade and plain words on the cold
open and the slides), `contrast` (WCAG AA of the design tokens, light
and dark), `hygiene` (no long dash and no private string in the sources), `unit`, `build` (public, under the
Pages base path), `video`, `browser` (Playwright at 390 and 1280 px on Chromium and WebKit against `out/`).
`node scripts/check.mjs lint unit` runs only the named steps.

## Design

Tokens are CSS variables in `app/styles/tokens.css`: a neutral scale, the court roles (offense, defense, ghost,
truth, ball), one accent, a type scale, and a dark theme that follows the system or `data-theme`. The faces are
Newsreader, Libre Franklin and IBM Plex Mono, served from `app/fonts` under the SIL Open Font License 1.1. The
licence texts are in `public/fonts` and ship with the site.
