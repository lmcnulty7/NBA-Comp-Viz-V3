// npm run check for one variant of the site, as the polish loop needs it (docs/POLISH_LOOP.md, "Gate").
//
//   public   node scripts/check.mjs: all thirteen steps; the build step leaves the public site in out/.
//   private  the steps that do not depend on the variant (types to unit), then the private build at the root of
//            a host (as the private site is served) and the browser suite against it; out/ holds the private site.
//
// It runs the repository's own scripts and changes none of them. Each call builds the site, so it runs only
// under the heavy-job lock (CLAUDE.md rule 7). The result names the step that failed.
import { spawnSync } from "node:child_process";
import path from "node:path";
import { WEB_DIR } from "./paths.mjs";

export const STATIC_STEPS = ["types", "typecheck", "lint", "data", "copy", "reading", "contrast", "hygiene", "unit"];

function run(label, command, args, env) {
  console.log(`[polish-check] ${label}`);
  const started = Date.now();
  const done = spawnSync(command, args, { cwd: WEB_DIR, env, encoding: "utf8", maxBuffer: 256 * 1024 * 1024 });
  const output = `${done.stdout ?? ""}${done.stderr ?? ""}`;
  // The output of a long run is kept short in the log: its tail says what failed.
  process.stdout.write(output.split("\n").slice(-40).join("\n"));
  const failedAt = /\[check\] failed at: (\S+)/.exec(output)?.[1] ?? null;
  return { status: done.status ?? 1, seconds: Math.round((Date.now() - started) / 100) / 10, failedAt };
}

/**
 * Run the checks of one variant. `extraEnv` is added to the environment (the loop sets POLISH_BASE, so the
 * guard test in the unit step checks the candidate's diff). Returns { ran, passed, failedAt, steps }.
 */
export function runCheck(variant, extraEnv = {}) {
  const node = process.execPath;
  const script = (name) => path.join(WEB_DIR, "scripts", name);
  const env = { ...process.env, ...extraEnv, NEXT_TELEMETRY_DISABLED: "1" };
  const steps = [];
  const finish = (failedAt) => ({ ran: true, variant, passed: failedAt === null, failedAt, steps });

  if (variant === "public") {
    const done = run("npm run check (public)", node, [script("check.mjs")], env);
    steps.push({ name: "check", passed: done.status === 0, seconds: done.seconds });
    return finish(done.status === 0 ? null : done.failedAt ?? "check");
  }
  if (variant !== "private") throw new Error(`polish-check: the variant is public or private, not ${variant}`);

  const plan = [
    ["check (static steps)", node, [script("check.mjs"), ...STATIC_STEPS], env],
    ["build (private)", node, [script("build.mjs"), "--variant", "private"], { ...env, VARIANT: "private", NEXT_PUBLIC_BASE_PATH: "" }],
    ["browser (private)", path.join(WEB_DIR, "node_modules", ".bin", "playwright"), ["test", "--config", "playwright.config.ts"], env],
  ];
  for (const [label, command, args, stepEnv] of plan) {
    const done = run(label, command, args, stepEnv);
    steps.push({ name: label, passed: done.status === 0, seconds: done.seconds });
    if (done.status !== 0) return finish(done.failedAt ?? label);
  }
  return finish(null);
}
