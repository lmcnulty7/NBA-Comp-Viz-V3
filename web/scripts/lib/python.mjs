// Runs the repository's Python modules. The interpreter is the PYTHON variable, else python3 on PATH.
import { spawnSync } from "node:child_process";
import { REPO_ROOT } from "./paths.mjs";

export function pythonCommand(env = process.env) {
  return (env.PYTHON ?? "").trim() || "python3";
}

let translated;

/**
 * True when this Node process is an Intel build running under Rosetta on an Apple Silicon Mac. A child process
 * then starts as x86_64 too, and a universal Python cannot load packages that were installed for arm64
 * (pydantic's compiled core, for one). Such a child is started through `arch -arm64` instead.
 */
export function isTranslated() {
  if (translated === undefined) {
    translated = false;
    if (process.platform === "darwin" && process.arch === "x64") {
      const probe = spawnSync("sysctl", ["-n", "sysctl.proc_translated"], { encoding: "utf8" });
      translated = probe.status === 0 && probe.stdout.trim() === "1";
    }
  }
  return translated;
}

function spawnPython(prefix, args, input) {
  const command = [...prefix, pythonCommand(), ...args];
  return spawnSync(command[0], command.slice(1), {
    cwd: REPO_ROOT,
    encoding: "utf8",
    input,
    maxBuffer: 64 * 1024 * 1024,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
  });
}

/** Run `python <args>` from the repository root. Returns { status, stdout, stderr }. */
export function runPython(args, { input } = {}) {
  let result = spawnPython(isTranslated() ? ["arch", "-arm64"] : [], args, input);
  if (isTranslated() && result.status !== 0 && /Bad CPU type/.test(result.stderr ?? "")) {
    result = spawnPython([], args, input); // an Intel-only Python: run it as it is
  }
  if (result.error) {
    throw new Error(
      `could not start Python (${pythonCommand()}): ${result.error.message}. ` +
        "Set PYTHON to the interpreter that has pydantic.",
    );
  }
  return { status: result.status ?? 1, stdout: result.stdout ?? "", stderr: result.stderr ?? "" };
}

/** Run `python -m <module> <args>` from the repository root. */
export function runPythonModule(module, args = [], options = {}) {
  return runPython(["-m", module, ...args], options);
}
