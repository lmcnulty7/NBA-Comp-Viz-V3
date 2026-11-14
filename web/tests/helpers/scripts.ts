import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

export const WEB_DIR = path.resolve(__dirname, "..", "..");
export const REPO_ROOT = path.resolve(WEB_DIR, "..");

export interface Run {
  status: number;
  stdout: string;
  stderr: string;
  output: string;
}

/** Run one of the web scripts with Node and return what it printed. */
export function runScript(name: string, args: string[] = [], env: Record<string, string> = {}): Run {
  const result = spawnSync(process.execPath, [path.join(WEB_DIR, "scripts", name), ...args], {
    cwd: WEB_DIR,
    encoding: "utf8",
    env: { ...process.env, ...env },
  });
  const stdout = result.stdout ?? "";
  const stderr = result.stderr ?? "";
  return { status: result.status ?? 1, stdout, stderr, output: stdout + stderr };
}

const made: string[] = [];
process.on("exit", () => {
  // Every folder a test made is removed when its worker ends, and with it the folders written beside it.
  for (const dir of made) {
    for (const suffix of ["", "-prepared", "-empty"]) rmSync(`${dir}${suffix}`, { recursive: true, force: true });
  }
});

export function tempDir(prefix: string): string {
  const dir = mkdtempSync(path.join(tmpdir(), `v3-web-${prefix}-`));
  made.push(dir);
  return dir;
}

interface PythonModule {
  runPython(args: string[]): { status: number; stdout: string; stderr: string };
  runPythonModule(module: string, args?: string[]): { status: number; stdout: string; stderr: string };
}

/** The Python helper of the scripts (it starts Python natively when Node runs under Rosetta). */
export async function python(): Promise<PythonModule> {
  return (await import("../../scripts/lib/python.mjs")) as unknown as PythonModule;
}

/** A fresh synthetic data root that validates against the contract, written by export/sample.py. */
export async function sampleRoot(): Promise<string> {
  const dir = tempDir("data");
  const { runPythonModule } = await python();
  const result = runPythonModule("export.sample", ["--out", dir]);
  if (result.status !== 0) throw new Error(`export.sample failed: ${result.stderr}`);
  return dir;
}
