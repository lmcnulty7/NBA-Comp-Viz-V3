import { defineConfig } from "@playwright/test";

// Unit tests: pure modules and the check scripts. Playwright is only the test runner here; no browser starts.
export default defineConfig({
  testDir: "tests/unit",
  testMatch: /.*\.test\.ts$/,
  outputDir: "test-results/unit",
  reporter: [["list"]],
  forbidOnly: true,
  fullyParallel: true,
  workers: 4,
  timeout: 60_000,
});
