import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'

// The lab's own tests: `lab/lab test`, which runs
// `npx vitest run --config lab/vitest.config.ts`. The app's vitest.config.ts
// includes only tests/**/*.test.ts, so `npm test` never runs these, and this
// file includes only lab/tests/**/*.test.ts, so these never run the app's.
//
// The same guards as the app's config, plus the lab's own: nothing a test
// loads may reach the real ComfyUI, the running app or the author's folders.
// ComfyUI and the app both point at port 9 (discard), where nothing listens,
// the server-side runner stays off, and every folder the lab would write to
// is a temporary one no test created by accident. A test that needs a folder
// makes its own under the system temp folder and sets it before loading a
// module. Every test here runs against a stand-in runner on port 0.
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const unset = path.join(os.tmpdir(), 'switchgen-lab-vitest-unset')

export default defineConfig({
  root: repo,
  test: {
    include: ['lab/tests/**/*.test.ts'],
    environment: 'node',
    testTimeout: 15_000,
    // The machine is short of memory while ComfyUI holds a model; two workers
    // at most keep the tests from pushing it into the out-of-memory killer.
    maxWorkers: 2,
    env: {
      COMFY_URL: 'http://127.0.0.1:9',
      SWITCHGEN_URL: 'http://127.0.0.1:9',
      SWITCHGEN_RUNNER: 'off',
      SWITCHGEN_OUTPUTS: path.join(unset, 'outputs'),
      SWITCHGEN_MODELS: path.join(unset, 'models'),
      SWITCHGEN_LAB_DIR: path.join(unset, 'lab'),
    },
  },
})
