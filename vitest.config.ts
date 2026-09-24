import os from 'node:os'
import path from 'node:path'
import { defineConfig } from 'vitest/config'

// The suite covers the pure modules under src/lib, the desks' job engines and
// the ledger with ComfyUI stood in (a fetch that answers from a table, a fake
// socket, a tab's storage and page events), a few components rendered to a
// string, the service worker run against a stand-in `self`, the Vite config,
// the install advice in bin/switchgen and the README read as text, and the
// server middlewares driven without a socket against temporary folders, some
// in a child process of their own. The downloader is driven against a fake
// aria2c the test writes, which fetches nothing, the reel's stitch against a
// fake ffmpeg and ffprobe, and the picture reader against a fake Python. The
// LoRA index the author's folder produces is replaced by rows the tests carry
// wherever it would decide an answer. Nothing in it needs ComfyUI, a model
// file, a running app or the author's own folders, so it runs the same on a
// clean CI runner. There is no DOM, so what only shows when a button is
// pressed in a browser is not covered here. Anything that needs a live
// ComfyUI stays in `npm run validate`.
//
// A test that waits for something (a debounced save, a child process, a
// socket reconnecting) waits for it by polling with a timeout of its own, not
// by sleeping a set time. Each test gets 15 s, well above those waits, so on a
// slow runner a wait that runs out reports what it was waiting for rather than
// a bare "timed out".
export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
    testTimeout: 15_000,
    // A server module under test must never reach the real ComfyUI or send it
    // work: its address points at a port nothing listens on (9, discard), and
    // the server-side runner stays off unless a test turns it on with its own
    // stand-in ComfyUI and temporary folders.
    //
    // Nor may it reach the author's own folders. Every test that loads a
    // server module sets its roots first (tempRoots in tests/http.ts); one
    // that forgot would otherwise find the defaults, which name the real
    // library and outputs, where the running app keeps its archive and its
    // queue. The archive, the thumbnails and the queue's folder are found
    // under the outputs root unless named, so a test that names only that
    // root still gets all three beside it.
    env: {
      COMFY_URL: 'http://127.0.0.1:9',
      SWITCHGEN_RUNNER: 'off',
      SWITCHGEN_OUTPUTS: path.join(os.tmpdir(), 'switchgen-vitest-unset', 'outputs'),
      SWITCHGEN_MODELS: path.join(os.tmpdir(), 'switchgen-vitest-unset', 'models'),
    },
  },
})
