import { defineConfig } from 'vitest/config'

// The suite covers the pure modules under src/lib, the desks' job engines and
// the ledger with ComfyUI stood in (a fetch that answers from a table, a fake
// socket, a tab's storage and page events), a few components rendered to a
// string, the service worker run against a stand-in `self`, the Vite config,
// the install advice in bin/switchgen and the README read as text, and the
// server middlewares driven without a socket against temporary folders, some
// in a child process of their own. The downloader is driven against a fake
// aria2c the test writes, which fetches nothing. The LoRA index the author's
// folder produces is replaced by rows the tests carry wherever it would decide
// an answer. Nothing in it needs ComfyUI, a model file, a running app or the
// author's own folders, so it runs the same on a clean CI runner. There is no
// DOM, so what only shows when a button is pressed in a browser is not
// covered here. Anything that needs a live ComfyUI stays in
// `npm run validate`.
export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
})
