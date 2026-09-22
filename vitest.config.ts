import { defineConfig } from 'vitest/config'

// The suite covers the pure modules under src/lib, the reel engine with
// ComfyUI stood in, and the server middlewares driven without a socket against
// temporary folders. Nothing in it needs ComfyUI, a model file, a running app
// or the author's own folders, so it runs the same on a clean CI runner.
// Anything that needs a live ComfyUI stays in `npm run validate`.
export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
})
