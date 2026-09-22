import { defineConfig } from 'vitest/config'

// The suite covers the pure modules under src/lib and the server's guard: the
// recipe, the archive store, availability, faults, add-on chaining. Anything
// that needs a running ComfyUI stays in `npm run validate`.
export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
})
