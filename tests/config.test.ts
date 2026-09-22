import { rmSync } from 'node:fs'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { tempRoots } from './http'

type Plugin = { name?: string; configureServer?: unknown; configurePreviewServer?: unknown }
type Config = {
  plugins: unknown[]
  server: { cors?: unknown }
  preview: { cors?: unknown }
}

let config: Config
let root = ''

beforeAll(async () => {
  // The config mounts every server module, and they read their roots when
  // they load: point them at empty temporary folders first.
  root = tempRoots().root
  config = (await import('../vite.config')).default as unknown as Config
})

afterAll(() => rmSync(root, { recursive: true, force: true }))

describe('the Vite config', () => {
  it('turns CORS off for the dev server and for preview', () => {
    // Vite's default lets a page on any localhost port read what this server answers.
    expect(config.server.cors).toBe(false)
    expect(config.preview.cors).toBe(false)
  })

  it('mounts the ComfyUI write guard in both servers, ahead of everything else', () => {
    const plugins = (config.plugins.flat(Infinity) as (Plugin | null | false)[]).filter((p): p is Plugin => !!p)
    const guard = plugins.find((p) => p.name === 'switchgen-comfy-guard')
    expect(guard).toBeDefined()
    expect(typeof guard!.configureServer).toBe('function')
    expect(typeof guard!.configurePreviewServer).toBe('function')
    // A plugin's middleware runs in the order the plugins are listed, and the
    // proxy runs after all of them.
    expect(plugins[0]).toBe(guard)
  })
})
