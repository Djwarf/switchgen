import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Loras from '../src/lib/loras'
import { FAMILIES } from '../src/lib/workflows'

// The add-on library as it is read from the server's model listing.
let loras: typeof Loras
const file = (name: string) => ({ name, rel: `Lora/${name}`, folder: 'Lora', size: 300_000_000, mtime: 0 })

beforeEach(async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) =>
      url === '/api/models'
        ? new Response(
            JSON.stringify({ files: [file('Wan22_I2V_NSFW_General_HIGH.safetensors'), file('Wan22_I2V_NSFW_General_LOW.safetensors')] }),
            { headers: { 'content-type': 'application/json' } },
          )
        : new Response('{}', { status: 404 }),
    ),
  )
  vi.resetModules()
  loras = await import('../src/lib/loras')
})
afterEach(() => vi.unstubAllGlobals())

describe('a Wan add-on whose size was read off its own file', () => {
  it('is known to be for the 14B, whatever case its name is in', async () => {
    const lib = await loras.loadLoraLibrary()
    for (const name of ['Wan22_I2V_NSFW_General_HIGH.safetensors', 'Wan22_I2V_NSFW_General_LOW.safetensors']) {
      const info = lib.all.find((l) => l.file === name)
      expect(info?.installed).toBe(true)
      expect(info?.arch).toBe('wan-14b')
    }
  })

  it('fits the 14B pairs and neither of the smaller Wan sizes', async () => {
    const lib = await loras.loadLoraLibrary()
    const info = lib.all.find((l) => l.file === 'Wan22_I2V_NSFW_General_HIGH.safetensors')!
    const fit = (id: string) => {
      const def = FAMILIES.find((f) => f.id === id)!
      return loras.fitFor(info, loras.targetFor(def, def.dualModel ? '' : def.models[0]!)).level
    }
    expect(fit('wan22-14b-t2v')).toBe('match')
    expect(fit('wan22-14b-i2v')).toBe('match')
    expect(fit('wan22-5b')).toBe('mismatch')
    expect(fit('wan21-vace-1_3b-gguf')).toBe('mismatch')
  })
})
