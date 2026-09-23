import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { LoraIndexEntry } from '../src/lib/loraIndex'
import type * as Loras from '../src/lib/loras'
import { FAMILIES } from '../src/lib/workflows'

// The checked-in caption index is rewritten from whatever LoRA folder the
// machine has, and a row there overrides the catalogue's trigger words. It is
// emptied here, so what is checked is the catalogue's own words on any machine.
vi.mock('../src/lib/loraIndex', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/lib/loraIndex')>()
  const rows = real.LORA_INDEX as LoraIndexEntry[]
  rows.splice(0, rows.length)
  return real
})

// The add-on library as it is read from the server's model listing.
let loras: typeof Loras
const file = (name: string) => ({ name, rel: `Lora/${name}`, folder: 'Lora', size: 300_000_000, mtime: 0 })

beforeEach(async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) =>
      url === '/api/models'
        ? new Response(
            JSON.stringify({
              files: [
                file('Wan22_I2V_NSFW_General_HIGH.safetensors'),
                file('Wan22_I2V_NSFW_General_LOW.safetensors'),
                file('detail-enhancer-il-pony.safetensors'),
                file('someone-elses-file.safetensors'),
              ],
            }),
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

describe('the words an add-on is said to answer to', () => {
  // Four catalogue rows once carried a heading from their model card
  // ("## Usage (Python)") as their trigger, and it was written into the prompt.
  it('are never a heading lifted from a model card', async () => {
    const lib = await loras.loadLoraLibrary()
    expect(lib.all.filter((l) => l.trigger.trim().startsWith('#')).map((l) => l.file)).toEqual([])
  })

  it('add nothing to the prompt for an add-on that has no trigger word', async () => {
    const lib = await loras.loadLoraLibrary()
    const stack = [{ file: 'detail-enhancer-il-pony.safetensors', strength: 0.6, enabled: true }]
    const target = { familyId: 'sdxl-illustrious', model: 'waiMatureIllustrious_v30.safetensors', arch: 'illustrious' as const }
    expect(loras.fitFor(lib.byFile.get('detail-enhancer-il-pony.safetensors')!, target).level).not.toBe('mismatch')
    expect(loras.triggersFor(stack, lib, target)).toEqual([])
    expect(loras.missingTriggers(stack, lib, 'a girl on a rooftop', target)).toEqual([])
  })
})

describe('what the add-on rack says about a file', () => {
  it('says an entry that has gone is no longer in the add-ons folder', async () => {
    const lib = await loras.loadLoraLibrary()
    const target = { familyId: 'sdxl-illustrious', model: 'waiMatureIllustrious_v30.safetensors', arch: 'illustrious' as const }
    const r = loras.resolveStack([{ file: 'gone-since.safetensors', strength: 0.5, enabled: true }], lib, target)
    expect(r.dropped[0]?.why).toBe('No longer in the add-ons folder.')
  })

  it('never calls an add-on a LoRA, whatever it is set against', async () => {
    const lib = await loras.loadLoraLibrary()
    const targets = FAMILIES.map((def) => loras.targetFor(def, def.dualModel ? '' : (def.models[0] ?? '')))
    for (const info of lib.all) {
      for (const target of targets) expect(loras.fitFor(info, target).why, `${info.file} on ${target.familyId}`).not.toContain('LoRA')
    }
    for (const target of targets) {
      const stack = [...lib.all.map((l) => ({ file: l.file, strength: 0.5, enabled: true })), { file: 'gone-since.safetensors', strength: 0.5, enabled: true }]
      for (const d of loras.resolveStack(stack, lib, target).dropped) expect(d.why, `${d.file} on ${target.familyId}`).not.toContain('LoRA')
    }
  })
})
