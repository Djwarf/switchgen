import { describe, expect, it, vi } from 'vitest'
import type { LoraIndexEntry } from '../src/lib/loraIndex'
import { decide, explain } from '../src/lib/recipe'
import { ANATOMY_HELPER, EXPLICIT, MICRO_DETAILS, NOOBAI, SCREENCAP, library } from './fixtures'

// The recipe reads triggers and training vocabulary from src/lib/loraIndex.ts,
// which `npm run index-loras` regenerates from the machine's own LoRA folder.
// Only the data is swapped for the fixture rows: the array is refilled in
// place before anything reads it, so byFilename, matchPrompt and the rest stay
// the real functions, reading the fixture instead of the author's folder.
vi.mock('../src/lib/loraIndex', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/lib/loraIndex')>()
  const { FIXTURE_INDEX } = await import('./fixtures')
  const rows = real.LORA_INDEX as LoraIndexEntry[]
  rows.splice(0, rows.length, ...FIXTURE_INDEX)
  return real
})

const loaders = (def: { graph: Record<string, { class_type: string; inputs: Record<string, unknown> }> }) =>
  Object.values(def.graph)
    .filter((n) => /Lora/.test(n.class_type))
    .map((n) => n.inputs.lora_name)

const base = { look: 'anime' as const, installed: [NOOBAI], loras: library() }
const prompt = 'anime screencap of a girl on a rooftop, 90s retro aesthetic, film grain'

describe('the fixture index', () => {
  it('is what the recipe reads, not the checked-in folder index', async () => {
    const { LORA_INDEX, byFilename } = await import('../src/lib/loraIndex')
    expect(LORA_INDEX.map((e) => e.file).sort()).toEqual([EXPLICIT, SCREENCAP].sort())
    expect(byFilename(SCREENCAP)?.triggerPhrase).toContain('anime screencap')
  })
})

describe('decide(): add-on offers', () => {
  it('offers a prompt-matched add-on rather than applying it', () => {
    const r = decide({ ...base, prompt, anatomy: 'off' })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.offers.map((o) => o.file)).toContain(SCREENCAP)
    expect(r.loras).toHaveLength(0)
    expect(loaders(r.def)).toHaveLength(0)
  })

  it('chains an accepted add-on into the graph and adds its trigger words', () => {
    const r = decide({ ...base, prompt, anatomy: 'off', addOns: { accepted: [SCREENCAP] } })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(loaders(r.def)).toContain(SCREENCAP)
    expect(r.loras.map((l) => l.file)).toContain(SCREENCAP)
    expect(r.params.positive.toLowerCase()).toContain('anime screencap')
  })

  it('keeps an accepted add-on after the prompt stops matching it', () => {
    const r = decide({ ...base, prompt: 'a red bicycle leaning on a brick wall', anatomy: 'off', addOns: { accepted: [SCREENCAP] } })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(loaders(r.def)).toContain(SCREENCAP)
  })

  it('lets a decline win over an accept', () => {
    // Offered without the decline, so the decline below is doing the work.
    const open = decide({ ...base, prompt, anatomy: 'off' })
    expect(open.ok && open.offers.some((o) => o.file === SCREENCAP)).toBe(true)
    const r = decide({ ...base, prompt, anatomy: 'off', addOns: { accepted: [SCREENCAP], declined: [SCREENCAP] } })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(loaders(r.def)).not.toContain(SCREENCAP)
    expect(r.offers.map((o) => o.file)).not.toContain(SCREENCAP)
  })
})

describe('decide(): the content gate', () => {
  const portrait = 'portrait of a woman, 1girl, smile, soft light, realistic photography'

  it('does not offer an explicit add-on at Standard for a plain portrait', () => {
    const r = decide({ ...base, prompt: portrait, anatomy: 'off' })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.offers.map((o) => o.file)).not.toContain(EXPLICIT)
  })

  it('offers it when the prompt itself is explicit', () => {
    const r = decide({ ...base, prompt: `${portrait}, nude, explicit`, anatomy: 'off' })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.offers.map((o) => o.file)).toContain(EXPLICIT)
  })

  it('treats "Also explicit anatomy" as an explicit brief, and "Sharper faces and hands" as not', () => {
    const at = (anatomy: 'off' | 'natural' | 'emphasised') =>
      decide({ ...base, look: 'photoreal', prompt: 'portrait of a woman', anatomy })
    const off = at('off')
    const nat = at('natural')
    const on = at('emphasised')
    expect(off.ok && nat.ok && on.ok).toBe(true)
    if (!off.ok || !nat.ok || !on.ok) return
    // The brief is what the ranking weights swing on. 'natural' once set it,
    // which ranked a request for sharper hands as a request for explicit work.
    expect(on.report.brief.explicit).toBe(true)
    expect(off.report.brief.explicit).toBe(false)
    expect(nat.report.brief.explicit).toBe(false)
  })
})

describe('the sharpness figure and the words under the button', () => {
  const helpers = library([SCREENCAP, EXPLICIT, ANATOMY_HELPER, MICRO_DETAILS])

  it('quotes the figure for the measured set, and withdraws it once an add-on joins', () => {
    const measured = decide({ ...base, loras: helpers, prompt, anatomy: 'natural' })
    expect(measured.ok).toBe(true)
    if (!measured.ok) return
    // Without this the assertion below would pass on a recipe that never had a figure.
    expect(measured.sharpness).not.toBeNull()

    const added = decide({ ...base, loras: helpers, prompt, anatomy: 'natural', addOns: { accepted: [SCREENCAP] } })
    expect(added.ok).toBe(true)
    if (!added.ok) return
    expect(loaders(added.def)).toContain(SCREENCAP)
    expect(added.sharpness).toBeNull()
    expect(explain(added)).not.toMatch(/as sharp as/)
  })

  it('claims no measurement for an add-on the reader chose at Standard', () => {
    const r = decide({ ...base, prompt, anatomy: 'off', addOns: { accepted: [SCREENCAP] } })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    const words = explain(r)
    // "Not measured here" is the honest line for it; nothing may say a measured set ran.
    expect(words).toContain('Not measured here')
    expect(words).not.toMatch(/measured (strengths|stack|set)|as sharp as|helper/)
  })

  it('does not say a model it cannot carry takes the helpers', () => {
    const klein = 'flux-2-klein-4b-fp8.safetensors'
    const r = decide({ ...base, look: 'photoreal', installed: [klein, NOOBAI], loras: helpers, prompt: 'portrait of a woman', anatomy: 'natural' })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.model).toBe(klein)
    const words = explain(r)
    expect(words).not.toContain('carries explicit anatomy')
    expect(words).toContain('do not attach')
    // The reason is in the notes, not the warnings; an empty list must not be pointed at.
    if (!r.warnings.length) expect(words).not.toMatch(/say why/)
  })
})
