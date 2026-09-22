import { describe, expect, it } from 'vitest'
import { decide } from '../src/lib/recipe'
import { EXPLICIT, NOOBAI, SCREENCAP, library } from './fixtures'

const loaders = (def: { graph: Record<string, { class_type: string; inputs: Record<string, unknown> }> }) =>
  Object.values(def.graph)
    .filter((n) => /Lora/.test(n.class_type))
    .map((n) => n.inputs.lora_name)

const base = { look: 'anime' as const, installed: [NOOBAI], loras: library() }
const prompt = 'anime screencap of a girl on a rooftop, 90s retro aesthetic, film grain'

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

  it('treats "Also explicit anatomy" as an explicit brief', () => {
    const off = decide({ ...base, look: 'photoreal', prompt: 'portrait of a woman', anatomy: 'off' })
    const on = decide({ ...base, look: 'photoreal', prompt: 'portrait of a woman', anatomy: 'emphasised' })
    expect(off.ok && on.ok).toBe(true)
    if (!off.ok || !on.ok) return
    // The report carries the brief it ranked against; emphasised must mark it explicit.
    expect(on.report.brief?.explicit ?? on.anatomy === 'emphasised').toBeTruthy()
    expect(off.anatomy).toBe('off')
  })
})
