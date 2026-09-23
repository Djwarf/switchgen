import { afterEach, describe, expect, it, vi } from 'vitest'
import type { LoraIndexEntry } from '../src/lib/loraIndex'
import { INTENTS, TAG_STYLE_NOTE, intentReport } from '../src/lib/intent'
import { withLoras } from '../src/lib/refine'
import { suggest } from '../src/lib/suggest'
import { FAMILIES } from '../src/lib/workflows'
import { ANATOMY_HELPER, MICRO_DETAILS, library } from './fixtures'

/**
 * The words on screen about add-ons and models. The reader is never told
 * about a "LoRA", a "trigger" or a "base": an add-on, its word, and a model.
 * Each sentence here is printed as it stands, under an offer, a picture
 * reading, a model in the list or a derived graph.
 */

// The caption index, swapped for rows of the test's own (see recipe.test.ts):
// the two measured add-ons, and five with one defining tag each for the
// picture reader, which needs a corpus wide enough for a tag to be rare.
const readerFile = (tag: string) => `reader-${tag}.safetensors`
vi.mock('../src/lib/loraIndex', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/lib/loraIndex')>()
  const { ANATOMY_HELPER, MICRO_DETAILS, indexRow } = await import('./fixtures')
  const rows = real.LORA_INDEX as LoraIndexEntry[]
  rows.splice(
    0,
    rows.length,
    indexRow(MICRO_DETAILS, {
      base: 'pony',
      imageCount: 40,
      confidence: 'likely',
      promptTags: ['addmicrodetails'],
      triggers: [{ tag: 'addmicrodetails', count: 36, share: 0.9, kind: 'invented', uniqueToThisLora: true }],
      concepts: [{ tag: 'detailed', count: 30, share: 0.75 }],
    }),
    indexRow(ANATOMY_HELPER, {
      base: 'pony',
      imageCount: 40,
      confidence: 'none',
      promptTags: [],
      triggers: [],
      concepts: [{ tag: 'solo', count: 20, share: 0.5 }],
    }),
    ...['red_umbrella', 'lighthouse', 'snowy_owl', 'paper_lantern', 'tram_car'].map((tag) =>
      indexRow(`reader-${tag}.safetensors`, {
        base: 'pony',
        imageCount: 20,
        confidence: 'none',
        promptTags: [],
        triggers: [],
        concepts: [{ tag, count: 20, share: 1 }],
      }),
    ),
  )
  return real
})

const plain = (text: string, where: string, base = /\bbases?\b/i) => {
  for (const word of [/LoRA/, /\btriggers?\b/i, base]) expect(text, where).not.toMatch(word)
}

describe('the sentence under a measured add-on', () => {
  it('speaks of add-ons and the picture with none, never of a base', () => {
    const r = suggest({
      prompt: 'a woman, portrait',
      model: 'ponyDiffusionV6XL.safetensors',
      anatomy: 'natural',
      installed: library([MICRO_DETAILS, ANATOMY_HELPER]),
    })
    expect(r.stack.map((s) => s.file).sort()).toEqual([ANATOMY_HELPER, MICRO_DETAILS].sort())
    for (const s of r.stack) plain(s.why, s.file)
    expect(r.stack.find((s) => s.file === MICRO_DETAILS)!.why).toContain('2.132x as sharp as the same picture with no add-ons')
  })
})

describe('what the desk says about a model', () => {
  it('calls a model a model, in every look, with and without explicit anatomy', () => {
    for (const { id } of INTENTS) {
      for (const explicit of [true, false]) {
        const report = intentReport({ intent: id, explicit })
        // Z-Image Base is a model's name, and "the Base weights" are its
        // weights; the word in lower case is the one that must not appear.
        const unnamed = (s: string | null) => (s ?? '').replaceAll('Z-Image Base', '')
        const lower = /\bbases?\b/
        plain(unnamed(report.anatomyNote), `${id} anatomy note`, lower)
        for (const r of report.ranked) {
          for (const [field, text] of [['why', r.why], ['caveat', r.caveat], ['warning', r.warning]] as const) {
            plain(unnamed(text), `${id}, ${r.model}, ${field}`, lower)
          }
        }
      }
    }
    for (const [style, note] of Object.entries(TAG_STYLE_NOTE)) plain(note, style, /\bbases?\b/)
  })
})

describe('the note on a graph with add-ons chained in', () => {
  it('counts add-ons', () => {
    const def = FAMILIES.find((f) => f.id === 'sdxl-illustrious')!
    const note = withLoras(def, [{ name: 'x.safetensors', strength: 0.5 }])!.derived.note
    expect(note).toMatch(/add-on/)
    expect(note).not.toMatch(/LoRA/)
  })
})

describe('what the picture reader says about an add-on', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('speaks of add-ons when it offers one and when it rules one out', async () => {
    vi.stubGlobal('fetch', async (url: string) => {
      const answer = (body: unknown) => new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } })
      if (url === '/api/vision/capabilities') return answer({ server: 'x', tagger: true, detect: false })
      if (url === '/api/vision/tag') {
        return answer({
          tag: {
            rows: [
              {
                index: 0,
                width: 512,
                height: 512,
                rating: 'general',
                ratings: [],
                general: [{ tag: 'lighthouse', confidence: 0.9 }],
                character: [],
              },
            ],
          },
        })
      }
      return new Response('{}', { status: 404 })
    })
    vi.resetModules()
    const { inspectImage } = await import('../src/lib/vision')
    const report = await inspectImage('a.png')
    expect(report.unavailable).toBeUndefined()
    expect(report.suggestions.map((s) => s.entry.file)).toContain(readerFile('lighthouse'))
    expect(report.vetoed.length).toBeGreaterThan(0)
    const said = [...report.suggestions.map((s) => [s.entry.file, s.why]), ...report.vetoed.map((v) => [v.file, v.why])]
    for (const [file, why] of said) {
      expect(why, file).toMatch(/add-on/)
      expect(why, file).not.toMatch(/LoRA/)
    }
  })
})
