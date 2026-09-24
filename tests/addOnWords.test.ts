import { readFileSync } from 'node:fs'
import path from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { LoraIndexEntry } from '../src/lib/loraIndex'
import { MEASURED_SINGLES } from '../src/components/loras/measured'
import { measuredText } from '../src/components/loras/measuredText'
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

describe('the line under a measured add-on in the rack', () => {
  it('compares it with the picture made with no add-ons, not with a base', () => {
    const line = measuredText(MEASURED_SINGLES[ANATOMY_HELPER]!, false)
    expect(line).toBe('0.81x as sharp as no add-ons at 0.3, 0.72x at 0.5, 0.44x at 0.8')
    plain(line, ANATOMY_HELPER)
  })

  it('says an add-on that takes a word was measured without it', () => {
    const line = measuredText(MEASURED_SINGLES[MICRO_DETAILS]!, true)
    expect(line.endsWith(', without its word')).toBe(true)
    plain(line, MICRO_DETAILS)
  })
})

describe('what the queue on the server says', () => {
  // Every sentence the queue and the desks say about work it holds, run as
  // the desks run them, and the copy that only exists as text in a page.
  const read = (...p: string[]) => readFileSync(path.resolve(import.meta.dirname, '..', ...p), 'utf8')

  /** The prose a file carries in its string literals: anything with two words in it that is not code. */
  const prose = (source: string) =>
    [...source.matchAll(/'((?:[^'\\\n]|\\.)*)'|`((?:[^`\\]|\\.)*)`/g)]
      .map((m) => m[1] ?? m[2] ?? '')
      .filter((s) => /[A-Za-z]+ [A-Za-z]+/.test(s) && !s.includes('${base'))

  it('says every wait, ending and hold in plain words, on every desk', async () => {
    const r = await import('../src/lib/runner')
    const statuses = ['waiting', 'releasing', 'sending', 'queued', 'running', 'filing', 'done', 'failed', 'stopped', 'lost', 'unsent', 'skipped'] as const
    const waits = [null, 'turn', 'before', 'heavy', 'queue', 'comfy', 'held', 'disk'] as const
    for (const desk of ['video', 'images', 'reel'] as const) {
      for (const status of statuses) {
        for (const w of waits) {
          for (const heavy of [true, false]) {
            const job = { id: 'j', desk, status, heavy, stopRequested: false, wait: w ? { for: w, ahead: 2 } : null } as unknown as Parameters<typeof r.waitLine>[0]
            const line = r.waitLine(job, desk)
            plain(`${line.stage} ${line.note ?? ''}`, `${desk} ${status} ${w} ${heavy}`)
          }
        }
      }
    }
    for (const s of [r.WAITS_ON_SERVER, r.RUNNER_LOST, r.RUNNER_UNSENT, r.OUTBOX_GIVEN_UP, r.OUTBOX_NOT_TAKEN, r.HELD_AFTER_LOSS, r.HELD_AFTER_UNSENT, r.HELD_AFTER_RESTART, r.HELD_AFTER_PAUSE]) plain(s, s)
    const codes = ['refused', 'failed', 'lost', 'unsent', 'ended-unsent', 'no-file', 'no-frame', 'stopped', 'skipped', 'internal'] as const
    for (const code of codes) {
      for (const sent of [true, false]) {
        const job = { id: 'j', status: 'failed', promptId: 'p', error: { code, message: null, node: null, nodeType: null, nodeErrors: null, mayExist: false, sent, after: null } } as unknown as Parameters<typeof r.runnerFault>[0]
        plain(r.runnerFault(job).message, code)
      }
    }
  })

  it('says why a page sends its own work, and why the server would not take it, in plain words', async () => {
    // @ts-expect-error the queue's parts are plain ESM without a declaration of their own
    const { REASONS } = await import('../server/runner/engine.mjs')
    const r = await import('../src/lib/runner')
    for (const reason of [REASONS.off, REASONS.held, REASONS.folder, REASONS.tripped, REASONS.stopped, REASONS.unwritable(new Error('EACCES: permission denied'))]) {
      expect(reason, 'a reason the queue gives').toEqual(expect.any(String))
      plain(reason, reason)
      plain(r.fallbackLine(reason), reason)
    }
    for (const file of ['server/runner.mjs', 'server/runner/engine.mjs', 'server/runner/routes.mjs', 'server/runner/filing.mjs', 'src/lib/runner.ts']) {
      const said = prose(read(...file.split('/')))
      expect(said.length, file).toBeGreaterThan(3)
      for (const s of said) plain(s, `${file}: ${s}`)
    }
  })

  it('says the desks\' own lines for the queue in plain words', async () => {
    const pictures = await import('../src/routes/Pictures')
    const { REEL_BUSY } = await import('../src/components/reel/engine')
    plain(pictures.BATCH_BUSY, 'BATCH_BUSY')
    plain(REEL_BUSY, 'REEL_BUSY')
    for (const [first, last, at] of [[2, 2, 1], [2, 3, 1], [3, 7, 2]]) plain(pictures.serverRestLine(first!, last!, at!)!, 'serverRestLine')
    const holds = [null, { press: true, rest: 0 }, { press: true, rest: 2 }, { press: false, rest: 1 }, { press: false, rest: 2 }]
    for (const stage of ['Handing it to the server', 'Waiting its turn', 'Queued', 'Drawing · step 3 of 20']) {
      for (const index of [1, 2, 3]) {
        for (const onHold of holds) {
          const status = stage === 'Queued' ? 'queued' : stage.startsWith('Drawing') ? 'running' : 'submitting'
          const line = pictures.serverWaitingLine({ status, stage, index, total: 3, runner: true, note: 'A note.', onHold })
          if (line) plain(line, `serverWaitingLine ${stage} ${index} ${JSON.stringify(onHold)}`)
        }
      }
    }
    // Copy the desks keep to themselves, read as the page carries it.
    const video = read('src', 'routes', 'Video.tsx')
    for (const name of ['HANDING', 'NOT_ANSWERED', 'BEHIND_HERE']) {
      const text = new RegExp(`const ${name} =\\s*'([^']*)'`).exec(video)?.[1]
      expect(text, name).toBeTruthy()
      plain(text!, name)
    }
    const handing = /const HANDING_LINE =\s*'([^']*)'/.exec(read('src', 'routes', 'Pictures.tsx'))?.[1]
    expect(handing).toMatch(/has not answered yet/)
    plain(handing!, 'HANDING_LINE')
    // The shell's notice about held work, on every room: its sentences, and the page it draws.
    plain(read('src', 'components', 'shell', 'RunnerHold.tsx').replace(/className="[^"]*"/g, '').replace(/const CONTROL =[^\n]*/, ''), 'RunnerHold')
    const reel = read('src', 'routes', 'Reel.tsx')
    for (const fn of ['HeldOnServer', 'OtherPass']) {
      const body = new RegExp(`function ${fn}\\([\\s\\S]*?\\n\\}\\n`).exec(reel)?.[0]
      expect(body, fn).toBeTruthy()
      plain(body!.replace(/className="[^"]*"/g, ''), fn)
    }
  })
})
