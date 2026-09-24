/**
 * Sealing and judging a small run end to end (TEST PLAN, D, seal and leak
 * check), with stand-in pictures: PNGs carrying a tEXt chunk that names the
 * ckpt, as ComfyUI writes them, and a JPEG stored sideways with an EXIF
 * orientation 6 (a stand-in, never the user's cat). The blind copies need
 * ffmpeg, so this file is skipped without it.
 */
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, it } from 'vitest'
import { runDir } from '../core/env.ts'
import { defineSuite } from '../core/suite.ts'
import { expand } from '../core/cells.ts'
import { addRef, ensureRefCopies, refIndex, setMask } from '../run/refs.ts'
import * as FP from '../suites/first-pass.ts'
import { sealRun } from '../judge/seal.ts'
import { metadataProblems, webpChunks } from '../judge/blind.ts'
import * as Events from '../judge/events.ts'
import * as Queue from '../judge/queue.ts'
import * as Aggregate from '../judge/aggregate.ts'
import { renderReport, toCsv } from '../judge/report.ts'
import { readLedger } from '../run/ledger.ts'
import { readReadings } from '../run/reader.ts'
import { HAS_FFMPEG, type Loose, png, removeTemp, sidewaysJpeg, tempEnv } from './helpers.ts'

const { appendEvents, readEvents, reduceEvents } = Events as Loose<typeof Events>
const { nextItem, remaining, writeSweep } = Queue as Loose<typeof Queue>
const { aggregate } = Aggregate as Loose<typeof Aggregate>

afterAll(removeTemp)

const env = tempEnv()
const root = env.labDir
const RUN = 'mini-1'
const png3 = (w: number, h: number, rgb: [number, number, number], text: string) => png(w, h, rgb, { text })

const GROUP = 'Four adult friends in their thirties sitting around a café table, all smiling at the camera, medium-wide shot.'
const CORE = ['noobai', 'klein', 'zbase']
const mini = defineSuite({
  id: 'mini', version: 1, study: 'mini-study', seeds: FP.SEEDS, steps: 28, sampler: FP.SAMPLER, shapes: FP.SHAPES,
  models: { noobai: FP.MODELS.noobai, klein: FP.MODELS.klein, zbase: FP.MODELS.zbase, qwenEdit: FP.MODELS.qwenEdit },
  core: CORE,
  refs: { cat: { describe: 'a grey cat with white paws' }, scene: { describe: 'a table', needsMask: true } },
  slots: [
    { ...FP.FRUIT, models: 'core' },
    { id: 'sensitivity.red', set: 'sensitivity', block: 'sensitivity', shape: 'square', op: 't2i', text: 'A red vintage car parked beside a lake in autumn.', condition: 'red', models: 'core' },
    { id: 'sensitivity.blue', set: 'sensitivity', block: 'sensitivity', shape: 'square', op: 't2i', text: 'A blue vintage car parked beside a lake in autumn.', condition: 'blue', models: 'core' },
    { id: 'detail.group', block: 'anatomy', shape: 'square', op: 't2i', text: GROUP, humans: true, models: 'core' },
    { id: 'detail.group.face', block: 'detail', shape: 'square', op: 'face', target: 'face', after: 'detail.group', text: GROUP, condition: 'with the face pass', humans: true, models: 'core' },
    { id: 'ref.cat.ghibli', block: 'reference', shape: 'square', op: 'i2i', source: { ref: 'cat' }, denoise: 0.65, text: 'A Studio Ghibli style illustration of {describe}.',
      perModelText: { qwenEdit: "Turn this photo into a Studio Ghibli style illustration. Keep the cat's markings." }, task: 'Restyle the reference as a Studio Ghibli illustration, keeping this cat.', models: [...CORE, 'qwenEdit'] },
    { id: 'edit.winter', block: 'edit', shape: 'square', op: 'edit', source: { ref: 'scene' }, text: 'Make it look like a snowy winter day; change nothing else.', models: 'edit' },
    { id: 'region.scene', block: 'region', shape: 'square', op: 'region', source: { ref: 'scene' }, denoise: 0.55, text: 'a small vase of yellow tulips',
      perModelText: { qwenEdit: 'Place a small vase of yellow tulips in the {position} of the picture; change nothing else.' }, task: 'Put a small vase of yellow tulips in the outlined area and change nothing else.', models: [...CORE, 'qwenEdit'] },
    { id: 'content.beach', block: 'content', shape: 'square', op: 't2i', text: 'A woman in her thirties relaxing on a beach towel, reading a book, on a summer afternoon.', humans: true, measuredOnly: true, models: 'core' },
    { ...FP.FRUIT, id: 'promptstyle.fruit.sentence', set: 'promptstyle.fruit', second: undefined, condition: 'sentence', models: ['noobai'] },
    { ...FP.FRUIT, id: 'promptstyle.fruit.tags', set: 'promptstyle.fruit', second: undefined, condition: 'tags', text: '3 apples, red apple, 1 pear, green pear, blue mug, wooden table', models: ['noobai'] },
  ],
  chains: [
    { id: 'chain.draft', name: 'Z-Image Base draft, NoobAI pass', block: 'following', compose: { slot: 'following.fruit', model: 'zbase' }, steps: [{ model: 'noobai', op: 'i2i', denoise: 0.4, text: '{text}' }] },
    { id: 'chain.faces', name: 'Klein, then faces', block: 'anatomy', also: ['detail'], compose: { slot: 'detail.group', model: 'klein' }, steps: [{ model: 'zbase', op: 'detailOnPicture', target: 'face', text: '{text}' }] },
  ],
  sweep: { models: ['zbase'], slots: ['following.fruit'], steps: [8, 28, 40], homeFor: { zbase: 30 } },
  samplerCheck: { models: ['zbase'], slots: ['following.fruit'] },
})

let seedByte = 0
const rng = (n: number) => { const out = Buffer.alloc(n); for (let i = 0; i < n; i++) { out[i] = createHash('sha256').update(String(seedByte++)).digest()[0] } return out }

describe.skipIf(!HAS_FFMPEG)('a small run, sealed and judged', () => {
  let x: any, ids: string[], dir: string, res: any, items: any, sealed: any, sealedText: string
  beforeAll(async () => {
    mkdirSync(env.outputs, { recursive: true })
    const cat = addRef(env, sidewaysJpeg(join(root, 'cat.jpg'), 160, 120), 'cat')
    assert.equal(cat.width, 120, 'the stand-in cat is upright: 120 wide')
    assert.equal(cat.height, 160)
    addRef(env, sidewaysJpeg(join(root, 'scene.jpg'), 192, 128), 'scene')
    setMask(env, 'scene', { x: 16, y: 32, w: 64, h: 48 })
    ensureRefCopies(env, ['cat', 'scene'])

    x = expand([mini], refIndex(env), { context: [] })
    assert.equal(x.blocked.length, 0, JSON.stringify(x.blocked))
    ids = [...new Set<string>(x.cells.map((c: { cellId: string }) => c.cellId))]
    dir = runDir(env, RUN)
    mkdirSync(dir, { recursive: true })
    writeFileSync(join(dir, 'plan.json'), JSON.stringify({ v: 1, run: RUN, study: 'mini-study', suites: [{ id: 'mini', version: 1 }], order: ids, reused: [], na: x.na }))
    // Make every picture, with a few failures and one removal.
    const byId = new Map(x.cells.map((c: { cellId: string }) => [c.cellId, c]))
    const fail = new Set<string>()
    const cellsOf = (pred: (c: Record<string, unknown>) => boolean) => x.cells.filter(pred).map((c: { cellId: string }) => c.cellId)
    const kleinFruit = cellsOf(c => c.slot === 'following.fruit' && c.model === 'klein' && c.set === 'following.fruit' && c.seed === 1001)
    const zbaseBlue = cellsOf(c => c.slot === 'sensitivity.blue' && c.model === 'zbase' && (c.seed === 1001 || c.seed === 2002))
    for (const id of [...kleinFruit, ...zbaseBlue]) fail.add(id)
    const removed = cellsOf(c => c.slot === 'content.beach' && c.model === 'noobai' && c.seed === 3003)[0]
    const lastModel = new Map<string, string>()
    let at = 1_700_000_000_000
    mkdirSync(join(env.outputs, '.lab', 'cells'), { recursive: true })
    for (const id of ids) {
      const c = byId.get(id) as Record<string, any>
      at += 1000
      if (fail.has(id)) {
        for (const attempt of [1, 2]) appendFileSync(join(dir, 'ledger.jsonl'), JSON.stringify({ t: 'ended', at, job: randomUUID(), cell: id, status: 'failed', error: { code: attempt === 1 ? 'comfy' : 'refused', message: 'x', node: null, nodeType: null }, files: [], primary: null, promptId: null, durationMs: 0, ranAt: null, finishedAt: null, cached: false, cold: false, attempt }) + '\n')
        continue
      }
      const name = `${id}_00001_.png`
      const w = Math.max(8, Math.round(c.width / 16)); const h = Math.max(8, Math.round(c.height / 16))
      const hue = parseInt(id.slice(0, 6), 16)
      writeFileSync(join(env.outputs, '.lab', 'cells', name), png3(w, h, [hue & 255, (hue >> 8) & 255, (hue >> 16) & 255], JSON.stringify({ ckpt_name: c.file, model: c.model })))
      const cold = lastModel.get('x') !== c.file
      lastModel.set('x', c.file)
      appendFileSync(join(dir, 'ledger.jsonl'), JSON.stringify({ t: 'ended', at, job: randomUUID(), cell: id, status: 'done', error: null, files: [], primary: { filename: name, subfolder: '.lab/cells', type: 'output' }, promptId: randomUUID(), durationMs: (c.steps * 400) + (c.model === 'klein' ? 3000 : 1000) + (cold ? 20000 : 0), ranAt: at, finishedAt: at, cached: false, cold, attempt: 1 }) + '\n')
    }
    appendFileSync(join(dir, 'ledger.jsonl'), JSON.stringify({ t: 'removed', at, cell: removed, why: 'quarantine' }) + '\n')
    // The picture reader: a few above general.
    const beach = cellsOf(c => c.slot === 'content.beach')
    for (const [i, id] of beach.entries()) appendFileSync(join(env.labDir, 'readings.jsonl'), JSON.stringify({ v: 1, cellId: id, rel: '', at, rating: i % 5 === 0 ? 'sensitive' : i === 7 ? 'questionable' : 'general', ratings: [], general: [], character: [], quarantined: false }) + '\n')

    res = await sealRun(env, RUN, { ffmpeg: 'ffmpeg', suites: [mini], rng })
    assert.ok(res.sets > 5, `sets ${res.sets}`)
    items = JSON.parse(readFileSync(join(dir, 'items.json'), 'utf8'))
    sealedText = readFileSync(join(dir, 'sealed.json'), 'utf8')
    sealed = JSON.parse(sealedText)
  }, 120_000)

  const setBySlot = (slot: string) => items.sets.find((s: any) => !s.secondOf && sealed.sets[s.setId].slots[0] === slot)
  const who = (s: any) => s.grids.map((g: any) => sealed.gridOf[g.itemId].contestant).sort()

  it('writes sealed.json once, heads items.json with its sha, and finishes a seal stopped before items.json', async () => {
      assert.equal(items.sealedSha, createHash('sha256').update(sealedText).digest('hex'), 'the sealed sha heads items.json')
    const again = await sealRun(env, RUN, { ffmpeg: 'ffmpeg', suites: [mini] })
    assert.deepEqual(again, res, 'sealing twice gives the first answer and writes nothing')
    // A seal that stopped after the key but before items.json finishes on the next call.
    renameSync(join(dir, 'items.json'), join(dir, 'items.json.pending'))
    assert.deepEqual(await sealRun(env, RUN, { ffmpeg: 'ffmpeg', suites: [mini] }), res)
    assert.ok(existsSync(join(dir, 'items.json')))

  })

  it('items.json names no model key, file, family, cell, rel, job or prompt id', () => {
    // Blind: no model key, file, family, job or prompt id in items.json.
    const itemsText = readFileSync(join(dir, 'items.json'), 'utf8')
    const secret = new Set<string>()
    for (const t of Object.values(sealed.tokens) as any[]) for (const s of [t.file, t.file.replace(/\.(safetensors|gguf)$/, ''), t.familyId, t.cellId, t.jobId, t.promptId, t.rel]) if (s && s.length >= 3) secret.add(s)
    for (const k of [...CORE, 'qwenEdit', 'chain.draft', 'chain.faces']) secret.add(k)
    for (const s of secret) assert.ok(!new RegExp(`(^|[^a-z0-9])${s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}([^a-z0-9]|$)`, 'i').test(itemsText), `items.json names ${s}`)
  })

  it('every blind copy holds picture chunks only, under a 128-bit token, with no ckpt_name', () => {
    // Blind copies: picture data only, and tokens of 128 bits.
    const view = readdirSync(join(dir, 'view'))
    assert.ok(view.length > 50)
    for (const f of view) {
      assert.match(f, /^[0-9a-f]{32}-(g|f)\.webp$/)
      const buf = readFileSync(join(dir, 'view', f))
      assert.deepEqual(metadataProblems(buf), [], f)
      assert.ok(!buf.includes('ckpt_name'))
    }
  })

  it('letters are drawn anew for each set', () => {
    // Letters are drawn per set.
    const letterRuns = items.sets.filter((s: any) => s.grids.length >= 3).map((s: any) => s.grids.map((g: any) => `${g.letter}=${sealed.gridOf[g.itemId].contestant}`).join(','))
    assert.equal(new Set(letterRuns).size, letterRuns.length, 'no two sets share letters for the same contestants')
  })

  it('each seed keeps its place in every grid', () => {
    // Seeds keep their places.
    for (const s of items.sets) for (const g of s.grids) {
      const toks: (string | null)[] = g.tiles ?? g.rows.map((r: any) => r[1])
      toks.forEach((t, i) => { if (t) assert.equal(sealed.tokens[t].seed, FP.SEEDS[i]) })
    }
  })

  it('N/A contestants are absent, and a grid with fewer than 3 pictures made is left out as not made', () => {
    // N/A contestants are absent; too-few grids are not made.
    assert.deepEqual(who(setBySlot('detail.group.face')), ['chain.faces', 'noobai', 'zbase'])
    assert.deepEqual(who(setBySlot('ref.cat.ghibli')), ['noobai', 'qwenEdit', 'zbase'])
    assert.deepEqual(who(setBySlot('sensitivity.red')), ['klein', 'noobai'], 'zbase has only 2 of 4 pairs')
    assert.ok(sealed.notMade.some((n: any) => n.contestant === 'zbase'))
    assert.deepEqual(who(setBySlot('following.fruit')), ['chain.draft', 'klein', 'noobai', 'zbase'])
    assert.deepEqual(who(setBySlot('detail.group')), ['chain.faces', 'klein', 'noobai', 'zbase'])
  })

  it('the phone gets the neutral task, the pinned photo and mask, and before/after conditions', () => {
    // The neutral task, never a model's own wording.
    assert.equal(setBySlot('ref.cat.ghibli').brief.task, 'Restyle the reference as a Studio Ghibli illustration, keeping this cat.')
    assert.ok(setBySlot('ref.cat.ghibli').brief.pinned.ref)
    const region = setBySlot('region.scene')
    assert.equal(region.mode, 'beforeAfter')
    assert.ok(region.brief.pinned.base && region.brief.pinned.mask)
    assert.equal(setBySlot('sensitivity.red').mode, 'beforeAfter')
    assert.deepEqual(setBySlot('sensitivity.red').brief.conditions, ['red', 'blue'])
    assert.equal(setBySlot('content.beach'), undefined, 'measured only: no set')
  })

  it('the pinned cat, stored sideways, is shown upright', () => {
    // The pinned cat is shown upright (stored sideways with an orientation flag).
    const pin = setBySlot('ref.cat.ghibli').brief.pinned.ref
    const dims = (buf: Buffer) => { const c = webpChunks(buf)[0]; const i = buf.indexOf(c.fourcc); return c.fourcc === 'VP8 ' ? [buf.readUInt16LE(i + 14) & 0x3fff, buf.readUInt16LE(i + 16) & 0x3fff] : [0, 0] }
    const [pw, ph] = dims(readFileSync(join(dir, 'view', `${pin}-f.webp`)))
    assert.ok(pw < ph, `upright cat ${pw}x${ph}`)
    // And turned, not squeezed: the blue band, on the left of the stored picture, runs along the top once upright.
    const px = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', join(dir, 'view', `${pin}-f.webp`), '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], { maxBuffer: 64 << 20 })
    const at = (x: number, y: number) => [...px.subarray((y * pw + x) * 3, (y * pw + x) * 3 + 3)]
    assert.ok(at(Math.round(pw / 2), 3)[2] > 150, `top middle ${at(Math.round(pw / 2), 3)}`)
    assert.ok(at(Math.round(pw / 2), ph - 4)[0] > 150, `bottom middle ${at(Math.round(pw / 2), ph - 4)}`)
    assert.ok(at(3, Math.round(ph * 0.75))[0] > 150, 'the left edge below the band is orange')
  })

  it('pairs: a chain against its start, the sweep, the sampler check and prompt style; the sweep key holds no token', () => {
    // Pairs: chain against its start, sweep, sampler, prompt style.
    const kinds = items.pairs.map((p: any) => p.kind).sort()
    assert.deepEqual([...new Set(kinds)], ['chain', 'promptstyle', 'sampler', 'sweep'])
    assert.equal(items.pairs.filter((p: any) => p.kind === 'sweep').length, 2)
    assert.ok(existsSync(join(dir, 'sweep-key.json')))
    assert.ok(!readFileSync(join(dir, 'sweep-key.json'), 'utf8').match(/[0-9a-f]{32}/), 'no token in the sweep key')
  })

  it('content checks come from the readings, and second looks repeat a grid with new tokens', () => {
    // Content checks from the readings.
    assert.ok(items.checks.length >= 2)
    assert.ok(items.checks.every((c: any) => c.rating !== 'general'))
    // Second looks.
    const seconds = items.sets.filter((s: any) => s.secondOf)
    assert.ok(seconds.length >= 1)
    for (const s of seconds) for (const g of s.grids) {
      assert.ok(g.repeats)
      const orig = items.sets.flatMap((x: any) => x.grids).find((y: any) => y.itemId === g.repeats)
      assert.equal(sealed.gridOf[orig.itemId].contestant, sealed.gridOf[g.itemId].contestant)
      assert.notDeepEqual(g.tiles ?? g.rows, orig.tiles ?? orig.rows, 'new tokens')
    }

  })

  it('is judged through the queue to the end, then aggregated and reported', { timeout: 120_000 }, () => {
    // Judge everything through the queue, as the phone would.
    const jdir = join(dir, 'judging')
    let now = 1_800_000_000_000
    let session = 's0'
    let guard = 0
    const stepFor = (c: string) => ({ noobai: 4, zbase: 4, klein: 3, qwenEdit: 5, 'chain.draft': 5, 'chain.faces': 2 } as Record<string, number>)[c] ?? 3
    for (;;) {
      if (guard++ > 400) throw new Error('queue does not end')
      const st = reduceEvents(readEvents(jdir))
      const a = nextItem(items, st, { judge: 'me', session, now })
      if (a.record.length) appendEvents(jdir, a.record, { allowServerKinds: true })
      if (a.done) break
      if (a.breakDue) { session = 's' + guard; now += 5 * 60_000; continue }
      const base = { v: 1, id: randomUUID(), at: now, judge: 'me', session, device: { w: 400, h: 800, dpr: 2 }, run: RUN, item: a.item.itemId, dwellMs: 8000 + (guard % 7) * 1000 }
      let e: Record<string, unknown>
      if (a.type === 'grid') {
        const c = sealed.gridOf[a.item.itemId].contestant
        const second = a.set.second ? { card: a.set.second, step: 3 } : undefined
        e = { ...base, kind: 'score', value: { step: stepFor(c), fail: c === 'klein' ? [0] : [], best: null, chips: [], recognised: c === 'noobai', ...(second ? { second } : {}) } }
      } else if (a.type === 'pair') e = { ...base, kind: 'pair', value: { winner: (guard % 3) as 0 | 1 | 2 } }
      else if (a.type === 'pick') e = { ...base, kind: 'pick', value: { letter: a.item.letters[0] } }
      else e = { ...base, kind: 'content', value: { agree: 'no' } }
      const r = appendEvents(jdir, [e])
      assert.equal(r.accepted, 1, JSON.stringify(r.rejected))
      now += 40_000
    }
    const st = reduceEvents(readEvents(jdir))
    assert.equal(remaining(items, st, 'me', now), 0)
    const sweep = writeSweep(env, RUN)
    assert.ok(sweep && sweep.models.zbase && sweep.sampler.zbase)

    const f = aggregate({ items, sealed, events: readEvents(jdir), readings: [...readReadings(env).values()], ledger: readLedger(dir), revealedAt: now })
    const cell = (c: string, b: string) => f.cells.find((x: any) => x.contestant === c && x.block === b)
    assert.equal(cell('klein', 'detail').score, null)
    assert.match(cell('klein', 'detail').na, /./, 'klein has no face pass: N/A with a reason')
    assert.equal(cell('klein', 'reference').score, null)
    assert.equal(cell('noobai', 'following').score, 4, 'second looks stay out of the means')
    assert.equal(cell('chain.draft', 'following').score, 5)
    assert.equal(cell('klein', 'following').failShare, 0.333, 'one Failed of the three pictures made')
    assert.equal(cell('noobai', 'following').recognisedShare, 1)
    assert.equal(cell('noobai', 'variation').score, 3, 'the second card lands on its own block')
    assert.ok(f.cost.byModel.zbase.warmN > 0 && f.cost.byModel.zbase.perStepSource === 'comfy')
    assert.equal(f.cost.byModel.noobai.perStepSource, 'estimated')
    assert.ok(f.content.byModel.noobai.removed === 1)
    // Content, as the picture reader rated it and after your checks (every check here was answered No).
    const rows = Object.values(f.content.byModel) as any[]
    assert.ok(rows.some((r) => r.raw.band !== null && r.raw.band < 5 && r.checked.band === 5), JSON.stringify(f.content.byModel))
    for (const r of rows) assert.ok(r.checked.general >= r.raw.general)
    assert.equal(rows.reduce((a, r) => a + r.checks.no, 0), items.checks.length)
    assert.equal(f.reliability.klein.refused, 1)
    assert.ok(f.judge.n >= 1)
    assert.equal(f.judge.agreeExact, 1)
    assert.ok(f.chains.find((c: any) => c.id === 'chain.draft').change[0].delta === 1)
    const csv = toCsv(f).trim().split('\n')
    assert.equal(csv.length - 1, f.contestants.length * f.blocks.length, 'one row per contestant and block')
    const html = renderReport(f)
    assert.match(html, /Capability map/)
    assert.match(html, /N\/A/)
  })
})
