/**
 * The whole first-pass study (TEST PLAN, D, with the four shipped nights and
 * all 13 models plus the editing model): each night planned with the real
 * planner, its pictures stood in by small PNGs that name their ckpt, sealed,
 * judged through the queue to the end, then aggregated and reported as one
 * study. Photos are generated grey JPEGs, never the user's. The blind copies
 * need ffmpeg, so this file is skipped without it; it takes about a minute
 * and a half.
 */
import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, it } from 'vitest'
import { runDir } from '../core/env.ts'
import { SHIPPED_SUITES, expand } from '../core/cells.ts'
import { planFrom, savePlan } from '../core/plan.ts'
import { addRef, ensureRefCopies, refIndex, setMask } from '../run/refs.ts'
import { appendDoneCell, readDoneCells, readLedger } from '../run/ledger.ts'
import { readReadings } from '../run/reader.ts'
import { sealRun } from '../judge/seal.ts'
import { appendEvents, appendServerEvent, readEvents, reduceEvents } from '../judge/events.ts'
import { nextItem, sweepJudged, writeSweep } from '../judge/queue.ts'
import { aggregate } from '../judge/aggregate.ts'
import { renderReport, toCsv } from '../judge/report.ts'
import { HAS_FFMPEG, greyJpeg, png, removeTemp, tempEnv } from './helpers.ts'

afterAll(removeTemp)

const RUNS: [string, string][] = [['cal-1', 'calibration'], ['core-1', 'first-pass-core'], ['exta-1', 'ext-edit'], ['extb-1', 'ext-range']]

describe.skipIf(!HAS_FFMPEG)('the four shipped nights, sealed, judged and reported as one study', () => {
  const env = tempEnv()
  let items: any[] = []
  let sealed: any[] = []
  const planned: Record<string, { order: number; reused: number }> = {}

  beforeAll(async () => {
    mkdirSync(join(env.outputs, '.lab', 'cells'), { recursive: true })
    addRef(env, greyJpeg(join(env.labDir, 'c.jpg'), 160, 120), 'cat')
    addRef(env, greyJpeg(join(env.labDir, 's.jpg'), 192, 128), 'scene')
    setMask(env, 'scene', { x: 20, y: 20, w: 60, h: 40 })
    ensureRefCopies(env, ['cat', 'scene'])
    let at = 1_700_000_000_000
    for (const [run, suiteId] of RUNS) {
      const suite = SHIPPED_SUITES.find((s) => s.id === suiteId)!
      const x = expand([suite], refIndex(env))
      assert.equal(x.blocked.length, 0, `${run} blocked: ${JSON.stringify(x.blocked[0])}`)
      const plan = planFrom(run, [suite], x, readDoneCells(env))
      savePlan(env, plan, x.graphs)
      planned[run] = { order: plan.order.length, reused: plan.reused.length }
      const dir = runDir(env, run)
      const byId = new Map(plan.cells.map((c) => [c.cellId, c]))
      for (const id of plan.order) {
        const c = byId.get(id)!
        at += 1000
        const name = `${id}_00001_.png`
        const v = parseInt(id.slice(0, 2), 16)
        writeFileSync(join(env.outputs, '.lab', 'cells', name), png(Math.max(8, c.width >> 7), Math.max(8, c.height >> 7), [v, v, v], { text: JSON.stringify({ ckpt_name: c.file }) }))
        const ms = 400 * c.steps + 800
        appendFileSync(join(dir, 'ledger.jsonl'), JSON.stringify({ t: 'ended', at, job: randomUUID(), cell: id, status: 'done', error: null, files: [], primary: { filename: name, subfolder: '.lab/cells', type: 'output' }, promptId: randomUUID(), durationMs: ms, ranAt: at, finishedAt: at, cached: false, cold: false, attempt: 1 }) + '\n')
        appendDoneCell(env, { cellId: id, rel: `.lab/cells/${name}`, durationMs: ms, cold: false, cached: false, finishedAt: at, run })
        if (parseInt(id.slice(-2), 16) < 6) appendFileSync(join(env.labDir, 'readings.jsonl'), JSON.stringify({ v: 1, cellId: id, rel: '', at, rating: 'sensitive', ratings: [], general: [], character: [], quarantined: false }) + '\n')
      }
      await sealRun(env, run, { ffmpeg: 'ffmpeg' })
    }
    items = RUNS.map(([run]) => JSON.parse(readFileSync(join(runDir(env, run), 'items.json'), 'utf8')))
    sealed = RUNS.map(([run]) => JSON.parse(readFileSync(join(runDir(env, run), 'sealed.json'), 'utf8')))
  }, 300_000)

  const slotsOf = (i: number) =>
    Object.fromEntries(items[i].sets.filter((s: any) => !s.secondOf).map((s: any) => [sealed[i].sets[s.setId].slots.join('+'), s.grids.map((g: any) => sealed[i].gridOf[g.itemId].contestant).sort()]))

  it('plans the nights with the counts for 13 models: 272, then 588 new of 676, then 648 new of 664, then 888 new of 944', () => {
    assert.deepEqual(planned, { 'cal-1': { order: 272, reused: 0 }, 'core-1': { order: 588, reused: 88 }, 'exta-1': { order: 648, reused: 16 }, 'extb-1': { order: 888, reused: 56 } })
  })

  it('calibration gives pairs only: 36 sweep pairs and 10 sampler pairs', () => {
    assert.equal(items[0].sets.length, 0)
    assert.equal(items[0].pairs.filter((p: any) => p.kind === 'sweep').length, 36)
    assert.equal(items[0].pairs.filter((p: any) => p.kind === 'sampler').length, 10)
  })

  it('core gives 10 sets of 13 grids, 2 of them before/after, and a second look for about 1 grid in 12', () => {
    const core = items[1].sets.filter((s: any) => !s.secondOf)
    assert.equal(core.length, 10, core.map((s: any) => sealed[1].sets[s.setId].slots.join('+')).join(', '))
    for (const s of core) assert.equal(s.grids.length, 13)
    assert.equal(core.filter((s: any) => s.mode === 'beforeAfter').length, 2)
    assert.equal(items[1].sets.filter((s: any) => s.secondOf).flatMap((s: any) => s.grids).length, Math.round(130 / 12))
  })

  it('ext-edit: the editing model joins the reference and region sets, Klein does not, and the chains land in their sets', () => {
    const ext = slotsOf(2)
    assert.ok(ext['ref.cat.ghibli'].includes('qwenEdit') && !ext['ref.cat.ghibli'].includes('klein'))
    assert.equal(ext['ref.cat.ghibli'].length, 13)
    assert.equal(ext['region.scene'].length, 13)
    assert.deepEqual(ext['character.cat.sofa+character.cat.library'], ['chain.cat-twice', 'qwenEdit'])
    assert.ok(ext['text.bakery'].includes('chain.text-fix') && ext['text.bakery'].includes('noobai'))
    assert.ok(ext['detail.group'].includes('chain.klein-faces'))
    assert.ok(ext['detail.group.face'].includes('chain.klein-faces') && !ext['detail.group.face'].includes('klein'))
    assert.ok(ext['layout.thumbnail'].includes('chain.thumbnail-3'))
    assert.equal(items[2].pairs.filter((p: any) => p.kind === 'chain').length, 6)
    const neg = ext['negative.party.without+negative.party.with']
    assert.deepEqual(neg, ['chroma', 'miaomiaoHarem', 'miaomiaoRealskin', 'noobai', 'oneObsession', 'pony', 'semireal', 'wai', 'zbase'])
  })

  it('ext-range: 14 prompt-style pairs, and the lighthouse set wide against tall with the square on demand', () => {
    const rng = slotsOf(3)
    assert.equal(items[3].pairs.filter((p: any) => p.kind === 'promptstyle').length, 14)
    const shapes = items[3].sets.find((s: any) => sealed[3].sets[s.setId].slots.includes('shapes.lighthouse.wide') && !s.secondOf)
    assert.deepEqual(shapes.brief.conditions, ['wide', 'tall'])
    assert.equal(shapes.grids[0].extra.label, 'square')
    assert.ok(!Object.keys(rng).some((k) => k.startsWith('content.')))
  })

  it('is judged through the queue to the end, then aggregated and reported with N/A reasons, measured costs and steadiness', { timeout: 300_000 }, () => {
    let now = 1_800_000_000_000
    for (const [i, [run]] of RUNS.entries()) {
      const jdir = join(runDir(env, run), 'judging')
      let session = 's0'
      for (let guard = 0; guard < 3000; guard++) {
        const st = reduceEvents(readEvents(jdir))
        const a: any = nextItem(items[i], st, { judge: 'me', session, now })
        if (a.record.length) appendEvents(jdir, a.record, { allowServerKinds: true })
        if (a.done) break
        if (a.breakDue) {
          session = `s${guard}`
          now += 3 * 60_000
          continue
        }
        const base = { v: 1, id: randomUUID(), at: now, judge: 'me', session, device: { w: 1, h: 1, dpr: 1 }, run, item: a.item.itemId, dwellMs: 9000 }
        const c = a.type === 'grid' ? sealed[i].gridOf[a.item.itemId].contestant : ''
        const e =
          a.type === 'grid'
            ? { ...base, kind: 'score', value: { step: 1 + (c.length % 5), fail: [], best: null, chips: [], recognised: false, ...(a.set.second ? { second: { card: a.set.second, step: 2 } } : {}) } }
            : a.type === 'pair'
              ? { ...base, kind: 'pair', value: { winner: guard % 3 } }
              : a.type === 'pick'
                ? { ...base, kind: 'pick', value: { letter: a.item.letters[0] } }
                : { ...base, kind: 'content', value: { agree: guard % 2 ? 'no' : 'yes' } }
        assert.equal(appendEvents(jdir, [e as never]).accepted, 1)
        now += 20_000
      }
      assert.ok((nextItem(items[i], reduceEvents(readEvents(jdir)), { judge: 'me', session, now }) as any).done, `${run} judged`)
    }
    assert.ok(sweepJudged(env, 'cal-1'))
    const sw: any = writeSweep(env, 'cal-1')
    assert.deepEqual(Object.keys(sw.models).sort(), ['klein', 'krea2', 'qwen21', 'zturbo'])
    assert.equal(sw.models.qwen21.against, '16v28')
    // The reveal, as the server writes it: one reveal event in every run's own log.
    for (const [run] of RUNS) {
      appendServerEvent(join(runDir(env, run), 'judging'), { v: 1, id: `reveal-${randomUUID()}`, at: now, judge: 'server', session: 'server', device: { w: 0, h: 0, dpr: 0 }, run, item: 'first-pass', kind: 'reveal', value: { early: false, remaining: 0 }, dwellMs: 0 } as never)
    }
    const f: any = aggregate({
      items, sealed,
      events: RUNS.flatMap(([run]) => readEvents(join(runDir(env, run), 'judging'))),
      readings: [...readReadings(env).values()],
      ledger: RUNS.flatMap(([run]) => readLedger(runDir(env, run))),
      revealedAt: now,
    } as never)
    const cell = (c: string, b: string) => f.cells.find((x: any) => x.contestant === c && x.block === b)
    assert.equal(f.afterReveal, 0, 'every run\'s judging came before its own reveal')
    assert.equal(f.contestants.filter((c: any) => c.kind === 'model').length, 14)
    assert.equal(f.contestants.filter((c: any) => c.kind === 'chain').length, 6)
    assert.match(cell('klein', 'negative').na, /negative/)
    assert.equal(cell('klein', 'negative').score, null)
    assert.match(cell('zturbo', 'negative').na, /guidance off|CFG 1/)
    assert.equal(cell('noobai', 'following').n, 4, 'fruit, negative.words, binding, spatial')
    assert.equal(f.cost.byModel.zturbo.perStepSource, 'comfy', 'the sweep measured Turbo per step')
    assert.equal(f.cost.byModel.zturbo.homeSource, 'comfy', 'Turbo at 8 steps was measured')
    assert.equal(f.cost.byModel.noobai.perStepSource, 'estimated')
    assert.ok(f.judge.n >= 20, `second looks ${f.judge.n}`)
    assert.equal(f.promptStyle.length, 7)
    assert.equal(toCsv(f).trim().split('\n').length - 1, f.contestants.length * f.blocks.length)
    const html = renderReport(f)
    assert.match(html, /Capability map/)
    assert.match(html, /one test per block: a first look, not a verdict/i)
    assert.match(html, /Where the lighthouse set ran, each grid got one step for wide and tall together/)
  })
})
