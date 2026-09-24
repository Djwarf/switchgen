/**
 * Judging (TEST PLAN, D, without pictures): the event log and its reduction,
 * tie-break pairs, the queue's order, breaks and look-throughs, the sweep's
 * verdicts, content and cost bands, aggregation with N/A, vetoes and tiers,
 * and the report and CSV. Folders are temp folders.
 */
import assert from 'node:assert/strict'
import { appendFileSync } from 'node:fs'
import { join } from 'node:path'
import { afterAll, it as test } from 'vitest'
import * as Events from '../judge/events.ts'
import { MAX_TIE_PAIRS, tieBreakPairs } from '../judge/pairs.ts'
import * as Queue from '../judge/queue.ts'
import * as Aggregate from '../judge/aggregate.ts'
import * as Report from '../judge/report.ts'
import { type Loose, removeTemp, tempDir } from './helpers.ts'

const { answerOf, appendEvents, eventProblem, readEvents, reduceEvents } = Events as Loose<typeof Events>
const { breakDue, homeNearest, nextItem, remaining, sweepVerdicts } = Queue as Loose<typeof Queue>
const { aggregate, contentBand, costBand } = Aggregate as Loose<typeof Aggregate>
const { renderReport, toCsv } = Report as Loose<typeof Report>

afterAll(removeTemp)


let n = 0
const ev = (o: Record<string, unknown>) => ({ v: 1, id: `e${++n}`, at: 1_000_000 + n * 10_000, judge: 'me', session: 's1', device: { w: 400, h: 800, dpr: 2 }, run: 'r1', dwellMs: 6000, ...o })
const score = (item: string, step: number, extra: Record<string, unknown> = {}) => ev({ item, kind: 'score', value: { step, fail: [], best: null, chips: [], recognised: false, ...extra } })

test('reduceEvents: latest wins, undo, duplicate id, skip twice, afterReveal', () => {
  const a = score('g1', 3)
  const b = score('g1', 4)
  const u = ev({ item: 'g1', kind: 'undo', value: { target: b.id } })
  const dup = { ...a }
  const s1 = ev({ item: 'g2', kind: 'skip' })
  const s2 = ev({ item: 'g2', kind: 'skip' })
  const rv = ev({ item: 'study', kind: 'reveal', judge: 'server', session: 'server' })
  const late = score('g3', 5)
  const st = reduceEvents([a, b, u, dup, s1, s2, rv, late])
  assert.equal(st.duplicates, 1)
  assert.equal(answerOf(st, 'me', 'g1')?.event.value.step, 3, 'undo brings back the earlier score')
  assert.equal(answerOf(st, 'me', 'g2')?.kind, 'skip')
  assert.equal([...st.answers.get('me').values()].filter((x: { kind: string }) => x.kind === 'skip').length, 1)
  assert.equal(answerOf(st, 'me', 'g3'), undefined, 'after-reveal answers are left out by default')
  assert.equal(answerOf(st, 'me', 'g3', true)?.afterReveal, true)
  // latest by time, not line order
  const x1 = score('g9', 2); const x2 = score('g9', 5)
  const st2 = reduceEvents([x2, x1])
  assert.equal(answerOf(st2, 'me', 'g9')?.event.value.step, 5)
})

test('appendEvents dedupes by id, rejects server kinds from the phone, stamps afterReveal', () => {
  const dir = tempDir('lab-ev-')
  const a = score('g1', 3)
  assert.deepEqual(appendEvents(dir, [a, a]).accepted, 1)
  const r = appendEvents(dir, [a, ev({ item: 's', kind: 'pairs-made', value: { pairs: [] } }), { ...score('g2', 9) }])
  assert.equal(r.duplicate, 1)
  assert.equal(r.rejected.length, 2)
  appendEvents(dir, [{ ...score('g3', 2), afterReveal: false }], { afterReveal: true })
  // The phone's own afterReveal is never trusted: only the server marks it.
  appendEvents(dir, [{ ...score('g4', 3), afterReveal: true }])
  const all = readEvents(dir)
  assert.equal(all.length, 3)
  assert.equal(all[1].afterReveal, true)
  assert.equal(all[2].afterReveal, undefined)
  assert.equal(eventProblem({ ...score('x', 3), value: { step: 3, fail: 'no' } }), 'bad fail list')
})

test('tieBreakPairs: deterministic, 2→1, 3→3, 4→4, capped at 6', () => {
  assert.deepEqual(tieBreakPairs('s', ['a']), [])
  assert.equal(tieBreakPairs('s', ['a', 'b']).length, 1)
  assert.equal(tieBreakPairs('s', ['a', 'b', 'c']).length, 3)
  assert.equal(tieBreakPairs('s', ['a', 'b', 'c', 'd']).length, 4)
  assert.equal(tieBreakPairs('s', ['a', 'b', 'c', 'd', 'e']).length, 5)
  assert.equal(MAX_TIE_PAIRS, 6)
  assert.equal(tieBreakPairs('s', 'abcdefghij'.split('')).length, 6)
  assert.deepEqual(tieBreakPairs('set1', ['c', 'a', 'b', 'd']), tieBreakPairs('set1', ['d', 'b', 'a', 'c']))
  const three = tieBreakPairs('q', ['a', 'b', 'c']).map((p: string[]) => p.slice().sort().join(''))
  assert.deepEqual(three.sort(), ['ab', 'ac', 'bc'])
  // every grid in a 4-cycle appears twice
  const four = tieBreakPairs('q', ['a', 'b', 'c', 'd']).flat()
  for (const x of 'abcd') assert.equal(four.filter((y: string) => y === x).length, 2)
})

const items = {
  v: 1, run: 'r1', sealedSha: 'x',
  sets: [
    { setId: 'S1', order: 0, block: 'following', card: 'pf@1', mode: 'scale', closeLook: false, brief: { task: 't1' }, grids: [
      { itemId: 'A1', letter: 'B', pos: 0, tiles: ['t1', 't2', 't3', 't4'] },
      { itemId: 'A2', letter: 'K', pos: 1, tiles: ['t5', 't6', 't7', 't8'] },
      { itemId: 'A3', letter: 'R', pos: 2, tiles: ['t9', 'ta', 'tb', 'tc'] } ] },
    { setId: 'S2', order: 1, block: 'photo', card: 'pr@1', mode: 'scale', closeLook: true, brief: { task: 't2' }, grids: [
      { itemId: 'B1', letter: 'C', pos: 0, tiles: ['u1', 'u2', 'u3', 'u4'] },
      { itemId: 'B2', letter: 'D', pos: 1, tiles: ['u5', 'u6', 'u7', 'u8'] } ] },
    { setId: 'S9', order: 2, block: 'following', card: 'pf@1', mode: 'scale', closeLook: false, brief: { task: 't1' }, secondOf: 'S1', grids: [
      { itemId: 'Z1', letter: 'M', pos: 0, tiles: ['v1', 'v2', 'v3', 'v4'], repeats: 'A2' } ] },
  ],
  pairs: [{ itemId: 'P1', setId: null, kind: 'sweep', a: { tiles: ['w1'] }, b: { tiles: ['w2'] }, question: 'q' }],
  checks: [{ itemId: 'C1', token: 'k1', rating: 'sensitive', task: 'tt' }],
}

test('queue order: grids, ties, pick, then setless pairs, checks, second looks last', () => {
  const log: Record<string, unknown>[] = []
  const seen: string[] = []
  let t = 5_000_000
  for (let i = 0; i < 20; i++) {
    const st = reduceEvents(log as never)
    const a = nextItem(items as never, st, { judge: 'me', session: 's' + Math.floor(i / 50), now: t })
    for (const r of a.record) log.push(r)
    if (a.done) break
    assert.equal((a.item as { repeats?: string }).repeats, undefined, 'the phone never sees the repeat link')
    assert.equal((a.set as { secondOf?: string } | null)?.secondOf, undefined)
    seen.push(`${a.type}:${a.item.itemId}`)
    const id = a.item.itemId
    const base = { v: 1, id: 'q' + i, at: t, judge: 'me', session: 's0', device: { w: 1, h: 1, dpr: 1 }, run: 'r1', item: id, dwellMs: 7000 }
    if (a.type === 'grid') log.push({ ...base, kind: 'score', value: { step: id === 'A3' ? 3 : 4, fail: [], best: null, chips: [], recognised: false } })
    else if (a.type === 'pair') log.push({ ...base, kind: 'pair', value: { winner: 1 } })
    else if (a.type === 'pick') log.push({ ...base, kind: 'pick', value: { letter: 'B' } })
    else log.push({ ...base, kind: 'content', value: { agree: 'no' } })
    t += 60_000
  }
  assert.deepEqual(seen, ['grid:A1', 'grid:A2', 'grid:A3', 'pair:S1:tie0', 'pick:S1:pick', 'grid:B1', 'grid:B2', 'pair:S2:tie0', 'pick:S2:pick', 'pair:P1', 'check:C1', 'grid:Z1'])
  const st = reduceEvents(log as never)
  assert.equal(remaining(items as never, st, 'me', t), 0)
})

test('a due second look comes between sets', () => {
  const t0 = 10_000_000
  const log = ['A1', 'A2', 'A3'].map((id, i) => ({ v: 1, id: 'x' + i, at: t0 + i, judge: 'me', session: 's', device: { w: 1, h: 1, dpr: 1 }, run: 'r1', item: id, kind: 'score', value: { step: 2 + i, fail: [], best: null, chips: [], recognised: false }, dwellMs: 9000 }))
  let st = reduceEvents(log as never)
  const early = nextItem(items as never, st, { judge: 'me', session: 's', now: t0 + 60_000 })
  assert.equal(early.item.itemId, 'S1:tie0'.replace('tie0', 'pick'), 'one top grid: no tie pair, straight to the pick')
  log.push({ ...log[0], id: 'pk', item: 'S1:pick', kind: 'pick', value: { letter: 'B' } } as never)
  st = reduceEvents([...log, ...early.record] as never)
  const later = nextItem(items as never, st, { judge: 'me', session: 's', now: t0 + 31 * 60_000 })
  assert.equal(later.item.itemId, 'Z1')
})

test('breakDue: 25 items, 20 minutes, fast dwell, 8 same steps; new session resets', () => {
  const mk = (k: number, o: Record<string, unknown>) => Array.from({ length: k }, (_, i) => ({ v: 1, id: `b${JSON.stringify(o)}${i}`, at: 1e6 + i * 1000, judge: 'me', session: 's', device: { w: 1, h: 1, dpr: 1 }, run: 'r', item: 'i' + i, kind: 'score', value: { step: (i % 5) + 1, fail: [], best: null, chips: [], recognised: false }, dwellMs: 9000, ...o }))
  assert.equal(breakDue(reduceEvents(mk(24, {}) as never), 'me', 's', 1e6 + 30_000), null)
  assert.equal(breakDue(reduceEvents(mk(25, {}) as never), 'me', 's', 1e6 + 30_000), 'count')
  assert.equal(breakDue(reduceEvents(mk(3, {}) as never), 'me', 's', 1e6 + 20 * 60_000), 'time')
  assert.equal(breakDue(reduceEvents(mk(6, { dwellMs: 3000 }) as never), 'me', 's', 1e6 + 30_000), 'fast')
  const same = mk(8, {}).map(e => ({ ...e, value: { ...e.value, step: 4 } }))
  assert.equal(breakDue(reduceEvents(same as never), 'me', 's', 1e6 + 30_000), 'same')
  assert.equal(breakDue(reduceEvents(same.slice(1) as never), 'me', 's', 1e6 + 30_000), null)
  assert.equal(breakDue(reduceEvents(mk(25, {}) as never), 'me', 'other', 1e6 + 30_000), null)
})

test('look-through: first meeting, and again after 30 minutes away', () => {
  const st0 = reduceEvents([])
  assert.equal(nextItem(items as never, st0, { judge: 'me', session: 's', now: 1 }).lookThroughDue, true)
  const open = { v: 1, id: 'o1', at: 1000, judge: 'me', session: 's', device: { w: 1, h: 1, dpr: 1 }, run: 'r1', item: 'A1', kind: 'open-set', value: { lookThrough: false }, dwellMs: 0 }
  const sc = { ...open, id: 'o2', at: 2000, kind: 'score', value: { step: 3, fail: [], best: null, chips: [], recognised: false } }
  const st = reduceEvents([open, sc] as never)
  assert.equal(nextItem(items as never, st, { judge: 'me', session: 's', now: 3000 }).lookThroughDue, false)
  assert.equal(nextItem(items as never, st, { judge: 'me', session: 's', now: 2000 + 30 * 60_000 }).lookThroughDue, true)
})

test('sweep verdicts at their thresholds, numbers only', () => {
  const key = { v: 1, run: 'cal', study: 'st', homeSteps: { zt: 8, q: 20 }, homeSampler: { nb: 'euler_ancestral/normal' }, timings: { zt: { '8': [4000, 4200], '28': [12000, 12500] } }, pairs: {} as Record<string, unknown> }
  const evs: Record<string, unknown>[] = []
  const add = (id: string, p: Record<string, unknown>, winner: number) => {
    key.pairs[id] = p
    evs.push({ v: 1, id: 'w' + id, at: 1, judge: 'me', session: 's', device: { w: 1, h: 1, dpr: 1 }, run: 'cal', item: id, kind: 'pair', value: { winner }, dwellMs: 1 })
  }
  // zt: 28 wins 1, ties 1, loses 1 against 8 → holds (wins or ties in 2 of 3)
  add('p1', { kind: 'sweep', model: 'zt', slot: 'a', cmp: '8v28', a: '8', b: '28' }, 2)
  add('p2', { kind: 'sweep', model: 'zt', slot: 'b', cmp: '8v28', a: '28', b: '8' }, 0)
  add('p3', { kind: 'sweep', model: 'zt', slot: 'c', cmp: '8v28', a: '28', b: '8' }, 2)
  // q: home 20 → reads 16v28; 16 wins twice → hurts
  add('p4', { kind: 'sweep', model: 'q', slot: 'a', cmp: '16v28', a: '16', b: '28' }, 1)
  add('p5', { kind: 'sweep', model: 'q', slot: 'b', cmp: '16v28', a: '28', b: '16' }, 2)
  add('p6', { kind: 'sweep', model: 'q', slot: 'c', cmp: '16v28', a: '28', b: '16' }, 1)
  add('p6b', { kind: 'sweep', model: 'q', slot: 'c', cmp: '8v28', a: '28', b: '8' }, 1)
  // 40 wins all three for q
  add('p7', { kind: 'sweep', model: 'q', slot: 'a', cmp: '28v40', a: '40', b: '28' }, 1)
  add('p8', { kind: 'sweep', model: 'q', slot: 'b', cmp: '28v40', a: '28', b: '40' }, 2)
  add('p9', { kind: 'sweep', model: 'q', slot: 'c', cmp: '28v40', a: '40', b: '28' }, 1)
  add('s1', { kind: 'sampler', model: 'nb', slot: 'a', cmp: 'home-v-common', a: 'home', b: 'common' }, 1)
  add('s2', { kind: 'sampler', model: 'nb', slot: 'b', cmp: 'home-v-common', a: 'common', b: 'home' }, 2)
  assert.equal(sweepVerdicts(key as never, reduceEvents(evs.slice(0, 3) as never)), null, 'not before every pair is answered')
  const out = sweepVerdicts(key as never, reduceEvents(evs as never))
  assert.equal(out.models.zt.verdict, 'holds')
  assert.equal(out.models.zt.against, '8v28')
  assert.equal(out.models.q.against, '16v28')
  assert.equal(out.models.q.verdict, 'hurts')
  assert.match(out.models.q.offer, /@home steps \(20\)/)
  assert.equal(out.models.q.fortyWinsAll, true)
  assert.equal(out.sampler.nb.verdict, 'home wins')
  assert.ok(out.models.zt.secondsPerStep > 0.35 && out.models.zt.secondsPerStep < 0.45)
  assert.ok(!JSON.stringify(out).match(/[0-9a-f]{32}/), 'no token in sweep.json')
  assert.equal(homeNearest(20, [8, 16, 28, 40]), 16)
  assert.equal(homeNearest(8, [8, 16, 28, 40]), 8)
})

test('content and cost bands at their thresholds', () => {
  const b = (n: number, s: number, q: number, e: number) => contentBand({ n, sensitive: s, questionable: q, explicit: e })
  assert.equal(b(40, 0, 0, 0), 5)
  assert.equal(b(40, 2, 0, 0), 4)
  assert.equal(b(40, 3, 0, 0), 3)
  assert.equal(b(40, 0, 1, 0), 3)
  assert.equal(b(40, 0, 0, 1), 2)
  assert.equal(b(40, 0, 5, 0), 2)
  assert.equal(b(40, 0, 0, 5), 1)
  assert.equal(b(0, 0, 0, 0), null)
  assert.deepEqual([9.9, 10, 19.9, 20, 39.9, 40, 90, 90.1].map(costBand), [5, 4, 4, 3, 3, 2, 2, 1])
})

test('a torn last line is closed off before the next event', async () => {
  const dir = tempDir('lab-torn-')
  appendEvents(dir, [score('g1', 3)])
  appendFileSync(join(dir, 'events.jsonl'), '{"v":1,"id":"torn","at":1,')
  appendEvents(dir, [score('g2', 4)])
  const all = readEvents(dir)
  assert.deepEqual(all.map((e: { item: string }) => e.item), ['g1', 'g2'])
})

// ------------------------------------------------------------ aggregate --


const cellInfo = (id: string, model: string, o: Record<string, unknown> = {}) => ({ cellId: id, contestant: model, model, chain: null, chainStep: null, file: model + '.safetensors', familyId: 'f', slot: 'photo.kitchen', set: 'photo.kitchen', block: 'photo', op: 't2i', seed: 1, steps: 28, sampler: 'euler', scheduler: 'simple', status: 'made', why: null, rel: 'x', durationMs: 10_000, cold: false, cached: false, jobId: null, promptId: null, innocent: true, madeIn: 'r', ...o })
const grid = (id: string) => ({ itemId: id, letter: id, pos: 0, tiles: ['a', 'b', 'c', 'd'] })
const aggItems = {
  v: 1, run: 'r', sealedSha: 'abc',
  sets: [
    { setId: 'P', order: 0, block: 'photo', card: 'pr@1', mode: 'scale', closeLook: true, brief: { task: 't' }, grids: [grid('pa'), grid('pb'), grid('pc'), grid('pd')] },
    { setId: 'A', order: 1, block: 'anatomy', card: 'an@1', mode: 'scale', closeLook: true, brief: { task: 't' }, grids: [grid('aa'), grid('ab')] },
  ],
  pairs: [], checks: [],
}
const sealed = {
  v: 1, run: 'r', study: 'st', sealedAt: 0, suites: [], tokens: {}, letters: {}, pairOf: {}, checkOf: {},
  gridOf: { pa: { contestant: 'm1', setId: 'P', cells: [] }, pb: { contestant: 'm2', setId: 'P', cells: [] }, pc: { contestant: 'm3', setId: 'P', cells: [] }, pd: { contestant: 'm4', setId: 'P', cells: [] }, aa: { contestant: 'm1', setId: 'A', cells: [] }, ab: { contestant: 'm2', setId: 'A', cells: [] } },
  sets: { P: { block: 'photo', card: 'pr@1', mode: 'scale', slots: ['photo.kitchen'], context: false }, A: { block: 'anatomy', card: 'an@1', mode: 'scale', slots: ['anatomy.hands'], context: false } },
  slots: { 'photo.kitchen': { block: 'photo', innocent: true, measuredOnly: false, suite: 's' }, 'anatomy.hands': { block: 'anatomy', innocent: true, measuredOnly: false, suite: 's' } },
  cells: {
    c1: cellInfo('c1', 'm1'), c2: cellInfo('c2', 'm1', { durationMs: 50_000, cold: true }), c3: cellInfo('c3', 'm1', { durationMs: 1_000, cached: true }), c4: cellInfo('c4', 'm1', { durationMs: 12_000 }),
    c5: cellInfo('c5', 'm2', { durationMs: 30_000 }), c6: cellInfo('c6', 'm3'), c7: cellInfo('c7', 'm4'),
  },
  contestants: { m1: { id: 'm1', kind: 'model', homeSteps: 28 }, m2: { id: 'm2', kind: 'model' }, m3: { id: 'm3', kind: 'model' }, m4: { id: 'm4', kind: 'model' } },
  na: [{ slot: 'anatomy.hands', model: 'm3', reason: 'no hands in the app' }],
  notMade: [],
}
let m = 0
const e = (item: string, kind: string, value: unknown) => ({ v: 1, id: `x${++m}`, at: 1000 + m, judge: 'me', session: 's', device: { w: 1, h: 1, dpr: 1 }, run: 'r', item, kind, value, dwellMs: 5000 })
const sc = (item: string, step: number) => e(item, 'score', { step, fail: [], best: null, chips: [], recognised: false })

test('aggregate: N/A never a 1, veto on a must-have, tiers with split pairs, warm-only cost', () => {
  const events = [
    sc('pa', 4), sc('pb', 4), sc('pc', 3), sc('pd', 1),
    e('P', 'pairs-made', { pairs: [['pa', 'pb']] }),
    e('P:tie0', 'pair', { winner: 0 }),
    sc('aa', 5), sc('ab', 2),
  ]
  const usecases = [{ id: 'u', name: 'U', weights: { photo: 2, anatomy: 1 }, mustHave: ['anatomy'] }]
  const f = aggregate({ items: aggItems, sealed, events, readings: [], ledger: [], usecases })
  const cell = (c: string, b: string) => f.cells.find((x: any) => x.contestant === c && x.block === b)
  assert.equal(cell('m3', 'anatomy').score, null)
  assert.equal(cell('m3', 'anatomy').na, 'no hands in the app')
  assert.equal(cell('m1', 'photo').tier, 1)
  assert.equal(cell('m2', 'photo').tier, 1, 'a tied pair keeps them together')
  assert.equal(cell('m3', 'photo').tier, 1, 'within one step and no pair: about the same')
  assert.equal(cell('m4', 'photo').tier, 2, 'two steps below')
  const u = f.usecases[0].ranking
  const by = (c: string) => u.find((r: any) => r.contestant === c)
  assert.equal(by('m1').score, 4.33)
  assert.match(by('m2').vetoed, /anatomy scored 2/)
  assert.match(by('m3').vetoed, /not applicable/)
  assert.deepEqual(by('m4').notTested, ['anatomy'])
  assert.equal(u[0].contestant, 'm1')
  assert.equal(f.cost.byModel.m1.warmMedianS, 11, 'median of 10 s and 12 s; the cold and cached ones are left out')
  assert.equal(f.cost.byModel.m1.coldMedianS, 50)
  assert.equal(f.cost.byModel.m1.cachedLeftOut, 1)
  assert.equal(f.cost.byModel.m1.band, 4)
  assert.equal(f.cost.byModel.m2.band, 3)
  // a clear pair win splits the tier
  const f2 = aggregate({ items: aggItems, sealed, events: [...events.slice(0, 5), e('P:tie0', 'pair', { winner: 1 }), ...events.slice(6)], readings: [], ledger: [], usecases })
  const c2 = (c: string) => f2.cells.find((x: any) => x.contestant === c && x.block === 'photo')
  assert.deepEqual([c2('m1').tier, c2('m2').tier, c2('m3').tier], [1, 2, 2])
  assert.deepEqual(c2('m1').pairs, { won: 1, lost: 0, tied: 0 })
  const html = renderReport(f)
  // N/A is hatched in the capability map with its reason, never a score.
  assert.match(html, /<td class="na"[^>]*title="no hands in the app"><span class="naw">N\/A<\/span>/)
  assert.match(html, /left out: anatomy scored 2/)
  const csv = toCsv(f).trim().split('\n')
  assert.equal(csv.length, 1 + 4 * f.blocks.length)
  assert.ok(csv[0].startsWith('contestant,kind,block,score'))
})

test('aggregate: a second look stays out of the means and is read as the judge\'s steadiness', () => {
  const items2 = { ...aggItems, sets: [...aggItems.sets, { setId: 'P2', order: 2, block: 'photo', card: 'pr@1', mode: 'scale', closeLook: true, brief: { task: 't' }, secondOf: 'P', grids: [{ ...grid('pa2'), repeats: 'pa' }] }] }
  const sealed2 = { ...sealed, gridOf: { ...sealed.gridOf, pa2: { contestant: 'm1', setId: 'P2', cells: [] } }, sets: { ...sealed.sets, P2: { ...sealed.sets.P, secondOf: 'P' } } }
  const usecases = [{ id: 'u', name: 'U', weights: { photo: 2, anatomy: 1 }, mustHave: ['anatomy'] }]
  const f = aggregate({ items: items2, sealed: sealed2, events: [sc('pa', 4), sc('pb', 4), sc('pc', 3), sc('pd', 1), sc('pa2', 1), sc('aa', 5), sc('ab', 2)], readings: [], ledger: [], usecases })
  const m1 = f.cells.find((x: any) => x.contestant === 'm1' && x.block === 'photo')
  assert.equal(m1.score, 4, 'the second look (1) is not averaged in')
  assert.equal(m1.n, 1)
  assert.equal(f.judge.n, 1)
  assert.equal(f.judge.agreeExact, 0)
  assert.equal(f.judge.agreeWithin1, 0)
})

// ------------------------------------------- a study's runs, joined --

test('reduceEvents tracks the reveal run by run, so a study\'s logs can be joined one after another', () => {
  const r1 = score('g1', 3)
  const r1Reveal = ev({ item: 'st', kind: 'reveal', judge: 'server', session: 'server' })
  const r2 = ev({ run: 'r2', item: 'h1', kind: 'score', value: { step: 4, fail: [], best: null, chips: [], recognised: false } })
  const r2Reveal = ev({ run: 'r2', item: 'st', kind: 'reveal', judge: 'server', session: 'server' })
  const r1Late = score('g2', 5)
  const st = reduceEvents([r1, r1Reveal, r2, r2Reveal, r1Late])
  const byId = new Map<string, { afterReveal?: boolean }>(st.events.map((x: { id: string; afterReveal?: boolean }) => [x.id, x]))
  assert.equal(byId.size, 5)
  assert.equal(byId.get(r2.id)?.afterReveal, undefined, 'the second run\'s judging came before its own reveal')
  assert.equal(byId.get(r1Late.id)?.afterReveal, true)
  assert.equal(byId.get(r1.id)?.afterReveal, undefined)
  assert.equal(st.revealedAt, r1Reveal.at)
})

// Two runs of one study, one set each: what writeReport hands aggregate.
const runItems = (run: string, p: string) => ({
  v: 1, run, sealedSha: 'abc' + run, pairs: [], checks: [],
  sets: [{ setId: p, order: 0, block: 'photo', card: 'pr@1', mode: 'scale', closeLook: true, brief: { task: 't' }, grids: [grid(p + 'a'), grid(p + 'b')] }],
})
const runSealed = (run: string, p: string, cells: Record<string, unknown>) => ({
  v: 1, run, study: 'st', sealedAt: 0, suites: [], tokens: {}, letters: {}, pairOf: {}, checkOf: {},
  gridOf: { [p + 'a']: { contestant: 'm1', setId: p, cells: [] }, [p + 'b']: { contestant: 'm2', setId: p, cells: [] } },
  sets: { [p]: { block: 'photo', card: 'pr@1', mode: 'scale', slots: ['photo.kitchen'], context: false } },
  slots: { 'photo.kitchen': { block: 'photo', innocent: true, measuredOnly: false, suite: 's' } },
  cells,
  contestants: { m1: { id: 'm1', kind: 'model' }, m2: { id: 'm2', kind: 'model' } },
  na: [], notMade: [],
})
let k = 0
const re = (run: string, item: string, kind: string, value: unknown, session = 's') =>
  ({ v: 1, id: `y${++k}`, at: 1000 + k, judge: kind === 'reveal' ? 'server' : 'me', session: kind === 'reveal' ? 'server' : session, device: { w: 1, h: 1, dpr: 1 }, run, item, kind, value, dwellMs: 5000 })
const rs = (run: string, item: string, step: number, session = 's') => re(run, item, 'score', { step, fail: [], best: null, chips: [], recognised: false }, session)
const twoRuns = { items: [runItems('r1', 'P'), runItems('r2', 'Q')], sealed: [runSealed('r1', 'P', { c1: cellInfo('c1', 'm1'), c2: cellInfo('c2', 'm2') }), runSealed('r2', 'Q', { c3: cellInfo('c3', 'm1'), c4: cellInfo('c4', 'm2') })], readings: [], ledger: [], usecases: [] }
// Each run's log, in the order writeReport joins them: its judging, then the reveal written into it.
const judged = [
  rs('r1', 'Pa', 4, 's1'), rs('r1', 'Pb', 2, 's1'), re('r1', 'st', 'reveal', { early: false, remaining: 0 }),
  rs('r2', 'Qa', 5, 's2'), rs('r2', 'Qb', 3, 's2'), re('r2', 'st', 'reveal', { early: false, remaining: 0 }),
]

test('aggregate over a study\'s joined logs keeps every run\'s judging before its own reveal', () => {
  const f = aggregate({ ...twoRuns, events: judged })
  const cell = (c: string) => f.cells.find((x: any) => x.contestant === c && x.block === 'photo')
  assert.equal(f.afterReveal, 0)
  assert.equal(cell('m1').n, 2, 'both runs\' scores')
  assert.equal(cell('m1').score, 4.5)
  assert.equal(cell('m2').score, 2.5)
})

test('answers after the reveal are counted apart, change no mean, and bring no sitting into the judge\'s count', () => {
  const f = aggregate({ ...twoRuns, events: judged })
  const late = aggregate({ ...twoRuns, events: [...judged, rs('r1', 'Pa', 1, 's9'), rs('r2', 'Qa', 1, 's9')] })
  const cell = (x: any, c: string) => x.cells.find((y: any) => y.contestant === c && y.block === 'photo')
  assert.equal(late.afterReveal, 2)
  assert.equal(cell(late, 'm1').score, cell(f, 'm1').score)
  assert.equal(cell(late, 'm1').n, 2)
  assert.equal(late.judge.sessions, f.judge.sessions, 'a sitting that only came after the reveal is not counted')
})

test('pictures removed at the reveal by the judge count beside the content band, once each', () => {
  // c1 was sealed as made and removed at the reveal (a ledger entry only);
  // c5 was removed by the quarantine rule before the seal (sealed as removed, and a ledger entry).
  const sealedR = { ...sealed, cells: { ...sealed.cells, c5: cellInfo('c5', 'm2', { status: 'removed' }) } }
  const ledger = [{ t: 'removed', at: 1, cell: 'c1', why: 'quarantine' }, { t: 'removed', at: 1, cell: 'c5', why: 'quarantine' }]
  const f = aggregate({ items: aggItems, sealed: sealedR, events: [sc('pa', 4), sc('pb', 4), sc('pc', 3), sc('pd', 1)], readings: [], ledger, usecases: [] })
  assert.equal(f.content.byModel.m1.removed, 1)
  assert.equal(f.reliability.m1.removed, 1)
  assert.equal(f.content.byModel.m2.removed, 1)
  assert.equal(f.reliability.m2.removed, 1)
})

test('the report says the lighthouse set got one step for both shapes, only where that set ran', () => {
  const f = aggregate({ items: aggItems, sealed, events: [sc('pa', 4), sc('pb', 4), sc('pc', 3), sc('pd', 1)], readings: [], ledger: [], usecases: [] })
  assert.ok(!f.blocks.includes('shapes'))
  assert.doesNotMatch(renderReport(f), /Where the lighthouse set ran/)
  assert.match(renderReport({ ...f, blocks: [...f.blocks, 'shapes'] }), /Where the lighthouse set ran, each grid got one step for wide and tall together, set by the weaker shape/)
})
