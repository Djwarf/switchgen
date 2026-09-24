/**
 * What the phone shows next, worked out from items.json and the judging log.
 *
 * The order, per judge:
 * 1. set by set, in the sealed order: every grid of the set, then its
 *    tie-break pairs (made once its grids are all answered), its chain pairs
 *    and "your pick";
 * 2. between sets, any second look that is due (its first showing at least
 *    30 minutes ago);
 * 3. the pairs that belong to no set (step sweep, sampler, prompt style);
 * 4. the content checks;
 * 5. last, any second look not yet due, after one short break card.
 *
 * {@link nextItem} is pure: it never writes. When it fixes a set's tie-break
 * pairs it returns the 'pairs-made' server event in `record`, which the caller
 * appends before answering, so the pairs stay fixed if a score changes later.
 *
 * Nothing here reads sealed.json. The step sweep's verdicts (sweep.json) are
 * worked out from sweep-key.json, which names models and step counts but holds
 * no picture token, letter or item-to-grid mapping of any scored set.
 */
import { randomUUID } from 'node:crypto'
import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { runDir, type LabEnv } from '../core/env.ts'
import { answerOf, atOf, readEvents, reduceEvents, type JudgeEvent, type JudgeState, type PairValue, type ScoreValue } from './events.ts'
import { PAIR_QUESTIONS, pickItemId, tieBreakPairs, tieItemId, type PairKind } from './pairs.ts'
import type { Brief, CheckItem, Grid, ItemSet, Items, PairItem, PairSide } from './seal.ts'

export const SECOND_LOOK_AFTER_MS = 30 * 60_000
export const LOOK_AGAIN_AFTER_MS = 30 * 60_000
export const BREAK_AFTER_ITEMS = 25
export const BREAK_AFTER_MS = 20 * 60_000
export const FAST_DWELL_MS = 4_000
export const SAME_STEPS = 8

/** Why a break card is due. The phone words each one. */
export type BreakWhy = 'count' | 'time' | 'fast' | 'same' | 'wait'

export type NextGrid = { type: 'grid'; itemId: string; setId: string; letter: string; pos: number; tiles?: (string | null)[]; rows?: [string | null, string | null][] }
export type NextPair = { type: 'pair'; itemId: string; setId: string | null; kind: PairKind; a: PairSide; b: PairSide; question: string }
export type NextPick = { type: 'pick'; itemId: string; setId: string; letters: string[] }
export type NextCheck = { type: 'check'; itemId: string; token: string; rating: string }
export type NextItemView = NextGrid | NextPair | NextPick | NextCheck

/** A set as the phone sees it: no link from a second look back to its first showing. */
export type SetView = Omit<ItemSet, 'grids' | 'secondOf'> & { grids: Omit<Grid, 'repeats'>[] }

export type NextAnswer =
  | { done: true; progress: { done: number; total: number }; record: JudgeEvent[] }
  | {
      done: false
      type: NextItemView['type']
      item: NextItemView
      set: SetView | null
      brief: Brief | null
      lookThroughDue: boolean
      breakDue: BreakWhy | null
      /** A plain sentence for the break card, when one is due. */
      breakText: string | null
      progress: { done: number; total: number }
      /** Server events to append before answering (a set's tie-break pairs). */
      record: JudgeEvent[]
    }

export const BREAK_TEXT: Record<BreakWhy, string> = {
  count: 'You have answered 25 items in a row.',
  time: 'You have been scoring for 20 minutes.',
  fast: 'Your recent answers came quickly, under 4 seconds each.',
  same: 'Your last 8 steps were all the same.',
  wait: 'A few grids come back for a second look. They work best after a short pause.',
}

type Unit =
  | { type: 'grid'; itemId: string; set: ItemSet; grid: Grid }
  | { type: 'pair'; itemId: string; set: ItemSet | null; pair: PairItem }
  | { type: 'pick'; itemId: string; set: ItemSet }
  | { type: 'check'; itemId: string; check: CheckItem }

/** Strip the fields the phone must not see from a set. */
export function setView(set: ItemSet): SetView {
  const { grids, secondOf: _secondOf, ...rest } = set
  void _secondOf
  return { ...rest, grids: grids.map(({ repeats: _r, ...g }) => { void _r; return g }) }
}

const sortedSets = (items: Items) => items.sets.slice().sort((a, b) => a.order - b.order)

/** The step a judge gave a grid, or null (skipped, unanswered or undone). */
export function stepOf(state: JudgeState, judge: string, itemId: string, includeAfterReveal = true): number | null {
  const a = answerOf(state, judge, itemId, includeAfterReveal)
  if (!a || a.kind !== 'score') return null
  const v = a.event.value as ScoreValue | undefined
  return typeof v?.step === 'number' ? v.step : null
}

function side(grid: Grid): PairSide {
  const tiles = grid.tiles ?? (grid.rows ?? []).map(r => r[1])
  return grid.rows ? { letter: grid.letter, tiles, rows: grid.rows } : { letter: grid.letter, tiles }
}

/** A set's tie-break pairs as items, from the fixed pairs. */
export function tiePairItems(set: ItemSet, pairs: [string, string][]): PairItem[] {
  const byId = new Map(set.grids.map(g => [g.itemId, g]))
  const out: PairItem[] = []
  pairs.forEach(([x, y], k) => {
    const a = byId.get(x)
    const b = byId.get(y)
    if (!a || !b) return
    out.push({ itemId: tieItemId(set.setId, k), setId: set.setId, kind: 'tie', a: side(a), b: side(b), question: PAIR_QUESTIONS.block })
  })
  return out
}

/** The grids of a set that share its top step, for one judge. */
export function topGrids(set: ItemSet, state: JudgeState, judge: string): string[] {
  let top = 0
  const steps = new Map<string, number>()
  for (const g of set.grids) {
    const s = stepOf(state, judge, g.itemId)
    if (s === null) continue
    steps.set(g.itemId, s)
    if (s > top) top = s
  }
  return [...steps].filter(([, s]) => s === top).map(([id]) => id)
}

type Plan = {
  units: Unit[]
  record: JudgeEvent[]
  /** Second-look grids by when they fall due. */
  seconds: { unit: Unit; dueAt: number | null }[]
}

const answered = (state: JudgeState, judge: string, itemId: string) => !!answerOf(state, judge, itemId, true)

/** Every unit for this judge, in order (second looks apart), with any pairs-made events to record. */
function plan(items: Items, state: JudgeState, judge: string, session: string, now: number): Plan {
  const units: Unit[] = []
  const record: JudgeEvent[] = []
  const all = sortedSets(items)
  const firsts = all.filter(s => !s.secondOf)
  for (const set of firsts) {
    for (const grid of set.grids.slice().sort((a, b) => a.pos - b.pos)) units.push({ type: 'grid', itemId: grid.itemId, set, grid })
    if (set.grids.every(g => answered(state, judge, g.itemId))) {
      let pairs = state.pairsMade.get(set.setId)
      if (!pairs) {
        pairs = tieBreakPairs(set.setId, topGrids(set, state, judge))
        record.push({
          v: 1, id: randomUUID(), at: now, judge, session, device: { w: 0, h: 0, dpr: 1 },
          run: items.run, item: set.setId, kind: 'pairs-made', value: { pairs }, dwellMs: 0,
        })
      }
      for (const p of tiePairItems(set, pairs)) units.push({ type: 'pair', itemId: p.itemId, set, pair: p })
    }
    for (const p of items.pairs) if (p.setId === set.setId) units.push({ type: 'pair', itemId: p.itemId, set, pair: p })
    if (set.grids.length >= 2) units.push({ type: 'pick', itemId: pickItemId(set.setId), set })
  }
  for (const p of items.pairs) {
    if (p.setId !== null) continue
    units.push({ type: 'pair', itemId: p.itemId, set: null, pair: p })
  }
  const seconds: Plan['seconds'] = []
  for (const set of all.filter(s => s.secondOf)) {
    for (const grid of set.grids.slice().sort((a, b) => a.pos - b.pos)) {
      const first = grid.repeats ? answerOf(state, judge, grid.repeats, true) : undefined
      seconds.push({ unit: { type: 'grid', itemId: grid.itemId, set, grid }, dueAt: first ? first.at + SECOND_LOOK_AFTER_MS : null })
    }
  }
  for (const check of items.checks) units.push({ type: 'check', itemId: check.itemId, check })
  return { units, record, seconds }
}

function view(u: Unit): NextItemView {
  switch (u.type) {
    case 'grid': {
      const g = u.grid
      const out: NextGrid = { type: 'grid', itemId: g.itemId, setId: u.set.setId, letter: g.letter, pos: g.pos }
      if (g.tiles) out.tiles = g.tiles
      if (g.rows) out.rows = g.rows
      return out
    }
    case 'pair':
      return { type: 'pair', itemId: u.itemId, setId: u.pair.setId, kind: u.pair.kind, a: u.pair.a, b: u.pair.b, question: u.pair.question }
    case 'pick':
      return { type: 'pick', itemId: u.itemId, setId: u.set.setId, letters: u.set.grids.map(g => g.letter).sort() }
    case 'check':
      return { type: 'check', itemId: u.itemId, token: u.check.token, rating: u.check.rating }
  }
}

function setOfUnit(u: Unit): ItemSet | null {
  return u.type === 'check' ? null : u.set
}

function briefOfUnit(u: Unit): Brief | null {
  if (u.type === 'check') return { task: u.check.task ?? '' }
  if (u.type === 'pair' && !u.set) return u.pair.brief ?? null
  return u.set?.brief ?? null
}

/** This judge's own events, oldest first, leaving out server events. */
function judgeEvents(state: JudgeState, judge: string): JudgeEvent[] {
  return state.events
    .filter(e => e.judge === judge && e.kind !== 'pairs-made' && e.kind !== 'reveal')
    .sort((a, b) => atOf(a) - atOf(b))
}

/**
 * Whether a break card is due in this session: 25 answers, 20 minutes, a
 * median dwell under 4 s over the recent answers, or 8 identical steps in a
 * row. The phone starts a new session when the judge continues after a break,
 * which starts every count again.
 */
export function breakDue(state: JudgeState, judge: string, session: string, now: number): BreakWhy | null {
  const mine = judgeEvents(state, judge).filter(e => e.session === session)
  if (!mine.length) return null
  const answers = mine.filter(e => ['score', 'pair', 'pick', 'content'].includes(e.kind) && !state.undone.has(e.id))
  const scores = answers.filter(e => e.kind === 'score')
  const last8 = scores.slice(-SAME_STEPS).map(e => (e.value as ScoreValue | undefined)?.step)
  if (last8.length === SAME_STEPS && last8.every(s => s === last8[0])) return 'same'
  const recent = answers.slice(-10).map(e => e.dwellMs).sort((a, b) => a - b)
  if (recent.length >= 6) {
    const mid = recent.length / 2
    const median = recent.length % 2 ? recent[Math.floor(mid)] : (recent[mid - 1] + recent[mid]) / 2
    if (median < FAST_DWELL_MS) return 'fast'
  }
  if (answers.length >= BREAK_AFTER_ITEMS) return 'count'
  if (now - atOf(mine[0]) >= BREAK_AFTER_MS) return 'time'
  return null
}

function setIdsOfItems(items: Items): Map<string, string> {
  const m = new Map<string, string>()
  for (const s of items.sets) {
    m.set(s.setId, s.setId)
    for (const g of s.grids) m.set(g.itemId, s.setId)
  }
  return m
}

/**
 * Whether the look-through should be offered before this unit: the first time
 * the judge meets the set, or when they come back to it after 30 minutes away.
 */
function lookThrough(items: Items, state: JudgeState, judge: string, u: Unit, now: number): boolean {
  if (u.type !== 'grid') return false
  const setId = u.set.setId
  const bySet = setIdsOfItems(items)
  const mine = judgeEvents(state, judge)
  const opened = mine.some(e => e.kind === 'open-set' && bySet.get(e.item) === setId)
  if (!opened) return true
  const last = mine.length ? atOf(mine[mine.length - 1]) : now
  return now - last >= LOOK_AGAIN_AFTER_MS
}

/**
 * The next thing for this judge. `now` is epoch ms. Returns {done:true} once
 * every item (including the tie-break pairs, picks, second looks and content
 * checks) has an answer or a skip.
 */
export function nextItem(items: Items, state: JudgeState, opts: { judge: string; session: string; now: number }): NextAnswer {
  const { judge, session, now } = opts
  const p = plan(items, state, judge, session, now)
  const allUnits = [...p.units, ...p.seconds.map(s => s.unit)]
  const done = allUnits.filter(u => answered(state, judge, u.itemId)).length
  const progress = { done, total: allUnits.length }
  const due = p.seconds.filter(s => !answered(state, judge, s.unit.itemId) && s.dueAt !== null && s.dueAt <= now)

  const answerWith = (u: Unit, why: BreakWhy | null): NextAnswer => {
    const set = setOfUnit(u)
    const b = why ?? breakDue(state, judge, session, now)
    return {
      done: false,
      type: u.type,
      item: view(u),
      set: set ? setView(set) : null,
      brief: briefOfUnit(u),
      lookThroughDue: lookThrough(items, state, judge, u, now),
      breakDue: b,
      breakText: b ? BREAK_TEXT[b] : null,
      progress,
      record: p.record,
    }
  }

  const open = p.units.filter(u => !answered(state, judge, u.itemId))
  const first = open[0]
  if (first) {
    // A due second look goes in between sets: before a set none of whose items is answered yet.
    const set = setOfUnit(first)
    const fresh = set && !set.secondOf && p.units.filter(u => setOfUnit(u) === set).every(u => !answered(state, judge, u.itemId))
    if (due.length && (fresh || first.type === 'check' || (first.type === 'pair' && !first.set))) return answerWith(due[0].unit, null)
    return answerWith(first, null)
  }
  if (due.length) return answerWith(due[0].unit, null)
  const waiting = p.seconds.filter(s => !answered(state, judge, s.unit.itemId))
  if (waiting.length) {
    // Nothing else is left: one break card, then the second looks, due or not.
    const answeredThisSession = state.events.some(e => e.judge === judge && e.session === session && e.kind !== 'pairs-made' && e.kind !== 'reveal')
    return answerWith(waiting[0].unit, answeredThisSession ? 'wait' : null)
  }
  return { done: true, progress, record: p.record }
}

/** How many items this judge has still to answer or skip (for the reveal gate). */
export function remaining(items: Items, state: JudgeState, judge: string, now = Date.now()): number {
  const p = plan(items, state, judge, 'remaining', now)
  const all = [...p.units, ...p.seconds.map(s => s.unit)]
  // A set's tie-break pairs are counted once its grids are all answered and the pairs exist.
  return all.filter(u => !answered(state, judge, u.itemId)).length
}

/** Every judge who has answered anything in this run, or the given default. */
export function judgesIn(state: JudgeState, fallback: string[] = []): string[] {
  const out = new Set<string>(fallback)
  for (const j of state.answers.keys()) out.add(j)
  return [...out]
}

// ------------------------------------------------------------------ sweep --

/**
 * sweep-key.json: written by the seal for a run with step sweep or sampler
 * pairs. It names models, prompts and step counts, and holds the measured
 * times, but no picture token and no letter: it is safe to read before the
 * reveal, and only the sweep's numbers are ever shown from it.
 */
export type SweepKey = {
  v: 1
  run: string
  study: string
  /** pair item id → what it compares. `a` and `b` say which arm each side is. */
  pairs: Record<string, {
    kind: 'sweep' | 'sampler'
    model: string
    slot: string
    /** '8v28', '16v28', '28v40', or 'home-v-common' for the sampler. */
    cmp: string
    a: string
    b: string
  }>
  /** model → step count → warm ComfyUI durations in ms (cold and cached pictures left out). */
  timings: Record<string, Record<string, number[]>>
  /** model → home step count. */
  homeSteps: Record<string, number>
  /** model → its home sampler, 'name/scheduler'. */
  homeSampler: Record<string, string>
}

export type SweepJson = {
  v: 1
  run: string
  study: string
  legend: string
  models: Record<string, {
    prompts: Record<string, Record<string, number | null>>
    verdict: 'holds' | 'hurts' | 'mixed'
    /** The comparison the verdict reads: the count nearest the model's home steps against 28. */
    against: string
    fortyWinsAll: boolean
    homeSteps: number
    secondsPerStep: number | null
    /** Seconds per picture by step count, the median of warm pictures, measured by ComfyUI. */
    secondsAt: Record<string, number | null>
    offer: string | null
  }>
  sampler: Record<string, {
    prompts: Record<string, number | null>
    home: string
    verdict: 'home wins' | 'common holds' | 'mixed'
    offer: string | null
  }>
}

export const SWEEP_KEY_FILE = 'sweep-key.json'
export const SWEEP_FILE = 'sweep.json'

function median(xs: number[]): number | null {
  if (!xs.length) return null
  const s = xs.slice().sort((a, b) => a - b)
  const m = s.length / 2
  return s.length % 2 ? s[Math.floor(m)] : (s[m - 1] + s[m]) / 2
}

/** Seconds per step: the slope of a straight line through the warm times by step count (ComfyUI's own timing). */
export function secondsPerStep(timings: Record<string, number[]>): number | null {
  const pts: [number, number][] = []
  for (const [steps, ms] of Object.entries(timings)) for (const v of ms) if (v > 0) pts.push([Number(steps), v / 1000])
  const xs = new Set(pts.map(p => p[0]))
  if (!pts.length) return null
  if (xs.size < 2) {
    const [x] = [...xs]
    const m = median(pts.map(p => p[1]))
    return m === null || !x ? null : Math.round((m / x) * 100) / 100
  }
  const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length
  const my = pts.reduce((s, p) => s + p[1], 0) / pts.length
  let num = 0
  let den = 0
  for (const [x, y] of pts) {
    num += (x - mx) * (y - my)
    den += (x - mx) ** 2
  }
  return den > 0 ? Math.round((num / den) * 100) / 100 : null
}

/** The step count, other than 28, nearest the model's home count (the lower one on a tie). */
export function homeNearest(home: number, counts: number[]): number | null {
  const others = counts.filter(c => c !== 28).sort((a, b) => a - b)
  let best: number | null = null
  for (const c of others) if (best === null || Math.abs(c - home) < Math.abs(best - home)) best = c
  return best
}

/**
 * The sweep's verdicts, once every sweep and sampler pair has an answer or a
 * skip from `judge` (any judge when omitted); null before that.
 *
 * - Each answer is scored from the 28-step side: 1 when 28 was better, 0 for
 *   can't tell, -1 when the other count was better, null when skipped. For the
 *   sampler, from the home sampler's side.
 * - "28 holds" when 28 wins or ties in at least 2 of the model's prompts against
 *   the count nearest its home steps; "28 hurts" when it loses in at least 2;
 *   otherwise mixed. Hurts offers "<model> @home steps" as an extra contestant.
 * - The home sampler "wins" when it wins every prompt (at least 2), which
 *   offers "<model> @home sampler".
 */
export function sweepVerdicts(key: SweepKey, state: JudgeState, judge?: string): SweepJson | null {
  const ids = Object.keys(key.pairs)
  if (!ids.length) return null
  const judges = judge ? [judge] : [...state.answers.keys()]
  const answerFor = (id: string) => {
    for (const j of judges) {
      const a = answerOf(state, j, id, true)
      if (a) return a
    }
    return undefined
  }
  if (!ids.every(id => answerFor(id))) return null
  const out: SweepJson = {
    v: 1,
    run: key.run,
    study: key.study,
    legend: 'Each number is the answer seen from the 28-step side (for the sampler, from the home sampler): 1 better, 0 cannot tell, -1 worse, null skipped. Times are ComfyUI execution times of warm pictures.',
    models: {},
    sampler: {},
  }
  const sweepCounts = new Map<string, Set<number>>()
  for (const [id, p] of Object.entries(key.pairs)) {
    const a = answerFor(id)
    const w = a && a.kind === 'pair' ? ((a.event.value as PairValue).winner) : null
    if (p.kind === 'sweep') {
      const m = (out.models[p.model] ??= {
        prompts: {}, verdict: 'mixed', against: '', fortyWinsAll: false,
        homeSteps: key.homeSteps[p.model] ?? 28, secondsPerStep: null, secondsAt: {}, offer: null,
      })
      const winnerArm = w === 1 ? p.a : w === 2 ? p.b : null
      const score = w === null ? null : w === 0 ? 0 : winnerArm === '28' ? 1 : -1
      ;(m.prompts[p.slot] ??= {})[p.cmp] = score
      const set = sweepCounts.get(p.model) ?? new Set<number>()
      for (const n of p.cmp.split('v').map(Number)) if (Number.isFinite(n)) set.add(n)
      sweepCounts.set(p.model, set)
    } else {
      const s = (out.sampler[p.model] ??= { prompts: {}, home: key.homeSampler[p.model] ?? 'home', verdict: 'mixed', offer: null })
      const winnerArm = w === 1 ? p.a : w === 2 ? p.b : null
      s.prompts[p.slot] = w === null ? null : w === 0 ? 0 : winnerArm === 'home' ? 1 : -1
    }
  }
  for (const [model, m] of Object.entries(out.models)) {
    const counts = [...(sweepCounts.get(model) ?? [])]
    const near = homeNearest(m.homeSteps, counts)
    m.against = near === null ? '' : `${Math.min(near, 28)}v${Math.max(near, 28)}`
    const vals = Object.values(m.prompts).map(p => p[m.against]).filter((v): v is number => typeof v === 'number')
    // For '28v40' the number is still seen from 28, so a win is a win either way round.
    const holds = vals.filter(v => v >= 0).length
    const hurts = vals.filter(v => v < 0).length
    m.verdict = holds >= 2 ? 'holds' : hurts >= 2 ? 'hurts' : 'mixed'
    const forty = Object.values(m.prompts).map(p => p['28v40'])
    m.fortyWinsAll = forty.length >= 2 && forty.every(v => v === -1)
    const t = key.timings[model] ?? {}
    m.secondsPerStep = secondsPerStep(t)
    for (const [steps, ms] of Object.entries(t)) {
      const med = median(ms.filter(v => v > 0))
      m.secondsAt[steps] = med === null ? null : Math.round(med / 100) / 10
    }
    m.offer = m.verdict === 'hurts' ? `${model} @home steps (${m.homeSteps})` : null
  }
  for (const [model, s] of Object.entries(out.sampler)) {
    const vals = Object.values(s.prompts)
    const decided = vals.filter((v): v is number => typeof v === 'number')
    if (decided.length >= 2 && decided.length === vals.length && decided.every(v => v === 1)) s.verdict = 'home wins'
    else if (decided.filter(v => v <= 0).length >= Math.max(1, Math.ceil(vals.length / 2))) s.verdict = 'common holds'
    else s.verdict = 'mixed'
    s.offer = s.verdict === 'home wins' ? `${model} @home sampler (${s.home})` : null
  }
  return out
}

/**
 * Work out and write runs/<run>/sweep.json once the run's sweep pairs are all
 * answered. Returns it, or null when there is no sweep or it is not judged yet.
 * sweep.json holds names and numbers only: no token, no letter.
 */
export function writeSweep(env: LabEnv, run: string): SweepJson | null {
  const dir = runDir(env, run)
  const keyFile = join(dir, SWEEP_KEY_FILE)
  if (!existsSync(keyFile)) return null
  const key = JSON.parse(readFileSync(keyFile, 'utf8')) as SweepKey
  const state = reduceEvents(readEvents(join(dir, 'judging')))
  const out = sweepVerdicts(key, state)
  if (!out) return null
  const tmp = join(dir, `.${SWEEP_FILE}.${process.pid}`)
  writeFileSync(tmp, JSON.stringify(out, null, 2) + '\n', { mode: 0o600 })
  renameSync(tmp, join(dir, SWEEP_FILE))
  return out
}

/** Whether a run's sweep and sampler pairs are all judged (true when it has none). */
export function sweepJudged(env: LabEnv, run: string): boolean {
  const dir = runDir(env, run)
  const keyFile = join(dir, SWEEP_KEY_FILE)
  if (!existsSync(keyFile)) return true
  const key = JSON.parse(readFileSync(keyFile, 'utf8')) as SweepKey
  const state = reduceEvents(readEvents(join(dir, 'judging')))
  return Object.keys(key.pairs).every(id => [...state.answers.values()].some(m => m.has(id)))
}
