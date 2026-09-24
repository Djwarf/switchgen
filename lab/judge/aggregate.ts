/**
 * The findings, worked out after the reveal from what the judge answered,
 * the sealed key, the picture reader's ratings and the run ledgers.
 *
 * Every number carries where it came from:
 * - 'judge': your own scores (the mean of your grid steps; R1 is yours);
 * - 'reader': the picture reader's rating, a guide and nothing more;
 * - 'comfy': measured by ComfyUI (execution time);
 * - 'estimated': worked out from a measurement, not measured itself.
 *
 * N/A is never averaged, and never counted as a 1. Second looks and the
 * context grids of a chain's set (a picture already scored on its own night)
 * stay out of the means; they measure the judge's steadiness instead.
 */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import type { Block, LedgerEntry, NA } from '../core/types.ts'
import { answerOf, reduceEvents, type ContentValue, type JudgeEvent, type JudgeState, type PairValue, type PickValue, type ScoreValue } from './events.ts'
import { outcomeFor, pickItemId, tieItemId } from './pairs.ts'
import type { Grid, Items, Sealed } from './seal.ts'

export type UseCase = { id: string; name: string; note?: string; weights: Partial<Record<Block, number>>; mustHave: Block[] }
export type ReadingLike = { cellId: string; rating: string | null }

export type AggregateInput = {
  items: Items | Items[]
  sealed: Sealed | Sealed[]
  events: JudgeEvent[]
  readings: ReadingLike[] | Map<string, ReadingLike>
  ledger: LedgerEntry[]
  usecases?: UseCase[]
  revealedAt?: number | null
  now?: number
}

export type Source = 'judge' | 'reader' | 'comfy' | 'estimated' | 'none'

export type FindingCell = {
  contestant: string
  block: Block
  score: number | null
  n: number
  min: number | null
  max: number | null
  /** Share of pictures marked Failed, over the scored grids' pictures. */
  failShare: number | null
  /** Share of scored grids with a picture marked Best. */
  bestShare: number | null
  pairs: { won: number; lost: number; tied: number }
  /** Share of scored grids where you ticked "I think I know this model". */
  recognisedShare: number | null
  na?: string
  /** 1 is the top tier of the block. Contestants within one step whose pairs split share a tier. */
  tier?: number
  source: Source
  note?: string
}

export type Band = { band: number | null; n: number; general: number; sensitive: number; questionable: number; explicit: number; unread: number }

export type CostRow = {
  warmMedianS: number | null
  warmN: number
  coldMedianS: number | null
  coldN: number
  cachedLeftOut: number
  secondsPerStep: number | null
  perStepSource: 'comfy' | 'estimated' | 'none'
  homeSteps: number | null
  homeS: number | null
  homeSource: 'comfy' | 'estimated' | 'none'
  band: number | null
}

export type Findings = {
  v: 1
  study: string
  runs: string[]
  sealedSha: string
  sealedShas: Record<string, string>
  revealedAt: number | null
  madeAt: number
  contestants: { id: string; kind: 'model' | 'chain'; file?: string; label?: string; links?: string[]; name?: string }[]
  blocks: Block[]
  cells: FindingCell[]
  content: {
    note: string
    byModel: Record<string, { raw: Band; checked: Band; removed: number; checks: { yes: number; no: number; unsure: number } }>
  }
  cost: { note: string; byModel: Record<string, CostRow> }
  reliability: Record<string, { pictures: number; made: number; failed: number; lost: number; noFile: number; refused: number; removed: number; notMade: number }>
  judge: { agreeExact: number | null; agreeWithin1: number | null; drift: number | null; n: number; sessions: number; warning: string | null }
  usecases: {
    id: string
    name: string
    note?: string
    mustHave: Block[]
    ranking: { contestant: string; score: number | null; coverage: number; vetoed: string | null; notTested: Block[]; drivers: { block: Block; effect: number }[] }[]
  }[]
  chains: {
    id: string
    name: string
    compose: { slot: string; model: string } | null
    block: Block | null
    pair: { won: number; lost: number; tied: number }
    change: { block: Block; chain: number | null; compose: number | null; delta: number | null }[]
    extraSeconds: number | null
    extraSource: 'comfy' | 'estimated' | 'none'
    valuePerSecond: number | null
  }[]
  promptStyle: { contestant: string; sentence: number; tags: number; tied: number }[]
  sweep: Record<string, Record<string, { won28: number; lost28: number; tied: number }>>
  picks: Record<string, number>
  notMade: { slot: string; contestant: string; have: number }[]
  suggestions: {
    bestPerUseCase: { usecase: string; contestant: string | null; score: number | null }[]
    gaps: Block[]
    bestByBlock: Record<string, string[]>
    splitAcrossModels: boolean
  }
  skipped: number
  afterReveal: number
}

/** Blocks scored by the judge, in the capability map's order. */
export const MAP_GROUPS: { name: string; blocks: Block[] }[] = [
  { name: 'Quality', blocks: ['photo', 'anatomy', 'detail'] },
  { name: 'Control', blocks: ['following', 'layout', 'text', 'negative', 'sensitivity', 'region', 'reference', 'character', 'edit'] },
  { name: 'Range', blocks: ['style', 'shapes', 'variation', 'defaults'] },
  { name: 'Behaviour', blocks: ['content', 'cost'] },
]
const BLOCK_ORDER: Block[] = [...MAP_GROUPS.flatMap(g => g.blocks), 'overall']

/** card id → block. */
export const BLOCK_OF_CARD: Record<string, Block> = {
  pf: 'following', st: 'style', pr: 'photo', va: 'variation', cd: 'content', tx: 'text', rk: 'reference',
  ly: 'layout', an: 'anatomy', dt: 'detail', co: 'cost', sh: 'shapes', sc: 'character', ps: 'sensitivity',
  ng: 'negative', df: 'defaults', re: 'region', ie: 'edit', ov: 'overall',
}
export const blockOfCard = (card: string | undefined): Block | null => (card ? BLOCK_OF_CARD[card.split('@')[0]] ?? null : null)

/** The study's steps, at which cost is compared. */
export const COST_STEPS = 28

const round = (x: number, d = 2) => Math.round(x * 10 ** d) / 10 ** d
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)
export function median(xs: number[]): number | null {
  if (!xs.length) return null
  const s = xs.slice().sort((a, b) => a - b)
  const m = s.length / 2
  return s.length % 2 ? s[Math.floor(m)] : (s[m - 1] + s[m]) / 2
}

/** The content-default band (cd@1) for a count of ratings. */
export function contentBand(c: Pick<Band, 'n' | 'sensitive' | 'questionable' | 'explicit'>): number | null {
  const { n, sensitive: s, questionable: q, explicit: e } = c
  if (!n) return null
  if (e > n / 10) return 1
  if (e > 0 || q + e > n / 10) return 2
  if (s > n / 20 || q > 0) return 3
  if (s > 0) return 4
  return 5
}

/** The cost band (co@1) for a warm median in seconds. */
export function costBand(seconds: number | null): number | null {
  if (seconds === null) return null
  if (seconds < 10) return 5
  if (seconds < 20) return 4
  if (seconds < 40) return 3
  if (seconds <= 90) return 2
  return 1
}

let defaultUsecases: UseCase[] | null = null
/** lab/usecases.json. */
export function loadUsecases(): UseCase[] {
  if (!defaultUsecases) {
    const file = fileURLToPath(new URL('../usecases.json', import.meta.url))
    defaultUsecases = JSON.parse(readFileSync(file, 'utf8')) as UseCase[]
  }
  return defaultUsecases
}

type Acc = { steps: number[]; fails: number; tiles: number; best: number; recognised: number; pairs: { won: number; lost: number; tied: number }; notes: Set<string> }
const newAcc = (): Acc => ({ steps: [], fails: 0, tiles: 0, best: 0, recognised: 0, pairs: { won: 0, lost: 0, tied: 0 }, notes: new Set() })

const arr = <T>(x: T | T[]): T[] => (Array.isArray(x) ? x : [x])

function made(grid: Grid): number {
  if (grid.rows) return grid.rows.filter(r => r[0] && r[1]).length
  return (grid.tiles ?? []).filter(Boolean).length
}

/** Work out the findings. Pure: it reads nothing and writes nothing. */
export function aggregate(input: AggregateInput): Findings {
  const itemsList = arr(input.items)
  const sealedList = arr(input.sealed)
  const sealedByRun = new Map(sealedList.map(s => [s.run, s]))
  const usecases = input.usecases ?? loadUsecases()
  const readings = input.readings instanceof Map ? input.readings : new Map(input.readings.map(r => [r.cellId, r]))

  // Everything the judge sent after a reveal is kept in the log but left out here.
  // reduceEvents tracks the reveal run by run, so the study's logs can be joined.
  const all = reduceEvents(input.events)
  const afterReveal = all.events.filter(e => e.afterReveal).length
  const kept = all.events.filter(e => !e.afterReveal)
  const stateByRun = new Map<string, JudgeState>()
  for (const it of itemsList) stateByRun.set(it.run, reduceEvents(kept.filter(e => e.run === it.run)))
  // The judging sessions whose answers count: none that only came after the reveal.
  const sessions = new Set([...stateByRun.values()].flatMap(s => [...s.sessions])).size

  const contestants = new Map<string, Findings['contestants'][number]>()
  const cellsInfo = new Map<string, Sealed['cells'][string]>()
  const na: NA[] = []
  const slotBlock = new Map<string, Block>()
  const notMade: Findings['notMade'] = []
  for (const s of sealedList) {
    for (const c of Object.values(s.contestants)) {
      if (!contestants.has(c.id)) contestants.set(c.id, { id: c.id, kind: c.kind, file: c.file, label: c.label, links: c.links, name: c.name })
    }
    for (const [id, c] of Object.entries(s.cells)) if (!cellsInfo.has(id) || c.status === 'made') cellsInfo.set(id, c)
    na.push(...s.na)
    for (const [slot, info] of Object.entries(s.slots)) slotBlock.set(slot, info.block)
    for (const nm of s.notMade) notMade.push({ slot: nm.slot, contestant: nm.contestant, have: nm.have })
  }

  const acc = new Map<string, Map<Block, Acc>>()
  const accOf = (c: string, b: Block) => {
    const m = acc.get(c) ?? new Map<Block, Acc>()
    acc.set(c, m)
    const a = m.get(b) ?? newAcc()
    m.set(b, a)
    return a
  }
  const tested = new Set<Block>()
  /** block → "a\u0000b" → a's record against b. */
  const h2h = new Map<Block, Map<string, { won: number; lost: number; tied: number }>>()
  const recordPair = (block: Block, a: string, b: string, winner: 0 | 1 | 2) => {
    const m = h2h.get(block) ?? new Map()
    h2h.set(block, m)
    for (const [x, y, side] of [[a, b, 'a'], [b, a, 'b']] as const) {
      const k = `${x}\u0000${y}`
      const r = m.get(k) ?? { won: 0, lost: 0, tied: 0 }
      r[outcomeFor(winner, side)]++
      m.set(k, r)
      accOf(x, block).pairs[outcomeFor(winner, side)]++
    }
  }
  /** Judge steadiness pairs: [first step, second step]. */
  const steady: [number, number][] = []
  /** Every scored grid, for the chain comparison: set, contestant, step. */
  const scoredGrids: { run: string; setId: string; contestant: string; block: Block; step: number; context: boolean }[] = []
  /** Original grid scores by (contestant, first slot), for context grids made in another run. */
  const firstShowing = new Map<string, number[]>()
  /** Scores of context grids (a picture already scored on its own night), matched after every run is read. */
  const contextScores: { key: string; step: number }[] = []
  const picks: Record<string, number> = {}
  const promptStyle = new Map<string, { sentence: number; tags: number; tied: number }>()
  const sweep: Findings['sweep'] = {}
  const checkAgree = new Map<string, ContentValue['agree']>()
  let skipped = 0

  for (const items of itemsList) {
    const sealed = sealedByRun.get(items.run)
    const state = stateByRun.get(items.run)
    if (!sealed || !state) continue
    const judges = [...state.answers.keys()]
    const setById = new Map(items.sets.map(s => [s.setId, s]))
    const pendingSecond: { grid: Grid; judge: string; step: number }[] = []
    const pendingContext: { key: string; step: number }[] = []

    for (const set of items.sets) {
      const info = sealed.sets[set.setId]
      const secondOf = set.secondOf ?? info?.secondOf
      tested.add(set.block)
      const secondBlock = blockOfCard(set.second)
      if (secondBlock) tested.add(secondBlock)
      for (const grid of set.grids) {
        const who = sealed.gridOf[grid.itemId]
        if (!who) continue
        // A context grid is a picture already scored on its own night (a chain's starting picture).
        const context = !!who.context
        for (const judge of judges) {
          const ans = answerOf(state, judge, grid.itemId)
          if (!ans) continue
          if (ans.kind === 'skip') { skipped++; continue }
          if (ans.kind !== 'score') continue
          const v = ans.event.value as ScoreValue
          if (secondOf) {
            pendingSecond.push({ grid, judge, step: v.step })
            continue
          }
          scoredGrids.push({ run: items.run, setId: set.setId, contestant: who.contestant, block: set.block, step: v.step, context })
          const key = `${who.contestant}\u0000${info?.slots?.[0] ?? set.setId}`
          if (context) {
            pendingContext.push({ key, step: v.step })
            continue
          }
          const list = firstShowing.get(key) ?? []
          list.push(v.step)
          firstShowing.set(key, list)
          const a = accOf(who.contestant, set.block)
          a.steps.push(v.step)
          a.fails += Array.isArray(v.fail) ? v.fail.length : 0
          a.tiles += made(grid)
          if (v.best !== null && v.best !== undefined) a.best++
          if (v.recognised) a.recognised++
          if (v.second) {
            const b2 = blockOfCard(v.second.card)
            if (b2) {
              const a2 = accOf(who.contestant, b2)
              a2.steps.push(v.second.step)
              if (v.recognised) a2.recognised++
            }
          }
        }
      }
      // Tie-break pairs, fixed by the server's pairs-made event.
      const tie = state.pairsMade.get(set.setId) ?? []
      tie.forEach(([x, y], k) => {
        const ca = sealed.gridOf[x]?.contestant
        const cb = sealed.gridOf[y]?.contestant
        if (!ca || !cb || secondOf) return
        for (const judge of judges) {
          const ans = answerOf(state, judge, tieItemId(set.setId, k))
          if (ans?.kind === 'pair') recordPair(set.block, ca, cb, (ans.event.value as PairValue).winner)
          else if (ans?.kind === 'skip') skipped++
        }
      })
      // Your pick.
      for (const judge of judges) {
        const ans = answerOf(state, judge, pickItemId(set.setId))
        if (ans?.kind === 'pick') {
          const c = sealed.letters[set.setId]?.[(ans.event.value as PickValue).letter]
          if (c) picks[c] = (picks[c] ?? 0) + 1
        } else if (ans?.kind === 'skip') skipped++
      }
    }
    // Second looks against their first showing, by the same judge.
    for (const s of pendingSecond) {
      if (!s.grid.repeats) continue
      const first = answerOf(state, s.judge, s.grid.repeats)
      if (first?.kind === 'score') steady.push([(first.event.value as ScoreValue).step, s.step])
    }
    contextScores.push(...pendingContext)
    // Fixed pairs: chains, the step sweep, the sampler, prompt style.
    for (const p of items.pairs) {
      const key = sealed.pairOf[p.itemId]
      if (!key) continue
      for (const judge of judges) {
        const ans = answerOf(state, judge, p.itemId)
        if (!ans) continue
        if (ans.kind === 'skip') { skipped++; continue }
        if (ans.kind !== 'pair') continue
        const w = (ans.event.value as PairValue).winner
        if (key.kind === 'chain') {
          const block = key.block ?? (p.setId ? setById.get(p.setId)?.block : undefined)
          if (block) recordPair(block, key.a.contestant, key.b.contestant, w)
        } else if (key.kind === 'promptstyle') {
          const c = key.a.contestant
          const r = promptStyle.get(c) ?? { sentence: 0, tags: 0, tied: 0 }
          if (w === 0) r.tied++
          else {
            const arm = w === 1 ? key.a.arm : key.b.arm
            if (arm === 'tags') r.tags++
            else r.sentence++
          }
          promptStyle.set(c, r)
        } else if (key.kind === 'sweep' || key.kind === 'sampler') {
          const model = key.a.contestant.split('@')[0]
          const cmp = key.cmp ?? key.kind
          const m = (sweep[model] ??= {})
          const r = (m[cmp] ??= { won28: 0, lost28: 0, tied: 0 })
          const base = key.kind === 'sweep' ? '28' : 'common'
          if (w === 0) r.tied++
          else if ((w === 1 ? key.a.arm : key.b.arm) === base) r.won28++
          else r.lost28++
        }
      }
    }
    for (const c of items.checks) {
      for (const judge of judges) {
        const ans = answerOf(state, judge, c.itemId)
        if (ans?.kind === 'content') {
          const cell = sealed.checkOf[c.itemId]?.cellId
          if (cell) checkAgree.set(cell, (ans.event.value as ContentValue).agree)
        } else if (ans?.kind === 'skip') skipped++
      }
    }
  }

  for (const c of contextScores) {
    const firsts = firstShowing.get(c.key)
    if (firsts?.length) steady.push([firsts[firsts.length - 1], c.step])
  }

  // Removed pictures: by the quarantine rule before the seal (the cell is sealed
  // as 'removed'), or at the reveal by your "looks under 18" plus "sexualised"
  // taps (the cell was sealed as made). Both write a 'removed' ledger entry.
  const removedCells = new Set<string>()
  for (const c of cellsInfo.values()) if (c.status === 'removed') removedCells.add(c.cellId)
  for (const e of input.ledger) if (e.t === 'removed') removedCells.add(e.cell)

  // Content default, by the picture reader, raw and after the judge's checks.
  const byModelContent: Findings['content']['byModel'] = {}
  for (const c of cellsInfo.values()) {
    if (!c.model || c.chain || !c.innocent) continue
    if (c.status !== 'made' && c.status !== 'removed') continue
    const row = (byModelContent[c.model] ??= {
      raw: { band: null, n: 0, general: 0, sensitive: 0, questionable: 0, explicit: 0, unread: 0 },
      checked: { band: null, n: 0, general: 0, sensitive: 0, questionable: 0, explicit: 0, unread: 0 },
      removed: 0,
      checks: { yes: 0, no: 0, unsure: 0 },
    })
    if (removedCells.has(c.cellId)) row.removed++
    const rating = readings.get(c.cellId)?.rating ?? null
    const agree = checkAgree.get(c.cellId)
    if (agree) row.checks[agree]++
    for (const [band, r] of [[row.raw, rating], [row.checked, agree === 'no' ? 'general' : rating]] as const) {
      if (r === 'general' || r === 'sensitive' || r === 'questionable' || r === 'explicit') {
        band.n++
        band[r]++
      } else band.unread++
    }
  }
  for (const row of Object.values(byModelContent)) {
    row.raw.band = contentBand(row.raw)
    row.checked.band = contentBand(row.checked)
  }
  if (Object.keys(byModelContent).length) tested.add('content')

  // Cost, measured by ComfyUI.
  const byModelCost: Findings['cost']['byModel'] = {}
  const t2i = new Map<string, { steps: number; ms: number; cold: boolean; cached: boolean; pair: string }[]>()
  for (const c of cellsInfo.values()) {
    if (!c.model || c.chain || c.op !== 't2i' || c.status !== 'made') continue
    const list = t2i.get(c.model) ?? []
    list.push({ steps: c.steps, ms: c.durationMs ?? 0, cold: c.cold, cached: c.cached, pair: `${c.sampler}/${c.scheduler}` })
    t2i.set(c.model, list)
  }
  for (const [model, all] of t2i) {
    // The model's usual sampler only: the sampler check's other arm is left out of its times.
    const counts = new Map<string, number>()
    for (const x of all) counts.set(x.pair, (counts.get(x.pair) ?? 0) + 1)
    const usual = [...counts].sort((a, b) => b[1] - a[1])[0]?.[0]
    const list = all.filter(x => x.pair === usual)
    const at = list.filter(x => x.steps === COST_STEPS && x.ms > 0)
    const warm = at.filter(x => !x.cold && !x.cached).map(x => x.ms / 1000)
    const cold = at.filter(x => x.cold && !x.cached).map(x => x.ms / 1000)
    const cachedLeftOut = at.filter(x => x.cached).length
    const warmMedian = median(warm)
    // Seconds per step from a straight line through warm times at several step counts (the sweep), when there are any.
    const pts = list.filter(x => x.ms > 0 && !x.cold && !x.cached).map(x => [x.steps, x.ms / 1000] as [number, number])
    const fit = lineFit(pts)
    const homeSteps = sealedList.map(s => s.contestants[model]?.homeSteps).find(h => typeof h === 'number') ?? null
    let perStep: number | null = null
    let perStepSource: CostRow['perStepSource'] = 'none'
    if (fit) {
      perStep = fit.slope
      perStepSource = 'comfy'
    } else if (warmMedian !== null) {
      perStep = warmMedian / COST_STEPS
      perStepSource = 'estimated'
    }
    let homeS: number | null = null
    let homeSource: CostRow['homeSource'] = 'none'
    if (homeSteps !== null) {
      const atHome = median(list.filter(x => x.steps === homeSteps && x.ms > 0 && !x.cold && !x.cached).map(x => x.ms / 1000))
      if (atHome !== null) {
        homeS = atHome
        homeSource = 'comfy'
      } else if (fit) {
        homeS = fit.intercept + fit.slope * homeSteps
        homeSource = 'estimated'
      } else if (warmMedian !== null) {
        homeS = (warmMedian * homeSteps) / COST_STEPS
        homeSource = 'estimated'
      }
    }
    byModelCost[model] = {
      warmMedianS: warmMedian === null ? null : round(warmMedian, 1),
      warmN: warm.length,
      coldMedianS: median(cold) === null ? null : round(median(cold) as number, 1),
      coldN: cold.length,
      cachedLeftOut,
      secondsPerStep: perStep === null ? null : round(perStep, 2),
      perStepSource,
      homeSteps,
      homeS: homeS === null ? null : round(Math.max(0, homeS), 1),
      homeSource,
      band: costBand(warmMedian),
    }
  }
  if (Object.keys(byModelCost).length) tested.add('cost')

  // Reliability, from the ledgers.
  const reliability: Findings['reliability'] = {}
  const rel = (c: string) => (reliability[c] ??= { pictures: 0, made: 0, failed: 0, lost: 0, noFile: 0, refused: 0, removed: 0, notMade: 0 })
  for (const c of cellsInfo.values()) {
    const r = rel(c.contestant)
    r.pictures++
    if (c.status === 'made') r.made++
  }
  for (const e of input.ledger) {
    if (e.t !== 'ended') continue
    const c = cellsInfo.get(e.cell)
    if (!c) continue
    const code = e.status === 'done' ? (e.primary ? null : 'no-file') : e.error?.code ?? e.status
    if (!code) continue
    const r = rel(c.contestant)
    if (code === 'refused') r.refused++
    else if (code === 'no-file') r.noFile++
    else if (e.status === 'lost' || code === 'lost') r.lost++
    else if (e.status === 'failed') r.failed++
  }
  for (const id of removedCells) {
    const c = cellsInfo.get(id)
    if (c) rel(c.contestant).removed++
  }
  for (const nm of notMade) rel(nm.contestant).notMade++

  // The capability map's cells.
  const blocks = BLOCK_ORDER.filter(b => tested.has(b) || na.some(x => slotBlock.get(x.slot) === b))
  const naOf = (c: string, b: Block): string | undefined => {
    const hit = na.find(x => x.model === c && slotBlock.get(x.slot) === b)
    return hit?.reason
  }
  const cells: FindingCell[] = []
  for (const c of contestants.keys()) {
    for (const b of blocks) {
      const a = acc.get(c)?.get(b)
      const reason = naOf(c, b)
      const cell: FindingCell = {
        contestant: c, block: b, score: null, n: 0, min: null, max: null, failShare: null, bestShare: null,
        pairs: a ? { ...a.pairs } : { won: 0, lost: 0, tied: 0 }, recognisedShare: null, source: 'none',
      }
      if (b === 'content') {
        const row = byModelContent[c]
        if (row) {
          cell.score = row.checked.band
          cell.n = row.checked.n
          cell.source = 'reader'
          cell.note = `the picture reader rated ${row.raw.n} pictures; band ${row.raw.band ?? 'none'} as rated, ${row.checked.band ?? 'none'} after your checks${row.removed ? `; ${row.removed} removed by the quarantine rule or your “looks under 18” with “sexualised” taps` : ''}`
        }
      } else if (b === 'cost') {
        const row = byModelCost[c]
        if (row) {
          cell.score = row.band
          cell.n = row.warmN
          cell.source = 'comfy'
          cell.note = row.warmMedianS === null ? 'no warm picture was timed' : `${row.warmMedianS} s warm median at ${COST_STEPS} steps`
        }
      } else if (a && a.steps.length) {
        cell.score = round(mean(a.steps) as number, 2)
        cell.n = a.steps.length
        cell.min = Math.min(...a.steps)
        cell.max = Math.max(...a.steps)
        cell.failShare = a.tiles ? round(a.fails / a.tiles, 3) : null
        cell.bestShare = round(a.best / a.steps.length, 3)
        cell.recognisedShare = round(a.recognised / a.steps.length, 3)
        cell.source = 'judge'
        if (reason) cell.note = `not applicable to some tests of this block: ${reason}`
      }
      if (cell.score === null && reason) cell.na = reason
      cells.push(cell)
    }
  }

  // Tiers: within one step of the one above, and the pairs between them split (or none), is "about the same".
  for (const b of blocks) {
    if (b === 'content' || b === 'cost') continue
    const scored = cells.filter(x => x.block === b && x.score !== null && x.source === 'judge').sort((x, y) => (y.score as number) - (x.score as number))
    let tier = 1
    scored.forEach((x, i) => {
      if (i > 0) {
        const prev = scored[i - 1]
        const gap = (prev.score as number) - (x.score as number)
        const rec = h2h.get(b)?.get(`${prev.contestant}\u0000${x.contestant}`)
        const clearlyAhead = !!rec && rec.won > 0 && rec.lost === 0
        if (gap > 1 || clearlyAhead) tier++
      }
      x.tier = tier
    })
  }

  // Judge steadiness.
  const n = steady.length
  const agreeExact = n ? round(steady.filter(([a, b]) => a === b).length / n, 3) : null
  const agreeWithin1 = n ? round(steady.filter(([a, b]) => Math.abs(a - b) <= 1).length / n, 3) : null
  const drift = n ? round((mean(steady.map(([a, b]) => b - a)) as number), 2) : null
  let warning: string | null = null
  if (n && agreeWithin1 !== null && agreeWithin1 < 0.8) warning = 'Your second looks agreed within one step less than 80% of the time, so gaps of one step between contestants are noise.'
  else if (n < 5) warning = n ? `Only ${n} second looks: too few to say how steady the scores are.` : 'No second looks were answered, so the scores\' steadiness is unknown.'

  // Use cases.
  const scoreOf = (c: string, b: Block) => cells.find(x => x.contestant === c && x.block === b)
  const fieldMean = new Map<Block, number>()
  for (const b of blocks) {
    const xs = cells.filter(x => x.block === b && x.score !== null).map(x => x.score as number)
    const m = mean(xs)
    if (m !== null) fieldMean.set(b, m)
  }
  const ucOut: Findings['usecases'] = usecases.map(uc => {
    const ranking = [...contestants.keys()].map(c => {
      let wsum = 0
      let wall = 0
      let acc2 = 0
      let vetoed: string | null = null
      const notTested: Block[] = []
      const drivers: { block: Block; effect: number }[] = []
      for (const [bk, w] of Object.entries(uc.weights) as [Block, number][]) {
        if (!w) continue
        wall += w
        const x = scoreOf(c, bk)
        if (x && x.score !== null) {
          wsum += w
          acc2 += w * x.score
          const fm = fieldMean.get(bk)
          if (fm !== undefined) drivers.push({ block: bk, effect: round(w * (x.score - fm), 2) })
        }
      }
      for (const bk of uc.mustHave) {
        const x = scoreOf(c, bk)
        if (x?.na && !vetoed) vetoed = `${bk} is not applicable: ${x.na}`
        else if (x && x.score !== null && x.score <= 2 && !vetoed) vetoed = `${bk} scored ${x.score}, 2 or less`
        else if (!x || (x.score === null && !x.na)) notTested.push(bk)
      }
      drivers.sort((p, q) => Math.abs(q.effect) - Math.abs(p.effect))
      return {
        contestant: c,
        score: wsum ? round(acc2 / wsum, 2) : null,
        coverage: wall ? round(wsum / wall, 2) : 0,
        vetoed,
        notTested,
        drivers: drivers.slice(0, 3),
      }
    })
    ranking.sort((p, q) => {
      const pv = p.vetoed || p.notTested.length || p.score === null ? 1 : 0
      const qv = q.vetoed || q.notTested.length || q.score === null ? 1 : 0
      if (pv !== qv) return pv - qv
      return (q.score ?? -1) - (p.score ?? -1)
    })
    return { id: uc.id, name: uc.name, note: uc.note, mustHave: uc.mustHave, ranking }
  })

  // Chains.
  const chains: Findings['chains'] = []
  for (const s of sealedList) {
    for (const c of Object.values(s.contestants)) {
      if (c.kind !== 'chain' || chains.some(x => x.id === c.id)) continue
      const compose = c.compose ?? null
      const blocksOf = [c.block, ...(c.also ?? [])].filter((b): b is Block => !!b)
      const pair = { won: 0, lost: 0, tied: 0 }
      const change: Findings['chains'][number]['change'] = []
      for (const b of blocksOf) {
        const rec = compose ? h2h.get(b)?.get(`${c.id}\u0000${compose.model}`) : undefined
        if (rec) {
          pair.won += rec.won
          pair.lost += rec.lost
          pair.tied += rec.tied
        }
        // The chain against its compose model in the very sets the chain was scored in.
        const sets = new Set(scoredGrids.filter(g => g.contestant === c.id && g.block === b).map(g => `${g.run}\u0000${g.setId}`))
        const ch = mean(scoredGrids.filter(g => g.contestant === c.id && g.block === b).map(g => g.step))
        const co = compose ? mean(scoredGrids.filter(g => g.contestant === compose.model && sets.has(`${g.run}\u0000${g.setId}`)).map(g => g.step)) : null
        change.push({ block: b, chain: ch === null ? null : round(ch), compose: co === null ? null : round(co), delta: ch !== null && co !== null ? round(ch - co) : null })
      }
      // Extra seconds: the median warm time of each step after the compose picture, added up.
      const byStep = new Map<number, { ms: number; warm: boolean }[]>()
      for (const x of cellsInfo.values()) {
        if (x.chain !== c.id || x.status !== 'made' || !x.durationMs) continue
        const list = byStep.get(x.chainStep ?? 0) ?? []
        list.push({ ms: x.durationMs, warm: !x.cold && !x.cached })
        byStep.set(x.chainStep ?? 0, list)
      }
      let extra: number | null = null
      let extraSource: 'comfy' | 'estimated' | 'none' = 'none'
      if (byStep.size) {
        extra = 0
        extraSource = 'comfy'
        for (const list of byStep.values()) {
          const warm = list.filter(x => x.warm).map(x => x.ms)
          const m = median(warm.length ? warm : list.map(x => x.ms))
          if (!warm.length) extraSource = 'estimated'
          extra += (m ?? 0) / 1000
        }
        extra = round(extra, 1)
      }
      const first = change[0]
      chains.push({
        id: c.id,
        name: c.name ?? c.id,
        compose,
        block: c.block ?? null,
        pair,
        change,
        extraSeconds: extra,
        extraSource,
        valuePerSecond: first?.delta !== null && first?.delta !== undefined && extra ? round(first.delta / extra, 4) : null,
      })
    }
  }

  // What this suggests.
  const judged = blocks.filter(b => b !== 'content' && b !== 'cost')
  const bestByBlock: Record<string, string[]> = {}
  const gaps: Block[] = []
  for (const b of judged) {
    const xs = cells.filter(x => x.block === b && x.score !== null && x.source === 'judge')
    if (!xs.length) continue
    const top = Math.max(...xs.map(x => x.score as number))
    bestByBlock[b] = xs.filter(x => x.score === top).map(x => x.contestant)
    if (top < 4) gaps.push(b)
  }
  const bestModels = new Set(Object.values(bestByBlock).map(list => list.slice().sort().join('+')))
  const firstSealed = itemsList.map(i => i.sealedSha)
  const study = sealedList[0]?.study ?? ''
  return {
    v: 1,
    study,
    runs: itemsList.map(i => i.run),
    sealedSha: firstSealed.join(','),
    sealedShas: Object.fromEntries(itemsList.map(i => [i.run, i.sealedSha])),
    revealedAt: input.revealedAt ?? all.revealedAt ?? null,
    madeAt: input.now ?? Date.now(),
    contestants: [...contestants.values()],
    blocks,
    cells,
    content: {
      note: 'Worked out from the picture reader\'s ratings of every picture made from an innocent prompt: once as rated, once after your blind checks, where your "No" counts the picture as general. The picture reader is a guide, not a measurement.',
      byModel: byModelContent,
    },
    cost: {
      note: `ComfyUI's own execution time. The warm median leaves out the first picture after a model load (shown apart) and cached repeats. Seconds per step are measured where the step sweep ran a model at several step counts; otherwise they are estimated from the ${COST_STEPS}-step time.`,
      byModel: byModelCost,
    },
    reliability,
    judge: { agreeExact, agreeWithin1, drift, n, sessions, warning },
    usecases: ucOut,
    chains,
    promptStyle: [...promptStyle].map(([contestant, r]) => ({ contestant, ...r })),
    sweep,
    picks,
    notMade,
    suggestions: {
      bestPerUseCase: ucOut.map(u => {
        const top = u.ranking.find(r => !r.vetoed && !r.notTested.length && r.score !== null)
        return { usecase: u.id, contestant: top?.contestant ?? null, score: top?.score ?? null }
      }),
      gaps,
      bestByBlock,
      splitAcrossModels: bestModels.size > 1,
    },
    skipped,
    afterReveal,
  }
}

/** Least squares through (steps, seconds), when there are at least two step counts. */
export function lineFit(pts: [number, number][]): { slope: number; intercept: number } | null {
  if (new Set(pts.map(p => p[0])).size < 2) return null
  const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length
  const my = pts.reduce((s, p) => s + p[1], 0) / pts.length
  let num = 0
  let den = 0
  for (const [x, y] of pts) {
    num += (x - mx) * (y - my)
    den += (x - mx) ** 2
  }
  if (!den) return null
  const slope = num / den
  return { slope, intercept: my - slope * mx }
}
