/**
 * The seal: after a run's last picture has ended and been read, turn the
 * pictures into blind judging items, and lock the key away.
 *
 * - Every picture shown gets a fresh 128-bit token per set it appears in, and
 *   a blind WebP copy (640 px for the grid, up to 2048 px full screen, the
 *   same full size for every picture of a set so nothing shows through pixel
 *   counts), made by ffmpeg with all metadata dropped, two at a time, niced.
 * - Each set draws its own letters and its own order, at random, once.
 *   Seeds keep fixed positions: 1001 top left to 4004 bottom right.
 * - A contestant that cannot take a test (N/A) is simply not in its set. A
 *   grid with fewer than 3 pictures made is left out and reported "not made".
 * - About 1 grid in 12 comes back later as a second look, under new letters
 *   and new tokens, to measure the judge's own steadiness.
 * - Every picture the picture reader rated above general becomes a blind
 *   content check.
 * - items.json holds only blind things: tokens, letters, the neutral task.
 *   sealed.json holds the key, is written once, and its sha256 heads
 *   items.json. sweep-key.json holds the step sweep's names and numbers.
 *
 * A set opens only when complete: the seal refuses while any picture of the
 * run is still to be made (unless asked for a partial seal, which leaves out
 * every set still waiting for a picture).
 */
import { createHash, randomBytes } from 'node:crypto'
import { existsSync, linkSync, copyFileSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { expand, maskKey, SHIPPED_SUITES } from '../core/cells.ts'
import { runDir, type LabEnv } from '../core/env.ts'
import { homeSteps } from '../core/params.ts'
import { setOf } from '../core/suite.ts'
import type { Block, Cell, ChainSpec, LedgerEntry, NA, Op, RefInfo, SetMode, Slot, Suite } from '../core/types.ts'
import { readDoneCells, readLedger, relOfFile, runState } from '../run/ledger.ts'
import { readReadings } from '../run/reader.ts'
import { refIndex } from '../run/refs.ts'
import { sizeOf } from '../run/imagesize.ts'
import { familyOwning } from '../../src/lib/workflows.ts'
import { blindCopy, createPool, FULL_EDGE, FULL_QUALITY, GRID_EDGE, GRID_QUALITY } from './blind.ts'
import { PAIR_QUESTIONS, type PairKind } from './pairs.ts'
import { SWEEP_KEY_FILE, type SweepKey } from './queue.ts'

/** What the phone is told about a set: the neutral task and the words to check against. */
export type Brief = {
  task: string
  checklist?: string[]
  expect?: string[]
  layout?: string
  /** The two labels of a before/after set, in row order. */
  conditions?: [string, string]
  /** Blind tokens of pictures pinned above the grid: a reference, its mask, a base. */
  pinned?: { ref?: string; mask?: string; base?: string }
}

export type Grid = {
  itemId: string
  letter: string
  /** Display position in the set, 0-based. */
  pos: number
  /** Scale sets: four tiles by seed, 1001 top left to 4004 bottom right. null where no picture was made. */
  tiles?: (string | null)[]
  /** Before/after sets: four rows by seed, each [before, after]. */
  rows?: [string | null, string | null][]
  /** Pictures available on demand beside the rows (the square lighthouse), by seed. */
  extra?: { label: string; tiles: (string | null)[] }
  /** A second look: the item id of the grid's first showing. Never sent to the phone. */
  repeats?: string
}

export type ItemSet = {
  setId: string
  order: number
  block: Block
  /** The card, as id@version: 'pf@1'. */
  card: string
  second?: string
  mode: SetMode
  /** The card asks for the worst picture full screen (R5). */
  closeLook: boolean
  brief: Brief
  grids: Grid[]
  /** A second-look set: the set its grids first appeared in. Never sent to the phone. */
  secondOf?: string
}

export type PairSide = { letter?: string; tiles: (string | null)[]; rows?: [string | null, string | null][] }

export type PairItem = {
  itemId: string
  setId: string | null
  kind: PairKind
  a: PairSide
  b: PairSide
  question: string
  /** For a pair outside any set, what was asked. */
  brief?: Brief
}

export type CheckItem = { itemId: string; token: string; rating: string; task?: string }

export type Items = {
  v: 1
  run: string
  sealedSha: string
  sets: ItemSet[]
  pairs: PairItem[]
  checks: CheckItem[]
}

export type TokenInfo = {
  /** '' for a reference photo or mask. */
  cellId: string
  contestant: string
  familyId: string
  file: string
  slot: string
  seed: number
  op: Op | 'ref' | 'mask'
  condition: string | null
  width: number
  height: number
  rel: string
  durationMs: number | null
  cold: boolean
  jobId: string | null
  promptId: string | null
  ref?: string
}

export type CellInfo = {
  cellId: string
  contestant: string
  model: string | null
  chain: string | null
  chainStep: number | null
  file: string
  familyId: string
  slot: string
  set: string
  block: Block
  op: Op
  seed: number
  steps: number
  sampler: string
  scheduler: string
  status: 'made' | 'removed' | 'failed' | 'missing'
  why: string | null
  rel: string | null
  durationMs: number | null
  cold: boolean
  cached: boolean
  jobId: string | null
  promptId: string | null
  innocent: boolean
  /** The run that made the picture (another run's for a reused one). */
  madeIn: string | null
}

export type ContestantInfo = {
  id: string
  kind: 'model' | 'chain'
  file?: string
  familyId?: string
  label?: string
  homeSteps?: number
  /** A chain's models in order: the compose model, then each step's. */
  links?: string[]
  name?: string
  /** A chain's compose model and slot. */
  compose?: { slot: string; model: string }
  block?: Block
  also?: Block[]
}

export type Sealed = {
  v: 1
  run: string
  study: string
  sealedAt: number
  suites: { id: string; version: number }[]
  tokens: Record<string, TokenInfo>
  gridOf: Record<string, { contestant: string; setId: string; cells: (string | null)[]; context?: boolean; repeats?: string }>
  letters: Record<string, Record<string, string>>
  pairOf: Record<string, {
    kind: PairKind
    a: { contestant: string; arm?: string; cells: (string | null)[] }
    b: { contestant: string; arm?: string; cells: (string | null)[] }
    slot?: string
    cmp?: string
    block?: Block
  }>
  checkOf: Record<string, { cellId: string; contestant: string; rating: string }>
  sets: Record<string, { block: Block; card: string; second?: string; mode: SetMode; slots: string[]; context: boolean; secondOf?: string }>
  slots: Record<string, { block: Block; second?: Block; innocent: boolean; measuredOnly: boolean; suite: string }>
  cells: Record<string, CellInfo>
  contestants: Record<string, ContestantInfo>
  na: NA[]
  notMade: { slot: string; setId: string | null; contestant: string; have: number }[]
}

// -------------------------------------------------------------- constants --

/** The card each block is scored on. */
export const CARD_OF: Record<Block, string> = {
  following: 'pf', style: 'st', photo: 'pr', variation: 'va', content: 'cd', text: 'tx', reference: 'rk',
  layout: 'ly', anatomy: 'an', detail: 'dt', cost: 'co', shapes: 'sh', character: 'sc', sensitivity: 'ps',
  negative: 'ng', defaults: 'df', region: 're', edit: 'ie', overall: 'ov',
}
export const CARD_VERSION = 1
export const cardRef = (b: Block): string => `${CARD_OF[b]}@${CARD_VERSION}`
/** Cards that ask for the worst picture full screen (R5), or a flip at full size. */
const CLOSE_LOOK = new Set(['pr', 'tx', 'an', 'dt'])
/** Blocks whose sets of two tests are before/after rows; any other set of two tests is judged as pairs (prompt style). */
const BEFORE_AFTER_BLOCKS = new Set<Block>(['character', 'sensitivity', 'negative', 'shapes', 'detail', 'region', 'edit'])
/** Blocks whose one-test sets show the source photo beside each result. */
const SOURCE_BESIDE = new Set<Block>(['region', 'edit'])
/** Letters a set draws from: no I, O or Q, which read as 1, 0 and O on a phone. */
export const LETTERS = 'ABCDEFGHJKLMNPRSTUVWXYZ'
export const SECOND_LOOK_SHARE = 1 / 12
export const SEALED_FILE = 'sealed.json'
export const ITEMS_FILE = 'items.json'
export const VIEW_DIR = 'view'
const STAGING_DIR = '.seal-staging'
const ABOVE_GENERAL = new Set(['sensitive', 'questionable', 'explicit'])

export type SealOptions = {
  ffmpeg: string
  /** Random bytes. Default crypto.randomBytes; tests pass a seeded one. */
  rng?: (n: number) => Buffer
  /** The suites to find the run's by id. Default the four shipped suites. */
  suites?: readonly Suite[]
  /** Seal although pictures are still to be made; every set waiting for one is left out. */
  partial?: boolean
  now?: () => number
  log?: (line: string) => void
  /** Run ffmpeg under nice (default true). */
  nice?: boolean
}

export type SealResult = { sealedSha: string; sets: number; items: number }

// ----------------------------------------------------------------- random --

class Draw {
  constructor(private readonly bytes: (n: number) => Buffer) {}
  hex(n: number): string {
    return this.bytes(n).toString('hex')
  }
  /** A whole number from 0 to max - 1, without bias. */
  int(max: number): number {
    if (max <= 1) return 0
    const limit = Math.floor(0x100000000 / max) * max
    for (;;) {
      const v = this.bytes(4).readUInt32LE(0)
      if (v < limit) return v % max
    }
  }
  shuffle<T>(xs: readonly T[]): T[] {
    const out = xs.slice()
    for (let i = out.length - 1; i > 0; i--) {
      const j = this.int(i + 1)
      ;[out[i], out[j]] = [out[j], out[i]]
    }
    return out
  }
  coin(): boolean {
    return this.int(2) === 1
  }
}

// --------------------------------------------------------------- pictures --

type Pic = {
  key: string
  kind: 'cell' | 'ref' | 'refbox' | 'mask'
  src: string
  width: number
  height: number
  cellId?: string
  ref?: { id: string; sha12: string; rel: string }
  outline?: { x: number; y: number; w: number; h: number } | null
}

type GridSpec = {
  contestant: string
  tiles?: (Pic | null)[]
  rows?: [Pic | null, Pic | null][]
  extra?: { label: string; tiles: (Pic | null)[] }
  cells: (string | null)[]
  context: boolean
  repeats?: GridSpec
  itemId?: string
}

type SetSpec = {
  key: string
  slots: Slot[]
  block: Block
  card: string
  second?: string
  mode: SetMode
  brief: Omit<Brief, 'pinned'>
  pinned: { ref?: Pic; mask?: Pic; base?: Pic }
  grids: GridSpec[]
  context: boolean
  order: number
  secondOf?: SetSpec
  setId?: string
  pending: boolean
}

type PairSpec = {
  kind: PairKind
  set: SetSpec | null
  a: { contestant: string; arm?: string; grid?: GridSpec; tiles: (Pic | null)[] }
  b: { contestant: string; arm?: string; grid?: GridSpec; tiles: (Pic | null)[] }
  question: string
  brief?: Omit<Brief, 'pinned'>
  slot?: string
  cmp?: string
  block?: Block
  sweep?: { model: string; a: string; b: string }
}

function readJson<T>(file: string): T {
  return JSON.parse(readFileSync(file, 'utf8')) as T
}

function writeAtomic(file: string, text: string): void {
  mkdirSync(path.dirname(file), { recursive: true })
  const tmp = `${file}.tmp-${process.pid}`
  writeFileSync(tmp, text, { mode: 0o600 })
  renameSync(tmp, file)
}

const madeCount = (g: GridSpec): number =>
  g.rows ? g.rows.filter(r => r[0] && r[1]).length : (g.tiles ?? []).filter(Boolean).length

/** A grid is scored only with at least 3 of its 4 pictures (R1). */
const MIN_MADE = 3

// ------------------------------------------------------------------- seal --

/**
 * Seal a run: blind copies, items.json, and the sealed key. Writes sealed.json
 * once: sealing a sealed run returns what it made the first time.
 */
export async function sealRun(env: LabEnv, run: string, opts: SealOptions): Promise<SealResult> {
  const dir = runDir(env, run)
  const sealedFile = path.join(dir, SEALED_FILE)
  const itemsFile = path.join(dir, ITEMS_FILE)
  const pendingItems = `${itemsFile}.pending`
  if (existsSync(sealedFile)) {
    // Sealed already. A seal that stopped between writing the key and putting items.json in place finishes here.
    if (!existsSync(itemsFile) && existsSync(pendingItems)) {
      const sha = createHash('sha256').update(readFileSync(sealedFile)).digest('hex')
      if (readJson<Items>(pendingItems).sealedSha === sha) renameSync(pendingItems, itemsFile)
    }
    if (!existsSync(itemsFile)) throw new Error(`${run} has a sealed key but no items.json, and the key cannot be made again. Ask for help before judging this run.`)
    const items = readJson<Items>(itemsFile)
    return { sealedSha: items.sealedSha, sets: items.sets.length, items: countItems(items) }
  }
  const log = opts.log ?? (() => {})
  const now = opts.now ?? Date.now
  const draw = new Draw(opts.rng ?? randomBytes)
  const planFile = path.join(dir, 'plan.json')
  if (!existsSync(planFile)) throw new Error(`There is no plan for ${run}.`)
  const plan = readJson<{
    run?: string
    study?: string
    suites?: { id: string; version?: number }[]
    order: string[]
    reused?: string[]
    na?: NA[]
    /** Every cell row of the run, reused ones included (the plan keeps them so the seal need not expand again). */
    cells?: Cell[]
    /** Cells drawn only because a chain starts from them. */
    context?: string[]
  }>(planFile)
  const all = opts.suites ?? SHIPPED_SUITES
  const runSuites: Suite[] = []
  for (const s of plan.suites ?? []) {
    const suite = all.find(x => x.id === s.id)
    if (!suite) throw new Error(`${run} was planned from suite ${s.id}, which the lab no longer has.`)
    runSuites.push(suite)
  }
  if (!runSuites.length) throw new Error(`The plan of ${run} names no suite.`)
  const study = plan.study ?? runSuites[0].study
  const ctxSuites = all.filter(s => s.study === study && !runSuites.includes(s))

  // The run's pictures, with the tests they belong to in this run's suites:
  // the plan's own rows when it kept them, else the suites expanded again.
  const refsIdx = refIndex(env)
  const planIds = new Set([...plan.order, ...(plan.reused ?? [])])
  const x: { cells: Cell[]; na: NA[]; context: Set<string> } = Array.isArray(plan.cells)
    ? { cells: plan.cells, na: plan.na ?? [], context: new Set(plan.context ?? []) }
    : (() => {
        const e = expand(runSuites, refsIdx, { context: ctxSuites })
        return { cells: e.cells, na: e.na, context: e.context }
      })()
  const rows = x.cells.filter(c => planIds.has(c.cellId))
  const seenIds = new Set(rows.map(r => r.cellId))
  for (const id of planIds) {
    if (seenIds.has(id)) continue
    // A cell the expansion no longer gives (a photo changed since the plan): its own file says what it was.
    const f = path.join(env.labDir, 'cells', `${id}.json`)
    if (!existsSync(f)) throw new Error(`Cell ${id} of ${run} has no file in the lab folder; plan the run again.`)
    const raw = readJson<Record<string, unknown>>(f)
    const cell = (raw.cell && typeof raw.cell === 'object' ? raw.cell : raw) as Cell
    rows.push(cell)
  }
  const contextIds = x.context

  // What became of each picture.
  const ledger = readLedger(dir)
  const global = readDoneCells(env)
  const st = runState(ledger, { order: plan.order, reused: plan.reused }, global)
  const pendingIds = new Set([...st.pending, ...st.inFlight.values()])
  if (pendingIds.size && !opts.partial) {
    throw new Error(`${pendingIds.size} picture${pendingIds.size === 1 ? ' is' : 's are'} of ${run} still to be made, so no set can open yet. Finish the run first.`)
  }
  const promptOfJob = new Map<string, string | null>()
  for (const e of ledger) if (e.t === 'ended') promptOfJob.set(e.job, e.promptId)
  const otherRuns = new Map<string, Map<string, { job: string; promptId: string | null }>>()
  const doneIn = (r: string, cellId: string) => {
    if (!otherRuns.has(r)) {
      const m = new Map<string, { job: string; promptId: string | null }>()
      try {
        for (const e of readLedger(runDir(env, r))) if (e.t === 'ended' && e.status === 'done' && !m.has(e.cell)) m.set(e.cell, { job: e.job, promptId: e.promptId })
      } catch { /* that run's folder is gone */ }
      otherRuns.set(r, m)
    }
    return otherRuns.get(r)?.get(cellId)
  }
  const readings = readReadings(env)

  // Slots, chains and contestants.
  const slotIndex = new Map<string, { slot: Slot; suite: Suite; order: number }>()
  let k = 0
  for (const s of [...runSuites, ...ctxSuites]) for (const sl of s.slots) if (!slotIndex.has(sl.id)) slotIndex.set(sl.id, { slot: sl, suite: s, order: k++ })
  const chains = new Map<string, { chain: ChainSpec; suite: Suite }>()
  for (const s of runSuites) for (const c of s.chains) chains.set(c.id, { chain: c, suite: s })
  const seeds = runSuites[0].seeds.slice()
  const models = runSuites[0].models
  const coreOrder = [...runSuites[0].core, ...Object.keys(models).filter(m => !runSuites[0].core.includes(m))]

  const cellInfo = new Map<string, CellInfo>()
  const outputs = path.resolve(env.outputs)
  const inside = (rel: string) => {
    const full = path.resolve(outputs, rel)
    return full.startsWith(outputs + path.sep) ? full : null
  }
  for (const r of rows) {
    if (cellInfo.has(r.cellId) && contextIds.has(r.cellId)) continue
    const slotDef = slotIndex.get(r.slot)?.slot
    const chain = r.chain ? chains.get(r.chain)?.chain : undefined
    const block: Block = chain ? chain.block : slotDef?.block ?? 'overall'
    let status: CellInfo['status'] = 'missing'
    let why: string | null = pendingIds.has(r.cellId) ? 'still to be made' : null
    let rel: string | null = null
    let durationMs: number | null = null
    let cold = false
    let cached = false
    let jobId: string | null = null
    let promptId: string | null = null
    let madeIn: string | null = null
    const d = st.done.get(r.cellId)
    const g = global.get(r.cellId)
    if (st.removed.has(r.cellId) || g?.removed) {
      status = 'removed'
      why = 'removed by the quarantine rule'
    } else if (d) {
      rel = d.rel
      const full = inside(d.rel)
      if (full && existsSync(full)) status = 'made'
      else why = 'the picture is not on disk'
      durationMs = d.durationMs > 0 ? d.durationMs : null
      cold = d.cold
      cached = d.cached
      if (d.how === 'reused' && g) {
        madeIn = g.run
        const hit = doneIn(g.run, r.cellId)
        jobId = hit?.job ?? null
        promptId = hit?.promptId ?? null
      } else {
        madeIn = run
        jobId = d.job
        promptId = d.job ? promptOfJob.get(d.job) ?? null : null
      }
    } else if (st.failed.has(r.cellId)) {
      status = 'failed'
      why = st.failed.get(r.cellId)?.why ?? 'failed'
    }
    cellInfo.set(r.cellId, {
      cellId: r.cellId,
      contestant: r.chain ?? r.model ?? '',
      model: r.model,
      chain: r.chain,
      chainStep: r.chainStep,
      file: r.file,
      familyId: r.familyId,
      slot: r.slot,
      set: r.set,
      block,
      op: r.op,
      seed: r.seed,
      steps: r.steps,
      sampler: r.sampler,
      scheduler: r.scheduler,
      status,
      why,
      rel,
      durationMs,
      cold,
      cached,
      jobId,
      promptId,
      innocent: slotDef?.innocent !== false,
      madeIn,
    })
  }

  // Pictures, measured once.
  const pics = new Map<string, Pic>()
  const cellPic = (cellId: string | null | undefined): Pic | null => {
    if (!cellId) return null
    const info = cellInfo.get(cellId)
    if (!info || info.status !== 'made' || !info.rel) return null
    const key = `cell:${cellId}`
    const had = pics.get(key)
    if (had) return had
    const src = inside(info.rel) as string
    let size: { width: number; height: number }
    try {
      size = sizeOf(readFileSync(src))
    } catch {
      // An unreadable picture counts as not made.
      info.status = 'missing'
      info.why = 'the picture on disk cannot be read'
      return null
    }
    const p: Pic = { key, kind: 'cell', src, width: size.width, height: size.height, cellId }
    pics.set(key, p)
    return p
  }
  const refPic = (sha12: string | null, kind: 'ref' | 'refbox' | 'mask'): Pic | null => {
    if (!sha12) return null
    const key = `${kind}:${sha12}`
    const had = pics.get(key)
    if (had) return had
    // A mask is named by its own fingerprint (maskKey), a photo by the photo's.
    const info = Object.values(refsIdx).find(r => (kind === 'mask' ? maskKey(r) : r.sha12) === sha12) as RefInfo | undefined
    const refsDir = path.join(outputs, '.lab', 'refs')
    let names: string[] = []
    try {
      names = readdirSync(refsDir)
    } catch { /* no copies yet */ }
    const name = kind === 'mask'
      ? names.find(n => n === `${sha12}.mask.png`)
      : names.find(n => n.startsWith(`${sha12}.`) && !n.endsWith('.mask.png'))
    let src = name ? path.join(refsDir, name) : null
    if (!src && info) {
      // The private original, when the copy under outputs has gone.
      const priv = path.join(env.labDir, 'refs', kind === 'mask' ? `${info.id}.mask.png` : `${info.id}.${info.ext}`)
      if (existsSync(priv)) src = priv
    }
    if (!src) return null
    const size = sizeOf(readFileSync(src))
    const p: Pic = {
      key, kind, src, width: size.width, height: size.height,
      ref: { id: info?.id ?? '', sha12, rel: name ? `.lab/refs/${name}` : '' },
      outline: kind === 'refbox' ? info?.rect ?? null : null,
    }
    pics.set(key, p)
    return p
  }
  const shaOf = (cellId: string | null | undefined, form: 'ref' | 'mask'): string | null => {
    const r = rows.find(c => c.cellId === cellId)
    const ph = r?.placeholders.find(p => p[2].startsWith(`${form}:`))
    return ph ? ph[2].slice(form.length + 1) : null
  }

  // Cells of the run by test, contestant and seed.
  const plain = rows.filter(r => r.chain === null && r.model !== null && r.set !== 'sweep' && r.set !== 'sampler')
  const bySlot = new Map<string, string>()
  for (const r of plain) {
    const key = `${r.slot}|${r.model}|${r.seed}`
    if (!bySlot.has(key) || !contextIds.has(r.cellId)) bySlot.set(key, r.cellId)
  }
  const cellAt = (slot: string, model: string, seed: number) => bySlot.get(`${slot}|${model}|${seed}`) ?? null
  const hasAny = (slot: string, model: string) => seeds.some(sd => cellAt(slot, model, sd) !== null)
  const isPending = (ids: (string | null)[]) => ids.some(id => id !== null && pendingIds.has(id))
  const contestantsOf = (list: Cell[]) => {
    const found = new Set(list.map(r => r.model as string))
    return coreOrder.filter(m => found.has(m)).concat([...found].filter(m => !coreOrder.includes(m)).sort())
  }

  const notMade: Sealed['notMade'] = []
  const sets: SetSpec[] = []
  const pairs: PairSpec[] = []
  const bySetKey = new Map<string, Cell[]>()
  for (const r of plain) {
    const list = bySetKey.get(r.set) ?? []
    list.push(r)
    bySetKey.set(r.set, list)
  }
  const orderOfSlot = (id: string) => slotIndex.get(id)?.order ?? 1e6
  for (const [setKey, list] of bySetKey) {
    const slotIds = [...new Set(list.map(r => r.slot))].sort((a, b) => orderOfSlot(a) - orderOfSlot(b))
    const slots = slotIds.map(id => slotIndex.get(id)?.slot).filter((s): s is Slot => !!s)
    if (!slots.length || slots.every(s => s.measuredOnly)) continue
    const primary = slots[0]
    const block = primary.block
    const context = list.every(r => contextIds.has(r.cellId))
    const who = contestantsOf(list)
    const spec: SetSpec = {
      key: setKey, slots, block, card: cardRef(block), mode: 'scale', brief: { task: '' }, pinned: {}, grids: [],
      context, order: Math.min(...slotIds.map(orderOfSlot)) + (context ? 1e5 : 0), pending: false,
    }
    if (primary.second) spec.second = cardRef(primary.second)
    const refId = primary.source && 'ref' in primary.source ? primary.source.ref : null
    const anyCell = list.find(r => r.slot === primary.id)?.cellId ?? null
    const refSha = refId ? shaOf(anyCell, 'ref') : null
    const words = (s: Slot) => s.task ?? s.text
    const common = { checklist: primary.checklist, expect: primary.expect, layout: primary.layout }

    if (slots.length >= 2 && !BEFORE_AFTER_BLOCKS.has(block)) {
      // Two wordings of one test (prompt style): a pair per contestant, never a scored set.
      const [sa, sb] = slots
      for (const m of who) {
        const ta = seeds.map(sd => cellAt(sa.id, m, sd))
        const tb = seeds.map(sd => cellAt(sb.id, m, sd))
        if (isPending([...ta, ...tb])) continue
        const pa = ta.map(cellPic)
        const pb = tb.map(cellPic)
        const ha = pa.filter(Boolean).length
        const hb = pb.filter(Boolean).length
        if (ha < MIN_MADE || hb < MIN_MADE) {
          notMade.push({ slot: ha < MIN_MADE ? sa.id : sb.id, setId: null, contestant: m, have: Math.min(ha, hb) })
          continue
        }
        pairs.push({
          kind: 'promptstyle', set: null, question: PAIR_QUESTIONS.brief, slot: setKey, block,
          brief: { task: words(sa), ...common },
          a: { contestant: m, arm: sa.condition ?? sa.id, tiles: pa },
          b: { contestant: m, arm: sb.condition ?? sb.id, tiles: pb },
        })
      }
      continue
    }

    if (slots.length >= 2) {
      spec.mode = 'beforeAfter'
      // Three shapes of one test (the lighthouse): one grid of wide | tall rows,
      // the square on demand, and so one step per grid for both shapes. The
      // plan's "mean of wide and tall, the worse shape named" would need a
      // step per shape; the sh card says the weaker shape sets the one step
      // instead, and the report does not name which shape was weaker.
      let pair = slots.slice(0, 2)
      let extra: Slot | null = null
      if (slots.length > 2) {
        const nonSquare = slots.filter(s => s.shape !== 'square')
        if (nonSquare.length === 2) {
          pair = nonSquare
          extra = slots.find(s => s.shape === 'square') ?? null
        }
      }
      const [s0, s1] = pair
      const t0 = words(s0)
      const t1 = words(s1)
      const conditions: [string, string] = [s0.condition ?? 'first', s1.condition ?? 'second']
      spec.brief = { task: t0 === t1 ? t0 : `${conditions[0]}: ${t0} ${conditions[1]}: ${t1}`, conditions, checklist: s0.checklist, expect: s0.expect, layout: s0.layout }
      if (refSha) spec.pinned.ref = refPic(refSha, 'ref') ?? undefined
      for (const m of who) {
        if (!hasAny(s0.id, m) || !hasAny(s1.id, m)) continue
        const ids = seeds.map(sd => [cellAt(s0.id, m, sd), cellAt(s1.id, m, sd)] as [string | null, string | null])
        const extraIds = extra ? seeds.map(sd => cellAt((extra as Slot).id, m, sd)) : []
        if (isPending([...ids.flat(), ...extraIds])) { spec.pending = true; continue }
        const g: GridSpec = { contestant: m, rows: ids.map(([a, b]) => [cellPic(a), cellPic(b)]), cells: ids.map(r => r[1]), context }
        if (extra) g.extra = { label: extra.condition ?? extra.shape, tiles: extraIds.map(cellPic) }
        spec.grids.push(g)
      }
    } else if (primary.after) {
      spec.mode = 'beforeAfter'
      const base = slotIndex.get(primary.after)?.slot
      spec.brief = { task: words(primary), conditions: [base?.condition ?? 'the base picture', primary.condition ?? 'after'], ...common }
      for (const m of who) {
        if (!hasAny(primary.after, m)) continue
        const ids = seeds.map(sd => [cellAt(primary.after as string, m, sd), cellAt(primary.id, m, sd)] as [string | null, string | null])
        if (isPending(ids.flat())) { spec.pending = true; continue }
        spec.grids.push({ contestant: m, rows: ids.map(([a, b]) => [cellPic(a), cellPic(b)]), cells: ids.map(r => r[1]), context })
      }
    } else if (refSha && SOURCE_BESIDE.has(block)) {
      spec.mode = 'beforeAfter'
      const region = block === 'region'
      const before = refPic(refSha, region ? 'refbox' : 'ref')
      spec.brief = { task: words(primary), conditions: region ? ['the photo, area outlined', 'the result'] : ['the photo', 'the result'], ...common }
      if (region) {
        spec.pinned.base = refPic(refSha, 'ref') ?? undefined
        spec.pinned.mask = refPic(shaOf(anyCell, 'mask') ?? refSha, 'mask') ?? undefined
      } else spec.pinned.ref = refPic(refSha, 'ref') ?? undefined
      for (const m of who) {
        const ids = seeds.map(sd => cellAt(primary.id, m, sd))
        if (isPending(ids)) { spec.pending = true; continue }
        spec.grids.push({ contestant: m, rows: ids.map(id => [before, cellPic(id)] as [Pic | null, Pic | null]), cells: ids, context })
      }
    } else {
      spec.brief = { task: words(primary), ...common }
      if (refSha) spec.pinned.ref = refPic(refSha, 'ref') ?? undefined
      for (const m of who) {
        const ids = seeds.map(sd => cellAt(primary.id, m, sd))
        if (isPending(ids)) { spec.pending = true; continue }
        spec.grids.push({ contestant: m, tiles: ids.map(cellPic), cells: ids, context })
      }
    }
    sets.push(spec)
  }

  // Chains: one more contestant in the set of the picture they start from, and a pair against it.
  const contestantInfo = new Map<string, ContestantInfo>()
  for (const [id, { chain }] of chains) {
    const crow = rows.filter(r => r.chain === id)
    if (!crow.length) continue
    const last = Math.max(...crow.map(r => r.chainStep ?? 0))
    const finals = seeds.map(sd => crow.find(r => r.seed === sd && r.chainStep === last)?.cellId ?? null)
    const composeIds = seeds.map(sd => crow.find(r => r.seed === sd && r.chainStep === 0)?.upstream ?? null)
    contestantInfo.set(id, {
      id, kind: 'chain', name: chain.name, compose: chain.compose, block: chain.block, also: chain.also,
      links: [chain.compose.model, ...chain.steps.map(s => s.model)],
    })
    const composeSlot = slotIndex.get(chain.compose.slot)?.slot
    const homeKey = composeSlot ? setOf(composeSlot) : chain.compose.slot
    const home = sets.find(s => s.key === homeKey)
    if (!home) continue
    if (isPending([...finals, ...composeIds])) { home.pending = true; continue }
    const add = (s: SetSpec): GridSpec => {
      const g: GridSpec = s.mode === 'scale'
        ? { contestant: id, tiles: finals.map(cellPic), cells: finals, context: false }
        : { contestant: id, rows: seeds.map((_, i) => [cellPic(composeIds[i]), cellPic(finals[i])] as [Pic | null, Pic | null]), cells: finals, context: false }
      s.grids.push(g)
      return g
    }
    const g = add(home)
    const composeGrid = home.grids.find(x => x.contestant === chain.compose.model)
    if (composeGrid && madeCount(g) >= MIN_MADE && madeCount(composeGrid) >= MIN_MADE) {
      pairs.push({
        kind: 'chain', set: home, question: PAIR_QUESTIONS.block, block: home.block,
        a: { contestant: id, grid: g, tiles: [] }, b: { contestant: chain.compose.model, grid: composeGrid, tiles: [] },
      })
    }
    const lastTarget = chain.steps[chain.steps.length - 1]?.target
    for (const b of chain.also ?? []) {
      const cands = sets.filter(s => s.block === b && s.slots[0].after === chain.compose.slot)
      const target = cands.find(s => s.slots[0].target === lastTarget) ?? cands[0]
      if (target) add(target)
    }
  }

  // Grids with too few pictures are not scored.
  for (const s of sets) {
    s.grids = s.grids.filter(g => {
      const n = madeCount(g)
      if (n >= MIN_MADE) return true
      notMade.push({ slot: s.slots[0].id, setId: null, contestant: g.contestant, have: n })
      return false
    })
  }
  const held = sets.filter(s => s.pending).map(s => s.key)
  const open = sets.filter(s => !s.pending && s.grids.length)
  const openPairs = pairs.filter(p => !p.set || open.includes(p.set))

  // The step sweep and the sampler check.
  const sweepKey: SweepKey = { v: 1, run, study, pairs: {}, timings: {}, homeSteps: {}, homeSampler: {} }
  const sweepPairs: PairSpec[] = []
  const armTiles = (list: Cell[]) => seeds.map(sd => list.find(r => r.seed === sd)?.cellId ?? null)
  for (const suite of runSuites) {
    if (suite.sweep) {
      Object.assign(sweepKey.homeSteps, suite.sweep.homeFor)
      for (const model of suite.sweep.models) {
        for (const slotId of suite.sweep.slots) {
          const arms = new Map<number, (string | null)[]>()
          for (const n of suite.sweep.steps) {
            const list = rows.filter(r => r.set === 'sweep' && r.model === model && r.slot === slotId && r.steps === n)
            if (list.length) arms.set(n, armTiles(list))
          }
          for (const [lo, hi] of [[8, 28], [16, 28], [28, 40]]) {
            const A = arms.get(lo)
            const B = arms.get(hi)
            if (!A || !B || isPending([...A, ...B])) continue
            const pa = A.map(cellPic)
            const pb = B.map(cellPic)
            if (pa.filter(Boolean).length < MIN_MADE || pb.filter(Boolean).length < MIN_MADE) {
              notMade.push({ slot: slotId, setId: null, contestant: `${model}@${pa.filter(Boolean).length < MIN_MADE ? lo : hi}`, have: Math.min(pa.filter(Boolean).length, pb.filter(Boolean).length) })
              continue
            }
            const slot = slotIndex.get(slotId)?.slot
            sweepPairs.push({
              kind: 'sweep', set: null, question: PAIR_QUESTIONS.overall, slot: slotId, cmp: `${lo}v${hi}`,
              brief: { task: slot ? slot.task ?? slot.text : '' },
              a: { contestant: model, arm: String(lo), tiles: pa }, b: { contestant: model, arm: String(hi), tiles: pb },
            })
          }
        }
        for (const r of rows) {
          const c = cellInfo.get(r.cellId)
          if (r.set !== 'sweep' || r.model !== model || !c || c.status !== 'made' || c.cold || c.cached || !c.durationMs) continue
          const t = (sweepKey.timings[model] ??= {})
          ;(t[String(r.steps)] ??= []).push(c.durationMs)
        }
      }
    }
    if (suite.samplerCheck) {
      for (const model of suite.samplerCheck.models) {
        for (const slotId of suite.samplerCheck.slots) {
          const home = rows.filter(r => r.set === 'sampler' && r.model === model && r.slot === slotId && r.condition === 'home sampler')
          const common = rows.filter(r => r.set === 'sampler' && r.model === model && r.slot === slotId && r.condition === 'common sampler')
          if (!home.length || !common.length) continue
          sweepKey.homeSampler[model] = `${home[0].sampler}/${home[0].scheduler}`
          const A = armTiles(home)
          const B = armTiles(common)
          if (isPending([...A, ...B])) continue
          const pa = A.map(cellPic)
          const pb = B.map(cellPic)
          if (pa.filter(Boolean).length < MIN_MADE || pb.filter(Boolean).length < MIN_MADE) {
            notMade.push({ slot: slotId, setId: null, contestant: `${model}@sampler`, have: Math.min(pa.filter(Boolean).length, pb.filter(Boolean).length) })
            continue
          }
          const slot = slotIndex.get(slotId)?.slot
          sweepPairs.push({
            kind: 'sampler', set: null, question: PAIR_QUESTIONS.overall, slot: slotId, cmp: 'home-v-common',
            brief: { task: slot ? slot.task ?? slot.text : '' },
            a: { contestant: model, arm: 'home', tiles: pa }, b: { contestant: model, arm: 'common', tiles: pb },
          })
        }
      }
    }
  }

  // Second looks: about 1 grid in 12, under new letters, in their own sets.
  const candidates = open.filter(s => !s.context).flatMap(s => s.grids.filter(g => !g.context).map(g => ({ s, g })))
  const want = candidates.length >= 6 ? Math.max(1, Math.round(candidates.length * SECOND_LOOK_SHARE)) : 0
  const chosen = draw.shuffle(candidates).slice(0, want)
  const seconds: SetSpec[] = []
  for (const s of open) {
    const mine = chosen.filter(c => c.s === s)
    if (!mine.length) continue
    seconds.push({
      ...s, key: `${s.key}#second`, order: s.order + 1e6, secondOf: s,
      grids: mine.map(({ g }) => ({ ...g, context: false, repeats: g })),
    })
  }

  // Content checks: every picture of this run the picture reader rated above general.
  const checkCells: { cellId: string; rating: string; task: string }[] = []
  for (const c of cellInfo.values()) {
    if (c.status !== 'made' || contextIds.has(c.cellId) || c.madeIn !== run) continue
    const rating = readings.get(c.cellId)?.rating
    if (!rating || !ABOVE_GENERAL.has(rating)) continue
    const slot = slotIndex.get(c.slot)?.slot
    checkCells.push({ cellId: c.cellId, rating, task: slot ? slot.task ?? slot.text : '' })
  }

  // Contestants' own facts, for the report after the reveal.
  for (const c of cellInfo.values()) {
    if (!c.model || contestantInfo.has(c.model)) continue
    const spec = models[c.model]
    const def = spec ? familyOwning(spec.file) : null
    const per = def ? ((def.perModel?.[spec.file] ?? {}) as { label?: unknown }) : {}
    contestantInfo.set(c.model, {
      id: c.model, kind: 'model', file: spec?.file ?? c.file, familyId: def?.id ?? c.familyId,
      // The file's own label; a family's label can name several files, so it is not used.
      label: typeof per.label === 'string' ? per.label : (spec?.file ?? c.file).replace(/\.(safetensors|gguf|ckpt)$/i, ''),
      homeSteps: def && spec ? homeSteps(def, spec.file) : undefined,
    })
  }

  // ---------------------------------------------------------------- tokens --
  const tokens: Record<string, TokenInfo> = {}
  type Copy = { token: string; pic: Pic; full: number }
  const copies: Copy[] = []
  const fullOf = (list: (Pic | null | undefined)[]) =>
    Math.min(FULL_EDGE, Math.max(1, ...list.filter((p): p is Pic => !!p).map(p => Math.max(p.width, p.height))))
  const mint = (pic: Pic | null | undefined, full: number, ctx: { slot: string; condition: string | null }, seen: Map<string, string>): string | null => {
    if (!pic) return null
    const had = seen.get(pic.key)
    if (had) return had
    const token = draw.hex(16)
    seen.set(pic.key, token)
    copies.push({ token, pic, full })
    const info = pic.cellId ? cellInfo.get(pic.cellId) : undefined
    tokens[token] = info
      ? {
          cellId: info.cellId, contestant: info.contestant, familyId: info.familyId, file: info.file, slot: info.slot, seed: info.seed,
          op: info.op, condition: ctx.condition, width: pic.width, height: pic.height, rel: info.rel ?? '', durationMs: info.durationMs,
          cold: info.cold, jobId: info.jobId, promptId: info.promptId,
        }
      : {
          cellId: '', contestant: '', familyId: '', file: '', slot: ctx.slot, seed: 0, op: pic.kind === 'mask' ? 'mask' : 'ref',
          condition: ctx.condition, width: pic.width, height: pic.height, rel: pic.ref?.rel ?? '', durationMs: null, cold: false,
          jobId: null, promptId: null, ref: pic.ref?.id,
        }
    return token
  }

  const itemSets: ItemSet[] = []
  const gridOf: Sealed['gridOf'] = {}
  const letters: Sealed['letters'] = {}
  const setsInfo: Sealed['sets'] = {}
  const ordered = [...open.sort((a, b) => a.order - b.order), ...seconds.sort((a, b) => a.order - b.order)]
  ordered.forEach((s, i) => {
    s.setId = draw.hex(9)
    const seen = new Map<string, string>()
    const gridPics = s.grids.flatMap(g => [...(g.tiles ?? []), ...(g.rows ?? []).flat(), ...(g.extra?.tiles ?? [])])
    const full = fullOf(gridPics)
    const conds = s.brief.conditions
    const pool = draw.shuffle([...LETTERS]).slice(0, s.grids.length).sort()
    const order = draw.shuffle(s.grids)
    const grids: Grid[] = order.map((g, pos) => {
      g.itemId = draw.hex(12)
      const out: Grid = { itemId: g.itemId, letter: pool[pos] ?? `Z${LETTERS[pos % LETTERS.length]}`, pos }
      if (g.tiles) out.tiles = g.tiles.map(p => mint(p, full, { slot: s.slots[0].id, condition: null }, seen))
      if (g.rows) out.rows = g.rows.map(([a, b]) => [
        mint(a, full, { slot: s.slots[0].id, condition: conds?.[0] ?? null }, seen),
        mint(b, full, { slot: s.slots[0].id, condition: conds?.[1] ?? null }, seen),
      ])
      if (g.extra) out.extra = { label: g.extra.label, tiles: g.extra.tiles.map(p => mint(p, full, { slot: s.slots[0].id, condition: g.extra?.label ?? null }, seen)) }
      if (g.repeats?.itemId) out.repeats = g.repeats.itemId
      gridOf[g.itemId] = { contestant: g.contestant, setId: s.setId as string, cells: g.cells, ...(g.context ? { context: true } : {}), ...(out.repeats ? { repeats: out.repeats } : {}) }
      ;(letters[s.setId as string] ??= {})[out.letter] = g.contestant
      return out
    })
    const pinned: Brief['pinned'] = {}
    const pin = (p: Pic | undefined) => mint(p, Math.min(FULL_EDGE, p ? Math.max(p.width, p.height) : FULL_EDGE), { slot: s.slots[0].id, condition: 'pinned' }, seen) ?? undefined
    if (s.pinned.ref) pinned.ref = pin(s.pinned.ref)
    if (s.pinned.base) pinned.base = pin(s.pinned.base)
    if (s.pinned.mask) pinned.mask = pin(s.pinned.mask)
    const brief: Brief = { ...s.brief }
    for (const k of Object.keys(brief) as (keyof Brief)[]) if (brief[k] === undefined) delete brief[k]
    if (Object.keys(pinned).length) brief.pinned = pinned
    const set: ItemSet = {
      setId: s.setId, order: i, block: s.block, card: s.card, mode: s.mode,
      closeLook: CLOSE_LOOK.has(CARD_OF[s.block]), brief, grids: grids.sort((a, b) => a.pos - b.pos),
    }
    if (s.second) set.second = s.second
    if (s.secondOf?.setId) set.secondOf = s.secondOf.setId
    itemSets.push(set)
    setsInfo[s.setId] = {
      block: s.block, card: s.card, ...(s.second ? { second: s.second } : {}), mode: s.mode,
      slots: s.slots.map(x => x.id), context: s.context, ...(s.secondOf?.setId ? { secondOf: s.secondOf.setId } : {}),
    }
  })

  // Pairs: chain pairs in their set (the set's own tokens), then the ones outside any set, shuffled.
  const itemPairs: PairItem[] = []
  const pairOf: Sealed['pairOf'] = {}
  const setItem = (s: SetSpec) => itemSets.find(x => x.setId === s.setId)
  for (const p of openPairs.filter(q => q.kind === 'chain')) {
    const s = p.set as SetSpec
    const item = setItem(s)
    const ga = item?.grids.find(g => g.itemId === p.a.grid?.itemId)
    const gb = item?.grids.find(g => g.itemId === p.b.grid?.itemId)
    if (!item || !ga || !gb) continue
    const flip = draw.coin()
    const [x, y] = flip ? [gb, ga] : [ga, gb]
    const [cx, cy] = flip ? [p.b, p.a] : [p.a, p.b]
    const side = (g: Grid): PairSide => (g.rows ? { letter: g.letter, tiles: g.rows.map(r => r[1]), rows: g.rows } : { letter: g.letter, tiles: g.tiles ?? [] })
    const itemId = draw.hex(12)
    itemPairs.push({ itemId, setId: item.setId, kind: 'chain', a: side(x), b: side(y), question: p.question })
    pairOf[itemId] = {
      kind: 'chain', block: s.block,
      a: { contestant: cx.contestant, cells: gridOf[x.itemId]?.cells ?? [] },
      b: { contestant: cy.contestant, cells: gridOf[y.itemId]?.cells ?? [] },
    }
  }
  for (const p of draw.shuffle([...openPairs.filter(q => q.kind !== 'chain'), ...sweepPairs])) {
    const flip = draw.coin()
    const [x, y] = flip ? [p.b, p.a] : [p.a, p.b]
    const seen = new Map<string, string>()
    const full = fullOf([...x.tiles, ...y.tiles])
    const tiles = (list: (Pic | null)[], arm: string | undefined) => list.map(pic => mint(pic, full, { slot: p.slot ?? '', condition: arm ?? null }, seen))
    const itemId = draw.hex(12)
    const pair: PairItem = { itemId, setId: null, kind: p.kind, a: { tiles: tiles(x.tiles, x.arm) }, b: { tiles: tiles(y.tiles, y.arm) }, question: p.question }
    if (p.brief) {
      const brief: Brief = { ...p.brief }
      for (const k of Object.keys(brief) as (keyof Brief)[]) if (brief[k] === undefined) delete brief[k]
      pair.brief = brief
    }
    itemPairs.push(pair)
    const cellsOf = (list: (Pic | null)[]) => list.map(pic => pic?.cellId ?? null)
    pairOf[itemId] = {
      kind: p.kind, slot: p.slot, cmp: p.cmp, block: p.block,
      a: { contestant: x.contestant, arm: x.arm, cells: cellsOf(x.tiles) },
      b: { contestant: y.contestant, arm: y.arm, cells: cellsOf(y.tiles) },
    }
    if (p.kind === 'sweep' || p.kind === 'sampler') {
      sweepKey.pairs[itemId] = { kind: p.kind, model: x.contestant, slot: p.slot ?? '', cmp: p.cmp ?? '', a: x.arm ?? '', b: y.arm ?? '' }
    }
  }

  // Content checks, shuffled so no model's pictures come together.
  const checks: CheckItem[] = []
  const checkOf: Sealed['checkOf'] = {}
  for (const c of draw.shuffle(checkCells)) {
    const pic = cellPic(c.cellId)
    if (!pic) continue
    const token = mint(pic, Math.min(FULL_EDGE, Math.max(pic.width, pic.height)), { slot: cellInfo.get(c.cellId)?.slot ?? '', condition: 'content check' }, new Map())
    if (!token) continue
    const itemId = draw.hex(12)
    checks.push({ itemId, token, rating: c.rating, task: c.task })
    checkOf[itemId] = { cellId: c.cellId, contestant: cellInfo.get(c.cellId)?.contestant ?? '', rating: c.rating }
  }

  // ----------------------------------------------------------- blind copies --
  const view = path.join(dir, VIEW_DIR)
  const staging = path.join(dir, STAGING_DIR)
  rmSync(view, { recursive: true, force: true })
  rmSync(staging, { recursive: true, force: true })
  mkdirSync(view, { recursive: true, mode: 0o700 })
  mkdirSync(staging, { recursive: true, mode: 0o700 })
  const pool = createPool()
  const made = new Map<string, Promise<string>>()
  const stageName = (key: string) => createHash('sha256').update(key).digest('hex').slice(0, 24) + '.webp'
  const staged = (key: string, make: (dst: string) => Promise<unknown>): Promise<string> => {
    let p = made.get(key)
    if (!p) {
      const dst = path.join(staging, stageName(key))
      p = pool(() => make(dst)).then(() => dst)
      made.set(key, p)
    }
    return p
  }
  const place = (from: string, to: string) => {
    try {
      linkSync(from, to)
    } catch {
      copyFileSync(from, to)
    }
  }
  log(`Making blind copies of ${pics.size} pictures for ${copies.length} places they are shown.`)
  try {
    await Promise.all(copies.map(async c => {
      const outline = c.pic.kind === 'refbox' ? c.pic.outline : null
      const g = await staged(`${c.pic.key}|g`, dst => blindCopy(opts.ffmpeg, c.pic.src, dst, GRID_EDGE, GRID_QUALITY, { outline, nice: opts.nice }))
      const f = await staged(`${c.pic.key}|f|${c.full}`, dst => blindCopy(opts.ffmpeg, c.pic.src, dst, c.full, FULL_QUALITY, { outline, upscale: true, nice: opts.nice }))
      place(g, path.join(view, `${c.token}-g.webp`))
      place(f, path.join(view, `${c.token}-f.webp`))
    }))
  } finally {
    rmSync(staging, { recursive: true, force: true })
  }

  // ------------------------------------------------------------------ write --
  const slotsInfo: Sealed['slots'] = {}
  for (const r of rows) {
    const hit = slotIndex.get(r.slot)
    if (hit && !slotsInfo[r.slot]) {
      slotsInfo[r.slot] = {
        block: hit.slot.block, ...(hit.slot.second ? { second: hit.slot.second } : {}), innocent: hit.slot.innocent !== false,
        measuredOnly: !!hit.slot.measuredOnly, suite: hit.suite.id,
      }
    }
  }
  for (const n of x.na) {
    if (slotsInfo[n.slot]) continue
    const hit = slotIndex.get(n.slot)
    const chain = chains.get(n.slot)?.chain
    if (hit) slotsInfo[n.slot] = { block: hit.slot.block, innocent: hit.slot.innocent !== false, measuredOnly: !!hit.slot.measuredOnly, suite: hit.suite.id }
    else if (chain) slotsInfo[n.slot] = { block: chain.block, innocent: true, measuredOnly: false, suite: chains.get(n.slot)?.suite.id ?? '' }
  }
  const sealed: Sealed = {
    v: 1,
    run,
    study,
    sealedAt: now(),
    suites: runSuites.map(s => ({ id: s.id, version: s.version })),
    tokens,
    gridOf,
    letters,
    pairOf,
    checkOf,
    sets: setsInfo,
    slots: slotsInfo,
    cells: Object.fromEntries(cellInfo),
    contestants: Object.fromEntries(contestantInfo),
    na: x.na,
    notMade,
  }
  if (held.length) (sealed as Sealed & { held?: string[] }).held = held
  const sealedText = JSON.stringify(sealed) + '\n'
  const sealedSha = createHash('sha256').update(sealedText).digest('hex')
  const items: Items = { v: 1, run, sealedSha, sets: itemSets, pairs: itemPairs, checks }
  if (Object.keys(sweepKey.pairs).length) writeAtomic(path.join(dir, SWEEP_KEY_FILE), JSON.stringify(sweepKey, null, 1) + '\n')
  // The items wait beside the key until the key is in place: a run is sealed once sealed.json is written,
  // and nothing can be judged before items.json appears.
  rmSync(itemsFile, { force: true })
  writeAtomic(pendingItems, JSON.stringify(items, null, 1) + '\n')
  writeAtomic(sealedFile, sealedText)
  renameSync(pendingItems, itemsFile)
  log(`Sealed ${run}: ${itemSets.length} sets, ${countItems(items)} items${held.length ? `, ${held.length} sets held back for pictures still to come` : ''}.`)
  return { sealedSha, sets: itemSets.length, items: countItems(items) }
}

/** Grids, fixed pairs and content checks (tie-break pairs and picks come during judging). */
export function countItems(items: Items): number {
  return items.sets.reduce((n, s) => n + s.grids.length, 0) + items.pairs.length + items.checks.length
}

/** The ledger entries of a run, for the report. */
export function ledgerOf(env: LabEnv, run: string): LedgerEntry[] {
  return readLedger(runDir(env, run))
}

/** A file's rel under outputs, as the runner names it. */
export const relOf = relOfFile
