/**
 * A run's plan: which pictures, in what order, which are already made, which
 * tests a model cannot take and why, which tests wait for a photo, and how
 * long the rest should take. Making a plan sends nothing anywhere.
 *
 * Every figure here is an estimate and is labelled so. The timings are seeded
 * from the archive's own medians (what ComfyUI measured for the user's past
 * pictures, scaled to 28 steps) where the archive has them, and guessed from
 * a sibling model where it does not; each entry says which.
 */
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, renameSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import type { ApiWorkflow } from '../../src/lib/comfy.ts'
import { canonicalJson, SHIPPED_SUITES, type Blocked, type Expansion } from './cells.ts'
import { runDir, type LabEnv } from './env.ts'
import type { Cell, DoneCell, NA, Op, Suite } from './types.ts'

// ---------------------------------------------------------------------------
// Suites by name
// ---------------------------------------------------------------------------

/** The shipped suites by id. */
export const SUITES: Readonly<Record<string, Suite>> = Object.fromEntries(SHIPPED_SUITES.map((s) => [s.id, s]))

const ALIASES: Record<string, string> = {
  cal: 'calibration',
  core: 'first-pass-core',
  'first-pass': 'first-pass-core',
  edit: 'ext-edit',
  range: 'ext-range',
}

/** A shipped suite by id or short name ('core', 'calibration', 'ext-edit'), or null. */
export function suiteById(id: string): Suite | null {
  return SUITES[id] ?? SUITES[ALIASES[id] ?? ''] ?? null
}

/** The run each shipped suite is planned as by default. */
export const DEFAULT_RUNS: Readonly<Record<string, string>> = {
  calibration: 'cal-1',
  'first-pass-core': 'core-1',
  'ext-edit': 'exta-1',
  'ext-range': 'extb-1',
}

// ---------------------------------------------------------------------------
// Timings
// ---------------------------------------------------------------------------

/**
 * How long a model takes: milliseconds per sampling step at 1024 x 1024 with
 * the model already loaded, and roughly how long loading it takes.
 */
export type Timing = { msPerStep: number; loadMs: number; source: 'archive' | 'guess' | 'measured'; note: string }
export type Timings = Record<string, Timing>

/**
 * The seed table, by weight file. 'archive' rows are the median ComfyUI time
 * of the user's own text-to-image pictures in the archive on 2026-09-24,
 * divided by their step count and scaled to 1024 x 1024. 'guess' rows have no
 * archive picture to go on. Load times are all guesses from file size.
 */
export const SEED_TIMINGS: Timings = {
  'NoobAI-XL-v1.1.safetensors': { msPerStep: 376, loadMs: 15_000, source: 'archive', note: '4 pictures, 28 steps, 1024 x 1216' },
  'semiRealIllustrious_v40.safetensors': { msPerStep: 371, loadMs: 15_000, source: 'archive', note: '4 pictures, 30 steps, 832 x 1216' },
  'ponyDiffusionV6XL.safetensors': { msPerStep: 402, loadMs: 15_000, source: 'archive', note: '4 pictures, 30 steps, 1024 x 1024' },
  'waiMatureIllustrious_v30.safetensors': { msPerStep: 422, loadMs: 15_000, source: 'archive', note: '4 pictures, 28 steps, 832 x 1216' },
  'oneObsession_anima29BV1.safetensors': { msPerStep: 1417, loadMs: 15_000, source: 'archive', note: '5 pictures, 30 steps, 1024 x 1024' },
  'miaomiaoHarem_29BBETA10.safetensors': { msPerStep: 1405, loadMs: 15_000, source: 'archive', note: '4 pictures, 30 steps, 1024 x 1024' },
  'miaomiaoRealskin_anima13.safetensors': { msPerStep: 966, loadMs: 12_000, source: 'archive', note: '4 pictures, 32 steps, 1024 x 1024' },
  'flux-2-klein-4b-fp8.safetensors': { msPerStep: 680, loadMs: 12_000, source: 'archive', note: '8 pictures, 26 and 32 steps, 1024 x 1024' },
  'Chroma1-HD-fp8mixed.safetensors': { msPerStep: 2529, loadMs: 25_000, source: 'archive', note: '4 pictures, 32 steps, 1024 x 1024' },
  'qwen_image_2.1_int8_convrot.safetensors': { msPerStep: 2510, loadMs: 25_000, source: 'archive', note: '3 pictures, 32 steps, 1024 x 1024' },
  'Z-Image-Turbo-fp8mix.safetensors': { msPerStep: 1037, loadMs: 18_000, source: 'archive', note: '4 pictures, 32 steps, 1024 x 1024' },
  'moodyCutieMixKrea2_v50_int8.safetensors': {
    msPerStep: 2000,
    loadMs: 30_000,
    source: 'guess',
    note: 'no archive picture; guessed from its 12.8 GiB of weights, near Qwen-Image 2.1',
  },
  'Z-Image-Base-bf16.safetensors': {
    msPerStep: 2075,
    loadMs: 25_000,
    source: 'guess',
    note: 'no archive picture; twice Z-Image Turbo, because guidance at CFG 4 runs two passes a step',
  },
  'qwen-image-edit-2511-Q4_K_M.gguf': {
    msPerStep: 5000,
    loadMs: 35_000,
    source: 'guess',
    note: 'no archive picture; twice Qwen-Image 2.1, because guidance at CFG 4 runs two passes a step, and more for the reference',
  },
}

const FALLBACK: Timing = { msPerStep: 1500, loadMs: 20_000, source: 'guess', note: 'no figure for this file' }

/** How much work an operation is, as a multiple of one plain generation (the app's own cost figures). */
const OP_COST: Record<Op, number> = {
  t2i: 1,
  i2i: 1,
  edit: 1,
  region: 1,
  face: 2,
  hand: 2,
  hires: 2.35,
  detailOnPicture: 1,
}

/** The pixels one pass works on, as a share of 1024 x 1024. */
function pixelShare(cell: Cell): number {
  const px = cell.width * cell.height
  if (cell.op === 'hires') return px / 2.25 / 1048576
  // A region is re-rendered at 1024 on its long side; a face pass works on a crop at most 1024.
  if (cell.op === 'region' || cell.op === 'detailOnPicture') return 1
  return px / 1048576
}

/** Estimated seconds for one picture, the model already loaded. */
export function cellSeconds(cell: Cell, timings: Timings = SEED_TIMINGS): number {
  const t = timings[cell.file] ?? FALLBACK
  return (t.msPerStep * cell.steps * pixelShare(cell) * OP_COST[cell.op]) / 1000
}

/**
 * Measured timings from finished cells: the median warm, uncached time per
 * step of each file's plain text-to-image pictures, scaled to 1024 x 1024.
 * Files with none keep their seed row.
 */
export function measuredTimings(cells: readonly Cell[], done: ReadonlyMap<string, DoneCell>, base: Timings = SEED_TIMINGS): Timings {
  const per = new Map<string, number[]>()
  for (const c of cells) {
    const d = done.get(c.cellId)
    if (!d || d.cold || d.cached || !(d.durationMs > 0) || c.op !== 't2i') continue
    per.set(c.file, [...(per.get(c.file) ?? []), d.durationMs / c.steps / pixelShare(c)])
  }
  const out: Timings = { ...base }
  for (const [file, xs] of per) {
    xs.sort((a, b) => a - b)
    const mid = xs.length % 2 ? xs[(xs.length - 1) / 2] : (xs[xs.length / 2 - 1] + xs[xs.length / 2]) / 2
    out[file] = { msPerStep: Math.round(mid), loadMs: (base[file] ?? FALLBACK).loadMs, source: 'measured', note: `${xs.length} warm pictures in the lab` }
  }
  return out
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

export type Plan = {
  v: 1
  run: string
  study: string
  suites: { id: string; version: number; sha: string }[]
  createdAt: number
  /** Cell ids still to make, in send order: model by model, each chain step after the picture it starts from. */
  order: string[]
  /** Cell ids already made by an earlier run (cells.jsonl), not made again. */
  reused: string[]
  na: NA[]
  estimate: {
    pictures: number
    newPictures: number
    seconds: number
    byModel: Record<string, { pictures: number; seconds: number }>
  }
  // Beyond the contract's list, additive:
  /** Every cell row of this run, reused ones included, so sealing never has to expand the suites again. */
  cells: Cell[]
  /** Tests that wait for a reference photo or its rectangle. While any is listed the run must not start. */
  blocked: Blocked[]
  /** Cells drawn only because a chain starts from them (they belong to another night's suite). */
  context: string[]
  /** Where each timing came from, per model key: 'archive', 'guess' or 'measured'. */
  timingSource: Record<string, Timing['source']>
  /** The words each cell sends without its prefix (the runner job's `prompt`). Never shown on the phone. */
  prompts: Record<string, string>
}

/** A short, stable fingerprint of a suite's content. */
export function suiteSha(s: Suite): string {
  return createHash('sha256').update(canonicalJson(s)).digest('hex').slice(0, 16)
}

const OP_ORDER: Op[] = ['t2i', 'face', 'hand', 'hires', 'i2i', 'region', 'edit', 'detailOnPicture']

/**
 * The send order of the cells still to make. A cell comes after the picture
 * it starts from (depth first), then model by model in the order the suites
 * name them, so each model loads once per depth; within a model, operation by
 * operation, then slot, then variant, then seed.
 */
export function sendOrder(cells: readonly Cell[], suites: readonly Suite[], skip: ReadonlySet<string> = new Set()): string[] {
  const first = new Map<string, Cell>()
  for (const c of cells) if (!first.has(c.cellId)) first.set(c.cellId, c)
  const depth = new Map<string, number>()
  const depthOf = (id: string, seen = new Set<string>()): number => {
    if (depth.has(id)) return depth.get(id) as number
    const c = first.get(id)
    if (!c?.upstream || seen.has(id) || !first.has(c.upstream)) return 0
    seen.add(id)
    const d = depthOf(c.upstream, seen) + 1
    depth.set(id, d)
    return d
  }
  const modelRank = new Map<string, number>()
  for (const c of cells) if (!modelRank.has(c.file)) modelRank.set(c.file, modelRank.size)
  const suiteRank = (id: string) => {
    const i = suites.findIndex((s) => s.id === id)
    return i < 0 ? -1 : i
  }
  const slotRank = (c: Cell) => {
    const s = suites.find((x) => x.id === c.suite) ?? SHIPPED_SUITES.find((x) => x.id === c.suite)
    return s ? s.slots.findIndex((x) => x.id === c.slot) : 0
  }
  const chainRank = (c: Cell) => {
    if (!c.chain) return -1
    const s = suites.find((x) => x.id === c.suite)
    return s ? s.chains.findIndex((x) => x.id === c.chain) : 0
  }
  const key = (c: Cell): number[] => [
    depthOf(c.cellId),
    modelRank.get(c.file) ?? 0,
    OP_ORDER.indexOf(c.op),
    c.chain ? 1 : 0,
    suiteRank(c.suite),
    chainRank(c),
    slotRank(c),
    c.steps,
    c.condition === 'home sampler' ? 0 : 1,
    c.seed,
  ]
  const todo = [...first.values()].filter((c) => !skip.has(c.cellId))
  const keys = new Map(todo.map((c) => [c.cellId, key(c)]))
  todo.sort((a, b) => {
    const ka = keys.get(a.cellId) as number[]
    const kb = keys.get(b.cellId) as number[]
    for (let i = 0; i < ka.length; i++) if (ka[i] !== kb[i]) return ka[i] - kb[i]
    return a.cellId < b.cellId ? -1 : a.cellId > b.cellId ? 1 : 0
  })
  return todo.map((c) => c.cellId)
}

/** The model key a cell's weight file has in the suites, for figures by model. */
function keyOfFile(suites: readonly Suite[]): (file: string) => string {
  const map = new Map<string, string>()
  for (const s of [...suites, ...SHIPPED_SUITES]) for (const [k, m] of Object.entries(s.models)) if (!map.has(m.file)) map.set(m.file, k)
  return (file) => map.get(file) ?? file
}

/**
 * Make a run's plan from an expansion. `cells` and `na` are expand()'s, as is
 * `blocked` when given; `done` is cells.jsonl by cell id; `timings` the table
 * to estimate with. Sends nothing.
 */
export function makePlan(
  run: string,
  suites: readonly Suite[],
  cells: readonly Cell[],
  na: readonly NA[],
  done: ReadonlyMap<string, DoneCell>,
  timings: Timings = SEED_TIMINGS,
  extra: {
    blocked?: readonly Blocked[]
    context?: ReadonlySet<string> | readonly string[]
    prompts?: ReadonlyMap<string, string>
  } = {},
): Plan {
  const studies = [...new Set(suites.map((s) => s.study))]
  if (studies.length !== 1) throw new Error(`A run covers one study; these suites name ${studies.length ? studies.join(', ') : 'none'}.`)
  const ids = [...new Set(cells.map((c) => c.cellId))]
  const reused = ids.filter((id) => done.has(id))
  const order = sendOrder(cells, suites, new Set(reused))

  const byId = new Map<string, Cell>()
  for (const c of cells) if (!byId.has(c.cellId)) byId.set(c.cellId, c)
  const keyOf = keyOfFile(suites)
  const byModel: Plan['estimate']['byModel'] = {}
  const timingSource: Plan['timingSource'] = {}
  let seconds = 0
  let lastFile: string | null = null
  for (const id of order) {
    const c = byId.get(id) as Cell
    const t = timings[c.file] ?? FALLBACK
    let s = cellSeconds(c, timings)
    if (c.file !== lastFile) s += t.loadMs / 1000
    lastFile = c.file
    seconds += s
    const k = keyOf(c.file)
    const row = (byModel[k] ??= { pictures: 0, seconds: 0 })
    row.pictures += 1
    row.seconds += s
    timingSource[k] = t.source
  }
  for (const row of Object.values(byModel)) row.seconds = Math.round(row.seconds)

  const context = extra.context ? [...extra.context].filter((id) => ids.includes(id)) : []
  return {
    v: 1,
    run,
    study: studies[0],
    suites: suites.map((s) => ({ id: s.id, version: s.version, sha: suiteSha(s) })),
    createdAt: Date.now(),
    order,
    reused,
    na: [...na],
    estimate: { pictures: ids.length, newPictures: order.length, seconds: Math.round(seconds), byModel },
    cells: [...cells],
    blocked: [...(extra.blocked ?? [])],
    context,
    timingSource,
    prompts: Object.fromEntries(ids.filter((id) => extra.prompts?.has(id)).map((id) => [id, extra.prompts?.get(id) as string])),
  }
}

/** makePlan straight from an expansion. */
export function planFrom(run: string, suites: readonly Suite[], x: Expansion, done: ReadonlyMap<string, DoneCell>, timings: Timings = SEED_TIMINGS): Plan {
  return makePlan(run, suites, x.cells, x.na, done, timings, { blocked: x.blocked, context: x.context, prompts: x.prompts })
}

// ---------------------------------------------------------------------------
// Words
// ---------------------------------------------------------------------------

const REF_WORDS: Record<string, string> = {
  cat: 'the photo of your cat',
  scene: 'the room or table photo',
}

/**
 * Why this run must not start yet, in plain words, or null when it may. A
 * night that needs a reference photo (or the rectangle drawn on it) refuses
 * until it is there, and says where to put it.
 */
export function startRefusal(plan: Pick<Plan, 'blocked'>, env?: Pick<LabEnv, 'labDir'>): string | null {
  if (!plan.blocked.length) return null
  const groups = new Map<string, { ref: string | null; kind: Blocked['kind']; why: string; tests: Set<string> }>()
  for (const b of plan.blocked) {
    const k = `${b.ref ?? ''}|${b.kind}`
    const row = groups.get(k) ?? { ref: b.ref, kind: b.kind, why: '', tests: new Set<string>() }
    // A test's own reason; a chain's only repeats the test it starts from.
    if (!row.why && !b.chain) row.why = b.why
    row.tests.add(b.chain ?? b.slot)
    groups.set(k, row)
  }
  const where = env ? path.join(env.labDir, 'refs') : "the lab's refs folder"
  const lines: string[] = []
  for (const { ref, kind, why, tests } of groups.values()) {
    const list = [...tests].join(', ')
    const one = tests.size === 1
    const photo = ref ? (REF_WORDS[ref] ?? `the ${ref} photo`) : 'a photo'
    switch (kind) {
      case 'photo':
        lines.push(`${list} ${one ? 'needs' : 'need'} ${photo}, which is not in the lab yet. Add it from your phone on the lab page (Photos), or drop the file into ${where}.`)
        break
      case 'rectangle':
        lines.push(`${list} ${one ? 'waits' : 'wait'} for a rectangle on ${photo}. Draw it on the lab page (Photos).`)
        break
      case 'description':
        // Every test of one photo shares its description, so one reason covers them all.
        lines.push(`${list} cannot use the description of ${photo}. ${why || 'Describe only the cat or object in the photo, with no one in it. Change it on the lab page (Photos).'}`)
        break
      case 'upstream':
        lines.push(`${list} cannot be made yet, because the picture ${one ? 'it starts' : 'they start'} from cannot.`)
        break
    }
  }
  return `This night cannot start yet.\n${lines.join('\n')}\nThen plan the night again, so the change is part of it.`
}

const hours = (s: number) => (s < 5400 ? `about ${Math.max(1, Math.round(s / 60))} minutes` : `about ${(s / 3600).toFixed(1)} hours`)

/** The plan in a few plain lines, for `lab plan`. Sends nothing. */
export function describePlan(plan: Plan): string {
  const e = plan.estimate
  const lines = [
    `Run ${plan.run} (${plan.suites.map((s) => s.id).join(', ')}): ${e.pictures} pictures, ${e.newPictures} to make, ${plan.reused.length} already made by an earlier run.`,
    `Estimated time: ${hours(e.seconds)} (an estimate from the archive and guesses, not a measurement).`,
  ]
  for (const [k, row] of Object.entries(e.byModel)) {
    lines.push(`  ${k}: ${row.pictures} pictures, ${hours(row.seconds)}${plan.timingSource[k] === 'guess' ? ' (guessed: no archive figure)' : ''}`)
  }
  if (plan.na.length) {
    lines.push(`Not applicable, from the app's registry (never scored, never averaged):`)
    const grouped = new Map<string, string[]>()
    for (const n of plan.na) {
      const k = `${n.model}: ${n.reason}`
      grouped.set(k, [...(grouped.get(k) ?? []), n.slot])
    }
    for (const [k, slots] of grouped) lines.push(`  ${k} (${slots.join(', ')})`)
  }
  const refusal = startRefusal(plan)
  if (refusal) lines.push(refusal)
  lines.push('Nothing is sent until you press Start on the lab page or run lab/lab start.')
  return lines.join('\n')
}

// ---------------------------------------------------------------------------
// On disk
// ---------------------------------------------------------------------------

/** Where a cell's graph and metadata are kept, shared by every run. */
export function cellFile(env: Pick<LabEnv, 'labDir'>, cellId: string): string {
  if (!/^[0-9a-f]{16}$/.test(cellId)) throw new Error(`"${cellId}" is not a cell id.`)
  return path.join(env.labDir, 'cells', `${cellId}.json`)
}

function writeAtomic(file: string, text: string) {
  mkdirSync(path.dirname(file), { recursive: true })
  const tmp = `${file}.${process.pid}.tmp`
  writeFileSync(tmp, text, { mode: 0o600 })
  renameSync(tmp, file)
}

/**
 * Write runs/<run>/plan.json and cells/<cellId>.json (the graph with its
 * placeholders, and the first cell row that names it) for every cell of the
 * plan. Writes only under the lab folder. Sends nothing.
 */
export function savePlan(env: LabEnv, plan: Plan, graphs: ReadonlyMap<string, ApiWorkflow>): string {
  const dir = runDir(env, plan.run)
  const written = new Set<string>()
  for (const c of plan.cells) {
    if (written.has(c.cellId)) continue
    written.add(c.cellId)
    const g = graphs.get(c.cellId)
    if (!g) throw new Error(`The plan names cell ${c.cellId}, whose graph is missing.`)
    // The same id is the same graph, so a file an earlier run wrote is kept as it is.
    const file = cellFile(env, c.cellId)
    if (!existsSync(file)) writeAtomic(file, JSON.stringify({ v: 1, cell: c, graph: g }))
  }
  const file = path.join(dir, 'plan.json')
  writeAtomic(file, JSON.stringify(plan, null, 1))
  return file
}
