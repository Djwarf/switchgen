/**
 * Suites into cells: one cell per picture, identified by the hash of the
 * graph that makes it.
 *
 * The same settings give the same graph, so the same id, in every run: the
 * calibration's 28-step pictures are the core's pictures, the prompt-style
 * sentences are the core's sentences, and a chain that starts from a core
 * picture starts from that very file. A registry change gives a new graph and
 * so a new id, and nothing stale is reused.
 *
 * A graph is hashed with its SaveImage prefix blanked and with placeholders
 * where pictures go in ('ref:<photo sha12>', 'mask:<mask sha12>',
 * 'cell:<cellId>'), so a new photo or a new mask is a new id.
 * finalizeGraph puts the real paths in just before sending, and verifyCell
 * checks the result is still exactly the graph that was hashed.
 */
import { createHash } from 'node:crypto'
import type { ApiWorkflow } from '../../src/lib/comfy.ts'
import { DETAIL_TUNING, DETECTORS, capabilitiesOf, hiresSize, hiresStepsFor, type DerivedDef } from '../../src/lib/refine.ts'
import { IMG2IMG, familyOwning, type FamilyDef } from '../../src/lib/workflows.ts'
import { CHAIN_TOKEN } from '../../server/runner/comfyRecord.mjs'
import calibration from '../suites/calibration.ts'
import extEdit from '../suites/ext-edit.ts'
import extRange from '../suites/ext-range.ts'
import firstPass from '../suites/first-pass.ts'
import { annotated, cellPrefix, maskRel, refRel } from './env.ts'
import { I2I_DEFAULTS, PLACEHOLDER, REGION_DEFAULTS, buildGraph, placeholdersOf, shapeFor, type CellSpec } from './graphs.ts'
import { naReason, negativeReason } from './na.ts'
import { labParams } from './params.ts'
import { chainStepText, checkSuite, childWords, descriptionProblem, findSlot, setOf } from './suite.ts'
import type { Cell, NA, Op, RefInfo, RefRect, Slot, Suite } from './types.ts'

export { CHAIN_TOKEN }

/** The four suites of the first study, in the order of their nights. */
export const SHIPPED_SUITES: readonly Suite[] = [calibration, firstPass, extEdit, extRange]

/** A graph ComfyUI takes at most this big (the runner's limit). */
export const GRAPH_MAX_BYTES = 1048576

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/** JSON with every object's keys sorted, so equal graphs give equal text. */
export function canonicalJson(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(canonicalJson).join(',')}]`
  if (v && typeof v === 'object') {
    const o = v as Record<string, unknown>
    return `{${Object.keys(o)
      .filter((k) => o[k] !== undefined)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${canonicalJson(o[k])}`)
      .join(',')}}`
  }
  return JSON.stringify(v) ?? 'null'
}

const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T

/** The first 16 hex characters of the sha256 of the graph, SaveImage prefixes blanked. */
export function cellIdOf(graph: ApiWorkflow): string {
  const g = clone(graph)
  for (const node of Object.values(g)) {
    if (node.class_type === 'SaveImage' && 'filename_prefix' in node.inputs) node.inputs.filename_prefix = ''
  }
  return createHash('sha256').update(canonicalJson(g)).digest('hex').slice(0, 16)
}

// ---------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------

/** The reference photos by id, from a list or from an id-keyed index. */
export type RefIndex = Record<string, RefInfo> | readonly RefInfo[]

function refMap(r: RefIndex | undefined): Map<string, RefInfo> {
  const out = new Map<string, RefInfo>()
  if (!r) return out
  const list = Array.isArray(r) ? (r as readonly RefInfo[]) : Object.entries(r as Record<string, RefInfo>).map(([id, v]) => ({ ...v, id: v.id ?? id }))
  for (const info of list) out.set(info.id, info)
  return out
}

/**
 * The hash a reference's mask goes by in its placeholder ('mask:<hash>') and
 * its copy's name (.lab/refs/<hash>.mask.png): the mask file's own hash, so a
 * rectangle drawn again gives a new cell id even when both rectangles crop to
 * the same frame. An index that gives no mask hash falls back to the photo's.
 */
export function maskKey(r: Pick<RefInfo, 'sha12' | 'maskSha12'>): string {
  return r.maskSha12 || r.sha12
}

/**
 * The files each placeholder stands for, under outputs: 'ref:<sha12>' is the
 * photo's copy, 'mask:<maskKey>' its mask's. For finalizeGraph's `refs`.
 */
export function refBindings(refs: RefIndex): Record<string, string> {
  const out: Record<string, string> = {}
  for (const r of refMap(refs).values()) {
    out[`ref:${r.sha12}`] = refRel(r.sha12, r.ext)
    if (r.mask) out[`mask:${maskKey(r)}`] = maskRel(maskKey(r))
  }
  return out
}

/**
 * Where a rectangle sits in the picture, in the words an instruction uses:
 * "lower left", "centre", "upper right". From the rectangle's middle, in
 * thirds of the upright picture.
 */
export function positionWords(rect: RefRect, size: { width: number; height: number }): string {
  const cx = (rect.x + rect.w / 2) / Math.max(1, size.width)
  const cy = (rect.y + rect.h / 2) / Math.max(1, size.height)
  const col = cx < 1 / 3 ? 0 : cx > 2 / 3 ? 2 : 1
  const row = cy < 1 / 3 ? 0 : cy > 2 / 3 ? 2 : 1
  return [
    ['upper left', 'top centre', 'upper right'],
    ['middle left', 'centre', 'middle right'],
    ['lower left', 'bottom centre', 'lower right'],
  ][row][col]
}

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

/** ComfyUI's FluxKontextImageScale sizes, which the editing graph scales its input to. */
const KONTEXT_SIZES: [number, number][] = [
  [672, 1568], [688, 1504], [720, 1456], [752, 1392], [800, 1328], [832, 1248], [880, 1184], [944, 1104], [1024, 1024],
  [1104, 944], [1184, 880], [1248, 832], [1328, 800], [1392, 752], [1456, 720], [1504, 688], [1568, 672],
]

/** ImageScaleToTotalPixels: the source at `megapixels` (of 1024 x 1024), both sides on a 16 grid. */
function fitPixels(src: { width: number; height: number }, megapixels: number): { width: number; height: number } {
  const scale = Math.sqrt((megapixels * 1024 * 1024) / Math.max(1, src.width * src.height))
  const side = (n: number) => Math.max(16, Math.round((n * scale) / 16) * 16)
  return { width: side(src.width), height: side(src.height) }
}

function kontextSize(src: { width: number; height: number }): { width: number; height: number } {
  const ratio = src.width / Math.max(1, src.height)
  let best = KONTEXT_SIZES[8]
  for (const s of KONTEXT_SIZES) if (Math.abs(ratio - s[0] / s[1]) < Math.abs(ratio - best[0] / best[1])) best = s
  return { width: best[0], height: best[1] }
}

/** The size the picture is expected to come out at. The true size is read from the file. */
function expectedSize(spec: CellSpec): { width: number; height: number } {
  const p = spec.params
  const src = spec.sourceSize ?? { width: p.width, height: p.height }
  switch (spec.op) {
    case 't2i':
    case 'face':
    case 'hand':
      return { width: p.width, height: p.height }
    case 'hires':
      return hiresSize({ width: p.width, height: p.height })
    case 'i2i':
      return fitPixels(src, spec.megapixels ?? I2I_DEFAULTS.megapixels)
    case 'edit':
      return kontextSize(src)
    case 'region':
    case 'detailOnPicture':
      return { width: src.width, height: src.height }
  }
}

// ---------------------------------------------------------------------------
// Expansion
// ---------------------------------------------------------------------------

/** A test that cannot be planned yet, because a reference photo or its mask is missing. */
export type Blocked = {
  suite: string
  slot: string
  model: string | null
  chain: string | null
  ref: string | null
  /** What is missing: the photo, the rectangle on it, a usable description, or the picture a chain starts from. */
  kind: 'photo' | 'rectangle' | 'description' | 'upstream'
  why: string
}

export type Expansion = {
  /** One row per picture per test. A picture two tests share (a reused cell) has a row in each. */
  cells: Cell[]
  /** Each cell's graph, once, with placeholders and a blank-able prefix. */
  graphs: Map<string, ApiWorkflow>
  na: NA[]
  /** Tests waiting for a reference photo or a mask; their cells are not in `cells`. */
  blocked: Blocked[]
  /** Cells pulled in only because a chain starts from them; they belong to another suite of the study. */
  context: Set<string>
  /**
   * The words each cell sends, without the model's trained prefix: the job's
   * `prompt` for the runner. A model's own wording (the editing model's
   * instruction) is here, so this never goes to the phone; the phone shows
   * the slot's neutral task.
   */
  prompts: Map<string, string>
}

/** The operation a model runs for a slot: the editing model edits wherever it has its own wording. */
export function opFor(slot: Slot, key: string, suite: Suite): Op {
  if (suite.models[key]?.role === 'edit' && slot.perModelText?.[key] !== undefined) return 'edit'
  return slot.op
}

/** The model keys a slot names, in order. */
export function resolveModels(slot: Slot, suite: Suite): string[] {
  const m = slot.models
  if (Array.isArray(m)) return [...m]
  const def = (k: string): FamilyDef | null => familyOwning(suite.models[k]?.file ?? '')
  const core = suite.core.filter((k) => !!def(k))
  switch (m) {
    case 'core':
      return [...suite.core]
    case 'all':
      return [...suite.core, ...Object.keys(suite.models).filter((k) => !suite.core.includes(k) && suite.models[k].role !== 'edit')]
    case 'edit':
      return Object.keys(suite.models).filter((k) => suite.models[k].role === 'edit')
    case 'negative-capable':
      return core.filter((k) => !negativeReason(def(k) as FamilyDef, suite.models[k].file))
    case 'i2i-capable':
      return core.filter((k) => !!IMG2IMG[(def(k) as FamilyDef).id])
    case 'refine-capable':
      return core.filter((k) => capabilitiesOf(def(k) as FamilyDef).refine)
  }
  return []
}

type Variant = { tag: string; steps?: number; sampler?: 'home'; set?: string; condition?: string }
const PLAIN: Variant = { tag: '' }

class Expander {
  readonly cells: Cell[] = []
  readonly graphs = new Map<string, ApiWorkflow>()
  readonly na: NA[] = []
  readonly blocked: Blocked[] = []
  readonly prompts = new Map<string, string>()
  private readonly rows = new Set<string>()
  private readonly naSeen = new Set<string>()
  private readonly blockedSeen = new Set<string>()
  private readonly own = new Set<string>()
  private readonly borrowed = new Set<string>()
  private readonly memo = new Map<string, Cell | null>()

  constructor(
    private readonly suites: readonly Suite[],
    private readonly context: readonly Suite[],
    private readonly refs: Map<string, RefInfo>,
  ) {}

  contextIds(): Set<string> {
    return new Set([...this.borrowed].filter((id) => !this.own.has(id)))
  }

  private noteNA(slot: string, model: string, reason: string) {
    const k = `${slot}|${model}`
    if (this.naSeen.has(k)) return
    this.naSeen.add(k)
    this.na.push({ slot, model, reason })
  }

  private block(b: Blocked) {
    const k = `${b.suite}|${b.slot}|${b.model}|${b.chain}|${b.why}`
    if (this.blockedSeen.has(k)) return
    this.blockedSeen.add(k)
    this.blocked.push(b)
  }

  private add(cell: Cell, graph: ApiWorkflow, prompt: string, isOwn: boolean) {
    const row = [cell.cellId, cell.suite, cell.slot, cell.set, cell.chain, cell.chainStep, cell.condition].join('|')
    if (!this.graphs.has(cell.cellId)) this.graphs.set(cell.cellId, graph)
    if (!this.prompts.has(cell.cellId)) this.prompts.set(cell.cellId, prompt)
    if (isOwn) this.own.add(cell.cellId)
    else this.borrowed.add(cell.cellId)
    if (this.rows.has(row)) return
    this.rows.add(row)
    this.cells.push(cell)
  }

  /** One cell of a slot, built once however many tests ask for it. */
  cellFor(suite: Suite, slot: Slot, key: string, seed: number, v: Variant): Cell | null {
    const memoKey = [suite.id, slot.id, key, seed, v.tag].join('|')
    if (this.memo.has(memoKey)) return this.memo.get(memoKey) ?? null
    this.memo.set(memoKey, null)
    const cell = this.make(suite, slot, key, seed, v)
    this.memo.set(memoKey, cell)
    return cell
  }

  private make(suite: Suite, slot: Slot, key: string, seed: number, v: Variant): Cell | null {
    const isOwn = this.suites.includes(suite)
    const spec = suite.models[key]
    const def = spec ? familyOwning(spec.file) : null
    if (!spec || !def) throw new Error(`Slot ${slot.id} names model ${key}, which is not a registry file.`)
    const file = spec.file
    const op = opFor(slot, key, suite)
    const na = naReason(file, op, slot)
    if (na) {
      if (isOwn) this.noteNA(slot.id, key, na)
      return null
    }

    let text = slot.perModelText?.[key] ?? slot.text
    let image: string | undefined
    let mask: string | undefined
    let rect: RefRect | undefined
    let sourceSize: { width: number; height: number } | undefined
    let upstream: string | null = null
    const refs: string[] = []
    const blocked = (ref: string | null, kind: Blocked['kind'], why: string) => {
      this.block({ suite: suite.id, slot: slot.id, model: key, chain: null, ref, kind, why })
      return null
    }

    const src = slot.source
    if (src && 'ref' in src) {
      const id = src.ref
      const ref = this.refs.get(id)
      if (!ref) return blocked(id, 'photo', `waits for the ${id} photo, which is not in the lab yet`)
      const wantsRect = op === 'region' || text.includes('{position}')
      if (wantsRect && !(ref.mask && ref.rect)) return blocked(id, 'rectangle', `waits for a rectangle to be drawn on the ${id} photo`)
      const describe = (ref.describe ?? '').trim() || suite.refs[id]?.describe || ''
      // The description, the user's own words included, names only the cat or object: no person at all.
      const problem = text.includes('{describe}') ? descriptionProblem(id, describe) : null
      if (problem) return blocked(id, 'description', `${problem} Change it on the lab page (Photos).`)
      text = text.split('{describe}').join(describe)
      if (ref.rect) text = text.split('{position}').join(positionWords(ref.rect, ref))
      image = `ref:${ref.sha12}`
      if (op === 'region') {
        mask = `mask:${maskKey(ref)}`
        rect = ref.rect ?? undefined
      }
      sourceSize = { width: ref.width, height: ref.height }
      refs.push(id)
    } else if (src && 'cell' in src) {
      const found = findSlot(src.cell.slot, suite, this.context)
      if (!found) throw new Error(`Slot ${slot.id} starts from slot ${src.cell.slot}, which no suite of study ${suite.study} has.`)
      const up = this.cellFor(found.suite, found.slot, src.cell.model, seed, PLAIN)
      if (!up) return blocked(null, 'upstream', `its starting picture (${src.cell.slot}) cannot be made yet`)
      image = `cell:${up.cellId}`
      upstream = up.cellId
      sourceSize = { width: up.width, height: up.height }
      refs.push(...up.refs)
    } else if (src) {
      throw new Error(`Slot ${slot.id} starts from a chain step, which only a chain can do.`)
    }

    const params = labParams(def, file, slot, suite, seed, { text, steps: v.steps, sampler: v.sampler ?? slot.sampler })
    const target = slot.target ?? (op === 'face' || op === 'hand' ? op : undefined)
    const cellSpec: CellSpec = { file, op, params, image, mask, rect, sourceSize, denoise: slot.denoise, megapixels: slot.megapixels, target }
    const built = buildGraph(cellSpec)
    if (!built.graph) {
      if (isOwn) this.noteNA(slot.id, key, built.reason)
      return null
    }
    const size = expectedSize(cellSpec)
    const cell: Cell = {
      cellId: cellIdOf(built.graph),
      study: suite.study,
      suite: suite.id,
      slot: slot.id,
      set: v.set ?? setOf(slot),
      model: key,
      chain: null,
      chainStep: null,
      file,
      familyId: def.id,
      op,
      seed,
      steps: params.steps,
      cfg: params.cfg,
      sampler: params.sampler,
      scheduler: schedulerOf(built.def, built.graph, params.scheduler),
      width: size.width,
      height: size.height,
      denoise: denoiseOf(op, cellSpec),
      condition: v.condition ?? slot.condition ?? null,
      upstream,
      refs,
      placeholders: built.placeholders,
      labOnly: built.labOnly,
    }
    this.add(cell, built.graph, text, isOwn)
    return cell
  }

  /** Every cell of one suite. */
  suite(s: Suite) {
    for (const slot of s.slots) {
      const variants: Variant[] = slot.stepsList?.length
        ? slot.stepsList.map((n) => ({ tag: `steps${n}`, steps: n }))
        : [PLAIN]
      for (const key of resolveModels(slot, s)) {
        for (const v of variants) for (const seed of s.seeds) this.cellFor(s, slot, key, seed, v)
      }
    }

    if (s.sweep) {
      for (const key of s.sweep.models) {
        for (const id of s.sweep.slots) {
          const slot = s.slots.find((x) => x.id === id) as Slot
          for (const n of s.sweep.steps) {
            const v: Variant = { tag: `sweep${n}`, steps: n, set: 'sweep', condition: `${n} steps` }
            for (const seed of s.seeds) this.cellFor(s, slot, key, seed, v)
          }
        }
      }
    }

    if (s.samplerCheck) {
      for (const key of s.samplerCheck.models) {
        for (const id of s.samplerCheck.slots) {
          const slot = s.slots.find((x) => x.id === id) as Slot
          const arms: Variant[] = [
            { tag: 'sampler-home', sampler: 'home', set: 'sampler', condition: 'home sampler' },
            { tag: 'sampler-common', set: 'sampler', condition: 'common sampler' },
          ]
          for (const v of arms) for (const seed of s.seeds) this.cellFor(s, slot, key, seed, v)
        }
      }
    }

    for (const chain of s.chains) this.chain(s, chain)
  }

  private chain(s: Suite, chain: Suite['chains'][number]) {
    const found = findSlot(chain.compose.slot, s, this.context)
    if (!found) throw new Error(`Chain ${chain.id} starts from slot ${chain.compose.slot}, which no suite of study ${s.study} has.`)
    for (const seed of s.seeds) {
      let prev = this.cellFor(found.suite, found.slot, chain.compose.model, seed, PLAIN)
      if (!prev) {
        const waits = this.blocked.find((b) => b.slot === found.slot.id && b.model === chain.compose.model)
        this.block({
          suite: s.id,
          slot: found.slot.id,
          model: null,
          chain: chain.id,
          ref: waits?.ref ?? null,
          kind: waits?.kind ?? 'upstream',
          why: waits ? `starts from ${found.slot.id}, which ${waits.why}` : `starts from ${found.slot.id}, which ${chain.compose.model} cannot make`,
        })
        continue
      }
      for (const [i, step] of chain.steps.entries()) {
        const spec = s.models[step.model]
        const def = spec ? familyOwning(spec.file) : null
        if (!spec || !def) throw new Error(`Chain ${chain.id} step ${i + 1} names model ${step.model}, which is not a registry file.`)
        const why = naReason(spec.file, step.op, { block: chain.block, target: step.target })
        if (why) {
          this.noteNA(chain.id, step.model, why)
          break
        }
        const text = chainStepText(step, found.slot.text)
        const bad = childWords(text)
        if (bad.length) throw new Error(`Chain ${chain.id} step ${i + 1} uses "${bad[0]}", which no lab prompt may use.`)
        const params = labParams(def, spec.file, { shape: found.slot.shape, text }, s, seed)
        const cellSpec: CellSpec = {
          file: spec.file,
          op: step.op,
          params,
          image: `cell:${prev.cellId}`,
          sourceSize: { width: prev.width, height: prev.height },
          denoise: step.denoise,
          megapixels: step.megapixels,
          target: step.target,
        }
        const built = buildGraph(cellSpec)
        if (!built.graph) {
          this.noteNA(chain.id, step.model, built.reason)
          break
        }
        const size = expectedSize(cellSpec)
        const cell: Cell = {
          cellId: cellIdOf(built.graph),
          study: s.study,
          suite: s.id,
          slot: found.slot.id,
          set: setOf(found.slot),
          model: null,
          chain: chain.id,
          chainStep: i,
          file: spec.file,
          familyId: def.id,
          op: step.op,
          seed,
          steps: params.steps,
          cfg: params.cfg,
          sampler: params.sampler,
          scheduler: schedulerOf(built.def, built.graph, params.scheduler),
          width: size.width,
          height: size.height,
          denoise: denoiseOf(step.op, cellSpec),
          condition: null,
          upstream: prev.cellId,
          refs: [...prev.refs],
          placeholders: built.placeholders,
          labOnly: built.labOnly,
        }
        this.add(cell, built.graph, text, true)
        prev = cell
      }
    }
  }
}

/** The scheduler that runs: the bound one, or the graph's own scheduler node where none is bound (Klein). */
function schedulerOf(def: FamilyDef | DerivedDef, graph: ApiWorkflow, asked: string): string {
  if (def.bindings.scheduler?.length) return asked
  const node = Object.values(graph).find((n) => /Scheduler$/.test(n.class_type))
  return node ? node.class_type : asked
}

/** The strength that applies to the source picture, or null where the operation draws from noise. */
function denoiseOf(op: Op, spec: CellSpec): number | null {
  switch (op) {
    case 'i2i':
      return spec.denoise ?? I2I_DEFAULTS.denoise
    case 'region':
      return spec.denoise ?? REGION_DEFAULTS.denoise
    case 'detailOnPicture':
      return DETAIL_TUNING[spec.target ?? 'face'].denoise
    default:
      return null
  }
}

/**
 * Every cell of the given suites, each built by the app's own functions.
 *
 * `refIndex` is the lab's reference photos (refs/index.json). A test whose
 * photo or mask is not there yet is listed in `blocked`, not guessed at.
 * `opts.context` is where a chain looks for a compose slot outside these
 * suites; by default the four shipped suites.
 *
 * Throws when a suite fails checkSuite: an unsafe or broken suite is never
 * expanded.
 */
export function expand(
  suites: readonly Suite[],
  refIndex: RefIndex = {},
  opts: { context?: readonly Suite[] } = {},
): Expansion {
  const context = [...suites, ...(opts.context ?? SHIPPED_SUITES).filter((c) => !suites.includes(c) && !suites.some((s) => s.id === c.id))]
  for (const s of suites) {
    const problems = checkSuite(
      s,
      context.filter((c) => c !== s),
    )
    if (problems.length) throw new Error(`Suite ${s.id} cannot be used:\n- ${problems.join('\n- ')}`)
  }
  const x = new Expander(suites, context, refMap(refIndex))
  for (const s of suites) x.suite(s)
  return { cells: x.cells, graphs: x.graphs, na: x.na, blocked: x.blocked, context: x.contextIds(), prompts: x.prompts }
}

// ---------------------------------------------------------------------------
// Before sending
// ---------------------------------------------------------------------------

export type Bind = {
  /** The finished picture the cell starts from (its rel under outputs), or 'token' when it is made in the same group. */
  upstream?: { rel: string } | 'token'
  /** Placeholder to rel under outputs, as refBindings gives. */
  refs: Record<string, string>
}

const SAFE_REL = /^\.lab\/[A-Za-z0-9._/-]+$/

/**
 * The graph as it is sent: real paths where the placeholders were, the
 * annotated form LoadImage reads ('.lab/cells/<id>_00001_.png [output]'), or
 * CHAIN_TOKEN where the upstream is made in the same group, and the SaveImage
 * prefix '.lab/cells/<cellId>'. `chainAt` lists the token's sites for the
 * job's chain.at, or null.
 */
export function finalizeGraph(graph: ApiWorkflow, cell: Cell, bind: Bind): { graph: ApiWorkflow; chainAt: [string, string][] | null } {
  const g = clone(graph)
  const chainAt: [string, string][] = []
  for (const [node, input, ph] of cell.placeholders) {
    if (g[node]?.inputs?.[input] !== ph) throw new Error(`Node ${node} input ${input} does not hold ${ph}; the graph is not this cell's.`)
    if (ph.startsWith('cell:')) {
      const up = ph.slice(5)
      if (up !== cell.upstream) throw new Error(`Cell ${cell.cellId} starts from ${up}, but names ${cell.upstream} as its upstream.`)
      if (bind.upstream === 'token') {
        g[node].inputs[input] = CHAIN_TOKEN
        chainAt.push([node, input])
      } else if (bind.upstream && typeof bind.upstream.rel === 'string') {
        const rel = bind.upstream.rel
        if (!SAFE_REL.test(rel) || rel.includes('..') || !rel.startsWith(`.lab/cells/${up}_`)) {
          throw new Error(`"${rel}" is not a picture of cell ${up}.`)
        }
        g[node].inputs[input] = annotated(rel)
      } else {
        throw new Error(`Cell ${cell.cellId} starts from cell ${up}, which is not made yet.`)
      }
    } else {
      const rel = bind.refs[ph]
      if (!rel) throw new Error(`No file was given for ${ph}.`)
      const bare = rel.replace(/ \[output\]$/, '')
      if (!SAFE_REL.test(bare) || bare.includes('..') || !bare.startsWith('.lab/refs/')) throw new Error(`"${rel}" is not a reference copy under .lab/refs.`)
      g[node].inputs[input] = annotated(bare)
    }
  }
  for (const node of Object.values(g)) {
    if (node.class_type === 'SaveImage') node.inputs.filename_prefix = cellPrefix(cell.cellId)
  }
  return { graph: g, chainAt: chainAt.length ? chainAt : null }
}

const OUTPUT_NODE = /^(SaveImage|PreviewImage|SaveAnimatedPNG|SaveAnimatedWEBP|SaveImageWebsocket|SaveVideo|SaveWEBM|SaveAudio|PreviewAny|VHS_VideoCombine)$/

const isLink = (v: unknown): v is [string, number] =>
  Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && typeof v[1] === 'number'

function reaches(graph: ApiWorkflow, from: string, to: string): boolean {
  const seen = new Set<string>()
  const stack = [from]
  while (stack.length) {
    const id = stack.pop() as string
    if (id === to) return true
    if (seen.has(id) || !graph[id]) continue
    seen.add(id)
    for (const v of Object.values(graph[id].inputs)) if (isLink(v)) stack.push(v[0])
  }
  return false
}

/**
 * Every reason not to send this finished graph for this cell; empty when it
 * may go. Checks that the bound inputs read back as the cell says, that
 * exactly one SaveImage saves it under the cell's own prefix, that the chain
 * token sits only where the cell's upstream goes, that no placeholder is left,
 * that it is the very graph the cell was hashed from, and that it is 1 MiB or
 * less.
 */
export function verifyCell(graph: ApiWorkflow, cell: Cell): string[] {
  const out: string[] = []
  const say = (m: string) => out.push(m)

  // Read back.
  const detector = Object.values(graph).find((n) => n.class_type === 'UltralyticsDetectorProvider')
  const target: 'face' | 'hand' =
    cell.op === 'hand' || detector?.inputs.model_name === DETECTORS.hand ? 'hand' : 'face'
  const shape = shapeFor(cell.file, cell.op, target)
  if ('reason' in shape) {
    say(`The app cannot build a ${cell.op} graph for ${cell.file}: ${shape.reason}.`)
  } else {
    const b = shape.def.bindings
    const want: [keyof FamilyDef['bindings'], unknown][] = [
      ['seed', cell.seed],
      ['steps', cell.steps],
      ['cfg', cell.cfg],
      ['sampler', cell.sampler],
    ]
    if (!shape.def.dualModel) want.push(['model', cell.file])
    if (b.scheduler?.length) want.push(['scheduler', cell.scheduler])
    if (cell.op === 't2i' || cell.op === 'face' || cell.op === 'hand') want.push(['width', cell.width], ['height', cell.height])
    if (cell.op === 'i2i' || cell.op === 'region') want.push(['denoise', cell.denoise])
    for (const [key, value] of want) {
      for (const [node, input] of b[key] ?? []) {
        if (!graph[node]) {
          say(`The ${key} setting's node ${node} is missing from the graph.`)
          continue
        }
        const got = graph[node].inputs[input]
        if (got !== value) say(`Node ${node} ${input} reads ${JSON.stringify(got)}, not the cell's ${key} ${JSON.stringify(value)}.`)
      }
    }
    if (cell.op === 'hires' && 'derived' in shape.def) {
      const [w] = b.width ?? []
      const [h] = b.height ?? []
      const size = w && h ? hiresSize({ width: Number(graph[w[0]]?.inputs[w[1]]), height: Number(graph[h[0]]?.inputs[h[1]]) }) : null
      if (!size || size.width !== cell.width || size.height !== cell.height) say(`The hires pass does not end at the cell's ${cell.width} x ${cell.height}.`)
      for (const [node, input] of (shape.def as DerivedDef).derived.extra.hiresSteps ?? []) {
        if (graph[node]?.inputs[input] !== hiresStepsFor(cell.steps)) say(`The hires pass runs ${String(graph[node]?.inputs[input])} steps, not ${hiresStepsFor(cell.steps)}.`)
      }
    }
  }

  // One SaveImage, under the cell's prefix, fed by the graph.
  const outputs = Object.entries(graph).filter(([, n]) => OUTPUT_NODE.test(n.class_type))
  const saves = outputs.filter(([, n]) => n.class_type === 'SaveImage')
  if (saves.length !== 1 || outputs.length !== 1) {
    say(`The graph must save exactly one picture with one SaveImage; it has ${saves.length} SaveImage and ${outputs.length - saves.length} other output nodes.`)
  }
  for (const [id, n] of saves) {
    if (n.inputs.filename_prefix !== cellPrefix(cell.cellId)) say(`SaveImage ${id} saves as "${String(n.inputs.filename_prefix)}", not "${cellPrefix(cell.cellId)}".`)
    const src = n.inputs.images
    if (!isLink(src) || !graph[src[0]]) say(`SaveImage ${id} is not fed by the graph.`)
  }
  const sampler = Object.entries(graph).find(([, n]) => /^(KSampler|KSamplerAdvanced|SamplerCustomAdvanced|FaceDetailer)$/.test(n.class_type))
  if (sampler && saves.length === 1 && !reaches(graph, saves[0][0], sampler[0])) say('The saved picture does not come from the graph\'s sampler.')

  // Placeholders, paths and the chain token.
  const leftover = placeholdersOf(graph)
  for (const [node, input, v] of leftover) say(`Node ${node} input ${input} still holds the placeholder ${v}.`)
  const tokenSites: string[] = []
  for (const [id, node] of Object.entries(graph)) {
    for (const [input, v] of Object.entries(node.inputs)) if (v === CHAIN_TOKEN) tokenSites.push(`${id}.${input}`)
  }
  const cellSites = cell.placeholders.filter(([, , ph]) => ph.startsWith('cell:')).map(([n, i]) => `${n}.${i}`)
  for (const site of tokenSites) if (!cellSites.includes(site)) say(`The chain token sits at ${site}, where the cell's upstream does not go.`)
  if (tokenSites.length && tokenSites.length !== cellSites.length) say('The chain token is at some of the upstream\'s sites but not all.')
  const tokenCount = JSON.stringify(graph).split(CHAIN_TOKEN).length - 1
  if (tokenCount !== tokenSites.length) say('The chain token appears inside a value, not as a whole input.')
  for (const [node, input, ph] of cell.placeholders) {
    const v = graph[node]?.inputs?.[input]
    if (v === CHAIN_TOKEN && ph.startsWith('cell:')) continue
    if (typeof v !== 'string') {
      say(`Node ${node} input ${input} holds no file.`)
      continue
    }
    const ok = ph.startsWith('cell:')
      ? new RegExp(`^\\.lab/cells/${ph.slice(5)}_\\d{5}_\\.png \\[output\\]$`).test(v)
      : ph.startsWith('mask:')
        ? v === `${maskRel(ph.slice(5))} [output]`
        : v.startsWith(`.lab/refs/${ph.slice(4)}.`) && v.endsWith(' [output]') && !v.includes('..')
    if (!ok && !PLACEHOLDER.test(v)) say(`Node ${node} input ${input} holds "${v}", which is not the file for ${ph}.`)
  }

  // The very graph that was hashed.
  const back = clone(graph)
  for (const [node, input, ph] of cell.placeholders) if (back[node]) back[node].inputs[input] = ph
  if (cellIdOf(back) !== cell.cellId) say('The graph is not the one this cell was hashed from.')

  const bytes = Buffer.byteLength(JSON.stringify(graph), 'utf8')
  if (bytes > GRAPH_MAX_BYTES) say(`The graph is ${bytes} bytes, over the 1 MiB limit.`)
  return out
}
