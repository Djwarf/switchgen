/**
 * What the reader took back by hand.
 *
 * The recipe decides everything. This module is the thin layer that lets one
 * decision be overruled without throwing the other forty away, and it is
 * deliberately small: an `Overrides` object holds only the keys the reader has
 * actually touched, so an empty object means "everything as decided" and a
 * missing key is never confused with a value that happens to equal the default.
 *
 * WHY THE MODEL IS NOT IN HERE. Changing the checkpoint changes the prompt
 * prefix it was trained with, which LoRAs can attach to it, which trigger
 * tokens belong in the positive, the sampler, the size and the pass list. All
 * of that lives in decide(), and re deriving a subset of it here would be a
 * second source of truth that goes stale the first time recipe.ts changes. So
 * the model picker does not write an override: it asks the desk to pin a model
 * and run decide() again with that file as the only installed candidate. See
 * `onPinModel` on {@link AdvancedPanel}.
 *
 * WHAT IS REBUILT AND WHEN. When no pass is switched on and the stack is the
 * one the recipe resolved, `def` is the recipe's own `def`, untouched: there is
 * one graph, and it is the one decide() built. Only a pass toggle or a stack
 * edit causes a rebuild, and the rebuild follows the desk's established order,
 * hires first because it changes the frame, then the detailers which work on
 * that frame's pixels, then the LoRAs which patch the model all of them sample
 * on.
 */
import {
  deriveAutoDetail,
  deriveHiresFix,
  hiresStepsFor,
  withLoras,
  writeExtras,
  type DerivedDef,
  type LoraSpec,
} from '../../lib/refine'
import {
  EMPTY_LIBRARY,
  resolveStack,
  targetFor,
  triggersFor,
  type LoraLibrary,
  type LoraStack,
  type LoraTarget,
} from '../../lib/loras'
import { plainWords, type Plan } from '../../lib/recipe'
import {
  IMG2IMG,
  deriveImg2Img,
  instantiate,
  type FamilyDef,
  type Params,
} from '../../lib/workflows'
import type { ApiWorkflow } from '../../lib/comfy'

// ---------------------------------------------------------------------------
// The shape
// ---------------------------------------------------------------------------

/** Face, hands, and the two pass render. Refine needs a mask, so it is not here. */
export type Passes = { face: boolean; hand: boolean; hires: boolean }

export const NO_PASSES: Passes = { face: false, hand: false, hires: false }

/**
 * Every key is optional and absent means "as decided". Nothing in this object
 * is ever filled in with a copy of the decided value, because then the panel
 * could not tell the two apart and the restore link would have nothing to say.
 */
export type Overrides = {
  steps?: number
  cfg?: number
  sampler?: string
  scheduler?: string
  width?: number
  height?: number
  seed?: number
  /** True pins the seed across runs. Absent means a fresh seed each time. */
  seedLocked?: boolean
  shift?: number
  clipSkip?: number
  negative?: string
  /** The positive exactly as it will be sent, prefix and triggers included. */
  positive?: string
  /** Image to image only. */
  denoise?: number
  megapixels?: number
  /** How many pictures, run one after another rather than in one batch. */
  runs?: number
  /** The whole LoRA stack, replacing the one the recipe resolved. */
  loras?: LoraStack
  passes?: Passes
}

export const NO_OVERRIDES: Overrides = {}

export type OverrideKey = keyof Overrides

/** Sensible bounds, matching what the desk has always accepted. */
export const BOUNDS = {
  steps: { min: 1, max: 150 },
  cfg: { min: 0, max: 30, step: 0.1 },
  size: { min: 256, max: 4096, step: 16 },
  shift: { min: 0, max: 12, step: 0.1 },
  clipSkip: { min: -12, max: -1 },
  denoise: { min: 0.05, max: 1, step: 0.01 },
} as const

/** The output budget buckets for image to image, where the source sets the shape. */
export const MEGAPIXELS: readonly number[] = [0.6, 1.0, 1.4, 2.0]

/** The four proportional buckets the shape picker has always offered. */
export const SHAPES: readonly { key: string; label: string; ratio: number }[] = [
  { key: 'portrait', label: 'Portrait', ratio: 2 / 3 },
  { key: 'square', label: 'Square', ratio: 1 },
  { key: 'landscape', label: 'Landscape', ratio: 3 / 2 },
  { key: 'wide', label: 'Wide', ratio: 16 / 9 },
]

// ---------------------------------------------------------------------------
// Small edits
// ---------------------------------------------------------------------------

export function setOverride<K extends OverrideKey>(
  ov: Overrides,
  key: K,
  value: Overrides[K],
): Overrides {
  return { ...ov, [key]: value }
}

/** Hand one value back to the recipe. */
export function clearOverride(ov: Overrides, key: OverrideKey): Overrides {
  if (!(key in ov)) return ov
  const next = { ...ov }
  delete next[key]
  return next
}

export function overrideKeys(ov: Overrides): OverrideKey[] {
  return (Object.keys(ov) as OverrideKey[]).filter((k) => ov[k] !== undefined)
}

export function anyOverride(ov: Overrides): boolean {
  return overrideKeys(ov).length > 0
}

const NUMBER_KEYS = ['steps', 'cfg', 'width', 'height', 'seed', 'shift', 'clipSkip', 'denoise', 'megapixels', 'runs'] as const
const STRING_KEYS = ['sampler', 'scheduler', 'negative', 'positive'] as const

/**
 * Overrides read back from storage, keeping only known keys with values of the
 * right kind. The desk keeps what was set by hand for the tab, so a reload does
 * not quietly put the recipe's values back under a loaded picture; what comes
 * back was written by some earlier build, and a string where a number belongs
 * would reach the graph as it stands. Anything unreadable is dropped, key by
 * key, which is the recipe's value again.
 */
export function sanitiseOverrides(raw: unknown): Overrides {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return NO_OVERRIDES
  const r = raw as Record<string, unknown>
  const out: Record<string, unknown> = {}
  for (const k of NUMBER_KEYS) {
    const v = r[k]
    if (typeof v === 'number' && Number.isFinite(v)) out[k] = v
  }
  for (const k of STRING_KEYS) if (typeof r[k] === 'string') out[k] = r[k]
  if (typeof r.seedLocked === 'boolean') out.seedLocked = r.seedLocked
  const p = r.passes as Record<string, unknown> | null | undefined
  if (p && typeof p === 'object' && [p.face, p.hand, p.hires].every((b) => typeof b === 'boolean')) {
    out.passes = { face: p.face, hand: p.hand, hires: p.hires }
  }
  if (Array.isArray(r.loras)) {
    out.loras = r.loras.flatMap((e: unknown) => {
      const s = e as Record<string, unknown> | null
      if (!s || typeof s.file !== 'string' || typeof s.strength !== 'number' || !Number.isFinite(s.strength)) return []
      const entry: LoraStack[number] = { file: s.file, strength: s.strength, enabled: s.enabled !== false }
      return typeof s.clipStrength === 'number' && Number.isFinite(s.clipStrength)
        ? [{ ...entry, clipStrength: s.clipStrength }]
        : [entry]
    })
  }
  return out as Overrides
}

// ---------------------------------------------------------------------------
// Settling
// ---------------------------------------------------------------------------

const snap16 = (n: number) => Math.max(16, Math.round(n / 16) * 16)

/**
 * The stack decide() actually resolved, as a stack the rack can edit.
 *
 * plan.loras is the resolved chain in load order, so every entry in it already
 * passed the architecture check and is on disk. Turning it back into a
 * LoraStack loses nothing: strength and order are the whole of it.
 */
export function decidedStack(plan: Plan): LoraStack {
  return plan.loras.map((l) => ({ file: l.file, strength: l.strength, enabled: true }))
}

function sameStack(a: LoraStack, b: LoraStack): boolean {
  if (a.length !== b.length) return false
  return a.every((e, i) => {
    const o = b[i]
    return (
      o !== undefined &&
      e.file === o.file &&
      e.enabled === o.enabled &&
      Math.abs(e.strength - o.strength) < 1e-9 &&
      (e.clipStrength ?? e.strength) === (o.clipStrength ?? o.strength)
    )
  })
}

/**
 * The family graph after image to image and before any LoRA chain.
 *
 * This reproduces step 6 of decide() exactly, and it has to: rebuilding from
 * `plan.base` alone would silently drop the image to image rewiring and render
 * from the prompt while the reader is looking at their own photograph.
 */
export function preLoraDef(plan: Plan): FamilyDef {
  if (!plan.params.image) return plan.base
  return IMG2IMG[plan.base.id] ?? deriveImg2Img(plan.base) ?? plan.base
}

export function targetOf(plan: Plan): LoraTarget {
  return targetFor(plan.base, plan.model)
}

/**
 * Rewrite the trigger tokens in a positive prompt for a changed stack.
 *
 * decide() builds the positive as prefix, then the reader's words, then one
 * trigger token per LoRA that needs one, and it drops any trigger the prompt
 * already contains. So every decided trigger is a token that is in the positive
 * and is not part of what the reader typed, which makes removing it by exact
 * token match safe. New triggers go on the end, in stack order, and only when
 * the prompt does not already carry them.
 */
function retrigger(
  positive: string,
  from: LoraStack,
  to: LoraStack,
  lib: LoraLibrary,
  target: LoraTarget,
): string {
  const old = new Set(triggersFor(from, lib, target).map((t) => t.toLowerCase()))
  const kept = positive
    .split(/\s*,\s*/)
    .map((p) => p.trim())
    .filter((p) => p && !old.has(p.toLowerCase()))
  const have = new Set(kept.map((p) => p.toLowerCase()))
  // The same words decide() would send, so a stack edited by hand cannot put
  // a model card's markdown into the prompt where the recipe would not.
  const add = plainWords(triggersFor(to, lib, target)).filter((t) => !have.has(t.toLowerCase()))
  return [...kept, ...add].join(', ')
}

export type Settled = {
  /** Ready for instantiate(def, params). */
  params: Params
  /** The graph that will run: passes and the LoRA chain already on it. */
  def: FamilyDef | DerivedDef
  /** The stack in force, decided or hand set. */
  stack: LoraStack
  /** What of that stack will actually be sent, after the architecture check. */
  specs: LoraSpec[]
  /** Stack entries the resolver refused, and why. */
  dropped: { file: string; label: string; why: string }[]
  /** Stack entries it allowed with a reservation. */
  cautions: { file: string; label: string; why: string }[]
  passes: Passes
  /** Rough multiple of one plain generation of this family. */
  cost: number
  /** True when the graph was rebuilt rather than taken from the recipe. */
  rebuilt: boolean
  /** How many pictures the run button will queue. */
  runs: number
  /** Whether the seed is pinned across runs. */
  seedLocked: boolean
}

/**
 * Fold the overrides onto the plan and hand back everything a run needs.
 *
 * Pure, and cheap enough to call on every render. The only expensive part is
 * the graph rebuild, which happens only when a pass is on or the stack was
 * edited, and the panel memoises it.
 */
export function settle(plan: Plan, ov: Overrides, lib: LoraLibrary = EMPTY_LIBRARY): Settled {
  const target = targetOf(plan)
  const decided = decidedStack(plan)
  const stack = ov.loras ?? decided
  const stackChanged = !sameStack(stack, decided)

  const resolved = resolveStack(stack, lib, target)
  const specs: LoraSpec[] = resolved.specs.map((s) => ({
    name: s.name,
    strength: s.strength,
    clipStrength: s.clipStrength,
  }))

  // A pass is on only where the plan can run it. A pass switched on for one
  // plan outlives it: the overrides are held across a change of mode and a
  // visit to another room, and the edit model carries no detail pass, nor
  // does a ComfyUI without the detector. Left on, the graph was built without
  // it while the record filed it as run, and the cost counted it.
  const wanted = ov.passes ?? NO_PASSES
  const passes: Passes = {
    face: wanted.face && plan.passes.face.available,
    hand: wanted.hand && plan.passes.hand.available,
    hires: wanted.hires && plan.passes.hires.available,
  }
  const anyPass = passes.face || passes.hand || passes.hires

  // One graph when nothing was touched: the recipe's own. A rebuild only
  // when a pass or the stack forces one.
  let def: FamilyDef | DerivedDef = plan.def
  const rebuilt = anyPass || stackChanged
  if (rebuilt) {
    let out: FamilyDef | DerivedDef = preLoraDef(plan)
    if (passes.hires) out = deriveHiresFix(out) ?? out
    if (passes.face) out = deriveAutoDetail(out, 'face') ?? out
    if (passes.hand) out = deriveAutoDetail(out, 'hand') ?? out
    if (specs.length) out = withLoras(out, specs) ?? out
    def = out
  }

  const params: Params = { ...plan.params }

  if (ov.steps !== undefined) params.steps = ov.steps
  if (ov.cfg !== undefined) params.cfg = ov.cfg
  if (ov.sampler !== undefined) params.sampler = ov.sampler
  if (ov.scheduler !== undefined) params.scheduler = ov.scheduler
  if (ov.width !== undefined) params.width = snap16(ov.width)
  if (ov.height !== undefined) params.height = snap16(ov.height)
  if (ov.seed !== undefined) params.seed = Math.max(0, Math.floor(ov.seed))
  if (ov.shift !== undefined) params.shift = ov.shift
  if (ov.clipSkip !== undefined) params.clipSkip = ov.clipSkip
  if (ov.negative !== undefined) params.negative = ov.negative
  if (params.image) {
    if (ov.denoise !== undefined) params.denoise = ov.denoise
    if (ov.megapixels !== undefined) params.megapixels = ov.megapixels
  }

  // The positive last, because a hand edit wins over the trigger rewrite: if
  // the reader has typed the prompt that will be sent, that is the prompt.
  if (stackChanged) params.positive = retrigger(params.positive, decided, stack, lib, target)
  if (ov.positive !== undefined) params.positive = ov.positive

  let cost = 1
  if (passes.hires) cost += 1.35
  if (passes.face) cost += 1
  if (passes.hand) cost += 1

  return {
    params,
    def,
    stack,
    specs,
    dropped: resolved.dropped,
    cautions: resolved.warnings,
    passes,
    cost,
    rebuilt,
    runs: ov.runs ?? 1,
    seedLocked: ov.seedLocked ?? false,
  }
}

/**
 * The graph exactly as it would be queued.
 *
 * `writeExtras` is the piece that is easy to forget: a two pass render carries
 * its own step count on a node instantiate() has no binding for, so a preview
 * built without it shows a second pass running at the graph's canned number
 * rather than the one derived from the steps on screen.
 */
export function buildGraph(settled: Settled): ApiWorkflow {
  const wf = instantiate(settled.def, settled.params)
  const derived = 'derived' in settled.def ? settled.def : null
  if (derived) {
    writeExtras(wf, derived, { hiresSteps: hiresStepsFor(settled.params.steps) })
  }
  return wf
}

/**
 * Four proportional rectangles sized against this family's own pixel budget,
 * with the bucket nearest its default ratio carrying the maker's exact numbers.
 * So Portrait on an Illustrious checkpoint really is 832 × 1216 rather than a
 * rounded approximation of it.
 */
export function shapesFor(plan: Plan, maxSide: number | null = null) {
  const dw = plan.params.width || 1024
  const dh = plan.params.height || 1024
  const area = dw * dh
  const r = dw / dh

  let nearest = 0
  let best = Infinity
  SHAPES.forEach((s, i) => {
    const gap = Math.abs(Math.log(s.ratio) - Math.log(r))
    if (gap < best) {
      best = gap
      nearest = i
    }
  })

  return SHAPES.map((s, i) => {
    if (i === nearest) return { ...s, width: dw, height: dh, maker: true }
    let w = snap16(Math.sqrt(area * s.ratio))
    let h = snap16(Math.sqrt(area / s.ratio))
    if (maxSide && Math.max(w, h) > maxSide) {
      const k = maxSide / Math.max(w, h)
      w = snap16(w * k)
      h = snap16(h * k)
    }
    return { ...s, width: w, height: h, maker: false }
  })
}

/** The maxSide a family's card declares, when it declares one. */
export function maxSideOf(plan: Plan): number | null {
  const per = (plan.base.perModel?.[plan.model] ?? {}) as Record<string, unknown>
  const v = per.maxSide
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}
