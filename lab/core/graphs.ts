/**
 * Every lab graph, built by the app's own functions and nothing else.
 *
 *   t2i     instantiate(def, p)
 *   i2i     instantiate(IMG2IMG[def.id], {...p, image, denoise, megapixels})
 *   edit    instantiate(BY_ID['qwen-image-edit'], {...p, image})
 *   region  instantiateRefine(deriveRefine(def), p, {image, mask, planRefine(...), ...})
 *   face    instantiate(deriveAutoDetail(def, 'face'), p)   (hand likewise)
 *   hires   instantiate(deriveHiresFix(def), p), then writeExtras(hiresSteps)
 *   detailOnPicture   lab-only (labShapes.ts), marked as such
 *
 * A finding about one of these graphs is therefore a finding about what the
 * app sends. Where a picture goes in, the graph holds a placeholder rather
 * than a file name: 'ref:<sha12>' for a reference photo, 'mask:<sha12>' for
 * its mask (the mask file's own hash, cells.ts maskKey), 'cell:<cellId>' for
 * another cell's picture. The cell's identity is
 * hashed over those placeholders; cells.ts puts the real paths in just before
 * a graph is sent.
 */
import { BY_ID, IMG2IMG, familyOwning, instantiate, type FamilyDef, type Params } from '../../src/lib/workflows.ts'
import type { ApiWorkflow } from '../../src/lib/comfy.ts'
import {
  REFINE_DEFAULTS,
  deriveAutoDetail,
  deriveHiresFix,
  deriveRefine,
  hiresStepsFor,
  instantiateRefine,
  planRefine,
  writeExtras,
  type DerivedDef,
} from '../../src/lib/refine.ts'
import { detailOnPicture } from './labShapes.ts'
import { REASONS, opReason } from './na.ts'
import type { Op, RefRect } from './types.ts'

/** The app's image-to-image strength (recipe.ts I2I) and pixel budget. */
export const I2I_DEFAULTS = { denoise: 0.65, megapixels: 1 } as const
/** The region pass the lab runs: the top of the app's useful band, the app's grow and feather. */
export const REGION_DEFAULTS = { denoise: 0.55, grow: REFINE_DEFAULTS.grow, feather: REFINE_DEFAULTS.feather } as const

/** A placeholder where a picture goes in. */
export const PLACEHOLDER = /^(ref|mask|cell):[0-9a-f]{8,64}$/

export type CellSpec = {
  /** The weight file. */
  file: string
  op: Op
  /** From labParams. */
  params: Params
  /** Placeholder of the picture that goes in: 'ref:<sha12>' or 'cell:<cellId>'. */
  image?: string
  /** Placeholder of the region's mask: 'mask:<sha12>'. */
  mask?: string
  /** The mask rectangle, in the source's upright pixels. */
  rect?: RefRect
  /** The source's upright size, for planning the region's crop. */
  sourceSize?: { width: number; height: number }
  denoise?: number
  megapixels?: number
  target?: 'face' | 'hand'
}

export type Built = {
  graph: ApiWorkflow
  /** [node, input, placeholder] for every picture that goes in. */
  placeholders: [string, string, string][]
  /** The family graph the cell was built from, derived where the operation derives one. */
  def: FamilyDef | DerivedDef
  labOnly: boolean
  note: string | null
}

export type NotBuilt = { graph: null; reason: string }

export type Shape = { def: FamilyDef | DerivedDef; labOnly: boolean; note: string | null }

/**
 * The family graph an operation runs for a file, or why there is none. The
 * same call builds a cell and later reads its graph back (verifyCell).
 */
export function shapeFor(file: string, op: Op, target?: 'face' | 'hand'): Shape | { reason: string } {
  const base = familyOwning(file)
  if (!base) return { reason: REASONS.unknown }
  const why = opReason(base, op, target)
  if (why) return { reason: why }
  const plain = (def: FamilyDef | DerivedDef | null | undefined, reason: string): Shape | { reason: string } =>
    def ? { def, labOnly: false, note: null } : { reason }
  switch (op) {
    case 't2i':
      return plain(base, REASONS.unknown)
    case 'i2i':
      return plain(IMG2IMG[base.id], REASONS.noI2i)
    case 'edit':
      return plain(base.mode === 'edit' ? base : BY_ID['qwen-image-edit'], REASONS.notEditor)
    case 'region':
      return plain(deriveRefine(base), REASONS.noRegion)
    case 'face':
      return plain(deriveAutoDetail(base, 'face'), REASONS.noFace)
    case 'hand':
      return plain(deriveAutoDetail(base, 'hand'), REASONS.noHand)
    case 'hires':
      return plain(deriveHiresFix(base), REASONS.noHires)
    case 'detailOnPicture': {
      const shape = detailOnPicture(base, target ?? 'face')
      return shape ? { def: shape.def, labOnly: true, note: shape.note } : { reason: REASONS.noDetailOnPicture }
    }
  }
}

/** Every [node, input, placeholder] in a graph, in the graph's order. */
export function placeholdersOf(graph: ApiWorkflow): [string, string, string][] {
  const out: [string, string, string][] = []
  for (const [id, node] of Object.entries(graph)) {
    for (const [input, v] of Object.entries(node.inputs)) {
      if (typeof v === 'string' && PLACEHOLDER.test(v)) out.push([id, input, v])
    }
  }
  return out
}

const isDerived = (def: FamilyDef | DerivedDef): def is DerivedDef => 'derived' in def && !!(def as DerivedDef).derived

/** Build one cell's graph, or say why the operation does not apply. */
export function buildGraph(spec: CellSpec): Built | NotBuilt {
  const shape = shapeFor(spec.file, spec.op, spec.target)
  if ('reason' in shape) return { graph: null, reason: shape.reason }
  const { def } = shape
  const p = spec.params
  const needsImage = spec.op === 'i2i' || spec.op === 'edit' || spec.op === 'region' || spec.op === 'detailOnPicture'
  if (needsImage && !spec.image) throw new Error(`A ${spec.op} cell needs the picture it works on.`)
  if (spec.image && !PLACEHOLDER.test(spec.image)) throw new Error(`"${spec.image}" is not a lab placeholder.`)

  let graph: ApiWorkflow
  switch (spec.op) {
    case 't2i':
    case 'face':
    case 'hand':
      graph = instantiate(def, p)
      break
    case 'i2i':
      graph = instantiate(def, {
        ...p,
        image: spec.image,
        denoise: spec.denoise ?? I2I_DEFAULTS.denoise,
        megapixels: spec.megapixels ?? I2I_DEFAULTS.megapixels,
      })
      break
    case 'edit':
      graph = instantiate(def, { ...p, image: spec.image })
      break
    case 'hires':
      graph = instantiate(def, p)
      if (isDerived(def)) writeExtras(graph, def, { hiresSteps: hiresStepsFor(p.steps) })
      break
    case 'region': {
      if (!spec.mask || !PLACEHOLDER.test(spec.mask)) throw new Error('A region cell needs its mask.')
      if (!spec.rect || !spec.sourceSize) throw new Error('A region cell needs the mask rectangle and the source size.')
      if (!isDerived(def)) throw new Error('The region pass did not derive.')
      const r = spec.rect
      const plan = planRefine({ x: r.x, y: r.y, width: r.w, height: r.h }, spec.sourceSize, {
        padding: REFINE_DEFAULTS.padding,
        targetLongEdge: REFINE_DEFAULTS.targetLongEdge,
      })
      graph = instantiateRefine(def, p, {
        image: spec.image as string,
        mask: spec.mask,
        crop: plan.crop,
        target: plan.target,
        denoise: spec.denoise ?? REGION_DEFAULTS.denoise,
        grow: REGION_DEFAULTS.grow,
        feather: REGION_DEFAULTS.feather,
        prompt: p.positive,
      })
      break
    }
    case 'detailOnPicture':
      graph = instantiate(def, { ...p, image: spec.image })
      break
  }

  const placeholders = placeholdersOf(graph)
  for (const want of [spec.image, spec.mask]) {
    if (want && !placeholders.some(([, , v]) => v === want)) {
      throw new Error(`The ${spec.op} graph did not take its picture (${want}); the app's binding may have moved.`)
    }
  }
  return { graph, placeholders, def, labOnly: shape.labOnly, note: shape.note }
}
