/**
 * The parameters one cell is made with, from the app's own registry.
 *
 * What is the same for everyone: the prompt, the seed, the step count and the
 * shape, and the common sampler. What stays each model's own, exactly as the
 * app would send it: the trained quality prefix, the house negative, the clip
 * skip, the shift and the guidance (CFG). A distilled model therefore runs at
 * CFG 1, which is guidance off, as it does in the app.
 */
import { defaultsFor, type FamilyDef, type Params } from '../../src/lib/workflows.ts'
import type { Slot, Suite } from './types.ts'

type SlotWords = Pick<Slot, 'shape' | 'text' | 'negativeAdd'>

/** The registry's per-file overrides, as loose data. */
function perOf(def: FamilyDef, file: string): Record<string, unknown> {
  return (def.perModel?.[file] ?? {}) as Record<string, unknown>
}

/**
 * The prompt with the file's trained prefix, joined the way the Pictures desk
 * joins it (recipe.ts): the prefix split into tags, then the prompt, with
 * ', ' between. A file that declares no prefix gets none.
 */
export function withPrefix(def: FamilyDef, file: string, text: string): string {
  const declared = perOf(def, file).positivePrefix
  const tokens =
    typeof declared === 'string'
      ? declared
          .split(',')
          .map((t) => t.trim())
          .filter(Boolean)
      : []
  return [...tokens, text].filter(Boolean).join(', ')
}

/** The house negative, with the slot's addition after it. */
export function negativeFor(def: FamilyDef, file: string, add?: string): string {
  const house = defaultsFor(def, file).negative ?? ''
  return [house.trim(), (add ?? '').trim()].filter(Boolean).join(', ')
}

/** The step count the registry gives a file: its home. */
export function homeSteps(def: FamilyDef, file: string): number {
  return defaultsFor(def, file).steps
}

/** The sampler and scheduler the registry gives a file. */
export function homeSampler(def: FamilyDef, file: string): { name: string; scheduler: string } {
  const d = defaultsFor(def, file)
  return { name: d.sampler, scheduler: d.scheduler }
}

export type LabParamOpts = {
  /** A step count other than the suite's (the step sweep). */
  steps?: number
  /** 'home' runs the file's own sampler and scheduler (the sampler check's other arm). */
  sampler?: 'home'
  /** The words to send instead of the slot's (a model's own wording, or a filled-in template). */
  text?: string
}

/**
 * The Params for one cell. Width and height come from the slot's shape; the
 * image-to-image, edit and region operations add their source in graphs.ts.
 */
export function labParams(
  def: FamilyDef,
  file: string,
  slot: SlotWords,
  suite: Pick<Suite, 'steps' | 'sampler' | 'shapes'>,
  seed: number,
  opts: LabParamOpts = {},
): Params {
  const d = defaultsFor(def, file)
  const per = perOf(def, file)
  const [width, height] = suite.shapes[slot.shape]
  const sampler = opts.sampler === 'home' ? { name: d.sampler, scheduler: d.scheduler } : suite.sampler
  const p: Params = {
    model: file,
    positive: withPrefix(def, file, opts.text ?? slot.text),
    negative: negativeFor(def, file, slot.negativeAdd),
    seed,
    steps: opts.steps ?? suite.steps,
    cfg: d.cfg,
    width,
    height,
    sampler: sampler.name,
    scheduler: sampler.scheduler,
  }
  if (typeof per.clipSkip === 'number') p.clipSkip = per.clipSkip
  if (typeof per.shift === 'number') p.shift = per.shift
  return p
}
