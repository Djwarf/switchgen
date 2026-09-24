/**
 * The lab's shared shapes. Every other lab module imports these, so they are
 * written once, here, and changed only together with the plan that defines
 * them (CONTRACTS, section C, in the lab plan).
 *
 * Nothing in src/ or server/ imports this file. The lab reads the app; the app
 * never reads the lab.
 */

/** One way of making a picture. `detailOnPicture` is lab-only: the app has no such graph. */
export type Op = 't2i' | 'i2i' | 'edit' | 'region' | 'face' | 'hand' | 'hires' | 'detailOnPicture'

export type ShapeName = 'square' | 'wide' | 'tall'

/** A building block of the capability map. */
export type Block =
  | 'following'
  | 'style'
  | 'photo'
  | 'variation'
  | 'content'
  | 'text'
  | 'reference'
  | 'layout'
  | 'anatomy'
  | 'detail'
  | 'cost'
  | 'shapes'
  | 'character'
  | 'sensitivity'
  | 'negative'
  | 'defaults'
  | 'region'
  | 'edit'
  | 'overall'

/** A contestant's weight file. `role: 'edit'` marks the instruction-editing model. */
export type ModelSpec = { file: string; role?: 'generate' | 'edit' }

/**
 * A reference photo the suite needs, by id ('cat', 'scene'). The file itself
 * comes from refs/index.json in the lab's private folder; `describe` is the
 * starting description, which the user can change on the lab page.
 */
export type RefSpec = { describe: string; needsMask?: boolean }

/** Where a slot's input picture comes from. */
export type Source = { ref: string } | { cell: { slot: string; model: string } } | { chainStep: number }

export type SetMode = 'scale' | 'beforeAfter' | 'pairs' | 'measure'

/** One test: a prompt, a shape, an operation and the models that take it. */
export type Slot = {
  id: string
  /** Slots sharing a set id form one set: character, sensitivity, negative pairs. */
  set?: string
  block: Block
  second?: Block
  shape: ShapeName
  op: Op
  text: string
  /** The neutral wording the phone shows. Defaults to `text`. */
  task?: string
  /** A model's own wording, by model key (Qwen-Edit's instruction, for example). */
  perModelText?: Record<string, string>
  /** Exact words that must appear in the picture, verbatim in `text`. */
  expect?: string[]
  checklist?: string[]
  layout?: string
  /** The before/after label of this member of a set. */
  condition?: string
  negativeAdd?: string
  models: 'core' | 'all' | 'negative-capable' | 'i2i-capable' | 'refine-capable' | 'edit' | string[]
  source?: Source
  denoise?: number
  megapixels?: number
  target?: 'face' | 'hand'
  /** For a detail slot: the slot id of the base picture. */
  after?: string
  humans?: boolean
  defaultsProbe?: boolean
  /** Default true. */
  innocent?: boolean
  measuredOnly?: boolean
  sampler?: 'home'
  stepsList?: number[]
}

/** A named chain: a compose picture, then one or more steps on it. */
export type ChainSpec = {
  id: string
  name: string
  block: Block
  also?: Block[]
  compose: { slot: string; model: string }
  steps: {
    model: string
    op: Op
    text: string
    denoise?: number
    megapixels?: number
    target?: 'face' | 'hand'
  }[]
}

export type Suite = {
  id: string
  version: number
  study: string
  seeds: number[]
  steps: number
  sampler: { name: string; scheduler: string }
  shapes: Record<ShapeName, [number, number]>
  models: Record<string, ModelSpec>
  core: string[]
  refs: Record<string, RefSpec>
  slots: Slot[]
  chains: ChainSpec[]
  sweep?: { models: string[]; slots: string[]; steps: number[]; homeFor: Record<string, number> }
  samplerCheck?: { models: string[]; slots: string[] }
}

/** One picture to make: one graph, identified by the hash of that graph. */
export type Cell = {
  cellId: string
  study: string
  suite: string
  slot: string
  set: string
  /** The model key, or null for a chain step (the contestant is then `chain`). */
  model: string | null
  chain: string | null
  chainStep: number | null
  file: string
  familyId: string
  op: Op
  seed: number
  steps: number
  cfg: number
  sampler: string
  scheduler: string
  width: number
  height: number
  denoise: number | null
  condition: string | null
  upstream: string | null
  refs: string[]
  /** [node, input, 'ref:<photo sha12>' | 'mask:<mask sha12>' | 'cell:<cellId>'] */
  placeholders: [string, string, string][]
  labOnly: boolean
}

/** A contestant that cannot take a test, with the registry's reason. */
export type NA = { slot: string; model: string; reason: string }

/** A file a finished job reported, as the runner lists it. */
export type FileRef = {
  filename: string
  subfolder: string
  type: string
  kind?: 'image' | 'video'
  cached?: boolean
}

export type JobError = { code: string; message: string; node: string | null; nodeType: string | null }

export type LedgerEntry =
  | { t: 'submitting'; at: number; group: string; jobs: { job: string; cell: string }[] }
  | { t: 'submitted'; at: number; group: string; replayed: boolean }
  | { t: 'refused'; at: number; group: string; status: number; error: string }
  | {
      t: 'ended'
      at: number
      job: string
      cell: string
      status: string
      error: JobError | null
      files: FileRef[]
      primary: FileRef | null
      promptId: string | null
      durationMs: number
      ranAt: number | null
      finishedAt: number | null
      cached: boolean
      cold: boolean
      attempt: number
    }
  | { t: 'recovered'; at: number; cell: string; rel: string }
  | { t: 'paused'; at: number; why: 'user' | 'until' | 'runner-off' }
  | { t: 'removed'; at: number; cell: string; why: 'quarantine' }

/** A cells.jsonl row: a finished cell any later run may reuse. */
export type DoneCell = {
  cellId: string
  rel: string
  durationMs: number
  cold: boolean
  cached: boolean
  finishedAt: number | null
  run: string
}

// ---------------------------------------------------------------------------
// Additions beyond the contract's list (additive only; nothing above changed)
// ---------------------------------------------------------------------------

/** A rectangle in the reference's upright pixels (EXIF orientation applied). */
export type RefRect = { x: number; y: number; w: number; h: number }

/**
 * One reference photo as refs/index.json records it. `width` and `height` are
 * the upright size, after the EXIF orientation flag is applied, because that
 * is how ComfyUI's LoadImage reads it and how the mask is drawn. `rect` is the
 * drawn mask rectangle, `maskSha12` the hash of the mask file drawn from it
 * (which names the mask's placeholder and copy, so each rectangle is its own
 * cell), `describe` the user's own description when they have changed the
 * suite's starting one.
 */
export type RefInfo = {
  id: string
  sha12: string
  ext: string
  width: number
  height: number
  mask: boolean
  rect?: RefRect | null
  maskSha12?: string | null
  describe?: string | null
}
