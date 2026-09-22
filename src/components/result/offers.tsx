/**
 * WHAT A FINISHED PICTURE CAN BE OFFERED, AND WHAT IT COSTS.
 *
 * This is the whole of requirement 3, in one pure function. The compose screen
 * used to ask, before a single pixel existed, whether to run a face pass, a
 * hand pass and a hires pass. Nobody can answer that. You cannot know whether
 * the hands came out wrong until you have looked at the hands.
 *
 * So the question moves to the only place it can honestly be asked: after the
 * picture. `offersFor` takes the graph the picture was actually made with and
 * returns the passes that graph can carry, each with its real cost attached.
 *
 * TWO RULES GOVERN EVERYTHING BELOW.
 *
 * First, availability is never guessed. Every pass is offered because its
 * derivation in lib/refine.ts returned a graph, and withheld because it
 * returned null. An absent row is the honest answer for a family that cannot
 * carry a pass. A greyed out row with an explanation underneath it is a row
 * nobody reads.
 *
 * Second, every number here is read out of something that measured it. The
 * cost multiples come from `derived.cost`, which lib/refine.ts sets as it
 * builds each graph, so they follow the graph rather than a comment about it.
 * The sharpness ratios come from the handoff run, through lib/recipe.ts's
 * MEASURED table. Nothing in this file is a number somebody thought sounded
 * about right.
 *
 * ON WHAT SHARPNESS IS. Laplacian variance measures sharpness. It does not
 * measure whether a hand has five fingers. A pass can rebuild a hand correctly
 * and soften the skin around it, and this metric will call that a loss. The
 * ratios below justify offering a pass rather than running it unasked. They are
 * never a claim about anatomical correctness, and the surface prints the caveat
 * next to them rather than leaving the reader to assume otherwise.
 */
import { MEASURED } from '../../lib/recipe'
import {
  capabilitiesOf,
  deriveAutoDetail,
  deriveHiresFix,
  deriveRefine,
  hiresSize,
  type DerivedDef,
} from '../../lib/refine'
import type { ImageFacts } from '../../lib/vision'
import type { FamilyDef } from '../../lib/workflows'

// ---------------------------------------------------------------------------
// The measurement this file adds
// ---------------------------------------------------------------------------

/**
 * THE REFINE PASS FIGURE, AND WHERE IT COMES FROM.
 *
 * 1.164x was stripped from this row as unsourceable, which was the right call at
 * the time: it was in neither lib/recipe.ts's MEASURED table nor
 * LORA_MEASUREMENTS.json. The source existed but was not written down.
 *
 * It is a real run: semiRealIllustrious_v40 at 832x1216, 30 steps, CFG 5, seed
 * 770411, on a face region found by bbox/face_yolov8m occupying 5.3% of the
 * frame. Laplacian variance over the grown mask box, 460.312 before and 535.691
 * after, so 1.164x. A control at half the render size gives 1.076x, which is
 * what shows the mechanism is resolution at the region rather than the pass
 * itself. Composite integrity was exact: 0.0 difference outside the crop.
 *
 * It now lives in LORA_MEASUREMENTS.json under `passes.regionRefine`, with its
 * method and control, so the next reader can trace it.
 *
 * The caveat travels with it: Laplacian variance measures SHARPNESS, not
 * anatomical correctness.
 */
const MEASURED_REFINE = { ratio: 1.164, region: 'the region you mark' } as const

/** `1.16x`. Ratios carry two decimals, because 1.14 and 1.16 are different answers. */
const ratio = (n: number) => `${n.toFixed(2)}x`

/**
 * `1x`, `2x`, `2.35x`. The cost multiple, trimmed rather than padded.
 *
 * Two decimals, not one. The hires derivation costs 2.35 and describes itself
 * as roughly 2.3x; rounding to one decimal prints 2.4x directly above its own
 * sentence saying 2.3, and a reader who notices that stops trusting both.
 */
const costLabel = (n: number) => `${Number(n.toFixed(2))}x`

// ---------------------------------------------------------------------------
// What comes back
// ---------------------------------------------------------------------------

export type ResultActionId =
  | 'refine'
  | 'hand'
  | 'face'
  | 'hires'
  | 'again'
  | 'source'
  // a clip's own rows, see videoOffers.ts
  | 'continue'
  | 'settings'
  | 'toPictures'
  | 'save'

export type ResultOffer = {
  id: ResultActionId
  /** What the reader is choosing, in the imperative. */
  label: string
  /** One or two sentences: what it does to this picture. Never a warning. */
  what: string
  /**
   * A measured figure, already worded, or null when nothing measured it. The
   * hand pass has no figure and says so rather than borrowing the face pass's.
   */
  measured: string | null
  /** Cost as a multiple of one ordinary generation, or null when it costs none. */
  cost: number | null
  /** The derivation's own cost sentence, for the title attribute. */
  costNote: string | null
  /**
   * `improve` works on the picture in front of the reader. `carry` starts the
   * next one. The two are printed apart because they answer different
   * questions: is this picture finished, and what comes after it.
   */
  group: 'improve' | 'carry'
}

export type OfferOptions = {
  /**
   * The finished picture's pixel size. Present, the hires row prints the size
   * it will actually produce, which is the one fact that makes "bigger" mean
   * something. Absent, the row still appears without it.
   */
  size?: { width: number; height: number } | null
  /**
   * Whether this desk can start an image to image from a finished picture.
   * False removes the row: the recipe would have nothing to route it to.
   */
  canSource?: boolean
  /**
   * What the detectors found, when the picture has been read. The hand and
   * face rows then say how many there are to fix and how big, which is the
   * one fact that tells a reader whether the pass has anything to do.
   */
  facts?: ImageFacts | null
  /**
   * Whether the desk can rebuild the graph that made this picture. False
   * withholds the four rows that re-render it from its record: the face, the
   * hands, the larger render and "make another". A region pass cannot be
   * rebuilt from what its record says, and rebuilding it anyway redrew the
   * whole original frame and threw the refine away, under a row promising
   * that only the faces would move. The refine and source rows work on the
   * file itself and stay.
   */
  rebuild?: boolean
}

/** "The detector found 2 hands, the larger 3.1% of the frame." or that it found none. */
function foundSentence(facts: ImageFacts | null | undefined, part: 'hand' | 'face'): string {
  if (!facts) return ''
  const list = facts[part]
  if (!list.length) return ` The detector found no ${part}s in this picture, so this pass would have nothing to draw.`
  const largest = Math.max(...list.map((d) => d.areaShare)) * 100
  return ` The detector found ${list.length} ${part}${list.length === 1 ? '' : 's'}, the largest ${largest.toFixed(1)}% of the frame.`
}

// ---------------------------------------------------------------------------
// offersFor
// ---------------------------------------------------------------------------

/**
 * The rows to print under a finished picture.
 *
 * `def` is the graph the picture was made with, which on a recipe is `plan.def`:
 * the derived def with image to image and the LoRA chain already applied. That
 * matters, because a def that already carries a hires pass cannot also carry a
 * refine, and asking the real graph is how this surface finds that out instead
 * of assuming.
 *
 * A null def means the desk does not know what made this picture, which is the
 * case for a picture opened out of the archive before its family is resolved.
 * The quality passes are then withheld, because offering a pass that might not
 * derive is exactly the disabled button this surface exists to avoid, and the
 * two rows that need no graph are still printed.
 */
export function offersFor(
  def: FamilyDef | DerivedDef | null,
  opts: OfferOptions = {},
): ResultOffer[] {
  const out: ResultOffer[] = []
  const caps = def ? capabilitiesOf(def) : null
  const rebuild = opts.rebuild !== false

  if (def && caps?.refine) {
    out.push({
      id: 'refine',
      label: 'Sharpen a region',
      what:
        'Mark one part of the picture. It is cut out, enlarged to full working resolution, drawn again and composited back. The only thing that adds real detail where no detector can find the region for you.',
      measured: `Measured ${ratio(MEASURED_REFINE.ratio)} sharpness on ${MEASURED_REFINE.region}. Sharpness, not anatomical correctness.`,
      ...costOf(deriveRefine(def)),
      group: 'improve',
    })
  }

  if (def && rebuild && caps?.handDetail) {
    out.push({
      id: 'hand',
      label: 'Fix the hands',
      what:
        'Finds every hand and draws it again, guided to 768 pixels and capped at 1024, with more freedom than a face, because hands come out wrong rather than merely soft. The picture is rendered again at the same seed, so the composition is the one in front of you.' +
        foundSentence(opts.facts, 'hand'),
      measured: 'Not measured. Judge it against the picture you already have.',
      ...costOf(deriveAutoDetail(def, 'hand')),
      group: 'improve',
    })
  }

  if (def && rebuild && caps?.faceDetail) {
    out.push({
      id: 'face',
      label: 'Fix the face',
      what:
        'Finds every face and draws it again, guided to 768 pixels and capped at 1024, where two eyes and a mouth finally have the cells to resolve. Rendered again at the same seed, so only the faces move.' +
        foundSentence(opts.facts, 'face'),
      measured: `Measured ${ratio(MEASURED.faceDetailer.ratio)} whole frame sharpness, which is why it is offered rather than run for you.`,
      ...costOf(deriveAutoDetail(def, 'face')),
      group: 'improve',
    })
  }

  if (def && rebuild && caps?.hires) {
    const big = opts.size ? hiresSize(opts.size) : null
    out.push({
      id: 'hires',
      label: 'Render it bigger',
      what: big
        ? `Composes at the size the model was trained on, upscales the latent, then draws the larger frame again at low strength. Every region gets 2.25 times the cells to resolve in. Output ${big.width} × ${big.height}.`
        : 'Composes at the size the model was trained on, upscales the latent, then draws the larger frame again at low strength. Every region gets 2.25 times the cells to resolve in.',
      measured: null,
      ...costOf(deriveHiresFix(def)),
      group: 'improve',
    })
  }

  if (rebuild) {
    out.push({
      id: 'again',
      label: 'Make another like this',
      what: 'The same recipe, a new seed. Nothing to fill in again.',
      measured: null,
      cost: 1,
      costNote: 'One ordinary generation.',
      group: 'carry',
    })
  }

  if (opts.canSource !== false) {
    out.push({
      id: 'source',
      label: 'Use it as a source',
      what: 'Puts this picture in the source slot, so the next one is drawn from it rather than from the prompt alone.',
      measured: null,
      cost: null,
      costNote: null,
      group: 'carry',
    })
  }

  return out
}

/**
 * The cost of a derivation, read off the graph it produced.
 *
 * `derived.cost` is a multiple of one ordinary generation of the family, set by
 * lib/refine.ts as it builds each graph: a refine replaces the generation and
 * stays at 1, a detail pass adds about one per detection, a hires pass adds
 * 1.35 for a second sampler over 2.25 times the pixels. Reading it here rather
 * than restating it means these figures cannot drift away from the graphs they
 * describe.
 */
function costOf(derived: DerivedDef | null): { cost: number | null; costNote: string | null } {
  if (!derived) return { cost: null, costNote: null }
  return { cost: derived.derived.cost, costNote: derived.derived.note || null }
}

/** Both figures on one row, for the surface to print in its own voice. */
export const offerFigures = { ratio, costLabel }

/** The caveat that has to sit next to every ratio printed here. */
export const SHARPNESS_CAVEAT = MEASURED.metricCaveat
