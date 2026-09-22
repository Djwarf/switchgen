/**
 * WHAT WAS ACTUALLY MEASURED, AND WHAT IT IS ALLOWED TO CLAIM.
 *
 * The catalogue in `src/lib/loras.ts` carries each author's recommended
 * strength, read off a model card. That is a claim by the person selling the
 * file. This module carries something different: Laplacian variance over the
 * whole frame, on this machine, on the same base, at the same seed, with the
 * same prompt, with only the LoRA stack changed. Where the two disagree this
 * one wins for the DEFAULT, because it is the one that was measured here.
 *
 * WHAT LAPLACIAN VARIANCE IS. The variance of a four neighbour second
 * derivative over the frame. It goes up when edges are crisp and down when the
 * picture is soft. It is a SHARPNESS number and nothing else. It cannot see
 * whether a hand has five fingers, whether labia are the right way round, or
 * whether a nipple sits where a nipple sits. A LoRA can draw anatomy correctly
 * and soften skin while doing it, and this number would mark that down. So:
 *
 *   These figures justify a default strength and a warning. They do not
 *   justify a word about anatomical accuracy, and nothing in this folder says
 *   one. Every surface that prints a ratio prints that caveat beside it.
 *
 * WHAT WAS FOUND. Every anatomy LoRA measured costs sharpness. One file pays
 * it back: add-micro-details, at 1.625x base on its own. The default stack is
 * therefore an anatomy LoRA held low plus that restorer, which measures 1.140x
 * base, so the stack is sharper than no LoRA at all while still carrying the
 * anatomy help. anatomy-helper degrades monotonically and steeply past 0.5,
 * which is where the 0.4 cap comes from rather than from taste.
 *
 * ONE BASE. Pony Diffusion V6 XL. Illustrious and NoobAI share the ecosystem
 * and the UNet key layout but were not measured, so on those the numbers are a
 * starting point and the UI says so instead of quietly reusing them.
 */
import type { LoraInfo, LoraStack, LoraTarget } from '../../lib/loras'
import { Kicker } from './bits'

// ---------------------------------------------------------------------------
// The measurements
// ---------------------------------------------------------------------------

/** The run every ratio below is a fraction of. */
export const MEASURED = {
  method: 'Laplacian variance, four neighbour kernel, over the whole frame',
  base: 'Pony Diffusion V6 XL',
  baseFile: 'ponyDiffusionV6XL.safetensors',
  baseline: 165.9,
  settings: '832 × 1216, 28 steps, CFG 7, dpmpp_2m and karras, seed 99 held, one prompt',
} as const

/** One measured point: this file at this strength came out at this fraction of base. */
export type MeasuredPoint = { strength: number; ratio: number }

/**
 * Measured singles, keyed by the filename LoraLoader uses.
 *
 * Strengths are the ones that were run, not a curve. Anything between them is
 * interpolated by `ratioAt` and is labelled as a prediction wherever it is
 * printed.
 */
export const MEASURED_SINGLES: Readonly<Record<string, readonly MeasuredPoint[]>> = {
  'anatomy-helper.safetensors': [
    { strength: 0.3, ratio: 0.812 },
    { strength: 0.5, ratio: 0.718 },
    { strength: 0.8, ratio: 0.437 },
  ],
  'detailed-pussy.safetensors': [{ strength: 0.6, ratio: 0.744 }],
  'real-nipples-and-areola-textures-gmr.safetensors': [{ strength: 0.6, ratio: 0.824 }],
  'add-micro-details-concept-illustrious-pony-noobai.safetensors': [
    { strength: 0.6, ratio: 1.625 },
  ],
}

/** Stacks that were run end to end, so their figure is read rather than predicted. */
export const MEASURED_STACKS: readonly {
  entries: readonly (readonly [string, number])[]
  ratio: number
  verdict: string
}[] = [
  {
    entries: [
      ['anatomy-helper.safetensors', 0.4],
      ['add-micro-details-concept-illustrious-pony-noobai.safetensors', 0.6],
    ],
    ratio: 1.14,
    verdict: 'Sharper than the base while still carrying anatomy help.',
  },
  {
    entries: [
      ['anatomy-helper.safetensors', 0.4],
      ['real-nipples-and-areola-textures-gmr.safetensors', 0.5],
      ['add-micro-details-concept-illustrious-pony-noobai.safetensors', 0.7],
    ],
    ratio: 0.919,
    verdict:
      'Slightly worse than using none. Three body add-ons overrun what the detail one repays. Raise that one, or drop a body add-on.',
  },
]

/**
 * The one measured detail restorer.
 *
 * It sits in the catalogue's anatomy category, so the anatomy cap below has to
 * know to leave it alone. Capping the only file that puts sharpness back at
 * the same number as the files that take it away would defeat the whole stack.
 */
export const RESTORERS: ReadonlySet<string> = new Set([
  'add-micro-details-concept-illustrious-pony-noobai.safetensors',
])

/**
 * The cap on an anatomy LoRA, in one number.
 *
 * anatomy-helper measures 0.812 of base at 0.3, 0.718 at 0.5 and 0.437 at 0.8.
 * The drop is monotonic and it accelerates, and by 0.8 more than half the
 * sharpness is gone: past that point the restorer cannot buy it back inside a
 * sane stack. 0.4 is the last strength where the loss stays inside what one
 * add-micro-details at 0.6 repays.
 */
export const ANATOMY_CAP = 0.4

/** The stack these measurements actually recommend, as files and strengths. */
export const RECOMMENDED_STACK: readonly (readonly [string, number])[] =
  MEASURED_STACKS[0].entries

/** Ratio of the recommended stack, measured rather than predicted. */
export const RECOMMENDED_RATIO = MEASURED_STACKS[0].ratio

/**
 * Bases these figures were taken on, or share a key layout with.
 *
 * Outside this set a LoRA from the anatomy shelf is a mismatch anyway and
 * `fitFor` refuses it, so there is nothing for the prediction to say.
 */
const MEASURED_LINEAGE: ReadonlySet<string> = new Set(['pony', 'illustrious', 'sdxl'])

/** True when this checkpoint is one the figures can be shown against at all. */
export function measurementsApply(target: LoraTarget): boolean {
  return MEASURED_LINEAGE.has(target.arch)
}

/**
 * True only when the checkpoint loaded is the one the figures were taken on.
 *
 * Deliberately a filename test and not an architecture test. Every Pony
 * finetune reports `arch: 'pony'`, and a finetune is a different set of
 * weights with a different sharpness of its own; treating one as the other is
 * how a measurement quietly becomes a folk belief.
 */
export function measuredOnThisBase(target: LoraTarget): boolean {
  const stem = (s: string) =>
    s
      .replace(/\\/g, '/')
      .split('/')
      .pop()!
      .replace(/\.(safetensors|ckpt|pt|pth|sft|bin|gguf)$/i, '')
      .toLowerCase()
  return !!target.model && stem(target.model) === stem(MEASURED.baseFile)
}

// ---------------------------------------------------------------------------
// Caps
// ---------------------------------------------------------------------------

/**
 * The measured ceiling for one LoRA, or null where there is no reason to hold
 * it down. Anatomy files are capped; the restorer and everything outside the
 * anatomy shelf are not.
 */
export function capFor(info: LoraInfo | undefined): number | null {
  if (!info) return null
  if (info.slider || info.recommended < 0) return null
  if (RESTORERS.has(info.file)) return null
  return info.category === 'anatomy' ? ANATOMY_CAP : null
}

/**
 * The strength a LoRA should be added at.
 *
 * The measured default where there is one, the cap where the author's
 * recommendation runs past it, and the author's own figure otherwise. On
 * this shelf the author's number is routinely above what measures well:
 * anatomy-helper is published at 0.6 and measures 0.718 of base there.
 */
export function openingStrength(info: LoraInfo | undefined, fallback: number): number {
  if (!info) return fallback
  const measured = RECOMMENDED_STACK.find(([file]) => file === info.file)
  if (measured) return measured[1]
  const cap = capFor(info)
  return cap === null ? fallback : Math.min(fallback, cap)
}

// ---------------------------------------------------------------------------
// Predicting a stack
// ---------------------------------------------------------------------------

/**
 * One LoRA's predicted ratio at an unmeasured strength.
 *
 * Interpolated in log space through the measured points, anchored at (0, 1)
 * because a LoRA at strength zero is not loaded and the frame is the base
 * frame exactly. Past the last measured point the final segment's slope is
 * continued, which is an extrapolation and is the least confident part of this
 * module. Returns null when nothing was measured for this file.
 */
export function ratioAt(file: string, strength: number): number | null {
  const points = MEASURED_SINGLES[file]
  if (!points || !points.length || strength <= 0) return points ? 1 : null
  const curve = [{ strength: 0, ratio: 1 }, ...points].slice().sort((a, b) => a.strength - b.strength)
  for (let i = 1; i < curve.length; i += 1) {
    const lo = curve[i - 1]
    const hi = curve[i]
    if (strength <= hi.strength || i === curve.length - 1) {
      const span = hi.strength - lo.strength
      if (span <= 0) return hi.ratio
      const t = (strength - lo.strength) / span
      return Math.exp(Math.log(lo.ratio) + t * (Math.log(hi.ratio) - Math.log(lo.ratio)))
    }
  }
  return curve[curve.length - 1].ratio
}

/**
 * How much two LoRAs get in each other's way, per extra LoRA.
 *
 * Multiplying the singles alone overstates a stack, because each LoRA patches
 * weights the previous one already moved. Calibrated against the only two
 * stacks that were measured end to end:
 *
 *   anatomy-helper 0.4 + micro-details 0.6    singles 1.241  measured 1.140
 *   the same plus nipples 0.5, details at 0.7 singles 1.145  measured 0.919
 *
 * One factor per extra LoRA fits both to within about three percent, which is
 * why the prediction is printed to two figures and called a prediction.
 */
const INTERFERENCE = 0.91

export type Prediction = {
  /** Predicted, or measured when the stack matches one that was run. */
  ratio: number
  measured: boolean
  /** Files in the stack with no measurement, so the figure is incomplete. */
  unmeasured: string[]
  /** How many enabled LoRAs the figure covers. */
  counted: number
  /** The file costing the most sharpness, for the advice line. Null when none does. */
  worst: { file: string; ratio: number } | null
}

const same = (a: number, b: number) => Math.abs(a - b) < 0.001

/**
 * Predict what a stack does to sharpness.
 *
 * Only entries that will actually be sent are counted, on the same terms
 * `resolveStack` uses: enabled, installed, non zero. A stack that exactly
 * matches one of the measured runs reports that run's own number and says it
 * was measured, because a read figure beats a modelled one every time.
 */
export function predictStack(
  entries: readonly { file: string; strength: number; enabled: boolean }[],
  byFile: ReadonlyMap<string, LoraInfo>,
): Prediction | null {
  const live = entries.filter(e => {
    if (!e.enabled || e.strength === 0) return false
    const info = byFile.get(e.file)
    return !!info && info.installed
  })
  if (!live.length) return null

  for (const run of MEASURED_STACKS) {
    if (run.entries.length !== live.length) continue
    const hit = run.entries.every(([file, strength]) =>
      live.some(e => e.file === file && same(e.strength, strength)),
    )
    if (hit) {
      return { ratio: run.ratio, measured: true, unmeasured: [], counted: live.length, worst: null }
    }
  }

  const unmeasured: string[] = []
  let product = 1
  let counted = 0
  let worst: Prediction['worst'] = null
  for (const e of live) {
    const r = ratioAt(e.file, e.strength)
    if (r === null) {
      unmeasured.push(e.file)
      continue
    }
    product *= r
    counted += 1
    if (r < 1 && (!worst || r < worst.ratio)) worst = { file: e.file, ratio: r }
  }
  if (!counted) return { ratio: 1, measured: false, unmeasured, counted: 0, worst: null }

  const ratio = product * INTERFERENCE ** Math.max(0, counted - 1)
  return { ratio, measured: false, unmeasured, counted, worst }
}

/** Whether a stack is already the measured recommendation. */
export function isRecommendedStack(stack: LoraStack): boolean {
  const live = stack.filter(e => e.enabled && e.strength !== 0)
  if (live.length !== RECOMMENDED_STACK.length) return false
  return RECOMMENDED_STACK.every(([file, strength]) =>
    live.some(e => e.file === file && same(e.strength, strength)),
  )
}

// ---------------------------------------------------------------------------
// Printing it
// ---------------------------------------------------------------------------

/** `1.14x`. Two decimals, because the third is inside the prediction's error. */
export const ratioText = (n: number) => `${n.toFixed(2)}x`

/**
 * The citation.
 *
 * It exists so a reader can tell the difference between a number somebody
 * measured on this machine and a number somebody felt strongly about. It
 * carries the caveat with it, because the caveat is the more important half:
 * a sharpness figure that a reader takes for an anatomy figure is worse than
 * no figure at all.
 */
export function MeasurementNote({ target }: { target: LoraTarget }) {
  return (
    <div className="mt-2 border-t border-grey-300 pt-1.5">
      <Kicker>Where these numbers come from</Kicker>
      <p className="mt-1 text-caption leading-snug text-grey-700">
        {MEASURED.method}, on {MEASURED.base} at {MEASURED.settings}. Only the add-ons changed
        between runs, so the differences are the stack's. Base frame: {MEASURED.baseline}.
      </p>
      <p className="mt-1 text-caption leading-snug italic text-warning">
        It measures SHARPNESS, not whether a body came out right. An add-on can draw a body correctly and soften the skin
        doing it, and this number marks that down. Use it to pick a strength. Judge the anatomy
        with your eyes.
      </p>
      {!measuredOnThisBase(target) ? (
        <p className="mt-1 text-caption leading-snug italic text-grey-700">
          Taken on {MEASURED.base} only, and that is not the checkpoint loaded here. Illustrious,
          NoobAI and the Pony finetunes take the same add-ons, so they load and the shape of
          the effect should carry. The exact figures will not. Treat them as a starting point.
        </p>
      ) : null}
    </div>
  )
}
