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
 * THE RUN. Pony Diffusion V6 XL (ponyDiffusionV6XL.safetensors), 832 × 1216,
 * 28 steps, CFG 7, dpmpp_2m and karras, seed 99 held, one prompt, base frame
 * 165.9. Illustrious and NoobAI share the ecosystem and the UNet key layout
 * but were not measured, so on those the numbers are a starting point and the
 * UI says so instead of quietly reusing them. The stacks themselves, with
 * their ratios and verdicts, live in `src/lib/recipe.ts` (MEASURED), which is
 * what the desk prints; this file holds only the single-file points the rack
 * row cites.
 */
import type { LoraInfo } from '../../lib/loras'

/** One measured point: this file at this strength came out at this fraction of base. */
export type MeasuredPoint = { strength: number; ratio: number }

/**
 * Measured singles, keyed by the filename LoraLoader uses.
 *
 * Strengths are the ones that were run, not a curve. Anything between them is
 * not claimed.
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

/** `1.14x`. Two decimals, because the third is inside the prediction's error. */
export const ratioText = (n: number) => `${n.toFixed(2)}x`
