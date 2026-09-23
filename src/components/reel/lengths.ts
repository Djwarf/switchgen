/**
 * The shot lengths the reel offers, and which of them the desk would refuse.
 *
 * Kept apart from the desk so the offer can be checked without mounting it.
 * The chips used to offer the usual lengths whatever the family, and the
 * memory verdict (lib/clipMemory) was only asked once a shot had been given
 * one. On the image-to-video 14B pair that verdict refuses any clip larger
 * than the one it came through at, so at its own size and frame rate the
 * 5 s and 7 s chips were accepted on the bench and then refused under the
 * strip, with every shot that follows the reel's length refused with them.
 * Each length now carries the verdict a shot of that length gets, asked the
 * way the desk asks it of the plan, and the chips say so where the choice is
 * made.
 */
import { clipMemory } from '../../lib/clipMemory'
import { snapLength } from '../../lib/continuation'
import type { Hardware } from '../../lib/hardware'
import type { FamilyDef } from '../../lib/workflows'
import { seconds, type ChipOption } from './bits'
import type { NumSpec } from './Bench'

/** One shot length on offer, and why the desk would refuse a shot that long, if it would. */
export type LengthChoice = {
  frames: number
  /** The memory verdict's sentence when a shot this long at the reel's size is refused. Null otherwise. */
  refused: string | null
}

/** A handful of legal shot lengths around the durations people actually cut to. */
export function lengthChoices(fps: number, spec: NumSpec): number[] {
  const out: number[] = []
  for (const s of [2, 3, 5, 7]) {
    const n = snapLength(Math.round(s * Math.max(1, fps)))
    if (n >= spec.min && n <= spec.max && !out.includes(n)) out.push(n)
  }
  return out.sort((a, b) => a - b)
}

/**
 * The lengths the bench and the strip offer for one family, the reel's own
 * length among them even when it is not one of the usual ones, each with the
 * verdict a shot that long gets at the reel's size. `hardware` is the reading
 * the desk prices the plan against; without one the verdict is the measured
 * machine's, as it is for the plan.
 */
export function lengthsFor(
  family: { def: Pick<FamilyDef, 'id' | 'dualModel'>; frames: NumSpec },
  reel: { fps: number; length: number; width: number; height: number },
  hardware: Hardware | null,
): LengthChoice[] {
  const list = lengthChoices(reel.fps, family.frames)
  const all = list.includes(reel.length) ? list : [...list, reel.length].sort((a, b) => a - b)
  return all.map((frames) => {
    const verdict = clipMemory(
      family.def,
      { width: Math.floor(reel.width), height: Math.floor(reel.height), frames },
      hardware,
    )
    return { frames, refused: verdict.level === 'refuse' ? verdict.reason : null }
  })
}

/** The lengths as chips. One the desk would refuse is greyed out, with its reason on the chip. */
export function lengthChips(choices: readonly LengthChoice[], fps: number): ChipOption<number>[] {
  return choices.map((c) => ({
    value: c.frames,
    label: seconds(c.frames, fps),
    title: c.refused ?? `${c.frames} frames`,
    disabled: c.refused !== null,
  }))
}
