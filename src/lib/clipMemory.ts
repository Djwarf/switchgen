/**
 * Will a clip on a two-model Wan 2.2 14B family survive its own decode?
 *
 * The memory verdict in hardware.ts counts weights, and the weights of a 14B
 * pair fit: two experts at about 9.3 GB each plus the encoder. What does not
 * fit is the working set of a long clip on top of them, and it arrives at the
 * very end, when the sampled latent is decoded. On 2026-09-22 a Wan 2.2 T2V
 * 14B clip sampled for fifteen minutes and was killed by earlyoom in its
 * final decode, with the ComfyUI process at 19.4 GB and the machine's swap
 * gone. The registry records two measured points on the same 30.5 GB
 * machine, and this module holds the clip against them rather than against
 * a model of memory nobody measured:
 *
 *   49 frames at 832 x 480   peaked near 20 GB and survived
 *   81 frames at 832 x 480   peaked at 28.1 GB, exactly where earlyoom fires
 *
 * The second point is an edge, not a limit: the same render was killed the
 * next time because memory from earlier runs had stayed resident in the
 * long-lived ComfyUI process. So a clip larger than the edge is refused, a
 * clip between the two is cautioned, and before any clip on these families
 * ComfyUI is asked to release its cached models (releaseComfyMemory), which
 * is the registry's own advice ("restart the service between heavy renders")
 * without the restart.
 *
 * On a machine with clearly more memory than the one measured, nothing is
 * refused: the figures were not taken there, and a refusal would be a claim
 * about a machine nobody measured.
 */
import type { Hardware } from './hardware'

type Point = { width: number; height: number; frames: number; peakGb: number }

/** The machine the two points were measured on. */
const MEASURED_RAM_GB = 30.5
const SURVIVED: Point = { width: 832, height: 480, frames: 49, peakGb: 20 }
const EDGE: Point = { width: 832, height: 480, frames: 81, peakGb: 28.1 }

/** Above this much RAM the measurements no longer describe the machine. */
const BEYOND_MEASURED_GB = MEASURED_RAM_GB * 1.25

const pixelFrames = (p: { width: number; height: number; frames: number }) => p.width * p.height * p.frames

export type ClipMemory = {
  level: 'ok' | 'caution' | 'refuse'
  /** A sentence for the desk, or null when there is nothing to say. */
  reason: string | null
  /** True when ComfyUI should release its cached models before this clip runs. */
  release: boolean
}

const OK: ClipMemory = { level: 'ok', reason: null, release: false }

/**
 * The verdict for one clip. Only the two-model families (dualModel) are held
 * to the measurements; every other family gets `ok` and no release.
 */
export function clipMemory(
  family: { dualModel: boolean },
  clip: { width: number; height: number; frames: number },
  hardware: Hardware | null,
): ClipMemory {
  if (!family.dualModel) return OK
  const size = pixelFrames(clip)
  const ramGb = hardware ? hardware.ram.total / 1024 ** 3 : null
  const roomier = ramGb !== null && ramGb >= BEYOND_MEASURED_GB

  if (size > pixelFrames(EDGE)) {
    const times = (size / pixelFrames(EDGE)).toFixed(1)
    if (roomier) {
      return {
        level: 'caution',
        release: true,
        reason: `This clip is ${times} times the size that peaked at ${EDGE.peakGb} GB on a ${MEASURED_RAM_GB} GB machine (${EDGE.frames} frames at ${EDGE.width} × ${EDGE.height}). This machine has ${ramGb.toFixed(0)} GB, which was not measured, so it may fit.`,
      }
    }
    return {
      level: 'refuse',
      release: true,
      reason: `Too large for memory. ${EDGE.frames} frames at ${EDGE.width} × ${EDGE.height} was measured to peak at ${EDGE.peakGb} GB of ${MEASURED_RAM_GB} GB, which is where the machine kills the process, and this clip is ${times} times that size. It would sample for several minutes and then be killed in its final decode. Shorten it or make it smaller.`,
    }
  }

  if (size > pixelFrames(SURVIVED)) {
    return {
      level: 'caution',
      release: true,
      reason: `Larger than the ${SURVIVED.frames} frames at ${SURVIVED.width} × ${SURVIVED.height} measured to fit (about ${SURVIVED.peakGb} GB). A clip this size has been killed for memory at its final decode, so ComfyUI's cached models are released before it runs.`,
    }
  }

  return { level: 'ok', reason: null, release: true }
}

/**
 * Ask ComfyUI to unload every cached model and free memory before its next
 * job. The flags are applied by ComfyUI's worker when the next prompt starts,
 * so this is sent immediately before queueing. Never throws: a failure only
 * means the clip runs with whatever is resident, as it did before.
 */
export async function releaseComfyMemory(): Promise<boolean> {
  try {
    const res = await fetch('/comfy/free', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ unload_models: true, free_memory: true }),
    })
    return res.ok
  } catch {
    return false
  }
}
