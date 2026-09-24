/**
 * Will a clip on a two-model Wan 2.2 14B family survive its own decode?
 *
 * The memory verdict in hardware.ts counts weights, and the weights of a 14B
 * pair fit: two experts at about 9.3 GB each plus the encoder. What does not
 * fit is the working set of a long clip on top of them, and it arrives at the
 * very end, when the sampled latent is decoded. The registry records what the
 * two pairs did on one 30.5 GB machine, and this module holds each pair to
 * its own record rather than to a model of memory nobody measured.
 *
 * The text-to-video pair loads no add-on of its own. Its record, bare:
 *
 *   49 frames at 832 x 480 ran near 20 GB and came through
 *   earlyoom kills ComfyUI at about 28.1 GB used of 30.5 GB
 *   a second render in the same long-lived ComfyUI process was killed at
 *   about 20 GB, because memory from earlier runs had stayed resident
 *
 * Nothing larger than the 49-frame clip was measured. The line drawn above
 * it is the family's own default, 81 frames at 832 x 480, about 1.65 times
 * the clip that fit: an estimate of where a clip stops fitting under the kill
 * line, not a measurement, and the copy says so. A clip larger than that line
 * is refused and a clip between the two is cautioned. Releasing ComfyUI's
 * cached models before the clip (releaseComfyMemory) is the registry's own
 * advice ("restart the service between heavy renders") without the restart,
 * and it answers only the second kind of kill: it clears what earlier runs
 * left behind and makes no room for the clip itself.
 *
 * The image-to-video pair always loads an add-on on each half, and it is
 * worse. The registry lists one result for each setting it tried, by length
 * and add-on count and not by size: with both add-ons 81 frames was killed
 * and 49 frames came through, and with an add-on on only one half 49 frames
 * was killed. The kills came 11 to 12 seconds after ComfyUI began loading the
 * decoder beside the two experts, so inside the run and out of reach of any
 * release. It also speaks of three runs all killed, beside the one that came
 * through, so how often 81 frames was tried is not clear: the copy gives no
 * count and no size for those runs, and a clip is measured against the
 * pair's default size, 832 x 480, which the copy names as such. The registry
 * calls the pair not reliable on 30 GB, warns that the one success is no
 * proof 49 frames is safe, and names the 5B as the model for a clip from a
 * start frame. So anything larger than 49 frames at 832 x 480 is refused,
 * and anything else is cautioned, never simply let through.
 *
 * Neither pair was measured with anything on the rack, and the registry says
 * of the image-to-video pair, which peaked at 28.1 GB with an add-on on one
 * half, that any addition tips it. Add-ons the reader chains make the verdict
 * one step stricter: what would pass is cautioned and what would be cautioned
 * is refused.
 *
 * On a machine with clearly more memory than the one measured, nothing is
 * refused: the figures were not taken there, and a refusal would be a claim
 * about a machine nobody measured.
 */
import { withDeadline } from './comfy'
import type { Hardware } from './hardware'

type Point = { width: number; height: number; frames: number; peakGb?: number }

/** The machine every point below was measured on. */
const MEASURED_RAM_GB = 30.5

/** The text-to-video pair, bare: the one clip its record measured. */
const SURVIVED: Point = { width: 832, height: 480, frames: 49, peakGb: 20 }
/**
 * Where the machine kills ComfyUI: earlyoom fires at 8% free, about 28.1 GB
 * used of 30.5. A property of the machine, not a peak any clip was measured at.
 */
const KILL_GB = 28.1
/**
 * The line above which a text-to-video clip is refused: the family's default
 * length and size. An estimate, not a measurement (see the header).
 */
const LINE_EST: Point = { width: 832, height: 480, frames: 81 }

/**
 * The image-to-video pair, with its own add-on on each half. Only the frame
 * counts are recorded; the size is the pair's default, which the runs are
 * taken to have used.
 */
const I2V_ID = 'wan22-14b-i2v'
const I2V_CAME_THROUGH: Point = { width: 832, height: 480, frames: 49 }
const I2V_KILLED: Point = { width: 832, height: 480, frames: 81 }
/** The model the registry names for a clip from a start frame instead, and what it was measured doing. */
const SAFER_I2V = 'Wan 2.2 TI2V 5B, which made a 49-frame clip from a start frame at 1280 × 704 and peaked at 8.6 GB'

/** Above this much RAM the measurements no longer describe the machine. */
const BEYOND_MEASURED_GB = MEASURED_RAM_GB * 1.25

const pixelFrames = (p: { width: number; height: number; frames: number }) => p.width * p.height * p.frames
const at = (p: Point) => `${p.frames} frames at ${p.width} × ${p.height}`

export type ClipMemory = {
  level: 'ok' | 'caution' | 'refuse'
  /** A sentence for the desk, or null when there is nothing to say. */
  reason: string | null
  /** True when ComfyUI should release its cached models before this clip runs. */
  release: boolean
}

const OK: ClipMemory = { level: 'ok', reason: null, release: false }
const caution = (reason: string): ClipMemory => ({ level: 'caution', reason, release: true })
const refuse = (reason: string): ClipMemory => ({ level: 'refuse', reason, release: true })

/**
 * The verdict for one clip. Only the two-model families (dualModel) are held
 * to the measurements; every other family gets `ok` and no release.
 *
 * `family.id` picks which pair's record applies; without it the clip is held
 * to the text-to-video pair's. `addOns` is how many add-on files from the
 * rack will be chained into the graph, the family's own not counted.
 */
export function clipMemory(
  family: { id?: string; dualModel: boolean },
  clip: { width: number; height: number; frames: number },
  hardware: Hardware | null,
  addOns = 0,
): ClipMemory {
  if (!family.dualModel) return OK
  const size = pixelFrames(clip)
  const ramGb = hardware ? hardware.ram.total / 1024 ** 3 : null
  const here = ramGb !== null && ramGb >= BEYOND_MEASURED_GB ? ramGb : null
  return family.id === I2V_ID ? imageToVideo(size, addOns > 0, here) : textToVideo(size, addOns > 0, here)
}

/** Said instead of a refusal on a machine clearly roomier than the one measured. */
const unmeasured = (ramGb: number) => `This machine has ${ramGb.toFixed(0)} GB, which was not measured, so it may fit.`

/**
 * The text-to-video pair against its record. `roomier` is the machine's
 * memory in GB when it is clearly more than the measured machine's.
 */
function textToVideo(size: number, rack: boolean, roomier: number | null): ClipMemory {
  const onTop = rack ? ' Its add-ons load on top of that, and the pair was measured with none.' : ''
  const measured = `${at(SURVIVED)} measured near ${SURVIVED.peakGb} GB`
  const killLine = `the machine kills ComfyUI at about ${KILL_GB} GB used`
  if (size > pixelFrames(LINE_EST)) {
    const times = (size / pixelFrames(SURVIVED)).toFixed(1)
    if (roomier !== null) {
      return caution(
        `This clip is ${times} times the ${measured} on a ${MEASURED_RAM_GB} GB machine, which kills ComfyUI at about ${KILL_GB} GB used.${onTop} ${unmeasured(roomier)}`,
      )
    }
    return refuse(
      `Too large for memory. On a ${MEASURED_RAM_GB} GB machine this pair made ${at(SURVIVED)} near ${SURVIVED.peakGb} GB, and ${killLine}. This clip is ${times} times that size, more than the family's default of ${at(LINE_EST)}, and nothing that large has been measured here.${onTop} It would sample for several minutes and is expected to be killed in its final decode. Shorten it or make it smaller.`,
    )
  }

  if (size > pixelFrames(SURVIVED)) {
    if (!rack) {
      return caution(
        `Larger than the ${at(SURVIVED)} measured to fit (about ${SURVIVED.peakGb} GB), and nothing this size has been measured on this pair. The machine kills ComfyUI at about ${KILL_GB} GB used, and a second render on this pair was killed at about ${SURVIVED.peakGb} GB once memory from earlier runs had built up. ComfyUI releases its cached models before this clip runs, which clears what earlier clips left behind and nothing more.`,
      )
    }
    const why = `The ${at(SURVIVED)} measured to fit (about ${SURVIVED.peakGb} GB) ran with no add-ons, and ${killLine}. This clip is larger than the one that fit, and its add-ons load on top.`
    if (roomier !== null) return caution(`${why} ${unmeasured(roomier)}`)
    return refuse(
      `Too large for memory with add-ons. ${why} Take the add-ons off, or make it no larger than ${at(SURVIVED)}.`,
    )
  }

  if (rack) {
    return caution(
      `The ${at(SURVIVED)} measured to fit (about ${SURVIVED.peakGb} GB) ran with no add-ons. The ones on the rack load on top of that, which nobody has measured.`,
    )
  }
  return { level: 'ok', reason: null, release: true }
}

/**
 * The image-to-video pair against its own record. Nothing here is said to be
 * fixed by the release: its kills happened inside the run, as the decoder
 * loaded.
 */
function imageToVideo(size: number, rack: boolean, roomier: number | null): ClipMemory {
  const larger = size > pixelFrames(I2V_CAME_THROUGH)
  const killed = `${I2V_KILLED.frames} frames`
  const cameThrough = `${I2V_CAME_THROUGH.frames} frames`
  const record = `On a ${MEASURED_RAM_GB} GB machine this pair, with an add-on on each half, was killed at ${killed} as its final decode began, and came through once at ${cameThrough}.`
  const largerThan = `This clip is larger than ${at(I2V_CAME_THROUGH)}, the pair's default size.`

  if (roomier !== null) {
    if (!larger && !rack) return { level: 'ok', reason: null, release: true }
    return caution(
      `${record}${larger ? ` ${largerThan}` : ''}${rack ? ' The add-ons on the rack load on top of the one it already carries on each half.' : ''} ${unmeasured(roomier)}`,
    )
  }

  if (larger) {
    return refuse(`Too large for memory. ${record} ${largerThan} Shorten it, or use ${SAFER_I2V}.`)
  }
  if (rack) {
    return refuse(
      `Too much for memory with add-ons. This pair already loads an add-on on each half. With those alone it was killed at ${killed}, and with only one of them it was killed at ${cameThrough}. Anything on the rack loads on top. Take the add-ons off, or use ${SAFER_I2V}.`,
    )
  }
  return caution(
    `This pair is not reliable on a ${MEASURED_RAM_GB} GB machine. At ${cameThrough} it came through once, and was killed once with an add-on on only one half. At ${killed} it was killed as its final decode began, while its own two models were loaded, and a release before the run cannot unload those. For a clip from a start frame the safer choice is ${SAFER_I2V}.`,
  )
}

/** How long one read of ComfyUI's queue may take before it counts as unanswered. */
const QUEUE_READ_MS = 10_000

type QueueItem = unknown[]
type QueueState = { running: QueueItem[]; pending: QueueItem[] }

/**
 * One read of ComfyUI's queue. Throws when ComfyUI does not answer with one:
 * a network error, a read that takes too long, or the proxy's empty 502 while
 * ComfyUI is down or restarting.
 */
async function readQueue(signal?: AbortSignal): Promise<QueueState> {
  return withDeadline(QUEUE_READ_MS, signal, async (s) => {
    const res = await fetch('/comfy/queue', { signal: s })
    if (!res.ok) throw new Error(`/queue -> HTTP ${res.status}`)
    const q = (await res.json()) as { queue_running?: unknown; queue_pending?: unknown }
    const list = (v: unknown) => (Array.isArray(v) ? (v.filter(Array.isArray) as QueueItem[]) : [])
    return { running: list(q.queue_running), pending: list(q.queue_pending) }
  })
}

/**
 * Wait until ComfyUI has nothing running and nothing queued.
 *
 * A release (below) is a flag ComfyUI's worker reads when it takes its NEXT
 * prompt, whichever prompt that is. Sent while other work is queued, it is
 * spent on that work, and the heavy clip it was meant for starts with every
 * model the earlier jobs loaded still resident, which is exactly how a 14B
 * clip gets killed at its final decode. So a clip that needs a release waits
 * for an empty queue, releases, and only then submits: sent in that order
 * with nothing between, the release is read by that clip.
 *
 * Polls the queue every two seconds. `onWait` hears how many jobs are ahead
 * each time it has to wait, or -1 while ComfyUI is not answering, for the
 * desk to say so. Resolves true once a read shows the queue empty, and false
 * if `signal` aborts first.
 *
 * It keeps waiting while ComfyUI does not answer. That is usually earlyoom
 * having killed it and systemd bringing it back, empty, a few seconds later:
 * waiting it out sends the clip onto a clean card. Giving up on the first
 * unanswered read sent the clip into the dead server, where it failed as if
 * the queue had refused it and left the lane with nothing to send again.
 */
export async function waitForIdleComfy(
  signal?: AbortSignal,
  onWait?: (ahead: number) => void,
): Promise<boolean> {
  for (;;) {
    if (signal?.aborted) return false
    let ahead: number
    try {
      const q = await readQueue(signal)
      ahead = q.running.length + q.pending.length
    } catch {
      if (signal?.aborted) return false
      ahead = -1
    }
    if (ahead === 0) return true
    // A stop that landed while the queue was being read has already fired its
    // abort event, so the sleep below would never hear it: check it here.
    if (signal?.aborted) return false
    onWait?.(ahead)
    await new Promise<void>((resolve) => {
      if (signal?.aborted) return resolve()
      const t = setTimeout(resolve, 2000)
      signal?.addEventListener('abort', () => { clearTimeout(t); resolve() }, { once: true })
    })
  }
}

/**
 * Ask ComfyUI to unload every cached model and free memory before its next
 * job. The flags are applied by ComfyUI's worker when the next prompt starts,
 * so this must be sent with an empty queue (see waitForIdleComfy) and
 * immediately before queueing. Never throws: a failure only means the clip
 * runs with whatever is resident, as it did before.
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

/**
 * Release again when a heavy clip's release may have been spent on other work.
 *
 * The clip released on an empty queue, but anything sent in the moment
 * between that release and the clip's own prompt (another tab, a picture from
 * this page, a job from ComfyUI's own page) is taken first and reads the flag
 * instead. Called once the clip's prompt is accepted: if ComfyUI lists any
 * prompt other than `promptId` as running or waiting, the release is sent
 * again, and ComfyUI applies it after the work in front and before the clip.
 * One that turns out to be spare costs the next job a model load. Never
 * throws; a queue that cannot be read is left alone.
 */
export async function releaseIfOthersAhead(promptId: string): Promise<void> {
  let q: QueueState
  try {
    q = await readQueue()
  } catch {
    return
  }
  const others = [...q.running, ...q.pending].some((item) => String(item[1]) !== promptId)
  if (others) await releaseComfyMemory()
}
