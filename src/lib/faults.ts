/**
 * What went wrong, in one shape, for every desk.
 *
 * ComfyUI reports a refusal per node: which node, what class, and a sentence
 * per bad input. comfy.ts attaches all of that to the error it throws. The
 * Pictures desk read it and printed "The trouble is in KSampler (node 3)";
 * the Video and Reel desks duck-typed `{message, cancelled}` and threw the
 * rest away, so a clip that failed on a bad frame said "Something went wrong"
 * where a picture would have named the node. One classifier now, and one set
 * of sentences.
 */
import { ComfyError } from './comfy'

export type Fault = {
  message: string
  cancelled: boolean
  /** True when ComfyUI no longer has any record of the job. */
  lost: boolean
  node: string | null
  nodeType: string | null
  /** Every per-input complaint ComfyUI made, joined. Null when it made none. */
  detail: string | null
  /**
   * True when ComfyUI did not answer at all: down, restarting, or the
   * connection dropped. The job was never judged, so it is not called failed.
   */
  unreachable?: boolean
  /**
   * True for a lost job ComfyUI says has ended without sending its result:
   * it may have finished, with its file on disk and no record yet.
   */
  mayExist?: boolean
}

/** The per-node complaints, flattened into one line. */
function nodeErrorDetail(nodeErrors: Record<string, unknown> | null | undefined): string | null {
  if (!nodeErrors) return null
  const lines = Object.values(nodeErrors).flatMap((n) => {
    const errs = (n as { errors?: { message?: string; details?: string }[] })?.errors ?? []
    return errs.map((e) => [e.message, e.details].filter(Boolean).join(': '))
  })
  const joined = lines.filter(Boolean).join('; ')
  return joined || null
}

export function faultOf(err: unknown): Fault {
  if (err instanceof ComfyError) {
    return {
      message: err.message,
      cancelled: err.cancelled,
      lost: false,
      node: err.node,
      nodeType: err.nodeType,
      detail: nodeErrorDetail(err.nodeErrors),
      ...(err.unreachable ? { unreachable: true } : {}),
    }
  }
  const e = err as { message?: unknown; cancelled?: unknown; lost?: unknown; mayExist?: unknown } | null
  return {
    message: err instanceof Error ? err.message : typeof e?.message === 'string' ? e.message : String(err),
    cancelled: e?.cancelled === true,
    lost: e?.lost === true,
    node: null,
    nodeType: null,
    detail: null,
    ...(e?.lost === true && e?.mayExist === true ? { mayExist: true } : {}),
  }
}

function outOfMemory(f: Fault): boolean {
  const text = `${f.message} ${f.detail ?? ''}`.toLowerCase()
  return text.includes('out of memory') || text.includes('cuda') || text.includes('alloc') || /\boom\b/.test(text)
}

export function faultTitle(f: Fault): string {
  if (f.cancelled) return 'Correction'
  if (f.lost) return 'We lost track of that job'
  // Nothing reached ComfyUI's queue, so the job neither failed nor was refused.
  if (f.unreachable) return 'ComfyUI is not answering'
  if (outOfMemory(f)) return 'The card ran out of memory'
  // Only the queue's refusal carries per-input detail. A node that failed
  // while running is named too, and calling that job rejected would be false:
  // ComfyUI accepted it and it broke part way.
  if (f.detail) return 'That job was rejected'
  return 'That job did not finish'
}

export type FaultBodyOptions = {
  /**
   * The desk has held what was waiting behind this job rather than send it on
   * by itself. The Video desk does this when a heavy clip is lost, since the
   * likeliest cause is a restart for lack of memory and the next clip needs as
   * much.
   */
  held?: boolean
}

export function faultBody(f: Fault, opts: FaultBodyOptions = {}): string {
  if (f.cancelled) return 'Job stopped. Nothing was saved.'
  // When the desk has held what was waiting behind a lost job, the lost job
  // is not one to invite a second go at: whatever took ComfyUI down may well
  // take it down again. Say instead that the rest waits for the reader.
  if (f.lost && opts.held) {
    const look = f.mayExist ? ' Look there before you run it again.' : ''
    return `${f.message}${look} Anything waiting its turn is held until you send it or call it off.`
  }
  // A job that may have finished is not one to simply run again: a long clip
  // may already be on disk, waiting to be filed.
  if (f.lost && f.mayExist) return `${f.message} The desk is free again; look there before you run it again.`
  if (f.lost) return `${f.message} The desk is free again, so you can try again.`
  if (outOfMemory(f)) {
    return 'This size needs more than the card has free. Try a smaller shape or a shorter clip, or close anything else using the GPU.'
  }
  if (f.detail) return `ComfyUI would not accept it: ${f.detail}`
  return f.message || 'ComfyUI did not say why. Check its log and try again.'
}

/**
 * "The trouble is in KSampler (node 3)." Null when ComfyUI named no node, and
 * for a stop: ComfyUI names the node an interrupt landed on too, and a job
 * the reader stopped had no trouble in it.
 */
export function faultWhere(f: Fault): string | null {
  if (f.cancelled || !f.nodeType) return null
  return `The trouble is in ${f.nodeType}${f.node ? ` (node ${f.node})` : ''}.`
}
