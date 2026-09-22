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
}

/** The per-node complaints, flattened into one line. */
export function nodeErrorDetail(nodeErrors: Record<string, unknown> | null | undefined): string | null {
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
    }
  }
  const e = err as { message?: unknown; cancelled?: unknown; lost?: unknown } | null
  return {
    message: err instanceof Error ? err.message : typeof e?.message === 'string' ? e.message : String(err),
    cancelled: e?.cancelled === true,
    lost: e?.lost === true,
    node: null,
    nodeType: null,
    detail: null,
  }
}

function outOfMemory(f: Fault): boolean {
  const text = `${f.message} ${f.detail ?? ''}`.toLowerCase()
  return text.includes('out of memory') || text.includes('cuda') || text.includes('alloc') || /\boom\b/.test(text)
}

export function faultTitle(f: Fault): string {
  if (f.cancelled) return 'Correction'
  if (f.lost) return 'We lost track of that job'
  if (outOfMemory(f)) return 'The card ran out of memory'
  if (f.detail || f.nodeType) return 'That job was rejected'
  return 'That job did not finish'
}

export function faultBody(f: Fault): string {
  if (f.cancelled) return 'Job stopped. Nothing was saved.'
  if (f.lost) return `${f.message} The desk is free again, so you can try again.`
  if (outOfMemory(f)) {
    return 'This size needs more than the card has free. Try a smaller shape or a shorter clip, or close anything else using the GPU.'
  }
  if (f.detail) return `ComfyUI would not accept it: ${f.detail}`
  return f.message || 'ComfyUI did not say why. Check its log and try again.'
}

/** "The trouble is in KSampler (node 3)." Null when ComfyUI named no node. */
export function faultWhere(f: Fault): string | null {
  if (!f.nodeType) return null
  return `The trouble is in ${f.nodeType}${f.node ? ` (node ${f.node})` : ''}.`
}
