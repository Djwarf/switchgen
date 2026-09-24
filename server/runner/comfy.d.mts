import type { ApiWorkflow } from './comfyRecord.mjs'

/** ComfyUI did not answer a read with its own JSON: down, restarting, too slow, or not ComfyUI. */
export class Unanswered extends Error {
  constructor(message?: string, options?: { cause?: unknown })
}

export type QueueIds = { running: string[]; pending: string[] }

export type SubmitResult =
  | { accepted: true }
  | {
      refused: true
      status: number
      message: string
      node: string | null
      nodeType: string | null
      nodeErrors: Record<string, unknown> | null
    }
  /** The connection was never made: nothing reached ComfyUI, so sending again is safe. */
  | { unreached: true }
  /** No clear answer: the prompt may be queued. Ask by its id; never send it again. */
  | { unknown: true; reason: string }

export type ComfyJob = {
  id: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
  execution_start_time?: number | null
}

export type SocketMessage =
  | {
      type:
        | 'status'
        | 'progress'
        | 'executing'
        | 'execution_start'
        | 'execution_cached'
        | 'execution_success'
        | 'execution_error'
        | 'execution_interrupted'
      data: Record<string, any>
    }
  | { type: 'preview'; promptId: string | null; mime: string; bytes: Buffer }

export type ComfySocket = {
  /**
   * Close for good. The real socket's promise settles once it has closed, or
   * after 2 s, so a runner taking over can wait before it opens its own on
   * the same client id; a caller that does not need that may ignore it.
   */
  close(): void | Promise<void>
}

export type Comfy = {
  /** The ComfyUI address in use, with no trailing slash. */
  readonly url: string
  /** Prompt ids running and waiting. Throws Unanswered when there is no queue to read. */
  readQueue(): Promise<QueueIds>
  /** POST /free {unload_models, free_memory}. Never throws. */
  free(): Promise<'ok' | 'unreached' | 'failed'>
  /** POST /prompt under the given prompt id, with a 30 s deadline. Never throws. */
  submit(graph: ApiWorkflow, promptId: string, clientId: string): Promise<SubmitResult>
  /** Null on a 404; throws Unanswered when ComfyUI does not answer. */
  getJob(id: string): Promise<ComfyJob | null>
  /**
   * Whether ComfyUI has the jobs list: false when a read of an id no job has
   * gets a 404 that is not ComfyUI's JSON, as from a ComfyUI older than the
   * list. Throws Unanswered when ComfyUI does not answer. A client without it
   * (a test's stand-in) is taken to have the list.
   */
  hasJobsList?(): Promise<boolean>
  /** The raw /history entry, or null when ComfyUI keeps none. Throws Unanswered when it does not answer. */
  history(id: string): Promise<Record<string, any> | null>
  /** POST /api/jobs/{id}/cancel. True only when ComfyUI says it cancelled; never throws. */
  cancel(id: string): Promise<boolean>
  /** POST /interrupt {prompt_id}. Does nothing for an empty id; never throws. */
  interrupt(id: string): Promise<void>
  /**
   * One socket for `clientId`, reconnecting with a wait from 0.5 s to 8 s.
   * Its first message declares preview metadata.
   */
  socket(clientId: string, on: (msg: SocketMessage) => void): ComfySocket
}

export function createComfy(opts?: {
  /** Defaults to COMFY_URL, else http://127.0.0.1:8188, read when this is called. */
  url?: string
  fetch?: typeof globalThis.fetch
  WebSocket?: new (url: string) => any
  /** A read's deadline, body included. 10 s unless a test shortens it. */
  readMs?: number
  /** A send's deadline. 30 s unless a test shortens it. */
  sendMs?: number
}): Comfy

/** One binary socket frame as a preview (events 1 and 4), or null. */
export function previewOf(buf: Buffer): Extract<SocketMessage, { type: 'preview' }> | null
