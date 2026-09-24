/**
 * The runner's line to ComfyUI: plain HTTP and one socket, straight to
 * COMFY_URL with no proxy between.
 *
 * This is the only part of the runner that does I/O with ComfyUI, and the
 * rules it keeps are the ones the page's transport (src/lib/comfy.ts) learned:
 *
 *   - Every read has a deadline that covers the body, 10 s, and a send has
 *     30 s. A ComfyUI that keeps its sockets open but stops answering (a
 *     machine swapping just before earlyoom acts) must cost a wait, not a
 *     request held open for good.
 *   - "No answer" is never read as "no such job". A read that is not
 *     ComfyUI's JSON throws {@link Unanswered}; only ComfyUI's own 404 for a
 *     job means it has none. Reading an outage as an absence would call a
 *     job lost, or a send unsent, while ComfyUI was only restarting.
 *   - A send is sorted by what is known about it, because the runner must
 *     never send a job twice. `unreached` means the connection was refused
 *     before a byte of the prompt was written, so sending again is safe.
 *     Anything else that is not a clear answer is `unknown`: the prompt may
 *     be in ComfyUI's queue, and the runner asks by the prompt id rather than
 *     sending again.
 *   - A stop is per prompt. POST /interrupt without a prompt id stops
 *     whatever is sampling, so {@link interrupt} never sends one without it.
 *
 * Every function takes the prompt id the runner minted, so ComfyUI can be
 * asked about a job whose answer never came back.
 */

import { randomUUID } from 'node:crypto'

/** Raised when ComfyUI did not answer a read with its own JSON. */
export class Unanswered extends Error {
  constructor(message, options) {
    super(message, options)
    this.name = 'Unanswered'
  }
}

/** How long a read may take, body included. */
const READ_MS = 10_000
/**
 * How long a send may take. ComfyUI checks the whole graph before it answers,
 * which for a large graph on a busy machine is slower than a read.
 */
const SEND_MS = 30_000

/**
 * Error codes that mean the connection was never made, so nothing of the
 * request reached ComfyUI. Each is from the connect or the name lookup; a
 * reset, a timeout or a socket closed part way are not here, because by then
 * the prompt may have been written.
 */
const UNREACHED = new Set(['ECONNREFUSED', 'EHOSTUNREACH', 'ENETUNREACH', 'ENOTFOUND', 'EAI_AGAIN'])

/** The socket's reconnect wait, doubling from the first to the last. */
const BACKOFF_FIRST_MS = 500
const BACKOFF_LAST_MS = 8000

/** The socket messages passed on; the rest (executed, progress_state, feature_flags) the runner does not use. */
const SOCKET_TYPES = new Set([
  'status',
  'progress',
  'executing',
  'execution_start',
  'execution_cached',
  'execution_success',
  'execution_error',
  'execution_interrupted',
])

/**
 * True when a failed fetch never reached ComfyUI. Node's fetch wraps the
 * socket's error as `cause`; a host that resolves to two addresses (localhost
 * as ::1 and 127.0.0.1) reports an AggregateError, whose own code is the
 * first attempt's, so every attempt is checked. "bad port" is fetch refusing
 * a port on its blocked list (9 is one) before it connects at all.
 */
function unreached(err) {
  const cause = err?.cause
  if (!cause) return false
  if (UNREACHED.has(cause.code)) return true
  if (Array.isArray(cause.errors) && cause.errors.length) return cause.errors.every((e) => UNREACHED.has(e?.code))
  return cause.message === 'bad port'
}

/** A short line saying why a request failed, for a log or a wait reason. */
function whyFailed(err, signal, ms) {
  if (signal?.aborted) return `no answer within ${ms >= 1000 ? `${Math.round(ms / 1000)} s` : `${ms} ms`}`
  const cause = err?.cause
  return String(cause?.code ?? cause?.message ?? err?.message ?? err)
}

/**
 * Run `work` with a signal that aborts after `ms`. The body must be read
 * inside `work`, so the deadline covers it as well as the headers.
 */
async function within(ms, work) {
  const ctl = new AbortController()
  const timer = setTimeout(() => ctl.abort(), ms)
  try {
    return await work(ctl.signal)
  } finally {
    clearTimeout(timer)
  }
}

/** Parse a body as JSON, or null when it is not JSON. */
function parse(text) {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v)

/**
 * ComfyUI's refusal of a graph, worded as the page words it (submit in
 * src/lib/comfy.ts), so a refusal reads the same whichever of the two sent
 * the job. One refused node is named outright; with several, the per-input
 * detail names them all and no single one is "the" trouble.
 */
function refusalOf(status, body) {
  const nodeErrors = isObject(body.node_errors) ? body.node_errors : null
  const detail =
    body.error?.message ??
    (typeof body.error === 'string' ? body.error : null) ??
    Object.values(nodeErrors ?? {})
      .flatMap((n) => (Array.isArray(n?.errors) ? n.errors : []).map((e) => e?.message))
      .filter(Boolean)
      .join('; ')
  const refused = Object.entries(nodeErrors ?? {})
  const only = refused.length === 1 ? refused[0] : null
  const onlyType = only?.[1]?.class_type
  return {
    refused: true,
    status,
    message: detail ? String(detail) : `The queue rejected the job (HTTP ${status}).`,
    node: only ? only[0] : null,
    nodeType: typeof onlyType === 'string' && onlyType ? onlyType : null,
    nodeErrors,
  }
}

/**
 * Split one binary socket frame into a preview, or null when it is not one.
 *
 *   event 1  PREVIEW_IMAGE                [0..4) event, [4..8) 1=JPEG 2=PNG, rest = bytes
 *   event 4  PREVIEW_IMAGE_WITH_METADATA  [0..4) event, [4..8) JSON length,
 *                                         then UTF-8 JSON, then bytes
 *
 * Big-endian throughout, as ComfyUI packs them (server.py send_image and
 * send_image_with_metadata). Only the second names its prompt; the first is
 * the old shape, and its prompt id is null for the runner to attribute to
 * whatever it knows is running.
 */
export function previewOf(buf) {
  if (!Buffer.isBuffer(buf) || buf.length < 8) return null
  const event = buf.readUInt32BE(0)
  if (event === 1) {
    return {
      type: 'preview',
      promptId: null,
      mime: buf.readUInt32BE(4) === 2 ? 'image/png' : 'image/jpeg',
      bytes: Buffer.from(buf.subarray(8)),
    }
  }
  if (event === 4) {
    const len = buf.readUInt32BE(4)
    if (8 + len > buf.length) return null
    const meta = parse(buf.subarray(8, 8 + len).toString('utf8'))
    return {
      type: 'preview',
      promptId: typeof meta?.prompt_id === 'string' && meta.prompt_id ? meta.prompt_id : null,
      mime: typeof meta?.image_type === 'string' && meta.image_type ? meta.image_type : 'image/jpeg',
      bytes: Buffer.from(buf.subarray(8 + len)),
    }
  }
  return null
}

/** A socket frame's bytes as a Buffer, whichever form the WebSocket in use hands them over in. */
async function bytesOf(data) {
  if (Buffer.isBuffer(data)) return data
  if (data instanceof ArrayBuffer) return Buffer.from(data)
  if (ArrayBuffer.isView(data)) return Buffer.from(data.buffer, data.byteOffset, data.byteLength)
  if (typeof data?.arrayBuffer === 'function') return Buffer.from(await data.arrayBuffer())
  return null
}

/**
 * A client for one ComfyUI. `url` defaults to COMFY_URL, read when this is
 * called. `fetch`, `WebSocket` and the two deadlines are there for tests,
 * which cannot wait 30 s for a send to run out.
 */
export function createComfy({ url, fetch: fetchImpl, WebSocket: WebSocketImpl, readMs = READ_MS, sendMs = SEND_MS } = {}) {
  const base = String(url ?? process.env.COMFY_URL ?? 'http://127.0.0.1:8188').replace(/\/+$/, '')
  const doFetch = fetchImpl ?? globalThis.fetch
  const Socket = WebSocketImpl ?? globalThis.WebSocket

  /**
   * One JSON read. Resolves `{status, body}` for an answer whose body is
   * JSON, and for a 404 whatever its body, and throws Unanswered for
   * anything else: no connection, the deadline, a gateway's 502 to 504, or a
   * body that is not JSON.
   */
  async function read(pathname) {
    return within(readMs, async (signal) => {
      let res
      let text
      try {
        res = await doFetch(`${base}${pathname}`, { signal })
        text = await res.text()
      } catch (err) {
        throw new Unanswered(`ComfyUI did not answer ${pathname}: ${whyFailed(err, signal, readMs)}`, { cause: err })
      }
      if (res.status === 502 || res.status === 503 || res.status === 504) {
        throw new Unanswered(`ComfyUI did not answer ${pathname}: HTTP ${res.status}`)
      }
      const body = parse(text)
      if (body === null && res.status !== 404) {
        throw new Unanswered(`ComfyUI answered ${pathname} with something that is not its JSON (HTTP ${res.status})`)
      }
      return { status: res.status, body }
    })
  }

  /** One POST whose answer does not matter beyond its status. Never throws. */
  async function post(pathname, body) {
    return within(readMs, async (signal) => {
      try {
        const res = await doFetch(`${base}${pathname}`, {
          method: 'POST',
          signal,
          ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
        })
        const text = await res.text().catch(() => '')
        return { ok: res.ok, status: res.status, body: parse(text) }
      } catch (err) {
        return { ok: false, status: 0, body: null, unreached: unreached(err) }
      }
    })
  }

  /** The prompt ids ComfyUI has running and waiting, in its own order. */
  async function readQueue() {
    const { status, body } = await read('/queue')
    // Any other status, or JSON that is not the queue, is still not a queue
    // to act on: a release sent on it could be spent on work it did not see.
    if (status < 200 || status >= 300 || !isObject(body)) {
      throw new Unanswered(`ComfyUI answered /queue with HTTP ${status} and no queue`)
    }
    const ids = (v) => (Array.isArray(v) ? v.filter((item) => Array.isArray(item) && item[1] != null).map((item) => String(item[1])) : [])
    return { running: ids(body.queue_running), pending: ids(body.queue_pending) }
  }

  /**
   * Ask ComfyUI to unload every model and free memory before its next job.
   * It only sets flags the worker reads when it takes its next prompt, so
   * the caller sends it on an empty queue, just before the heavy prompt.
   */
  async function free() {
    const r = await post('/free', { unload_models: true, free_memory: true })
    if (r.ok) return 'ok'
    return r.unreached ? 'unreached' : 'failed'
  }

  /** Queue `graph` under `promptId`, and say what is known about whether it landed. */
  async function submit(graph, promptId, clientId) {
    return within(sendMs, async (signal) => {
      let res
      try {
        res = await doFetch(`${base}/prompt`, {
          method: 'POST',
          signal,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: graph, client_id: clientId, prompt_id: promptId }),
        })
      } catch (err) {
        if (!signal.aborted && unreached(err)) return { unreached: true }
        return { unknown: true, reason: whyFailed(err, signal, sendMs) }
      }
      let text
      try {
        text = await res.text()
      } catch (err) {
        // The answer began and was cut off: whatever it said, it was said
        // after ComfyUI had the prompt.
        return { unknown: true, reason: whyFailed(err, signal, sendMs) }
      }
      const body = parse(text)
      if (res.ok) {
        // ComfyUI's acceptance is JSON. Anything else did not come from its
        // queue, and cannot say whether the queue has the prompt.
        return isObject(body) ? { accepted: true } : { unknown: true, reason: `HTTP ${res.status} with no JSON` }
      }
      if (res.status === 400 && isObject(body) && (body.error != null || body.node_errors != null)) {
        return refusalOf(res.status, body)
      }
      return { unknown: true, reason: `HTTP ${res.status}` }
    })
  }

  /**
   * One job as ComfyUI's jobs API reports it, or null when ComfyUI says it has
   * none. Only a 404 means that; anything else that is not an answer throws.
   * A 404 is taken whatever its body, as the page takes it. A ComfyUI too old
   * for the jobs API answers every id so, running or not, which is why the
   * runner does not run on one (see hasJobsList).
   */
  async function getJob(id) {
    const { status, body } = await read(`/api/jobs/${encodeURIComponent(id)}`)
    if (status === 404) return null
    if (status < 200 || status >= 300 || !isObject(body) || typeof body.status !== 'string') {
      throw new Unanswered(`ComfyUI answered /api/jobs with HTTP ${status} and no job`)
    }
    const start = body.execution_start_time
    return {
      id: typeof body.id === 'string' && body.id ? body.id : id,
      status: body.status,
      execution_start_time: typeof start === 'number' ? start : null,
    }
  }

  /**
   * Whether this ComfyUI has the jobs list (/api/jobs) the runner follows its
   * work by. Asked about an id no job has: a ComfyUI with the list answers
   * its own JSON 404, and one older than it answers aiohttp's plain one, as
   * it does for any route it lacks. On that one getJob is null for every id,
   * running or not, so the runner would call each job it sent lost. Throws
   * Unanswered when ComfyUI does not answer.
   */
  async function hasJobsList() {
    const { status, body } = await read(`/api/jobs/${randomUUID()}`)
    return !(status === 404 && body === null)
  }

  /** The raw /history entry for one prompt, or null when ComfyUI keeps none. */
  async function history(id) {
    const { status, body } = await read(`/history/${encodeURIComponent(id)}`)
    if (status < 200 || status >= 300 || !isObject(body)) {
      throw new Unanswered(`ComfyUI answered /history with HTTP ${status} and no history`)
    }
    const entry = body[id]
    return isObject(entry) ? entry : null
  }

  /**
   * Cancel one prompt, running or waiting. True only when ComfyUI says it
   * cancelled it; a job already ended, an unknown id and no answer are all
   * false, and it never throws.
   */
  async function cancel(id) {
    if (typeof id !== 'string' || !id) return false
    const r = await post(`/api/jobs/${encodeURIComponent(id)}/cancel`)
    return r.ok && r.body?.cancelled === true
  }

  /**
   * Stop the sampling of one running prompt. ComfyUI does nothing when that
   * prompt is not the one running. Never throws, and never sends a bare
   * interrupt, which would stop whatever is sampling, ours or not.
   */
  async function interrupt(id) {
    if (typeof id !== 'string' || !id) return
    await post('/interrupt', { prompt_id: id })
  }

  /**
   * One socket for this client id, kept open until `close()`.
   *
   * ComfyUI keeps one socket per client id and forgets the older when a
   * newer arrives, so this opens its next socket only after the last has
   * closed. The first thing it sends declares preview metadata, which makes
   * ComfyUI name the prompt on each preview (server.py only honours it as the
   * first message). Nothing heard here decides how a job ends; it is
   * progress, previews and a hint to look sooner.
   */
  function socket(clientId, on) {
    const wsUrl = `${base.replace(/^http/i, 'ws')}/ws?clientId=${encodeURIComponent(clientId)}`
    let closed = false
    let ws = null
    let timer = null
    let backoff = BACKOFF_FIRST_MS
    let whenClosed = null
    let closing = null
    let warned = false

    const deliver = (msg) => {
      try {
        const r = on(msg)
        if (r && typeof r.then === 'function') r.then(undefined, (err) => console.warn(`[switchgen-runner] socket listener failed: ${err?.message ?? err}`))
      } catch (err) {
        console.warn(`[switchgen-runner] socket listener failed: ${err?.message ?? err}`)
      }
    }

    const heard = async (data) => {
      if (closed) return
      if (typeof data === 'string') {
        const msg = parse(data)
        if (isObject(msg) && SOCKET_TYPES.has(msg.type)) deliver({ type: msg.type, data: isObject(msg.data) ? msg.data : {} })
        return
      }
      let buf
      try {
        buf = await bytesOf(data)
      } catch {
        return
      }
      const preview = buf ? previewOf(buf) : null
      if (preview && !closed) deliver(preview)
    }

    const schedule = () => {
      if (closed || timer) return
      timer = setTimeout(openSocket, backoff)
      backoff = Math.min(backoff * 2, BACKOFF_LAST_MS)
    }

    function openSocket() {
      timer = null
      if (closed) return
      let s
      try {
        s = new Socket(wsUrl)
      } catch (err) {
        // No WebSocket in this runtime, or an address it will not take. The
        // runner still follows every job by asking; it loses progress only,
        // so this is said once, not at every retry.
        if (!warned) console.warn(`[switchgen-runner] could not open ComfyUI's socket: ${err?.message ?? err}`)
        warned = true
        schedule()
        return
      }
      ws = s
      try {
        s.binaryType = 'arraybuffer'
      } catch {
        /* a WebSocket that hands frames over as it likes; bytesOf reads each form */
      }
      s.onopen = () => {
        if (ws !== s) return
        backoff = BACKOFF_FIRST_MS
        try {
          s.send(JSON.stringify({ type: 'feature_flags', data: { supports_preview_metadata: true } }))
        } catch {
          /* it closed between open and send; onclose opens the next */
        }
      }
      s.onerror = () => {
        /* onclose always follows, and reconnects */
      }
      s.onclose = () => {
        if (ws !== s) return
        ws = null
        if (whenClosed) whenClosed()
        schedule()
      }
      s.onmessage = (ev) => {
        if (ws === s) void heard(ev?.data)
      }
    }

    openSocket()

    return {
      /**
       * Close for good. The promise settles once the socket has closed, or
       * after two seconds. A runner taking over waits for it before opening
       * its own socket on the same client id: when a socket closes, ComfyUI
       * forgets its client id even if a newer socket has registered that id
       * since (the `finally` of server.py's websocket_handler), and the newer
       * one would then hear nothing addressed to it.
       */
      close() {
        if (closing) return closing
        closed = true
        if (timer) {
          clearTimeout(timer)
          timer = null
        }
        const s = ws
        if (!s) {
          closing = Promise.resolve()
          return closing
        }
        closing = new Promise((resolve) => {
          const done = () => {
            clearTimeout(limit)
            whenClosed = null
            ws = null
            resolve()
          }
          const limit = setTimeout(done, 2000)
          whenClosed = done
          try {
            s.close()
          } catch {
            done()
          }
        })
        return closing
      },
    }
  }

  return { url: base, readQueue, free, submit, getJob, hasJobsList, history, cancel, interrupt, socket }
}
