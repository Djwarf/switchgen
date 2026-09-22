/**
 * ComfyUI transport.
 *
 * Talks to the ComfyUI service over HTTP and one WebSocket. Nothing from
 * ComfyUI's Python internals is imported, so `comfy update` cannot break this
 * layer. Vite proxies /comfy -> :8188 and /comfy-ws -> /ws, so the browser sees
 * a single origin.
 *
 * Three rules hold this module together:
 *
 *   1. ONE SOCKET. ComfyUI's `websocket_handler` does `sockets.pop(sid)` when a
 *      clientId reconnects (server.py:274). A second socket on the same
 *      clientId therefore evicts the first, and the first job's listeners go
 *      deaf forever. So the whole application shares a single lazily-opened,
 *      auto-reconnecting socket and messages are routed by `prompt_id`.
 *
 *   2. CANCEL IS PER PROMPT. `POST /interrupt` kills whatever is sampling right
 *      now, which is almost never the job the user pressed stop on. Use
 *      `cancelJob(promptId)` -> `POST /api/jobs/{id}/cancel`, which is
 *      prompt-targeted and cancels running *or* pending jobs.
 *
 *   3. VIDEOS ARRIVE UNDER "images". SaveWEBM returns `ui.PreviewVideo`, which
 *      serialises as `{"images":[...], "animated":[true]}`. Classification is
 *      done by `collectFiles()` from three independent signals; do not simplify
 *      it back to a key check.
 */

const HTTP = '/comfy'
const WS = '/comfy-ws'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ApiNode = { class_type: string; inputs: Record<string, unknown> }
export type ApiWorkflow = Record<string, ApiNode>

/** Anything ComfyUI can serve from /view. */
export type FileRef = {
  filename: string
  /** '' for the outputs root. */
  subfolder: string
  /** 'output' | 'input' | 'temp'. */
  type: string
}

/** A file a run produced, classified for the UI. */
export type OutputFile = FileRef & { kind: 'image' | 'video' }

export type ProgressEvent =
  /** Accepted by the queue. Nothing has started yet. */
  | { phase: 'queued'; promptId: string }
  /** A node is executing. `value`/`max` are sampler steps when known, else 0/1. */
  | { phase: 'running'; node: string | null; value: number; max: number }
  /**
   * A latent preview frame, as an object URL.
   *
   * The URL is owned by this module and is revoked when the next preview for
   * the same prompt arrives, and 60 seconds after the job ends. Render it;
   * do not store it beyond the job.
   */
  | { phase: 'preview'; url: string }
  | { phase: 'done'; files: OutputFile[] }
  /**
   * Terminal failure. `cancelled` is true when the job was stopped on purpose
   * (`execution_interrupted`), which wants different copy from a real error.
   */
  | { phase: 'error'; message: string; cancelled: boolean; node: string | null }

/** Connection state of the shared socket, for the offline notice. */
export type ConnectionState = 'connecting' | 'open' | 'closed'

/** One job as ComfyUI's own queue reports it (GET /api/jobs). */
export type ServerJobStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled'

export type ServerJob = {
  id: string
  status: ServerJobStatus
  priority?: number
  create_time?: number
  execution_start_time?: number | null
  execution_end_time?: number | null
  outputs_count?: number
  previewable_outputs_count?: number
  preview_output?: (FileRef & { nodeId?: string; mediaType?: string }) | null
}

export type ServerJobsPage = {
  jobs: ServerJob[]
  pagination: { offset: number; limit: number | null; total: number; has_more: boolean }
}

/** A finished run recovered from GET /history. */
export type PastRun = {
  promptId: string
  /** The exact API graph that was submitted. Every parameter is in here. */
  graph: ApiWorkflow
  files: OutputFile[]
  status: 'success' | 'error' | 'cancelled' | 'unknown'
  /** Epoch ms, from the history status messages. Null when absent. */
  startedAt: number | null
  finishedAt: number | null
  clientId: string | null
}

/**
 * The binding shape `registry.ts` uses, restated locally so this module does
 * not depend on generated data. Compatible with `FamilyDef['bindings']`.
 */
export type BindingMap = Partial<Record<string, [nodeId: string, input: string][]>>

/** An error from the queue or from execution, with the bits the UI needs. */
export class ComfyError extends Error {
  /** True when the job was deliberately stopped rather than failing. */
  readonly cancelled: boolean
  readonly promptId: string | null
  /** The node that failed, so the implicated field can be surfaced. */
  readonly node: string | null
  readonly nodeType: string | null
  /** ComfyUI's per-node validation detail, when the queue rejected the graph. */
  readonly nodeErrors: Record<string, unknown> | null

  constructor(
    message: string,
    opts: {
      cancelled?: boolean
      promptId?: string | null
      node?: string | null
      nodeType?: string | null
      nodeErrors?: Record<string, unknown> | null
    } = {},
  ) {
    super(message)
    this.name = 'ComfyError'
    this.cancelled = opts.cancelled ?? false
    this.promptId = opts.promptId ?? null
    this.node = opts.node ?? null
    this.nodeType = opts.nodeType ?? null
    this.nodeErrors = opts.nodeErrors ?? null
  }
}

// ---------------------------------------------------------------------------
// Identity and plain HTTP helpers
// ---------------------------------------------------------------------------

/**
 * One client id for the lifetime of the page. ComfyUI addresses execution
 * messages to the submitting client, so every job in this tab must use it —
 * and exactly one socket may hold it at a time (see rule 1).
 */
const clientId: string =
  globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2)

/** Build a browser-loadable URL for a file ComfyUI holds. */
export function fileUrl(f: FileRef): string {
  const q = new URLSearchParams({
    filename: f.filename,
    subfolder: f.subfolder ?? '',
    type: f.type ?? 'output',
  })
  return `${HTTP}/view?${q}`
}

/** Path of a file relative to its root — what the local delete API wants. */
export function relPath(f: FileRef): string {
  return f.subfolder ? `${f.subfolder}/${f.filename}` : f.filename
}

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${HTTP}${path}`)
  if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`)
  return r.json() as Promise<T>
}

/** Node catalogue — used to discover which models and samplers are installed. */
export function objectInfo(): Promise<Record<string, any>> {
  return getJson('/object_info')
}

export async function systemStats(): Promise<any> {
  try {
    return await getJson('/system_stats')
  } catch {
    return null
  }
}

/** Pull the option list for one node input, e.g. which checkpoints exist. */
export function optionsFor(info: Record<string, any>, node: string, field: string): string[] {
  const input = info?.[node]?.input
  const spec = input?.required?.[field]?.[0] ?? input?.optional?.[field]?.[0]
  return Array.isArray(spec) ? (spec as string[]) : []
}

/** True when the file is still on disk. Missing files 404; HEAD is enough. */
export async function headFile(f: FileRef): Promise<boolean> {
  try {
    const r = await fetch(fileUrl(f), { method: 'HEAD' })
    return r.ok
  } catch {
    return false
  }
}

const SAFE_NAME = /[^a-zA-Z0-9._-]+/g

/**
 * Upload a picture into ComfyUI's input folder so a graph can reference it by
 * filename (LoadImage).
 *
 * The name is made unique. Uploading with `overwrite: true` under the raw
 * basename means a second `photo.png` silently replaces the first, and every
 * archive record pointing at it then shows the wrong picture.
 *
 * @param file  the picture. A Blob is accepted so the player can send a frame.
 * @param name  optional original name, used to keep a readable suffix.
 * @returns the name to write into LoadImage, subfolder-prefixed when ComfyUI
 *          filed it under one.
 */
export async function uploadImage(file: File | Blob, name?: string): Promise<string> {
  const original = name ?? (file instanceof File ? file.name : 'frame.png')
  const base = (original.split(/[/\\]/).pop() || 'image.png').replace(SAFE_NAME, '_').slice(-64)
  const unique = `sg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}_${base}`

  const fd = new FormData()
  fd.append('image', file, unique)
  fd.append('overwrite', 'true')
  const r = await fetch(`${HTTP}/upload/image`, { method: 'POST', body: fd })
  if (!r.ok) throw new ComfyError(`ComfyUI refused the file (HTTP ${r.status}).`)
  const body = await r.json()
  return body.subfolder ? `${body.subfolder}/${body.name}` : body.name
}

// ---------------------------------------------------------------------------
// Output classification
// ---------------------------------------------------------------------------

const VIDEO_EXT = /\.(webm|mp4|mkv|gif|webp|avi|mov)$/i

/**
 * Turn one node's `ui` output dict into typed files.
 *
 * SaveWEBM returns `ui.PreviewVideo`, which serialises as
 * `{"images":[{...webm}], "animated":[true]}` — verified against a live
 * /history entry. Keying off the container name alone tags every clip as an
 * image, so three independent signals are used and any one is sufficient:
 * the container key, the sibling `animated` flag, and the file extension.
 */
export function collectFiles(output: Record<string, any> | undefined | null): OutputFile[] {
  const animated = Array.isArray(output?.animated) ? (output.animated as unknown[]) : []
  const anyAnimated = animated.some(Boolean)
  const out: OutputFile[] = []
  for (const [key, val] of Object.entries(output ?? {})) {
    if (key === 'animated' || !Array.isArray(val)) continue
    ;(val as any[]).forEach((f, i) => {
      if (!f?.filename) return
      const isVideo =
        key !== 'images' ||
        anyAnimated ||
        animated[i] === true ||
        VIDEO_EXT.test(String(f.filename))
      out.push({
        filename: String(f.filename),
        subfolder: String(f.subfolder ?? ''),
        type: String(f.type ?? 'output'),
        kind: isVideo ? 'video' : 'image',
      })
    })
  }
  return out
}

/** Every file across every node of a history entry's `outputs`. */
function filesOfOutputs(outputs: Record<string, any> | undefined | null): OutputFile[] {
  const out: OutputFile[] = []
  for (const node of Object.values(outputs ?? {})) out.push(...collectFiles(node as any))
  return out
}

// ---------------------------------------------------------------------------
// The shared socket
// ---------------------------------------------------------------------------

type Listener = (e: ProgressEvent) => void

type PromptState = {
  listeners: Set<Listener>
  files: OutputFile[]
  previewUrl: string | null
  /** The done/error event, replayed to anyone who watches late. */
  terminal: ProgressEvent | null
  settling: boolean
  /**
   * True when some of this prompt's messages may never have reached us: the
   * socket dropped while it was in flight, it was submitted while the socket
   * was down, or the first watcher came late. `executed` messages are not
   * replayed, so `files` may then hold only some of the outputs, and the
   * /history record has to be read for the full list.
   */
  gap: boolean
  touched: number
  reaper: ReturnType<typeof setTimeout> | null
}

const prompts = new Map<string, PromptState>()
const connectionListeners = new Set<(s: ConnectionState) => void>()

let sock: WebSocket | null = null
let backoff = 500
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let connection: ConnectionState = 'closed'
/** The prompt ComfyUI says it is executing, used to attribute bare previews. */
let executingPrompt: string | null = null
/** Queue depth from the server's `status` broadcast, for the shell. */
let queueRemaining = 0

/** Prompt states with no listener and no terminal event beyond this are junk. */
const MAX_IDLE_PROMPTS = 64
/** How long a finished prompt's result and preview URL stay available. */
const TERMINAL_GRACE_MS = 60_000

function setConnection(next: ConnectionState) {
  if (connection === next) return
  connection = next
  for (const l of connectionListeners) {
    try {
      l(next)
    } catch {
      /* a broken listener must not stop the others */
    }
  }
}

function stateFor(promptId: string): PromptState {
  let st = prompts.get(promptId)
  if (!st) {
    st = {
      listeners: new Set(),
      files: [],
      previewUrl: null,
      terminal: null,
      settling: false,
      gap: false,
      touched: Date.now(),
      reaper: null,
    }
    prompts.set(promptId, st)
    pruneStates()
  }
  st.touched = Date.now()
  return st
}

/** Keep the map from growing if we ever see prompts nobody is watching. */
function pruneStates() {
  if (prompts.size <= MAX_IDLE_PROMPTS) return
  const idle = [...prompts.entries()]
    .filter(([, s]) => s.listeners.size === 0)
    .sort((a, b) => a[1].touched - b[1].touched)
  for (const [id, s] of idle) {
    if (prompts.size <= MAX_IDLE_PROMPTS) break
    dropState(id, s)
  }
}

function dropState(id: string, st: PromptState) {
  if (st.reaper) clearTimeout(st.reaper)
  revokePreview(st)
  prompts.delete(id)
}

function revokePreview(st: PromptState) {
  if (!st.previewUrl) return
  try {
    URL.revokeObjectURL(st.previewUrl)
  } catch {
    /* already gone */
  }
  st.previewUrl = null
}

function emit(st: PromptState, e: ProgressEvent) {
  for (const l of [...st.listeners]) {
    try {
      l(e)
    } catch {
      /* one bad consumer must not break the others or the socket */
    }
  }
}

type Failure = { ok: false; message: string; cancelled: boolean; node: string | null }
type Outcome = { ok: true } | Failure

const STOPPED: Failure = {
  ok: false,
  message: 'Job stopped. Nothing was saved.',
  cancelled: true,
  node: null,
}

const FAILED: Failure = {
  ok: false,
  message: 'ComfyUI reported an error. The job did not finish.',
  cancelled: false,
  node: null,
}

/**
 * Finish a prompt once. Replayed to late watchers, then cleaned up.
 *
 * When a job reports success but some of its `executed` messages may have been
 * missed (see `gap`), the files come from /history rather than from whatever
 * the socket delivered. A reel shot writes two files, the clip and the frame
 * the next shot starts from, and settling with only one of them loses the
 * other for good. The same record says whether the run really succeeded, which
 * matters when the missed message was the error.
 */
async function settle(promptId: string, outcome: Outcome) {
  const st = prompts.get(promptId)
  if (!st || st.terminal || st.settling) return
  st.settling = true

  if (outcome.ok && (st.files.length === 0 || st.gap)) {
    const run = await fetchPastRun(promptId).catch(() => null)
    if (run?.status === 'cancelled') outcome = STOPPED
    else if (run?.status === 'error') outcome = FAILED
    else if (run) st.files = run.files
  }

  const event: ProgressEvent = outcome.ok
    ? { phase: 'done', files: st.files }
    : {
        phase: 'error',
        message: outcome.message,
        cancelled: outcome.cancelled,
        node: outcome.node,
      }

  st.terminal = event
  st.settling = false
  if (executingPrompt === promptId) executingPrompt = null
  emit(st, event)

  st.reaper = setTimeout(() => {
    const cur = prompts.get(promptId)
    if (cur) dropState(promptId, cur)
  }, TERMINAL_GRACE_MS)
}

function ensureSocket(): void {
  if (sock && (sock.readyState === WebSocket.OPEN || sock.readyState === WebSocket.CONNECTING)) return
  if (reconnectTimer) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }

  setConnection('connecting')
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${proto}//${location.host}${WS}?clientId=${clientId}`)
  ws.binaryType = 'arraybuffer'
  sock = ws

  ws.onopen = () => {
    backoff = 500
    setConnection('open')
    // Declaring this as the FIRST message makes ComfyUI send previews as
    // PREVIEW_IMAGE_WITH_METADATA, which carries prompt_id and node_id. Without
    // it, previews arrive bare and can only be attributed to whatever is
    // executing — wrong the moment two jobs overlap. (server.py:302)
    try {
      ws.send(JSON.stringify({ type: 'feature_flags', data: { supports_preview_metadata: true } }))
    } catch {
      /* the socket closed between open and send; the reconnect path handles it */
    }
    void reconcileWatched()
  }

  ws.onclose = () => {
    if (sock === ws) sock = null
    // Anything still in flight may now miss messages that are never replayed.
    for (const st of prompts.values()) if (!st.terminal) st.gap = true
    setConnection('closed')
    reconnectTimer = setTimeout(ensureSocket, backoff)
    backoff = Math.min(backoff * 2, 8000)
  }

  ws.onerror = () => {
    /* onclose always follows; the reconnect is handled there */
  }

  ws.onmessage = (ev: MessageEvent) => {
    if (typeof ev.data === 'string') handleText(ev.data)
    else if (ev.data instanceof ArrayBuffer) handleBinary(ev.data)
  }
}

function handleText(raw: string) {
  let msg: any
  try {
    msg = JSON.parse(raw)
  } catch {
    return
  }
  const d = msg?.data ?? {}
  const id: string | undefined = d.prompt_id

  switch (msg.type) {
    case 'status':
      queueRemaining = d?.status?.exec_info?.queue_remaining ?? queueRemaining
      return

    case 'execution_start':
      if (!id) return
      executingPrompt = id
      emit(stateFor(id), { phase: 'running', node: null, value: 0, max: 1 })
      return

    case 'executing': {
      if (!id) return
      if (d.node === null || d.node === undefined) {
        // main.py sends this when a prompt finishes, success or not.
        void settle(id, { ok: true })
      } else {
        executingPrompt = id
        emit(stateFor(id), { phase: 'running', node: String(d.node), value: 0, max: 1 })
      }
      return
    }

    case 'progress': {
      if (!id) return
      executingPrompt = id
      emit(stateFor(id), {
        phase: 'running',
        node: d.node == null ? null : String(d.node),
        value: Number(d.value ?? 0),
        max: Number(d.max ?? 1) || 1,
      })
      return
    }

    case 'executed': {
      if (!id) return
      const st = stateFor(id)
      st.files.push(...collectFiles(d.output))
      return
    }

    case 'execution_success': {
      if (!id) return
      // ComfyUI sends this before it writes the /history record, and the
      // `executing` message with a null node after. When the files have to
      // come from that record, settle on the later message instead, or the
      // lookup finds nothing yet and the job resolves without its files.
      const st = prompts.get(id)
      if (st && (st.gap || st.files.length === 0)) return
      void settle(id, { ok: true })
      return
    }

    case 'execution_interrupted':
      if (id) void settle(id, { ...STOPPED, node: d.node_id == null ? null : String(d.node_id) })
      return

    case 'execution_error':
      if (id)
        void settle(id, {
          ok: false,
          message: String(d.exception_message ?? d.exception_type ?? 'ComfyUI reported an error.'),
          cancelled: false,
          node: d.node_id == null ? null : String(d.node_id),
        })
      return

    default:
      // progress_state, execution_cached, feature_flags, b_preview and any
      // future message type. Nothing here needs them.
      return
  }
}

const utf8 = new TextDecoder()

/**
 * Binary frames, both preview shapes.
 *
 *  event 1  PREVIEW_IMAGE                [0..4) event, [4..8) 1=JPEG 2=PNG, rest = bytes
 *  event 4  PREVIEW_IMAGE_WITH_METADATA  [0..4) event, [4..8) json length,
 *                                        then UTF-8 JSON, then bytes
 *
 * Verified against server.py:1314-1373.
 */
function handleBinary(buf: ArrayBuffer) {
  if (buf.byteLength < 8) return
  const view = new DataView(buf)
  const event = view.getUint32(0)

  let mime = 'image/jpeg'
  let bytes: ArrayBuffer
  let promptId: string | null = executingPrompt

  if (event === 1) {
    mime = view.getUint32(4) === 2 ? 'image/png' : 'image/jpeg'
    bytes = buf.slice(8)
  } else if (event === 4) {
    const len = view.getUint32(4)
    if (8 + len > buf.byteLength) return
    let meta: any = null
    try {
      meta = JSON.parse(utf8.decode(new Uint8Array(buf, 8, len)))
    } catch {
      meta = null
    }
    if (typeof meta?.image_type === 'string') mime = meta.image_type
    if (typeof meta?.prompt_id === 'string') promptId = meta.prompt_id
    bytes = buf.slice(8 + len)
  } else {
    return
  }

  if (!promptId) return
  const st = prompts.get(promptId)
  // No listener, no decode. A four-minute Wan run at 1 preview a second is
  // 240 blobs nobody asked for.
  if (!st || st.listeners.size === 0 || st.terminal) return

  revokePreview(st)
  st.previewUrl = URL.createObjectURL(new Blob([bytes], { type: mime }))
  emit(st, { phase: 'preview', url: st.previewUrl })
}

/**
 * Settle one prompt from its /history record. True when the record was final
 * and the prompt has been settled from it; false when there is no record yet,
 * or none at all.
 */
async function reconcileOne(id: string): Promise<boolean> {
  const run = await fetchPastRun(id).catch(() => null)
  if (!run) return false
  if (run.status === 'success') {
    // A success record lists every output, so it replaces whatever the
    // socket managed to deliver before it dropped.
    const st = prompts.get(id)
    if (st) {
      st.files = run.files
      st.gap = false
    }
    void settle(id, { ok: true })
    return true
  }
  if (run.status === 'cancelled') {
    void settle(id, STOPPED)
    return true
  }
  if (run.status === 'error') {
    void settle(id, FAILED)
    return true
  }
  return false
}

/**
 * After a reconnect we may have missed the end of a job. Ask the server about
 * everything still being watched and settle whatever has finished. A prompt
 * with no record at all is left to the lost-job watch in `run()`.
 */
async function reconcileWatched(): Promise<void> {
  const open = [...prompts.entries()].filter(([, s]) => !s.terminal && s.listeners.size > 0)
  for (const [id] of open) await reconcileOne(id)
}

// ---------------------------------------------------------------------------
// Public transport API
// ---------------------------------------------------------------------------

/** Open the shared socket now, so the first job does not wait for a handshake. */
export function connect(): void {
  ensureSocket()
}

/** Current socket state, and a subscription for the offline notice. */
export function connectionState(): ConnectionState {
  return connection
}

export function watchConnection(on: (s: ConnectionState) => void): () => void {
  connectionListeners.add(on)
  ensureSocket()
  on(connection)
  return () => {
    connectionListeners.delete(on)
  }
}

/**
 * Queue a workflow. Resolves with the prompt id as soon as ComfyUI accepts it.
 *
 * Throws a {@link ComfyError} carrying ComfyUI's own validation detail when the
 * queue rejects the graph, so the implicated field can be surfaced.
 */
export async function submit(workflow: ApiWorkflow): Promise<string> {
  ensureSocket()
  let res: Response
  try {
    res = await fetch(`${HTTP}/prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: workflow, client_id: clientId }),
    })
  } catch {
    throw new ComfyError('We could not reach ComfyUI. It may not be running.')
  }

  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const nodeErrors = (body?.node_errors ?? null) as Record<string, unknown> | null
    const detail =
      body?.error?.message ??
      (typeof body?.error === 'string' ? body.error : null) ??
      Object.values(nodeErrors ?? {})
        .flatMap((n: any) => (n?.errors ?? []).map((e: any) => e?.message))
        .filter(Boolean)
        .join('; ')
    throw new ComfyError(detail || `The queue rejected the job (HTTP ${res.status}).`, {
      nodeErrors,
    })
  }

  const body = await res.json()
  const promptId = String(body.prompt_id)
  const st = stateFor(promptId)
  // ComfyUI addresses this prompt's messages to our client id from now on; a
  // socket that is not open yet drops them.
  if (connection !== 'open') st.gap = true
  return promptId
}

/**
 * Subscribe to one prompt's progress. Returns an unsubscribe function.
 *
 * Safe to call at any point in a job's life: events that arrived before the
 * first watcher are buffered (files) or replayed (the terminal event), so a
 * late subscriber still gets `done` or `error`.
 */
export function watch(promptId: string, on: Listener): () => void {
  ensureSocket()
  // A prompt this tab did not submit may already be part way through, and
  // the outputs it reported before now are not sent again.
  const late = !prompts.has(promptId)
  const st = stateFor(promptId)
  if (late) st.gap = true
  st.listeners.add(on)

  if (st.terminal) {
    const terminal = st.terminal
    queueMicrotask(() => {
      if (st.listeners.has(on)) on(terminal)
    })
  }

  return () => {
    st.listeners.delete(on)
    if (st.terminal && st.listeners.size === 0) revokePreview(st)
  }
}

/**
 * A job ComfyUI has forgotten, as opposed to one it refused or one that failed.
 *
 * Deliberately not a {@link ComfyError}: that is the queue or the graph
 * speaking, and this is the server having no record of the job at all, which
 * wants different words. `lost` is what `faults.ts` reads to tell the two apart.
 */
export class LostJob extends Error {
  readonly lost = true
  readonly cancelled = false
  readonly promptId: string

  constructor(message: string, promptId: string) {
    super(message)
    this.name = 'LostJob'
    this.promptId = promptId
  }
}

/** How often the lost-job watch asks the server about the prompts it follows. */
const LOST_POLL_MS = 5000
/** Consecutive absences from /api/jobs before the direct lookup. */
const LOST_MISSES = 2
/** A prompt accepted seconds ago may not be listed yet. */
const LOST_GRACE_MS = 20_000
/** Ticks to wait for a history record the server says already exists. */
const LOST_TERMINAL_WAITS = 3

type Followed = {
  since: number
  sighted: boolean
  misses: number
  terminalWaits: number
  lose: (err: LostJob) => void
}

/** Every prompt a `run()` is waiting on, checked together on one timer. */
const followed = new Map<string, Followed>()
let lostTimer: ReturnType<typeof setInterval> | null = null
let lostChecking = false

/**
 * Watch the server's own queue for a prompt that has vanished.
 *
 * A prompt settles on a socket event carrying its id, and some endings never
 * send one. A ComfyUI that restarts mid job has forgotten it, and its /history
 * starts empty, so the reconnect pass finds nothing to settle either. Without
 * this the caller's await never returns and the desk stays busy until reload.
 *
 * The evidence is two consecutive absences from the server's list of pending
 * and running jobs, confirmed by a direct lookup that finds no record. When
 * the lookup finds a finished job instead, its /history record settles it.
 */
function followForLoss(promptId: string, lose: (err: LostJob) => void): () => void {
  followed.set(promptId, { since: Date.now(), sighted: false, misses: 0, terminalWaits: 0, lose })
  if (lostTimer === null) lostTimer = setInterval(() => void checkForLoss(), LOST_POLL_MS)
  return () => {
    followed.delete(promptId)
    if (followed.size === 0 && lostTimer !== null) {
      clearInterval(lostTimer)
      lostTimer = null
    }
  }
}

async function checkForLoss(): Promise<void> {
  if (lostChecking || followed.size === 0) return
  lostChecking = true
  try {
    let live: Set<string>
    try {
      const page = await listJobs({ status: ['pending', 'in_progress'], limit: 100 })
      live = new Set(page.jobs.map((j) => j.id))
    } catch {
      return // an unreachable server is the offline notice's business
    }

    for (const [id, f] of [...followed]) {
      // Each await below can outlive the run it is checking.
      const current = () => followed.get(id) === f && !prompts.get(id)?.terminal
      if (!current()) continue
      if (live.has(id)) {
        f.sighted = true
        f.misses = 0
        continue
      }
      if (!f.sighted && Date.now() - f.since < LOST_GRACE_MS) continue
      f.misses += 1
      if (f.misses < LOST_MISSES) continue

      let server: ServerJob | null
      try {
        server = await getJob(id)
      } catch {
        continue // could not ask; try again on the next tick
      }
      if (!current()) continue

      if (server && (server.status === 'pending' || server.status === 'in_progress')) {
        f.sighted = true
        f.misses = 0
        continue
      }
      if (server) {
        if (await reconcileOne(id)) continue
        if (!current()) continue
        f.misses = 0
        f.terminalWaits += 1
        if (f.terminalWaits < LOST_TERMINAL_WAITS) continue
        f.lose(
          new LostJob(
            'ComfyUI says this job has ended but never sent the result. Look in the archive: the file may be on disk anyway.',
            id,
          ),
        )
        continue
      }
      f.lose(
        new LostJob(
          'We lost track of this job. ComfyUI has no record of it any more, which usually means it restarted. Nothing was saved.',
          id,
        ),
      )
    }
  } finally {
    lostChecking = false
  }
}

/**
 * Submit and follow a workflow to the end.
 *
 * Resolves with the produced files. Rejects with a {@link ComfyError}; check
 * `err.cancelled` to tell a deliberate stop from a failure. Rejects with a
 * {@link LostJob} when ComfyUI no longer knows the job, so no caller has to
 * race its own watch against this promise.
 *
 * Concurrent calls are safe — that is the whole point of the shared socket.
 * There is no AbortSignal: cancellation is `cancelJob(promptId)`, which is the
 * only mechanism that cannot kill somebody else's job.
 */
export function run(workflow: ApiWorkflow, on: (e: ProgressEvent) => void): Promise<OutputFile[]> {
  return new Promise<OutputFile[]>((resolve, reject) => {
    let stop: (() => void) | null = null
    let unfollow: (() => void) | null = null
    let promptId: string | null = null
    let settled = false

    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      stop?.()
      unfollow?.()
      fn()
    }

    const handler: Listener = (e) => {
      try {
        on(e)
      } catch {
        /* a throwing consumer must not strand the promise */
      }
      if (e.phase === 'done') finish(() => resolve(e.files))
      else if (e.phase === 'error')
        finish(() =>
          reject(
            new ComfyError(e.message, { cancelled: e.cancelled, promptId, node: e.node }),
          ),
        )
    }

    submit(workflow).then(
      (id) => {
        promptId = id
        // Register before announcing, so a synchronous consumer cannot miss
        // anything that is already in flight.
        stop = watch(id, handler)
        if (!settled) {
          try {
            on({ phase: 'queued', promptId: id })
          } catch {
            /* as above */
          }
        }
        if (!settled) unfollow = followForLoss(id, (err) => finish(() => reject(err)))
      },
      (err) => finish(() => reject(err)),
    )
  })
}

/**
 * Stop one job, running or queued, by its prompt id.
 *
 * `POST /api/jobs/{id}/cancel` is atomic and prompt-targeted (server.py:974):
 * a running job is interrupted, a pending one is dequeued, and an id that has
 * already finished returns `{cancelled: false}` with a 200.
 *
 * @returns true when the server actually stopped something. False means the
 *          job had already finished — treat that as success, not an error.
 *
 * A job that was still waiting is settled here, as stopped, because nothing
 * else ever will: dequeuing sends no message naming the prompt, only the new
 * queue length, and writes no /history record. A running job is left to its
 * `execution_interrupted`, because the interrupt only lands at ComfyUI's next
 * check for it, and a job on its last node can still finish and save first.
 *
 * Fallback: on a ComfyUI old enough not to route /api/jobs at all (the request
 * 404s, which the real endpoint never does for an unknown id) we fall back to
 * `POST /interrupt` — but ONLY when the prompt is the one currently executing,
 * because /interrupt kills whatever is sampling and would otherwise destroy an
 * unrelated job.
 */
export async function cancelJob(promptId: string): Promise<boolean> {
  const r = await fetch(`${HTTP}/api/jobs/${encodeURIComponent(promptId)}/cancel`, {
    method: 'POST',
  })
  if (r.ok) {
    const body = await r.json().catch(() => null)
    const cancelled = Boolean(body?.cancelled)
    if (cancelled) void settleIfDequeued(promptId)
    return cancelled
  }
  if (r.status === 404 || r.status === 405) {
    if (executingPrompt !== promptId) return false
    const legacy = await fetch(`${HTTP}/interrupt`, { method: 'POST' })
    return legacy.ok
  }
  throw new ComfyError(`ComfyUI would not stop that job (HTTP ${r.status}).`, { promptId })
}

/**
 * After a successful cancel, find out which kind it was. The server's answer
 * is the same `{cancelled: true}` for both, so ask about the job itself: a
 * dequeued job is gone without a trace, a running one is still listed until
 * the interrupt lands, and one that has already ended has a record to settle
 * from.
 */
async function settleIfDequeued(promptId: string): Promise<void> {
  const st = prompts.get(promptId)
  if (!st || st.terminal) return
  let server: ServerJob | null
  try {
    server = await getJob(promptId)
  } catch {
    return // could not ask; the lost-job watch in run() still has it
  }
  if (server === null) void settle(promptId, STOPPED)
  else if (server.status !== 'pending' && server.status !== 'in_progress') void reconcileOne(promptId)
}

/** The server's own view of the queue. */
export async function listJobs(
  opts: { status?: ServerJobStatus[]; limit?: number; offset?: number } = {},
): Promise<ServerJobsPage> {
  const q = new URLSearchParams()
  if (opts.status?.length) q.set('status', opts.status.join(','))
  if (opts.limit !== undefined) q.set('limit', String(opts.limit))
  if (opts.offset !== undefined) q.set('offset', String(opts.offset))
  const suffix = q.toString() ? `?${q}` : ''
  const page = await getJson<ServerJobsPage>(`/api/jobs${suffix}`)
  return {
    jobs: Array.isArray(page?.jobs) ? page.jobs : [],
    pagination: page?.pagination ?? { offset: 0, limit: null, total: 0, has_more: false },
  }
}

/**
 * One job by id, or null when the server has never heard of it.
 *
 * Only a 404 means that. Any other failure throws, because a proxy that
 * cannot reach ComfyUI answers 502, and reading that as "no such job" would
 * report a job lost while the server was merely down.
 */
export async function getJob(promptId: string): Promise<ServerJob | null> {
  const r = await fetch(`${HTTP}/api/jobs/${encodeURIComponent(promptId)}`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`/api/jobs/${promptId} -> HTTP ${r.status}`)
  return (await r.json()) as ServerJob
}

// ---------------------------------------------------------------------------
// Reading past generations back out of /history
// ---------------------------------------------------------------------------

/** Raw history page, newest last. Keys are prompt ids. */
function historyPage(max = 200): Promise<Record<string, any>> {
  return getJson<Record<string, any>>(`/history?max_items=${Math.max(1, Math.floor(max))}`)
}

function timestampOf(status: any, event: string): number | null {
  const messages: any[] = Array.isArray(status?.messages) ? status.messages : []
  for (const m of messages) {
    if (Array.isArray(m) && m[0] === event && typeof m[1]?.timestamp === 'number') return m[1].timestamp
  }
  return null
}

/**
 * Normalise one raw /history entry.
 *
 * `prompt` is the tuple `[number, promptId, graph, extraData, outputNodeIds]`;
 * `prompt[2]` is the complete API graph that was submitted, so every parameter
 * of a past generation is recoverable from it. Verified live.
 */
function readPastRun(promptId: string, raw: any): PastRun | null {
  const tuple = raw?.prompt
  if (!Array.isArray(tuple)) return null
  const graph = tuple[2]
  if (!graph || typeof graph !== 'object') return null

  const statusStr = raw?.status?.status_str
  const interrupted = (raw?.status?.messages ?? []).some(
    (m: any) => Array.isArray(m) && m[0] === 'execution_interrupted',
  )
  const status: PastRun['status'] = interrupted
    ? 'cancelled'
    : statusStr === 'success'
      ? 'success'
      : statusStr === 'error'
        ? 'error'
        : 'unknown'

  return {
    promptId,
    graph: graph as ApiWorkflow,
    files: filesOfOutputs(raw?.outputs),
    status,
    startedAt: timestampOf(raw?.status, 'execution_start') ?? tuple[3]?.create_time ?? null,
    finishedAt:
      timestampOf(raw?.status, 'execution_success') ??
      timestampOf(raw?.status, 'execution_error') ??
      timestampOf(raw?.status, 'execution_interrupted'),
    clientId: typeof tuple[3]?.client_id === 'string' ? tuple[3].client_id : null,
  }
}

/** One past run by prompt id, or null when it is not in history. */
async function fetchPastRun(promptId: string): Promise<PastRun | null> {
  const page = await getJson<Record<string, any>>(`/history/${encodeURIComponent(promptId)}`)
  const raw = page?.[promptId]
  return raw ? readPastRun(promptId, raw) : null
}

/** The last `max` runs ComfyUI remembers, newest first. */
export async function pastRuns(max = 200): Promise<PastRun[]> {
  const page = await historyPage(max)
  const runs: PastRun[] = []
  for (const [id, raw] of Object.entries(page ?? {})) {
    const run = readPastRun(id, raw)
    if (run) runs.push(run)
  }
  // /history is oldest-first; the archive reads newest-first everywhere.
  return runs.reverse()
}

/**
 * Read parameters back out of a submitted graph, through a family's bindings.
 *
 * Give it `def.bindings` and it returns `{ positive, seed, steps, ... }` with
 * whatever the graph actually held — the recovery path behind "import my past
 * runs" and behind rebuilding a record whose metadata was lost.
 *
 * Link values (`[nodeId, slot]` tuples) are skipped: they are wiring, not a
 * value a person chose.
 */
export function readBoundParams(
  graph: ApiWorkflow,
  bindings: BindingMap,
): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [key, binds] of Object.entries(bindings)) {
    if (!binds?.length) continue
    for (const [nodeId, input] of binds) {
      const value = graph[nodeId]?.inputs?.[input]
      if (value === undefined || Array.isArray(value)) continue
      out[key] = value
      break
    }
  }
  return out
}

