import { useSyncExternalStore } from 'react'

/**
 * Fetching a file into the models tree through the local server.
 *
 * POST /api/download answers with an event stream rather than JSON, so this
 * reads the body as a stream instead of using EventSource, which cannot POST.
 * The server keeps the partial file when the request is aborted, so a
 * cancelled fetch resumes rather than starting again. The LoRA picker used to
 * own this reader; the vision tagger and the catalogue need it too.
 */

export type FetchProgress = {
  state: 'starting' | 'downloading' | 'done' | 'error' | 'cancelled'
  /** 0 to 1. Zero until the server has the content length. */
  pct: number
  done: number
  total: number
  /** Bytes per second, as aria2c reports it. */
  speed: number
  etaSec: number | null
  error: string | null
}

export type DownloadSpec = {
  url: string
  filename: string
  /** Destination relative to the models root, folder or full path. */
  dest: string
}

/** One frame of the download stream, as the server names them. */
export type DownloadEvent =
  | { event: 'plan'; family: string | null; files: PlanFile[] }
  | { event: 'start' | 'progress' | 'file' | 'skip' | 'error'; job: DownloadJob }
  | { event: 'done'; family: string | null; files: string[]; bytes: number }

export type PlanFile = { filename: string; dest: string; sizeBytes: number | null; gated: boolean }

export type DownloadJob = {
  id: string
  family: string | null
  filename: string
  dest: string
  state: 'starting' | 'downloading' | 'done' | 'error' | 'cancelled'
  done: number
  total: number
  pct: number
  speed: number
  etaSec: number | null
  fileIndex: number
  fileCount: number
  error: string | null
}

/**
 * POST a download request and read its event stream. A refused request
 * answers JSON, and that error is thrown with the server's own sentence.
 */
export async function streamDownload(
  body: Record<string, unknown>,
  onEvent: (e: DownloadEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  // A rejected plan answers JSON, not a stream: bad URL, a clashing download,
  // a destination outside the models root, a disk too full to take it. So does
  // a plan with nothing to do, which is a success with no stream behind it.
  const type = res.headers.get('content-type') ?? ''
  if (!res.ok || !res.body || !type.includes('event-stream')) {
    let data: { error?: string; nothingToDo?: boolean; family?: string | null } = {}
    try {
      data = (await res.json()) as typeof data
    } catch {
      /* not JSON either */
    }
    if (res.ok && data.nothingToDo) {
      onEvent({ event: 'done', family: data.family ?? null, files: [], bytes: 0 })
      return
    }
    throw new Error(data.error ?? `HTTP ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() ?? ''
    for (const block of blocks) {
      let event = 'message'
      let data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      if (!data) continue
      let payload: Record<string, unknown>
      try {
        payload = JSON.parse(data) as Record<string, unknown>
      } catch {
        continue
      }
      if (event === 'plan') {
        onEvent({ event, family: (payload.family as string | null) ?? null, files: (payload.files as PlanFile[]) ?? [] })
      } else if (event === 'done') {
        onEvent({
          event,
          family: (payload.family as string | null) ?? null,
          files: (payload.files as string[]) ?? [],
          bytes: typeof payload.bytes === 'number' ? payload.bytes : 0,
        })
      } else if (event === 'start' || event === 'progress' || event === 'file' || event === 'skip' || event === 'error') {
        onEvent({ event, job: payload as unknown as DownloadJob })
      }
    }
  }
}

function progressOf(job: DownloadJob, state: FetchProgress['state']): FetchProgress {
  return {
    state,
    pct: typeof job.pct === 'number' ? job.pct : 0,
    done: typeof job.done === 'number' ? job.done : 0,
    total: typeof job.total === 'number' ? job.total : 0,
    speed: typeof job.speed === 'number' ? job.speed : 0,
    etaSec: typeof job.etaSec === 'number' ? job.etaSec : null,
    error: null,
  }
}

export async function downloadFile(
  spec: DownloadSpec,
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  let failure: string | null = null
  await streamDownload({ url: spec.url, filename: spec.filename, dest: spec.dest }, (e) => {
    if (e.event === 'start') onProgress(progressOf(e.job, 'starting'))
    else if (e.event === 'progress') onProgress(progressOf(e.job, 'downloading'))
    else if (e.event === 'file' || e.event === 'skip') onProgress(progressOf(e.job, 'done'))
    else if (e.event === 'error') {
      failure = e.job.error ?? 'the download failed'
      onProgress({ state: 'error', pct: 0, done: 0, total: 0, speed: 0, etaSec: null, error: failure })
    }
  }, signal)
  if (failure) throw new Error(failure)
}

// ---------------------------------------------------------------------------
// Whole families, in a store that outlives the panel that started them
// ---------------------------------------------------------------------------

export type PlanRun = {
  family: string
  state: 'starting' | 'running' | 'done' | 'error' | 'cancelled'
  files: PlanFile[]
  /** The file being fetched right now, with its own progress. */
  current: (FetchProgress & { filename: string; jobId: string; index: number; count: number }) | null
  finished: string[]
  error: string | null
  abort: AbortController
}

let runs: ReadonlyMap<string, PlanRun> = new Map()
const runListeners = new Set<() => void>()

function emitRuns(): void {
  for (const fn of [...runListeners]) {
    try { fn() } catch { /* one broken subscriber must not stop the rest */ }
  }
}

function patchRun(family: string, patch: Partial<PlanRun>): void {
  const prev = runs.get(family)
  if (!prev) return
  const next = new Map(runs)
  next.set(family, { ...prev, ...patch })
  runs = next
  emitRuns()
}

export function subscribeDownloads(fn: () => void): () => void {
  runListeners.add(fn)
  return () => { runListeners.delete(fn) }
}

export function downloadRuns(): ReadonlyMap<string, PlanRun> {
  return runs
}

/**
 * Fetch everything a family is missing. One run per family at a time; a
 * second request for a running family is ignored. `onDone` fires once every
 * file has landed, which is when a desk should re-read its catalogue.
 */
export function startPlan(
  req: { family: string; model?: string | null; include?: string[]; force?: boolean },
  onDone?: () => void,
): void {
  const existing = runs.get(req.family)
  if (existing && (existing.state === 'starting' || existing.state === 'running')) return
  const abort = new AbortController()
  const next = new Map(runs)
  next.set(req.family, { family: req.family, state: 'starting', files: [], current: null, finished: [], error: null, abort })
  runs = next
  emitRuns()

  const body: Record<string, unknown> = { family: req.family }
  if (req.model) body.model = req.model
  if (req.include?.length) body.include = req.include
  if (req.force) body.force = true

  void streamDownload(body, (e) => {
    if (e.event === 'plan') patchRun(req.family, { state: 'running', files: e.files })
    else if (e.event === 'start' || e.event === 'progress') {
      patchRun(req.family, {
        state: 'running',
        current: {
          ...progressOf(e.job, e.event === 'start' ? 'starting' : 'downloading'),
          filename: e.job.filename,
          jobId: e.job.id,
          index: e.job.fileIndex,
          count: e.job.fileCount,
        },
      })
    } else if (e.event === 'file' || e.event === 'skip') {
      const run = runs.get(req.family)
      patchRun(req.family, { current: null, finished: [...(run?.finished ?? []), e.job.filename] })
    } else if (e.event === 'error') {
      patchRun(req.family, {
        state: e.job.state === 'cancelled' ? 'cancelled' : 'error',
        error: e.job.error ?? 'the download failed',
        current: null,
      })
    } else if (e.event === 'done') {
      patchRun(req.family, { state: 'done', current: null })
      onDone?.()
    }
  }, abort.signal).catch((err: unknown) => {
    const run = runs.get(req.family)
    if (run && run.state !== 'done' && run.state !== 'cancelled') {
      patchRun(req.family, {
        state: abort.signal.aborted ? 'cancelled' : 'error',
        error: abort.signal.aborted ? null : err instanceof Error ? err.message : String(err),
        current: null,
      })
    }
  })
}

/** Stop a family's run. The partial file is removed on the server unless asked to keep it. */
export async function cancelPlan(family: string, keepPartial = false): Promise<void> {
  const run = runs.get(family)
  if (!run) return
  patchRun(family, { state: 'cancelled', current: null })
  run.abort.abort()
  if (run.current?.jobId) {
    try {
      await fetch('/api/download/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: run.current.jobId, keepPartial }),
      })
    } catch {
      /* the server will notice the closed stream and stop on its own */
    }
  }
}

/** Forget a finished, failed or cancelled run, so the panel shows the family plain again. */
export function forgetPlan(family: string): void {
  const run = runs.get(family)
  if (!run || run.state === 'running' || run.state === 'starting') return
  const next = new Map(runs)
  next.delete(family)
  runs = next
  emitRuns()
}

/**
 * Fetch the tagger's files, one after the other, reporting the whole as one
 * progress figure. `install` is what /api/vision/capabilities hands back when
 * the tagger is missing: verified URLs and byte counts.
 */
export async function installFiles(
  files: readonly { filename: string; dest: string; url: string; sizeBytes: number }[],
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  const total = files.reduce((n, f) => n + f.sizeBytes, 0)
  let before = 0
  for (const f of files) {
    await downloadFile({ url: f.url, filename: f.filename, dest: f.dest }, (p) => {
      const done = before + p.done
      onProgress({ ...p, done, total, pct: total ? done / total : p.pct })
    }, signal)
    before += f.sizeBytes
  }
  onProgress({ state: 'done', pct: 1, done: total, total, speed: 0, etaSec: null, error: null })
}

/** The runs, live, for a component. */
export function useDownloads(): ReadonlyMap<string, PlanRun> {
  return useSyncExternalStore(subscribeDownloads, downloadRuns, downloadRuns)
}
