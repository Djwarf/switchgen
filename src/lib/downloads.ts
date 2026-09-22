import { useSyncExternalStore } from 'react'

/**
 * Fetching a file into the models tree through the local server.
 *
 * POST /api/download answers with an event stream rather than JSON, so this
 * reads the body as a stream instead of using EventSource, which cannot POST.
 * Stopping a fetch names the file to the server by its job id before hanging
 * up, and says whether to keep the partial, so a kept one resumes rather than
 * starting again. The LoRA picker used to own this reader; the vision tagger
 * and the catalogue need it too.
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
async function streamDownload(
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

async function postCancel(id: string, keepPartial: boolean): Promise<void> {
  try {
    await fetch('/api/download/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, keepPartial }),
    })
  } catch {
    /* hanging up afterwards still tells the server the reader has gone */
  }
}

type Stream = {
  /** Settles when the server ends the stream or refuses the request. */
  done: Promise<void>
  /** Cancel the file the server is on, by its id, then hang up. */
  stop: (keepPartial: boolean) => Promise<void>
}

/**
 * A download stream that can be stopped for real. Aborting the fetch alone
 * only closes the socket, and the server is not guaranteed to notice: aria2c
 * went on to the end of the file, and a plan went on to its next one. So a
 * stop names the job to POST /api/download/cancel first. The id arrives with
 * the 'start' frame, and a stop pressed before it (the server still checking
 * the plan) waits for that frame rather than hanging up on a job it cannot
 * name. After a stop, frames are not passed on, since the caller has already
 * said on screen that it stopped; a 'done' still is, because then every file
 * landed before the stop reached the server.
 */
function openStream(body: Record<string, unknown>, onEvent: (e: DownloadEvent) => void): Stream {
  const abort = new AbortController()
  let jobId: string | null = null
  let ended = false
  let nextId: ((id: string | null) => void) | null = null
  let stopping: Promise<void> | null = null

  const done = streamDownload(body, (e) => {
    if (e.event === 'start') {
      jobId = e.job.id
      nextId?.(jobId)
      nextId = null
    } else if (e.event === 'file' || e.event === 'skip' || e.event === 'error') {
      jobId = null
    }
    if (!stopping || e.event === 'done') onEvent(e)
  }, abort.signal).finally(() => {
    ended = true
    nextId?.(null)
    nextId = null
  })

  const stop = (keepPartial: boolean): Promise<void> => {
    stopping ??= (async () => {
      const id = jobId ?? (ended ? null : await new Promise<string | null>((resolve) => { nextId = resolve }))
      if (id) await postCancel(id, keepPartial)
      abort.abort()
    })()
    return stopping
  }

  return { done, stop }
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

/**
 * Fetch one file. Aborting `signal` rejects at once, so the caller can say it
 * stopped, while the stop itself goes on behind: the server is told to cancel
 * the job and keep the partial, so fetching the file again resumes it.
 */
export async function downloadFile(
  spec: DownloadSpec,
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  signal?.throwIfAborted()
  let failure: string | null = null
  const stream = openStream({ url: spec.url, filename: spec.filename, dest: spec.dest }, (e) => {
    if (e.event === 'start') onProgress(progressOf(e.job, 'starting'))
    else if (e.event === 'progress') onProgress(progressOf(e.job, 'downloading'))
    else if (e.event === 'file' || e.event === 'skip') onProgress(progressOf(e.job, 'done'))
    else if (e.event === 'error') {
      failure = e.job.error ?? 'the download failed'
      onProgress({ state: 'error', pct: 0, done: 0, total: 0, speed: 0, etaSec: null, error: failure })
    }
  })
  await new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      void stream.stop(true)
      reject(signal?.reason ?? new DOMException('Stopped', 'AbortError'))
    }
    signal?.addEventListener('abort', onAbort, { once: true })
    void stream.done.then(resolve, reject).finally(() => signal?.removeEventListener('abort', onAbort))
  })
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
  /** This run's own stream. Also what tells a run apart from a later one for the same family. */
  stop: (keepPartial: boolean) => Promise<void>
}

let runs: ReadonlyMap<string, PlanRun> = new Map()
const runListeners = new Set<() => void>()
const landedListeners = new Set<(family: string) => void>()

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

function subscribeDownloads(fn: () => void): () => void {
  runListeners.add(fn)
  return () => { runListeners.delete(fn) }
}

function downloadRuns(): ReadonlyMap<string, PlanRun> {
  return runs
}

/**
 * Hear about every family whose files all landed, with its catalogue id.
 * A fetch can take long enough for the reader to leave the room and come
 * back, so the news is published here rather than handed to whichever panel
 * started the run: that panel, and the desk it would have refreshed, may no
 * longer be on screen. Returns the unsubscribe.
 */
export function onPlanLanded(fn: (family: string) => void): () => void {
  landedListeners.add(fn)
  return () => { landedListeners.delete(fn) }
}

/**
 * Fetch everything a family is missing. One run per family at a time; a
 * second request for a running family is ignored. When every file has
 * landed the run reads 'done' and onPlanLanded's listeners hear of it.
 */
export function startPlan(req: { family: string; model?: string | null; include?: string[]; force?: boolean }): void {
  const existing = runs.get(req.family)
  if (existing && (existing.state === 'starting' || existing.state === 'running')) return

  const body: Record<string, unknown> = { family: req.family }
  if (req.model) body.model = req.model
  if (req.include?.length) body.include = req.include
  if (req.force) body.force = true

  // A family stopped and fetched again gets a new run while the old stream
  // may still be winding down, so each stream patches only its own run.
  const own = () => runs.get(req.family)?.stop === stream.stop
  const stream = openStream(body, (e) => {
    if (e.event === 'done') {
      // The files are on disk whatever became of the run on screen.
      if (own()) patchRun(req.family, { state: 'done', current: null })
      for (const fn of [...landedListeners]) {
        try { fn(req.family) } catch { /* one broken listener must not stop the rest */ }
      }
      return
    }
    if (!own()) return
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
    }
  })

  const next = new Map(runs)
  next.set(req.family, { family: req.family, state: 'starting', files: [], current: null, finished: [], error: null, stop: stream.stop })
  runs = next
  emitRuns()

  // A stop marks the run cancelled before it hangs up, so a run still going
  // when the stream fails has failed.
  stream.done.catch((err: unknown) => {
    const run = runs.get(req.family)
    if (own() && run && (run.state === 'starting' || run.state === 'running')) {
      patchRun(req.family, { state: 'error', error: err instanceof Error ? err.message : String(err), current: null })
    }
  })
}

/**
 * Stop a family's run. The server cancels the file it is on, by id, even when
 * the stop comes before the first file is named; the partial file is removed
 * unless asked to keep it. Resolves once the server has been told.
 */
export async function cancelPlan(family: string, keepPartial = false): Promise<void> {
  const run = runs.get(family)
  if (!run || (run.state !== 'starting' && run.state !== 'running')) return
  patchRun(family, { state: 'cancelled', current: null })
  await run.stop(keepPartial)
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
