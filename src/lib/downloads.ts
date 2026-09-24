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
 *
 * A fetch is the server's, not the page's. The server goes on with it when
 * the stream closes, which on a phone is the usual case: a reload, a tab the
 * browser discarded, a locked screen, a change of network. So a lost stream
 * is not a failed fetch. The page asks GET /api/download/status instead, and
 * a reloaded page takes up the plans it finds running there.
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
 * A family fetch as GET /api/download/status lists it: the one running, or
 * else the one started last, kept for ten minutes after it ends. The server
 * may add the plan's whole file list and, on the file it is on, the job id
 * and speed; the page takes them when they are there, and otherwise reads
 * the id and speed off the matching entry in `downloads`.
 */
export type ServerPlan = {
  family: string | null
  state: 'running' | 'done' | 'error' | 'cancelled'
  current: {
    filename: string
    index: number
    count: number
    done: number
    total: number
    pct: number
    etaSec: number | null
    jobId?: string
    speed?: number
  } | null
  finished: string[]
  error: string | null
  files?: PlanFile[]
}

/** GET /api/download/status: the files being fetched now, and the plans. */
export type DownloadStatus = { downloads: DownloadJob[]; plans: ServerPlan[] }

/**
 * The server answered the request without a stream, so nothing was started
 * for it and there is nothing to follow. Anything else that ends a stream
 * early is the connection, and the fetch may well be going on without it.
 */
class Refused extends Error {}

/** How often a fetch the page follows through the server is asked about. */
const POLL_MS = 2000

/** Settles after `ms`, or rejects at once when `signal` aborts. */
function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason)
      return
    }
    const onAbort = () => {
      clearTimeout(timer)
      reject(signal?.reason)
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

/**
 * What the server is fetching now and the plans it ran lately, or null when
 * it did not answer (restarting, or the phone off the network). A server
 * that answers without `plans` lists none.
 */
export async function readDownloadStatus(): Promise<DownloadStatus | null> {
  try {
    const res = await fetch('/api/download/status', { headers: { Accept: 'application/json' } })
    if (!res.ok || !(res.headers.get('content-type') ?? '').includes('json')) return null
    const data = (await res.json()) as Partial<DownloadStatus> | null
    return {
      downloads: Array.isArray(data?.downloads) ? data.downloads : [],
      plans: Array.isArray(data?.plans) ? data.plans : [],
    }
  } catch {
    return null
  }
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
    throw new Refused(data.error ?? `HTTP ${res.status}`)
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
 * only closes the socket, and a plan that has begun goes on without its page
 * (see the top of this file). So a stop names the job to POST
 * /api/download/cancel first. The id arrives with
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

/** Said when the stream is lost before the server named the file it started. */
const LOST_BEFORE_START =
  'Lost touch with the server before the fetch began, and nothing is fetching for it there now.'

/** The files the models listing names at all (server/api.mjs WEIGHTS): weights, not the tagger's model or tag list. */
const LISTED = /\.(safetensors|gguf|ckpt|pt|pth|sft|bin)$/i

/**
 * Whether the models listing has a whole file at `dest` (relative to the
 * models root), or null when it did not answer. The listing leaves out a file
 * aria2c has not finished, so a name there is a file that landed.
 */
async function landedAt(dest: string): Promise<boolean | null> {
  try {
    const res = await fetch('/api/models', { headers: { Accept: 'application/json' } })
    if (!res.ok) return null
    const { files } = (await res.json()) as { files?: { rel?: string }[] }
    return (files ?? []).some((f) => (f.rel ?? '').replace(/\\/g, '/') === dest)
  } catch {
    return null
  }
}

/**
 * Follow one file through the server's list after its stream was lost:
 * progress while the server lists the job, and then how it ended. The status
 * lists only family plans, so a file fetched by address is looked for in the
 * models listing once its job has gone. Resolves once it landed; rejects with
 * a plain sentence when it cannot be found there.
 */
async function followFile(
  job: { id: string; filename: string; dest: string },
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  let total = 0
  const landedNow = () => {
    onProgress({ state: 'done', pct: 1, done: total, total, speed: 0, etaSec: null, error: null })
  }
  for (;;) {
    await wait(POLL_MS, signal)
    const status = await readDownloadStatus()
    // No answer is not an ending: the phone may be between networks.
    if (!status) continue
    const listed = status.downloads.find((j) => j.id === job.id)
    if (listed && (listed.state === 'starting' || listed.state === 'downloading')) {
      const p = progressOf(listed, listed.state)
      total = p.total || total
      onProgress(p)
      continue
    }
    // The job leaves the list once its file is done or has failed; one still
    // listed as done, failed or cancelled is on its way out.
    if (listed) continue
    if (status.plans.some((p) => p.finished.includes(job.filename))) return landedNow()
    if (!LISTED.test(job.dest)) {
      throw new Error(
        `Lost touch with the server during the fetch, and it no longer lists ${job.filename}, so whether it ` +
          'landed is not known here. Fetching it again finishes it, or finds it already there.',
      )
    }
    const landed = await landedAt(job.dest)
    if (landed === null) continue
    if (landed) return landedNow()
    throw new Error(
      `Lost touch with the server during the fetch, and ${job.filename} is not in the models folder now. ` +
        'Fetching it again takes up any part already on disk.',
    )
  }
}

/**
 * Fetch one file. Aborting `signal` rejects at once, so the caller can say it
 * stopped, while the stop itself goes on behind: the server is told to cancel
 * the job and keep the partial, so fetching the file again resumes it.
 *
 * When the stream is lost part way, the fetch goes on at the server, so this
 * follows it there rather than reporting the connection as a failed fetch.
 */
export async function downloadFile(
  spec: DownloadSpec,
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  signal?.throwIfAborted()
  let failure: string | null = null
  let started: { id: string; filename: string; dest: string } | null = null
  /** The stream said how the fetch ended, with 'done' or 'error'. */
  let told = false
  /** The stream is gone and the server's list is followed instead. */
  let following = false
  const stream = openStream({ url: spec.url, filename: spec.filename, dest: spec.dest }, (e) => {
    if (e.event === 'start') {
      started = { id: e.job.id, filename: e.job.filename, dest: e.job.dest }
      onProgress(progressOf(e.job, 'starting'))
    } else if (e.event === 'progress') onProgress(progressOf(e.job, 'downloading'))
    else if (e.event === 'file' || e.event === 'skip') onProgress(progressOf(e.job, 'done'))
    else if (e.event === 'error') {
      failure = e.job.error ?? 'the download failed'
      onProgress({ state: 'error', pct: 0, done: 0, total: 0, speed: 0, etaSec: null, error: failure })
    }
    if (e.event === 'done' || e.event === 'error') told = true
  })
  const lost = async (): Promise<void> => {
    following = true
    // Lost before the server named the job: it may still have begun, so the
    // server's list is asked once before anything is said about it.
    if (!started) {
      await wait(POLL_MS, signal)
      const status = await readDownloadStatus()
      const job = status?.downloads.find((j) => j.family === null && j.filename === spec.filename)
      if (!job) throw new Error(status ? LOST_BEFORE_START : 'Lost touch with the server before the fetch began.')
      started = { id: job.id, filename: job.filename, dest: job.dest }
    }
    return followFile(started, onProgress, signal)
  }
  const outcome = stream.done.then(
    () => (told ? undefined : lost()),
    (err: unknown) => {
      if (err instanceof Refused || signal?.aborted) throw err
      return lost()
    },
  )
  await new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      // A followed fetch has no stream left to stop, so it is cancelled by
      // the id its stream named, keeping the partial all the same.
      if (following && started) void postCancel(started.id, true)
      else void stream.stop(true)
      reject(signal?.reason ?? new DOMException('Stopped', 'AbortError'))
    }
    signal?.addEventListener('abort', onAbort, { once: true })
    void outcome.then(resolve, reject).finally(() => signal?.removeEventListener('abort', onAbort))
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
  /** Stop this run at the server: the file it is on, and every file after it. */
  stop: (keepPartial: boolean) => Promise<void>
  /** Tells this run apart from a later one for the same family. */
  serial: number
  /**
   * The page follows this run by asking the server every couple of seconds
   * rather than through a stream of its own: it was taken up after a reload,
   * its stream was lost, or the page came back from the background.
   */
  followed: boolean
  /** A followed run whose last question went unanswered; what it shows is the server's last answer. */
  outOfTouch: boolean
}

let runs: ReadonlyMap<string, PlanRun> = new Map()
let serials = 0
const runListeners = new Set<() => void>()
const landedListeners = new Set<(family: string) => void>()
/** Runs whose landing has been announced, so a stream and the server's list cannot both announce it. */
const announced = new Set<number>()

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

/** The runs as they stand. */
export function downloadRuns(): ReadonlyMap<string, PlanRun> {
  return runs
}

const live = (run: PlanRun | undefined): run is PlanRun => run?.state === 'starting' || run?.state === 'running'

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

function announceLanded(family: string, serial: number): void {
  if (announced.has(serial)) return
  announced.add(serial)
  for (const fn of [...landedListeners]) {
    try { fn(family) } catch { /* one broken listener must not stop the rest */ }
  }
}

/**
 * Fetch everything a family is missing. One run per family at a time; a
 * second request for a running family is ignored. When every file has
 * landed the run reads 'done' and onPlanLanded's listeners hear of it.
 */
export function startPlan(req: { family: string; model?: string | null; include?: string[]; force?: boolean }): void {
  if (live(runs.get(req.family))) return

  const body: Record<string, unknown> = { family: req.family }
  if (req.model) body.model = req.model
  if (req.include?.length) body.include = req.include
  if (req.force) body.force = true

  // A family stopped and fetched again gets a new run while the old stream
  // may still be winding down, so each stream patches only its own run, and
  // only while the page reads the run through it: once the run is followed
  // through the server, the server's list is what it shows.
  const serial = ++serials
  const mine = () => runs.get(req.family)?.serial === serial
  const own = () => mine() && !runs.get(req.family)?.followed
  const stream = openStream(body, (e) => {
    if (e.event === 'done') {
      // The files are on disk whatever became of the run on screen.
      if (mine()) patchRun(req.family, { state: 'done', current: null, outOfTouch: false })
      announceLanded(req.family, serial)
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
  next.set(req.family, {
    family: req.family, state: 'starting', files: [], current: null, finished: [], error: null,
    stop: stream.stop, serial, followed: false, outOfTouch: false,
  })
  runs = next
  emitRuns()

  // A stop marks the run cancelled before it hangs up, and a refusal is the
  // server's own answer, so a run still going when its stream ends any other
  // way lost its connection, not its fetch: follow it through the server.
  const lost = () => {
    if (own()) follow(req.family)
  }
  stream.done.then(lost, (err: unknown) => {
    if (!(err instanceof Refused)) {
      lost()
      return
    }
    if (own() && live(runs.get(req.family))) {
      patchRun(req.family, { state: 'error', error: err.message, current: null })
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
  if (!live(run)) return
  patchRun(family, { state: 'cancelled', current: null, outOfTouch: false })
  await run.stop(keepPartial)
}

/** Forget a finished, failed or cancelled run, so the panel shows the family plain again. */
export function forgetPlan(family: string): void {
  const run = runs.get(family)
  if (!run || live(run)) return
  const next = new Map(runs)
  next.delete(family)
  runs = next
  emitRuns()
}

// ---------------------------------------------------------------------------
// Following a run through the server's list
// ---------------------------------------------------------------------------

/** Said when a followed run's plan is no longer listed. */
export const PLAN_GONE =
  'Lost track of this fetch: the server no longer lists it, either because it restarted or because the fetch ' +
  'ended more than ten minutes ago. Dismiss this to read the disk again.'

/**
 * The server's entry for a family: the running plan when there is one, since
 * only one plan runs per family, and otherwise the last one listed, which is
 * the one that ended most recently.
 */
export function planFor(plans: readonly ServerPlan[], family: string): ServerPlan | null {
  let last: ServerPlan | null = null
  for (const p of plans) {
    if (p.family !== family) continue
    if (p.state === 'running') return p
    last = p
  }
  return last
}

const num = (n: unknown): number => (typeof n === 'number' && Number.isFinite(n) ? n : 0)

/** The file a running plan is on, with the id and speed of its job when the server lists it. */
function currentOf(plan: ServerPlan, jobs: readonly DownloadJob[]): PlanRun['current'] {
  const c = plan.current
  const job = jobs.find((j) => j.family === plan.family && (!c || j.filename === c.filename))
  if (!c && !job) return null
  const state = job?.state === 'starting' ? 'starting' : 'downloading'
  if (!c) return { ...progressOf(job!, state), filename: job!.filename, jobId: job!.id, index: job!.fileIndex, count: job!.fileCount }
  return {
    state,
    pct: num(c.pct),
    done: num(c.done),
    total: num(c.total),
    speed: num(c.speed ?? job?.speed),
    etaSec: typeof c.etaSec === 'number' ? c.etaSec : null,
    error: null,
    filename: c.filename,
    jobId: c.jobId ?? job?.id ?? '',
    index: num(c.index),
    count: num(c.count),
  }
}

/**
 * What a followed run shows, read off the server's list. A run whose stream
 * was lost before the server said it began ('starting') is the server's only
 * if a plan for its family is running: a lost request stops a plan during its
 * checks, and an ended plan listed for the family is then an earlier one.
 */
export function followedFields(
  run: Pick<PlanRun, 'family' | 'state' | 'files'>,
  status: DownloadStatus,
): Pick<PlanRun, 'state' | 'current' | 'finished' | 'error' | 'files' | 'outOfTouch'> {
  const plan = planFor(status.plans, run.family)
  if (!plan || (run.state === 'starting' && plan.state !== 'running')) {
    return {
      state: 'error',
      current: null,
      finished: [],
      error: run.state === 'starting' ? LOST_BEFORE_START : PLAN_GONE,
      files: run.files,
      outOfTouch: false,
    }
  }
  const current = plan.state === 'running' ? currentOf(plan, status.downloads) : null
  const finished = Array.isArray(plan.finished) ? plan.finished : []
  // A run taken up after a reload never saw the plan's list of files. The
  // server's list is used when it sends one; otherwise the run names the
  // files it knows of, those landed and the one being fetched.
  const files = run.files.length
    ? run.files
    : Array.isArray(plan.files) && plan.files.length
      ? plan.files
      : [...finished, ...(current ? [current.filename] : [])].map((filename) => ({
          filename, dest: '', sizeBytes: null, gated: false,
        }))
  return {
    state: plan.state,
    current,
    finished,
    error: plan.state === 'error' ? plan.error ?? 'the download failed' : null,
    files,
    outOfTouch: false,
  }
}

/**
 * Stop a followed run: find the file its plan is on in the server's list and
 * cancel that job by id, as a stream's own stop does. Between two files no
 * job is listed for a moment, so it asks again a few times.
 */
async function cancelOnServer(family: string, keepPartial: boolean): Promise<void> {
  for (let tries = 0; tries < 5; tries++) {
    const status = await readDownloadStatus()
    if (status) {
      const onIt = planFor(status.plans, family)?.current?.jobId
      const job = status.downloads.find((j) => (onIt ? j.id === onIt : j.family === family))
      if (job?.state === 'starting' || job?.state === 'downloading') {
        await postCancel(job.id, keepPartial)
        return
      }
      // Already stopping, or nothing is running for the family any more.
      if (job?.state === 'cancelled') return
      if (!job && planFor(status.plans, family)?.state !== 'running') return
    }
    await wait(1000)
  }
}

/**
 * Follow a live run through the server's list from now on. The first look
 * is a poll away rather than at once, which gives a plan whose request was
 * lost during its checks the moment it needs to show up in the list.
 */
function follow(family: string): void {
  const run = runs.get(family)
  if (!live(run) || run.followed) return
  patchRun(family, { followed: true, stop: (keepPartial) => cancelOnServer(family, keepPartial) })
  schedulePoll()
}

let pollTimer: ReturnType<typeof setTimeout> | null = null

function schedulePoll(): void {
  if (pollTimer !== null) return
  if (![...runs.values()].some((r) => r.followed && live(r))) return
  pollTimer = setTimeout(() => {
    pollTimer = null
    void resumeRuns()
  }, POLL_MS)
}

function applyStatus(status: DownloadStatus | null): void {
  for (const run of [...runs.values()]) {
    if (!run.followed || !live(run)) continue
    if (!status) {
      if (!run.outOfTouch) patchRun(run.family, { outOfTouch: true })
      continue
    }
    const fields = followedFields(run, status)
    patchRun(run.family, fields)
    if (fields.state === 'done') announceLanded(run.family, run.serial)
  }
  // A plan running at the server that no run here knows of: started before a
  // reload, or from another tab. Taken up so it shows, with its Stop. So is
  // one whose run here has failed, which covers a run this page lost track
  // of: the server running a plan for the family is the later news.
  for (const plan of status?.plans ?? []) {
    const family = plan.family
    if (typeof family !== 'string' || plan.state !== 'running') continue
    const here = runs.get(family)
    if (here && here.state !== 'error') continue
    const run: PlanRun = {
      family, state: 'running', files: [], current: null, finished: [], error: null,
      stop: (keepPartial) => cancelOnServer(family, keepPartial),
      serial: ++serials, followed: true, outOfTouch: false,
    }
    const next = new Map(runs)
    next.set(family, { ...run, ...followedFields(run, status!) })
    runs = next
    emitRuns()
  }
  schedulePoll()
}

let resuming: Promise<void> | null = null

/**
 * Ask the server where its fetches stand: bring every followed run up to
 * date, and take up the plans running there that no run here knows of. Runs
 * once when the page loads, again whenever it comes back into view, and
 * every couple of seconds while a followed run is going.
 */
export function resumeRuns(): Promise<void> {
  resuming ??= (async () => {
    try {
      applyStatus(await readDownloadStatus())
    } finally {
      resuming = null
    }
  })()
  return resuming
}

// A phone that slept may hold a stream open that will never speak again: the
// connection went with the network it was on, and nothing tells the page. So
// when the page comes back into view, every run that has begun is followed
// through the server from then on, and anything started meanwhile is taken
// up. Not in a test, which has no document.
if (typeof document !== 'undefined') {
  void resumeRuns()
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return
    for (const run of [...runs.values()]) if (run.state === 'running') follow(run.family)
    void resumeRuns()
  })
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
