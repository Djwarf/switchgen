/**
 * The lab's only door into the app: the routes it reads and the two it
 * writes to, and nothing else in the lab knows them. Every request carries a
 * JSON content type and no Origin header; guard.mjs lets a request with no
 * Origin through, as it does the app's own server-side callers.
 *
 * The lab never talks to ComfyUI through here, and never sends the lane word
 * (POST /api/runner/lane): a held lane is the user's to answer, on the app.
 */

/** The routes, for a stand-in to serve. */
export const ROUTES = {
  capabilities: '/api/capabilities',
  snapshot: '/api/runner',
  groups: '/api/runner/groups',
  stop: (groupId: string) => `/api/runner/groups/${encodeURIComponent(groupId)}/stop`,
  tag: '/api/vision/tag',
} as const

/** The reader takes at most this many pictures a call (vision.mjs MAX_IMAGES). */
export const TAG_MAX = 24

export type Capabilities = {
  runner: boolean
  runnerDesks: string[]
  runnerReason: string | null
  [k: string]: unknown
}

export type OutputFile = { filename: string; subfolder: string; type: string; kind?: 'image' | 'video'; cached?: boolean }

export type RunnerError = {
  code: string
  message: string | null
  node: string | null
  nodeType: string | null
  [k: string]: unknown
}

export type RunnerStatus =
  | 'waiting'
  | 'releasing'
  | 'sending'
  | 'queued'
  | 'running'
  | 'filing'
  | 'done'
  | 'failed'
  | 'stopped'
  | 'lost'
  | 'unsent'
  | 'skipped'

/** An ended job: nothing will change it again. */
export const TERMINAL: ReadonlySet<string> = new Set(['done', 'failed', 'stopped', 'lost', 'unsent', 'skipped'])

export type JobView = {
  id: string
  groupId: string
  desk: string
  status: RunnerStatus
  error: RunnerError | null
  files: OutputFile[]
  primary: OutputFile | null
  promptId: string | null
  durationMs: number
  ranAt: number | null
  finishedAt: number | null
  endedAt: number | null
  seq: number
  meta: Record<string, unknown> | null
  wait?: { for: string; ahead?: number } | null
  [k: string]: unknown
}

export type GroupView = {
  id: string
  desk: string
  kind: string
  state: 'active' | 'ended'
  jobIds: string[]
  [k: string]: unknown
}

export type Snapshot = {
  v: 1
  available: boolean
  reason: string | null
  boot: string
  rev: number
  comfy?: { answering: boolean | null; since: number }
  lane: { held: null | { why: string; scope: 'heavy' | 'all'; jobId: string | null; since: number } }
  groups: GroupView[]
  jobs: JobView[]
  progress?: Record<string, unknown>
}

export type LabJobBody = {
  id: string
  label: string
  prompt: string
  kind: 'image'
  primary: 'image'
  orFirst: true
  noFile: 'fail'
  heavy: false
  graph: Record<string, { class_type: string; inputs: Record<string, unknown> }>
  record: { desk: 'lab'; mode: string }
  chain?: { after: string; at: [string, string][] }
  meta: { lab: { run: string; cell: string } }
}

export type GroupBody = {
  v: 1
  group: { id: string; desk: 'lab'; kind: 'set'; label: string; device: string }
  jobs: LabJobBody[]
}

export type SubmitResult =
  | { ok: true; status: 200; rev: number; replayed: boolean; group: GroupView; jobs: JobView[] }
  | { ok: false; status: number; error: string; busy?: string; reason?: string }

export type Tag = { tag: string; confidence: number }

export type TagRow = {
  index: number
  kind?: string
  rel: string
  width?: number
  height?: number
  rating?: 'general' | 'sensitive' | 'questionable' | 'explicit' | null
  ratings?: Tag[]
  general?: Tag[]
  character?: Tag[]
  /** The reader could not read this one: why. */
  error?: string
}

export type RunnerClient = {
  readonly appUrl: string
  capabilities(): Promise<Capabilities>
  snapshot(): Promise<Snapshot>
  /** POST the group. A network failure throws AppUnreachable: the answer is unknown, and the ledger decides. */
  submit(body: GroupBody): Promise<SubmitResult>
  /** True when the runner stopped it, false when it does not know the group. */
  stopGroup(id: string): Promise<boolean>
  tag(rels: readonly string[]): Promise<TagRow[] | { busy: 'memory'; message: string | null }>
}

/** The app did not answer at all (not running, or the connection dropped). */
export class AppUnreachable extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options)
    this.name = 'AppUnreachable'
  }
}

/** The app answered with something the lab cannot use. */
export class AppError extends Error {
  status: number
  body: unknown
  constructor(status: number, message: string, body: unknown) {
    super(message)
    this.name = 'AppError'
    this.status = status
    this.body = body
  }
}

const SHORT_MS = 20_000
const SUBMIT_MS = 180_000
const TAG_MS = 15 * 60_000

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>

export function runnerClient(appUrl: string, fetchImpl: FetchLike = fetch): RunnerClient {
  const base = appUrl.replace(/\/+$/, '')

  async function call(method: 'GET' | 'POST', route: string, body: unknown, timeoutMs: number): Promise<{ status: number; json: unknown }> {
    const init: RequestInit = {
      method,
      headers: method === 'POST' ? { 'Content-Type': 'application/json', Accept: 'application/json' } : { Accept: 'application/json' },
      signal: AbortSignal.timeout(timeoutMs),
    }
    if (method === 'POST') init.body = JSON.stringify(body ?? {})
    let res: Response
    try {
      res = await fetchImpl(base + route, init)
    } catch (err) {
      throw new AppUnreachable(`The app at ${base} did not answer (${(err as Error)?.message ?? err}).`, { cause: err })
    }
    let text: string
    try {
      text = await res.text()
    } catch (err) {
      throw new AppUnreachable(`The app at ${base} stopped answering part way (${(err as Error)?.message ?? err}).`, { cause: err })
    }
    let json: unknown = null
    if (text) {
      try {
        json = JSON.parse(text)
      } catch {
        json = { error: text.slice(0, 300) }
      }
    }
    return { status: res.status, json }
  }

  const errorOf = (json: unknown, status: number) => {
    const e = (json as { error?: unknown } | null)?.error
    return typeof e === 'string' && e ? e : `the app answered ${status}`
  }

  return {
    appUrl: base,

    async capabilities() {
      const { status, json } = await call('GET', ROUTES.capabilities, null, SHORT_MS)
      if (status !== 200 || !json || typeof json !== 'object') throw new AppError(status, errorOf(json, status), json)
      const c = json as Partial<Capabilities>
      return {
        ...(json as Record<string, unknown>),
        runner: c.runner === true,
        runnerDesks: Array.isArray(c.runnerDesks) ? c.runnerDesks.filter((d): d is string => typeof d === 'string') : [],
        runnerReason: typeof c.runnerReason === 'string' ? c.runnerReason : null,
      }
    },

    async snapshot() {
      const { status, json } = await call('GET', ROUTES.snapshot, null, SHORT_MS)
      const s = json as Partial<Snapshot> | null
      if (status !== 200 || !s || s.v !== 1 || !Array.isArray(s.jobs) || !Array.isArray(s.groups)) {
        throw new AppError(status, status === 200 ? 'the runner answered with something that is not its list of work' : errorOf(json, status), json)
      }
      return {
        ...(s as Snapshot),
        lane: s.lane && typeof s.lane === 'object' ? s.lane : { held: null },
      }
    },

    async submit(body) {
      const { status, json } = await call('POST', ROUTES.groups, body, SUBMIT_MS)
      const j = (json ?? {}) as Record<string, unknown>
      if (status === 200) {
        return {
          ok: true,
          status: 200,
          rev: Number(j.rev) || 0,
          replayed: j.replayed === true,
          group: j.group as GroupView,
          jobs: Array.isArray(j.jobs) ? (j.jobs as JobView[]) : [],
        }
      }
      return {
        ok: false,
        status,
        error: errorOf(json, status),
        ...(typeof j.busy === 'string' ? { busy: j.busy } : {}),
        ...(typeof j.reason === 'string' ? { reason: j.reason } : {}),
      }
    },

    async stopGroup(id) {
      const { status, json } = await call('POST', ROUTES.stop(id), {}, SHORT_MS)
      if (status === 200) return true
      if (status === 404) return false
      throw new AppError(status, errorOf(json, status), json)
    },

    async tag(rels) {
      if (rels.length > TAG_MAX) throw new Error(`the picture reader takes at most ${TAG_MAX} pictures a call`)
      const images = rels.map((rel) => ({ kind: 'output', rel }))
      const { status, json } = await call('POST', ROUTES.tag, { images }, TAG_MS)
      const j = (json ?? {}) as { tag?: { rows?: unknown }; rejected?: unknown; busy?: unknown; error?: unknown }
      if (status === 503 && j.busy === 'memory') {
        return { busy: 'memory' as const, message: typeof j.error === 'string' ? j.error : null }
      }
      const rejected = Array.isArray(j.rejected) ? (j.rejected as { index?: unknown; error?: unknown }[]) : []
      const refused = (): TagRow[] =>
        rejected
          .filter((r) => typeof r?.index === 'number' && rels[r.index as number] !== undefined)
          .map((r) => ({ index: r.index as number, rel: rels[r.index as number], error: typeof r.error === 'string' ? r.error : 'not readable' }))
      // Every picture refused (none found, say): an answer about each, not a failure of the call.
      if (status === 400 && rejected.length) return refused()
      if (status !== 200) throw new AppError(status, errorOf(json, status), json)
      const rows = Array.isArray(j.tag?.rows) ? (j.tag.rows as Record<string, unknown>[]) : []
      const out: TagRow[] = []
      for (const r of rows) {
        const index = typeof r.index === 'number' ? r.index : -1
        const rel = typeof r.rel === 'string' ? r.rel : rels[index]
        if (rel === undefined) continue
        out.push({ ...(r as Partial<TagRow>), index, rel } as TagRow)
      }
      return [...out, ...refused()]
    },
  }
}
