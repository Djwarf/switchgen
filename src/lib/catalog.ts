/**
 * The model catalogue, as server/downloads.mjs serves it.
 *
 * The catalogue knows 104 families; this app carries a verified graph for
 * fifteen of them. The panel that reads this joins the two by id and offers
 * only what can actually run here once fetched. Every verdict about fit
 * (RAM, disk, a gated file) is the server's, quoting the numbers it used.
 */
import { onPlanLanded } from './downloads'

export type CatalogFile = {
  filename: string
  kind: string
  dest: string
  url: string | null
  sizeBytes: number | null
  sizeApprox: boolean
  gated: boolean
  optional: boolean
  note: string
  /** True for a file on disk under this name, and for one counted as it (see `conflict`). */
  installed: boolean
  installedBytes: number
  /** A fetch that stopped part way at `dest`, which the next fetch resumes. */
  partial: boolean
  /**
   * A whole file at `dest` under this name that is shorter than the one the
   * catalogue lists, and not a stopped fetch: most likely another build of
   * it. The server counts it as installed and never fetches over it, since a
   * graph loads the file by its name and that is the file it would get.
   */
  conflict: boolean
}

export type CatalogInstalled = {
  chosenModel: string | null
  defaultModel: string | null
  residentBytes: number
  files: CatalogFile[]
  missing: string[]
  missingBytes: number
  ready: boolean
  gatedMissing: string[]
  /** Files on disk under the catalogue's name but not its size, used as they are (see CatalogFile.conflict). */
  asIs: string[]
  installedCount: number
  fileCount: number
}

export type CatalogFamily = {
  id: string
  label: string
  mode: string
  group: string
  models: string[]
  verified: boolean
  requiresBytes: number | null
  deps: string[]
  notes: string
  incomplete?: string[]
  installed: CatalogInstalled
}

export type Catalog = {
  root: string
  counts: { families: number; ready: number; deps: number; depsInstalled: number }
  families: CatalogFamily[]
}

export type CatalogPlan = {
  family: { id: string; label: string; mode: string; models: string[] }
  chosenModel: string | null
  download: { filename: string; dest: string; url: string | null; sizeBytes: number | null; gated: boolean; kind: string; resumingFrom: number }[]
  totalBytes: number
  alreadyInstalled: string[]
  /** The ones among alreadyInstalled that are not the catalogue's file, used as they are. */
  asIs: string[]
  incomplete: string[]
  gated: { files: string[]; tokenPresent: boolean }
  hardware: { ramTotal: number; ramFree: number; diskFree: number | null; diskTotal: number | null }
  fits: boolean
  verdict: string
  reasons: string[]
  blockers: string[]
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { Accept: 'application/json', ...(init?.headers ?? {}) } })
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('json')) throw new Error('the catalogue is not served here')
  const data = (await res.json()) as T & { error?: string }
  if (!res.ok) throw new Error(String(data.error ?? `HTTP ${res.status}`))
  return data
}

let cache: Promise<Catalog> | null = null

/** The whole catalogue, annotated with what is on disk. Cached until a download lands. */
export function fetchCatalog(fresh = false): Promise<Catalog> {
  if (fresh || !cache) {
    cache = json<Catalog>('/api/catalog?slim=1').catch((err: unknown) => {
      cache = null
      throw err
    })
  }
  return cache
}

export function forgetCatalog(): void {
  cache = null
}

// A family landing changes what the catalogue calls installed, whether or not
// a panel is open to re-read it, so the next reader asks the server again.
onPlanLanded(() => forgetCatalog())

/** The server's plan for one family: what to fetch, how big, and whether it fits. */
export function fetchPlan(family: string, model?: string | null): Promise<CatalogPlan> {
  const q = new URLSearchParams({ family })
  if (model) q.set('model', model)
  return json<CatalogPlan>(`/api/catalog/plan?${q}`)
}

/** `1.4 GB`, `217 MB`. */
export function bytesText(n: number | null | undefined): string {
  if (!n) return '0 MB'
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`
  return `${Math.max(1, Math.round(n / 1024 ** 2))} MB`
}

/**
 * A gated file the server has no token for. The fit check passes such a
 * plan, but the server refuses that file when the fetch reaches it, and a
 * fetch ends at its first refused file.
 */
export function noToken(plan: CatalogPlan): string | null {
  const gated = plan.gated.files
  if (!gated.length || plan.gated.tokenPresent) return null
  const one = gated.length === 1
  return (
    `${gated.join(', ')} ${one ? 'is' : 'are'} gated on HuggingFace and no token is on the server, so the ` +
    `fetch would stop at ${one ? 'it' : 'the first of them'}.`
  )
}

/**
 * Why a fetch was held back, for the question before fetching anyway. It is
 * not always that the family will not fit: too little RAM free right now,
 * too little disk, a URL that failed its last check, or a gated file with no
 * token each hold it back, and each asks something different of the reader.
 * So the server's own sentences are quoted rather than summed up.
 *
 * A file on disk under the catalogue's name but not its size is not fetched
 * over, and the verdict of a plan that will not run here leaves it out, so
 * the question names it: once fetched, the family counts as installed and
 * loads that file, which nothing here has checked.
 */
export function heldBack(plan: CatalogPlan): string {
  const parts = plan.fits
    ? []
    : ['The server holds this back.', ...(plan.blockers.length ? plan.blockers : [plan.verdict])]
  const token = noToken(plan)
  if (token) parts.push(token)
  const asIs = plan.asIs ?? []
  if (asIs.length) {
    const one = asIs.length === 1
    parts.push(
      `The fetch leaves ${asIs.join(', ')} as ${one ? 'it is' : 'they are'} on disk: not the size the catalogue ` +
        `lists, and nothing here has checked that ${one ? 'it loads' : 'they load'}.`,
    )
  }
  return parts.join(' ')
}
