/**
 * The model catalogue, as server/downloads.mjs serves it.
 *
 * The catalogue knows 104 families; this app carries a verified graph for
 * fifteen of them. The panel that reads this joins the two by id and offers
 * only what can actually run here once fetched. Every verdict about fit
 * (RAM, disk, a gated file) is the server's, quoting the numbers it used.
 */

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
  installed: boolean
  installedBytes: number
  partial: boolean
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

/** The server's plan for one family: what to fetch, how big, and whether it fits. */
export function fetchPlan(family: string, model?: string | null): Promise<CatalogPlan> {
  const q = new URLSearchParams({ family })
  if (model) q.set('model', model)
  return json<CatalogPlan>(`/api/catalog/plan?${q}`)
}

export async function cancelDownload(id: string, keepPartial = false): Promise<void> {
  await json('/api/download/cancel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, keepPartial }),
  })
}

/** `1.4 GB`, `217 MB`. */
export function bytesText(n: number | null | undefined): string {
  if (!n) return '0 MB'
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`
  return `${Math.max(1, Math.round(n / 1024 ** 2))} MB`
}
