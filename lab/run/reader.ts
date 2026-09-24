/**
 * The picture reader's pass over a run: the app's WD14 tagger (POST
 * /api/vision/tag), 24 pictures a call, only once the run's last lab job has
 * ended. Its rating, the four rating confidences and the general and
 * character tags go to <lab>/readings.jsonl by cell; a picture read once is
 * never read again.
 *
 * Its figures are a guide, hidden until the reveal, and always worded as
 * "the picture reader rated", never as safe or unsafe.
 *
 * The quarantine rule: a picture the reader rates questionable or explicit
 * that also carries a minor-coded tag is deleted from <outputs>/.lab at once,
 * never gets a blind copy, and its cell is marked removed (in the run's
 * ledger and in cells.jsonl, so no later run makes or reuses it).
 */
import fs from 'node:fs'
import path from 'node:path'
import type { LabEnv } from '../core/env.ts'
import { runDir } from '../core/env.ts'
import { appendDoneCell, appendJsonl, appendLedger, readDoneCells, readJsonl } from './ledger.ts'
import type { RunnerClient, Tag, TagRow } from './runnerClient.ts'
import { TAG_MAX } from './runnerClient.ts'

/** Tags that, with a questionable or explicit rating, remove a picture. */
export const QUARANTINE_TAGS: readonly string[] = [
  'child',
  'children',
  'loli',
  'shota',
  'toddler',
  'baby',
  'male_child',
  'female_child',
  'aged_down',
  'kindergarten',
  'school_uniform',
]

/** The ratings that, with one of those tags, remove a picture. */
export const QUARANTINE_RATINGS: readonly string[] = ['questionable', 'explicit']

export type Reading = {
  v: 1
  cellId: string
  rel: string
  at: number
  rating: string | null
  ratings: Tag[]
  general: Tag[]
  character: Tag[]
  quarantined: boolean
}

export const readingsPath = (env: LabEnv) => path.join(env.labDir, 'readings.jsonl')

/** The readings kept so far, by cell; the first reading of a cell stands. */
export function readReadings(env: LabEnv): Map<string, Reading> {
  const out = new Map<string, Reading>()
  for (const r of readJsonl<Reading>(readingsPath(env)).rows) {
    if (r && typeof r.cellId === 'string' && !out.has(r.cellId)) out.set(r.cellId, r)
  }
  return out
}

const norm = (t: string) => t.trim().toLowerCase().replace(/\s+/g, '_')
const QUARANTINE = new Set(QUARANTINE_TAGS)

/** The minor-coded tags a row carries, if its rating also puts it under the rule. */
export function quarantineTags(row: Pick<TagRow, 'rating' | 'general'>): string[] {
  if (!row.rating || !QUARANTINE_RATINGS.includes(row.rating)) return []
  return (row.general ?? []).map((g) => norm(g.tag)).filter((t) => QUARANTINE.has(t))
}

export const shouldQuarantine = (row: Pick<TagRow, 'rating' | 'general'>) => quarantineTags(row).length > 0

/**
 * Delete a cell's pictures under <outputs>/.lab/cells and mark it removed:
 * a 'removed' entry in the run's ledger and a removed row in cells.jsonl.
 * Only files named for this cell are touched.
 */
export function removeCell(env: LabEnv, run: string, cellId: string, now: () => number = Date.now): string[] {
  if (!/^[0-9a-f]{8,64}$/.test(cellId)) throw new Error(`"${cellId}" is not a cell id`)
  const dir = path.join(env.outputs, '.lab', 'cells')
  const gone: string[] = []
  let names: string[] = []
  try {
    names = fs.readdirSync(dir)
  } catch {
    names = []
  }
  for (const name of names) {
    if (!name.startsWith(`${cellId}_`)) continue
    const full = path.join(dir, name)
    try {
      if (fs.lstatSync(full).isFile()) {
        fs.unlinkSync(full)
        gone.push(`.lab/cells/${name}`)
      }
    } catch {
      /* already gone */
    }
  }
  const at = now()
  appendLedger(runDir(env, run), { t: 'removed', at, cell: cellId, why: 'quarantine' })
  const row = readDoneCells(env).get(cellId)
  appendDoneCell(env, {
    cellId,
    rel: row?.rel ?? '',
    durationMs: row?.durationMs ?? 0,
    cold: row?.cold ?? false,
    cached: row?.cached ?? false,
    finishedAt: row?.finishedAt ?? null,
    run,
    removed: true,
  })
  return gone
}

export type ReadOptions = {
  /** The run whose ledger a removal is written to. */
  run: string
  /** Each cell's picture under outputs; by default from cells.jsonl. */
  rels?: ReadonlyMap<string, string>
  sleep?: (ms: number) => Promise<void>
  now?: () => number
  log?: (line: string) => void
  /** How long to keep waiting while the reader says the machine is short of memory. Default 2 h. */
  maxWaitMs?: number
  /** Checked between calls: true stops the pass, leaving the rest pending. */
  shouldStop?: () => boolean
}

export type ReadResult = {
  /** Cells read now or before. */
  read: string[]
  /** Cells the quarantine rule removed now or before. */
  quarantined: string[]
  /** Cells not read: the reader was busy too long, could not read them, or their picture was missing. */
  pending: string[]
  /** Why some are pending, in words, when there is one reason for all of them. */
  why?: string
}

const FIRST_BACKOFF_MS = 30_000
const MAX_BACKOFF_MS = 5 * 60_000
const DEFAULT_MAX_WAIT_MS = 2 * 60 * 60_000

/** Read every cell not read yet, applying the quarantine rule as the answers come. */
export async function readCells(env: LabEnv, client: Pick<RunnerClient, 'tag'>, cellIds: readonly string[], opts: ReadOptions): Promise<ReadResult> {
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)))
  const now = opts.now ?? Date.now
  const log = opts.log ?? (() => {})
  const maxWait = opts.maxWaitMs ?? DEFAULT_MAX_WAIT_MS
  const done = readReadings(env)
  const rows = readDoneCells(env)
  const relOf = (id: string): string | null => {
    const r = opts.rels ? opts.rels.get(id) : rows.get(id)?.rel
    return typeof r === 'string' && r ? r : null
  }

  const read: string[] = []
  const quarantined: string[] = []
  const pending: string[] = []
  const todo: { cellId: string; rel: string }[] = []
  for (const cellId of new Set(cellIds)) {
    const had = done.get(cellId)
    if (had || rows.get(cellId)?.removed) {
      // Read before; or removed by the rule, even if the process stopped
      // between the deletion and writing its reading.
      read.push(cellId)
      if (had?.quarantined || rows.get(cellId)?.removed) quarantined.push(cellId)
      continue
    }
    const rel = relOf(cellId)
    if (!rel) {
      pending.push(cellId)
      continue
    }
    todo.push({ cellId, rel })
  }

  let waited = 0
  let backoff = FIRST_BACKOFF_MS
  let why: string | undefined
  for (let i = 0; i < todo.length; ) {
    if (opts.shouldStop?.()) {
      why = 'the reading was stopped'
      pending.push(...todo.slice(i).map((t) => t.cellId))
      break
    }
    const chunk = todo.slice(i, i + TAG_MAX)
    const answer = await client.tag(chunk.map((c) => c.rel))
    if (!Array.isArray(answer)) {
      if (waited >= maxWait) {
        why = `the machine was too short of memory for the picture reader for ${Math.round(waited / 60_000)} minutes, so the lab stopped waiting`
        pending.push(...todo.slice(i).map((t) => t.cellId))
        break
      }
      const pause = Math.min(backoff, maxWait - waited)
      log(`The picture reader is waiting for memory; trying again in ${Math.round(pause / 1000)} s.`)
      await sleep(pause)
      waited += pause
      backoff = Math.min(backoff * 2, MAX_BACKOFF_MS)
      continue
    }
    backoff = FIRST_BACKOFF_MS
    const byIndex = new Map<number, TagRow>()
    for (const row of answer) if (typeof row.index === 'number') byIndex.set(row.index, row)
    for (let k = 0; k < chunk.length; k++) {
      const { cellId, rel } = chunk[k]
      const row = byIndex.get(k)
      if (!row || row.error || row.rel !== rel) {
        pending.push(cellId)
        continue
      }
      const bad = shouldQuarantine(row)
      const reading: Reading = {
        v: 1,
        cellId,
        rel,
        at: now(),
        rating: row.rating ?? null,
        ratings: row.ratings ?? [],
        general: row.general ?? [],
        character: row.character ?? [],
        quarantined: bad,
      }
      // The deletion comes first: a reading that says quarantined is never
      // written while the picture is still there.
      if (bad) {
        removeCell(env, opts.run, cellId, now)
        quarantined.push(cellId)
      }
      appendJsonl(readingsPath(env), reading)
      read.push(cellId)
    }
    i += chunk.length
  }
  return { read, quarantined, pending, ...(why ? { why } : {}) }
}
