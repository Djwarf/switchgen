/**
 * The run's ledger: runs/<run>/ledger.jsonl, one JSON entry a line, each
 * written and flushed to disk before the action it names, so a run that dies
 * at any point (a crash, a reboot, the lab process killed) is rebuilt from it
 * exactly: what was about to be sent, what the runner took, what ended how.
 *
 * Beside it, the lab's own index of finished pictures, cells.jsonl, shared by
 * every run: a picture made once (the same graph, the same seed) is never
 * made again, only reused.
 */
import fs from 'node:fs'
import path from 'node:path'
import type { LabEnv } from '../core/env.ts'
import type { DoneCell, LedgerEntry } from '../core/types.ts'

export const LEDGER_FILE = 'ledger.jsonl'

/**
 * Append one line and flush it to disk before returning. A line left torn by
 * a crash mid-write is closed off first, so the next entry starts clean.
 */
export function appendJsonl(file: string, row: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  const fd = fs.openSync(file, 'a+', 0o600)
  try {
    const size = fs.fstatSync(fd).size
    let lead = ''
    if (size > 0) {
      const last = Buffer.alloc(1)
      fs.readSync(fd, last, 0, 1, size - 1)
      if (last[0] !== 0x0a) lead = '\n'
    }
    fs.writeSync(fd, lead + JSON.stringify(row) + '\n')
    fs.fsyncSync(fd)
  } finally {
    fs.closeSync(fd)
  }
}

/** Every whole line of a JSON-lines file; a torn or unreadable line is counted and left out. */
export function readJsonl<T>(file: string): { rows: T[]; torn: number } {
  let text: string
  try {
    text = fs.readFileSync(file, 'utf8')
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return { rows: [], torn: 0 }
    throw err
  }
  const rows: T[] = []
  let torn = 0
  for (const line of text.split('\n')) {
    if (!line.trim()) continue
    try {
      rows.push(JSON.parse(line) as T)
    } catch {
      torn += 1
    }
  }
  return { rows, torn }
}

/** Write an entry to runs/<run>/ledger.jsonl, flushed to disk before this returns. */
export function appendLedger(dir: string, e: LedgerEntry): void {
  appendJsonl(path.join(dir, LEDGER_FILE), e)
}

export function readLedger(dir: string): LedgerEntry[] {
  return readJsonl<LedgerEntry>(path.join(dir, LEDGER_FILE)).rows.filter((e) => e && typeof (e as { t?: unknown }).t === 'string')
}

// ------------------------------------------------------------ cells.jsonl --

/** A cells.jsonl row. `removed` marks a picture the quarantine rule deleted: it is never made again, nor reused. */
export type DoneRow = DoneCell & { removed?: boolean }

export const cellsIndexPath = (env: LabEnv) => path.join(env.labDir, 'cells.jsonl')

export function appendDoneCell(env: LabEnv, row: DoneRow): void {
  appendJsonl(cellsIndexPath(env), row)
}

/** The lab's finished pictures, by cell id. A removal stays, whatever comes after it. */
export function readDoneCells(env: LabEnv): Map<string, DoneRow> {
  const out = new Map<string, DoneRow>()
  for (const r of readJsonl<DoneRow>(cellsIndexPath(env)).rows) {
    if (!r || typeof r.cellId !== 'string') continue
    const had = out.get(r.cellId)
    if (had?.removed) continue
    if (r.removed) {
      out.set(r.cellId, { ...(had ?? r), removed: true })
      continue
    }
    if (typeof r.rel !== 'string' || !r.rel) continue
    // The first picture kept for a cell stays its picture.
    if (!had) out.set(r.cellId, r)
  }
  return out
}

// -------------------------------------------------------------- the state --

type Ended = Extract<LedgerEntry, { t: 'ended' }>
type FileLike = { filename: string; subfolder?: string | null; type?: string | null }

/** `subfolder/filename`, the path under outputs. */
export const relOfFile = (f: FileLike) => (f.subfolder ? `${f.subfolder}/${f.filename}` : f.filename)

export type DoneInfo = {
  rel: string
  job: string | null
  /** ComfyUI's execution time; 0 = not measured (a picture recovered from disk, or a reused one without it). */
  durationMs: number
  cold: boolean
  cached: boolean
  finishedAt: number | null
  how: 'made' | 'recovered' | 'reused'
}

export type FailInfo = {
  /** Why it will not be made in this run: 'refused', 'no-file', or the last ending after its second try. */
  why: string
  message: string | null
  attempts: number
}

export type RunState = {
  /** Cells of the plan still to send. */
  pending: Set<string>
  /** Jobs sent or about to be, with no ending yet: job id → cell id. */
  inFlight: Map<string, string>
  done: Map<string, DoneInfo>
  /** Cells that will not be sent again in this run. */
  failed: Map<string, FailInfo>
  /** Endings that count as a try, by cell (a stop, a skip or a missing upstream frame do not). */
  attempts: Map<string, number>
  /** Pictures the quarantine rule deleted. */
  removed: Set<string>
  paused: boolean
  pausedWhy: 'user' | 'until' | 'runner-off' | null
  /**
   * Groups sent before a pause by the user or at the --until time: the lab
   * asked the runner to stop them, so a stop seen on their jobs is the lab's
   * own, even when the process that asked has ended since.
   */
  pausedAfter: Set<string>
  /** Groups written as 'submitting' with no answer recorded. */
  unanswered: Map<string, { job: string; cell: string }[]>
  /** Every group this run wrote, with its jobs. */
  groups: Map<string, { jobs: { job: string; cell: string }[]; answer: 'none' | 'submitted' | 'refused' }>
  /** The group each job was sent in. */
  jobGroup: Map<string, string>
  /** The cell each job was for, whatever became of it. */
  jobCell: Map<string, string>
}

/** An ending that is never tried again: ComfyUI refused the graph, or the run wrote no file. */
export const FINAL_CODES = new Set(['refused', 'no-file'])
/** Endings that send the cell back to be sent again without counting a try. */
const REQUEUE_STATUSES = new Set(['stopped', 'skipped'])
/** The most tries a cell gets: the first, and one more after a failure or loss. */
export const MAX_ATTEMPTS = 2

/** Whether an ending counts as a try at the cell. */
export function countsAsAttempt(e: { status: string; error: { code: string } | null }): boolean {
  if (e.status === 'done') return true
  if (REQUEUE_STATUSES.has(e.status)) return false
  if (e.status === 'failed' && e.error?.code === 'no-frame') return false
  return true
}

/**
 * The run as its ledger leaves it. `plan` names the cells (its order and the
 * ones it reuses); `global`, the lab's finished pictures from cells.jsonl,
 * turns a cell made in another run into a reused one.
 */
export function runState(
  entries: readonly LedgerEntry[],
  plan: { order: readonly string[]; reused?: readonly string[] },
  global?: ReadonlyMap<string, DoneRow>,
): RunState {
  const inFlight = new Map<string, string>()
  const done = new Map<string, DoneInfo>()
  const finalFail = new Map<string, FailInfo>()
  const attempts = new Map<string, number>()
  const removed = new Set<string>()
  const groups: RunState['groups'] = new Map()
  const jobGroup = new Map<string, string>()
  const jobCell = new Map<string, string>()
  let paused = false
  let pausedWhy: RunState['pausedWhy'] = null
  const pausedAfter = new Set<string>()

  const dropJobsOf = (cell: string) => {
    for (const [job, c] of inFlight) if (c === cell) inFlight.delete(job)
  }

  for (const e of entries) {
    switch (e.t) {
      case 'submitting': {
        groups.set(e.group, { jobs: e.jobs.map((j) => ({ job: j.job, cell: j.cell })), answer: 'none' })
        for (const j of e.jobs) {
          jobGroup.set(j.job, e.group)
          jobCell.set(j.job, j.cell)
          if (!done.has(j.cell)) inFlight.set(j.job, j.cell)
        }
        paused = false
        pausedWhy = null
        break
      }
      case 'submitted': {
        const g = groups.get(e.group)
        if (g) g.answer = 'submitted'
        break
      }
      case 'refused': {
        const g = groups.get(e.group)
        if (g) {
          g.answer = 'refused'
          for (const j of g.jobs) inFlight.delete(j.job)
        }
        break
      }
      case 'ended': {
        const ended = e as Ended
        inFlight.delete(ended.job)
        jobCell.set(ended.job, ended.cell)
        if (done.has(ended.cell)) break
        const counted = countsAsAttempt(ended)
        const n = (attempts.get(ended.cell) ?? 0) + (counted ? 1 : 0)
        if (counted) attempts.set(ended.cell, n)
        if (ended.status === 'done' && ended.primary) {
          done.set(ended.cell, {
            rel: relOfFile(ended.primary),
            job: ended.job,
            durationMs: ended.durationMs ?? 0,
            cold: !!ended.cold,
            cached: !!ended.cached,
            finishedAt: ended.finishedAt ?? null,
            how: 'made',
          })
          finalFail.delete(ended.cell)
          dropJobsOf(ended.cell)
          break
        }
        const code = ended.status === 'done' ? 'no-file' : ended.error?.code ?? ended.status
        if (FINAL_CODES.has(code) || (counted && n >= MAX_ATTEMPTS)) {
          finalFail.set(ended.cell, { why: code, message: ended.error?.message ?? null, attempts: n })
        }
        break
      }
      case 'recovered': {
        if (!done.has(e.cell)) {
          done.set(e.cell, { rel: e.rel, job: null, durationMs: 0, cold: false, cached: false, finishedAt: null, how: 'recovered' })
        }
        finalFail.delete(e.cell)
        dropJobsOf(e.cell)
        break
      }
      case 'removed': {
        removed.add(e.cell)
        break
      }
      case 'paused': {
        paused = true
        pausedWhy = e.why
        // A runner that was off is only waited for; nothing of ours was stopped.
        if (e.why !== 'runner-off') for (const g of groups.keys()) pausedAfter.add(g)
        break
      }
      default:
        break
    }
  }

  const cells = new Set<string>([...plan.order, ...(plan.reused ?? [])])
  const pending = new Set<string>()
  const failed = new Map<string, FailInfo>()
  const flying = new Set(inFlight.values())
  for (const cell of cells) {
    const row = global?.get(cell)
    if (removed.has(cell) || row?.removed) {
      removed.add(cell)
      done.delete(cell)
      continue
    }
    if (done.has(cell)) continue
    if (row) {
      done.set(cell, {
        rel: row.rel,
        job: null,
        durationMs: row.durationMs ?? 0,
        cold: !!row.cold,
        cached: !!row.cached,
        finishedAt: row.finishedAt ?? null,
        how: 'reused',
      })
      continue
    }
    if (flying.has(cell)) continue
    const f = finalFail.get(cell)
    if (f) {
      failed.set(cell, f)
      continue
    }
    pending.add(cell)
  }

  const unanswered = new Map<string, { job: string; cell: string }[]>()
  for (const [id, g] of groups) {
    if (g.answer !== 'none') continue
    const open = g.jobs.filter((j) => inFlight.has(j.job))
    if (open.length) unanswered.set(id, g.jobs)
  }

  return { pending, inFlight, done, failed, attempts, removed, paused, pausedWhy, pausedAfter, unanswered, groups, jobGroup, jobCell }
}
