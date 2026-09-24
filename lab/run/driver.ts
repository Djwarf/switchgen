/**
 * The driver: sends a planned run's pictures through the app's runner, as
 * desk 'lab' kind 'set', one lab group at a time, model by model, and
 * records every step in the run's ledger before it is taken.
 *
 * Nothing is sent until start() is called (the Start button on the lab page,
 * or `lab/lab start`). Before the first group it checks, and refuses with the
 * reason in words when any check fails:
 *   - the runner is on and has the 'lab' desk;
 *   - the calibration pairs of the study are judged (unless skipped);
 *   - no other run is being made, and no other process sends this one;
 *   - every photo the night needs is there (and its marked area, for a
 *     region test), the very one the run was planned with: each cell names
 *     its photo and its mask by their hashes;
 *   - every picture still to make passes the lab's own checks (verifyCell)
 *     and ComfyUI's list of nodes (checkGraph against GET /object_info).
 *
 * Then it takes up what the ledger says: a group written as 'submitting'
 * with no answer is POSTed again with the identical body (the runner answers
 * replayed when it had taken it); a job the runner has since pruned is found
 * on disk by its cell id, or counted lost.
 *
 * Each group holds the next model's pending pictures (at most 200), in the
 * plan's order, with each chain step right after the picture it works on.
 * The runner is read every 5 s; each ended job is written to the ledger
 * (and to cells.jsonl when it made its picture). A failed or lost picture is
 * tried once more, in a later group with a new job id; a refusal by ComfyUI
 * or a run that wrote no file is never retried; a skipped or stopped one goes
 * back to be sent. A held lane is reported and waited on (the lab never
 * sends the lane word), as is a runner that is off.
 *
 * When the last picture has ended: the picture reader, then the seal.
 *
 * The status it reports never names a model, a weight file or a family:
 * counts, states and plain sentences only.
 */
import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import type { LabEnv } from '../core/env.ts'
import { maskRel, runDir } from '../core/env.ts'
import type { Cell, JobError, LedgerEntry } from '../core/types.ts'
import { finalizeGraph, maskKey, verifyCell } from '../core/cells.ts'
import type { Plan } from '../core/plan.ts'
import { startRefusal, suiteById } from '../core/plan.ts'
import type { ObjectInfo } from '../core/validate.ts'
import { checkGraph, fetchObjectInfo } from '../core/validate.ts'
import type { ApiWorkflow } from '../../src/lib/comfy.ts'
import type { DoneRow, RunState } from './ledger.ts'
import { appendDoneCell, appendLedger, countsAsAttempt, readDoneCells, readLedger, relOfFile, runState } from './ledger.ts'
import type { Held } from './lock.ts'
import { LockHeld, acquireDriverLock, liveDrivers } from './lock.ts'
import { readCells } from './reader.ts'
import type { RefCopy } from './refs.ts'
import { ensureRefCopies, listRefs, missingSentence, noAreaSentence } from './refs.ts'
import type { GroupBody, JobView, LabJobBody, RunnerClient, Snapshot } from './runnerClient.ts'
import { AppUnreachable, TERMINAL } from './runnerClient.ts'
import type { CellFacts } from './timing.ts'
import { coldFlag, isCached, previousRan } from './timing.ts'

export type DriverState = 'idle' | 'waiting-runner' | 'held' | 'sending' | 'paused' | 'reading' | 'sealing' | 'made' | 'error'

export type DriverStatus = {
  run: string
  state: DriverState
  made: number
  total: number
  failed: number
  etaSeconds: number | null
  until: string | null
  message: string | null
}

export type DriverOptions = {
  env: LabEnv
  run: string
  client: RunnerClient
  now?: () => number
  sleep?: (ms: number) => Promise<void>
  log?: (line: string) => void
  /** How often the runner is read. Default 5 s. */
  pollMs?: number
  /** ComfyUI's node list for checkGraph. Default: GET {COMFY_URL}/object_info, a read. */
  objectInfo?: () => Promise<ObjectInfo>
  /** What seals the run once it is read. Default: the judge's sealRun. */
  seal?: (env: LabEnv, run: string) => Promise<unknown>
  /** The ffmpeg the default seal runs. Default SWITCHGEN_FFMPEG or 'ffmpeg'. */
  ffmpeg?: string
  /** New group and job ids (lowercase v4 UUIDs). */
  newId?: () => string
  /** How long a runner that is off is waited for before the run is paused. Default 30 min. */
  runnerOffLimitMs?: number
}

export type Driver = {
  start(opts?: { until?: string; skipCalibration?: boolean }): Promise<void>
  pause(): Promise<void>
  status(): DriverStatus
}

/** The most jobs a group may hold (routes.mjs JOBS_MAX). */
export const GROUP_MAX = 200
export const DEVICE = 'switchgen-lab'
const POLL_MS = 5000
const RUNNER_OFF_LIMIT_MS = 30 * 60_000
const BACKOFF_MAX_MS = 60_000

// ------------------------------------------------------------- run files --

/** The parts of plan.json the driver reads (core/plan.ts writes it). */
export type PlanLike = {
  run?: string
  study?: string
  suites?: { id: string; version?: number; sha?: string }[]
  createdAt?: string | number
  order: string[]
  reused?: string[]
  estimate?: { pictures?: number; newPictures?: number; seconds?: number }
  /** This run's own cell rows. */
  cells?: Cell[]
  /** Tests waiting for a photo or its rectangle: the run must not start while any is listed. */
  blocked?: Plan['blocked']
  /** Each cell's words without the prefix: the runner job's prompt. */
  prompts?: Record<string, string>
}

export const planPath = (env: LabEnv, run: string) => path.join(runDir(env, run), 'plan.json')
export const sentPath = (env: LabEnv, run: string, group: string) => path.join(runDir(env, run), 'sent', `${group}.json`)
export const cellFilePath = (env: LabEnv, cellId: string) => path.join(env.labDir, 'cells', `${cellId}.json`)

export function readPlan(env: LabEnv, run: string): PlanLike | null {
  try {
    const p = JSON.parse(fs.readFileSync(planPath(env, run), 'utf8'))
    if (p && Array.isArray(p.order)) return p as PlanLike
  } catch {
    /* not planned */
  }
  return null
}

export type CellFile = { cell: Cell; graph: ApiWorkflow; text: string }

/**
 * A cell as cells/<cellId>.json keeps it: the built graph and the cell. Both
 * {cell, graph} and the cell's fields beside `graph` are read; the slot's
 * words, when kept, are the runner job's prompt.
 */
export function loadCellFile(env: LabEnv, cellId: string): CellFile {
  const raw = JSON.parse(fs.readFileSync(cellFilePath(env, cellId), 'utf8')) as Record<string, unknown>
  const graph = raw.graph as ApiWorkflow | undefined
  const cell = (raw.cell && typeof raw.cell === 'object' ? raw.cell : raw) as Cell
  if (!graph || typeof graph !== 'object' || !cell || cell.cellId !== cellId) throw new Error(`cells/${cellId}.json is not that cell's file`)
  const t = [raw.text, raw.prompt, (raw.cell as { text?: unknown } | undefined)?.text].find((v) => typeof v === 'string')
  return { cell, graph, text: typeof t === 'string' ? t : '' }
}

function writeSynced(file: string, text: string): void {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  const tmp = `${file}.tmp-${process.pid}`
  const fd = fs.openSync(tmp, 'w', 0o600)
  try {
    fs.writeSync(fd, text)
    fs.fsyncSync(fd)
  } finally {
    fs.closeSync(fd)
  }
  fs.renameSync(tmp, file)
}

// ------------------------------------------------------ the calibration --

const isCalibration = (p: PlanLike) => (p.suites ?? []).some((s) => /calibration/i.test(s.id))

/**
 * Whether a run may start as far as the calibration night goes: the
 * calibration run itself always may; any other run of its study waits for
 * the calibration pairs to be judged (the judge writes sweep.json then).
 */
export function calibrationGate(env: LabEnv, run: string): { ok: true } | { ok: false; why: string } {
  const plan = readPlan(env, run)
  if (!plan) return { ok: false, why: `The ${run} run has not been planned yet. Plan it first (lab/lab plan).` }
  if (isCalibration(plan)) return { ok: true }
  const runs = path.join(env.labDir, 'runs')
  let names: string[] = []
  try {
    names = fs.readdirSync(runs)
  } catch {
    names = []
  }
  const cals = names.filter((n) => {
    if (n === run) return false
    const p = readPlanQuiet(env, n)
    return !!p && isCalibration(p) && (p.study ?? null) === (plan.study ?? null)
  })
  if (!cals.length) {
    return { ok: false, why: 'The calibration night for this study has not been planned or made yet. Make and judge it first, or start with "skip calibration".' }
  }
  if (cals.some((n) => fs.existsSync(path.join(runs, n, 'sweep.json')))) return { ok: true }
  return { ok: false, why: `The calibration pairs (${cals.join(', ')}) are not judged yet. Judge them first, or start with "skip calibration".` }
}

function readPlanQuiet(env: LabEnv, run: string): PlanLike | null {
  try {
    return readPlan(env, run)
  } catch {
    return null
  }
}

/** Whether `run` may start now: planned, the calibration gate met (unless skipped), and no other run live. */
export function canStart(env: LabEnv, run: string, opts: { skipCalibration?: boolean } = {}): { ok: true } | { ok: false; why: string } {
  const plan = readPlan(env, run)
  if (!plan) return { ok: false, why: `The ${run} run has not been planned yet. Plan it first (lab/lab plan).` }
  const blocked = plan.blocked?.length ? startRefusal({ blocked: plan.blocked }, env) : null
  if (blocked) return { ok: false, why: blocked }
  const other = liveDrivers(env).find((d) => d.run !== run)
  if (other) return { ok: false, why: `The ${other.run} run is being made now. Pause it before starting another.` }
  if (!opts.skipCalibration) {
    const gate = calibrationGate(env, run)
    if (!gate.ok) return gate
  }
  return { ok: true }
}

// ------------------------------------------------------------------ until --

/** The next time the clock reads HH:MM (24-hour, local) after `from`. Throws on anything else. */
export function untilTime(hhmm: string, from: number): number {
  const m = /^([01]?\d|2[0-3]):([0-5]\d)$/.exec(String(hhmm).trim())
  if (!m) throw new Error(`"${hhmm}" is not a time. Use the 24-hour clock, like 07:30.`)
  const d = new Date(from)
  d.setHours(Number(m[1]), Number(m[2]), 0, 0)
  if (d.getTime() <= from) d.setDate(d.getDate() + 1)
  return d.getTime()
}

// ----------------------------------------------------------------- driver --

const placeholderSha = (v: string, kind: 'ref' | 'mask') => (v.startsWith(`${kind}:`) ? v.slice(kind.length + 1) : null)

const REPLACED = 'A photo this run was planned with has been replaced or removed since. Plan the run again.'

/** A run name no run has yet, after this one's: exta-1 gives exta-2. Null when none fits the rules. */
function freshRunName(env: LabEnv, run: string): string | null {
  const m = /^(.*?)(\d+)$/.exec(run)
  const stem = m ? m[1] : `${run}-`
  for (let n = m ? Number(m[2]) + 1 : 2; n < 1000; n++) {
    try {
      if (!fs.existsSync(runDir(env, `${stem}${n}`))) return `${stem}${n}`
    } catch {
      return null
    }
  }
  return null
}

/**
 * How to go on when a started run cannot use the photos the lab has now: a
 * started run keeps its plan (the server refuses to plan it again), so the
 * way on is a new run of the same nights under another name, with the command.
 */
function newRunWords(env: LabEnv, run: string, plan: PlanLike): string {
  const ids = (plan.suites ?? []).map((s) => s.id)
  const name = freshRunName(env, run)
  let cmd: string | null = null
  if (name && ids.length === 1 && ids[0] === 'smoke') cmd = `lab/lab smoke --run ${name}`
  else if (name && ids.length && ids.every((id) => suiteById(id))) cmd = `lab/lab plan ${ids.join(' ')} --run ${name}`
  return `plan a new run under another name${cmd ? `: ${cmd}` : ''}.`
}

const jobError = (e: JobView['error']): JobError | null =>
  e ? { code: String(e.code ?? 'failed'), message: typeof e.message === 'string' ? e.message : '', node: e.node ?? null, nodeType: e.nodeType ?? null } : null

class Refusal extends Error {}

export function createDriver(o: DriverOptions): Driver {
  const { env, run, client } = o
  const now = o.now ?? Date.now
  const sleep = o.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)))
  const log = o.log ?? (() => {})
  const pollMs = o.pollMs ?? POLL_MS
  const newId = o.newId ?? (() => randomUUID())
  const offLimit = o.runnerOffLimitMs ?? RUNNER_OFF_LIMIT_MS
  const objectInfo = o.objectInfo ?? (() => fetchObjectInfo(env.comfyUrl))
  const ffmpeg = o.ffmpeg ?? process.env.SWITCHGEN_FFMPEG ?? 'ffmpeg'
  // Loaded only when a run is sealed.
  const seal = o.seal ?? (async (e: LabEnv, r: string) => (await import('../judge/seal.ts')).sealRun(e, r, { ffmpeg }))
  const dir = runDir(env, run)

  const status: DriverStatus = { run, state: 'idle', made: 0, total: 0, failed: 0, etaSeconds: null, until: null, message: null }
  let running: Promise<void> | null = null
  let pauseAsked = false
  /** Why the loop is winding down, once it is. */
  let stopping: 'user' | 'until' | 'runner-off' | null = null
  /** A job of this run was stopped by someone other than the lab (the app's own controls). */
  let stoppedOutside = false
  let pausedWritten = false
  let wake: (() => void) | null = null
  let lock: Held | null = null

  // Per start.
  let plan: PlanLike = { order: [] }
  let cells = new Map<string, CellFile>()
  let children = new Map<string, string[]>()
  let planCells: string[] = []
  let state: RunState | null = null
  let globalDone: ReadonlyMap<string, DoneRow> = new Map()
  let blocked = new Set<string>()
  let refs: Record<string, RefCopy> = {}
  let shaToRef = new Map<string, string>()
  /** Each marked area's hash (cells.ts maskKey), to the photo it is drawn on. */
  let maskToRef = new Map<string, string>()
  /** What to say for a photo that is not the one planned: a started run cannot be planned again. */
  let replaced = REPLACED
  let sessionStart = 0
  let madeAtStart = 0
  let groupsMade = 0
  let groupsGuess = 0

  const set = (s: DriverState, message: string | null) => {
    status.state = s
    status.message = message
  }

  /** A sleep that pause() cuts short. */
  const nap = (ms: number) =>
    new Promise<void>((resolve) => {
      let done = false
      const finish = () => {
        if (done) return
        done = true
        wake = null
        resolve()
      }
      wake = finish
      sleep(ms).then(finish, finish)
    })

  const write = (e: LedgerEntry) => appendLedger(dir, e)

  function refresh(): RunState {
    globalDone = readDoneCells(env)
    state = runState(readLedger(dir), plan, globalDone)
    blocked = blockedCells(state)
    const made = state.done.size + state.removed.size
    status.made = made
    status.total = planCells.length
    status.failed = state.failed.size + blocked.size
    const left = Math.max(0, status.total - made - status.failed)
    const madeNow = made - madeAtStart
    if (!left) status.etaSeconds = 0
    else if (madeNow >= 3 && sessionStart) status.etaSeconds = Math.round(((now() - sessionStart) / 1000 / madeNow) * left)
    else if (plan.estimate?.seconds && plan.estimate.newPictures) status.etaSeconds = Math.round((plan.estimate.seconds / plan.estimate.newPictures) * left)
    else status.etaSeconds = null
    return state
  }

  /** Where a finished picture is, made in this run or any other, or null. */
  function doneRel(s: RunState, cellId: string): string | null {
    const here = s.done.get(cellId)?.rel
    if (here) return here
    const row = globalDone.get(cellId)
    return row && !row.removed && row.rel ? row.rel : null
  }

  /** Pending cells that can never be sent: the picture they work on will not be made. */
  function blockedCells(s: RunState): Set<string> {
    const out = new Set<string>()
    let grew = true
    while (grew) {
      grew = false
      for (const id of s.pending) {
        if (out.has(id)) continue
        const up = cells.get(id)?.cell.upstream
        if (!up) continue
        const upGone = s.failed.has(up) || s.removed.has(up) || out.has(up) || !!globalDone.get(up)?.removed || (!doneRel(s, up) && !cells.has(up))
        if (upGone) {
          out.add(id)
          grew = true
        }
      }
    }
    return out
  }

  const cellFacts = (cellId: string): CellFacts | null => {
    const c = cells.get(cellId)?.cell
    if (c) return { file: c.file, op: c.op }
    try {
      const f = loadCellFile(env, cellId)
      return { file: f.cell.file, op: f.cell.op }
    } catch {
      return null
    }
  }

  // ---------------------------------------------------------- the binding --

  /**
   * The paths the graph's picture placeholders become, keyed by placeholder.
   * A photo that is not ready to send is refused in words that name it.
   */
  function refBinding(cell: Cell): Record<string, string> {
    const out: Record<string, string> = {}
    for (const [, , v] of cell.placeholders) {
      const r = placeholderSha(v, 'ref')
      const m = placeholderSha(v, 'mask')
      if (!r && !m) continue
      // A photo by its own hash, a marked area by the mask's.
      const id = r ? shaToRef.get(r) : maskToRef.get(m as string)
      if (!id) throw new Refusal(replaced)
      const copy = refs[id]
      if (!copy) throw new Refusal(`The "${id}" photo was not copied where ComfyUI reads it. Press Start again, and the lab copies it before sending.`)
      if (r) {
        if (copy.sha12 !== r) throw new Refusal(replaced)
        out[v] = copy.ref
      } else {
        if (!copy.mask) throw new Refusal(noAreaSentence(id))
        if (copy.mask !== maskRel(m as string)) throw new Refusal(replaced)
        out[v] = copy.mask
      }
    }
    return out
  }

  /** The graph to send for a cell, bound to its upstream's picture or the chain token. */
  function bind(cf: CellFile, upstream: { rel: string } | 'token' | undefined) {
    const made = finalizeGraph(cf.graph, cf.cell, { upstream, refs: refBinding(cf.cell) })
    return { graph: made.graph, chainAt: made.chainAt, problems: verifyCell(made.graph, cf.cell) }
  }

  // ------------------------------------------------------------ preflight --

  async function preflight(): Promise<void> {
    set('idle', 'Checking everything before sending anything.')
    const p = readPlan(env, run)
    if (!p) throw new Refusal(`The ${run} run has not been planned yet. Plan it first (lab/lab plan).`)
    plan = p
    planCells = [...new Set([...p.order, ...(p.reused ?? [])])]
    status.total = planCells.length
    // A night that waits for a photo, or the rectangle drawn on it, says so and does not start.
    const waits = p.blocked?.length ? startRefusal({ blocked: p.blocked }, env) : null
    if (waits) throw new Refusal(waits)

    // The runner: on, with the lab's desk.
    let caps
    try {
      caps = await client.capabilities()
    } catch (err) {
      if (err instanceof AppUnreachable) throw new Refusal(`The app is not answering at ${client.appUrl}, so nothing can be sent. Start the app and try again.`)
      throw new Refusal(`The app would not say what it can do (${(err as Error).message}).`)
    }
    if (!caps.runner) throw new Refusal(`The app's runner is off${caps.runnerReason ? `: ${caps.runnerReason}` : '.'} The lab sends its work only through the runner.`)
    if (!caps.runnerDesks.includes('lab')) {
      throw new Refusal("The app's runner has no lab desk, so the lab cannot send its work without it landing in History or on a desk. It needs the runner update that adds the lab desk.")
    }

    // The cells: the graph from cells/<id>.json, the row as this run planned it.
    cells = new Map()
    const rows = new Map((p.cells ?? []).map((c) => [c.cellId, c]))
    for (const id of planCells) {
      let cf: CellFile
      try {
        cf = loadCellFile(env, id)
      } catch {
        throw new Refusal(`A picture this run was planned with is missing from the lab folder (cells/${id}.json). Plan the run again.`)
      }
      const row = rows.get(id)
      if (row) cf = { ...cf, cell: row }
      const words = p.prompts?.[id]
      if (typeof words === 'string') cf = { ...cf, text: words }
      cells.set(id, cf)
    }
    children = new Map()
    for (const id of p.order) {
      const up = cells.get(id)?.cell.upstream
      if (up) children.set(up, [...(children.get(up) ?? []), id])
    }

    const s = refresh()
    const ledger = readLedger(dir)
    // A run that has sent anything keeps its plan: the lab will not plan it again.
    const started = ledger.length > 0
    const goOn = started ? newRunWords(env, run, p) : ''
    replaced = started ? `A photo this run was planned with has been replaced or removed since it started, and a started run keeps its plan. Put the photo back as it was, or ${goOn}` : REPLACED

    // The photos, each checked by its hash, and its marked area by the mask's
    // own hash: the words a cell was planned with are in its plan, so a
    // description saved since changes nothing here. A night not started yet
    // takes new words because Start plans it again first.
    const infos = listRefs(env)
    shaToRef = new Map(infos.map((r) => [r.sha12, r.id]))
    maskToRef = new Map(infos.flatMap((r) => (r.mask ? [[maskKey(r), r.id] as const] : [])))
    const byId = new Map(infos.map((r) => [r.id, r]))
    const needed = new Set<string>()
    const problems: string[] = []
    const todo = [...s.pending, ...s.inFlight.values()]
    for (const id of todo) {
      const c = cells.get(id)?.cell
      if (!c) continue
      const photos: string[] = []
      for (const r of c.refs ?? []) {
        const refId = byId.has(r) ? r : shaToRef.get(r)
        if (!refId) problems.push(byId.size && /^[0-9a-f]{12}$/.test(r) ? replaced : missingSentence(env, r))
        else {
          needed.add(refId)
          photos.push(refId)
        }
      }
      for (const [, , v] of c.placeholders) {
        const r = placeholderSha(v, 'ref')
        const m = placeholderSha(v, 'mask')
        if (r) {
          const refId = shaToRef.get(r)
          if (refId) needed.add(refId)
          else problems.push(replaced)
          continue
        }
        if (!m) continue
        const refId = maskToRef.get(m)
        if (refId) {
          needed.add(refId)
          continue
        }
        // Not the area this run was planned with: gone, or drawn again since.
        const photo = photos[0]
        if (!photo) problems.push(replaced)
        else if (!byId.get(photo)?.mask) problems.push(noAreaSentence(photo))
        else if (started) problems.push(`The area marked on the "${photo}" photo changed after this run started, and a started run keeps its plan. To use the new area, ${goOn}`)
        else problems.push(`The area marked on the "${photo}" photo changed after this run was planned. Plan the run again.`)
      }
    }
    if (problems.length) throw new Refusal([...new Set(problems)].join(' '))
    try {
      refs = ensureRefCopies(env, [...needed])
    } catch (err) {
      throw new Refusal((err as Error).message)
    }

    // Every picture still to make, against the lab's own checks and ComfyUI's node list.
    let info: ObjectInfo
    try {
      info = await objectInfo()
    } catch (err) {
      throw new Refusal(`ComfyUI did not answer the lab's check of its node list (${(err as Error).message}). Nothing was sent.`)
    }
    const found: string[] = []
    for (const id of todo) {
      const cf = cells.get(id)
      if (!cf) continue
      const up = cf.cell.upstream
      const upRel = up ? doneRel(s, up) ?? `.lab/cells/${up}_00001_.png` : undefined
      let b
      try {
        b = bind(cf, upRel ? { rel: upRel } : undefined)
      } catch (err) {
        // A photo that is not ready is said in its own words, not in preflight.txt.
        if (err instanceof Refusal) throw err
        found.push(`${id}: ${(err as Error).message}`)
        continue
      }
      for (const m of [...b.problems, ...checkGraph(b.graph, info)]) found.push(`${id}: ${m}`)
    }
    if (found.length) {
      const file = path.join(dir, 'preflight.txt')
      writeSynced(file, found.join('\n') + '\n')
      throw new Refusal(`${new Set(found.map((f) => f.slice(0, f.indexOf(':')))).size} of the pictures to make failed the lab's own check, so nothing was sent. The details are in ${file}.`)
    }

    // A picture removed by the quarantine rule in another run is removed here
    // too. Written only once every check has passed: a refused start leaves the
    // ledger empty, so a run that has sent nothing can still be planned again.
    const global = readDoneCells(env)
    const marked = new Set(ledger.flatMap((e) => (e.t === 'removed' ? [e.cell] : [])))
    for (const id of planCells) {
      if (global.get(id)?.removed && !marked.has(id)) write({ t: 'removed', at: now(), cell: id, why: 'quarantine' })
    }
  }

  // ------------------------------------------------------------ recording --

  /** Write the ending of a job the runner has finished. */
  function recordEnded(job: string, cell: string, jv: JobView, snap: Snapshot): void {
    const s = state as RunState
    const cached = isCached(jv)
    const cold = cached ? false : coldFlag(previousRan(snap.jobs, jv), jv, cellFacts)
    const error = jobError(jv.error)
    const counted = countsAsAttempt({ status: jv.status, error })
    const files = (jv.files ?? []).map((f) => ({ ...f }))
    const primary = jv.primary ? { ...jv.primary } : null
    write({
      t: 'ended',
      at: now(),
      job,
      cell,
      status: jv.status,
      error,
      files,
      primary,
      promptId: jv.promptId ?? null,
      durationMs: Number(jv.durationMs) || 0,
      ranAt: jv.ranAt ?? null,
      finishedAt: jv.finishedAt ?? null,
      cached,
      cold,
      attempt: (s.attempts.get(cell) ?? 0) + (counted ? 1 : 0),
    })
    if (jv.status === 'done' && primary) {
      const row: DoneRow = {
        cellId: cell,
        rel: relOfFile(primary),
        durationMs: Number(jv.durationMs) || 0,
        cold,
        cached,
        finishedAt: jv.finishedAt ?? null,
        run,
      }
      appendDoneCell(env, row)
    }
  }

  /** The newest picture on disk for a cell, as a path under outputs, or null. */
  function onDisk(cellId: string): string | null {
    const d = path.join(env.outputs, '.lab', 'cells')
    let best: { name: string; t: number } | null = null
    let names: string[] = []
    try {
      names = fs.readdirSync(d)
    } catch {
      return null
    }
    for (const name of names) {
      if (!name.startsWith(`${cellId}_`) || !/\.(png|webp|jpe?g)$/i.test(name)) continue
      try {
        const st = fs.statSync(path.join(d, name))
        if (st.isFile() && st.size > 0 && (!best || st.mtimeMs > best.t)) best = { name, t: st.mtimeMs }
      } catch {
        /* gone */
      }
    }
    return best ? `.lab/cells/${best.name}` : null
  }

  /** A picture found on disk for a job the runner no longer lists; its time is not known. */
  function recovered(cell: string, rel: string): void {
    write({ t: 'recovered', at: now(), cell, rel })
    appendDoneCell(env, { cellId: cell, rel, durationMs: 0, cold: false, cached: false, finishedAt: null, run })
  }

  /** A job the runner no longer lists: its picture from disk, or counted lost. */
  function recoverOrLose(job: string, cell: string): void {
    const s = state as RunState
    const rel = onDisk(cell)
    if (rel) {
      recovered(cell, rel)
      return
    }
    write({
      t: 'ended',
      at: now(),
      job,
      cell,
      status: 'lost',
      error: { code: 'lost', message: 'The runner no longer lists this job, and no picture for it is on disk.', node: null, nodeType: null },
      files: [],
      primary: null,
      promptId: null,
      durationMs: 0,
      ranAt: null,
      finishedAt: null,
      cached: false,
      cold: false,
      attempt: (s.attempts.get(cell) ?? 0) + 1,
    })
  }

  /** Write every ending the snapshot shows for this run's jobs in flight. */
  function recordEndings(snap: Snapshot): void {
    const s = refresh()
    const jobs = new Map(snap.jobs.map((j) => [j.id, j]))
    for (const [job, cell] of s.inFlight) {
      const g = s.jobGroup.get(job)
      const jv = jobs.get(job)
      if (jv) {
        if (TERMINAL.has(jv.status)) {
          recordEnded(job, cell, jv, snap)
          // A stop counts as made on the app only when the lab did not ask
          // for it, in this process or an earlier one: a pause written after
          // the job's group was sent means the lab asked, even if the process
          // that asked ended before it saw the stop land.
          const ours = stopping || pauseAsked || (g !== undefined && s.pausedAfter.has(g))
          if (jv.status === 'stopped' && !ours) stoppedOutside = true
        }
        continue
      }
      if (g && s.unanswered.has(g)) continue // settled by reconcile()
      // A runner that is off lists what it could read of its saved list, which
      // may be nothing: a missing job means something only from a running one.
      if (!snap.available) continue
      // The runner took it (a group and its jobs are taken in one commit) and
      // no longer lists it: pruned after it ended.
      recoverOrLose(job, cell)
    }
    refresh()
  }

  /** Our jobs the runner has not finished. */
  function liveJobs(snap: Snapshot): JobView[] {
    const s = state as RunState
    return snap.jobs.filter((j) => s.inFlight.has(j.id) && !TERMINAL.has(j.status))
  }

  // ------------------------------------------------------------- sending --

  /** Handle the runner's answer to a POST already written as 'submitting'. */
  async function answer(group: string, res: Awaited<ReturnType<RunnerClient['submit']>>): Promise<'ok' | 'retry' | 'stop'> {
    if (res.ok) {
      write({ t: 'submitted', at: now(), group, replayed: res.replayed })
      return 'ok'
    }
    write({ t: 'refused', at: now(), group, status: res.status, error: res.error })
    if (res.status === 503) {
      set('waiting-runner', `The runner is not taking work just now${res.reason ? ` (${res.reason})` : ''}. The lab will try again.`)
      return 'retry'
    }
    if (res.status === 507) throw new Refusal(`The server's disk is full, so the runner could not take the lab's work. ${res.error}`)
    throw new Refusal(`The runner refused the lab's work (${res.status}): ${res.error}`)
  }

  /**
   * Groups written as 'submitting' with no answer: POSTed again with the
   * identical body, which the runner answers as a replay when it took them
   * the first time and takes fresh when it never did. A group the runner no
   * longer has whose pictures are on disk was taken, made and since pruned:
   * its pictures are kept from disk and it is not sent again.
   */
  async function reconcile(snap: Snapshot): Promise<'none' | 'ok' | 'retry'> {
    const s = refresh()
    if (!s.unanswered.size) return 'none'
    const groups = new Set(snap.groups.map((g) => g.id))
    const jobs = new Set(snap.jobs.map((j) => j.id))
    for (const [group, members] of s.unanswered) {
      const held = groups.has(group) || members.some((m) => jobs.has(m.job))
      let body: GroupBody | null = null
      try {
        body = JSON.parse(fs.readFileSync(sentPath(env, run, group), 'utf8')) as GroupBody
      } catch {
        body = null
      }
      const disk = held ? [] : members.map((m) => ({ ...m, rel: onDisk(m.cell) }))
      if (!held && (!body || disk.some((d) => d.rel))) {
        // The runner once had it and has let it go: what is on disk is kept, the rest is lost.
        for (const d of disk) {
          if (d.rel) recovered(d.cell, d.rel)
          else recoverOrLose(d.job, d.cell)
        }
        continue
      }
      if (!body) {
        write({ t: 'submitted', at: now(), group, replayed: true })
        continue
      }
      log(`Sending part ${group.slice(0, 8)} again with the same body: its answer was lost.`)
      let res
      try {
        res = await client.submit(body)
      } catch (err) {
        if (err instanceof AppUnreachable) return 'retry'
        throw err
      }
      // The runner lists it, whatever it answers now (a 409 once some of its
      // jobs are pruned, a 503 if it has just gone off): it has the work.
      if (!res.ok && held) {
        write({ t: 'submitted', at: now(), group, replayed: true })
        continue
      }
      if ((await answer(group, res)) === 'retry') return 'retry'
    }
    refresh()
    return 'ok'
  }

  const keyOf = (c: Cell) => (c.chain ? `chain:${c.chain}` : `model:${c.model ?? c.file}`)

  /** The next group's cells: the next model's pending pictures, chain steps right after their upstream. */
  function nextCells(): string[] {
    const s = state as RunState
    const ready = (id: string) => s.pending.has(id) && !blocked.has(id)
    const upDone = (id: string) => {
      const up = cells.get(id)?.cell.upstream
      return !up || !!doneRel(s, up)
    }
    const order = plan.order.filter((id) => cells.has(id))
    const first = order.find((id) => ready(id) && upDone(id))
    if (!first) return []
    const key = keyOf(cells.get(first)!.cell)
    const picked: string[] = []
    const inGroup = new Set<string>()
    const add = (id: string) => {
      if (picked.length >= GROUP_MAX || inGroup.has(id)) return
      picked.push(id)
      inGroup.add(id)
      for (const child of children.get(id) ?? []) if (ready(child)) add(child)
    }
    for (const id of order) {
      if (picked.length >= GROUP_MAX) break
      if (!ready(id) || !upDone(id) || inGroup.has(id)) continue
      if (keyOf(cells.get(id)!.cell) !== key) continue
      add(id)
    }
    return picked
  }

  /** Roughly how many groups the pending pictures make, for the "part k of n" labels. */
  function guessGroups(): number {
    const s = state as RunState
    const byKey = new Map<string, number>()
    for (const id of s.pending) {
      const c = cells.get(id)?.cell
      if (!c || c.upstream) continue
      const k = keyOf(c)
      byKey.set(k, (byKey.get(k) ?? 0) + 1)
    }
    let n = 0
    for (const v of byKey.values()) n += Math.ceil(v / GROUP_MAX)
    return n
  }

  async function sendGroup(ids: string[]): Promise<'ok' | 'retry'> {
    const s = state as RunState
    const group = newId()
    const jobOf = new Map<string, string>()
    const jobs: LabJobBody[] = []
    const position = new Map(planCells.map((id, i) => [id, i + 1]))
    for (const id of ids) {
      const cf = cells.get(id)!
      const up = cf.cell.upstream
      const upRel = up ? doneRel(s, up) : null
      const upstream = up ? (upRel ? { rel: upRel } : jobOf.has(up) ? ('token' as const) : null) : undefined
      if (upstream === null) continue // its upstream is not made and not in this group: a later group
      const b = bind(cf, upstream)
      if (b.problems.length) throw new Refusal(`A picture failed the lab's own check just before sending (${b.problems.length} problem(s)); nothing more was sent.`)
      const job = newId()
      jobOf.set(id, job)
      const body: LabJobBody = {
        id: job,
        label: `Lab picture ${position.get(id) ?? 0} of ${planCells.length}`,
        prompt: cf.text.slice(0, 4000),
        kind: 'image',
        primary: 'image',
        orFirst: true,
        noFile: 'fail',
        heavy: false,
        graph: b.graph,
        record: { desk: 'lab', mode: cf.cell.op },
        meta: { lab: { run, cell: id } },
      }
      if (upstream === 'token') {
        if (!b.chainAt?.length) throw new Refusal("A chained picture's graph has no place for the picture before it; nothing more was sent.")
        body.chain = { after: jobOf.get(up as string)!, at: b.chainAt.map(([n, i]) => [n, i] as [string, string]) }
      }
      jobs.push(body)
    }
    if (!jobs.length) return 'ok'
    // Nothing is written or sent once a pause is asked: a group written after
    // the pause would not count as stopped by the lab.
    if (pauseAsked) return 'ok'
    groupsMade += 1
    groupsGuess = Math.max(groupsGuess, groupsMade)
    const body: GroupBody = {
      v: 1,
      group: { id: group, desk: 'lab', kind: 'set', label: `Lab ${run} · part ${groupsMade} of ${groupsGuess}`, device: DEVICE },
      jobs,
    }
    // The body first, then the ledger's word, then the POST: a crash at any
    // point leaves enough to send the identical body again.
    writeSynced(sentPath(env, run, group), JSON.stringify(body))
    write({ t: 'submitting', at: now(), group, jobs: jobs.map((j) => ({ job: j.id, cell: j.meta.lab.cell })) })
    set('sending', `Sending part ${groupsMade}: ${jobs.length} picture${jobs.length === 1 ? '' : 's'}.`)
    let res
    try {
      res = await client.submit(body)
    } catch (err) {
      if (err instanceof AppUnreachable) {
        set('waiting-runner', 'The app did not answer while the lab sent its work; the lab will check whether it arrived.')
        return 'retry'
      }
      throw err
    }
    const out = await answer(group, res)
    refresh()
    return out === 'retry' ? 'retry' : 'ok'
  }

  // ------------------------------------------------------------- the loop --

  async function stopOurs(snap: Snapshot | null): Promise<void> {
    const s = state as RunState
    const groups = new Set<string>()
    for (const j of snap ? liveJobs(snap) : []) groups.add(j.groupId)
    if (!snap) for (const job of s.inFlight.keys()) groups.add(s.jobGroup.get(job) ?? '')
    for (const g of groups) {
      if (!g) continue
      try {
        await client.stopGroup(g)
      } catch {
        /* the loop keeps reading; a stop that did not land is asked again next pass */
      }
    }
  }

  function writePaused(why: 'user' | 'until' | 'runner-off'): void {
    if (pausedWritten) return
    pausedWritten = true
    write({ t: 'paused', at: now(), why })
  }

  async function loop(untilAt: number | null): Promise<void> {
    let offSince: number | null = null
    let backoff = pollMs
    let lastSnap: Snapshot | null = null
    for (;;) {
      if (!stopping && pauseAsked) stopping = 'user'
      if (!stopping && untilAt != null && now() >= untilAt) {
        stopping = 'until'
        writePaused('until')
        await stopOurs(lastSnap)
      }

      let snap: Snapshot
      try {
        snap = await client.snapshot()
      } catch (err) {
        offSince ??= now()
        if (stopping) {
          set('paused', stopping === 'user' ? 'Paused. The app is not answering, so the lab could not confirm the stop.' : 'Stopped at the time asked. The app is not answering.')
          return
        }
        if (now() - offSince >= offLimit) {
          writePaused('runner-off')
          set('paused', 'The app was not answering for a long while, so the lab stopped waiting. Press Start when it is back.')
          return
        }
        set('waiting-runner', `The app is not answering (${err instanceof AppUnreachable ? 'no connection' : (err as Error).message}). The lab keeps trying.`)
        await nap(backoff)
        backoff = Math.min(backoff * 2, BACKOFF_MAX_MS)
        continue
      }
      lastSnap = snap
      if (!snap.available) {
        offSince ??= now()
        recordEndings(snap)
        if (stopping) {
          set('paused', 'Paused. The runner is off.')
          return
        }
        if (now() - offSince >= offLimit) {
          writePaused('runner-off')
          set('paused', 'The runner was off for a long while, so the lab stopped waiting. Press Start when it is back on.')
          return
        }
        set('waiting-runner', `The runner is off${snap.reason ? `: ${snap.reason}` : '.'} The lab waits for it.`)
        await nap(backoff)
        backoff = Math.min(backoff * 2, BACKOFF_MAX_MS)
        continue
      }
      offSince = null
      backoff = pollMs

      recordEndings(snap)
      const settled = await reconcile(snap)
      if (settled === 'retry') {
        await nap(pollMs)
        continue
      }
      // Groups sent again: read the runner afresh before going on.
      if (settled === 'ok') continue
      const live = liveJobs(snap)

      if (!stopping && stoppedOutside) {
        // Someone stopped the lab's pictures on the app: taken as a pause,
        // so the lab does not send them again behind their back.
        stopping = 'user'
        pauseAsked = true
        writePaused('user')
        if (live.length) await stopOurs(snap)
      }

      if (stopping) {
        if (live.length) {
          await stopOurs(snap) // asked again until it lands
          set('paused', 'Stopping: waiting for the picture being made to end.')
          await nap(pollMs)
          continue
        }
        set(
          'paused',
          stoppedOutside
            ? "Paused, because the lab's pictures were stopped on the app. Press Start to go on from here."
            : stopping === 'user'
              ? 'Paused. Press Start to go on from here.'
              : `Stopped at ${status.until} as asked. Press Start to go on.`,
        )
        return
      }

      if (live.length) {
        const held = snap.lane?.held
        if (held && live.some((j) => j.wait?.for === 'held')) {
          set('held', 'The runner is holding its waiting work after a restart or a lost job. Answer it on the app (send or stop); the lab waits and never answers it itself.')
        } else {
          set('sending', `Making pictures: ${status.made} of ${status.total} done.`)
        }
        await nap(pollMs)
        continue
      }

      // Only one lab group at a time, whoever sent it.
      const otherLab = snap.groups.some((g) => g.desk === 'lab' && g.state === 'active' && !(state as RunState).groups.has(g.id))
      if (otherLab) {
        set('sending', "Another lab run's pictures are being made; waiting for them to end.")
        await nap(pollMs)
        continue
      }

      const ids = nextCells()
      if (!ids.length) {
        // Nothing more to send. Jobs still listed as sent but not yet seen on
        // the runner are waited for; otherwise the run is made.
        if ((state as RunState).inFlight.size) {
          await nap(pollMs)
          continue
        }
        return
      }
      // A pause asked while the runner was being read, with nothing in
      // flight: the next pass winds down instead of sending a new group.
      if (pauseAsked) continue
      if ((await sendGroup(ids)) === 'retry') await nap(pollMs)
    }
  }

  async function finish(): Promise<void> {
    const s = refresh()
    const inPlan = new Set(planCells)
    const rels = new Map([...s.done].filter(([id]) => inPlan.has(id)).map(([id, d]) => [id, d.rel]))
    const notMade = status.failed
    set('reading', 'The picture reader is reading the pictures.')
    // The reader waits up to 5 minutes at a time while the machine is short of
    // memory: a pause cuts that wait short, or skips it once asked.
    const wait = (ms: number) => (pauseAsked ? Promise.resolve() : nap(ms))
    let result
    try {
      result = await readCells(env, client, [...rels.keys()], { run, rels, sleep: wait, now, log, shouldStop: () => pauseAsked })
    } catch (err) {
      set('made', `All pictures that could be made are made, but the picture reader could not run (${(err as Error).message}). The run is not sealed yet: run lab/lab read ${run}, then lab/lab seal ${run}.`)
      return
    }
    refresh()
    if (pauseAsked && result.pending.length) {
      set('paused', `Paused while reading. All pictures that could be made are made; press Start to finish reading and sealing.`)
      return
    }
    if (result.pending.length) {
      set('made', `All pictures that could be made are made. The picture reader could not read ${result.pending.length} of them${result.why ? ` (${result.why})` : ''}, so the run is not sealed yet: run lab/lab read ${run}, then lab/lab seal ${run}.`)
      return
    }
    set('sealing', 'Sealing: making the blind copies for judging.')
    try {
      await seal(env, run)
    } catch (err) {
      set('error', `The pictures are made and read, but sealing failed (${(err as Error).message}). Run lab/lab seal ${run} to try again.`)
      return
    }
    const removed = result.quarantined.length
    set(
      'made',
      `Made and sealed: ready to judge.${notMade ? ` ${notMade} picture${notMade === 1 ? '' : 's'} could not be made.` : ''}${removed ? ` ${removed} picture${removed === 1 ? ' was' : 's were'} removed by the content rule.` : ''}`,
    )
  }

  async function go(opts: { until?: string; skipCalibration?: boolean }): Promise<void> {
    pauseAsked = false
    pausedWritten = false
    stopping = null
    stoppedOutside = false
    groupsMade = 0
    let untilAt: number | null = null
    status.until = null
    try {
      if (opts.until) {
        try {
          untilAt = untilTime(opts.until, now())
        } catch (err) {
          throw new Refusal((err as Error).message)
        }
        status.until = opts.until.trim()
      }
      if (!opts.skipCalibration) {
        const gate = calibrationGate(env, run)
        if (!gate.ok) throw new Refusal(gate.why)
      }
      try {
        lock = acquireDriverLock(env, run, now)
      } catch (err) {
        if (err instanceof LockHeld) throw new Refusal(err.message)
        throw err
      }
      await preflight()
      if (pauseAsked) {
        set('paused', 'Paused before anything was sent.')
        return
      }
      const s = refresh()
      sessionStart = now()
      madeAtStart = s.done.size + s.removed.size
      groupsGuess = guessGroups()
      await loop(untilAt)
      if (status.state === 'paused' || status.state === 'error') return
      await finish()
    } catch (err) {
      if (err instanceof Refusal) {
        set('error', err.message)
        log(err.message)
        return
      }
      set('error', `The lab stopped on an error it did not expect: ${(err as Error)?.message ?? err}`)
      log(String((err as Error)?.stack ?? err))
    } finally {
      lock?.release()
      lock = null
    }
  }

  return {
    start(opts = {}) {
      if (running) return running
      running = go(opts).finally(() => {
        running = null
      })
      return running
    },

    async pause() {
      pauseAsked = true
      if (!running) return
      if (state && state.inFlight.size) {
        writePaused('user')
        let snap: Snapshot | null = null
        try {
          snap = await client.snapshot()
        } catch {
          snap = null
        }
        await stopOurs(snap)
      } else if (status.state !== 'reading' && status.state !== 'sealing') {
        writePaused('user')
      }
      if (status.state === 'reading') set('reading', 'Pausing: the picture reader stops after the pictures it is reading now.')
      else if (status.state !== 'sealing') set('paused', 'Pausing: stopping the pictures in progress.')
      wake?.()
    },

    status() {
      return { ...status }
    },
  }
}
