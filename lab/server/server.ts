/**
 * The lab's own server: the lab page, the blind judging routes and the driver
 * that sends a run's work, in one long-lived process (`lab/lab serve`).
 *
 * It listens on 127.0.0.1:5274 by default, an origin of its own, so the app's
 * service worker (which keeps every page on the app's origin as its shell)
 * never touches it. The phone reaches it through `tailscale serve`, which the
 * user runs once by hand (`lab/lab phone` prints the line).
 *
 * The same guards as the app's own servers, from server/guard.mjs:
 * - every request passes hostAllowed (an address, localhost, this machine's
 *   name, any tailnet name, SWITCHGEN_ALLOWED_HOSTS);
 * - every write passes guardMutation (its own pages only, and the body type
 *   the route reads) and a capped read of its body.
 *
 * Blindness is structural. Judging routes read items.json and the events log,
 * which hold blind tokens, letters and neutral wording only. The one function
 * here that opens sealed.json, readSealed, refuses until the study is
 * revealed, and no route calls it before then.
 *
 * Nothing is sent to the app until Start is pressed: the driver is made and
 * started only by the start route.
 */
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { fileURLToPath } from 'node:url'
import { guardMutation, hostAllowed, readBody, reqUrl, safely, send } from '../../server/guard.mjs'
import { RUN_ID, labEnv, runDir, type LabEnv } from '../core/env.ts'
import { appendEvents, appendServerEvent, readEvents, reduceEvents, type JudgeEvent, type JudgeState } from '../judge/events.ts'
import { liveDrivers } from '../run/lock.ts'
import { readDoneCells, readLedger, runState, type RunState } from '../run/ledger.ts'
import { REF_ID, addRef, listRefs, refFile, refIndex, refsDir, setDescribe, setMask } from '../run/refs.ts'
import { readReadings, removeCell, type Reading } from '../run/reader.ts'
import { cleanCopy } from '../run/imagesize.ts'
import { runnerClient } from '../run/runnerClient.ts'
import { calibrationGate as driverGate, canStart, createDriver, type DriverStatus } from '../run/driver.ts'
import { nextItem, remaining, setView, writeSweep, SWEEP_KEY_FILE } from '../judge/queue.ts'
import { aggregate } from '../judge/aggregate.ts'
import { renderReport, toCsv } from '../judge/report.ts'
import { sealRun, type Items, type Sealed } from '../judge/seal.ts'
import { DEFAULT_RUNS, SEED_TIMINGS, measuredTimings, planFrom, savePlan, startRefusal, suiteById, type Plan } from '../core/plan.ts'
import type { DoneCell, Suite } from '../core/types.ts'
import { SHIPPED_SUITES, expand } from '../core/cells.ts'
import { descriptionProblem } from '../core/suite.ts'

/** The smoke run's study (lab/lab smoke): its own, with no calibration night to wait for. */
export const SMOKE_STUDY = 'smoke'

/** The shipped suites, by id: calibration, first-pass-core, ext-edit, ext-range. */
export const SUITES: Readonly<Record<string, Suite>> = Object.freeze(Object.fromEntries(SHIPPED_SUITES.map((s) => [s.id, s])))

// ------------------------------------------------------------------ types --

type Req = http.IncomingMessage
type Res = http.ServerResponse
type Next = () => void

/** What the server needs of a driver: the contract's three calls. */
export type LabDriver = {
  start(opts: { until?: string; skipCalibration?: boolean }): Promise<void>
  pause(): Promise<void>
  status(): DriverStatus
}

export type LabCtx = {
  env: LabEnv
  /** Names allowed besides addresses and localhost. Default: this machine's name, any tailnet name, SWITCHGEN_ALLOWED_HOSTS. */
  allowedHosts?: string[]
  /** Makes the in-process driver for a run. Default: the real driver, talking to the app's runner. Tests pass a stand-in. */
  makeDriver?: (run: string) => LabDriver
  now?: () => number
  log?: (line: string) => void
  /**
   * Runs the blind check for a run and writes its marker (blindMarkerPath).
   * Default: `lab/bin/lab.ts check-blind <run>` in a process of its own, so
   * this server never opens a sealed key before the reveal. Tests pass their own.
   */
  blindCheck?: (run: string) => Promise<void>
  /** No blind gate (the blind check itself drives the handler this way). */
  skipBlindGate?: boolean
  /** Never write: the judging routes answer without fixing tie pairs (the blind check). */
  readOnly?: boolean
}

export type RunRow = {
  run: string
  study: string
  suite: string
  state: 'planned' | 'running' | 'paused' | 'made' | 'sealed' | 'judging' | 'revealed'
  made: number
  total: number
  failed: number
  judged: number
  toJudge: number
  /** Reference photos (or their marked areas) the run needs and does not have, by id. */
  needsRefs: string[]
  /** The photos it waited for have arrived: Start plans the night again to take them in. */
  replan: boolean
  /** True for the calibration night (the step sweep and sampler check). */
  calibration: boolean
  /** For any other night: whether the calibration pairs are all judged. Null for the calibration itself. */
  gate: 'met' | 'unmet' | null
  live: boolean
}

// ------------------------------------------------------------- constants --

const UI_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'ui')
const UI_FILES: Record<string, string> = {
  'judge.js': 'text/javascript; charset=utf-8',
  'judge.css': 'text/css; charset=utf-8',
  'refs.js': 'text/javascript; charset=utf-8',
}
const CARDS_FILE = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'cards', 'blocks.json')

const REF_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const REF_UPLOAD_MAX = 20 * 1024 * 1024
const EVENTS_BODY_MAX = 512 * 1024
const EVENTS_PER_POST = 500
const SMALL_BODY_MAX = 16 * 1024
/** A blind token: 128 bits, as hex or base64url. */
const TOKEN = /^[A-Za-z0-9_-]{22,64}$/
const JUDGE_NAME = /^[A-Za-z0-9._:@~ -]{1,80}$/
const STUDY_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
/** A 24-hour time, 7:30 or 07:30: the same times the CLI and the driver take. */
const UNTIL = /^([01]?\d|2[0-3]):[0-5]\d$/
const ACTIVE: ReadonlySet<DriverStatus['state']> = new Set(['waiting-runner', 'held', 'sending', 'reading', 'sealing'])

/**
 * The page's policy: its own scripts, styles and pictures only. The page
 * builds every element with DOM calls, so no inline script is needed.
 */
const PAGE_CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' blob: data:",
  "connect-src 'self'",
  "frame-ancestors 'none'",
  "base-uri 'none'",
  "form-action 'none'",
  "object-src 'none'",
].join('; ')

/** The report is one self-contained page: its own inline styles and script, nothing fetched. */
const REPORT_CSP = [
  "default-src 'none'",
  "script-src 'unsafe-inline'",
  "style-src 'unsafe-inline'",
  "img-src data:",
  "frame-ancestors 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join('; ')

export function defaultAllowedHosts(env: NodeJS.ProcessEnv = process.env): string[] {
  return [
    'localhost',
    os.hostname(),
    '.ts.net',
    ...(env.SWITCHGEN_ALLOWED_HOSTS ?? '').split(',').map((h) => h.trim()).filter(Boolean),
  ]
}

// --------------------------------------------------------------- reading --

function readJson<T>(file: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8')) as T
  } catch {
    return null
  }
}

function writeJsonAtomic(file: string, value: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  const tmp = path.join(path.dirname(file), `.tmp-${process.pid}-${Date.now()}-${path.basename(file)}`)
  const fd = fs.openSync(tmp, 'w', 0o600)
  try {
    fs.writeSync(fd, JSON.stringify(value, null, 2) + '\n')
    fs.fsyncSync(fd)
  } finally {
    fs.closeSync(fd)
  }
  fs.renameSync(tmp, file)
}

/** Every run that has a plan, by name. */
export function runIds(env: LabEnv): string[] {
  let names: string[]
  try {
    names = fs.readdirSync(path.join(env.labDir, 'runs'))
  } catch {
    return []
  }
  return names.filter((n) => RUN_ID.test(n) && fs.existsSync(path.join(env.labDir, 'runs', n, 'plan.json'))).sort()
}

export function loadPlan(env: LabEnv, run: string): Plan | null {
  if (!RUN_ID.test(run)) return null
  const p = readJson<Plan>(path.join(runDir(env, run), 'plan.json'))
  return p && p.v === 1 && Array.isArray(p.order) ? p : null
}

export function loadItems(env: LabEnv, run: string): Items | null {
  if (!RUN_ID.test(run)) return null
  const it = readJson<Items>(path.join(runDir(env, run), 'items.json'))
  return it && it.v === 1 && Array.isArray(it.sets) ? it : null
}

const judgingDir = (env: LabEnv, run: string) => path.join(runDir(env, run), 'judging')

export function loadEvents(env: LabEnv, run: string): JudgeEvent[] {
  return readEvents(judgingDir(env, run))
}

/** The run's cells: the ones it sends and the ones it reuses. */
function cellsOfPlan(p: Plan): Set<string> {
  return new Set([...p.order, ...(p.reused ?? [])])
}

export function stateOfRun(env: LabEnv, run: string, p: Plan): RunState {
  return runState(readLedger(runDir(env, run)), p, readDoneCells(env))
}

// --------------------------------------------------------------- studies --

/** A study's own folder: its reveal marker and, after the reveal, its report. */
export const studyDir = (env: LabEnv, study: string) => path.join(env.labDir, 'studies', study)
const revealFile = (env: LabEnv, study: string) => path.join(studyDir(env, study), 'revealed.json')

export type RevealInfo = { v: 1; study: string; at: number; early: boolean; remaining: number }

export function revealOf(env: LabEnv, study: string): RevealInfo | null {
  if (!STUDY_ID.test(study)) return null
  const r = readJson<RevealInfo>(revealFile(env, study))
  return r && r.v === 1 ? r : null
}

export function runsOfStudy(env: LabEnv, study: string): string[] {
  return runIds(env).filter((r) => loadPlan(env, r)?.study === study)
}

/**
 * Why nothing more may be planned or started in this study: it has been
 * revealed, so every answer given in it from now on is stamped afterReveal
 * and left out of the results. Null when the study is not revealed. The
 * smoke study is the exception: a smoke run only checks that the lab works,
 * and a new one after its reveal is still that check.
 */
export function revealedRefusal(env: LabEnv, study: string): string | null {
  if (study === SMOKE_STUDY || !revealOf(env, study)) return null
  return `Study ${study} has been revealed, so anything made for it now would be scored after the reveal and left out of the results.`
}

/** How to start without the calibration night's answers, from the page or the command line. */
export const SKIP_CALIBRATION_HINT = 'To start anyway, tick “Start without the calibration night’s answers” on the lab page, or add --skip-calibration to lab/lab start.'

/**
 * The driver's calibration refusal with its own closing hint (“or start with
 * "skip calibration"”) replaced by `hint`, so the way to skip it is said
 * once, in the words of the place the user started from.
 */
export function calibrationRefusal(why: string, hint: string = SKIP_CALIBRATION_HINT): string {
  return `${why.replace(/,?\s*or start with [“"]skip calibration[”"]\.?\s*$/i, '.')} ${hint}`
}

/**
 * The run's sealed key. Refused until the study is revealed: this is the one
 * place sealed.json is opened, and blindness depends on nothing else calling
 * it earlier.
 */
export function readSealed(env: LabEnv, run: string): Sealed {
  const p = loadPlan(env, run)
  if (!p) throw new Error(`There is no run called ${run}.`)
  if (!revealOf(env, p.study)) throw new Error(`Study ${p.study} is not revealed, so its sealed key stays shut.`)
  const s = readJson<Sealed>(path.join(runDir(env, run), 'sealed.json'))
  if (!s) throw new Error(`Run ${run} has no sealed key: it has not been sealed.`)
  return s
}

// ------------------------------------------------------------- the suites --

/** Suite ids a plan was made from. */
const suiteIdsOf = (p: Plan): string[] => (p.suites ?? []).map((s) => s.id)

/** True when the plan holds a calibration suite: one with a step sweep or a sampler check. */
export function isCalibration(p: Plan): boolean {
  return suiteIdsOf(p).some((id) => {
    const s = SUITES[id]
    return !!s && (!!s.sweep || !!s.samplerCheck)
  })
}

/**
 * The photos a suite's tests start from: each ref a slot's source names, and
 * whether a region test needs its marked area. A ref the suite only declares
 * (the shared table) is not needed.
 */
export function refsUsed(s: Suite): { id: string; mask: boolean; describe: string }[] {
  const out = new Map<string, { id: string; mask: boolean; describe: string }>()
  for (const slot of s.slots) {
    const src = slot.source
    if (!src || !('ref' in src)) continue
    const spec = s.refs?.[src.ref]
    const had = out.get(src.ref)
    out.set(src.ref, {
      id: src.ref,
      mask: !!had?.mask || !!spec?.needsMask || slot.op === 'region',
      describe: had?.describe ?? spec?.describe ?? '',
    })
  }
  return [...out.values()]
}

/** The photos a run's tests start from. */
export function refsNeeded(p: Plan, own: readonly Suite[] = []): { id: string; mask: boolean; describe: string }[] {
  const out = new Map<string, { id: string; mask: boolean; describe: string }>()
  for (const id of suiteIdsOf(p)) {
    const s = suiteById(id) ?? own.find((x) => x.id === id)
    if (!s) continue
    for (const r of refsUsed(s)) {
      const had = out.get(r.id)
      out.set(r.id, { id: r.id, mask: r.mask || !!had?.mask, describe: had?.describe || r.describe })
    }
  }
  return [...out.values()]
}

/**
 * The photos a run is still waiting for, by id: those its plan was blocked on
 * that are still missing, or still have no area marked where one is needed.
 * A photo that has arrived since the plan was made is not listed: Start plans
 * the night again (nothing having been sent) and takes it in.
 */
export function missingRefs(env: LabEnv, p: Plan): string[] {
  const have = new Map(listRefs(env).map((r) => [r.id, r]))
  const out = new Set<string>()
  for (const b of p.blocked ?? []) {
    if (!b.ref) continue
    const r = have.get(b.ref)
    if (!r || (/rectangle|mask|area/i.test(b.why) && !r.mask)) out.add(b.ref)
  }
  return [...out]
}

// ---------------------------------------------------------------- planning --

/**
 * runs/<run>/suites.json: a suite the run was planned from that is not one of
 * the shipped ones (the smoke run's), kept so the run can be planned again
 * and sealed without it.
 */
const ownSuitesPath = (env: LabEnv, run: string) => path.join(runDir(env, run), 'suites.json')

export function ownSuites(env: LabEnv, run: string): Suite[] {
  const v = readJson<Suite[]>(ownSuitesPath(env, run))
  return Array.isArray(v) ? v.filter((s) => s && typeof s.id === 'string' && Array.isArray(s.slots)) : []
}

/** The suites a plan was made from: shipped ones by id, others from the run's own suites.json. */
export function suitesOfPlan(env: LabEnv, p: Plan): Suite[] | null {
  const own = ownSuites(env, p.run)
  const out: Suite[] = []
  for (const id of suiteIdsOf(p)) {
    const s = suiteById(id) ?? own.find((x) => x.id === id)
    if (!s) return null
    out.push(s)
  }
  return out.length ? out : null
}

/** Seal a run, finding its suites among the shipped ones and its own. */
export function sealWithSuites(env: LabEnv, run: string, ffmpeg: string, log?: (l: string) => void) {
  return sealRun(env, run, { ffmpeg, suites: [...SHIPPED_SUITES, ...ownSuites(env, run)], ...(log ? { log } : {}) })
}

/**
 * Plan `run` from these suites with the photos and words the lab has now,
 * and write plan.json and each cell's graph. Sends nothing. Refused once the
 * run has sent anything: a started run keeps its plan.
 */
export function planRun(env: LabEnv, run: string, suites: readonly Suite[]): Plan {
  for (const study of new Set(suites.map((s) => s.study))) {
    const shut = revealedRefusal(env, study)
    if (shut) throw new Error(`${shut} Nothing was planned.`)
  }
  if (readLedger(runDir(env, run)).length) {
    throw new Error(`The ${run} run has already started, so its plan stays as it is. Plan a new run under another name.`)
  }
  const x = expand(suites, refIndex(env))
  const done = new Map<string, DoneCell>()
  for (const [id, row] of readDoneCells(env)) if (!row.removed) done.set(id, row)
  const plan = planFrom(run, suites, x, done, measuredTimings(x.cells, done, SEED_TIMINGS))
  const own = suites.filter((s) => !suiteById(s.id))
  if (own.length) writeJsonAtomic(ownSuitesPath(env, run), own)
  savePlan(env, plan, x.graphs)
  return plan
}

/**
 * The plan a run starts from. A run that has sent nothing yet is planned
 * again first, with the photos and words the lab has now, so a photo added
 * since (from the phone or into the refs folder) or a description edited
 * since is part of it. A started run keeps its plan. The lab page's Start,
 * `lab/lab start` and `lab/lab run` all start from this.
 */
export function planForStart(env: LabEnv, run: string, planned: Plan): { plan: Plan; replanned: boolean } {
  if (readLedger(runDir(env, run)).length) return { plan: planned, replanned: false }
  const suites = suitesOfPlan(env, planned)
  if (!suites) return { plan: planned, replanned: false }
  return { plan: planRun(env, run, suites), replanned: true }
}

// --------------------------------------------------------------- judging --

type JudgeProgress = { judged: number; toJudge: number }

/** How far a run's judging has got, for whoever has answered most. */
export function judgingProgress(env: LabEnv, run: string, items: Items | null, now: number): JudgeProgress {
  if (!items) return { judged: 0, toJudge: 0 }
  const state = reduceEvents(loadEvents(env, run))
  const judges = [...state.answers.keys()]
  const judge = judges.sort((a, b) => (state.answers.get(b)?.size ?? 0) - (state.answers.get(a)?.size ?? 0))[0] ?? 'you'
  const n = nextItem(items, state, { judge, session: 'progress', now })
  const left = remaining(items, state, judge, now)
  return { judged: Math.max(0, n.progress.total - left), toJudge: left }
}

/**
 * Whether the calibration night's pairs are judged, for a run of the study.
 * The driver's own rule, so the page and the driver agree: the calibration
 * pairs are judged once sweep.json is written, which the events route does
 * as the last of them arrives.
 */
export function calibrationGate(env: LabEnv, run: string): { met: boolean; why: string | null } {
  const g = driverGate(env, run)
  return g.ok ? { met: true, why: null } : { met: false, why: g.why }
}

// ------------------------------------------------------------------ rows --

/** Every planned run, as the lab page and `lab/lab status` list them. */
export function runRows(env: LabEnv, now: () => number = Date.now): RunRow[] {
  const ctx = { env, now, live: new Map<string, LiveRun>() }
  return runIds(env).map((r) => rowOf(ctx, r)).filter((r): r is RunRow => !!r)
}

function rowOf(ctx: Required<Pick<LabCtx, 'env' | 'now'>> & { live: Map<string, LiveRun> }, run: string): RunRow | null {
  const { env } = ctx
  const p = loadPlan(env, run)
  if (!p) return null
  const st = stateOfRun(env, run, p)
  const total = [...cellsOfPlan(p)].filter((c) => !st.removed.has(c)).length
  const items = loadItems(env, run)
  const prog = judgingProgress(env, run, items, ctx.now())
  const revealed = !!revealOf(env, p.study)
  const inProcess = ctx.live.get(run)
  const liveHere = !!inProcess && ACTIVE.has(inProcess.driver.status().state)
  const liveElsewhere = liveDrivers(env).some((l) => l.run === run && l.pid !== process.pid)
  const live = liveHere || liveElsewhere
  const made = st.done.size
  let state: RunRow['state']
  if (revealed && items) state = 'revealed'
  else if (items) state = prog.judged > 0 ? 'judging' : 'sealed'
  else if (live) state = 'running'
  else if (total > 0 && st.pending.size === 0 && st.inFlight.size === 0) state = 'made'
  else if (readLedger(runDir(env, run)).length > 0) state = 'paused'
  else state = 'planned'
  const calibration = isCalibration(p)
  return {
    run,
    study: p.study,
    suite: suiteIdsOf(p).join(', '),
    state,
    made,
    total,
    failed: st.failed.size,
    judged: prog.judged,
    toJudge: prog.toJudge,
    needsRefs: made >= total && total > 0 ? [] : missingRefs(env, p),
    replan: readLedger(runDir(env, run)).length === 0 && (p.blocked ?? []).length > 0 && missingRefs(env, p).length === 0,
    calibration,
    gate: calibration || p.study === SMOKE_STUDY ? null : calibrationGate(env, run).met ? 'met' : 'unmet',
    live,
  }
}

/** A run's status when no driver in this process is sending it. */
export function restingStatus(env: LabEnv, run: string): DriverStatus | null {
  const p = loadPlan(env, run)
  if (!p) return null
  const st = stateOfRun(env, run, p)
  const total = [...cellsOfPlan(p)].filter((c) => !st.removed.has(c)).length
  const other = liveDrivers(env).find((l) => l.run === run && l.pid !== process.pid)
  let state: DriverStatus['state'] = 'idle'
  let message: string | null = null
  if (other) {
    state = 'sending'
    message = `Being sent by another lab process (pid ${other.pid}). Pause it there, or stop that process.`
  } else if (total > 0 && st.pending.size === 0 && st.inFlight.size === 0) {
    state = 'made'
  } else if (st.paused) {
    state = 'paused'
    message =
      st.pausedWhy === 'until'
        ? 'Stopped at the time you set. Press Start to go on.'
        : st.pausedWhy === 'runner-off'
          ? 'Paused because the app’s runner was off. Press Start to go on.'
          : 'Paused. Press Start to go on.'
  }
  return { run, state, made: st.done.size, total, failed: st.failed.size, etaSeconds: null, until: null, message }
}

// -------------------------------------------------------------- the reveal --

export type RevealCheck = {
  remaining: number
  /** Runs not made and sealed yet, the nights not planned yet included. */
  waiting: string[]
  /** Of those, the shipped nights of the study that have no run yet, by their usual run name. */
  unplanned: string[]
}

/**
 * What is left before a normal reveal: unanswered items, runs not yet
 * sealed, and the study's shipped nights that have no run yet. The reveal
 * covers the whole study, and nothing can be planned in it afterwards, so
 * revealing before any of these is an early reveal.
 */
export function revealCheck(env: LabEnv, study: string, now: number): RevealCheck {
  let remaining = 0
  const waiting: string[] = []
  const planned = new Set<string>()
  for (const run of runsOfStudy(env, study)) {
    const p = loadPlan(env, run)
    if (p) for (const id of suiteIdsOf(p)) planned.add(id)
    const items = loadItems(env, run)
    if (!items) {
      waiting.push(run)
      continue
    }
    remaining += judgingProgress(env, run, items, now).toJudge
  }
  const unplanned = SHIPPED_SUITES.filter((s) => s.study === study && !planned.has(s.id)).map((s) => DEFAULT_RUNS[s.id] ?? `${s.id}-1`)
  return { remaining, waiting: [...waiting, ...unplanned], unplanned }
}

/**
 * Reveal a study: write the marker, log a reveal event in each run's judging
 * log, carry out the judge's own removals, and make each run's report.
 * Everything scored after this is stamped afterReveal and left out.
 */
export function revealStudy(env: LabEnv, study: string, early: boolean, now: number, log: (l: string) => void = () => {}): RevealInfo {
  const check = revealCheck(env, study, now)
  const info: RevealInfo = { v: 1, study, at: now, early, remaining: check.remaining + check.waiting.length }
  writeJsonAtomic(revealFile(env, study), info)
  for (const run of runsOfStudy(env, study)) {
    if (!loadItems(env, run)) continue
    appendServerEvent(judgingDir(env, run), {
      v: 1,
      id: `reveal-${randomUUID()}`,
      at: now,
      judge: 'server',
      session: 'server',
      device: { w: 0, h: 0, dpr: 0 },
      run,
      item: study,
      kind: 'reveal',
      value: { early, remaining: info.remaining },
      dwellMs: 0,
    })
  }
  for (const run of runsOfStudy(env, study)) {
    if (!loadItems(env, run)) continue
    try {
      applyJudgeRemovals(env, run, now)
    } catch (err) {
      log(`Could not finish the judge's removals for ${run}: ${(err as Error).message}`)
    }
  }
  try {
    writeReport(env, study)
  } catch (err) {
    log(`Could not make the report for study ${study}: ${(err as Error).message}`)
  }
  return info
}

/**
 * The study's report, over every sealed run of it at once: findings.json,
 * report.html and cells.csv in <lab>/studies/<study>/. Only after the reveal.
 */
export function writeReport(env: LabEnv, study: string): { findings: string; report: string; csv: string; runs: string[] } {
  const reveal = revealOf(env, study)
  if (!reveal) throw new Error(`Study ${study} is not revealed yet, so it has no report.`)
  const runs = runsOfStudy(env, study).filter((r) => !!loadItems(env, r))
  if (!runs.length) throw new Error(`Study ${study} has no sealed run to report on.`)
  const sealed = runs.map((r) => readSealed(env, r))
  const dir = studyDir(env, study)
  // The picture reader's results for the study's pictures are kept with the
  // study the first time a report is made: prune forgets a deleted picture's
  // reading, and a picture made again later is read again, so a report made
  // after that still has the readings of the pictures that were judged.
  const keptReadings = path.join(dir, 'readings.json')
  const all = readReadings(env)
  const readings = new Map<string, Reading>()
  for (const s of sealed) for (const id of Object.keys(s.cells ?? {})) if (all.has(id)) readings.set(id, all.get(id) as Reading)
  const kept = readJson<unknown>(keptReadings)
  for (const r of Array.isArray(kept) ? (kept as Reading[]) : []) if (r && typeof r.cellId === 'string') readings.set(r.cellId, r)
  writeJsonAtomic(keptReadings, [...readings.values()])
  const f = aggregate({
    items: runs.map((r) => loadItems(env, r) as Items),
    sealed,
    events: runs.flatMap((r) => loadEvents(env, r)),
    readings: [...readings.values()],
    ledger: runs.flatMap((r) => readLedger(runDir(env, r))),
    revealedAt: reveal.at,
  })
  const out = {
    findings: path.join(dir, 'findings.json'),
    report: path.join(dir, 'report.html'),
    csv: path.join(dir, 'cells.csv'),
    runs,
  }
  writeJsonAtomic(out.findings, f)
  fs.writeFileSync(out.report, renderReport(f), { mode: 0o600 })
  fs.writeFileSync(out.csv, toCsv(f), { mode: 0o600 })
  return out
}

// ------------------------------------------------------------ blind gate --

/** runs/<run>/blind-check.json: what `lab/lab check-blind` found, for which items.json. */
export const blindMarkerPath = (env: LabEnv, run: string) => path.join(runDir(env, run), 'blind-check.json')

export type BlindMarker = { v: 1; run: string; at: number; itemsSha: string; ok: boolean; checked: number; problems: number }

/** The sha256 of items.json as it is on disk, or null when there is none. */
export function itemsSha(env: LabEnv, run: string): string | null {
  try {
    return createHash('sha256').update(fs.readFileSync(path.join(runDir(env, run), 'items.json'))).digest('hex')
  } catch {
    return null
  }
}

/** The blind check's verdict on the run's current items.json, or null when it has not run on them. */
export function blindVerdict(env: LabEnv, run: string): BlindMarker | null {
  const m = readJson<BlindMarker>(blindMarkerPath(env, run))
  const sha = itemsSha(env, run)
  return m && m.v === 1 && sha && m.itemsSha === sha ? m : null
}

export function writeBlindMarker(env: LabEnv, run: string, r: { ok: boolean; checked: number; problems: number }, now = Date.now()): BlindMarker {
  const m: BlindMarker = { v: 1, run, at: now, itemsSha: itemsSha(env, run) ?? '', ok: r.ok, checked: r.checked, problems: r.problems }
  writeJsonAtomic(blindMarkerPath(env, run), m)
  return m
}

/** The default blind check: the CLI's check-blind, in its own process. */
function spawnBlindCheck(env: LabEnv, run: string, log: (l: string) => void): Promise<void> {
  const cli = path.join(env.repoRoot, 'lab', 'bin', 'lab.ts')
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ['--import', 'tsx', cli, 'check-blind', run], {
      cwd: env.repoRoot,
      env: { ...process.env, SWITCHGEN_LAB_DIR: env.labDir, SWITCHGEN_OUTPUTS: env.outputs },
      stdio: ['ignore', 'ignore', 'pipe'],
    })
    let err = ''
    child.stderr?.on('data', (d: Buffer) => {
      err += d.toString()
    })
    child.on('error', (e) => {
      log(`The blind check for ${run} could not start: ${e.message}`)
      resolve()
    })
    child.on('close', (code) => {
      if (code !== 0 && code !== 1) log(`The blind check for ${run} stopped (${code}): ${err.trim().slice(0, 300)}`)
      resolve()
    })
  })
}

// ------------------------------------------------------ judge's removals --

const removalsFile = (env: LabEnv, run: string) => path.join(judgingDir(env, run), 'removals.jsonl')

type Removal = { at: number; token: string; item: string; event: string }

/**
 * How long a judge's removal waits before its blind copies are deleted. The
 * phone offers Undo for 4 s after the score is sent, and the undo then has
 * to reach the server; until this has passed the copies stay, so an Undo
 * brings the item back whole.
 */
export const REMOVAL_UNDO_MS = 30_000

/**
 * The defaults card's rule: "looks under 18" together with "sexualised"
 * removes those pictures, as the quarantine rule does. The removal is written
 * down as the score arrives. Its blind copies go once the undo window has
 * closed with the score still standing (settleJudgeRemovals); the originals
 * go at the reveal, when the token can be matched to its cell without
 * opening the key early.
 */
function recordJudgeRemovals(env: LabEnv, run: string, items: Items, events: readonly JudgeEvent[], now: number): number {
  let n = 0
  for (const e of events) {
    if (e.kind !== 'score') continue
    const d = (e.value as { defaults?: { who?: unknown; extra?: unknown; flag?: unknown } } | undefined)?.defaults
    if (!d || d.who !== 'looks under 18' || !Array.isArray(d.extra) || !d.extra.includes('sexualised')) continue
    const grid = gridById(items, e.item)
    if (!grid) continue
    const tiles = gridTokens(grid)
    const flag = Array.isArray(d.flag) ? d.flag.filter((i): i is number => Number.isInteger(i)) : tiles.map((_, i) => i)
    for (const i of flag) {
      for (const token of tiles[i] ?? []) {
        if (!token || !TOKEN.test(token)) continue
        fs.mkdirSync(judgingDir(env, run), { recursive: true })
        fs.appendFileSync(removalsFile(env, run), JSON.stringify({ at: now, token, item: e.item, event: e.id } satisfies Removal) + '\n')
        n++
      }
    }
  }
  return n
}

function readRemovals(env: LabEnv, run: string): Removal[] {
  let text: string
  try {
    text = fs.readFileSync(removalsFile(env, run), 'utf8')
  } catch {
    return []
  }
  return text
    .split('\n')
    .filter(Boolean)
    .map((l) => {
      try {
        return JSON.parse(l) as Removal
      } catch {
        return null
      }
    })
    .filter((r): r is Removal => !!r && typeof r.token === 'string' && TOKEN.test(r.token) && typeof r.event === 'string')
}

/**
 * The removals whose score still stands: it is in the log, no undo names it,
 * and it is still its judge's latest answer to the item.
 */
function standingRemovals(env: LabEnv, run: string): Removal[] {
  const rows = readRemovals(env, run)
  if (!rows.length) return []
  const state = reduceEvents(loadEvents(env, run))
  const byId = new Map(state.events.map((e) => [e.id, e]))
  return rows.filter((r) => {
    const e = byId.get(r.event)
    if (!e || state.undone.has(e.id)) return false
    return state.answers.get(e.judge)?.get(e.item)?.event.id === e.id
  })
}

function deleteBlindCopies(env: LabEnv, run: string, token: string): number {
  let n = 0
  for (const size of ['g', 'f']) {
    try {
      fs.unlinkSync(path.join(runDir(env, run), 'view', `${token}-${size}.webp`))
      n++
    } catch {
      /* already gone */
    }
  }
  return n
}

/**
 * Delete the blind copies of every removal whose undo window has closed with
 * its score still standing. An undone removal keeps its copies. Returns how
 * many files went.
 */
export function settleJudgeRemovals(env: LabEnv, run: string, now: number): number {
  let n = 0
  for (const r of standingRemovals(env, run)) if (now - r.at >= REMOVAL_UNDO_MS) n += deleteBlindCopies(env, run, r.token)
  return n
}

/** At the reveal: delete the pictures of the removals still standing, originals too, and mark their cells removed. */
function applyJudgeRemovals(env: LabEnv, run: string, now: number): number {
  const rows = standingRemovals(env, run)
  if (!rows.length) return 0
  const sealed = readSealed(env, run)
  const done = new Set<string>()
  for (const r of rows) {
    deleteBlindCopies(env, run, r.token)
    const cell = sealed.tokens[r.token]?.cellId
    if (!cell || done.has(cell)) continue
    removeCell(env, run, cell, () => now)
    done.add(cell)
  }
  return done.size
}

// ----------------------------------------------------------- items shape --

type Grid = Items['sets'][number]['grids'][number]

function gridById(items: Items, itemId: string): Grid | null {
  for (const s of items.sets) for (const g of s.grids) if (g.itemId === itemId) return g
  return null
}

/** Per position, the tokens shown there (one, or before and after). */
function gridTokens(g: Grid): (string | null)[][] {
  if (Array.isArray(g.rows)) return g.rows.map((r) => [...r])
  return (g.tiles ?? []).map((t) => [t])
}

/** Every id an event may name in this run: items, sets and the virtual pick and tie items. */
function knownIds(items: Items): Set<string> {
  const ids = new Set<string>()
  for (const s of items.sets) {
    ids.add(s.setId)
    ids.add(`${s.setId}:pick`)
    for (let k = 0; k < 6; k++) ids.add(`${s.setId}:tie${k}`)
    for (const g of s.grids) ids.add(g.itemId)
  }
  for (const p of items.pairs ?? []) ids.add(p.itemId)
  for (const c of items.checks ?? []) ids.add(c.itemId)
  return ids
}

// ------------------------------------------------------------- plumbing --

function secure(res: Res): void {
  res.setHeader('X-Content-Type-Options', 'nosniff')
  res.setHeader('Referrer-Policy', 'no-referrer')
  res.setHeader('X-Frame-Options', 'DENY')
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin')
}

function json(res: Res, code: number, body: unknown, cache = 'no-store'): void {
  res.setHeader('Cache-Control', cache)
  send(res, code, body)
}

function fail(res: Res, code: number, error: string, extra: Record<string, unknown> = {}): void {
  json(res, code, { error, ...extra })
}

/** Send a file's bytes. Never with Content-Disposition: the page shows them, nothing is offered as a download. */
function sendFile(res: Res, file: string, type: string, cache: string, extra: Record<string, string> = {}): boolean {
  let bytes: Buffer
  try {
    const st = fs.lstatSync(file)
    if (!st.isFile()) return false
    bytes = fs.readFileSync(file)
  } catch {
    return false
  }
  sendBytes(res, bytes, type, cache, extra)
  return true
}

function sendBytes(res: Res, bytes: Buffer, type: string, cache: string, extra: Record<string, string> = {}): void {
  res.statusCode = 200
  res.setHeader('Content-Type', type)
  res.setHeader('Content-Length', bytes.length)
  res.setHeader('Cache-Control', cache)
  for (const [k, v] of Object.entries(extra)) res.setHeader(k, v)
  res.end(bytes)
}

/** Read a raw body up to `limit` bytes; null when it is larger. */
async function readRaw(req: Req, limit: number): Promise<Buffer | null> {
  const declared = Number(req.headers['content-length'])
  if (Number.isFinite(declared) && declared > limit) return null
  const chunks: Buffer[] = []
  let size = 0
  for await (const c of req as AsyncIterable<Buffer | string>) {
    const b = typeof c === 'string' ? Buffer.from(c) : c
    size += b.length
    if (size > limit) return null
    chunks.push(b)
  }
  return Buffer.concat(chunks)
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

// ---------------------------------------------------------- the handler --

type LiveRun = { run: string; driver: LabDriver; error: string | null; startedAt: number }

export type LabHandler = ((req: Req, res: Res, next: Next) => void) & {
  /** Pause every run this process is sending (the server is closing). */
  pauseAll(): Promise<void>
}

/**
 * The lab's routes as one Connect-style handler, so tests can drive it
 * without a socket. A path it does not serve is passed to `next`.
 */
export function createLabHandler(input: LabCtx): LabHandler {
  const env = input.env
  const allowed = input.allowedHosts ?? defaultAllowedHosts()
  const now = input.now ?? Date.now
  const log = input.log ?? ((line: string) => console.log(line))
  const makeDriver =
    input.makeDriver ??
    ((run: string): LabDriver =>
      createDriver({
        env,
        run,
        client: runnerClient(env.appUrl),
        log: (l: string) => log(`[${run}] ${l}`),
        seal: (e: LabEnv, r: string) => sealWithSuites(e, r, process.env.SWITCHGEN_FFMPEG ?? 'ffmpeg'),
      }))
  const live = new Map<string, LiveRun>()
  const ctx = { env, now, live }
  const blindCheck = input.blindCheck ?? ((run: string) => spawnBlindCheck(env, run, log))
  const checking = new Map<string, Promise<void>>()

  /**
   * Judging opens only once the blind check has passed on this items.json.
   * Answers the request itself (409, saying why) and returns false until then.
   */
  const blindOpen = (res: Res, run: string): boolean => {
    if (input.skipBlindGate) return true
    const v = blindVerdict(env, run)
    if (v?.ok) return true
    if (v && !v.ok) {
      fail(res, 409, `The blind check found ${v.problems} problem${v.problems === 1 ? '' : 's'} in what this run would show you, so judging stays closed. Run lab/lab check-blind ${run} on the machine to see where (it never prints what leaked).`, { blindFailed: true })
      return false
    }
    if (!checking.has(run)) {
      log(`${run}: checking that nothing the judge sees gives a model away.`)
      checking.set(run, blindCheck(run).finally(() => checking.delete(run)))
    }
    fail(res, 409, 'Checking that nothing on the judging pages gives a model away. This takes a few seconds.', { checking: true })
    return false
  }

  /** The run this process is sending, if any. */
  const activeHere = (): LiveRun | null => {
    for (const l of live.values()) if (ACTIVE.has(l.driver.status().state)) return l
    return null
  }

  /** Carry out the judge's removals whose undo window has closed. */
  const settle = (run: string): void => {
    try {
      settleJudgeRemovals(env, run, now())
    } catch (err) {
      log(`Could not finish a removal in ${run}: ${(err as Error).message}`)
    }
  }

  async function handle(req: Req, res: Res, next: Next): Promise<void> {
    secure(res)
    if (!hostAllowed(req.headers.host, allowed)) return fail(res, 403, 'This lab server does not answer to that host name.')
    const url = reqUrl(req)
    if (!url) return fail(res, 400, 'That address could not be read.')
    const p = url.pathname
    const method = req.method ?? 'GET'
    if (method !== 'GET' && method !== 'HEAD' && method !== 'POST') return fail(res, 405, 'Only GET and POST are served here.')
    if (method === 'POST' && !p.startsWith('/api/lab/')) return fail(res, 405, 'Nothing here takes a POST.')

    // ------------------------------------------------------------ page --
    if (method !== 'POST' && (p === '/' || p === '/index.html')) {
      if (sendFile(res, path.join(UI_DIR, 'index.html'), 'text/html; charset=utf-8', 'no-cache', { 'Content-Security-Policy': PAGE_CSP })) return
      return fail(res, 500, 'The lab page is missing from lab/ui.')
    }
    let m = /^\/ui\/([^/]+)$/.exec(p)
    if (m && method !== 'POST') {
      const type = Object.hasOwn(UI_FILES, m[1]) ? UI_FILES[m[1]] : undefined
      if (type && sendFile(res, path.join(UI_DIR, m[1]), type, 'no-cache')) return
      return fail(res, 404, 'No such file.')
    }

    // ---------------------------------------------------------- report --
    m = /^\/report\/([^/]+)$/.exec(p)
    if (m && method !== 'POST') return reportPage(res, decodeURIComponent(m[1]))

    if (!p.startsWith('/api/lab/')) return next()
    const route = p.slice('/api/lab'.length)

    if (method === 'POST' && route !== '/refs') {
      if (!guardMutation(req, res)) return
    }

    // ---------------------------------------------------------- health --
    if (route === '/health' && method !== 'POST') return json(res, 200, { server: 'switchgen-lab' })
    if (route === '/cards' && method !== 'POST') {
      const cards = readJson<unknown[]>(CARDS_FILE)
      return cards ? json(res, 200, cards, 'no-cache') : fail(res, 500, 'The scorecards file lab/cards/blocks.json is missing.')
    }

    // ---------------------------------------------------------- nights --
    if (route === '/suites' && method !== 'POST') return json(res, 200, suitesView())
    if (route === '/plan' && method === 'POST') return planRoute(req, res)

    // ------------------------------------------------------------ runs --
    if (route === '/runs' && method !== 'POST') {
      const rows = runIds(env).map((r) => rowOf(ctx, r)).filter((r): r is RunRow => !!r)
      return json(res, 200, rows)
    }
    m = /^\/runs\/([^/]+)\/(status|start|pause|next|events)$/.exec(route)
    if (m) {
      const run = decodeURIComponent(m[1])
      const plan = loadPlan(env, run)
      if (!plan) return fail(res, 404, `There is no run called ${run}.`)
      const what = m[2]
      if (what === 'status' && method !== 'POST') {
        const l = live.get(run)
        const s = l ? l.driver.status() : restingStatus(env, run)
        if (l && l.error && s && s.state !== 'error' && !ACTIVE.has(s.state)) return json(res, 200, { ...s, message: s.message ?? l.error })
        return json(res, 200, s)
      }
      if (what === 'start' && method === 'POST') return start(req, res, run, plan)
      if (what === 'pause' && method === 'POST') return pause(req, res, run)
      if (what === 'next' && method !== 'POST') return nextRoute(res, url, run)
      if (what === 'events' && method === 'POST') return eventsRoute(req, res, run, plan)
      return fail(res, 405, 'Not with that method.')
    }
    m = /^\/runs\/([^/]+)\/sets\/([^/]+)$/.exec(route)
    if (m && method !== 'POST') {
      const run = decodeURIComponent(m[1])
      const items = loadItems(env, run)
      if (!items) return fail(res, 404, `Run ${run} is not ready to score yet.`)
      if (!blindOpen(res, run)) return
      const set = items.sets.find((s) => s.setId === decodeURIComponent(m![2]))
      if (!set) return fail(res, 404, 'No such set.')
      return json(res, 200, setView(set))
    }

    // ---------------------------------------------------------- images --
    m = /^\/img\/([A-Za-z0-9_-]+)-(g|f)\.webp$/.exec(route)
    if (m && method !== 'POST') {
      const [, token, size] = m
      if (!TOKEN.test(token)) return fail(res, 404, 'No such picture.')
      for (const run of runIds(env)) {
        const file = path.join(runDir(env, run), 'view', `${token}-${size}.webp`)
        if (sendFile(res, file, 'image/webp', 'private, max-age=31536000, immutable')) return
      }
      return fail(res, 404, 'No such picture.')
    }

    // ------------------------------------------------------------ refs --
    if (route === '/refs' && method !== 'POST') return json(res, 200, listRefs(env))
    if (route === '/refs' && method === 'POST') return uploadRef(req, res)
    if (route === '/refs/needed' && method !== 'POST') return json(res, 200, refsNeededView())
    m = /^\/refs\/([^/]+)\/(image|mask\.png|mask|describe)$/.exec(route)
    if (m) {
      const id = decodeURIComponent(m[1])
      if (!REF_ID.test(id)) return fail(res, 404, 'No such photo.')
      const what = m[2]
      if (what === 'image' && method !== 'POST') {
        const f = refFile(env, id)
        if (!f) return fail(res, 404, 'No such photo.')
        // The copy without the camera's metadata (place, time, camera), as
        // ComfyUI gets it: the page is reached over the tailnet. The
        // orientation flag stays, so it still shows upright.
        let bytes: Buffer
        try {
          if (!fs.lstatSync(f.path).isFile()) return fail(res, 404, 'No such photo.')
          bytes = cleanCopy(fs.readFileSync(f.path))
        } catch {
          return fail(res, 404, 'No such photo.')
        }
        const current = url.searchParams.get('v') === f.info.sha12
        return sendBytes(res, bytes, f.mime, current ? 'private, max-age=31536000, immutable' : 'private, no-cache')
      }
      if (what === 'mask.png' && method !== 'POST') {
        if (sendFile(res, path.join(refsDir(env), `${id}.mask.png`), 'image/png', 'private, no-cache')) return
        return fail(res, 404, 'This photo has no marked area.')
      }
      if (what === 'mask' && method === 'POST') {
        const body = await readBody(req, SMALL_BODY_MAX)
        const r = isObj(body) && isObj(body.rect) ? body.rect : null
        const nums = r ? [r.x, r.y, r.w, r.h] : []
        if (!r || !nums.every((n) => typeof n === 'number' && Number.isFinite(n))) return fail(res, 400, 'Send {rect:{x,y,w,h}} in the photo’s pixels.')
        try {
          return json(res, 200, setMask(env, id, { x: r.x as number, y: r.y as number, w: r.w as number, h: r.h as number }))
        } catch (err) {
          return fail(res, 400, (err as Error).message)
        }
      }
      if (what === 'describe' && method === 'POST') {
        const body = await readBody(req, SMALL_BODY_MAX)
        const text = isObj(body) ? body.describe : undefined
        if (!(typeof text === 'string' || text === null)) return fail(res, 400, 'Send {describe:"…"}, or null to go back to the suite’s words.')
        // Refused now, not only when a night is planned or started, so the
        // page never says a description it cannot use was saved.
        const problem = text === null ? null : descriptionProblem(id, text)
        if (problem) return fail(res, 400, problem)
        try {
          return json(res, 200, setDescribe(env, id, text))
        } catch (err) {
          return fail(res, 400, (err as Error).message)
        }
      }
      return fail(res, 405, 'Not with that method.')
    }

    // --------------------------------------------------------- studies --
    m = /^\/studies\/([^/]+)\/(sweep|reveal|findings\.json)$/.exec(route)
    if (m) {
      const study = decodeURIComponent(m[1])
      if (!STUDY_ID.test(study) || !runsOfStudy(env, study).length) return fail(res, 404, `There is no study called ${study}.`)
      if (m[2] === 'sweep' && method !== 'POST') return sweepRoute(res, study)
      if (m[2] === 'reveal' && method === 'POST') return revealRoute(req, res, study)
      if (m[2] === 'findings.json' && method !== 'POST') return findingsRoute(res, study)
      return fail(res, 405, 'Not with that method.')
    }

    return fail(res, 404, 'No such route.')
  }

  // --------------------------------------------------------------- planning --

  function suitesView() {
    const plans = runIds(env).map((r) => loadPlan(env, r)).filter((p): p is Plan => !!p)
    return SHIPPED_SUITES.map((s) => ({
      id: s.id,
      study: s.study,
      version: s.version,
      defaultRun: DEFAULT_RUNS[s.id] ?? `${s.id}-1`,
      runs: plans.filter((p) => suiteIdsOf(p).includes(s.id)).map((p) => p.run),
    }))
  }

  /** Plan a shipped night from the page. Sends nothing. */
  async function planRoute(req: Req, res: Res): Promise<void> {
    const body = await readBody(req, SMALL_BODY_MAX)
    if (!isObj(body) || typeof body.suite !== 'string') return fail(res, 400, 'Send {suite, run?}.')
    const suite = suiteById(body.suite)
    if (!suite) return fail(res, 404, `There is no suite called ${body.suite}.`)
    const run = typeof body.run === 'string' && body.run ? body.run : DEFAULT_RUNS[suite.id] ?? `${suite.id}-1`
    if (!RUN_ID.test(run)) return fail(res, 400, `"${run}" is not a run name. Use letters, digits and dashes.`)
    if (activeHere()?.run === run) return fail(res, 409, `The ${run} run is being made now.`)
    let plan: Plan
    try {
      plan = planRun(env, run, [suite])
    } catch (err) {
      return fail(res, 409, (err as Error).message)
    }
    log(`Planned ${run} from ${suite.id}: ${plan.estimate.newPictures} pictures to make.`)
    return json(res, 200, {
      run,
      study: plan.study,
      pictures: plan.estimate.pictures,
      newPictures: plan.estimate.newPictures,
      reused: plan.reused.length,
      seconds: plan.estimate.seconds,
      na: plan.na.length,
      refusal: startRefusal(plan, env),
    })
  }

  // ---------------------------------------------------------- start/pause --

  async function start(req: Req, res: Res, run: string, planned: Plan): Promise<void> {
    const body = await readBody(req, SMALL_BODY_MAX)
    if (!isObj(body) || body.confirm !== 'start') return fail(res, 400, 'Starting needs {confirm:"start"}, so nothing starts by accident.')
    const until = typeof body.until === 'string' ? body.until.trim() : body.until
    if (until !== undefined && until !== null && until !== '' && !(typeof until === 'string' && UNTIL.test(until))) {
      return fail(res, 400, 'until must be a 24-hour time like 07:30.')
    }
    const shut = revealedRefusal(env, planned.study)
    if (shut) return fail(res, 409, `${shut} ${run} was not started.`)
    // The smoke run has no calibration night of its own to wait for.
    const skipCalibration = body.skipCalibration === true || planned.study === SMOKE_STUDY
    const here = activeHere()
    if (here && here.run !== run) return fail(res, 409, `The ${here.run} run is being made now. Pause it before starting another.`)
    if (here && here.run === run) return fail(res, 409, `The ${run} run is already being sent.`)
    const elsewhere = liveDrivers(env).find((l) => l.pid !== process.pid)
    if (elsewhere) return fail(res, 409, `The ${elsewhere.run} run is being sent by another lab process (pid ${elsewhere.pid}). Pause it there first.`)
    let plan: Plan
    try {
      const p = planForStart(env, run, planned)
      plan = p.plan
      if (p.replanned) log(`${run}: planned again before starting (${plan.estimate.newPictures} pictures to make).`)
    } catch (err) {
      return fail(res, 409, `Could not plan ${run} again before starting: ${(err as Error).message}`)
    }
    const refusal = startRefusal(plan, env)
    if (refusal) return fail(res, 409, refusal, { needsRefs: missingRefs(env, plan) })
    const may = canStart(env, run, { skipCalibration })
    if (!may.ok) {
      const gateOnly = !skipCalibration && !driverGate(env, run).ok
      return fail(res, 409, gateOnly ? calibrationRefusal(may.why) : may.why)
    }
    let l = live.get(run)
    if (!l) {
      l = { run, driver: makeDriver(run), error: null, startedAt: now() }
      live.set(run, l)
    }
    const entry = l
    entry.error = null
    entry.startedAt = now()
    const opts: { until?: string; skipCalibration?: boolean } = {}
    if (typeof until === 'string' && until) opts.until = until
    if (skipCalibration) opts.skipCalibration = true
    log(`Start pressed for ${run}${opts.until ? `, until ${opts.until}` : ''}.`)
    entry.driver.start(opts).then(
      () => log(`${run}: the driver finished.`),
      (err: unknown) => {
        entry.error = (err as Error)?.message ?? String(err)
        log(`${run}: the driver stopped: ${entry.error}`)
      },
    )
    // Let a refusal that comes at once (the runner is off, the lock is taken) show in the answer.
    await new Promise((r) => setImmediate(r))
    return json(res, 202, { run, status: entry.driver.status() })
  }

  async function pause(req: Req, res: Res, run: string): Promise<void> {
    await readBody(req, SMALL_BODY_MAX)
    const l = live.get(run)
    if (l) {
      try {
        await l.driver.pause()
      } catch (err) {
        return fail(res, 500, `Could not pause: ${(err as Error).message}`)
      }
      return json(res, 200, { run, status: l.driver.status() })
    }
    const other = liveDrivers(env).find((x) => x.run === run && x.pid !== process.pid)
    if (other) {
      // `lab/lab run` turns an interrupt into a pause: it stops the group in progress and writes it to the ledger.
      try {
        process.kill(other.pid, 'SIGINT')
      } catch (err) {
        return fail(res, 500, `Could not reach the lab process (pid ${other.pid}) sending this run: ${(err as Error).message}`)
      }
      return json(res, 200, { run, status: restingStatus(env, run), message: `Asked the lab process (pid ${other.pid}) sending this run to pause.` })
    }
    return json(res, 200, { run, status: restingStatus(env, run), message: 'This run is not being sent.' })
  }

  // -------------------------------------------------------------- judging --

  function nextRoute(res: Res, url: URL, run: string): void {
    const items = loadItems(env, run)
    if (!items) return fail(res, 409, `Run ${run} is not ready to score: its pictures are not all made and sealed yet.`)
    if (!blindOpen(res, run)) return
    const judge = url.searchParams.get('judge') || 'you'
    const session = url.searchParams.get('session') || 'page'
    if (!JUDGE_NAME.test(judge) || !JUDGE_NAME.test(session)) return fail(res, 400, 'judge and session must be short plain names.')
    if (!input.readOnly) settle(run)
    const state: JudgeState = reduceEvents(loadEvents(env, run))
    const answer = nextItem(items, state, { judge, session, now: now() })
    // A set's tie-break pairs are fixed the first time they are worked out, so
    // a score changed later cannot change which pairs were asked.
    if (answer.record.length && !input.readOnly) appendEvents(judgingDir(env, run), answer.record, { allowServerKinds: true })
    const { record: _record, ...phone } = answer
    void _record
    return json(res, 200, phone)
  }

  async function eventsRoute(req: Req, res: Res, run: string, plan: Plan): Promise<void> {
    const body = await readBody(req, EVENTS_BODY_MAX)
    if (body === null) return fail(res, 413, 'Too many events in one go, or not JSON.')
    if (!isObj(body) || !Array.isArray(body.events)) return fail(res, 400, 'Send {events:[…]}.')
    if (body.events.length > EVENTS_PER_POST) return fail(res, 413, `At most ${EVENTS_PER_POST} events in one go.`)
    const items = loadItems(env, run)
    if (!items) return fail(res, 409, `Run ${run} is not ready to score yet.`)
    const ids = knownIds(items)
    const rejected: { index: number; error: string }[] = []
    const keep: unknown[] = []
    const index: number[] = []
    body.events.forEach((e, i) => {
      if (!isObj(e)) return rejected.push({ index: i, error: 'not an object' })
      if (e.run !== run) return rejected.push({ index: i, error: `the event is for run ${String(e.run)}, not ${run}` })
      if (typeof e.item !== 'string' || !ids.has(e.item)) return rejected.push({ index: i, error: 'no such item in this run' })
      keep.push(e)
      index.push(i)
    })
    const revealed = !!revealOf(env, plan.study)
    const r = appendEvents(judgingDir(env, run), keep, { afterReveal: revealed })
    for (const x of r.rejected) rejected.push({ index: index[x.index], error: x.error })
    if (r.accepted && fs.existsSync(path.join(runDir(env, run), SWEEP_KEY_FILE))) {
      try {
        writeSweep(env, run)
      } catch (err) {
        log(`Could not work out the sweep for ${run}: ${(err as Error).message}`)
      }
    }
    if (r.accepted) {
      try {
        const n = recordJudgeRemovals(env, run, items, keep.filter((e) => isObj(e) && e.kind === 'score') as JudgeEvent[], now())
        // The blind copies go once the undo window has closed, if the score still stands then.
        if (n) setTimeout(() => settle(run), REMOVAL_UNDO_MS + 1000).unref()
      } catch (err) {
        log(`Could not record a removal in ${run}: ${(err as Error).message}`)
      }
      settle(run)
    }
    return json(res, 200, { accepted: r.accepted, duplicate: r.duplicate, ...(rejected.length ? { rejected } : {}) })
  }

  // --------------------------------------------------------------- studies --

  function sweepRoute(res: Res, study: string): void {
    for (const run of runsOfStudy(env, study)) {
      const p = loadPlan(env, run)
      if (!p || !isCalibration(p)) continue
      const sweep = readJson<unknown>(path.join(runDir(env, run), 'sweep.json'))
      if (sweep) return json(res, 200, sweep)
    }
    return fail(res, 404, 'The sweep result is not ready: its pairs are not all judged yet.')
  }

  async function revealRoute(req: Req, res: Res, study: string): Promise<void> {
    const body = await readBody(req, SMALL_BODY_MAX)
    if (body === null || !isObj(body)) return fail(res, 400, 'Send {} or {confirm:"reveal early"}.')
    const had = revealOf(env, study)
    if (had) return json(res, 200, had)
    // The driver does not look for a reveal while it sends, so a run still
    // being made would go on and everything it made would be left out.
    const making = new Set([activeHere()?.run, ...liveDrivers(env).map((l) => l.run)])
    const busy = runsOfStudy(env, study).find((r) => making.has(r))
    if (busy) return fail(res, 409, `The ${busy} run is being made now. Pause it first: the reveal is final, and anything made for the study after it is left out of the results.`)
    const check = revealCheck(env, study, now())
    const early = body.confirm === 'reveal early'
    if ((check.remaining > 0 || check.waiting.length > 0) && !early) {
      const notSealed = check.waiting.filter((r) => !check.unplanned.includes(r))
      const words = [
        check.remaining ? `${check.remaining} item${check.remaining === 1 ? ' is' : 's are'} still to score` : '',
        notSealed.length ? `${notSealed.join(', ')} ${notSealed.length === 1 ? 'is' : 'are'} not made and sealed yet` : '',
        check.unplanned.length ? `${check.unplanned.join(', ')} ${check.unplanned.length === 1 ? 'is' : 'are'} not planned yet` : '',
      ].filter(Boolean)
      const after = check.waiting.length ? ' Nothing more can be made for the study after the reveal.' : ''
      return fail(res, 409, `${words.join('; ')}. To reveal now anyway, send confirm "reveal early".${after}`, check)
    }
    const info = revealStudy(env, study, early && (check.remaining > 0 || check.waiting.length > 0), now(), log)
    return json(res, 200, info)
  }

  function findingsRoute(res: Res, study: string): void {
    if (!revealOf(env, study)) return fail(res, 409, 'The findings open after the reveal.')
    const file = path.join(studyDir(env, study), 'findings.json')
    if (sendFile(res, file, 'application/json', 'no-cache')) return
    return fail(res, 404, `The findings are not made yet. Run lab/lab report ${study} on the machine.`)
  }

  function reportPage(res: Res, study: string): void {
    if (!STUDY_ID.test(study) || !runsOfStudy(env, study).length) return fail(res, 404, 'No such study.')
    if (!revealOf(env, study)) return fail(res, 409, 'The report opens after the reveal.')
    const file = path.join(studyDir(env, study), 'report.html')
    if (sendFile(res, file, 'text/html; charset=utf-8', 'no-cache', { 'Content-Security-Policy': REPORT_CSP })) return
    return fail(res, 404, `The report is not made yet. Run lab/lab report ${study} on the machine.`)
  }

  // ------------------------------------------------------------------ refs --

  async function uploadRef(req: Req, res: Res): Promise<void> {
    if (!guardMutation(req, res, REF_TYPES)) return
    const name = String(req.headers['x-lab-name'] ?? '').trim()
    if (!name) return fail(res, 400, 'Name the photo in the X-Lab-Name header, for example scene.')
    const bytes = await readRaw(req, REF_UPLOAD_MAX)
    if (!bytes) return fail(res, 413, 'The photo is over 20 MB. The lab page converts a large photo to JPEG first.')
    if (!bytes.length) return fail(res, 400, 'The photo is empty.')
    try {
      const info = addRef(env, bytes, name)
      log(`Photo ${info.id} saved (${info.width} × ${info.height}).`)
      return json(res, 200, info)
    } catch (err) {
      return fail(res, 400, (err as Error).message)
    }
  }

  function refsNeededView() {
    const have = new Map(listRefs(env).map((r) => [r.id, r]))
    const byId = new Map<string, { id: string; describe: string; needsMask: boolean; present: boolean; runs: string[] }>()
    const add = (id: string, describe: string, mask: boolean, run: string | null) => {
      const had = byId.get(id)
      const row = had ?? { id, describe, needsMask: false, present: have.has(id) && (!mask || !!have.get(id)?.mask), runs: [] }
      row.needsMask ||= mask
      row.present = have.has(id) && (!row.needsMask || !!have.get(id)?.mask)
      if (run && !row.runs.includes(run)) row.runs.push(run)
      byId.set(id, row)
    }
    for (const s of Object.values(SUITES)) for (const r of refsUsed(s)) add(r.id, r.describe, r.mask, null)
    for (const run of runIds(env)) {
      const p = loadPlan(env, run)
      if (p) for (const n of refsNeeded(p)) add(n.id, n.describe, n.mask, run)
    }
    return { dropDir: refsDir(env), refs: [...byId.values()] }
  }

  const handler = safely(handle) as LabHandler
  handler.pauseAll = async () => {
    for (const l of live.values()) if (ACTIVE.has(l.driver.status().state)) await l.driver.pause()
  }
  return handler
}

// ------------------------------------------------------------ the server --

/**
 * Listen on the lab's own port. Anything the handler passes on is a 404.
 * Resolves once listening.
 */
export async function startServer(env: LabEnv = labEnv(), extra: Partial<LabCtx> = {}): Promise<{ server: http.Server; pauseAll(): Promise<void> }> {
  const handler = createLabHandler({ env, ...extra })
  const server = http.createServer((req, res) => {
    handler(req, res, () => {
      if (!res.headersSent) {
        res.statusCode = 404
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ error: 'No such page.' }))
      }
    })
  })
  server.requestTimeout = 120_000
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(env.port, env.host, () => {
      server.off('error', reject)
      resolve()
    })
  })
  return { server, pauseAll: () => handler.pauseAll() }
}
