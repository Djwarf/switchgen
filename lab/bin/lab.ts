/**
 * The model lab's commands: `lab/lab <command>` (lab/lab is the bash
 * launcher; it runs this with tsx from the repo root). `lab/lab help` lists
 * them.
 *
 * Nothing here sends work anywhere until you ask: `plan` and `smoke` write a
 * plan and send nothing; pictures are made only after Start on the lab page,
 * `lab/lab start <run>` or `lab/lab run <run>`. `check` only reads ComfyUI's
 * node list (GET /object_info). `read` asks the app's picture reader about a
 * run's finished pictures, and only when you run it.
 */
import { spawn, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { cellRel, labEnv, runDir, RUN_ID, type LabEnv } from '../core/env.ts'
import { checkSuite, descriptionProblem } from '../core/suite.ts'
import type { Cell, ChainSpec, Slot, Suite } from '../core/types.ts'
import { expand, finalizeGraph, refBindings, verifyCell } from '../core/cells.ts'
import { DEFAULT_RUNS, SEED_TIMINGS, describePlan, startRefusal, suiteById, type Plan } from '../core/plan.ts'
import { checkGraph, fetchObjectInfo } from '../core/validate.ts'
import { naReason } from '../core/na.ts'
import { cellsIndexPath, readDoneCells } from '../run/ledger.ts'
import { liveDrivers } from '../run/lock.ts'
import { addRef, listRefs, refIdFrom, refIndex, setDescribe, setMask } from '../run/refs.ts'
import { readCells, readingsPath } from '../run/reader.ts'
import { runnerClient } from '../run/runnerClient.ts'
import { calibrationGate, canStart, createDriver, type DriverStatus } from '../run/driver.ts'
import type { Sealed } from '../judge/seal.ts'
import { metadataProblems } from '../judge/blind.ts'
import {
  SMOKE_STUDY,
  SUITES,
  calibrationRefusal,
  createLabHandler,
  loadItems,
  loadPlan,
  planForStart,
  planRun,
  revealOf,
  revealedRefusal,
  runIds,
  runRows,
  ownSuites,
  refsNeeded,
  runsOfStudy,
  sealWithSuites,
  startServer,
  studyDir,
  writeBlindMarker,
  writeReport,
  type RunRow,
} from '../server/server.ts'

// ------------------------------------------------------------------ args --

type Args = { _: string[]; flags: Record<string, string | true> }

export function parseArgs(argv: readonly string[]): Args {
  const out: Args = { _: [], flags: {} }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--') {
      out._.push(...argv.slice(i + 1))
      break
    }
    if (a.startsWith('--')) {
      const eq = a.indexOf('=')
      if (eq > 2) {
        out.flags[a.slice(2, eq)] = a.slice(eq + 1)
        continue
      }
      const key = a.slice(2)
      const next = argv[i + 1]
      if (VALUE_FLAGS.has(key) && next !== undefined && !next.startsWith('--')) {
        out.flags[key] = next
        i++
      } else out.flags[key] = true
      continue
    }
    out._.push(a)
  }
  return out
}

/** Flags that take a value; every other flag is a switch. */
const VALUE_FLAGS = new Set(['run', 'until', 'name', 'judge'])

const flag = (a: Args, k: string): string | null => (typeof a.flags[k] === 'string' ? (a.flags[k] as string) : null)
const has = (a: Args, k: string): boolean => a.flags[k] !== undefined

class Usage extends Error {}

function need<T>(v: T | null | undefined, words: string): T {
  if (v === null || v === undefined || v === '') throw new Usage(words)
  return v
}

function runArg(a: Args, i = 0): string {
  const run = need(a._[i], 'Name the run, for example core-1.')
  if (!RUN_ID.test(run)) throw new Usage(`"${run}" is not a run name. Use letters, digits and dashes, like core-1.`)
  return run
}

const say = (line = '') => process.stdout.write(line + '\n')

function table(rows: string[][]): string {
  const widths = rows[0].map((_, i) => Math.max(...rows.map((r) => (r[i] ?? '').length)))
  return rows.map((r) => r.map((c, i) => (c ?? '').padEnd(widths[i])).join('  ').trimEnd()).join('\n')
}

function duration(seconds: number | null | undefined): string {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return 'unknown'
  const m = Math.round(seconds / 60)
  if (m < 60) return `${m} min`
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`
}

// ---------------------------------------------------------------- check --

/** Import specifiers in a source file: from '…', import('…'), require('…'), export … from '…'. */
const IMPORT_RE = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|\brequire\s*\(\s*)(['"])([^'"\n]+)\1/g

/**
 * Files under src/ and server/ that import anything from lab/. The app must
 * never load the lab: the lab reads the app, never the other way round.
 */
export function labImports(repoRoot: string): { file: string; spec: string }[] {
  const labRoot = path.join(repoRoot, 'lab') + path.sep
  const out: { file: string; spec: string }[] = []
  const walk = (dir: string) => {
    let names: fs.Dirent[]
    try {
      names = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const d of names) {
      const full = path.join(dir, d.name)
      if (d.isDirectory()) {
        if (d.name === 'node_modules' || d.name.startsWith('.')) continue
        walk(full)
      } else if (/\.(m?[jt]sx?|cjs|cts|mts)$/.test(d.name)) {
        const text = fs.readFileSync(full, 'utf8')
        for (const m of text.matchAll(IMPORT_RE)) {
          const spec = m[2]
          const target = spec.startsWith('.') ? path.resolve(path.dirname(full), spec) : spec.startsWith('/') ? spec : null
          const bare = /^(?:\.\/)?lab\//.test(spec)
          if (bare || (target && (target + path.sep).startsWith(labRoot))) out.push({ file: path.relative(repoRoot, full), spec })
        }
      }
    }
  }
  walk(path.join(repoRoot, 'src'))
  walk(path.join(repoRoot, 'server'))
  return out
}

async function cmdCheck(env: LabEnv, a: Args): Promise<number> {
  let bad = 0
  const fault = (line: string) => {
    bad++
    say(`  ✗ ${line}`)
  }

  say('The app never loads the lab:')
  const imports = labImports(env.repoRoot)
  if (imports.length) for (const i of imports) fault(`${i.file} imports ${i.spec}`)
  else say('  ✓ nothing under src/ or server/ imports from lab/')
  const rootVitest = fs.readFileSync(path.join(env.repoRoot, 'vitest.config.ts'), 'utf8')
  if (/lab\//.test(rootVitest)) fault('the app’s vitest.config.ts mentions lab/')
  else say('  ✓ the app’s tests never include lab/tests')
  const rootTs = fs.readFileSync(path.join(env.repoRoot, 'tsconfig.json'), 'utf8')
  if (/lab/.test(rootTs)) fault('the root tsconfig.json mentions lab')
  else say('  ✓ the root tsconfig.json does not reference lab/')

  say('')
  say('The suites:')
  const all = Object.values(SUITES)
  for (const s of all) {
    const problems = checkSuite(s, all.filter((o) => o !== s))
    if (problems.length) for (const p of problems) fault(`${s.id}: ${p}`)
    else say(`  ✓ ${s.id} passes its checks`)
  }

  say('')
  say('Every graph, built with the app’s own functions and made ready to send as the driver makes it:')
  let built: ReturnType<typeof expand>
  const refs = refIndex(env)
  try {
    built = expand(all, refs)
  } catch (err) {
    fault(`could not build the cells: ${(err as Error).message}`)
    return 1
  }
  const bindRefs = refBindings(refs)
  const ready = new Map<string, { cell: Cell; graph: Parameters<typeof checkGraph>[0] }>()
  // Problems grouped by what they say, so one fault in a family graph reads once.
  const grouped = new Map<string, string[]>()
  const note = (problem: string, where: string) => grouped.set(problem, [...(grouped.get(problem) ?? []), where])
  for (const cell of built.cells) {
    if (ready.has(cell.cellId)) continue
    const g = built.graphs.get(cell.cellId)
    const where = `${cell.slot} (${cell.op}, ${cell.model ?? cell.chain})`
    if (!g) {
      note('no graph was built', where)
      continue
    }
    try {
      const up = cell.upstream ? { rel: cellRel(cell.upstream) } : undefined
      const made = finalizeGraph(g, cell, { upstream: up, refs: bindRefs })
      for (const p of verifyCell(made.graph, cell)) note(p.replace(/[0-9a-f]{16}/g, '<cell>'), where)
      ready.set(cell.cellId, { cell, graph: made.graph })
    } catch (err) {
      note((err as Error).message.replace(/[0-9a-f]{16}/g, '<cell>'), where)
    }
  }
  const report = () => {
    for (const [problem, wheres] of grouped) {
      const uniq = [...new Set(wheres)]
      fault(`${problem} (${wheres.length}×: ${uniq.slice(0, 3).join(', ')}${uniq.length > 3 ? `, and ${uniq.length - 3} more` : ''})`)
    }
    grouped.clear()
  }
  report()
  say(`  ${ready.size} distinct graphs ready to send; ${built.na.length} not-applicable pairings, each with the registry’s reason; ${built.blocked.length} test${built.blocked.length === 1 ? '' : 's'} waiting for a photo.`)

  if (has(a, 'no-comfy')) {
    say('')
    say('ComfyUI’s node list was not read (--no-comfy).')
  } else {
    say('')
    say(`ComfyUI’s node list (GET ${env.comfyUrl}/object_info, a read):`)
    let info: Awaited<ReturnType<typeof fetchObjectInfo>> | null = null
    try {
      info = await fetchObjectInfo(env.comfyUrl)
    } catch (err) {
      fault(`could not read it: ${(err as Error).message}`)
    }
    if (info) {
      let ok = 0
      for (const { cell, graph } of ready.values()) {
        const problems = checkGraph(graph, info)
        if (problems.length) for (const p of problems) note(p, `${cell.op} on ${cell.file}`)
        else ok++
      }
      report()
      say(`  ${ok} of ${ready.size} graphs use only nodes, inputs and choices ComfyUI has.`)
    }
  }

  say('')
  say(bad ? `${bad} problem${bad === 1 ? '' : 's'}.` : 'All checks pass.')
  return bad ? 1 : 0
}

// ----------------------------------------------------------------- plan --

function printPlan(env: LabEnv, plan: Plan): void {
  say(describePlan(plan))
  const needs = refsNeeded(plan, ownSuites(env, plan.run))
  if (!needs.length) return
  const have = new Map(listRefs(env).map((r) => [r.id, r]))
  say('')
  for (const n of needs) {
    const r = have.get(n.id)
    if (!r) say(`The "${n.id}" photo: not here yet (the lab page’s Photos screen, or ${path.join(env.labDir, 'refs')}).`)
    else say(`The "${n.id}" photo: here (${r.width} × ${r.height}, upright)${n.mask ? (r.mask ? ', area marked' : ', no area marked yet') : ''}.`)
  }
}

function suitesFrom(ids: string[]): Suite[] {
  if (!ids.length) throw new Usage(`Name at least one suite: ${Object.keys(SUITES).join(', ')}.`)
  return ids.map((id) => {
    const s = suiteById(id)
    if (!s) throw new Usage(`There is no suite called ${id}. The suites are ${Object.keys(SUITES).join(', ')}.`)
    return s
  })
}

function cmdPlan(env: LabEnv, a: Args): number {
  const suites = suitesFrom(a._)
  const run = flag(a, 'run') ?? (suites.length === 1 ? DEFAULT_RUNS[suites[0].id] : null)
  if (!run) throw new Usage('Name the run with --run, for example --run core-1.')
  if (!RUN_ID.test(run)) throw new Usage(`"${run}" is not a run name. Use letters, digits and dashes, like core-1.`)
  const plan = planRun(env, run, suites)
  printPlan(env, plan)
  return 0
}

// ---------------------------------------------------------------- smoke --


/** The suite's picture models, fastest first by the plan's timing table (milliseconds per step). */
function fastFirst(models: Suite['models']): string[] {
  const ms = (k: string) => SEED_TIMINGS[models[k].file]?.msPerStep ?? Number.POSITIVE_INFINITY
  return Object.keys(models)
    .filter((k) => models[k].role !== 'edit')
    .sort((x, y) => ms(x) - ms(y) || x.localeCompare(y))
}

/**
 * One cell per operation at 8 steps, on the fastest model that can take it:
 * t2i, face, hand and hires from nothing; i2i and a region edit on the cat
 * photo (the region needs its marked area); edit with the editing model; and
 * the lab-only detailOnPicture as a chain step on the t2i picture.
 */
export function smokeSuite(env: LabEnv): { suite: Suite; left: string[] } {
  const base = SUITES['first-pass-core'] ?? Object.values(SUITES)[0]
  const refs = new Map(listRefs(env).map((r) => [r.id, r]))
  const left: string[] = []
  const models = base.models
  const order = fastFirst(models)
  const pick = (op: Slot['op'], slot: Pick<Slot, 'block' | 'negativeAdd' | 'target'>) =>
    order.find((k) => naReason(models[k].file, op, slot) === null) ?? null
  const editor = Object.keys(models).find((k) => models[k].role === 'edit') ?? null
  const slots: Slot[] = []
  const add = (s: Omit<Slot, 'shape' | 'models'> & { model: string | null }) => {
    const { model, ...rest } = s
    if (!model) {
      left.push(`${s.op}: no model can take it`)
      return
    }
    slots.push({ ...rest, shape: 'square', models: [model] })
  }
  const t2iModel = pick('t2i', { block: 'overall' })
  add({ id: 'smoke.t2i', block: 'overall', op: 't2i', model: t2iModel, text: 'A red apple on a wooden table, soft morning light.' })
  add({ id: 'smoke.face', block: 'detail', op: 'face', target: 'face', model: pick('face', { block: 'detail', target: 'face' }), text: 'A portrait photograph of a woman in her forties smiling, soft window light.', humans: true })
  add({ id: 'smoke.hand', block: 'detail', op: 'hand', target: 'hand', model: pick('hand', { block: 'detail', target: 'hand' }), text: 'Close-up of an adult woman’s hand holding a cup of coffee.', humans: true })
  add({ id: 'smoke.hires', block: 'detail', op: 'hires', model: pick('hires', { block: 'detail' }), text: 'A lighthouse on a rocky coast at dusk.' })
  const cat = refs.get('cat')
  const catSpec = base.refs?.cat
  if (cat) {
    add({ id: 'smoke.i2i', block: 'reference', op: 'i2i', source: { ref: 'cat' }, denoise: 0.65, model: pick('i2i', { block: 'reference' }), text: 'A watercolour illustration of {describe}.' })
    if (editor) {
      add({ id: 'smoke.edit', block: 'edit', op: 'edit', source: { ref: 'cat' }, model: editor, text: 'Make it look like a snowy winter day; change nothing else.' })
    }
    if (cat.mask) {
      add({ id: 'smoke.region', block: 'region', op: 'region', source: { ref: 'cat' }, denoise: 0.55, model: pick('region', { block: 'region' }), text: 'a small vase of yellow tulips' })
    } else left.push('region: the cat photo has no marked area yet (draw one on the Photos page)')
  } else left.push('i2i, edit and region: the cat photo is not in the lab')
  const chains: ChainSpec[] = []
  if (t2iModel && naReason(models[t2iModel].file, 'face', { block: 'detail', target: 'face' }) === null) {
    chains.push({
      id: 'smoke.chain',
      name: 'Face detail on a finished picture (lab-only shape)',
      block: 'detail',
      compose: { slot: 'smoke.t2i', model: t2iModel },
      steps: [{ model: t2iModel, op: 'detailOnPicture', target: 'face', text: '{text}' }],
    })
  }
  const suite: Suite = {
    id: 'smoke',
    version: 1,
    study: SMOKE_STUDY,
    seeds: [1001],
    steps: 8,
    sampler: base.sampler,
    shapes: base.shapes,
    models,
    core: [],
    refs: cat && catSpec ? { cat: { ...catSpec, needsMask: !!cat.mask } } : {},
    slots,
    chains,
  }
  return { suite, left }
}

function cmdSmoke(env: LabEnv, a: Args): number {
  const run = flag(a, 'run') ?? 'smoke-1'
  if (!RUN_ID.test(run)) throw new Usage(`"${run}" is not a run name.`)
  const { suite, left } = smokeSuite(env)
  const problems = checkSuite(suite)
  if (problems.length) throw new Error(`The smoke suite does not pass its own checks: ${problems.join(' ')}`)
  const plan = planRun(env, run, [suite])
  say(`The smoke run: one picture per operation, ${suite.steps} steps, one seed, on the fastest model that can take each (by the plan's timing table).`)
  for (const s of suite.slots) say(`  ${s.op.padEnd(16)} ${s.id}`)
  for (const c of suite.chains) for (const st of c.steps) say(`  ${st.op.padEnd(16)} ${c.id} (lab-only shape, not in the app)`)
  for (const l of left) say(`  left out: ${l}`)
  say('')
  printPlan(env, plan)
  say('The smoke run has no calibration night; it starts without one.')
  return 0
}

// ------------------------------------------------------- start and run --

const labBase = (env: LabEnv) => `http://${env.host.includes(':') ? `[${env.host}]` : env.host}:${env.port}`

/** The lab server's answer, or null when it is not running. Only ever the lab's own port. */
async function labServer(env: LabEnv, method: 'GET' | 'POST', route: string, body?: unknown): Promise<{ status: number; json: any } | null> {
  try {
    const res = await fetch(labBase(env) + route, {
      method,
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(10_000),
    })
    let json: any = null
    try {
      json = await res.json()
    } catch {
      json = null
    }
    return { status: res.status, json }
  } catch {
    return null
  }
}

async function labServerUp(env: LabEnv): Promise<boolean> {
  const r = await labServer(env, 'GET', '/api/lab/health')
  return !!r && r.status === 200 && r.json?.server === 'switchgen-lab'
}

function startFlags(a: Args, run: string, env: LabEnv): { until?: string; skipCalibration?: boolean } {
  const until = flag(a, 'until')
  if (until !== null && !/^([01]?\d|2[0-3]):[0-5]\d$/.test(until)) throw new Usage(`--until takes a 24-hour time like 07:30, not "${until}".`)
  const out: { until?: string; skipCalibration?: boolean } = {}
  if (until) out.until = until
  if (has(a, 'skip-calibration') || loadPlan(env, run)?.study === SMOKE_STUDY) out.skipCalibration = true
  return out
}

/**
 * Why `run` may not start now, in words, or null when it may: the checks the
 * lab page's Start makes. Its study must not be revealed; a run that has sent
 * nothing is planned again first with the photos the lab has now; then
 * nothing may wait for a photo, the calibration pairs must be judged (unless
 * skipped) and no other run may be being made.
 */
export function startProblem(env: LabEnv, run: string, opts: { skipCalibration?: boolean }, log: (line: string) => void = say): string | null {
  const planned = loadPlan(env, run)
  if (!planned) return `There is no plan for ${run}. Plan it first: lab/lab plan <suite> --run ${run}.`
  const shut = revealedRefusal(env, planned.study)
  if (shut) return `${shut} ${run} was not started.`
  let plan: Plan
  try {
    const p = planForStart(env, run, planned)
    plan = p.plan
    if (p.replanned) log(`Planned ${run} again before starting, with the photos the lab has now (${plan.estimate.newPictures} pictures to make).`)
  } catch (err) {
    return `Could not plan ${run} again before starting: ${(err as Error).message}`
  }
  const waits = startRefusal(plan, env)
  if (waits) return waits
  const may = canStart(env, run, { skipCalibration: !!opts.skipCalibration })
  if (may.ok) return null
  const gateOnly = !opts.skipCalibration && !calibrationGate(env, run).ok
  return gateOnly ? calibrationRefusal(may.why, 'To start anyway, add --skip-calibration.') : may.why
}

async function cmdStart(env: LabEnv, a: Args): Promise<number> {
  const run = runArg(a)
  if (!loadPlan(env, run)) throw new Usage(`There is no plan for ${run}. Plan it first: lab/lab plan <suite> --run ${run}.`)
  const opts = startFlags(a, run, env)
  if (await labServerUp(env)) {
    const r = await labServer(env, 'POST', `/api/lab/runs/${encodeURIComponent(run)}/start`, { confirm: 'start', ...opts })
    if (r && r.status === 202) {
      say(`Started ${run} in the lab server. Follow it on the lab page, or with lab/lab status.`)
      return 0
    }
    say(`The lab server refused: ${r?.json?.error ?? `it answered ${r?.status}`}`)
    return 1
  }
  // The background process writes only to its log, so a refusal it would
  // meet at once is said here instead, and nothing is started.
  const problem = startProblem(env, run, opts)
  if (problem) {
    say(`${run} was not started:`)
    say(problem)
    return 1
  }
  const log = path.join(runDir(env, run), 'driver.log')
  fs.mkdirSync(path.dirname(log), { recursive: true })
  const out = fs.openSync(log, 'a', 0o600)
  const args = ['tsx', path.join(env.repoRoot, 'lab', 'bin', 'lab.ts'), 'run', run]
  if (opts.until) args.push('--until', opts.until)
  if (opts.skipCalibration) args.push('--skip-calibration')
  const child = spawn('npx', args, { cwd: env.repoRoot, detached: true, stdio: ['ignore', out, out], env: process.env })
  child.unref()
  fs.closeSync(out)
  say(`The lab server is not running, so ${run} was started in the background (pid ${child.pid}).`)
  say(`Its log: ${log}. Pause it with lab/lab pause ${run}.`)
  return 0
}

async function cmdRun(env: LabEnv, a: Args): Promise<number> {
  const run = runArg(a)
  if (!loadPlan(env, run)) throw new Usage(`There is no plan for ${run}. Plan it first: lab/lab plan <suite> --run ${run}.`)
  const opts = startFlags(a, run, env)
  const stamp = () => new Date().toTimeString().slice(0, 8)
  const problem = startProblem(env, run, opts, (l) => say(`${stamp()} ${l}`))
  if (problem) {
    say(`${stamp()} ${run} was not started:`)
    say(problem)
    return 1
  }
  const ffmpeg = process.env.SWITCHGEN_FFMPEG ?? 'ffmpeg'
  const driver = createDriver({
    env,
    run,
    client: runnerClient(env.appUrl),
    log: (l) => say(`${stamp()} ${l}`),
    seal: (e, r) => sealWithSuites(e, r, ffmpeg),
  })
  let interrupts = 0
  const onInterrupt = () => {
    interrupts++
    if (interrupts > 1) {
      say('Leaving now. The ledger keeps the run; start it again to go on.')
      process.exit(130)
    }
    say(`${stamp()} Pausing: stopping the group in progress. Press Ctrl-C again to leave at once.`)
    driver.pause().catch((err) => say(`Could not pause: ${(err as Error).message}`))
  }
  process.on('SIGINT', onInterrupt)
  process.on('SIGTERM', onInterrupt)
  let last = ''
  const show = (s: DriverStatus) => {
    const line = `${s.state} · ${s.made} of ${s.total} made${s.failed ? ` · ${s.failed} failed` : ''}${s.etaSeconds !== null ? ` · about ${duration(s.etaSeconds)} left (estimate)` : ''}${s.message ? ` · ${s.message}` : ''}`
    if (line !== last) say(`${stamp()} ${line}`)
    last = line
  }
  const timer = setInterval(() => show(driver.status()), 5000)
  say(`${stamp()} Starting ${run}${opts.until ? `, stopping at ${opts.until}` : ''}.`)
  try {
    await driver.start(opts)
  } finally {
    clearInterval(timer)
  }
  const s = driver.status()
  show(s)
  return s.state === 'error' ? 1 : 0
}

async function cmdPause(env: LabEnv, a: Args): Promise<number> {
  const run = runArg(a)
  if (await labServerUp(env)) {
    const r = await labServer(env, 'POST', `/api/lab/runs/${encodeURIComponent(run)}/pause`, {})
    say(r?.status === 200 ? (r.json?.message ?? `Asked the lab server to pause ${run}.`) : `The lab server answered: ${r?.json?.error ?? r?.status}`)
    return r?.status === 200 ? 0 : 1
  }
  const holder = liveDrivers(env).find((l) => l.run === run)
  if (!holder) {
    say(`${run} is not being sent.`)
    return 0
  }
  process.kill(holder.pid, 'SIGINT')
  say(`Asked the lab process sending ${run} (pid ${holder.pid}) to pause.`)
  return 0
}

function cmdStatus(env: LabEnv): number {
  const rows: RunRow[] = runRows(env)
  if (!rows.length) {
    say('No runs are planned yet. Plan one with lab/lab plan <suite> --run <name>.')
    return 0
  }
  say(
    table([
      ['run', 'study', 'state', 'made', 'failed', 'scored', 'waiting for'],
      ...rows.map((r) => [
        r.run,
        r.study,
        r.state,
        `${r.made}/${r.total}`,
        String(r.failed),
        r.judged + r.toJudge ? `${r.judged}/${r.judged + r.toJudge}` : '-',
        r.needsRefs.length ? `photo ${r.needsRefs.join(', ')}` : r.gate === 'unmet' ? 'calibration answers' : '',
      ]),
    ]),
  )
  return 0
}

// ----------------------------------------------------------------- refs --

async function cmdRefs(env: LabEnv, a: Args): Promise<number> {
  const [sub, ...rest] = a._
  if (!sub || sub === 'list') {
    const refs = listRefs(env)
    if (!refs.length) say(`No photos yet. Add one with lab/lab refs add <file> --name cat, on the lab page, or by putting it in ${path.join(env.labDir, 'refs')}.`)
    for (const r of refs) {
      say(`${r.id}: ${r.width} × ${r.height} (upright)${r.mask ? `, area marked${r.rect ? ` ${r.rect.x},${r.rect.y},${r.rect.w},${r.rect.h}` : ''}` : ', no area marked'}`)
      if (r.describe) say(`  described as: ${r.describe}`)
    }
    return 0
  }
  if (sub === 'add') {
    const file = need(rest[0], 'Name the photo file: lab/lab refs add <file> [--name id].')
    const bytes = fs.readFileSync(file)
    const info = addRef(env, bytes, flag(a, 'name') ?? refIdFrom(path.basename(file)))
    say(`Saved as ${info.id}: ${info.width} × ${info.height}, the right way up.`)
    return 0
  }
  if (sub === 'mask') {
    const id = need(rest[0], 'Name the photo: lab/lab refs mask <id> x,y,w,h.')
    const nums = need(rest[1], 'Give the area as x,y,w,h in the photo’s upright pixels.').split(',').map(Number)
    if (nums.length !== 4 || !nums.every((n) => Number.isFinite(n))) throw new Usage('Give the area as four numbers: x,y,w,h.')
    const info = setMask(env, id, { x: nums[0], y: nums[1], w: nums[2], h: nums[3] })
    say(`Area marked on ${info.id}${info.rect ? `: ${info.rect.x},${info.rect.y},${info.rect.w},${info.rect.h}` : ''}.`)
    return 0
  }
  if (sub === 'describe') {
    const id = need(rest[0], 'Name the photo: lab/lab refs describe <id> <words…>.')
    const text = rest.slice(1).join(' ').trim()
    const problem = text ? descriptionProblem(id, text) : null
    if (problem) {
      say(`Not saved. ${problem}`)
      return 1
    }
    const info = setDescribe(env, id, text || null)
    say(info.describe ? `${info.id} is now described as: ${info.describe}` : `${info.id} goes back to the suite’s own description.`)
    return 0
  }
  throw new Usage('lab/lab refs [list | add <file> [--name id] | mask <id> x,y,w,h | describe <id> <words…>]')
}

// --------------------------------------------------- read, seal, check --

async function cmdRead(env: LabEnv, a: Args): Promise<number> {
  const run = runArg(a)
  const plan = loadPlan(env, run)
  if (!plan) throw new Usage(`There is no run called ${run}.`)
  const done = readDoneCells(env)
  const ids = [...new Set([...plan.order, ...(plan.reused ?? [])])].filter((c) => done.has(c) && !done.get(c)?.removed)
  say(`Asking the app’s picture reader about ${ids.length} picture${ids.length === 1 ? '' : 's'} of ${run}, 24 at a time.`)
  const r = await readCells(env, runnerClient(env.appUrl), ids, { run, log: (l) => say(l) })
  say(`${r.read.length} read, ${r.quarantined.length} removed by the content rule, ${r.pending.length} not read${r.why ? ` (${r.why})` : ''}.`)
  return r.pending.length ? 1 : 0
}

async function cmdSeal(env: LabEnv, a: Args): Promise<number> {
  const run = runArg(a)
  const ffmpeg = process.env.SWITCHGEN_FFMPEG ?? 'ffmpeg'
  const r = await sealWithSuites(env, run, ffmpeg, (l) => say(l))
  say(`Sealed ${run}: ${r.sets} sets, ${r.items} items to score. Judging opens once lab/lab check-blind ${run} passes (the lab page runs it for you).`)
  return 0
}

/**
 * Strings that would give a contestant away, from the sealed key: the model
 * keys and weight files (with and without their extension), family ids,
 * cell ids, file names under .lab/cells, and the runner's job and prompt ids.
 */
export function sealedStrings(sealed: Sealed): string[] {
  const out = new Set<string>()
  const add = (v: unknown) => {
    if (typeof v !== 'string') return
    const s = v.trim()
    if (s.length >= 3) out.add(s)
  }
  for (const t of Object.values(sealed.tokens ?? {})) {
    add(t.contestant)
    add(t.file)
    if (typeof t.file === 'string') add(t.file.replace(/\.(safetensors|gguf|ckpt|pt|bin)$/i, ''))
    add(t.familyId)
    add(t.cellId)
    add(t.rel)
    if (typeof t.rel === 'string') add(path.basename(t.rel))
    add(t.jobId)
    add(t.promptId)
  }
  for (const letters of Object.values(sealed.letters ?? {})) for (const c of Object.values(letters)) add(c)
  out.add('.lab/cells')
  return [...out]
}

/** The sealed strings found in `text`, matched as whole words, case-insensitively. */
export function leaksIn(text: string, strings: readonly string[]): string[] {
  const found: string[] = []
  const lower = text.toLowerCase()
  for (const s of strings) {
    const needle = s.toLowerCase()
    let at = lower.indexOf(needle)
    while (at >= 0) {
      const before = at === 0 ? '' : lower[at - 1]
      const after = lower[at + needle.length] ?? ''
      const edge = (c: string) => !c || !/[a-z0-9]/.test(c)
      if (edge(before) && edge(after)) {
        found.push(s)
        break
      }
      at = lower.indexOf(needle, at + 1)
    }
  }
  return found
}

export type BlindReport = { ok: boolean; checked: number; problems: string[] }

/**
 * Scan everything the judge can be shown for sealed strings: items.json, the
 * lab server's answers for the run (driven in-process, no socket), and every
 * blind copy's bytes and chunks. A problem names where, never the string.
 */
export async function checkBlind(env: LabEnv, run: string): Promise<BlindReport> {
  const dir = runDir(env, run)
  const problems: string[] = []
  let checked = 0
  const sealedFile = path.join(dir, 'sealed.json')
  if (!fs.existsSync(sealedFile)) return { ok: false, checked, problems: [`${run} is not sealed yet.`] }
  // The one read of the key outside the reveal: to know what must not appear.
  // Nothing from it is printed; problems say where a leak is, not what it is.
  const sealed = JSON.parse(fs.readFileSync(sealedFile, 'utf8')) as Sealed
  const strings = sealedStrings(sealed)
  const items = loadItems(env, run)
  if (!items) return { ok: false, checked, problems: [`${run} has no items.json.`] }

  const scan = (where: string, text: string) => {
    checked++
    const hits = leaksIn(text, strings)
    if (hits.length) problems.push(`${where} holds ${hits.length} sealed string${hits.length === 1 ? '' : 's'}.`)
  }
  scan('items.json', fs.readFileSync(path.join(dir, 'items.json'), 'utf8'))

  const handler = createLabHandler({
    env,
    allowedHosts: ['localhost'],
    log: () => {},
    makeDriver: () => {
      throw new Error('no driver in a blind check')
    },
    skipBlindGate: true,
    readOnly: true,
  })
  const routes = [
    '/api/lab/runs',
    `/api/lab/runs/${run}/status`,
    `/api/lab/runs/${run}/next?judge=check-blind&session=check-blind`,
    ...items.sets.map((s) => `/api/lab/runs/${run}/sets/${encodeURIComponent(s.setId)}`),
    '/api/lab/refs',
  ]
  for (const route of routes) {
    const r = await drive(handler, route)
    scan(`the answer to GET ${route.replace(/\?.*$/, '')}`, r.body)
  }

  const view = path.join(dir, 'view')
  let names: string[] = []
  try {
    names = fs.readdirSync(view)
  } catch {
    problems.push('There are no blind copies (view/ is missing).')
  }
  const pictureChunk = /^[A-Za-z0-9_-]{22,64}-(g|f)\.webp$/
  for (const name of names) {
    if (!pictureChunk.test(name)) {
      problems.push(`view/ holds a file that is not a blind copy (${name.length} characters long).`)
      continue
    }
    const buf = fs.readFileSync(path.join(view, name))
    checked++
    const meta = metadataProblems(buf)
    if (meta.length) problems.push(`A blind copy carries more than picture data: ${meta.join(', ')}.`)
    // Anything after the RIFF chunk is invisible to a chunk reader but still served.
    const riff = buf.length >= 12 && buf.toString('ascii', 0, 4) === 'RIFF' ? buf.readUInt32LE(4) : -1
    if (riff >= 0 && buf.length > 8 + riff + (riff & 1)) problems.push('A blind copy has data after the end of its picture.')
    // Bytes are not words: any sealed string of six characters or more, anywhere.
    const hay = buf.toString('latin1').toLowerCase()
    if (strings.some((s) => s.length >= 6 && hay.includes(s.toLowerCase()))) problems.push('A blind copy’s bytes hold a sealed string.')
  }
  return { ok: problems.length === 0, checked, problems }
}

/** GET a route from the handler, with no socket. */
async function drive(handler: ReturnType<typeof createLabHandler>, url: string): Promise<{ status: number; body: string }> {
  const { Readable } = await import('node:stream')
  const { EventEmitter } = await import('node:events')
  const req = Object.assign(Readable.from([]), { method: 'GET', url, headers: { host: 'localhost' } })
  return new Promise((resolve) => {
    let body = ''
    let status = 200
    const res = Object.assign(new EventEmitter(), {
      req,
      headersSent: false,
      destroyed: false,
      setHeader() {},
      getHeader() {
        return undefined
      },
      writeHead(code: number) {
        status = code
        return res
      },
      write(chunk: string | Buffer) {
        body += chunk.toString()
        return true
      },
      end(chunk?: string | Buffer) {
        if (chunk !== undefined) body += Buffer.isBuffer(chunk) ? chunk.toString('latin1') : chunk
        resolve({ status, body })
      },
    })
    Object.defineProperty(res, 'statusCode', { get: () => status, set: (v: number) => (status = v) })
    handler(req as never, res as never, () => resolve({ status: 404, body: '' }))
  })
}

async function cmdCheckBlind(env: LabEnv, a: Args): Promise<number> {
  const run = runArg(a)
  const r = await checkBlind(env, run)
  if (loadItems(env, run)) writeBlindMarker(env, run, { ok: r.ok, checked: r.checked, problems: r.problems.length })
  if (r.ok) {
    say(`Blind: nothing sealed appears in ${r.checked} answers, files and pictures of ${run}.`)
    return 0
  }
  say(`Not blind (${r.problems.length} problem${r.problems.length === 1 ? '' : 's'}; what leaked is not printed, so you stay blind):`)
  for (const p of r.problems) say(`  ✗ ${p}`)
  return 1
}

// ------------------------------------------------ serve, phone, report --

async function cmdServe(env: LabEnv): Promise<number> {
  const { server, pauseAll } = await startServer(env)
  say(`The lab page is at ${labBase(env)}/ (this machine only; lab/lab phone says how to reach it from your phone).`)
  say('Nothing is sent until you press Start on a run.')
  let closing = false
  const close = async () => {
    if (closing) process.exit(130)
    closing = true
    say('Closing: pausing any run in progress first.')
    try {
      await pauseAll()
    } catch (err) {
      say(`Could not pause: ${(err as Error).message}`)
    }
    server.close(() => process.exit(0))
    setTimeout(() => process.exit(0), 3000).unref()
  }
  process.on('SIGINT', close)
  process.on('SIGTERM', close)
  await new Promise(() => {})
  return 0
}

function cmdPhone(env: LabEnv): number {
  say('To reach the lab page from your phone over Tailscale, run this once on this machine.')
  say('The lab never runs it for you; it needs sudo:')
  say('')
  say(`  sudo tailscale serve --bg --https=8443 http://127.0.0.1:${env.port}`)
  say('')
  say(`Then open https://${os.hostname()}.<your tailnet>.ts.net:8443/ on the phone (tailscale status names the tailnet).`)
  say('Only devices on your tailnet can reach it. Never use tailscale funnel, which would put it on the public internet.')
  say('To stop serving it: sudo tailscale serve --https=8443 off')
  return 0
}

/** A study named directly, or the study of a named run. */
function studyArg(env: LabEnv, a: Args): string {
  const name = need(a._[0], 'Name the run or the study, for example core-1 or first-pass.')
  const p = RUN_ID.test(name) ? loadPlan(env, name) : null
  if (p) return p.study
  if (runsOfStudy(env, name).length) return name
  throw new Usage(`There is no run or study called ${name}.`)
}

function cmdReport(env: LabEnv, a: Args): number {
  const study = studyArg(env, a)
  if (!revealOf(env, study)) {
    say(`Study ${study} is not revealed yet, so there is no report. Reveal it on the lab page first.`)
    return 1
  }
  const out = writeReport(env, study)
  say(`The report for study ${study} (runs ${out.runs.join(', ')}):`)
  say(`  ${out.report}`)
  say(`  ${out.findings}`)
  say(`  ${out.csv}`)
  say(`The lab page serves it at ${labBase(env)}/report/${study}.`)
  return 0
}

function cmdFindings(env: LabEnv, a: Args): number {
  const study = studyArg(env, a)
  const src = path.join(studyDir(env, study), 'findings.json')
  if (!fs.existsSync(src)) {
    say(`Study ${study} has no findings yet: they are made at the reveal (or with lab/lab report ${study} after it).`)
    return 1
  }
  const dst = path.join(env.repoRoot, 'lab', 'findings', `${study}.json`)
  fs.mkdirSync(path.dirname(dst), { recursive: true })
  fs.copyFileSync(src, dst)
  say(`Copied to ${path.relative(env.repoRoot, dst)}. It names models and scores; commit it only if you want to.`)
  return 0
}

/**
 * Rewrite a JSON-lines file of cell rows without the rows `drop` picks: to a
 * new file renamed over the old one, read again if it grew meanwhile.
 * Returns the ids of the cells whose rows were taken out.
 */
function dropCellRows(file: string, drop: (row: { cellId: string; quarantined?: unknown }) => boolean): Set<string> {
  for (let attempt = 0; attempt < 5; attempt++) {
    let text: string
    try {
      text = fs.readFileSync(file, 'utf8')
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return new Set()
      throw err
    }
    const gone = new Set<string>()
    const kept = text.split('\n').filter((line) => {
      let row: { cellId?: unknown; quarantined?: unknown } | null = null
      try {
        row = line.trim() ? JSON.parse(line) : null
      } catch {
        row = null
      }
      if (!row || typeof row.cellId !== 'string' || !drop(row as { cellId: string })) return true
      gone.add(row.cellId)
      return false
    })
    if (!gone.size) return gone
    const tmp = path.join(path.dirname(file), `.tmp-${process.pid}-${Date.now()}-${path.basename(file)}`)
    const fd = fs.openSync(tmp, 'w', 0o600)
    try {
      fs.writeSync(fd, kept.join('\n'))
      fs.fsyncSync(fd)
    } finally {
      fs.closeSync(fd)
    }
    if (fs.statSync(file).size !== Buffer.byteLength(text)) {
      fs.rmSync(tmp, { force: true })
      continue
    }
    fs.renameSync(tmp, file)
    return gone
  }
  throw new Error(`The file ${file} kept changing while prune ran, so no pictures were removed. Try again when no run is being made.`)
}

/**
 * Take these cells out of the lab's list of finished pictures (cells.jsonl)
 * and out of the picture reader's results (readings.jsonl), so a later plan
 * makes them again instead of reusing a file that is gone, and the picture
 * made again goes through the reader and its content rule afresh.
 * A cell the content rule removed keeps its rows: it is never made again.
 * Returns how many cells were taken out of the list.
 */
export function forgetCells(env: LabEnv, ids: ReadonlySet<string>): number {
  const removed = new Set([...readDoneCells(env)].flatMap(([id, r]) => (r.removed ? [id] : [])))
  const forget = (id: string) => ids.has(id) && !removed.has(id)
  // The readings first: if this stops halfway, a picture still listed is only read again.
  dropCellRows(readingsPath(env), (r) => forget(r.cellId) && r.quarantined !== true)
  return dropCellRows(cellsIndexPath(env), (r) => forget(r.cellId)).size
}

function cmdPrune(env: LabEnv, a: Args): number {
  const run = runArg(a)
  const plan = loadPlan(env, run)
  if (!plan) throw new Usage(`There is no run called ${run}.`)
  if (!revealOf(env, plan.study)) {
    say(`Study ${plan.study} is not revealed yet, so its pictures are kept.`)
    return 1
  }
  // A smoke run may be planned and started after its study's reveal.
  if (!loadItems(env, run) && !revealedRefusal(env, plan.study)) {
    say(`${run} is not made and sealed yet, and it can still be started, so its pictures are kept.`)
    return 1
  }
  // A picture another run may still need is kept: a run of a study not
  // revealed yet, or a run not sealed yet whatever its study (a smoke run
  // planned after the smoke study's reveal reuses the earlier one's pictures).
  const keep = new Set<string>()
  for (const other of runIds(env)) {
    if (other === run) continue
    const p = loadPlan(env, other)
    if (p && (!revealOf(env, p.study) || !loadItems(env, other))) for (const c of [...p.order, ...(p.reused ?? [])]) keep.add(c)
  }
  const cells = new Set([...plan.order, ...(plan.reused ?? [])].filter((c) => !keep.has(c)))
  const dir = path.join(env.outputs, '.lab', 'cells')
  let names: string[] = []
  try {
    names = fs.readdirSync(dir)
  } catch {
    names = []
  }
  const files = names.filter((n) => cells.has(n.split('_')[0]))
  let bytes = 0
  for (const f of files) {
    try {
      bytes += fs.statSync(path.join(dir, f)).size
    } catch {
      /* gone */
    }
  }
  if (!has(a, 'yes')) {
    say(`${files.length} original${files.length === 1 ? '' : 's'} of ${run} (${(bytes / 1e6).toFixed(0)} MB) under ${dir} can go. The blind copies and the report stay.`)
    say('A later run that needs the same pictures makes them again.')
    say(`Nothing was removed. Run lab/lab prune ${run} --yes to remove them.`)
    return 0
  }
  // A run being made adds to the list of finished pictures, which prune rewrites.
  const busy = liveDrivers(env)[0]
  if (busy) {
    say(`The ${busy.run} run is being made now, so nothing was removed. Prune once it has stopped.`)
    return 1
  }
  // The list first: if this stops halfway, a later plan makes a picture again
  // rather than reusing a file that is gone.
  forgetCells(env, cells)
  for (const f of files) {
    try {
      fs.unlinkSync(path.join(dir, f))
    } catch {
      /* gone */
    }
  }
  say(`Removed ${files.length} original${files.length === 1 ? '' : 's'} (${(bytes / 1e6).toFixed(0)} MB). A later run that needs the same pictures makes them again.`)
  return 0
}

function cmdSpawn(env: LabEnv, cmd: string, args: string[]): number {
  const r = spawnSync('npx', [cmd, ...args], { cwd: env.repoRoot, stdio: 'inherit' })
  return r.status ?? 1
}

// ----------------------------------------------------------------- main --

const HELP = `The model lab. Nothing is sent anywhere until you press Start.

  lab/lab check [--no-comfy]         the app never loads lab/; the suites; every graph against ComfyUI's node list (a read)
  lab/lab typecheck                  the lab's own type check
  lab/lab test                       the lab's own tests
  lab/lab plan <suite…> [--run <id>] plan a run and print its counts and estimate; sends nothing
                                     (one suite alone gets its usual run name: cal-1, core-1, exta-1, extb-1)
  lab/lab smoke [--run smoke-1]      plan one picture per operation at 8 steps; sends nothing
  lab/lab start <run> [--until HH:MM] [--skip-calibration]
                                     start sending (in the lab server if it runs, else in the background)
  lab/lab run <run> [--until HH:MM] [--skip-calibration]
                                     send in this terminal; Ctrl-C pauses
  lab/lab pause <run>                stop the group in progress and pause
  lab/lab status                     every run: made, failed, scored
  lab/lab refs [list]                the reference photos
  lab/lab refs add <file> [--name id]
  lab/lab refs mask <id> x,y,w,h     the area a region edit may change, in upright pixels
  lab/lab refs describe <id> <words…>
  lab/lab read <run>                 ask the app's picture reader about the run's pictures
  lab/lab seal <run>                 make the blind copies and the judging items
  lab/lab check-blind <run>          scan what the judge sees for anything sealed
  lab/lab serve                      the lab page on 127.0.0.1:5274
  lab/lab phone                      print the tailscale line that puts the page on your phone
  lab/lab report <run|study>         after the reveal: the study's report.html, findings.json, cells.csv
  lab/lab findings <run|study>       copy the study's findings.json into lab/findings/ (only when you ask)
  lab/lab prune <run> [--yes]        after the reveal: remove the run's original pictures

The suites: ${Object.keys(SUITES).join(', ')}.`

export async function main(argv: readonly string[]): Promise<number> {
  const [cmd, ...rest] = argv
  const a = parseArgs(rest)
  if (!cmd || cmd === 'help' || cmd === '--help' || cmd === '-h') {
    say(HELP)
    return 0
  }
  const env = labEnv()
  switch (cmd) {
    case 'check':
      return cmdCheck(env, a)
    case 'typecheck':
      return cmdSpawn(env, 'tsc', ['-p', 'lab/tsconfig.json'])
    case 'test':
      return cmdSpawn(env, 'vitest', ['run', '--config', 'lab/vitest.config.ts'])
    case 'plan':
      return cmdPlan(env, a)
    case 'smoke':
      return cmdSmoke(env, a)
    case 'start':
      return cmdStart(env, a)
    case 'run':
      return cmdRun(env, a)
    case 'pause':
      return cmdPause(env, a)
    case 'status':
      return cmdStatus(env)
    case 'refs':
      return cmdRefs(env, a)
    case 'read':
      return cmdRead(env, a)
    case 'seal':
      return cmdSeal(env, a)
    case 'check-blind':
      return cmdCheckBlind(env, a)
    case 'serve':
      return cmdServe(env)
    case 'phone':
      return cmdPhone(env)
    case 'report':
      return cmdReport(env, a)
    case 'findings':
      return cmdFindings(env, a)
    case 'prune':
      return cmdPrune(env, a)
    default:
      throw new Usage(`There is no command called ${cmd}. lab/lab help lists them.`)
  }
}

const invoked = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url : false
if (invoked) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (err) => {
      if (err instanceof Usage) {
        process.stderr.write(err.message + '\n')
        process.exit(2)
      }
      process.stderr.write(`${(err as Error)?.message ?? err}\n`)
      process.exit(1)
    },
  )
}
