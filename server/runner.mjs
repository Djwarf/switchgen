/**
 * SwitchGen's queue on the server, "the runner": waiting work goes on while
 * the phone sleeps.
 *
 * A page used to keep its own waiting work (the Video lane, the reel's walk
 * from shot to shot, a batch of pictures) and send each job to ComfyUI
 * itself, so a locked phone or a discarded tab stopped the work where it
 * stood. With the runner, a page hands a whole Make, Run or Render to this
 * process in one request and only watches after that; this process sends the
 * work to ComfyUI in turn, under the page's memory rules, files each result
 * in the archive, and tells every open page on every device what happened.
 *
 * It lives beside the archive, in the app server, in a folder of its own that
 * belongs to that archive, and runs only while this process holds both the
 * archive's lock and its folder's: a second server started on the same
 * outputs, or pointed at the same folder, stands back, says so in
 * /api/capabilities, and its pages send their work themselves, as they did
 * before there was a queue. So does every page when SWITCHGEN_RUNNER=off,
 * and while the ComfyUI it talks to is one older than the jobs list
 * (/api/jobs) the queue follows its work by.
 * A queue that is not running still answers its saved list, read-only, and
 * the queue that next runs holds whatever waited meanwhile until the reader
 * says.
 *
 * The pieces live in server/runner/: store.mjs keeps the list of work on
 * disk, engine.mjs decides what goes to ComfyUI and when, filing.mjs files
 * what comes back, routes.mjs answers /api/runner, and comfy.mjs and
 * comfyRecord.mjs speak to ComfyUI and read its answers.
 */
import path from 'node:path'
import { archiveApi } from './archive.mjs'
import { reqUrl, safely, send } from './guard.mjs'
import { createComfy } from './runner/comfy.mjs'
import { DESKS, REASONS, createEngine } from './runner/engine.mjs'
import { createRoutes } from './runner/routes.mjs'
import { endHandovers } from './runner/store.mjs'

/**
 * Vite loads this file afresh each time its config reloads, in the same
 * process, so the runner in service is kept here rather than in a module
 * variable. A new load waits for the one before to hand over before it reads
 * the list of work, so there is never a second dispatcher; the folder's lock
 * passes from one to the next in place, so another server on the same folder
 * never sees it free in between.
 */
const REGISTRY = Symbol.for('switchgen.runner')

function registry() {
  let reg = globalThis[REGISTRY]
  if (!reg || typeof reg !== 'object') {
    reg = globalThis[REGISTRY] = { latest: null, current: null, hooked: false }
  }
  return reg
}

/** The desks that send their work through the queue: SWITCHGEN_RUNNER_DESKS, or all three. */
function desksFromEnv() {
  const raw = process.env.SWITCHGEN_RUNNER_DESKS
  if (raw === undefined) return [...DESKS]
  return raw.split(',').map((d) => d.trim()).filter((d) => DESKS.includes(d))
}

/**
 * The queue's folder when SWITCHGEN_RUNNER_DIR does not name one: beside the
 * archive and named after it, so two archives in one folder never share a
 * list of work. The default archive keeps the folder it always had.
 */
export function runnerDirFor(archiveFile) {
  const name = path.basename(archiveFile)
  return path.join(path.dirname(archiveFile), name === 'archive.json' ? 'runner' : `${name}.runner`)
}

function settleFromEnv() {
  const n = Number(process.env.SWITCHGEN_RUNNER_SETTLE_MS)
  return process.env.SWITCHGEN_RUNNER_SETTLE_MS !== undefined && Number.isFinite(n) && n >= 0 ? n : 1000
}

/**
 * A queue over a folder of its own, a ComfyUI client and the archive.
 * Everything it touches comes in through `opts`, so a test can build one
 * against a scripted ComfyUI and step it with tick().
 */
export function createRunner(opts) {
  const engine = createEngine(opts)
  const routes = createRoutes(engine)
  engine.start()
  return {
    handler: routes.handler,
    tick: engine.tick,
    status: engine.status,
    snapshot: () => {
      engine.ensureActive()
      return engine.snapshot()
    },
    retire: async (o) => {
      await engine.retire(o)
      routes.close()
    },
    /** Close the socket to ComfyUI, as the process exits. */
    closeSocket: () => engine.closeSocket(),
  }
}

/** Close the runner's socket as the process ends. Its list of work needs nothing written: every commit is on disk already. */
function hookExit(reg) {
  if (reg.hooked) return
  reg.hooked = true
  process.once('exit', () => {
    try { void reg.current?.closeSocket?.() } catch { /* ending anyway */ }
  })
}

/** A client that refuses everything, for a runner that must not reach ComfyUI at all. */
const INERT = {
  url: '',
  readQueue: () => Promise.reject(new Error('the queue is off')),
  free: async () => 'unreached',
  submit: async () => ({ unreached: true }),
  getJob: () => Promise.reject(new Error('the queue is off')),
  history: () => Promise.reject(new Error('the queue is off')),
  cancel: async () => false,
  interrupt: async () => {},
  socket: () => ({ close() {} }),
}

export function switchgenRunner() {
  const serve = async (server) => {
    const reg = registry()
    const before = reg.latest
    const ready = (async () => {
      const old = await before?.catch(() => null)
      if (old) {
        try { await old.retire({ handover: true }) } catch (err) { console.warn(`[switchgen-runner] the queue before this one did not hand over cleanly: ${err?.message ?? err}`) }
      }
      const on = process.env.SWITCHGEN_RUNNER !== 'off'
      const opts = {
        dir: process.env.SWITCHGEN_RUNNER_DIR || runnerDirFor(archiveApi.archiveFile),
        archive: archiveApi,
        outputs: archiveApi.outputsRoot,
        desks: desksFromEnv(),
        settleMs: settleFromEnv(),
        enabled: on,
      }
      let runner
      try {
        runner = createRunner({ ...opts, comfy: on ? createComfy() : INERT })
      } catch (err) {
        console.warn(`[switchgen-runner] the queue could not start: ${err?.stack ?? err}; pages send their own work`)
        runner = createRunner({ ...opts, comfy: INERT, enabled: false })
      }
      // The new runner has taken the folder over, if it runs at all. One that
      // is off, stands back or could not start leaves it: let it go now, and
      // look again, so waiting work is noted as waiting on a queue that is off.
      endHandovers()
      try { runner.status() } catch { /* said when it is next asked */ }
      reg.current = runner
      return runner
    })()
    reg.latest = ready
    hookExit(reg)

    // Mounted at once, so its place in the middleware order is fixed; a
    // request that arrives before the queue has taken over waits for it.
    server.middlewares.use(
      safely(async (req, res, next) => {
        const url = reqUrl(req)
        if (!url) return send(res, 400, { error: 'the request path is not a valid URL' })
        const p = url.pathname
        if (p !== '/api/runner' && !p.startsWith('/api/runner/')) return next()
        const runner = await ready
        return runner.handler(req, res, next)
      }),
    )
    await ready
  }
  return {
    name: 'switchgen-runner',
    configureServer: serve,
    configurePreviewServer: serve,
  }
}

/**
 * The queue's own word on whether pages should hand it their work, for
 * /api/capabilities: active, the desks it takes, and why not when it is not.
 */
export function runnerStatus() {
  const current = globalThis[REGISTRY]?.current
  if (current) return current.status()
  return { active: false, desks: [], reason: REASONS.stopped }
}
