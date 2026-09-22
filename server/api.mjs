/**
 * SwitchGen local API, mounted into Vite's dev and preview servers.
 *
 * Exists because the browser cannot stat files or free disk, and ComfyUI's
 * /experiment/models endpoint silently omits .gguf files, so model footprints
 * computed from it would be wrong for exactly the quantised models most likely
 * to strain a machine. Local-only; every path is confined to the roots below,
 * and the confinement is re-checked after realpath so a symlink planted in a
 * root cannot reach outside it.
 */
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { confineReal, guardMutation, readBody, send, tools } from './guard.mjs'

const run = promisify(execFile)

const MODELS = process.env.SWITCHGEN_MODELS ?? '/mnt/storage/ai/models'
const OUTPUTS = process.env.SWITCHGEN_OUTPUTS ?? '/mnt/storage/ai/outputs'
const WEIGHTS = /\.(safetensors|gguf|ckpt|pt|pth|sft|bin)$/i

/** Longest a probe of the machine may hang before we give up on it. */
const PROBE_MS = 5000

async function walk(dir, root = dir, out = []) {
  let entries
  try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return out }
  for (const e of entries) {
    const full = path.join(dir, e.name)
    if (e.isDirectory()) await walk(full, root, out)
    else if (WEIGHTS.test(e.name)) {
      try {
        const st = await fs.stat(full)
        out.push({
          name: e.name,
          rel: path.relative(root, full),
          folder: path.relative(root, dir) || '.',
          size: st.size,
          mtime: st.mtimeMs,
        })
      } catch { /* vanished mid-walk */ }
    }
  }
  return out
}

async function gpuInfo() {
  try {
    const { stdout } = await run('nvidia-smi', [
      '--query-gpu=name,memory.total,memory.used,memory.free',
      '--format=csv,noheader,nounits',
    ], { timeout: PROBE_MS, killSignal: 'SIGKILL' })
    const [name, total, used, free] = stdout.trim().split('\n')[0].split(',').map(s => s.trim())
    return { name, vramTotal: +total * 1048576, vramUsed: +used * 1048576, vramFree: +free * 1048576 }
  } catch {
    return null
  }
}

async function diskFree(dir) {
  try {
    const { stdout } = await run('df', ['-B1', '--output=avail,size', dir], { timeout: PROBE_MS, killSignal: 'SIGKILL' })
    const [avail, size] = stdout.trim().split('\n')[1].trim().split(/\s+/).map(Number)
    return { free: avail, total: size }
  } catch { return null }
}

/** Per-core CPU busy fraction, from the delta between two samples. */
let lastCpu = null
function cpuSample() {
  const now = os.cpus().map(c => {
    const t = c.times
    return { idle: t.idle, total: t.user + t.nice + t.sys + t.idle + t.irq }
  })
  let overall = 0
  const perCore = now.map((c, i) => {
    const prev = lastCpu?.[i]
    if (!prev) return 0
    const dTotal = c.total - prev.total
    const dIdle = c.idle - prev.idle
    return dTotal > 0 ? Math.max(0, Math.min(1, 1 - dIdle / dTotal)) : 0
  })
  lastCpu = now
  overall = perCore.length ? perCore.reduce((a, b) => a + b, 0) / perCore.length : 0
  return { overall, perCore }
}

/** One nvidia-smi call, with the extra live fields a status readout wants. */
async function gpuLive() {
  try {
    const { stdout } = await run('nvidia-smi', [
      '--query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw,power.limit,clocks.sm,fan.speed',
      '--format=csv,noheader,nounits',
    ], { timeout: PROBE_MS, killSignal: 'SIGKILL' })
    const p = stdout.trim().split('\n')[0].split(',').map(s => s.trim())
    const num = v => (v === '[N/A]' || v === '' ? null : Number(v))
    return {
      name: p[0],
      utilGpu: num(p[1]),
      utilMem: num(p[2]),
      vramTotal: num(p[3]) * 1048576,
      vramUsed: num(p[4]) * 1048576,
      vramFree: (num(p[3]) - num(p[4])) * 1048576,
      tempC: num(p[5]),
      powerW: num(p[6]),
      powerLimitW: num(p[7]),
      clockMhz: num(p[8]),
      fanPct: num(p[9]),
    }
  } catch { return null }
}

async function snapshot() {
  const [gpu, disk] = await Promise.all([gpuLive(), diskFree(MODELS)])
  const total = os.totalmem()
  const free = os.freemem()
  return {
    t: Date.now(),
    cpu: { ...cpuSample(), cores: os.cpus().length, model: os.cpus()[0]?.model?.trim() ?? 'unknown', load: os.loadavg() },
    ram: { total, free, used: total - free },
    gpu,
    disk,
    uptime: os.uptime(),
  }
}

/**
 * What this server can do, for a client deciding whether to offer a feature.
 *
 * The archive used to infer this by probing a path it knew nothing served and
 * checking whether the answer was JSON. Nothing served that path, so the answer
 * was Vite's SPA fallback: HTML, every time, and file deletion was switched off
 * on every page load while POST /api/delete sat there working. This is the real
 * answer to that question.
 *
 * `downloads` and `stitch` used to be declared true on the grounds that
 * switchgenDownloads() and switchgenReel() are mounted alongside this plugin.
 * Mounted is not the same as working: those two spawn aria2c and ffmpeg, and a
 * machine without them answered "yes" to a question it had never asked. Each
 * is now the result of running the binary, cached for a minute in guard.mjs.
 */
async function capabilities() {
  const t = await tools()
  return {
    server: 'switchgen',
    deleteFiles: true,
    stitch: !!(t.ffmpeg && t.ffprobe),
    downloads: !!t.aria2c,
    models: true,
    hardware: true,
    hardwareStream: true,
    // Served by switchgenArchive(), registered beside this plugin. It spawns
    // nothing, so mounted and working are the same thing for it.
    archive: true,
    gpu: t.gpu,
    tools: { aria2c: t.aria2c, ffmpeg: t.ffmpeg, ffprobe: t.ffprobe },
  }
}

/** Paths this middleware owns. Anything else under /api/ belongs elsewhere. */
const OWNED = ['/api/capabilities', '/api/hardware', '/api/models', '/api/delete']
/** The method each owned path answers, so a wrong one gets 405 and not HTML. */
const METHODS = new Map([
  ['/api/capabilities', 'GET'],
  ['/api/hardware', 'GET'],
  ['/api/hardware/stream', 'GET'],
  ['/api/models', 'GET'],
  ['/api/delete', 'POST'],
])

export function switchgenApi() {
  const handler = async (req, res, next) => {
    const url = new URL(req.url, 'http://local')
    if (!url.pathname.startsWith('/api/')) return next()

    try {
      if (url.pathname === '/api/capabilities' && req.method === 'GET') {
        return send(res, 200, { ...(await capabilities()), roots: { models: MODELS, outputs: OUTPUTS } })
      }

      if (url.pathname === '/api/hardware' && req.method === 'GET') {
        const cpus = os.cpus()
        const [gpu, disk] = await Promise.all([gpuInfo(), diskFree(MODELS)])
        return send(res, 200, {
          cpu: { cores: cpus.length, model: cpus[0]?.model?.trim() ?? 'unknown' },
          ram: { total: os.totalmem(), free: os.freemem() },
          gpu,
          disk,
          platform: `${os.type()} ${os.release()}`,
        })
      }

      // Real-time device status. One SSE stream per viewer, torn down on
      // disconnect so a closed tab cannot leave nvidia-smi polling forever.
      //
      // The teardown listens on the response, not the request. An
      // IncomingMessage is destroyed once its message is complete, which for a
      // bodyless GET can be immediately: its 'close' is not the client going
      // away. The response stays open until we end it, so its 'close' is.
      if (url.pathname === '/api/hardware/stream' && req.method === 'GET') {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
          'X-Accel-Buffering': 'no',
        })
        cpuSample() // prime the delta so the first real sample is meaningful
        let closed = false
        let inFlight = false
        const tick = async () => {
          if (closed || inFlight) return
          inFlight = true
          try {
            const data = await snapshot()
            if (closed || res.writableEnded) return
            res.write(`data: ${JSON.stringify(data)}\n\n`)
          } catch { stop() } finally { inFlight = false }
        }
        const everyMs = Math.min(5000, Math.max(250, Number(url.searchParams.get('ms')) || 1000))
        const timer = setInterval(tick, everyMs)
        const stop = () => {
          if (closed) return
          closed = true
          clearInterval(timer)
          res.off('close', stop)
          res.off('error', stop)
          try { res.end() } catch { /* already gone */ }
        }
        res.on('close', stop)
        res.on('error', stop)
        void tick()
        return
      }

      if (url.pathname === '/api/models' && req.method === 'GET') {
        return send(res, 200, { root: MODELS, files: await walk(MODELS) })
      }

      // Deleting a model or an output is destructive, so it is POST-with-intent
      // rather than a bare DELETE that a stray prefetch could trigger.
      if (url.pathname === '/api/delete' && req.method === 'POST') {
        if (!guardMutation(req, res)) return
        const b = await readBody(req)
        if (!b) return send(res, 400, { error: 'body must be JSON under 1 MB' })
        const { kind, rel } = b
        const root = kind === 'output' ? OUTPUTS : kind === 'model' ? MODELS : null
        if (!root || typeof rel !== 'string' || !rel.trim()) {
          return send(res, 400, { error: 'kind must be "model" or "output", rel must be a non-empty string' })
        }
        // unlink follows symlinks, so a lexical check is not enough: resolve
        // the real path and require that to land back inside the root.
        const full = await confineReal(root, rel)
        if (!full) return send(res, 400, { error: 'path escapes root' })
        let st
        try { st = await fs.lstat(full) } catch { return send(res, 404, { error: 'not found' }) }
        if (!st.isFile()) return send(res, 400, { error: 'not a regular file' })
        await fs.unlink(full)
        // Report what was actually removed. If `rel` named a link, the file
        // that went is the one it pointed at, and saying so is the honest answer.
        return send(res, 200, { deleted: path.relative(path.resolve(root), full), requested: rel, freed: st.size })
      }

      // Unknown paths inside our own namespace answer JSON, not Vite's SPA
      // fallback. A client cannot tell "this server does not do that" from
      // "this is not that server" if the 404 arrives as HTML.
      const method = METHODS.get(url.pathname)
      if (method && method !== req.method) {
        res.setHeader('Allow', method)
        return send(res, 405, { error: `${url.pathname} takes ${method}, not ${req.method}` })
      }
      if (OWNED.some(p => url.pathname === p || url.pathname.startsWith(p + '/'))) {
        return send(res, 404, { error: `no such endpoint: ${req.method} ${url.pathname}` })
      }

      // Not ours, and not in our namespace. Hand off: other middleware (the
      // downloader, the reel) owns /api/* routes of its own, and swallowing
      // unknown paths here would 404 them.
      return next()
    } catch (err) {
      if (res.headersSent) { try { res.end() } catch { /* gone */ } return }
      return send(res, 500, { error: String(err?.message ?? err) })
    }
  }

  return {
    name: 'switchgen-api',
    configureServer(server) { server.middlewares.use(handler) },
    configurePreviewServer(server) { server.middlewares.use(handler) },
  }
}
