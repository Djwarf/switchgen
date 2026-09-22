/**
 * SwitchGen model catalogue + downloader, mounted next to the local API.
 *
 * The browser cannot stat the model tree, cannot read free RAM, and cannot run
 * aria2c, so the honest "will this model actually run here?" answer has to be
 * computed server-side. That question is the point of this file: we deleted the
 * Wan 2.2 14B pair after it was OOM-killed mid-generation (28.3 GB of weights
 * against 30.5 GB of total RAM), and a catalogue that hides that is worse than
 * no catalogue. Every verdict below quotes the real numbers it used.
 *
 * Local-only, same as api.mjs: every write path is confined to the models root,
 * and the HuggingFace token is passed to aria2c through a 0600 input file so it
 * never reaches argv, a log line, or a response body.
 */
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import crypto from 'node:crypto'
import { spawn, execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { fileURLToPath } from 'node:url'
import { TOOLS, confine, guardMutation, readBody, send, sse, sseOpen } from './guard.mjs'

const run = promisify(execFile)
const HERE = path.dirname(fileURLToPath(import.meta.url))

const MODELS = process.env.SWITCHGEN_MODELS ?? '/mnt/storage/ai/models'
const CATALOG = process.env.SWITCHGEN_CATALOG ?? path.join(HERE, 'catalog.json')
const ARIA2C = TOOLS.aria2c
const HF_TOKEN_FILE = process.env.HF_TOKEN_FILE ?? path.join(os.homedir(), '.cache/huggingface/token')
const WEIGHTS = /\.(safetensors|gguf|ckpt|pt|pth|sft|bin)$/i

const GIB = 1073741824
/** Working set a generation needs on top of the weights: samplers, latents, the
 *  Python process itself. Measured conservatively; used only to widen the gap. */
const WORKING_SET = 2 * GIB
/** No model may claim more than this share of *total* RAM. Above it the machine
 *  starts swapping or earlyoom fires, which is what killed Wan 2.2 14B. */
const RAM_CEILING = 0.85
/** Leave this much disk unclaimed after a download completes. */
const DISK_RESERVE = 8 * GIB

// ---------------------------------------------------------------- helpers ---

/** Sizes for humans: GB once it is really gigabytes, MB below that. */
const human = n => (n >= GIB ? (n / GIB).toFixed(1) + ' GB' : n >= 1048576 ? Math.round(n / 1048576) + ' MB' : n + ' B')

async function walk(dir, root = dir, out = []) {
  let entries
  try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return out }
  for (const e of entries) {
    const full = path.join(dir, e.name)
    if (e.isDirectory()) await walk(full, root, out)
    else if (WEIGHTS.test(e.name)) {
      try {
        const st = await fs.stat(full)
        out.push({ name: e.name, rel: path.relative(root, full), size: st.size })
      } catch { /* vanished mid-walk */ }
    }
  }
  return out
}

/** Filename -> {rel, size}, so a weight counts as installed wherever it sits in
 *  the tree. ComfyUI resolves by folder root, but users move files around and a
 *  re-download of something already on disk is the expensive mistake. */
let installedCache = { at: 0, byName: new Map() }
async function installedIndex(maxAgeMs = 4000) {
  if (Date.now() - installedCache.at < maxAgeMs) return installedCache.byName
  const files = await walk(MODELS)
  const byName = new Map()
  for (const f of files) {
    const prev = byName.get(f.name)
    if (!prev || f.size > prev.size) byName.set(f.name, f)
  }
  installedCache = { at: Date.now(), byName }
  return byName
}

let catalogCache = null
async function catalog() {
  if (catalogCache) return catalogCache
  const doc = JSON.parse(await fs.readFile(CATALOG, 'utf8'))
  doc.depsByName = new Map(doc.deps.map(d => [d.filename, d]))
  catalogCache = doc
  return catalogCache
}

async function diskFree(dir) {
  try {
    const { stdout } = await run('df', ['-B1', '--output=avail,size', dir])
    const [avail, size] = stdout.trim().split('\n')[1].trim().split(/\s+/).map(Number)
    return { free: avail, total: size }
  } catch { return null }
}

async function vramInfo() {
  try {
    const { stdout } = await run('nvidia-smi', [
      '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits',
    ])
    const [name, total, used] = stdout.trim().split('\n')[0].split(',').map(s => s.trim())
    return { name, total: +total * 1048576, used: +used * 1048576, free: (+total - +used) * 1048576 }
  } catch { return null }
}

/** The token is read on demand and never cached, logged, or echoed. */

/**
 * The HuggingFace token may only ever be sent to HuggingFace.
 *
 * Without this check, POST /api/download accepts a caller supplied {url, gated}
 * and hands the Bearer token to whatever host was named. The middleware has no
 * auth and the app is reachable across the tailnet, so that is a token
 * exfiltration path for any peer. Host is re-checked after redirects because
 * redirect: 'follow' would otherwise carry the header off-site.
 */
const TOKEN_HOSTS = new Set(['huggingface.co', 'cdn-lfs.huggingface.co', 'cdn-lfs-us-1.huggingface.co'])

function tokenAllowedFor(url) {
  try {
    const h = new URL(url).hostname.toLowerCase()
    return TOKEN_HOSTS.has(h) || h.endsWith('.huggingface.co')
  } catch { return false }
}

async function hfToken() {
  try {
    const t = (await fs.readFile(HF_TOKEN_FILE, 'utf8')).trim()
    return t || null
  } catch { return null }
}

// ------------------------------------------------------------- annotation ---

/** One dep, annotated with what is actually on this disk. */
function annotateDep(dep, index) {
  const hit = index.get(dep.filename)
  const want = dep.sizeBytes ?? null
  // A short file is a partial or a different quant under the same name; either
  // way it is not the file the graph expects, so do not call it installed.
  const complete = !!hit && (want == null || dep.sizeApprox || hit.size >= want * 0.995)
  return {
    filename: dep.filename,
    kind: dep.kind,
    dest: dep.dest,
    enumValue: dep.enumValue,
    url: dep.url,
    httpStatus: dep.httpStatus,
    sizeBytes: want,
    sizeApprox: !!dep.sizeApprox,
    gated: !!dep.gated,
    optional: !!dep.optional,
    note: dep.note || '',
    installed: complete,
    installedPath: hit ? hit.rel : null,
    installedBytes: hit ? hit.size : 0,
    partial: !!hit && !complete,
  }
}

/**
 * The entries of fam.models are alternative main weights — different quants or
 * variants of the same architecture — so exactly one of them is needed, not all.
 * Everything else in fam.deps (text encoders, VAEs, clip vision) is required.
 * Fetching every listed variant is how you turn a 12 GB model into a 36 GB one.
 */
function pickModel(fam, want, index) {
  const models = fam.models ?? []
  if (!models.length) return null
  if (want && models.includes(want)) return want
  const onDisk = models.find(m => index.has(m))
  if (onDisk) return onDisk
  for (const node of Object.values(fam.graph ?? {})) {
    if (['UNETLoader', 'UnetLoaderGGUF', 'CheckpointLoaderSimple', 'CheckpointLoader', 'ImageOnlyCheckpointLoader']
      .includes(node.class_type)) {
      const v = Object.values(node.inputs ?? {})[0]
      if (typeof v === 'string' && models.includes(v)) return v
    }
  }
  return models[0]
}

/** Every weight filename this family's validated graph actually loads. */
function graphWeights(fam) {
  const out = new Set()
  for (const node of Object.values(fam.graph ?? {})) {
    for (const v of Object.values(node.inputs ?? {})) {
      if (typeof v === 'string' && WEIGHTS.test(v)) { out.add(v); out.add(v.split('/').pop()) }
    }
  }
  return out
}

function annotateFamily(fam, cat, index, wantModel, include = []) {
  const chosen = pickModel(fam, wantModel, index)
  const fallbackIndex = new Map()
  const defaultModel = pickModel(fam, null, fallbackIndex)
  const models = new Set(fam.models ?? [])
  const inGraph = graphWeights(fam)
  const wanted = new Set(include)
  const all = fam.deps.map(n => cat.depsByName.get(n)).filter(Boolean).map(d => annotateDep(d, index))
  // The graph is the authority on what actually loads. A dep the graph never
  // names, and that is not the chosen main weight, is an alternate quant, a
  // speed LoRA or an aux model: offered, never fetched behind the user's back.
  const required = f => f.filename === chosen
    || wanted.has(f.filename)
    || (!models.has(f.filename) && (inGraph.has(f.enumValue) || inGraph.has(f.filename)) && !f.optional)
  const files = all.filter(required)
  const alternatives = all.filter(f => !required(f) && models.has(f.filename))
  const optional = all.filter(f => !required(f) && !models.has(f.filename))
  const missing = files.filter(f => !f.installed)
  const missingBytes = missing.reduce((a, f) => a + (f.sizeBytes ?? 0), 0)
  // requiresBytes was computed by the group agents for the family's default
  // weight. Pick a different variant and that number stops being true, so fall
  // back to the real sum of the files this configuration actually loads.
  const selectedBytes = files.reduce((a, f) => a + (f.sizeBytes ?? 0), 0)
  const resident = (chosen === defaultModel && fam.requiresBytes) ? fam.requiresBytes : selectedBytes
  return {
    chosenModel: chosen,
    defaultModel,
    residentBytes: resident,
    residentFrom: (chosen === defaultModel && fam.requiresBytes) ? 'family.requiresBytes' : 'sum of selected files',
    selectedBytes,
    files,
    alternatives,
    optional,
    installedCount: files.length - missing.length,
    fileCount: files.length,
    missing: missing.map(f => f.filename),
    missingBytes,
    ready: missing.length === 0 && !(fam.incomplete?.length),
    gatedMissing: missing.filter(f => f.gated).map(f => f.filename),
  }
}

// ------------------------------------------------------------ the fit gate ---

/**
 * Does this family actually run on this machine? requiresBytes is the resident
 * weight footprint the group agents computed; RAM is the thing that kills a
 * run, VRAM only makes it slow (ComfyUI offloads), disk only blocks the fetch.
 */
function fitVerdict(fam, ann, hw) {
  const need = ann.residentBytes || fam.requiresBytes || 0
  const { ramTotal, ramFree, disk, vram } = hw
  const reasons = []
  const blockers = []

  if (!need) {
    reasons.push('This family did not record a weight footprint, so RAM fit could not be checked.')
  } else if (need > ramTotal * RAM_CEILING) {
    blockers.push(
      `Weights need ${human(need)} resident, which is more than ${Math.round(RAM_CEILING * 100)}% of this machine's ` +
      `${human(ramTotal)} of total RAM. It would be OOM-killed mid-generation, the way the Wan 2.2 14B pair was.`)
  } else if (need + WORKING_SET > ramFree) {
    blockers.push(
      `Weights need ${human(need)} plus about ${human(WORKING_SET)} of working set, but only ${human(ramFree)} of RAM ` +
      `is free right now (of ${human(ramTotal)} total). Close something, or pick a smaller quantisation.`)
  } else {
    reasons.push(`Weights need ${human(need)}; ${human(ramFree)} of RAM is free of ${human(ramTotal)} total.`)
  }

  if (disk && ann.missingBytes) {
    if (ann.missingBytes + DISK_RESERVE > disk.free) {
      blockers.push(
        `The download is ${human(ann.missingBytes)} but only ${human(disk.free)} of disk is free ` +
        `(${human(DISK_RESERVE)} is held back as headroom).`)
    } else {
      reasons.push(`Download is ${human(ann.missingBytes)} against ${human(disk.free)} free disk.`)
    }
  } else if (!ann.missingBytes) {
    reasons.push('Every file this family needs is already on disk.')
  }

  if (fam.incomplete?.length) {
    blockers.push(
      `No verified download URL exists for ${fam.incomplete.join(', ')}, so this family cannot be completed ` +
      `automatically — that weight has to be placed by hand.`)
  }
  for (const f of ann.files) {
    if (!f.installed && f.httpStatus && f.httpStatus !== 200 && !f.gated) {
      blockers.push(`${f.filename} answered HTTP ${f.httpStatus} when its URL was last checked.`)
    }
  }

  if (vram && need) {
    if (need > vram.free) {
      reasons.push(
        `Weights (${human(need)}) exceed the ${human(vram.free)} free on the ${vram.name}, so ComfyUI will offload ` +
        `to RAM: it will run, just slower.`)
    } else {
      reasons.push(`Weights fit the ${human(vram.free)} free on the ${vram.name}.`)
    }
  }

  const fits = blockers.length === 0
  const gatedMissing = ann.gatedMissing.length
  const verdict = fits
    ? (ann.missing.length === 0
        ? `Ready to run: every file is already installed. ${reasons.join(' ')}`
        : `Fits this machine. ${ann.missing.length} file${ann.missing.length === 1 ? '' : 's'} to fetch, ` +
          `${human(ann.missingBytes)}. ${reasons.join(' ')}` +
          (gatedMissing
            ? ` ${gatedMissing} of them ${gatedMissing === 1 ? 'is' : 'are'} gated and need${gatedMissing === 1 ? 's' : ''} a HuggingFace token.`
            : ''))
    : `Will not run here. ${blockers.join(' ')}`

  return { fits, verdict, reasons, blockers }
}

// -------------------------------------------------------------- downloads ---

/** id -> live download record. Nothing here ever holds the token. */
const jobs = new Map()
/**
 * Plans in flight. A plan is one POST /api/download, sequential inside itself;
 * nothing capped how many could run at once, and each spawns aria2c at eight
 * connections. Two is enough to fetch a family while a LoRA lands.
 */
const MAX_ACTIVE_PLANS = 2
let activePlans = 0

function publicJob(j) {
  return {
    id: j.id,
    family: j.family,
    filename: j.filename,
    dest: j.dest,
    state: j.state,
    done: j.done,
    total: j.total,
    pct: j.total ? Math.min(1, j.done / j.total) : 0,
    speed: j.speed,
    etaSec: j.etaSec,
    fileIndex: j.fileIndex,
    fileCount: j.fileCount,
    startedAt: j.startedAt,
    error: j.error ?? null,
    gated: !!j.gated,
  }
}

/**
 * Content-Length for a URL, so progress is honest even when the catalogue size
 * is approximate. Uses Node's own fetch rather than shelling out to curl, so a
 * bearer token never appears in argv. Returns nulls rather than guessing.
 */
async function remoteSize(url, token) {
  const ac = new AbortController()
  const t = setTimeout(() => ac.abort(), 30000)
  try {
    // A one-byte ranged GET rather than a HEAD: undici drops Content-Length from
    // HEAD responses, but Content-Range on a 206 carries the true total.
    const headers = { Range: 'bytes=0-0' }
      // Defence in depth: callers can no longer set `gated`, but a catalogue entry
      // could still name a non-HuggingFace host. Never attach the token to one, and
      // never let a redirect replay it somewhere else.
      const mayAuth = !!token && tokenAllowedFor(url)
      if (mayAuth) headers.Authorization = `Bearer ${token}`
      let r = await fetch(url, {
        method: 'GET', redirect: mayAuth ? 'manual' : 'follow', signal: ac.signal, headers,
      })
      for (let hop = 0; mayAuth && r.status >= 300 && r.status < 400 && hop < 5; hop++) {
        const loc = r.headers.get('location')
        if (!loc) break
        const next = new URL(loc, url).toString()
        const h2 = { Range: 'bytes=0-0' }
        if (tokenAllowedFor(next)) h2.Authorization = `Bearer ${token}`
        r = await fetch(next, { method: 'GET', redirect: 'manual', signal: ac.signal, headers: h2 })
      }
    let size = null
    const cr = r.headers.get('content-range')
    const m = cr && /\/(\d+)\s*$/.exec(cr)
    if (m) size = Number(m[1])
    if (size == null) {
      const len = r.headers.get('content-length') ?? r.headers.get('x-linked-size')
      if (len && r.status === 200) size = Number(len)
    }
    try { await r.body?.cancel() } catch {}
    return { size, status: r.status === 206 ? 200 : r.status }
  } catch {
    return { size: null, status: null }
  } finally { clearTimeout(t) }
}

/**
 * Fetch one file with aria2c. Resumable (-c) with 8 connections; the URL and any
 * Authorization header go through a 0600 input file so the token cannot be read
 * out of `ps`. Progress comes from stat()ing the part file rather than scraping
 * aria2c's console, which gives exact byte counts.
 */
function fetchFile(job, onProgress) {
  return new Promise((resolve, reject) => { (async () => {
    const full = confine(MODELS, job.dest)
    if (!full) return reject(new Error('destination escapes the models root'))
    await fs.mkdir(path.dirname(full), { recursive: true })

    const token = job.gated ? await hfToken() : null
    if (job.gated && !token) {
      return reject(new Error(`${job.filename} is gated and no HuggingFace token was found at ${HF_TOKEN_FILE}`))
    }

    const listFile = path.join(os.tmpdir(), `switchgen-${job.id}.aria2in`)
    const lines = [job.url, `  dir=${path.dirname(full)}`, `  out=${path.basename(full)}`]
    if (token) lines.push(`  header=Authorization: Bearer ${token}`)
    await fs.writeFile(listFile, lines.join('\n') + '\n', { mode: 0o600 })

    const child = spawn(ARIA2C, [
      '-i', listFile,
      '-x8', '-s8', '-c',
      '--auto-file-renaming=false',
      '--allow-overwrite=false',
      '--console-log-level=error',
      '--summary-interval=0',
      '--file-allocation=none',
    ], { stdio: ['ignore', 'pipe', 'pipe'] })
    job.pid = child.pid
    job.state = 'downloading'

    let stderr = ''
    child.stderr.on('data', d => {
      stderr = (stderr + d.toString().replace(/Bearer\s+\S+/gi, 'Bearer [redacted]')).slice(-4000)
    })
    child.stdout.on('data', () => {})

    const samples = []
    const poll = setInterval(async () => {
      try {
        const st = await fs.stat(full)
        const now = Date.now()
        samples.push({ t: now, b: st.size })
        while (samples.length > 6) samples.shift()
        const first = samples[0]
        const dt = (now - first.t) / 1000
        job.done = st.size
        job.speed = dt > 0.4 ? Math.max(0, (st.size - first.b) / dt) : job.speed
        job.etaSec = job.speed > 0 && job.total ? Math.max(0, Math.round((job.total - st.size) / job.speed)) : null
        onProgress?.(job)
      } catch { /* aria2c has not created the file yet */ }
    }, 500)

    const finish = async (err) => {
      clearInterval(poll)
      try { await fs.unlink(listFile) } catch {}
      job.pid = null
      if (err) return reject(err)
      try {
        const st = await fs.stat(full)
        job.done = st.size
        if (job.total && st.size < job.total * 0.995) {
          return reject(new Error(`${job.filename} stopped at ${st.size} of ${job.total} bytes`))
        }
      } catch { return reject(new Error(`${job.filename} was not written`)) }
      resolve()
    }

    child.on('error', e => finish(new Error(`aria2c could not start: ${e.message}`)))
    child.on('close', code => {
      if (job.state === 'cancelled') return finish(new Error('cancelled'))
      if (code === 0) return finish(null)
      // aria2c's own message is safe to surface: the token lives in the input
      // file, never in argv or its log lines.
      finish(new Error(`aria2c exited ${code}${stderr.trim() ? `: ${stderr.trim().split('\n').pop()}` : ''}`))
    })
  })().catch(reject) })
}

/** Run a queue of files through aria2c, streaming SSE for each. */
async function runPlan(res, queue, familyId, mine = new Set()) {
  const ids = []
  let bytes = 0
  for (let i = 0; i < queue.length; i++) {
    const f = queue[i]
    const id = crypto.randomUUID()
    ids.push(id)
    const job = {
      id, family: familyId, filename: f.filename, dest: f.dest, url: f.url,
      gated: !!f.gated, total: f.sizeBytes ?? 0, done: 0, speed: 0, etaSec: null,
      state: 'starting', startedAt: Date.now(), fileIndex: i + 1, fileCount: queue.length,
      error: null, pid: null,
    }
    jobs.set(id, job)
    mine.add(id)
    sse(res, 'start', publicJob(job))

    try {
      // Trust the wire over the catalogue for the progress denominator.
      const token = job.gated && tokenAllowedFor(job.url) ? await hfToken() : null
      const { size, status } = await remoteSize(job.url, token)
      if (size) job.total = size
      if (status && status >= 400) throw new Error(`${job.filename} answered HTTP ${status}`)
      const already = await fs.stat(confine(MODELS, job.dest)).catch(() => null)
      if (already && job.total && already.size >= job.total) {
        job.done = already.size
        job.state = 'done'
        bytes += job.done
        sse(res, 'skip', publicJob(job))
        jobs.delete(id)
        continue
      }
      await fetchFile(job, j => sse(res, 'progress', publicJob(j)))
      job.state = 'done'
      bytes += job.done
      sse(res, 'file', publicJob(job))
    } catch (err) {
      job.state = job.state === 'cancelled' ? 'cancelled' : 'error'
      job.error = String(err?.message ?? err)
      sse(res, 'error', publicJob(job))
      jobs.delete(id)
      try { res.end() } catch {}
      return
    }
    jobs.delete(id)
    installedCache = { at: 0, byName: new Map() }
  }
  sse(res, 'done', { ids, family: familyId, files: queue.map(f => f.filename), bytes })
  try { res.end() } catch {}
}

// ---------------------------------------------------------------- handler ---

export const downloadsMiddleware = async (req, res, next) => {
  const url = new URL(req.url, 'http://local')
  const p = url.pathname
  if (!p.startsWith('/api/catalog') && !p.startsWith('/api/download')) return next()

  try {
    // ---- GET /api/catalog -------------------------------------------------
    if (p === '/api/catalog' && req.method === 'GET') {
      const [cat, index] = await Promise.all([catalog(), installedIndex()])
      const slim = url.searchParams.get('slim') === '1'
      const families = cat.families.map(fam => {
        const ann = annotateFamily(fam, cat, index, null)
        const base = slim
          ? { id: fam.id, label: fam.label, mode: fam.mode, group: fam.group, models: fam.models,
              verified: fam.verified, requiresBytes: fam.requiresBytes, deps: fam.deps,
              notes: fam.notes, incomplete: fam.incomplete }
          : fam
        return { ...base, installed: ann, requiresBytes: fam.requiresBytes }
      })
      const deps = cat.deps.map(d => annotateDep(d, index))
      return send(res, 200, {
        root: MODELS,
        generatedAt: cat.generatedAt,
        counts: {
          families: families.length,
          ready: families.filter(f => f.installed.ready).length,
          deps: deps.length,
          depsInstalled: deps.filter(d => d.installed).length,
        },
        families,
        deps,
      })
    }

    // ---- GET /api/catalog/plan?family=id ----------------------------------
    if (p === '/api/catalog/plan' && req.method === 'GET') {
      const id = url.searchParams.get('family')
      if (!id) return send(res, 400, { error: 'family is required' })
      const [cat, index] = await Promise.all([catalog(), installedIndex()])
      const fam = cat.families.find(f => f.id === id)
      if (!fam) return send(res, 404, { error: `no family "${id}"` })

      const ann = annotateFamily(fam, cat, index, url.searchParams.get('model'),
        (url.searchParams.get('include') ?? '').split(',').filter(Boolean))
      const [disk, vram] = await Promise.all([diskFree(MODELS), vramInfo()])
      const hw = { ramTotal: os.totalmem(), ramFree: os.freemem(), disk, vram }
      const fit = fitVerdict(fam, ann, hw)
      const missing = ann.files.filter(f => !f.installed)
      const token = await hfToken()

      return send(res, 200, {
        family: { id: fam.id, label: fam.label, mode: fam.mode, group: fam.group, models: fam.models, notes: fam.notes },
        chosenModel: ann.chosenModel,
        defaultModel: ann.defaultModel,
        residentBytes: ann.residentBytes,
        residentFrom: ann.residentFrom,
        alternatives: ann.alternatives.map(f => ({
          filename: f.filename, sizeBytes: f.sizeBytes, gated: f.gated, installed: f.installed,
          note: f.note, reason: 'another main weight for this family; pick one with ?model=',
        })),
        optional: ann.optional.map(f => ({
          filename: f.filename, kind: f.kind, sizeBytes: f.sizeBytes, gated: f.gated,
          installed: f.installed, note: f.note,
          reason: 'not referenced by this family\u2019s validated graph; add it with ?include=',
        })),
        requiresBytes: fam.requiresBytes,
        files: ann.files,
        download: missing.map(f => ({
          filename: f.filename, dest: f.dest, url: f.url, sizeBytes: f.sizeBytes,
          sizeApprox: f.sizeApprox, gated: f.gated, kind: f.kind,
          resumingFrom: f.partial ? f.installedBytes : 0,
        })),
        totalBytes: ann.missingBytes,
        alreadyInstalled: ann.files.filter(f => f.installed).map(f => f.filename),
        incomplete: fam.incomplete ?? [],
        gated: { files: ann.gatedMissing, tokenPresent: !!token },
        hardware: {
          ramTotal: hw.ramTotal, ramFree: hw.ramFree,
          diskFree: disk?.free ?? null, diskTotal: disk?.total ?? null,
          vram: vram ? { name: vram.name, total: vram.total, free: vram.free } : null,
          cpu: os.cpus()[0]?.model?.trim() ?? 'unknown',
        },
        fits: fit.fits,
        verdict: fit.verdict,
        reasons: fit.reasons,
        blockers: fit.blockers,
      })
    }

    // ---- GET /api/download/status -----------------------------------------
    if (p === '/api/download/status' && req.method === 'GET') {
      return send(res, 200, { downloads: [...jobs.values()].map(publicJob) })
    }

    // ---- POST /api/download/cancel ----------------------------------------
    if (p === '/api/download/cancel' && req.method === 'POST') {
      if (!guardMutation(req, res)) return
      const b = await readBody(req)
      if (!b || typeof b.id !== 'string') return send(res, 400, { error: 'id must be a string' })
      const job = jobs.get(b.id)
      if (!job) return send(res, 404, { error: 'no such download' })
      job.state = 'cancelled'
      if (job.pid) { try { process.kill(job.pid, 'SIGTERM') } catch {} }
      // Give aria2c a moment to release the file, then drop the partial and its
      // control file unless the caller wants to resume later.
      const full = confine(MODELS, job.dest)
      let removed = []
      if (full && b.keepPartial !== true) {
        await new Promise(r => setTimeout(r, 400))
        for (const f of [full, full + '.aria2']) {
          try { await fs.unlink(f); removed.push(path.relative(MODELS, f)) } catch {}
        }
      }
      jobs.delete(b.id)
      return send(res, 200, { cancelled: b.id, removed, keptPartial: b.keepPartial === true })
    }

    // ---- POST /api/download -----------------------------------------------
    if (p === '/api/download' && req.method === 'POST') {
      if (!guardMutation(req, res)) return
      const b = await readBody(req)
      if (!b) return send(res, 400, { error: 'body must be JSON under 1 MB' })
      if (activePlans >= MAX_ACTIVE_PLANS) {
        return send(res, 429, { error: `${MAX_ACTIVE_PLANS} downloads are already running; wait for one to finish` })
      }

      let queue = []
      let familyId = null
      let famIncomplete = []

      if (typeof b.family === 'string') {
        const [cat, index] = await Promise.all([catalog(), installedIndex()])
        const fam = cat.families.find(f => f.id === b.family)
        if (!fam) return send(res, 404, { error: `no family "${b.family}"` })
        familyId = fam.id
        const ann = annotateFamily(fam, cat, index, typeof b.model === 'string' ? b.model : null,
          Array.isArray(b.include) ? b.include.filter(x => typeof x === 'string') : [])
        const [disk, vram] = await Promise.all([diskFree(MODELS), vramInfo()])
        const fit = fitVerdict(fam, ann, { ramTotal: os.totalmem(), ramFree: os.freemem(), disk, vram })
        if (!fit.fits && b.force !== true) {
          return send(res, 409, { error: 'this family does not fit this machine', fits: false, verdict: fit.verdict, blockers: fit.blockers })
        }
        famIncomplete = fam.incomplete ?? []
        queue = ann.files.filter(f => !f.installed).map(f => ({
          filename: f.filename, dest: f.dest, url: f.url, sizeBytes: f.sizeBytes, gated: f.gated,
        }))
      } else if (typeof b.url === 'string') {
        if (!/^https:\/\//.test(b.url)) return send(res, 400, { error: 'url must be https' })
        const filename = typeof b.filename === 'string' && b.filename ? b.filename : path.basename(new URL(b.url).pathname)
        const dest = typeof b.dest === 'string' && b.dest
          ? (b.dest.endsWith('/') || !path.extname(b.dest) ? path.posix.join(b.dest, filename) : b.dest)
          : filename
        if (!confine(MODELS, dest)) return send(res, 400, { error: 'dest escapes the models root' })
        const cat = await catalog()
        const known = cat.depsByName.get(path.basename(dest))
        queue = [{ filename, dest, url: b.url, sizeBytes: known?.sizeBytes ?? null, gated: !!known?.gated /* never trust the caller: this decides whether a token is attached */ }]
        // The family path is checked by fitVerdict. A bare URL gets the same
        // disk check, using the wire size when the catalogue has none, so a
        // LoRA fetch cannot fill the disk that the next generation needs.
        const disk = await diskFree(MODELS)
        let bytes = known?.sizeBytes ?? null
        if (bytes == null) {
          const token = queue[0].gated && tokenAllowedFor(b.url) ? await hfToken() : null
          bytes = (await remoteSize(b.url, token)).size
        }
        if (disk && bytes != null && bytes + DISK_RESERVE > disk.free) {
          return send(res, 409, {
            error: `The download is ${human(bytes)} but only ${human(disk.free)} of disk is free ` +
                   `(${human(DISK_RESERVE)} is held back as headroom).`,
            fits: false,
          })
        }
      } else {
        return send(res, 400, { error: 'send {family} or {url, dest, filename}' })
      }

      const clash = queue.find(f => [...jobs.values()].some(j => j.dest === f.dest))
      if (clash) {
        return send(res, 409, { error: `${clash.filename} is already downloading`, dest: clash.dest })
      }

      if (!queue.length) {
        if (famIncomplete.length) {
          return send(res, 409, {
            error: 'this family cannot be completed automatically',
            incomplete: famIncomplete,
            detail: `No verified download URL exists for ${famIncomplete.join(', ')}; that weight has to be placed ` +
                    `under the models root by hand.`,
          })
        }
        return send(res, 200, { nothingToDo: true, family: familyId })
      }
      for (const f of queue) {
        if (!f.url) return send(res, 409, { error: `no verified URL for ${f.filename}; it must be placed by hand` })
      }

      sseOpen(res)
      sse(res, 'plan', { family: familyId, files: queue.map(f => ({ filename: f.filename, dest: f.dest, sizeBytes: f.sizeBytes, gated: f.gated })) })
      const mine = new Set()
      req.on('close', () => {
        for (const j of jobs.values()) {
          if (mine.has(j.id) && j.state === 'downloading' && j.pid) {
            // the client hung up; leave the partial so -c can resume it
            j.state = 'cancelled'
            try { process.kill(j.pid, 'SIGTERM') } catch {}
          }
        }
      })
      activePlans += 1
      try {
        await runPlan(res, queue, familyId, mine)
      } finally {
        activePlans -= 1
      }
      return
    }

    return send(res, 404, { error: 'no such endpoint' })
  } catch (err) {
    if (res.headersSent) { try { res.end() } catch {} ; return }
    return send(res, 500, { error: String(err?.message ?? err) })
  }
}

/** Vite plugin wrapper, matching switchgenApi()'s shape. */
export function switchgenDownloads() {
  return {
    name: 'switchgen-downloads',
    configureServer(server) { server.middlewares.use(downloadsMiddleware) },
    configurePreviewServer(server) { server.middlewares.use(downloadsMiddleware) },
  }
}

export default downloadsMiddleware
