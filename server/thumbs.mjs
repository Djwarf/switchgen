/**
 * SwitchGen thumbnails: small pictures of the outputs, made once and kept.
 *
 * The archive grid and the picture picker used to load every original to fill
 * a cell a few hundred pixels wide. The renders here were PNGs of 1.26 MB at
 * the median when that was measured, and the picker asks for up to 120 of
 * them at once. One 1248x1824 render of 2,815,365 B came out at 58,842 B as
 * the 512-wide WebP below, made in about a tenth of a second.
 *
 * GET /api/thumb?rel=<path under the outputs folder>&w=<256|512|1024>
 *
 *   A picture is scaled down to that width, never up; a clip gives its first
 *   frame. Each is made by ffmpeg the first time it is asked for and kept
 *   under `<outputs>/.switchgen/thumbs`, keyed by the file's size and
 *   modification time. ComfyUI hands a deleted file's name to the next render,
 *   so a name alone would show the old picture for the new one; with the
 *   stamp in the key, a replaced file is a new thumbnail. The answer carries
 *   the same stamp as its ETag and `no-cache`, so a browser keeps its copy and
 *   asks each time whether it still holds, which costs one stat here.
 *
 *   When no thumbnail can be made (no ffmpeg, a file it cannot read), a
 *   picture is redirected to the original, which is what the page loaded
 *   before this existed. A clip is answered 503 instead: redirecting an image
 *   tag to a video would only download the clip to show nothing, and the
 *   archive grabs a clip's frame in the browser when this says no.
 *
 * Same posture as the other servers: the path is confined to the outputs
 * folder after resolving links, only the fixed widths are made, so the cache
 * holds at most three files per output, and no more than two ffmpeg run at
 * once, so a picker full of new pictures cannot take the machine from a
 * generation.
 */
import { createHash, randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { TOOLS, confineReal, reqUrl, safely, send, tools } from './guard.mjs'

const run = promisify(execFile)

const OUTPUTS = process.env.SWITCHGEN_OUTPUTS ?? '/mnt/storage/ai/outputs'
const THUMBS = process.env.SWITCHGEN_THUMBS ?? path.join(OUTPUTS, '.switchgen', 'thumbs')

/** The widths made. Keep in step with THUMB_WIDTHS in src/lib/thumbs.ts. */
const WIDTHS = [256, 512, 1024]
const IMAGE = /\.(png|jpe?g|webp|gif|avif|bmp)$/i
const VIDEO = /\.(webm|mp4|mkv|mov|m4v)$/i
/**
 * Part of every thumbnail's name and ETag. Change it when the recipe in
 * render() changes, and every thumbnail made the old way is made again.
 */
const RECIPE = 'r1'
const MAX_RUNNING = 2
/** Longest one ffmpeg may take before it is killed and the fallback served. */
const RENDER_MS = 30000
/** How long a file ffmpeg could not read is left alone before it is tried again. */
const RETRY_MS = 10 * 60000

// --------------------------------------------------------------- the cache --

/** The folder for one output's thumbnails, named for its path under the outputs root. */
function folderOf(rel) {
  const key = createHash('sha1').update(String(rel).split(path.sep).join('/')).digest('hex').slice(0, 24)
  return path.join(THUMBS, key)
}

async function exists(p) {
  try { await fs.access(p); return true } catch { return false }
}

/**
 * Remove every thumbnail made from `rel`, a path under the outputs root.
 * Best effort: the caller has already done what it came to do.
 */
export async function forgetThumbs(rel) {
  if (typeof rel !== 'string' || !rel) return
  try { await fs.rm(folderOf(rel), { recursive: true, force: true }) } catch { /* not ours to worry about */ }
}

/** Drop the thumbnails of this width made from an earlier file under the same name. */
async function prune(folder, width, keep) {
  let names
  try { names = await fs.readdir(folder) } catch { return }
  await Promise.all(names
    .filter(n => n.startsWith(`${width}-`) && n.endsWith('.webp') && n !== keep)
    .map(n => fs.unlink(path.join(folder, n)).catch(() => {})))
}

// ---------------------------------------------------------------- the work --

let running = 0
const waiting = []
/** One render per thumbnail, however many requests arrive for it at once. */
const making = new Map()
/** Thumbnails ffmpeg failed on, and when, so a bad file is not retried on every request. */
const failed = new Map()

function slot() {
  if (running < MAX_RUNNING) {
    running += 1
    return Promise.resolve()
  }
  return new Promise(resolve => waiting.push(resolve))
}

function release() {
  const next = waiting.shift()
  // Handed straight to the next in line, so the count does not change.
  if (next) next()
  else running -= 1
}

/**
 * One frame, scaled to fit `width` across and four times that down, as WebP.
 * Written beside its final name and renamed into place, so a reader never
 * finds half a file, and a render that fails leaves nothing behind.
 */
async function render(src, out, width) {
  if (await exists(out)) return true
  await fs.mkdir(path.dirname(out), { recursive: true })
  const part = `${out}.${randomUUID().slice(0, 8)}.part`
  const args = [
    '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
    '-i', src,
    '-map', '0:v:0', '-frames:v', '1', '-an', '-sn',
    '-vf', `scale='min(${width},iw)':'min(${width * 4},ih)':force_original_aspect_ratio=decrease`,
    '-c:v', 'libwebp', '-quality', '75',
    '-f', 'webp', part,
  ]
  try {
    await run(TOOLS.ffmpeg, args, { timeout: RENDER_MS, killSignal: 'SIGKILL' })
    await fs.rename(part, out)
  } catch (err) {
    await fs.unlink(part).catch(() => {})
    console.warn(`[switchgen-thumbs] no thumbnail for ${src}: ${String(err?.stderr || err?.message || err).trim().split('\n')[0]}`)
    return false
  }
  void prune(path.dirname(out), width, path.basename(out))
  return true
}

function make(src, out, width) {
  let job = making.get(out)
  if (!job) {
    job = (async () => {
      await slot()
      try { return await render(src, out, width) } finally { release() }
    })().finally(() => making.delete(out))
    making.set(out, job)
  }
  return job
}

/** Is there a thumbnail at `out` now, making it if it can be made? */
async function ensure(src, out, width) {
  if (await exists(out)) return true
  const at = failed.get(out)
  if (at !== undefined && Date.now() - at < RETRY_MS) return false
  if (!(await tools()).ffmpeg) return false
  const ok = await make(src, out, width)
  if (ok) failed.delete(out)
  else {
    if (failed.size > 5000) failed.clear()
    failed.set(out, Date.now())
  }
  return ok
}

// ------------------------------------------------------------------ answer --

/** Does If-None-Match name this ETag? A list and weak tags are both allowed there. */
function matches(header, etag) {
  if (!header) return false
  return String(header).split(',').some(t => {
    const tag = t.trim()
    return tag === '*' || tag.replace(/^W\//, '') === etag
  })
}

/**
 * What the page gets when there is no thumbnail. Never kept by a cache: the
 * next request should find the thumbnail if one can be made by then.
 */
function fallback(res, rel, video) {
  res.setHeader('Cache-Control', 'no-store')
  if (video) return send(res, 503, { error: 'no frame could be taken from this clip here' })
  const dir = path.posix.dirname(rel)
  const q = new URLSearchParams({
    filename: path.posix.basename(rel),
    subfolder: dir === '.' ? '' : dir,
    type: 'output',
  })
  // ComfyUI serves the original through the /comfy proxy in vite.config.ts.
  res.statusCode = 302
  res.setHeader('Location', `/comfy/view?${q}`)
  res.end()
}

export function switchgenThumbs() {
  const handler = async (req, res, next) => {
    const url = reqUrl(req)
    if (!url) return send(res, 400, { error: 'the request path is not a valid URL' })
    if (url.pathname !== '/api/thumb') return next()

    try {
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.setHeader('Allow', 'GET')
        return send(res, 405, { error: `/api/thumb takes GET, not ${req.method}` })
      }
      const width = Number(url.searchParams.get('w'))
      if (!WIDTHS.includes(width)) return send(res, 400, { error: `w must be one of ${WIDTHS.join(', ')}` })
      const rel = url.searchParams.get('rel') ?? ''
      // A name that starts with a dot is not an output: the archive lists
      // none, and this cache lives under one.
      if (!rel || rel.length > 1024 || rel.includes('\0') || rel.split(/[/\\]/).some(s => s.startsWith('.'))) {
        return send(res, 400, { error: 'rel must name a file under the outputs folder' })
      }
      const video = VIDEO.test(rel)
      if (!video && !IMAGE.test(rel)) return send(res, 415, { error: 'only pictures and clips have thumbnails' })

      const full = await confineReal(OUTPUTS, rel)
      if (!full) return send(res, 400, { error: 'path escapes the outputs folder' })
      let st
      try { st = await fs.stat(full) } catch { return send(res, 404, { error: 'not found' }) }
      if (!st.isFile()) return send(res, 404, { error: 'not a file' })

      // Keyed by the resolved path, so a link and the file it names share
      // their thumbnails, and a delete through either can drop them.
      const inside = path.relative(path.resolve(OUTPUTS), full).split(path.sep).join('/')
      const stamp = `${width}-${Math.trunc(st.mtimeMs)}-${st.size}-${RECIPE}`
      const etag = `"${stamp}"`
      if (matches(req.headers['if-none-match'], etag)) {
        res.statusCode = 304
        res.setHeader('ETag', etag)
        res.setHeader('Cache-Control', 'no-cache')
        return res.end()
      }

      const out = path.join(folderOf(inside), `${stamp}.webp`)
      if (!(await ensure(full, out, width))) return fallback(res, inside, video)
      let body
      // Pruned between the check and the read by a newer file's render.
      try { body = await fs.readFile(out) } catch { return fallback(res, inside, video) }
      res.statusCode = 200
      res.setHeader('Content-Type', 'image/webp')
      res.setHeader('Content-Length', body.length)
      res.setHeader('Cache-Control', 'no-cache')
      res.setHeader('ETag', etag)
      return res.end(req.method === 'HEAD' ? undefined : body)
    } catch (err) {
      if (res.headersSent) { try { res.end() } catch { /* gone */ } return }
      return send(res, 500, { error: String(err?.message ?? err) })
    }
  }

  return {
    name: 'switchgen-thumbs',
    configureServer(server) { server.middlewares.use(safely(handler)) },
    configurePreviewServer(server) { server.middlewares.use(safely(handler)) },
  }
}
