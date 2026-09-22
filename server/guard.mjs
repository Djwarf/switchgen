/**
 * What every SwitchGen middleware shares, and the one check none of them had.
 *
 * Four servers grew four byte-identical copies of `send`, `confine`,
 * `confineReal` and `readBody`. They live here now, once. Alongside them is
 * the guard that was missing from all four: Vite binds this app to every
 * interface and allows the whole tailnet, and ComfyUI's own CORS handling runs
 * AFTER plugin middleware, so a page on any other origin could POST to
 * /api/delete with a text/plain body and unlink model weights. Nothing here
 * adds a login; the app is for a private network. It adds the two checks a
 * browser lets a server make for free: the request came from this server's
 * own pages, and the body is the type the route reads.
 *
 * `tools()` is the other thing this file owns. /api/capabilities used to
 * declare `downloads: true` and `stitch: true` by construction, on the grounds
 * that the middlewares were mounted. Mounted is not the same as working: the
 * binaries they spawn may not be installed. So each is probed, and the answer
 * is the probe's.
 */
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { isIP } from 'node:net'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { gzipSync } from 'node:zlib'

const run = promisify(execFile)

// ---------------------------------------------------------------- plumbing --

/** Below this a body is not worth the compressor's time or the header bytes. */
const GZIP_MIN = 1024

/**
 * Does the client take gzip? `gzip;q=0` is a refusal, not an offer, and `*`
 * offers every coding the header does not name.
 */
function takesGzip(req) {
  const accept = String(req?.headers?.['accept-encoding'] ?? '').toLowerCase()
  let star = false
  for (const part of accept.split(',')) {
    const [name, ...params] = part.split(';').map(s => s.trim())
    const q = params.find(p => p.startsWith('q='))
    const offered = q === undefined || Number(q.slice(2)) > 0
    if (name === 'gzip' || name === 'x-gzip') return offered
    if (name === '*') star = offered
  }
  return star
}

/**
 * Answer with JSON, gzipped when it is big enough to matter and the client
 * takes it.
 *
 * Vite compresses what it serves itself, but these routes answer before its
 * compressor is mounted, so a whole-archive pull went out as plain text: on a
 * phone over Tailscale, every byte of it. The archive file here, 126,517 B
 * when this was written, packed to 18,638 B in about 1.3 ms.
 */
export function send(res, code, body) {
  const json = JSON.stringify(body) ?? ''
  res.statusCode = code
  res.setHeader('Content-Type', 'application/json')
  if (json.length < GZIP_MIN) {
    res.end(json)
    return
  }
  // An answer this size differs by what the client accepts, so anything that
  // caches it between here and the browser has to key on that.
  res.setHeader('Vary', 'Accept-Encoding')
  if (!takesGzip(res.req)) {
    res.end(json)
    return
  }
  const packed = gzipSync(json)
  res.setHeader('Content-Encoding', 'gzip')
  res.setHeader('Content-Length', packed.length)
  res.end(packed)
}

/**
 * The request's URL, or null when it cannot be read as one.
 *
 * `new URL(req.url, base)` throws for a path such as `//`: two slashes open
 * a host, and an empty host is not a URL. Every handler here is async, so a
 * parse at the top of one, outside its own try, turned that throw into an
 * unhandled rejection, and Node ends the process on those. One GET for `//`
 * from anything that could reach the port stopped the app for every device.
 */
export function reqUrl(req) {
  try { return new URL(req.url ?? '/', 'http://local') } catch { return null }
}

/**
 * Mount an async handler so nothing it throws can reach the process.
 *
 * Connect ignores the promise a handler returns, so a rejection that escapes
 * the handler's own try is unhandled, and an unhandled rejection exits Node.
 * Each handler catches its own errors; this is the backstop for the lines
 * that run before its try, where the `//` crash above came from.
 */
export function safely(handler) {
  return (req, res, next) => {
    const fail = err => {
      try {
        if (res.headersSent) res.end()
        else send(res, 500, { error: String(err?.message ?? err) })
      } catch { /* the socket is gone */ }
    }
    try {
      Promise.resolve(handler(req, res, next)).catch(fail)
    } catch (err) {
      fail(err)
    }
  }
}

/** Resolve `rel` inside `root`, refusing anything that escapes it lexically. */
export function confine(root, rel) {
  const full = path.resolve(root, String(rel).replace(/^[/\\]+/, ''))
  const base = path.resolve(root)
  if (full !== base && !full.startsWith(base + path.sep)) return null
  return full
}

/**
 * Confinement that survives symlinks. `confine` alone is lexical: a link inside
 * a root pointing at /etc passes it, and `fs.unlink` follows links. This walks
 * up to the nearest component that actually exists, resolves that for real, and
 * re-checks, so a missing leaf cannot skip the check on the directories above
 * it either.
 *
 * The root is resolved for real too, and the file is measured against that.
 * It used to be taken as written while the file was resolved, so a root that
 * is a link, or sits under one (an outputs folder linked to a data disk, or a
 * /home that is a link to /var/home), made every file in it look as if it had
 * climbed out: every delete, reel and picture read under it was refused.
 *
 * The answer is spelled under the root as configured, not under its real
 * path, so a caller's `path.relative(root, full)` still reads as a path inside
 * the root. Everything below the root in it is already resolved.
 */
export async function confineReal(root, rel) {
  const base = path.resolve(root)
  const full = confine(base, rel)
  if (!full) return null
  let realBase
  try { realBase = await fs.realpath(base) } catch { return null }
  let head = full
  const tail = []
  for (;;) {
    let real
    try {
      real = await fs.realpath(head)
    } catch {
      const parent = path.dirname(head)
      if (parent === head) return null
      tail.unshift(path.basename(head))
      head = parent
      continue
    }
    const inside = path.relative(realBase, real)
    if (!confine(realBase, inside)) return null
    const back = path.join(base, inside)
    return tail.length ? path.join(back, ...tail) : back
  }
}

/**
 * Read a JSON body with a cap, so a request cannot buffer the box to death.
 * `{}` for an empty body, null for one that is too large or not JSON.
 */
export async function readBody(req, limit = 1048576) {
  const chunks = []
  let size = 0
  for await (const c of req) {
    size += c.length
    if (size > limit) return null
    chunks.push(c)
  }
  const raw = Buffer.concat(chunks).toString()
  if (!raw.trim()) return {}
  try { return JSON.parse(raw) } catch { return null }
}

export function sseOpen(res) {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  })
}

export function sse(res, event, data) {
  try { res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`) } catch { /* client gone */ }
}

// ------------------------------------------------------------------- guard --

/**
 * Origins a proxy in front of this server is allowed to present. A TLS
 * terminator such as Tailscale Serve rewrites Host, and a browser behind it
 * still sends its own Origin; when the two disagree the Sec-Fetch-Site header
 * settles it for any modern browser, and this list settles it for the rest.
 */
const TRUSTED = new Set(
  (process.env.SWITCHGEN_TRUSTED_ORIGINS ?? '')
    .split(',')
    .map(s => s.trim().toLowerCase())
    .filter(Boolean),
)

const MUTATING = new Set(['POST', 'PUT', 'PATCH', 'DELETE'])

/**
 * Did this request come from one of this server's own pages?
 *
 * Sec-Fetch-Site is the browser's own verdict and cannot be forged by a page,
 * so it is consulted first. Older browsers send Origin on every POST, and that
 * must match Host. A request with neither header did not come from a browser
 * at all: curl, a script, a health check. Those are allowed, because the
 * threat this guards against is a hostile page in someone's browser, not a
 * peer with shell access to the tailnet, who could reach ComfyUI directly.
 */
export function sameOrigin(req) {
  const site = req.headers['sec-fetch-site']
  if (site) return site === 'same-origin' || site === 'none'
  const origin = req.headers.origin
  if (!origin) return true
  let host
  try { host = new URL(origin).host } catch { return false }
  const self = String(req.headers.host ?? '')
  if (host.toLowerCase() === self.toLowerCase()) return true
  return TRUSTED.has(String(origin).toLowerCase())
}

/**
 * Vite's allowedHosts rule, for the one path Vite does not apply it to.
 *
 * An IP address or localhost is always allowed; a name must be listed, and an
 * entry with a leading dot allows that domain and everything under it. This
 * mirrors Vite's own check so the two cannot disagree about a host.
 */
export function hostAllowed(hostHeader, allowedHosts) {
  if (hostHeader === undefined) return true
  const h = String(hostHeader).trim().toLowerCase()
  if (h.startsWith('[')) {
    const end = h.indexOf(']')
    return end > 0 && isIP(h.slice(1, end)) === 6
  }
  const name = h.includes(':') ? h.slice(0, h.indexOf(':')) : h
  if (isIP(name) === 4) return true
  if (name === 'localhost' || name.endsWith('.localhost')) return true
  return allowedHosts.some(a => {
    const allowed = String(a).toLowerCase()
    return allowed === name || (allowed.startsWith('.') && (allowed.slice(1) === name || name.endsWith(allowed)))
  })
}

/**
 * May this WebSocket handshake reach ComfyUI through the proxy?
 *
 * Vite checks Host on HTTP requests only. Its proxy takes an upgrade straight
 * off the HTTP server, where that check never runs, and rewrites Origin to
 * ComfyUI's own on the way through, so ComfyUI's check passes too: any page
 * the user had open could watch the live feed. Two checks close it. The Host check is the one Vite skipped, and
 * stops a hostile name rebound to this machine's address. The origin check
 * stops a page elsewhere that names this machine by an allowed name, which a
 * browser lets any page do for a WebSocket.
 */
export function upgradeAllowed(req, allowedHosts) {
  return hostAllowed(req.headers.host, allowedHosts) && sameOrigin(req)
}

/**
 * The check every mutating route runs before it reads a byte of body.
 *
 * Answers the request itself and returns false when it must not proceed. The
 * content-type check is the CSRF half: a cross-site HTML form can only send
 * text/plain, multipart or urlencoded bodies without a preflight, so a route
 * that insists on application/json cannot be driven by one at all.
 */
export function guardMutation(req, res, accept = ['application/json']) {
  if (!MUTATING.has(req.method)) return true
  if (!sameOrigin(req)) {
    send(res, 403, { error: 'cross-site request refused: this server takes writes only from its own pages' })
    return false
  }
  const ct = String(req.headers['content-type'] ?? '').toLowerCase()
  if (!accept.some(a => ct.startsWith(a))) {
    send(res, 415, { error: `body must be ${accept.join(' or ')}` })
    return false
  }
  return true
}

// ------------------------------------------------------------------- tools --

/** The binaries the servers spawn. One definition, overridable per binary. */
export const TOOLS = {
  aria2c: process.env.SWITCHGEN_ARIA2C ?? '/usr/bin/aria2c',
  ffmpeg: process.env.SWITCHGEN_FFMPEG ?? '/usr/bin/ffmpeg',
  ffprobe: process.env.SWITCHGEN_FFPROBE ?? '/usr/bin/ffprobe',
}

const PROBE_MS = 3000
/** A binary does not appear or vanish often. Probe once a minute at most. */
const PROBE_TTL = 60000

async function works(bin, args) {
  try {
    await run(bin, args, { timeout: PROBE_MS, killSignal: 'SIGKILL' })
    return true
  } catch {
    return false
  }
}

let probed = { at: 0, value: null }

/**
 * Which of the binaries actually run here. Each path is returned when its
 * version call succeeds and null when it does not, so a client can print the
 * path that was tried. ffmpeg takes `-version` with one dash; `--version` is
 * an error to it, which is exactly the kind of thing a probe exists to learn.
 */
export async function tools() {
  if (probed.value && Date.now() - probed.at < PROBE_TTL) return probed.value
  const [aria2c, ffmpeg, ffprobe, gpu] = await Promise.all([
    works(TOOLS.aria2c, ['--version']),
    works(TOOLS.ffmpeg, ['-version']),
    works(TOOLS.ffprobe, ['-version']),
    works('nvidia-smi', ['-L']),
  ])
  probed = {
    at: Date.now(),
    value: {
      aria2c: aria2c ? TOOLS.aria2c : null,
      ffmpeg: ffmpeg ? TOOLS.ffmpeg : null,
      ffprobe: ffprobe ? TOOLS.ffprobe : null,
      gpu,
    },
  }
  return probed.value
}
