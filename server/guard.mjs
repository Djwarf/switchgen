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
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const run = promisify(execFile)

// ---------------------------------------------------------------- plumbing --

export function send(res, code, body) {
  res.statusCode = code
  res.setHeader('Content-Type', 'application/json')
  res.end(JSON.stringify(body))
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
 */
export async function confineReal(root, rel) {
  const base = path.resolve(root)
  const full = confine(base, rel)
  if (!full) return null
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
    const back = confine(base, path.relative(base, real))
    if (!back) return null
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
