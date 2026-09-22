/**
 * SwitchGen reel assembly: many short clips, one long piece.
 *
 * A shot is a generation. A reel is the cut. The graph layer chains shots so
 * each opens on the last frame of the one before; this file is what turns that
 * chain into a single file you can actually watch.
 *
 * Why the server and not a ComfyUI graph: ConcatenateVideo would re-decode
 * every clip through the GPU path and re-encode the lot, on a machine where the
 * GPU is already the scarce thing and earlyoom is armed. ffmpeg's concat
 * demuxer copies the bitstreams instead. Ten clips of identical shape assemble
 * in under a second and lose nothing, because nothing is re-encoded. Re-encode
 * happens only when the inputs genuinely disagree, or when a crossfade asks for
 * frames that do not exist in either source.
 *
 * Local-only, same posture as api.mjs and downloads.mjs: every path is confined
 * to the outputs root, the confinement is re-checked after realpath so a
 * symlink planted in the outputs tree cannot read or write outside it, and no
 * path carrying a control character is passed to ffmpeg at all.
 */
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import crypto from 'node:crypto'
import { spawn, execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { TOOLS, confineReal, guardMutation, readBody, reqUrl, safely, send, sse, sseOpen } from './guard.mjs'

const run = promisify(execFile)

const OUTPUTS = process.env.SWITCHGEN_OUTPUTS ?? '/mnt/storage/ai/outputs'
const FFMPEG = TOOLS.ffmpeg
const FFPROBE = TOOLS.ffprobe

/** Containers we will read. Anything else is refused before ffmpeg is spawned. */
const VIDEO = /\.(webm|mp4|mkv|mov|m4v)$/i
/** A reel is a cut, not a bulk job. Past this the UI is doing something wrong. */
const MAX_CLIPS = 200
/** Longest crossfade we will honour, and the share of the shorter clip it may eat. */
const MAX_FADE = 5
const FADE_SHARE = 0.45
/** Longest a single ffprobe may hang before we stop waiting on it. */
const PROBE_MS = 20000

/**
 * Control characters are refused in any path that reaches ffmpeg.
 *
 * The concat demuxer reads a line-oriented list file. Quoting the path handles
 * quotes and spaces, but nothing quotes a newline: a clip literally named
 * "a\nfile /etc/shadow\n.webm" closes its own `file` line and opens a second
 * one, naming a file outside the outputs root that the confinement never saw.
 * There is no legitimate reason for a generated clip to carry one.
 */
const CONTROL = /[\u0000-\u001f\u007f]/

// ------------------------------------------------------------------ plumbing

/** "24/1" into a usable pair. Rationals stay exact; ffmpeg wants them that way. */
function rational(s) {
  if (typeof s !== 'string' || !s.includes('/')) return { text: String(s ?? ''), value: Number(s) || 0 }
  const [n, d] = s.split('/').map(Number)
  return { text: s, value: d ? n / d : 0 }
}

const num = v => (v == null || v === 'N/A' || v === '' ? null : Number(v))

// -------------------------------------------------------------------- probe

/**
 * Real numbers for one clip. Frame count comes from counted packets rather than
 * the container's nb_frames header, which Matroska simply does not write: the
 * count is authoritative and costs about 50 ms on a five second clip.
 *
 * The timeout matters more than it looks: a truncated or adversarial file can
 * hold ffprobe open indefinitely, and a stitch holds the single assembly slot
 * while it probes. Without it one bad file wedges the whole endpoint.
 */
async function probe(full, rel) {
  const { stdout } = await run(FFPROBE, [
    '-v', 'error', '-count_packets',
    '-show_entries',
    'stream=index,codec_type,codec_name,profile,width,height,pix_fmt,color_range,' +
    'r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,duration,channels,sample_rate:' +
    'format=duration,size,format_name,bit_rate',
    '-print_format', 'json', full,
  ], { maxBuffer: 16 << 20, timeout: PROBE_MS, killSignal: 'SIGKILL' })

  const parsed = JSON.parse(stdout)
  const streams = parsed.streams ?? []
  const v = streams.find(s => s.codec_type === 'video')
  const a = streams.find(s => s.codec_type === 'audio')
  if (!v) throw new Error(`${rel} has no video stream`)

  const fps = rational(v.r_frame_rate)
  const frames = num(v.nb_read_packets) ?? num(v.nb_frames) ?? null
  const fmtDur = num(parsed.format?.duration)
  const strDur = num(v.duration)
  const duration = fmtDur ?? strDur ?? (frames && fps.value ? frames / fps.value : 0)

  return {
    rel,
    full,
    name: path.basename(rel),
    container: (parsed.format?.format_name ?? '').split(',')[0],
    ext: path.extname(rel).toLowerCase(),
    size: num(parsed.format?.size),
    bitrate: num(parsed.format?.bit_rate),
    duration,
    frames: frames ?? (duration && fps.value ? Math.round(duration * fps.value) : null),
    fps: fps.value,
    fpsExact: fps.text,
    avgFps: rational(v.avg_frame_rate).value,
    codec: v.codec_name,
    profile: v.profile ?? null,
    width: num(v.width),
    height: num(v.height),
    pixFmt: v.pix_fmt ?? null,
    colorRange: v.color_range ?? null,
    audio: a ? { codec: a.codec_name, channels: num(a.channels), sampleRate: num(a.sample_rate) } : null,
  }
}

/**
 * Can these be copied end to end, or do they have to be re-encoded? Every
 * reason is named, because "re-encoding, this will take a while" is only a fair
 * thing to tell someone if you also say which clip disagreed and how.
 */
function compatibility(clips) {
  const reasons = []
  if (clips.length < 2) return { identical: true, reasons }
  const head = clips[0]
  const keys = [
    ['codec', 'codec'],
    ['width', 'width'],
    ['height', 'height'],
    ['pixFmt', 'pixel format'],
    ['fpsExact', 'frame rate'],
    ['container', 'container'],
    ['colorRange', 'colour range'],
    ['profile', 'codec profile'],
  ]
  for (const c of clips.slice(1)) {
    for (const [k, label] of keys) {
      if (c[k] !== head[k]) reasons.push(`${c.name}: ${label} ${c[k]} against ${head[k]} in ${head.name}`)
    }
    const mine = c.audio ? `${c.audio.codec}/${c.audio.channels}ch` : 'silent'
    const theirs = head.audio ? `${head.audio.codec}/${head.audio.channels}ch` : 'silent'
    if (mine !== theirs) reasons.push(`${c.name}: audio ${mine} against ${theirs} in ${head.name}`)
  }
  return { identical: reasons.length === 0, reasons }
}

// ------------------------------------------------------------- output naming

/** ComfyUI's counter style, so a reel sorts next to the clips that made it. */
async function nextOutput(dir, base, ext) {
  let entries = []
  try { entries = await fs.readdir(dir) } catch { /* created below */ }
  const seen = new RegExp(`^${base.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}_(\\d{5})_\\${ext}$`)
  let n = 0
  for (const e of entries) {
    const m = seen.exec(e)
    if (m) n = Math.max(n, Number(m[1]))
  }
  return path.join(dir, `${base}_${String(n + 1).padStart(5, '0')}_${ext}`)
}

// ------------------------------------------------------------ encoder tables

const ENCODERS = {
  vp9: { ext: '.webm', crf: 32, video: crf => ['-c:v', 'libvpx-vp9', '-crf', String(crf), '-b:v', '0', '-row-mt', '1', '-deadline', 'good', '-cpu-used', '4'], audio: () => ['-c:a', 'libopus', '-b:a', '128k'] },
  av1: { ext: '.webm', crf: 32, video: crf => ['-c:v', 'libsvtav1', '-crf', String(crf), '-preset', '6'], audio: () => ['-c:a', 'libopus', '-b:a', '128k'] },
  h264: { ext: '.mp4', crf: 20, video: crf => ['-c:v', 'libx264', '-crf', String(crf), '-preset', 'medium'], audio: () => ['-c:a', 'aac', '-b:a', '160k'] },
}

/** Keep the reel in the codec the shots already are, unless asked otherwise. */
function encoderFor(codec) {
  if (codec === 'vp9' || codec === 'vp8') return 'vp9'
  if (codec === 'av1' || codec === 'libdav1d') return 'av1'
  if (codec === 'h264' || codec === 'hevc') return 'h264'
  return 'vp9'
}

// ------------------------------------------------------------- filter graphs

/**
 * Normalise one input to the reel's shape. Letterbox rather than crop: a shot
 * that came out at a different aspect is a mistake to show, not to hide.
 */
function normalise(i, w, h, fps) {
  return `[${i}:v]scale=${w}:${h}:force_original_aspect_ratio=decrease,` +
    `pad=${w}:${h}:-1:-1:color=black,setsar=1,settb=AVTB,fps=${fps},` +
    `format=yuv420p,setpts=PTS-STARTPTS[v${i}]`
}

/**
 * Butt-joined concat through the filter graph. Used when the clips disagree in
 * shape: the demuxer cannot copy those, so they get decoded, normalised, joined.
 */
function concatGraph(clips, w, h, fps, withAudio) {
  const parts = clips.map((_, i) => normalise(i, w, h, fps))
  if (withAudio) {
    clips.forEach((_, i) => parts.push(`[${i}:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a${i}]`))
  }
  const pairs = clips.map((_, i) => (withAudio ? `[v${i}][a${i}]` : `[v${i}]`)).join('')
  parts.push(`${pairs}concat=n=${clips.length}:v=1:a=${withAudio ? 1 : 0}[vout]${withAudio ? '[aout]' : ''}`)
  return { filter: parts.join(';'), duration: clips.reduce((s, c) => s + c.duration, 0) }
}

/**
 * Crossfaded assembly. Each transition eats `t` seconds off the running length,
 * so the reel is shorter than the sum of its shots by `t` times the joins. The
 * offsets are computed against that running length, not against raw durations,
 * which is the usual way these graphs come out wrong.
 */
function xfadeGraph(clips, w, h, fps, t, transition, withAudio) {
  const parts = clips.map((_, i) => normalise(i, w, h, fps))
  if (withAudio) {
    clips.forEach((_, i) => parts.push(`[${i}:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a${i}]`))
  }

  let cur = 'v0'
  let acur = 'a0'
  let acc = clips[0].duration
  for (let i = 1; i < clips.length; i++) {
    const last = i === clips.length - 1
    const out = last ? 'vout' : `x${i}`
    const offset = Math.max(0, acc - t)
    parts.push(`[${cur}][v${i}]xfade=transition=${transition}:duration=${t}:offset=${offset.toFixed(6)}[${out}]`)
    cur = out
    if (withAudio) {
      const aout = last ? 'aout' : `ax${i}`
      parts.push(`[${acur}][a${i}]acrossfade=d=${t}:c1=tri:c2=tri[${aout}]`)
      acur = aout
    }
    acc = acc + clips[i].duration - t
  }
  // A reel of one has no joins, so nothing above ran and the labels never moved.
  if (clips.length === 1) {
    parts.push('[v0]null[vout]')
    if (withAudio) parts.push('[a0]anull[aout]')
  }
  return { filter: parts.join(';'), duration: acc }
}

// ------------------------------------------------------------- ffmpeg runner

/**
 * Spawn ffmpeg, parse `-progress`, report a fraction of the expected output.
 * ffmpeg writes progress to stdout and diagnostics to stderr, so the two never
 * have to be untangled. The child dies with the request: a closed tab must not
 * leave a VP9 encode holding twelve cores.
 */
function encode(args, expectedSeconds, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const child = spawn(FFMPEG, args, { stdio: ['ignore', 'pipe', 'pipe'] })
    let out = ''
    let err = ''
    let killed = false

    const stop = () => { killed = true; try { child.kill('SIGKILL') } catch { /* already gone */ } }
    // The tab can close while we are still probing and naming the output, in
    // which case the signal is already aborted and 'abort' will never fire again.
    if (signal?.aborted) stop()
    else signal?.addEventListener('abort', stop, { once: true })

    child.stdout.on('data', d => {
      out += d
      const blocks = out.split('\n')
      out = blocks.pop() ?? ''
      for (const line of blocks) {
        const eq = line.indexOf('=')
        if (eq < 0) continue
        const key = line.slice(0, eq).trim()
        const value = line.slice(eq + 1).trim()
        if (key === 'out_time_us' || key === 'out_time_ms') {
          const seconds = Number(value) / (key === 'out_time_us' ? 1e6 : 1e3)
          if (Number.isFinite(seconds) && seconds >= 0) {
            onProgress({
              seconds: Math.min(seconds, expectedSeconds),
              fraction: expectedSeconds > 0 ? Math.min(1, seconds / expectedSeconds) : 0,
            })
          }
        } else if (key === 'frame' || key === 'fps' || key === 'speed') {
          onProgress({ [key]: key === 'frame' ? Number(value) : value })
        }
      }
    })
    child.stderr.on('data', d => { err += d; if (err.length > 65536) err = err.slice(-65536) })

    child.on('error', e => { signal?.removeEventListener('abort', stop); reject(e) })
    child.on('close', code => {
      signal?.removeEventListener('abort', stop)
      if (killed) return reject(Object.assign(new Error('cancelled'), { cancelled: true }))
      if (code === 0) return resolve()
      const tail = err.trim().split('\n').filter(Boolean).slice(-4).join('; ')
      reject(new Error(`ffmpeg exited ${code}${tail ? `: ${tail}` : ''}`))
    })
  })
}

// ----------------------------------------------------------------- the route

/**
 * One stitch at a time. VP9 saturates every core it is given, and two encodes
 * racing each other on a box where earlyoom is armed is how you lose both.
 */
let busy = null

/**
 * Claim the assembly slot, run the stitch, and release the slot whatever
 * happens.
 *
 * The release has to be a `finally` around the whole thing rather than a set of
 * early-return helpers. Naming the output touches the disk (mkdir, readdir), so
 * an ENOSPC or an ENOTDIR throws on a line that no hand-written refusal path
 * covers, and the old shape left `busy` set forever: every later stitch 409'd
 * with "a reel is already being assembled" and /api/reel/probe reported a
 * phantom reel until Vite was restarted.
 */
async function stitch(req, res, url) {
  const b = await readBody(req)
  if (!b) return send(res, 400, { error: 'body must be JSON under 1 MB' })

  const clipRefs = Array.isArray(b.clips) ? b.clips : null
  if (!clipRefs || clipRefs.length === 0) return send(res, 400, { error: 'clips must be a non-empty array of filenames' })
  if (clipRefs.length > MAX_CLIPS) return send(res, 400, { error: `a reel is capped at ${MAX_CLIPS} clips, got ${clipRefs.length}` })
  if (!clipRefs.every(c => typeof c === 'string' && c.trim())) return send(res, 400, { error: 'every clip must be a non-empty string' })

  if (busy) return send(res, 409, { error: 'a reel is already being assembled', since: { ...busy } })
  // Claim the slot before the first await below, not after. Two requests
  // arriving together would otherwise both clear the check, both pick
  // reel_00004_, and both write it.
  busy = { startedAt: Date.now(), clips: clipRefs.length, out: null }
  try {
    return await assemble(b, clipRefs, res, url)
  } finally {
    busy = null
  }
}

async function assemble(b, clipRefs, res, url) {
  // Listen on the response, not the request. An IncomingMessage emits 'close'
  // as soon as its body has been read, which for a POST is long before the
  // client goes away: a listener attached after readBody would never fire, and
  // a closed tab would leave a VP9 encode holding every core. The response
  // stays open until we end it, so its 'close' is the real disconnect.
  //
  // It is armed here, before the probes, because probing 200 clips is itself
  // long enough for a tab to close, and each probe is a child process.
  const abort = new AbortController()
  const onClose = () => abort.abort()
  res.on('close', onClose)
  let listFile = null

  try {
    // Resolve and vet every path before anything is spawned.
    const resolved = []
    for (const rel of clipRefs) {
      if (CONTROL.test(rel)) return send(res, 400, { error: 'a clip name cannot contain control characters' })
      if (!VIDEO.test(rel)) return send(res, 400, { error: `${rel} is not a video file this reel can read` })
      const full = await confineReal(OUTPUTS, rel)
      if (!full) return send(res, 400, { error: `${rel} escapes the outputs root` })
      // Re-check the resolved path: `rel` may be clean while the symlink it
      // names points at something whose own name is not.
      if (CONTROL.test(full)) return send(res, 400, { error: `${rel} resolves to a path with control characters in it` })
      let st
      try { st = await fs.stat(full) } catch { return send(res, 404, { error: `${rel} not found under the outputs root` }) }
      if (!st.isFile()) return send(res, 400, { error: `${rel} is not a regular file` })
      resolved.push({ rel: path.relative(path.resolve(OUTPUTS), full), full })
    }

    let clips
    try {
      clips = await Promise.all(resolved.map(r => probe(r.full, r.rel)))
    } catch (e) {
      return send(res, 422, { error: String(e?.message ?? e) })
    }

    const compat = compatibility(clips)
    const warnings = []

    // Audio rides along only when every clip has it. Half a reel with sound is
    // worse than none: the drop-outs read as a bug in the render.
    const withAudio = clips.every(c => c.audio)
    if (!withAudio && clips.some(c => c.audio)) warnings.push('some clips carry audio and some do not, so the reel is silent')

    const w = clips[0].width
    const h = clips[0].height
    const fpsExact = clips[0].fpsExact
    const fps = b.fps != null ? Number(b.fps) : clips[0].fps
    if (!Number.isFinite(fps) || fps <= 0 || fps > 240) return send(res, 400, { error: 'fps must be a positive number no greater than 240' })
    const fpsChanged = Math.abs(fps - clips[0].fps) > 1e-6

    // Clamp the crossfade against the shortest clip. Asking for two seconds of
    // dissolve out of a 1.5 second shot has no honest answer.
    let fade = Number(b.crossfade) || 0
    if (fade < 0) return send(res, 400, { error: 'crossfade cannot be negative' })
    if (fade > 0 && clips.length > 1) {
      const shortest = Math.min(...clips.map(c => c.duration))
      const ceiling = Math.min(MAX_FADE, shortest * FADE_SHARE)
      if (fade > ceiling) {
        warnings.push(`crossfade trimmed from ${fade}s to ${ceiling.toFixed(2)}s by the ${shortest.toFixed(2)}s shot`)
        fade = ceiling
      }
    }
    if (clips.length === 1) fade = 0
    const transition = typeof b.transition === 'string' && /^[a-z]+$/.test(b.transition) ? b.transition : 'fade'

    const native = encoderFor(clips[0].codec)
    const encName = typeof b.codec === 'string' ? b.codec.toLowerCase() : native
    const enc = ENCODERS[encName]
    if (!enc) return send(res, 400, { error: `codec must be one of ${Object.keys(ENCODERS).join(', ')}` })
    const crf = b.crf != null ? Number(b.crf) : enc.crf
    if (!Number.isFinite(crf) || crf < 0 || crf > 63) return send(res, 400, { error: 'crf must be between 0 and 63' })

    // Asking for the codec the shots already are is not a reason to re-encode
    // them. Only a genuine change of codec, shape, rate or a crossfade is.
    const wantsEncode = fade > 0 || !compat.identical || fpsChanged || encName !== native
    if (!wantsEncode && b.crf != null) warnings.push('crf was ignored: these clips copy without being re-encoded, so their quality is unchanged')

    // The reel lands beside the clips that made it, in the same subfolder as the
    // first shot unless a prefix says otherwise.
    if (b.prefix != null && (typeof b.prefix !== 'string' || CONTROL.test(b.prefix))) {
      return send(res, 400, { error: 'prefix must be a string without control characters' })
    }
    const prefix = typeof b.prefix === 'string' && b.prefix.trim() ? b.prefix.trim() : path.posix.join(path.dirname(clips[0].rel).replace(/\\/g, '/'), 'reel')
    const prefixDir = await confineReal(OUTPUTS, path.dirname(prefix))
    if (!prefixDir) return send(res, 400, { error: 'prefix escapes the outputs root' })
    const base = path.basename(prefix).replace(/[^\w.-]/g, '_').replace(/^\.+$/, '') || 'reel'
    const ext = wantsEncode ? enc.ext : clips[0].ext
    await fs.mkdir(prefixDir, { recursive: true })
    const outPath = await nextOutput(prefixDir, base, ext)
    const outRel = path.relative(path.resolve(OUTPUTS), outPath)

    // Everything validated. Name the reel on the lock, open the stream, work.
    busy.out = outRel
    const wantsJson = url.searchParams.get('json') === '1'

    let args
    let expected
    let mode

    if (!wantsEncode) {
      mode = 'copy'
      expected = clips.reduce((s, c) => s + c.duration, 0)
      listFile = path.join(os.tmpdir(), `switchgen-reel-${crypto.randomUUID()}.txt`)
      // The concat demuxer's own quoting: a single quote closes, escapes, reopens.
      // Control characters, the newline above all, are refused before we get
      // here, because no amount of quoting contains them in a line-based format.
      const list = clips.map(c => `file '${c.full.replace(/'/g, "'\\''")}'`).join('\n') + '\n'
      await fs.writeFile(listFile, list, { mode: 0o600 })
      args = [
        '-hide_banner', '-nostdin', '-loglevel', 'error', '-progress', 'pipe:1', '-nostats',
        '-fflags', '+genpts',
        '-f', 'concat', '-safe', '0', '-protocol_whitelist', 'file', '-i', listFile,
        '-c', 'copy', '-y', outPath,
      ]
    } else {
      mode = fade > 0 ? 'crossfade' : 'encode'
      const graph = fade > 0
        ? xfadeGraph(clips, w, h, fps, fade, transition, withAudio)
        : concatGraph(clips, w, h, fps, withAudio)
      expected = graph.duration
      args = [
        '-hide_banner', '-nostdin', '-loglevel', 'error', '-progress', 'pipe:1', '-nostats',
        ...clips.flatMap(c => ['-i', c.full]),
        '-filter_complex', graph.filter,
        '-map', '[vout]',
        ...(withAudio ? ['-map', '[aout]'] : ['-an']),
        ...enc.video(crf),
        ...(withAudio ? enc.audio() : []),
        '-pix_fmt', 'yuv420p',
        '-y', outPath,
      ]
    }

    const plan = {
      mode,
      out: outRel,
      clips: clips.map(c => ({ rel: c.rel, name: c.name, duration: c.duration, frames: c.frames, fps: c.fps, width: c.width, height: c.height, codec: c.codec })),
      expectedSeconds: expected,
      expectedFrames: Math.round(expected * fps),
      crossfade: fade,
      transition: fade > 0 ? transition : null,
      fps,
      fpsExact: fpsChanged ? String(fps) : fpsExact,
      width: w,
      height: h,
      audio: withAudio,
      codec: wantsEncode ? encName : clips[0].codec,
      crf: wantsEncode ? crf : null,
      reasons: compat.reasons,
      warnings,
      command: [FFMPEG, ...args].join(' '),
    }

    if (!wantsJson) {
      sseOpen(res)
      sse(res, 'plan', plan)
    }

    const started = Date.now()
    let last = 0
    try {
      await encode(args, expected, p => {
        if (wantsJson) return
        // One frame of progress per 100 ms is plenty for a bar, and keeps a long
        // VP9 encode from spending its time writing SSE.
        const now = Date.now()
        if (p.seconds == null && now - last < 400) return
        if (now - last < 100) return
        last = now
        sse(res, 'progress', { ...p, elapsed: (now - started) / 1000 })
      }, abort.signal)

      const result = await probe(outPath, outRel)
      const summary = {
        ...plan,
        done: true,
        elapsed: (Date.now() - started) / 1000,
        actual: {
          rel: outRel,
          seconds: result.duration,
          frames: result.frames,
          size: result.size,
          codec: result.codec,
          width: result.width,
          height: result.height,
          fps: result.fps,
          audio: result.audio,
        },
      }
      if (wantsJson) return send(res, 200, summary)
      sse(res, 'done', summary)
      return res.end()
    } catch (e) {
      const cancelled = e?.cancelled === true
      // A half-written reel is not a file anyone wants in the gallery.
      try { await fs.unlink(outPath) } catch { /* never created */ }
      const payload = { error: cancelled ? 'cancelled' : String(e?.message ?? e), out: outRel, mode, command: plan.command }
      if (wantsJson) return send(res, cancelled ? 499 : 500, payload)
      if (!cancelled) sse(res, 'error', payload)
      return res.end()
    }
  } finally {
    res.off('close', onClose)
    if (listFile) { try { await fs.unlink(listFile) } catch { /* gone */ } }
  }
}

// -------------------------------------------------------------------- export

export function switchgenReel() {
  const handler = async (req, res, next) => {
    // An unreadable path is refused rather than parsed where it can throw.
    // See reqUrl in guard.mjs for what that throw used to do.
    const url = reqUrl(req)
    if (!url) return send(res, 400, { error: 'the request path is not a valid URL' })
    if (!url.pathname.startsWith('/api/reel')) return next()

    try {
      // Real durations for the timeline. The UI shows a reel length before a
      // single frame is encoded, so the number has to come from the files.
      if (url.pathname === '/api/reel/probe' && req.method === 'GET') {
        const refs = url.searchParams.getAll('file')
        if (refs.length === 0) return send(res, 400, { error: 'pass one or more ?file= parameters, relative to the outputs root' })
        if (refs.length > MAX_CLIPS) return send(res, 400, { error: `at most ${MAX_CLIPS} files per probe` })

        // 200 files is 200 ffprobe children. If the tab goes, stop spawning.
        let gone = false
        const onClose = () => { gone = true }
        res.on('close', onClose)
        try {
          const clips = []
          for (const rel of refs) {
            if (gone) return
            if (CONTROL.test(rel)) return send(res, 400, { error: 'a clip name cannot contain control characters' })
            if (!VIDEO.test(rel)) return send(res, 400, { error: `${rel} is not a video file this reel can read` })
            const full = await confineReal(OUTPUTS, rel)
            if (!full) return send(res, 400, { error: `${rel} escapes the outputs root` })
            if (CONTROL.test(full)) return send(res, 400, { error: `${rel} resolves to a path with control characters in it` })
            try { if (!(await fs.stat(full)).isFile()) throw new Error('not a regular file') }
            catch { return send(res, 404, { error: `${rel} not found under the outputs root` }) }
            try { clips.push(await probe(full, path.relative(path.resolve(OUTPUTS), full))) }
            catch (e) { return send(res, 422, { error: String(e?.message ?? e) }) }
          }
          if (gone) return

          const compat = compatibility(clips)
          return send(res, 200, {
            root: OUTPUTS,
            clips: clips.map(({ full, ...rest }) => rest),
            total: {
              seconds: clips.reduce((s, c) => s + c.duration, 0),
              frames: clips.reduce((s, c) => s + (c.frames ?? 0), 0),
              size: clips.reduce((s, c) => s + (c.size ?? 0), 0),
            },
            // Whether a stitch would copy or re-encode, and why. The UI can tell
            // the truth about the wait before anyone commits to it.
            assembly: { copy: compat.identical, reasons: compat.reasons },
            busy: busy ? { ...busy } : null,
          })
        } finally {
          res.off('close', onClose)
        }
      }

      if (url.pathname === '/api/reel/stitch' && req.method === 'POST') {
        if (!guardMutation(req, res)) return
        return await stitch(req, res, url)
      }

      // Other middleware owns /api/* paths too, so unknown ones are handed on
      // rather than swallowed. Only our own namespace answers 404 here.
      if (url.pathname === '/api/reel' || url.pathname.startsWith('/api/reel/')) {
        return send(res, 404, { error: `no such endpoint: ${req.method} ${url.pathname}` })
      }
      return next()
    } catch (err) {
      if (res.headersSent) { try { res.end() } catch { /* gone */ } return }
      return send(res, 500, { error: String(err?.message ?? err) })
    }
  }

  return {
    name: 'switchgen-reel',
    configureServer(server) { server.middlewares.use(safely(handler)) },
    configurePreviewServer(server) { server.middlewares.use(safely(handler)) },
  }
}
