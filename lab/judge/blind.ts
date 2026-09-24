/**
 * Blind copies: the only pictures the phone ever sees.
 *
 * ComfyUI's PNGs carry the whole graph in text chunks (the weight file by
 * name), and a phone photo carries camera, place and time. A blind copy is a
 * fresh WebP made by ffmpeg with every metadata stream and chapter dropped,
 * then checked chunk by chunk: anything but picture data fails the copy.
 *
 * A reference photo is turned upright here, from its own EXIF orientation
 * flag, the way ComfyUI's LoadImage reads it and the way a browser shows it.
 * ffmpeg's own automatic rotation is switched off (-noautorotate) so the turn
 * is made exactly once, by this code, whatever ffmpeg version is installed.
 */
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { sizeOf } from '../run/imagesize.ts'
import type { RefRect } from '../core/types.ts'

/** The grid copy's long edge. */
export const GRID_EDGE = 640
/** The full-screen copy's longest long edge. */
export const FULL_EDGE = 2048
export const GRID_QUALITY = 80
export const FULL_QUALITY = 88
/** At most this many ffmpeg processes at a time, each under nice. */
export const FFMPEG_AT_ONCE = 2

export type BlindOptions = {
  /** Draw this rectangle (in the upright source's pixels) as an outline: the painted area of a region edit. */
  outline?: RefRect | null
  /** Scale up to `longEdge` when the source is smaller (so every copy in a set has the same size). */
  upscale?: boolean
  /** Run ffmpeg under `nice -n 10`. Default true. */
  nice?: boolean
  timeoutMs?: number
}

/** The ffmpeg filters that turn a picture with this EXIF orientation upright. */
export function orientFilters(orientation: number): string[] {
  switch (orientation) {
    case 2: return ['hflip']
    case 3: return ['hflip', 'vflip']
    case 4: return ['vflip']
    case 5: return ['transpose=0']
    case 6: return ['transpose=1']
    case 7: return ['transpose=3']
    case 8: return ['transpose=2']
    default: return []
  }
}

/** The copy's size: the long edge set to `longEdge` (never above the source's unless `upscale`), the aspect kept. */
export function fitSize(width: number, height: number, longEdge: number, upscale = false): { width: number; height: number } {
  const long = Math.max(width, height)
  const L = upscale ? longEdge : Math.min(longEdge, long)
  if (width >= height) return { width: L, height: Math.max(1, Math.round((height * L) / width)) }
  return { width: Math.max(1, Math.round((width * L) / height)), height: L }
}

export type WebpChunk = { fourcc: string; size: number; flags?: number }

/** The chunks of a WebP file, in order. Throws when it is not a WebP. */
export function webpChunks(buf: Uint8Array): WebpChunk[] {
  const b = Buffer.from(buf.buffer, buf.byteOffset, buf.byteLength)
  if (b.length < 12 || b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WEBP') throw new Error('not a WebP file')
  const end = Math.min(b.length, 8 + b.readUInt32LE(4))
  const out: WebpChunk[] = []
  let i = 12
  while (i + 8 <= end) {
    const fourcc = b.toString('ascii', i, i + 4)
    const size = b.readUInt32LE(i + 4)
    const c: WebpChunk = { fourcc, size }
    if (fourcc === 'VP8X' && size >= 1) c.flags = b[i + 8]
    out.push(c)
    i += 8 + size + (size & 1)
  }
  return out
}

const PICTURE_CHUNKS = new Set(['VP8 ', 'VP8L', 'VP8X', 'ALPH'])

/**
 * Everything in a WebP that is not picture data: EXIF, XMP, a colour profile,
 * animation, or an unknown chunk, and the VP8X flags that announce them.
 * Empty when the file is clean. A file that is not a WebP is one problem.
 */
export function metadataProblems(buf: Uint8Array): string[] {
  let chunks: WebpChunk[]
  try {
    chunks = webpChunks(buf)
  } catch (e) {
    return [(e as Error).message]
  }
  const out: string[] = []
  for (const c of chunks) {
    if (!PICTURE_CHUNKS.has(c.fourcc)) out.push(`a ${JSON.stringify(c.fourcc)} chunk`)
    if (c.fourcc === 'VP8X' && c.flags !== undefined) {
      if (c.flags & 0x20) out.push('the VP8X header announces a colour profile')
      if (c.flags & 0x08) out.push('the VP8X header announces EXIF')
      if (c.flags & 0x04) out.push('the VP8X header announces XMP')
      if (c.flags & 0x02) out.push('the VP8X header announces animation')
    }
  }
  if (!chunks.some(c => c.fourcc === 'VP8 ' || c.fourcc === 'VP8L')) out.push('no picture data')
  return out
}

/** Whether this ffmpeg runs and has the libwebp encoder. */
export function hasFfmpeg(ffmpeg = 'ffmpeg'): Promise<boolean> {
  return new Promise(resolve => {
    let out = ''
    let child
    try {
      child = spawn(ffmpeg, ['-hide_banner', '-encoders'], { stdio: ['ignore', 'pipe', 'ignore'] })
    } catch {
      resolve(false)
      return
    }
    child.stdout.on('data', (d: Buffer) => { out += d.toString() })
    child.on('error', () => resolve(false))
    child.on('close', code => resolve(code === 0 && /\blibwebp\b/.test(out)))
  })
}

const NICE = ['/usr/bin/nice', '/bin/nice'].find(p => existsSync(p)) ?? null

function run(cmd: string, args: string[], timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'ignore', 'pipe'] })
    let err = ''
    child.stderr.on('data', (d: Buffer) => { if (err.length < 4000) err += d.toString() })
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs)
    child.on('error', e => { clearTimeout(timer); reject(e) })
    child.on('close', code => {
      clearTimeout(timer)
      if (code === 0) resolve()
      else reject(new Error(`ffmpeg failed (${code ?? 'killed'}): ${err.trim().split('\n').slice(-3).join(' ')}`))
    })
  })
}

/**
 * Make one blind WebP copy of `src` at `dst`: upright, the long edge
 * `longEdge`, no metadata. The copy is written beside `dst` first and moved
 * into place only once it has passed the chunk check. Returns its size.
 */
export async function blindCopy(
  ffmpeg: string,
  src: string,
  dst: string,
  longEdge: number,
  quality: number,
  opts: BlindOptions = {},
): Promise<{ width: number; height: number }> {
  const bytes = readFileSync(src)
  const size = sizeOf(bytes)
  const out = fitSize(size.width, size.height, longEdge, !!opts.upscale)
  const filters = [...orientFilters(size.orientation)]
  const r = opts.outline
  if (r && r.w > 0 && r.h > 0) {
    const t = Math.max(3, Math.round(Math.max(size.width, size.height) / 220))
    const x = Math.max(0, Math.round(r.x))
    const y = Math.max(0, Math.round(r.y))
    const w = Math.max(1, Math.round(r.w))
    const h = Math.max(1, Math.round(r.h))
    filters.push(`drawbox=x=${x - 1}:y=${y - 1}:w=${w + 2}:h=${h + 2}:color=black@0.9:t=${t + 2}`)
    filters.push(`drawbox=x=${x}:y=${y}:w=${w}:h=${h}:color=0xFFD400@1:t=${t}`)
  }
  filters.push(`scale=${out.width}:${out.height}:flags=lanczos`)
  mkdirSync(dirname(dst), { recursive: true })
  const tmp = join(dirname(dst), `.${process.pid}.${Math.random().toString(36).slice(2)}.webp`)
  const args = [
    '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
    '-noautorotate', '-i', src,
    '-map', '0:v:0', '-map_metadata', '-1', '-map_chapters', '-1',
    '-frames:v', '1', '-vf', filters.join(','),
    '-fflags', '+bitexact', '-flags:v', '+bitexact',
    '-c:v', 'libwebp', '-quality', String(quality), '-compression_level', '4',
    '-f', 'webp', tmp,
  ]
  const useNice = opts.nice !== false && NICE !== null
  try {
    await run(useNice ? (NICE as string) : ffmpeg, useNice ? ['-n', '10', ffmpeg, ...args] : args, opts.timeoutMs ?? 120_000)
    const problems = metadataProblems(readFileSync(tmp))
    if (problems.length) throw new Error(`the blind copy of a picture still carries ${problems.join(', ')}`)
    renameSync(tmp, dst)
  } finally {
    rmSync(tmp, { force: true })
  }
  return out
}

/** A queue that runs at most `limit` tasks at a time. */
export function createPool(limit = FFMPEG_AT_ONCE): <T>(task: () => Promise<T>) => Promise<T> {
  let active = 0
  const waiting: (() => void)[] = []
  const release = () => {
    active--
    waiting.shift()?.()
  }
  return async <T>(task: () => Promise<T>): Promise<T> => {
    if (active >= limit) await new Promise<void>(res => waiting.push(res))
    active++
    try {
      return await task()
    } finally {
      release()
    }
  }
}
