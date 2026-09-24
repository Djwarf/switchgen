/**
 * Shared helpers for the lab's own tests (`lab/lab test`).
 *
 * Nothing here reaches the network, the running app, ComfyUI or the author's
 * folders. Every folder a test writes to is made under the system temp folder
 * and removed when the file's tests end. Test pictures are generated here, in
 * code or with PIL or ffmpeg when those are installed; the user's own photos
 * are never read.
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { crc32, deflateSync } from 'node:zlib'
import type { LabEnv } from '../core/env.ts'

/** The repo these tests sit in: lab/tests is two folders down. */
export const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')

// ------------------------------------------------------------- folders --

const made: string[] = []

/** A fresh folder under the system temp folder, removed by `removeTemp`. */
export function tempDir(prefix = 'lab-test-'): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix))
  made.push(dir)
  return dir
}

/** Remove every folder `tempDir` made in this file. */
export function removeTemp(): void {
  for (const d of made.splice(0)) fs.rmSync(d, { recursive: true, force: true })
}

/** A LabEnv whose lab folder and outputs are fresh temp folders, and whose app and ComfyUI are on port 9. */
export function tempEnv(over: Partial<LabEnv> = {}): LabEnv {
  const root = tempDir('lab-env-')
  const env: LabEnv = {
    repoRoot: REPO,
    labDir: path.join(root, 'lab'),
    outputs: path.join(root, 'outputs'),
    appUrl: 'http://127.0.0.1:9',
    comfyUrl: 'http://127.0.0.1:9',
    port: 0,
    host: '127.0.0.1',
    ...over,
  }
  fs.mkdirSync(env.labDir, { recursive: true })
  fs.mkdirSync(env.outputs, { recursive: true })
  return env
}

// ------------------------------------------------------------ tools --

function works(cmd: string, args: string[]): boolean {
  try {
    execFileSync(cmd, args, { stdio: 'ignore', timeout: 20_000 })
    return true
  } catch {
    return false
  }
}

/** True when python3 with PIL is installed (the cross-checks that decode pictures need it). */
export const HAS_PIL = works('python3', ['-c', 'import PIL'])
/** True when ffmpeg is installed (the blind copies need it). */
export const HAS_FFMPEG = works('ffmpeg', ['-hide_banner', '-version'])

// ------------------------------------------------------------ pictures --

export function pngChunk(type: string, data: Uint8Array): Buffer {
  const head = Buffer.alloc(8)
  head.writeUInt32BE(data.length, 0)
  head.write(type, 4, 'ascii')
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])) >>> 0, 0)
  return Buffer.concat([head, data, crc])
}

const PNG_SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

/**
 * A real RGB PNG of one colour. `text` goes in a tEXt 'prompt' chunk, the
 * way ComfyUI's SaveImage writes the graph (and its ckpt_name) into every
 * picture; `orientation` adds an eXIf chunk.
 */
export function png(w: number, h: number, rgb: [number, number, number] = [128, 128, 128], opts: { text?: string; orientation?: number } = {}): Buffer {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(w, 0)
  ihdr.writeUInt32BE(h, 4)
  ihdr[8] = 8
  ihdr[9] = 2
  const row = Buffer.alloc(1 + w * 3)
  for (let x = 0; x < w; x++) {
    row[1 + x * 3] = rgb[0]
    row[2 + x * 3] = rgb[1]
    row[3 + x * 3] = rgb[2]
  }
  const raw = Buffer.concat(Array.from({ length: h }, () => row))
  const parts = [PNG_SIG, pngChunk('IHDR', ihdr)]
  if (opts.orientation) parts.push(pngChunk('eXIf', tiff({ orientation: opts.orientation })))
  if (opts.text !== undefined) parts.push(pngChunk('tEXt', Buffer.from('prompt\0' + opts.text, 'latin1')))
  parts.push(pngChunk('IDAT', deflateSync(raw)), pngChunk('IEND', Buffer.alloc(0)))
  return Buffer.concat(parts)
}

/**
 * A little-endian TIFF block (EXIF's body) holding an orientation and, when
 * given, a camera model: what a phone writes, minus the rest.
 */
export function tiff(o: { orientation?: number; model?: string }): Buffer {
  const entries: { tag: number; type: number; count: number; value: Buffer }[] = []
  if (o.model !== undefined) {
    const s = Buffer.from(o.model + '\0', 'latin1')
    entries.push({ tag: 0x0110, type: 2, count: s.length, value: s })
  }
  if (o.orientation !== undefined) {
    const v = Buffer.alloc(4)
    v.writeUInt16LE(o.orientation, 0)
    entries.push({ tag: 0x0112, type: 3, count: 1, value: v })
  }
  const ifdSize = 2 + entries.length * 12 + 4
  let extraAt = 8 + ifdSize
  const extras: Buffer[] = []
  const ifd = Buffer.alloc(ifdSize)
  ifd.writeUInt16LE(entries.length, 0)
  entries.forEach((e, k) => {
    const at = 2 + k * 12
    ifd.writeUInt16LE(e.tag, at)
    ifd.writeUInt16LE(e.type, at + 2)
    ifd.writeUInt32LE(e.count, at + 4)
    if (e.value.length <= 4) e.value.copy(ifd, at + 8)
    else {
      ifd.writeUInt32LE(extraAt, at + 8)
      extras.push(e.value)
      extraAt += e.value.length
    }
  })
  const head = Buffer.from([0x49, 0x49, 42, 0, 8, 0, 0, 0])
  return Buffer.concat([head, ifd, ...extras])
}

function jpegSegment(marker: number, body: Buffer): Buffer {
  const head = Buffer.from([0xff, marker, 0, 0])
  head.writeUInt16BE(body.length + 2, 2)
  return Buffer.concat([head, body])
}

/**
 * A JPEG with a real marker structure (SOI, APP0, APP1 Exif, a comment, SOF,
 * SOS, EOI) but a token scan, stored at `width` x `height`: enough for every
 * header reader and for cleanCopy, not for a decoder. `trailer` is left after
 * the end marker, the way a phone appends a second picture.
 */
export function jpegHeader(o: { width: number; height: number; orientation?: number; model?: string; progressive?: boolean; comment?: string; trailer?: string }): Buffer {
  const parts: Buffer[] = [Buffer.from([0xff, 0xd8])]
  parts.push(jpegSegment(0xe0, Buffer.from('JFIF\0\x01\x01\0\0\x01\0\x01\0\0', 'latin1')))
  if (o.orientation !== undefined || o.model !== undefined) {
    parts.push(jpegSegment(0xe1, Buffer.concat([Buffer.from('Exif\0\0', 'latin1'), tiff({ orientation: o.orientation, model: o.model })])))
  }
  if (o.comment) parts.push(jpegSegment(0xfe, Buffer.from(o.comment, 'latin1')))
  const sof = Buffer.alloc(15)
  sof[0] = 8
  sof.writeUInt16BE(o.height, 1)
  sof.writeUInt16BE(o.width, 3)
  sof[5] = 3
  for (let c = 0; c < 3; c++) {
    sof[6 + c * 3] = c + 1
    sof[7 + c * 3] = 0x11
    sof[8 + c * 3] = 0
  }
  parts.push(jpegSegment(o.progressive ? 0xc2 : 0xc0, sof))
  parts.push(jpegSegment(0xda, Buffer.from([1, 1, 0, 0, 63, 0])))
  parts.push(Buffer.from([0x12, 0x34, 0xff, 0x00, 0x56]))
  parts.push(Buffer.from([0xff, 0xd9]))
  if (o.trailer) parts.push(Buffer.from(o.trailer, 'latin1'))
  return Buffer.concat(parts)
}

function riff(chunks: Buffer[]): Buffer {
  const body = Buffer.concat([Buffer.from('WEBP', 'ascii'), ...chunks])
  const head = Buffer.alloc(8)
  head.write('RIFF', 0, 'ascii')
  head.writeUInt32LE(body.length, 4)
  return Buffer.concat([head, body])
}

function riffChunk(fourcc: string, data: Buffer): Buffer {
  const head = Buffer.alloc(8)
  head.write(fourcc, 0, 'ascii')
  head.writeUInt32LE(data.length, 4)
  return Buffer.concat([head, data, data.length & 1 ? Buffer.alloc(1) : Buffer.alloc(0)])
}

/** WebP headers of each kind: lossy (VP8), lossless (VP8L) and extended (VP8X, with an EXIF chunk for the orientation). */
export function webpHeader(kind: 'VP8' | 'VP8L' | 'VP8X', width: number, height: number, orientation?: number): Buffer {
  if (kind === 'VP8') {
    const d = Buffer.alloc(10)
    d[3] = 0x9d
    d[4] = 0x01
    d[5] = 0x2a
    d.writeUInt16LE(width & 0x3fff, 6)
    d.writeUInt16LE(height & 0x3fff, 8)
    return riff([riffChunk('VP8 ', d)])
  }
  if (kind === 'VP8L') {
    const d = Buffer.alloc(5)
    d[0] = 0x2f
    d.writeUInt32LE(((width - 1) & 0x3fff) | (((height - 1) & 0x3fff) << 14), 1)
    return riff([riffChunk('VP8L', d)])
  }
  const d = Buffer.alloc(10)
  d[0] = orientation ? 0x08 : 0
  d.writeUIntLE(width - 1, 4, 3)
  d.writeUIntLE(height - 1, 7, 3)
  const chunks = [riffChunk('VP8X', d), riffChunk('VP8L', Buffer.from([0x2f, 0, 0, 0, 0]))]
  if (orientation) chunks.push(riffChunk('EXIF', tiff({ orientation })))
  return riff(chunks)
}

/**
 * A stand-in for the user's cat photo: 4080 x 3060 stored sideways with EXIF
 * orientation 6, so its upright size is 3060 x 4080, and a camera model in
 * its EXIF. Header only; never the real photo.
 */
export function sidewaysCatStandIn(): Buffer {
  return jpegHeader({ width: 4080, height: 3060, orientation: 6, model: 'PhoneModel X', comment: 'secret comment', trailer: 'TRAILER with secret data' })
}

/**
 * Real, decodable pictures made with PIL, each with the metadata a phone or
 * an editor leaves: camera maker and model, GPS, comments, and an orientation
 * flag on the sideways ones. Needs HAS_PIL.
 */
export function pilFixtures(dir: string): string {
  fs.mkdirSync(dir, { recursive: true })
  execFileSync('python3', [
    '-c',
    `
import sys, os
from PIL import Image
d = sys.argv[1]
def pic(w, h, mode='RGB'):
    im = Image.new(mode, (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            c = ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
            px[x, y] = c + ((x * 11) % 256,) if mode == 'RGBA' else c
    return im
def phone_exif(o):
    ex = Image.Exif()
    ex[274] = o
    ex[271] = 'PhoneMaker'
    ex[272] = 'PhoneModel X'
    ex[0x8825] = {1: 'N', 2: (52.0, 22.0, 1.0)}
    return ex
pic(120, 80).save(os.path.join(d, 'plain.jpg'), quality=95)
pic(120, 80).save(os.path.join(d, 'side.jpg'), quality=95, exif=phone_exif(6), comment=b'secret comment')
pic(120, 80).save(os.path.join(d, 'prog.jpg'), quality=95, exif=phone_exif(6), progressive=True)
pic(120, 80).save(os.path.join(d, 'plain.png'))
ex = Image.Exif(); ex[274] = 8; ex[271] = 'Maker'
from PIL import PngImagePlugin
meta = PngImagePlugin.PngInfo(); meta.add_text('Comment', 'hello secret')
pic(120, 80).save(os.path.join(d, 'side.png'), exif=ex, pnginfo=meta)
pic(120, 80).save(os.path.join(d, 'lossy.webp'), quality=90)
pic(333, 777, 'RGBA').save(os.path.join(d, 'odd-alpha.webp'), lossless=True)
ex = Image.Exif(); ex[274] = 5; ex[271] = 'Maker'
pic(120, 80).save(os.path.join(d, 'side.webp'), exif=ex, lossless=True)
`,
    dir,
  ])
  return dir
}

/** What PIL reads from a picture: its stored and upright size, EXIF tags, text, GPS and one upright pixel. */
export function pilRead(file: string): { size: [number, number]; upright: [number, number]; exif: Record<string, string>; gps: boolean; info: Record<string, unknown>; px: number[] } {
  const out = execFileSync('python3', [
    '-c',
    `
import sys, json
from PIL import Image, ImageOps
im = Image.open(sys.argv[1]); ex = im.getexif()
up = ImageOps.exif_transpose(im)
info = {k: v for k, v in im.info.items() if isinstance(v, (str, int, float))}
print(json.dumps({"size": im.size, "upright": up.size, "exif": {str(k): str(v) for k, v in ex.items()}, "gps": bool(ex.get_ifd(0x8825)), "info": info, "px": list(up.convert('RGB').getpixel((5, 5)))}))
`,
    file,
  ])
  return JSON.parse(out.toString())
}

/**
 * A real JPEG of `w` x `h` made by ffmpeg (a blue band on the left quarter,
 * orange elsewhere), then marked sideways with an EXIF orientation 6 segment:
 * a phone's portrait photo in miniature. Needs HAS_FFMPEG.
 */
export function sidewaysJpeg(file: string, w: number, h: number): Buffer {
  execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', `color=orange:s=${w}x${h},drawbox=x=0:y=0:w=${Math.round(w / 4)}:h=${h}:color=blue:t=fill`, '-frames:v', '1', file])
  const j = fs.readFileSync(file)
  const t = Buffer.alloc(26)
  t.write('II', 0, 'ascii')
  t.writeUInt16LE(42, 2)
  t.writeUInt32LE(8, 4)
  t.writeUInt16LE(1, 8)
  t.writeUInt16LE(0x0112, 10)
  t.writeUInt16LE(3, 12)
  t.writeUInt32LE(1, 14)
  t.writeUInt16LE(6, 18)
  const payload = Buffer.concat([Buffer.from('Exif\0\0', 'latin1'), t])
  const seg = Buffer.alloc(4)
  seg.writeUInt16BE(0xffe1, 0)
  seg.writeUInt16BE(payload.length + 2, 2)
  const out = Buffer.concat([j.subarray(0, 2), seg, payload, j.subarray(2)])
  fs.writeFileSync(file, out)
  return out
}

/** A plain grey JPEG made by ffmpeg. Needs HAS_FFMPEG. */
export function greyJpeg(file: string, w: number, h: number): Buffer {
  execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', `color=gray:s=${w}x${h}`, '-frames:v', '1', file])
  return fs.readFileSync(file)
}

// ---------------------------------------------------------------- words --

/** True when `text` holds `word` as a whole word (not inside a longer one), ignoring case. */
export function hasWord(text: string, word: string): boolean {
  const esc = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return new RegExp(`(^|[^a-z0-9])${esc}([^a-z0-9]|$)`, 'i').test(text)
}

// ---------------------------------------------------------------- types --

/**
 * A module's functions with their argument and answer types loosened. The
 * judging tests build events, items and sealed keys as plain objects, the way
 * the phone and the seal write them to disk, and read the answers the same
 * way; the modules themselves stay strictly typed.
 */
export type Loose<T> = { [K in keyof T]: T[K] extends (...a: never[]) => unknown ? (...a: any[]) => any : T[K] }
