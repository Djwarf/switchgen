/**
 * The size of a PNG, JPEG or WebP from its header, as it is shown.
 *
 * A phone stores a portrait photo sideways and says so in an EXIF
 * orientation flag. Browsers turn it upright, and so does ComfyUI (LoadImage
 * and LoadImageMask both run ImageOps.exif_transpose), so every size the lab
 * measures is the upright one: a mask drawn on the lab page and the crop the
 * region graph makes both live in that frame. `stored` keeps the size the
 * pixels are written at.
 *
 * `cleanCopy` makes the copy ComfyUI reads from the outputs folder, which
 * ComfyUI's /view serves to the tailnet: the photo with its camera, place and
 * text metadata taken out, keeping only the orientation flag (and a JPEG's
 * colour profile), so it still loads upright.
 */
import { crc32 } from 'node:zlib'

export type ImageKind = 'png' | 'jpeg' | 'webp'

export type ImageSize = {
  /** Upright width, after the orientation flag. */
  width: number
  /** Upright height, after the orientation flag. */
  height: number
  kind: ImageKind
  /** EXIF orientation, 1 to 8; 1 when there is none. */
  orientation: number
  /** The size the pixels are stored at. */
  stored: { width: number; height: number }
}

const u16be = (b: Uint8Array, i: number) => (b[i] << 8) | b[i + 1]
const u16le = (b: Uint8Array, i: number) => b[i] | (b[i + 1] << 8)
const u24le = (b: Uint8Array, i: number) => b[i] | (b[i + 1] << 8) | (b[i + 2] << 16)
const u32be = (b: Uint8Array, i: number) => ((b[i] << 24) >>> 0) + ((b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3])
const u32le = (b: Uint8Array, i: number) => ((b[i + 3] << 24) >>> 0) + ((b[i + 2] << 16) | (b[i + 1] << 8) | b[i])
const ascii = (b: Uint8Array, i: number, n: number) => String.fromCharCode(...b.subarray(i, i + n))

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

/** Which of the three formats these bytes are, or null. */
export function sniff(bytes: Uint8Array): ImageKind | null {
  if (bytes.length >= 24 && PNG_SIG.every((v, i) => bytes[i] === v)) return 'png'
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'jpeg'
  if (bytes.length >= 16 && ascii(bytes, 0, 4) === 'RIFF' && ascii(bytes, 8, 4) === 'WEBP') return 'webp'
  return null
}

export const EXT_OF: Record<ImageKind, 'png' | 'jpg' | 'webp'> = { png: 'png', jpeg: 'jpg', webp: 'webp' }
export const MIME_OF: Record<ImageKind, string> = { png: 'image/png', jpeg: 'image/jpeg', webp: 'image/webp' }

/**
 * The orientation in a TIFF block (EXIF's body), 1 to 8, or 1 when it has
 * none or cannot be read. Accepts the block with or without the "Exif\0\0"
 * lead some PNG and WebP writers leave on it.
 */
export function exifOrientation(block: Uint8Array): number {
  let t = block
  if (t.length >= 6 && ascii(t, 0, 4) === 'Exif' && t[4] === 0 && t[5] === 0) t = t.subarray(6)
  if (t.length < 8) return 1
  const order = ascii(t, 0, 2)
  if (order !== 'II' && order !== 'MM') return 1
  const le = order === 'II'
  const r16 = (i: number) => (le ? u16le(t, i) : u16be(t, i))
  const r32 = (i: number) => (le ? u32le(t, i) : u32be(t, i))
  if (r16(2) !== 42) return 1
  const ifd = r32(4)
  if (ifd < 8 || ifd + 2 > t.length) return 1
  const count = r16(ifd)
  for (let k = 0; k < count; k++) {
    const e = ifd + 2 + k * 12
    if (e + 12 > t.length) return 1
    if (r16(e) !== 0x0112) continue
    const type = r16(e + 2)
    // SHORT is the type the standard names; a LONG is read too, since some
    // writers use it and readers take it.
    const v = type === 3 ? r16(e + 8) : type === 4 ? r32(e + 8) : 0
    return v >= 1 && v <= 8 ? v : 1
  }
  return 1
}

/** A TIFF block holding only an orientation, big-endian. */
export function orientationTiff(orientation: number): Buffer {
  const b = Buffer.alloc(26)
  b.write('MM', 0, 'ascii')
  b.writeUInt16BE(42, 2)
  b.writeUInt32BE(8, 4)
  b.writeUInt16BE(1, 8)
  b.writeUInt16BE(0x0112, 10)
  b.writeUInt16BE(3, 12)
  b.writeUInt32BE(1, 14)
  b.writeUInt16BE(orientation, 18)
  b.writeUInt32BE(0, 22)
  return b
}

const turned = (o: number) => o >= 5 && o <= 8

function sized(kind: ImageKind, width: number, height: number, orientation: number): ImageSize {
  if (!(width > 0 && height > 0)) throw new Error(`the ${kind} header gives no size`)
  const o = orientation >= 1 && orientation <= 8 ? orientation : 1
  return turned(o)
    ? { width: height, height: width, kind, orientation: o, stored: { width, height } }
    : { width, height, kind, orientation: o, stored: { width, height } }
}

// ------------------------------------------------------------------ JPEG --

const isSof = (m: number) => m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc
const standalone = (m: number) => m === 0x01 || m === 0xd8 || (m >= 0xd0 && m <= 0xd7)

type JpegSegment = { marker: number; start: number; end: number; body: Uint8Array }

/** The segments before the first scan, and where that scan's marker starts. */
function jpegHead(b: Uint8Array): { segments: JpegSegment[]; sos: number } {
  const segments: JpegSegment[] = []
  let i = 2
  while (i < b.length) {
    if (b[i] !== 0xff) throw new Error('the JPEG is damaged: a segment does not start with a marker')
    const start = i
    while (i < b.length && b[i] === 0xff) i++
    if (i >= b.length) break
    const marker = b[i]
    i++
    if (standalone(marker)) continue
    if (marker === 0xd9) throw new Error('the JPEG ends before its picture data')
    if (i + 2 > b.length) throw new Error('the JPEG is cut short')
    const len = u16be(b, i)
    if (len < 2 || i + len > b.length) throw new Error('the JPEG is cut short')
    if (marker === 0xda) return { segments, sos: start }
    segments.push({ marker, start, end: i + len, body: b.subarray(i + 2, i + len) })
    i += len
  }
  throw new Error('the JPEG has no picture data')
}

/** Where the JPEG's EOI marker ends, walking every scan from `sos`. */
function jpegEnd(b: Uint8Array, sos: number): number {
  let i = sos
  while (i < b.length) {
    // A marker: the SOS itself, or the tables between a progressive JPEG's scans.
    if (b[i] !== 0xff) break
    while (i < b.length && b[i] === 0xff) i++
    if (i >= b.length) break
    const marker = b[i]
    i++
    if (marker === 0xd9) return i
    if (standalone(marker)) continue
    if (i + 2 > b.length) break
    i += u16be(b, i)
    if (marker !== 0xda) continue
    // Entropy-coded data runs to the next marker that is not a stuffed zero,
    // a restart, or a fill byte.
    while (i < b.length) {
      if (b[i] !== 0xff) { i++; continue }
      const n = b[i + 1]
      if (n === 0x00 || (n >= 0xd0 && n <= 0xd7)) { i += 2; continue }
      if (n === 0xff) { i++; continue }
      break
    }
  }
  // No end marker found: keep everything, as decoders do.
  return b.length
}

function jpegSize(b: Uint8Array): ImageSize {
  const { segments } = jpegHead(b)
  let orientation = 1
  let w = 0
  let h = 0
  for (const s of segments) {
    if (s.marker === 0xe1 && s.body.length > 6 && ascii(s.body, 0, 4) === 'Exif' && s.body[4] === 0 && s.body[5] === 0 && orientation === 1) {
      orientation = exifOrientation(s.body.subarray(6))
    }
    if (isSof(s.marker) && !w && s.body.length >= 5) {
      h = u16be(s.body, 1)
      w = u16be(s.body, 3)
    }
  }
  return sized('jpeg', w, h, orientation)
}

// ------------------------------------------------------------------- PNG --

type PngChunk = { type: string; start: number; end: number; data: Uint8Array }

function pngChunks(b: Uint8Array): PngChunk[] {
  const out: PngChunk[] = []
  let i = 8
  while (i + 12 <= b.length) {
    const len = u32be(b, i)
    const type = ascii(b, i + 4, 4)
    const end = i + 12 + len
    if (end > b.length) throw new Error('the PNG is cut short')
    out.push({ type, start: i, end, data: b.subarray(i + 8, i + 8 + len) })
    i = end
    if (type === 'IEND') break
  }
  return out
}

function pngSize(b: Uint8Array): ImageSize {
  if (ascii(b, 12, 4) !== 'IHDR') throw new Error('the PNG has no header chunk')
  const w = u32be(b, 16)
  const h = u32be(b, 20)
  let orientation = 1
  for (const c of pngChunks(b)) {
    if (c.type === 'eXIf') {
      orientation = exifOrientation(c.data)
      break
    }
    if (c.type === 'IDAT') break
  }
  return sized('png', w, h, orientation)
}

function pngChunk(type: string, data: Uint8Array): Buffer {
  const head = Buffer.alloc(8)
  head.writeUInt32BE(data.length, 0)
  head.write(type, 4, 'ascii')
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])) >>> 0, 0)
  return Buffer.concat([head, data, crc])
}

// ------------------------------------------------------------------ WebP --

type RiffChunk = { fourcc: string; start: number; end: number; data: Uint8Array }

function webpChunks(b: Uint8Array): RiffChunk[] {
  const out: RiffChunk[] = []
  const riffEnd = Math.min(b.length, 8 + u32le(b, 4))
  let i = 12
  while (i + 8 <= riffEnd) {
    const fourcc = ascii(b, i, 4)
    const len = u32le(b, i + 4)
    const dataEnd = i + 8 + len
    if (dataEnd > b.length) throw new Error('the WebP is cut short')
    const end = Math.min(riffEnd, dataEnd + (len & 1))
    out.push({ fourcc, start: i, end, data: b.subarray(i + 8, dataEnd) })
    i = end
  }
  return out
}

function webpSize(b: Uint8Array): ImageSize {
  const chunks = webpChunks(b)
  const first = chunks[0]
  if (!first) throw new Error('the WebP has no image chunk')
  let w = 0
  let h = 0
  let orientation = 1
  const d = first.data
  if (first.fourcc === 'VP8 ') {
    if (d.length < 10 || d[3] !== 0x9d || d[4] !== 0x01 || d[5] !== 0x2a) throw new Error('the WebP (lossy) header is damaged')
    w = u16le(d, 6) & 0x3fff
    h = u16le(d, 8) & 0x3fff
  } else if (first.fourcc === 'VP8L') {
    if (d.length < 5 || d[0] !== 0x2f) throw new Error('the WebP (lossless) header is damaged')
    const bits = u32le(d, 1)
    w = (bits & 0x3fff) + 1
    h = ((bits >>> 14) & 0x3fff) + 1
  } else if (first.fourcc === 'VP8X') {
    if (d.length < 10) throw new Error('the WebP (extended) header is damaged')
    w = u24le(d, 4) + 1
    h = u24le(d, 7) + 1
    const exif = chunks.find((c) => c.fourcc === 'EXIF')
    if (exif) orientation = exifOrientation(exif.data)
  } else {
    throw new Error(`the WebP starts with an unknown chunk ${JSON.stringify(first.fourcc)}`)
  }
  return sized('webp', w, h, orientation)
}

// ------------------------------------------------------------------ both --

/** Upright size, format and orientation of a PNG, JPEG or WebP. Throws on anything else. */
export function sizeOf(bytes: Uint8Array): ImageSize {
  const kind = sniff(bytes)
  if (kind === 'png') return pngSize(bytes)
  if (kind === 'jpeg') return jpegSize(bytes)
  if (kind === 'webp') return webpSize(bytes)
  throw new Error('not a PNG, JPEG or WebP picture')
}

/** Kept in a JPEG copy: JFIF, the colour profile, Adobe's colour transform, and every table. */
function keepJpegSegment(s: JpegSegment): boolean {
  if (s.marker === 0xe0 || s.marker === 0xee) return true
  if (s.marker === 0xe2) return s.body.length >= 12 && ascii(s.body, 0, 12) === 'ICC_PROFILE\0'
  if (s.marker >= 0xe1 && s.marker <= 0xef) return false
  if (s.marker === 0xfe) return false
  return true
}

function cleanJpeg(b: Uint8Array): Buffer {
  const { segments, sos } = jpegHead(b)
  let orientation = 1
  for (const s of segments) {
    if (s.marker === 0xe1 && s.body.length > 6 && ascii(s.body, 0, 4) === 'Exif' && s.body[4] === 0 && s.body[5] === 0) {
      orientation = exifOrientation(s.body.subarray(6))
      break
    }
  }
  const parts: Uint8Array[] = [Buffer.from([0xff, 0xd8])]
  const kept = segments.filter(keepJpegSegment)
  const jfif = kept.length && kept[0].marker === 0xe0 ? 1 : 0
  for (const s of kept.slice(0, jfif)) parts.push(b.subarray(s.start, s.end))
  if (orientation !== 1) {
    const tiff = orientationTiff(orientation)
    const head = Buffer.alloc(4 + 6)
    head[0] = 0xff
    head[1] = 0xe1
    head.writeUInt16BE(2 + 6 + tiff.length, 2)
    head.write('Exif\0\0', 4, 'latin1')
    parts.push(head, tiff)
  }
  for (const s of kept.slice(jfif)) parts.push(b.subarray(s.start, s.end))
  // The picture data up to its end marker; anything after it (a phone's
  // second picture, a maker's trailer) is left behind.
  parts.push(b.subarray(sos, jpegEnd(b, sos)))
  return Buffer.concat(parts)
}

const PNG_DROP = new Set(['tEXt', 'zTXt', 'iTXt', 'tIME', 'eXIf'])

function cleanPng(b: Uint8Array): Buffer {
  const { orientation } = pngSize(b)
  const parts: Uint8Array[] = [b.subarray(0, 8)]
  let placed = orientation === 1
  for (const c of pngChunks(b)) {
    if (!placed && (c.type === 'IDAT' || c.type === 'IEND')) {
      parts.push(pngChunk('eXIf', orientationTiff(orientation)))
      placed = true
    }
    if (PNG_DROP.has(c.type)) continue
    parts.push(b.subarray(c.start, c.end))
  }
  return Buffer.concat(parts)
}

function cleanWebp(b: Uint8Array): Buffer {
  const chunks = webpChunks(b)
  if (chunks[0]?.fourcc !== 'VP8X') return Buffer.from(b.subarray(0, Math.min(b.length, 8 + u32le(b, 4))))
  const { orientation } = webpSize(b)
  const parts: Uint8Array[] = []
  const vp8x = Buffer.from(chunks[0].data)
  // Flags: 0x08 EXIF, 0x04 XMP.
  vp8x[0] = (vp8x[0] & ~0x0c) | (orientation !== 1 ? 0x08 : 0)
  const head = Buffer.alloc(8)
  head.write('VP8X', 0, 'ascii')
  head.writeUInt32LE(vp8x.length, 4)
  parts.push(head, vp8x)
  if (vp8x.length & 1) parts.push(Buffer.alloc(1))
  for (const c of chunks.slice(1)) {
    if (c.fourcc === 'EXIF' || c.fourcc === 'XMP ') continue
    parts.push(b.subarray(c.start, c.end))
  }
  if (orientation !== 1) {
    const tiff = orientationTiff(orientation)
    const h = Buffer.alloc(8)
    h.write('EXIF', 0, 'ascii')
    h.writeUInt32LE(tiff.length, 4)
    parts.push(h, tiff)
  }
  const body = Buffer.concat(parts)
  const riff = Buffer.alloc(12)
  riff.write('RIFF', 0, 'ascii')
  riff.writeUInt32LE(4 + body.length, 4)
  riff.write('WEBP', 8, 'ascii')
  return Buffer.concat([riff, body])
}

/**
 * The picture with its metadata taken out: no camera, place, time, text or
 * XMP, no trailing second picture. The orientation flag stays (rewritten on
 * its own), so the copy still loads upright, and a JPEG keeps its colour
 * profile. The same bytes in always give the same bytes out.
 */
export function cleanCopy(bytes: Uint8Array): Buffer {
  const kind = sniff(bytes)
  if (kind === 'jpeg') return cleanJpeg(bytes)
  if (kind === 'png') return cleanPng(bytes)
  if (kind === 'webp') return cleanWebp(bytes)
  throw new Error('not a PNG, JPEG or WebP picture')
}
