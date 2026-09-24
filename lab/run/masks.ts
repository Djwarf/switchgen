/**
 * The mask for a region test: a white rectangle on black, as a PNG the size
 * of the upright reference.
 *
 * The app's region graph reads the mask's red channel (refine.ts: an opaque
 * picture read on its alpha channel comes back empty), and a grey PNG with no
 * alpha reads as red = grey in ComfyUI, so white is "redraw" and black is
 * "keep". Written with node:zlib alone.
 */
import { crc32, deflateSync } from 'node:zlib'

export type Rect = { x: number; y: number; w: number; h: number }

/** The smallest side a marked area may have, in pixels. */
export const MIN_SIDE = 8

/**
 * The rectangle in whole pixels, cut to the picture. Throws, in words the lab
 * page shows, when it is not a rectangle or leaves too little inside.
 */
export function normalizeRect(width: number, height: number, rect: Rect): Rect {
  const nums = [rect?.x, rect?.y, rect?.w, rect?.h]
  if (!nums.every((n) => typeof n === 'number' && Number.isFinite(n))) {
    throw new Error('The area must be four numbers: x, y, width and height, in pixels.')
  }
  const x0 = Math.max(0, Math.round(rect.x))
  const y0 = Math.max(0, Math.round(rect.y))
  const x1 = Math.min(width, Math.round(rect.x + rect.w))
  const y1 = Math.min(height, Math.round(rect.y + rect.h))
  if (x1 - x0 < MIN_SIDE || y1 - y0 < MIN_SIDE) {
    throw new Error(`The area must be at least ${MIN_SIDE} pixels wide and tall, inside the ${width}×${height} photo.`)
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
}

function chunk(type: string, data: Uint8Array): Buffer {
  const head = Buffer.alloc(8)
  head.writeUInt32BE(data.length, 0)
  head.write(type, 4, 'ascii')
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])) >>> 0, 0)
  return Buffer.concat([head, data, crc])
}

/** An 8-bit grey PNG of w×h, white inside `rect` and black outside. */
export function rectMaskPng(w: number, h: number, rect: Rect): Buffer {
  if (!(Number.isInteger(w) && Number.isInteger(h) && w > 0 && h > 0)) throw new Error('the mask size must be whole pixels')
  const r = normalizeRect(w, h, rect)
  const stride = w + 1
  const raw = Buffer.alloc(stride * h)
  // Each row starts with filter byte 0 (none); black rows are already zero.
  const lit = Buffer.alloc(stride)
  lit.fill(0xff, 1 + r.x, 1 + r.x + r.w)
  for (let y = r.y; y < r.y + r.h; y++) lit.copy(raw, y * stride)
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(w, 0)
  ihdr.writeUInt32BE(h, 4)
  ihdr[8] = 8 // bit depth
  ihdr[9] = 0 // grey
  ihdr[10] = 0
  ihdr[11] = 0
  ihdr[12] = 0
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ])
}
