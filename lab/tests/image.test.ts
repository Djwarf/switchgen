/**
 * Reference pictures (TEST PLAN, C, refs): sizes read from the header, upright
 * after the EXIF orientation flag; the clean copy ComfyUI reads, with the
 * camera's metadata gone and the orientation kept; the rectangle mask.
 * Every picture here is generated; the user's photos are never read. The
 * cross-checks that decode pictures use PIL and are skipped without it.
 */
import fs from 'node:fs'
import path from 'node:path'
import { inflateSync } from 'node:zlib'
import { afterAll, describe, expect, it } from 'vitest'
import { cleanCopy, exifOrientation, orientationTiff, sizeOf } from '../run/imagesize.ts'
import { normalizeRect, rectMaskPng } from '../run/masks.ts'
import { HAS_PIL, jpegHeader, pilFixtures, pilRead, png, removeTemp, sidewaysCatStandIn, tempDir, webpHeader } from './helpers.ts'

afterAll(removeTemp)

const dims = (b: Buffer) => {
  const s = sizeOf(b)
  return [s.width, s.height]
}

/** The grey values of an 8-bit grey PNG with no filtering, row by row. */
function greyRows(bytes: Buffer): { w: number; h: number; at: (x: number, y: number) => number } {
  const w = bytes.readUInt32BE(16)
  const h = bytes.readUInt32BE(20)
  let i = 8
  const idat: Buffer[] = []
  while (i < bytes.length) {
    const len = bytes.readUInt32BE(i)
    if (bytes.toString('ascii', i + 4, i + 8) === 'IDAT') idat.push(bytes.subarray(i + 8, i + 8 + len))
    i += 12 + len
  }
  const raw = inflateSync(Buffer.concat(idat))
  expect(raw.length).toBe((w + 1) * h)
  for (let y = 0; y < h; y++) expect(raw[y * (w + 1)]).toBe(0)
  return { w, h, at: (x, y) => raw[y * (w + 1) + 1 + x] }
}

describe('sizeOf', () => {
  it('reads a PNG, and turns it upright for orientations 5 to 8 only', () => {
    expect(dims(png(120, 80))).toEqual([120, 80])
    for (let o = 1; o <= 8; o++) {
      const s = sizeOf(png(120, 80, [1, 2, 3], { orientation: o }))
      expect(s.orientation).toBe(o)
      expect(s.stored).toEqual({ width: 120, height: 80 })
      expect([s.width, s.height], `orientation ${o}`).toEqual(o >= 5 ? [80, 120] : [120, 80])
    }
  })
  it('reads baseline and progressive JPEGs, with the EXIF orientation', () => {
    expect(dims(jpegHeader({ width: 120, height: 80 }))).toEqual([120, 80])
    for (const progressive of [false, true]) {
      for (let o = 1; o <= 8; o++) {
        const s = sizeOf(jpegHeader({ width: 120, height: 80, orientation: o, progressive }))
        expect(s.kind).toBe('jpeg')
        expect([s.width, s.height], `${progressive ? 'progressive' : 'baseline'} ${o}`).toEqual(o >= 5 ? [80, 120] : [120, 80])
      }
    }
  })
  it('reads lossy, lossless and extended WebP headers, the last with its orientation', () => {
    expect(dims(webpHeader('VP8', 120, 80))).toEqual([120, 80])
    expect(dims(webpHeader('VP8L', 333, 777))).toEqual([333, 777])
    expect(dims(webpHeader('VP8X', 333, 777))).toEqual([333, 777])
    for (let o = 1; o <= 8; o++) {
      const s = sizeOf(webpHeader('VP8X', 120, 80, o))
      expect([s.width, s.height, s.orientation], `orientation ${o}`).toEqual(o >= 5 ? [80, 120, o] : [120, 80, o])
    }
  })
  it('gives the cat stand-in, stored 4080 x 3060 with orientation 6, as 3060 x 4080 upright', () => {
    const s = sizeOf(sidewaysCatStandIn())
    expect(s).toMatchObject({ width: 3060, height: 4080, orientation: 6, stored: { width: 4080, height: 3060 } })
  })
  it('refuses what is not a picture', () => {
    expect(() => sizeOf(Buffer.from('hello world, not a picture'))).toThrow(/not a PNG, JPEG or WebP/)
    expect(() => sizeOf(Buffer.from([0xff, 0xd8, 0xff, 0xd9]))).toThrow()
  })
})

describe('exifOrientation', () => {
  it('tolerates junk and reads both byte orders', () => {
    expect(exifOrientation(Buffer.from('nonsense'))).toBe(1)
    expect(exifOrientation(Buffer.alloc(0))).toBe(1)
    for (let o = 1; o <= 8; o++) {
      expect(exifOrientation(orientationTiff(o))).toBe(o)
      expect(exifOrientation(Buffer.concat([Buffer.from('Exif\0\0', 'latin1'), orientationTiff(o)]))).toBe(o)
    }
  })
})

describe('cleanCopy', () => {
  it('keeps a JPEG\'s orientation and picture data, and drops the camera, the comment and the trailer', () => {
    const src = sidewaysCatStandIn()
    const out = cleanCopy(src)
    expect(sizeOf(out)).toMatchObject({ width: 3060, height: 4080, orientation: 6 })
    for (const secret of ['PhoneModel', 'secret comment', 'TRAILER']) expect(out.includes(Buffer.from(secret)), secret).toBe(false)
    expect(out.includes(Buffer.from([0x12, 0x34, 0xff, 0x00, 0x56]))).toBe(true)
    expect(out.subarray(-2)).toEqual(Buffer.from([0xff, 0xd9]))
    expect(cleanCopy(src)).toEqual(out)
  })
  it('keeps a PNG\'s orientation and pixels, and drops its text', () => {
    const src = png(12, 10, [9, 8, 7], { orientation: 8, text: '{"ckpt_name":"NoobAI-XL-v1.1.safetensors"}' })
    const out = cleanCopy(src)
    expect(sizeOf(out)).toMatchObject({ width: 10, height: 12, orientation: 8 })
    expect(out.includes(Buffer.from('ckpt_name'))).toBe(false)
    expect(out.includes(Buffer.from('tEXt'))).toBe(false)
  })
  it('keeps an extended WebP\'s orientation and drops the rest of its EXIF', () => {
    const out = cleanCopy(webpHeader('VP8X', 120, 80, 5))
    expect(sizeOf(out)).toMatchObject({ width: 80, height: 120, orientation: 5 })
  })
  it.skipIf(!HAS_PIL)('checked with PIL: upright size and pixels kept, maker, GPS, comments and trailing data gone, the same bytes every time', () => {
    const dir = pilFixtures(tempDir('lab-pil-'))
    for (const f of ['side.jpg', 'prog.jpg', 'side.png', 'side.webp', 'plain.jpg', 'plain.png', 'lossy.webp', 'odd-alpha.webp']) {
      let input = fs.readFileSync(path.join(dir, f))
      if (f === 'side.jpg') input = Buffer.concat([input, Buffer.from('TRAILER with secret MPF data')])
      const out = cleanCopy(input)
      const tmp = path.join(dir, `clean-${f}`)
      fs.writeFileSync(tmp, out)
      const a = pilRead(path.join(dir, f))
      const b = pilRead(tmp)
      expect(b.upright, f).toEqual(a.upright)
      expect(b.px, f).toEqual(a.px)
      expect(Object.keys(b.exif).every((k) => k === '274'), `${f}: ${Object.keys(b.exif)}`).toBe(true)
      expect(b.gps, f).toBe(false)
      expect(JSON.stringify(b.info), f).not.toMatch(/secret/)
      expect(sizeOf(out).orientation, f).toBe(sizeOf(input).orientation)
      for (const s of ['secret', 'Maker', 'PhoneModel']) expect(out.includes(Buffer.from(s)), `${f} ${s}`).toBe(false)
      expect(cleanCopy(input), `${f} deterministic`).toEqual(out)
    }
    // The sideways ones really are sideways, so the orientation was worth keeping.
    expect(pilRead(path.join(dir, 'side.jpg')).upright).toEqual([80, 120])
    expect(pilRead(path.join(dir, 'side.jpg')).gps).toBe(true)
  })
})

describe('rectMaskPng', () => {
  it('inflates to 255 inside the rectangle and 0 outside', () => {
    const m = greyRows(rectMaskPng(40, 30, { x: 10, y: 5, w: 12, h: 8 }))
    expect([m.w, m.h]).toEqual([40, 30])
    for (let y = 0; y < 30; y++) for (let x = 0; x < 40; x++) expect(m.at(x, y), `${x},${y}`).toBe(x >= 10 && x < 22 && y >= 5 && y < 13 ? 255 : 0)
  })
  it('is drawn at the cat\'s upright size, quickly and small', () => {
    const t = Date.now()
    const b = rectMaskPng(3060, 4080, { x: 100, y: 200, w: 800, h: 600 })
    expect(Date.now() - t).toBeLessThan(3000)
    expect(b.length).toBeLessThan(200_000)
    expect(dims(b)).toEqual([3060, 4080])
  })
  it('cuts the rectangle to the picture and refuses one too small or not numbers', () => {
    expect(normalizeRect(100, 100, { x: -5, y: 90.4, w: 50, h: 50 })).toEqual({ x: 0, y: 90, w: 45, h: 10 })
    expect(() => rectMaskPng(40, 30, { x: 39, y: 0, w: 5, h: 5 })).toThrow(/at least 8 pixels/)
    expect(() => normalizeRect(40, 30, { x: 'a' as unknown as number, y: 0, w: 10, h: 10 })).toThrow(/four numbers/)
  })
  it.skipIf(!HAS_PIL)('checked with PIL: an 8-bit grey picture, white inside', () => {
    const f = path.join(tempDir('lab-mask-'), 'mask.png')
    fs.writeFileSync(f, rectMaskPng(40, 30, { x: 10, y: 5, w: 12, h: 8 }))
    const r = pilRead(f)
    expect(r.size).toEqual([40, 30])
    expect(r.px).toEqual([0, 0, 0])
  })
})
