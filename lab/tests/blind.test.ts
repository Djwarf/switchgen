/**
 * Blind copies (TEST PLAN, D, leak check): the chunk check that refuses any
 * metadata, and, with ffmpeg, copies made upright from a picture stored
 * sideways, matching ffmpeg's own auto-rotation, with a region's outline
 * drawn where the area is in the upright picture. Every picture is generated.
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { blindCopy, fitSize, metadataProblems, orientFilters, webpChunks } from '../judge/blind.ts'
import { sizeOf } from '../run/imagesize.ts'
import { HAS_FFMPEG, png, removeTemp, sidewaysJpeg, tempDir, webpHeader } from './helpers.ts'

afterAll(removeTemp)

/** Decode a WebP copy to raw RGB with ffmpeg. */
function raw(file: string): { w: number; h: number; px: Buffer } {
  const { width: w, height: h } = sizeOf(fs.readFileSync(file))
  const px = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', file, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], { maxBuffer: 64 << 20 })
  expect(px.length).toBe(w * h * 3)
  return { w, h, px }
}

describe('the chunk check', () => {
  it('finds EXIF, XMP, a colour profile and unknown chunks, and passes picture data alone', () => {
    expect(metadataProblems(webpHeader('VP8', 16, 16))).toEqual([])
    expect(metadataProblems(webpHeader('VP8L', 16, 16))).toEqual([])
    const withExif = webpHeader('VP8X', 16, 16, 6)
    expect(metadataProblems(withExif).join(' ')).toMatch(/EXIF/)
    expect(webpChunks(withExif).map((c) => c.fourcc)).toEqual(['VP8X', 'VP8L', 'EXIF'])
    expect(metadataProblems(png(4, 4))).toEqual(['not a WebP file'])
  })
  it('turns each orientation upright and fits the long edge', () => {
    expect(orientFilters(1)).toEqual([])
    expect(orientFilters(6)).toEqual(['transpose=1'])
    expect(orientFilters(8)).toEqual(['transpose=2'])
    expect(fitSize(3060, 4080, 640)).toEqual({ width: 480, height: 640 })
    expect(fitSize(100, 50, 640)).toEqual({ width: 100, height: 50 })
    expect(fitSize(100, 50, 640, true)).toEqual({ width: 640, height: 320 })
  })
})

describe.skipIf(!HAS_FFMPEG)('blind copies made with ffmpeg', () => {
  it('a JPEG stored sideways (orientation 6) comes out upright, as ffmpeg\'s own auto-rotation shows it, with no metadata', async () => {
    const dir = tempDir('lab-blind-')
    const src = path.join(dir, 'side.jpg')
    sidewaysJpeg(src, 160, 120)
    const dst = path.join(dir, 'copy.webp')
    const size = await blindCopy('ffmpeg', src, dst, 160, 100, { nice: false })
    expect(size).toEqual({ width: 120, height: 160 })
    const buf = fs.readFileSync(dst)
    expect(metadataProblems(buf)).toEqual([])
    // ffmpeg's own reading of the same file, turned by its display matrix, made the same way.
    const ref = path.join(dir, 'ref.webp')
    execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', src, '-frames:v', '1', '-vf', 'scale=120:160:flags=lanczos', '-map_metadata', '-1', '-fflags', '+bitexact', '-flags:v', '+bitexact', '-c:v', 'libwebp', '-quality', '100', '-compression_level', '4', '-f', 'webp', ref])
    const a = raw(dst)
    const b = raw(ref)
    expect([a.w, a.h]).toEqual([120, 160])
    expect([b.w, b.h]).toEqual([120, 160])
    expect(a.px.equals(b.px)).toBe(true)
    // The blue band, on the left of the stored picture, is along the top once upright.
    const at = (x: number, y: number) => [...a.px.subarray((y * a.w + x) * 3, (y * a.w + x) * 3 + 3)]
    expect(at(60, 5)[2]).toBeGreaterThan(150)
    expect(at(60, 150)[0]).toBeGreaterThan(150)
  })
  it('a region\'s outline is drawn at the rectangle in upright pixels', async () => {
    const dir = tempDir('lab-outline-')
    const src = path.join(dir, 'side.jpg')
    sidewaysJpeg(src, 160, 120)
    const dst = path.join(dir, 'outlined.webp')
    const rect = { x: 20, y: 100, w: 60, h: 40 }
    await blindCopy('ffmpeg', src, dst, 160, 100, { nice: false, outline: rect })
    const a = raw(dst)
    const at = (x: number, y: number) => [...a.px.subarray((y * a.w + x) * 3, (y * a.w + x) * 3 + 3)]
    const yellow = (p: number[]) => p[0] > 200 && p[1] > 170 && p[2] < 100
    expect(yellow(at(rect.x + 1, rect.y + 20)), 'left edge').toBe(true)
    expect(yellow(at(rect.x + 30, rect.y + 1)), 'top edge').toBe(true)
    expect(yellow(at(rect.x + 30, rect.y + 20)), 'inside is left alone').toBe(false)
    // The same numbers read in the stored (sideways) frame would put the box elsewhere.
    expect(yellow(at(rect.y + 1, 100 - rect.x)), 'not in the stored frame').toBe(false)
  })
})
