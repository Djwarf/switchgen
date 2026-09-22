import { describe, expect, it } from 'vitest'
import { offerNotes, offersFor } from '../src/components/result/offers'
import type { HistoryEntry } from '../src/lib/history'
import { deriveHiresFix, rebuildable, regionOrigin } from '../src/lib/refine'
import { FAMILIES } from '../src/lib/workflows'

// A region pass redraws one area of a finished picture. Its record cannot be
// replayed, so what the desk offers under it, and where "again" leads, are
// worked out from the picture it was drawn on.
const il = FAMILIES.find((f) => f.id === 'sdxl-illustrious')!
const ids = (offers: { id: string }[]) => offers.map((o) => o.id)

describe('the rows under a finished picture', () => {
  it('offers a region on a picture made larger, drawn with the plain graph the bench uses', () => {
    const bigger = deriveHiresFix(il)!
    expect(ids(offersFor(bigger))).not.toContain('refine')
    expect(ids(offersFor(bigger, { region: il }))).toContain('refine')
  })

  it('leaves out a pass ComfyUI cannot run, and says why instead', () => {
    expect(ids(offersFor(il, { blocks: { face: 'x' } }))).not.toContain('face')
    expect(offerNotes(il, { blocks: { face: 'x' } })).toEqual(['x'])
  })

  it('makes another region pass on the picture it came from, and costs nothing until the bench draws', () => {
    const again = offersFor(il, { rebuild: false, region: il, regionPass: { from: 'No. 40' } }).find((o) => o.id === 'again')
    expect(again).toBeDefined()
    expect(again!.cost).toBeNull()
    expect(again!.what).toContain('No. 40')
  })

  it('offers no "again" when that picture is gone, and says so', () => {
    const opts = { rebuild: false, region: il, regionPass: { from: null } }
    expect(ids(offersFor(il, opts))).not.toContain('again')
    expect(offerNotes(il, opts).join(' ')).toContain('that picture is no longer here')
  })
})

describe('the picture a region pass was drawn on', () => {
  const pass = (source: HistoryEntry['source']) => ({ variant: 'refine' as const, source })
  const original = { id: 'orig', missing: false } as HistoryEntry
  const ref = { filename: 'orig.png', subfolder: '', type: 'output' }

  it('is its record while that record and its file are here', () => {
    expect(regionOrigin(pass({ name: 'in.png', ref, fromEntryId: 'orig' }), [original])).toEqual({ kind: 'record', entry: original })
  })

  it('falls back to the output file, then to ComfyUI\'s input copy, then to nothing', () => {
    expect(regionOrigin(pass({ name: 'in.png', ref, fromEntryId: 'orig' }), [{ ...original, missing: true }])?.kind).toBe('output')
    expect(regionOrigin(pass({ name: 'in.png', ref }), [])?.kind).toBe('output')
    expect(regionOrigin(pass({ name: 'in.png' }), [])?.kind).toBe('input')
    expect(regionOrigin(pass(undefined), [])?.kind).toBe('gone')
  })

  it('is not asked of anything but a region pass', () => {
    expect(regionOrigin({ variant: null, source: { name: 'in.png' } }, [])).toBeNull()
    expect(regionOrigin({ variant: 'img2img', source: { name: 'in.png' } }, [])).toBeNull()
  })

  it('cannot be rebuilt from the region pass\'s own record', () => {
    expect(rebuildable({ variant: 'refine', mode: 'i2i', megapixels: 1 })).toBe(false)
    expect(rebuildable({ variant: null, mode: 't2i', megapixels: undefined })).toBe(true)
  })
})
