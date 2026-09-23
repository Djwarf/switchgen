import { describe, expect, it } from 'vitest'
import { offerNotes, offersFor } from '../src/components/result/offers'
import type { HistoryEntry } from '../src/lib/history'
import { deriveHiresFix, rebuildable, regionOrigin, regionPicture } from '../src/lib/refine'
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

describe('the picture the Archive hands the region bench', () => {
  const regionPassRecord = {
    id: 'pass',
    no: 41,
    at: 1_700_000_000_000,
    desk: 'images',
    kind: 'image',
    mode: 'i2i',
    file: { filename: 'pass.png', subfolder: '', type: 'output' },
    familyId: 'sdxl-illustrious',
    familyLabel: 'Illustrious',
    variant: 'refine',
    model: 'm.safetensors',
    modelLabel: 'M',
    prompt: 'freckles across the cheeks',
    negative: null,
    seed: 7,
    steps: 20,
    cfg: 5,
    sampler: 'euler',
    scheduler: 'normal',
    width: 1024,
    height: 1024,
    promptId: 'p2',
    durationMs: 1000,
    source: { name: 'sub/in.png', fromEntryId: 'orig' },
  } as HistoryEntry
  const original = {
    ...regionPassRecord,
    id: 'orig',
    no: 40,
    file: { filename: 'orig.png', subfolder: '', type: 'output' },
    variant: null,
    prompt: 'a lighthouse at dusk',
    source: undefined,
  } as HistoryEntry

  it('is the source record, with the pass\'s words, and never the pass itself', () => {
    const picture = regionPicture(regionPassRecord, { kind: 'record', entry: original })!
    expect(picture.id).toBe('orig')
    expect(picture.file).toEqual(original.file)
    expect(picture.prompt).toBe('freckles across the cheeks')
    expect(picture.variant).toBeNull()
  })

  it('is a record of the output file when only the file is left, claiming no maker', () => {
    const ref = { filename: 'orig.png', subfolder: 'day', type: 'output' }
    const picture = regionPicture(regionPassRecord, { kind: 'output', ref, fromEntryId: 'orig' })!
    expect(picture.id).toBe('orig')
    expect(picture.file).toEqual(ref)
    expect(picture.variant).toBeNull()
    expect(picture.source).toBeUndefined()
    expect([picture.model, picture.modelLabel, picture.familyId, picture.familyLabel]).toEqual(['', '', '', ''])
    expect([picture.width, picture.height]).toEqual([null, null])
    expect(picture.prompt).toBe('freckles across the cheeks')
  })

  it('reads ComfyUI\'s input copy by its folder and name', () => {
    const picture = regionPicture(regionPassRecord, { kind: 'input', name: 'sub/in.png' })!
    expect(picture.file).toEqual({ filename: 'in.png', subfolder: 'sub', type: 'input' })
    expect(picture.id).toBe('')
    expect(regionPicture(regionPassRecord, { kind: 'input', name: 'in.png' })!.file).toEqual({ filename: 'in.png', subfolder: '', type: 'input' })
  })

  it('is nothing when that picture is gone', () => {
    expect(regionPicture(regionPassRecord, { kind: 'gone' })).toBeNull()
  })
})
