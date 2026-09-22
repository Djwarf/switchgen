import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { HistoryEntry } from '../src/lib/history'
import type * as Loras from '../src/lib/loras'
import type * as Session from '../src/lib/session'

// Reuse of a record into its desk, with a localStorage for the desks and the
// add-on racks to be saved in.
const disk = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (k: string) => disk.get(k) ?? null,
  setItem: (k: string, v: string) => void disk.set(k, String(v)),
  removeItem: (k: string) => void disk.delete(k),
  key: (i: number) => [...disk.keys()][i] ?? null,
  get length() {
    return disk.size
  },
})

let session: typeof Session
let loras: typeof Loras
beforeEach(async () => {
  disk.clear()
  vi.resetModules()
  session = await import('../src/lib/session')
  loras = await import('../src/lib/loras')
})

const record = (extra: Partial<HistoryEntry>): HistoryEntry => ({
  id: 'r1',
  no: 40,
  at: 1_700_000_000_000,
  desk: 'images',
  kind: 'image',
  mode: 't2i',
  file: { filename: 'r1.png', subfolder: '', type: 'output' },
  familyId: 'sdxl-illustrious',
  familyLabel: 'Illustrious',
  variant: null,
  model: 'm.safetensors',
  modelLabel: 'M',
  prompt: 'a lighthouse at dusk',
  negative: null,
  seed: 7,
  steps: 20,
  cfg: 5,
  sampler: 'euler',
  scheduler: 'normal',
  width: 1024,
  height: 1024,
  promptId: 'p1',
  durationMs: 1000,
  ...extra,
})

describe('reusing a region pass', () => {
  it('loads its words and settings, never the whole picture as image to image, and says where it came from', () => {
    const r = session.compositionFromEntry(
      record({ mode: 'i2i', variant: 'refine', source: { name: 'orig.png' }, denoise: 0.45 }),
    )
    expect(r.composition.mode).toBe('t2i')
    expect(r.composition.source).toBeNull()
    expect(r.composition.denoise).toBeNull()
    expect(r.composition.touched).not.toContain('denoise')
    expect(r.composition.prompt).toBe('a lighthouse at dusk')
    expect(r.composition.seed).toBe(7)
    expect(r.notes.map((n) => n.field)).toContain('region')
  })
})

describe('reusing a record with add-ons', () => {
  it('names a picture\'s add-ons as the library does, and sends the reader to the Pictures desk', () => {
    const r = session.compositionFromEntry(record({ loras: [{ name: 'add-detail-xl.safetensors', strength: 0.8 }] }))
    const note = r.notes.find((n) => n.field === 'loras')!.reason
    expect(note).toContain('Add detail xl at 0.8')
    expect(note).toContain('Pictures desk')
    expect(note).not.toMatch(/\brack\b/)
  })

  it('puts a clip\'s add-ons back on its family\'s rack, and undoes the rack with the desk', () => {
    const saved = [{ file: 'old.safetensors', strength: 0.5, enabled: true }]
    loras.saveStack('wan22-5b', saved)
    const before = session.deskStore('video').get()
    const clip = record({
      desk: 'video',
      kind: 'video',
      mode: 't2v',
      familyId: 'wan22-5b',
      familyLabel: 'Wan 2.2 5B',
      model: 'wan2.2_ti2v_5B_fp16.safetensors',
      loras: [{ name: 'motion.safetensors', strength: 0.7, half: 'both' }],
    })
    const r = session.reuseIntoDesk(clip)
    expect(loras.loadStack('wan22-5b')).toEqual([{ file: 'motion.safetensors', strength: 0.7, enabled: true }])
    expect(r.notes.some((n) => n.reason.includes('Not carried over'))).toBe(false)
    expect(r.notes.find((n) => n.field === 'loras')?.reason).toBe('The add-on rack now holds what this clip used.')
    expect(session.deskStore('video').get().prompt).toBe('a lighthouse at dusk')

    r.undo()
    expect(loras.loadStack('wan22-5b')).toEqual(saved)
    expect(session.deskStore('video').get()).toEqual(before)
  })
})
