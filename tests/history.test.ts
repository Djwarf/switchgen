import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as History from '../src/lib/history'

type H = typeof History
let h: H

// Enough of a browser for the archive: a localStorage, and a window whose
// storage listeners a test can call, standing in for another tab's write.
const KEY = 'switchgen.archive.v2'
const disk = new Map<string, string>()
const storageListeners: ((e: { key: string | null; newValue: string | null }) => void)[] = []
vi.stubGlobal('localStorage', {
  getItem: (k: string) => disk.get(k) ?? null,
  setItem: (k: string, v: string) => void disk.set(k, String(v)),
  removeItem: (k: string) => void disk.delete(k),
  key: (i: number) => [...disk.keys()][i] ?? null,
  get length() {
    return disk.size
  },
})
vi.stubGlobal('window', {
  addEventListener: (type: string, fn: (e: { key: string | null; newValue: string | null }) => void) => {
    if (type === 'storage') storageListeners.push(fn)
  },
  removeEventListener: () => {},
})
vi.stubGlobal('document', { addEventListener: () => {}, removeEventListener: () => {}, visibilityState: 'visible' })

/** What this tab last wrote. */
const stored = () => JSON.parse(disk.get(KEY) ?? 'null')
/** Another tab wrote this envelope. */
const arrive = (env: unknown) => {
  for (const fn of storageListeners) fn({ key: KEY, newValue: JSON.stringify(env) })
}
/** A record as the server or another tab holds it. */
const held = (id: string, extra: Record<string, unknown> = {}) => ({
  ...record(id.charCodeAt(0)),
  id,
  no: 1,
  at: 1_700_000_000_000 + id.charCodeAt(0),
  prompt: `p-${id}`,
  ...extra,
})

const record = (i: number): History.NewEntry => ({
  desk: 'images',
  kind: 'image',
  mode: 't2i',
  file: { filename: `f${i}.png`, subfolder: '', type: 'output' },
  familyId: 'x',
  familyLabel: 'X',
  variant: null,
  model: 'm',
  modelLabel: 'M',
  prompt: `p${i}`,
  negative: null,
  seed: i,
  steps: 1,
  cfg: 1,
  sampler: 's',
  scheduler: 's',
  width: 1,
  height: 1,
  promptId: `pid${i}`,
  durationMs: 0,
})

beforeEach(async () => {
  vi.useFakeTimers()
  disk.clear()
  storageListeners.length = 0
  vi.resetModules()
  h = await import('../src/lib/history')
})

afterEach(() => {
  // A save still due belongs to this test's module, not the next one's.
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('the archive store and the server', () => {
  it('reports local adds, edits and removals as deltas the sync can push', () => {
    const deltas: History.CommitDelta[] = []
    h.onCommit((d) => deltas.push(d))
    const a = h.add(record(1))
    expect(deltas.at(-1)).toMatchObject({ upserted: [expect.objectContaining({ id: a.id })], removed: [], remote: false })
    h.star(a.id)
    expect(deltas.at(-1)?.upserted[0]?.starred).toBe(true)
    h.remove(a.id)
    expect(deltas.at(-1)).toMatchObject({ upserted: [], removed: [a.id], remote: false })
  })

  it('knows which records the server has never stamped', () => {
    const a = h.add(record(1))
    const b = h.add(record(2))
    expect(h.unsynced().map((e) => e.id).sort()).toEqual([a.id, b.id].sort())
    h.applyServerMeta([{ id: a.id, no: a.no, rev: 1 }, { id: b.id, no: 99, rev: 2 }])
    expect(h.unsynced()).toHaveLength(0)
    expect(h.get(b.id)?.no).toBe(99)
  })

  it('marks server-originated changes as remote so they are not echoed back', () => {
    const deltas: History.CommitDelta[] = []
    h.onCommit((d) => deltas.push(d))
    const a = h.add(record(1))
    h.applyServerMeta([{ id: a.id, no: a.no, rev: 1 }])
    expect(deltas.at(-1)?.remote).toBe(true)
    h.mergeFromServer([{ ...h.get(a.id)!, prompt: 'edited elsewhere', rev: 4 }], [], 100)
    expect(deltas.at(-1)?.remote).toBe(true)
    expect(h.get(a.id)?.prompt).toBe('edited elsewhere')
  })

  it('honours a server removal only for a record the server had stamped', () => {
    const synced = h.add(record(1))
    h.applyServerMeta([{ id: synced.id, no: synced.no, rev: 1 }])
    const local = h.add(record(2))
    h.mergeFromServer([], [synced.id, local.id], 100)
    expect(h.get(synced.id)).toBeUndefined()
    expect(h.get(local.id)).toBeDefined()
  })

  it('follows the server for edition numbers', () => {
    h.mergeFromServer([], [], 500)
    expect(h.add(record(1)).no).toBeGreaterThanOrEqual(500)
  })

  it('finds tags with the tag: field, spaces or underscores', () => {
    const a = h.add(record(1))
    h.update(a.id, { tags: ['red_hair', 'smile'] })
    expect(h.matches(h.get(a.id)!, 'tag:red_hair')).toBe(true)
    expect(h.matches(h.get(a.id)!, 'tag:red hair')).toBe(true)
    expect(h.matches(h.get(a.id)!, 'tag:blue')).toBe(false)
    expect(h.matches(h.get(a.id)!, 'is:tagged')).toBe(true)
  })
})

describe('an edit made while its push was in flight', () => {
  it('keeps the edit and its mark, and takes only the number the server gave', () => {
    const a = h.add(record(1))
    const sent = h.unsynced()
    h.star(a.id)
    h.applyServerMeta([{ id: a.id, no: 77, rev: 1 }], sent)
    expect(h.get(a.id)?.starred).toBe(true)
    expect(h.get(a.id)?.no).toBe(77)
    expect(h.get(a.id)?.rev).toBeUndefined()
    expect(h.unsynced().map((e) => e.id)).toEqual([a.id])
  })

  it('keeps a second edit to a stamped record waiting after the first is stamped', () => {
    const a = h.add(record(1))
    h.applyServerMeta([{ id: a.id, no: a.no, rev: 1 }], h.unsynced())
    h.star(a.id)
    const sent = h.unsynced()
    h.update(a.id, { note: 'written during the push' })
    h.applyServerMeta([{ id: a.id, no: a.no, rev: 2 }], sent)
    expect(h.get(a.id)?.note).toBe('written during the push')
    expect(h.get(a.id)?.pending).toBe(true)
    expect(h.unsynced().map((e) => e.id)).toEqual([a.id])
  })
})

describe('undo and refusals', () => {
  it('puts a removed record back unstamped, so the server takes it as a record arriving', () => {
    const a = h.add(record(1))
    h.applyServerMeta([{ id: a.id, no: a.no, rev: 3 }], h.unsynced())
    const removed = h.remove(a.id)!
    expect(removed.rev).toBe(3)
    h.restore(removed)
    expect(h.get(a.id)).toBeDefined()
    expect(h.get(a.id)?.rev).toBeUndefined()
    expect(h.unsynced().map((e) => e.id)).toEqual([a.id])
  })

  it('drops the records the server refused, without sending anything', () => {
    const deltas: History.CommitDelta[] = []
    h.onCommit((d) => deltas.push(d))
    const a = h.add(record(1))
    const b = h.add(record(2))
    h.forget([a.id])
    expect(h.get(a.id)).toBeUndefined()
    expect(h.get(b.id)).toBeDefined()
    expect(deltas.at(-1)).toMatchObject({ removed: [a.id], remote: true })
  })
})

describe('another tab\'s write', () => {
  const settled = () => vi.advanceTimersByTime(400)

  it('does not lose a record filed here since this tab last wrote', () => {
    const older = { v: 2, nextNo: 1, entries: [] }
    const r = h.add(record(1))
    arrive(older)
    expect(h.get(r.id)).toBeDefined()
    expect(h.unsynced().map((e) => e.id)).toContain(r.id)
    settled()
    expect(stored().entries.map((e: { id: string }) => e.id)).toContain(r.id)
  })

  it('does not lose a star made here, or its mark', () => {
    h.mergeFromServer([held('p0', { rev: 5 })], [], 2, { rev: 5, epoch: 'E1', boot: 'B1' })
    h.flush()
    const older = stored()
    h.star('p0')
    arrive(older)
    expect(h.get('p0')?.starred).toBe(true)
    expect(h.get('p0')?.pending).toBe(true)
  })

  it('is the word on a record this tab had already written: a removal there stands', () => {
    const r = h.add(record(1))
    settled()
    const mine = stored()
    arrive({ ...mine, entries: mine.entries.filter((e: { id: string }) => e.id !== r.id), gone: [r.id] })
    expect(h.get(r.id)).toBeUndefined()
    expect(h.gone()).toContain(r.id)
  })

  it('never hands out an edition number twice on an older count', () => {
    h.add(record(1))
    const before = h.add(record(2)).no
    arrive({ ...stored(), v: 2, entries: [], nextNo: 1 })
    expect(h.add(record(3)).no).toBeGreaterThan(before)
  })

  it('takes its removals as they stand, so one it saw settled is not sent again', () => {
    h.mergeFromServer([held('X', { rev: 5 }), held('Y', { rev: 6, no: 2 })], [], 3, { rev: 6, epoch: 'E1', boot: 'B1' })
    h.flush()
    const base = stored()
    // The other tab removed X and wrote before the server acknowledged it.
    arrive({ ...base, entries: base.entries.filter((e: { id: string }) => e.id !== 'X'), gone: ['X'] })
    expect(h.gone()).toEqual(['X'])
    expect(h.get('X')).toBeUndefined()
    // It then had the removal acknowledged, undid it, and the server stamped X again.
    arrive({ ...base, entries: [...base.entries.filter((e: { id: string }) => e.id !== 'X'), held('X', { rev: 8 })], gone: undefined, sync: { rev: 8, epoch: 'E1', boot: 'B1' } })
    expect(h.gone()).toEqual([])
    expect(h.get('X')?.rev).toBe(8)
    h.star('Y')
    settled()
    expect(stored().gone).toBeUndefined()
  })

  it('does not bring back a record removed here before this tab wrote, and not a settled removal either', () => {
    h.mergeFromServer([held('Y', { rev: 6 })], [], 3, { rev: 6, epoch: 'E1', boot: 'B1' })
    h.flush()
    const now = stored()
    h.remove('Y')
    arrive(now)
    expect(h.get('Y')).toBeUndefined()
    expect(h.gone()).toContain('Y')
    h.acknowledgeRemoved(['Y'])
    arrive({ ...now, entries: [], gone: ['Y'] })
    expect(h.gone()).not.toContain('Y')
  })
})

describe('a server that lost part of its log', () => {
  it('sends back everything a new log lacks, and takes the new log\'s copy even at an equal rev', () => {
    h.mergeFromServer([held('a', { rev: 3 }), held('b', { rev: 4, no: 2 }), held('c', { rev: 5, no: 3 })], [], 4, { rev: 5, epoch: 'E1', boot: 'B1' })
    expect(h.unsynced()).toEqual([])
    h.mergeFromServer([held('b', { rev: 4, prompt: 'from the other browser' })], [], 2, { rev: 4, epoch: 'E2', boot: 'B2' }, { lostAbove: 0 })
    expect(h.unsynced().map((e) => e.id).sort()).toEqual(['a', 'c'])
    expect(h.get('a')?.rev).toBeUndefined()
    expect(h.get('c')?.rev).toBeUndefined()
    expect(h.get('b')?.prompt).toBe('from the other browser')
    expect(h.syncCursor()).toEqual({ rev: 4, epoch: 'E2', boot: 'B2' })
  })

  it('sends back only what was stamped in the lost tail and not written or removed since', () => {
    h.mergeFromServer(
      [held('u', { rev: 90 }), held('v', { rev: 105 }), held('w', { rev: 101 }), held('x', { rev: 104 }), held('y', { rev: 103 }), held('z', { rev: 102 })],
      [],
      200,
      { rev: 105, epoch: 'E1', boot: 'B1' },
    )
    h.mergeFromServer(
      [held('u', { rev: 90 }), held('y', { rev: 50, prompt: 'older' }), held('z', { rev: 104, prompt: 'written after the restart' })],
      ['w', 'v'],
      200,
      { rev: 106, epoch: 'E1', boot: 'B2' },
      { lostAbove: 100, removedRev: { w: 106, v: 90 } },
    )
    expect(h.unsynced().map((e) => e.id).sort()).toEqual(['v', 'x', 'y'])
    // y's lost edit is newer than the copy the server kept.
    expect(h.get('y')?.prompt).toBe('p-y')
    // z was written again after the restart, at a rev the lost load had used.
    expect(h.get('z')?.prompt).toBe('written after the restart')
    // w was removed after the restart; v's removal is older than its lost re-add.
    expect(h.get('w')).toBeUndefined()
    expect(h.get('v')).toBeDefined()
    expect(h.get('u')?.rev).toBe(90)
    expect(h.highestRev()).toBe(106)
  })

  it('saves the log it knows even before the log has a record', () => {
    h.mergeFromServer([], [], 1, { rev: 0, epoch: 'E1' })
    h.flush()
    expect(stored()?.sync?.epoch).toBe('E1')
  })
})
