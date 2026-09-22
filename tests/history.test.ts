import { beforeEach, describe, expect, it, vi } from 'vitest'
import type * as History from '../src/lib/history'

type H = typeof History
let h: H

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
  vi.resetModules()
  h = await import('../src/lib/history')
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
