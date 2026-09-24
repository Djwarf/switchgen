import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as History from '../src/lib/history'

type H = typeof History
let h: H

// Enough of a browser for the archive: a localStorage, and a window whose
// storage listeners a test can call, standing in for another tab's write.
const KEY = 'switchgen.archive.v2'
const disk = new Map<string, string>()
const storageListeners: ((e: { key: string | null; newValue: string | null }) => void)[] = []
/** The longest archive this browser has room for; longer, and it says it is full, as a phone's does. */
let room = Infinity
vi.stubGlobal('localStorage', {
  getItem: (k: string) => disk.get(k) ?? null,
  setItem: (k: string, v: string) => {
    if (k === KEY && String(v).length > room) throw Object.assign(new Error('full'), { name: 'QuotaExceededError' })
    disk.set(k, String(v))
  },
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
  room = Infinity
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

describe('a desk\'s record that took the place of one filed after the fact', () => {
  // The server folds a record the recovery pass filed on another device into
  // the desk's record for the same file, and says what it kept from it. This
  // browser is stamped at the rev the folded record was stored at, and a pull
  // passes over a rev it already holds, so what was kept is added here.
  const kept = { starred: true, note: 'n', tags: ['t'], rating: 'general' } as const

  it('takes what the server kept, stamped, with nothing left to send', () => {
    const a = h.add(record(1))
    const sent = h.unsynced()
    h.applyServerMeta([{ id: a.id, no: 4, rev: 7, kept: { ...kept, tags: [...kept.tags] } }], sent)
    expect(h.get(a.id)).toMatchObject({ starred: true, note: 'n', tags: ['t'], rating: 'general', rev: 7, no: 4 })
    expect(h.unsynced()).toEqual([])
    // A pull at that rev has nothing to add.
    expect(h.mergeFromServer([{ ...h.get(a.id)! }], [], 5)).toBe(0)
  })

  it('adds it to a copy edited while the push was out, which is sent whole next', () => {
    const a = h.add(record(1))
    const sent = h.unsynced()
    h.update(a.id, { tags: ['mine'] })
    h.applyServerMeta([{ id: a.id, no: a.no, rev: 7, kept: { note: 'n', tags: ['t'] } }], sent)
    expect(h.get(a.id)?.note).toBe('n')
    expect(h.get(a.id)?.tags).toEqual(['mine'])
    // The next push replaces the server's copy whole, so it carries the note.
    const next = h.unsynced().find((e) => e.id === a.id)
    expect(next?.note).toBe('n')
    expect(next?.tags).toEqual(['mine'])
  })

  it('keeps a field the copy has over the one the server kept', () => {
    const a = h.add(record(1))
    h.update(a.id, { note: 'own' })
    const sent = h.unsynced()
    h.applyServerMeta([{ id: a.id, no: a.no, rev: 7, kept: { note: 'n' } }], sent)
    expect(h.get(a.id)?.note).toBe('own')
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

describe('one job filed by two writers', () => {
  // Two tabs following one shot, a page taking up a job the page before it
  // left, or the recovery pass finding a file its desk is about to file.
  it('files it once: the second call gets the first record back', () => {
    const a = h.add(record(1))
    const b = h.add(record(1))
    expect(b.id).toBe(a.id)
    expect(h.all()).toHaveLength(1)
  })

  it('files a new job under a reused name as a record of its own', () => {
    // ComfyUI hands a deleted file's name to the next render.
    h.add(record(1))
    h.add({ ...record(1), promptId: 'another-job' })
    expect(h.all()).toHaveLength(2)
  })

  it('gives the desk\'s account to a record the recovery pass filed first, keeping its identity and the reader\'s star', () => {
    const [first] = h.addMany([{ ...record(2), recovered: true, promptId: '' }])
    h.star(first!.id)
    h.applyServerMeta([{ id: first!.id, no: first!.no, rev: 9 }])
    const desk = h.add(record(2))
    expect(desk.id).toBe(first!.id)
    expect(desk.no).toBe(first!.no)
    const now = h.get(first!.id)!
    expect(now.recovered).toBeUndefined()
    expect(now.starred).toBe(true)
    expect(now.rev).toBe(9)
    // The server has the minimal copy, so the desk's has to be sent.
    expect(now.pending).toBe(true)
    expect(now.prompt).toBe('p2')
    expect(now.promptId).toBe('pid2')
    expect(h.all()).toHaveLength(1)
  })

  it('files nothing twice from one recovery batch, and returns only what it made', () => {
    h.add(record(3))
    const made = h.addMany([
      { ...record(3), recovered: true },
      { ...record(4), recovered: true },
      { ...record(4), recovered: true },
    ])
    expect(made.map((e) => e.file.filename)).toEqual(['f4.png'])
    expect(h.all()).toHaveLength(2)
  })
})

describe('a record filed again on request', () => {
  it('keeps its refiled mark through a save and a fresh load, and nothing that is not a plain true', async () => {
    const a = h.add({ ...record(5), recovered: true, refiled: true })
    const b = h.add({ ...record(6), recovered: true })
    h.flush()
    const env = stored()
    expect(env.entries.find((e: { id: string }) => e.id === a.id)?.refiled).toBe(true)
    // The server takes a removed file back on this mark alone, so only a
    // true sent on purpose may carry it.
    env.entries.find((e: { id: string }) => e.id === b.id).refiled = 'yes'
    disk.set(KEY, JSON.stringify(env))
    vi.resetModules()
    h = await import('../src/lib/history')
    expect(h.get(a.id)?.refiled).toBe(true)
    expect(h.get(b.id)?.refiled).toBeUndefined()
  })
})

describe('the files deleting a record takes with it', () => {
  const out = (filename: string, type = 'output') => ({ filename, subfolder: 'video', type })
  const entry = (id: string, file: ReturnType<typeof out>, files?: ReturnType<typeof out>[]) =>
    ({ ...record(1), id, file, files }) as History.HistoryEntry
  const clip = out('wan_00001_.webm')
  const frame = out('wan_00001_.webm.frame.png')
  const preview = out('ComfyUI_temp_0001.png', 'temp')
  const shared = out('shared.png')
  const other = entry('other', shared)
  const shot = entry('shot', clip, [clip, frame, preview, shared])

  it('is every output its run wrote, its own file first, and not what another record names', () => {
    expect(h.filesToDelete(shot, { others: [other] })).toEqual({ go: [clip, frame], kept: [] })
  })

  it('keeps what it is asked to keep', () => {
    const keep = (rel: string) => rel === 'video/wan_00001_.webm.frame.png'
    expect(h.filesToDelete(shot, { others: [other], keep })).toEqual({ go: [clip], kept: [frame] })
  })

  it('is the record\'s own file when it names no others', () => {
    expect(h.filesToDelete(entry('lone', clip), { others: [] })).toEqual({ go: [clip], kept: [] })
  })
})

describe('deleting a record\'s files', () => {
  const clip = { filename: 'c.webm', subfolder: 'reel', type: 'output' }
  const frame = { filename: 'c.webm.frame.png', subfolder: 'reel', type: 'output' }
  const third = { filename: 'c2.png', subfolder: 'reel', type: 'output' }
  let asked: string[]
  const realFetch = globalThis.fetch
  const answering = (byRel: Record<string, [number, unknown]>) => {
    asked = []
    globalThis.fetch = (async (_url: string, init: RequestInit) => {
      const rel = JSON.parse(String(init.body)).rel as string
      asked.push(rel)
      const [status, body] = byRel[rel]!
      return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
    }) as typeof fetch
  }
  afterEach(() => {
    globalThis.fetch = realFetch
  })

  it('deletes every file the run wrote, reports one that would not go, and removes the record', async () => {
    const e = h.add({ ...record(1), file: clip, files: [clip, frame, third] })
    answering({ 'reel/c.webm': [200, { freed: 10 }], 'reel/c.webm.frame.png': [500, { error: 'EACCES' }], 'reel/c2.png': [404, { error: 'not found' }] })
    const [{ result }] = await h.deleteFiles([e])
    expect(asked).toEqual(['reel/c.webm', 'reel/c.webm.frame.png', 'reel/c2.png'])
    expect(result).toEqual({ ok: true, freed: 10, gone: 2, left: ['reel/c.webm.frame.png'], kept: [] })
    expect(h.get(e.id)).toBeUndefined()
  })

  it('touches nothing else, and keeps the record, when its own file will not go', async () => {
    const e = h.add({ ...record(1), file: clip, files: [clip, frame] })
    answering({ 'reel/c.webm': [400, { error: 'not a regular file' }] })
    const [{ result }] = await h.deleteFiles([e])
    expect(result).toEqual({ ok: false, reason: 'not a regular file' })
    expect(asked).toEqual(['reel/c.webm'])
    expect(h.get(e.id)).toBeDefined()
  })
})

describe('the words for what was deleted', () => {
  it('counts one file with its record', async () => {
    const { deletedText } = await import('../src/components/archive/deletion')
    expect(deletedText([{ gone: 1, left: [], kept: [] }])).toBe('One file has gone from the outputs folder, and its record with it.')
  })

  it('counts the files, not the records', async () => {
    const { deletedText } = await import('../src/components/archive/deletion')
    expect(deletedText([{ gone: 2, left: [], kept: [] }])).toBe('2 files have gone from the outputs folder, and their record with them.')
    expect(deletedText([{ gone: 1, left: [], kept: [] }, { gone: 2, left: [], kept: [] }])).toBe(
      '3 files have gone from the outputs folder, and their 2 records with them.',
    )
  })

  it('names a file that would not go, and a last frame kept for the reel', async () => {
    const { deletedText } = await import('../src/components/archive/deletion')
    const said = deletedText([{ gone: 1, left: ['reel/a.png'], kept: ['reel/c.webm.frame.png'] }])
    expect(said).toContain('One more, reel/a.png, could not be deleted and is still on disk.')
    expect(said).toContain('reel/c.webm.frame.png stays on disk, because the reel still opens a shot on it.')
    const many = deletedText([{ gone: 1, left: ['a', 'b'], kept: ['c', 'd'] }])
    expect(many).toContain('2 more could not be deleted and are still on disk: a, b.')
    expect(many).toContain('2 last frames stay on disk, because the reel still opens shots on them.')
  })
})

describe('the search index', () => {
  it('reads again only the record that changed, not the whole archive', () => {
    h.addMany(Array.from({ length: 2000 }, (_, i) => ({ ...record(i), prompt: i % 2 ? `rain on glass ${i}` : `sun ${i}` })))
    const first = h.search('rain')
    expect(first).toHaveLength(1000)
    h.star(h.all()[5]!.id)
    const lower = vi.spyOn(String.prototype, 'toLowerCase')
    try {
      expect(h.search('rain')).toHaveLength(1000)
      // The starred record's own haystack, and the words asked for.
      expect(lower.mock.calls.length).toBeLessThan(10)
    } finally {
      lower.mockRestore()
    }
  })

  it('finds a record by what was changed on it', () => {
    const a = h.add(record(1))
    const b = h.add(record(2))
    expect(h.search('umbrella')).toEqual([])
    h.update(b.id, { note: 'umbrella' })
    expect(h.search('umbrella').map((e) => e.id)).toEqual([b.id])
    expect(h.search('umbrella').map((e) => e.id)).not.toContain(a.id)
  })
})

describe('a browser that runs out of room', () => {
  // A phone's browser keeps far less than a desktop's. What it sheds decides
  // whether a record made here, and not yet on the server, survives a reload.
  const padded = (i: number) => ({ ...record(i), prompt: `a long prompt, written out in full ${'x'.repeat(200)} ${i}` })
  const ids = (list: readonly { id: string }[]) => list.map((e) => e.id)

  /**
   * 600 records, the newest 500 held by the server as well; the oldest 100
   * were never sent (made while the server could not be reached, say), and
   * the very oldest is starred.
   */
  function filled(serverBacked: boolean) {
    h.addMany(Array.from({ length: 600 }, (_, i) => padded(i)))
    if (serverBacked) {
      h.setServerBacked(true)
      const sent = h.all().slice(0, 500)
      h.applyServerMeta(sent.map((e, i) => ({ id: e.id, no: e.no, rev: i + 1 })), sent)
    }
    const starred = h.all().at(-1)!
    h.star(starred.id)
    h.flush()
    // Room for about four fifths of what is saved now.
    room = Math.floor(disk.get(KEY)!.length * 0.8)
    return starred
  }

  it('with the server behind it, sheds only what the server holds, keeps every record on screen, and says so', () => {
    const starred = filled(true)
    const deltas: History.CommitDelta[] = []
    h.onCommit((d) => deltas.push(d))
    h.add(padded(1000))
    h.flush()

    const saved = stored().entries as History.HistoryEntry[]
    const savedIds = new Set(ids(saved))
    const onlyHere = h.all().filter((e) => e.rev === undefined || e.pending)
    expect(onlyHere.length).toBeGreaterThan(100)
    for (const e of onlyHere) expect(savedIds.has(e.id)).toBe(true)
    expect(savedIds.has(starred.id)).toBe(true)
    expect(h.all()).toHaveLength(601)
    expect(deltas.every((d) => d.removed.length === 0)).toBe(true)
    expect(h.quotaIssue()).toContain('Nothing was removed')
    expect(saved.length).toBe(401)

    // Later saves keep to the room found, rather than failing on the whole
    // list, and still keep every record the server does not have.
    h.add(padded(1001))
    h.flush()
    expect(stored().entries).toHaveLength(401)
    expect(h.all()).toHaveLength(602)
    const later = new Set(ids(stored().entries))
    for (const e of h.all()) if (e.rev === undefined || e.pending) expect(later.has(e.id)).toBe(true)
  })

  it('on its own, removes the oldest unstarred records and says how many', () => {
    const starred = filled(false)
    h.add(padded(1000))
    h.flush()
    expect(h.all()).toHaveLength(401)
    expect(ids(h.all())).toContain(starred.id)
    expect(h.quotaIssue()).toContain('removed the 200 oldest')
  })

  it('says it could not save when even the shorter copy does not fit', () => {
    filled(true)
    room = 0
    h.add(padded(1000))
    h.flush()
    expect(h.quotaIssue()).toContain('could not save')
    expect(h.quotaIssue()).toContain('Nothing was removed')
  })

  it('says so on its own too', () => {
    filled(false)
    room = 0
    h.add(padded(1000))
    h.flush()
    expect(h.quotaIssue()).toContain('could not save')
  })
})
