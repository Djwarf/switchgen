/**
 * Run parts (TEST PLAN, C, besides the driver): the ledger and its state, the
 * lock, cold and warm timing, the picture reader and its quarantine, the
 * reference photos, and the one client that knows the app's routes. Folders
 * are temp folders; the reader and the client talk to stand-ins on port 0.
 */
import fs from 'node:fs'
import http from 'node:http'
import type { AddressInfo } from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { appendDoneCell, appendLedger, readDoneCells, readLedger, runState } from '../run/ledger.ts'
import { LockHeld, acquireDriverLock, acquireLock, liveDrivers, machineBoot, readLock } from '../run/lock.ts'
import { coldFlag, previousRan } from '../run/timing.ts'
import { QUARANTINE_TAGS, readCells, readReadings } from '../run/reader.ts'
import { addRef, ensureRefCopies, listRefs, refIdFrom, refIndex, refProblems, refUpdatedAt, renameRef, setDescribe, setMask } from '../run/refs.ts'
import { AppUnreachable, TAG_MAX, runnerClient } from '../run/runnerClient.ts'
import { sizeOf } from '../run/imagesize.ts'
import { jpegHeader, png, removeTemp, sidewaysCatStandIn, tempEnv, webpHeader } from './helpers.ts'
import { standInRunner } from './standInRunner.ts'

afterAll(removeTemp)

const ended = (job: string, cell: string, status: string, code: string | null = null, extra: Record<string, unknown> = {}) => ({
  t: 'ended' as const, at: 1, job, cell, status, error: code ? { code, message: 'x', node: null, nodeType: null } : null,
  files: status === 'done' ? [{ filename: `${cell}_00001_.png`, subfolder: '.lab/cells', type: 'output' }] : [],
  primary: status === 'done' ? { filename: `${cell}_00001_.png`, subfolder: '.lab/cells', type: 'output' } : null,
  promptId: null, durationMs: 1000, ranAt: 1, finishedAt: 2, cached: false, cold: false, attempt: 1, ...extra,
})

describe('ledger', () => {
  it('reads past a torn line, and runState sorts every ending', () => {
    const env = tempEnv()
    const dir = path.join(env.labDir, 'runs', 'r1')
    appendLedger(dir, { t: 'submitting', at: 1, group: 'g1', jobs: ['a', 'b', 'c', 'd', 'e', 'f'].map((cell, i) => ({ job: `j${i + 1}`, cell })) })
    fs.appendFileSync(path.join(dir, 'ledger.jsonl'), '{"t":"subm')
    appendLedger(dir, { t: 'submitted', at: 2, group: 'g1', replayed: false })
    expect(readLedger(dir).map((e) => e.t)).toEqual(['submitting', 'submitted'])
    appendLedger(dir, ended('j1', 'a', 'done'))
    appendLedger(dir, ended('j2', 'b', 'failed', 'refused'))
    appendLedger(dir, ended('j3', 'c', 'failed', 'failed'))
    appendLedger(dir, ended('j4', 'd', 'lost', 'lost'))
    appendLedger(dir, ended('j5', 'e', 'skipped', 'skipped'))
    appendLedger(dir, ended('j6', 'f', 'failed', 'no-file'))
    const plan = { order: ['a', 'b', 'c', 'd', 'e', 'f', 'g'] }
    let s = runState(readLedger(dir), plan)
    expect([...s.done.keys()]).toEqual(['a'])
    expect([...s.failed.keys()].sort()).toEqual(['b', 'f'])
    expect([...s.pending].sort()).toEqual(['c', 'd', 'e', 'g'])
    expect(s.attempts.get('c')).toBe(1)
    expect(s.attempts.get('e')).toBeUndefined()
    expect(s.inFlight.size).toBe(0)
    // Retried once: a second failure is final; a stop goes back without counting.
    appendLedger(dir, { t: 'submitting', at: 3, group: 'g2', jobs: [{ job: 'k3', cell: 'c' }, { job: 'k4', cell: 'd' }, { job: 'k7', cell: 'g' }] })
    s = runState(readLedger(dir), plan)
    expect([...s.unanswered.keys()]).toEqual(['g2'])
    expect(s.inFlight.size).toBe(3)
    appendLedger(dir, { t: 'submitted', at: 4, group: 'g2', replayed: true })
    appendLedger(dir, ended('k3', 'c', 'failed', 'failed'))
    appendLedger(dir, ended('k4', 'd', 'done'))
    appendLedger(dir, ended('k7', 'g', 'stopped', 'stopped'))
    appendLedger(dir, { t: 'paused', at: 5, why: 'user' })
    s = runState(readLedger(dir), plan)
    expect(s.failed.get('c')?.attempts).toBe(2)
    expect(s.done.has('d')).toBe(true)
    expect(s.pending.has('g')).toBe(true)
    expect(s.attempts.get('g')).toBeUndefined()
    expect(s.paused).toBe(true)
    // A refused group gives its cells back, and a new submission ends the pause.
    appendLedger(dir, { t: 'submitting', at: 6, group: 'g3', jobs: [{ job: 'm7', cell: 'g' }] })
    appendLedger(dir, { t: 'refused', at: 7, group: 'g3', status: 503, error: 'busy' })
    s = runState(readLedger(dir), plan)
    expect(s.pending.has('g')).toBe(true)
    expect(s.paused).toBe(false)
    appendLedger(dir, { t: 'recovered', at: 8, cell: 'g', rel: '.lab/cells/g_00001_.png' })
    appendLedger(dir, { t: 'removed', at: 9, cell: 'a', why: 'quarantine' })
    s = runState(readLedger(dir), plan)
    expect(s.done.get('g')?.how).toBe('recovered')
    expect(s.removed.has('a')).toBe(true)
    expect(s.done.has('a')).toBe(false)
  })

  it('reuses cells another run made, and keeps a removal even when the cell is written again', () => {
    const env = tempEnv()
    appendDoneCell(env, { cellId: 'x', rel: '.lab/cells/x_00001_.png', durationMs: 5, cold: false, cached: false, finishedAt: 1, run: 'cal-1' })
    appendDoneCell(env, { cellId: 'y', rel: '.lab/cells/y_00001_.png', durationMs: 5, cold: false, cached: false, finishedAt: 1, run: 'cal-1' })
    appendDoneCell(env, { cellId: 'y', rel: '', durationMs: 0, cold: false, cached: false, finishedAt: null, run: 'cal-1', removed: true })
    appendDoneCell(env, { cellId: 'y', rel: '.lab/cells/y_00002_.png', durationMs: 5, cold: false, cached: false, finishedAt: 1, run: 'core-1' })
    const g = readDoneCells(env)
    expect(g.get('x')?.rel).toBe('.lab/cells/x_00001_.png')
    expect(g.get('y')?.removed).toBe(true)
    const s = runState([], { order: ['x', 'y', 'z'] }, g)
    expect(s.done.get('x')?.how).toBe('reused')
    expect(s.removed.has('y')).toBe(true)
    expect([...s.pending]).toEqual(['z'])
  })
})

describe('lock', () => {
  it('has one holder, refuses another run while one is live, and takes over a dead pid or an old boot', () => {
    const env = tempEnv()
    const a = acquireDriverLock(env, 'core-1')
    expect(() => acquireDriverLock(env, 'core-1')).toThrow(LockHeld)
    expect(() => acquireDriverLock(env, 'exta-1')).toThrow(/core-1 run is being made/)
    expect(liveDrivers(env)).toHaveLength(1)
    a.release()
    expect(liveDrivers(env)).toHaveLength(0)
    const f = path.join(env.labDir, 'runs', 'core-1', 'driver.lock')
    fs.mkdirSync(path.dirname(f), { recursive: true })
    fs.writeFileSync(f, JSON.stringify({ pid: 2 ** 22 + 12345, boot: machineBoot(), host: os.hostname(), run: 'core-1', at: 1 }))
    const b = acquireLock(f, 'core-1')
    expect(readLock(f)?.pid).toBe(process.pid)
    b.release()
    // pid 1 is alive, but the lock was written before the machine last started.
    fs.writeFileSync(f, JSON.stringify({ pid: 1, boot: 'an-old-boot', host: os.hostname(), run: 'core-1', at: 1 }))
    const c = acquireLock(f, 'core-1')
    expect(readLock(f)?.pid).toBe(process.pid)
    c.release()
    // pid 1 on this boot is a live holder.
    fs.writeFileSync(f, JSON.stringify({ pid: 1, boot: machineBoot(), host: os.hostname(), run: 'core-1', at: 1 }))
    expect(() => acquireLock(f, 'core-1')).toThrow(LockHeld)
  })
})

describe('coldFlag', () => {
  it('is cold after a model change, a foreign desk\'s job, or the same file with another operation; a cached job is passed over', () => {
    const facts: Record<string, { file: string; op: string }> = { A: { file: 'm1', op: 't2i' }, B: { file: 'm1', op: 't2i' }, C: { file: 'm2', op: 't2i' }, D: { file: 'm1', op: 'i2i' } }
    const cellOf = (c: string) => facts[c]
    const lab = (id: string, cell: string, ranAt: number, fin: number, cached = false) => ({ id, desk: 'lab', ranAt, finishedAt: fin, primary: { cached }, meta: { lab: { run: 'r', cell } } })
    const jobs = [lab('1', 'A', 0, 10), lab('2', 'B', 11, 20), { id: 'u', desk: 'images', ranAt: 21, finishedAt: 30, primary: {}, meta: null }, lab('3', 'B', 31, 40), lab('4', 'C', 41, 50), lab('5', 'A', 51, 52, true), lab('6', 'C', 53, 60), lab('7', 'D', 61, 70), lab('8', 'A', 71, 80)]
    const cold = (id: string) => {
      const j = jobs.find((x) => x.id === id)!
      return coldFlag(previousRan(jobs as never, j as never), j as never, cellOf as never)
    }
    expect(cold('1')).toBe(true)
    expect(cold('2')).toBe(false)
    expect(cold('3')).toBe(true)
    expect(cold('4')).toBe(true)
    expect(cold('6')).toBe(false)
    expect(cold('7')).toBe(true)
    // The same file right after, with another operation: its graph loads again.
    expect(cold('8')).toBe(true)
  })
})

describe('the picture reader', () => {
  it('reads in calls of 24 or fewer, backs off on memory, stores by cell, and quarantines', async () => {
    const env = tempEnv()
    const ids: string[] = []
    fs.mkdirSync(path.join(env.outputs, '.lab', 'cells'), { recursive: true })
    for (let i = 0; i < 50; i++) {
      const id = i.toString(16).padStart(16, '0')
      ids.push(id)
      fs.writeFileSync(path.join(env.outputs, '.lab', 'cells', `${id}_00001_.png`), png(8, 8))
      appendDoneCell(env, { cellId: id, rel: `.lab/cells/${id}_00001_.png`, durationMs: 1, cold: false, cached: false, finishedAt: 1, run: 'r1' })
    }
    const calls: number[] = []
    let busy = 2
    const slept: number[] = []
    const client = {
      async tag(rels: readonly string[]) {
        calls.push(rels.length)
        if (busy-- > 0) return { busy: 'memory' as const, message: 'short' }
        return rels.map((rel, index) => {
          const n = parseInt(rel.slice(11, 27), 16)
          if (n === 3) return { index, rel, rating: 'questionable' as const, ratings: [], general: [{ tag: 'child', confidence: 0.6 }], character: [] }
          if (n === 4) return { index, rel, rating: 'general' as const, ratings: [], general: [{ tag: 'child', confidence: 0.6 }], character: [] }
          if (n === 5) return { index, rel, rating: 'explicit' as const, ratings: [], general: [{ tag: 'school uniform', confidence: 0.6 }], character: [] }
          if (n === 6) return { index, rel, error: 'broken' }
          return { index, rel, rating: 'general' as const, ratings: [{ tag: 'general', confidence: 0.9 }], general: [{ tag: 'outdoors', confidence: 0.5 }], character: [] }
        })
      },
    }
    const r = await readCells(env, client as never, ids, { run: 'r1', sleep: async (ms: number) => void slept.push(ms) })
    expect(calls.every((n) => n <= TAG_MAX && n <= 24)).toBe(true)
    expect(slept).toEqual([30_000, 60_000])
    expect(r.quarantined.sort()).toEqual([ids[3], ids[5]].sort())
    expect(r.pending).toEqual([ids[6]])
    expect(r.read).toHaveLength(49)
    expect(fs.existsSync(path.join(env.outputs, '.lab', 'cells', `${ids[3]}_00001_.png`))).toBe(false)
    expect(fs.existsSync(path.join(env.outputs, '.lab', 'cells', `${ids[5]}_00001_.png`))).toBe(false)
    expect(fs.existsSync(path.join(env.outputs, '.lab', 'cells', `${ids[4]}_00001_.png`))).toBe(true)
    const L = readLedger(path.join(env.labDir, 'runs', 'r1'))
    expect(L.filter((e) => e.t === 'removed').map((e) => (e as { cell: string }).cell).sort()).toEqual([ids[3], ids[5]].sort())
    expect(readReadings(env).get(ids[0])?.rating).toBe('general')
    expect(readDoneCells(env).get(ids[3])?.removed).toBe(true)
    // Again: only the one that failed is read, and the removals stay.
    calls.length = 0
    const r2 = await readCells(env, client as never, ids, { run: 'r1', sleep: async () => {} })
    expect(calls).toEqual([1])
    expect(r2.quarantined).toHaveLength(2)
    expect(QUARANTINE_TAGS).toEqual(expect.arrayContaining(['child', 'children', 'loli', 'shota', 'toddler', 'baby', 'male_child', 'female_child', 'aged_down', 'kindergarten', 'school_uniform']))
  })
  it('gives up after two hours of memory refusals and leaves the cells pending', async () => {
    const env = tempEnv()
    appendDoneCell(env, { cellId: 'a'.repeat(16), rel: 'x.png', durationMs: 1, cold: false, cached: false, finishedAt: 1, run: 'r' })
    const slept: number[] = []
    const r = await readCells(env, { async tag() { return { busy: 'memory' as const, message: null } } } as never, ['a'.repeat(16)], { run: 'r', sleep: async (ms: number) => void slept.push(ms) })
    expect(slept.slice(0, 2)).toEqual([30_000, 60_000])
    expect(Math.max(...slept)).toBeLessThanOrEqual(5 * 60_000)
    expect(slept.reduce((a, b) => a + b, 0)).toBe(2 * 3600_000)
    expect(r.pending).toEqual(['a'.repeat(16)])
    expect(r.why).toBeTruthy()
  })
})

describe('reference photos', () => {
  it('are measured upright, named, marked, described and copied under .lab/refs', () => {
    const env = tempEnv()
    const info = addRef(env, sidewaysCatStandIn(), 'Cat')
    expect(info).toMatchObject({ id: 'cat', width: 3060, height: 4080, ext: 'jpg', mask: false })
    expect(info.sha12).toMatch(/^[0-9a-f]{12}$/)
    expect(refProblems(env, [{ id: 'cat' }, { id: 'scene', mask: true }])).toHaveLength(1)
    expect(refProblems(env, [{ id: 'scene' }])[0]).toMatch(/scene.*not in the lab yet.*scene\.jpg/)
    // A scene photo dropped in the folder by hand, named with capitals.
    fs.writeFileSync(path.join(env.labDir, 'refs', 'Scene.PNG'), png(120, 80, [10, 20, 30], { orientation: 8 }))
    const list = listRefs(env)
    expect(list.map((r) => r.id)).toEqual(['cat', 'scene'])
    expect([list[1].width, list[1].height]).toEqual([80, 120])
    expect(refProblems(env, [{ id: 'scene', mask: true }])[0]).toMatch(/no area marked/)
    // The rectangle is in upright pixels, and the mask is drawn at the upright size.
    const m = setMask(env, 'scene', { x: 10, y: 20, w: 30, h: 40 })
    expect(m).toMatchObject({ mask: true, rect: { x: 10, y: 20, w: 30, h: 40 } })
    const mask = fs.readFileSync(path.join(env.labDir, 'refs', 'scene.mask.png'))
    expect([sizeOf(mask).width, sizeOf(mask).height]).toEqual([80, 120])
    expect(refProblems(env, [{ id: 'scene', mask: true }])).toEqual([])
    const d = setDescribe(env, 'cat', 'a tortoiseshell tabby cat with a white belly and white paws, wearing a dark collar')
    expect(d.describe).toMatch(/^a tortoiseshell/)
    const copies = ensureRefCopies(env, ['cat', 'scene'])
    expect(copies.cat.ref).toBe(`.lab/refs/${info.sha12}.jpg`)
    // The mask's copy is named by the mask's own hash, so a rectangle drawn again is a new file.
    expect(m.maskSha12).toMatch(/^[0-9a-f]{12}$/)
    expect(copies.scene.mask).toBe(`.lab/refs/${m.maskSha12}.mask.png`)
    expect(fs.readFileSync(path.join(env.outputs, copies.scene.mask!))).toEqual(mask)
    const copy = fs.readFileSync(path.join(env.outputs, copies.cat.ref))
    expect(sizeOf(copy)).toMatchObject({ orientation: 6, width: 3060, height: 4080 })
    expect(copy.includes(Buffer.from('PhoneModel'))).toBe(false)
    expect(fs.existsSync(path.join(env.outputs, copies.scene.mask!))).toBe(true)
    expect(refIndex(env).cat.describe).toMatch(/^a tortoiseshell/)
  })
  it('a replaced photo loses its mask and its old file; a mask deleted by hand clears the area; non-pictures are refused', () => {
    const env = tempEnv()
    fs.mkdirSync(path.join(env.labDir, 'refs'), { recursive: true })
    fs.writeFileSync(path.join(env.labDir, 'refs', 'Scene.PNG'), png(120, 80))
    setMask(env, 'scene', { x: 10, y: 10, w: 20, h: 20 })
    addRef(env, jpegHeader({ width: 120, height: 80, orientation: 6 }), 'scene')
    const sc = listRefs(env).find((r) => r.id === 'scene')!
    expect(sc).toMatchObject({ mask: false, ext: 'jpg', width: 80, height: 120 })
    expect(fs.existsSync(path.join(env.labDir, 'refs', 'Scene.PNG'))).toBe(false)
    setMask(env, 'scene', { x: 10, y: 10, w: 20, h: 20 })
    expect(refIndex(env).scene.mask).toBe(true)
    fs.rmSync(path.join(env.labDir, 'refs', 'scene.mask.png'))
    expect(refIndex(env).scene).toMatchObject({ mask: false, rect: null })
    expect(() => addRef(env, Buffer.from('nope'), 'x')).toThrow(/JPEG, PNG or WebP/)
    expect(() => refIdFrom('...')).toThrow(/not a usable photo name/)
  })
  it('a room or table photo dropped by hand is the scene; renameRef moves a camera-named one with its mask', () => {
    const env = tempEnv()
    fs.mkdirSync(path.join(env.labDir, 'refs'), { recursive: true })
    fs.writeFileSync(path.join(env.labDir, 'refs', 'room.jpg'), jpegHeader({ width: 120, height: 80 }))
    expect(listRefs(env).map((r) => r.id)).toEqual(['scene'])
    fs.rmSync(path.join(env.labDir, 'refs', 'room.jpg'))
    fs.writeFileSync(path.join(env.labDir, 'refs', 'table.webp'), webpHeader('VP8L', 96, 64))
    expect(listRefs(env).map((r) => [r.id, r.ext, r.width])).toEqual([['scene', 'webp', 96]])
    fs.rmSync(path.join(env.labDir, 'refs', 'table.webp'))
    fs.writeFileSync(path.join(env.labDir, 'refs', 'IMG_2034.PNG'), png(120, 80, [1, 2, 3], { orientation: 8 }))
    expect(listRefs(env).map((r) => r.id)).toEqual(['img-2034'])
    setMask(env, 'img-2034', { x: 1, y: 1, w: 20, h: 20 })
    const r = renameRef(env, 'img-2034', 'scene')
    expect(r).toMatchObject({ id: 'scene', mask: true, rect: { x: 1, y: 1, w: 20, h: 20 } })
    expect(listRefs(env).map((x) => x.id)).toEqual(['scene'])
    expect(fs.readdirSync(path.join(env.labDir, 'refs')).sort()).toEqual(['index.json', 'scene.mask.png', 'scene.png'])
  })
})

describe('a photo\'s description is not a change to the photo', () => {
  // A started night keeps the words it was planned with, and one not started
  // yet is planned again at Start, so a new description moves nothing.
  it('setDescribe saves the words and leaves the photo\'s change time alone; a new rectangle moves it', () => {
    const env = tempEnv()
    addRef(env, sidewaysCatStandIn(), 'cat', () => 1000)
    expect(refUpdatedAt(env, 'cat')).toBe(1000)
    expect(setDescribe(env, 'cat', '  a grey   cat asleep ').describe).toBe('a grey cat asleep')
    expect(refIndex(env).cat.describe).toBe('a grey cat asleep')
    expect(refUpdatedAt(env, 'cat')).toBe(1000)
    expect(setDescribe(env, 'cat', null).describe).toBeNull()
    expect(refUpdatedAt(env, 'cat')).toBe(1000)
    setMask(env, 'cat', { x: 10, y: 10, w: 100, h: 100 }, () => 7000)
    expect(refUpdatedAt(env, 'cat')).toBe(7000)
    setDescribe(env, 'cat', 'a grey cat asleep')
    expect(refUpdatedAt(env, 'cat')).toBe(7000)
  })
})

describe('runnerClient', () => {
  it('talks to the app with no Origin and JSON on every POST, and reads its answers', async () => {
    const seen: { method?: string; url?: string; headers: http.IncomingHttpHeaders; body: string }[] = []
    const srv = http.createServer((req, res) => {
      let body = ''
      req.on('data', (c) => (body += c))
      req.on('end', () => {
        seen.push({ method: req.method, url: req.url, headers: req.headers, body })
        res.setHeader('Content-Type', 'application/json')
        if (req.url === '/api/capabilities') return res.end(JSON.stringify({ runner: true, runnerDesks: ['images', 'lab'], runnerReason: null }))
        if (req.url === '/api/runner') return res.end(JSON.stringify({ v: 1, available: true, reason: null, boot: 'b', rev: 1, lane: { held: null }, groups: [], jobs: [] }))
        if (req.url === '/api/runner/groups') {
          res.statusCode = 503
          return res.end(JSON.stringify({ error: 'off', busy: 'runner', reason: 'off' }))
        }
        if (req.url?.endsWith('/stop')) {
          res.statusCode = req.url.includes('nope') ? 404 : 200
          return res.end('{}')
        }
        if (req.url === '/api/vision/tag') {
          res.statusCode = 503
          return res.end(JSON.stringify({ error: 'short of memory', busy: 'memory' }))
        }
        res.statusCode = 404
        res.end('{}')
      })
    })
    await new Promise<void>((r) => srv.listen(0, '127.0.0.1', r))
    try {
      const c = runnerClient(`http://127.0.0.1:${(srv.address() as AddressInfo).port}/`)
      expect((await c.capabilities()).runnerDesks).toEqual(['images', 'lab'])
      expect((await c.snapshot()).v).toBe(1)
      const r = await c.submit({ v: 1, group: { id: 'g', desk: 'lab', kind: 'set', label: 'x', device: 'd' }, jobs: [] } as never)
      expect(r).toMatchObject({ ok: false, status: 503, busy: 'runner' })
      expect(await c.stopGroup('abc')).toBe(true)
      expect(await c.stopGroup('nope')).toBe(false)
      expect(await c.tag(['a.png'])).toEqual({ busy: 'memory', message: 'short of memory' })
      for (const s of seen) {
        expect(s.headers.origin).toBeUndefined()
        if (s.method === 'POST') expect(s.headers['content-type']).toBe('application/json')
      }
      expect(JSON.parse(seen.find((s) => s.url === '/api/vision/tag')!.body).images[0]).toEqual({ kind: 'output', rel: 'a.png' })
    } finally {
      srv.close()
    }
    await expect(runnerClient('http://127.0.0.1:9').snapshot()).rejects.toBeInstanceOf(AppUnreachable)
  })
  it('the stand-in runner accepts what the client sends and answers the tag call', async () => {
    const env = tempEnv()
    const si = await standInRunner(env.outputs)
    try {
      const c = runnerClient(si.url)
      expect((await c.capabilities()).runnerDesks).toContain('lab')
      si.tagBusy = 1
      expect(await c.tag(['.lab/cells/x.png'])).toMatchObject({ busy: 'memory' })
      const rows = await c.tag(['.lab/cells/x.png'])
      expect(Array.isArray(rows) && rows[0]).toMatchObject({ rel: '.lab/cells/x.png', rating: 'general' })
      expect(si.calls.every((x) => x.headers.origin === undefined)).toBe(true)
    } finally {
      await si.close()
    }
  })
})
