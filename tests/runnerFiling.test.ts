import { chmodSync, existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import type { ArchiveApi } from '../server/archive.mjs'
import { call } from './http'
import { archiveRoutes, groupBody, harness, loadArchive, recordOfDesk, runnerEnv, type Harness } from './runnerFake'

/**
 * Filing: the queue is the one filer of its own work. Each job becomes one
 * archive record under its own id, from the template the page built, with
 * ComfyUI's times, and the job is done only once that record is on disk.
 */

let restoreEnv = () => {}
beforeAll(() => {
  restoreEnv = runnerEnv()
})
afterAll(() => restoreEnv())

const asRoot = process.getuid?.() === 0
const archiveFile = (h: Harness) => path.join(h.outputs, '.switchgen', 'archive.json')
const recordsOnDisk = (h: Harness): Record<string, any> => {
  try {
    return JSON.parse(readFileSync(archiveFile(h), 'utf8')).records
  } catch {
    return {}
  }
}

/** Submit one job of `desk`, send it, and finish it as `finish` says. */
async function oneJob(h: Harness, desk: 'video' | 'images' | 'reel', finish: Parameters<Harness['comfy']['finish']>[1], spec = {}) {
  const r = await h.submit(groupBody({ desk, jobs: [spec] }))
  expect(r.status, r.body).toBe(200)
  const id = r.json().jobs[0].id as string
  await h.tick()
  const pid = h.job(id).promptId!
  h.comfy.finish(pid, finish)
  return { id, pid }
}

describe('a finished job\'s record', () => {
  it('is one record under the job\'s own id, with every field of the template and ComfyUI\'s times', async () => {
    const h = await harness()
    const template = recordOfDesk('video', {
      loras: [{ name: 'detail.safetensors', strength: 0.8 }],
      positive: 'a lighthouse at dusk, film grain',
      // What the server fills, or a browser's own marks: never taken from a page.
      id: 'page-made',
      no: 99,
      rev: 4,
      pending: true,
      recovered: true,
      promptId: 'from-the-page',
      durationMs: 1,
      at: 2,
    })
    const { id, pid } = await oneJob(
      h,
      'video',
      { files: [{ filename: 'clip.webm', subfolder: 'video', video: true }, { filename: 'clip.frame.png', subfolder: 'video' }], start: 20_000, end: 81_000 },
      { record: template },
    )
    await h.tick()
    const job = h.job(id)
    expect(job).toMatchObject({ status: 'done', entryId: id, entryNo: 1, durationMs: 61_000, ranAt: 20_000, finishedAt: 81_000, promptId: pid })
    const records = recordsOnDisk(h)
    expect(Object.keys(records)).toEqual([id])
    const r = records[id]
    const { id: _i, no: _n, rev: _r, pending: _p, recovered: _rc, promptId: _pi, durationMs: _d, at: _a, ...kept } = template as Record<string, unknown>
    expect(r).toMatchObject(kept)
    expect(r).toMatchObject({
      id,
      no: 1,
      kind: 'video',
      promptId: pid,
      durationMs: 61_000,
      at: 81_000,
      file: { filename: 'clip.webm', subfolder: 'video', type: 'output' },
    })
    expect(r.files.map((f: { filename: string }) => f.filename)).toEqual(['clip.webm', 'clip.frame.png'])
    expect(r.recovered).toBeUndefined()
    expect(r.pending).toBeUndefined()
    await h.runner.retire()
  })

  it('claims no time when ComfyUI kept no start, even with the time it was queued', async () => {
    const h = await harness()
    const { id } = await oneJob(h, 'images', { files: [{ filename: 'a.png' }], start: null, createTime: 5000, end: 9000 })
    await h.tick()
    expect(h.job(id)).toMatchObject({ status: 'done', durationMs: 0 })
    expect(recordsOnDisk(h)[id]).toMatchObject({ durationMs: 0, at: 9000 })
    expect(recordsOnDisk(h)[id].files).toBeUndefined()
    await h.runner.retire()
  })

  it('is dated when the runner saw the ending where ComfyUI kept no end stamp', async () => {
    const h = await harness()
    const { id, pid } = await oneJob(h, 'images', { files: [{ filename: 'b.png' }] })
    const entry = h.comfy.records.get(pid) as any
    entry.status.messages = entry.status.messages.filter((m: unknown[]) => m[0] !== 'execution_success')
    await h.tick(1, 500)
    expect(h.job(id).status).toBe('done')
    expect(recordsOnDisk(h)[id]).toMatchObject({ at: h.clock.t, durationMs: 0 })
    await h.runner.retire()
  })
})

describe('an answer from ComfyUI\'s cache', () => {
  it('is done as a repeat of the record that names the file, and files nothing new', async () => {
    const h = await harness()
    const first = await oneJob(h, 'video', { files: [{ filename: 'same.webm', video: true, node: '9' }] })
    await h.tick()
    const again = await oneJob(h, 'video', { files: [{ filename: 'same.webm', video: true, node: '9', write: false }], cachedNodes: ['9'] })
    await h.tick()
    expect(h.job(again.id)).toMatchObject({ status: 'done', repeatOf: first.id, entryId: first.id, entryNo: 1, durationMs: 0 })
    expect(h.job(again.id).primary).toMatchObject({ filename: 'same.webm', cached: true })
    expect(Object.keys(recordsOnDisk(h))).toEqual([first.id])
    await h.runner.retire()
  })

  it('is filed as the desk\'s record when the file\'s only record was filed after the fact', async () => {
    const h = await harness()
    const routes = archiveRoutes(h.archive)
    await call(routes, { method: 'POST', url: '/api/archive/upsert', body: { records: [{ id: 'found-later', at: 1, file: { filename: 'same.webm', subfolder: '', type: 'output' }, recovered: true }] } })
    const again = await oneJob(h, 'video', { files: [{ filename: 'same.webm', video: true, node: '9' }], cachedNodes: ['9'] })
    await h.tick()
    expect(h.job(again.id)).toMatchObject({ status: 'done', repeatOf: null, entryId: again.id, entryNo: 1 })
    expect(Object.keys(recordsOnDisk(h))).toEqual([again.id])
    await h.runner.retire()
  })

  it('is filed as usual when the file was not answered from the cache, whatever its name', async () => {
    const h = await harness()
    const first = await oneJob(h, 'video', { files: [{ filename: 'same.webm', video: true, node: '9' }] })
    await h.tick()
    const again = await oneJob(h, 'video', { files: [{ filename: 'same.webm', video: true, node: '9' }] })
    await h.tick()
    expect(h.job(again.id)).toMatchObject({ status: 'done', repeatOf: null, entryId: again.id, entryNo: 2 })
    expect(Object.keys(recordsOnDisk(h)).sort()).toEqual([first.id, again.id].sort())
    await h.runner.retire()
  })
})

describe('what the archive already holds', () => {
  it('a record the reader removed between two attempts stays removed, and the job is done with nothing filed', async () => {
    const base = await harness()
    await base.runner.retire()
    const real = base.archive
    let crash = () => {}
    const once: ArchiveApi = { ...real, fileOnce: async (r) => { const out = await real.fileOnce(r); crash(); return out } }
    const h = await harness({ root: base.root, outputs: base.outputs, archive: once })
    crash = () => void h.runner.retire()
    const { id } = await oneJob(h, 'video', { files: [{ filename: 'gone.webm', video: true }] })
    await h.tick()
    expect(h.onDisk().jobs[id].status).toBe('filing')
    const routes = archiveRoutes(real)
    const removed = await call(routes, { method: 'POST', url: '/api/archive/remove', body: { ids: [id] } })
    expect(removed.json().removed).toBe(1)
    const h2 = await h.restart({ archive: real })
    await h2.tick()
    expect(h2.job(id)).toMatchObject({ status: 'done', entryId: null, entryNo: null })
    const pull = (await call(routes, { url: '/api/archive?since=0' })).json()
    expect(pull.records.map((r: { id: string }) => r.id)).not.toContain(id)
    expect(pull.removed).toContain(id)
    await h2.runner.retire()
  })

  it('a record the recovery pass filed for the same file is replaced, keeping its number, star and note', async () => {
    const h = await harness()
    const routes = archiveRoutes(h.archive)
    const r = await h.submit(groupBody({ desk: 'video', jobs: [{}] }))
    const id = r.json().jobs[0].id
    await h.tick()
    const clip = { filename: 'found.webm', subfolder: 'video', type: 'output' }
    await call(routes, { method: 'POST', url: '/api/archive/upsert', body: { records: [{ id: 'filler', at: 1, file: { filename: 'x.png', subfolder: '', type: 'output' } }] } })
    const rec = await call(routes, { method: 'POST', url: '/api/archive/upsert', body: { records: [{ id: 'recovered-1', at: 1, file: clip, recovered: true, starred: true, note: 'keep' }] } })
    const no = rec.json().assigned[0].no
    expect(no).toBe(2)
    h.comfy.finish(h.job(id).promptId!, { files: [{ filename: 'found.webm', subfolder: 'video', video: true }] })
    await h.tick()
    expect(h.job(id)).toMatchObject({ status: 'done', entryId: id, entryNo: no })
    const records = recordsOnDisk(h)
    expect(records['recovered-1']).toBeUndefined()
    expect(records[id]).toMatchObject({ no, starred: true, note: 'keep' })
    expect(records[id].recovered).toBeUndefined()
    await h.runner.retire()
  })

  it('a record the recovery pass filed for the file, and the reader removed, stays removed when the queue files the job', async () => {
    // The queue was off while ComfyUI made the clip; the recovery pass filed
    // it, and the reader took it out, before the queue came back to file it.
    const h = await harness()
    const routes = archiveRoutes(h.archive)
    const id = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    await h.tick()
    h.comfy.finish(h.job(id).promptId!, { files: [{ filename: 'found.webm', subfolder: 'video', video: true }] })
    const clip = { filename: 'found.webm', subfolder: 'video', type: 'output' }
    const rec = await call(routes, { method: 'POST', url: '/api/archive/upsert', body: { records: [{ id: 'recovered-1', at: 1, file: clip, recovered: true }] } })
    expect(rec.json().assigned).toHaveLength(1)
    const rm = await call(routes, { method: 'POST', url: '/api/archive/remove', body: { ids: ['recovered-1'] } })
    expect(rm.json().removed).toBe(1)
    await h.tick()
    await h.tick()
    expect(h.job(id)).toMatchObject({ status: 'done', entryId: null, entryNo: null })
    expect(recordsOnDisk(h)[id]).toBeUndefined()
    const listed = (await call(routes, { url: '/api/outputs' })).json().files
    expect(listed.find((f: { rel: string }) => f.rel === 'video/found.webm')).toMatchObject({ dismissed: true })
    const pull = (await call(routes, { url: '/api/archive?since=0' })).json()
    expect(pull.removed).toContain('recovered-1')
    await h.runner.retire()
  })

  it('a record the reader removed does not keep out a file written again after the removal', async () => {
    const h = await harness()
    const routes = archiveRoutes(h.archive)
    const clip = { filename: 'again.webm', subfolder: '', type: 'output' }
    await call(routes, { method: 'POST', url: '/api/archive/upsert', body: { records: [{ id: 'old', at: 1, file: clip }] } })
    await call(routes, { method: 'POST', url: '/api/archive/remove', body: { ids: ['old'] } })
    // The file's time must come after the removal's.
    await new Promise((r) => setTimeout(r, 20))
    const { id } = await oneJob(h, 'video', { files: [{ filename: 'again.webm', video: true }] })
    await h.tick()
    expect(h.job(id)).toMatchObject({ status: 'done', entryId: id })
    expect(Object.keys(recordsOnDisk(h))).toEqual([id])
    await h.runner.retire()
  })

  it('a file ComfyUI answered from its cache is filed again after the reader removed its record', async () => {
    const h = await harness()
    const routes = archiveRoutes(h.archive)
    const a = await oneJob(h, 'video', { files: [{ filename: 'same.webm', video: true, node: '9' }] })
    await h.tick()
    expect(h.job(a.id)).toMatchObject({ status: 'done', entryId: a.id })
    await call(routes, { method: 'POST', url: '/api/archive/remove', body: { ids: [a.id] } })
    const b = await oneJob(h, 'video', { files: [{ filename: 'same.webm', video: true, node: '9', write: false }], cachedNodes: ['9'] })
    await h.tick()
    expect(h.job(b.id)).toMatchObject({ status: 'done', entryId: b.id })
    expect(Object.keys(recordsOnDisk(h))).toEqual([b.id])
    await h.runner.retire()
  })
})

describe('done, and not before', () => {
  it('is committed only once the archive\'s file on disk holds the record', async () => {
    const base = await harness()
    await base.runner.retire()
    const real = base.archive
    const seen: Record<string, unknown> = {}
    let h: Harness | null = null
    let id = ''
    const watched: ArchiveApi = {
      ...real,
      fileOnce: async (r) => {
        const out = await real.fileOnce(r)
        // The archive answers before its write lands.
        seen.onDiskAtAnswer = id in recordsOnDisk(h!)
        return out
      },
      durable: async () => {
        seen.statusBeforeDurable = h!.onDisk().jobs[id].status
        return real.durable()
      },
      unclaim: (rels) => {
        // Called just after the job's last commit.
        if (!('doneWithRecordOnDisk' in seen)) {
          seen.doneWithRecordOnDisk = h!.onDisk().jobs[id]?.status === 'done' && id in recordsOnDisk(h!)
        }
        real.unclaim(rels)
      },
    }
    h = await harness({ root: base.root, outputs: base.outputs, archive: watched })
    id = (await oneJob(h, 'images', { files: [{ filename: 'c.png' }] })).id
    await h.tick()
    expect(h.job(id).status).toBe('done')
    expect(seen).toEqual({ onDiskAtAnswer: false, statusBeforeDurable: 'filing', doneWithRecordOnDisk: true })
    await h.runner.retire()
  })

  it('waits, filing, when the archive could not write, and is done once it can', async () => {
    const base = await harness()
    await base.runner.retire()
    const real = base.archive
    let fail = true
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const flaky: ArchiveApi = { ...real, durable: async () => (fail ? false : real.durable()) }
    const h = await harness({ root: base.root, outputs: base.outputs, archive: flaky })
    const { id } = await oneJob(h, 'images', { files: [{ filename: 'd.png' }] })
    await h.tick(3)
    expect(h.job(id)).toMatchObject({ status: 'filing', wait: { for: 'disk' } })
    fail = false
    await h.tick()
    expect(h.job(id)).toMatchObject({ status: 'done', entryId: id })
    warn.mockRestore()
    await h.runner.retire()
  })

  it('waits, filing, while another server holds the archive, and is filed once it is free', async () => {
    const base = await harness()
    await base.runner.retire()
    const real = base.archive
    let busy = true
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const held: ArchiveApi = {
      ...real,
      fileOnce: async (r) => {
        if (busy) throw new Error('Another SwitchGen server (process 4242) is using this archive, so this one cannot share it.')
        return real.fileOnce(r)
      },
    }
    const h = await harness({ root: base.root, outputs: base.outputs, archive: held })
    const { id } = await oneJob(h, 'video', { files: [{ filename: 'held.webm', subfolder: 'v', video: true }] })
    await h.tick(3)
    expect(h.job(id)).toMatchObject({ status: 'filing', wait: { for: 'disk' }, entryId: null })
    expect(h.job(id).primary).toMatchObject({ filename: 'held.webm' })
    // Meanwhile the recovery pass is told the file is being filed.
    const listed = (await call(archiveRoutes(real), { url: '/api/outputs' })).json().files
    expect(listed.find((f: { rel: string }) => f.rel === 'v/held.webm')).toMatchObject({ filed: true })
    busy = false
    await h.tick()
    expect(h.job(id)).toMatchObject({ status: 'done', entryId: id })
    expect(Object.keys(recordsOnDisk(h))).toEqual([id])
    warn.mockRestore()
    await h.runner.retire()
  })

  it('claims a job\'s files again when a new load of the archive takes the list up with it still filing', async () => {
    const base = await harness()
    await base.runner.retire()
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const busy: ArchiveApi = { ...base.archive, fileOnce: async () => { throw new Error('busy') } }
    const h = await harness({ root: base.root, outputs: base.outputs, archive: busy })
    const { id } = await oneJob(h, 'video', { files: [{ filename: 'later.webm', subfolder: 'v', video: true }] })
    await h.tick()
    expect(h.job(id).status).toBe('filing')
    await h.runner.retire()
    // Vite's config reloaded: a new load of the archive, which has claimed nothing.
    const fresh = await loadArchive(h.outputs)
    const stalled: ArchiveApi = { ...fresh, fileOnce: async () => { throw new Error('busy') } }
    const h2 = await harness({ root: h.root, outputs: h.outputs, archive: stalled, comfy: h.comfy })
    const listed = (await call(archiveRoutes(fresh), { url: '/api/outputs' })).json().files
    expect(listed.find((f: { rel: string }) => f.rel === 'v/later.webm')).toMatchObject({ filed: true })
    warn.mockRestore()
    await h2.runner.retire()
  })
})

describe('a run that wrote no file', () => {
  it('fails a picture as no-file, and the batch goes on to the next picture', async () => {
    const h = await harness()
    const [P1, P2] = (await h.submit(groupBody({ desk: 'images', jobs: [{}, {}] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick()
    h.comfy.finish(h.job(P1).promptId!, { files: [] })
    await h.tick()
    expect(h.job(P1)).toMatchObject({ status: 'failed', entryId: null, error: { code: 'no-file', sent: true } })
    expect(h.job(P2).status).toBe('queued')
    h.comfy.finish(h.job(P2).promptId!, { files: [{ filename: 'p2.png' }] })
    await h.tick()
    expect(h.job(P2).status).toBe('done')
    expect(Object.keys(recordsOnDisk(h))).toEqual([P2])
    await h.runner.retire()
  })

  it('is done with nothing filed for a desk that asked for that', async () => {
    const h = await harness()
    const { id } = await oneJob(h, 'video', { files: [] })
    await h.tick()
    expect(h.job(id)).toMatchObject({ status: 'done', entryId: null, entryNo: null, error: null })
    // A reel shot takes only a clip: a still alone is not what it keeps.
    const shot = await oneJob(h, 'reel', { files: [{ filename: 'only.png' }] })
    await h.tick()
    expect(h.job(shot.id)).toMatchObject({ status: 'done', entryId: null, primary: null })
    expect(recordsOnDisk(h)).toEqual({})
    await h.runner.retire()
  })

  it('files the first file of any kind for a desk that takes any', async () => {
    const h = await harness()
    const { id } = await oneJob(h, 'images', { files: [{ filename: 'moving.webm', video: true }] })
    await h.tick()
    expect(h.job(id)).toMatchObject({ status: 'done', primary: { filename: 'moving.webm', kind: 'video' } })
    expect(recordsOnDisk(h)[id].kind).toBe('video')
    await h.runner.retire()
  })
})

describe('a refused picture', () => {
  it('ends the batch, and the pictures after it are skipped, never sent', async () => {
    const h = await harness()
    const g = (await h.submit(groupBody({ desk: 'images', jobs: [{}, {}, {}] }))).json()
    const [P1, P2, P3] = g.jobs.map((j: { id: string }) => j.id)
    await h.tick()
    h.comfy.finish(h.job(P1).promptId!, { files: [{ filename: 'p1.png' }] })
    h.comfy.sends.push(() => ({ answer: { refused: true, status: 400, message: 'bad node', node: '3', nodeType: 'KSampler', nodeErrors: null }, land: false }))
    await h.tick()
    expect(h.job(P1).status).toBe('done')
    expect(h.job(P2)).toMatchObject({ status: 'failed', error: { code: 'refused', message: 'bad node' } })
    expect(h.job(P3)).toMatchObject({ status: 'skipped', promptId: null, error: { code: 'skipped', sent: false, after: { jobId: P2, index: 2 } } })
    expect(h.group(g.group.id)).toMatchObject({ state: 'ended', endedBy: { jobId: P2, why: 'refused' } })
    await h.tick(3, 2000)
    expect(h.comfy.prompts()).toHaveLength(2)
    await h.runner.retire()
  })

  it('and a failed picture ends it too, but a no-file picture never does', async () => {
    const h = await harness()
    const [P1, P2, P3] = (await h.submit(groupBody({ desk: 'images', jobs: [{}, {}, {}] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick()
    h.comfy.finish(h.job(P1).promptId!, { files: [] })
    await h.tick()
    h.comfy.finish(h.job(P2).promptId!, { error: 'CUDA out of memory' })
    await h.tick()
    expect(h.job(P2)).toMatchObject({ status: 'failed', error: { code: 'failed', message: 'CUDA out of memory', node: '3', nodeType: 'VAEDecode' } })
    expect(h.job(P3)).toMatchObject({ status: 'skipped', error: { after: { jobId: P2, index: 2 } } })
    await h.runner.retire()
  })
})

describe('the breaker', () => {
  it('stops the queue after five failed passes in a row, and it takes no more work', async () => {
    const base = await harness()
    await base.runner.retire()
    const broken: ArchiveApi = { ...base.archive, fileOnce: async () => undefined as never }
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const h = await harness({ root: base.root, outputs: base.outputs, archive: broken })
    await oneJob(h, 'images', { files: [{ filename: 'e.png' }] })
    await h.tick(4)
    expect(h.runner.status().active).toBe(true)
    await h.tick()
    expect(h.runner.status()).toEqual({ active: false, desks: [], reason: 'The queue stopped after repeated faults; see the server log.' })
    expect(h.snap()).toMatchObject({ available: false, reason: 'The queue stopped after repeated faults; see the server log.' })
    const r = await h.submit(groupBody({ desk: 'images', jobs: [{}] }))
    expect(r.status).toBe(503)
    expect(r.json()).toMatchObject({ busy: 'runner', reason: 'The queue stopped after repeated faults; see the server log.' })
    warn.mockRestore()
    await h.runner.retire()
  })

  it('counts only passes in a row: one that goes through starts the count again', async () => {
    const base = await harness()
    await base.runner.retire()
    let broken = true
    const flaky: ArchiveApi = { ...base.archive, fileOnce: async (r) => (broken ? (undefined as never) : base.archive.fileOnce(r)) }
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const h = await harness({ root: base.root, outputs: base.outputs, archive: flaky })
    const { id } = await oneJob(h, 'images', { files: [{ filename: 'f.png' }] })
    await h.tick(4)
    broken = false
    await h.tick()
    expect(h.job(id).status).toBe('done')
    broken = true
    await oneJob(h, 'images', { files: [{ filename: 'g.png' }] })
    await h.tick(4)
    expect(h.runner.status().active).toBe(true)
    warn.mockRestore()
    await h.runner.retire()
  })
})

describe('work waiting when the breaker stopped the queue', () => {
  it('is noted in the folder, and held once a queue runs it again, since its pages sent their own work meanwhile', async () => {
    const base = await harness()
    await base.runner.retire()
    const broken: ArchiveApi = { ...base.archive, fileOnce: async () => undefined as never }
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      const h = await harness({ root: base.root, outputs: base.outputs, archive: broken })
      const [A] = (await h.submit(groupBody({ desk: 'images', jobs: [{}, {}] }))).json().jobs.map((j: { id: string }) => j.id)
      const C = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
      await h.tick()
      h.comfy.finish(h.job(A).promptId!, { files: [{ filename: 'e.png' }] })
      await h.tick(5)
      expect(h.runner.status().reason).toBe('The queue stopped after repeated faults; see the server log.')
      expect(h.job(C).status).toBe('waiting')
      expect(existsSync(path.join(h.dir, 'paused.json'))).toBe(true)
      // The app server restarted, with an archive that files again.
      const again = await h.restart({ archive: base.archive })
      expect(again.runner.status().active).toBe(true)
      expect(again.snap().lane.held).toMatchObject({ why: 'paused', scope: 'all' })
      expect(existsSync(path.join(h.dir, 'paused.json'))).toBe(false)
      const sent = again.comfy.prompts().length
      await again.tick(2, 2000)
      expect(again.comfy.prompts()).toHaveLength(sent)
      expect(again.job(C)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
      await again.runner.retire()
    } finally {
      warn.mockRestore()
    }
  })
})

describe('the archive, as the queue uses it', () => {
  it('says it does not hold the archive while a live process does, and takes it once that lock is gone', async () => {
    const h = await harness()
    await h.runner.retire()
    const onLinux = existsSync('/proc/1/stat') && existsSync('/proc/sys/kernel/random/boot_id')
    if (!onLinux) return
    const stat = readFileSync('/proc/1/stat', 'utf8')
    const start = stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19]
    const boot = readFileSync('/proc/sys/kernel/random/boot_id', 'utf8').trim()
    const fresh = await loadArchive(path.join(h.root, 'held'))
    const freshLock = `${fresh.archiveFile}.lock`
    mkdirSync(path.dirname(freshLock), { recursive: true })
    writeFileSync(freshLock, JSON.stringify({ pid: 1, boot, start }))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    expect(fresh.holds()).toBe(false)
    await expect(fresh.fileOnce({ id: 'x', at: 1, file: { filename: 'x.png', subfolder: '', type: 'output' } })).rejects.toThrow(/Another SwitchGen server \(process 1\)/)
    unlinkSync(freshLock)
    expect(fresh.holds()).toBe(true)
    expect(JSON.parse(readFileSync(freshLock, 'utf8')).pid).toBe(process.pid)
    warn.mockRestore()
  })

  it('refuses what is not a record', async () => {
    const h = await harness()
    await h.runner.retire()
    await expect(h.archive.fileOnce({ id: 'no-time', file: { filename: 'a.png', subfolder: '', type: 'output' } })).rejects.toThrow('not a record')
    await expect(h.archive.fileOnce({ id: 'no-file', at: 1 })).rejects.toThrow('not a record')
  })

  it('names the record for a file, a desk\'s before one filed after the fact', async () => {
    const h = await harness()
    await h.runner.retire()
    const routes = archiveRoutes(h.archive)
    const file = { filename: 'n.png', subfolder: 'pics', type: 'output' }
    expect(await h.archive.recordNaming('pics/n.png')).toBeNull()
    await call(routes, { method: 'POST', url: '/api/archive/upsert', body: { records: [{ id: 'rec', at: 1, file, recovered: true }] } })
    expect(await h.archive.recordNaming('pics/n.png')).toMatchObject({ id: 'rec', recovered: true })
    await h.archive.fileOnce({ id: 'desk', at: 2, file, promptId: 'p1', durationMs: 1234 })
    expect(await h.archive.recordNaming('pics/n.png')).toMatchObject({ id: 'desk', recovered: false, promptId: 'p1', durationMs: 1234 })
    expect(await h.archive.recordNaming('pics/other.png')).toBeNull()
    // An archive from before records were folded together can hold both, whichever came first.
    const older = path.join(h.root, 'older')
    mkdirSync(path.join(older, '.switchgen'), { recursive: true })
    writeFileSync(path.join(older, '.switchgen', 'archive.json'), JSON.stringify({
      v: 3,
      epoch: 'e',
      rev: 2,
      nextNo: 4,
      records: {
        rec1: { id: 'rec1', no: 1, rev: 1, at: 1, file, recovered: true },
        desk: { id: 'desk', no: 2, rev: 2, at: 2, file, promptId: 'p2' },
        rec2: { id: 'rec2', no: 3, rev: 3, at: 3, file, recovered: true },
      },
      tombstones: {},
      dismissed: {},
    }))
    const both = await loadArchive(older)
    expect(await both.recordNaming('pics/n.png')).toEqual({ id: 'desk', no: 2, recovered: false, promptId: 'p2', durationMs: 0 })
  })

  it('answers existed for an id it holds, and files it once', async () => {
    const h = await harness()
    await h.runner.retire()
    const rec = { id: 'once', at: 1, file: { filename: 'o.png', subfolder: '', type: 'output' } }
    expect(await h.archive.fileOnce(rec)).toEqual({ entryId: 'once', no: 1, existed: false })
    expect(await h.archive.fileOnce({ ...rec, at: 5 })).toEqual({ entryId: 'once', no: 1, existed: true })
    expect(await h.archive.durable()).toBe(true)
    expect(Object.keys(recordsOnDisk(h))).toEqual(['once'])
  })

  it('keeps a file claimed until it is let go, however many times it was claimed', async () => {
    const h = await harness()
    await h.runner.retire()
    const routes = archiveRoutes(h.archive)
    writeFileSync(path.join(h.outputs, 'claimed.png'), 'x')
    const filed = async () => (await call(routes, { url: '/api/outputs' })).json().files.find((f: { rel: string }) => f.rel === 'claimed.png')?.filed
    h.archive.claim(['claimed.png'])
    h.archive.claim(['claimed.png'])
    expect(await filed()).toBe(true)
    h.archive.unclaim(['claimed.png'])
    expect(await filed()).toBeUndefined()
  })

  it.skipIf(asRoot)('says a write did not land while the archive\'s folder cannot be written, and did once it can', async () => {
    const h = await harness()
    await h.runner.retire()
    await h.archive.fileOnce({ id: 'first', at: 1, file: { filename: 'f.png', subfolder: '', type: 'output' } })
    expect(await h.archive.durable()).toBe(true)
    const folder = path.dirname(archiveFile(h))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    chmodSync(folder, 0o500)
    try {
      await h.archive.fileOnce({ id: 'second', at: 1, file: { filename: 's.png', subfolder: '', type: 'output' } })
      expect(await h.archive.durable()).toBe(false)
    } finally {
      chmodSync(folder, 0o700)
    }
    expect(await h.archive.durable()).toBe(true)
    expect(Object.keys(recordsOnDisk(h)).sort()).toEqual(['first', 'second'])
    warn.mockRestore()
  })
})
