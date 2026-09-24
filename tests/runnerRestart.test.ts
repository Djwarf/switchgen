import { existsSync, mkdtempSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import type { ArchiveApi } from '../server/archive.mjs'
import { call } from './http'
import { endHandovers } from '../server/runner/store.mjs'
import { clearRegistry, groupBody, harness, loadArchive, mountPlugin, runnerEnv, standIn, type Harness } from './runnerFake'

/**
 * The queue after its process stops, at any point: every change of status is
 * on disk before the thing it allows, so a runner that takes the list up
 * again never sends a job twice and files each one once. "Restart" here is a
 * new runner on the same folder after the old one has retired, which is what
 * Vite does when its config reloads, and what the next process does after a
 * crash; "crash at a point" is a retire called from inside that step.
 */

let restoreEnv = () => {}
beforeAll(() => {
  restoreEnv = runnerEnv()
})
afterAll(() => restoreEnv())

const recordsOnDisk = (h: Harness) => {
  try {
    return JSON.parse(readFileSync(path.join(h.outputs, '.switchgen', 'archive.json'), 'utf8')).records as Record<string, any>
  } catch {
    return {}
  }
}

describe('a job left sending', () => {
  it('is asked about by its id and taken as queued, never sent again', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    h.comfy.hooks.submit = () => void h.runner.retire()
    await h.tick()
    expect(h.onDisk().jobs[A].status).toBe('sending')
    const h2 = await h.restart()
    await h2.tick()
    const [pid] = h2.comfy.prompts()
    expect(h2.job(A)).toMatchObject({ status: 'queued', promptId: pid })
    expect(h2.comfy.prompts()).toHaveLength(1)
    await h2.runner.retire()
  })

  it('is unsent when ComfyUI never had it, and is not sent again', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    h.comfy.sends.push(() => {
      void h.runner.retire()
      return { answer: { unknown: true, reason: 'the process ended' }, land: false }
    })
    await h.tick()
    const h2 = await h.restart()
    await h2.tick(4, 2000)
    expect(h2.job(A)).toMatchObject({ status: 'unsent', error: { code: 'unsent', sent: false } })
    expect(h2.comfy.prompts()).toHaveLength(1)
    await h2.runner.retire()
  })
})

describe('a job left releasing', () => {
  it('waits again, releases again, and is sent once', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    h.comfy.hooks.free = () => void h.runner.retire()
    await h.tick()
    expect(h.onDisk().jobs[A].status).toBe('releasing')
    expect(h.comfy.prompts()).toEqual([])
    const h2 = await h.restart()
    expect(h2.job(A).status).toBe('waiting')
    await h2.tick()
    expect(h2.job(A).status).toBe('queued')
    expect(h2.comfy.count('free')).toBe(2)
    expect(h2.comfy.prompts()).toHaveLength(1)
    await h2.runner.retire()
  })
})

describe('waiting work across a restart of the app server', () => {
  it('is sent in the same order, by itself', async () => {
    const h = await harness()
    const ids = (await h.submit(groupBody({ desk: 'video', jobs: [{}, {}, {}] }))).json().jobs.map((j: { id: string }) => j.id)
    const h2 = await h.restart()
    expect(h2.snap().lane.held).toBeNull()
    for (const id of ids) {
      await h2.tick()
      const pid = h2.job(id).promptId!
      expect(h2.job(id).status).toBe('queued')
      h2.comfy.finish(pid, { files: [{ filename: `${id}.webm`, video: true }] })
    }
    await h2.tick()
    expect(h2.comfy.prompts()).toEqual(ids.map((id: string) => h2.job(id).promptId))
    await h2.runner.retire()
  })

  it('keeps a held lane held', async () => {
    const h = await harness()
    const [A, B] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick()
    h.comfy.restart()
    await h.tick(3, 21_000)
    expect(h.job(A).status).toBe('lost')
    const h2 = await h.restart()
    expect(h2.snap().lane.held).toMatchObject({ why: 'lost', jobId: A })
    await h2.tick(5, 2000)
    expect(h2.job(B)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect(h2.comfy.prompts()).toHaveLength(1)
    await h2.runner.retire()
  })
})

describe('after the machine itself restarted', () => {
  it('holds all the waiting work until the reader says, then sends it', async () => {
    const h = await harness({ boot: 'boot-1' })
    const [A] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    const P = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    const h2 = await h.restart({ boot: 'boot-2' })
    expect(h2.snap().lane.held).toMatchObject({ why: 'restart', scope: 'all', jobId: null })
    await h2.tick(15, 2000)
    expect(h2.comfy.log).toEqual([])
    expect(h2.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect(h2.job(P)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect((await h2.post('/api/runner/lane', { action: 'send' })).status).toBe(200)
    await h2.tick(1, 2000)
    expect(h2.comfy.prompts()).toHaveLength(1)
    await h2.runner.retire()
  })

  it('takes the place of a hold for a lost clip', async () => {
    const h = await harness({ boot: 'boot-1' })
    const [A] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick()
    h.comfy.restart()
    await h.tick(3, 21_000)
    expect(h.snap().lane.held).toMatchObject({ why: 'lost', jobId: A })
    const h2 = await h.restart({ boot: 'boot-2' })
    expect(h2.snap().lane.held).toMatchObject({ why: 'restart', scope: 'all' })
    await h2.runner.retire()
  })

  it('holds nothing when nothing waits, and nothing after an ordinary restart', async () => {
    const h = await harness({ boot: 'boot-1' })
    const h2 = await h.restart({ boot: 'boot-2' })
    expect(h2.snap().lane.held).toBeNull()
    await h2.submit(groupBody({ desk: 'images', jobs: [{}] }))
    const h3 = await h2.restart({ boot: 'boot-2' })
    expect(h3.snap().lane.held).toBeNull()
    await h3.tick()
    expect(h3.comfy.prompts()).toHaveLength(1)
    await h3.runner.retire()
  })
})

describe('a crash at each point of a job', () => {
  /**
   * Retire the runner at one point, start another on the same folder with
   * the archive as it is, and run to the end: one prompt and one record,
   * under one number, whatever the point.
   */
  async function crashAt(point: 'sending' | 'prompt' | 'history' | 'fileOnce' | 'durable') {
    let current: Harness | null = null
    const retire = () => void current!.runner.retire()
    const base = await harness()
    await base.runner.retire()
    const real = base.archive
    const archive: ArchiveApi = {
      ...real,
      fileOnce: async (r) => {
        const out = await real.fileOnce(r)
        if (point === 'fileOnce') retire()
        return out
      },
      durable: async () => {
        if (point === 'durable') retire()
        return real.durable()
      },
    }
    const h = await harness({ root: base.root, outputs: base.outputs, comfy: base.comfy, clock: base.clock, archive })
    current = h
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id as string
    let queueReads = 0
    if (point === 'sending') h.comfy.hooks.submit = retire
    // The third read is the one just after the prompt was accepted.
    if (point === 'prompt') h.comfy.hooks.queue = () => void (++queueReads === 3 && retire())
    await h.tick()
    h.comfy.hooks = {}
    const pid = h.comfy.prompts()[0]!
    h.comfy.finish(pid, { files: [{ filename: `${point}.webm`, video: true }] })
    if (point === 'history') h.comfy.hooks.history = retire
    await h.tick(2, 2000)
    // Stopped part way, one way or another.
    expect(h.onDisk().jobs[A].status, point).not.toBe('done')
    const after = await h.restart({ archive: real })
    await after.tick(4, 2000)
    const job = after.job(A)
    expect(job.status, point).toBe('done')
    expect(after.comfy.accepted.map((a) => a.id), point).toEqual([pid])
    const records = recordsOnDisk(after)
    expect(Object.keys(records), point).toEqual([A])
    expect(records[A].no, point).toBe(job.entryNo)
    expect(records[A].promptId, point).toBe(pid)
    await after.runner.retire()
  }

  it('after the sending commit', () => crashAt('sending'))
  it('after the prompt was accepted', () => crashAt('prompt'))
  it('after the history read', () => crashAt('history'))
  it('after the archive took the record', () => crashAt('fileOnce'))
  it('before the archive\'s write was on disk', () => crashAt('durable'))
})

describe('taking the list up', () => {
  it('sets aside a list it cannot read and starts empty', async () => {
    const h = await harness()
    await h.runner.retire()
    writeFileSync(path.join(h.dir, 'state.json'), '{not json')
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const h2 = await harness({ root: h.root, outputs: h.outputs, archive: h.archive, clock: h.clock })
    expect(h2.runner.status().active).toBe(true)
    warn.mockRestore()
    expect(readdirSync(h.dir)).toContain(`state.broken-${h.clock.t}.json`)
    expect(readFileSync(path.join(h.dir, `state.broken-${h.clock.t}.json`), 'utf8')).toBe('{not json')
    expect(h2.snap().jobs).toEqual([])
    await h2.runner.retire()
  })

  it('fails a job still to be done whose saved graph has gone', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    await h.runner.retire()
    unlinkSync(path.join(h.dir, 'jobs', `${A}.json`))
    const h2 = await harness({ root: h.root, outputs: h.outputs, archive: h.archive, comfy: h.comfy })
    expect(h2.job(A)).toMatchObject({ status: 'failed', error: { code: 'internal', sent: false } })
    await h2.tick()
    expect(h2.comfy.prompts()).toEqual([])
    await h2.runner.retire()
  })

  it('stays off when the archive says yes but its lock does not name this process', async () => {
    const h = await harness()
    await h.runner.retire()
    const lock = `${h.archive.archiveFile}.lock`
    const cases: [string, () => void][] = [
      ['another pid', () => writeFileSync(lock, JSON.stringify({ pid: 1 }))],
      ['no lock at all', () => unlinkSync(lock)],
    ]
    for (const [what, stage] of cases) {
      stage()
      const h2 = await harness({ root: h.root, outputs: h.outputs, archive: { ...h.archive, holds: () => true }, comfy: h.comfy })
      expect(h2.runner.status(), what).toEqual({ active: false, desks: [], reason: 'Another SwitchGen server holds the archive, and the queue with it.' })
      expect(h2.comfy.sockets, what).toBe(0)
      const r = await h2.submit(groupBody({ desk: 'images', jobs: [{}] }))
      expect(r.status, what).toBe(503)
      await h2.runner.retire()
    }
  })

  it('never reads or writes its folder while another server holds the archive', async () => {
    const base = await harness()
    await base.runner.retire()
    const h = await harness({ archive: { ...base.archive, holds: () => false } })
    expect(h.runner.status().reason).toBe('Another SwitchGen server holds the archive, and the queue with it.')
    expect(() => readdirSync(h.dir)).toThrow()
    await h.runner.retire()
  })
})

describe('the queue\'s folder, held by one queue at a time', () => {
  const FOLDER = 'Another SwitchGen server runs the queue from the same folder.'

  it('stands a second queue on the same folder back, with its own archive, until the first lets go', async () => {
    const a = await harness()
    const A = (await a.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      // Another archive, as a second server given SWITCHGEN_ARCHIVE of its own
      // has, over the same folder of work.
      const other = await loadArchive(path.join(a.outputs, 'b'))
      const b = await harness({ root: a.root, outputs: a.outputs, dir: a.dir, archive: other, comfy: a.comfy, clock: a.clock })
      expect(b.runner.status()).toEqual({ active: false, desks: [], reason: FOLDER })
      // It shows the list read-only, and notes no pause: the other queue runs that work.
      expect(b.snap()).toMatchObject({ available: false, reason: FOLDER })
      expect(b.snap().jobs.map((j) => j.id)).toEqual([A])
      expect(existsSync(path.join(a.dir, 'paused.json'))).toBe(false)
      await a.tick()
      await b.tick(3, 2000)
      expect(a.comfy.prompts()).toEqual([a.job(A).promptId])
      expect(JSON.parse(readFileSync(path.join(a.dir, 'lock'), 'utf8')).pid).toBe(process.pid)
      expect((await b.submit(groupBody({ desk: 'video', jobs: [{}] }))).status).toBe(503)

      await a.runner.retire()
      expect(existsSync(path.join(a.dir, 'lock'))).toBe(false)
      await b.tick()
      expect(b.runner.status().active).toBe(true)
      expect(b.snap().lane.held).toBeNull()
      expect(existsSync(path.join(a.dir, 'lock'))).toBe(true)
      await b.runner.retire()
    } finally {
      warn.mockRestore()
    }
  })

  it('takes over a lock left by a process that has gone', async () => {
    const h = await harness()
    await h.runner.retire()
    // Above the largest process number Linux gives out, so no process has it.
    writeFileSync(path.join(h.dir, 'lock'), JSON.stringify({ pid: 2 ** 22 + 7, owner: 'x' }))
    const h2 = await harness({ root: h.root, outputs: h.outputs, archive: h.archive })
    expect(h2.runner.status().active).toBe(true)
    expect(JSON.parse(readFileSync(path.join(h.dir, 'lock'), 'utf8')).pid).toBe(process.pid)
    await h2.runner.retire()
  })
})

describe('work that waited while the queue was not running', () => {
  const OFF = 'Turned off with SWITCHGEN_RUNNER=off.'
  const HELD = 'Another SwitchGen server holds the archive, and the queue with it.'

  it('is listed read-only while the queue is off, and held once it runs again until the reader says', async () => {
    const h = await harness()
    const [A, B] = (await h.submit(groupBody({ desk: 'images', jobs: [{}, {}] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.runner.retire()

    const off = await harness({ root: h.root, outputs: h.outputs, archive: h.archive, comfy: h.comfy, clock: h.clock, enabled: false })
    const snap = (await off.get('/api/runner')).json()
    expect(snap).toMatchObject({ available: false, reason: OFF, lane: { held: null } })
    expect(snap.jobs.map((j: { id: string; status: string; wait: unknown }) => [j.id, j.status, j.wait])).toEqual([[A, 'waiting', null], [B, 'waiting', null]])
    expect((await off.get(`/api/runner/jobs/${A}`)).status).toBe(200)
    expect((await off.post('/api/runner/lane', { action: 'send' })).status).toBe(503)
    expect(existsSync(path.join(h.dir, 'paused.json'))).toBe(true)
    await off.runner.retire()

    const on = await harness({ root: h.root, outputs: h.outputs, archive: h.archive, comfy: h.comfy, clock: h.clock })
    expect(on.snap().lane.held).toMatchObject({ why: 'paused', scope: 'all', jobId: null })
    expect(existsSync(path.join(h.dir, 'paused.json'))).toBe(false)
    await on.tick(3, 2000)
    expect(on.comfy.prompts()).toEqual([])
    expect(on.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect((await on.post('/api/runner/lane', { action: 'send' })).status).toBe(200)
    await on.tick(1, 2000)
    expect(on.comfy.prompts()).toHaveLength(1)
    await on.runner.retire()

    // An ordinary restart after that holds nothing.
    const again = await harness({ root: h.root, outputs: h.outputs, archive: h.archive, comfy: h.comfy, clock: h.clock })
    expect(again.snap().lane.held).toBeNull()
    await again.runner.retire()
  })

  it('is listed while the queue stands back for the archive, and held once it runs', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    await h.runner.retire()
    let holds = false
    const b = await harness({ root: h.root, outputs: h.outputs, archive: { ...h.archive, holds: () => holds && h.archive.holds() }, comfy: h.comfy, clock: h.clock })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      expect(b.snap()).toMatchObject({ available: false, reason: HELD })
      expect(b.snap().jobs.map((j) => [j.id, j.status])).toEqual([[A, 'waiting']])
      holds = true
      await b.tick()
      expect(b.snap()).toMatchObject({ available: true, lane: { held: { why: 'paused', scope: 'all' } } })
      await b.tick(2, 2000)
      expect(b.comfy.prompts()).toEqual([])
      await b.runner.retire()
    } finally {
      warn.mockRestore()
    }
  })

  it('is not held when another queue ran that folder while this one stood back', async () => {
    const a = await harness()
    const A = (await a.submit(groupBody({ desk: 'images', jobs: [{}, {}] }))).json().jobs[0].id
    let holds = false
    const b = await harness({ root: a.root, outputs: a.outputs, dir: a.dir, archive: { ...a.archive, holds: () => holds && a.archive.holds() }, comfy: a.comfy, clock: a.clock })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      expect(b.snap()).toMatchObject({ available: false, reason: HELD })
      expect(b.snap().jobs.map((j) => j.id)).toContain(A)
      await b.tick(2, 2000)
      expect(existsSync(path.join(a.dir, 'paused.json'))).toBe(false)
      await a.runner.retire()
      holds = true
      await b.tick()
      expect(b.runner.status().active).toBe(true)
      expect(b.snap().lane.held).toBeNull()
      await b.tick()
      expect(b.comfy.prompts()).toHaveLength(1)
      await b.runner.retire()
    } finally {
      warn.mockRestore()
    }
  })
})

describe('the hold on work that waited while the queue was off', () => {
  /** A queue running again after it was off with this work waiting, so a paused hold stands. */
  async function pausedWith(desk: 'video' | 'images', jobs: object[]) {
    const h = await harness()
    const ids = (await h.submit(groupBody({ desk, jobs }))).json().jobs.map((j: { id: string }) => j.id) as string[]
    await h.runner.retire()
    const off = await harness({ root: h.root, outputs: h.outputs, archive: h.archive, comfy: h.comfy, clock: h.clock, enabled: false })
    await off.runner.retire()
    const on = await harness({ root: h.root, outputs: h.outputs, archive: h.archive, comfy: h.comfy, clock: h.clock })
    return { on, ids }
  }

  it('does not hold work made after it began, and the reader\'s word to stop leaves that work alone', async () => {
    const { on, ids: [A] } = await pausedWith('video', [{ heavy: true }])
    const held = on.snap().lane.held!
    expect(held).toMatchObject({ why: 'paused', scope: 'all' })
    on.clock.t += 1000
    const P = (await on.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    await on.tick(2, 1000)
    expect(on.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect(on.job(P).promptId).toBeTruthy()
    expect(on.comfy.prompts()).toEqual([on.job(P).promptId])
    const word = await on.post('/api/runner/lane', { action: 'stop', since: held.since })
    expect(word.status).toBe(200)
    expect(word.json().stopped).toEqual([A])
    expect(on.job(P).status).not.toBe('stopped')
    expect(on.snap().lane.held).toBeNull()
    await on.runner.retire()
  })

  it('lets go once the work it held is gone, with work made after it still waiting', async () => {
    const { on, ids: [A] } = await pausedWith('images', [{}])
    on.clock.t += 1000
    const P = (await on.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    expect(on.job(P).wait?.for).not.toBe('held')
    const stop = await on.post(`/api/runner/jobs/${A}/stop`)
    expect(stop.status, stop.body).toBe(200)
    expect(on.snap().lane.held).toBeNull()
    await on.runner.retire()
  })

  it('takes in work made under it that then waited through another time the queue was off', async () => {
    const { on, ids: [A] } = await pausedWith('images', [{}])
    const first = on.snap().lane.held!
    on.clock.t += 1000
    const B = (await on.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await on.runner.retire()
    const off = await harness({ root: on.root, outputs: on.outputs, archive: on.archive, comfy: on.comfy, clock: on.clock, enabled: false })
    await off.runner.retire()
    on.clock.t += 1000
    const back = await harness({ root: on.root, outputs: on.outputs, archive: on.archive, comfy: on.comfy, clock: on.clock })
    const second = back.snap().lane.held!
    expect(second).toMatchObject({ why: 'paused', scope: 'all' })
    expect(second.since).toBeGreaterThan(first.since)
    await back.tick(2, 1000)
    expect(back.comfy.prompts()).toEqual([])
    expect(back.job(A).wait).toEqual({ for: 'held' })
    expect(back.job(B).wait).toEqual({ for: 'held' })
    await back.runner.retire()
  })

  // The lane has room for one hold, so a heavy clip lost while this one
  // stands is added to it: it then covers the heavy work too (heavyAfter),
  // names the lost clip, and keeps its since, so it still covers no light
  // work made after it began.
  it('takes in the heavy work behind a heavy clip lost while it stands, since the lane has room for one hold', async () => {
    const { on, ids: [A] } = await pausedWith('images', [{}])
    const first = on.snap().lane.held!
    on.clock.t += 1000
    const [X, Y] = (await on.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await on.tick()
    expect(on.job(X).promptId).toBeTruthy()
    on.comfy.restart()
    await on.tick(3, 21_000)
    expect(on.job(X).status).toBe('lost')
    const held = on.snap().lane.held!
    expect(held).toMatchObject({ why: 'paused', scope: 'all', jobId: X, since: first.since })
    expect(held.heavyAfter).toBeGreaterThan(first.since)
    await on.tick(3, 2000)
    expect(on.job(Y)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect(on.comfy.prompts()).toHaveLength(1)
    expect(on.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    await on.runner.retire()
  })

  it('holds the heavy work behind a clip lost while it stands, not light work made after it began, and a word to stop leaves that work alone', async () => {
    const { on, ids: [A] } = await pausedWith('video', [{ heavy: true }])
    const first = on.snap().lane.held!
    expect(first).toMatchObject({ why: 'paused', scope: 'all', jobId: null })
    on.clock.t += 1000
    const X = (await on.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await on.tick()
    expect(on.job(X).promptId).toBeTruthy()
    // Made while X runs: a heavy clip and a light picture, neither of which waited through the pause.
    on.clock.t += 1000
    const P = (await on.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    const L = (await on.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    await on.tick()
    on.comfy.restart()
    await on.tick(3, 21_000)
    expect(on.job(X).status).toBe('lost')
    const held = on.snap().lane.held!
    expect(held).toMatchObject({ why: 'paused', scope: 'all', jobId: X, since: first.since })
    expect(held.heavyAfter).toBeGreaterThan(first.since)
    expect(on.job(P).wait).toEqual({ for: 'held' })
    expect(on.job(L).wait?.for).not.toBe('held')
    const word = await on.post('/api/runner/lane', { action: 'stop', since: held.since })
    expect(word.status).toBe(200)
    expect([...word.json().stopped].sort()).toEqual([A, P].sort())
    expect(on.job(L).status).not.toBe('stopped')
    expect(on.snap().lane.held).toBeNull()
    await on.runner.retire()
  })

  it('sends light work made after a clip was lost while it stands, and the heavy work behind the loss waits', async () => {
    const { on, ids: [A] } = await pausedWith('video', [{}])
    const first = on.snap().lane.held!
    on.clock.t += 1000
    const [X, Y] = (await on.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await on.tick()
    on.comfy.restart()
    await on.tick(3, 21_000)
    expect(on.job(X).status).toBe('lost')
    on.clock.t += 1000
    const L = (await on.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    await on.tick(3, 2000)
    expect(on.job(L).promptId).toBeTruthy()
    expect(on.job(Y)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect(on.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect(on.snap().lane.held).toMatchObject({ why: 'paused', since: first.since, jobId: X })
    await on.runner.retire()
  })

  it('becomes the hold for the lost clip once the work that waited through the pause is gone', async () => {
    const { on, ids: [A] } = await pausedWith('images', [{}])
    const first = on.snap().lane.held!
    on.clock.t += 1000
    const [X, Y] = (await on.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await on.tick()
    on.comfy.restart()
    await on.tick(3, 21_000)
    const added = on.snap().lane.held!
    expect(added).toMatchObject({ why: 'paused', since: first.since, jobId: X })
    const stop = await on.post(`/api/runner/jobs/${A}/stop`)
    expect(stop.status, stop.body).toBe(200)
    const now = on.snap().lane.held!
    expect(now).toEqual({ why: 'lost', scope: 'heavy', jobId: X, since: added.heavyAfter })
    expect(on.job(Y).wait).toEqual({ for: 'held' })
    // A word naming the hold as it stood before is refused; one naming it as it stands now is taken.
    expect((await on.post('/api/runner/lane', { action: 'send', since: first.since })).status).toBe(409)
    expect((await on.post('/api/runner/lane', { action: 'send', since: now.since })).status).toBe(200)
    await on.tick(3, 2000)
    expect(on.job(Y).promptId).toBeTruthy()
    await on.runner.retire()
  })
})

describe('the folder handed from one queue to the next in one process', () => {
  const HELD = 'Another SwitchGen server holds the archive, and the queue with it.'
  afterEach(() => endHandovers())

  it('stays held all through, so a server standing back for the archive notes no pause, and the next queue holds nothing', async () => {
    const a = await harness()
    const [A] = (await a.submit(groupBody({ desk: 'images', jobs: [{}, {}] }))).json().jobs.map((j: { id: string }) => j.id)
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      // A second server on the same outputs, with the same folder, which
      // another server's archive lock keeps standing back.
      const b = await harness({ root: a.root, outputs: a.outputs, dir: a.dir, archive: { ...a.archive, holds: () => false }, comfy: a.comfy, clock: a.clock })
      expect(b.snap()).toMatchObject({ available: false, reason: HELD })
      await a.runner.retire({ handover: true })
      expect(existsSync(path.join(a.dir, 'lock'))).toBe(true)
      // It looks while the next queue has not yet started.
      b.snap()
      await b.tick(2, 10_000)
      expect(existsSync(path.join(a.dir, 'paused.json'))).toBe(false)

      const next = await harness({ root: a.root, outputs: a.outputs, dir: a.dir, archive: a.archive, comfy: a.comfy, clock: a.clock })
      expect(next.runner.status().active).toBe(true)
      expect(next.snap().lane.held).toBeNull()
      expect(JSON.parse(readFileSync(path.join(a.dir, 'lock'), 'utf8')).pid).toBe(process.pid)
      await next.tick()
      expect(next.job(A).status).toBe('queued')
      await next.runner.retire()
      expect(existsSync(path.join(a.dir, 'lock'))).toBe(false)
      await b.runner.retire()
    } finally {
      warn.mockRestore()
    }
  })

  it('is let go at an ordinary retire, and then a server standing back notes the waiting work', async () => {
    const a = await harness()
    await a.submit(groupBody({ desk: 'images', jobs: [{}] }))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      const b = await harness({ root: a.root, outputs: a.outputs, dir: a.dir, archive: { ...a.archive, holds: () => false }, comfy: a.comfy, clock: a.clock })
      expect(existsSync(path.join(a.dir, 'paused.json'))).toBe(false)
      await a.runner.retire()
      expect(existsSync(path.join(a.dir, 'lock'))).toBe(false)
      b.snap()
      expect(existsSync(path.join(a.dir, 'paused.json'))).toBe(true)
      await b.runner.retire()
    } finally {
      warn.mockRestore()
    }
  })
})

describe('the plugin loaded again in one process', () => {
  let si: Awaited<ReturnType<typeof standIn>> | null = null
  let dirBefore: { dir: string | undefined } | null = null
  /** A folder of work of the test's own, so what one test leaves waiting is no other's. */
  const freshDir = () => {
    dirBefore ??= { dir: process.env.SWITCHGEN_RUNNER_DIR }
    const dir = path.join(mkdtempSync(path.join(os.tmpdir(), 'switchgen-runner-')), 'runner')
    process.env.SWITCHGEN_RUNNER_DIR = dir
    return dir
  }
  afterEach(async () => {
    await clearRegistry()
    endHandovers()
    await si?.close()
    si = null
    if (dirBefore) {
      if (dirBefore.dir === undefined) delete process.env.SWITCHGEN_RUNNER_DIR
      else process.env.SWITCHGEN_RUNNER_DIR = dirBefore.dir
      dirBefore = null
    }
  })

  it('takes the folder over in place, and lets it go, noting the waiting work, when the next queue does not run', async () => {
    si = await standIn()
    process.env.COMFY_URL = si.url
    process.env.SWITCHGEN_RUNNER_SETTLE_MS = '0'
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      const { switchgenRunner } = await import('../server/runner.mjs')
      const dir = freshDir()
      const lockOf = () => JSON.parse(readFileSync(path.join(dir, 'lock'), 'utf8'))
      const one = await mountPlugin(switchgenRunner())
      // Someone else's prompt keeps a heavy clip waiting over both handovers.
      si.running.push('someone-else')
      expect((await call(one, { method: 'POST', url: '/api/runner/groups', body: groupBody({ desk: 'video', jobs: [{ heavy: true }] }) })).status).toBe(200)
      const first = lockOf()
      const two = await mountPlugin(switchgenRunner())
      expect(lockOf()).toMatchObject({ pid: process.pid })
      expect(lockOf().owner).not.toBe(first.owner)
      expect((await call(two, { url: '/api/runner' })).json()).toMatchObject({ available: true, lane: { held: null } })
      expect(existsSync(path.join(dir, 'paused.json'))).toBe(false)

      // Loaded again turned off: the queue before lets the folder go, and the
      // clip still waiting is noted as waiting on a queue that is off.
      process.env.SWITCHGEN_RUNNER = 'off'
      const three = await mountPlugin(switchgenRunner())
      expect((await call(three, { url: '/api/runner' })).json()).toMatchObject({ available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.' })
      expect(existsSync(path.join(dir, 'lock'))).toBe(false)
      expect(existsSync(path.join(dir, 'paused.json'))).toBe(true)
    } finally {
      process.env.SWITCHGEN_RUNNER = 'on'
      delete process.env.SWITCHGEN_RUNNER_SETTLE_MS
      warn.mockRestore()
    }
  }, 30_000)

  it('keeps the folder held while the queue before finishes a pass ComfyUI is slow to answer', async () => {
    si = await standIn()
    process.env.COMFY_URL = si.url
    process.env.SWITCHGEN_RUNNER_SETTLE_MS = '0'
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    let release = () => {}
    try {
      const { switchgenRunner } = await import('../server/runner.mjs')
      const { archiveApi } = await import('../server/archive.mjs')
      const dir = freshDir()
      const one = await mountPlugin(switchgenRunner())
      si.running.push('someone-else')
      expect((await call(one, { method: 'POST', url: '/api/runner/groups', body: groupBody({ desk: 'video', jobs: [{ heavy: true }] }) })).status).toBe(200)
      await vi.waitFor(async () => {
        expect((await call(one, { url: '/api/runner' })).json().jobs[0].wait).toEqual({ for: 'queue', ahead: 1 })
      }, { timeout: 5000, interval: 20 })
      // Another server on the same folder, standing back for the archive.
      const b = await harness({ dir, archive: { ...archiveApi, holds: () => false } })
      expect(b.snap().available).toBe(false)
      expect(existsSync(path.join(dir, 'paused.json'))).toBe(false)

      // ComfyUI stops answering the queue's reads; the next pass waits on one.
      release = si.stall('/queue')
      const reads = si.log.filter((l) => l === 'GET /queue').length
      await vi.waitFor(() => expect(si!.log.filter((l) => l === 'GET /queue').length).toBeGreaterThan(reads), { timeout: 5000, interval: 20 })
      // Loaded again: the queue before retires, and waits for its pass.
      const two = mountPlugin(switchgenRunner())
      await vi.waitFor(async () => {
        expect((await call(one, { url: '/api/runner' })).json().available).toBe(false)
      }, { timeout: 5000, interval: 20 })
      expect(existsSync(path.join(dir, 'lock'))).toBe(true)
      b.snap()
      expect(existsSync(path.join(dir, 'paused.json'))).toBe(false)

      release()
      const mounted = await two
      expect((await call(mounted, { url: '/api/runner' })).json()).toMatchObject({ available: true, lane: { held: null } })
      expect(existsSync(path.join(dir, 'paused.json'))).toBe(false)
      await b.runner.retire()
    } finally {
      release()
      delete process.env.SWITCHGEN_RUNNER_SETTLE_MS
      warn.mockRestore()
    }
  }, 30_000)

  it('hands over: one socket, one dispatcher, and the old mount answers as the retired queue', async () => {
    // A ComfyUI slow to answer a socket's close, so a new runner that did not
    // wait for the old one's socket to close would open its own first.
    si = await standIn({ closeEchoMs: 300 })
    process.env.COMFY_URL = si.url
    process.env.SWITCHGEN_RUNNER_SETTLE_MS = '0'
    try {
      const { switchgenRunner } = await import('../server/runner.mjs')
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const one = await mountPlugin(switchgenRunner())
      // Someone else's prompt keeps a heavy clip waiting over the handover.
      si.running.push('someone-else')
      const b = groupBody({ desk: 'video', jobs: [{ heavy: true }] })
      expect((await call(one, { method: 'POST', url: '/api/runner/groups', body: b })).status).toBe(200)
      await vi.waitFor(async () => {
        const s = (await call(one, { url: '/api/runner' })).json()
        expect(s.jobs[0].wait).toEqual({ for: 'queue', ahead: 1 })
      }, { timeout: 5000, interval: 20 })

      await vi.waitFor(() => expect(si!.order).toEqual(['open 1']), { timeout: 5000, interval: 20 })
      const two = await mountPlugin(switchgenRunner())
      // ComfyUI forgets a client id when its socket closes, even one a newer
      // socket has taken since: the new socket opens only once the old has gone.
      await vi.waitFor(() => expect(si!.order).toEqual(['open 1', 'close 1', 'open 2']), { timeout: 5000, interval: 20 })
      const old = (await call(one, { url: '/api/runner' })).json()
      expect(old).toMatchObject({ available: false, reason: 'The queue on the server is not running.' })
      expect((await call(one, { method: 'POST', url: '/api/runner/groups', body: groupBody({ desk: 'images', jobs: [{}] }) })).status).toBe(503)
      const fresh = (await call(two, { url: '/api/runner' })).json()
      expect(fresh.available).toBe(true)
      expect(fresh.boot).not.toBe(old.boot)

      si.running.length = 0
      await vi.waitFor(() => expect(si!.log.filter((l) => l === 'POST /prompt')).toHaveLength(1), { timeout: 5000, interval: 20 })
      // Two dispatchers would each have sent it: give a second one its chance.
      await new Promise((resolve) => setTimeout(resolve, 1500))
      expect(si.log.filter((l) => l === 'POST /prompt')).toHaveLength(1)

      const clientId = JSON.parse(readFileSync(path.join(process.env.SWITCHGEN_RUNNER_DIR!, 'state.json'), 'utf8')).clientId
      await vi.waitFor(() => expect(si!.open()).toBe(1), { timeout: 5000, interval: 20 })
      expect(si.socketUrls.at(-1)).toBe(`/ws?clientId=${clientId}`)
      expect(JSON.parse(si.frames.at(-1)![0]!)).toEqual({ type: 'feature_flags', data: { supports_preview_metadata: true } })
      warn.mockRestore()
    } finally {
      delete process.env.SWITCHGEN_RUNNER_SETTLE_MS
    }
  }, 30_000)
})
