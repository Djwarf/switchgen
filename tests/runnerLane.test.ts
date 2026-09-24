import { readFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { HEAVY, groupBody, harness, open, revEvents, runnerEnv, type Harness } from './runnerFake'

/**
 * The queue's memory rules, against a scripted ComfyUI: a heavy job waits for
 * ComfyUI's queue to empty, releases memory, reads the queue again and only
 * then sends; one job of ours is ComfyUI's at a time; a heavy job lost holds
 * the heavy work behind it until the reader says. The runner is stepped by
 * hand and its clock is the test's, so nothing here waits in real time.
 */

let restoreEnv = () => {}
beforeAll(() => {
  restoreEnv = runnerEnv()
})
afterAll(() => restoreEnv())

const LIVE = new Set(['releasing', 'sending', 'queued', 'running'])

/**
 * Every commit the runner makes, heard on its stream, replayed in order: the
 * most jobs that were ComfyUI's business at once after any one commit.
 */
function depthWatch(h: Harness) {
  const s = open(h.runner.handler, { url: '/api/runner/stream' })
  return {
    most(): number {
      const evs = revEvents(s.reply)
      const status = new Map<string, string>()
      for (const j of evs[0]!.data.jobs) status.set(j.id, j.status)
      let most = 0
      let i = 1
      while (i < evs.length) {
        const rev = evs[i]!.data.rev
        // All the events of one commit, then the count.
        for (; i < evs.length && (evs[i]!.data.rev === rev || evs[i]!.data.rev === undefined); i++) {
          if (evs[i]!.event === 'job') status.set(evs[i]!.data.job.id, evs[i]!.data.job.status)
        }
        most = Math.max(most, [...status.values()].filter((st) => LIVE.has(st)).length)
      }
      return most
    },
    close: () => s.hangUp(),
  }
}

describe('two heavy jobs from two desks', () => {
  it('release, read again, send under the saved id; the second gets nothing until the first has ended', async () => {
    const h = await harness()
    const depth = depthWatch(h)
    const a = await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))
    expect(a.status, a.body).toBe(200)
    const b = await h.submit(groupBody({ desk: 'reel', jobs: [{ heavy: true }] }))
    expect(b.status, b.body).toBe(200)
    const A = a.json().jobs[0].id as string
    const B = b.json().jobs[0].id as string

    // The prompt goes under the id already on disk with 'sending'.
    let onDiskAtSend: any = null
    h.comfy.hooks.submit = (pid) => {
      onDiskAtSend = { pid, job: JSON.parse(readFileSync(path.join(h.dir, 'state.json'), 'utf8')).jobs[A] }
    }
    await h.tick()
    const pa = h.job(A).promptId!
    expect(h.job(A).status).toBe('queued')
    expect(h.comfy.log).toEqual(['queue', 'free', 'queue', `prompt:${pa}`, 'queue'])
    expect(onDiskAtSend).toMatchObject({ pid: pa, job: { status: 'sending', promptIdInternal: pa } })
    expect(h.job(B)).toMatchObject({ status: 'waiting', wait: { for: 'turn' } })

    // A runs: B is not released for while it does.
    h.comfy.hooks.submit = undefined
    let aWhenFreed: string[] = []
    h.comfy.hooks.free = () => {
      aWhenFreed.push(h.job(A).status)
    }
    h.comfy.run(pa)
    await h.tick(3, 2000)
    expect(h.job(A).status).toBe('running')
    expect(h.comfy.count('free')).toBe(1)
    expect(h.job(B).status).toBe('waiting')

    h.comfy.finish(pa, { files: [{ filename: 'a.webm', video: true }] })
    await h.tick(1, 2000)
    expect(h.job(A).status).toBe('done')
    const pb = h.job(B).promptId!
    expect(h.job(B).status).toBe('queued')
    expect(aWhenFreed).toEqual(['done'])
    // From A's ending on: B's release, one queue read, B's prompt.
    const tail = h.comfy.log.slice(h.comfy.log.indexOf(`history:${pa}`))
    expect(tail).toEqual([`history:${pa}`, 'queue', 'free', 'queue', `prompt:${pb}`, 'queue'])
    expect(depth.most()).toBe(1)
    depth.close()
    await h.runner.retire()
  })
})

describe('ComfyUI busy or away', () => {
  it('waits on work that is not ours with no release and no prompt, and goes once the queue is empty', async () => {
    const h = await harness()
    h.comfy.running.push('someone-else')
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await h.tick()
    expect(h.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'queue', ahead: 1 } })
    expect(h.comfy.log).toEqual(['queue'])
    // Read at most every 2 s while it waits.
    await h.tick(1, 500)
    expect(h.comfy.log).toEqual(['queue'])
    h.comfy.running.length = 0
    await h.tick(1, 2000)
    expect(h.job(A).status).toBe('queued')
    expect(h.comfy.count('free')).toBe(1)
    expect(h.comfy.prompts()).toHaveLength(1)
    await h.runner.retire()
  })

  it('waits for ComfyUI while its queue cannot be read, and sends nothing', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    h.comfy.unanswered = 3
    await h.tick(3, 2000)
    expect(h.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'comfy' } })
    expect(h.comfy.log).toEqual(['queue', 'queue', 'queue'])
    expect(h.snap().comfy.answering).toBe(false)
    await h.tick(1, 2000)
    expect(h.job(A).status).toBe('queued')
    expect(h.snap().comfy.answering).toBe(true)
    await h.runner.retire()
  })
})

describe('work that arrives while memory is released', () => {
  it('sends the heavy job back to wait, with no prompt, and releases again once the queue is empty', async () => {
    let h: Harness | null = null
    let settles = 0
    h = await harness({
      sleep: async () => {
        // Someone else's prompt reaches ComfyUI's queue during the first settle only.
        if (settles++ === 0) h!.comfy.land('someone-else')
      },
    })
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await h.tick()
    expect(h.comfy.log).toEqual(['queue', 'free', 'queue'])
    expect(h.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'queue', ahead: 1 }, promptId: null })
    await h.tick(2, 2000)
    expect(h.comfy.prompts()).toEqual([])
    h.comfy.finish('someone-else')
    await h.tick(1, 2000)
    const pid = h.job(A).promptId!
    expect(h.job(A).status).toBe('queued')
    expect(h.comfy.log.slice(-5)).toEqual(['queue', 'free', 'queue', `prompt:${pid}`, 'queue'])
    await h.runner.retire()
  })
})

describe('releases beyond the first', () => {
  it('sends one more after acceptance for work listed then, and one for each other prompt later seen running ahead', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    // Someone else's prompt slips in between the release and ours.
    h.comfy.sends.push((graph, pid) => {
      h.comfy.land('slipped-in')
      h.comfy.land(pid, graph)
      return { answer: { accepted: true }, land: false }
    })
    await h.tick()
    expect(h.job(A).status).toBe('queued')
    expect(h.comfy.count('free')).toBe(2)

    // The prompt that slipped in runs ahead of A: it was released for already.
    h.comfy.run('slipped-in')
    await h.tick(3, 2000)
    expect(h.comfy.count('free')).toBe(2)

    // Another prompt put at the front runs ahead: one release for it, however long it runs.
    h.comfy.finish('slipped-in')
    h.comfy.running.push('jumped-ahead')
    await h.tick(4, 2000)
    expect(h.comfy.count('free')).toBe(3)
    h.comfy.finish('jumped-ahead')
    h.comfy.running.push('and-another')
    await h.tick(2, 2000)
    expect(h.comfy.count('free')).toBe(4)
    expect(h.job(A).status).toBe('queued')
    expect(h.comfy.prompts()).toHaveLength(1)
    await h.runner.retire()
  })

  it('reads the queue at most every 2 s for prompts running ahead', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await h.tick()
    h.comfy.running.push('ahead')
    h.comfy.log.length = 0
    await h.tick(5, 100)
    expect(h.comfy.count('queue')).toBeLessThanOrEqual(1)
    expect(h.job(A).status).toBe('queued')
    await h.runner.retire()
  })
})

describe('a light job behind a heavy one', () => {
  it('waits while the heavy job waits for the queue, and goes only once the heavy one has ended', async () => {
    const h = await harness()
    h.comfy.running.push('someone-else')
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await h.tick()
    const P = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    await h.tick(3, 2000)
    expect(h.job(P)).toMatchObject({ status: 'waiting', wait: { for: 'heavy' } })
    expect(h.comfy.prompts()).toEqual([])

    h.comfy.running.length = 0
    await h.tick(1, 2000)
    const pa = h.job(A).promptId!
    expect(h.job(A).status).toBe('queued')
    await h.tick(3, 2000)
    // One job of ours at a time: the picture waits for the clip to end.
    expect(h.job(P).status).toBe('waiting')
    expect(h.comfy.prompts()).toEqual([pa])

    h.comfy.finish(pa, { files: [{ filename: 'a.webm', video: true }] })
    await h.tick(1, 2000)
    expect(h.job(A).status).toBe('done')
    const pp = h.job(P).promptId!
    expect(h.job(P).status).toBe('queued')
    expect(h.comfy.prompts()).toEqual([pa, pp])
    expect(h.comfy.log.indexOf(`history:${pa}`)).toBeLessThan(h.comfy.log.indexOf(`prompt:${pp}`))
    // A light job is sent at once: no release for it.
    expect(h.comfy.count('free')).toBe(1)
    await h.runner.retire()
  })
})

describe('the heavy job picked', () => {
  it('stays the pick while it waits for the queue, even over light work from a group served less lately', async () => {
    const h = await harness()
    const [A1, A2] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick()
    h.comfy.finish(h.job(A1).promptId!, { files: [{ filename: 'a1.webm', video: true }] })
    // Someone else's work arrives as the first clip ends: the second must wait for it.
    h.comfy.running.push('someone-else')
    await h.tick(1, 2000)
    expect(h.job(A1).status).toBe('done')
    expect(h.job(A2)).toMatchObject({ status: 'waiting', wait: { for: 'queue', ahead: 1 } })
    // A picture, from a batch never served, which would otherwise go first.
    const P = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    await h.tick(3, 2000)
    expect(h.job(P)).toMatchObject({ status: 'waiting', wait: { for: 'heavy' } })
    expect(h.comfy.prompts()).toHaveLength(1)
    h.comfy.running.length = 0
    await h.tick(1, 2000)
    expect(h.job(A2).status).toBe('queued')
    expect(h.job(P).status).toBe('waiting')
    await h.runner.retire()
  })
})

describe('a heavy job lost', () => {
  /** A heavy clip sent, then lost to a ComfyUI restart, with `rest` waiting behind it in the same group. */
  async function lose(rest: { heavy: boolean }[]) {
    const h = await harness()
    const ids = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, ...rest] }))).json().jobs.map((j: { id: string }) => j.id) as string[]
    await h.tick()
    expect(h.job(ids[0]!).status).toBe('queued')
    h.comfy.restart()
    // Past the grace a new prompt gets, then absent twice.
    await h.tick(3, 21_000)
    expect(h.job(ids[0]!).status).toBe('lost')
    return { h, ids }
  }

  it('holds the heavy work behind it, lets light work go, and sends the held work on the reader\'s word', async () => {
    const { h, ids } = await lose([{ heavy: true }])
    const [A, B] = ids as [string, string]
    expect(h.snap().lane.held).toMatchObject({ why: 'lost', scope: 'heavy', jobId: A })
    expect(h.job(B)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })

    // Thirty seconds of an empty queue: nothing for the held clip.
    const freesBefore = h.comfy.count('free')
    const P = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    await h.tick(15, 2000)
    expect(h.comfy.count('free')).toBe(freesBefore)
    expect(h.job(P).status).toBe('queued')
    // The lost clip's prompt and the picture's; none for the held clip.
    expect(h.comfy.prompts()).toEqual([h.job(A).promptId, h.job(P).promptId])
    h.comfy.finish(h.job(P).promptId!, { files: [{ filename: 'p.png' }] })
    await h.tick(1, 2000)
    expect(h.job(P).status).toBe('done')
    expect(h.job(B).status).toBe('waiting')

    const word = await h.post('/api/runner/lane', { action: 'send' })
    expect(word.status, word.body).toBe(200)
    expect(word.json()).toEqual({ lane: { held: null }, stopped: [] })
    await h.tick(1, 2000)
    expect(h.job(B).status).toBe('queued')
    expect(h.comfy.count('free')).toBe(freesBefore + 1)
    await h.runner.retire()
  })

  it('stops the held work on the reader\'s word, none of it ever sent', async () => {
    const { h, ids } = await lose([{ heavy: true }, { heavy: true }])
    const [, B, C] = ids as [string, string, string]
    const prompts = h.comfy.prompts().length
    const word = await h.post('/api/runner/lane', { action: 'stop' })
    expect(word.status, word.body).toBe(200)
    expect(word.json().stopped.sort()).toEqual([B, C].sort())
    await h.tick(5, 2000)
    for (const id of [B, C]) {
      expect(h.job(id)).toMatchObject({ status: 'stopped', promptId: null, error: { code: 'stopped', sent: false } })
    }
    expect(h.comfy.prompts()).toHaveLength(prompts)
    expect(h.snap().lane.held).toBeNull()
    // Nothing is held now.
    expect((await h.post('/api/runner/lane', { action: 'send' })).status).toBe(409)
    await h.runner.retire()
  })

  it('holds nothing when no heavy work is waiting', async () => {
    const { h, ids } = await lose([{ heavy: false }])
    expect(h.snap().lane.held).toBeNull()
    await h.tick(1, 2000)
    expect(h.job(ids[1]!).status).toBe('queued')
    await h.runner.retire()
  })

  it('refuses a word given to a hold that has changed since the page showed it, and changes nothing', async () => {
    const { h, ids } = await lose([{ heavy: true }, { heavy: true }])
    const [, B, C] = ids as [string, string, string]
    const held = h.snap().lane.held!
    expect(held).toMatchObject({ why: 'lost', scope: 'heavy' })
    const stale = await h.post('/api/runner/lane', { action: 'stop', since: held.since - 1 })
    expect(stale.status).toBe(409)
    expect(stale.json()).toEqual({ error: 'The hold has changed since this page showed it.' })
    expect(h.snap().lane.held).toEqual(held)
    for (const id of [B, C]) expect(h.job(id)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    // A since that is not a time is no word at all.
    expect((await h.post('/api/runner/lane', { action: 'stop', since: String(held.since) })).status).toBe(400)
    expect(h.snap().lane.held).toEqual(held)
    // The word for the hold the page showed is taken.
    const word = await h.post('/api/runner/lane', { action: 'send', since: held.since })
    expect(word.status, word.body).toBe(200)
    expect(word.json()).toEqual({ lane: { held: null }, stopped: [] })
    expect(h.snap().lane.held).toBeNull()
    await h.tick(1, 2000)
    expect(h.job(B).status).toBe('queued')
    await h.runner.retire()
  })

  it('lets the hold go by itself once nothing it covers is waiting', async () => {
    const { h, ids } = await lose([{ heavy: true }])
    expect(h.snap().lane.held).not.toBeNull()
    expect((await h.post(`/api/runner/jobs/${ids[1]}/stop`)).status).toBe(200)
    expect(h.snap().lane.held).toBeNull()
    await h.runner.retire()
  })
})

describe('a prompt ComfyUI does not list yet', () => {
  it('is given twenty seconds before it is looked for, then two absences before it is called lost', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await h.tick()
    const pid = h.job(A).promptId!
    // ComfyUI took it but does not list it yet.
    h.comfy.pending.length = 0
    await h.tick(3, 6000)
    expect(h.job(A).status).toBe('queued')
    expect(h.comfy.log).not.toContain(`history:${pid}`)
    await h.tick(1, 3000)
    expect(h.job(A).status).toBe('queued')
    await h.tick(1, 1000)
    expect(h.job(A)).toMatchObject({ status: 'lost', error: { code: 'lost', sent: true } })
    expect(h.comfy.log.filter((l) => l === `history:${pid}`)).toHaveLength(1)
    await h.runner.retire()
  })
})

describe('a stopped job taken out of the queue', () => {
  it('ends stopped, not lost, and holds nothing', async () => {
    const h = await harness()
    const [A, B] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick()
    const stop = await h.post(`/api/runner/jobs/${A}/stop`)
    expect(stop.json().job).toMatchObject({ stopRequested: true, stopLanded: true })
    await h.tick(3, 21_000)
    expect(h.job(A)).toMatchObject({ status: 'stopped', error: { code: 'stopped', sent: true } })
    expect(h.snap().lane.held).toBeNull()
    expect(h.job(B).status).toBe('queued')
    await h.runner.retire()
  })
})

describe('the least weight the server gives a job', () => {
  it('runs a two-sampler graph as heavy whatever the page said', async () => {
    const h = await harness()
    const r = await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: false, graph: HEAVY() }] }))
    expect(r.json().jobs[0].heavy).toBe(true)
    await h.tick()
    expect(h.comfy.log.slice(0, 3)).toEqual(['queue', 'free', 'queue'])
    // And a one-sampler graph the page called light stays light: sent at once.
    const P = (await h.submit(groupBody({ desk: 'images', jobs: [{ heavy: false }] }))).json().jobs[0]
    expect(P.heavy).toBe(false)
    await h.runner.retire()
  })
})
