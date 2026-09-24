import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { groupBody, harness, runnerEnv, type Harness } from './runnerFake'

/**
 * Stop, from any page on any device, at every point of a job's life. A job
 * not yet sent is never sent; one ComfyUI has is cancelled by its own prompt
 * id; one that finished before the stop landed is filed, since the file is
 * made and the archive should say so.
 */

let restoreEnv = () => {}
beforeAll(() => {
  restoreEnv = runnerEnv()
})
afterAll(() => restoreEnv())

const stop = (h: Harness, id: string, headers: Record<string, string> = {}) => h.post(`/api/runner/jobs/${id}/stop`, {}, headers)
const NEVER_SENT = { status: 'stopped', promptId: null, error: { code: 'stopped', sent: false, message: 'Stopped before it was sent.' } }

describe('a stop', () => {
  it('ends a waiting job at once, never sent', async () => {
    const h = await harness()
    const [A, B] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, {}] }))).json().jobs.map((j: { id: string }) => j.id)
    const r = await stop(h, B)
    expect(r.status).toBe(200)
    expect(r.json().job).toMatchObject(NEVER_SENT)
    await h.tick(3, 2000)
    expect(h.comfy.prompts()).toEqual([h.job(A).promptId])
    await h.runner.retire()
  })

  it('that lands while memory is released ends the job there, with no prompt', async () => {
    let h: Harness | null = null
    let A = ''
    h = await harness({ sleep: async () => void (await stop(h!, A)) })
    A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await h.tick()
    expect(h.job(A)).toMatchObject(NEVER_SENT)
    expect(h.comfy.count('free')).toBe(1)
    expect(h.comfy.prompts()).toEqual([])
    await h.runner.retire()
  })

  it('that lands while memory is released, with work arriving in ComfyUI\'s queue meanwhile, still ends the job', async () => {
    let h: Harness | null = null
    let A = ''
    h = await harness({
      sleep: async () => {
        h!.comfy.land('someone-else')
        await stop(h!, A)
      },
    })
    A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await h.tick()
    expect(h.job(A)).toMatchObject(NEVER_SENT)
    await h.tick(3, 2000)
    expect(h.comfy.prompts()).toEqual([])
    await h.runner.retire()
  })

  it('that lands before a release or a send that reached no one ends the job, never sent', async () => {
    for (const at of ['free', 'submit'] as const) {
      const h = await harness()
      const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
      if (at === 'free') h.comfy.frees.push('unreached')
      else h.comfy.sends.push(() => ({ answer: { unreached: true }, land: false }))
      h.comfy.hooks[at] = async () => void (await stop(h, A))
      await h.tick()
      h.comfy.hooks = {}
      expect(h.job(A), at).toMatchObject({ ...NEVER_SENT })
      await h.tick(3, 2000)
      expect(h.comfy.accepted, at).toEqual([])
      await h.runner.retire()
    }
  })

  it('found asked for on a job left releasing at a restart ends it, never sent', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    h.comfy.hooks.free = async () => {
      await stop(h, A)
      void h.runner.retire()
    }
    await h.tick()
    expect(h.onDisk().jobs[A]).toMatchObject({ status: 'releasing', stopRequested: true })
    h.comfy.hooks = {}
    const h2 = await h.restart()
    expect(h2.job(A)).toMatchObject(NEVER_SENT)
    await h2.tick(3, 2000)
    expect(h2.comfy.prompts()).toEqual([])
    await h2.runner.retire()
  })

  it('while the prompt is on its way cancels it the moment it is accepted', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    h.comfy.hooks.submit = async () => {
      const r = await stop(h, A)
      expect(r.json().job).toMatchObject({ status: 'sending', stopRequested: true, stopLanded: false })
    }
    await h.tick()
    const pid = h.comfy.prompts()[0]!
    expect(h.comfy.log.slice(-1)).toEqual([`cancel:${pid}`])
    expect(h.job(A)).toMatchObject({ status: 'queued', stopRequested: true, stopLanded: true })
    await h.tick(3, 21_000)
    expect(h.job(A)).toMatchObject({ status: 'stopped', promptId: pid, error: { code: 'stopped', sent: true } })
    await h.runner.retire()
  })

  it('on a send with no clear answer that ComfyUI never had ends it never sent, and holds nothing', async () => {
    const h = await harness()
    const [A, B] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    h.comfy.sends.push(() => ({ answer: { unknown: true, reason: 'ECONNRESET' }, land: false }))
    await h.tick()
    expect(h.job(A).status).toBe('sending')
    expect((await stop(h, A)).json().job.stopRequested).toBe(true)
    await h.tick(2)
    expect(h.job(A)).toMatchObject(NEVER_SENT)
    expect(h.snap().lane.held).toBeNull()
    await h.tick(1, 2000)
    expect(h.job(B).status).toBe('queued')
    await h.runner.retire()
  })

  it('on a queued job takes it out of ComfyUI\'s queue, and it ends stopped once gone', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    await h.tick()
    const pid = h.job(A).promptId!
    const r = await stop(h, A)
    expect(r.json().job).toMatchObject({ status: 'queued', stopRequested: true, stopLanded: true })
    expect(h.comfy.pending).toEqual([])
    await h.tick(3, 21_000)
    expect(h.job(A)).toMatchObject({ status: 'stopped', error: { code: 'stopped', sent: true } })
    expect(h.comfy.log).not.toContain(`interrupt:${pid}`)
    await h.runner.retire()
  })

  it('on a running job ends it stopped, from ComfyUI\'s record of the interruption', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    await h.tick()
    const pid = h.job(A).promptId!
    h.comfy.run(pid, 10_000)
    await h.tick()
    expect(h.job(A)).toMatchObject({ status: 'running', ranAt: 10_000 })
    await stop(h, A)
    await h.tick()
    expect(h.job(A)).toMatchObject({ status: 'stopped', error: { code: 'stopped', sent: true }, finishedAt: 13_000 })
    await h.runner.retire()
  })

  it('never interrupts where ComfyUI cannot cancel while the prompt only waits, since an interrupt stops whatever is sampling', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    h.comfy.cancel = async (id: string) => (h.comfy.log.push(`cancel:${id}`), false)
    h.comfy.running.push('someone-else')
    await h.tick()
    const pa = h.job(A).promptId!
    const r = await stop(h, A)
    expect(r.json().job).toMatchObject({ status: 'queued', stopRequested: true, stopLanded: false })
    expect(h.comfy.log.filter((l) => l.startsWith('interrupt'))).toEqual([])
    // Asked again once it runs, from any page: now it is this prompt that samples.
    h.comfy.running.length = 0
    h.comfy.run(pa)
    await h.tick()
    await stop(h, A)
    expect(h.comfy.log.filter((l) => l.startsWith('interrupt'))).toEqual([`interrupt:${pa}`])
    await h.runner.retire()
  })

  it('that loses the race to the ending leaves the job done and filed', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    await h.tick()
    const pid = h.job(A).promptId!
    h.comfy.run(pid)
    await h.tick()
    h.comfy.finish(pid, { files: [{ filename: 'late.webm', video: true }] })
    const r = await stop(h, A)
    expect(r.json().job).toMatchObject({ status: 'running', stopRequested: true, stopLanded: false })
    expect(h.comfy.log).not.toContain(`interrupt:${pid}`)
    await h.tick()
    expect(h.job(A)).toMatchObject({ status: 'done', entryId: A, stopRequested: true })
    await h.runner.retire()
  })

  it('changes nothing on a job already ended, and answers 404 for one the queue never had', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    await h.tick()
    h.comfy.finish(h.job(A).promptId!, { files: [{ filename: 'a.png' }] })
    await h.tick()
    const rev = h.snap().rev
    const r = await stop(h, A)
    expect(r.status).toBe(200)
    expect(r.json().job).toMatchObject({ status: 'done', stopRequested: false })
    expect(h.snap().rev).toBe(rev)
    expect((await stop(h, '5a1d7c0e-0000-4000-8000-000000000000')).status).toBe(404)
    await h.runner.retire()
  })
})

describe('interrupting a prompt ComfyUI cannot cancel', () => {
  it('sends an interrupt naming the prompt only while it runs', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    h.comfy.cancel = async (id: string) => (h.comfy.log.push(`cancel:${id}`), false)
    await h.tick()
    const pid = h.job(A).promptId!
    h.comfy.run(pid)
    await h.tick()
    await stop(h, A)
    expect(h.comfy.log.slice(-3)).toEqual([`cancel:${pid}`, `getJob:${pid}`, `interrupt:${pid}`])
    await h.tick()
    expect(h.job(A).status).toBe('stopped')
    await h.runner.retire()
  })
})

describe('a group stop', () => {
  it('stops every live job of the group, from any device, and ends the group', async () => {
    const h = await harness()
    const g = (await h.submit(groupBody({ desk: 'images', jobs: [{}, {}, {}], device: 'phone' }))).json()
    const [Q1, Q2, Q3] = g.jobs.map((j: { id: string }) => j.id)
    await h.tick()
    const p1 = h.job(Q1).promptId!
    h.comfy.run(p1)
    await h.tick()
    const r = await h.post(`/api/runner/groups/${g.group.id}/stop`, {}, { 'x-switchgen-device': 'laptop' })
    expect(r.status, r.body).toBe(200)
    expect(r.json().group).toMatchObject({ state: 'ended', endedBy: { jobId: Q1, why: 'stopped' } })
    expect(r.json().jobs.map((j: { id: string }) => j.id)).toEqual([Q1, Q2, Q3])
    for (const id of [Q2, Q3]) expect(h.job(id)).toMatchObject(NEVER_SENT)
    expect(h.job(Q1)).toMatchObject({ stopRequested: true, stopLanded: true })
    await h.tick()
    expect(h.job(Q1)).toMatchObject({ status: 'stopped', error: { sent: true } })
    expect(h.comfy.prompts()).toEqual([p1])
    await h.runner.retire()
  })

  it('stops a job in the middle of its release too, and it is never sent', async () => {
    let h: Harness | null = null
    let gid = ''
    h = await harness({ sleep: async () => void (await h!.post(`/api/runner/groups/${gid}/stop`, {})) })
    const g = (await h.submit(groupBody({ desk: 'reel', jobs: [{ heavy: true }, { heavy: true }] }))).json()
    gid = g.group.id
    await h.tick(3, 2000)
    for (const j of g.jobs) expect(h.job(j.id)).toMatchObject(NEVER_SENT)
    expect(h.comfy.prompts()).toEqual([])
    expect(h.group(gid)).toMatchObject({ state: 'ended', endedBy: { jobId: g.jobs[0].id, why: 'stopped' } })
    await h.runner.retire()
  })

  it('answers 404 for a group the queue never had', async () => {
    const h = await harness()
    expect((await h.post('/api/runner/groups/5a1d7c0e-0000-4000-8000-000000000000/stop', {})).status).toBe(404)
    await h.runner.retire()
  })
})

describe('a stop ComfyUI did not take', () => {
  /** A cancel that says no, as comfy.mjs does on any network error, timeout or refusal, `times` times. */
  const refuseCancel = (h: Harness, times: number) => {
    const real = h.comfy.cancel.bind(h.comfy)
    let left = times
    h.comfy.cancel = async (id: string) => {
      if (left-- > 0) {
        h.comfy.log.push(`cancel:${id}`)
        return false
      }
      return real(id)
    }
  }

  it('is asked again while the job runs, and the job ends stopped, not done and filed', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    await h.tick()
    h.comfy.run(h.job(A).promptId!, 10_000)
    await h.tick()
    refuseCancel(h, 1)
    // ...and the read that would have found the job ended goes unanswered.
    h.comfy.unanswered = 1
    const r = await stop(h, A)
    expect(r.json().job).toMatchObject({ status: 'running', stopRequested: true, stopLanded: false })
    expect(h.comfy.count('cancel')).toBe(1)
    await h.tick(1, 2000)
    expect(h.comfy.count('cancel')).toBe(2)
    await h.tick(2, 2000)
    expect(h.job(A)).toMatchObject({ status: 'stopped', error: { code: 'stopped', sent: true }, entryId: null })
    await h.runner.retire()
  })

  it('is asked again by the next runner after a restart, while the prompt still waits in ComfyUI\'s queue', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    h.comfy.running.push('someone-else')
    await h.tick()
    const pid = h.job(A).promptId!
    expect(h.comfy.pending).toContain(pid)
    refuseCancel(h, 1)
    await stop(h, A)
    expect(h.job(A)).toMatchObject({ status: 'queued', stopRequested: true, stopLanded: false })
    const h2 = await h.restart()
    await h2.tick()
    expect(h2.job(A)).toMatchObject({ stopLanded: true })
    expect(h2.comfy.pending).not.toContain(pid)
    await h2.tick(3, 21_000)
    expect(h2.job(A).status).toBe('stopped')
    await h2.runner.retire()
  })
})

describe('a stop whose answer from ComfyUI was lost', () => {
  it('on a queued heavy clip that ComfyUI took out of its queue ends it stopped, and holds nothing', async () => {
    const h = await harness()
    const [A, B] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick(1, 2000)
    const pid = h.job(A).promptId!
    expect(h.job(A).status).toBe('queued')
    // ComfyUI, stalled, takes the cancel, but its answer never comes back:
    // comfy.mjs says no on a timeout.
    const real = h.comfy.cancel.bind(h.comfy)
    h.comfy.cancel = async (id: string) => {
      await real(id)
      return false
    }
    const r = await stop(h, A)
    expect(r.json().job).toMatchObject({ status: 'queued', stopRequested: true, stopLanded: false })
    expect(h.comfy.pending).not.toContain(pid)
    // Past the grace a prompt gets, then gone from the queue and from the
    // history, as a prompt taken out of the queue is.
    await h.tick(4, 21_000)
    expect(h.job(A)).toMatchObject({ status: 'stopped', error: { code: 'stopped', sent: true } })
    expect(h.snap().lane.held).toBeNull()
    await h.tick(1, 2000)
    expect(h.job(B).status).not.toBe('waiting')
    await h.runner.retire()
  })
})

describe('a stop while ComfyUI does not answer', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  /** A cancel ComfyUI never answers; `reached` settles once the runner has asked. */
  const stall = (h: Harness) => {
    let reached = () => {}
    const asked = new Promise<void>((resolve) => { reached = resolve })
    h.comfy.cancel = () => {
      reached()
      return new Promise<boolean>(() => {})
    }
    return asked
  }

  for (const what of ['job', 'group'] as const) {
    it(`of a ${what} answers the page after 3 s, well before the page gives up at 15 s, with the stop asked for and not yet landed`, async () => {
      const h = await harness()
      const g = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json()
      const A = g.jobs[0].id as string
      await h.tick()
      expect(h.job(A).status).toBe('queued')
      const asked = stall(h)
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
      let answer: Awaited<ReturnType<Harness['post']>> | null = null
      void h.post(what === 'job' ? `/api/runner/jobs/${A}/stop` : `/api/runner/groups/${g.group.id}/stop`, {}).then((r) => {
        answer = r
      })
      await asked
      await vi.advanceTimersByTimeAsync(2900)
      expect(answer).toBeNull()
      await vi.advanceTimersByTimeAsync(100)
      expect(answer).not.toBeNull()
      const r = answer!
      expect(r.status, r.body).toBe(200)
      const view = what === 'job' ? r.json().job : r.json().jobs[0]
      expect(view).toMatchObject({ id: A, status: 'queued', stopRequested: true, stopLanded: false })
      vi.useRealTimers()
      await h.runner.retire()
    })
  }

  it('waits no longer than the runner is told to', async () => {
    const h = await harness({ stopWaitMs: 50 })
    const g = (await h.submit(groupBody({ desk: 'images', jobs: [{}, {}] }))).json()
    await h.tick()
    const asked = stall(h)
    // Well under the 3 s the runner waits unless told otherwise.
    const late = new Promise<'late'>((resolve) => setTimeout(() => resolve('late'), 2000).unref())
    const job = await Promise.race([stop(h, g.jobs[0].id), late])
    await asked
    expect(job).not.toBe('late')
    const group = await Promise.race([h.post(`/api/runner/groups/${g.group.id}/stop`, {}), late])
    expect(group).not.toBe('late')
    if (group === 'late') return
    expect(group.status).toBe(200)
    expect(group.json().jobs.map((j: { status: string; stopRequested: boolean }) => [j.status, j.stopRequested])).toEqual([
      ['queued', true],
      ['stopped', false],
    ])
    await h.runner.retire()
  })
})
