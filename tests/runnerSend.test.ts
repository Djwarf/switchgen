import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { Unanswered } from '../server/runner/comfy.mjs'
import { groupBody, harness, runnerEnv } from './runnerFake'

/**
 * A send is sorted by what is known about it, because the queue must never
 * send a job twice. Refused: the job fails with ComfyUI's words. Never
 * reached: it waits and goes again under a new id. Anything unclear: ComfyUI
 * is asked about it by its id, and it is never sent again.
 */

let restoreEnv = () => {}
beforeAll(() => {
  restoreEnv = runnerEnv()
})
afterAll(() => restoreEnv())

describe('a send that never reached ComfyUI', () => {
  it('goes back to waiting and is sent again under a new id, releasing again first', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    h.comfy.sends.push(() => ({ answer: { unreached: true }, land: false }))
    await h.tick()
    expect(h.job(A)).toMatchObject({ status: 'waiting', wait: { for: 'comfy' }, promptId: null, attempt: 1 })
    expect(h.onDisk().jobs[A].promptIdInternal).toBeNull()
    await h.tick(1, 2000)
    const [first, second] = h.comfy.prompts()
    expect(second).toBeDefined()
    expect(second).not.toBe(first)
    expect(h.job(A)).toMatchObject({ status: 'queued', promptId: second, attempt: 2 })
    expect(h.comfy.accepted.map((a) => a.id)).toEqual([second])
    expect(h.comfy.count('free')).toBe(2)
    await h.runner.retire()
  })

  it('tries a light job again at the pace the queue is read, not on every pass', async () => {
    const h = await harness()
    const P = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    h.comfy.sends.push(() => ({ answer: { unreached: true }, land: false }))
    await h.tick()
    expect(h.job(P)).toMatchObject({ status: 'waiting', wait: { for: 'comfy' } })
    await h.tick(3, 500)
    expect(h.comfy.prompts()).toHaveLength(1)
    await h.tick(1, 600)
    expect(h.comfy.prompts()).toHaveLength(2)
    expect(h.job(P).status).toBe('queued')
    await h.runner.retire()
  })
})

describe('a send with no clear answer', () => {
  it('is found by its id and taken as queued, with no second prompt', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    h.comfy.sends.push(() => ({ answer: { unknown: true, reason: 'ECONNRESET' }, land: true }))
    await h.tick()
    expect(h.job(A)).toMatchObject({ status: 'sending', promptId: null })
    await h.tick()
    const [pid] = h.comfy.prompts()
    expect(h.job(A)).toMatchObject({ status: 'queued', promptId: pid })
    expect(h.comfy.prompts()).toHaveLength(1)
    expect(h.comfy.log).toContain(`getJob:${pid}`)
    await h.runner.retire()
  })

  it('is called unsent when ComfyUI answers twice that it has no such job and keeps no record, and holds the heavy work behind it', async () => {
    const h = await harness()
    const [A, B] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    h.comfy.sends.push(() => ({ answer: { unknown: true, reason: 'no answer within 30 s' }, land: false }))
    await h.tick()
    await h.tick()
    expect(h.job(A).status).toBe('sending')
    await h.tick()
    expect(h.job(A)).toMatchObject({ status: 'unsent', promptId: null, error: { code: 'unsent', sent: false } })
    const [pid] = h.comfy.prompts()
    expect(h.comfy.log.filter((l) => l === `history:${pid}`)).toHaveLength(1)
    expect(h.snap().lane.held).toMatchObject({ why: 'unsent', scope: 'heavy', jobId: A })
    await h.tick(10, 2000)
    expect(h.job(B)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect(h.comfy.prompts()).toHaveLength(1)
    await h.runner.retire()
  })

  it('keeps asking while ComfyUI does not answer, and never reads an outage as no record', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: false }] }))).json().jobs[0].id
    h.comfy.sends.push(() => ({ answer: { unknown: true, reason: 'HTTP 502' }, land: false }))
    await h.tick()
    h.comfy.unanswered = 3
    await h.tick(3)
    expect(h.job(A)).toMatchObject({ status: 'sending', wait: { for: 'comfy' } })
    // Absent once, then the history read itself goes unanswered: still asking.
    await h.tick()
    expect(h.job(A).status).toBe('sending')
    h.comfy.hooks.history = () => {
      h.comfy.hooks.history = undefined
      throw new Unanswered('ComfyUI did not answer /history: restarting')
    }
    await h.tick()
    expect(h.job(A)).toMatchObject({ status: 'sending', wait: { for: 'comfy' } })
    await h.tick()
    expect(h.job(A).status).toBe('unsent')
    expect(h.comfy.prompts()).toHaveLength(1)
    await h.runner.retire()
  })

  it('found in ComfyUI\'s history after all, is followed to its ending and filed', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    h.comfy.sends.push((graph, pid) => {
      // It ran and ended before the answer came back, which was lost.
      h.comfy.accepted.push({ id: pid, graph })
      h.comfy.finish(pid, { files: [{ filename: 'fast.webm', video: true }] })
      return { answer: { unknown: true, reason: 'ECONNRESET' }, land: false }
    })
    // A ComfyUI with no jobs API answers 404 for every id; /history still has it.
    const getJob = h.comfy.getJob.bind(h.comfy)
    h.comfy.getJob = async (id: string) => (await getJob(id), null)
    await h.tick()
    const [pid] = h.comfy.prompts()
    expect(h.job(A).status).toBe('sending')
    await h.tick(2)
    expect(h.job(A)).toMatchObject({ status: 'done', entryId: A, promptId: pid })
    expect(h.comfy.prompts()).toHaveLength(1)
    await h.runner.retire()
  })
})

describe('an ending ComfyUI has no record of', () => {
  it('is read three times before it is called ended without its result, which may exist', async () => {
    const h = await harness()
    const A = (await h.submit(groupBody({ desk: 'video', jobs: [{}] }))).json().jobs[0].id
    await h.tick()
    const pid = h.job(A).promptId!
    h.comfy.finish(pid, { files: [{ filename: 'a.webm', video: true }] })
    // Ended, as the jobs API says, with a history entry that says nothing yet.
    const entry = h.comfy.records.get(pid) as any
    entry.status = { status_str: null, completed: false, messages: [] }
    await h.tick(2)
    expect(h.job(A).status).toBe('queued')
    await h.tick()
    expect(h.job(A)).toMatchObject({ status: 'failed', error: { code: 'ended-unsent', mayExist: true, sent: true } })
    expect(h.comfy.log.filter((l) => l === `history:${pid}`)).toHaveLength(3)
    await h.runner.retire()
  })
})

describe('a send ComfyUI refused', () => {
  it('fails with ComfyUI\'s words and node, holds nothing, and is never sent again', async () => {
    const h = await harness()
    const [A, B] = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }, { heavy: true }] }))).json().jobs.map((j: { id: string }) => j.id)
    const nodeErrors = { '5': { errors: [{ message: 'Value not in list' }], class_type: 'LoadImage' } }
    h.comfy.sends.push(() => ({
      answer: { refused: true, status: 400, message: 'Prompt outputs failed validation', node: '5', nodeType: 'LoadImage', nodeErrors },
      land: false,
    }))
    await h.tick()
    expect(h.job(A)).toMatchObject({
      status: 'failed',
      promptId: null,
      error: { code: 'refused', message: 'Prompt outputs failed validation', node: '5', nodeType: 'LoadImage', nodeErrors, sent: false },
    })
    expect(h.snap().lane.held).toBeNull()
    await h.tick(1, 2000)
    expect(h.job(B).status).toBe('queued')
    expect(h.comfy.prompts()).toHaveLength(2)
    await h.runner.retire()
  })
})

describe('turns between groups', () => {
  it('serves the group served least recently first, so a picture goes after the shot on the press, not the whole reel', async () => {
    const h = await harness()
    const reel = (await h.submit(groupBody({ desk: 'reel', jobs: [{}, {}, {}] }))).json().jobs.map((j: { id: string }) => j.id)
    await h.tick()
    expect(h.job(reel[0]).status).toBe('queued')
    const P = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id
    h.comfy.finish(h.job(reel[0]).promptId!, { files: [{ filename: 's1.webm', video: true }] })
    await h.tick()
    expect(h.job(reel[0]).status).toBe('done')
    expect(h.job(P).status).toBe('queued')
    expect(h.job(reel[1])).toMatchObject({ status: 'waiting' })
    await h.runner.retire()
  })
})
