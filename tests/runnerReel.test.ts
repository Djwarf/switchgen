import { randomUUID } from 'node:crypto'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { CHAINED, CHAIN_TOKEN, groupBody, harness, runnerEnv, type Finish, type Harness } from './runnerFake'

/**
 * A reel's pass on the server. Each shot is sent only once the one before it
 * has landed; a shot that opens on the last frame of the one before is handed
 * over with a placeholder where that frame goes, and the server puts the real
 * frame in, from the shot's own handoff tap, before anything is released or
 * sent for it.
 */

let restoreEnv = () => {}
beforeAll(() => {
  restoreEnv = runnerEnv()
})
afterAll(() => restoreEnv())

/** A pass of `n` heavy shots, each after the first opening on the one before. */
function chainedPass(n: number, shotIds: string[] = Array.from({ length: n }, () => randomUUID())) {
  const ids = Array.from({ length: n }, () => randomUUID())
  return {
    ids,
    shotIds,
    body: groupBody({
      desk: 'reel',
      jobs: ids.map((id, i) => ({
        id,
        heavy: true,
        shotId: shotIds[i],
        ...(i > 0 ? { graph: CHAINED(), chain: { after: ids[i - 1]!, at: [['20', 'image']] as [string, string][] } } : {}),
      })),
    }),
  }
}

const sentGraph = (h: Harness, jobId: string) => h.comfy.accepted.find((a) => a.id === h.job(jobId).promptId)!.graph

describe('a chained pass', () => {
  it('sends each shot after the one before has landed, opening on that shot\'s tapped last frame', async () => {
    const h = await harness()
    const { ids, body } = chainedPass(3)
    const [J1, J2, J3] = ids as [string, string, string]
    expect((await h.submit(body)).status).toBe(200)
    await h.tick()
    expect(h.job(J1).status).toBe('queued')
    expect(h.job(J2)).toMatchObject({ status: 'waiting', wait: { for: 'before' }, openedOn: null })
    expect(h.comfy.prompts()).toHaveLength(1)

    // Shot 1 lands with a poster still of its own besides the tap.
    h.comfy.finish(h.job(J1).promptId!, {
      files: [{ filename: 'shot1.webm', subfolder: 'reel', video: true }, { filename: 'poster.png', subfolder: 'reel' }],
      frame: { filename: 'shot1.frame_00001_.png', subfolder: 'reel' },
    })
    let atSend: any = null
    h.comfy.hooks.submit = () => {
      atSend ??= JSON.parse(readFileSync(path.join(h.dir, 'state.json'), 'utf8')).jobs[J2]
    }
    await h.tick(1, 2000)
    const ref = 'reel/shot1.frame_00001_.png [output]'
    expect(h.job(J1)).toMatchObject({ status: 'done', frame: { filename: 'shot1.frame_00001_.png', subfolder: 'reel' } })
    expect(h.job(J2)).toMatchObject({ status: 'queued', openedOn: ref })
    // The frame it opens on was on disk before its prompt went.
    expect(atSend).toMatchObject({ status: 'sending', openedOn: ref })
    const g2 = sentGraph(h, J2)
    expect(g2['20'].inputs.image).toBe(ref)
    expect(JSON.stringify(g2)).not.toContain(CHAIN_TOKEN)
    // The saved graph is never rewritten.
    const saved = JSON.parse(readFileSync(path.join(h.dir, 'jobs', `${J2}.json`), 'utf8'))
    expect(saved.graph['20'].inputs.image).toBe(CHAIN_TOKEN)

    h.comfy.finish(h.job(J2).promptId!, {
      files: [{ filename: 'shot2.webm', subfolder: 'reel', video: true }],
      frame: { filename: 'shot2.frame_00001_.png', subfolder: 'reel' },
    })
    await h.tick(1, 2000)
    expect(sentGraph(h, J3)['20'].inputs.image).toBe('reel/shot2.frame_00001_.png [output]')
    // Each shot's prompt went only after the one before had ended.
    const log = h.comfy.log
    expect(log.indexOf(`history:${h.job(J1).promptId}`)).toBeLessThan(log.indexOf(`prompt:${h.job(J2).promptId}`))
    expect(log.indexOf(`history:${h.job(J2).promptId}`)).toBeLessThan(log.indexOf(`prompt:${h.job(J3).promptId}`))
    await h.runner.retire()
  })

  it('opens on a frame named like the tap when a graph has no tap node', async () => {
    const h = await harness()
    const { ids, body } = chainedPass(2)
    await h.submit(body)
    await h.tick()
    h.comfy.finish(h.job(ids[0]!).promptId!, {
      files: [
        { filename: 'shot1.webm', subfolder: 'reel', video: true },
        { filename: 'chain_00001_.png', subfolder: 'reel' },
        { filename: 'late.png', subfolder: 'reel' },
      ],
    })
    await h.tick(1, 2000)
    expect(h.job(ids[1]!)).toMatchObject({ status: 'queued', openedOn: 'reel/chain_00001_.png [output]' })
    await h.runner.retire()
  })

  it('opens on the frame of a shot answered from the cache, done as a repeat', async () => {
    const h = await harness()
    // The same shot made once before, and filed.
    const before = (await h.submit(groupBody({ desk: 'reel', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await h.tick()
    const files = { files: [{ filename: 'shot1.webm', subfolder: 'reel', video: true, node: '9' }], frame: { filename: 'shot1.frame.png', subfolder: 'reel' } }
    h.comfy.finish(h.job(before).promptId!, files)
    await h.tick(1, 2000)
    expect(h.job(before).status).toBe('done')
    const { ids, body } = chainedPass(2)
    await h.submit(body)
    await h.tick(1, 2000)
    h.comfy.finish(h.job(ids[0]!).promptId!, { ...files, cachedNodes: ['9', '__cont_frame'] })
    await h.tick(1, 2000)
    expect(h.job(ids[0]!)).toMatchObject({ status: 'done', repeatOf: before, frame: { filename: 'shot1.frame.png', cached: true } })
    expect(h.job(ids[1]!)).toMatchObject({ status: 'queued', openedOn: 'reel/shot1.frame.png [output]' })
    await h.runner.retire()
  })
})

describe('a shot with nothing to open on', () => {
  const cases: [string, Finish][] = [
    ['wrote no still at all', { files: [{ filename: 'shot1.webm', subfolder: 'reel', video: true }] }],
    ['wrote a still that is not on disk', { files: [{ filename: 'shot1.webm', subfolder: 'reel', video: true }], frame: { filename: 'gone.png', subfolder: 'reel', write: false } }],
  ]
  for (const [what, finish] of cases) {
    it(`fails no-frame when the shot before ${what}, and the rest of the pass is skipped, never sent`, async () => {
      const h = await harness()
      const { ids, body } = chainedPass(3)
      const [J1, J2, J3] = ids as [string, string, string]
      await h.submit(body)
      await h.tick()
      h.comfy.finish(h.job(J1).promptId!, finish)
      await h.tick(3, 2000)
      expect(h.job(J1).status).toBe('done')
      expect(h.job(J2)).toMatchObject({ status: 'failed', promptId: null, openedOn: null, error: { code: 'no-frame', sent: false } })
      expect(h.job(J3)).toMatchObject({ status: 'skipped', promptId: null, error: { code: 'skipped', after: { jobId: J2, index: 2 } } })
      expect(h.group(body.group.id)).toMatchObject({ state: 'ended', endedBy: { jobId: J2, why: 'no-frame' } })
      expect(h.comfy.prompts()).toHaveLength(1)
      expect(h.comfy.count('free')).toBe(1)
      await h.runner.retire()
    })
  }

  it('is never opened on a shot that did not finish: the pass ends with that shot', async () => {
    const h = await harness()
    const { ids, body } = chainedPass(2)
    await h.submit(body)
    await h.tick()
    h.comfy.finish(h.job(ids[0]!).promptId!, { error: 'Allocation on device' })
    await h.tick(2, 2000)
    expect(h.job(ids[0]!).status).toBe('failed')
    expect(h.job(ids[1]!)).toMatchObject({ status: 'skipped', error: { after: { jobId: ids[0], index: 1 } } })
    expect(h.group(body.group.id)?.endedBy).toEqual({ jobId: ids[0], why: 'failed' })
    await h.runner.retire()
  })
})

describe('one lane for the reel and the Video desk', () => {
  it('sends their heavy work one at a time, and a lost reel shot holds a waiting clip', async () => {
    const h = await harness()
    const { ids, body } = chainedPass(2)
    await h.submit(body)
    const clip = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id
    await h.tick()
    expect(h.job(ids[0]!).status).toBe('queued')
    await h.tick(3, 2000)
    expect(h.job(clip).status).toBe('waiting')
    h.comfy.restart()
    await h.tick(3, 21_000)
    expect(h.job(ids[0]!).status).toBe('lost')
    expect(h.snap().lane.held).toMatchObject({ why: 'lost', scope: 'heavy', jobId: ids[0] })
    await h.tick(5, 2000)
    expect(h.job(clip)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect(h.comfy.prompts()).toHaveLength(1)
    await h.runner.retire()
  })
})

describe('two passes over the same shots', () => {
  it('refuses a pass that shares a shot with one still going, and takes it once that one has ended', async () => {
    const h = await harness()
    const shared = randomUUID()
    const first = chainedPass(2, [randomUUID(), shared])
    expect((await h.submit(first.body)).status).toBe(200)
    // The same request again is the same pass.
    const again = await h.submit(first.body)
    expect(again.status).toBe(200)
    expect(again.json().replayed).toBe(true)
    const overlap = chainedPass(1, [shared])
    const r = await h.submit(overlap.body)
    expect(r.status).toBe(409)
    expect(r.json()).toEqual({
      error: 'A reel is already being rendered, from this page or another. Stop it there or wait for it to finish.',
      busy: 'reel',
    })
    expect((await h.submit(chainedPass(1).body)).status).toBe(200)
    await h.post(`/api/runner/groups/${first.body.group.id}/stop`)
    expect((await h.submit(overlap.body)).status).toBe(200)
    await h.runner.retire()
  })
})
