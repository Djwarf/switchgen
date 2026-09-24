import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Engine from '../src/components/reel/engine'

/**
 * The reel with the queue on the server: a pass is handed over whole, each
 * shot that opens on the one before carrying the frame's place, and the
 * walk only watches while the server renders. The queue's library is stood
 * in by a store the test sets and by mocks for the hand-over, stop and
 * dismiss; ComfyUI's run() is mocked, and a pass the server takes must never
 * call it. The in-page walk, the fallback, runs against the same mock.
 */

const h = vi.hoisted(() => {
  let snap: any = null
  const listeners = new Set<() => void>()
  const fresh = () => ({ v: 1, available: true, reason: null, boot: 'b', rev: 0, comfy: { answering: true, since: 0 }, lane: { held: null }, groups: [], jobs: [], progress: {}, connected: true })
  snap = fresh()
  const h = {
    get snap() {
      return snap
    },
    reset() {
      snap = fresh()
      listeners.clear()
    },
    set(next: any) {
      snap = { ...snap, ...next }
      for (const f of [...listeners]) f()
    },
    /** What a read of the state brings, when a test says; by default it brings nothing new. */
    onRefresh: null as null | (() => void | Promise<void>),
    store: {
      subscribe: (fn: () => void) => {
        listeners.add(fn)
        return () => listeners.delete(fn)
      },
      snapshot: () => snap,
      start: () => {},
      refresh: async () => {
        await h.onRefresh?.()
      },
    },
    fns: {} as Record<string, any>,
    givenUp: [] as any[],
    /** Hand-overs of this tab still waiting for the server's answer. */
    pending: [] as any[],
  }
  return h
})

vi.mock('../src/lib/runner', async (orig) => ({
  ...(await orig<any>()),
  runnerStore: h.store,
  runnerAvailable: (...a: any[]) => h.fns.available(...a),
  submitGroup: (...a: any[]) => h.fns.submit(...a),
  stopGroup: (...a: any[]) => h.fns.stop(...a),
  dismiss: (...a: any[]) => h.fns.dismiss(...a),
  withdraw: (...a: any[]) => h.fns.withdraw(...a),
  givenUpBatches: () => h.givenUp,
  outboxPending: () => h.pending,
  forgetGivenUp: (id: string) => {
    h.givenUp = h.givenUp.filter((g) => g.groupId !== id)
  },
}))
vi.mock('../src/lib/comfy', async (orig) => ({ ...(await orig<any>()), run: (...a: any[]) => h.fns.run(...a) }))

let engine: typeof Engine
let cont: any
let wf: any
let session: any
let splice: any

const fam = (id: string) => wf.FAMILIES.find((f: any) => f.id === id)!
function plan(prompts: string[], id = 'wan22-5b') {
  const def = fam(id)
  const model = def.dualModel ? '' : def.models[0]
  const d = wf.defaultsFor(def, model)
  const params = { model, positive: '', negative: d.negative ?? '', seed: 7, steps: d.steps, cfg: d.cfg, width: d.width, height: d.height, sampler: d.sampler, scheduler: d.scheduler, length: 17, fps: d.fps || 24 }
  return cont.shotPlan({ base: def, params, shots: prompts.map((prompt) => ({ prompt })), freshSeeds: false }).jobs
}
const ctx = (refuse = -1, release = true) => ({
  familyLabel: 'Fam',
  modelLabel: 'Mod',
  compositionFor: (job: any) => session.newComposition('video', { mode: 't2v', familyId: 'wan22-5b', model: job.params.model, prompt: job.params.positive }),
  memory: (job: any) => (job.index === refuse ? { level: 'refuse' as const, reason: 'Too big.', release: false } : { level: 'ok' as const, reason: null, release }),
})

function view(body: any, i: number, patch: any = {}) {
  const j = body.jobs[i]
  return {
    id: j.id, groupId: body.group.id, desk: 'reel', kind: 'video', seq: i, index: i + 1, total: body.jobs.length, label: j.label, prompt: j.prompt, device: 'd', heavy: j.heavy,
    status: 'waiting', wait: null, stopRequested: false, stopLanded: false, promptId: null, attempt: 0, createdAt: 1000, sentAt: null, ranAt: null, finishedAt: null, endedAt: null,
    files: [], primary: null, frame: null, openedOn: null, entryId: null, entryNo: null, repeatOf: null, durationMs: 0, error: null, meta: j.meta, dismissed: false,
    ...patch,
  }
}
function groupOf(body: any, patch: any = {}) {
  return { id: body.group.id, desk: 'reel', kind: 'pass', label: body.group.label ?? 'L', device: 'd', createdAt: 1000, state: 'active', endedBy: null, endedAt: null, jobIds: body.jobs.map((j: any) => j.id), dismissed: false, ...patch }
}
const clip = (n: number) => ({ filename: `reel_${n}_.webm`, subfolder: 'reel', type: 'output', kind: 'video' })
const frame = (n: number) => ({ filename: `reel_${n}.frame_.png`, subfolder: 'reel', type: 'output', kind: 'image' })
const err = (code: string, sent = true, message: string | null = null) => ({ code, message, node: null, nodeType: null, nodeErrors: null, mayExist: false, sent, after: null })
const taken = (store: { body?: any }) => async (b: any) => {
  store.body = b
  return { ok: true, replayed: false, group: groupOf(b), jobs: b.jobs.map((_: any, i: number) => view(b, i)) }
}

beforeEach(async () => {
  vi.resetModules()
  h.reset()
  h.givenUp = []
  h.pending = []
  h.onRefresh = null
  // As lib/runner's: where the store says the queue is off it reads the state afresh, and answers from that.
  h.fns.available = vi.fn(async () => {
    if (h.snap.boot && !h.snap.available) await h.store.refresh()
    return h.snap.boot && !h.snap.available ? { ok: false, reason: h.snap.reason } : { ok: true, reason: null }
  })
  h.fns.submit = vi.fn()
  h.fns.stop = vi.fn(async () => true)
  h.fns.dismiss = vi.fn(async () => true)
  h.fns.withdraw = vi.fn()
  h.fns.run = vi.fn(() => new Promise(() => {}))
  session = await import('../src/lib/session')
  cont = await import('../src/lib/continuation')
  wf = await import('../src/lib/workflows')
  splice = (await import('../server/runner/comfyRecord.mjs')).splice
  engine = await import('../src/components/reel/engine')
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

/** The desk's notice of this strip's held shots, as drawn. */
const heldNotice = (html: string) => /<div class="notice notice-warning[^"]*"><strong>(?:Shots|A shot) held back\.<\/strong>[^]*?<\/div>/.exec(html)?.[0] ?? ''

describe('a pass the server renders', () => {
  it('hands over one pass: chained shots carry the token and chain.after, nothing goes in page', async () => {
    const jobs = plan(['a', 'b', 'c'])
    let body: any
    h.fns.submit.mockImplementation(async (b: any) => { body = b; return { ok: true, replayed: false, group: groupOf(b), jobs: b.jobs.map((_: any, i: number) => view(b, i)) } })
    engine.reelRun.renderAll(['s1', 's2', 's3'], jobs, ctx() as any)
    expect(engine.reelRun.snapshot().status).toBe('running')
    expect(engine.reelRun.snapshot().runnerGroupId).toBeTruthy()
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalledTimes(1))
    expect(h.fns.run).not.toHaveBeenCalled()
    expect(body.group).toMatchObject({ desk: 'reel', kind: 'pass' })
    expect(body.jobs).toHaveLength(3)
    expect(body.jobs[0].chain).toBeUndefined()
    expect(body.jobs[1].chain.after).toBe(body.jobs[0].id)
    expect(body.jobs[2].chain.after).toBe(body.jobs[1].id)
    expect(body.jobs[1].chain.at.length).toBeGreaterThan(0)
    for (const [n, input] of body.jobs[1].chain.at) expect(body.jobs[1].graph[n].inputs[input]).toBe('switchgen:chain:previous-frame')
    // equivalence with a direct build
    const ref = cont.annotatedRef(frame(1))
    expect(splice(body.jobs[1].graph, body.jobs[1].chain.at, ref)).toEqual(cont.instantiateShot(jobs[1], frame(1)))
    expect(body.jobs[0]).toMatchObject({ kind: 'video', primary: 'video', orFirst: false, noFile: 'done', heavy: true, label: 'Shot 1 of 3' })
    expect(body.jobs[1].meta).toMatchObject({ shotId: 's2', index: 1, chained: true, made: { openedOn: null } })
    expect(body.jobs[0].record.desk).toBe('video')
    expect(body.jobs[0].record).not.toHaveProperty('file')
    expect(engine.reelRun.snapshot().runnerGroupId).toBe(body.group.id)
    expect(engine.waitingInPage(engine.reelRun.snapshot())).toBe(0)
    expect(engine.waitingOnServer(engine.reelRun.snapshot())).toBe(3)

    // shot 1 runs
    h.set({ groups: [groupOf(body)], jobs: [view(body, 0, { status: 'running', promptId: 'p1', sentAt: 1100 }), view(body, 1, { wait: { for: 'before' } }), view(body, 2, { wait: { for: 'before' } })], progress: { [body.jobs[0].id]: { id: body.jobs[0].id, value: 3, max: 10, node: '3', classType: 'KSampler', pass: null, at: 1, previewN: 2 } } })
    let st = engine.reelRun.snapshot()
    expect(st.states.s1).toMatchObject({ status: 'running', stage: 'Drawing', value: 3, max: 10, promptId: 'p1' })
    expect(st.states.s1.previewUrl).toContain('/preview?n=2')
    expect(st.states.s2.status).toBe('waiting')
    expect(st.currentShotId).toBe('s1')

    // shot 1 done, shot 2 done with openedOn, shot 3 failed
    h.set({ jobs: [
      view(body, 0, { status: 'done', promptId: 'p1', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e1', durationMs: 5000, finishedAt: 2000, endedAt: 2001 }),
      view(body, 1, { status: 'queued', promptId: 'p2', openedOn: cont.annotatedRef(frame(1)) }),
      view(body, 2, { wait: { for: 'before' } }),
    ] })
    st = engine.reelRun.snapshot()
    expect(st.states.s1).toMatchObject({ status: 'done', clip: clip(1), frame: frame(1), entryId: 'e1', durationMs: 5000 })
    expect(st.states.s2).toMatchObject({ status: 'queued', stage: 'Queued', promptId: 'p2' })
    h.set({ groups: [groupOf(body, { state: 'ended', endedAt: 4000, endedBy: { jobId: body.jobs[2].id, why: 'failed' } })], jobs: [
      view(body, 0, { status: 'done', promptId: 'p1', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e1', durationMs: 5000, finishedAt: 2000, endedAt: 2001 }),
      view(body, 1, { status: 'done', promptId: 'p2', openedOn: cont.annotatedRef(frame(1)), files: [clip(2), frame(2)], primary: clip(2), frame: frame(2), entryId: 'e2', durationMs: 0, finishedAt: 3000, endedAt: 3001 }),
      view(body, 2, { status: 'failed', promptId: 'p3', endedAt: 3900, error: { code: 'failed', message: 'boom', node: '8', nodeType: 'VAEDecode', nodeErrors: null, mayExist: false, sent: true, after: null } }),
    ] })
    st = engine.reelRun.snapshot()
    expect(st.status).toBe('error')
    expect(st.note).toBe('Shot 3 failed.')
    expect(st.states.s2.made?.openedOn).toBe(cont.annotatedRef(frame(1)))
    expect(engine.currencyOf(1, st.order, jobs, st.states)).toBe('current')
    expect(st.states.s3).toMatchObject({ status: 'error', stage: 'Failed', error: 'boom', detail: 'The trouble is in VAEDecode (node 8).' })
    expect(st.runnerGroupId).toBe(body.group.id)
    expect(engine.reelRun.busy()).toBe(false)
    const saved = JSON.parse(session.store.get('switchgen.reelrun.v1'))
    expect(saved.shots.map((s: any) => s.shotId).sort()).toEqual(['s1', 's2'])
    expect(saved.pending).toBeNull()
    expect(saved.press).toBeNull()
  })

  it('refuses a lone shot whose upstream outside the pass has no frame, sending nothing', async () => {
    const jobs = plan(['a', 'b'])
    engine.reelRun.renderOne(1, ['s1', 's2'], jobs, ctx() as any)
    await vi.waitFor(() => expect(engine.reelRun.snapshot().status).not.toBe('running'))
    const st = engine.reelRun.snapshot()
    expect(h.fns.submit).not.toHaveBeenCalled()
    expect(st.status).toBe('error')
    expect(st.note).toBe('Shot 2 had nothing to continue from.')
    expect(st.states.s2).toMatchObject({ status: 'error', stage: 'Not built' })
    expect(st.states.s2.error).toMatch(/opens on shot 1's last frame/)
  })

  it('uses the real frame when the upstream is outside the pass', async () => {
    const jobs = plan(['a', 'b'])
    // give s1 a done clip via a first pass in the store? simpler: seed saved run
    const saved = JSON.stringify({ shots: [{ shotId: 's1', clip: clip(1), frame: frame(1), files: [clip(1), frame(1)], entryId: 'e1', durationMs: 1, finishedAt: 1, made: { signature: cont.jobSignature(jobs[0]), seed: 7, openedOn: null, frames: 17, fps: 24, width: 1, height: 1 } }], pending: null, press: null })
    vi.resetModules()
    session = await import('../src/lib/session')
    session.store.set('switchgen.reelrun.v1', saved)
    engine = await import('../src/components/reel/engine')
    let body: any
    h.fns.submit.mockImplementation(async (b: any) => { body = b; return { ok: true, replayed: false, group: groupOf(b), jobs: b.jobs.map((_: any, i: number) => view(b, i)) } })
    engine.reelRun.renderOne(1, ['s1', 's2'], jobs, ctx() as any)
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled())
    expect(body.jobs).toHaveLength(1)
    expect(body.jobs[0].chain).toBeUndefined()
    expect(body.jobs[0].meta.made.openedOn).toBe(cont.annotatedRef(frame(1)))
    expect(JSON.stringify(body.jobs[0].graph)).not.toContain('switchgen:chain')
  })

  it('409 busy reel shows the line and puts the shots back', async () => {
    h.fns.submit.mockResolvedValue({ ok: false, fallback: false, status: 409, error: 'x', busy: 'reel' })
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx() as any)
    await vi.waitFor(() => expect(engine.reelRun.snapshot().status).toBe('stopped'))
    expect(engine.reelRun.snapshot().note).toBe(engine.REEL_BUSY)
    expect(engine.reelRun.snapshot().states.s1.status).toBe('waiting')
  })

  it('falls back to the page when the server says so', async () => {
    h.fns.submit.mockResolvedValue({ ok: false, fallback: true, reason: 'no' })
    engine.reelRun.renderAll(['s1'], plan(['a']), { ...ctx(), memory: () => ({ level: 'ok', reason: null, release: false }) } as any)
    await vi.waitFor(() => expect(h.fns.run).toHaveBeenCalled())
    expect(engine.reelRun.snapshot().runnerGroupId).toBeNull()
  })

  it('asks afresh where the store says the queue is off, reading as the walk meanwhile, and walks in page when it still is', async () => {
    h.set({ available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.' })
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx(-1, false) as any)
    expect(h.fns.available).toHaveBeenCalledWith('reel')
    expect(engine.reelRun.busy()).toBe(true)
    const st = engine.reelRun.snapshot()
    expect(st.runnerGroupId).toBeNull()
    expect(st.states.s1.stage).not.toBe('Handing it to the server')
    await vi.waitFor(() => expect(h.fns.run).toHaveBeenCalled(), LONG)
    expect(h.fns.submit).not.toHaveBeenCalled()
    expect(engine.reelRun.snapshot().runnerGroupId).toBeNull()
  })

  it('hands the pass over where the store last said the queue is off but it has come back since', async () => {
    h.set({ available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.' })
    h.onRefresh = () => h.set({ available: true, reason: null })
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined(), LONG)
    expect(h.fns.run).not.toHaveBeenCalled()
    expect(engine.reelRun.snapshot().runnerGroupId).toBe(got.body.group.id)
  })

  it('a Stop while the page asks whether the queue is back ends the pass here, and nothing is walked or sent', async () => {
    h.set({ available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.' })
    let answer: (v: any) => void = () => {}
    h.fns.available = vi.fn(() => new Promise((res) => (answer = res)))
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx() as any)
    engine.reelRun.stop()
    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot()).toMatchObject({ status: 'stopped', note: 'Stopped before shot 1.' })
    answer({ ok: false, reason: null })
    await new Promise((res) => setTimeout(res, 10))
    expect(h.fns.run).not.toHaveBeenCalled()
    expect(h.fns.submit).not.toHaveBeenCalled()
  })

  it('stop calls stopGroup; a stopped running shot keeps its earlier clip', async () => {
    let body: any
    h.fns.submit.mockImplementation(async (b: any) => { body = b; return { ok: true, replayed: false, group: groupOf(b), jobs: b.jobs.map((_: any, i: number) => view(b, i)) } })
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx() as any)
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled())
    await vi.waitFor(() => expect(engine.reelRun.snapshot().runnerGroupId).toBe(body.group.id))
    engine.reelRun.stop()
    expect(h.fns.stop).toHaveBeenCalledWith(body.group.id)
    h.set({ groups: [groupOf(body, { state: 'ended', endedAt: 5000 })], jobs: [
      view(body, 0, { status: 'stopped', promptId: 'p1', endedAt: 4000, error: { code: 'stopped', message: null, node: null, nodeType: null, nodeErrors: null, mayExist: false, sent: true, after: null } }),
      view(body, 1, { status: 'stopped', endedAt: 4000, error: { code: 'stopped', message: null, node: null, nodeType: null, nodeErrors: null, mayExist: false, sent: false, after: null } }),
    ] })
    expect(engine.reelRun.snapshot().status).toBe('stopped')
    expect(engine.reelRun.snapshot().note).toBe('Stopped during shot 1.')
  })

  it('takes up a live pass for this strip on load, and folds its clips in', async () => {
    const jobs = plan(['a', 'b'])
    const store = await import('../src/components/reel/store')
    store.reel.patch({ shots: [{ ...store.newShot('a'), id: 's1' }, { ...store.newShot('b'), id: 's2' }] })
    const body = { group: { id: 'g1', label: 'Reel, 2 shots' }, jobs: [
      { id: 'j1', label: 'Shot 1 of 2', prompt: 'a', heavy: true, meta: { shotId: 's1', index: 0, chained: false, made: { signature: cont.jobSignature(jobs[0]), seed: 7, openedOn: null, frames: 17, fps: 24, width: 1, height: 1 } } },
      { id: 'j2', label: 'Shot 2 of 2', prompt: 'b', heavy: true, meta: { shotId: 's2', index: 1, chained: true, made: { signature: cont.jobSignature(jobs[1]), seed: 8, openedOn: null, frames: 16, fps: 24, width: 1, height: 1 } } },
    ] }
    h.set({ groups: [groupOf(body)], jobs: [view(body, 0, { status: 'done', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e1', finishedAt: 2000, endedAt: 2000, promptId: 'p1' }), view(body, 1, { status: 'running', promptId: 'p2' })] })
    const st = engine.reelRun.snapshot()
    expect(st.runnerGroupId).toBe('g1')
    expect(st.status).toBe('running')
    expect(st.states.s1).toMatchObject({ status: 'done', clip: clip(1) })
    expect(st.states.s2).toMatchObject({ status: 'running' })
    expect(engine.reelRun.busy()).toBe(true)
  })
})

describe('passNote', () => {
  it('matches the walk sentences', () => {
    expect(engine.passNote([0, 1, 2], [{ kind: 'done' }, { kind: 'failed', nextContinues: true }, null]).note).toBe('Shot 2 failed, and shot 3 was going to open on its last frame, so the rest of this pass was not sent.')
    expect(engine.passNote([0, 2], [{ kind: 'done' }, { kind: 'stopped-before' }]).note).toBe('Stopped before shot 3. Shot 1 finished in this pass and is on disk and in the archive.')
    expect(engine.passNote([0], [{ kind: 'done', unchanged: true }]).note).toMatch(/^Shot 1 came back as the clip already on disk/)
  })
})

describe('more', () => {
  it('pending, then the outbox gives it up', async () => {
    h.fns.submit.mockResolvedValue({ ok: false, fallback: false, pending: true })
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx())
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled())
    await new Promise((r) => setTimeout(r, 5))
    expect(engine.reelRun.snapshot().status).toBe('running')
    expect(engine.reelRun.snapshot().states.s1.stage).toBe('Handing it to the server')
    const id = h.fns.submit.mock.calls[0][0].group.id
    h.givenUp = [{ groupId: id, desk: 'reel', label: 'L', at: 1, line: 'The server never answered for this batch, so it was not sent.' }]
    h.set({})
    expect(engine.reelRun.snapshot().status).toBe('stopped')
    expect(engine.reelRun.snapshot().note).toBe('The server never answered for this pass, so it was not sent.')
    expect(engine.reelRun.snapshot().states.s1.status).toBe('waiting')
  })

  it('no-frame from the server shows the page sentence and skips the rest', async () => {
    let body: any
    h.fns.submit.mockImplementation(async (b: any) => { body = b; return { ok: true, replayed: false, group: groupOf(b), jobs: b.jobs.map((_: any, i: number) => view(b, i)) } })
    engine.reelRun.renderAll(['s1', 's2', 's3'], plan(['a', 'b', 'c']), ctx())
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled())
    h.set({ groups: [groupOf(body, { state: 'ended', endedAt: 5000 })], jobs: [
      view(body, 0, { status: 'done', files: [clip(1)], primary: clip(1), frame: null, entryId: 'e1', finishedAt: 2000, endedAt: 2000, promptId: 'p1' }),
      view(body, 1, { status: 'failed', endedAt: 2100, error: err('no-frame', false) }),
      view(body, 2, { status: 'skipped', endedAt: 2100, error: { ...err('skipped', false), after: { jobId: body.jobs[1].id, index: 1 } } }),
    ] })
    const st = engine.reelRun.snapshot()
    expect(st.note).toBe('Shot 2 had nothing to continue from, so the rest of this pass was not sent.')
    expect(st.states.s2).toMatchObject({ status: 'error', stage: 'Not built' })
    expect(st.states.s2.error).toBe("This shot opens on shot 1's last frame, and shot 1 has not produced one yet. Render the shot before it, or pin an opening frame here.")
    expect(st.states.s3.status).toBe('waiting')
  })

  it('a lost heavy shot that held the lane says the rest waits', async () => {
    let body: any
    h.fns.submit.mockImplementation(async (b: any) => { body = b; return { ok: true, replayed: false, group: groupOf(b), jobs: b.jobs.map((_: any, i: number) => view(b, i)) } })
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx())
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled())
    h.set({ lane: { held: { why: 'lost', scope: 'heavy', jobId: body.jobs[0].id, since: 1 } }, groups: [groupOf(body, { state: 'ended' })], jobs: [view(body, 0, { status: 'lost', promptId: 'p1', endedAt: 3000, error: err('lost') })] })
    const st = engine.reelRun.snapshot()
    expect(st.states.s1.error).toMatch(/Anything waiting its turn is held until you send it or call it off/)
    expect(st.note).toBe('Shot 1 failed.')
  })

  it('an ended pass is taken up on load once, and dismissed later', async () => {
    vi.useFakeTimers()
    try {
      const jobs = plan(['a'])
      const store = await import('../src/components/reel/store')
      store.reel.patch({ shots: [{ ...store.newShot('a'), id: 's1' }] })
      const body = { group: { id: 'g9' }, jobs: [{ id: 'j9', label: 'Shot 1 of 1', prompt: 'a', heavy: true, meta: { shotId: 's1', index: 0, chained: false, made: { signature: cont.jobSignature(jobs[0]), seed: 7, openedOn: null, frames: 17, fps: 24, width: 1, height: 1 } } }] }
      h.set({ groups: [groupOf(body, { state: 'ended', endedAt: 1500 })], jobs: [view(body, 0, { status: 'done', files: [clip(9), frame(9)], primary: clip(9), frame: frame(9), entryId: 'e9', finishedAt: 1400, endedAt: 1400, promptId: 'p9', repeatOf: null })] })
      const st = engine.reelRun.snapshot()
      expect(st.status).toBe('done')
      expect(st.note).toBeNull()
      expect(st.states.s1).toMatchObject({ status: 'done', clip: clip(9) })
      expect(engine.currencyOf(0, ['s1'], jobs, st.states)).toBe('current')
      h.set({})
      expect(engine.reelRun.snapshot().runnerGroupId).toBe('g9')
      await vi.advanceTimersByTimeAsync(10_001)
      expect(h.fns.dismiss).toHaveBeenCalledWith(['j9'])
    } finally {
      vi.useRealTimers()
    }
  })

  it('a storage write from another tab does not undo a live pass', async () => {
    let body: any
    h.fns.submit.mockImplementation(async (b: any) => { body = b; return { ok: true, replayed: false, group: groupOf(b), jobs: b.jobs.map((_: any, i: number) => view(b, i)) } })
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx())
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled())
    h.set({ groups: [groupOf(body)], jobs: [view(body, 0, { status: 'running', promptId: 'p1' }), view(body, 1)] })
    // clear from elsewhere does nothing while live
    engine.reelRun.clear()
    expect(engine.reelRun.snapshot().runnerGroupId).toBe(body.group.id)
    expect(engine.reelRun.snapshot().states.s1.status).toBe('running')
    // pressing again is refused while live
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx())
    expect(h.fns.submit).toHaveBeenCalledTimes(1)
  })
})

describe('a shot refused at the press', () => {
it('a refused shot cuts the pass after the shots before it, with the walk note', async () => {
  let body: any
  h.fns.submit.mockImplementation(async (b: any) => { body = b; return { ok: true, replayed: false, group: { id: b.group.id, desk: 'reel', kind: 'pass', label: 'L', device: 'd', createdAt: 1000, state: 'active', endedBy: null, endedAt: null, jobIds: b.jobs.map((j: any) => j.id), dismissed: false }, jobs: [] } })
  engine.reelRun.renderAll(['s1', 's2', 's3'], plan(['a', 'b', 'c']), ctx(1))
  expect(engine.waitingInPage(engine.reelRun.snapshot())).toBe(0)
  await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled())
  expect(body.jobs).toHaveLength(1)
  const done = { id: body.jobs[0].id, groupId: body.group.id, desk: 'reel', kind: 'video', seq: 0, index: 0, total: 1, label: 'x', prompt: '', device: 'd', heavy: true, status: 'done', wait: null, stopRequested: false, stopLanded: false, promptId: 'p1', attempt: 1, createdAt: 1000, sentAt: 1100, ranAt: 1200, finishedAt: 2000, endedAt: 2000, files: [], primary: { filename: 'c.webm', subfolder: '', type: 'output', kind: 'video' }, frame: null, openedOn: null, entryId: 'e1', entryNo: 1, repeatOf: null, durationMs: 800, error: null, meta: body.jobs[0].meta, dismissed: false }
  h.set({ groups: [{ id: body.group.id, desk: 'reel', kind: 'pass', label: 'L', device: 'd', createdAt: 1000, state: 'ended', endedBy: null, endedAt: 2100, jobIds: [done.id], dismissed: false }], jobs: [done] })
  const st = engine.reelRun.snapshot()
  expect(st.note).toBe('Shot 2 was not sent to ComfyUI, so the rest of this pass was not sent.')
  expect(st.states.s2).toMatchObject({ status: 'error', stage: 'Not sent', error: 'Too big.' })
  expect(st.states.s1.status).toBe('done')
  expect(st.status).toBe('error')
})
it('a refused first shot sends nothing', async () => {
  engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx(0))
  await vi.waitFor(() => expect(engine.reelRun.snapshot().status).toBe('error'))
  expect(h.fns.submit).not.toHaveBeenCalled()
  expect(engine.reelRun.snapshot().note).toBe('Shot 1 was not sent to ComfyUI, so the rest of this pass was not sent.')
})
it('stop while asking whether the server can take it sends nothing', async () => {
  engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx(-1))
  engine.reelRun.stop()
  await vi.waitFor(() => expect(engine.reelRun.snapshot().status).toBe('stopped'))
  expect(h.fns.submit).not.toHaveBeenCalled()
  expect(engine.reelRun.snapshot().note).toBe('Stopped before shot 1.')
})

})

describe('the closing note', () => {
  /** Walk a pass in the page, the fallback, with shot `failAt` failing, and return its note. */
  async function walkInPage(prompts: string[], family: string, failAt: number) {
    h.set({ available: false })
    const { ComfyError } = await import('../src/lib/comfy')
    let n = 0
    h.fns.run = vi.fn(async () => {
      const i = n++
      if (i === failAt) throw new ComfyError('boom')
      return [clip(i + 1), frame(i + 1)]
    })
    const ids = prompts.map((_, i) => `s${i + 1}`)
    engine.reelRun.renderAll(ids, plan(prompts, family), ctx(-1, false) as any)
    await vi.waitFor(() => expect(engine.reelRun.snapshot().status).not.toBe('running'), { timeout: 5000, interval: 5 })
    expect(h.fns.submit).not.toHaveBeenCalled()
    return engine.reelRun.snapshot().note
  }

  it('reads the same from the page\'s own walk and from passNote, for a shot the next one opens on', async () => {
    const note = await walkInPage(['a', 'b', 'c'], 'wan22-5b', 1)
    expect(note).toBe(engine.passNote([0, 1, 2], [{ kind: 'done' }, { kind: 'failed', nextContinues: true }, null]).note)
    expect(note).toBe('Shot 2 failed, and shot 3 was going to open on its last frame, so the rest of this pass was not sent.')
  })

  it('reads the same from both for a shot nothing opens on', async () => {
    const note = await walkInPage(['a', 'b', 'c'], 'wan22-14b-t2v', 1)
    expect(note).toBe(engine.passNote([0, 1, 2], [{ kind: 'done' }, { kind: 'failed', nextContinues: false }, null]).note)
    expect(note).toBe('Shot 2 failed, so the rest of this pass was not sent.')
  })

  it('says a clip from the cache came back unchanged, with the time its record holds', async () => {
    const history = await import('../src/lib/history')
    history.restore({ desk: 'video', mode: 't2v', kind: 'video', id: 'e-old', no: 3, at: 5, file: clip(1), promptId: 'p0', durationMs: 4321, familyId: 'wan22-5b', familyLabel: 'F', modelLabel: 'M', model: 'm', prompt: 'a', negative: null, seed: 7, steps: 1, cfg: 1, sampler: 's', scheduler: 's', width: 1, height: 1, variant: null } as never)
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined())
    const body = got.body
    h.set({ groups: [groupOf(body, { state: 'ended', endedAt: 3000 })], jobs: [view(body, 0, { status: 'done', promptId: 'p1', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e-old', repeatOf: 'e-old', durationMs: 0, finishedAt: 2000, endedAt: 2000 })] })
    const st = engine.reelRun.snapshot()
    expect(st.note).toMatch(/^Shot 1 came back as the clip already on disk\./)
    expect(st.states.s1).toMatchObject({ status: 'done', durationMs: 4321, entryId: 'e-old' })
  })

  it('keeps the earlier clip of a shot stopped while it was drawing, and says so', async () => {
    const jobs = plan(['a', 'b'])
    const saved = JSON.stringify({ shots: [{ shotId: 's1', clip: clip(1), frame: frame(1), files: [clip(1), frame(1)], entryId: 'e1', durationMs: 1, finishedAt: 1, made: { signature: 'old', seed: 7, openedOn: null, frames: 17, fps: 24, width: 1, height: 1 } }], pending: null, press: null })
    vi.resetModules()
    session = await import('../src/lib/session')
    session.store.set('switchgen.reelrun.v1', saved)
    engine = await import('../src/components/reel/engine')
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1', 's2'], jobs, ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined())
    const body = got.body
    h.set({ groups: [groupOf(body)], jobs: [view(body, 0, { status: 'running', promptId: 'p1' }), view(body, 1)] })
    h.set({
      groups: [groupOf(body, { state: 'ended', endedAt: 5000, endedBy: { jobId: body.jobs[0].id, why: 'stopped' } })],
      jobs: [view(body, 0, { status: 'stopped', promptId: 'p1', endedAt: 4000, error: err('stopped', true) }), view(body, 1, { status: 'stopped', endedAt: 4000, error: err('stopped', false) })],
    })
    const st = engine.reelRun.snapshot()
    expect(st.status).toBe('stopped')
    expect(st.note).toBe('Stopped during shot 1. Its earlier clip is kept.')
    expect(st.states.s1).toMatchObject({ clip: clip(1), entryId: 'e1' })
  })

  it('says stopped before a shot that the reader\'s word stopped while it waited', async () => {
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined())
    const body = got.body
    h.set({
      groups: [groupOf(body, { state: 'ended', endedAt: 5000 })],
      jobs: [view(body, 0, { status: 'done', promptId: 'p1', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e1', finishedAt: 2000, endedAt: 2000 }), view(body, 1, { status: 'stopped', endedAt: 4000, error: err('stopped', false) })],
    })
    expect(engine.reelRun.snapshot().note).toBe('Stopped before shot 2. Shot 1 finished in this pass and is on disk and in the archive.')
  })
})

describe('the Reel room with a pass on the server', () => {
  async function drawn() {
    const { renderToStaticMarkup } = await import('react-dom/server')
    const { createElement } = await import('react')
    const storage = new Map<string, string>()
    const kept = { getItem: (k: string) => storage.get(k) ?? null, setItem: (k: string, v: string) => void storage.set(k, String(v)), removeItem: (k: string) => void storage.delete(k) }
    vi.stubGlobal('localStorage', kept)
    vi.stubGlobal('sessionStorage', kept)
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {}, location: { hash: '' }, history: {}, matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }) })
    vi.stubGlobal('document', { addEventListener() {}, removeEventListener() {}, visibilityState: 'visible' })
    const Reel = (await import('../src/routes/Reel')).default
    return () => renderToStaticMarkup(createElement(Reel))
  }

  it('offers the reader\'s word on shots the server holds, in the words for why it holds them', async () => {
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined())
    const body = got.body
    const draw = await drawn()
    const { HELD_AFTER_RESTART } = await import('../src/lib/runner')
    h.set({ lane: { held: { why: 'lost', scope: 'heavy', jobId: 'x', since: 1 } }, groups: [groupOf(body)], jobs: [view(body, 0, { wait: { for: 'held' } }), view(body, 1, { wait: { for: 'before' } })] })
    let html = draw()
    // The word itself is the shell's notice's, with its own handling of a word
    // that does not get through: the desk names the shots and points to it.
    expect(html).toContain('Send them or call them off in the notice “Work held on the server” at the top of the page.')
    expect(heldNotice(html)).toContain('held back.')
    expect(heldNotice(html)).not.toContain('<button')
    expect(html).not.toContain('Send them anyway')
    expect(html).not.toContain('Stop them')
    expect(html).toContain('was lost')
    expect(html).toContain('on this desk and the Video desk')
    // The machine restarted after the shots were made (at 1000).
    h.set({ lane: { held: { why: 'restart', scope: 'all', jobId: null, since: 5000 } } })
    html = draw()
    expect(html).toContain(HELD_AFTER_RESTART)
    expect(html).toContain('on every desk')
    expect(heldNotice(html)).toContain('held back.')
    expect(heldNotice(html)).not.toContain('<button')
    vi.unstubAllGlobals()
  })

  it('says a pass still being handed over cannot be taken while the queue is off', async () => {
    h.fns.submit.mockResolvedValue({ ok: false, fallback: false, pending: true })
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx() as any)
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled(), LONG)
    const draw = await drawn()
    h.set({ available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.' })
    expect(engine.reelRun.snapshot()).toMatchObject({ status: 'running', runnerGroupId: h.fns.submit.mock.calls[0][0].group.id })
    const html = draw()
    expect(html).toContain('This pass is being handed to the SwitchGen server.')
    expect(html).toContain('The queue on the server is not running, so it cannot take the pass yet.')
    h.set({ available: true, reason: null })
    expect(draw()).not.toContain('cannot take the pass yet')
    vi.unstubAllGlobals()
  })

  it('names a pass from another browser that is not this strip\'s, with a Stop', async () => {
    const draw = await drawn()
    h.set({ groups: [{ id: 'g-other', desk: 'reel', kind: 'pass', label: 'L', device: 'someone-else', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: [], dismissed: false }] })
    const html = draw()
    expect(html).toContain('Another reel is on the press on the server,')
    expect(html).toContain('sent from another browser')
    expect(html).toContain('Stop it')
    vi.unstubAllGlobals()
  })
})

/** A wait with room for a runner several times slower than this one. */
const LONG = { timeout: 5000, interval: 10 }

describe('a pass the outbox gave up on', () => {
  const tick = () => new Promise((res) => setTimeout(res, 0))

  it('is said on the strip after a reload, naming the pass, once', async () => {
    const { OUTBOX_GIVEN_UP } = await import('../src/lib/runner')
    const store = await import('../src/components/reel/store')
    store.reel.patch({ shots: [{ ...store.newShot('a'), id: 's1' }, { ...store.newShot('b'), id: 's2' }] })
    expect(engine.reelRun.snapshot().status).toBe('idle')
    h.givenUp = [{ groupId: 'gone', desk: 'reel', label: 'Reel, 2 shots', at: Date.now() - 15 * 60_000, line: OUTBOX_GIVEN_UP }]
    h.set({})
    const st = engine.reelRun.snapshot()
    expect(st).toMatchObject({ status: 'stopped', note: 'Reel, 2 shots: The server never answered for this pass, so it was not sent.', order: ['s1', 's2'], queue: [], runnerGroupId: null })
    await vi.waitFor(() => expect(h.givenUp).toEqual([]), LONG)
    h.set({})
    expect(engine.reelRun.snapshot().note).toBe('Reel, 2 shots: The server never answered for this pass, so it was not sent.')
    await tick()
    expect(engine.reelRun.snapshot().status).toBe('stopped')
  })

  it('leaves other desks\' alone, and lets one the server lists after all go without a word', async () => {
    const { OUTBOX_GIVEN_UP } = await import('../src/lib/runner')
    h.givenUp = [
      { groupId: 'pics', desk: 'images', label: 'M, 3 pictures', at: 1, line: OUTBOX_GIVEN_UP },
      { groupId: 'listed', desk: 'reel', label: 'Reel, 2 shots', at: 1, line: OUTBOX_GIVEN_UP },
    ]
    h.set({ groups: [{ id: 'listed', desk: 'reel', kind: 'pass', label: 'L', device: 'd', createdAt: 1, state: 'ended', endedBy: null, endedAt: 2, jobIds: [], dismissed: true }] })
    expect(engine.reelRun.snapshot()).toMatchObject({ status: 'idle', note: null })
    await vi.waitFor(() => expect(h.givenUp.map((g: any) => g.groupId)).toEqual(['pics']), LONG)
  })

  it('waits while another pass is on the press, then is said after how that pass ended', async () => {
    const { OUTBOX_NOT_TAKEN } = await import('../src/lib/runner')
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined(), LONG)
    const body = got.body
    h.givenUp = [{ groupId: 'older', desk: 'reel', label: 'Reel, shot 1', at: 1, line: OUTBOX_NOT_TAKEN }]
    h.set({ groups: [groupOf(body)], jobs: [view(body, 0, { status: 'running', promptId: 'p1' })] })
    expect(engine.reelRun.snapshot().status).toBe('running')
    expect(h.givenUp).toHaveLength(1)
    h.set({ groups: [groupOf(body, { state: 'ended', endedAt: 3000 })], jobs: [view(body, 0, { status: 'failed', promptId: 'p1', endedAt: 2000, error: err('failed', true, 'boom') })] })
    h.set({})
    const st = engine.reelRun.snapshot()
    expect(st.status).toBe('error')
    expect(st.note).toBe(`Shot 1 failed. Reel, shot 1: ${OUTBOX_NOT_TAKEN.replace('this batch', 'this pass')}`)
    expect(st.note).toBe('Shot 1 failed. Reel, shot 1: The queue on the server had stopped when this pass reached it, so it was not taken.')
  })

  it('is kept for later while the strip walks in the page', async () => {
    h.set({ available: false })
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx() as any)
    expect(engine.reelRun.busy()).toBe(true)
    h.givenUp = [{ groupId: 'x', desk: 'reel', label: 'Reel, shot 1', at: 1, line: 'L' }]
    h.set({})
    expect(engine.reelRun.snapshot()).toMatchObject({ status: 'running', note: null })
    await tick()
    expect(h.givenUp).toHaveLength(1)
  })
})

describe('a shot lost on the server, and the hold that follows it', () => {
  const lost = (body: any, i: number) => view(body, i, { status: 'lost', promptId: `p${i}`, endedAt: 3000, error: err('lost', true) })
  async function pressed() {
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined(), LONG)
    return got.body
  }

  it('says the rest is held only while the lane is held for it', async () => {
    const body = await pressed()
    h.set({ lane: { held: { why: 'lost', scope: 'heavy', jobId: body.jobs[0].id, since: 1 } }, groups: [groupOf(body, { state: 'ended', endedAt: 3000 })], jobs: [lost(body, 0)] })
    expect(engine.reelRun.snapshot().status).not.toBe('running')
    expect(engine.reelRun.snapshot().states.s1.error).toMatch(/held until you send it or call it off/)
    // The word is given, on this device or another.
    h.set({ lane: { held: null } })
    const s1 = engine.reelRun.snapshot().states.s1
    expect(s1.status).toBe('error')
    expect(s1.error).not.toMatch(/held/)
    expect(s1.error).toMatch(/We lost track of this job/)
  })

  it('says it once the lane is held, when the hold comes after the ending', async () => {
    const body = await pressed()
    h.set({ groups: [groupOf(body, { state: 'ended', endedAt: 3000 })], jobs: [lost(body, 0)] })
    expect(engine.reelRun.snapshot().states.s1.error).not.toMatch(/held/)
    h.set({ lane: { held: { why: 'lost', scope: 'heavy', jobId: body.jobs[0].id, since: 1 } } })
    expect(engine.reelRun.snapshot().states.s1.error).toMatch(/held until you send it or call it off/)
  })

  it('leaves alone a fault a later render wrote on the shot', async () => {
    const body = await pressed()
    h.set({ lane: { held: { why: 'lost', scope: 'heavy', jobId: body.jobs[0].id, since: 1 } }, groups: [groupOf(body, { state: 'ended', endedAt: 3000 })], jobs: [lost(body, 0)] })
    // A later press, in the page, is refused at the press.
    h.set({ available: false })
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx(0) as any)
    await vi.waitFor(() => expect(engine.reelRun.snapshot().status).not.toBe('running'), LONG)
    expect(engine.reelRun.snapshot().states.s1.error).toBe('Too big.')
    h.set({ lane: { held: null } })
    expect(engine.reelRun.snapshot().states.s1.error).toBe('Too big.')
  })
})

describe('the Reel room and what the server holds', () => {
  async function drawn() {
    const { renderToStaticMarkup } = await import('react-dom/server')
    const { createElement } = await import('react')
    const storage = new Map<string, string>()
    const kept = { getItem: (k: string) => storage.get(k) ?? null, setItem: (k: string, v: string) => void storage.set(k, String(v)), removeItem: (k: string) => void storage.delete(k) }
    vi.stubGlobal('localStorage', kept)
    vi.stubGlobal('sessionStorage', kept)
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {}, location: { hash: '' }, history: {}, matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }) })
    vi.stubGlobal('document', { addEventListener() {}, removeEventListener() {}, visibilityState: 'visible' })
    const Reel = (await import('../src/routes/Reel')).default
    return () => renderToStaticMarkup(createElement(Reel))
  }
  const other = { id: 'g-other', desk: 'reel', kind: 'pass', label: 'L', device: 'someone-else', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: ['o1', 'o2', 'o3'], dismissed: false }
  const theirs = (id: string, patch: any) => ({
    id, groupId: 'g-other', desk: 'reel', kind: 'video', seq: 1, index: 1, total: 3, label: 'x', prompt: '', device: 'someone-else', heavy: true, status: 'waiting', wait: null, stopRequested: false, stopLanded: false, promptId: null, attempt: 0,
    createdAt: 1, sentAt: null, ranAt: null, finishedAt: null, endedAt: null, files: [], primary: null, frame: null, openedOn: null, entryId: null, entryNo: null, repeatOf: null, durationMs: 0, error: null, meta: null, dismissed: false, ...patch,
  })

  it('counts the held shots of another browser\'s pass as the hold covers them, and leaves the word to the shell', async () => {
    const draw = await drawn()
    h.set({ lane: { held: { why: 'lost', scope: 'heavy', jobId: 'v', since: 1 } }, groups: [other], jobs: [theirs('o1', { wait: { for: 'held' } }), theirs('o2', { wait: { for: 'before' } }), theirs('o3', { heavy: false })] })
    let html = draw()
    expect(html).toContain('The server holds 2 of its shots until someone says.')
    expect(html).toContain('Stop it')
    expect(html).not.toContain('Send them anyway')
    h.set({ lane: { held: { why: 'restart', scope: 'all', jobId: null, since: 1 } } })
    html = draw()
    expect(html).toContain('The server holds 3 of its shots until someone says.')
    h.set({ lane: { held: null } })
    html = draw()
    expect(html).not.toContain('until someone says')
    expect(html).toContain('Stop it')
    vi.unstubAllGlobals()
  })

  it('says nothing held of this strip\'s shots when the hold was made before them, as the server rules', async () => {
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined(), LONG)
    const body = got.body
    const draw = await drawn()
    // The shots were made at 1000, after the queue came back and held what waited then.
    h.set({ lane: { held: { why: 'paused', scope: 'all', jobId: null, since: 500 } }, groups: [groupOf(body)], jobs: [view(body, 0, { wait: { for: 'turn' } }), view(body, 1, { wait: { for: 'before' } })] })
    const html = draw()
    expect(heldNotice(html)).toBe('')
    expect(html).not.toContain('held until you say')
    vi.unstubAllGlobals()
  })

  it('words this strip\'s hold after the queue was off, with no lost clip', async () => {
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined(), LONG)
    const body = got.body
    const draw = await drawn()
    h.set({ lane: { held: { why: 'paused', scope: 'all', jobId: null, since: 5000 } }, groups: [groupOf(body)], jobs: [view(body, 0, { wait: { for: 'held' } }), view(body, 1, { wait: { for: 'before' } })] })
    const html = draw()
    expect(html).toContain('The queue on the server was off while these shots waited, so they are held until you say.')
    expect(html).toContain('on every desk')
    expect(html).toContain('Send them or call them off in the notice “Work held on the server” at the top of the page.')
    expect(heldNotice(html)).toContain('held back.')
    expect(heldNotice(html)).not.toContain('<button')
    expect(html).not.toContain('was lost')
    expect(html).not.toContain('as much memory')
    vi.unstubAllGlobals()
  })
})

/** The body of a two-shot pass for shots s1 and s2, as the server lists it, the second opening on the first's frame. */
function passBody(jobs: any[]) {
  return {
    group: { id: 'g1', label: 'Reel, 2 shots' },
    jobs: [
      { id: 'j1', label: 'Shot 1 of 2', prompt: 'a', heavy: true, meta: { shotId: 's1', index: 0, chained: false, made: { signature: cont.jobSignature(jobs[0]), seed: 7, openedOn: null, frames: 17, fps: 24, width: 1, height: 1 } } },
      { id: 'j2', label: 'Shot 2 of 2', prompt: 'b', heavy: true, meta: { shotId: 's2', index: 1, chained: true, made: { signature: cont.jobSignature(jobs[1]), seed: 8, openedOn: cont.annotatedRef(frame(1)), frames: 17, fps: 24, width: 1, height: 1 } } },
    ],
  }
}
/** The server's list with that pass ended, both shots done, the second opening on the first's frame. */
const endedState = (body: any) => ({
  groups: [groupOf(body, { state: 'ended', endedAt: 5000 })],
  jobs: [
    view(body, 0, { status: 'done', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e1', finishedAt: 2000, endedAt: 2000, promptId: 'p1' }),
    view(body, 1, { status: 'done', files: [clip(2), frame(2)], primary: clip(2), frame: frame(2), entryId: 'e2', finishedAt: 3000, endedAt: 3000, promptId: 'p2', openedOn: cont.annotatedRef(frame(1)) }),
  ],
})
/** The strip laid out with shots of these ids. */
async function strip(ids: string[]) {
  const store = await import('../src/components/reel/store')
  store.reel.patch({ shots: ids.map((id, i) => ({ ...store.newShot(String.fromCharCode(97 + i)), id })) })
}

describe('a press made before the server\'s state has been read', () => {
  it('reads it first, and folds in the pass the server made, rendering nothing again', async () => {
    h.set({ boot: '', available: false })
    await strip(['s1', 's2'])
    const jobs = plan(['a', 'b'])
    const body = passBody(jobs)
    h.onRefresh = () =>
      h.set({
        boot: 'b',
        available: true,
        groups: [groupOf(body, { state: 'ended', endedAt: 5000 })],
        jobs: [
          view(body, 0, { status: 'done', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e1', finishedAt: 2000, endedAt: 2000, promptId: 'p1' }),
          view(body, 1, { status: 'done', files: [clip(2), frame(2)], primary: clip(2), frame: frame(2), entryId: 'e2', finishedAt: 3000, endedAt: 3000, promptId: 'p2', openedOn: cont.annotatedRef(frame(1)) }),
        ],
      })
    engine.reelRun.renderAll(['s1', 's2'], jobs, ctx(-1, false) as any)
    expect(engine.reelRun.busy()).toBe(true)
    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false), LONG)
    expect(h.fns.run).not.toHaveBeenCalled()
    expect(h.fns.submit).not.toHaveBeenCalled()
    const st = engine.reelRun.snapshot()
    expect(st.states.s1).toMatchObject({ status: 'done', clip: clip(1) })
    expect(st.states.s2).toMatchObject({ status: 'done', clip: clip(2) })
  })

  it('does not render again a shot pressed on its own when the read brings its clip', async () => {
    h.set({ boot: '', available: false })
    await strip(['s1', 's2'])
    const jobs = plan(['a', 'b'])
    const body = passBody(jobs)
    h.onRefresh = () => h.set({ boot: 'b', available: true, ...endedState(body) })
    engine.reelRun.renderOne(1, ['s1', 's2'], jobs, ctx(-1, false) as any)
    expect(engine.reelRun.busy()).toBe(true)
    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false), LONG)
    await new Promise((res) => setTimeout(res, 20))
    expect(h.fns.submit).not.toHaveBeenCalled()
    expect(h.fns.run).not.toHaveBeenCalled()
    expect(engine.reelRun.snapshot().states.s2).toMatchObject({ status: 'done', clip: clip(2) })
  })

  it('still renders again a shot pressed on its own whose clip the reader already had', async () => {
    await strip(['s1', 's2'])
    const jobs = plan(['a', 'b'])
    const body = passBody(jobs)
    // The pass was folded in once, with the store read.
    h.set({ ...endedState(body) })
    expect(engine.reelRun.snapshot().states.s2.clip).toEqual(clip(2))
    engine.reelRun.dismissNote()
    // A store not read yet, as after a reload, whose read lists the same pass.
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    h.set({ boot: '', available: false, groups: [], jobs: [] })
    h.onRefresh = () => h.set({ boot: 'b', available: true, ...endedState(body) })
    engine.reelRun.renderOne(1, ['s1', 's2'], jobs, ctx(-1, false) as any)
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled(), LONG)
  })

  it('folds in a pass that ended while the queue is off, rendering nothing again', async () => {
    h.set({ boot: '', available: false })
    await strip(['s1', 's2'])
    const jobs = plan(['a', 'b'])
    const body = passBody(jobs)
    h.onRefresh = () => h.set({ boot: 'b', available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.', ...endedState(body) })
    engine.reelRun.renderAll(['s1', 's2'], jobs, ctx(-1, false) as any)
    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false), LONG)
    expect(h.fns.run).not.toHaveBeenCalled()
    expect(h.fns.submit).not.toHaveBeenCalled()
    expect(engine.reelRun.snapshot().states.s1).toMatchObject({ status: 'done', clip: clip(1) })
    expect(engine.reelRun.snapshot().states.s2).toMatchObject({ status: 'done', clip: clip(2) })
  })

  it('takes up a pass the server is still rendering, in place of the press', async () => {
    h.set({ boot: '', available: false })
    await strip(['s1', 's2'])
    const jobs = plan(['a', 'b'])
    const body = passBody(jobs)
    h.onRefresh = () => h.set({ boot: 'b', available: true, groups: [groupOf(body)], jobs: [view(body, 0, { status: 'running', promptId: 'p1' }), view(body, 1)] })
    engine.reelRun.renderAll(['s1', 's2'], jobs, ctx(-1, false) as any)
    await vi.waitFor(() => expect(engine.reelRun.snapshot().runnerGroupId).toBe('g1'), LONG)
    expect(h.fns.run).not.toHaveBeenCalled()
    expect(h.fns.submit).not.toHaveBeenCalled()
    expect(engine.reelRun.snapshot().status).toBe('running')
  })

  it('walks in the page once the read says the queue is off', async () => {
    h.set({ boot: '', available: false })
    h.onRefresh = () => h.set({ boot: 'b', available: false, reason: 'Turned off with SWITCHGEN_RUNNER=off.' })
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx(-1, false) as any)
    await vi.waitFor(() => expect(h.fns.run).toHaveBeenCalled(), LONG)
    expect(h.fns.submit).not.toHaveBeenCalled()
    expect(engine.reelRun.snapshot().runnerGroupId).toBeNull()
  })

  it('asks whether the queue takes it when the read brings no answer, and hands it over', async () => {
    h.set({ boot: '', available: false })
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx(-1, false) as any)
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled(), LONG)
    expect(h.fns.available).toHaveBeenCalled()
    expect(h.fns.run).not.toHaveBeenCalled()
  })

  it('walks at once where the server has said it has no queue', async () => {
    h.set({ boot: '', available: false, reason: 'This server has no queue of its own.' })
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx(-1, false) as any)
    expect(engine.reelRun.snapshot().runnerGroupId).toBeNull()
    await vi.waitFor(() => expect(h.fns.run).toHaveBeenCalled(), LONG)
  })
})

describe('a pass the server keeps while its queue is off', () => {
  const OFF = 'Turned off with SWITCHGEN_RUNNER=off.'

  it('is not taken up while the queue is off, and is once it is back', async () => {
    await strip(['s1', 's2'])
    const body = passBody(plan(['a', 'b']))
    h.set({ available: false, reason: OFF, groups: [groupOf(body)], jobs: [view(body, 0, { status: 'running', promptId: 'p1' }), view(body, 1)] })
    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot().runnerGroupId).toBeNull()
    h.set({ available: true, reason: null })
    expect(engine.reelRun.snapshot().runnerGroupId).toBe('g1')
    expect(engine.reelRun.busy()).toBe(true)
  })

  it('is let go from the press when the queue goes off, with no stop asked while it is, and the stop asked again once it is back', async () => {
    await strip(['s1', 's2'])
    const got: { body?: any } = {}
    h.fns.submit.mockImplementation(taken(got))
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx() as any)
    await vi.waitFor(() => expect(got.body).toBeDefined(), LONG)
    const body = got.body
    h.set({ groups: [groupOf(body)], jobs: [view(body, 0, { status: 'done', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e1', finishedAt: 2000, endedAt: 2000, promptId: 'p1' }), view(body, 1, { status: 'running', promptId: 'p2' })] })
    expect(engine.reelRun.busy()).toBe(true)
    h.fns.stop.mockResolvedValue(false)
    engine.reelRun.stop()
    expect(h.fns.stop).toHaveBeenCalledTimes(1)
    h.set({ available: false, reason: OFF })
    expect(engine.reelRun.busy()).toBe(false)
    let st = engine.reelRun.snapshot()
    expect(st.status).toBe('idle')
    expect(st.states.s1).toMatchObject({ status: 'done', clip: clip(1) })
    expect(st.states.s2.status).toBe('waiting')
    h.set({})
    expect(h.fns.stop).toHaveBeenCalledTimes(1)
    h.fns.stop.mockResolvedValue(true)
    h.set({ available: true, reason: null })
    st = engine.reelRun.snapshot()
    expect(st.runnerGroupId).toBe(body.group.id)
    expect(st.stopRequested).toBe(true)
    expect(h.fns.stop).toHaveBeenCalledTimes(2)
  })

  it('is not taken up while the queue is off when listed as ended with a shot still running, and is once it is back', async () => {
    await strip(['s1', 's2'])
    const body = passBody(plan(['a', 'b']))
    const one = view(body, 0, { status: 'done', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e1', finishedAt: 2000, endedAt: 2000, promptId: 'p1' })
    // Stopped just before the queue went off: the list, as last saved, has the pass ended and its shot still stopping.
    h.set({ available: false, reason: OFF, groups: [groupOf(body, { state: 'ended', endedAt: 5000 })], jobs: [one, view(body, 1, { status: 'running', promptId: 'p2', stopRequested: true })] })
    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot()).toMatchObject({ status: 'idle', runnerGroupId: null })
    h.set({})
    expect(engine.reelRun.busy()).toBe(false)
    // The page's own work is free to go.
    engine.reelRun.renderOne(0, ['s1', 's2'], plan(['a', 'b']), ctx(-1, false) as any)
    await vi.waitFor(() => expect(h.fns.run).toHaveBeenCalled(), LONG)
    expect(h.fns.stop).not.toHaveBeenCalled()
  })

  it('takes up, once the queue is back, a pass listed as ended with a shot still running, and ends it with that shot', async () => {
    await strip(['s1', 's2'])
    const body = passBody(plan(['a', 'b']))
    const one = view(body, 0, { status: 'done', files: [clip(1), frame(1)], primary: clip(1), frame: frame(1), entryId: 'e1', finishedAt: 2000, endedAt: 2000, promptId: 'p1' })
    h.set({ available: false, reason: OFF, groups: [groupOf(body, { state: 'ended', endedAt: 5000 })], jobs: [one, view(body, 1, { status: 'running', promptId: 'p2', stopRequested: true })] })
    expect(engine.reelRun.snapshot().runnerGroupId).toBeNull()
    h.set({ available: true, reason: null })
    expect(engine.reelRun.snapshot().runnerGroupId).toBe('g1')
    expect(engine.reelRun.busy()).toBe(true)
    h.set({ jobs: [one, view(body, 1, { status: 'stopped', promptId: 'p2', endedAt: 5500, error: err('stopped', true) })] })
    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot().states.s1).toMatchObject({ status: 'done', clip: clip(1) })
  })

  it('folds in one that ended, at rest, and does not ask the server to put it away while the queue is off', async () => {
    vi.useFakeTimers()
    await strip(['s1', 's2'])
    const body = passBody(plan(['a', 'b']))
    h.set({ available: false, reason: OFF, ...endedState(body) })
    expect(engine.reelRun.snapshot().states.s2).toMatchObject({ status: 'done', clip: clip(2) })
    expect(engine.reelRun.busy()).toBe(false)
    await vi.advanceTimersByTimeAsync(10_001)
    expect(h.fns.dismiss).not.toHaveBeenCalled()
  })

  it('is taken up once the page\'s own walk ends, when the queue came back meanwhile, with no word from the server after', async () => {
    await strip(['s1', 's2'])
    const body = passBody(plan(['a', 'b']))
    let finish: (v: any) => void = () => {}
    h.fns.run.mockImplementation(() => new Promise((res) => (finish = res)))
    h.set({ available: false, reason: OFF, groups: [groupOf(body)], jobs: [view(body, 0, { wait: { for: 'held' } }), view(body, 1)] })
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx(-1, false) as any)
    await vi.waitFor(() => expect(h.fns.run).toHaveBeenCalled(), LONG)
    // Back, and holding the pass, while the page walks its own shot.
    h.set({ available: true, reason: null, lane: { held: { why: 'paused', scope: 'all', jobId: null, since: 5000 } } })
    expect(engine.reelRun.snapshot().runnerGroupId).toBeNull()
    finish([clip(7)])
    await vi.waitFor(() => expect(engine.reelRun.snapshot().runnerGroupId).toBe('g1'), LONG)
  })

  it('is named in the Reel room with the server\'s reason, and with no Stop', async () => {
    const { renderToStaticMarkup } = await import('react-dom/server')
    const { createElement } = await import('react')
    const storage = new Map<string, string>()
    const kept = { getItem: (k: string) => storage.get(k) ?? null, setItem: (k: string, v: string) => void storage.set(k, String(v)), removeItem: (k: string) => void storage.delete(k) }
    vi.stubGlobal('localStorage', kept)
    vi.stubGlobal('sessionStorage', kept)
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {}, location: { hash: '' }, history: {}, matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }) })
    vi.stubGlobal('document', { addEventListener() {}, removeEventListener() {}, visibilityState: 'visible' })
    const Reel = (await import('../src/routes/Reel')).default
    const other = { id: 'g-other', desk: 'reel', kind: 'pass', label: 'L', device: 'someone-else', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds: [], dismissed: false }
    h.set({ available: false, reason: OFF, groups: [other] })
    let html = renderToStaticMarkup(createElement(Reel))
    expect(html).toContain('Another reel waits on the server,')
    expect(html).toContain('turned off with SWITCHGEN_RUNNER=off.')
    expect(html).not.toContain('Stop it')
    h.set({ available: true, reason: null })
    html = renderToStaticMarkup(createElement(Reel))
    expect(html).toContain('Stop it')
  })
})

describe('Stop on a pass the server has not listed', () => {
  const STOPS = 'switchgen.reel.stops.v1'
  /** The tab's own storage, kept across a reload, with the engine loaded over it. */
  async function inTab(tab: Map<string, string>) {
    vi.stubGlobal('sessionStorage', { getItem: (k: string) => tab.get(k) ?? null, setItem: (k: string, v: string) => void tab.set(k, String(v)), removeItem: (k: string) => void tab.delete(k) })
    vi.resetModules()
    session = await import('../src/lib/session')
    engine = await import('../src/components/reel/engine')
  }

  it('takes it out of the outbox and ends it here at once, keeps the Stop in the tab, and asks it once the server lists the pass', async () => {
    const tab = new Map<string, string>()
    await inTab(tab)
    h.fns.submit.mockResolvedValue({ ok: false, fallback: false, pending: true })
    engine.reelRun.renderAll(['s1', 's2'], plan(['a', 'b']), ctx() as any)
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled(), LONG)
    await new Promise((res) => setTimeout(res, 5))
    const b = h.fns.submit.mock.calls[0][0]
    h.set({ available: false, reason: 'off' })
    expect(engine.reelRun.busy()).toBe(true)
    engine.reelRun.stop()
    // The press is free for the page's own work.
    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot()).toMatchObject({ status: 'stopped', note: 'Stopped before shot 1.' })
    expect(h.fns.withdraw).toHaveBeenCalledWith(b.group.id)
    expect(JSON.parse(tab.get(STOPS)!)[0].groupId).toBe(b.group.id)
    expect(h.fns.stop).not.toHaveBeenCalled()
    // An earlier try reached the server after all: listed once the queue is back.
    h.set({ available: true, reason: null, groups: [groupOf(b)], jobs: [view(b, 0), view(b, 1)] })
    expect(h.fns.stop).toHaveBeenCalledWith(b.group.id)
    expect(h.fns.stop).toHaveBeenCalledTimes(1)
    h.set({})
    expect(h.fns.stop).toHaveBeenCalledTimes(1)
    await vi.waitFor(() => expect(tab.has(STOPS)).toBe(false), LONG)
    expect(engine.reelRun.snapshot().status).toBe('stopped')
  })

  it('takes a pass still being handed over out of the outbox, and stops it if the answer says the server took it', async () => {
    let resolve: (v: any) => void = () => {}
    h.fns.submit.mockImplementation(() => new Promise((res) => (resolve = res)))
    engine.reelRun.renderAll(['s1'], plan(['a']), ctx() as any)
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled(), LONG)
    const b = h.fns.submit.mock.calls[0][0]
    engine.reelRun.stop()
    expect(h.fns.withdraw).toHaveBeenCalledWith(b.group.id)
    expect(engine.reelRun.snapshot().status).toBe('stopped')
    resolve({ ok: true, replayed: false, group: groupOf(b), jobs: [view(b, 0)] })
    await new Promise((res) => setTimeout(res, 5))
    expect(engine.reelRun.snapshot().status).toBe('stopped')
    h.set({ groups: [groupOf(b)], jobs: [view(b, 0)] })
    expect(h.fns.stop).toHaveBeenCalledWith(b.group.id)
  })

  it('is asked by the page loaded after the tab was thrown away, once the server lists the pass', async () => {
    const tab = new Map<string, string>([[STOPS, JSON.stringify([{ groupId: 'gX', at: Date.now() }])]])
    await inTab(tab)
    await strip(['s1', 's2'])
    const body = { ...passBody(plan(['a', 'b'])), group: { id: 'gX', label: 'L' } }
    h.set({ groups: [groupOf(body)], jobs: [view(body, 0, { status: 'running', promptId: 'p1' }), view(body, 1)] })
    expect(engine.reelRun.snapshot()).toMatchObject({ runnerGroupId: 'gX', stopRequested: true })
    expect(h.fns.stop).toHaveBeenCalledTimes(1)
    expect(h.fns.stop).toHaveBeenCalledWith('gX')
  })
})

describe('a pass the outbox gave up on, from a strip at rest', () => {
  it('names the shots it held, and gives it no time of its own', async () => {
    // The tab's own storage, kept across the reload below.
    const tab = new Map<string, string>()
    vi.stubGlobal('sessionStorage', { getItem: (k: string) => tab.get(k) ?? null, setItem: (k: string, v: string) => void tab.set(k, String(v)), removeItem: (k: string) => void tab.delete(k) })
    await strip(['s1', 's2', 's3'])
    h.fns.submit.mockResolvedValue({ ok: false, fallback: false, pending: true })
    engine.reelRun.renderOne(0, ['s1', 's2', 's3'], plan(['a', 'b', 'c']), ctx() as any)
    await vi.waitFor(() => expect(h.fns.submit).toHaveBeenCalled(), LONG)
    const id = h.fns.submit.mock.calls[0][0].group.id
    h.pending = [{ groupId: id, desk: 'reel', label: 'Reel, shot 1', at: 1, jobIds: [] }]
    await vi.waitFor(() => expect(JSON.parse(tab.get('switchgen.reel.handed.v1') ?? 'null')).toEqual([{ groupId: id, shots: ['s1'] }]), LONG)
    // The page goes; a new one loads in the same tab, and the outbox has given the pass up.
    vi.resetModules()
    session = await import('../src/lib/session')
    h.reset()
    engine = await import('../src/components/reel/engine')
    await strip(['s1', 's2', 's3'])
    const at = Date.now() - 15 * 60_000
    h.givenUp = [{ groupId: id, desk: 'reel', label: 'Reel, shot 1', at, line: 'The server never answered for this batch, so it was not sent.' }]
    h.pending = []
    h.set({})
    const st = engine.reelRun.snapshot()
    expect(st.note).toBe('Reel, shot 1: The server never answered for this pass, so it was not sent. It held shot 1, and nothing is rendering it.')
    expect(st.startedAt).toBe(at)
    expect(st.finishedAt).toBe(at)
    const { renderToStaticMarkup } = await import('react-dom/server')
    const { createElement } = await import('react')
    const { ReelProgress } = await import('../src/components/reel/Progress')
    const html = renderToStaticMarkup(createElement(ReelProgress, { run: st, jobs: [], fps: 24, width: 1, height: 1, familyId: 'x', records: [], now: Date.now(), onStop: () => {} } as any))
    expect(html).not.toContain('0 of 0')
    expect(html).not.toContain('Elapsed')
    expect(html).toContain('Stopped')
    expect(html).toContain('It held shot 1')
    await vi.waitFor(() => {
      h.set({})
      expect(tab.has('switchgen.reel.handed.v1')).toBe(false)
    }, LONG)
  })

  it('never reads a whole minute as 60 s', async () => {
    const { duration } = await import('../src/components/reel/bits')
    expect(duration(59_600)).toBe('1 min')
    expect(duration(119_600)).toBe('2 min')
    expect(duration(61_000)).toBe('1 min 1 s')
    expect(duration(59_400)).toBe('59 s')
    expect(duration(9_940)).toBe('9.9 s')
  })

  it('times an ended pass from its own start to its own end, not to the page\'s clock', async () => {
    const { renderToStaticMarkup } = await import('react-dom/server')
    const { createElement } = await import('react')
    const { ReelProgress } = await import('../src/components/reel/Progress')
    const run = { id: 'r', status: 'done', startedAt: 1000, finishedAt: 61_000, order: ['s1'], states: {}, currentShotId: null, queue: ['s1'], stopRequested: false, note: null, elsewhere: null, runnerGroupId: null }
    const html = renderToStaticMarkup(createElement(ReelProgress, { run, jobs: [], fps: 24, width: 1, height: 1, familyId: 'x', records: [], now: 10_000_000, onStop: () => {} } as any))
    expect(html).toContain('Elapsed</span><span class="tabular-nums text-ink">1 min</span>')
  })
})
