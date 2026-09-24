import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Engine from '../src/components/reel/engine'
import type * as Session from '../src/lib/session'

/**
 * The reel engine against the ways a phone lets a tab go and come back: a
 * tab frozen while another took the reel over, a stand-in record the
 * recovery pass filed first, a reload that left shots unsent, a Wan 14B shot
 * that samples in two passes, and a ComfyUI that is restarting. The queue and
 * the memory calls are stand-ins; the engine, the plan and the archive are the
 * real modules.
 */
const m = vi.hoisted(() => ({ run: vi.fn(), cancelJob: vi.fn(), getJob: vi.fn(), fetchPastRun: vi.fn(), others: vi.fn(), wait: vi.fn(), release: vi.fn() }))
vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
  getJob: m.getJob,
  fetchPastRun: m.fetchPastRun,
}))
vi.mock('../src/lib/clipMemory', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/clipMemory')>()),
  releaseIfOthersAhead: m.others,
  waitForIdleComfy: m.wait,
  releaseComfyMemory: m.release,
}))

/**
 * vi.waitFor with room for a slow runner: its own default gives up after one
 * second, and the engine walks on real timers here.
 */
const waitFor = <T>(fn: () => T) => vi.waitFor(fn, { timeout: 5000, interval: 10 })

const KEY = 'switchgen.reelrun.v1'
let engine: typeof Engine
async function load(saved: unknown) {
  vi.resetModules()
  const session = await import('../src/lib/session')
  session.store.set(KEY, JSON.stringify(saved))
  engine = await import('../src/components/reel/engine')
  return session
}
const savedRun = (session: typeof Session) => JSON.parse(session.store.get(KEY) ?? 'null')

async function planOf(prompts: string[]) {
  const { shotPlan } = await import('../src/lib/continuation')
  const { FAMILIES, defaultsFor } = await import('../src/lib/workflows')
  const T2V = FAMILIES.find((f) => f.id === 'wan22-14b-t2v')!
  const d = defaultsFor(T2V, '')
  const params = { model: '', positive: '', negative: d.negative ?? '', seed: 500, steps: d.steps, cfg: d.cfg, width: d.width, height: d.height, sampler: d.sampler, scheduler: d.scheduler, length: 17, fps: d.fps || 24 }
  return shotPlan({ base: T2V, params, shots: prompts.map((prompt) => ({ prompt, seed: undefined })), freshSeeds: false }).jobs
}
async function ctxOf(release = false): Promise<Engine.RunContext> {
  const { newComposition } = await import('../src/lib/session')
  return {
    familyLabel: 'x',
    modelLabel: 'y',
    compositionFor: (job) => newComposition('video', { mode: 't2v', familyId: 'wan22-14b-t2v', model: '', prompt: job.params.positive }),
    memory: () => ({ level: 'ok', reason: null, release }),
  }
}
const clipFiles = (n: number) => [
  { filename: `${n}.webm`, subfolder: 'reel', type: 'output', kind: 'video' },
  { filename: `${n}.png`, subfolder: 'reel', type: 'output', kind: 'image' },
]
function hanging(promptId: string) {
  let finish: (files: any[]) => void = () => {}
  let fail: (e: Error) => void = () => {}
  let on: any = null
  m.run.mockImplementationOnce(
    (_wf: unknown, cb: any) =>
      new Promise<any[]>((resolve, reject) => {
        on = cb
        cb({ phase: 'queued', promptId })
        finish = resolve
        fail = reject
      }),
  )
  return { finish: (files: any[]) => finish(files), fail: (e: Error) => fail(e), emit: (e: any) => on(e) }
}
function stubWindow() {
  const win = new EventTarget()
  vi.stubGlobal('window', win)
  return { fire: (type: string, extra: Record<string, unknown> = {}) => win.dispatchEvent(Object.assign(new Event(type), extra)) }
}
function stubDocument() {
  const doc = Object.assign(new EventTarget(), { visibilityState: 'hidden' as string })
  vi.stubGlobal('document', doc)
  return {
    show: () => {
      doc.visibilityState = 'visible'
      doc.dispatchEvent(new Event('visibilitychange'))
    },
    resume: () => doc.dispatchEvent(new Event('resume')),
  }
}
const takeover = (session: typeof Session) => {
  const theirs = {
    shots: [],
    pending: { ...savedRun(session).pending, owner: 'other', beat: Date.now(), released: false },
    press: { owner: 'other', beat: Date.now(), released: false, shotId: 's1' },
  }
  session.store.set(KEY, JSON.stringify(theirs))
  return theirs
}

beforeEach(() => {
  for (const f of Object.values(m)) f.mockReset()
  m.wait.mockResolvedValue(true)
  m.release.mockResolvedValue(true)
  m.others.mockResolvedValue(undefined)
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('finding 9: a thawed tab after a takeover', () => {
  it('lands its job with no pageshow: hands over, files nothing, sends nothing more', async () => {
    stubWindow()
    const session = await load({ shots: [], pending: null })
    const history = await import('../src/lib/history')
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1', 's2'], await planOf(['a', 'b']), await ctxOf())
    await waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'))
    const theirs = takeover(session)
    run.finish(clipFiles(1))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(m.run).toHaveBeenCalledTimes(1)
    expect(savedRun(session)).toEqual(theirs)
    expect(history.all()).toEqual([])
    expect(engine.reelRun.snapshot().note).toMatch(/another tab took the reel over/)
    expect(engine.reelRun.snapshot().note).not.toMatch(/already on disk/)
  })

  it('also when that tab had already filed and settled the shot', async () => {
    stubWindow()
    const session = await load({ shots: [], pending: null })
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1', 's2'], await planOf(['a', 'b']), await ctxOf())
    await waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'))
    const settled = { shots: [{ shotId: 's1', clip: clipFiles(1)[0], frame: clipFiles(1)[1], files: clipFiles(1), entryId: 'e1', durationMs: 5, finishedAt: 1, made: { signature: 's', seed: 1, openedOn: null, frames: 17, fps: 16, width: 832, height: 480 } }], pending: null, press: null }
    session.store.set(KEY, JSON.stringify(settled))
    run.finish(clipFiles(1))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(m.run).toHaveBeenCalledTimes(1)
    expect(savedRun(session)).toEqual(settled)
    expect(engine.reelRun.snapshot().states.s1?.status).toBe('done')
  })

  it('lets go when the page is shown again, before its job lands', async () => {
    stubWindow()
    const doc = stubDocument()
    const session = await load({ shots: [], pending: null })
    hanging('p1')
    engine.reelRun.renderAll(['s1', 's2'], await planOf(['a', 'b']), await ctxOf())
    await waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'))
    const theirs = takeover(session)
    doc.show()
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(savedRun(session)).toEqual(theirs)
  })

  it('lets go on a resume event', async () => {
    stubWindow()
    const doc = stubDocument()
    const session = await load({ shots: [], pending: null })
    hanging('p1')
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf())
    await waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'))
    const theirs = takeover(session)
    doc.resume()
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(savedRun(session)).toEqual(theirs)
  })

  it('lets go when another tab writes while it walks (storage event)', async () => {
    const win = stubWindow()
    const session = await load({ shots: [], pending: null })
    hanging('p1')
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf())
    await waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'))
    const theirs = takeover(session)
    win.fire('storage', { key: KEY, newValue: JSON.stringify(theirs) })
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(savedRun(session)).toEqual(theirs)
  })

  it('the heartbeat will not write over a takeover', async () => {
    vi.useFakeTimers()
    stubWindow()
    const session = await load({ shots: [], pending: null })
    hanging('p1')
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf())
    await vi.advanceTimersByTimeAsync(10)
    expect(savedRun(session)?.pending?.promptId).toBe('p1')
    const theirs = takeover(session)
    await vi.advanceTimersByTimeAsync(6000)
    expect(savedRun(session)).toEqual(theirs)
    expect(engine.reelRun.busy()).toBe(false)
  })

  it('a hold nobody took keeps going through a show and a heartbeat', async () => {
    stubWindow()
    const doc = stubDocument()
    const session = await load({ shots: [], pending: null })
    const history = await import('../src/lib/history')
    const r1 = hanging('p1')
    const r2 = hanging('p2')
    engine.reelRun.renderAll(['s1', 's2'], await planOf(['a', 'b']), await ctxOf())
    await waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'))
    doc.show()
    expect(engine.reelRun.busy()).toBe(true)
    r1.finish(clipFiles(1))
    await waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p2'))
    r2.finish(clipFiles(2))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(m.run).toHaveBeenCalledTimes(2)
    expect(history.all()).toHaveLength(2)
    expect(engine.reelRun.snapshot().status).toBe('done')
  })
})

describe('finding 39: the recovered stand-in', () => {
  it('is replaced by the reel record, not kept as a cached answer', async () => {
    stubWindow()
    await load({ shots: [], pending: null })
    const history = await import('../src/lib/history')
    const rec = history.add({ ...(await import('../src/lib/session')).recordOf((await ctxOf()).compositionFor({ params: { positive: '' } } as any), { file: clipFiles(1)[0], kind: 'video', promptId: '', durationMs: 0, seed: 0, familyLabel: '', modelLabel: '', at: 1 }), recovered: true } as any)
    expect(history.all()[0].recovered).toBe(true)
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf())
    await waitFor(() => expect(m.run).toHaveBeenCalled())
    run.finish(clipFiles(1))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    const all = history.all()
    expect(all).toHaveLength(1)
    expect(all[0].id).toBe(rec.id)
    expect(all[0].recovered).toBeUndefined()
    expect(all[0].promptId).toBe('p1')
    expect(engine.reelRun.snapshot().note).toBeNull()
    expect(engine.reelRun.snapshot().states.s1.entryId).toBe(rec.id)
  })

  it('a record of the same job from another tab is not called a cached answer', async () => {
    stubWindow()
    await load({ shots: [], pending: null })
    const history = await import('../src/lib/history')
    const { recordOf } = await import('../src/lib/session')
    history.add(recordOf((await ctxOf()).compositionFor({ params: { positive: '' } } as any), { file: clipFiles(1)[0], kind: 'video', promptId: 'p1', durationMs: 123, seed: 0, familyLabel: '', modelLabel: '', at: 1 }))
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf())
    await waitFor(() => expect(m.run).toHaveBeenCalled())
    run.finish(clipFiles(1))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(history.all()).toHaveLength(1)
    expect(engine.reelRun.snapshot().note).toBeNull()
  })

  it('a record of a different job for the same file is still a cached answer', async () => {
    stubWindow()
    await load({ shots: [], pending: null })
    const history = await import('../src/lib/history')
    const { recordOf } = await import('../src/lib/session')
    history.add(recordOf((await ctxOf()).compositionFor({ params: { positive: '' } } as any), { file: clipFiles(1)[0], kind: 'video', promptId: 'p0', durationMs: 123, seed: 0, familyLabel: '', modelLabel: '', at: 1 }))
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf())
    await waitFor(() => expect(m.run).toHaveBeenCalled())
    run.finish(clipFiles(1))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(engine.reelRun.snapshot().note).toMatch(/already on disk/)
  })
})

describe('finding 21: after a reload', () => {
  const made = { signature: 's', seed: 1, openedOn: null, frames: 17, fps: 24, width: 832, height: 480 }
  const pendingOf = (extra: Record<string, unknown>) => ({
    shotId: 's1', promptId: 'p-left', label: 'Shot 1', order: ['s1', 's2', 's3'], startedAt: Date.now() - 1000,
    made, frames: 17, composition: { kind: 'video' }, familyLabel: 'x', modelLabel: 'y', owner: 'old', beat: 0, released: true, ...extra,
  })
  it('names the shots after it that were never sent', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'completed' })
    m.fetchPastRun.mockResolvedValue({ promptId: 'p-left', graph: {}, status: 'success', startedAt: 1, finishedAt: 2, clientId: null, error: null, files: clipFiles(9) })
    await load({ shots: [], pending: pendingOf({}) })
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    const note = engine.reelRun.snapshot().note
    expect(note).toMatch(/Shots 2 and 3 after it have no clip/)
    expect(note).toMatch(/Render what is missing/)
  })
  it('says nothing more when every later shot has a clip', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'completed' })
    m.fetchPastRun.mockResolvedValue({ promptId: 'p-left', graph: {}, status: 'success', startedAt: 1, finishedAt: 2, clientId: null, error: null, files: clipFiles(9) })
    const shot = (id: string, n: number) => ({ shotId: id, clip: clipFiles(n)[0], frame: clipFiles(n)[1], files: clipFiles(n), entryId: null, durationMs: 5, finishedAt: 1, made })
    await load({ shots: [shot('s2', 2), shot('s3', 3)], pending: pendingOf({}) })
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(engine.reelRun.snapshot().note).toBe('Shot 1 was left on the press by the page before this one. It has finished and is filed.')
  })
  it('says Render the reel when the followed shot failed and nothing has a clip', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'failed' })
    await load({ shots: [], pending: pendingOf({}) })
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(engine.reelRun.snapshot().note).toMatch(/Render the reel sends them/)
  })
  it('pure helpers', async () => {
    await load({ shots: [], pending: null })
    expect(engine.unsentAfter(['a', 'b', 'c'], 'a', { b: { clip: {} } } as never)).toEqual([3])
    expect(engine.unsentAfter(['a', 'b'], 'zz', {})).toEqual([])
    expect(engine.unsentLine([2], true)).toBe('Shot 2 after it has no clip. Anything that page was still to send went with it, so Render what is missing sends it from here.')
    expect(engine.unsentLine([], true)).toBe('')
    expect(engine.waitingInPage({ status: 'running', elsewhere: null, queue: ['a', 'b', 'c'], states: { a: { status: 'queued', promptId: null }, b: { status: 'waiting', promptId: 'old' }, c: { status: 'done' } } } as never)).toBe(2)
    expect(engine.waitingInPage({ status: 'running', elsewhere: null, queue: ['a'], states: { a: { status: 'queued', promptId: 'p' } } } as never)).toBe(0)
    expect(engine.waitingInPage({ status: 'idle', elsewhere: null, queue: ['a'], states: { a: { status: 'waiting' } } } as never)).toBe(0)
  })
})

describe('C8, C7, C5, C14', () => {
  it('drawnFraction and samplerPass', async () => {
    await load({ shots: [], pending: null })
    const { samplerPass, instantiateShot } = await import('../src/lib/continuation')
    expect(engine.drawnFraction({ status: 'running', value: 5, max: 10, pass: { index: 2, count: 2 } })).toBe(0.75)
    expect(engine.drawnFraction({ status: 'running', value: 5, max: 10, pass: { index: 1, count: 2 } })).toBe(0.25)
    expect(engine.drawnFraction({ status: 'running', value: 5, max: 10, pass: null })).toBe(0.5)
    expect(engine.drawnFraction({ status: 'running', value: 0, max: 1, pass: { index: 2, count: 2 } })).toBe(0)
    const jobs = await planOf(['a'])
    const wf = instantiateShot(jobs[0])
    const ids = Object.entries(wf).filter(([, n]: any) => n.class_type === 'KSamplerAdvanced')
    expect(ids).toHaveLength(2)
    const passes = ids.map(([id]) => samplerPass(wf, id))
    expect(passes).toEqual(expect.arrayContaining([{ index: 1, count: 2 }, { index: 2, count: 2 }]))
    expect(samplerPass(wf, null)).toBeNull()
  })

  it('a shot on a 14B pair reports its pass through the run', async () => {
    stubWindow()
    await load({ shots: [], pending: null })
    const { instantiateShot, samplerPass } = await import('../src/lib/continuation')
    const jobs = await planOf(['a'])
    const wf = instantiateShot(jobs[0])
    const second = Object.keys(wf).find((id) => samplerPass(wf, id)?.index === 2)!
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1'], jobs, await ctxOf())
    await waitFor(() => expect(m.run).toHaveBeenCalled())
    run.emit({ phase: 'running', node: second, value: 3, max: 10 })
    expect(engine.reelRun.snapshot().states.s1.pass).toEqual({ index: 2, count: 2 })
    run.emit({ phase: 'running', node: 'nope', value: 0, max: 1 })
    expect(engine.reelRun.snapshot().states.s1.pass).toEqual({ index: 2, count: 2 })
    run.finish(clipFiles(1))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
  })

  it('asks for a second release when others are ahead, only for a shot that released', async () => {
    stubWindow()
    await load({ shots: [], pending: null })
    const run = hanging('p1')
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf(true))
    await waitFor(() => expect(m.others).toHaveBeenCalledWith('p1'))
    run.finish(clipFiles(1))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    const run2 = hanging('p2')
    engine.reelRun.renderAll(['s1'], await planOf(['b']), await ctxOf(false))
    await waitFor(() => expect(m.run).toHaveBeenCalledTimes(2))
    run2.finish(clipFiles(2))
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(m.others).toHaveBeenCalledTimes(1)
  })

  it('words a ComfyUI that is not answering', async () => {
    stubWindow()
    await load({ shots: [], pending: null })
    let say: any
    m.wait.mockImplementation((_s: any, on: any) => { say = on; return new Promise(() => {}) })
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf(true))
    await waitFor(() => expect(say).toBeDefined())
    say(-1)
    expect(engine.reelRun.snapshot().states.s1.stage).toMatch(/not answering and may be restarting/)
    say(2)
    expect(engine.reelRun.snapshot().states.s1.stage).toBe('Waiting for ComfyUI to finish 2 other jobs')
    engine.reelRun.stop()
  })

  it('names the real control when a picked-up clip is not in history', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'completed' })
    m.fetchPastRun.mockResolvedValue({ promptId: 'p-left', graph: {}, status: 'success', startedAt: 1, finishedAt: 2, clientId: null, error: null, files: [] })
    const made = { signature: 's', seed: 1, openedOn: null, frames: 17, fps: 24, width: 832, height: 480 }
    await load({ shots: [], pending: { shotId: 's1', promptId: 'p-left', label: 'Shot 1', order: ['s1'], startedAt: 1, made, frames: 17, composition: { kind: 'video' }, familyLabel: 'x', modelLabel: 'y', owner: 'old', beat: 0, released: true } })
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(engine.reelRun.snapshot().note).toMatch(/open the Archive and press "Look for files with no record"/)
  })
})

describe('finding 9: a removal by the other tab', () => {
  it('lets go when another tab removed the saved run while this one walks', async () => {
    const win = stubWindow()
    const session = await load({ shots: [], pending: null })
    hanging('p1')
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf())
    await waitFor(() => expect(savedRun(session)?.pending?.promptId).toBe('p1'))
    win.fire('storage', { key: KEY, newValue: null })
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
  })
})

describe('finding 13: old pins', () => {
  it('reads a pinned original back as its thumbnail', async () => {
    const { smallPreview } = await import('../src/components/reel/store')
    const { fileUrl } = await import('../src/lib/comfy')
    expect(smallPreview(fileUrl({ filename: 'a.png', subfolder: 'x', type: 'output' }))).toBe('/api/thumb?rel=x%2Fa.png&w=512')
    expect(smallPreview(fileUrl({ filename: 'a.png', subfolder: '', type: 'input' }))).toBe(fileUrl({ filename: 'a.png', subfolder: '', type: 'input' }))
    expect(smallPreview('/api/thumb?rel=a.png&w=512')).toBe('/api/thumb?rel=a.png&w=512')
  })
})

describe('finding 9: waiting for an idle ComfyUI through a takeover', () => {
  it('frees no memory and sends nothing once another tab took over', async () => {
    stubWindow()
    const session = await load({ shots: [], pending: null })
    let idle: (v: boolean) => void = () => {}
    m.wait.mockImplementation(() => new Promise((r) => { idle = r }))
    engine.reelRun.renderAll(['s1'], await planOf(['a']), await ctxOf(true))
    await waitFor(() => expect(m.wait).toHaveBeenCalled())
    const theirs = { shots: [], pending: null, press: { owner: 'other', beat: Date.now(), released: false, shotId: 's1' } }
    session.store.set(KEY, JSON.stringify(theirs))
    idle(true)
    await waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(m.release).not.toHaveBeenCalled()
    expect(m.run).not.toHaveBeenCalled()
    expect(savedRun(session)).toEqual(theirs)
  })
})
