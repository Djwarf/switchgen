import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Followed, OutputFile } from '../src/lib/comfy'
import type { HistoryEntry } from '../src/lib/history'
import type * as Desk from '../src/routes/Pictures'

/**
 * The Pictures press against a phone that reloads a tab it put in the
 * background: the picture ComfyUI has is kept for the tab and followed by the
 * next page, timed from when it began running, and a picture ComfyUI answers
 * from its cache is shown as the record it already has. ComfyUI is stood in
 * for; the desk, the tab's storage (a Map) and the archive are real.
 */
const m = vi.hoisted(() => ({
  run: vi.fn(),
  cancelJob: vi.fn(),
  followPrompt: vi.fn(),
  fetchPastRun: vi.fn(),
  listJobs: vi.fn(),
}))
vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
  followPrompt: m.followPrompt,
  fetchPastRun: m.fetchPastRun,
  listJobs: m.listJobs,
}))

const KEY = 'switchgen.picturesent.v1'
const LEFT_KEY = 'switchgen.picturesent.v1.left'
const V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const tab = new Map<string, string>()
const WAIT = { timeout: 5000, interval: 5 }

beforeEach(() => {
  for (const f of Object.values(m)) f.mockReset()
  m.fetchPastRun.mockResolvedValue(null)
  m.cancelJob.mockResolvedValue(true)
  tab.clear()
  vi.stubGlobal('sessionStorage', {
    getItem: (k: string) => tab.get(k) ?? null,
    setItem: (k: string, v: string) => void tab.set(k, String(v)),
    removeItem: (k: string) => void tab.delete(k),
  })
  vi.resetModules()
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

const composition = {
  desk: 'images', mode: 't2i', familyId: 'sdxl-illustrious', model: 'm.safetensors',
  prompt: 'a lighthouse at dusk', negative: null, seed: 1, steps: 20, cfg: 5,
  sampler: 'euler', scheduler: 'normal', width: 1024, height: 1024, noLora: false,
}
const filing = { composition, seed: 1, familyLabel: 'Illustrious', modelLabel: 'M', variant: null, passes: { face: false, hand: false, hires: false }, loras: [] }
const plan = (label = 'first') => ({ graph: {}, label, ...filing }) as unknown as Desk.RunPlan
const png = (filename: string, extra: Partial<OutputFile> = {}): OutputFile => ({ filename, subfolder: '', type: 'output', kind: 'image', ...extra })
/** A run ComfyUI queues as `promptId`, which ends when the test says. */
function queued(promptId: string) {
  let finish: (files: OutputFile[]) => void = () => {}
  let on: (e: unknown) => void = () => {}
  m.run.mockImplementationOnce((_graph: unknown, onProgress: (e: unknown) => void) => {
    on = onProgress
    onProgress({ phase: 'queued', promptId })
    return new Promise<OutputFile[]>((resolve) => {
      finish = resolve
    })
  })
  return { finish: (files: OutputFile[]) => finish(files), emit: (e: unknown) => on(e) }
}
/** A run ComfyUI queues as `promptId` and answers at once with `files`. */
const lands = (promptId: string, files: OutputFile[]) =>
  m.run.mockImplementationOnce((_graph: unknown, onProgress: (e: unknown) => void) => {
    onProgress({ phase: 'queued', promptId })
    return Promise.resolve(files)
  })
/**
 * Wait on the event loop, not on timers: vi.waitFor moves a faked clock on
 * each time it asks, and the clock is what a test of timing reads.
 */
async function settled(done: () => boolean) {
  for (let i = 0; i < 1000 && !done(); i++) await new Promise((resolve) => setImmediate(resolve))
  expect(done()).toBe(true)
}
const saved = async () => (await import('../src/routes/Pictures')).readSent(tab.get(KEY) ?? null)

describe('a picture sent from this page', () => {
  it('is kept for the tab once ComfyUI has it, and let go once it lands', async () => {
    const desk = await import('../src/routes/Pictures')
    const run = queued('p1')
    desk.startRuns([plan()])
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('queued'), WAIT)
    expect((await saved())?.jobs[0]?.promptId).toBe('p1')
    run.finish([png('one.png')])
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('done'), WAIT)
    expect(tab.has(KEY)).toBe(false)
  })

  it('is timed from when it began running, not from the press', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    vi.setSystemTime(100_000)
    const desk = await import('../src/routes/Pictures')
    const run = queued('p1')
    desk.startRuns([plan()])
    // The page asks first whether the queue on the server has come back.
    await vi.waitFor(() => expect(m.run).toHaveBeenCalledTimes(1), WAIT)
    // Ten seconds behind other work in ComfyUI's queue, then thirty ms to make.
    vi.setSystemTime(110_000)
    run.emit({ phase: 'running', node: null, value: 1, max: 20 })
    vi.setSystemTime(110_030)
    run.finish([png('one.png')])
    await settled(() => desk.pressSnapshot().current !== null)
    expect(desk.pressSnapshot().current!.durationMs).toBe(30)
    expect(m.fetchPastRun).not.toHaveBeenCalled()
  })

  it('takes ComfyUI\'s own times when it never heard the picture begin, and none when there are none', async () => {
    const desk = await import('../src/routes/Pictures')
    m.fetchPastRun.mockResolvedValueOnce({ startedAt: 1000, finishedAt: 4000 })
    lands('p1', [png('one.png')])
    desk.startRuns([plan()])
    await vi.waitFor(() => expect(desk.pressSnapshot().current?.file.filename).toBe('one.png'), WAIT)
    expect(desk.pressSnapshot().current!.durationMs).toBe(3000)
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('done'), WAIT)

    lands('p2', [png('two.png')])
    desk.startRuns([plan()])
    await vi.waitFor(() => expect(desk.pressSnapshot().current?.file.filename).toBe('two.png'), WAIT)
    expect(desk.pressSnapshot().current!.durationMs).toBe(0)
  })

  it('shows the record a picture ComfyUI answered from its cache already has, and files nothing', async () => {
    const desk = await import('../src/routes/Pictures')
    const history = await import('../src/lib/history')
    lands('p1', [png('c.png')])
    desk.startRuns([plan()])
    await vi.waitFor(() => expect(desk.pressSnapshot().current?.file.filename).toBe('c.png'), WAIT)
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('done'), WAIT)
    const first = desk.pressSnapshot().current!
    const count = history.all().length

    lands('p2', [png('c.png', { cached: true })])
    desk.startRuns([plan()])
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.promptId).toBe('p2'), WAIT)
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('done'), WAIT)
    expect(history.all()).toHaveLength(count)
    expect(desk.pressSnapshot().current!.id).toBe(first.id)
  })
})

describe('a picture the page before this one sent', () => {
  const left = (released: boolean) => ({
    writer: 'other',
    released,
    jobs: [{ id: 'j9', promptId: 'p9', startedAt: 5, label: 'first', ...filing }],
  })
  /** followPrompt as ComfyUI would answer it, ended when the test says. */
  function following() {
    let end: (r: Followed) => void = () => {}
    m.followPrompt.mockImplementation(() => new Promise<Followed>((resolve) => { end = resolve }))
    return (r: Followed) => end(r)
  }

  it('is followed as the press\'s own job when that page handed it on, and filed', async () => {
    const end = following()
    tab.set(KEY, JSON.stringify(left(true)))
    const desk = await import('../src/routes/Pictures')
    expect(desk.pressSnapshot().job).toMatchObject({ promptId: 'p9', status: 'queued' })
    const again = await saved()
    expect(again?.released).toBe(false)
    expect(again?.writer).not.toBe('other')
    expect(m.followPrompt).toHaveBeenCalledWith('p9', expect.anything())

    m.fetchPastRun.mockResolvedValue({ startedAt: 1000, finishedAt: 4000 })
    end({ status: 'done', files: [png('nine.png')] })
    await vi.waitFor(() => expect(desk.pressSnapshot().current?.file.filename).toBe('nine.png'), WAIT)
    expect(desk.pressSnapshot().current!.durationMs).toBe(3000)
    expect(tab.has(KEY)).toBe(false)
  })

  // A copied tab, a page that crashed, or an iPhone that closed the tab in
  // the background: the page that wrote it may still be following it, so it
  // is not followed on a guess, and not dropped without a word either.
  it('is not followed when that page did not hand it on, but put to the reader', async () => {
    following()
    tab.set(KEY, JSON.stringify(left(false)))
    const desk = await import('../src/routes/Pictures')
    expect(desk.pressSnapshot().job).toBeNull()
    expect(m.followPrompt).not.toHaveBeenCalled()
    expect(desk.pressSnapshot().left.map((j) => j.promptId)).toEqual(['p9'])
    expect(tab.has(KEY)).toBe(false)
    expect(desk.readSent(tab.get(LEFT_KEY) ?? null)?.jobs.map((j) => j.id)).toEqual(['j9'])
  })

  it('is followed from here and filed at the reader\'s word', async () => {
    const end = following()
    tab.set(KEY, JSON.stringify(left(false)))
    const desk = await import('../src/routes/Pictures')
    desk.followLeftSent()
    expect(m.followPrompt).toHaveBeenCalledWith('p9', expect.anything())
    expect(desk.pressSnapshot().left).toEqual([])
    expect(tab.has(LEFT_KEY)).toBe(false)
    // This page's own now, for a page after it to take up.
    expect((await saved())?.jobs.map((j) => j.id)).toEqual(['j9'])
    end({ status: 'done', files: [png('nine.png')] })
    await vi.waitFor(() => expect(desk.pressSnapshot().current?.file.filename).toBe('nine.png'), WAIT)
    expect(tab.has(KEY)).toBe(false)
  })

  it('is no longer offered once the reader leaves it, and still offered by a page loaded before they answer', async () => {
    following()
    tab.set(KEY, JSON.stringify(left(false)))
    let desk = await import('../src/routes/Pictures')
    // A reload before the reader answers: the next page offers it again.
    vi.resetModules()
    desk = await import('../src/routes/Pictures')
    expect(desk.pressSnapshot().left.map((j) => j.id)).toEqual(['j9'])
    expect(desk.pressSnapshot().job).toBeNull()
    expect(m.followPrompt).not.toHaveBeenCalled()
    desk.forgetLeftSent()
    expect(desk.pressSnapshot().left).toEqual([])
    expect(tab.has(LEFT_KEY)).toBe(false)
    expect(m.followPrompt).not.toHaveBeenCalled()
  })

  it('is stopped by Stop, and gone after a stop reads as stopped', async () => {
    const end = following()
    tab.set(KEY, JSON.stringify(left(true)))
    const desk = await import('../src/routes/Pictures')
    desk.stopPress()
    await vi.waitFor(() => expect(m.cancelJob).toHaveBeenCalledWith('p9'), WAIT)
    end({ status: 'lost' })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('cancelled'), WAIT)
  })
})

describe('a picture on its way to ComfyUI', () => {
  const entry = (extra: Record<string, unknown>) => ({
    writer: 'other',
    released: true,
    jobs: [{ id: 'j9', promptId: 'p9', startedAt: 5, label: 'second', ...filing, ...extra }],
  })
  /** A run that waits: it hands the test its progress callback and the id it was asked to send under. */
  function held() {
    const got: { on: (e: unknown) => void; sendAs: string | undefined; atSend: Desk.SentPicture | undefined } = {
      on: () => {},
      sendAs: undefined,
      atSend: undefined,
    }
    m.run.mockImplementationOnce((_graph: unknown, onProgress: (e: unknown) => void, opts?: { promptId?: string }) => {
      got.on = onProgress
      got.sendAs = opts?.promptId
      got.atSend = JSON.parse(tab.get(KEY) ?? '{"jobs":[]}').jobs[0]
      return new Promise<OutputFile[]>(() => {})
    })
    return got
  }

  it('is kept for the tab under the id it goes with, and where it stands in its batch, before it goes', async () => {
    const desk = await import('../src/routes/Pictures')
    const run = held()
    desk.startRuns([plan('one'), plan('two'), plan('three')])
    await vi.waitFor(() => expect(m.run).toHaveBeenCalled(), WAIT)
    expect(run.sendAs).toMatch(V4)
    expect(run.atSend).toMatchObject({ promptId: run.sendAs, sending: true, index: 1, total: 3 })
    run.on({ phase: 'queued', promptId: run.sendAs })
    const kept = (await saved())!.jobs[0]!
    expect(kept).toMatchObject({ promptId: run.sendAs, index: 1, total: 3 })
    expect('sending' in kept).toBe(false)
  })

  it('is stopped the moment ComfyUI names it when Stop came first, and not kept for a later page', async () => {
    const desk = await import('../src/routes/Pictures')
    const run = held()
    desk.startRuns([plan()])
    await vi.waitFor(() => expect(m.run).toHaveBeenCalled(), WAIT)
    expect(tab.has(KEY)).toBe(true)
    desk.stopPress()
    run.on({ phase: 'queued', promptId: run.sendAs })
    expect(tab.has(KEY)).toBe(false)
    await vi.waitFor(() => expect(m.cancelJob).toHaveBeenCalledWith(run.sendAs), WAIT)
  })

  it('is shown as not sent when the page before went while sending it and ComfyUI never had it', async () => {
    let end: (r: Followed) => void = () => {}
    m.followPrompt.mockImplementation(() => new Promise<Followed>((resolve) => { end = resolve }))
    tab.set(KEY, JSON.stringify(entry({ sending: true, index: 2, total: 3 })))
    const desk = await import('../src/routes/Pictures')
    // Not queued until ComfyUI shows it has it, and one of one here.
    expect(desk.pressSnapshot().job).toMatchObject({ promptId: 'p9', status: 'submitting', index: 1, total: 1 })
    end({ status: 'lost' })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('error'), WAIT)
    expect(desk.pressSnapshot().fault?.message).toContain('still sending this picture')
    expect(desk.pressSnapshot().fault?.message).not.toContain('restarted')
    expect(m.run).not.toHaveBeenCalled()
    expect(tab.has(KEY)).toBe(false)
  })

  it('is lost to a restart, not unsent, once ComfyUI had it', async () => {
    let end: (r: Followed) => void = () => {}
    m.followPrompt.mockImplementation((_id: string, opts: { onState?: (s: 'queued' | 'running') => void }) => {
      opts.onState?.('queued')
      return new Promise<Followed>((resolve) => { end = resolve })
    })
    tab.set(KEY, JSON.stringify(entry({ sending: true })))
    const desk = await import('../src/routes/Pictures')
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('queued'), WAIT)
    end({ status: 'lost' })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('error'), WAIT)
    expect(desk.pressSnapshot().fault?.message).toContain('usually means it restarted')
  })

  it('drops the on-its-way mark from the saved entry once ComfyUI has it, so the next page follows a picture that got there', async () => {
    m.followPrompt.mockImplementation((_id: string, opts: { onState?: (s: 'queued' | 'running') => void }) => {
      opts.onState?.('running')
      return new Promise<Followed>(() => {})
    })
    tab.set(KEY, JSON.stringify(entry({ sending: true })))
    const desk = await import('../src/routes/Pictures')
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('running'), WAIT)
    const kept = JSON.parse(tab.get(KEY) ?? '{"jobs":[]}').jobs[0]
    expect(kept).toMatchObject({ id: 'j9', promptId: 'p9' })
    expect('sending' in kept).toBe(false)
  })

  it('says what the page before never sent of its batch, until a new press', async () => {
    let end: (r: Followed) => void = () => {}
    m.followPrompt.mockImplementation(() => new Promise<Followed>((resolve) => { end = resolve }))
    tab.set(KEY, JSON.stringify(entry({ index: 2, total: 3 })))
    const desk = await import('../src/routes/Pictures')
    expect(desk.pressSnapshot().unsent).toContain('Picture 3 was never sent')
    end({ status: 'cancelled' })
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.status).toBe('cancelled'), WAIT)
    lands('p2', [png('two.png')])
    desk.startRuns([plan()])
    expect(desk.pressSnapshot().unsent).toBeNull()
  })

  it('names the rest of a batch in a line, and nothing for the last of one or a place not known', async () => {
    const { batchRestLine } = await import('../src/routes/Pictures')
    expect(batchRestLine(1, 3)).toContain('Pictures 2 and 3 were never sent')
    expect(batchRestLine(2, 5)).toContain('Pictures 3 to 5 were never sent')
    expect(batchRestLine(2, 3)).toContain('Picture 3 was never sent')
    expect(batchRestLine(3, 3)).toBeNull()
    expect(batchRestLine(undefined, undefined)).toBeNull()
  })
})

describe('the screen through a batch', () => {
  it('is let sleep once ComfyUI has the last picture, not when that picture lands', async () => {
    vi.stubGlobal('navigator', { wakeLock: { request: async () => ({ released: false, release: async () => {}, addEventListener() {} }) } })
    const desk = await import('../src/routes/Pictures')
    const { awakeReasons } = await import('../src/lib/wakeLock')
    const first = queued('p1')
    queued('p2')
    desk.startRuns([plan('one'), plan('two')])
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.promptId).toBe('p1'), WAIT)
    // The second picture still waits in the page.
    expect(awakeReasons()).toEqual(['pictures batch'])
    first.finish([png('one.png')])
    await vi.waitFor(() => expect(desk.pressSnapshot().job?.promptId).toBe('p2'), WAIT)
    // In ComfyUI's hands and still rendering, with nothing left here to send.
    expect(desk.pressSnapshot().job?.status).toBe('queued')
    expect(awakeReasons()).toEqual([])
  })
})

describe('asking how many jobs are ahead', () => {
  it('never asks again while the last ask is unanswered', async () => {
    vi.useFakeTimers()
    m.listJobs.mockImplementation(() => new Promise(() => {}))
    const desk = await import('../src/routes/Pictures')
    const stop = desk.watchAhead(() => {}, 5000)
    await vi.advanceTimersByTimeAsync(30_000)
    expect(m.listJobs).toHaveBeenCalledTimes(1)
    stop()
  })
})

describe('the desk\'s pure parts', () => {
  const rec = (id: string, over: Partial<HistoryEntry>) => ({ id, desk: 'images', kind: 'image', file: png(`${id}.png`), ...over }) as HistoryEntry

  it('times a picture only from a start it saw', async () => {
    const { ranFor } = await import('../src/routes/Pictures')
    expect(ranFor(null, 5)).toBe(0)
    expect(ranFor(10, 25)).toBe(15)
  })

  it('matches a record only to a file ComfyUI says came from its cache', async () => {
    const { cachedRecordFor } = await import('../src/routes/Pictures')
    const r = rec('c', { file: png('c.png') })
    expect(cachedRecordFor(png('c.png', { cached: true }), [r])).toBe(r)
    expect(cachedRecordFor(png('c.png'), [r])).toBeNull()
  })

  it('puts the newest picture of this desk still on disk on an empty plate', async () => {
    const { plateFallback } = await import('../src/routes/Pictures')
    const a = rec('a', {})
    const list = [rec('v', { desk: 'video', kind: 'video' }), rec('gone', { missing: true }), a, rec('b', {})]
    expect(plateFallback(list)).toBe(a)
  })

  it('reads the desk\'s held settings back field by field, and drops what cannot be read', async () => {
    const { parseHeld } = await import('../src/routes/Pictures')
    const { INTENTS } = await import('../src/lib/intent')
    const { ANATOMY_LEVELS } = await import('../src/lib/recipe')
    const good = { look: INTENTS[0]!.id, anatomy: ANATOMY_LEVELS[0]!.id, mode: 't2i', seed0: 7, overrides: { steps: 'x', cfg: 4 } }
    const held = parseHeld(JSON.stringify(good))
    expect(held).toMatchObject({ look: good.look, anatomy: good.anatomy, mode: 't2i', seed0: 7 })
    expect(held!.overrides).toEqual({ cfg: 4 })
    expect(parseHeld(JSON.stringify({ ...good, look: 'nope' }))).toBeNull()
    expect(parseHeld(JSON.stringify({ ...good, mode: 'sideways' }))).toBeNull()
    expect(parseHeld(JSON.stringify({ ...good, seed0: -1 }))).toBeNull()
    expect(parseHeld('{')).toBeNull()
  })

  it('keeps only overrides of the right kind', async () => {
    const { sanitiseOverrides } = await import('../src/components/advanced/overrides')
    const out = sanitiseOverrides({
      steps: '20',
      cfg: 5,
      sampler: 3,
      loras: [{ file: 'a', strength: 0.5, enabled: true }, { file: 1 }],
      passes: { face: true, hand: false, hires: false },
    })
    expect(out).toEqual({ cfg: 5, loras: [{ file: 'a', strength: 0.5, enabled: true }], passes: { face: true, hand: false, hires: false } })
  })
})
