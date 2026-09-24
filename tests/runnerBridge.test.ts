import { readFileSync } from 'node:fs'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as RunnerMod from '../src/lib/runner'

/**
 * The section bar and the queue on the server. App.tsx's fourth bridge
 * reports the queue's jobs from every device, so each shows once in the
 * ledger, counts as SwitchGen's own and not someone else's work in ComfyUI's
 * queue, and stops the way its desk stops. The queue's store is stood in, so
 * the bridge reads what a test sets; the ledger and the mirror are real, and
 * ComfyUI's queue is a route table.
 */

const m = vi.hoisted(() => {
  const listeners = new Set<() => void>()
  let snap: any = null
  return {
    listeners,
    get snap() {
      return snap
    },
    set(next: any) {
      snap = { ...snap, ...next }
      for (const fn of [...listeners]) fn()
    },
    reset(fresh: any) {
      snap = fresh
      listeners.clear()
    },
    stopGroup: vi.fn(async () => true),
    stopJob: vi.fn(async () => true),
  }
})

vi.mock('../src/lib/runner', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/runner')>()),
  runnerStore: {
    subscribe: (fn: () => void) => {
      m.listeners.add(fn)
      return () => m.listeners.delete(fn)
    },
    snapshot: () => m.snap,
    start: () => {},
    refresh: async () => {},
  },
  stopGroup: m.stopGroup,
  stopJob: m.stopJob,
}))

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
/** How many jobs ComfyUI says it is running and has waiting, as its jobs API counts them. */
let comfyQueue = { running: 0, pending: 0 }

const fresh = () => ({ v: 1, available: true, reason: null, boot: 'b', rev: 1, comfy: { answering: true, since: 0 }, lane: { held: null }, groups: [], jobs: [], progress: {}, connected: true })

function job(over: Partial<RunnerMod.RunnerJob>): RunnerMod.RunnerJob {
  return {
    id: 'j', groupId: 'g', desk: 'images', kind: 'image', seq: 1, index: 1, total: 1, label: 'A picture', prompt: 'p', device: 'another-browser',
    heavy: false, status: 'waiting', wait: { for: 'turn' }, stopRequested: false, stopLanded: false, promptId: null, attempt: 0,
    createdAt: Date.now() - 1000, sentAt: null, ranAt: null, finishedAt: null, endedAt: null, files: [], primary: null, frame: null,
    openedOn: null, entryId: null, entryNo: null, repeatOf: null, durationMs: 0, error: null, meta: null, dismissed: false, ...over,
  }
}
function group(id: string, kind: RunnerMod.RunnerGroup['kind'], jobIds: string[], desk: RunnerMod.RunnerGroup['desk'] = 'images'): RunnerMod.RunnerGroup {
  return { id, desk, kind, label: 'x', device: 'another-browser', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds, dismissed: false }
}

let app: typeof import('../src/App')
let ledger: typeof import('../src/components/shell/jobs').jobs
let mirror: typeof import('../src/components/shell/mirror').mirror

beforeEach(async () => {
  m.reset(fresh())
  m.stopGroup.mockClear()
  m.stopJob.mockClear()
  comfyQueue = { running: 0, pending: 0 }
  vi.stubGlobal('fetch', async (url: string) => {
    if (url.startsWith('/comfy/api/jobs?')) {
      const q = new URLSearchParams(url.split('?')[1])
      const total = q.get('status') === 'in_progress' ? comfyQueue.running : comfyQueue.pending
      return json({ jobs: [], pagination: { offset: 0, limit: 1, total, has_more: false } })
    }
    return json({ error: 'no route' }, 599)
  })
  vi.resetModules()
  app = await import('../src/App')
  ledger = (await import('../src/components/shell/jobs')).jobs
  mirror = (await import('../src/components/shell/mirror')).mirror
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

const reportsOf = (desk: 'images' | 'video' | 'reel') => app.runnerBridgeOf(desk).read()

describe('the queue\'s bridges', () => {
  it('report another browser\'s work as rows of the ledger, each on its own desk\'s bridge', () => {
    m.set({
      groups: [group('g1', 'batch', ['p1']), group('g2', 'clips', ['c1'], 'video')],
      jobs: [job({ id: 'p1', groupId: 'g1', status: 'queued', promptId: 'pp1' }), job({ id: 'c1', groupId: 'g2', desk: 'video', kind: 'video', heavy: true })],
    })
    expect(reportsOf('images').map((r) => r.key)).toEqual(['p1'])
    expect(reportsOf('video').map((r) => r.key)).toEqual(['c1'])
    expect(reportsOf('reel')).toEqual([])
    const bridge = app.runnerBridgeOf('images')
    expect(bridge).toMatchObject({ desk: 'images', kind: 'image' })
    expect(app.runnerBridgeOf('reel')).toMatchObject({ desk: 'reel', kind: 'video' })
    const stop = mirror(bridge)
    const row = ledger.get(bridge.seen.get('p1')!)
    expect(row).toMatchObject({ desk: 'images', kind: 'image', status: 'queued', promptId: 'pp1', label: 'A picture' })
    stop()
  })

  it('report only the picture or shot in hand of a batch or pass, and every clip', () => {
    m.set({
      groups: [group('g1', 'batch', ['p1', 'p2', 'p3']), group('g2', 'clips', ['c1', 'c2'], 'video')],
      jobs: [
        job({ id: 'p1', groupId: 'g1', status: 'queued', promptId: 'x' }),
        job({ id: 'p2', groupId: 'g1', wait: { for: 'before' } }),
        job({ id: 'p3', groupId: 'g1', wait: { for: 'before' } }),
        job({ id: 'c1', groupId: 'g2', desk: 'video', kind: 'video' }),
        job({ id: 'c2', groupId: 'g2', desk: 'video', kind: 'video' }),
      ],
    })
    expect(reportsOf('images').map((r) => r.key)).toEqual(['p1'])
    expect(reportsOf('video').map((r) => r.key)).toEqual(['c1', 'c2'])
  })

  it('count the queue\'s work as SwitchGen\'s own once ComfyUI has it, so nothing shows as someone else\'s', async () => {
    comfyQueue = { running: 1, pending: 0 }
    m.set({ groups: [group('g1', 'clips', ['c1'], 'video')], jobs: [job({ id: 'c1', groupId: 'g1', desk: 'video', kind: 'video', status: 'running', promptId: 'pc1' })] })
    const stop = mirror(app.runnerBridgeOf('video'))
    const off = ledger.subscribe(() => {})
    await vi.waitFor(() => expect(ledger.snapshot().server.known).toBe(true), { timeout: 5000, interval: 5 })
    expect(ledger.snapshot().server).toMatchObject({ running: 1, foreign: 0 })
    off()
    stop()
  })

  it('route the bar\'s Stop to the whole batch or pass, and to the one clip', async () => {
    m.set({
      groups: [group('g1', 'batch', ['p1']), group('g2', 'clips', ['c1'], 'video'), group('g3', 'pass', ['s1'], 'reel')],
      jobs: [
        job({ id: 'p1', groupId: 'g1', status: 'queued', promptId: 'a' }),
        job({ id: 'c1', groupId: 'g2', desk: 'video', status: 'queued', promptId: 'b' }),
        job({ id: 's1', groupId: 'g3', desk: 'reel', status: 'queued', promptId: 'c' }),
      ],
    })
    const stops = (['images', 'video', 'reel'] as const).map((d) => mirror(app.runnerBridgeOf(d)))
    await ledger.cancel(app.runnerBridgeOf('images').seen.get('p1')!)
    await ledger.cancel(app.runnerBridgeOf('video').seen.get('c1')!)
    await ledger.cancel(app.runnerBridgeOf('reel').seen.get('s1')!)
    expect(m.stopGroup.mock.calls).toEqual([['g1'], ['g3']])
    expect(m.stopJob.mock.calls).toEqual([['c1']])
    // A job the queue no longer lists is stopped by its own id.
    app.runnerBridgeOf('images').stop!('gone')
    expect(m.stopJob).toHaveBeenLastCalledWith('gone')
    for (const s of stops) s()
  })

  it('give Stop back, and say so, when the word to the server did not get through', async () => {
    const { renderToString } = await import('react-dom/server')
    const { createElement } = await import('react')
    const notice = await import('../src/components/shell/Notice')
    const rail = () => renderToString(createElement(notice.NoticeRail))
    m.stopGroup.mockResolvedValueOnce(false)
    m.stopJob.mockResolvedValueOnce(false)
    // A batch's head held on the server, and a clip waiting its turn: neither
    // has a prompt id, and no step will come to say the stop did not take.
    m.set({
      groups: [group('g1', 'batch', ['p1', 'p2']), group('g2', 'clips', ['c1'], 'video')],
      jobs: [
        job({ id: 'p1', groupId: 'g1', wait: { for: 'held' } }),
        job({ id: 'p2', groupId: 'g1', wait: { for: 'before' } }),
        job({ id: 'c1', groupId: 'g2', desk: 'video', kind: 'video', heavy: true, wait: { for: 'held' } }),
      ],
      lane: { held: { why: 'restart', scope: 'all', jobId: null, since: 1 } },
    })
    const pictures = app.runnerBridgeOf('images')
    const clips = app.runnerBridgeOf('video')
    const stops = [mirror(pictures), mirror(clips)]
    const p1 = pictures.seen.get('p1')!
    await ledger.cancel(p1)
    expect(m.stopGroup).toHaveBeenCalledWith('g1')
    expect(ledger.get(p1)).toMatchObject({ status: 'submitting', cancelling: false })
    expect(rail()).toContain('Could not stop that job')
    const c1 = clips.seen.get('c1')!
    await ledger.cancel(c1)
    expect(m.stopJob).toHaveBeenCalledWith('c1')
    expect(ledger.get(c1)).toMatchObject({ status: 'submitting', cancelling: false })
    // The word's own answer goes back to the ledger.
    const told = clips.stop!('c1')
    expect(told).toBeInstanceOf(Promise)
    expect(await told).toBe(true)
    for (const s of stops) s()
  })

  it('give a job its ending however long ago it came, while the ledger still shows it live', () => {
    m.set({ groups: [group('g1', 'clips', ['c1'], 'video')], jobs: [job({ id: 'c1', groupId: 'g1', desk: 'video', kind: 'video', status: 'running', promptId: 'p' })] })
    const bridge = app.runnerBridgeOf('video')
    const stop = mirror(bridge)
    const row = bridge.seen.get('c1')!
    expect(ledger.get(row)!.status).toBe('running')
    // The phone slept through the ending: an hour later it wakes.
    const long = Date.now() - 3_600_000
    m.set({ jobs: [job({ id: 'c1', groupId: 'g1', desk: 'video', kind: 'video', status: 'done', promptId: 'p', entryId: 'e1', endedAt: long, finishedAt: long })] })
    expect(ledger.get(row)!.status).toBe('done')
    expect(ledger.get(row)!.entryId).toBe('e1')
    stop()
  })

  it('list a job that ended just now but was never seen live, without announcing it; and leave out one put away', () => {
    const now = Date.now()
    m.set({
      groups: [group('g1', 'clips', ['c1', 'c2', 'c3'], 'video')],
      jobs: [
        job({ id: 'c1', groupId: 'g1', desk: 'video', status: 'done', promptId: 'p', endedAt: now - 2000 }),
        job({ id: 'c2', groupId: 'g1', desk: 'video', status: 'failed', promptId: 'q', endedAt: now - 2000, dismissed: true }),
        job({ id: 'c3', groupId: 'g1', desk: 'video', status: 'done', promptId: 'r', endedAt: now - 60_000 }),
      ],
    })
    expect(reportsOf('video').map((r) => r.key)).toEqual(['c1'])
    const bridge = app.runnerBridgeOf('video')
    const before = ledger.snapshot().jobs.length
    const stop = mirror(bridge)
    expect(bridge.seen.get('c1')).toBe('')
    expect(ledger.snapshot().jobs.length).toBe(before)
    stop()
  })
})

describe('the bar while the queue on the server is off', () => {
  const OFF = 'Turned off with SWITCHGEN_RUNNER=off.'
  const drawn = async (name: 'slug' | 'rail') => {
    const { renderToString } = await import('react-dom/server')
    const { createElement } = await import('react')
    if (name === 'slug') return renderToString(createElement((await import('../src/components/shell/RunningSlug')).RunningSlug))
    return renderToString(createElement((await import('../src/components/shell/Notice')).NoticeRail))
  }
  /** The classes of the slug's row: the job's line and Stop, or the reason in its place. */
  const slugRow = (html: string) => /<div class="(flex min-w-0 items-center gap-3[^"]*)">/.exec(html)?.[1]?.split(' ')
  /** A batch whose first picture waits on the server. */
  const parked = (over: Partial<RunnerMod.RunnerJob> = {}) => ({ groups: [group('g1', 'batch', ['p1'])], jobs: [job({ id: 'p1', groupId: 'g1', ...over })] })

  it('offers no Stop for work parked on it, and says why in its place, until the queue runs again', async () => {
    m.set({ available: false, reason: OFF, ...parked() })
    const bridge = app.runnerBridgeOf('images')
    const off = mirror(bridge)
    const id = bridge.seen.get('p1')!
    expect(ledger.get(id)!.noStop).toBe(`${OFF} It can be stopped once the queue on the server is running again.`)
    const slug = await drawn('slug')
    expect(slug).toContain(OFF)
    expect(slug).not.toContain('Hold to stop')
    // Below a wide screen the reason takes a line of its own and wraps there
    // in full, not cut to a few words with the rest in a title a touch
    // screen cannot open; only on a wide screen is it cut short.
    const reason = /<span class="([^"]*)" title="[^"]*">/.exec(slug)?.[1]?.split(' ')
    expect(reason).toContain('max-lg:basis-full')
    expect(reason).toContain('lg:truncate')
    expect(reason).not.toContain('truncate')
    expect(slugRow(slug)).toContain('max-lg:flex-wrap')
    // Nor is a stop sent from the ledger, which the server would refuse.
    await ledger.cancel(id)
    expect(m.stopGroup).not.toHaveBeenCalled()
    expect(m.stopJob).not.toHaveBeenCalled()
    expect(ledger.get(id)!.cancelling).toBe(false)
    m.set({ available: true, reason: null })
    expect(ledger.get(id)!.noStop).toBeNull()
    const running = await drawn('slug')
    expect(running).toContain('Hold to stop')
    // With Stop back, it stays on the job's own line.
    expect(slugRow(running)).not.toContain('max-lg:flex-wrap')
    off()
  })

  it('says a stop the server already holds goes through once the queue runs again', () => {
    m.set({ available: false, reason: null, ...parked({ stopRequested: true }) })
    const bridge = app.runnerBridgeOf('images')
    const off = mirror(bridge)
    expect(ledger.get(bridge.seen.get('p1')!)!.noStop).toBe('The queue on the server is not running. The stop goes through once the queue on the server is running again.')
    off()
  })

  it('takes back a notice that a stop did not get through once there is no Stop to hold again', async () => {
    m.stopGroup.mockResolvedValueOnce(false)
    m.set(parked())
    const bridge = app.runnerBridgeOf('images')
    const off = mirror(bridge)
    const id = bridge.seen.get('p1')!
    await ledger.cancel(id)
    expect(m.stopGroup).toHaveBeenCalledWith('g1')
    expect(await drawn('rail')).toContain('Could not stop that job')
    m.set({ available: false, reason: 'Another SwitchGen server holds the archive, and the queue with it.' })
    expect(await drawn('rail')).not.toContain('Could not stop that job')
    off()
  })

  it('posts no such notice for a stop that was out when the queue went off', async () => {
    let answer: (v: boolean) => void = () => {}
    m.stopGroup.mockImplementationOnce(() => new Promise<boolean>((res) => (answer = res)))
    m.set(parked())
    const bridge = app.runnerBridgeOf('images')
    const off = mirror(bridge)
    const id = bridge.seen.get('p1')!
    const asked = ledger.cancel(id)
    m.set({ available: false, reason: OFF })
    answer(false)
    await asked
    expect(await drawn('rail')).not.toContain('Could not stop that job')
    expect(ledger.get(id)!.cancelling).toBe(false)
    off()
  })

  it('says nothing in Stop\'s place for work that has ended, nor before the server has said where it stands', () => {
    m.set({ available: false, reason: OFF, groups: [group('g1', 'clips', ['c1'], 'video')], jobs: [job({ id: 'c1', groupId: 'g1', desk: 'video', kind: 'video', status: 'done', promptId: 'p', endedAt: Date.now() - 1000 })] })
    expect(reportsOf('video')).toMatchObject([{ key: 'c1', noStop: null }])
    m.set({ boot: '', available: false, reason: null, ...parked() })
    expect(reportsOf('images')).toMatchObject([{ key: 'p1', noStop: null }])
  })
})

describe('the desks\' own bridges', () => {
  // They are not exported, and mount only once a desk's code has loaded in a
  // page, which this suite has no DOM to do; the page is read instead.
  const source = readFileSync(path.resolve(import.meta.dirname, '..', 'src', 'App.tsx'), 'utf8')
  const body = (name: string) => new RegExp(`function ${name}\\([\\s\\S]*?\\n\\}\\n`).exec(source)?.[0] ?? ''

  it('leave the queue\'s work to its own bridge, so nothing is reported twice', () => {
    expect(body('pictureBridgeOf')).toContain('if (!job || job.runner) return []')
    expect(body('videoBridgeOf')).toContain('.filter((job) => !job.runner)')
    expect(body('reelBridgeOf')).toContain('if (run.runnerGroupId) return []')
  })

  it('mount one queue bridge per desk beside them', () => {
    expect(source).toContain('for (const desk of RUNNER_DESKS) stops.push(mirror(runnerBridgeOf(desk)))')
    expect(source).toMatch(/const RUNNER_DESKS = \['images', 'video', 'reel'\] as const/)
  })
})
