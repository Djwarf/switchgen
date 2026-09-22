import { beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Engine from '../src/components/reel/engine'

// A shot left on the press by another tab or an earlier page, saved as the
// reel's pending entry, and what this page does with it on load.
const m = vi.hoisted(() => ({ run: vi.fn(), cancelJob: vi.fn(), getJob: vi.fn(), fetchPastRun: vi.fn() }))
vi.mock('../src/lib/comfy', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/comfy')>()),
  run: m.run,
  cancelJob: m.cancelJob,
  getJob: m.getJob,
  fetchPastRun: m.fetchPastRun,
}))

const KEY = 'switchgen.reelrun.v1'
const made = { signature: 's', seed: 1, openedOn: null, frames: 17, fps: 24, width: 832, height: 480 }
const pendingOf = (extra: Record<string, unknown>) => ({
  shotId: 'shot2',
  promptId: 'p-left',
  label: 'Shot 2',
  order: ['shot1', 'shot2'],
  startedAt: Date.now() - 1000,
  made,
  frames: 17,
  composition: { kind: 'video' },
  familyLabel: 'x',
  modelLabel: 'y',
  ...extra,
})

let engine: typeof Engine
/** Save a run as another page left it, then load the engine as this page does. */
async function load(saved: unknown) {
  vi.resetModules()
  const session = await import('../src/lib/session')
  session.store.set(KEY, JSON.stringify(saved))
  engine = await import('../src/components/reel/engine')
  return session
}
beforeEach(() => {
  for (const f of Object.values(m)) f.mockReset()
})

describe('one tab follows the shot on the press', () => {
  it('leaves a shot another live tab follows alone, and keeps its entry through a save', async () => {
    const session = await load({ shots: [], pending: pendingOf({ owner: 'other', beat: Date.now(), released: false }) })
    expect(engine.reelRun.busy()).toBe(false)
    expect(engine.reelRun.snapshot().status).toBe('idle')
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(m.getJob).not.toHaveBeenCalled()
    engine.reelRun.clear()
    expect(JSON.parse(session.store.get(KEY)!).pending.owner).toBe('other')
  })

  it('picks up a shot its page released on the way out', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'in_progress' })
    await load({ shots: [], pending: pendingOf({ owner: 'other', beat: Date.now(), released: true }) })
    expect(engine.reelRun.busy()).toBe(true)
    expect(engine.reelRun.snapshot().states.shot2?.status).toBe('running')
  })

  it('picks up an entry saved before entries were stamped', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'in_progress' })
    await load({ shots: [], pending: pendingOf({}) })
    expect(engine.reelRun.busy()).toBe(true)
  })
})

describe('stopping a picked-up shot', () => {
  it('reads Stopped when the stop takes it out of the queue', async () => {
    let gone = false
    m.getJob.mockImplementation(async () => (gone ? null : { id: 'p-left', status: 'pending' }))
    m.cancelJob.mockImplementation(async () => {
      gone = true
      return true
    })
    await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true }) })
    await vi.waitFor(() => expect(m.getJob).toHaveBeenCalled())
    engine.reelRun.stop()
    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false), { timeout: 10_000 })
    const run = engine.reelRun.snapshot()
    expect(run.status).toBe('stopped')
    expect(run.states.shot2?.status).toBe('stopped')
    expect(run.note).toBe('Shot 2 was stopped.')
  }, 15_000)

  it('files a finished picked-up shot from its own history record', async () => {
    m.getJob.mockResolvedValue({ id: 'p-left', status: 'completed' })
    m.fetchPastRun.mockResolvedValue({
      promptId: 'p-left',
      graph: {},
      status: 'success',
      startedAt: 1,
      finishedAt: 2,
      clientId: null,
      error: null,
      files: [{ filename: 'c.webm', subfolder: 'reel', type: 'output', kind: 'video' }],
    })
    await load({ shots: [], pending: pendingOf({ owner: 'old', beat: 0, released: true }) })
    await vi.waitFor(() => expect(engine.reelRun.busy()).toBe(false))
    expect(m.fetchPastRun).toHaveBeenCalledWith('p-left')
    expect(engine.reelRun.snapshot().states.shot2?.status).toBe('done')
  })
})
