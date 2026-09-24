import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Desk from '../src/routes/Pictures'

/**
 * The Pictures desk with the queue's real store and the real ask of whether
 * it runs, over a stand-in fetch. A page loaded while the queue was off, with
 * nothing in it, has no stream open, so nothing tells it when the queue comes
 * back: the desk asks at each press rather than going by the store's last
 * word. Only the hand-over, following a job and ComfyUI's run are stood in;
 * nothing is sent anywhere.
 */
const m = vi.hoisted(() => ({ run: vi.fn(), submitGroup: vi.fn(), follow: vi.fn() }))
vi.mock('../src/lib/comfy', async (importOriginal) => ({ ...(await importOriginal<typeof import('../src/lib/comfy')>()), run: m.run }))
vi.mock('../src/lib/runner', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/runner')>()),
  submitGroup: m.submitGroup,
  follow: m.follow,
}))

const OFF = 'The queue is off.'
const tab = new Map<string, string>()
/** Whether the server's queue runs, as both its capabilities and its state say. */
let on = false
const json = (x: unknown) => new Response(JSON.stringify(x), { status: 200, headers: { 'content-type': 'application/json' } })

beforeEach(() => {
  for (const f of [m.run, m.submitGroup, m.follow]) f.mockReset()
  on = false
  tab.clear()
  vi.stubGlobal('sessionStorage', {
    getItem: (k: string) => tab.get(k) ?? null,
    setItem: (k: string, v: string) => void tab.set(k, String(v)),
    removeItem: (k: string) => void tab.delete(k),
  })
  vi.stubGlobal('fetch', async (url: string) => {
    if (url === '/api/capabilities') return json({ runner: on, runnerDesks: on ? ['images'] : [], runnerReason: on ? null : OFF })
    if (url === '/api/runner') return json({ v: 1, boot: 'b1', rev: 1, available: on, reason: on ? null : OFF, jobs: [], groups: [], lane: { held: null } })
    return new Response('no', { status: 404 })
  })
  vi.resetModules()
})
afterEach(() => vi.unstubAllGlobals())

const composition = {
  desk: 'images', mode: 't2i', familyId: 'sdxl-illustrious', model: 'm.safetensors',
  prompt: 'a lighthouse', negative: null, seed: 1, steps: 20, cfg: 5,
  sampler: 'euler', scheduler: 'normal', width: 1024, height: 1024, noLora: false,
}
const plan = (label: string) =>
  ({ graph: { '3': { class_type: 'KSampler', inputs: {} } }, label, composition, seed: 1, familyLabel: 'I', modelLabel: 'M', variant: null, passes: { face: false, hand: false, hires: false }, loras: [] }) as unknown as Desk.RunPlan

/** The page as loaded while the queue was off: capabilities and state read, and no stream open. */
async function loadedWhileOff() {
  const caps = await import('../src/lib/capabilities')
  const runner = await import('../src/lib/runner')
  await caps.serverCapabilities()
  runner.runnerStore.start()
  await runner.runnerStore.refresh()
  expect(runner.runnerStore.snapshot()).toMatchObject({ boot: 'b1', available: false, connected: false })
  return { runner, desk: await import('../src/routes/Pictures') }
}

describe('a queue that was off when the page loaded', () => {
  it('is handed the next batch once it is back, with no word to the page', async () => {
    const { runner, desk } = await loadedWhileOff()
    on = true
    m.submitGroup.mockResolvedValue({ ok: true, replayed: false, group: {}, jobs: [] })
    m.follow.mockImplementation(() => new Promise(() => {}))
    m.run.mockImplementation(() => new Promise(() => {}))
    await desk.startRuns([plan('one'), plan('two')])
    expect(m.submitGroup).toHaveBeenCalledTimes(1)
    expect(m.run).not.toHaveBeenCalled()
    expect(runner.runnerStore.snapshot().available).toBe(true)
  })

  it('has the batch sent from the page while it is still off', async () => {
    const { desk } = await loadedWhileOff()
    m.run.mockImplementation(() => new Promise(() => {}))
    await desk.startRuns([plan('one'), plan('two')])
    await vi.waitFor(() => expect(m.run).toHaveBeenCalledTimes(1), { timeout: 5000, interval: 5 })
    expect(m.submitGroup).not.toHaveBeenCalled()
    expect(desk.pressSnapshot().job?.runner).toBeUndefined()
  })
})
