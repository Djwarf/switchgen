import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as Capabilities from '../src/lib/capabilities'
import type * as Vision from '../src/lib/vision'

/**
 * Asking the local server what it can do, while it restarts. Both probes keep
 * their answer in module state, so each test loads them fresh, and the clock
 * is faked so the hold on a failure can be stepped past.
 */
let caps: typeof Capabilities
let vision: typeof Vision
let fetchStub: ReturnType<typeof vi.fn>

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
const refuse = () => Promise.reject(new TypeError('fetch failed'))

beforeEach(async () => {
  vi.useFakeTimers()
  fetchStub = vi.fn(refuse)
  vi.stubGlobal('fetch', fetchStub)
  vi.resetModules()
  caps = await import('../src/lib/capabilities')
  vision = await import('../src/lib/vision')
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('the server probe', () => {
  it('asks again after a failure, passes the real answer on, and stops asking once it has one', async () => {
    const heard: Capabilities.ServerCapabilities[] = []
    caps.askUntilAnswered(caps.serverCapabilities, (c) => c.reason !== null, (c) => heard.push(c))
    await vi.advanceTimersByTimeAsync(0)
    expect(heard).toHaveLength(1)
    expect(heard[0]!.downloads).toBe(false)

    fetchStub.mockImplementation(async () => json({ downloads: true, tools: { aria2c: '/usr/bin/aria2c' } }))
    await vi.advanceTimersByTimeAsync(caps.RETRY_FAILED_MS)
    expect(fetchStub).toHaveBeenCalledTimes(2)
    expect(heard).toHaveLength(2)
    expect(heard[1]!.downloads).toBe(true)
    expect(heard[1]!.reason).toBeNull()

    await vi.advanceTimersByTimeAsync(3 * caps.RETRY_FAILED_MS)
    expect(fetchStub).toHaveBeenCalledTimes(2)
  })

  it('stops asking when the one listening goes away', async () => {
    const stop = caps.askUntilAnswered(caps.serverCapabilities, (c) => c.reason !== null, () => {})
    await vi.advanceTimersByTimeAsync(0)
    stop()
    await vi.advanceTimersByTimeAsync(3 * caps.RETRY_FAILED_MS)
    expect(fetchStub).toHaveBeenCalledTimes(1)
  })
})

describe('the vision probe', () => {
  it('holds a failure only briefly, and keeps an answer', async () => {
    fetchStub.mockImplementation(async () => json({ error: 'bad gateway' }, 502))
    const first = await vision.capabilities()
    expect(first.server).toBeNull()
    expect(first.tagger).toBe(false)
    await vision.capabilities()
    expect(fetchStub).toHaveBeenCalledTimes(1)

    fetchStub.mockImplementation(async () => json({ server: 'switchgen-vision', tagger: true, install: null, reason: null }))
    await vi.advanceTimersByTimeAsync(caps.RETRY_FAILED_MS)
    expect((await vision.capabilities()).tagger).toBe(true)
    expect(fetchStub).toHaveBeenCalledTimes(2)

    await vi.advanceTimersByTimeAsync(3 * caps.RETRY_FAILED_MS)
    await vision.capabilities()
    expect(fetchStub).toHaveBeenCalledTimes(2)
  })

  it('re-asks a server that does not answer, and takes a real answer with a reason as final', async () => {
    const heard: Vision.VisionCapabilities[] = []
    vision.watchCapabilities((c) => heard.push(c))
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(caps.RETRY_FAILED_MS)
    expect(fetchStub).toHaveBeenCalledTimes(2)
    expect(heard.every((c) => c.server === null)).toBe(true)

    const answer = {
      server: 'switchgen-vision',
      tagger: false,
      install: { repo: 'x', files: [], missing: ['model.onnx'] },
      reason: 'the tagger is not downloaded yet',
    }
    fetchStub.mockImplementation(async () => json(answer))
    await vi.advanceTimersByTimeAsync(caps.RETRY_FAILED_MS)
    expect(heard.at(-1)).toMatchObject(answer)
    const asked = fetchStub.mock.calls.length
    await vi.advanceTimersByTimeAsync(3 * caps.RETRY_FAILED_MS)
    expect(fetchStub).toHaveBeenCalledTimes(asked)
  })
})

describe('a server that stays silent', () => {
  it('asks at the short wait a few times, then less and less often, never more than the longest wait apart', async () => {
    const start = Date.now()
    const at: number[] = []
    fetchStub.mockImplementation(() => {
      at.push((Date.now() - start) / 1000)
      return refuse()
    })
    caps.askUntilAnswered(caps.serverCapabilities, (c) => c.reason !== null, () => {})
    await vi.advanceTimersByTimeAsync(20 * 60_000)
    // A fixed wait asked a seventh time at 60 s, and every ten seconds after.
    expect(at.slice(0, 8)).toEqual([0, 10, 20, 30, 40, 50, 70, 110])
    const gaps = at.slice(1).map((t, i) => t - at[i]!)
    expect(Math.max(...gaps) * 1000).toBeLessThanOrEqual(caps.RETRY_LONGEST_MS)
    expect(gaps.at(-1)! * 1000).toBe(caps.RETRY_LONGEST_MS)
  })

  it('waits the short wait five times, then twice as long each time, up to the longest', () => {
    for (let n = 0; n < caps.QUICK_RETRIES; n++) expect(caps.retryDelay(n)).toBe(caps.RETRY_FAILED_MS)
    expect(caps.retryDelay(5)).toBe(2 * caps.RETRY_FAILED_MS)
    expect(caps.retryDelay(50)).toBe(caps.RETRY_LONGEST_MS)
  })
})

describe('a tab out of sight', () => {
  /** A document whose visibility a test sets, and a window, both able to fire events. */
  function browser(visibilityState: 'hidden' | 'visible') {
    const doc = Object.assign(new EventTarget(), { visibilityState })
    const win = new EventTarget()
    vi.stubGlobal('document', doc)
    vi.stubGlobal('window', win)
    return { doc, win }
  }

  it('does not ask while hidden, asks at once when shown, and stops listening once answered', async () => {
    const { doc, win } = browser('hidden')
    const heard: Capabilities.ServerCapabilities[] = []
    caps.askUntilAnswered(caps.serverCapabilities, (c) => c.reason !== null, (c) => heard.push(c))
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchStub).toHaveBeenCalledTimes(1)
    // Nobody is looking, so nothing is asked, however long it stays hidden.
    await vi.advanceTimersByTimeAsync(10 * 60_000)
    expect(fetchStub).toHaveBeenCalledTimes(1)

    doc.visibilityState = 'visible'
    fetchStub.mockImplementation(async () => json({ downloads: true }))
    doc.dispatchEvent(new Event('visibilitychange'))
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchStub).toHaveBeenCalledTimes(2)
    expect(heard.at(-1)!.downloads).toBe(true)

    win.dispatchEvent(new Event('online'))
    await vi.advanceTimersByTimeAsync(60_000)
    expect(fetchStub).toHaveBeenCalledTimes(2)
  })

  it('takes the held failure when the network comes back inside the hold, and asks nothing new', async () => {
    const { win } = browser('visible')
    const heard: Capabilities.ServerCapabilities[] = []
    caps.askUntilAnswered(caps.serverCapabilities, (c) => c.reason !== null, (c) => heard.push(c))
    await vi.advanceTimersByTimeAsync(2000)
    expect(heard).toHaveLength(1)
    win.dispatchEvent(new Event('online'))
    await vi.advanceTimersByTimeAsync(0)
    expect(heard).toHaveLength(2)
    expect(heard[1]!.reason).not.toBeNull()
    expect(fetchStub).toHaveBeenCalledTimes(1)
  })
})
