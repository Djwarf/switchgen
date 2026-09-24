import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * The archive's live stream after the server behind it answers with an HTTP
 * error. A browser reconnects an event stream by itself after a network
 * error, but closes it for good on any answer that is not the stream, and
 * behind tailscale serve a restarting server answers 502. The stream is stood
 * in for, and a test says when it errs and whether the browser gave it up.
 */
class FakeStream {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 2
  static made: FakeStream[] = []
  readyState = FakeStream.CONNECTING
  onmessage: ((ev: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  constructor(public url: string) {
    FakeStream.made.push(this)
  }
  close() {
    this.readyState = FakeStream.CLOSED
  }
}

const PULL = { rev: 0, nextNo: 1, full: true, records: [], removed: [] }
/** Every URL fetched, in order. */
let asked: string[] = []
/** Pulls still to fail, as a fetch that cannot reach the server. */
let failing = 0
const pulls = () => asked.filter((u) => u.startsWith('/api/archive?since='))
/** The browser gives the stream up, as it does on an HTTP error, and says so. */
const giveUp = (s: FakeStream) => {
  s.readyState = FakeStream.CLOSED
  s.onerror?.()
}

beforeEach(() => {
  // The retry waits on a timer, which is moved on by hand.
  vi.useFakeTimers()
  vi.resetModules()
  FakeStream.made = []
  asked = []
  failing = 0
  vi.stubGlobal('EventSource', FakeStream)
  vi.stubGlobal('fetch', async (url: string) => {
    asked.push(url)
    if (url.startsWith('/api/archive?since=')) {
      if (failing > 0) {
        failing -= 1
        throw new TypeError('Failed to fetch')
      }
      return new Response(JSON.stringify(PULL), { status: 200, headers: { 'content-type': 'application/json' } })
    }
    return new Response(JSON.stringify({ error: 'not here' }), { status: 404, headers: { 'content-type': 'application/json' } })
  })
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

async function started() {
  const sync = await import('../src/lib/archiveSync')
  await sync.startArchiveSync()
  expect(FakeStream.made).toHaveLength(1)
  return FakeStream.made[0]!
}

describe('the archive\'s live stream', () => {
  it('is opened again, and the archive pulled, after the browser gives it up on an HTTP error', async () => {
    const first = await started()
    const before = pulls().length
    giveUp(first)
    await vi.advanceTimersByTimeAsync(4900)
    expect(FakeStream.made).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(200)
    expect(FakeStream.made).toHaveLength(2)
    expect(pulls().length).toBeGreaterThan(before)
  })

  it('is left to the browser while it is still reconnecting by itself', async () => {
    const first = await started()
    first.readyState = FakeStream.CONNECTING
    first.onerror?.()
    await vi.advanceTimersByTimeAsync(70_000)
    expect(FakeStream.made).toHaveLength(1)
  })

  it('opens one stream for one error, however often the dropped one speaks again', async () => {
    const first = await started()
    giveUp(first)
    await vi.advanceTimersByTimeAsync(5100)
    expect(FakeStream.made).toHaveLength(2)
    giveUp(first)
    await vi.advanceTimersByTimeAsync(70_000)
    expect(FakeStream.made).toHaveLength(2)
  })

  it('is opened again by whichever later pull gets through, when the first retry does not', async () => {
    const first = await started()
    failing = 1
    giveUp(first)
    // The first retry cannot reach the server, and waits longer for the next.
    await vi.advanceTimersByTimeAsync(5100)
    expect(failing).toBe(0)
    expect(FakeStream.made).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(10_000)
    expect(FakeStream.made).toHaveLength(2)
  })
})
