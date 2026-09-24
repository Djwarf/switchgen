import { readFileSync } from 'node:fs'
import path from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type * as RunnerMod from '../src/lib/runner'

/**
 * The shell's notice about work the queue on the server holds, shown on
 * every room while the lane is held, with the two answers to it. The
 * queue's store is stood in, so the notice reads what a test sets; it is
 * drawn to a string, so what only happens when a button is pressed in a
 * browser is not covered here.
 */

const m = vi.hoisted(() => ({
  snap: null as unknown as RunnerMod.RunnerSnapshot,
  laneWord: vi.fn(async (_action: 'send' | 'stop', _since: number) => true),
  /** A read of the state: a test has it bring the state as the server now has it. */
  refresh: vi.fn(async (_opts?: { fresh?: boolean }) => {}),
  /** The library itself, for a test that has the notice use its real store and word. */
  real: null as unknown as typeof RunnerMod,
  live: false,
}))

vi.mock('../src/lib/runner', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/lib/runner')>()
  m.real = real
  return {
    ...real,
    runnerStore: { subscribe: () => () => {}, snapshot: () => (m.live ? m.real.runnerStore.snapshot() : m.snap), start: () => {}, refresh: m.refresh },
    useRunner: () => m.snap,
    laneWord: m.laneWord,
  }
})

const blank = (): RunnerMod.RunnerSnapshot => ({
  v: 1, available: true, reason: null, boot: 'b', rev: 1, comfy: { answering: true, since: 0 }, lane: { held: null }, groups: [], jobs: [], progress: {}, connected: true,
})

function job(id: string, over: Partial<RunnerMod.RunnerJob>): RunnerMod.RunnerJob {
  return {
    id, groupId: 'g', desk: 'images', kind: 'image', seq: 1, index: 1, total: 1, label: 'A picture', prompt: 'p', device: 'x',
    heavy: false, status: 'waiting', wait: { for: 'turn' }, stopRequested: false, stopLanded: false, promptId: null, attempt: 0,
    createdAt: 1, sentAt: null, ranAt: null, finishedAt: null, endedAt: null, files: [], primary: null, frame: null,
    openedOn: null, entryId: null, entryNo: null, repeatOf: null, durationMs: 0, error: null, meta: null, dismissed: false, ...over,
  }
}
const hold = (why: NonNullable<RunnerMod.RunnerLane['held']>['why'], scope: 'all' | 'heavy', jobId: string | null = null) => ({ held: { why, scope, jobId, since: 1 } })

let hold$: typeof import('../src/components/shell/RunnerHold')
const drawn = () => renderToStaticMarkup(createElement(hold$.RunnerHold))
/** The page's text, one space between words. */
const said = () => drawn().replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ')

beforeEach(async () => {
  m.snap = blank()
  m.live = false
  m.laneWord.mockReset()
  m.laneWord.mockResolvedValue(true)
  m.refresh.mockReset()
  m.refresh.mockResolvedValue(undefined)
  vi.resetModules()
  hold$ = await import('../src/components/shell/RunnerHold')
})

describe('the notice about work held on the server', () => {
  it('shows nothing while nothing is held', () => {
    m.snap = { ...blank(), jobs: [job('a', { wait: { for: 'held' } })] }
    expect(drawn()).toBe('')
    expect(hold$.heldCounts(m.snap)).toEqual({ images: 0, video: 0, reel: 0 })
  })

  it('counts every waiting job a hold on all the work covers, desk by desk, whatever its wait says', () => {
    m.snap = {
      ...blank(),
      lane: hold('restart', 'all'),
      jobs: [
        // A batch: only its first picture says held; the rest wait for the one before, held all the same.
        job('a', { wait: { for: 'held' } }),
        job('b', { wait: { for: 'before' } }),
        job('c', { wait: { for: 'before' } }),
        job('d', { desk: 'video', kind: 'video', heavy: true, wait: { for: 'held' } }),
        // Running, so the hold does not hold it.
        job('e', { desk: 'reel', kind: 'video', heavy: true, status: 'running', wait: null, promptId: 'p' }),
        job('f', { status: 'done', wait: null }),
      ],
    }
    expect(hold$.heldCounts(m.snap)).toEqual({ images: 3, video: 1, reel: 0 })
    const html = drawn()
    expect(said()).toContain('Work held on the server')
    expect(said()).toContain('The machine restarted while this work waited')
    expect(said()).toContain('Waiting: 3 pictures and 1 clip.')
    expect(html).toContain('Send them')
    expect(html).toContain('Call them off')
    // Both answers are big enough for a thumb.
    expect(html.match(/\[@media\(pointer:coarse\)\]:min-h-11/g)).toHaveLength(2)
    expect(said()).not.toContain('is not running just now')
    expect(m.laneWord).not.toHaveBeenCalled()
  })

  it('counts only the heavy waiting jobs under a hold on heavy work, whatever their wait', () => {
    m.snap = {
      ...blank(),
      lane: hold('lost', 'heavy', 'x'),
      jobs: [
        job('a', { wait: { for: 'turn' } }),
        job('d', { desk: 'video', kind: 'video', heavy: true, wait: { for: 'held' } }),
        job('e', { desk: 'reel', kind: 'video', heavy: true, wait: { for: 'before' } }),
      ],
    }
    expect(hold$.heldCounts(m.snap)).toEqual({ images: 0, video: 1, reel: 1 })
    expect(said()).toContain('A heavy job was lost')
    expect(said()).toContain('Waiting: 1 clip and 1 shot.')
  })

  it('counts, under a hold on the work that waited through a pause, a heavy clip made after it once a clip was lost, and not a picture made after it', () => {
    m.snap = {
      ...blank(),
      lane: { held: { why: 'paused', scope: 'all', jobId: 'x', since: 50, heavyAfter: 90 } },
      jobs: [
        job('a', { wait: { for: 'held' }, createdAt: 40 }),
        job('p', { desk: 'video', kind: 'video', heavy: true, wait: { for: 'held' }, createdAt: 100 }),
        job('l', { wait: { for: 'turn' }, createdAt: 100 }),
      ],
    }
    expect(hold$.heldCounts(m.snap)).toEqual({ images: 1, video: 1, reel: 0 })
    expect(said()).toContain('Waiting: 1 picture and 1 clip.')
  })

  it('names a clip that may never have reached ComfyUI', () => {
    m.snap = { ...blank(), lane: hold('unsent', 'heavy', 'x'), jobs: [job('d', { desk: 'video', heavy: true, wait: { for: 'held' } })] }
    expect(said()).toContain('A heavy job may never have reached ComfyUI')
  })

  it('names a queue that was off, and says while it still is that it cannot take the word', () => {
    m.snap = {
      ...blank(),
      available: false,
      reason: 'Turned off with SWITCHGEN_RUNNER=off.',
      lane: hold('paused', 'all'),
      jobs: [
        job('a', { wait: { for: 'turn' } }),
        job('d', { desk: 'video', kind: 'video', heavy: true, wait: { for: 'held' } }),
        job('e', { desk: 'reel', kind: 'video', heavy: true, wait: { for: 'before' } }),
      ],
    }
    expect(said()).toContain('The queue on the server was off while this work waited')
    expect(said()).toContain('is not running just now')
    expect(said()).toContain('Waiting: 1 picture, 1 clip and 1 shot.')
    // Still offered: the page's copy can be behind the server's.
    expect(said()).toContain('Send them')
  })

  it('is drawn by the shell on every room, not by one desk', () => {
    // The shell itself cannot be drawn without a DOM; its source is read.
    const shell = readFileSync(path.resolve(import.meta.dirname, '..', 'src', 'components', 'shell', 'Shell.tsx'), 'utf8')
    expect(shell).toMatch(/import \{ RunnerHold \} from '\.\/RunnerHold'/)
    expect(shell).toMatch(/<RunnerHold \/>/)
  })
})

describe('the word given on the notice', () => {
  const heldSince = (since: number, why: NonNullable<RunnerMod.RunnerLane['held']>['why'] = 'restart') => ({ held: { why, scope: 'all' as const, jobId: null, since } })
  /** The state a read brings back. */
  const readBrings = (lane: RunnerMod.RunnerLane) => m.refresh.mockImplementation(async () => { m.snap = { ...m.snap, lane } })

  it('names the hold the notice shows, and is taken', async () => {
    m.snap = { ...blank(), lane: heldSince(5) }
    expect(await hold$.answerHold('stop', 5)).toEqual({ is: 'taken' })
    expect(m.laneWord).toHaveBeenCalledWith('stop', 5)
    // What it changed comes on the stream.
    expect(m.refresh).not.toHaveBeenCalled()
  })

  it('says another hold stands, and which, when the server refused the word for the one shown', async () => {
    m.snap = { ...blank(), lane: heldSince(5) }
    m.laneWord.mockResolvedValue(false)
    readBrings({ held: { why: 'lost', scope: 'heavy', jobId: 'x', since: 9 } })
    expect(await hold$.answerHold('stop', 5)).toEqual({ is: 'moved', since: 9 })
    expect(m.refresh).toHaveBeenCalledTimes(1)
    // A read that leaves after the refusal, not one already out.
    expect(m.refresh).toHaveBeenCalledWith({ fresh: true })
  })

  it('says the hold moved when a read already out when the word was refused still showed the old one', async () => {
    // The real store and word, over a server stood in by fetch.
    const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
    const state = (since: number, rev: number) => ({ ...blank(), rev, lane: heldSince(since) })
    let releaseFirst: () => void = () => {}
    let reads = 0
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url === '/api/runner' && (init?.method ?? 'GET') === 'GET') {
        reads++
        if (reads === 1) {
          await new Promise<void>((res) => (releaseFirst = res))
          return json(state(1, 1))
        }
        return json(state(2, 2))
      }
      if (url === '/api/runner/lane') return json({ error: 'The hold has changed since this page showed it.' }, 409)
      return json({ error: 'no route' }, 404)
    })
    try {
      m.live = true
      m.laneWord.mockImplementation((action, since) => m.real.laneWord(action, since))
      m.refresh.mockImplementation((opts) => m.real.runnerStore.refresh(opts))
      // A read left before the hold changed, and is still out.
      const first = m.real.runnerStore.refresh()
      await vi.waitFor(() => expect(reads).toBe(1), { timeout: 5000, interval: 5 })
      const answer = hold$.answerHold('send', 1)
      // The word was refused, and the notice has asked for the hold as it now stands.
      await vi.waitFor(() => expect(m.refresh).toHaveBeenCalled(), { timeout: 5000, interval: 5 })
      releaseFirst()
      await first
      expect(await answer).toEqual({ is: 'moved', since: 2 })
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('says the word did not get through while the same hold stands, whether refused or not answered', async () => {
    m.snap = { ...blank(), lane: heldSince(5) }
    m.laneWord.mockResolvedValue(false)
    readBrings(heldSince(5))
    expect(await hold$.answerHold('send', 5)).toEqual({ is: 'refused' })
    m.laneWord.mockRejectedValue(new TypeError('fetch failed'))
    expect(await hold$.answerHold('send', 5)).toEqual({ is: 'refused' })
  })

  it('says nothing more of a hold another device answered first', async () => {
    m.snap = { ...blank(), lane: heldSince(5) }
    m.laneWord.mockResolvedValue(false)
    readBrings({ held: null })
    expect(await hold$.answerHold('send', 5)).toEqual({ is: 'gone' })
  })

  it('says nothing of a word for an earlier hold on the notice for a hold no word was given to', () => {
    m.snap = { ...blank(), lane: heldSince(9, 'lost'), jobs: [job('d', { desk: 'video', kind: 'video', heavy: true, wait: { for: 'held' } })] }
    expect(said()).toContain('Work held on the server')
    expect(said()).not.toContain('The hold changed while you answered')
  })

  it('gives each button the since of the hold drawn', () => {
    // A press cannot be made without a DOM; the source is read.
    const src = readFileSync(path.resolve(import.meta.dirname, '..', 'src', 'components', 'shell', 'RunnerHold.tsx'), 'utf8')
    expect(src).toContain('laneWord(action, since)')
    expect(src).toMatch(/since=\{held\.since\}/)
    expect(src).toContain('answerHold(action, since)')
  })
})
