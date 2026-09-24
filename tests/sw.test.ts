import { readFileSync } from 'node:fs'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * The service worker, run against a stand-in `self`, `caches` and `fetch`.
 * It is plain script, so its source is evaluated here as the browser would,
 * with the fetch listener captured to be called by hand.
 */
const SOURCE = readFileSync(new URL('../public/sw.js', import.meta.url), 'utf8')

type FetchEvent = { request: unknown; respondWith: ReturnType<typeof vi.fn>; waitUntil?: ReturnType<typeof vi.fn> }
let onFetch: (e: FetchEvent) => void
let fetchStub: ReturnType<typeof vi.fn>
let put: ReturnType<typeof vi.fn>
/** What the caches hold, by URL, whichever cache it is asked of. */
let kept: Map<string, Response>
/** Every cache the worker opened, by name, in order. */
let opened: string[]

beforeEach(() => {
  const listeners: Record<string, (e: FetchEvent) => void> = {}
  const self = {
    location: { origin: 'http://h' },
    addEventListener: (type: string, fn: (e: FetchEvent) => void) => {
      listeners[type] = fn
    },
    skipWaiting: () => Promise.resolve(),
    clients: { claim: () => Promise.resolve() },
  }
  put = vi.fn(() => Promise.resolve())
  kept = new Map()
  opened = []
  const match = vi.fn(async (r: string | Request) => kept.get(typeof r === 'string' ? new URL(r, 'http://h').href : r.url))
  const cache = { match, put, keys: vi.fn(async () => []), delete: vi.fn(), add: vi.fn() }
  const caches = {
    open: vi.fn(async (name: string) => {
      opened.push(name)
      return cache
    }),
    keys: vi.fn(async () => []),
    delete: vi.fn(),
  }
  fetchStub = vi.fn(async () => new Response('x', { status: 200 }))
  new Function('self', 'caches', 'fetch', SOURCE)(self, caches, fetchStub)
  onFetch = listeners.fetch!
})

const send = (request: unknown) => {
  const event: FetchEvent = { request, respondWith: vi.fn(), waitUntil: vi.fn() }
  onFetch(event)
  return event
}
const answer = (e: FetchEvent) => e.respondWith.mock.calls[0]![0] as Promise<Response>

afterEach(() => {
  vi.useRealTimers()
})

describe('the service worker and generated files', () => {
  it('leaves a ranged request for a clip to the browser', () => {
    const e = send(new Request('http://h/comfy/view?filename=c.webm', { headers: { range: 'bytes=0-' } }))
    expect(e.respondWith).not.toHaveBeenCalled()
    expect(fetchStub).not.toHaveBeenCalled()
  })

  it('leaves anything a media element asks for to the browser', () => {
    // Node's Request cannot be given a destination, so this is the shape a
    // <video> element's request has.
    const e = send({ method: 'GET', url: 'http://h/comfy/view?filename=c.webm', mode: 'no-cors', destination: 'video', headers: new Headers() })
    expect(e.respondWith).not.toHaveBeenCalled()
  })

  it('asks the server first for a picture, and keeps a whole one', async () => {
    const request = new Request('http://h/comfy/view?filename=a.png')
    const e = send(request)
    expect(e.respondWith).toHaveBeenCalledOnce()
    const res = (await e.respondWith.mock.calls[0]![0]) as Response
    expect(res.status).toBe(200)
    expect(fetchStub).toHaveBeenCalledWith(request, { cache: 'no-cache' })
    expect(put).toHaveBeenCalledOnce()
    expect(put.mock.calls[0]![0]).toBe(request)
  })
})

describe('a file asked for while ComfyUI is restarting', () => {
  // The proxy in front of ComfyUI answers an empty 502 or 504 while it is
  // down, where a fetch that threw used to be the only way to the copy kept.
  const view = 'http://h/comfy/view?filename=a.png&type=output'

  it.each([502, 504])('shows the copy kept here for a %i', async (status) => {
    const copy = new Response('kept picture', { status: 200 })
    kept.set(view, copy)
    fetchStub.mockResolvedValueOnce(new Response('', { status }))
    const res = await answer(send(new Request(view)))
    expect(res).toBe(copy)
  })

  it('passes a 502 on when nothing is kept', async () => {
    fetchStub.mockResolvedValueOnce(new Response('', { status: 502 }))
    expect((await answer(send(new Request(view)))).status).toBe(502)
  })

  it('passes ComfyUI\'s own 500 on', async () => {
    kept.set(view, new Response('kept picture', { status: 200 }))
    fetchStub.mockResolvedValueOnce(new Response('boom', { status: 500 }))
    expect((await answer(send(new Request(view)))).status).toBe(500)
  })

  it('passes a thumbnail\'s 503 on, which is the thumbnail server\'s own answer', async () => {
    const thumb = 'http://h/api/thumb?rel=a.png&w=512'
    kept.set(thumb, new Response('kept thumbnail', { status: 200 }))
    fetchStub.mockResolvedValueOnce(new Response('busy', { status: 503 }))
    expect((await answer(send(new Request(thumb)))).status).toBe(503)
  })

  it('keeps files and thumbnails in the caches named for them', async () => {
    await answer(send(new Request(view)))
    await answer(send(new Request('http://h/api/thumb?rel=a.png&w=512')))
    // Each is opened again to trim it once a copy is kept.
    expect([...new Set(opened)]).toEqual(['switchgen-v2-media', 'switchgen-v2-thumbs'])
  })
})

describe('a page load on a weak signal', () => {
  const navigate = { method: 'GET', url: 'http://h/pictures', mode: 'navigate', headers: new Headers() }

  it('shows the kept shell once the network has kept it waiting', async () => {
    vi.useFakeTimers()
    const shell = new Response('<html>kept</html>', { status: 200 })
    kept.set('http://h/', shell)
    fetchStub.mockReturnValueOnce(new Promise(() => {}))
    let got: Response | undefined
    void answer(send(navigate)).then((r) => {
      got = r
    })
    await vi.advanceTimersByTimeAsync(3900)
    expect(got).toBeUndefined()
    await vi.advanceTimersByTimeAsync(200)
    expect(got).toBe(shell)
  })

  it('shows the network\'s page when it answers first', async () => {
    vi.useFakeTimers()
    kept.set('http://h/', new Response('<html>kept</html>', { status: 200 }))
    const fresh = new Response('<html>fresh</html>', { status: 200 })
    fetchStub.mockResolvedValueOnce(fresh)
    const e = send(navigate)
    await vi.advanceTimersByTimeAsync(0)
    expect(await answer(e)).toBe(fresh)
  })

  // Over Tailscale Serve the proxy answers for the app server while it is
  // stopped or rebuilding, with an empty 502, and an app opened from the home
  // screen showed that as a white page with no way to reload it.
  it.each([502, 503, 504])('shows the kept shell when a gateway answers %i for the app server', async (status) => {
    const shell = new Response('<html>kept</html>', { status: 200 })
    kept.set('http://h/', shell)
    fetchStub.mockResolvedValueOnce(new Response('', { status }))
    expect(await answer(send(navigate))).toBe(shell)
  })

  it('passes the gateway\'s answer on when no shell is kept', async () => {
    fetchStub.mockResolvedValueOnce(new Response('', { status: 502 }))
    expect((await answer(send(navigate))).status).toBe(502)
  })

  it('passes the server\'s own error on, and does not hide it behind the shell', async () => {
    kept.set('http://h/', new Response('<html>kept</html>', { status: 200 }))
    fetchStub.mockResolvedValueOnce(new Response('boom', { status: 500 }))
    expect((await answer(send(navigate))).status).toBe(500)
  })

  it('keeps nothing of a gateway\'s answer, and waits on no refresh once the shell is shown', async () => {
    kept.set('http://h/', new Response('<html>kept</html>', { status: 200 }))
    // Same-origin, as the proxy's answer is in a browser: only its status keeps it out of the cache.
    fetchStub.mockResolvedValueOnce(Object.defineProperty(new Response('', { status: 502 }), 'type', { value: 'basic' }))
    const e = send(navigate)
    await answer(e)
    expect(put).not.toHaveBeenCalled()
    expect(e.waitUntil).not.toHaveBeenCalled()
  })
})
