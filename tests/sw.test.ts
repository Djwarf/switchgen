import { readFileSync } from 'node:fs'
import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * The service worker, run against a stand-in `self`, `caches` and `fetch`.
 * It is plain script, so its source is evaluated here as the browser would,
 * with the fetch listener captured to be called by hand.
 */
const SOURCE = readFileSync(new URL('../public/sw.js', import.meta.url), 'utf8')

type FetchEvent = { request: unknown; respondWith: ReturnType<typeof vi.fn> }
let onFetch: (e: FetchEvent) => void
let fetchStub: ReturnType<typeof vi.fn>
let put: ReturnType<typeof vi.fn>

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
  const cache = { match: vi.fn(async () => undefined), put, keys: vi.fn(async () => []), delete: vi.fn(), add: vi.fn() }
  const caches = { open: vi.fn(async () => cache), keys: vi.fn(async () => []), delete: vi.fn() }
  fetchStub = vi.fn(async () => new Response('x', { status: 200 }))
  new Function('self', 'caches', 'fetch', SOURCE)(self, caches, fetchStub)
  onFetch = listeners.fetch!
})

const send = (request: unknown) => {
  const event: FetchEvent = { request, respondWith: vi.fn() }
  onFetch(event)
  return event
}

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
