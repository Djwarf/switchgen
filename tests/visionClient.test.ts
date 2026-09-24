import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * What the page makes of the picture reader turning a reading away for want
 * of memory (server/vision.mjs answers 503 with busy 'memory'): the server's
 * own sentence, shown as it is, and a pause rather than a failure.
 */
const SAID = 'The picture reader needs about 1.9 GB of memory and only 1.2 GB is free. Running out makes the system stop its largest program (usually ComfyUI) to get memory back, so nothing was read. Try again once more memory is free, for example when a render has finished.'
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

beforeEach(() => {
  vi.resetModules()
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url === '/api/vision/capabilities') return json({ server: 'switchgen-vision', tagger: true, detect: true, install: null, reason: null })
    return json({ error: SAID, busy: 'memory' }, 503)
  }))
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('a reading refused for memory', () => {
  it('reads as the server\'s sentence when one picture is inspected', async () => {
    const vision = await import('../src/lib/vision')
    const r = await vision.inspectImage('a.png')
    expect(r.unavailable).toBe(SAID)
    expect(r.busy).toBe('memory')
  })

  it('pauses a tagging run with the same sentence, as its own kind of stop', async () => {
    const vision = await import('../src/lib/vision')
    const err = await vision.tagImages([{ kind: 'output', rel: 'a.png' }]).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(vision.VisionBusy)
    expect((err as Error).message).toBe(SAID)
  })
})
