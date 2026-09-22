import { rmSync } from 'node:fs'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { call, mounted, tempRoots, type Handler } from './http'

// Every server module reads its roots from the environment when it loads, so
// they are pointed at empty temporary folders first and imported after.
let root = ''
const handlers: Record<string, Handler> = {}

beforeAll(async () => {
  root = tempRoots().root
  const [api, archive, thumbs, downloads, reel, vision] = await Promise.all([
    import('../server/api.mjs'),
    import('../server/archive.mjs'),
    import('../server/thumbs.mjs'),
    import('../server/downloads.mjs'),
    import('../server/reel.mjs'),
    import('../server/vision.mjs'),
  ])
  handlers.api = mounted(api.switchgenApi())
  handlers.archive = mounted(archive.switchgenArchive())
  handlers.thumbs = mounted(thumbs.switchgenThumbs())
  handlers.downloads = mounted(downloads.switchgenDownloads())
  handlers.reel = mounted(reel.switchgenReel())
  handlers.vision = mounted(vision.switchgenVision())
})

afterAll(async () => {
  // Let the archive's debounced write land before its folder goes.
  await new Promise((resolve) => setTimeout(resolve, 400))
  rmSync(root, { recursive: true, force: true })
})

describe('a request path that is not a URL', () => {
  // `new URL('//', base)` throws. Parsed outside a handler's own try, that
  // throw was an unhandled rejection, and Node ends the process on those: one
  // GET for // stopped the app for every device.
  it('is answered 400 by every middleware, which does not reject', async () => {
    for (const name of ['api', 'archive', 'thumbs', 'downloads', 'reel', 'vision']) {
      const r = await call(handlers[name]!, { url: '//' })
      expect(r.passed, name).toBe(false)
      expect(r.status, name).toBe(400)
      expect(r.json().error, name).toMatch(/not a valid URL/)
    }
  })

  it('leaves paths outside a middleware\'s own namespace to the next one', async () => {
    for (const name of ['archive', 'thumbs', 'downloads', 'reel', 'vision']) {
      expect((await call(handlers[name]!, { url: '/api/somewhere-else' })).passed, name).toBe(true)
    }
    expect((await call(handlers.api!, { url: '/index.html' })).passed).toBe(true)
  })
})

describe('the archive server', () => {
  const record = (id: string, extra: Record<string, unknown> = {}) => ({
    id,
    at: 1_700_000_000_000,
    file: { filename: `${id}.png`, subfolder: '', type: 'output' },
    prompt: 'a lighthouse at dusk',
    ...extra,
  })
  const post = (url: string, body: unknown) => call(handlers.archive!, { method: 'POST', url, body })
  const pull = async () => (await call(handlers.archive!, { url: '/api/archive' })).json()

  it('writes to the temporary archive, never the real one', async () => {
    expect(String((await pull()).file).startsWith(root)).toBe(true)
  })

  it('refuses to bring back a record removed after the writer last saw it', async () => {
    const put = (await post('/api/archive/upsert', { records: [record('r1')] })).json()
    expect(put.refused).toEqual([])
    const seen = put.assigned[0].rev as number

    // Another device removes it.
    expect((await post('/api/archive/remove', { ids: ['r1'] })).json().removed).toBe(1)

    // A tab that last saw the record at `seen` sends its edit afterwards.
    const stale = (await post('/api/archive/upsert', { records: [record('r1', { rev: seen, prompt: 'edited' })] })).json()
    expect(stale.refused).toEqual(['r1'])
    expect(stale.assigned).toEqual([])
    const after = await pull()
    expect(after.records.map((r: { id: string }) => r.id)).not.toContain('r1')
    expect(after.removed).toContain('r1')
  })

  it('lets an undo, which carries no stamp, and the restore route bring it back', async () => {
    await post('/api/archive/upsert', { records: [record('r2')] })
    const stamped = (await pull()).records.find((r: { id: string }) => r.id === 'r2')
    expect((await post('/api/archive/remove', { ids: ['r2'] })).json().removed).toBe(1)

    // Undo sends the record as it was before the server ever stamped it.
    const undo = (await post('/api/archive/upsert', { records: [record('r2')] })).json()
    expect(undo.assigned.map((a: { id: string }) => a.id)).toEqual(['r2'])

    expect((await post('/api/archive/remove', { ids: ['r2'] })).json().removed).toBe(1)
    const restored = (await post('/api/archive/restore', { records: [{ ...stamped }] })).json()
    expect(restored.assigned.map((a: { id: string }) => a.id)).toEqual(['r2'])
    expect((await pull()).records.map((r: { id: string }) => r.id)).toContain('r2')
  })

  it('refuses a write from another origin before reading it', async () => {
    const r = await call(handlers.archive!, {
      method: 'POST',
      url: '/api/archive/upsert',
      headers: { origin: 'http://evil.example', host: '127.0.0.1:5273' },
      body: { records: [record('r3')] },
    })
    expect(r.status).toBe(403)
    expect((await pull()).records.map((x: { id: string }) => x.id)).not.toContain('r3')
  })
})
