import { mkdirSync, rmSync, statSync, symlinkSync, writeFileSync, existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import type * as History from '../src/lib/history'
import { call, mounted, tempRoots, type Handler } from './http'

/**
 * Deleting a file, the one thing in the app that cannot be undone, on both
 * sides: POST /api/delete in server/api.mjs, and history.deleteFiles, which
 * reads its answers. The two are held together by the 404 body a file that is
 * already gone gets, and by the names of the service worker's caches, which
 * the page empties of a deleted file's copies. Both are checked here, so a
 * change to one side that the other does not follow fails.
 */
let root = ''
let outputs = ''
let api: Handler
let h: typeof History

beforeAll(async () => {
  const roots = tempRoots()
  root = roots.root
  outputs = roots.outputs
  api = mounted((await import('../server/api.mjs')).switchgenApi())
  h = await import('../src/lib/history')
})

afterAll(() => rmSync(root, { recursive: true, force: true }))

afterEach(() => {
  vi.unstubAllGlobals()
})

const del = (rel: string, headers: Record<string, string> = {}) =>
  call(api, { method: 'POST', url: '/api/delete', body: { kind: 'output', rel }, headers })
const put = (rel: string, bytes: number) => {
  mkdirSync(path.dirname(path.join(outputs, rel)), { recursive: true })
  writeFileSync(path.join(outputs, rel), Buffer.alloc(bytes, 1))
}

describe('POST /api/delete', () => {
  it('deletes a file and says how much it freed', async () => {
    put('a.png', 123)
    const r = await del('a.png')
    expect(r.status).toBe(200)
    expect(r.json().freed).toBe(123)
    expect(existsSync(path.join(outputs, 'a.png'))).toBe(false)
  })

  it('answers a file already gone with the exact words the page reads', async () => {
    const r = await del('never-there.png')
    expect(r.status).toBe(404)
    expect(r.json()).toEqual({ error: 'not found' })
  })

  it('refuses a folder', async () => {
    mkdirSync(path.join(outputs, 'folder'), { recursive: true })
    expect((await del('folder')).status).toBe(400)
  })

  it('refuses a link that leads out of the outputs folder, and leaves what it points at', async () => {
    const outside = path.join(root, 'outside.png')
    writeFileSync(outside, 'keep me')
    symlinkSync(outside, path.join(outputs, 'escape.png'))
    expect((await del('escape.png')).status).toBe(400)
    expect(readFileSync(outside, 'utf8')).toBe('keep me')
  })

  it('refuses a request from another site before reading it', async () => {
    put('b.png', 10)
    const r = await del('b.png', { origin: 'http://evil.example', host: '127.0.0.1:5273' })
    expect(r.status).toBe(403)
    expect(statSync(path.join(outputs, 'b.png')).size).toBe(10)
  })
})

describe('deleting a record\'s file from the page', () => {
  const entry = (filename: string): History.NewEntry => ({
    desk: 'images', kind: 'image', mode: 't2i', file: { filename, subfolder: '', type: 'output' },
    familyId: 'x', familyLabel: 'X', variant: null, model: 'm', modelLabel: 'M', prompt: filename, negative: null,
    seed: 1, steps: 1, cfg: 1, sampler: 's', scheduler: 's', width: 1, height: 1, promptId: filename, durationMs: 0,
  })
  /** The page's fetch, answered by the real route, unless `answer` says otherwise for a file. */
  const throughTheServer = (answer: (rel: string) => Response | null = () => null) =>
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body))
      const said = answer(body.rel)
      if (said) return said
      const r = await call(api, { method: init.method, url, body })
      return new Response(r.body, { status: r.status, headers: { 'content-type': 'application/json' } })
    })
  const json = (body: unknown, status: number) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

  it('removes the record once its file has gone', async () => {
    throughTheServer()
    put('c.png', 50)
    const e = h.add(entry('c.png'))
    const [{ result }] = await h.deleteFiles([e])
    expect(result).toMatchObject({ ok: true, freed: 50, gone: 1 })
    expect(h.get(e.id)).toBeUndefined()
  })

  it('removes the record of a file that was already gone, on the server\'s own 404', async () => {
    throughTheServer()
    const e = h.add(entry('gone-already.png'))
    const [{ result }] = await h.deleteFiles([e])
    expect(result).toMatchObject({ ok: true, freed: 0 })
    expect(h.get(e.id)).toBeUndefined()
  })

  it('keeps the record, and says why, when the file would not go', async () => {
    throughTheServer(() => json({ error: 'EACCES: permission denied' }, 500))
    const e = h.add(entry('stuck.png'))
    const [{ result }] = await h.deleteFiles([e])
    expect(result).toEqual({ ok: false, reason: 'EACCES: permission denied' })
    expect(h.get(e.id)).toBeDefined()
  })

  it('says deleting is not offered here only when there is no delete route at all', async () => {
    const e = h.add(entry('static-host.png'))
    for (const [status, body] of [[404, { error: 'no such endpoint: POST /api/delete' }], [405, { error: 'takes GET' }]] as const) {
      throughTheServer(() => json(body, status))
      const [{ result }] = await h.deleteFiles([e])
      expect(result).toMatchObject({ ok: false, unsupported: true })
    }
    expect(h.get(e.id)).toBeDefined()
  })
})

describe('the offline copies of a deleted file', () => {
  it('are looked for in the caches the service worker keeps them in', async () => {
    // The names the worker opens for a file and for a thumbnail.
    const opened = new Set<string>()
    const listeners: Record<string, (e: unknown) => void> = {}
    const cache = { match: async () => undefined, put: async () => {}, keys: async () => [], delete: async () => true }
    const workerCaches = { open: async (name: string) => (opened.add(name), cache), keys: async () => [], delete: async () => true }
    const worker = {
      location: { origin: 'http://h' },
      addEventListener: (type: string, fn: (e: unknown) => void) => void (listeners[type] = fn),
      skipWaiting: async () => {},
      clients: { claim: async () => {} },
    }
    const source = readFileSync(new URL('../public/sw.js', import.meta.url), 'utf8')
    new Function('self', 'caches', 'fetch', source)(worker, workerCaches, async () => new Response('x', { status: 200 }))
    for (const url of ['http://h/comfy/view?filename=d.png&type=output', 'http://h/api/thumb?rel=d.png&w=512']) {
      let answer: Promise<unknown> = Promise.resolve()
      listeners.fetch!({ request: new Request(url), respondWith: (p: Promise<unknown>) => void (answer = p) })
      await answer
    }

    // The names the page empties once the file is deleted.
    const asked = new Set<string>()
    vi.stubGlobal('caches', {
      has: async (name: string) => (asked.add(name), false),
      open: async () => cache,
    })
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify({ freed: 1 }), { status: 200, headers: { 'content-type': 'application/json' } }))
    const e = h.add({ ...{ desk: 'images', kind: 'image', mode: 't2i', familyId: 'x', familyLabel: 'X', variant: null, model: 'm', modelLabel: 'M', prompt: 'd', negative: null, seed: 1, steps: 1, cfg: 1, sampler: 's', scheduler: 's', width: 1, height: 1, promptId: 'd', durationMs: 0 }, file: { filename: 'd.png', subfolder: '', type: 'output' } } as History.NewEntry)
    await h.deleteFiles([e])
    await vi.waitFor(() => expect(asked.size).toBe(2), { timeout: 2000 })

    expect([...asked].sort()).toEqual([...opened].sort())
  })
})
