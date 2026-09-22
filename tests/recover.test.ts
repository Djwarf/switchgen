import { rmSync, utimesSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import type * as History from '../src/lib/history'
import type * as Recover from '../src/lib/recover'
import { call, mounted, tempRoots, type Handler } from './http'

/**
 * Filing the outputs no record describes, against the real archive server.
 * ComfyUI is not there, so every file is filed without settings.
 */
let root = ''
let server: Handler
let h: typeof History
let recover: typeof Recover

beforeAll(async () => {
  const roots = tempRoots()
  root = roots.root
  server = mounted((await import('../server/archive.mjs')).switchgenArchive())
  vi.stubGlobal('fetch', async (url: string) => {
    if (url !== '/api/outputs') return new Response('{}', { status: 404, headers: { 'content-type': 'application/json' } })
    const r = await call(server, { url })
    return new Response(r.body, { status: r.status, headers: { 'content-type': 'application/json' } })
  })
  // A record for kept.png, removed from the archive with its file kept.
  const t = Date.now() / 1000 - 60
  for (const name of ['kept.png', 'stray.png']) {
    writeFileSync(path.join(roots.outputs, name), 'x')
    utimesSync(path.join(roots.outputs, name), t, t)
  }
  const post = (url: string, body: unknown) => call(server, { method: 'POST', url, body })
  await post('/api/archive/upsert', { records: [{ id: 'K', at: 1, file: { filename: 'kept.png', subfolder: '', type: 'output' } }] })
  await post('/api/archive/remove', { ids: ['K'] })
  h = await import('../src/lib/history')
  recover = await import('../src/lib/recover')
})

afterAll(async () => {
  await new Promise((resolve) => setTimeout(resolve, 400))
  vi.unstubAllGlobals()
  rmSync(root, { recursive: true, force: true })
})

const filed = () => h.all().map((e) => e.file.filename).sort()

describe('recovering unfiled outputs', () => {
  it('leaves out a file whose record was removed, and counts it', async () => {
    const r = await recover.recoverUnfiled()
    expect(r).toEqual({ filed: 1, fromHistory: 0, removed: 1 })
    expect(filed()).toEqual(['stray.png'])
  })

  it('files it when the reader asks for removed files by name', async () => {
    const r = await recover.recoverUnfiled({ includeRemoved: true })
    expect(r.filed).toBe(1)
    expect(r.removed).toBe(0)
    expect(filed()).toEqual(['kept.png', 'stray.png'])
  })
})
