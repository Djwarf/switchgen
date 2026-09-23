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
let outputs = ''
let server: Handler
/** While set, asking ComfyUI how the files were made waits for it. */
let gate: Promise<void> | null = null
let askedComfy = false
let h: typeof History
let recover: typeof Recover

beforeAll(async () => {
  const roots = tempRoots()
  root = roots.root
  outputs = roots.outputs
  server = mounted((await import('../server/archive.mjs')).switchgenArchive())
  vi.stubGlobal('fetch', async (url: string) => {
    if (url !== '/api/outputs') {
      askedComfy = true
      if (gate) await gate
      return new Response('{}', { status: 404, headers: { 'content-type': 'application/json' } })
    }
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
  vi.useRealTimers()
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

describe('what a recovery pass sends', () => {
  const post = (url: string, body: unknown) => call(server, { method: 'POST', url, body })
  const aged = (name: string) => {
    const t = Date.now() / 1000 - 60
    writeFileSync(path.join(outputs, name), 'x')
    utimesSync(path.join(outputs, name), t, t)
  }
  // The server lists the outputs folder at most every 4 s; files written by
  // a test are seen once the clock has moved past that. Only ever forward:
  // the listing remembers the time it was made.
  let clock = 0
  const later = () => {
    clock = Math.max(clock, Date.now()) + 5000
    vi.setSystemTime(clock)
  }
  const recordOf = (name: string) => h.all().find((e) => e.file.filename === name)

  it('marks as refiled only the records made from files whose record was removed', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    aged('late.png')
    aged('gone.png')
    await post('/api/archive/upsert', { records: [{ id: 'G', at: 1, file: { filename: 'gone.png', subfolder: '', type: 'output' } }] })
    await post('/api/archive/remove', { ids: ['G'] })
    later()

    const r = await recover.recoverUnfiled({ includeRemoved: true })
    expect(r.filed).toBe(2)
    expect(recordOf('gone.png')?.refiled).toBe(true)
    expect(recordOf('late.png')?.refiled).toBeUndefined()
    expect(recordOf('kept.png')?.refiled).toBe(true)
    expect(recordOf('stray.png')?.refiled).toBeUndefined()

    // The server takes both: the mark is what lets the removed one back in.
    const sent = (await post('/api/archive/upsert', { records: [recordOf('gone.png'), recordOf('late.png')] })).json()
    expect(sent.refused).toEqual([])
    expect(sent.assigned).toHaveLength(2)
    vi.useRealTimers()
  })

  it('does not file a second record for a file its desk filed while ComfyUI was being asked', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    aged('raced.png')
    aged('calm.png')
    later()
    let open = () => {}
    gate = new Promise<void>((resolve) => { open = resolve })
    askedComfy = false

    const pass = recover.recoverUnfiled()
    await vi.waitFor(() => { if (!askedComfy) throw new Error('not asked yet') })
    // The desk that made raced.png files it now, with how it was made.
    const desk = h.add({
      desk: 'images', kind: 'image', mode: 't2i',
      file: { filename: 'raced.png', subfolder: '', type: 'output' },
      familyId: 'x', familyLabel: 'X', variant: null, model: 'm', modelLabel: 'M',
      prompt: 'the desk\'s words', negative: null, seed: 1, steps: 1, cfg: 1, sampler: 's', scheduler: 's',
      width: 1, height: 1, promptId: 'desk-job', durationMs: 0,
    })
    open()
    gate = null
    const r = await pass

    expect(r.filed).toBe(1)
    const raced = h.all().filter((e) => e.file.filename === 'raced.png')
    expect(raced).toHaveLength(1)
    expect(raced[0]!.id).toBe(desk.id)
    expect(raced[0]!.prompt).toBe('the desk\'s words')
    expect(recordOf('calm.png')?.recovered).toBe(true)
    vi.useRealTimers()
  })
})
