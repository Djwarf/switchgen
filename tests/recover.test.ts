import { rmSync, utimesSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import type { PastRun } from '../src/lib/comfy'
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
/** What ComfyUI's /history answers while set; unset, ComfyUI has forgotten everything. */
let comfyHistory: Record<string, unknown> | null = null
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
      if (comfyHistory && url.startsWith('/comfy/history')) {
        return new Response(JSON.stringify(comfyHistory), { status: 200, headers: { 'content-type': 'application/json' } })
      }
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

  it('files a name two runs have written with the run under way when the file on disk was written', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    // Written at 60 s past the epoch, in ms as the server lists it.
    const at = (name: string) => {
      writeFileSync(path.join(outputs, name), 'x')
      utimesSync(path.join(outputs, name), 60, 60)
    }
    at('rerun.png')
    at('forgot.png')
    /** A /history entry as ComfyUI keeps it, times in ms. */
    const run = (id: string, name: string, startedAt: number, finishedAt: number) => [
      id,
      {
        prompt: [1, id, { '9': { class_type: 'SaveImage', inputs: { filename_prefix: 'x' } } }, {}, ['9']],
        outputs: { '9': { images: [{ filename: name, subfolder: '', type: 'output' }] } },
        status: {
          status_str: 'success',
          messages: [['execution_start', { timestamp: startedAt }], ['execution_success', { timestamp: finishedAt }]],
        },
      },
    ]
    // Oldest first, as /history lists them. rerun.png was made, deleted, and
    // made again under the same name; forgot.png's own run is forgotten, and
    // only one that made an earlier file under its name is remembered.
    comfyHistory = Object.fromEntries([
      run('old', 'rerun.png', 500, 1000),
      run('new', 'rerun.png', 55_000, 59_000),
      run('older', 'forgot.png', 500, 1000),
    ])
    later()
    try {
      const r = await recover.recoverUnfiled()
      expect(r.filed).toBe(2)
      expect(recordOf('rerun.png')?.promptId).toBe('new')
      expect(recordOf('forgot.png')?.promptId).toBe('')
      expect(recordOf('forgot.png')?.familyId).toBe('unknown')
    } finally {
      comfyHistory = null
      vi.useRealTimers()
    }
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
    await vi.waitFor(() => { if (!askedComfy) throw new Error('not asked yet') }, { timeout: 5000 })
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

describe('the run that wrote a file', () => {
  const run = (promptId: string, startedAt: number | null, finishedAt: number | null): PastRun => ({
    promptId, startedAt, finishedAt, graph: {}, files: [], status: 'success', clientId: null, error: null,
  })

  it('is the one under way when the file was written, newest first', () => {
    const runs = [run('new', 4000, 5000), run('old', 500, 1000)]
    expect(recover.runThatWrote(runs, 4900)?.promptId).toBe('new')
  })

  it('is none when every run remembered ended long before the file was written', () => {
    expect(recover.runThatWrote([run('old', 500, 1000)], 60_000)).toBeUndefined()
  })

  it('is not a later run that found the file already made', () => {
    const runs = [run('cached', 70_000, 70_100), run('writer', 500, 1000)]
    expect(recover.runThatWrote(runs, 900)?.promptId).toBe('writer')
  })

  it('is a run with no times at all', () => {
    expect(recover.runThatWrote([run('untimed', null, null)], 123_456)?.promptId).toBe('untimed')
  })
})
