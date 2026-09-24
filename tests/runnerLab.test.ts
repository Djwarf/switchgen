import { randomUUID } from 'node:crypto'
import { readFileSync, rmSync } from 'node:fs'
import path from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { call, tempRoots, type Handler } from './http'
import { CHAIN_TOKEN, clearRegistry, groupBody, harness, mountPlugin, runnerEnv, standIn, type Harness } from './runnerFake'

/**
 * The model lab's desk on the queue (lab/ in the repo). A lab group is of
 * kind 'set': its pictures go independently, except that one chained to
 * another waits for that one to end and then opens on the picture it stands
 * for. Its jobs are never filed in the archive, no busy lock of the Pictures
 * desk applies to it, it is named in /api/capabilities, and no desk of the
 * app shows its work. Runners are built against the scripted ComfyUI and
 * stepped by hand; the capabilities tests talk to a stand-in ComfyUI on a
 * local port. Nothing here reaches a real ComfyUI or the running app.
 */

// For the last describe, which reads the app's bridges and the shell's hold:
// the queue's store is stood in, so they read what the test sets. Nothing else
// here imports it.
const m = vi.hoisted(() => ({ snap: null as any }))
vi.mock('../src/lib/runner', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/lib/runner')>()),
  runnerStore: {
    subscribe: () => () => {},
    snapshot: () => m.snap,
    start: () => {},
    refresh: async () => {},
  },
  useRunner: () => m.snap,
}))

let restoreEnv = () => {}
let roots = ''
beforeAll(() => {
  roots = tempRoots().root
  restoreEnv = runnerEnv()
})
afterAll(() => {
  restoreEnv()
  // tempRoots' folder; runnerFake removes its own.
  if (roots) rmSync(roots, { recursive: true, force: true })
})

// -------------------------------------------------------- a lab group --

/** A cell id as the lab makes one: 16 hex characters, naming no model. */
const newCell = () => randomUUID().replace(/-/g, '').slice(0, 16)
const fileOf = (cell: string) => `${cell}_00001_.png`
const LAB_DIR = '.lab/cells'

/** One sampler and one SaveImage under the cell's own prefix, as the lab builds a picture. */
const picture = (cell: string) => ({
  '1': { class_type: 'KSampler', inputs: {} },
  '9': { class_type: 'SaveImage', inputs: { images: ['1', 0], filename_prefix: `${LAB_DIR}/${cell}` } },
})
/** Two samplers: a graph the server always runs as heavy, whatever the lab says. */
const heavyPicture = (cell: string) => ({
  ...picture(cell),
  '1': { class_type: 'KSamplerAdvanced', inputs: {} },
  '2': { class_type: 'KSamplerAdvanced', inputs: {} },
})
/** A picture that works on the one before it in the set: its LoadImage holds the chain token. */
const chainedPicture = (cell: string) => ({ ...picture(cell), '20': { class_type: 'LoadImage', inputs: { image: CHAIN_TOKEN } } })

type LabJob = { id?: string; cell?: string; graph?: Record<string, unknown>; after?: string }

/** A POST /api/runner/groups body as lab/run/driver.ts builds one. */
function labBody(jobs: LabJob[], o: { groupId?: string; kind?: string } = {}) {
  return {
    v: 1 as const,
    group: { id: o.groupId ?? randomUUID(), desk: 'lab', kind: o.kind ?? 'set', label: 'Lab core-1 · part 1 of 1', device: 'switchgen-lab' },
    jobs: jobs.map((j, i) => {
      const cell = j.cell ?? newCell()
      return {
        id: j.id ?? randomUUID(),
        label: `Lab picture ${i + 1} of ${jobs.length}`,
        prompt: 'Three red apples and one green pear on a wooden table.',
        kind: 'image' as const,
        primary: 'image' as const,
        orFirst: true,
        noFile: 'fail' as const,
        heavy: false,
        graph: j.graph ?? (j.after ? chainedPicture(cell) : picture(cell)),
        record: { desk: 'lab', mode: 't2i' },
        ...(j.after ? { chain: { after: j.after, at: [['20', 'image']] as [string, string][] } } : {}),
        meta: { lab: { run: 'core-1', cell } },
      }
    }),
  }
}

/** Jobs of a set, with their ids and cells made up front so a chain can name them. */
function cells(n: number) {
  return Array.from({ length: n }, () => ({ id: randomUUID(), cell: newCell() }))
}

/** ComfyUI finishes a lab picture: its one file, under .lab/cells, written to disk. */
const labFile = (cell: string, node = '9') => ({ filename: fileOf(cell), subfolder: LAB_DIR, node })

const recordsOnDisk = (h: Harness): Record<string, unknown> => {
  try {
    return JSON.parse(readFileSync(path.join(h.outputs, '.switchgen', 'archive.json'), 'utf8')).records ?? {}
  } catch {
    return {}
  }
}

const sentGraph = (h: Harness, jobId: string) => h.comfy.accepted.find((a) => a.id === h.job(jobId).promptId)!.graph

// ----------------------------------------------------------------- tests --

describe('the lab desk takes its work in', () => {
  it('takes a lab set while a Pictures batch is being made, and the batch lock still holds for pictures', async () => {
    const h = await harness()
    const batch = await h.submit(groupBody({ desk: 'images', jobs: [{}, {}] }))
    expect(batch.status, batch.body).toBe(200)

    const lab = await h.submit(labBody([{}, {}]))
    expect(lab.status, lab.body).toBe(200)
    expect(lab.json().group).toMatchObject({ desk: 'lab', kind: 'set', state: 'active', device: 'switchgen-lab' })
    // A second set too: keeping to one at a time is the lab's own rule.
    expect((await h.submit(labBody([{}]))).status).toBe(200)

    // A second batch of pictures is still refused while the first is made.
    const again = await h.submit(groupBody({ desk: 'images', jobs: [{}] }))
    expect(again.status).toBe(409)
    expect(again.json()).toMatchObject({ busy: 'images' })
    await h.runner.retire()
  })

  it('takes a Pictures batch while a lab set is being made', async () => {
    const h = await harness()
    expect((await h.submit(labBody([{}, {}, {}]))).status).toBe(200)
    await h.tick()
    const batch = await h.submit(groupBody({ desk: 'images', jobs: [{}] }))
    expect(batch.status, batch.body).toBe(200)
    await h.runner.retire()
  })

  it('refuses a lab group of another kind, and names the lab among the desks', async () => {
    const h = await harness()
    const batch = await h.submit(labBody([{}], { kind: 'batch' }))
    expect(batch.status).toBe(400)
    expect(batch.json().error).toBe('a lab group must be of kind set')
    const body = labBody([{}])
    const wrong = await h.submit({ ...body, group: { ...body.group, desk: 'studio' } })
    expect(wrong.status).toBe(400)
    expect(wrong.json().error).toBe('group.desk must be images, video, reel or lab')
    await h.runner.retire()
  })

  it('refuses lab work, and does not name the desk, where the queue does not take it', async () => {
    const h = await harness({ desks: ['video', 'images', 'reel'] })
    expect(h.runner.status().desks).toEqual(['video', 'images', 'reel'])
    const r = await h.submit(labBody([{}]))
    expect(r.status).toBe(400)
    expect(r.json().error).toBe('the lab desk does not send its work through the queue on this server')
    await h.runner.retire()
  })
})

describe('a lab set as it runs', () => {
  it('files nothing in the archive, keeps each picture\'s file and time, and one failure ends only that picture', async () => {
    const h = await harness()
    const fileOnce = vi.spyOn(h.archive, 'fileOnce')
    const recordNaming = vi.spyOn(h.archive, 'recordNaming')
    const [A, B, C] = cells(3) as [{ id: string; cell: string }, { id: string; cell: string }, { id: string; cell: string }]
    const r = await h.submit(labBody([A, B, C]))
    expect(r.status, r.body).toBe(200)
    const groupId = r.json().group.id as string

    await h.tick()
    expect(h.job(A.id).status).toBe('queued')
    h.comfy.finish(h.job(A.id).promptId!, { error: 'CUDA out of memory' })
    await h.tick(1, 2000)
    expect(h.job(A.id)).toMatchObject({ status: 'failed', error: { code: 'failed' } })
    // Nothing after it was skipped, and the set is still being made.
    expect(h.job(B.id).status).toBe('queued')
    expect(h.group(groupId)).toMatchObject({ state: 'active', endedBy: null })

    h.comfy.finish(h.job(B.id).promptId!, { files: [labFile(B.cell)], start: 20_000, end: 27_500 })
    await h.tick(1, 2000)
    expect(h.job(B.id)).toMatchObject({
      status: 'done',
      entryId: null,
      entryNo: null,
      repeatOf: null,
      durationMs: 7_500,
      ranAt: 20_000,
      finishedAt: 27_500,
      primary: { filename: fileOf(B.cell), subfolder: LAB_DIR, type: 'output', kind: 'image' },
      files: [{ filename: fileOf(B.cell), subfolder: LAB_DIR, type: 'output', kind: 'image' }],
    })

    // Answered from ComfyUI's cache: still no record looked up or made, and
    // the time and the cached mark stay for the lab to leave out of its costs.
    h.comfy.finish(h.job(C.id).promptId!, { files: [labFile(C.cell)], cachedNodes: ['9'] })
    await h.tick(1, 2000)
    expect(h.job(C.id)).toMatchObject({ status: 'done', entryId: null, repeatOf: null, durationMs: 3_000, primary: { cached: true } })

    expect(h.group(groupId)).toMatchObject({ state: 'ended', endedBy: null })
    expect(fileOnce).not.toHaveBeenCalled()
    expect(recordNaming).not.toHaveBeenCalled()
    expect(recordsOnDisk(h)).toEqual({})
    await h.runner.retire()
  })

  it('fails a picture as no-file when ComfyUI wrote none, and goes on', async () => {
    const h = await harness()
    const [A, B] = cells(2) as [{ id: string; cell: string }, { id: string; cell: string }]
    await h.submit(labBody([A, B]))
    await h.tick()
    h.comfy.finish(h.job(A.id).promptId!, { files: [] })
    await h.tick(1, 2000)
    expect(h.job(A.id)).toMatchObject({ status: 'failed', error: { code: 'no-file' } })
    expect(h.job(B.id).status).toBe('queued')
    await h.runner.retire()
  })

  it('opens a chained picture on the one before it, at the literal .lab/cells path of the file that picture stands for', async () => {
    const h = await harness()
    const [A, B] = cells(2) as [{ id: string; cell: string }, { id: string; cell: string }]
    await h.submit(labBody([A, { ...B, after: A.id }]))
    await h.tick()
    expect(h.job(A.id).status).toBe('queued')
    expect(h.job(B.id)).toMatchObject({ status: 'waiting', wait: { for: 'before' }, openedOn: null })

    // The picture A stands for is its SaveImage's; a second image it wrote
    // (last, so the one a reel would take as its frame) is not what B works on.
    h.comfy.finish(h.job(A.id).promptId!, { files: [labFile(A.cell, '9'), { filename: 'extra_00001_.png', subfolder: LAB_DIR, node: '12' }] })
    await h.tick(1, 2000)
    const ref = `${LAB_DIR}/${fileOf(A.cell)} [output]`
    expect(h.job(A.id)).toMatchObject({ status: 'done', primary: { filename: fileOf(A.cell) } })
    expect(h.job(B.id)).toMatchObject({ status: 'queued', openedOn: ref })
    const g = sentGraph(h, B.id)
    expect(g['20'].inputs.image).toBe(ref)
    expect(JSON.stringify(g)).not.toContain(CHAIN_TOKEN)
    await h.runner.retire()
  })

  it('holds a chained picture back while the one it works on waits, and sends the rest of the set meanwhile', async () => {
    const h = await harness()
    // A heavy clip of the Video desk, sent, then lost to a ComfyUI restart:
    // the heavy work behind it is held until the reader says.
    const X = (await h.submit(groupBody({ desk: 'video', jobs: [{ heavy: true }] }))).json().jobs[0].id as string
    await h.tick()
    expect(h.job(X).status).toBe('queued')
    // The set: A heavy (two samplers), B works on A, C stands alone.
    const [A, B, C] = cells(3) as [{ id: string; cell: string }, { id: string; cell: string }, { id: string; cell: string }]
    const r = await h.submit(labBody([{ ...A, graph: heavyPicture(A.cell) }, { ...B, after: A.id }, C]))
    expect(r.status, r.body).toBe(200)
    expect(h.job(A.id).heavy).toBe(true)
    h.comfy.restart()
    await h.tick(3, 21_000)
    expect(h.job(X).status).toBe('lost')
    expect(h.snap().lane.held).toMatchObject({ why: 'lost', scope: 'heavy', jobId: X })

    // A is held. B waits for A rather than opening on nothing; C goes.
    expect(h.job(A.id)).toMatchObject({ status: 'waiting', wait: { for: 'held' } })
    expect(h.job(B.id)).toMatchObject({ status: 'waiting', wait: { for: 'before' }, error: null })
    expect(h.job(C.id).status).toBe('queued')
    h.comfy.finish(h.job(C.id).promptId!, { files: [labFile(C.cell)] })
    await h.tick(1, 2000)
    expect(h.job(C.id).status).toBe('done')
    expect(h.job(B.id)).toMatchObject({ status: 'waiting', wait: { for: 'before' } })

    // The reader lets the held work go: A is made, then B opens on it.
    expect((await h.post('/api/runner/lane', { action: 'send' })).status).toBe(200)
    await h.tick(1, 2000)
    expect(h.job(A.id).status).toBe('queued')
    h.comfy.finish(h.job(A.id).promptId!, { files: [labFile(A.cell)] })
    await h.tick(1, 2000)
    expect(h.job(B.id)).toMatchObject({ status: 'queued', openedOn: `${LAB_DIR}/${fileOf(A.cell)} [output]` })
    // B's prompt went only once A had ended.
    const log = h.comfy.log
    expect(log.indexOf(`history:${h.job(A.id).promptId}`)).toBeLessThan(log.indexOf(`prompt:${h.job(B.id).promptId}`))
    await h.runner.retire()
  })

  it('fails a picture chained to one that failed, as no-frame, never sending it, and the rest of the set goes on', async () => {
    const h = await harness()
    const [A, B, C] = cells(3) as [{ id: string; cell: string }, { id: string; cell: string }, { id: string; cell: string }]
    const groupId = (await h.submit(labBody([A, { ...B, after: A.id }, C]))).json().group.id as string
    await h.tick()
    h.comfy.finish(h.job(A.id).promptId!, { error: 'CUDA out of memory' })
    await h.tick(1, 2000)
    expect(h.job(A.id).status).toBe('failed')
    expect(h.job(B.id)).toMatchObject({
      status: 'failed',
      promptId: null,
      error: { code: 'no-frame', message: 'The picture it works on did not finish, or is not on disk, so there was nothing to open.' },
    })
    await h.tick(1, 2000)
    expect(h.job(C.id).status).toBe('queued')
    h.comfy.finish(h.job(C.id).promptId!, { files: [labFile(C.cell)] })
    await h.tick(1, 2000)
    expect(h.job(C.id).status).toBe('done')
    expect(h.comfy.prompts()).toEqual([h.job(A.id).promptId, h.job(C.id).promptId])
    expect(h.group(groupId)).toMatchObject({ state: 'ended', endedBy: null })
    await h.runner.retire()
  })
})

describe('the lab\'s ended pictures on the list', () => {
  it('are counted apart, so a night of them never pushes the desks\' own ended work off the list', async () => {
    const h = await harness()
    const P = (await h.submit(groupBody({ desk: 'images', jobs: [{}] }))).json().jobs[0].id as string
    await h.tick()
    h.comfy.finish(h.job(P).promptId!, { files: [{ filename: 'p.png', subfolder: 'pics' }] })
    await h.tick(1, 2000)
    expect(h.job(P).status).toBe('done')

    // Three full lab groups, each stopped before anything was sent: 600
    // ended lab pictures, more than the desks' 500 on their own.
    const groups: string[][] = []
    for (let i = 0; i < 3; i++) {
      h.clock.t += 1000
      const r = await h.submit(labBody(Array.from({ length: 200 }, () => ({}))))
      expect(r.status, r.body.slice(0, 200)).toBe(200)
      groups.push(r.json().jobs.map((j: { id: string }) => j.id))
      expect((await h.post(`/api/runner/groups/${r.json().group.id}/stop`)).status).toBe(200)
    }
    await h.tick(1, 61_000)

    const listed = h.snap().jobs
    expect(listed.find((j) => j.id === P)).toMatchObject({ status: 'done', desk: 'images' })
    const lab = listed.filter((j) => j.desk === 'lab')
    expect(lab).toHaveLength(200)
    expect(lab.map((j) => j.id)).toEqual(groups[2])
    // The saved graphs of the pictures let go went with them.
    expect(() => readFileSync(path.join(h.dir, 'jobs', `${groups[0]![0]}.json`))).toThrow()
    await h.runner.retire()
  })
})

describe('the lab desk in /api/capabilities', () => {
  let si: Awaited<ReturnType<typeof standIn>> | null = null
  let apiHandler: Handler
  const caps = async () => (await call(apiHandler, { url: '/api/capabilities' })).json()

  beforeAll(async () => {
    si = await standIn()
    process.env.COMFY_URL = si.url
    const { switchgenApi } = await import('../server/api.mjs')
    apiHandler = await mountPlugin(switchgenApi())
  })
  afterEach(async () => {
    await clearRegistry()
    delete process.env.SWITCHGEN_RUNNER_DESKS
  })
  afterAll(async () => {
    await si?.close()
  })

  it('is listed with the app\'s three by default', async () => {
    const { switchgenRunner } = await import('../server/runner.mjs')
    await mountPlugin(switchgenRunner())
    expect(await caps()).toMatchObject({ runner: true, runnerDesks: ['video', 'images', 'reel', 'lab'], runnerReason: null })
  })

  it('is left out when SWITCHGEN_RUNNER_DESKS leaves it out, and its work is refused', async () => {
    process.env.SWITCHGEN_RUNNER_DESKS = 'video,images,reel'
    const { switchgenRunner } = await import('../server/runner.mjs')
    const h = await mountPlugin(switchgenRunner())
    expect((await caps()).runnerDesks).toEqual(['video', 'images', 'reel'])
    const r = await call(h, { method: 'POST', url: '/api/runner/groups', body: labBody([{}]) })
    expect(r.status).toBe(400)
    expect(r.json().error).toBe('the lab desk does not send its work through the queue on this server')
  })

  it('is the only desk when SWITCHGEN_RUNNER_DESKS names only lab', async () => {
    process.env.SWITCHGEN_RUNNER_DESKS = ' lab '
    const { switchgenRunner } = await import('../server/runner.mjs')
    await mountPlugin(switchgenRunner())
    expect((await caps()).runnerDesks).toEqual(['lab'])
  })
})

describe('the app\'s desks and the lab\'s work', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify({ error: 'no route' }), { status: 599 }))
    vi.resetModules()
  })
  afterEach(() => vi.unstubAllGlobals())

  const job = (over: Record<string, unknown>) => ({
    id: 'j', groupId: 'g', desk: 'images', kind: 'image', seq: 1, index: 1, total: 1, label: 'x', prompt: 'p', device: 'd',
    heavy: false, status: 'queued', wait: null, stopRequested: false, stopLanded: false, promptId: 'p1', attempt: 1,
    createdAt: Date.now() - 1000, sentAt: Date.now() - 500, ranAt: null, finishedAt: null, endedAt: null, files: [], primary: null, frame: null,
    openedOn: null, entryId: null, entryNo: null, repeatOf: null, durationMs: 0, error: null, meta: null, dismissed: false, ...over,
  })
  const group = (id: string, desk: string, kind: string, jobIds: string[]) => ({ id, desk, kind, label: 'x', device: 'd', createdAt: 1, state: 'active', endedBy: null, endedAt: null, jobIds, dismissed: false })

  it('show none of it: each desk\'s bridge reads only its own desk\'s jobs', async () => {
    m.snap = {
      v: 1, available: true, reason: null, boot: 'b', rev: 1, comfy: { answering: true, since: 0 }, lane: { held: null }, progress: {}, connected: true,
      groups: [group('g1', 'images', 'batch', ['p1']), group('g2', 'lab', 'set', ['l1', 'l2'])],
      jobs: [
        job({ id: 'p1', groupId: 'g1', seq: 1 }),
        job({ id: 'l1', groupId: 'g2', desk: 'lab', seq: 2, promptId: 'pl1' }),
        job({ id: 'l2', groupId: 'g2', desk: 'lab', seq: 3, status: 'waiting', promptId: null }),
      ],
    }
    const app = await import('../src/App')
    expect(app.runnerBridgeOf('images').read().map((r) => r.key)).toEqual(['p1'])
    expect(app.runnerBridgeOf('video').read()).toEqual([])
    expect(app.runnerBridgeOf('reel').read()).toEqual([])
  })

  it('are counted by name on the shell\'s hold, which answers for them too', async () => {
    // A lost heavy lab picture holds the heavy work behind it: the lab's own
    // pictures, and a clip of the Video desk. A light picture is not held.
    const waiting = { status: 'waiting', promptId: null, sentAt: null, attempt: 0, wait: { for: 'held' } }
    m.snap = {
      v: 1, available: true, reason: null, boot: 'b', rev: 1, comfy: { answering: true, since: 0 }, progress: {}, connected: true,
      lane: { held: { why: 'lost', scope: 'heavy', jobId: 'gone', since: 5 } },
      groups: [group('g1', 'images', 'batch', ['p1']), group('g2', 'lab', 'set', ['l1', 'l2']), group('g3', 'video', 'clips', ['v1'])],
      jobs: [
        job({ ...waiting, id: 'p1', groupId: 'g1' }),
        job({ ...waiting, id: 'l1', groupId: 'g2', desk: 'lab', heavy: true }),
        job({ ...waiting, id: 'l2', groupId: 'g2', desk: 'lab', heavy: true, wait: { for: 'turn' } }),
        job({ ...waiting, id: 'v1', groupId: 'g3', desk: 'video', kind: 'video', heavy: true }),
      ],
    }
    const hold = await import('../src/components/shell/RunnerHold')
    expect(hold.heldCounts(m.snap)).toEqual({ images: 0, video: 1, reel: 0, lab: 2 })
    const said = renderToStaticMarkup(createElement(hold.RunnerHold)).replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ')
    expect(said).toContain('A heavy job was lost')
    expect(said).toContain('Waiting: 1 clip and 2 lab pictures.')

    // Only the lab's: the band still says what it holds.
    m.snap = { ...m.snap, jobs: m.snap.jobs.filter((j: { desk: string }) => j.desk === 'lab').slice(0, 1) }
    expect(hold.heldCounts(m.snap)).toEqual({ images: 0, video: 0, reel: 0, lab: 1 })
    expect(renderToStaticMarkup(createElement(hold.RunnerHold))).toContain('Waiting: 1 lab picture.')
  })
})
