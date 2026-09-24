/**
 * Both sides of the lab's lane at once: the lab's driver and client
 * (lab/run) against the app's own queue (server/runner.mjs), not the lab's
 * stand-in. The runner is built by the app tests' harness (tests/runnerFake.ts,
 * used here as it is) over a scripted ComfyUI that reaches nothing, in
 * temporary folders; the client's fetch is handed to the runner's handler in
 * process, so no port is opened. ComfyUI's node list is the lab's fixture,
 * the picture reader's answer is stood in, and the seal records its call.
 */
import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { LabEnv } from '../core/env.ts'
import { CHAIN_TOKEN, expand } from '../core/cells.ts'
import type { Cell } from '../core/types.ts'
import firstPass from '../suites/first-pass.ts'
import extEdit from '../suites/ext-edit.ts'
import { createDriver } from '../run/driver.ts'
import { readDoneCells, readLedger, runState } from '../run/ledger.ts'
import { runnerClient } from '../run/runnerClient.ts'
import type { ApiWorkflow } from '../../src/lib/comfy.ts'
import { call, type Handler } from '../../tests/http.ts'
import { harness, runnerEnv, type Harness } from '../../tests/runnerFake.ts'
import { objectInfoFrom } from './fixtures/objectInfo.ts'
import { removeTemp, tempDir } from './helpers.ts'

let restoreEnv = () => {}
let envRoot = ''
beforeAll(() => {
  // Before any server module loads: the runner's roots in a temporary folder, and the queue on.
  restoreEnv = runnerEnv()
  envRoot = path.dirname(process.env.SWITCHGEN_OUTPUTS!)
})
afterAll(() => {
  restoreEnv()
  // runnerEnv's own folder, which it does not remove.
  if (envRoot && path.basename(envRoot).startsWith('switchgen-runner-')) fs.rmSync(envRoot, { recursive: true, force: true })
  removeTemp()
})

const fp = expand([firstPass], {})
const ee = expand([extEdit], {})
const all = new Map<string, Cell>([...fp.cells, ...ee.cells].map((c) => [c.cellId, c]))
const graphs = new Map<string, ApiWorkflow>([...fp.graphs, ...ee.graphs])
const pick = (pred: (c: Cell) => boolean) => [...all.values()].filter(pred)
const noobFruit = pick((c) => c.suite === 'first-pass-core' && c.model === 'noobai' && c.slot === 'following.fruit').map((c) => c.cellId)
const kleinFruit = pick((c) => c.suite === 'first-pass-core' && c.model === 'klein' && c.slot === 'following.fruit').map((c) => c.cellId)
const thumb = pick((c) => c.chain === 'chain.thumbnail-3')
const thumbCompose = [...new Set(thumb.filter((c) => c.chainStep === 0).map((c) => c.upstream!))]
const thumbSteps = thumb.sort((a, b) => a.chainStep! - b.chainStep!).map((c) => c.cellId)

/** The client's requests, answered by the runner's own handler; capabilities and the reader as the app answers them. */
function throughRunner(h: Harness, asked: string[]) {
  return async (input: string, init: RequestInit = {}): Promise<Response> => {
    const u = new URL(input)
    const method = init.method ?? 'GET'
    asked.push(`${method} ${u.pathname}`)
    const body = typeof init.body === 'string' ? JSON.parse(init.body) : undefined
    if (u.pathname === '/api/capabilities') {
      // As server/api.mjs builds it from the queue's own status.
      const s = h.runner.status()
      return Response.json({ server: 'switchgen', runner: s.active, runnerDesks: s.active ? s.desks : [], runnerReason: s.active ? null : s.reason })
    }
    if (u.pathname === '/api/vision/tag') {
      const rows = body.images.map((im: { rel: string }, index: number) => ({ index, kind: 'output', rel: im.rel, width: 1, height: 1, rating: 'general', ratings: [], general: [], character: [] }))
      return Response.json({ tag: { rows } })
    }
    const r = await call(h.runner.handler as Handler, { method, url: u.pathname + u.search, body })
    if (r.passed) return Response.json({ error: 'no such endpoint' }, { status: 404 })
    return new Response(r.body, { status: r.status, headers: { 'content-type': 'application/json' } })
  }
}

/** ComfyUI makes every prompt it holds: the one picture under the cell's own prefix, as SaveImage writes it. */
function makeAll(h: Harness): void {
  for (const id of [...h.comfy.pending, ...h.comfy.running]) {
    const graph = h.comfy.accepted.find((a) => a.id === id)?.graph as Record<string, { class_type: string; inputs: Record<string, unknown> }>
    const [node, save] = Object.entries(graph).find(([, n]) => n.class_type === 'SaveImage')!
    const prefix = String(save.inputs.filename_prefix)
    h.comfy.finish(id, { files: [{ filename: `${path.basename(prefix)}_00001_.png`, subfolder: path.dirname(prefix), node }] })
  }
}

/**
 * A lab folder with these cells planned as run r1, and the app's queue over
 * the scripted ComfyUI. `driver()` makes a lab process; each of its waits is
 * two passes of the queue with ComfyUI making what it holds in between, and
 * `onWait` runs first.
 */
async function lane(order: string[]) {
  const root = tempDir('lab-lane-')
  const h = await harness({ root, outputs: path.join(root, 'outputs'), desks: ['video', 'images', 'reel', 'lab'] })
  const env: LabEnv = { repoRoot: '', labDir: path.join(root, 'lab'), outputs: h.outputs, appUrl: 'http://app.invalid', comfyUrl: 'http://127.0.0.1:9', port: 0, host: '127.0.0.1' }
  fs.mkdirSync(path.join(env.labDir, 'cells'), { recursive: true })
  for (const id of order) fs.writeFileSync(path.join(env.labDir, 'cells', `${id}.json`), JSON.stringify({ v: 1, cell: all.get(id), graph: graphs.get(id), text: 'slot text' }))
  const dir = path.join(env.labDir, 'runs', 'r1')
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'plan.json'), JSON.stringify({ v: 1, run: 'r1', study: 'first-pass', suites: [{ id: 'calibration', version: 1, sha: 'x' }], createdAt: Date.now() + 1e9, order, reused: [], estimate: { pictures: order.length, newPictures: order.length, seconds: order.length * 30 } }))
  const asked: string[] = []
  const sealed: string[] = []
  let clock = Date.parse('2026-09-24T22:00:00')
  const driver = (onWait: (n: number) => void = () => {}) => {
    let n = 0
    return createDriver({
      env, run: 'r1', client: runnerClient(env.appUrl, throughRunner(h, asked)), now: () => clock, pollMs: 5000,
      sleep: async (ms: number) => {
        clock += ms
        onWait(++n)
        await h.tick(1, 1000)
        makeAll(h)
        await h.tick(1, 1000)
      },
      objectInfo: async () => objectInfoFrom(graphs.values()),
      seal: async (_e, r) => void sealed.push(r),
    })
  }
  return { h, env, dir, asked, sealed, driver }
}

describe('the lab\'s driver and the app\'s queue agree', () => {
  it('a night with a chain in it: sent as sets, chained on the literal file, filed nowhere, and every ending read back', async () => {
    const order = [...noobFruit, ...thumbCompose, ...thumbSteps]
    const { h, env, dir, asked, sealed, driver } = await lane(order)
    try {
      const d = driver()
      await d.start({})
      const st = d.status()
      expect(st.state, st.message ?? '').toBe('made')
      expect([st.made, st.failed, st.total]).toEqual([order.length, 0, order.length])
      expect(sealed).toEqual(['r1'])

      // The queue took every group as a lab set, and ended every picture done.
      const snap = h.snap()
      const ours = snap.groups.filter((g) => g.desk === 'lab')
      expect(ours.length).toBeGreaterThanOrEqual(2)
      for (const g of ours) expect(g).toMatchObject({ kind: 'set', device: 'switchgen-lab', state: 'ended', endedBy: null })
      const jobs = snap.jobs.filter((j) => j.desk === 'lab')
      expect(jobs).toHaveLength(order.length)
      for (const j of jobs) expect(j.status).toBe('done')

      // A chained picture opened on the file its upstream stands for, at the literal .lab/cells path; no token reached ComfyUI.
      for (const c of thumbSteps.map((id) => all.get(id)!)) {
        const job = jobs.find((j) => (j.meta as { lab: { cell: string } }).lab.cell === c.cellId)!
        const want = `.lab/cells/${c.upstream}_00001_.png [output]`
        expect(job.openedOn).toBe(want)
        const sent = h.comfy.accepted.find((a) => a.id === job.promptId)!.graph
        expect(JSON.stringify(sent)).toContain(JSON.stringify(want))
      }
      for (const a of h.comfy.accepted) expect(JSON.stringify(a.graph)).not.toContain(CHAIN_TOKEN)

      // Nothing was filed in the archive.
      expect(jobs.every((j) => j.entryId === null)).toBe(true)
      const archive = path.join(h.outputs, '.switchgen', 'archive.json')
      const records = fs.existsSync(archive) ? JSON.parse(fs.readFileSync(archive, 'utf8')).records ?? {} : {}
      expect(records).toEqual({})

      // What the lab read back from the queue: each picture's file and times, into its ledger and its list of finished pictures.
      const ledger = readLedger(dir)
      const ended = ledger.filter((e) => e.t === 'ended') as { cell: string; status: string; primary: { filename: string; subfolder: string } | null; durationMs: number | null; ranAt: number | null; finishedAt: number | null; cached: boolean }[]
      expect(ended).toHaveLength(order.length)
      for (const e of ended) {
        expect(e.status).toBe('done')
        expect(e.primary).toMatchObject({ filename: `${e.cell}_00001_.png`, subfolder: '.lab/cells' })
        expect(e.durationMs).toBeGreaterThan(0)
        expect(typeof e.ranAt).toBe('number')
        expect(typeof e.finishedAt).toBe('number')
        expect(e.cached).toBe(false)
      }
      expect(runState(ledger, { order }).done.size).toBe(order.length)
      const done = readDoneCells(env)
      for (const id of order) expect(done.get(id)?.rel).toBe(`.lab/cells/${id}_00001_.png`)

      // The lab used only its own doors into the app, and never the lane.
      expect(new Set(asked)).toEqual(new Set(['GET /api/capabilities', 'GET /api/runner', 'POST /api/runner/groups', 'POST /api/vision/tag']))
    } finally {
      await h.runner.retire()
    }
  })

  it('a pause stops the lab\'s set on the queue; Start sends what was stopped again and ends made', async () => {
    const order = [...noobFruit, ...kleinFruit]
    const { h, dir, asked, driver } = await lane(order)
    try {
      let d1: ReturnType<typeof driver> | null = null
      d1 = driver((n) => {
        if (n === 2) void d1!.pause()
      })
      await d1.start({})
      expect(d1.status().state).toBe('paused')
      expect(d1.status().made).toBeLessThan(order.length)
      const stopped = h.snap().jobs.filter((j) => j.desk === 'lab' && j.status === 'stopped')
      expect(stopped.length).toBeGreaterThan(0)
      expect(asked).toContain('POST /api/runner/groups/' + stopped[0].groupId + '/stop')
      // The queue's word for what the lab stopped is 'stopped', and the lab takes it as its own pause.
      const L = readLedger(dir)
      expect(L.filter((e) => e.t === 'paused').map((e) => (e as { why: string }).why)).toEqual(['user'])
      expect(L.some((e) => e.t === 'ended' && e.status === 'stopped')).toBe(true)
      expect(L.some((e) => e.t === 'ended' && e.status === 'failed')).toBe(false)

      const d2 = driver()
      await d2.start({})
      const st = d2.status()
      expect(st.state, st.message ?? '').toBe('made')
      expect(st.message ?? '').not.toMatch(/stopped on the app/)
      expect([st.made, st.failed]).toEqual([order.length, 0])
      // A stopped picture went again as a new job; none of the lab's jobs is left open on the queue.
      const again = h.snap().jobs.filter((j) => j.desk === 'lab')
      expect(again.filter((j) => j.status === 'done')).toHaveLength(order.length)
      expect(again.every((j) => ['done', 'stopped'].includes(j.status))).toBe(true)
      expect(readLedger(dir).filter((e) => e.t === 'paused')).toHaveLength(1)
    } finally {
      await h.runner.retire()
    }
  })
})
