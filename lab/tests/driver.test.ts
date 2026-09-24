/**
 * The driver (TEST PLAN, C, RUN) against the stand-in runner: preflight and
 * refusals, one lab group at a time model by model, neutral labels, the
 * write-ahead ledger through lost answers, crashes and pruning, every ending,
 * pause, --until, held lanes, chains in one group, cold flags, and a whole
 * run from the real planner to the reader and the seal.
 *
 * The stand-in listens on 127.0.0.1 port 0; ComfyUI's node list is a fixture
 * built from the lab's own graphs; the clock and sleep are fake; the seal is a
 * stand-in that records the call. Nothing reaches the app or ComfyUI.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { afterAll, afterEach, describe, expect, it, vi } from 'vitest'
import type { LabEnv } from '../core/env.ts'
import { CHAIN_TOKEN, expand, finalizeGraph } from '../core/cells.ts'
import { planFrom, savePlan } from '../core/plan.ts'
import type { Cell } from '../core/types.ts'
import extEdit from '../suites/ext-edit.ts'
import firstPass from '../suites/first-pass.ts'
import { createDriver, sentPath, GROUP_MAX } from '../run/driver.ts'
import { LEDGER_FILE, appendDoneCell, appendLedger, readDoneCells, readLedger, runState } from '../run/ledger.ts'
import { runnerClient, type RunnerClient } from '../run/runnerClient.ts'
import { addRef, refIndex, setDescribe, setMask, type RefCopy } from '../run/refs.ts'
import type { ApiWorkflow } from '../../src/lib/comfy.ts'
import type { ObjectInfo } from '../core/validate.ts'
import { objectInfoFrom } from './fixtures/objectInfo.ts'
import { removeTemp, sidewaysCatStandIn, tempEnv } from './helpers.ts'
import { StandInRunner, standInRunner } from './standInRunner.ts'

// The copies of the photos the driver sends, as the copy step made them: a
// test may change them, to reach the refusals only a race between the checks
// and the copy could reach.
const copies = vi.hoisted(() => ({ tamper: null as null | ((c: Record<string, RefCopy>) => Record<string, RefCopy>) }))
vi.mock('../run/refs.ts', async (importOriginal) => {
  const m = await importOriginal<typeof import('../run/refs.ts')>()
  return {
    ...m,
    ensureRefCopies: (env: LabEnv, ids: readonly string[]) => {
      const made = m.ensureRefCopies(env, ids)
      return copies.tamper ? copies.tamper(made) : made
    },
  }
})

const open: StandInRunner[] = []
afterEach(async () => {
  copies.tamper = null
  for (const s of open.splice(0)) await s.close()
})
afterAll(removeTemp)

const fp = expand([firstPass], {})
const ee = expand([extEdit], {})
const all = new Map<string, Cell>([...fp.cells, ...ee.cells].map((c) => [c.cellId, c]))
const graphs = new Map<string, ApiWorkflow>([...fp.graphs, ...ee.graphs])
const pick = (pred: (c: Cell) => boolean) => [...all.values()].filter(pred)
const FILES = [...new Set([...all.values()].map((c) => c.file))]
const FAMILIES = [...new Set([...all.values()].map((c) => c.familyId))]
const KEYS = Object.keys(firstPass.models)

function setup(order: string[], reused: string[] = [], createdAt = Date.now() + 1e9) {
  const env = tempEnv({ appUrl: '' })
  fs.mkdirSync(path.join(env.labDir, 'cells'), { recursive: true })
  for (const id of new Set([...order, ...reused])) fs.writeFileSync(path.join(env.labDir, 'cells', `${id}.json`), JSON.stringify({ v: 1, cell: all.get(id), graph: graphs.get(id), text: 'slot text' }))
  const dir = path.join(env.labDir, 'runs', 'r1')
  fs.mkdirSync(dir, { recursive: true })
  // A calibration plan opens its own gate; createdAt in the future keeps later photo edits from counting as changes.
  fs.writeFileSync(path.join(dir, 'plan.json'), JSON.stringify({ v: 1, run: 'r1', study: 'first-pass', suites: [{ id: 'calibration', version: 1, sha: 'x' }], createdAt, order, reused, estimate: { pictures: order.length, newPictures: order.length, seconds: order.length * 30 } }))
  return { env, dir }
}

type Harness = Awaited<ReturnType<typeof harness>>

type HarnessOptions = {
  reused?: string[]
  desks?: string[]
  step?: (s: StandInRunner, n: number) => void
  env?: LabEnv
  dir?: string
  objectInfo?: () => Promise<ObjectInfo>
  /** The driver's sleep, given the fake one (which moves the clock and steps the stand-in). */
  sleep?: (ms: number, fake: (ms: number) => Promise<void>) => Promise<void>
  /** The client the driver gets, built on the real one. */
  client?: (base: RunnerClient) => RunnerClient
  /** A stand-in already running: a second process of the lab against the same app. */
  si?: StandInRunner
}

async function harness(order: string[], opts: HarnessOptions = {}) {
  const { env, dir } = opts.env && opts.dir ? { env: opts.env, dir: opts.dir } : setup(order, opts.reused)
  const si = opts.si ?? (await standInRunner(env.outputs))
  if (!opts.si) open.push(si)
  if (opts.desks) si.desks = opts.desks
  env.appUrl = si.url
  let clock = Date.parse('2026-09-24T22:00:00')
  let n = 0
  const sealed: string[] = []
  const fake = async (ms: number) => {
    clock += ms
    n++
    if (opts.step) opts.step(si, n)
    else si.step()
  }
  const base = runnerClient(si.url)
  const d = createDriver({
    env, run: 'r1', client: opts.client ? opts.client(base) : base, now: () => clock, pollMs: 5000,
    sleep: opts.sleep ? (ms: number) => opts.sleep!(ms, fake) : fake,
    objectInfo: opts.objectInfo ?? (async () => objectInfoFrom(graphs.values())),
    seal: async (_e, r) => void sealed.push(r),
  })
  return { env, dir, si, d, sealed, tick: () => clock }
}

const cellsOf = (h: Harness) => h.si.posts().flatMap((p) => p.body.jobs as { id: string; meta: { lab: { cell: string } } }[])
const noobFruit = pick((c) => c.suite === 'first-pass-core' && c.model === 'noobai' && c.slot === 'following.fruit').map((c) => c.cellId)
const kleinFruit = pick((c) => c.suite === 'first-pass-core' && c.model === 'klein' && c.slot === 'following.fruit').map((c) => c.cellId)
const thumb = pick((c) => c.chain === 'chain.thumbnail-3')
const thumbCompose = [...new Set(thumb.filter((c) => c.chainStep === 0).map((c) => c.upstream!))]
const thumbSteps = thumb.sort((a, b) => a.chainStep! - b.chainStep!).map((c) => c.cellId)

describe('preflight', () => {
  it('sends nothing before start(), and refuses without the lab desk, with an empty ledger', async () => {
    const h = await harness(noobFruit, { desks: ['images', 'video', 'reel'] })
    expect(h.d.status().state).toBe('idle')
    expect(h.si.calls).toEqual([])
    await h.d.start({})
    expect(h.d.status().state).toBe('error')
    expect(h.d.status().message).toMatch(/no lab desk/)
    expect(h.si.posts()).toHaveLength(0)
    expect(readLedger(h.dir)).toHaveLength(0)
  })
  it('refuses while the runner is off, and says why', async () => {
    const h = await harness(noobFruit)
    h.si.runner = false
    await h.d.start({})
    expect(h.d.status()).toMatchObject({ state: 'error' })
    expect(h.d.status().message).toMatch(/runner is off/)
    expect(h.si.posts()).toHaveLength(0)
  })
  it('a plan waiting for a photo gives the planner\'s refusal and sends nothing', async () => {
    const h = await harness(noobFruit)
    const plan = JSON.parse(fs.readFileSync(path.join(h.dir, 'plan.json'), 'utf8'))
    plan.blocked = [{ suite: 'ext-edit', slot: 'edit.winter', model: 'qwenEdit', chain: null, ref: 'scene', kind: 'photo', why: 'no scene photo' }]
    fs.writeFileSync(path.join(h.dir, 'plan.json'), JSON.stringify(plan))
    await h.d.start({})
    expect(h.d.status().state).toBe('error')
    expect(h.d.status().message).toMatch(/room or table photo/)
    expect(h.d.status().message).toContain(path.join(h.env.labDir, 'refs'))
    expect(h.si.calls).toHaveLength(0)
  })
  it('the calibration gate: a later night waits for the calibration\'s sweep.json unless skipped', async () => {
    const h = await harness(noobFruit)
    const plan = JSON.parse(fs.readFileSync(path.join(h.dir, 'plan.json'), 'utf8'))
    plan.suites = [{ id: 'first-pass-core', version: 1, sha: 'x' }]
    fs.writeFileSync(path.join(h.dir, 'plan.json'), JSON.stringify(plan))
    await h.d.start({})
    expect(h.d.status().state).toBe('error')
    expect(h.d.status().message).toMatch(/calibration/i)
    expect(h.si.posts()).toHaveLength(0)
    const cal = path.join(h.env.labDir, 'runs', 'cal-1')
    fs.mkdirSync(cal, { recursive: true })
    fs.writeFileSync(path.join(cal, 'plan.json'), JSON.stringify({ run: 'cal-1', study: 'first-pass', suites: [{ id: 'calibration' }], order: [] }))
    await h.d.start({})
    expect(h.d.status().message).toMatch(/not judged yet/)
    expect(h.si.posts()).toHaveLength(0)
    fs.writeFileSync(path.join(cal, 'sweep.json'), '{}')
    await h.d.start({})
    expect(h.d.status().state).toBe('made')
  })
  it('skipCalibration passes the gate', async () => {
    const h = await harness(noobFruit)
    const plan = JSON.parse(fs.readFileSync(path.join(h.dir, 'plan.json'), 'utf8'))
    plan.suites = [{ id: 'first-pass-core', version: 1, sha: 'x' }]
    fs.writeFileSync(path.join(h.dir, 'plan.json'), JSON.stringify(plan))
    await h.d.start({ skipCalibration: true })
    expect(h.d.status().state).toBe('made')
  })
  it('a graph ComfyUI cannot run (a node class it lacks) stops the run before anything is sent', async () => {
    const h = await harness(noobFruit, {
      objectInfo: async () => {
        const info = objectInfoFrom(graphs.values())
        delete info.KSampler
        return info
      },
    })
    await h.d.start({})
    expect(h.d.status().state).toBe('error')
    expect(h.d.status().message).toMatch(/failed the lab's own check, so nothing was sent/)
    expect(fs.readFileSync(path.join(h.dir, 'preflight.txt'), 'utf8')).toMatch(/no node class "KSampler"/)
    expect(h.si.posts()).toHaveLength(0)
  })
  it('a cell file changed on disk after planning stops the run before anything is sent', async () => {
    const h = await harness(noobFruit)
    const bad = JSON.parse(fs.readFileSync(path.join(h.env.labDir, 'cells', `${noobFruit[0]}.json`), 'utf8'))
    // A seed ComfyUI would take gladly: only the lab's own readback and hash can see the file was changed.
    for (const n of Object.values(bad.graph) as { class_type: string; inputs: Record<string, unknown> }[]) if (n.class_type === 'KSampler') n.inputs.seed = 7
    fs.writeFileSync(path.join(h.env.labDir, 'cells', `${noobFruit[0]}.json`), JSON.stringify(bad))
    await h.d.start({})
    expect(h.d.status().state).toBe('error')
    expect(h.d.status().message).toMatch(/failed the lab's own check, so nothing was sent/)
    expect(h.si.posts()).toHaveLength(0)
  })
  it('a missing reference photo refuses in plain words and sends nothing', async () => {
    const x = expand([extEdit], { scene: { id: 'scene', sha12: 'abcdefabcdef', ext: 'jpg', width: 800, height: 600, mask: true, rect: { x: 10, y: 10, w: 100, h: 100 } } })
    const region = x.cells.filter((c) => c.slot === 'region.scene').slice(0, 2)
    for (const c of region) {
      all.set(c.cellId, c)
      graphs.set(c.cellId, x.graphs.get(c.cellId)!)
    }
    const h = await harness(region.map((c) => c.cellId))
    await h.d.start({})
    expect(h.d.status().state).toBe('error')
    expect(h.d.status().message).toMatch(/"scene" photo is not in the lab yet/)
    expect(h.si.posts()).toHaveLength(0)
  })
})

describe('sending', () => {
  it('a full run: one group at a time, model by model, chains by token, neutral labels, then the reader and the seal', async () => {
    const order = [...noobFruit, ...kleinFruit, ...thumbCompose, ...thumbSteps]
    let maxActive = 0
    const h = await harness(order, {
      step: (si) => {
        si.step()
        maxActive = Math.max(maxActive, [...si.groups.values()].filter((g) => g.state === 'active').length)
      },
    })
    await h.d.start({})
    const st = h.d.status()
    expect(st.state, st.message ?? '').toBe('made')
    expect([st.made, st.failed, st.total]).toEqual([order.length, 0, order.length])
    expect(h.sealed).toEqual(['r1'])
    expect(maxActive).toBe(1)
    const posts = h.si.posts()
    const keys = posts.map((p) => [...new Set(p.body.jobs.map((j: { meta: { lab: { cell: string } } }) => {
      const c = all.get(j.meta.lab.cell)!
      return c.chain ? 'chain' : c.model
    }))])
    expect(keys).toEqual([['noobai'], ['klein'], ['qwen21', 'chain']])
    // The chain's steps ride in the compose group, joined by the token at exactly chain.at.
    const last = posts[2].body.jobs
    const chained = last.filter((j: { chain?: unknown }) => j.chain)
    expect(chained).toHaveLength(thumbSteps.length)
    for (const j of chained) {
      expect(j.chain.at.length).toBeGreaterThan(0)
      for (const [n, i] of j.chain.at) expect(j.graph[n].inputs[i]).toBe(CHAIN_TOKEN)
      expect(JSON.stringify(j.graph).split(CHAIN_TOKEN).length - 1).toBe(j.chain.at.length)
      const after = last.findIndex((x: { id: string }) => x.id === j.chain.after)
      expect(after).toBeGreaterThanOrEqual(0)
      expect(after).toBeLessThan(last.findIndex((x: { id: string }) => x.id === j.id))
    }
    // Labels, meta, record and prompt hold no weight file, family id or model key.
    const text = JSON.stringify(posts.map((p) => [p.body.group.label, p.body.jobs.map((j: Record<string, unknown>) => [j.label, j.meta, j.record, j.prompt])]))
    for (const f of [...FILES, ...FAMILIES]) expect(text.includes(f), f).toBe(false)
    for (const k of KEYS) expect(new RegExp(`\\b${k}\\b`, 'i').test(text), k).toBe(false)
    expect(posts[0].body.group.label).toBe('Lab r1 · part 1 of 3')
    expect(posts[0].body.jobs[0].label).toMatch(/^Lab picture 1 of \d+$/)
    for (const p of posts) {
      expect(p.body.group).toMatchObject({ desk: 'lab', kind: 'set', device: 'switchgen-lab' })
      expect(p.body.jobs.length).toBeLessThanOrEqual(GROUP_MAX)
      for (const j of p.body.jobs) {
        expect(j).toMatchObject({ heavy: false, noFile: 'fail', kind: 'image', primary: 'image' })
        expect(j.record.desk).toBe('lab')
        expect(j.meta).toEqual({ lab: { run: 'r1', cell: expect.stringMatching(/^[0-9a-f]{16}$/) } })
        const save = Object.values(j.graph as Record<string, { class_type: string; inputs: Record<string, unknown> }>).filter((n) => n.class_type === 'SaveImage')
        expect(save).toHaveLength(1)
        expect(save[0].inputs.filename_prefix).toBe(`.lab/cells/${j.meta.lab.cell}`)
      }
    }
    // cells.jsonl and cold flags: the first picture of each model cold, the rest warm.
    expect(readDoneCells(h.env).size).toBe(order.length)
    const ended = readLedger(h.dir).filter((e) => e.t === 'ended') as { cell: string; cold: boolean }[]
    expect(ended[0].cold).toBe(true)
    expect(ended[1].cold).toBe(false)
    expect(ended.find((e) => kleinFruit.includes(e.cell))!.cold).toBe(true)
    expect(h.si.calls.some((c) => c.url === '/api/vision/tag')).toBe(true)
    expect(h.si.calls.some((c) => c.url.startsWith('/api/runner/lane'))).toBe(false)
  })

  it('status, read at every poll through a chain and a held lane, never names a model, file or family', async () => {
    const order = [...noobFruit, ...kleinFruit, ...thumbCompose, ...thumbSteps]
    const seen: ReturnType<Harness['d']['status']>[] = []
    let d: Harness['d'] | null = null
    const h = await harness(order, {
      step: (si, n) => {
        si.holdLane(n > 2 && n < 8)
        seen.push(d!.status())
        si.step()
      },
    })
    d = h.d
    await h.d.start({})
    expect(seen.some((s) => s.state === 'held'), JSON.stringify(seen.map((s) => s.state))).toBe(true)
    expect(h.d.status().state).toBe('made')
    const text = JSON.stringify([...seen, h.d.status()])
    for (const f of [...FILES, ...FAMILIES]) expect(text.includes(f), f).toBe(false)
    for (const k of KEYS) expect(new RegExp(`\\b${k}\\b`, 'i').test(text), k).toBe(false)
    expect(seen.some((s) => typeof s.etaSeconds === 'number' && s.etaSeconds > 0)).toBe(true)
    expect(h.si.calls.some((c) => c.url.startsWith('/api/runner/lane'))).toBe(false)
  })

  it('a night bigger than one group is sent 200 at a time, from the real planner\'s files, with the plan\'s words as prompts', async () => {
    const env = tempEnv({ appUrl: '' })
    const seeds = Array.from({ length: 51 }, (_, i) => 1001 + i)
    const big = { ...firstPass, id: 'big', core: ['noobai'], seeds, chains: [], sweep: undefined, samplerCheck: undefined, slots: firstPass.slots.filter((s) => ['following.fruit', 'style.ghibli', 'photo.kitchen', 'text.bakery'].includes(s.id)) }
    const x = expand([big], {}, { context: [] })
    const plan = planFrom('r1', [big], x, new Map())
    expect(plan.order).toHaveLength(204)
    savePlan(env, plan, x.graphs)
    const h = await harness(plan.order, { env, dir: path.join(env.labDir, 'runs', 'r1') })
    await h.d.start({ skipCalibration: true })
    expect(h.d.status().state, h.d.status().message ?? '').toBe('made')
    expect(h.si.posts().map((p) => p.body.jobs.length)).toEqual([200, 4])
    const jobs = cellsOf(h) as unknown as { prompt: string; meta: { lab: { cell: string } } }[]
    for (const j of jobs) expect(j.prompt).toBe(plan.prompts[j.meta.lab.cell])
    expect(jobs.some((j) => j.prompt.startsWith('Three red apples'))).toBe(true)
    expect(h.sealed).toEqual(['r1'])
  })

  it('a region and an edit are sent with the photo and mask copies under .lab/refs; a mask redrawn after planning refuses', async () => {
    const envR = tempEnv()
    addRef(envR, sidewaysCatStandIn(), 'scene')
    setMask(envR, 'scene', { x: 100, y: 200, w: 800, h: 600 })
    const x = expand([extEdit], refIndex(envR))
    const pickCells = [...x.cells.filter((c) => c.slot === 'region.scene').slice(0, 2), ...x.cells.filter((c) => c.op === 'edit' && c.refs.includes('scene')).slice(0, 1)]
    expect(pickCells).toHaveLength(3)
    for (const c of pickCells) {
      all.set(c.cellId, c)
      graphs.set(c.cellId, x.graphs.get(c.cellId)!)
    }
    const h = await harness(pickCells.map((c) => c.cellId))
    fs.cpSync(path.join(envR.labDir, 'refs'), path.join(h.env.labDir, 'refs'), { recursive: true })
    await h.d.start({})
    expect(h.d.status().state, h.d.status().message ?? '').toBe('made')
    const { sha12: sha, maskSha12: maskSha } = refIndex(h.env).scene
    // The photo's copy goes by the photo's hash, the mask's by the mask's own.
    expect(maskSha).toMatch(/^[0-9a-f]{12}$/)
    const text = JSON.stringify(cellsOf(h).map((j) => (j as unknown as { graph: unknown }).graph))
    expect(text).toContain(`.lab/refs/${sha}.jpg [output]`)
    expect(text).toContain(`.lab/refs/${maskSha}.mask.png [output]`)
    expect(text).not.toContain(`.lab/refs/${sha}.mask.png`)
    expect(fs.existsSync(path.join(h.env.outputs, `.lab/refs/${sha}.jpg`))).toBe(true)
    expect(fs.existsSync(path.join(h.env.outputs, `.lab/refs/${maskSha}.mask.png`))).toBe(true)
    // The copy ComfyUI reads keeps the orientation and loses the camera.
    const copy = fs.readFileSync(path.join(h.env.outputs, `.lab/refs/${sha}.jpg`))
    expect(copy.includes(Buffer.from('PhoneModel'))).toBe(false)
    const h2 = await harness(pickCells.map((c) => c.cellId))
    fs.cpSync(path.join(envR.labDir, 'refs'), path.join(h2.env.labDir, 'refs'), { recursive: true })
    const plan = JSON.parse(fs.readFileSync(path.join(h2.dir, 'plan.json'), 'utf8'))
    plan.createdAt = Date.now() - 1000
    fs.writeFileSync(path.join(h2.dir, 'plan.json'), JSON.stringify(plan))
    setMask(h2.env, 'scene', { x: 50, y: 50, w: 400, h: 400 })
    await h2.d.start({})
    expect(h2.d.status().state).toBe('error')
    expect(h2.d.status().message).toMatch(/changed after this run was planned/)
    expect(h2.si.posts()).toHaveLength(0)
  })
})

describe('endings', () => {
  it('failed and lost are retried once with a new job id; refused and no-file are not; a second failure is final', async () => {
    const h = await harness([...noobFruit])
    h.si.fates.set(noobFruit[0], [{ status: 'failed', code: 'failed' }, { status: 'failed', code: 'failed' }])
    h.si.fates.set(noobFruit[1], [{ status: 'lost', code: 'lost' }])
    h.si.fates.set(noobFruit[2], [{ status: 'failed', code: 'refused' }])
    h.si.fates.set(noobFruit[3], [{ status: 'failed', code: 'no-file' }])
    await h.d.start({})
    const sent = (cell: string) => cellsOf(h).filter((j) => j.meta.lab.cell === cell).map((j) => j.id)
    expect(sent(noobFruit[0])).toHaveLength(2)
    expect(new Set(sent(noobFruit[0])).size).toBe(2)
    expect(sent(noobFruit[1])).toHaveLength(2)
    expect(sent(noobFruit[2])).toHaveLength(1)
    expect(sent(noobFruit[3])).toHaveLength(1)
    const s = runState(readLedger(h.dir), { order: noobFruit })
    expect([...s.failed.keys()].sort()).toEqual([noobFruit[0], noobFruit[2], noobFruit[3]].sort())
    expect(s.done.has(noobFruit[1])).toBe(true)
    expect(h.d.status().failed).toBe(3)
  })
  it('a skipped job goes back to pending and is sent again without counting as a try', async () => {
    const h = await harness([...noobFruit], {
      step: (si, n) => {
        if (n === 1) for (const j of si.jobs.values()) if (j.meta.lab.cell === noobFruit[2] && j.status === 'waiting') Object.assign(j, { status: 'skipped', endedAt: si.clock, error: { code: 'skipped', message: 'skipped', node: null, nodeType: null } })
        si.step()
      },
    })
    await h.d.start({})
    expect(h.d.status()).toMatchObject({ state: 'made', made: 4, failed: 0 })
    const L = readLedger(h.dir).filter((e) => e.t === 'ended' && e.cell === noobFruit[2]) as { status: string; attempt: number }[]
    expect(L.map((e) => [e.status, e.attempt])).toEqual([['skipped', 0], ['done', 1]])
  })
  it('a cached picture is recorded cached and never cold', async () => {
    const h = await harness([...noobFruit])
    h.si.fates.set(noobFruit[1], [{ status: 'done', cached: true, durationMs: 5 }])
    await h.d.start({})
    const e = readLedger(h.dir).find((x) => x.t === 'ended' && x.cell === noobFruit[1]) as { cached: boolean; cold: boolean }
    expect(e).toMatchObject({ cached: true, cold: false })
    expect(readDoneCells(h.env).get(noobFruit[1])?.cached).toBe(true)
  })
  it('an upstream that fails leaves its chain step not made (no-frame, not counted), then blocked once its retry fails too', async () => {
    const order = [...thumbCompose.slice(0, 1), ...thumb.filter((c) => c.upstream === thumbCompose[0] || thumb.some((u) => u.upstream === thumbCompose[0] && c.upstream === u.cellId)).map((c) => c.cellId)]
    expect(order).toHaveLength(3)
    const h = await harness(order)
    h.si.fates.set(thumbCompose[0], [{ status: 'failed', code: 'failed' }, { status: 'failed', code: 'failed' }])
    await h.d.start({})
    const L = readLedger(h.dir).filter((e) => e.t === 'ended') as { cell: string; status: string; error: { code: string } | null; attempt: number }[]
    const first = L.find((e) => e.cell === order[1])!
    expect(first.error?.code).toBe('no-frame')
    expect(first.attempt).toBe(0)
    const s = runState(readLedger(h.dir), { order })
    expect(s.failed.has(thumbCompose[0])).toBe(true)
    expect(h.d.status().failed).toBe(order.length)
    expect(h.si.posts()).toHaveLength(2)
  })
})

describe('the write-ahead ledger', () => {
  it('an answer lost after the runner kept the work: the identical body again, replayed, no duplicate job', async () => {
    const h = await harness([...noobFruit])
    h.si.dropNext = 1
    await h.d.start({})
    const posts = h.si.posts()
    expect(posts).toHaveLength(2)
    expect(posts[1].body).toEqual(posts[0].body)
    expect(readLedger(h.dir).some((e) => e.t === 'submitted' && e.replayed === true)).toBe(true)
    expect(h.si.jobs.size).toBe(4)
    expect(h.d.status().made).toBe(4)
    // The saved body was on disk before the POST.
    expect(fs.existsSync(sentPath(h.env, 'r1', posts[0].body.group.id))).toBe(true)
  })

  function savedBody(cells: string[], gid: string, prefix: string) {
    return {
      v: 1,
      group: { id: gid, desk: 'lab', kind: 'set', label: 'Lab r1 · part 1 of 1', device: 'switchgen-lab' },
      jobs: cells.map((cell, i) => ({
        id: `${prefix}${i}-2222-4222-8222-222222222222`, label: `Lab picture ${i + 1} of ${cells.length}`, prompt: 'slot text', kind: 'image', primary: 'image', orFirst: true, noFile: 'fail', heavy: false,
        graph: finalizeGraph(graphs.get(cell)!, all.get(cell)!, { refs: {} }).graph, record: { desk: 'lab', mode: 't2i' }, meta: { lab: { run: 'r1', cell } },
      })),
    }
  }

  it('a restart after a crash between "submitting" and the POST sends the saved body, identical', async () => {
    const h = await harness([...noobFruit])
    const body = savedBody(noobFruit, '11111111-1111-4111-8111-111111111111', '2222222')
    fs.mkdirSync(path.dirname(sentPath(h.env, 'r1', body.group.id)), { recursive: true })
    fs.writeFileSync(sentPath(h.env, 'r1', body.group.id), JSON.stringify(body))
    appendLedger(h.dir, { t: 'submitting', at: 1, group: body.group.id, jobs: body.jobs.map((j) => ({ job: j.id, cell: j.meta.lab.cell })) })
    await h.d.start({})
    expect(h.si.posts()[0].body).toEqual(body)
    expect(h.si.posts()).toHaveLength(1)
    expect(h.d.status().made).toBe(4)
  })

  it('a re-POST of a group the runner still lists is recorded as submitted, never refused, whatever it answers', async () => {
    for (const fault of ['409', '503'] as const) {
      const h = await harness([...noobFruit])
      const body = savedBody(noobFruit, '55555555-5555-4555-8555-555555555555', '6666666')
      const taken = await runnerClient(h.si.url).submit(body as never)
      expect(taken.ok).toBe(true)
      const saved = fault === '409' ? { ...body, group: { ...body.group, label: 'Lab r1 · part 1 of 2' } } : body
      if (fault === '503') h.si.offPosts = 1
      fs.mkdirSync(path.dirname(sentPath(h.env, 'r1', body.group.id)), { recursive: true })
      fs.writeFileSync(sentPath(h.env, 'r1', body.group.id), JSON.stringify(saved))
      appendLedger(h.dir, { t: 'submitting', at: 1, group: body.group.id, jobs: body.jobs.map((j) => ({ job: j.id, cell: j.meta.lab.cell })) })
      await h.d.start({})
      const L = readLedger(h.dir)
      expect(L.some((e) => e.t === 'refused'), fault).toBe(false)
      expect(L.some((e) => e.t === 'submitted' && e.group === body.group.id), fault).toBe(true)
      expect(h.d.status().made, fault).toBe(4)
      expect(h.si.groups.size, fault).toBe(1)
    }
  })

  it('a submitted group whose jobs were pruned: a picture on disk is recovered (and in cells.jsonl), the rest lost and sent again', async () => {
    const h = await harness([...noobFruit])
    const gid = '33333333-3333-4333-8333-333333333333'
    appendLedger(h.dir, { t: 'submitting', at: 1, group: gid, jobs: noobFruit.map((cell, i) => ({ job: `4444444${i}-4444-4444-8444-444444444444`, cell })) })
    appendLedger(h.dir, { t: 'submitted', at: 2, group: gid, replayed: false })
    fs.mkdirSync(path.join(h.env.outputs, '.lab/cells'), { recursive: true })
    fs.writeFileSync(path.join(h.env.outputs, `.lab/cells/${noobFruit[0]}_00001_.png`), 'png')
    await h.d.start({})
    const L = readLedger(h.dir)
    expect(L.some((e) => e.t === 'recovered' && e.cell === noobFruit[0])).toBe(true)
    expect(L.filter((e) => e.t === 'ended' && e.status === 'lost')).toHaveLength(3)
    expect(readDoneCells(h.env).get(noobFruit[0])).toMatchObject({ rel: `.lab/cells/${noobFruit[0]}_00001_.png`, durationMs: 0 })
    expect(h.d.status().made).toBe(4)
    expect(cellsOf(h).map((j) => j.meta.lab.cell).sort()).toEqual(noobFruit.slice(1).sort())
  })

  it('a 503 on a POST is written as refused 503, and the next group has new group and job ids', async () => {
    const h = await harness([...noobFruit])
    h.si.offPosts = 1
    await h.d.start({})
    const posts = h.si.posts()
    expect(posts).toHaveLength(2)
    expect(posts[1].body.group.id).not.toBe(posts[0].body.group.id)
    expect(posts[1].body.jobs[0].id).not.toBe(posts[0].body.jobs[0].id)
    expect(readLedger(h.dir).some((e) => e.t === 'refused' && e.status === 503)).toBe(true)
    expect(h.d.status().made).toBe(4)
  })

  it('a runner that stays off pauses the run (runner-off) and never turns jobs in flight into lost', async () => {
    const h = await harness([...noobFruit], {
      step: (si, n) => {
        if (n >= 2) {
          si.available = false
          si.jobs.clear()
        }
        si.step()
      },
    })
    await h.d.start({})
    expect(h.d.status().state).toBe('paused')
    const L = readLedger(h.dir)
    expect(L.some((e) => e.t === 'paused' && e.why === 'runner-off')).toBe(true)
    expect(L.some((e) => e.t === 'ended' && e.status === 'lost')).toBe(false)
  })
})

describe('pause, until and holds', () => {
  it('pause() stops the group, writes paused (user), and start() carries on to the end', async () => {
    const order = [...noobFruit, ...kleinFruit]
    let d: Harness['d'] | null = null
    const h = await harness(order, {
      step: (si, n) => {
        si.step()
        if (n === 3) void d!.pause()
      },
    })
    d = h.d
    await h.d.start({})
    expect(h.d.status().state).toBe('paused')
    const L = readLedger(h.dir)
    expect(L.some((e) => e.t === 'paused' && e.why === 'user')).toBe(true)
    expect(h.si.calls.some((c) => c.method === 'POST' && /\/api\/runner\/groups\/[^/]+\/stop$/.test(c.url))).toBe(true)
    expect(runState(L, { order }).paused).toBe(true)
    expect(h.d.status().made).toBeLessThan(order.length)
    await h.d.start({})
    expect(h.d.status()).toMatchObject({ state: 'made', made: order.length })
  })
  it('--until stops the group at the deadline on a fake clock and writes paused (until)', async () => {
    const order = [...noobFruit, ...kleinFruit]
    const h = await harness(order)
    const until = new Date(h.tick() + 60_000)
    const hhmm = `${String(until.getHours()).padStart(2, '0')}:${String(until.getMinutes()).padStart(2, '0')}`
    await h.d.start({ until: hhmm })
    expect(h.d.status().state).toBe('paused')
    expect(h.d.status().message).toMatch(/Stopped at/)
    expect(readLedger(h.dir).some((e) => e.t === 'paused' && e.why === 'until')).toBe(true)
    expect(h.si.calls.some((c) => /\/stop$/.test(c.url))).toBe(true)
    expect(h.d.status().made).toBeLessThan(order.length)
    expect(h.tick()).toBeGreaterThanOrEqual(until.getTime())
  })
  it('a held lane is shown as held and waited on; the lab never asks for the lane', async () => {
    const states: string[] = []
    let d: Harness['d'] | null = null
    const h = await harness([...noobFruit], {
      step: (si, n) => {
        si.holdLane(n < 6)
        states.push(d!.status().state)
        si.step()
      },
    })
    d = h.d
    await h.d.start({})
    expect(states).toContain('held')
    expect(h.d.status().state).toBe('made')
    expect(h.si.calls.some((c) => c.url.includes('/lane'))).toBe(false)
  })
  it('a stop the lab asked for, from a process that ended before it saw the stop land, is not taken as a stop made on the app', async () => {
    const order = [...noobFruit, ...kleinFruit]
    const { env, dir } = setup(order)
    let d1: Harness['d'] | null = null
    const h1 = await harness(order, {
      env, dir,
      step: (si, n) => {
        si.step()
        if (n === 3) void d1!.pause()
      },
    })
    d1 = h1.d
    await h1.d.start({})
    expect(h1.d.status().state).toBe('paused')
    // The process ended right after the pause (Ctrl-C on lab/lab serve): the
    // stops reached the runner, but no ending was written.
    const L = readLedger(dir)
    const at = L.findIndex((e) => e.t === 'paused')
    const kept = L.filter((e, i) => !(i > at && e.t === 'ended' && e.status === 'stopped'))
    expect(kept.length).toBeLessThan(L.length)
    fs.writeFileSync(path.join(dir, LEDGER_FILE), kept.map((e) => JSON.stringify(e)).join('\n') + '\n')
    expect(runState(kept, { order }).pausedAfter.size).toBe(1)
    // A new process, the same app: Start goes on to the end, with one pause in the ledger.
    const h2 = await harness(order, { env, dir, si: h1.si })
    await h2.d.start({})
    const st = h2.d.status()
    expect(st.message ?? '').not.toMatch(/stopped on the app/)
    expect(st.state, st.message ?? '').toBe('made')
    expect(st.made).toBe(order.length)
    expect(readLedger(dir).filter((e) => e.t === 'paused' && e.why === 'user')).toHaveLength(1)
  })
  it('runState: only a pause by the user or --until marks the groups sent before it as stopped by the lab', () => {
    const g1 = '11111111-1111-4111-8111-111111111111'
    const g2 = '22222222-2222-4222-8222-222222222222'
    const s = runState([
      { t: 'submitting', at: 1, group: g1, jobs: [] },
      { t: 'paused', at: 2, why: 'runner-off' },
      { t: 'submitting', at: 3, group: g2, jobs: [] },
      { t: 'paused', at: 4, why: 'until' },
    ], { order: [] })
    expect([...s.pausedAfter].sort()).toEqual([g1, g2])
    const t = runState([
      { t: 'submitting', at: 1, group: g1, jobs: [] },
      { t: 'paused', at: 2, why: 'runner-off' },
      { t: 'submitting', at: 3, group: g2, jobs: [] },
    ], { order: [] })
    expect(t.pausedAfter.size).toBe(0)
  })
  it('a stop made on the app pauses the run instead of sending the pictures again', async () => {
    const h = await harness([...noobFruit], {
      step: (si, n) => {
        if (n === 3) si.stopFromApp()
        si.step()
      },
    })
    await h.d.start({})
    expect(h.d.status().state).toBe('paused')
    expect(h.d.status().message).toMatch(/stopped on the app/)
    expect(h.si.posts()).toHaveLength(1)
    assert.ok(readLedger(h.dir).some((e) => e.t === 'paused' && e.why === 'user'))
  })
})

describe('a pause while the picture reader waits for memory', () => {
  // The reader waits 30 s and more between tries while the machine is short
  // of memory; here such a wait never ends by itself.
  const never = (ms: number, fake: (ms: number) => Promise<void>) => (ms >= 30_000 ? new Promise<void>(() => {}) : fake(ms))
  const within = <T>(p: Promise<T>, ms: number) => Promise.race([p.then(() => 'ended'), new Promise((r) => setTimeout(() => r('still waiting'), ms))])

  it('cuts the wait short: the run pauses at once and says so', async () => {
    const h = await harness([...noobFruit], { sleep: never })
    h.si.tagBusy = 1000
    const going = h.d.start({})
    for (let i = 0; i < 200 && !(h.d.status().state === 'reading' && h.si.calls.some((c) => c.url === '/api/vision/tag')); i++) await new Promise((r) => setTimeout(r, 10))
    expect(h.d.status().state).toBe('reading')
    await h.d.pause()
    expect(await within(going, 2000)).toBe('ended')
    expect(h.d.status().state).toBe('paused')
    expect(h.d.status().message).toMatch(/Paused while reading/)
  })
  it('a pause asked during a read call skips the wait that follows it, and says it is pausing', async () => {
    let d: Harness['d'] | null = null
    const messages: string[] = []
    const h = await harness([...noobFruit], {
      sleep: never,
      client: (base) => ({
        ...base,
        async tag(rels) {
          void d!.pause()
          messages.push(d!.status().message ?? '')
          return base.tag(rels)
        },
      }),
    })
    d = h.d
    h.si.tagBusy = 1000
    expect(await within(h.d.start({}), 2000)).toBe('ended')
    expect(h.d.status().state).toBe('paused')
    expect(messages[0]).toMatch(/Pausing: the picture reader stops/)
  })
})

describe('a description saved after a run started', () => {
  // Two cells that use the cat photo, planned at 2000 with the suite's words;
  // the photo arrived at 1000. A started run keeps the words it was planned
  // with (a run not started yet is planned again at Start, by the server).
  async function catRun() {
    const envR = tempEnv()
    addRef(envR, sidewaysCatStandIn(), 'cat', () => 1000)
    const x = expand([extEdit], refIndex(envR))
    const cat = x.cells.filter((c) => c.refs.includes('cat') && !c.upstream && x.prompts.get(c.cellId)?.includes('tortoiseshell')).slice(0, 2)
    expect(cat).toHaveLength(2)
    for (const c of cat) {
      all.set(c.cellId, c)
      graphs.set(c.cellId, x.graphs.get(c.cellId)!)
    }
    const { env, dir } = setup(cat.map((c) => c.cellId), [], 2000)
    fs.cpSync(path.join(envR.labDir, 'refs'), path.join(env.labDir, 'refs'), { recursive: true })
    return { env, dir, order: cat.map((c) => c.cellId) }
  }
  it('a paused run goes on to the end with the words it was planned with, never the new ones', async () => {
    const { env, dir, order } = await catRun()
    let d: Harness['d'] | null = null
    const h = await harness(order, {
      env, dir,
      step: (si, n) => {
        si.step()
        if (n === 1) void d!.pause()
      },
    })
    d = h.d
    await h.d.start({ skipCalibration: true })
    expect(h.d.status().state).toBe('paused')
    expect(readLedger(dir).length).toBeGreaterThan(0)
    setDescribe(env, 'cat', 'a grey cat asleep on a blue cushion')
    await h.d.start({ skipCalibration: true })
    expect(h.d.status().state, h.d.status().message ?? '').toBe('made')
    const sent = cellsOf(h).map((j) => JSON.stringify((j as unknown as { graph: unknown }).graph))
    expect(sent.length).toBeGreaterThan(0)
    for (const g of sent) {
      expect(g).toContain('tortoiseshell')
      expect(g).not.toContain('grey cat asleep')
    }
  })
  it('the suite\'s own words saved from the page do not stop the run', async () => {
    const { env, dir, order } = await catRun()
    // What the page sends: the words it shows, the suite's own until changed.
    setDescribe(env, 'cat', extEdit.refs.cat.describe)
    const h = await harness(order, { env, dir })
    await h.d.start({})
    expect(h.d.status().state, h.d.status().message ?? '').toBe('made')
  })
})

describe('a rectangle drawn again', () => {
  // Two region cells of the scene photo (upright 3060 x 4080), planned at
  // 2000 with the rectangle drawn at 1000.
  const PLANNED = { x: 100, y: 200, w: 800, h: 600 }
  async function regionRun() {
    const envR = tempEnv()
    addRef(envR, sidewaysCatStandIn(), 'scene', () => 1000)
    setMask(envR, 'scene', PLANNED, () => 1000)
    const x = expand([extEdit], refIndex(envR))
    const cells = x.cells.filter((c) => c.slot === 'region.scene').slice(0, 2)
    expect(cells).toHaveLength(2)
    for (const c of cells) {
      all.set(c.cellId, c)
      graphs.set(c.cellId, x.graphs.get(c.cellId)!)
    }
    const { env, dir } = setup(cells.map((c) => c.cellId), [], 2000)
    fs.cpSync(path.join(envR.labDir, 'refs'), path.join(env.labDir, 'refs'), { recursive: true })
    return { env, dir, order: cells.map((c) => c.cellId) }
  }

  it('near the whole frame gives new region cells, sent with the new mask under its own hash', async () => {
    const envR = tempEnv()
    addRef(envR, sidewaysCatStandIn(), 'scene')
    const { width: W, height: H } = refIndex(envR).scene
    const regions = () => expand([extEdit], refIndex(envR)).cells.filter((c) => c.slot === 'region.scene' && c.op === 'region')
    setMask(envR, 'scene', { x: 2, y: 2, w: W - 4, h: H - 4 })
    const first = refIndex(envR).scene.maskSha12
    const before = regions()
    setMask(envR, 'scene', { x: 3, y: 3, w: W - 6, h: H - 6 })
    const second = refIndex(envR).scene.maskSha12
    expect(first).toMatch(/^[0-9a-f]{12}$/)
    expect(second).toMatch(/^[0-9a-f]{12}$/)
    expect(second).not.toBe(first)
    // Both rectangles crop to the whole frame: only the mask's own hash tells the cells apart.
    const x = expand([extEdit], refIndex(envR))
    const after = x.cells.filter((c) => c.slot === 'region.scene' && c.op === 'region')
    expect(after.length).toBe(before.length)
    const old = new Set(before.map((c) => c.cellId))
    expect(after.filter((c) => old.has(c.cellId))).toEqual([])
    const two = after.slice(0, 2)
    for (const c of two) {
      all.set(c.cellId, c)
      graphs.set(c.cellId, x.graphs.get(c.cellId)!)
    }
    const h = await harness(two.map((c) => c.cellId))
    fs.cpSync(path.join(envR.labDir, 'refs'), path.join(h.env.labDir, 'refs'), { recursive: true })
    await h.d.start({})
    expect(h.d.status().state, h.d.status().message ?? '').toBe('made')
    const photo = refIndex(h.env).scene.sha12
    const text = JSON.stringify(cellsOf(h).map((j) => (j as unknown as { graph: unknown }).graph))
    expect(text).toContain(`.lab/refs/${second}.mask.png [output]`)
    expect(text).not.toContain(`.lab/refs/${photo}.mask.png`)
    const copy = fs.readFileSync(path.join(h.env.outputs, '.lab', 'refs', `${second}.mask.png`))
    expect(copy).toEqual(fs.readFileSync(path.join(h.env.labDir, 'refs', 'scene.mask.png')))
  })

  it('after a run started, says how to go on (a new run, with the command) and never "Plan the run again"; the planned rectangle again goes on', async () => {
    const { env, dir, order } = await regionRun()
    const plan = JSON.parse(fs.readFileSync(path.join(dir, 'plan.json'), 'utf8'))
    plan.suites = [{ id: 'ext-edit', version: 1, sha: 'x' }]
    fs.writeFileSync(path.join(dir, 'plan.json'), JSON.stringify(plan))
    appendLedger(dir, { t: 'paused', at: 1500, why: 'user' })
    setMask(env, 'scene', { x: 50, y: 50, w: 400, h: 400 })
    const h = await harness(order, { env, dir })
    await h.d.start({ skipCalibration: true })
    const msg = h.d.status().message ?? ''
    expect(h.d.status().state).toBe('error')
    expect(msg).toMatch(/area marked on the "scene" photo changed after this run started, and a started run keeps its plan/)
    expect(msg).toContain('lab/lab plan ext-edit --run r2')
    expect(msg).not.toMatch(/Plan the run again/)
    expect(h.si.posts()).toHaveLength(0)
    setMask(env, 'scene', PLANNED)
    await h.d.start({ skipCalibration: true })
    expect(h.d.status().state, h.d.status().message ?? '').toBe('made')
  })

  it('before a run started, the refusal writes nothing to the ledger, not even a picture another run removed', async () => {
    const { env, dir, order } = await regionRun()
    appendDoneCell(env, { cellId: order[0], rel: '', durationMs: 0, cold: false, cached: false, finishedAt: null, run: 'other-1', removed: true })
    setMask(env, 'scene', { x: 50, y: 50, w: 400, h: 400 })
    const h = await harness(order, { env, dir })
    await h.d.start({})
    expect(h.d.status().state).toBe('error')
    expect(h.d.status().message).toMatch(/changed after this run was planned\. Plan the run again\./)
    // An empty ledger: the run has sent nothing, so the server may still plan it again.
    expect(readLedger(dir)).toHaveLength(0)
    expect(h.si.posts()).toHaveLength(0)
  })
})

describe('a pause asked while the runner is being read', () => {
  it('before the first group: nothing is sent or written after the pause', async () => {
    let d: Harness['d'] | null = null
    let asked = false
    const h = await harness([...noobFruit], {
      client: (base) => ({
        ...base,
        async snapshot() {
          if (!asked) {
            asked = true
            void d!.pause()
          }
          return base.snapshot()
        },
      }),
    })
    d = h.d
    await h.d.start({})
    expect(h.d.status().state).toBe('paused')
    expect(h.si.posts()).toHaveLength(0)
    const L = readLedger(h.dir)
    expect(L.some((e) => e.t === 'submitting')).toBe(false)
    expect(L.some((e) => e.t === 'paused' && e.why === 'user')).toBe(true)
  })
  it('between groups: the next group is not sent, and the ledger has no group after the pause', async () => {
    const order = [...noobFruit, ...kleinFruit]
    let d: Harness['d'] | null = null
    let si: StandInRunner | null = null
    let asked = false
    const h = await harness(order, {
      client: (base) => ({
        ...base,
        async snapshot() {
          const snap = await base.snapshot()
          const first = snap.jobs.filter((j) => noobFruit.includes((j.meta as { lab?: { cell?: string } } | null)?.lab?.cell ?? ''))
          if (!asked && si!.posts().length === 1 && first.length === noobFruit.length && first.every((j) => j.status === 'done')) {
            asked = true
            void d!.pause()
          }
          return snap
        },
      }),
    })
    d = h.d
    si = h.si
    await h.d.start({})
    expect(asked, 'the pause was asked with the first group made').toBe(true)
    expect(h.d.status().state).toBe('paused')
    expect(h.si.posts()).toHaveLength(1)
    const L = readLedger(h.dir)
    const at = L.findIndex((e) => e.t === 'paused')
    expect(at).toBeGreaterThanOrEqual(0)
    expect(L.slice(at).some((e) => e.t === 'submitting')).toBe(false)
    expect(h.d.status().made).toBe(noobFruit.length)
  })
})

describe('a photo that is not ready when the graphs are bound is named, in words', () => {
  // The two checks before the copy pass; the copy step is then made to leave
  // the photo out, drop its mask or copy other bytes, as a race could.
  async function regionRun() {
    const envR = tempEnv()
    addRef(envR, sidewaysCatStandIn(), 'scene', () => 1000)
    setMask(envR, 'scene', { x: 100, y: 200, w: 800, h: 600 }, () => 1000)
    const x = expand([extEdit], refIndex(envR))
    const cells = x.cells.filter((c) => c.slot === 'region.scene').slice(0, 2)
    expect(cells).toHaveLength(2)
    for (const c of cells) {
      all.set(c.cellId, c)
      graphs.set(c.cellId, x.graphs.get(c.cellId)!)
    }
    const { env, dir } = setup(cells.map((c) => c.cellId), [], 2000)
    fs.cpSync(path.join(envR.labDir, 'refs'), path.join(env.labDir, 'refs'), { recursive: true })
    return { env, dir, order: cells.map((c) => c.cellId) }
  }
  it('a photo left out of the copies', async () => {
    const { env, dir, order } = await regionRun()
    copies.tamper = () => ({})
    const h = await harness(order, { env, dir })
    await h.d.start({})
    const msg = h.d.status().message ?? ''
    expect(h.d.status().state).toBe('error')
    expect(msg).toMatch(/^The "scene" photo was not copied where ComfyUI reads it\./)
    expect(msg).not.toMatch(/undefined|No file was given|preflight\.txt/)
    expect(fs.existsSync(path.join(dir, 'preflight.txt'))).toBe(false)
    expect(h.si.posts()).toHaveLength(0)
  })
  it('a mask that went missing between the checks and the copy', async () => {
    const { env, dir, order } = await regionRun()
    copies.tamper = (c) => Object.fromEntries(Object.entries(c).map(([k, v]) => [k, { ...v, mask: null }]))
    const h = await harness(order, { env, dir })
    await h.d.start({})
    expect(h.d.status().state).toBe('error')
    expect(h.d.status().message).toBe('The "scene" photo has no area marked for redrawing yet. Draw the box on the lab page.')
    expect(h.si.posts()).toHaveLength(0)
  })
  it('a copy of other bytes than the ones planned', async () => {
    const { env, dir, order } = await regionRun()
    copies.tamper = (c) => Object.fromEntries(Object.entries(c).map(([k, v]) => [k, { ...v, sha12: '000000000000', ref: `.lab/refs/000000000000.${v.ext}`, mask: v.mask && '.lab/refs/000000000000.mask.png' }]))
    const h = await harness(order, { env, dir })
    await h.d.start({})
    expect(h.d.status().message).toMatch(/replaced or removed since\. Plan the run again\./)
    expect(h.si.posts()).toHaveLength(0)
  })
  it('untouched copies still make the run', async () => {
    const { env, dir, order } = await regionRun()
    const h = await harness(order, { env, dir })
    await h.d.start({})
    expect(h.d.status().state, h.d.status().message ?? '').toBe('made')
  })
})
