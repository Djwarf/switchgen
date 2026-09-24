/**
 * The command line, run through its main(): `lab/lab start` with no lab
 * server running refuses, in words, a start the driver would refuse, and
 * starts nothing in the background; `lab/lab prune --yes` forgets the
 * pictures it deletes and keeps those a run not sealed yet reuses; `lab/lab
 * refs describe` refuses a description that brings in a person. The
 * background process is stood in (spawn is a mock that records its call),
 * the lab server's port is 9, where nothing listens, and the lab folder and
 * outputs are temporary ones.
 */
import fs from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { runDir } from '../core/env.ts'
import { suiteById } from '../core/plan.ts'
import { appendDoneCell, readDoneCells } from '../run/ledger.ts'
import { addRef, refIndex } from '../run/refs.ts'
import { jpegHeader, removeTemp, tempEnv } from './helpers.ts'

const spawned = vi.hoisted(() => [] as unknown[][])
vi.mock('node:child_process', async (importOriginal) => {
  const real = await importOriginal<typeof import('node:child_process')>()
  const spawn = vi.fn((...a: unknown[]) => {
    spawned.push(a)
    return { pid: 4242, unref() {} }
  })
  return { ...real, default: { ...real, spawn }, spawn }
})

const env = tempEnv()
const keys = ['SWITCHGEN_LAB_DIR', 'SWITCHGEN_OUTPUTS', 'SWITCHGEN_LAB_PORT', 'SWITCHGEN_LAB_HOST'] as const
const before = Object.fromEntries(keys.map((k) => [k, process.env[k]]))
let said = ''
beforeAll(() => {
  Object.assign(process.env, { SWITCHGEN_LAB_DIR: env.labDir, SWITCHGEN_OUTPUTS: env.outputs, SWITCHGEN_LAB_PORT: '9', SWITCHGEN_LAB_HOST: '127.0.0.1' })
  vi.spyOn(process.stdout, 'write').mockImplementation((chunk: string | Uint8Array) => {
    said += String(chunk)
    return true
  })
})
afterAll(() => {
  vi.restoreAllMocks()
  for (const [k, v] of Object.entries(before)) {
    if (v === undefined) delete process.env[k]
    else process.env[k] = v
  }
  removeTemp()
})

describe('lab/lab start without the lab server', () => {
  it('refuses a night the calibration gate holds back, says why, and starts nothing', async () => {
    const cp = await import('node:child_process')
    expect(vi.isMockFunction(cp.spawn), 'the background process is stood in').toBe(true)
    const { main } = await import('../bin/lab.ts')
    const { planRun } = await import('../server/server.ts')
    planRun(env, 'core-1', [suiteById('first-pass-core')!])
    said = ''
    expect(await main(['start', 'core-1'])).toBe(1)
    expect(spawned).toEqual([])
    expect(said).toMatch(/core-1 was not started/)
    expect(said).toMatch(/calibration/i)
    // Said once, in the command line's words, not also as the page's "skip calibration".
    expect(said.match(/--skip-calibration/g)).toHaveLength(1)
    expect(said).not.toMatch(/skip calibration"/)
    expect(fs.existsSync(path.join(env.labDir, 'runs', 'core-1', 'driver.log'))).toBe(false)
  })
  it('refuses a night waiting for a photo, and names it', async () => {
    const { main } = await import('../bin/lab.ts')
    const { planRun } = await import('../server/server.ts')
    planRun(env, 'exta-1', [suiteById('ext-edit')!])
    said = ''
    expect(await main(['start', 'exta-1', '--skip-calibration'])).toBe(1)
    expect(spawned).toEqual([])
    expect(said).toMatch(/cat/)
  })
  it('hands a night that may start to the background process', async () => {
    const cp = await import('node:child_process')
    if (!vi.isMockFunction(cp.spawn)) throw new Error('spawn is not stood in, so this test does not start anything')
    const { main } = await import('../bin/lab.ts')
    said = ''
    expect(await main(['start', 'core-1', '--skip-calibration'])).toBe(0)
    expect(spawned).toHaveLength(1)
    expect(spawned[0][1]).toEqual(expect.arrayContaining(['run', 'core-1', '--skip-calibration']))
    expect(said).toMatch(/started in the background/)
  })
})

describe('lab/lab prune --yes', () => {
  it('deletes the originals of a revealed run and forgets them, so a later plan makes them again; a removal stays', async () => {
    const { main } = await import('../bin/lab.ts')
    const { studyDir } = await import('../server/server.ts')
    const [a, b, c] = ['aaaaaaaaaaaaaaaa', 'bbbbbbbbbbbbbbbb', 'cccccccccccccccc']
    const run = 'core-9'
    fs.mkdirSync(runDir(env, run), { recursive: true })
    fs.writeFileSync(path.join(runDir(env, run), 'plan.json'), JSON.stringify({ v: 1, run, study: 'pruned-study', suites: [], createdAt: 0, order: [a, b], reused: [], na: [], estimate: { pictures: 2, newPictures: 2, seconds: 1, byModel: {} }, cells: [], blocked: [], context: [], timingSource: {} }))
    fs.mkdirSync(studyDir(env, 'pruned-study'), { recursive: true })
    fs.writeFileSync(path.join(studyDir(env, 'pruned-study'), 'revealed.json'), JSON.stringify({ v: 1, study: 'pruned-study', at: 1, early: false, remaining: 0 }))
    const cells = path.join(env.outputs, '.lab', 'cells')
    fs.mkdirSync(cells, { recursive: true })
    for (const id of [a, b, c]) {
      fs.writeFileSync(path.join(cells, `${id}_00001_.png`), 'png')
      appendDoneCell(env, { cellId: id, rel: `.lab/cells/${id}_00001_.png`, durationMs: 1, cold: false, cached: false, finishedAt: 1, run })
    }
    appendDoneCell(env, { cellId: b, removed: true } as never)
    said = ''
    expect(await main(['prune', run, '--yes'])).toBe(0)
    expect(said).toMatch(/Removed 2 originals/)
    expect(fs.readdirSync(cells)).toEqual([`${c}_00001_.png`])
    const done = readDoneCells(env)
    expect(done.has(a)).toBe(false)
    expect(done.get(b)?.removed).toBe(true)
    expect(done.has(c)).toBe(true)
  })
})

describe('lab/lab prune and a smoke run planned after the smoke reveal', () => {
  // smoke-1 is made, sealed and revealed; smoke-2, planned after that reveal,
  // reuses its pictures and is not made yet.
  it('pruning smoke-1 keeps every picture smoke-2 reuses, on disk and in the list', async () => {
    const { main, smokeSuite } = await import('../bin/lab.ts')
    const { planRun, studyDir, SMOKE_STUDY } = await import('../server/server.ts')
    const { suite } = smokeSuite(env)
    const one = planRun(env, 'smoke-1', [suite])
    fs.writeFileSync(path.join(runDir(env, 'smoke-1'), 'items.json'), JSON.stringify({ v: 1, run: 'smoke-1', sealedSha: 'x', sets: [], pairs: [], checks: [] }))
    fs.mkdirSync(studyDir(env, SMOKE_STUDY), { recursive: true })
    fs.writeFileSync(path.join(studyDir(env, SMOKE_STUDY), 'revealed.json'), JSON.stringify({ v: 1, study: SMOKE_STUDY, at: 1, early: true, remaining: 0 }))
    const cells = path.join(env.outputs, '.lab', 'cells')
    fs.mkdirSync(cells, { recursive: true })
    for (const id of one.order) {
      fs.writeFileSync(path.join(cells, `${id}_00001_.png`), 'png')
      appendDoneCell(env, { cellId: id, rel: `.lab/cells/${id}_00001_.png`, durationMs: 1, cold: false, cached: false, finishedAt: 1, run: 'smoke-1' })
    }
    const two = planRun(env, 'smoke-2', [suite])
    expect(two.reused.length).toBeGreaterThan(0)
    said = ''
    expect(await main(['prune', 'smoke-1', '--yes'])).toBe(0)
    const done = readDoneCells(env)
    for (const id of two.reused) {
      expect(fs.existsSync(path.join(cells, `${id}_00001_.png`)), id).toBe(true)
      expect(done.has(id), id).toBe(true)
    }
  })
  it('smoke-2 itself is not pruned before it is made and sealed', async () => {
    const { main } = await import('../bin/lab.ts')
    const { loadPlan } = await import('../server/server.ts')
    said = ''
    expect(await main(['prune', 'smoke-2', '--yes'])).toBe(1)
    expect(said).toMatch(/smoke-2 is not made and sealed yet/)
    const reused = loadPlan(env, 'smoke-2')!.reused
    for (const id of reused) expect(fs.existsSync(path.join(env.outputs, '.lab', 'cells', `${id}_00001_.png`)), id).toBe(true)
  })
})

describe('lab/lab refs describe', () => {
  it('refuses a description that brings in a person, says it was not saved, and keeps the old words', async () => {
    const { main } = await import('../bin/lab.ts')
    addRef(env, jpegHeader({ width: 120, height: 80 }), 'cat')
    said = ''
    expect(await main(['refs', 'describe', 'cat', 'a', 'grey', 'cat'])).toBe(0)
    expect(refIndex(env).cat.describe).toBe('a grey cat')
    said = ''
    expect(await main(['refs', 'describe', 'cat', 'a', 'cat', 'on', 'my', 'lap'])).toBe(1)
    expect(said).toMatch(/^Not saved\. /)
    expect(said).toMatch(/brings a person into the picture \("my", "lap"\)/)
    expect(refIndex(env).cat.describe).toBe('a grey cat')
  })
})
