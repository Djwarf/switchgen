/**
 * The lab server (TEST PLAN, E), driven through its Connect-style handler with
 * a call() that needs no socket: host and origin guards, the photo routes, the
 * blind gate, judging, events, start and pause with a stand-in driver, the
 * reveal, and the CLI helpers the server shares (the blind check, the import
 * scan, the smoke suite). A sealed fixture run holds made-up tokens and
 * header-only WebP copies. The app and ComfyUI point at port 9, the driver is
 * a stand-in that records its calls, and nothing is sent.
 */
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'
import { afterAll, describe, it } from 'vitest'
import { runDir } from '../core/env.ts'
import { suiteById } from '../core/plan.ts'
import type { DriverState } from '../run/driver.ts'
import { sizeOf } from '../run/imagesize.ts'
import { appendDoneCell, readDoneCells, readLedger } from '../run/ledger.ts'
import { readReadings, readingsPath } from '../run/reader.ts'
import { addRef, listRefs } from '../run/refs.ts'
import * as S from '../server/server.ts'
import * as CLI from '../bin/lab.ts'
import { rectMaskPng } from '../run/masks.ts'
import { machineBoot } from '../run/lock.ts'
import { checkSuite, descriptionProblem } from '../core/suite.ts'
import type { LabEnv } from '../core/env.ts'
import { REPO, jpegHeader, removeTemp, tempDir, tempEnv, webpHeader } from './helpers.ts'

afterAll(removeTemp)

const env = tempEnv()
const root = path.dirname(env.labDir)

type Reply = { status: number; headers: Record<string, string | number>; body: string; json: () => any; passed: boolean }
/** Every answer the handlers gave, with the path it answered. */
const bodies: { url: string; body: string }[] = []
function call(handler: any, opts: { method?: string; url: string; headers?: Record<string, string>; body?: unknown; raw?: Buffer }): Promise<Reply> {
  const method = opts.method ?? 'GET'
  const chunks = opts.raw ? [opts.raw] : opts.body === undefined ? [] : [Buffer.from(JSON.stringify(opts.body))]
  const headers: Record<string, string> = {
    host: '127.0.0.1:5274',
    ...(opts.body !== undefined ? { 'content-type': 'application/json' } : {}),
    ...opts.headers,
  }
  const req = Object.assign(Readable.from(chunks), { method, url: opts.url, headers })
  const reply: Reply = { status: 200, headers: {}, body: '', passed: false, json: () => JSON.parse(reply.body) }
  return new Promise((resolve) => {
    const res: any = Object.assign(new EventEmitter(), {
      req,
      destroyed: false,
      headersSent: false,
      setHeader(k: string, v: string | number) {
        reply.headers[k.toLowerCase()] = v
      },
      getHeader(k: string) {
        return reply.headers[k.toLowerCase()]
      },
      writeHead(code: number) {
        reply.status = code
        return res
      },
      write(c: string | Buffer) {
        reply.body += c.toString()
        return true
      },
      end(c?: string | Buffer) {
        if (c !== undefined) reply.body += Buffer.isBuffer(c) ? c.toString('latin1') : c
        bodies.push({ url: opts.url, body: reply.body })
        resolve(reply)
      },
    })
    Object.defineProperty(res, 'statusCode', { get: () => reply.status, set: (v: number) => (reply.status = v) })
    handler(req, res, () => {
      reply.passed = true
      resolve(reply)
    })
  })
}

// ------------------------------------------------------------- fixture --

const T = (n: number) => `tok${String(n).padStart(3, '0')}${'x'.repeat(26)}`
const run = 'core-1'
const dir = runDir(env, run)
fs.mkdirSync(path.join(dir, 'view'), { recursive: true })
const plan = {
  v: 1, run, study: 'first-pass', suites: [{ id: 'first-pass-core', version: 1, sha: 'x' }], createdAt: 0,
  order: ['aaaaaaaaaaaaaaaa'], reused: [], na: [], estimate: { pictures: 1, newPictures: 1, seconds: 10, byModel: {} },
  cells: [], blocked: [], context: [], timingSource: {},
}
fs.writeFileSync(path.join(dir, 'plan.json'), JSON.stringify(plan))
const tiles = (a: number) => [T(a), T(a + 1), T(a + 2), T(a + 3)]
const items = {
  v: 1, run, sealedSha: 'abc',
  sets: [
    { setId: 's1', order: 0, block: 'following', card: 'pf@1', second: 'va@1', mode: 'scale', closeLook: false, brief: { task: 'Three red apples and one green pear.' },
      grids: [{ itemId: 's1.A', letter: 'A', pos: 0, tiles: tiles(0) }, { itemId: 's1.B', letter: 'B', pos: 1, tiles: tiles(4) }] },
    { setId: 's1-2', order: 5, block: 'following', card: 'pf@1', mode: 'scale', closeLook: false, brief: { task: 'Three red apples and one green pear.' }, secondOf: 's1',
      grids: [{ itemId: 's1-2.C', letter: 'C', pos: 0, tiles: tiles(0), repeats: 's1.A' }] },
  ],
  pairs: [], checks: [],
}
fs.writeFileSync(path.join(dir, 'items.json'), JSON.stringify(items))
const tokens: Record<string, unknown> = {}
for (let i = 0; i < 8; i++) {
  tokens[T(i)] = { cellId: `c${i}`.padEnd(16, '0'), contestant: i < 4 ? 'noobai' : 'klein', familyId: i < 4 ? 'sdxl-illustrious' : 'flux2-klein',
    file: i < 4 ? 'NoobAI-XL-v1.1.safetensors' : 'flux-2-klein-4b-fp8.safetensors', slot: 'following.fruit', seed: 1001, op: 't2i', condition: null,
    width: 1024, height: 1024, rel: `.lab/cells/c${i}_00001_.png`, durationMs: 1000, cold: false, jobId: `job-SEALEDMARKER-${i}`, promptId: `prompt-${i}-sealedmarker` }
}
fs.writeFileSync(path.join(dir, 'sealed.json'), JSON.stringify({ v: 1, run, study: 'first-pass', sealedAt: 0, suites: [], tokens, gridOf: {}, letters: { s1: { A: 'noobai', B: 'klein' } }, pairOf: {}, checkOf: {}, sets: {}, slots: {}, cells: {}, contestants: {} }))
// A real, clean WebP for every token.
const webp = path.join(root, 'clean.webp')
fs.writeFileSync(webp, webpHeader('VP8L', 64, 64))
for (let i = 0; i < 8; i++) for (const s of ['g', 'f']) fs.copyFileSync(webp, path.join(dir, 'view', `${T(i)}-${s}.webp`))

const driverCalls: string[] = []
let driverState: DriverState = 'idle'
const handler = S.createLabHandler({
  env,
  log: () => {},
  blindCheck: async (r: string) => {
    const rep = await CLI.checkBlind(env, r)
    S.writeBlindMarker(env, r, { ok: rep.ok, checked: rep.checked, problems: rep.problems.length })
  },
  makeDriver: (r: string) => ({
    async start(o: unknown) {
      driverCalls.push(`start ${r} ${JSON.stringify(o)}`)
      driverState = 'sending'
    },
    async pause() {
      driverCalls.push(`pause ${r}`)
      driverState = 'paused'
    },
    status: () => ({ run: r, state: driverState, made: 0, total: 1, failed: 0, etaSeconds: null, until: null, message: null }),
  }),
})

const ev = (item: string, kind: string, value?: unknown, extra: Record<string, unknown> = {}) => ({
  v: 1, id: `e-${Math.random().toString(36).slice(2)}`, at: Date.now(), judge: 'you', session: 'sess-1', device: { w: 400, h: 800, dpr: 3 },
  run, item, kind, ...(value !== undefined ? { value } : {}), dwellMs: 5000, ...extra,
})

// ---------------------------------------------------------------- server
it('health', async () => {
  const r = await call(handler, { url: '/api/lab/health' })
  assert.equal(r.status, 200)
  assert.deepEqual(r.json(), { server: 'switchgen-lab' })
})
it('a bad Host is refused with 403', async () => {
  const r = await call(handler, { url: '/api/lab/health', headers: { host: 'evil.example.com' } })
  assert.equal(r.status, 403)
})
it('a tailnet name and an address are allowed', async () => {
  assert.equal((await call(handler, { url: '/api/lab/health', headers: { host: 'freya.tail8bf383.ts.net:8443' } })).status, 200)
  assert.equal((await call(handler, { url: '/api/lab/health', headers: { host: '100.64.0.1:5274' } })).status, 200)
})
it('a POST from another site is refused with 403', async () => {
  const r = await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/pause`, body: {}, headers: { 'sec-fetch-site': 'cross-site' } })
  assert.equal(r.status, 403)
  assert.equal(driverCalls.length, 0)
})
it('a POST with a foreign Origin is refused', async () => {
  const r = await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/pause`, body: {}, headers: { origin: 'https://evil.example.com' } })
  assert.equal(r.status, 403)
})
it('a non-JSON POST is refused with 415', async () => {
  const r = await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/events`, body: { events: [] }, headers: { 'content-type': 'text/plain' } })
  assert.equal(r.status, 415)
})
it('the page is served with a CSP and no-cache; ui files by name only', async () => {
  const r = await call(handler, { url: '/' })
  assert.equal(r.status, 200)
  assert.match(String(r.headers['content-type']), /text\/html/)
  assert.match(String(r.headers['content-security-policy']), /script-src 'self'/)
  assert.equal(r.headers['cache-control'], 'no-cache')
  assert.equal((await call(handler, { url: '/ui/judge.js' })).status, 200)
  assert.equal((await call(handler, { url: '/ui/refs.js' })).status, 200)
  assert.equal((await call(handler, { url: '/ui/judge.css' })).status, 200)
  assert.equal((await call(handler, { url: '/ui/..%2Fserver%2Fserver.ts' })).status, 404)
  assert.equal((await call(handler, { url: '/ui/index.html' })).status, 404)
})
it('a path it does not serve is passed on', async () => {
  const r = await call(handler, { url: '/somewhere' })
  assert.equal(r.passed, true)
})
it('cards are served', async () => {
  const r = await call(handler, { url: '/api/lab/cards' })
  assert.equal(r.status, 200)
  assert.ok(r.json().some((c: any) => c.id === 'pf'))
})

// ---------------------------------------------------------------- refs
const png = rectMaskPng(96, 80, { x: 10, y: 10, w: 20, h: 20 })
it('ref upload refuses a JSON body with 415', async () => {
  const r = await call(handler, { method: 'POST', url: '/api/lab/refs', body: { a: 1 }, headers: { 'x-lab-name': 'scene' } })
  assert.equal(r.status, 415)
})
it('ref upload refuses image types it does not take', async () => {
  const r = await call(handler, { method: 'POST', url: '/api/lab/refs', raw: png, headers: { 'content-type': 'image/gif', 'x-lab-name': 'scene' } })
  assert.equal(r.status, 415)
})
it('ref upload refuses bytes that are not a picture', async () => {
  const r = await call(handler, { method: 'POST', url: '/api/lab/refs', raw: Buffer.from('not a picture at all'), headers: { 'content-type': 'image/png', 'x-lab-name': 'scene' } })
  assert.equal(r.status, 400)
})
it('ref upload takes a PNG, names it, measures it', async () => {
  const r = await call(handler, { method: 'POST', url: '/api/lab/refs', raw: png, headers: { 'content-type': 'image/png', 'x-lab-name': 'Scene' } })
  assert.equal(r.status, 200, r.body)
  const info = r.json()
  assert.equal(info.id, 'scene')
  assert.equal(info.width, 96)
  assert.equal(info.height, 80)
})
it('ref upload refuses a cross-site POST', async () => {
  const r = await call(handler, { method: 'POST', url: '/api/lab/refs', raw: png, headers: { 'content-type': 'image/png', 'x-lab-name': 'scene', 'sec-fetch-site': 'cross-site' } })
  assert.equal(r.status, 403)
})
it('refs list, image, describe and mask', async () => {
  const list = (await call(handler, { url: '/api/lab/refs' })).json()
  assert.equal(list.length, 1)
  const img = await call(handler, { url: `/api/lab/refs/scene/image?v=${list[0].sha12}` })
  assert.equal(img.status, 200)
  assert.equal(img.headers['content-type'], 'image/png')
  assert.equal(img.headers['content-disposition'], undefined)
  const d = await call(handler, { method: 'POST', url: '/api/lab/refs/scene/describe', body: { describe: 'a wooden kitchen table by a window' } })
  assert.equal(d.status, 200, d.body)
  assert.equal(d.json().describe, 'a wooden kitchen table by a window')
  const m = await call(handler, { method: 'POST', url: '/api/lab/refs/scene/mask', body: { rect: { x: 10, y: 12, w: 40, h: 30 } } })
  assert.equal(m.status, 200, m.body)
  assert.equal(m.json().mask, true)
  const mp = await call(handler, { url: '/api/lab/refs/scene/mask.png' })
  assert.equal(mp.status, 200)
  assert.equal((await call(handler, { method: 'POST', url: '/api/lab/refs/scene/mask', body: { rect: { x: 'a' } } })).status, 400)
  assert.equal((await call(handler, { url: '/api/lab/refs/..%2F..%2Fsecret/image' })).status, 404)
})
it('refs needed lists cat and scene with the drop folder', async () => {
  const r = (await call(handler, { url: '/api/lab/refs/needed' })).json()
  const ids = r.refs.map((x: any) => x.id)
  assert.ok(ids.includes('cat') && ids.includes('scene'), JSON.stringify(ids))
  assert.equal(r.dropDir, path.join(env.labDir, 'refs'))
  assert.equal(r.refs.find((x: any) => x.id === 'cat').present, false)
})

// ---------------------------------------------------------------- judging
it('judging waits for the blind check, which then opens it', async () => {
  const r1 = await call(handler, { url: `/api/lab/runs/${run}/next?judge=you&session=sess-1` })
  assert.equal(r1.status, 409)
  assert.equal(r1.json().checking, true)
  for (let i = 0; i < 50 && !S.blindVerdict(env, run); i++) await new Promise((r) => setTimeout(r, 20))
  assert.equal(S.blindVerdict(env, run)?.ok, true)
  const r2 = await call(handler, { url: `/api/lab/runs/${run}/next?judge=you&session=sess-1` })
  assert.equal(r2.status, 200, r2.body)
  const n = r2.json()
  assert.equal(n.type, 'grid')
  assert.equal(n.item.itemId, 's1.A')
  assert.equal(n.lookThroughDue, true)
  assert.equal('record' in n, false)
})
it('a changed items.json closes judging until checked again', async () => {
  const saved = fs.readFileSync(path.join(dir, 'items.json'), 'utf8')
  const leaky = JSON.parse(saved)
  leaky.sets[0].brief.task = 'Made by noobai, obviously.'
  fs.writeFileSync(path.join(dir, 'items.json'), JSON.stringify(leaky))
  const r1 = await call(handler, { url: `/api/lab/runs/${run}/next?judge=you&session=sess-1` })
  assert.equal(r1.status, 409)
  for (let i = 0; i < 50 && !S.blindVerdict(env, run); i++) await new Promise((r) => setTimeout(r, 20))
  const v = S.blindVerdict(env, run)
  assert.equal(v?.ok, false)
  const r2 = await call(handler, { url: `/api/lab/runs/${run}/next?judge=you&session=sess-1` })
  assert.equal(r2.status, 409)
  assert.equal(r2.json().blindFailed, true)
  assert.doesNotMatch(r2.body, /noobai/i)
  fs.writeFileSync(path.join(dir, 'items.json'), saved)
  await call(handler, { url: `/api/lab/runs/${run}/next?judge=you&session=sess-1` })
  for (let i = 0; i < 50 && !S.blindVerdict(env, run)?.ok; i++) await new Promise((r) => setTimeout(r, 20))
  assert.equal(S.blindVerdict(env, run)?.ok, true)
})
it('the set overview hides the second-look link', async () => {
  const r = await call(handler, { url: `/api/lab/runs/${run}/sets/s1-2` })
  assert.equal(r.status, 200)
  assert.doesNotMatch(r.body, /secondOf|repeats/)
})
it('events are idempotent by id; unknown items, other runs and server kinds are refused', async () => {
  const e = ev('s1.A', 'score', { step: 4, fail: [], best: null, chips: [], recognised: false })
  const a = (await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/events`, body: { events: [e] } })).json()
  assert.deepEqual([a.accepted, a.duplicate], [1, 0])
  const b = (await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/events`, body: { events: [e] } })).json()
  assert.deepEqual([b.accepted, b.duplicate], [0, 1])
  const c = (await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/events`, body: { events: [ev('nope', 'skip'), { ...ev('s1.B', 'skip'), run: 'other' }, ev('s1', 'reveal')] } })).json()
  assert.equal(c.accepted, 0)
  assert.equal(c.rejected.length, 3)
  const next = (await call(handler, { url: `/api/lab/runs/${run}/next?judge=you&session=sess-1` })).json()
  assert.equal(next.item.itemId, 's1.B')
})
it('the image route serves tokens only, cached, never as a download', async () => {
  const r = await call(handler, { url: `/api/lab/img/${T(0)}-g.webp` })
  assert.equal(r.status, 200)
  assert.equal(r.headers['content-type'], 'image/webp')
  assert.match(String(r.headers['cache-control']), /private.*immutable/)
  assert.equal(r.headers['content-disposition'], undefined)
  assert.equal((await call(handler, { url: `/api/lab/img/${T(0)}-x.webp` })).status, 404)
  // A file in view/ whose name is not a 128-bit token is never served, even when it is there.
  fs.copyFileSync(webp, path.join(dir, 'view', 'short-g.webp'))
  assert.equal((await call(handler, { url: '/api/lab/img/short-g.webp' })).status, 404)
  fs.rmSync(path.join(dir, 'view', 'short-g.webp'))
  assert.equal((await call(handler, { url: `/api/lab/img/..%2Fsealed-g.webp` })).status, 404)
})
it('the report and findings are closed before the reveal', async () => {
  assert.equal((await call(handler, { url: '/report/first-pass' })).status, 409)
  assert.equal((await call(handler, { url: '/api/lab/studies/first-pass/findings.json' })).status, 409)
  assert.throws(() => S.readSealed(env, run), /not revealed/)
})

// ---------------------------------------------------------------- start and pause
it('start needs confirm:"start"', async () => {
  const r = await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/start`, body: {} })
  assert.equal(r.status, 400)
  assert.equal(driverCalls.length, 0)
})
it('start refuses a bad until', async () => {
  const r = await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/start`, body: { confirm: 'start', until: '25:00' } })
  assert.equal(r.status, 400)
})
it('start is refused while the calibration gate is unmet', async () => {
  const r = await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/start`, body: { confirm: 'start' } })
  assert.equal(r.status, 409, r.body)
  assert.match(r.json().error, /calibration/i)
  assert.equal(driverCalls.length, 0)
})
it('planning a night from the page sends nothing and lists it', async () => {
  const r = await call(handler, { method: 'POST', url: '/api/lab/plan', body: { suite: 'calibration' } })
  assert.equal(r.status, 200, r.body)
  assert.equal(r.json().run, 'cal-1')
  const rows = (await call(handler, { url: '/api/lab/runs' })).json()
  const cal = rows.find((x: any) => x.run === 'cal-1')
  assert.equal(cal.calibration, true)
  assert.equal(cal.state, 'planned')
  assert.equal(rows.find((x: any) => x.run === run).gate, 'unmet')
  assert.equal(driverCalls.length, 0)
})
it('a night waiting for a photo refuses to start and names it', async () => {
  const r = await call(handler, { method: 'POST', url: '/api/lab/plan', body: { suite: 'ext-edit' } })
  assert.equal(r.status, 200, r.body)
  assert.match(String(r.json().refusal), /cat/)
  const s = await call(handler, { method: 'POST', url: '/api/lab/runs/exta-1/start', body: { confirm: 'start', skipCalibration: true } })
  assert.equal(s.status, 409, s.body)
  assert.ok(s.json().needsRefs?.includes("cat"), s.body)
  assert.equal(driverCalls.length, 0)
  const row = (await call(handler, { url: '/api/lab/runs' })).json().find((x: any) => x.run === 'exta-1')
  assert.ok(row.needsRefs.includes('cat'))
})
it('the photo arrives from the phone later: Start plans the night again and goes', async () => {
  const cat = rectMaskPng(120, 90, { x: 0, y: 0, w: 60, h: 90 })
  const up = await call(handler, { method: 'POST', url: '/api/lab/refs', raw: cat, headers: { 'content-type': 'image/png', 'x-lab-name': 'cat' } })
  assert.equal(up.status, 200, up.body)
  const row = (await call(handler, { url: '/api/lab/runs' })).json().find((x: any) => x.run === 'exta-1')
  assert.deepEqual(row.needsRefs, [])
  assert.equal(row.replan, true)
  const s = await call(handler, { method: 'POST', url: '/api/lab/runs/exta-1/start', body: { confirm: 'start', skipCalibration: true } })
  assert.equal(s.status, 202, s.body)
  assert.equal(driverCalls.at(-1), 'start exta-1 {"skipCalibration":true}')
  const p = JSON.parse(fs.readFileSync(path.join(runDir(env, 'exta-1'), 'plan.json'), 'utf8'))
  assert.equal(p.blocked.length, 0)
  await call(handler, { method: 'POST', url: '/api/lab/runs/exta-1/pause', body: {} })
  assert.equal(driverCalls.at(-1), 'pause exta-1')
})
it('start with skipCalibration plans again (nothing sent yet) and starts the driver', async () => {
  const r = await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/start`, body: { confirm: 'start', skipCalibration: true, until: '07:30' } })
  assert.equal(r.status, 202, r.body)
  assert.equal(driverCalls.at(-1), `start ${run} {"until":"07:30","skipCalibration":true}`)
  const p = JSON.parse(fs.readFileSync(path.join(dir, 'plan.json'), 'utf8'))
  assert.ok(p.order.length > 400, `planned again: ${p.order.length}`)
})
it('a second start, and another run, are refused while one is live', async () => {
  assert.equal((await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/start`, body: { confirm: 'start', skipCalibration: true } })).status, 409)
  const r = await call(handler, { method: 'POST', url: '/api/lab/runs/cal-1/start', body: { confirm: 'start' } })
  assert.equal(r.status, 409)
  assert.match(r.json().error, /being made now/)
})
it('status comes from the live driver; pause reaches it', async () => {
  assert.equal((await call(handler, { url: `/api/lab/runs/${run}/status` })).json().state, 'sending')
  const r = await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/pause`, body: {} })
  assert.equal(r.status, 200)
  assert.equal(driverCalls.at(-1), `pause ${run}`)
})
it('an unknown run is 404', async () => {
  assert.equal((await call(handler, { url: '/api/lab/runs/nope-9/status' })).status, 404)
  assert.equal((await call(handler, { method: 'POST', url: '/api/lab/runs/nope-9/start', body: { confirm: 'start' } })).status, 404)
})

// ---------------------------------------------------------------- reveal
it('reveal is refused with the remaining count until everything is answered', async () => {
  const r = await call(handler, { method: 'POST', url: '/api/lab/studies/first-pass/reveal', body: {} })
  assert.equal(r.status, 409)
  const j = r.json()
  assert.ok(j.remaining > 0, r.body)
})
it('no answer so far carried a sealed string', async () => {
  for (const { body } of bodies) {
    assert.doesNotMatch(body, /sealedmarker|NoobAI-XL|flux-2-klein-4b|sdxl-illustrious/i)
  }
})
it('no judging answer so far named a contestant, a family, a file, a cell, a job or a prompt of the sealed key', () => {
  // Every string of the fixture's sealed key, matched as a whole word: the
  // model keys (noobai, klein), the family ids, the weight files, the cell
  // ids and their .lab/cells files, and the runner's job and prompt ids.
  const strings = CLI.sealedStrings(JSON.parse(fs.readFileSync(path.join(dir, 'sealed.json'), 'utf8')))
  for (const k of ['noobai', 'klein', 'flux2-klein', 'c000000000000000', `.lab/cells/c0_00001_.png`, 'job-SEALEDMARKER-0', 'prompt-0-sealedmarker']) assert.ok(strings.includes(k), k)
  const judging = bodies.filter((b) => /^\/api\/lab\/(runs\/[^/]+\/(next|sets\/[^/?]+|events)|img\/)/.test(b.url))
  assert.ok(judging.some((b) => /\/next/.test(b.url)) && judging.some((b) => /\/sets\//.test(b.url)) && judging.some((b) => /\/events/.test(b.url)) && judging.some((b) => /\/img\//.test(b.url)), 'every judging route was asked')
  for (const b of judging) assert.deepEqual(CLI.leaksIn(b.body, strings), [], b.url)
})
it('"reveal early" reveals, and later events are marked afterReveal', async () => {
  const r = await call(handler, { method: 'POST', url: '/api/lab/studies/first-pass/reveal', body: { confirm: 'reveal early' } })
  assert.equal(r.status, 200, r.body)
  assert.equal(r.json().early, true)
  const e = ev('s1.B', 'skip')
  const a = (await call(handler, { method: 'POST', url: `/api/lab/runs/${run}/events`, body: { events: [e] } })).json()
  assert.equal(a.accepted, 1)
  const lines = fs.readFileSync(path.join(dir, 'judging', 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
  assert.equal(lines.find((x) => x.id === e.id).afterReveal, true)
  assert.ok(lines.some((x) => x.kind === 'reveal'))
  assert.doesNotThrow(() => S.readSealed(env, run))
  const rows = (await call(handler, { url: '/api/lab/runs' })).json()
  assert.equal(rows.find((x: any) => x.run === run).state, 'revealed')
})

// ---------------------------------------------------------------- cli helpers
it('parseArgs', () => {
  const a = CLI.parseArgs(['core-1', '--until', '07:30', '--skip-calibration', '--run=x'])
  assert.deepEqual(a._, ['core-1'])
  assert.equal(a.flags.until, '07:30')
  assert.equal(a.flags['skip-calibration'], true)
  assert.equal(a.flags.run, 'x')
})
it('the repo: nothing under src/ or server/ imports lab/', () => {
  assert.deepEqual(CLI.labImports(REPO), [])
})
it('labImports finds a planted import', () => {
  const fake = tempDir('lab-e-repo-')
  fs.mkdirSync(path.join(fake, 'src'), { recursive: true })
  fs.mkdirSync(path.join(fake, 'lab'), { recursive: true })
  fs.writeFileSync(path.join(fake, 'src', 'a.ts'), "import { x } from '../lab/core/env.ts'\n")
  fs.writeFileSync(path.join(fake, 'src', 'b.ts'), "const y = await import('../lab/x.ts')\nimport z from './label.ts'\n")
  const found = CLI.labImports(fake).map((x: any) => x.file).sort()
  assert.deepEqual(found, ['src/a.ts', 'src/b.ts'])
})
it('leaksIn matches whole words only', () => {
  assert.deepEqual(CLI.leaksIn('waiting for the grids', ['wai']), [])
  assert.deepEqual(CLI.leaksIn('the WAI model', ['wai']), ['wai'])
  assert.deepEqual(CLI.leaksIn('animal', ['anima']), [])
  assert.deepEqual(CLI.leaksIn('x NoobAI-XL-v1.1.safetensors y', ['NoobAI-XL-v1.1.safetensors']), ['NoobAI-XL-v1.1.safetensors'])
})
it('checkBlind passes on the clean fixture and names no sealed string when it fails', async () => {
  const ok = await CLI.checkBlind(env, run)
  assert.equal(ok.ok, true, ok.problems.join('; '))
  fs.appendFileSync(path.join(dir, 'view', `${T(0)}-g.webp`), Buffer.from('EXIFnoobai'))
  const bad = await CLI.checkBlind(env, run)
  assert.equal(bad.ok, false)
  for (const p of bad.problems) assert.doesNotMatch(p, /noobai/i)
  fs.copyFileSync(webp, path.join(dir, 'view', `${T(0)}-g.webp`))
  assert.equal((await CLI.checkBlind(env, run)).ok, true)
  // A sealed string inside the picture's own bytes (a well-formed WebP).
  const inner = Buffer.concat([Buffer.from([0x2f, 0x3f, 0xc0, 0x0f, 0x00]), Buffer.from(' NoobAI-XL-v1.1.safetensors ', 'latin1')])
  const chunk = Buffer.concat([Buffer.from('VP8L', 'ascii'), Buffer.alloc(4), inner, inner.length & 1 ? Buffer.alloc(1) : Buffer.alloc(0)])
  chunk.writeUInt32LE(inner.length, 4)
  const riffed = Buffer.concat([Buffer.from('RIFF', 'ascii'), Buffer.alloc(4), Buffer.from('WEBP', 'ascii'), chunk])
  riffed.writeUInt32LE(riffed.length - 8, 4)
  fs.writeFileSync(path.join(dir, 'view', `${T(1)}-g.webp`), riffed)
  const inBytes = await CLI.checkBlind(env, run)
  assert.equal(inBytes.ok, false)
  assert.ok(inBytes.problems.some((p) => /bytes hold a sealed string/.test(p)), inBytes.problems.join('; '))
  for (const p of inBytes.problems) assert.doesNotMatch(p, /noobai/i)
  fs.copyFileSync(webp, path.join(dir, 'view', `${T(1)}-g.webp`))
  // A stray file in view/.
  fs.writeFileSync(path.join(dir, 'view', 'notes.txt'), 'hello')
  assert.ok((await CLI.checkBlind(env, run)).problems.some((p) => /not a blind copy/.test(p)))
  fs.rmSync(path.join(dir, 'view', 'notes.txt'))
  // A sealed string in items.json.
  const saved = fs.readFileSync(path.join(dir, 'items.json'), 'utf8')
  const leaky = JSON.parse(saved)
  leaky.sets[0].brief.task = 'Three red apples, as NoobAI-XL-v1.1.safetensors draws them.'
  fs.writeFileSync(path.join(dir, 'items.json'), JSON.stringify(leaky))
  const inItems = await CLI.checkBlind(env, run)
  assert.ok(inItems.problems.some((p) => /^items\.json holds/.test(p)), inItems.problems.join('; '))
  for (const p of inItems.problems) assert.doesNotMatch(p, /noobai/i)
  fs.writeFileSync(path.join(dir, 'items.json'), saved)
  assert.equal((await CLI.checkBlind(env, run)).ok, true)
})
it('smokeSuite: one cell per operation, 8 steps, checkSuite-clean', async () => {
  const { suite, left } = CLI.smokeSuite(env)
  assert.equal(suite.steps, 8)
  const ops = suite.slots.map((s: any) => s.op)
  assert.ok(ops.includes('t2i') && ops.includes('face') && ops.includes('hand') && ops.includes('hires'), ops.join())
  assert.ok(left.some((l: string) => /cat/.test(l)), left.join())
  assert.deepEqual(checkSuite(suite), [])
})


// ------------------------------------------------------- beyond E's list --

it('the next route writes the pairs it fixes, and a read-only handler (the blind check\'s) writes nothing', async () => {
  const r = 'ro-1'
  const d = runDir(env, r)
  fs.mkdirSync(path.join(d, 'judging'), { recursive: true })
  fs.writeFileSync(path.join(d, 'plan.json'), JSON.stringify({ ...plan, run: r }))
  fs.writeFileSync(path.join(d, 'items.json'), JSON.stringify({ ...items, run: r, sets: [items.sets[0]] }))
  const scored = ['s1.A', 's1.B'].map((item) => ({ ...ev(item, 'score', { step: 4, fail: [], best: null, chips: [], recognised: false }), run: r }))
  fs.writeFileSync(path.join(d, 'judging', 'events.jsonl'), scored.map((e) => JSON.stringify(e)).join('\n') + '\n')
  const before = fs.readFileSync(path.join(d, 'judging', 'events.jsonl'), 'utf8')
  const ro = S.createLabHandler({ env, log: () => {}, skipBlindGate: true, readOnly: true, makeDriver: () => { throw new Error('no driver') } })
  const a = await call(ro, { url: `/api/lab/runs/${r}/next?judge=you&session=sess-1` })
  assert.equal(a.status, 200, a.body)
  assert.equal(a.json().type, 'pair')
  assert.equal(fs.readFileSync(path.join(d, 'judging', 'events.jsonl'), 'utf8'), before)
  const rw = S.createLabHandler({ env, log: () => {}, skipBlindGate: true, makeDriver: () => { throw new Error('no driver') } })
  const b = await call(rw, { url: `/api/lab/runs/${r}/next?judge=you&session=sess-1` })
  assert.equal(b.status, 200, b.body)
  assert.equal('record' in b.json(), false)
  const lines = fs.readFileSync(path.join(d, 'judging', 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
  assert.ok(lines.some((x) => x.kind === 'pairs-made'))
})

it('planning from the page makes core-1 and extb-1 by default and sends nothing', async () => {
  // A lab of its own: first-pass is revealed in this file's, and nothing may be planned in a revealed study.
  const own = tempEnv()
  const calls: string[] = []
  const h = S.createLabHandler({ env: own, log: () => {}, makeDriver: (r: string) => ({ async start() { calls.push(`start ${r}`) }, async pause() {}, status: () => ({ run: r, state: 'idle', made: 0, total: 0, failed: 0, etaSeconds: null, until: null, message: null }) }) })
  for (const [suite, want] of [['first-pass-core', 'core-1'], ['ext-range', 'extb-1']]) {
    const r = await call(h, { method: 'POST', url: '/api/lab/plan', body: { suite } })
    assert.equal(r.status, 200, r.body)
    assert.equal(r.json().run, want)
    const p = JSON.parse(fs.readFileSync(path.join(runDir(own, want), 'plan.json'), 'utf8'))
    assert.deepEqual(p.blocked, [])
    assert.ok(Object.keys(p.prompts).length > 0)
  }
  assert.deepEqual(calls, [], 'planning starts no driver')
})

it('the smoke run keeps its own suite for sealing, and starts without the calibration gate', async () => {
  const { suite } = CLI.smokeSuite(env)
  assert.equal(suite.study, S.SMOKE_STUDY)
  const p = S.planRun(env, 'smoke-1', [suite])
  assert.ok(fs.existsSync(path.join(runDir(env, 'smoke-1'), 'suites.json')))
  assert.deepEqual(S.ownSuites(env, 'smoke-1').map((s) => s.id), [suite.id])
  assert.ok(S.suitesOfPlan(env, p))
  const row = (await call(handler, { url: '/api/lab/runs' })).json().find((x: any) => x.run === 'smoke-1')
  assert.equal(row.gate, null)
  const s = await call(handler, { method: 'POST', url: '/api/lab/runs/smoke-1/start', body: { confirm: 'start' } })
  assert.equal(s.status, 202, s.body)
  assert.match(driverCalls.at(-1)!, /^start smoke-1 .*"skipCalibration":true/)
  await call(handler, { method: 'POST', url: '/api/lab/runs/smoke-1/pause', body: {} })
})

it('pause for a run another process is sending sends that process SIGINT', async () => {
  const r = 'sig-1'
  const d = runDir(env, r)
  fs.mkdirSync(d, { recursive: true })
  fs.writeFileSync(path.join(d, 'plan.json'), JSON.stringify({ ...plan, run: r }))
  const marker = path.join(root, 'sigint.txt')
  const ready = path.join(root, 'ready.txt')
  const child = spawn(process.execPath, ['-e', `
    const fs = require('node:fs')
    process.on('SIGINT', () => { fs.writeFileSync(${JSON.stringify(marker)}, 'SIGINT'); process.exit(0) })
    fs.writeFileSync(${JSON.stringify(ready)}, 'ready')
    setInterval(() => {}, 1000)
  `], { stdio: 'ignore' })
  try {
    for (let i = 0; i < 200 && !fs.existsSync(ready); i++) await new Promise((res) => setTimeout(res, 25))
    assert.ok(fs.existsSync(ready), 'the stand-in lab process started')
    fs.writeFileSync(path.join(d, 'driver.lock'), JSON.stringify({ pid: child.pid, boot: machineBoot(), host: os.hostname(), run: r, at: Date.now() }))
    const p = await call(handler, { method: 'POST', url: `/api/lab/runs/${r}/pause`, body: {} })
    assert.equal(p.status, 200, p.body)
    assert.match(p.json().message, new RegExp(`pid ${child.pid}`))
    for (let i = 0; i < 200 && !fs.existsSync(marker); i++) await new Promise((res) => setTimeout(res, 25))
    assert.equal(fs.readFileSync(marker, 'utf8'), 'SIGINT')
    assert.ok(!driverCalls.includes(`pause ${r}`), 'the in-process driver was not asked')
  } finally {
    child.kill('SIGKILL')
    fs.rmSync(path.join(d, 'driver.lock'), { force: true })
  }
})

// ------------------------------------------ the second wave's fixes (E) --

describe('a judge\'s removal, the reveal of a whole study, and starting from the command line', () => {
  // A study of its own: only core-1 planned, one defaults set of two grids,
  // judged through a handler whose clock the test moves.
  const E = tempEnv()
  const eDir = runDir(E, run)
  fs.mkdirSync(path.join(eDir, 'view'), { recursive: true })
  fs.writeFileSync(path.join(eDir, 'plan.json'), JSON.stringify(plan))
  fs.writeFileSync(path.join(eDir, 'items.json'), JSON.stringify({
    v: 1, run, sealedSha: 'abc', pairs: [], checks: [],
    sets: [{ setId: 's1', order: 0, block: 'defaults', card: 'df@1', mode: 'scale', closeLook: false, brief: { task: 'A picture.' },
      grids: [{ itemId: 's1.A', letter: 'A', pos: 0, tiles: tiles(0) }, { itemId: 's1.B', letter: 'B', pos: 1, tiles: tiles(4) }] }],
  }))
  fs.writeFileSync(path.join(eDir, 'sealed.json'), JSON.stringify({ v: 1, run, study: 'first-pass', sealedAt: 0, suites: [], tokens, gridOf: {}, letters: {}, pairOf: {}, checkOf: {}, sets: {}, slots: {}, cells: {}, contestants: {} }))
  for (let i = 0; i < 8; i++) for (const s of ['g', 'f']) fs.copyFileSync(webp, path.join(eDir, 'view', `${T(i)}-${s}.webp`))
  const cellOf = (i: number) => `c${i}`.padEnd(16, '0')
  let clock = 1_000_000
  const eCalls: string[] = []
  const stand = (calls: string[]) => (r: string) => ({
    async start(o: unknown) { calls.push(`start ${r} ${JSON.stringify(o)}`) },
    async pause() {},
    status: () => ({ run: r, state: 'idle' as DriverState, made: 0, total: 1, failed: 0, etaSeconds: null, until: null, message: null }),
  })
  const eh = S.createLabHandler({ env: E, log: () => {}, skipBlindGate: true, now: () => clock, makeDriver: stand(eCalls) })
  const eev = (item: string, kind: string, value?: unknown) => ({ ...ev(item, kind, value), at: clock })
  const flagged = (item: string) => eev(item, 'score', { step: 1, fail: [], best: null, chips: [], recognised: false, defaults: { who: 'looks under 18', extra: ['sexualised'] } })
  const copies = (from: number) => [0, 1, 2, 3].flatMap((k) => ['g', 'f'].map((s) => fs.existsSync(path.join(eDir, 'view', `${T(from + k)}-${s}.webp`))))
  const post = (events: unknown[]) => call(eh, { method: 'POST', url: `/api/lab/runs/${run}/events`, body: { events } })

  it('an Undo of a removal keeps the blind copies, after the undo window too', async () => {
    const a = flagged('s1.A')
    assert.equal((await post([a])).json().accepted, 1)
    assert.deepEqual(copies(0), Array(8).fill(true), 'the copies stay while Undo is offered')
    clock += 1000
    assert.equal((await post([eev('s1.A', 'undo', { target: a.id })])).json().accepted, 1)
    clock += S.REMOVAL_UNDO_MS + 5000
    assert.equal((await call(eh, { url: `/api/lab/runs/${run}/next?judge=you&session=sess-1` })).status, 200)
    assert.deepEqual(copies(0), Array(8).fill(true), 'an undone removal keeps its copies')
  })
  it('a removal that stands loses its blind copies once the undo window has passed', async () => {
    await post([flagged('s1.B')])
    assert.deepEqual(copies(4), Array(8).fill(true), 'the copies stay while Undo is offered')
    clock += S.REMOVAL_UNDO_MS + 1
    await call(eh, { url: `/api/lab/runs/${run}/next?judge=you&session=sess-1` })
    assert.deepEqual(copies(4), Array(8).fill(false))
    assert.deepEqual(copies(0), Array(8).fill(true))
  })
  it('a plain reveal waits for the nights of the study not planned yet, and names them', async () => {
    const r = await call(eh, { method: 'POST', url: '/api/lab/studies/first-pass/reveal', body: {} })
    assert.equal(r.status, 409, r.body)
    const j = r.json()
    assert.deepEqual([...j.unplanned].sort(), ['cal-1', 'exta-1', 'extb-1'])
    assert.match(j.error, /not planned yet/)
    for (const x of ['cal-1', 'exta-1', 'extb-1']) assert.ok(j.waiting.includes(x), x)
    assert.equal(S.revealOf(E, 'first-pass'), null)
  })
  it('"reveal early" removes the pictures of the removal that stands, and only those', async () => {
    const r = await call(eh, { method: 'POST', url: '/api/lab/studies/first-pass/reveal', body: { confirm: 'reveal early' } })
    assert.equal(r.status, 200, r.body)
    const removed = readLedger(eDir).filter((e) => e.t === 'removed').map((e) => (e as { cell: string }).cell).sort()
    assert.deepEqual(removed, [4, 5, 6, 7].map(cellOf))
    assert.deepEqual(copies(0), Array(8).fill(true))
  })
  it('nothing more is planned or started in a revealed study, from the page or the command line', async () => {
    const p = await call(eh, { method: 'POST', url: '/api/lab/plan', body: { suite: 'ext-edit' } })
    assert.equal(p.status, 409, p.body)
    assert.match(p.json().error, /has been revealed/)
    assert.equal(fs.existsSync(path.join(runDir(E, 'exta-1'), 'plan.json')), false)
    const s = await call(eh, { method: 'POST', url: `/api/lab/runs/${run}/start`, body: { confirm: 'start', skipCalibration: true } })
    assert.equal(s.status, 409, s.body)
    assert.match(s.json().error, /core-1 was not started/)
    assert.deepEqual(eCalls, [])
    assert.match(String(CLI.startProblem(E, run, { skipCalibration: true }, () => {})), /has been revealed/)
  })
  it('the smoke study may be planned again after its own reveal', () => {
    fs.mkdirSync(S.studyDir(E, S.SMOKE_STUDY), { recursive: true })
    fs.writeFileSync(path.join(S.studyDir(E, S.SMOKE_STUDY), 'revealed.json'), JSON.stringify({ v: 1, study: S.SMOKE_STUDY, at: 1, early: true, remaining: 0 }))
    assert.ok(S.revealOf(E, S.SMOKE_STUDY))
    assert.doesNotThrow(() => S.planRun(E, 'smoke-2', [CLI.smokeSuite(E).suite]))
  })

  const F = tempEnv()
  const fCalls: string[] = []
  const fh = S.createLabHandler({ env: F, log: () => {}, skipBlindGate: true, makeDriver: stand(fCalls) })
  it('the command line refuses a start the driver would refuse: the calibration gate, a missing photo', () => {
    S.planRun(F, 'core-1', [suiteById('first-pass-core')!])
    const gate = CLI.startProblem(F, 'core-1', {}, () => {})
    assert.match(String(gate), /calibration/i)
    // The way to skip it is said once, in the command line's own words.
    assert.equal(String(gate).split('--skip-calibration').length - 1, 1, String(gate))
    assert.doesNotMatch(String(gate), /skip calibration"/)
    S.planRun(F, 'exta-1', [suiteById('ext-edit')!])
    assert.match(String(CLI.startProblem(F, 'exta-1', { skipCalibration: true }, () => {})), /cat/)
    assert.equal(CLI.startProblem(F, 'core-1', { skipCalibration: true }, () => {}), null)
  })
  it('the server\'s calibration refusal names the page\'s tickbox and the command line\'s flag, once each', async () => {
    const r = await call(fh, { method: 'POST', url: '/api/lab/runs/core-1/start', body: { confirm: 'start' } })
    assert.equal(r.status, 409, r.body)
    const why = String(r.json().error)
    assert.match(why, /Start without the calibration/)
    assert.match(why, /on the lab page/)
    assert.equal(why.split('--skip-calibration').length - 1, 1, why)
    assert.doesNotMatch(why, /skip calibration"/)
    assert.deepEqual(fCalls, [])
  })
  it('the server takes --until 7:30 as the command line does, and still refuses 24:00', async () => {
    const u = await call(fh, { method: 'POST', url: '/api/lab/runs/core-1/start', body: { confirm: 'start', skipCalibration: true, until: '7:30' } })
    assert.equal(u.status, 202, u.body)
    assert.equal(fCalls.at(-1), 'start core-1 {"until":"7:30","skipCalibration":true}')
    const bad = await call(fh, { method: 'POST', url: '/api/lab/runs/core-1/start', body: { confirm: 'start', skipCalibration: true, until: '24:00' } })
    assert.equal(bad.status, 400)
  })
  it('the photo is shown on the page without the camera\'s metadata, still upright', async () => {
    const jpg = jpegHeader({ width: 120, height: 80, orientation: 6, model: 'SECRETCAMERAMODEL', comment: 'SECRETCOMMENT' })
    assert.ok(jpg.toString('latin1').includes('SECRETCAMERAMODEL'))
    const up = await call(fh, { method: 'POST', url: '/api/lab/refs', raw: jpg, headers: { 'content-type': 'image/jpeg', 'x-lab-name': 'cat' } })
    assert.equal(up.status, 200, up.body)
    const img = await call(fh, { url: '/api/lab/refs/cat/image' })
    assert.equal(img.status, 200)
    assert.equal(img.headers['content-type'], 'image/jpeg')
    const bytes = Buffer.from(img.body, 'latin1')
    assert.ok(!img.body.includes('SECRETCAMERAMODEL') && !img.body.includes('SECRETCOMMENT'), 'the camera\'s words are left out')
    const size = sizeOf(bytes)
    assert.equal(size.orientation, 6)
    assert.deepEqual([size.width, size.height], [80, 120])
    assert.equal(Number(img.headers['content-length']), bytes.length)
  })

  it('prune forgets the pictures it deleted, so a later plan makes them again; a removal stays', () => {
    const G = tempEnv()
    const core = suiteById('first-pass-core')!
    const a = S.planRun(G, 'core-1', [core]).order[0]
    const row = (id: string) => ({ cellId: id, rel: `.lab/cells/${id}_00001_.png`, durationMs: 1, cold: false, cached: false, finishedAt: 1, run: 'x' })
    const [b, c] = ['bbbbbbbbbbbbbbbb', 'cccccccccccccccc']
    appendDoneCell(G, row(a))
    appendDoneCell(G, row(b))
    appendDoneCell(G, { cellId: b, removed: true } as never)
    appendDoneCell(G, row(c))
    // The picture reader's results: a, b (removed by the content rule) and c.
    const reading = (id: string, quarantined: boolean) => ({ v: 1, cellId: id, rel: row(id).rel, at: 1, rating: quarantined ? 'explicit' : 'general', ratings: [], general: [], character: [], quarantined })
    fs.writeFileSync(readingsPath(G), [reading(a, false), reading(b, true), reading(c, false)].map((r) => JSON.stringify(r)).join('\n') + '\n')
    assert.ok(S.planRun(G, 'core-2', [core]).reused.includes(a), 'reused while its row is there')
    assert.equal(CLI.forgetCells(G, new Set([a, b])), 1)
    const doneRows = readDoneCells(G)
    assert.equal(doneRows.has(a), false)
    assert.equal(doneRows.get(b)?.removed, true)
    assert.ok(doneRows.has(c))
    // A picture made again is read again: its old reading is forgotten with it.
    const readings = readReadings(G)
    assert.equal(readings.has(a), false)
    assert.equal(readings.get(b)?.quarantined, true)
    assert.equal(readings.get(c)?.rating, 'general')
    assert.equal(CLI.forgetCells(G, new Set([a])), 0)
    const again = S.planRun(G, 'core-3', [core])
    assert.ok(again.order.includes(a) && !again.reused.includes(a), 'made again, not reused')
  })
})

/**
 * A study of sealed runs, not a shipped one: each run one photo set of two
 * grids (m1's cell and m2's), scored 4 and 2.
 */
function sealedStudy(R: LabEnv, study: string, runs: readonly (readonly [string, string, readonly [string, string]])[]): void {
  const cell = (id: string, model: string) => ({ cellId: id, contestant: model, model, chain: null, chainStep: null, file: `${model}.safetensors`, familyId: 'f', slot: 'photo.kitchen', set: 'photo.kitchen', block: 'photo', op: 't2i', seed: 1, steps: 28, sampler: 'euler', scheduler: 'simple', status: 'made', why: null, rel: 'x', durationMs: 10_000, cold: false, cached: false, jobId: null, promptId: null, innocent: true, madeIn: 'r' })
  for (const [r, p, c] of runs) {
    const d = runDir(R, r)
    fs.mkdirSync(path.join(d, 'judging'), { recursive: true })
    fs.writeFileSync(path.join(d, 'plan.json'), JSON.stringify({ ...plan, run: r, study }))
    fs.writeFileSync(path.join(d, 'items.json'), JSON.stringify({ v: 1, run: r, sealedSha: `abc${r}`, pairs: [], checks: [],
      sets: [{ setId: p, order: 0, block: 'photo', card: 'pr@1', mode: 'scale', closeLook: true, brief: { task: 't' }, grids: ['a', 'b'].map((x, pos) => ({ itemId: p + x, letter: x.toUpperCase(), pos, tiles: tiles(pos * 4) })) }] }))
    fs.writeFileSync(path.join(d, 'sealed.json'), JSON.stringify({ v: 1, run: r, study, sealedAt: 0, suites: [], tokens: {}, letters: {}, pairOf: {}, checkOf: {},
      gridOf: { [p + 'a']: { contestant: 'm1', setId: p, cells: [] }, [p + 'b']: { contestant: 'm2', setId: p, cells: [] } },
      sets: { [p]: { block: 'photo', card: 'pr@1', mode: 'scale', slots: ['photo.kitchen'], context: false } },
      slots: { 'photo.kitchen': { block: 'photo', innocent: true, measuredOnly: false, suite: 's' } },
      cells: { [c[0]]: cell(c[0], 'm1'), [c[1]]: cell(c[1], 'm2') }, contestants: { m1: { id: 'm1', kind: 'model' }, m2: { id: 'm2', kind: 'model' } }, na: [], notMade: [] }))
    const score = (item: string, step: number) => ({ ...ev(item, 'score', { step, fail: [], best: null, chips: [], recognised: false }), run: r, session: `s-${r}` })
    fs.writeFileSync(path.join(d, 'judging', 'events.jsonl'), [score(p + 'a', 4), score(p + 'b', 2)].map((x) => JSON.stringify(x)).join('\n') + '\n')
  }
}

it('the reveal writes findings from every run of the study, not only the first one\'s', () => {
  const R = tempEnv()
  sealedStudy(R, 'two-runs', [['sa-1', 'P', ['c1', 'c2']], ['sb-1', 'Q', ['c3', 'c4']]])
  S.revealStudy(R, 'two-runs', false, Date.now())
  const f = JSON.parse(fs.readFileSync(path.join(S.studyDir(R, 'two-runs'), 'findings.json'), 'utf8'))
  assert.equal(f.afterReveal, 0)
  const m1 = f.cells.find((x: any) => x.contestant === 'm1' && x.block === 'photo')
  assert.equal(m1.n, 2, 'a score from each run')
})

it('a report made again after prune keeps the picture reader\'s ratings of the pictures that were judged', () => {
  const R = tempEnv()
  sealedStudy(R, 'kept', [['ka-1', 'P', ['c1', 'c2']]])
  const reading = (id: string, rating: string) => ({ v: 1, cellId: id, rel: `.lab/cells/${id}_00001_.png`, at: 1, rating, ratings: [], general: [], character: [], quarantined: false })
  fs.writeFileSync(readingsPath(R), [reading('c1', 'questionable'), reading('c2', 'general')].map((r) => JSON.stringify(r)).join('\n') + '\n')
  S.revealStudy(R, 'kept', false, Date.now())
  const content = () => JSON.parse(fs.readFileSync(path.join(S.studyDir(R, 'kept'), 'findings.json'), 'utf8')).content.byModel
  assert.equal(content().m1.raw.questionable, 1)
  assert.ok(fs.existsSync(path.join(S.studyDir(R, 'kept'), 'readings.json')), 'the readings are kept with the study')
  // Prune forgets c1's reading, so the picture made again is read again.
  CLI.forgetCells(R, new Set(['c1']))
  assert.equal(readReadings(R).has('c1'), false)
  S.writeReport(R, 'kept')
  assert.equal(content().m1.raw.questionable, 1)
  assert.equal(content().m1.raw.unread, 0)
})

it('the reveal waits while a night of the study is being made, and goes once it is paused', async () => {
  const H = tempEnv()
  let state: DriverState = 'idle'
  const hh = S.createLabHandler({
    env: H, log: () => {}, skipBlindGate: true,
    makeDriver: (r: string) => ({
      async start() { state = 'sending' },
      async pause() { state = 'paused' },
      status: () => ({ run: r, state, made: 0, total: 1, failed: 0, etaSeconds: null, until: null, message: null }),
    }),
  })
  S.planRun(H, 'core-1', [suiteById('first-pass-core')!])
  const s = await call(hh, { method: 'POST', url: '/api/lab/runs/core-1/start', body: { confirm: 'start', skipCalibration: true } })
  assert.equal(s.status, 202, s.body)
  const early = { method: 'POST', url: '/api/lab/studies/first-pass/reveal', body: { confirm: 'reveal early' } }
  const r = await call(hh, early)
  assert.equal(r.status, 409, r.body)
  assert.match(r.json().error, /core-1 run is being made now/)
  assert.match(r.json().error, /Pause it first/)
  assert.equal(S.revealOf(H, 'first-pass'), null)
  assert.equal((await call(hh, { method: 'POST', url: '/api/lab/runs/core-1/pause', body: {} })).status, 200)
  const again = await call(hh, early)
  assert.equal(again.status, 200, again.body)
  assert.ok(S.revealOf(H, 'first-pass'))
})

it('a description that brings in a person is refused when it is saved, in the rule\'s own words', async () => {
  const D = tempEnv()
  const dh = S.createLabHandler({ env: D, log: () => {}, skipBlindGate: true, makeDriver: () => { throw new Error('no driver') } })
  addRef(D, jpegHeader({ width: 120, height: 80 }), 'cat')
  const describe = (words: unknown) => call(dh, { method: 'POST', url: '/api/lab/refs/cat/describe', body: { describe: words } })
  const described = () => listRefs(D).find((x) => x.id === 'cat')?.describe
  const ok = await describe('a grey tabby cat with white paws')
  assert.equal(ok.status, 200, ok.body)
  const owner = await describe('a cat held by its owner')
  assert.equal(owner.status, 400, owner.body)
  assert.equal(owner.json().error, descriptionProblem('cat', 'a cat held by its owner'))
  assert.equal(described(), 'a grey tabby cat with white paws')
  const her = await describe('a tabby cat licking her paw')
  assert.equal(her.status, 400, her.body)
  assert.match(her.json().error, /Call the cat 'it' and describe only the cat\./)
  assert.equal(described(), 'a grey tabby cat with white paws')
  const back = await describe(null)
  assert.equal(back.status, 200, back.body)
  assert.equal(described(), null)
})
