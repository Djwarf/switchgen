/**
 * A stand-in for the app and its server-side runner, for the lab's tests.
 *
 * An in-process node:http server on 127.0.0.1, port 0, answering the routes
 * lab/run/runnerClient.ts knows (contracts B) the way the runner would once
 * it has the 'lab' desk (contracts A):
 * - GET /api/capabilities, with a configurable runnerDesks;
 * - GET /api/runner, the snapshot;
 * - POST /api/runner/groups, checked by the same rules as routes.mjs
 *   validateGroup, for desk 'lab' kind 'set'; an identical body again is
 *   answered replayed:true, a different one with the same id 409;
 * - POST /api/runner/groups/:id/stop;
 * - POST /api/vision/tag, with scripted rows or 503 busy:'memory'.
 * Any other path, /api/runner/lane among them, is recorded and answered 404,
 * so a test can prove the lab never asked for it.
 *
 * `step()` moves one job: it finishes the running one, or starts the next
 * waiting one. A chained job waits until its upstream has ended (the 'set'
 * rule of contracts A); an upstream that did not finish done fails it with
 * no-frame, as openChain does. A finished job writes a small real PNG under
 * the temp outputs folder at .lab/cells/<cellId>_00001_.png, with a tEXt
 * 'prompt' chunk naming the ckpt, as ComfyUI's SaveImage would (the leak
 * check looks for it).
 *
 * Scripted faults: a job's fate per attempt (failed, lost, refused, no-file,
 * cached), a dropped answer after the work was kept, 503 for the next POSTs or
 * while the runner is off, pruning of ended jobs, and a held lane.
 *
 * Nothing here talks to ComfyUI, the running app or the network.
 */
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import http from 'node:http'
import type { AddressInfo } from 'node:net'
import path from 'node:path'
import { CHAIN_TOKEN } from '../core/cells.ts'
import { png } from './helpers.ts'

export const TERMINAL = new Set(['done', 'failed', 'stopped', 'lost', 'unsent', 'skipped'])
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const KIND_OF_DESK: Record<string, string> = { images: 'batch', video: 'clips', reel: 'pass', lab: 'set' }
const isObject = (v: unknown): v is Record<string, any> => v !== null && typeof v === 'object' && !Array.isArray(v)
const bytes = (v: unknown) => Buffer.byteLength(JSON.stringify(v) ?? '')
const isString = (v: unknown, max: number) => typeof v === 'string' && v.length <= max

/** Every place a graph holds CHAIN_TOKEN as an input's whole value. */
function tokenSites(graph: Record<string, any>): [string, string][] {
  const out: [string, string][] = []
  for (const [id, n] of Object.entries(graph)) for (const [k, v] of Object.entries(n?.inputs ?? {})) if (v === CHAIN_TOKEN) out.push([id, k])
  return out
}

/** The rules of server/runner/routes.mjs validateGroup, with contracts A's 'lab' desk of kind 'set'. */
export function validateLabGroup(body: any, desks: readonly string[]): string | null {
  if (!isObject(body) || body.v !== 1) return 'body must be {v: 1, group, jobs}'
  const g = body.group
  if (!isObject(g)) return 'group must be an object'
  if (typeof g.id !== 'string' || !UUID.test(g.id)) return 'group.id must be a lowercase v4 UUID'
  if (!Object.hasOwn(KIND_OF_DESK, g.desk)) return 'group.desk must be images, video, reel or lab'
  if (!desks.includes(g.desk)) return `the ${g.desk} desk does not send its work through the queue on this server`
  if (g.kind !== KIND_OF_DESK[g.desk]) return `a ${g.desk} group must be of kind ${KIND_OF_DESK[g.desk]}`
  if (!isString(g.label, 200)) return 'group.label must be a string of at most 200 characters'
  if (!isString(g.device, 64)) return 'group.device must be a string of at most 64 characters'
  if (!Array.isArray(body.jobs) || body.jobs.length < 1 || body.jobs.length > 200) return 'jobs must be a list of 1 to 200'
  const seen = new Set<string>()
  for (let i = 0; i < body.jobs.length; i++) {
    const j = body.jobs[i]
    const at = `jobs[${i}]`
    if (!isObject(j)) return `${at} must be an object`
    if (typeof j.id !== 'string' || !UUID.test(j.id)) return `${at}.id must be a lowercase v4 UUID`
    if (seen.has(j.id) || j.id === g.id) return `${at}.id is used twice`
    if (!isString(j.label, 200)) return `${at}.label must be a string of at most 200 characters`
    if (!isString(j.prompt, 4000)) return `${at}.prompt must be a string of at most 4000 characters`
    if (j.kind !== 'image' && j.kind !== 'video') return `${at}.kind must be image or video`
    if (j.primary !== 'image' && j.primary !== 'video') return `${at}.primary must be image or video`
    if (typeof j.orFirst !== 'boolean') return `${at}.orFirst must be true or false`
    if (j.noFile !== 'fail' && j.noFile !== 'done') return `${at}.noFile must be fail or done`
    if (typeof j.heavy !== 'boolean') return `${at}.heavy must be true or false`
    if (!isObject(j.graph) || !Object.keys(j.graph).length) return `${at}.graph must be a graph of nodes`
    for (const [node, n] of Object.entries(j.graph)) {
      if (!isObject(n) || typeof n.class_type !== 'string' || (n.inputs !== undefined && !isObject(n.inputs))) return `${at}.graph node ${node} must have a class_type and inputs`
    }
    if (bytes(j.graph) > 1048576) return `${at}.graph is over 1 MiB`
    if (!isObject(j.record)) return `${at}.record must be an object`
    if (typeof j.record.desk !== 'string' || typeof j.record.mode !== 'string') return `${at}.record must name its desk and mode`
    if (bytes(j.record) > 64 * 1024) return `${at}.record is over 64 KB`
    if (j.meta !== undefined && j.meta !== null) {
      if (!isObject(j.meta)) return `${at}.meta must be an object`
      if (bytes(j.meta) > 16 * 1024) return `${at}.meta is over 16 KB`
    }
    const count = JSON.stringify(j.graph).split(CHAIN_TOKEN).length - 1
    if (j.chain !== undefined && j.chain !== null) {
      const c = j.chain
      if (!isObject(c) || typeof c.after !== 'string' || !UUID.test(c.after)) return `${at}.chain.after must be a job id`
      if (!seen.has(c.after)) return `${at}.chain.after must name an earlier job of this group`
      if (!Array.isArray(c.at) || !c.at.length) return `${at}.chain.at must list at least one place`
      const sites: string[] = []
      for (const s of c.at) {
        if (!Array.isArray(s) || s.length !== 2 || typeof s[0] !== 'string' || typeof s[1] !== 'string') return `${at}.chain.at must hold [node, input] pairs`
        if (j.graph[s[0]]?.inputs?.[s[1]] !== CHAIN_TOKEN) return `${at}.chain.at names ${s[0]}.${s[1]}, which does not hold the frame's place`
        if (sites.includes(`${s[0]}.${s[1]}`)) return `${at}.chain.at names ${s[0]}.${s[1]} twice`
        sites.push(`${s[0]}.${s[1]}`)
      }
      if (tokenSites(j.graph).length !== sites.length || count !== sites.length) return `${at}.graph holds the frame's place somewhere chain.at does not name`
    } else if (count > 0) return `${at}.graph holds the frame's place but the job has no chain`
    seen.add(j.id)
  }
  return null
}

/** How a job ends, per attempt, keyed by its cell. */
export type Fate = { status: 'done' | 'failed' | 'lost'; code?: string; cached?: boolean; durationMs?: number }

export type Call = { method: string; url: string; body: any; headers: http.IncomingHttpHeaders }

export class StandInRunner {
  server!: http.Server
  url = ''
  desks: string[] = ['images', 'video', 'reel', 'lab']
  runner = true
  available = true
  calls: Call[] = []
  groups = new Map<string, any>()
  jobs = new Map<string, any>()
  specs = new Map<string, string>()
  seq = 0
  rev = 0
  clock = 1000
  lane: { held: unknown } = { held: null }
  /** Per cell id: fates in order of attempt; a cell with none left finishes done. */
  fates = new Map<string, Fate[]>()
  /** Answer the next N group POSTs with 503 busy:'runner'. */
  offPosts = 0
  /** Keep the next N group POSTs' work but drop the connection before answering. */
  dropNext = 0
  /** Answer the next N tag calls with 503 busy:'memory'. */
  tagBusy = 0
  tagRows: (rel: string) => Record<string, unknown> = () => ({ rating: 'general', ratings: [], general: [], character: [] })
  /** Jobs of other desks shown in the snapshot (the user's own work). */
  foreign: any[] = []

  constructor(public outputs: string) {}

  async listen(): Promise<this> {
    this.server = http.createServer((req, res) => this.handle(req, res))
    await new Promise<void>((r) => this.server.listen(0, '127.0.0.1', r))
    this.url = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`
    return this
  }

  close(): Promise<void> {
    this.server.closeAllConnections?.()
    return new Promise((r) => this.server.close(() => r()))
  }

  /** The job as GET /api/runner shows it: no graph, chain or spec hash. */
  view(j: any): any {
    const { specSha: _s, graph: _g, chain: _c, stopRequested: _r, ...v } = j
    return v
  }

  snapshot(): any {
    return {
      v: 1,
      available: this.available,
      reason: this.available ? null : 'The queue is off.',
      boot: 'boot-1',
      rev: this.rev,
      comfy: { up: true },
      lane: this.lane,
      groups: [...this.groups.values()],
      jobs: [...this.foreign, ...[...this.jobs.values()].map((j) => this.view(j))].sort((a, b) => a.seq - b.seq),
      progress: {},
    }
  }

  private send(res: http.ServerResponse, code: number, body: unknown): void {
    res.statusCode = code
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify(body))
  }

  private handle(req: http.IncomingMessage, res: http.ServerResponse): void {
    let raw = ''
    req.on('data', (c) => (raw += c))
    req.on('end', () => {
      let body: any = null
      try {
        body = raw ? JSON.parse(raw) : null
      } catch {
        body = raw
      }
      const url = req.url ?? ''
      this.calls.push({ method: req.method ?? '', url, body, headers: req.headers })
      // The app's guard: JSON only, and no page of another site.
      if (req.method === 'POST' && req.headers['content-type'] !== 'application/json') return this.send(res, 415, { error: 'json only' })
      if (req.headers.origin) return this.send(res, 403, { error: 'origin' })
      if (url === '/api/capabilities' && req.method === 'GET') {
        return this.send(res, 200, { runner: this.runner, runnerDesks: this.runner ? this.desks : [], runnerReason: this.runner ? null : 'The queue is off on this server.' })
      }
      if (url === '/api/runner' && req.method === 'GET') return this.send(res, 200, this.snapshot())
      if (url === '/api/runner/groups' && req.method === 'POST') return this.submit(body, res)
      const stop = /^\/api\/runner\/groups\/([^/]+)\/stop$/.exec(url)
      if (stop && req.method === 'POST') return this.stop(decodeURIComponent(stop[1]), res)
      if (url === '/api/vision/tag' && req.method === 'POST') {
        if (this.tagBusy > 0) {
          this.tagBusy--
          return this.send(res, 503, { error: 'The picture reader is short of memory.', busy: 'memory' })
        }
        const rows = body.images.map((im: any, index: number) => ({ index, kind: 'output', rel: im.rel, width: 1, height: 1, ...this.tagRows(im.rel) }))
        return this.send(res, 200, { tag: { rows } })
      }
      return this.send(res, 404, { error: 'no such endpoint' })
    })
  }

  private submit(b: any, res: http.ServerResponse): void {
    if (!this.available || this.offPosts > 0) {
      if (this.offPosts > 0) this.offPosts--
      return this.send(res, 503, { error: 'The queue is off.', busy: 'runner', reason: 'The queue is off.' })
    }
    const problem = validateLabGroup(b, this.runner ? this.desks : [])
    if (problem) return this.send(res, 400, { error: problem })
    const g = b.group
    if (g.desk !== 'lab') return this.send(res, 400, { error: 'this stand-in takes lab groups only' })
    const sha = createHash('sha256').update(JSON.stringify(b)).digest('hex')
    if (this.groups.has(g.id)) {
      if (this.specs.get(g.id) !== sha) return this.send(res, 409, { error: 'A different group with this id is already here.', conflict: 'id' })
      const held = this.groups.get(g.id)
      return this.send(res, 200, { rev: this.rev, replayed: true, group: held, jobs: held.jobIds.map((id: string) => this.jobs.get(id)).filter(Boolean).map((j: any) => this.view(j)) })
    }
    if (b.jobs.some((j: any) => this.jobs.has(j.id))) return this.send(res, 409, { error: 'A job with one of these ids is already here.', conflict: 'id' })
    this.specs.set(g.id, sha)
    this.rev++
    this.groups.set(g.id, { id: g.id, desk: g.desk, kind: g.kind, label: g.label, device: g.device, state: 'active', jobIds: b.jobs.map((j: any) => j.id), createdAt: this.clock })
    for (const j of b.jobs) {
      this.jobs.set(j.id, {
        id: j.id, groupId: g.id, desk: 'lab', status: 'waiting', seq: ++this.seq, label: j.label, prompt: j.prompt, error: null, files: [], primary: null,
        promptId: null, durationMs: 0, ranAt: null, finishedAt: null, endedAt: null, meta: j.meta, wait: null, graph: structuredClone(j.graph),
        chain: j.chain ?? null, record: j.record, heavy: j.heavy,
      })
    }
    if (this.dropNext > 0) {
      this.dropNext--
      res.socket?.destroy()
      return
    }
    const held = this.groups.get(g.id)
    return this.send(res, 200, { rev: this.rev, replayed: false, group: held, jobs: held.jobIds.map((id: string) => this.view(this.jobs.get(id))) })
  }

  private stop(id: string, res: http.ServerResponse): void {
    const g = this.groups.get(id)
    if (!g) return this.send(res, 404, { error: 'no such group' })
    for (const jid of g.jobIds) {
      const j = this.jobs.get(jid)
      if (!j) continue
      if (j.status === 'waiting') Object.assign(j, { status: 'stopped', endedAt: this.clock, error: { code: 'stopped', message: 'Stopped before it was sent.', node: null, nodeType: null } })
      else if (!TERMINAL.has(j.status)) j.stopRequested = true
    }
    g.state = 'ended'
    this.rev++
    return this.send(res, 200, { group: g })
  }

  /** Stop a group as the user would from the app (not asked for by the lab). */
  stopFromApp(): void {
    for (const g of this.groups.values()) {
      if (g.state !== 'active') continue
      for (const id of g.jobIds) {
        const j = this.jobs.get(id)
        if (j.status === 'waiting') Object.assign(j, { status: 'stopped', endedAt: this.clock, error: { code: 'stopped', message: 'Stopped on the app.', node: null, nodeType: null } })
        else if (j.status === 'running') j.stopRequested = true
      }
      g.state = 'ended'
    }
  }

  private endGroups(): void {
    for (const g of this.groups.values()) if (g.state === 'active' && g.jobIds.every((id: string) => TERMINAL.has(this.jobs.get(id)?.status))) g.state = 'ended'
  }

  /** Mark every waiting job held (or not) the way the snapshot shows a held lane. */
  holdLane(held: boolean): void {
    this.lane = held ? { held: { why: 'restart', scope: 'all', jobId: null, since: 1 } } : { held: null }
    for (const j of this.jobs.values()) j.wait = held && j.status === 'waiting' ? { for: 'held' } : null
  }

  /** One pass: finish the running job, or start the next eligible one. Nothing runs while the lane is held. */
  step(): void {
    this.clock += 1000
    this.rev++
    const running = [...this.jobs.values()].find((j) => j.status === 'running')
    if (running) return this.finish(running)
    if (this.lane.held) return
    const next = [...this.jobs.values()]
      .filter((j) => j.status === 'waiting' && this.groups.get(j.groupId)?.state === 'active')
      .sort((a, b) => a.seq - b.seq)
      // Contracts A, kind 'set': a chained job waits until its upstream has ended.
      .find((j) => !j.chain || TERMINAL.has(this.jobs.get(j.chain.after)?.status))
    if (!next) return
    if (next.chain) {
      const up = this.jobs.get(next.chain.after)
      if (up.status !== 'done') {
        Object.assign(next, { status: 'failed', endedAt: this.clock, error: { code: 'no-frame', message: 'The picture this one starts from was not made.', node: null, nodeType: null } })
        this.endGroups()
        return
      }
      for (const [n, i] of next.chain.at) next.graph[n].inputs[i] = `${up.primary.subfolder}/${up.primary.filename} [output]`
    }
    Object.assign(next, { status: 'running', ranAt: this.clock, promptId: `p-${next.id.slice(0, 8)}` })
  }

  private finish(j: any): void {
    if (j.stopRequested) {
      Object.assign(j, { status: 'stopped', finishedAt: this.clock, endedAt: this.clock, error: { code: 'stopped', message: 'Stopped.', node: null, nodeType: null } })
      this.endGroups()
      return
    }
    const cell = j.meta.lab.cell
    const fates = this.fates.get(cell) ?? []
    const fate = fates.shift() ?? { status: 'done' }
    this.fates.set(cell, fates)
    if (fate.status === 'done') {
      const save = Object.values(j.graph).find((n: any) => n.class_type === 'SaveImage') as any
      const prefix: string = save.inputs.filename_prefix
      const sub = path.dirname(prefix)
      const name = `${path.basename(prefix)}_00001_.png`
      fs.mkdirSync(path.join(this.outputs, sub), { recursive: true })
      const ck = Object.values(j.graph).find((n: any) => n.inputs?.ckpt_name || n.inputs?.unet_name) as any
      fs.writeFileSync(path.join(this.outputs, sub, name), png(8, 8, [90, 120, 150], { text: JSON.stringify({ ckpt_name: ck?.inputs?.ckpt_name ?? ck?.inputs?.unet_name ?? 'x' }) }))
      const f = { filename: name, subfolder: sub, type: 'output', kind: 'image', ...(fate.cached ? { cached: true } : {}) }
      Object.assign(j, { status: 'done', files: [f], primary: f, durationMs: fate.durationMs ?? 900, finishedAt: this.clock, endedAt: this.clock })
    } else {
      const code = fate.code ?? fate.status
      Object.assign(j, { status: fate.status, finishedAt: this.clock, endedAt: this.clock, error: { code, message: `${code} happened`, node: '4', nodeType: 'KSampler' } })
    }
    this.endGroups()
  }

  /** Run everything submitted so far to its end. */
  drain(max = 10000): void {
    for (let i = 0; i < max; i++) {
      this.step()
      if ([...this.jobs.values()].every((j) => TERMINAL.has(j.status))) return
    }
  }

  /** The runner forgets ended jobs (48 h or 500 kept), and groups left empty. */
  prune(): void {
    for (const [id, j] of this.jobs) {
      if (!TERMINAL.has(j.status)) continue
      this.jobs.delete(id)
      for (const g of this.groups.values()) g.jobIds = g.jobIds.filter((x: string) => x !== id)
    }
    for (const [id, g] of this.groups) if (!g.jobIds.length) this.groups.delete(id)
  }

  /** The group POSTs the lab made, in order. */
  posts(): Call[] {
    return this.calls.filter((c) => c.method === 'POST' && c.url === '/api/runner/groups')
  }
}

/** A stand-in runner listening on 127.0.0.1, port 0, writing pictures under `outputs`. */
export function standInRunner(outputs: string): Promise<StandInRunner> {
  return new StandInRunner(outputs).listen()
}
