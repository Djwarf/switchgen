import { randomUUID } from 'node:crypto'
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Unanswered } from '../server/runner/comfy.mjs'
import type { Comfy, SocketMessage, SubmitResult } from '../server/runner/comfy.mjs'
import type { ArchiveApi } from '../server/archive.mjs'
import type { Runner, RunnerOptions, JobView, GroupView, Snapshot } from '../server/runner.mjs'
import { call, open, type Reply } from './http'

/**
 * A scripted ComfyUI for the queue on the server, and a way to build a runner
 * against it that a test steps by hand.
 *
 * The fake keeps a queue and a history the way ComfyUI does, and logs every
 * call the runner makes, in order: `queue`, `free`, `prompt:<id>`,
 * `getJob:<id>`, `history:<id>`, `cancel:<id>` and `interrupt:<id>`. It
 * reaches nothing: the runner under test is handed this object in place of
 * the real client, so no request leaves the process. Files a finished run
 * "wrote" are written under the test's own temporary outputs root.
 */

export const CHAIN_TOKEN = 'switchgen:chain:previous-frame'

/** A file a finished run wrote. `node` is the output node that wrote it. */
export type FakeFile = { filename: string; subfolder?: string; video?: boolean; node?: string; write?: boolean }

export type Finish = {
  files?: FakeFile[]
  /** Output nodes ComfyUI answered from its cache. */
  cachedNodes?: string[]
  /** ComfyUI's execution_start stamp; null leaves the message out. */
  start?: number | null
  /** The stamp of the ending. */
  end?: number
  /** The time the prompt was queued, which the page falls back to and the runner never uses. */
  createTime?: number
  error?: string
  interrupted?: boolean
  /** The handoff still, written by the `__cont_frame` tap. */
  frame?: FakeFile | null
}

/** What one scripted send does: its answer, and whether the prompt landed in the queue. */
export type SendScript = (graph: unknown, promptId: string) => { answer: SubmitResult; land: boolean } | Promise<{ answer: SubmitResult; land: boolean }>

type Hook = (arg: string) => void | Promise<void>

export class FakeComfy implements Comfy {
  readonly url = 'fake:comfy'
  log: string[] = []
  running: string[] = []
  pending: string[] = []
  /** ComfyUI's /history, by prompt id. */
  records = new Map<string, Record<string, unknown>>()
  /** Every prompt that landed in the queue, in order. */
  accepted: { id: string; graph: any }[] = []
  /** Scripted sends, taken one per /prompt; with none left a send lands and is accepted. */
  sends: SendScript[] = []
  /** Scripted answers to /free, one per call; 'ok' when none is left. */
  frees: ('ok' | 'unreached' | 'failed')[] = []
  /** Reads (queue, getJob, history) still to answer as a ComfyUI that is down would. */
  unanswered = 0
  /** Called before the fake answers a call of that name, with its prompt id. */
  hooks: Partial<Record<'queue' | 'free' | 'submit' | 'getJob' | 'history' | 'cancel' | 'interrupt', Hook>> = {}
  /** Sockets open now, and every client id one was opened for. */
  sockets = 0
  socketIds: string[] = []
  private listeners: ((m: SocketMessage) => void)[] = []
  private started = new Map<string, number>()

  constructor(private outputs: string | null = null) {}

  private async hook(name: keyof FakeComfy['hooks'], arg = ''): Promise<void> {
    await this.hooks[name]?.(arg)
  }

  private down(what: string): void {
    if (this.unanswered > 0) {
      this.unanswered--
      throw new Unanswered(`ComfyUI did not answer ${what}: fake outage`)
    }
  }

  async readQueue() {
    this.log.push('queue')
    await this.hook('queue')
    this.down('/queue')
    return { running: [...this.running], pending: [...this.pending] }
  }

  async free() {
    this.log.push('free')
    await this.hook('free')
    return this.frees.shift() ?? 'ok'
  }

  async submit(graph: any, promptId: string): Promise<SubmitResult> {
    this.log.push(`prompt:${promptId}`)
    await this.hook('submit', promptId)
    const script = this.sends.shift()
    const out = script ? await script(graph, promptId) : { answer: { accepted: true } as SubmitResult, land: true }
    if (out.land) this.land(promptId, graph)
    return out.answer
  }

  /** Put a prompt in the queue as if it had been sent, by the runner or by anyone. */
  land(promptId: string, graph: any = {}): void {
    this.pending.push(promptId)
    this.accepted.push({ id: promptId, graph })
  }

  async getJob(id: string) {
    this.log.push(`getJob:${id}`)
    await this.hook('getJob', id)
    this.down('/api/jobs')
    if (this.running.includes(id)) return { id, status: 'in_progress' as const, execution_start_time: this.started.get(id) ?? null }
    if (this.pending.includes(id)) return { id, status: 'pending' as const, execution_start_time: null }
    const h = this.records.get(id) as any
    if (h) {
      const s = h.status?.status_str
      const interrupted = (h.status?.messages ?? []).some((m: any[]) => m[0] === 'execution_interrupted')
      return { id, status: interrupted ? ('cancelled' as const) : s === 'success' ? ('completed' as const) : ('failed' as const), execution_start_time: null }
    }
    return null
  }

  async history(id: string) {
    this.log.push(`history:${id}`)
    await this.hook('history', id)
    this.down('/history')
    return (this.records.get(id) as Record<string, any> | undefined) ?? null
  }

  async cancel(id: string) {
    this.log.push(`cancel:${id}`)
    await this.hook('cancel', id)
    const i = this.pending.indexOf(id)
    if (i >= 0) {
      // A waiting prompt taken out of the queue leaves no record.
      this.pending.splice(i, 1)
      return true
    }
    if (this.running.includes(id)) {
      this.finish(id, { interrupted: true })
      return true
    }
    return false
  }

  async interrupt(id: string) {
    this.log.push(`interrupt:${id}`)
    await this.hook('interrupt', id)
    if (this.running.includes(id)) this.finish(id, { interrupted: true })
  }

  socket(clientId: string, on: (m: SocketMessage) => void) {
    this.sockets++
    this.socketIds.push(clientId)
    this.listeners.push(on)
    let open = true
    return {
      close: () => {
        if (!open) return
        open = false
        this.sockets--
        this.listeners = this.listeners.filter((l) => l !== on)
      },
    }
  }

  /** A message on every open socket. */
  emit(msg: SocketMessage): void {
    for (const on of [...this.listeners]) on(msg)
  }

  /** ComfyUI takes a prompt out of the queue and starts it. */
  run(id: string, startedAt: number | null = null): void {
    const i = this.pending.indexOf(id)
    if (i >= 0) this.pending.splice(i, 1)
    if (!this.running.includes(id)) this.running.push(id)
    if (startedAt !== null) this.started.set(id, startedAt)
  }

  /** A prompt ends, with a history record and any files written to disk. */
  finish(id: string, f: Finish = {}): void {
    for (const list of [this.running, this.pending]) {
      const i = list.indexOf(id)
      if (i >= 0) list.splice(i, 1)
    }
    const outputs: Record<string, any> = {}
    const put = (file: FakeFile, node: string) => {
      const ref = { filename: file.filename, subfolder: file.subfolder ?? '', type: 'output' }
      const prior = outputs[node]?.images ?? []
      outputs[node] = file.video ? { images: [...prior, ref], animated: [true] } : { images: [...prior, ref] }
      if (this.outputs && file.write !== false) {
        mkdirSync(path.join(this.outputs, file.subfolder ?? ''), { recursive: true })
        writeFileSync(path.join(this.outputs, file.subfolder ?? '', file.filename), 'x')
      }
    }
    ;(f.files ?? []).forEach((file, i) => put(file, file.node ?? String(i + 10)))
    if (f.frame) put(f.frame, '__cont_frame')
    const start = f.start === undefined ? 10_000 : f.start
    const end = f.end ?? 13_000
    const messages: unknown[][] = []
    if (start !== null) messages.push(['execution_start', { prompt_id: id, timestamp: start }])
    if (f.cachedNodes?.length) messages.push(['execution_cached', { nodes: f.cachedNodes, prompt_id: id, timestamp: start ?? end }])
    if (f.interrupted) messages.push(['execution_interrupted', { prompt_id: id, timestamp: end, node_id: '1', node_type: 'KSampler' }])
    else if (f.error) messages.push(['execution_error', { prompt_id: id, timestamp: end, exception_message: f.error, node_id: 3, node_type: 'VAEDecode' }])
    else messages.push(['execution_success', { prompt_id: id, timestamp: end }])
    const graph = this.accepted.find((a) => a.id === id)?.graph ?? {}
    this.records.set(id, {
      prompt: [1, id, graph, { client_id: 'fake', create_time: f.createTime ?? 1 }, []],
      outputs,
      status: { status_str: f.error || f.interrupted ? 'error' : 'success', completed: !f.error && !f.interrupted, messages },
    })
  }

  /** ComfyUI restarted: its queue and history are gone, and it answers nothing for `reads` reads. */
  restart(reads = 0): void {
    this.running.length = 0
    this.pending.length = 0
    this.records.clear()
    this.unanswered = reads
  }

  /** The prompts sent, as the log says, in order. */
  prompts(): string[] {
    return this.log.filter((l) => l.startsWith('prompt:')).map((l) => l.slice('prompt:'.length))
  }

  count(what: string): number {
    return this.log.filter((l) => l === what || l.startsWith(`${what}:`)).length
  }
}

// ---------------------------------------------------------------- graphs --

/** Two KSamplerAdvanced passes: a pair of models, which the server always runs as heavy. */
export const HEAVY = () => ({
  '1': { class_type: 'KSamplerAdvanced', inputs: { return_with_leftover_noise: 'enable' } },
  '2': { class_type: 'KSamplerAdvanced', inputs: { return_with_leftover_noise: 'disable' } },
  '9': { class_type: 'SaveWEBM', inputs: {} },
})
export const LIGHT = () => ({ '1': { class_type: 'KSampler', inputs: {} }, '9': { class_type: 'SaveImage', inputs: {} } })
/** A heavy shot that opens on the frame of the shot before it. */
export const CHAINED = () => ({ ...HEAVY(), '20': { class_type: 'LoadImage', inputs: { image: CHAIN_TOKEN, upload: 'image' } } })

export const recordOfDesk = (desk: 'video' | 'images', extra: Record<string, unknown> = {}) => ({
  desk,
  mode: desk === 'video' ? 't2v' : 't2i',
  prompt: 'a lighthouse at dusk',
  familyId: 'fam',
  familyLabel: 'Family',
  variant: null,
  model: 'model.safetensors',
  modelLabel: 'Model',
  negative: null,
  seed: 7,
  steps: 20,
  cfg: 4,
  sampler: 'euler',
  scheduler: 'simple',
  width: 832,
  height: 480,
  ...extra,
})

export type JobSpec = {
  id?: string
  heavy?: boolean
  graph?: Record<string, unknown>
  record?: Record<string, unknown>
  chain?: { after: string; at: [string, string][] }
  shotId?: string
  meta?: Record<string, unknown>
  noFile?: 'fail' | 'done'
  orFirst?: boolean
  label?: string
}

const KIND = { video: 'clips', images: 'batch', reel: 'pass' } as const

/** A POST /api/runner/groups body, as a desk builds one. */
export function groupBody(o: { desk: 'video' | 'images' | 'reel'; jobs: JobSpec[]; groupId?: string; device?: string; label?: string }) {
  const { desk } = o
  return {
    v: 1 as const,
    group: { id: o.groupId ?? randomUUID(), desk, kind: KIND[desk], label: o.label ?? `${desk} group`, device: o.device ?? 'device-a' },
    jobs: o.jobs.map((j, i) => ({
      id: j.id ?? randomUUID(),
      label: j.label ?? `job ${i + 1}`,
      prompt: 'a lighthouse at dusk',
      kind: desk === 'images' ? ('image' as const) : ('video' as const),
      primary: desk === 'images' ? ('image' as const) : ('video' as const),
      orFirst: j.orFirst ?? desk !== 'reel',
      noFile: j.noFile ?? (desk === 'images' ? ('fail' as const) : ('done' as const)),
      heavy: j.heavy ?? false,
      graph: j.graph ?? (j.heavy ? HEAVY() : LIGHT()),
      record: j.record ?? recordOfDesk(desk === 'images' ? 'images' : 'video'),
      ...(j.chain ? { chain: j.chain } : {}),
      ...(desk === 'reel'
        ? { meta: { shotId: j.shotId ?? randomUUID(), made: { seed: 7, openedOn: null }, ...j.meta } }
        : j.meta
          ? { meta: j.meta }
          : {}),
    })),
  }
}

// --------------------------------------------------------------- harness --

let loads = 0

type ArchiveModule = { archiveApi: ArchiveApi; switchgenArchive: () => { configurePreviewServer?: unknown } }

/** Each load of the archive a harness made, by its api, for a test that wants its routes too. */
const modules = new WeakMap<ArchiveApi, ArchiveModule>()

/** A fresh load of server/archive.mjs, on its own archive file under `outputs`. */
export async function loadArchive(outputs: string): Promise<ArchiveApi> {
  process.env.SWITCHGEN_OUTPUTS = outputs
  process.env.SWITCHGEN_ARCHIVE = path.join(outputs, '.switchgen', 'archive.json')
  // A query makes a fresh copy of the module: a new load, on the file named now.
  const spec = `../server/archive.mjs?load=${++loads}`
  const mod = (await import(/* @vite-ignore */ spec)) as ArchiveModule
  modules.set(mod.archiveApi, mod)
  return mod.archiveApi
}

/** The archive's own routes (/api/archive, /api/outputs) for a load made by loadArchive. */
export function archiveRoutes(api: ArchiveApi) {
  const mod = modules.get(api)
  if (!mod) throw new Error('not an archive loadArchive made')
  let handler: ((req: unknown, res: unknown, next: () => void) => unknown) | null = null
  const hook = mod.switchgenArchive().configurePreviewServer as (server: unknown) => void
  hook({ middlewares: { use: (fn: typeof handler) => { handler = fn } } })
  return handler!
}

/**
 * Point every root a server module reads at a temporary folder and turn the
 * queue on, returning what puts the environment back. Called before any
 * server module is imported: runner.mjs loads the archive, which reads its
 * roots as it loads.
 */
export function runnerEnv(): () => void {
  const keys = ['SWITCHGEN_RUNNER', 'SWITCHGEN_OUTPUTS', 'SWITCHGEN_ARCHIVE', 'SWITCHGEN_MODELS', 'SWITCHGEN_RUNNER_DIR', 'SWITCHGEN_RUNNER_DESKS', 'COMFY_URL']
  const before = Object.fromEntries(keys.map((k) => [k, process.env[k]]))
  const root = mkdtempSync(path.join(os.tmpdir(), 'switchgen-runner-'))
  process.env.SWITCHGEN_RUNNER = 'on'
  process.env.SWITCHGEN_OUTPUTS = path.join(root, 'outputs')
  process.env.SWITCHGEN_ARCHIVE = path.join(root, 'outputs', '.switchgen', 'archive.json')
  process.env.SWITCHGEN_MODELS = path.join(root, 'models')
  process.env.SWITCHGEN_RUNNER_DIR = path.join(root, 'runner')
  delete process.env.SWITCHGEN_RUNNER_DESKS
  // Nothing here may reach a real ComfyUI; the runners under test get the fake.
  process.env.COMFY_URL = 'http://127.0.0.1:9'
  return () => {
    for (const [k, v] of Object.entries(before)) {
      if (v === undefined) delete process.env[k]
      else process.env[k] = v
    }
  }
}

export type Harness = {
  root: string
  outputs: string
  dir: string
  archive: ArchiveApi
  comfy: FakeComfy
  clock: { t: number }
  runner: Runner
  post: (url: string, body?: unknown, headers?: Record<string, string>) => Promise<Reply>
  get: (url: string) => Promise<Reply>
  submit: (body: unknown) => Promise<Reply>
  job: (id: string) => JobView
  group: (id: string) => GroupView | undefined
  snap: () => Snapshot
  /** Passes over the queue, the fake clock moved on by `ms` before each. */
  tick: (times?: number, ms?: number) => Promise<void>
  /** The same folder, ComfyUI and archive, under a new runner: an app server restarted. */
  restart: (over?: Partial<HarnessOptions>) => Promise<Harness>
  /** The list of work as it stands on disk. */
  onDisk: () => any
}

export type HarnessOptions = {
  root?: string
  outputs?: string
  dir?: string
  archive?: ArchiveApi
  comfy?: FakeComfy
  clock?: { t: number }
  boot?: string | null
  settleMs?: number
  sleep?: (ms: number) => Promise<void>
  desks?: RunnerOptions['desks']
  enabled?: boolean
  /** How long a stop from a page waits on ComfyUI's answers to its cancels; the runner's own cap when left out. */
  stopWaitMs?: number
}

/**
 * A runner over a temporary folder, the fake ComfyUI and a load of the
 * archive of its own: no timers (autoTick off), no settle, and a clock the
 * test moves. Step it with tick().
 */
export async function harness(o: HarnessOptions = {}): Promise<Harness> {
  // Loaded first, while the environment still names runnerEnv's folder: it
  // loads the archive module too, and that load must not share a file with
  // the archive loaded for this runner below.
  const { createRunner } = await import('../server/runner.mjs')
  const root = o.root ?? mkdtempSync(path.join(os.tmpdir(), 'switchgen-runner-'))
  const outputs = o.outputs ?? path.join(root, 'outputs')
  mkdirSync(outputs, { recursive: true })
  const dir = o.dir ?? path.join(outputs, '.switchgen', 'runner')
  const archive = o.archive ?? (await loadArchive(outputs))
  const comfy = o.comfy ?? new FakeComfy(outputs)
  const clock = o.clock ?? { t: 1_000_000 }
  const runner = createRunner({
    dir,
    comfy,
    archive,
    outputs,
    autoTick: false,
    settleMs: o.settleMs ?? 0,
    sleep: o.sleep ?? (async () => {}),
    now: () => clock.t,
    machineBoot: () => (o.boot === undefined ? 'boot-1' : o.boot),
    ...(o.desks ? { desks: o.desks } : {}),
    ...(o.enabled !== undefined ? { enabled: o.enabled } : {}),
    ...(o.stopWaitMs !== undefined ? { stopWaitMs: o.stopWaitMs } : {}),
  })
  const h: Harness = {
    root,
    outputs,
    dir,
    archive,
    comfy,
    clock,
    runner,
    post: (url, body = {}, headers = {}) => call(runner.handler, { method: 'POST', url, body, headers }),
    get: (url) => call(runner.handler, { url }),
    submit: (body) => call(runner.handler, { method: 'POST', url: '/api/runner/groups', body }),
    job: (id) => {
      const j = runner.snapshot().jobs.find((x) => x.id === id)
      if (!j) throw new Error(`no job ${id} in the snapshot`)
      return j
    },
    group: (id) => runner.snapshot().groups.find((g) => g.id === id),
    snap: () => runner.snapshot(),
    tick: async (times = 1, ms = 0) => {
      for (let i = 0; i < times; i++) {
        clock.t += ms
        await runner.tick()
      }
    },
    restart: async (over = {}) => {
      await runner.retire()
      return harness({ root, outputs, dir, archive, comfy, clock, boot: o.boot, stopWaitMs: o.stopWaitMs, ...over })
    },
    onDisk: () => JSON.parse(readFileSync(path.join(dir, 'state.json'), 'utf8')),
  }
  return h
}

/** The events a stream reply holds so far, with the ones that carry a rev. */
export function revEvents(reply: Reply): { event: string; data: any }[] {
  const out: { event: string; data: any }[] = []
  for (const block of reply.body.split('\n\n')) {
    const event = /^event: (.*)$/m.exec(block)?.[1]
    const data = /^data: (.*)$/m.exec(block)?.[1]
    if (event && data !== undefined) out.push({ event, data: JSON.parse(data) })
  }
  return out
}

export { call, open }

// ------------------------------------------------------ a stand-in on a port --

export type StandIn = {
  url: string
  /** `METHOD /path` of every request, in order. */
  log: string[]
  /** Prompt ids ComfyUI has waiting, which /queue lists; a test adds or clears them. */
  pending: string[]
  running: string[]
  /** Sockets open now, and the address each one asked for. */
  open: () => number
  socketUrls: string[]
  /** The text frames each socket sent, by socket. */
  frames: string[][]
  /** `open <n>` as each socket is taken, `close <n>` as the stand-in answers its close, in order, from 1. */
  order: string[]
  /**
   * Leave every request for `path` (as `/queue`) unanswered, as a stalled
   * ComfyUI does, until the function returned is called. The request is
   * logged as it arrives.
   */
  stall: (path: string) => () => void
  close: () => Promise<void>
}

/**
 * A stand-in ComfyUI on a free local port, for the runner's real client: a
 * queue, /free, /prompt that takes the client's prompt id, the jobs API, an
 * empty /history, and a bare WebSocket endpoint that completes the handshake,
 * reads the client's frames and answers a close. It sends no work anywhere.
 */
export async function standIn(o: { closeEchoMs?: number } = {}): Promise<StandIn> {
  const http = await import('node:http')
  const { createHash } = await import('node:crypto')
  const log: string[] = []
  const pending: string[] = []
  const running: string[] = []
  const socketUrls: string[] = []
  const frames: string[][] = []
  const order: string[] = []
  const live = new Set<import('node:net').Socket>()
  const gates = new Map<string, { wait: Promise<void>; open: () => void }>()
  const json = (res: import('node:http').ServerResponse, code: number, body: unknown) => {
    res.writeHead(code, { 'content-type': 'application/json' })
    res.end(JSON.stringify(body))
  }
  const server = http.createServer(async (req, res) => {
    let body = ''
    for await (const c of req) body += c
    const p = new URL(req.url ?? '/', 'http://x').pathname
    log.push(`${req.method} ${p}`)
    const gate = gates.get(p)
    if (gate) await gate.wait
    if (p === '/queue') return json(res, 200, { queue_running: running.map((id, i) => [i, id, {}, {}, []]), queue_pending: pending.map((id, i) => [i + 10, id, {}, {}, []]) })
    if (p === '/free' || p === '/interrupt') return json(res, 200, {})
    if (p === '/prompt') {
      const id = JSON.parse(body).prompt_id as string
      pending.push(id)
      return json(res, 200, { prompt_id: id, number: 1, node_errors: {} })
    }
    const cancel = /^\/api\/jobs\/([^/]+)\/cancel$/.exec(p)
    if (cancel) return json(res, 200, { cancelled: false })
    const job = /^\/api\/jobs\/([^/]+)$/.exec(p)
    if (job) {
      const id = decodeURIComponent(job[1]!)
      if (running.includes(id)) return json(res, 200, { id, status: 'in_progress', execution_start_time: 1 })
      if (pending.includes(id)) return json(res, 200, { id, status: 'pending' })
      return json(res, 404, { error: 'Job not found' })
    }
    if (p.startsWith('/history/')) return json(res, 200, {})
    return json(res, 404, { error: 'no route' })
  })
  server.on('upgrade', (req, duplex) => {
    const sock = duplex as import('node:net').Socket
    const key = String(req.headers['sec-websocket-key'])
    const accept = createHash('sha1').update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest('base64')
    sock.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`)
    live.add(sock)
    socketUrls.push(req.url ?? '')
    const n = socketUrls.length
    order.push(`open ${n}`)
    const got: string[] = []
    frames.push(got)
    let buf = Buffer.alloc(0)
    sock.on('data', (chunk: Buffer) => {
      buf = Buffer.concat([buf, chunk])
      // Client frames are masked; read each whole one.
      for (;;) {
        if (buf.length < 2) return
        const op = buf[0]! & 0x0f
        let len = buf[1]! & 0x7f
        let off = 2
        if (len === 126) {
          if (buf.length < 4) return
          len = buf.readUInt16BE(2)
          off = 4
        }
        if (buf.length < off + 4 + len) return
        const mask = buf.subarray(off, off + 4)
        const payload = Buffer.from(buf.subarray(off + 4, off + 4 + len)).map((b, i) => b ^ mask[i % 4]!)
        buf = buf.subarray(off + 4 + len)
        if (op === 1) got.push(Buffer.from(payload).toString('utf8'))
        if (op === 8) {
          // ComfyUI answers a close in its own time; a stand-in may be told to take a while.
          setTimeout(() => {
            order.push(`close ${n}`)
            try {
              sock.write(Buffer.from([0x88, 0]))
            } catch {
              /* gone */
            }
            sock.end()
          }, o.closeEchoMs ?? 0)
        }
      }
    })
    sock.on('close', () => live.delete(sock))
    sock.on('error', () => live.delete(sock))
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const port = (server.address() as import('node:net').AddressInfo).port
  return {
    url: `http://127.0.0.1:${port}`,
    log,
    pending,
    running,
    open: () => live.size,
    socketUrls,
    frames,
    order,
    stall: (p) => {
      let open = () => {}
      const wait = new Promise<void>((resolve) => { open = resolve })
      const gate = { wait, open: () => { if (gates.get(p) === gate) gates.delete(p); open() } }
      gates.set(p, gate)
      return gate.open
    },
    close: async () => {
      for (const g of [...gates.values()]) g.open()
      for (const s of live) s.destroy()
      server.closeAllConnections()
      await new Promise<void>((resolve) => server.close(() => resolve()))
    },
  }
}

/** Mount a SwitchGen plugin as Vite's preview server does, and wait for it to be ready. */
export async function mountPlugin(plugin: { configurePreviewServer?: unknown }): Promise<(req: unknown, res: unknown, next: () => void) => unknown> {
  let handler: ((req: unknown, res: unknown, next: () => void) => unknown) | null = null
  const hook = plugin.configurePreviewServer as (server: unknown) => unknown
  await hook({ middlewares: { use: (fn: typeof handler) => { handler = fn } } })
  if (!handler) throw new Error('the plugin mounted no middleware')
  return handler
}

/** Retire whatever runner the plugin registry holds, and forget the registry. */
export async function clearRegistry(): Promise<void> {
  const key = Symbol.for('switchgen.runner')
  const reg = (globalThis as Record<symbol, any>)[key]
  const current = await reg?.latest?.catch(() => null)
  await current?.retire?.()
  delete (globalThis as Record<symbol, any>)[key]
}
