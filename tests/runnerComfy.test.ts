import { createHash } from 'node:crypto'
import http from 'node:http'
import net from 'node:net'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { Unanswered, createComfy, previewOf } from '../server/runner/comfy.mjs'

/**
 * The queue's own line to ComfyUI, against a stand-in on a local port: what
 * each kind of answer, and each kind of silence, is taken to mean. The one
 * rule that matters most: a send is `unreached` only when no byte of it can
 * have arrived, since that is the only case in which sending again is safe.
 */

type Route = (req: http.IncomingMessage, res: http.ServerResponse, body: string) => void
let routes: Record<string, Route> = {}
let seen: { method: string; path: string; body: string }[] = []
let server: http.Server
let url = ''
const upgrades: { url: string; sock: net.Socket; got: string[] }[] = []

const json = (res: http.ServerResponse, code: number, body: unknown) => {
  res.writeHead(code, { 'content-type': 'application/json' })
  res.end(JSON.stringify(body))
}
const text = (res: http.ServerResponse, code: number, body: string, type = 'text/plain') => {
  res.writeHead(code, { 'content-type': type })
  res.end(body)
}

/** One unmasked frame from the server: opcode 1 text, 2 binary. */
function frame(op: number, payload: Buffer): Buffer {
  const len = payload.length
  const head = len < 126 ? Buffer.from([0x80 | op, len]) : Buffer.from([0x80 | op, 126, len >> 8, len & 255])
  return Buffer.concat([head, payload])
}

beforeAll(async () => {
  server = http.createServer(async (req, res) => {
    let body = ''
    for await (const c of req) body += c
    const p = new URL(req.url ?? '/', 'http://x').pathname
    seen.push({ method: req.method ?? '', path: p, body })
    const route = routes[`${req.method} ${p}`]
    if (route) return route(req, res, body)
    // An old ComfyUI, or a route it does not have: aiohttp's plain 404.
    text(res, 404, '404: Not Found')
  })
  server.on('upgrade', (req, sock: net.Socket) => {
    const key = String(req.headers['sec-websocket-key'])
    const accept = createHash('sha1').update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest('base64')
    sock.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`)
    const entry = { url: req.url ?? '', sock, got: [] as string[] }
    upgrades.push(entry)
    sock.on('data', (buf: Buffer) => {
      const op = buf[0]! & 0x0f
      let len = buf[1]! & 0x7f
      let off = 2
      if (len === 126) {
        len = buf.readUInt16BE(2)
        off = 4
      }
      const mask = buf.subarray(off, off + 4)
      const payload = Buffer.from(buf.subarray(off + 4, off + 4 + len)).map((b, i) => b ^ mask[i % 4]!)
      if (op === 1) entry.got.push(Buffer.from(payload).toString('utf8'))
      if (op === 8) {
        try {
          sock.write(Buffer.from([0x88, 0]))
        } catch {
          /* gone */
        }
        sock.end()
      }
    })
    sock.on('error', () => {})
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  url = `http://127.0.0.1:${(server.address() as net.AddressInfo).port}`
})

afterEach(() => {
  routes = {}
  seen = []
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

afterAll(async () => {
  for (const u of upgrades) u.sock.destroy()
  server.closeAllConnections()
  await new Promise<void>((resolve) => server.close(() => resolve()))
})

/** A port nothing listens on: one this machine just handed out and took back. */
async function closedPort(): Promise<number> {
  const s = net.createServer()
  await new Promise<void>((resolve) => s.listen(0, '127.0.0.1', resolve))
  const port = (s.address() as net.AddressInfo).port
  await new Promise<void>((resolve) => s.close(() => resolve()))
  return port
}

describe('a ComfyUI nobody can reach', () => {
  it('is unreached for a send and a release, unanswered for a read, and never a throw for a stop', async () => {
    const dead = createComfy({ url: `http://127.0.0.1:${await closedPort()}` })
    expect(await dead.submit({}, 'p', 'c')).toEqual({ unreached: true })
    expect(await dead.free()).toBe('unreached')
    await expect(dead.readQueue()).rejects.toBeInstanceOf(Unanswered)
    await expect(dead.getJob('x')).rejects.toBeInstanceOf(Unanswered)
    await expect(dead.history('x')).rejects.toBeInstanceOf(Unanswered)
    expect(await dead.cancel('x')).toBe(false)
    await expect(dead.interrupt('x')).resolves.toBeUndefined()
  })

  it('is unreached by name too, where the name stands for more than one address', async () => {
    const port = await closedPort()
    expect(await createComfy({ url: `http://localhost:${port}` }).submit({}, 'p', 'c')).toEqual({ unreached: true })
  })

  it('is unreached, never unknown, at the address every test is given, a port fetch will not open', async () => {
    const c = createComfy()
    expect(c.url).toBe('http://127.0.0.1:9')
    expect(await c.submit({}, 'p', 'c')).toEqual({ unreached: true })
    await expect(c.readQueue()).rejects.toBeInstanceOf(Unanswered)
  })
})

describe('sending a prompt', () => {
  it('posts the graph under the runner\'s prompt id and client id, and takes ComfyUI\'s JSON as accepted', async () => {
    routes['POST /prompt'] = (_q, res, body) => json(res, 200, { prompt_id: JSON.parse(body).prompt_id, number: 3, node_errors: {} })
    const c = createComfy({ url })
    expect(await c.submit({ '1': { class_type: 'KSampler', inputs: {} } }, 'pid-1', 'client-1')).toEqual({ accepted: true })
    expect(JSON.parse(seen[0]!.body)).toEqual({ prompt: { '1': { class_type: 'KSampler', inputs: {} } }, client_id: 'client-1', prompt_id: 'pid-1' })
  })

  it('is unknown when the connection dies mid request, or the answer is cut off after it began', async () => {
    const c = createComfy({ url })
    routes['POST /prompt'] = (req) => req.socket.destroy()
    expect(await c.submit({}, 'p', 'c')).toMatchObject({ unknown: true })
    routes['POST /prompt'] = (req, res) => {
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': '100' })
      res.write('{"prompt_')
      setTimeout(() => req.socket.destroy(), 20)
    }
    expect(await c.submit({}, 'p', 'c')).toMatchObject({ unknown: true })
  })

  it('is unknown for a gateway\'s answer, a 500, a page, or a 400 that is not ComfyUI\'s refusal', async () => {
    const c = createComfy({ url })
    const cases: [Route, unknown][] = [
      [(_q, res) => text(res, 502, 'Bad Gateway'), { unknown: true, reason: 'HTTP 502' }],
      [(_q, res) => json(res, 500, { error: 'boom' }), { unknown: true, reason: 'HTTP 500' }],
      [(_q, res) => text(res, 200, '<html>', 'text/html'), { unknown: true }],
      [(_q, res) => text(res, 400, '<html>', 'text/html'), { unknown: true }],
      [(_q, res) => json(res, 400, { message: 'something else' }), { unknown: true }],
      [(_q, res) => json(res, 200, ['a list']), { unknown: true }],
    ]
    for (const [route, want] of cases) {
      routes['POST /prompt'] = route
      expect(await c.submit({}, 'p', 'c')).toMatchObject(want as object)
    }
  })

  it('is refused for a 400 naming what ComfyUI would not take, worded as the page words it', async () => {
    const c = createComfy({ url })
    const refuse = (body: unknown) => {
      routes['POST /prompt'] = (_q, res) => json(res, 400, body)
      return c.submit({}, 'p', 'c')
    }
    const one = { '5': { errors: [{ message: 'Value not in list' }], class_type: 'LoadImage' } }
    expect(await refuse({ error: { type: 'prompt_outputs_failed_validation', message: 'Prompt outputs failed validation' }, node_errors: one })).toEqual({
      refused: true,
      status: 400,
      message: 'Prompt outputs failed validation',
      node: '5',
      nodeType: 'LoadImage',
      nodeErrors: one,
    })
    expect(await refuse({ error: 'no outputs' })).toMatchObject({ refused: true, message: 'no outputs', node: null, nodeErrors: null })
    const several = await refuse({ node_errors: { '5': { errors: [{ message: 'A' }], class_type: 'X' }, '6': { errors: [{ message: 'B' }] } } })
    expect(several).toMatchObject({ refused: true, message: 'A; B', node: null, nodeType: null })
    expect(await refuse({ error: {}, node_errors: {} })).toMatchObject({ refused: true, message: 'The queue rejected the job (HTTP 400).', nodeErrors: {} })
  })

  it('is unknown, with how long it waited, when ComfyUI takes the prompt and says nothing', async () => {
    routes['POST /prompt'] = () => {}
    const c = createComfy({ url, sendMs: 300 })
    expect(await c.submit({}, 'p', 'c')).toEqual({ unknown: true, reason: 'no answer within 300 ms' })
  })
})

describe('reading the queue', () => {
  it('gives the prompt ids in ComfyUI\'s order, dropping items with none', async () => {
    routes['GET /queue'] = (_q, res) =>
      json(res, 200, { queue_running: [[1, 'r1', {}, {}, []]], queue_pending: [[2, 'q1', {}], [3], [4, null], 'odd', [5, 'q2']] })
    expect(await createComfy({ url }).readQueue()).toEqual({ running: ['r1'], pending: ['q1', 'q2'] })
  })

  it('throws Unanswered for anything that is not a queue', async () => {
    const c = createComfy({ url })
    for (const route of [
      (_q, res) => text(res, 502, 'Bad Gateway'),
      (_q, res) => text(res, 200, '<html>', 'text/html'),
      (_q, res) => json(res, 500, { error: 'x' }),
      (_q, res) => json(res, 200, [['r1']]),
    ] as Route[]) {
      routes['GET /queue'] = route
      await expect(c.readQueue()).rejects.toBeInstanceOf(Unanswered)
    }
  })

  it('throws Unanswered when the answer stalls part way, within the read deadline', async () => {
    routes['GET /queue'] = (_q, res) => {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.write('{"queue_running":')
    }
    await expect(createComfy({ url, readMs: 300 }).readQueue()).rejects.toThrow(/no answer within 300 ms/)
  })
})

describe('asking about one job', () => {
  it('gives the job\'s id, status and start, and nothing else', async () => {
    routes['GET /api/jobs/j1'] = (_q, res) => json(res, 200, { id: 'j1', status: 'in_progress', execution_start_time: 1234, outputs_count: 2, workflow: {} })
    routes['GET /api/jobs/j2'] = (_q, res) => json(res, 200, { status: 'pending' })
    const c = createComfy({ url })
    expect(await c.getJob('j1')).toEqual({ id: 'j1', status: 'in_progress', execution_start_time: 1234 })
    expect(await c.getJob('j2')).toEqual({ id: 'j2', status: 'pending', execution_start_time: null })
  })

  it('is null for ComfyUI\'s 404, JSON or plain, as an old ComfyUI with no jobs API answers', async () => {
    routes['GET /api/jobs/j3'] = (_q, res) => json(res, 404, { error: 'Job not found' })
    const c = createComfy({ url })
    expect(await c.getJob('j3')).toBeNull()
    expect(await c.getJob('j4')).toBeNull()
  })

  it('throws Unanswered for a gateway\'s answer or a body with no status', async () => {
    const c = createComfy({ url })
    routes['GET /api/jobs/j5'] = (_q, res) => text(res, 503, 'Service Unavailable')
    await expect(c.getJob('j5')).rejects.toBeInstanceOf(Unanswered)
    routes['GET /api/jobs/j6'] = (_q, res) => json(res, 200, { id: 'j6' })
    await expect(c.getJob('j6')).rejects.toBeInstanceOf(Unanswered)
  })

  it('reads its history entry, null only for ComfyUI\'s empty answer', async () => {
    const entry = { prompt: [1, 'h1', {}, {}, []], outputs: {}, status: { status_str: 'success', messages: [] } }
    routes['GET /history/h1'] = (_q, res) => json(res, 200, { h1: entry })
    routes['GET /history/h2'] = (_q, res) => json(res, 200, {})
    routes['GET /history/h4'] = (_q, res) => text(res, 502, 'Bad Gateway')
    const c = createComfy({ url })
    expect(await c.history('h1')).toEqual(entry)
    expect(await c.history('h2')).toBeNull()
    // No route at all is not "no record".
    await expect(c.history('h3')).rejects.toBeInstanceOf(Unanswered)
    await expect(c.history('h4')).rejects.toBeInstanceOf(Unanswered)
  })
})

describe('asking whether ComfyUI has the jobs list', () => {
  /** A ComfyUI that answers every read with one reply, noting each address asked. */
  const answering = (status: number, body: string, type: string, asked: string[]) =>
    createComfy({
      url,
      fetch: (async (u: string) => {
        asked.push(u)
        return new Response(body, { status, headers: { 'content-type': type } })
      }) as typeof fetch,
    })

  it('is no for aiohttp\'s plain 404, which a ComfyUI older than the list gives every id, and yes for ComfyUI\'s own JSON one', async () => {
    const asked: string[] = []
    expect(await answering(404, '404: Not Found', 'text/plain', asked).hasJobsList!()).toBe(false)
    expect(await answering(404, JSON.stringify({ error: 'Job not found' }), 'application/json', asked).hasJobsList!()).toBe(true)
    // An id no job has, and a new one each time.
    expect(asked).toHaveLength(2)
    for (const u of asked) expect(u).toMatch(new RegExp(`^${url}/api/jobs/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`))
    expect(asked[0]).not.toBe(asked[1])
    // The stand-in's own plain 404, over a real connection.
    expect(await createComfy({ url }).hasJobsList!()).toBe(false)
    expect(seen.at(-1)?.path).toMatch(/^\/api\/jobs\/[0-9a-f-]{36}$/)
  })

  it('throws Unanswered where nothing answers, or a gateway answers for ComfyUI', async () => {
    const dead = createComfy({ url: `http://127.0.0.1:${await closedPort()}` })
    await expect(dead.hasJobsList!()).rejects.toBeInstanceOf(Unanswered)
    await expect(answering(502, 'Bad Gateway', 'text/plain', []).hasJobsList!()).rejects.toBeInstanceOf(Unanswered)
  })
})

describe('releasing memory', () => {
  it('asks ComfyUI to unload its models and free memory', async () => {
    routes['POST /free'] = (_q, res) => text(res, 200, '')
    expect(await createComfy({ url }).free()).toBe('ok')
    expect(JSON.parse(seen[0]!.body)).toEqual({ unload_models: true, free_memory: true })
  })

  it('is failed for an error or a silence, and unreached only where nothing was reached', async () => {
    routes['POST /free'] = (_q, res) => json(res, 500, { error: 'x' })
    expect(await createComfy({ url }).free()).toBe('failed')
    routes['POST /free'] = () => {}
    expect(await createComfy({ url, readMs: 300 }).free()).toBe('failed')
    expect(await createComfy({ url: `http://127.0.0.1:${await closedPort()}` }).free()).toBe('unreached')
  })
})

describe('stopping a prompt', () => {
  it('cancels by id, and says so only when ComfyUI did', async () => {
    routes['POST /api/jobs/k1/cancel'] = (_q, res) => json(res, 200, { cancelled: true })
    routes['POST /api/jobs/k2/cancel'] = (_q, res) => json(res, 200, { cancelled: false })
    routes['POST /api/jobs/k3/cancel'] = (_q, res) => json(res, 503, { cancelled: true })
    const c = createComfy({ url })
    expect(await c.cancel('k1')).toBe(true)
    expect(await c.cancel('k2')).toBe(false)
    expect(await c.cancel('k3')).toBe(false)
    expect(await c.cancel('k4')).toBe(false)
    expect(await createComfy({ url: `http://127.0.0.1:${await closedPort()}` }).cancel('k1')).toBe(false)
  })

  it('interrupts only a named prompt, never with no name, which would stop whatever is sampling', async () => {
    routes['POST /interrupt'] = (_q, res) => text(res, 200, '')
    const c = createComfy({ url })
    await c.interrupt('')
    await c.interrupt(undefined as unknown as string)
    expect(seen).toEqual([])
    await c.interrupt('k1')
    expect(seen).toEqual([{ method: 'POST', path: '/interrupt', body: '{"prompt_id":"k1"}' }])
  })
})

describe('previews', () => {
  const meta = Buffer.from(JSON.stringify({ prompt_id: 'pp', node_id: '3', image_type: 'image/png' }))
  const withMeta = Buffer.concat([Buffer.from([0, 0, 0, 4]), Buffer.from([0, 0, 0, meta.length]), meta, Buffer.from([9, 9, 9])])
  const bare = Buffer.concat([Buffer.from([0, 0, 0, 1, 0, 0, 0, 2]), Buffer.from([7])])

  it('reads both of ComfyUI\'s binary preview frames', () => {
    expect(previewOf(withMeta)).toEqual({ type: 'preview', promptId: 'pp', mime: 'image/png', bytes: Buffer.from([9, 9, 9]) })
    expect(previewOf(bare)).toEqual({ type: 'preview', promptId: null, mime: 'image/png', bytes: Buffer.from([7]) })
    expect(previewOf(Buffer.from([0, 0, 0, 1, 0, 0, 0, 1, 5]))!.mime).toBe('image/jpeg')
  })

  it('ignores a frame whose metadata runs past its end, another event, and a scrap', () => {
    expect(previewOf(Buffer.from([0, 0, 0, 4, 0, 0, 1, 0, 1]))).toBeNull()
    expect(previewOf(Buffer.from([0, 0, 0, 3, 0, 0, 0, 0, 1]))).toBeNull()
    expect(previewOf(Buffer.from([0, 0, 0, 1]))).toBeNull()
  })
})

// ------------------------------------------------------------- the socket --

class FakeSocket {
  static all: FakeSocket[] = []
  binaryType = ''
  sent: string[] = []
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((ev: { data: unknown }) => void) | null = null
  /** Whether close() answers with a close event, as a live socket does. */
  static answersClose = true
  constructor(public url: string) {
    FakeSocket.all.push(this)
  }
  open() {
    this.onopen?.()
  }
  send(x: string) {
    this.sent.push(x)
  }
  close() {
    if (FakeSocket.answersClose) this.onclose?.()
  }
  drop() {
    this.onerror?.()
    this.onclose?.()
  }
}

describe('the socket', () => {
  const FLAGS = '{"type":"feature_flags","data":{"supports_preview_metadata":true}}'

  function fresh() {
    FakeSocket.all = []
    FakeSocket.answersClose = true
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  }

  it('opens wss for https, asks for bytes as ArrayBuffers, and declares preview metadata first, again on each reconnect', async () => {
    fresh()
    const c = createComfy({ url: 'https://comfy.example:8443/', WebSocket: FakeSocket })
    const s = c.socket('client 1', () => {})
    const first = FakeSocket.all[0]!
    expect(first.url).toBe('wss://comfy.example:8443/ws?clientId=client%201')
    expect(first.binaryType).toBe('arraybuffer')
    first.open()
    expect(first.sent).toEqual([FLAGS])
    first.drop()
    await vi.advanceTimersByTimeAsync(500)
    FakeSocket.all[1]!.open()
    expect(FakeSocket.all[1]!.sent).toEqual([FLAGS])
    await s.close()
  })

  it('waits twice as long each time it cannot connect, from half a second to eight, and starts over once it has', async () => {
    fresh()
    const s = createComfy({ url: 'http://comfy.example', WebSocket: FakeSocket }).socket('c', () => {})
    const waits: number[] = []
    for (let i = 0; i < 6; i++) {
      const n = FakeSocket.all.length
      FakeSocket.all[n - 1]!.drop()
      let waited = 0
      while (FakeSocket.all.length === n) {
        await vi.advanceTimersByTimeAsync(100)
        waited += 100
      }
      waits.push(waited)
    }
    expect(waits).toEqual([500, 1000, 2000, 4000, 8000, 8000])
    FakeSocket.all.at(-1)!.open()
    FakeSocket.all.at(-1)!.drop()
    await vi.advanceTimersByTimeAsync(499)
    const n = FakeSocket.all.length
    await vi.advanceTimersByTimeAsync(1)
    expect(FakeSocket.all.length).toBe(n + 1)
    await s.close()
  })

  it('closes for good: no reconnect, a promise that settles, and a second close that does no harm', async () => {
    fresh()
    const s = createComfy({ url: 'http://comfy.example', WebSocket: FakeSocket }).socket('c', () => {})
    FakeSocket.all[0]!.open()
    await s.close()
    await s.close()
    await vi.advanceTimersByTimeAsync(20_000)
    expect(FakeSocket.all).toHaveLength(1)
    // A socket that never says it closed is let go after two seconds.
    FakeSocket.answersClose = false
    const t = createComfy({ url: 'http://comfy.example', WebSocket: FakeSocket }).socket('d', () => {})
    FakeSocket.all.at(-1)!.open()
    let settled = false
    const closing = Promise.resolve(t.close()).then(() => (settled = true))
    await vi.advanceTimersByTimeAsync(1999)
    expect(settled).toBe(false)
    await vi.advanceTimersByTimeAsync(1)
    await closing
    expect(settled).toBe(true)
  })

  it('passes on the messages the runner uses, in any form a frame comes in, and nothing else', async () => {
    fresh()
    const heard: any[] = []
    const s = createComfy({ url: 'http://comfy.example', WebSocket: FakeSocket }).socket('c', (m) => void heard.push(m))
    const ws = FakeSocket.all[0]!
    ws.open()
    const e1 = Uint8Array.from([0, 0, 0, 1, 0, 0, 0, 1, 5, 6])
    ws.onmessage!({ data: e1.buffer })
    ws.onmessage!({ data: e1 })
    ws.onmessage!({ data: new Blob([e1]) })
    ws.onmessage!({ data: Uint8Array.from([0, 0, 0, 3, 0, 0, 0, 0, 1]).buffer })
    ws.onmessage!({ data: Uint8Array.from([0, 0, 0, 4, 0, 0, 1, 0, 1]).buffer })
    ws.onmessage!({ data: JSON.stringify({ type: 'execution_start', data: { prompt_id: 'x', timestamp: 1 } }) })
    ws.onmessage!({ data: JSON.stringify({ type: 'executed', data: { node: '9' } }) })
    ws.onmessage!({ data: JSON.stringify({ type: 'feature_flags', data: {} }) })
    ws.onmessage!({ data: JSON.stringify({ type: 'status', data: 'odd' }) })
    ws.onmessage!({ data: 'not json' })
    await vi.advanceTimersByTimeAsync(0)
    expect(heard.filter((m) => m.type === 'preview').map((m) => [...m.bytes])).toEqual([[5, 6], [5, 6], [5, 6]])
    expect(heard.filter((m) => m.type !== 'preview')).toEqual([
      { type: 'execution_start', data: { prompt_id: 'x', timestamp: 1 } },
      { type: 'status', data: {} },
    ])
    await s.close()
  })

  it('goes on delivering past a listener that throws or rejects', async () => {
    fresh()
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    let n = 0
    const s = createComfy({ url: 'http://comfy.example', WebSocket: FakeSocket }).socket('c', () => {
      n++
      if (n === 1) throw new Error('boom')
      return Promise.reject(new Error('later')) as unknown as void
    })
    const ws = FakeSocket.all[0]!
    ws.open()
    for (let i = 0; i < 3; i++) ws.onmessage!({ data: JSON.stringify({ type: 'status', data: {} }) })
    await vi.advanceTimersByTimeAsync(0)
    expect(n).toBe(3)
    expect(warn).toHaveBeenCalledTimes(3)
    await s.close()
  })

  it('says once, and does not throw, where there is no WebSocket at all', async () => {
    fresh()
    vi.stubGlobal('WebSocket', undefined)
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const s = createComfy({ url: 'http://comfy.example' }).socket('c', () => {})
    await vi.advanceTimersByTimeAsync(20_000)
    expect(warn).toHaveBeenCalledTimes(1)
    expect(String(warn.mock.calls[0]![0])).toMatch(/could not open ComfyUI's socket/)
    await s.close()
  })

  it('works over a real socket: the first frame, text and binary messages, and a reconnect after a drop', async () => {
    const heard: any[] = []
    const before = upgrades.length
    const s = createComfy({ url }).socket('client-9', (m) => void heard.push(m))
    const WAIT = { timeout: 5000, interval: 20 }
    await vi.waitFor(() => expect(upgrades[before]?.got).toEqual([FLAGS]), WAIT)
    const one = upgrades[before]!
    expect(one.url).toBe('/ws?clientId=client-9')
    const meta = Buffer.from(JSON.stringify({ prompt_id: 'pp', image_type: 'image/png' }))
    one.sock.write(frame(1, Buffer.from(JSON.stringify({ type: 'progress', data: { prompt_id: 'p', node: '3', value: 2, max: 10 } }))))
    one.sock.write(frame(2, Buffer.concat([Buffer.from([0, 0, 0, 4, 0, 0, 0, meta.length]), meta, Buffer.from([1, 2])])))
    one.sock.write(frame(2, Buffer.from([0, 0, 0, 1, 0, 0, 0, 1, 3])))
    await vi.waitFor(() => expect(heard).toHaveLength(3), WAIT)
    expect(heard[0]).toEqual({ type: 'progress', data: { prompt_id: 'p', node: '3', value: 2, max: 10 } })
    expect(heard[1]).toMatchObject({ type: 'preview', promptId: 'pp', mime: 'image/png' })
    expect([...heard[1].bytes]).toEqual([1, 2])
    expect(heard[2]).toMatchObject({ type: 'preview', promptId: null, mime: 'image/jpeg' })
    one.sock.destroy()
    await vi.waitFor(() => expect(upgrades[before + 1]?.got).toEqual([FLAGS]), WAIT)
    await s.close()
    await new Promise((resolve) => setTimeout(resolve, 700))
    expect(upgrades).toHaveLength(before + 2)
  })
})
