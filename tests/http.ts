import { EventEmitter } from 'node:events'
import { mkdtempSync, mkdirSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'

/**
 * Drive a Connect-style middleware without a socket or a port: a request that
 * is a readable stream, so readBody can consume it, and a response that
 * records what the handler said. Nothing here listens anywhere.
 */
export type Handler = (req: unknown, res: unknown, next: () => void) => unknown

export type Reply = {
  status: number
  headers: Record<string, string | number>
  body: string
  /** True when the handler passed the request on instead of answering. */
  passed: boolean
  json: () => any
}

export function call(
  handler: Handler,
  opts: { method?: string; url: string; headers?: Record<string, string>; body?: unknown },
): Promise<Reply> {
  const method = opts.method ?? 'GET'
  const raw = opts.body === undefined ? [] : [Buffer.from(JSON.stringify(opts.body))]
  const headers: Record<string, string> = { ...(opts.body !== undefined ? { 'content-type': 'application/json' } : {}), ...opts.headers }
  const req = Object.assign(Readable.from(raw), { method, url: opts.url, headers })

  return new Promise((resolve) => {
    const reply: Reply = {
      status: 200,
      headers: {},
      body: '',
      passed: false,
      json: () => JSON.parse(reply.body),
    }
    const res = Object.assign(new EventEmitter(), {
      req,
      destroyed: false,
      headersSent: false,
      setHeader(k: string, v: string | number) {
        reply.headers[k.toLowerCase()] = v
      },
      getHeader(k: string) {
        return reply.headers[k.toLowerCase()]
      },
      writeHead(code: number, h: Record<string, string> = {}) {
        reply.status = code
        for (const [k, v] of Object.entries(h)) reply.headers[k.toLowerCase()] = v
        res.headersSent = true
        return res
      },
      write(chunk: string | Buffer) {
        res.headersSent = true
        reply.body += chunk.toString()
        return true
      },
      end(chunk?: string | Buffer) {
        res.headersSent = true
        if (chunk !== undefined) reply.body += chunk.toString()
        res.emit('finish')
        resolve(reply)
      },
    })
    // An accessor, so `res.statusCode = 400` lands in the reply. Object.assign
    // would copy a getter's value rather than the accessor itself.
    Object.defineProperty(res, 'statusCode', {
      get: () => reply.status,
      set: (v: number) => {
        reply.status = v
      },
    })
    handler(req, res, () => {
      reply.passed = true
      resolve(reply)
    })
  })
}

/** The middleware a SwitchGen plugin mounts, taken the way Vite's preview server takes it. */
export function mounted(plugin: { configurePreviewServer?: unknown }): Handler {
  let handler: Handler | null = null
  const hook = plugin.configurePreviewServer as (server: unknown) => void
  hook({ middlewares: { use: (fn: Handler) => { handler = fn } } })
  if (!handler) throw new Error('the plugin mounted no middleware')
  return handler
}

/**
 * Fresh models, outputs and ComfyUI folders under the system temp folder, set
 * in the environment the server modules read when they load. Import the
 * modules after calling this, never at the top of a test file, or they keep
 * their defaults, which name the author's own folders.
 */
export function tempRoots(): { root: string; models: string; outputs: string } {
  const root = mkdtempSync(path.join(os.tmpdir(), 'switchgen-test-'))
  const models = path.join(root, 'models')
  const outputs = path.join(root, 'outputs')
  const comfy = path.join(root, 'comfy')
  for (const dir of [models, outputs, comfy]) mkdirSync(dir, { recursive: true })
  Object.assign(process.env, {
    SWITCHGEN_MODELS: models,
    SWITCHGEN_OUTPUTS: outputs,
    SWITCHGEN_ARCHIVE: path.join(outputs, '.switchgen', 'archive.json'),
    SWITCHGEN_THUMBS: path.join(outputs, '.switchgen', 'thumbs'),
    SWITCHGEN_COMFY: comfy,
    SWITCHGEN_COMFY_INPUT: path.join(comfy, 'input'),
    SWITCHGEN_PYTHON: path.join(comfy, 'no-python'),
    SWITCHGEN_WD14: path.join(models, 'wd14'),
    SWITCHGEN_LORA_DIR: path.join(models, 'Lora'),
    HF_TOKEN_FILE: path.join(root, 'no-token'),
  })
  return { root, models, outputs }
}
