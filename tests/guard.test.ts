import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
// @ts-expect-error the server is plain ESM without a declaration for its helpers
import { confineReal, guardMutation, guardOrigin, hostAllowed, proxyWriteGuard, safely, sameOrigin, upgradeAllowed } from '../server/guard.mjs'
import { call } from './http'

const req = (headers: Record<string, string>, method = 'POST') => ({ headers, method })

function res() {
  const r = { statusCode: 200, headers: {} as Record<string, string>, body: '', setHeader(k: string, v: string) { r.headers[k] = v }, end(b: string) { r.body = b } }
  return r
}

describe('the same-origin guard', () => {
  it('trusts the browser\'s own verdict first', () => {
    expect(sameOrigin(req({ 'sec-fetch-site': 'same-origin', origin: 'http://x', host: 'y' }))).toBe(true)
    expect(sameOrigin(req({ 'sec-fetch-site': 'cross-site', origin: 'http://freya:5273', host: 'freya:5273' }))).toBe(false)
  })

  it('matches Origin to Host when there is no verdict, case-insensitively', () => {
    expect(sameOrigin(req({ origin: 'http://FREYA:5273', host: 'freya:5273' }))).toBe(true)
    expect(sameOrigin(req({ origin: 'http://evil.test', host: 'freya:5273' }))).toBe(false)
    expect(sameOrigin(req({ origin: 'not a url', host: 'h' }))).toBe(false)
  })

  it('lets a request with neither header through: curl and scripts', () => {
    expect(sameOrigin(req({}))).toBe(true)
  })

  it('answers 403 for cross-site and 415 for the wrong body type, and passes the rest', () => {
    const a = res()
    expect(guardMutation(req({ origin: 'http://evil.test', host: 'h', 'content-type': 'application/json' }), a)).toBe(false)
    expect(a.statusCode).toBe(403)
    const b = res()
    expect(guardMutation(req({ 'content-type': 'text/plain' }), b)).toBe(false)
    expect(b.statusCode).toBe(415)
    const c = res()
    expect(guardMutation(req({ 'content-type': 'application/json; charset=utf-8' }), c)).toBe(true)
    expect(guardMutation(req({}, 'GET'), res())).toBe(true)
  })
})

describe('the WebSocket handshake check', () => {
  // What vite.config.ts builds: the machine's own names, its addresses and the tailnet.
  const allowed = ['localhost', 'freya', '.ts.net', '192.168.1.20']
  const handshake = (host: string, origin?: string, site?: string) => ({
    headers: { host, ...(origin ? { origin } : {}), ...(site ? { 'sec-fetch-site': site } : {}) },
  })

  it('lets this server\'s own page through by address, by name and by tailnet name', () => {
    expect(upgradeAllowed(handshake('192.168.1.20:5273', 'http://192.168.1.20:5273', 'same-origin'), allowed)).toBe(true)
    expect(upgradeAllowed(handshake('freya:5273', 'http://freya:5273', 'same-origin'), allowed)).toBe(true)
    expect(upgradeAllowed(handshake('freya.tail1234.ts.net', 'https://freya.tail1234.ts.net', 'same-origin'), allowed)).toBe(true)
  })

  it('refuses a hostile name rebound to this machine, even from its own page', () => {
    expect(upgradeAllowed(handshake('attacker.example:5273', 'http://attacker.example:5273', 'same-origin'), allowed)).toBe(false)
  })

  it('refuses a page elsewhere that names this machine by an allowed address', () => {
    expect(upgradeAllowed(handshake('127.0.0.1:5273', 'http://evil.example', 'cross-site'), allowed)).toBe(false)
    expect(upgradeAllowed(handshake('127.0.0.1:5273', 'http://evil.example'), allowed)).toBe(false)
  })

  it('matches a dotted entry as a domain, not as a suffix of any name', () => {
    expect(hostAllowed('evilts.net', ['.ts.net'])).toBe(false)
    expect(hostAllowed('ts.net', ['.ts.net'])).toBe(true)
    expect(hostAllowed('a.b.ts.net:443', ['.ts.net'])).toBe(true)
    expect(hostAllowed('[::1]:5273', [])).toBe(true)
    expect(hostAllowed('10.0.0.5', [])).toBe(true)
    expect(hostAllowed('LocalHost:5273', [])).toBe(true)
    expect(hostAllowed('freya.lan', ['freya'])).toBe(false)
  })
})

describe('confineReal', () => {
  let tmp = ''
  let real = ''
  let link = ''
  let outside = ''

  beforeAll(() => {
    tmp = mkdtempSync(path.join(os.tmpdir(), 'switchgen-confine-'))
    real = path.join(tmp, 'data', 'outputs')
    outside = path.join(tmp, 'elsewhere')
    mkdirSync(path.join(real, 'sub'), { recursive: true })
    mkdirSync(outside, { recursive: true })
    writeFileSync(path.join(real, 'sub', 'a.png'), 'x')
    writeFileSync(path.join(outside, 'secret'), 'x')
    // An outputs folder that is a link to a data disk.
    link = path.join(tmp, 'outputs')
    symlinkSync(real, link)
    // A link inside the root that points out of it.
    symlinkSync(outside, path.join(real, 'escape'))
  })

  afterAll(() => rmSync(tmp, { recursive: true, force: true }))

  it('finds a file under a real root', async () => {
    expect(await confineReal(real, 'sub/a.png')).toBe(path.join(real, 'sub', 'a.png'))
  })

  it('finds the same file under a linked root, spelled under the root as configured', async () => {
    expect(await confineReal(link, 'sub/a.png')).toBe(path.join(link, 'sub', 'a.png'))
    expect(await confineReal(link, '/sub/a.png')).toBe(path.join(link, 'sub', 'a.png'))
  })

  it('allows a leaf that does not exist yet under a linked root', async () => {
    expect(await confineReal(link, 'sub/new/b.png')).toBe(path.join(link, 'sub', 'new', 'b.png'))
  })

  it('refuses a link inside the root that leads out of it, and anything under it', async () => {
    expect(await confineReal(real, 'escape/secret')).toBeNull()
    expect(await confineReal(link, 'escape/missing.png')).toBeNull()
  })

  it('refuses a climb out of the root', async () => {
    expect(await confineReal(real, '../elsewhere/secret')).toBeNull()
    expect(await confineReal(real, 'sub/../../../elsewhere/secret')).toBeNull()
  })

  it('refuses everything when the root itself is missing', async () => {
    expect(await confineReal(path.join(tmp, 'no-such-root'), 'a.png')).toBeNull()
  })
})

describe('safely', () => {
  it('answers 500 for a handler that rejects, instead of letting it reach the process', async () => {
    const r = await call(safely(async () => { throw new Error('boom') }), { url: '/api/x' })
    expect(r.status).toBe(500)
    expect(r.json()).toEqual({ error: 'boom' })
  })

  it('answers 500 for a handler that throws before it returns a promise', async () => {
    const r = await call(safely(() => { throw new Error('sooner') }), { url: '/api/x' })
    expect(r.status).toBe(500)
    expect(r.json().error).toBe('sooner')
  })
})

describe('writes bound for ComfyUI through the proxy', () => {
  // The proxy rewrites Origin to ComfyUI's own, so ComfyUI cannot tell which
  // page sent a write. The guard in front of it holds those writes to the
  // rule the /api routes follow.
  const through = (url: string, headers: Record<string, string>, method = 'POST') =>
    call(proxyWriteGuard('/comfy'), { method, url, headers })

  it('refuses a same-site page on another port of this machine', async () => {
    const r = await through('/comfy/prompt', { 'sec-fetch-site': 'same-site', origin: 'http://127.0.0.1:8080', host: '127.0.0.1:5273' })
    expect(r.status).toBe(403)
    expect(r.passed).toBe(false)
  })

  it('refuses an Origin that does not match Host when the browser gives no verdict', async () => {
    const r = await through('/comfy/free', { origin: 'http://127.0.0.1:8080', host: '127.0.0.1:5273' })
    expect(r.status).toBe(403)
    expect(r.passed).toBe(false)
  })

  it('matches the prefix the way the proxy does, from the start of the raw URL', async () => {
    const r = await through('/comfyprompt', { 'sec-fetch-site': 'cross-site', origin: 'http://evil.example', host: '127.0.0.1:5273' })
    expect(r.status).toBe(403)
    expect(r.passed).toBe(false)
  })

  it('passes this server\'s own writes, every read, and paths that are not the proxy\'s', async () => {
    expect((await through('/comfy/prompt', { 'sec-fetch-site': 'same-origin', origin: 'http://127.0.0.1:5273', host: '127.0.0.1:5273' })).passed).toBe(true)
    expect((await through('/comfy/view?filename=a.png', { 'sec-fetch-site': 'same-site' }, 'GET')).passed).toBe(true)
    // /api routes run their own guard, with the body type as well.
    expect((await through('/api/x', { 'sec-fetch-site': 'cross-site', origin: 'http://evil.example' })).passed).toBe(true)
  })

  it('shares one origin rule with the /api routes', () => {
    expect(guardOrigin(req({ 'sec-fetch-site': 'cross-site' }, 'GET'), res())).toBe(true)
    const r = res()
    expect(guardOrigin(req({ 'sec-fetch-site': 'same-site' }), r)).toBe(false)
    expect(r.statusCode).toBe(403)
    const m = res()
    expect(guardMutation(req({ 'sec-fetch-site': 'same-site', 'content-type': 'application/json' }), m)).toBe(false)
    expect(m.statusCode).toBe(403)
    expect(JSON.parse(m.body).error).toMatch(/cross-site request refused/)
  })
})
