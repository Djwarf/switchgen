import { describe, expect, it } from 'vitest'
// @ts-expect-error the server is plain ESM without a declaration for its helpers
import { guardMutation, sameOrigin } from '../server/guard.mjs'

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
