/**
 * The lab page's own scripts (lab/ui/judge.js and lab/ui/refs.js), run in a
 * small stand-in for a browser: just enough of document and window for the
 * page to build its elements, with fetch answered by the test. Nothing is
 * served or sent.
 */
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import { describe, expect, it } from 'vitest'
import { REPO } from './helpers.ts'

/** An element: children, attributes, listeners and text, nothing more. */
class El {
  tagName: string
  childNodes: El[] = []
  attrs: Record<string, string> = {}
  listeners: Record<string, ((e: unknown) => unknown)[]> = {}
  style: Record<string, string> = {}
  dataset: Record<string, string> = {}
  className = ''
  hidden = false
  value = ''
  own: string | null = null
  classList = { toggle() {}, add() {}, remove() {}, contains: () => false }
  constructor(tag: string, text: string | null = null) {
    this.tagName = tag
    this.own = text
  }
  append(...kids: El[]) {
    this.childNodes.push(...kids)
  }
  replaceChildren(...kids: El[]) {
    this.childNodes = [...kids]
  }
  setAttribute(k: string, v: string) {
    this.attrs[k] = v
  }
  removeAttribute(k: string) {
    delete this.attrs[k]
  }
  addEventListener(type: string, fn: (e: unknown) => unknown) {
    ;(this.listeners[type] ??= []).push(fn)
  }
  querySelectorAll() {
    return []
  }
  get textContent(): string {
    return this.own ?? this.childNodes.map((c) => c.textContent).join('')
  }
  set textContent(v: string) {
    this.own = String(v)
    this.childNodes = []
  }
  /** Every element under this one, in order. */
  all(): El[] {
    return this.childNodes.flatMap((c) => [c, ...c.all()])
  }
}

type Answer = { status: number; body: unknown }

/** The page, loaded fresh, with `answer` giving the server's reply to each request. */
function page(answer: (method: string, url: string, body: unknown) => Answer) {
  const ids: Record<string, El> = { view: new El('main'), toast: new El('div'), net: new El('span') }
  const document = {
    createElement: (tag: string) => new El(tag),
    createTextNode: (t: string) => new El('#text', t),
    getElementById: (id: string) => ids[id] ?? null,
    querySelectorAll: () => [],
    addEventListener() {},
    body: new El('body'),
  }
  const requests: string[] = []
  const w: Record<string, unknown> = {
    document,
    Node: El,
    location: { hash: '#/' },
    addEventListener() {},
    setInterval: () => 0,
    clearInterval() {},
    setTimeout: () => 0,
    clearTimeout() {},
    scrollTo() {},
    localStorage: { getItem: () => null, setItem() {} },
    crypto: globalThis.crypto,
    console,
    fetch: async (url: string, init: { method?: string; body?: string } = {}) => {
      const method = init.method ?? 'GET'
      requests.push(`${method} ${url}`)
      const a = answer(method, url, init.body ? JSON.parse(init.body) : undefined)
      return { ok: a.status >= 200 && a.status < 300, status: a.status, json: async () => a.body }
    },
  }
  w.window = w
  vm.createContext(w)
  vm.runInContext(fs.readFileSync(path.join(REPO, 'lab', 'ui', 'judge.js'), 'utf8'), w, { filename: 'judge.js' })
  vm.runInContext(fs.readFileSync(path.join(REPO, 'lab', 'ui', 'refs.js'), 'utf8'), w, { filename: 'refs.js' })
  const lab = w.Lab as { render: () => void }
  return {
    view: ids.view,
    toast: ids.toast,
    requests,
    go: (hash: string) => {
      ;(w.location as { hash: string }).hash = hash
      lab.render()
    },
  }
}

const settle = async () => {
  for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r))
}

describe('the study page\'s reveal', () => {
  it('names the night that holds the reveal back, never "0 items are still to score"', async () => {
    const p = page((method, url) => {
      if (method === 'GET' && url === '/api/lab/runs') return { status: 200, body: [{ run: 'core-1', study: 'first-pass', state: 'judging', toJudge: 0 }] }
      if (method === 'POST' && url === '/api/lab/studies/first-pass/reveal') {
        return { status: 409, body: { error: 'Not everything is made, sealed and scored yet.', remaining: 0, waiting: ['cal-1'], unplanned: [] } }
      }
      return { status: 404, body: { error: 'no' } }
    })
    p.go('#/study/first-pass')
    await settle()
    const reveal = p.view.all().find((e) => e.tagName === 'button' && e.textContent === 'Reveal')
    expect(reveal, 'the Reveal button is on the page').toBeDefined()
    await reveal!.listeners.click[0]({})
    await settle()
    expect(p.requests).toContain('POST /api/lab/studies/first-pass/reveal')
    const text = p.view.textContent
    expect(text).toContain('cal-1 is not made and sealed yet')
    expect(text).not.toMatch(/0 items? (is|are) still to score/)
  })
})

describe('the Photos page\'s description', () => {
  it('a description the server refuses says "Not saved." and why, never "saved"', async () => {
    const posted: unknown[] = []
    const p = page((method, url, body) => {
      if (method === 'GET' && url === '/api/lab/refs') return { status: 200, body: [{ id: 'cat', sha12: 'abcdefabcdef', ext: 'jpg', width: 120, height: 80, mask: false, describe: null }] }
      if (method === 'GET' && url === '/api/lab/refs/needed') return { status: 200, body: { refs: [{ id: 'cat', present: true, describe: 'a tabby cat', needsMask: false, runs: [] }], dropDir: '/x/refs' } }
      if (method === 'POST' && url === '/api/lab/refs/cat/describe') {
        posted.push(body)
        return { status: 400, body: { error: 'X' } }
      }
      return { status: 404, body: { error: 'no' } }
    })
    p.go('#/refs')
    await settle()
    const box = p.view.all().find((e) => e.tagName === 'textarea')
    const save = p.view.all().find((e) => e.tagName === 'button' && e.textContent === 'Save description')
    expect(box, 'the description box is on the page').toBeDefined()
    expect(save, 'the Save description button is on the page').toBeDefined()
    box!.value = 'a cat held by its owner'
    await save!.listeners.click[0]({})
    await settle()
    expect(posted).toEqual([{ describe: 'a cat held by its owner' }])
    expect(p.toast.textContent).toBe('Not saved. X')
  })
})
