import { readFileSync } from 'node:fs'
import path from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * What the Reel and Video desks say while ComfyUI is not answering: that
 * nothing can be queued, in the words the Pictures desk uses. A shot or a
 * light clip pressed anyway only failed, and after earlyoom took ComfyUI down
 * the press looked ready all the while systemd was bringing it back.
 */
const SENTENCE = 'ComfyUI is not answering, so nothing can be queued.'

class FakeSocket {
  static OPEN = 1
  static CONNECTING = 0
  readyState = 0
  binaryType = ''
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((ev: { data: unknown }) => void) | null = null
  url: string
  constructor(url: string) {
    this.url = url
    queueMicrotask(() => {
      this.readyState = 1
      this.onopen?.()
    })
  }
  send() {}
  close() {
    this.readyState = 3
    this.onclose?.()
  }
}

describe('the Reel room', () => {
  const kept = new Map<string, string>()
  const storage = {
    getItem: (k: string) => kept.get(k) ?? null,
    setItem: (k: string, v: string) => void kept.set(k, String(v)),
    removeItem: (k: string) => void kept.delete(k),
    key: (i: number) => [...kept.keys()][i] ?? null,
    get length() {
      return kept.size
    },
  }

  beforeEach(() => {
    kept.clear()
    // Nothing answers: the room is drawn once, without its catalogue.
    vi.stubGlobal('fetch', async () => new Response('{}', { status: 599, headers: { 'content-type': 'application/json' } }))
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.stubGlobal('location', { protocol: 'http:', host: 'harness', hash: '' })
    vi.stubGlobal('localStorage', storage)
    vi.stubGlobal('sessionStorage', storage)
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {}, location: { hash: '' }, history: {}, matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }) })
    vi.stubGlobal('document', { addEventListener() {}, removeEventListener() {}, visibilityState: 'visible' })
    vi.resetModules()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  /** The room with one written shot and nothing rendered, as it draws now. */
  async function room() {
    const { reel } = await import('../src/components/reel/store')
    reel.setShot(reel.add(), { prompt: 'a tram crossing a bridge at night' })
    const Reel = (await import('../src/routes/Reel')).default
    return () => renderToStaticMarkup(createElement(Reel))
  }

  it('says nothing can be queued while ComfyUI is not answering, and offers no shortcut to start it', async () => {
    const draw = await room()
    const comfy = await import('../src/lib/comfy')
    expect(comfy.connectionState()).toBe('closed')
    const html = draw()
    expect(html).toContain(SENTENCE)
    expect(html).not.toContain('Control and Enter starts')
    expect(html).toMatch(/<button type="button" class="press" disabled="">/)
  })

  it('says what a render costs once ComfyUI answers', async () => {
    const draw = await room()
    const comfy = await import('../src/lib/comfy')
    comfy.connect()
    await vi.waitFor(() => expect(comfy.connectionState()).toBe('open'), { timeout: 2000, interval: 5 })
    const html = draw()
    expect(html).not.toContain(SENTENCE)
    expect(html).toContain('One generation of several minutes.')
  })
})

describe('the Video desk', () => {
  // Its refusal is private to the page and read when Make is pressed, which
  // this suite has no DOM to do, so the page is read instead.
  const page = readFileSync(path.resolve(import.meta.dirname, '..', 'src', 'routes', 'Video.tsx'), 'utf8')

  it('refuses to queue while ComfyUI is not answering, with the Pictures desk\'s words', () => {
    const reason = /function reasonFor\([\s\S]*?\n\}\n/.exec(page)?.[0] ?? ''
    expect(reason).toContain(`if (offline) return '${SENTENCE}'`)
    // Last, so a blank prompt or a clip too big is still what is said.
    expect(reason.indexOf('if (offline)')).toBeGreaterThan(reason.indexOf("'Describe the shot first.'"))
  })

  it('asks the socket at every place it decides, the press as well as what the button shows', () => {
    const calls = page.split('\n').filter((l) => l.includes('reasonFor(') && !l.includes('function reasonFor'))
    expect(calls.length).toBeGreaterThanOrEqual(2)
    for (const c of calls) expect(c).toMatch(/reasonFor\([^\n]*, connection(State\(\))? === 'closed'\)/)
  })
})
