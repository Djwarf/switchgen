import { readFileSync } from 'node:fs'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * What a page served over plain http, as the phone reaches it on the tailnet,
 * still does for itself: keep the screen on where the browser allows it, and
 * copy text where there is no navigator.clipboard. Also, that the rooms load
 * as chunks of their own.
 */

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('keeping the screen on', () => {
  type Sentinel = { released: boolean; release: ReturnType<typeof vi.fn>; addEventListener: (t: string, fn: () => void) => void; drop: () => void }
  let sentinels: Sentinel[]
  let request: ReturnType<typeof vi.fn>
  let onVisible: (() => void)[]
  let doc: { visibilityState: string; addEventListener: (t: string, fn: () => void) => void }

  beforeEach(() => {
    vi.resetModules()
    sentinels = []
    onVisible = []
    request = vi.fn(async () => {
      const gone: (() => void)[] = []
      const s: Sentinel = {
        released: false,
        release: vi.fn(async () => {
          s.released = true
        }),
        addEventListener: (_t, fn) => void gone.push(fn),
        // The browser lets the lock go by itself when the page is hidden.
        drop: () => {
          s.released = true
          for (const fn of gone) fn()
        },
      }
      sentinels.push(s)
      return s
    })
    doc = {
      visibilityState: 'visible',
      addEventListener: (t, fn) => {
        if (t === 'visibilitychange') onVisible.push(fn)
      },
    }
    vi.stubGlobal('document', doc)
  })
  const settle = () => new Promise((resolve) => setTimeout(resolve, 0))

  it('asks once for every hold, and lets go when the last hold goes, once', async () => {
    vi.stubGlobal('navigator', { wakeLock: { request } })
    const { holdAwake, wakeLockAvailable } = await import('../src/lib/wakeLock')
    expect(wakeLockAvailable()).toBe(true)
    const lane = holdAwake('video lane')
    const reel = holdAwake('reel')
    await settle()
    expect(request).toHaveBeenCalledTimes(1)
    expect(request).toHaveBeenCalledWith('screen')
    lane()
    await settle()
    expect(sentinels[0]!.release).not.toHaveBeenCalled()
    reel()
    reel()
    await settle()
    expect(sentinels[0]!.release).toHaveBeenCalledTimes(1)
  })

  it('does nothing where the browser offers no wake lock, as over plain http', async () => {
    vi.stubGlobal('navigator', {})
    const { holdAwake, wakeLockAvailable } = await import('../src/lib/wakeLock')
    expect(wakeLockAvailable()).toBe(false)
    const letGo = holdAwake('batch')
    expect(() => letGo()).not.toThrow()
  })

  it('asks again when the page is shown after the browser let the lock go', async () => {
    vi.stubGlobal('navigator', { wakeLock: { request } })
    const { holdAwake } = await import('../src/lib/wakeLock')
    const hold = holdAwake('reel')
    await settle()
    expect(request).toHaveBeenCalledTimes(1)
    doc.visibilityState = 'hidden'
    sentinels[0]!.drop()
    doc.visibilityState = 'visible'
    for (const fn of onVisible) fn()
    await settle()
    expect(request).toHaveBeenCalledTimes(2)
    hold()
    await settle()
    expect(sentinels[1]!.release).toHaveBeenCalledTimes(1)
  })
})

describe('copying text', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('uses the clipboard where the page has one', async () => {
    const writeText = vi.fn(async () => {})
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    const { copyText } = await import('../src/lib/clipboard')
    expect(await copyText('a prompt')).toBe(true)
    expect(writeText).toHaveBeenCalledWith('a prompt')
  })

  /** Just enough of a document for a field to be put in, selected, copied from and taken out. */
  function page(copies: boolean) {
    const children: unknown[] = []
    const execCommand = vi.fn(() => copies)
    const field = {
      value: '',
      style: {} as Record<string, string>,
      tabIndex: 0,
      setAttribute() {},
      focus() {},
      select() {},
      setSelectionRange() {},
      remove() {
        children.splice(children.indexOf(field), 1)
      },
    }
    vi.stubGlobal('document', {
      body: { appendChild: (el: unknown) => void children.push(el) },
      createElement: () => field,
      getSelection: () => null,
      activeElement: null,
      execCommand,
    })
    return { children, execCommand, field }
  }

  it.each([true, false])('falls back to selecting the text where there is no clipboard (copied: %s)', async (copies) => {
    vi.stubGlobal('navigator', {})
    const p = page(copies)
    const { copyText } = await import('../src/lib/clipboard')
    expect(await copyText('a prompt')).toBe(copies)
    expect(p.execCommand).toHaveBeenCalledWith('copy')
    expect(p.field.value).toBe('a prompt')
    expect(p.children).toEqual([])
  })
})

describe('the rooms', () => {
  const app = readFileSync(path.resolve(import.meta.dirname, '..', 'src', 'App.tsx'), 'utf8')

  it('load as chunks of their own, not in the first one', () => {
    expect(app).not.toMatch(/^import (?!type)[^\n]*from '\.\/routes\//m)
    for (const room of ['./routes/Pictures', './routes/Video', './routes/Reel', './routes/ArchivePage']) {
      expect(app, room).toContain(`import('${room}')`)
    }
  })
})
