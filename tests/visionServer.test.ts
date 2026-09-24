import { chmodSync, closeSync, mkdirSync, openSync, readFileSync, rmSync, truncateSync, unlinkSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { call, mounted, open, tempRoots, type Handler } from './http'

/**
 * What /api/vision/capabilities says about a tagger that is on disk but not
 * all there, and how readings take their turn. Nothing here runs Python: the
 * interpreter is a shell script that logs when it starts and ends, waits a
 * moment and answers with no tags.
 */
let root = ''
let wd14 = ''
let outputs = ''
let readerLog = ''
let h: Handler
let memoryRefusal: (peak: number, free: number, total: number, floorPercent?: number, margin?: number) => string | null

/** A stand-in reader: `start` and `end` lines in the log, and an empty answer. */
const fakeReader = (log: string) => `#!/bin/sh
echo "start $$" >> ${JSON.stringify(log)}
sleep "\${FAKE_READER_SECONDS:-0.3}"
echo "end $$" >> ${JSON.stringify(log)}
printf '%s' '{"tag":{"rows":[]},"detect":null}'
`

beforeAll(async () => {
  const roots = tempRoots()
  root = roots.root
  outputs = roots.outputs
  wd14 = process.env.SWITCHGEN_WD14!
  readerLog = path.join(root, 'reader.log')
  const python = path.join(root, 'python')
  writeFileSync(python, fakeReader(readerLog))
  chmodSync(python, 0o755)
  process.env.SWITCHGEN_PYTHON = python
  const vision = await import('../server/vision.mjs')
  memoryRefusal = vision.memoryRefusal
  h = mounted(vision.switchgenVision())
})

beforeEach(() => {
  rmSync(wd14, { recursive: true, force: true })
  mkdirSync(wd14, { recursive: true })
})

afterAll(() => rmSync(root, { recursive: true, force: true }))

const model = () => path.join(wd14, 'model.onnx')
const tags = () => path.join(wd14, 'selected_tags.csv')
const caps = async () => (await call(h, { url: '/api/vision/capabilities' })).json()

describe('a tagger file that is on disk but short', () => {
  it('offers no fetch for a short file a fetch would refuse to touch, and says to move it aside', async () => {
    writeFileSync(model(), Buffer.alloc(1000))
    const c = await caps()
    expect(c.tagger).toBe(false)
    expect(c.install).toBeNull()
    expect(c.reason).toContain(model())
    expect(c.reason).toContain('move it aside')
  })

  it('offers the fetch again for a download that stopped part way', async () => {
    writeFileSync(model(), Buffer.alloc(1000))
    writeFileSync(`${model()}.aria2`, 'control')
    const c = await caps()
    expect(c.install.missing).toContain('model.onnx')
    expect(c.reason).toContain('did not finish')
    unlinkSync(`${model()}.aria2`)
  })

  it('names both files when both are short', async () => {
    writeFileSync(model(), Buffer.alloc(1000))
    writeFileSync(tags(), 'tag_id,name\n')
    const c = await caps()
    expect(c.install).toBeNull()
    expect(c.reason).toContain(`${model()} and ${tags()} are on disk`)
    expect(c.reason).toContain('move them aside')
  })
})

const GIB = 1073741824

describe('the memory a reading needs', () => {
  it('refuses a reading that would leave too little, with both figures', () => {
    const said = memoryRefusal(1.9 * GIB, 3.1 * GIB, 32 * GIB, 8, 0.5 * GIB)
    expect(said).toContain('1.9 GB')
    expect(said).toContain('3.1 GB')
    // What would be left, and the 8% of 32 GB at which earlyoom acts.
    expect(said).toContain('about 1.2 GB')
    expect(said).toContain('2.6 GB')
  })

  it('lets a reading that fits start', () => {
    expect(memoryRefusal(1.9 * GIB, 5.1 * GIB, 32 * GIB, 8, 0.5 * GIB)).toBeNull()
  })

  it('says only so much is free when the reading would not fit at all', () => {
    expect(memoryRefusal(1.9 * GIB, 1.2 * GIB, 32 * GIB, 8, 0.5 * GIB)).toContain('only 1.2 GB is free')
  })

  it('takes a floor of 0 as no line at all', () => {
    expect(memoryRefusal(1.9 * GIB, 3.1 * GIB, 32 * GIB, 0, 0.5 * GIB)).toBeNull()
  })

  // The line is 8% of 32 GB, about 2.6 GB, and the margin above it 0.5 GB.
  // What a reading would leave is said to be under the line when it is, and
  // too close to it only when it would stay above it.
  it('says a reading would leave less than the line, not that it is close to it', () => {
    const said = memoryRefusal(1.9 * GIB, 4.0 * GIB, 32 * GIB, 8, 0.5 * GIB)
    expect(said).toContain('leave about 2.1 GB, under the 2.6 GB')
    expect(said).not.toContain('too close')
  })

  it('calls a reading that would stay just above the line too close to it', () => {
    expect(memoryRefusal(1.9 * GIB, 4.8 * GIB, 32 * GIB, 8, 0.5 * GIB)).toContain('That is too close to the 2.6 GB')
  })

  it('keeps the other two answers either side of those', () => {
    expect(memoryRefusal(1.9 * GIB, 1.5 * GIB, 32 * GIB, 8, 0.5 * GIB)).toContain('only 1.5 GB is free')
    expect(memoryRefusal(1.9 * GIB, 6 * GIB, 32 * GIB, 8, 0.5 * GIB)).toBeNull()
  })
})

describe('readings take their turn', () => {
  const pictures = ['a.png', 'b.png', 'c.png']

  beforeEach(() => {
    // A whole tagger on disk, as sparse files of the sizes the server checks.
    for (const [name, bytes] of [['model.onnx', 378536310], ['selected_tags.csv', 308468]] as const) {
      closeSync(openSync(path.join(wd14, name), 'w'))
      truncateSync(path.join(wd14, name), bytes)
    }
    for (const name of pictures) writeFileSync(path.join(outputs, name), 'png')
    writeFileSync(readerLog, '')
    // All the memory there is, so the guard never decides these.
    vi.spyOn(os, 'freemem').mockReturnValue(os.totalmem())
  })

  afterEach(() => {
    vi.restoreAllMocks()
    delete process.env.FAKE_READER_SECONDS
  })

  const tag = (rel: string) =>
    open(h, { method: 'POST', url: '/api/vision/tag', body: { images: [{ kind: 'output', rel }] } })
  const lines = () => readFileSync(readerLog, 'utf8').split('\n').filter(Boolean)
  /** The most readers the log shows alive at once. */
  const mostAtOnce = () => {
    let alive = 0
    let most = 0
    for (const line of lines()) {
      alive += line.startsWith('start') ? 1 : -1
      most = Math.max(most, alive)
    }
    return most
  }

  it('runs one reader at a time, whoever asks', async () => {
    expect((await caps()).tagger).toBe(true)
    const replies = await Promise.all(pictures.map((p) => tag(p).done))
    expect(replies.map((r) => r.status)).toEqual([200, 200, 200])
    expect(lines().filter((l) => l.startsWith('start'))).toHaveLength(3)
    expect(mostAtOnce()).toBe(1)
  }, 20_000)

  it('never starts a reader for a page that went while it waited', async () => {
    process.env.FAKE_READER_SECONDS = '1'
    const first = tag('a.png')
    await vi.waitFor(() => expect(lines()).toHaveLength(1), { timeout: 5000, interval: 20 })
    const second = tag('b.png')
    // Long enough for the second request to reach the line and wait in it.
    // Wherever it is when the page goes, it must never start a reader.
    await new Promise((r) => setTimeout(r, 200))
    second.hangUp()
    expect((await first.done).status).toBe(200)
    await second.done
    expect(lines().filter((l) => l.startsWith('start'))).toHaveLength(1)
  }, 20_000)
})
