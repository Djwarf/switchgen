import { mkdirSync, rmSync, unlinkSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest'
import { call, mounted, tempRoots, type Handler } from './http'

/**
 * What /api/vision/capabilities says about a tagger that is on disk but not
 * all there. Nothing here runs Python: the interpreter only has to exist, so
 * it is pointed at this Node binary.
 */
let root = ''
let wd14 = ''
let h: Handler

beforeAll(async () => {
  root = tempRoots().root
  wd14 = process.env.SWITCHGEN_WD14!
  process.env.SWITCHGEN_PYTHON = process.execPath
  h = mounted((await import('../server/vision.mjs')).switchgenVision())
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
