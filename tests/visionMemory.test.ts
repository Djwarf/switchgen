import { chmodSync, closeSync, mkdirSync, openSync, readFileSync, rmSync, truncateSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { call, mounted, tempRoots, type Handler } from './http'

/**
 * The memory guard in front of the picture reader, through the route. It is
 * a file of its own because the line earlyoom acts at is read once, when
 * server/vision.mjs loads: here it is set at 99% of RAM, which no machine has
 * free, so every reading is refused before a reader is started. The reader is
 * a shell script that logs any start, so the test can see that none was.
 */
let root = ''
let log = ''
let h: Handler

beforeAll(async () => {
  const roots = tempRoots()
  root = roots.root
  log = path.join(root, 'reader.log')
  writeFileSync(log, '')
  const python = path.join(root, 'python')
  writeFileSync(python, `#!/bin/sh\necho start >> ${JSON.stringify(log)}\nprintf '%s' '{"tag":{"rows":[]},"detect":{"rows":[]}}'\n`)
  chmodSync(python, 0o755)
  process.env.SWITCHGEN_PYTHON = python
  process.env.SWITCHGEN_MEMORY_FLOOR_PERCENT = '99'
  // A whole tagger and one detector on disk, so /inspect is offered at all.
  const wd14 = process.env.SWITCHGEN_WD14!
  mkdirSync(wd14, { recursive: true })
  for (const [name, bytes] of [['model.onnx', 378536310], ['selected_tags.csv', 308468]] as const) {
    closeSync(openSync(path.join(wd14, name), 'w'))
    truncateSync(path.join(wd14, name), bytes)
  }
  const bbox = path.join(roots.models, 'ultralytics', 'bbox')
  mkdirSync(bbox, { recursive: true })
  writeFileSync(path.join(bbox, 'face_yolov8m.pt'), 'weights')
  writeFileSync(path.join(roots.outputs, 'a.png'), 'png')
  h = mounted((await import('../server/vision.mjs')).switchgenVision())
})

afterAll(() => {
  delete process.env.SWITCHGEN_MEMORY_FLOOR_PERCENT
  rmSync(root, { recursive: true, force: true })
})

describe('a reading there is no memory for', () => {
  it('is refused as busy, with the figures, and no reader is started', async () => {
    const r = await call(h, { method: 'POST', url: '/api/vision/inspect', body: { image: { kind: 'output', rel: 'a.png' } } })
    expect(r.status).toBe(503)
    const body = r.json()
    expect(body.busy).toBe('memory')
    expect(body.error).toMatch(/needs about 1\.9 GB/)
    expect(readFileSync(log, 'utf8')).toBe('')
  })
})
