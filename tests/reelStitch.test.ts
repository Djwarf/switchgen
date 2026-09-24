import { chmodSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest'
import { call, mounted, tempRoots, type Handler } from './http'

/**
 * Stitching a reel, the Reel room's export (POST /api/reel/stitch), against a
 * fake ffprobe that answers from a table of clips and a fake ffmpeg that logs
 * what it was asked and writes a stand-in file, or fails when told to. CI has
 * no ffmpeg, and nothing here needs one: what is checked is the choice between
 * copying and encoding, the concat list, the refusals and the naming.
 */
let root = ''
let outputs = ''
let log = ''
let failFlag = ''
let h: Handler

/** What the fake ffprobe says of each clip, by file name. */
const clip = (over: Record<string, unknown> = {}) => ({
  codec_type: 'video', codec_name: 'vp9', profile: 'Profile 0', width: 832, height: 480, pix_fmt: 'yuv420p',
  color_range: 'tv', r_frame_rate: '16/1', avg_frame_rate: '16/1', nb_read_packets: '81', duration: '5.0625', ...over,
})
const TABLE: Record<string, Record<string, unknown>> = {
  'a.webm': clip(),
  'b.webm': clip(),
  "it's.webm": clip(),
  'fast.webm': clip({ r_frame_rate: '24/1', avg_frame_rate: '24/1' }),
  'deep.webm': clip({ pix_fmt: 'yuv444p' }),
}

const FAKE_FFPROBE = (table: string) => `#!${process.execPath}
const fs = require('node:fs')
const path = require('node:path')
const table = JSON.parse(fs.readFileSync(${JSON.stringify(table)}, 'utf8'))
const file = process.argv[process.argv.length - 1]
const v = table[path.basename(file)] ?? table['a.webm']
process.stdout.write(JSON.stringify({ streams: [v], format: { duration: v.duration, size: '1000', format_name: 'matroska,webm' } }))
`
const FAKE_FFMPEG = (logFile: string, flag: string) => `#!${process.execPath}
const fs = require('node:fs')
const args = process.argv.slice(2)
const entry = { args }
const f = args.indexOf('-f')
if (f >= 0 && args[f + 1] === 'concat') entry.list = fs.readFileSync(args[args.indexOf('-i') + 1], 'utf8')
fs.appendFileSync(${JSON.stringify(logFile)}, JSON.stringify(entry) + '\\n')
const out = args[args.length - 1]
fs.writeFileSync(out, 'part of a reel')
if (fs.existsSync(${JSON.stringify(flag)})) { process.stderr.write('Conversion failed!\\n'); process.exit(1) }
process.stdout.write('out_time_us=1000000\\nprogress=end\\n')
`

beforeAll(async () => {
  const roots = tempRoots()
  root = roots.root
  outputs = roots.outputs
  log = path.join(root, 'ffmpeg.log')
  failFlag = path.join(root, 'ffmpeg-fails')
  const table = path.join(root, 'clips.json')
  writeFileSync(table, JSON.stringify(TABLE))
  const bin = path.join(root, 'bin')
  mkdirSync(bin)
  for (const [name, body] of [['ffprobe', FAKE_FFPROBE(table)], ['ffmpeg', FAKE_FFMPEG(log, failFlag)]] as const) {
    writeFileSync(path.join(bin, name), body)
    chmodSync(path.join(bin, name), 0o755)
  }
  process.env.SWITCHGEN_FFPROBE = path.join(bin, 'ffprobe')
  process.env.SWITCHGEN_FFMPEG = path.join(bin, 'ffmpeg')
  mkdirSync(path.join(outputs, 'shots'), { recursive: true })
  for (const name of Object.keys(TABLE)) writeFileSync(path.join(outputs, 'shots', name), 'clip')
  h = mounted((await import('../server/reel.mjs')).switchgenReel())
})

afterAll(() => rmSync(root, { recursive: true, force: true }))

beforeEach(() => {
  writeFileSync(log, '')
  rmSync(failFlag, { force: true })
  for (const f of readdirSync(path.join(outputs, 'shots'))) if (f.startsWith('reel_')) rmSync(path.join(outputs, 'shots', f))
})

/**
 * One stitch, answered as JSON. Sent the moment the last one answered, with
 * no wait between: the server lets go of the assembly slot as its answer goes
 * out (see 'one stitch straight after another' below).
 */
const stitch = (body: Record<string, unknown>) => call(h, { method: 'POST', url: '/api/reel/stitch?json=1', body })
const shots = (...names: string[]) => names.map((n) => `shots/${n}`)
const ran = () => readFileSync(log, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l) as { args: string[]; list?: string })

describe('a reel of clips that agree', () => {
  it('is copied, not encoded, in the order given', async () => {
    const r = await stitch({ clips: shots('b.webm', 'a.webm') })
    expect(r.status).toBe(200)
    expect(r.json().mode).toBe('copy')
    expect(r.json().reasons).toEqual([])
    const [{ list }] = ran()
    expect(list!.split('\n').filter(Boolean)).toEqual([
      `file '${path.join(outputs, 'shots', 'b.webm')}'`,
      `file '${path.join(outputs, 'shots', 'a.webm')}'`,
    ])
  })

  it('quotes an apostrophe in a clip\'s name the way the concat list reads it', async () => {
    const r = await stitch({ clips: shots('a.webm', "it's.webm") })
    expect(r.status).toBe(200)
    const [{ list }] = ran()
    expect(list).toContain(`file '${path.join(outputs, 'shots', 'it')}'\\''s.webm'`)
  })

  it('never writes over an earlier reel', async () => {
    const first = (await stitch({ clips: shots('a.webm', 'b.webm') })).json()
    const second = (await stitch({ clips: shots('a.webm', 'b.webm') })).json()
    expect(first.out).toBe(path.join('shots', 'reel_00001_.webm'))
    expect(second.out).toBe(path.join('shots', 'reel_00002_.webm'))
  })
})

describe('one stitch straight after another', () => {
  // The slot used to be let go only after the answer, once the concat list
  // was removed from disk, so a second export pressed the moment the first
  // answered was refused as another reel being assembled.
  it('takes a second stitch sent the moment the first answers', async () => {
    const one = await stitch({ clips: shots('a.webm', 'b.webm') })
    expect(one.status).toBe(200)
    const two = await stitch({ clips: shots('a.webm', 'b.webm') })
    expect(two.status, two.body).toBe(200)
    expect(two.json().out).not.toBe(one.json().out)
  })

  it('does the same after a stitch that streamed its progress', async () => {
    const streamed = await call(h, { method: 'POST', url: '/api/reel/stitch', body: { clips: shots('a.webm', 'b.webm') } })
    expect(streamed.status).toBe(200)
    expect(streamed.body).toContain('event: done')
    const next = await stitch({ clips: shots('a.webm', 'b.webm') })
    expect(next.status, next.body).toBe(200)
  })

  it('does the same after a stitch whose ffmpeg failed', async () => {
    writeFileSync(failFlag, '')
    expect((await stitch({ clips: shots('a.webm', 'fast.webm') })).status).toBe(500)
    rmSync(failFlag, { force: true })
    const next = await stitch({ clips: shots('a.webm', 'b.webm') })
    expect(next.status, next.body).toBe(200)
  })
})

describe('a reel that has to be encoded', () => {
  it('names the clip whose frame rate differs', async () => {
    const r = (await stitch({ clips: shots('a.webm', 'fast.webm') })).json()
    expect(r.mode).toBe('encode')
    expect(r.reasons.join(' ')).toContain('fast.webm: frame rate 24/1 against 16/1 in a.webm')
    expect(ran()[0]!.args).toContain('-filter_complex')
  })

  it('names the clip whose pixel format differs', async () => {
    const r = (await stitch({ clips: shots('a.webm', 'deep.webm') })).json()
    expect(r.mode).toBe('encode')
    expect(r.reasons.join(' ')).toContain('deep.webm: pixel format yuv444p against yuv420p in a.webm')
  })

  it('crossfades when asked, though the clips agree', async () => {
    const r = (await stitch({ clips: shots('a.webm', 'b.webm'), crossfade: 0.5 })).json()
    expect(r.mode).toBe('crossfade')
    expect(r.crossfade).toBe(0.5)
    expect(ran()[0]!.args.join(' ')).toContain('xfade=transition=fade:duration=0.5')
  })

  it('leaves no half-written reel behind when ffmpeg fails', async () => {
    writeFileSync(failFlag, '')
    const r = await stitch({ clips: shots('a.webm', 'fast.webm') })
    expect(r.status).toBe(500)
    expect(r.json().error).toMatch(/ffmpeg exited 1/)
    expect(existsSync(path.join(outputs, r.json().out))).toBe(false)
  })
})

describe('a clip name that is not a clip in the outputs folder', () => {
  it('is refused before anything runs when it holds a newline', async () => {
    const r = await stitch({ clips: ['shots/a.webm', "shots/a.webm\nfile '/etc/passwd'\n.webm"] })
    expect(r.status).toBe(400)
    expect(r.json().error).toMatch(/control characters/)
    expect(ran()).toEqual([])
  })

  it('is refused when it leads out of the outputs folder', async () => {
    const r = await stitch({ clips: ['shots/a.webm', '../../elsewhere.webm'] })
    expect(r.status).toBe(400)
    expect(r.json().error).toMatch(/escapes the outputs root/)
    expect(ran()).toEqual([])
  })
})
