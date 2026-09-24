import { afterEach, describe, expect, it, vi } from 'vitest'
import * as server from '../server/runner/comfyRecord.mjs'
import * as client from '../src/lib/comfy'
import * as cont from '../src/lib/continuation'
import * as runnerLib from '../src/lib/runner'
import { jobs as ledger } from '../src/components/shell/jobs'
import { FAMILIES, defaultsFor, type Params } from '../src/lib/workflows'
import type { ShotSpec } from '../src/lib/continuation'

/**
 * The queue on the server reads ComfyUI's answers and builds reel graphs the
 * way the page does, because it files what the page used to file and sends
 * what the page used to send. The same /history entries go through both
 * readers here, and the frame the server puts into a chained shot is checked
 * against the graph the page would have built with that frame in hand.
 */

afterEach(() => {
  vi.unstubAllGlobals()
})

const twoPass = {
  '1': { class_type: 'KSamplerAdvanced', inputs: { return_with_leftover_noise: 'enable' } },
  '2': { class_type: 'KSamplerAdvanced', inputs: { return_with_leftover_noise: 'disable' } },
  '3': { class_type: 'KSampler', inputs: {} },
}

const fixtures: Record<string, any> = {
  // A reel shot: the clip, the tap, a poster frame answered from the cache.
  video: {
    prompt: [1, 'video', twoPass, { create_time: 1000, client_id: 'c' }, []],
    outputs: {
      '9': { images: [{ filename: 'reel_001.webm', subfolder: 'switchgen/reel', type: 'output' }], animated: [true] },
      __cont_frame: { images: [{ filename: 'reel_001.frame_00001_.png', subfolder: 'switchgen/reel', type: 'output' }] },
      '12': { images: [{ filename: 'poster.png', subfolder: '', type: 'output' }] },
    },
    status: {
      status_str: 'success',
      completed: true,
      messages: [
        ['execution_start', { prompt_id: 'video', timestamp: 2000 }],
        ['execution_cached', { nodes: ['12'], prompt_id: 'video', timestamp: 2001 }],
        ['execution_success', { prompt_id: 'video', timestamp: 9000 }],
      ],
    },
  },
  // No execution_start: the page falls back to the time it was queued.
  noStart: {
    prompt: [1, 'noStart', twoPass, { create_time: 1000 }, []],
    outputs: {
      '9': { gifs: [{ filename: 'a.mp4', subfolder: 'x', type: 'output' }] },
      '4': { images: [{ filename: 'chain_b.png', subfolder: 'x', type: 'output' }, { filename: 'c.png', subfolder: 'x', type: 'temp' }] },
    },
    status: { status_str: 'success', messages: [['execution_success', { timestamp: 5000 }]] },
  },
  failed: {
    prompt: [1, 'failed', twoPass, {}, []],
    outputs: {},
    status: {
      status_str: 'error',
      messages: [
        ['execution_start', { timestamp: 10 }],
        ['execution_error', { timestamp: 20, node_id: 7, node_type: 'VAEDecode', exception_message: 'Allocation on device', exception_type: 'OutOfMemoryError' }],
      ],
    },
  },
  // Interrupted is 'cancelled' though ComfyUI files it as an error.
  stopped: {
    prompt: [1, 'stopped', twoPass, {}, []],
    outputs: {},
    status: { status_str: 'error', messages: [['execution_start', { timestamp: 10 }], ['execution_interrupted', { timestamp: 30, node_id: '1', node_type: 'KSamplerAdvanced' }]] },
  },
  cachedAll: {
    prompt: [1, 'cachedAll', twoPass, {}, []],
    outputs: { '9': { images: [{ filename: 'x.png', subfolder: '', type: 'output' }] }, __cont_frame: { images: [{ filename: 'x.frame.png', subfolder: '', type: 'output' }] } },
    status: { status_str: 'success', messages: [['execution_cached', { nodes: ['9', '__cont_frame'] }], ['execution_success', { timestamp: 3 }]] },
  },
  // An animated PNG: a clip by its flag alone.
  apng: {
    prompt: [1, 'apng', twoPass, {}, []],
    outputs: { '9': { images: [{ filename: 'loop_00001_.png', subfolder: '', type: 'output' }], animated: [true] } },
    status: { status_str: 'success', messages: [['execution_success', { timestamp: 4 }]] },
  },
  webp: {
    prompt: [1, 'webp', twoPass, {}, []],
    outputs: { '9': { images: [{ filename: 'loop.webp', subfolder: '', type: 'output' }, { filename: 'still.png', subfolder: '', type: 'output' }], animated: [false, false] } },
    status: { status_str: 'success', messages: [['execution_success', { timestamp: 4 }]] },
  },
  running: { prompt: [1, 'running', twoPass, {}, []], outputs: {}, status: { status_str: null, messages: [] } },
  noGraph: { prompt: [1, 'noGraph'], outputs: {}, status: {} },
  odd: { prompt: 'nope' },
}

describe('a finished run, read on the server and in the page', () => {
  it('gives the same files, cached marks, status, end and error; the start differs only where ComfyUI kept none', async () => {
    vi.stubGlobal('fetch', async (u: string) => {
      const id = decodeURIComponent(String(u).split('/').pop()!)
      return new Response(JSON.stringify({ [id]: fixtures[id] }), { status: 200, headers: { 'content-type': 'application/json' } })
    })
    for (const [id, raw] of Object.entries(fixtures)) {
      const page = await client.fetchPastRun(id)
      const ours = server.readPastRun(id, raw)
      if (!page) {
        expect(ours, id).toBeNull()
        continue
      }
      expect(ours, id).not.toBeNull()
      expect(ours!.files, id).toEqual(page.files)
      expect(ours!.status, id).toBe(page.status)
      expect(ours!.finishedAt, id).toBe(page.finishedAt)
      expect(ours!.error, id).toEqual(page.error)
      if (id === 'noStart') {
        expect(page.startedAt).toBe(1000)
        expect(ours!.startedAt).toBeNull()
      } else {
        expect(ours!.startedAt, id).toBe(page.startedAt)
      }
      expect(server.chainFrameOf(ours!.files), id).toEqual(cont.chainFrameOf(page.files))
      for (const f of ours!.files) {
        expect(server.annotatedRef(f)).toBe(cont.annotatedRef(f))
        expect(server.relOf(f)).toBe(client.relPath(f))
      }
    }
  })

  it('takes the frame from the tap, over a poster still and over a later still named like one', () => {
    const run = server.readPastRun('video', fixtures.video)!
    expect(run.frame).toEqual({ filename: 'reel_001.frame_00001_.png', subfolder: 'switchgen/reel', type: 'output', kind: 'image' })
    // A tap whose file is not named like one, and a still named like one after it.
    const later = structuredClone(fixtures.video)
    later.outputs.__cont_frame = { images: [{ filename: 'ComfyUI_00007_.png', subfolder: 'switchgen/reel', type: 'output' }] }
    later.outputs.poster_after = { images: [{ filename: 'chain_late.png', subfolder: '', type: 'output' }] }
    expect(server.readPastRun('video', later)!.frame!.filename).toBe('ComfyUI_00007_.png')
    // With no tap, the page's own filename rule.
    expect(server.readPastRun('noStart', fixtures.noStart)!.frame!.filename).toBe('chain_b.png')
    expect(server.readPastRun('failed', fixtures.failed)!.frame).toBeNull()
  })

  it('takes the last still where none is named like a frame, as the page does', () => {
    const still = (filename: string) => ({ filename, subfolder: '', type: 'output', kind: 'image' as const })
    const clip = { filename: 'a.webm', subfolder: '', type: 'output', kind: 'video' as const }
    for (const files of [[still('a.png'), still('b.png')], [clip, still('x.png'), still('y.png')], [still('chain_1.png'), still('z.png'), still('chain_2.png')], [clip]]) {
      expect(server.chainFrameOf(files)).toEqual(cont.chainFrameOf(files))
    }
    expect(server.chainFrameOf([still('a.png'), still('b.png')])!.filename).toBe('b.png')
    expect(server.readPastRun('apng', fixtures.apng)!.files[0]!.kind).toBe('video')
  })

  it('marks a tap answered from the cache as cached', () => {
    const run = server.readPastRun('cachedAll', fixtures.cachedAll)!
    expect(run.frame).toMatchObject({ filename: 'x.frame.png', cached: true })
    expect(run.files.every((f) => f.cached === true)).toBe(true)
    expect(server.readPastRun('video', fixtures.video)!.files.find((f) => f.filename === 'reel_001.webm')!.cached).toBeUndefined()
  })

  it('names the failing node as a string, with its class', () => {
    expect(server.readPastRun('failed', fixtures.failed)!.error).toEqual({ message: 'Allocation on device', node: '7', nodeType: 'VAEDecode' })
    expect(server.readPastRun('stopped', fixtures.stopped)).toMatchObject({ status: 'cancelled', error: null, finishedAt: 30 })
    expect(server.readPastRun('running', fixtures.running)!.status).toBe('unknown')
  })
})

describe('the sampling pass', () => {
  const graphs: Record<string, any> = { twoPass, onePass: { a: { class_type: 'KSamplerAdvanced', inputs: { return_with_leftover_noise: 'enable' } } } }

  it('is read the same on the server, in the reel and in the ledger', () => {
    for (const [name, graph] of Object.entries(graphs)) {
      for (const node of [...Object.keys(graph), 'missing', null]) {
        const ours = server.samplerPass(graph, node)
        expect(ours, `${name} ${node}`).toEqual(cont.samplerPass(graph, node))
        // The ledger's own copy, through the one door it has.
        const id = ledger.start({ desk: 'video', kind: 'video', label: 'x', prompt: '' })
        ledger.apply(id, { phase: 'running', node, value: 2, max: 10 }, { graph })
        expect(ledger.get(id)!.pass, `${name} ${node} ledger`).toEqual(ours)
      }
    }
    expect(server.samplerPass(twoPass as any, '1')).toEqual({ index: 1, count: 2 })
    expect(server.samplerPass(twoPass as any, '2')).toEqual({ index: 2, count: 2 })
  })

  it('makes two advanced samplers heavy whatever the page says, and one or none not', () => {
    expect(server.heavyFloor(twoPass as any)).toBe(true)
    expect(server.heavyFloor(graphs.onePass)).toBe(false)
    expect(server.heavyFloor({ a: { class_type: 'KSampler', inputs: {} } } as any)).toBe(false)
  })
})

describe('the frame\'s place in a chained shot', () => {
  const T = server.CHAIN_TOKEN
  const graph = () => ({
    '5': { class_type: 'LoadImage', inputs: { image: T, upload: 'image' } },
    '6': { class_type: 'X', inputs: { a: ['5', 0], image: T } },
  })

  it('is spelled the same on both sides', () => {
    expect(runnerLib.CHAIN_TOKEN).toBe(server.CHAIN_TOKEN)
    expect(runnerLib.FRAME_NODE).toBe(server.FRAME_NODE)
    expect(runnerLib.tokenSites(graph() as any)).toEqual(server.tokenSites(graph() as any))
  })

  it('is found in the graph\'s own order, and anywhere at all by hasToken', () => {
    expect(server.tokenSites(graph() as any)).toEqual([['5', 'image'], ['6', 'image']])
    expect(server.hasToken({ a: { inputs: { deep: { x: [T] } } } })).toBe(true)
    expect(server.hasToken({ a: { inputs: { t: `before ${T}` } } })).toBe(true)
    expect(server.hasToken({ a: { inputs: { t: 'plain' } } })).toBe(false)
  })

  it('is filled in a copy, leaving the saved graph as it was', () => {
    const g = graph()
    const out = server.splice(g as any, server.tokenSites(g as any), 'switchgen/chain/f.png [output]')
    expect(out['5']!.inputs.image).toBe('switchgen/chain/f.png [output]')
    expect(out['6']!.inputs.image).toBe('switchgen/chain/f.png [output]')
    expect(g['5'].inputs.image).toBe(T)
    // A place named twice does no harm.
    expect(server.splice(g as any, [['5', 'image'], ['5', 'image'], ['6', 'image']], 'r')['5']!.inputs.image).toBe('r')
  })

  it('refuses a place without it, a token left anywhere, and an empty frame', () => {
    const g = graph()
    expect(() => server.splice(g as any, [['5', 'image']], 'r')).toThrow(/still holds/)
    expect(() => server.splice(g as any, [['5', 'upload'], ['6', 'image']], 'r')).toThrow(/does not hold/)
    expect(() => server.splice(g as any, [['9', 'image']], 'r')).toThrow(/does not hold/)
    expect(() => server.splice({ a: { class_type: 'X', inputs: { t: `x ${T}` } } } as any, [], 'r')).toThrow(/still holds/)
    expect(() => server.splice(g as any, server.tokenSites(g as any), '')).toThrow(/No frame/)
    expect(() => server.splice(g as any, 'nope' as any, 'r')).toThrow(/not a list/)
    expect(g['5'].inputs.image).toBe(T)
  })

  it('once filled, is the graph the page would have built with the frame in hand', () => {
    const frame = { filename: 'reel_001.frame_00001_.png', subfolder: 'switchgen/reel', type: 'output', kind: 'image' as const }
    const checked: string[] = []
    for (const def of FAMILIES) {
      const model = def.dualModel ? '' : def.models[0]!
      const d = defaultsFor(def, model)
      const params = { model, positive: '', negative: d.negative ?? '', seed: 7, steps: d.steps, cfg: d.cfg, width: d.width, height: d.height, sampler: d.sampler, scheduler: d.scheduler, length: d.length || undefined, fps: d.fps || undefined } as Params
      const seconds: [string, ShotSpec][] = [
        ['plain', { prompt: 'b' }],
        ['bookend', { prompt: 'b', endImage: 'end.png' }],
        ['vace', { prompt: 'b', referenceImage: 'ref.png' }],
      ]
      for (const [variant, second] of seconds) {
        const job = cont.shotPlan({ base: def, params, shots: [{ prompt: 'a' }, second] }).jobs[1]
        if (!job || job.start.from !== 'previous') continue
        const withToken = cont.instantiateShot(job, server.CHAIN_TOKEN)
        const sites = server.tokenSites(withToken as any)
        expect(sites.length, `${def.id} ${variant}`).toBeGreaterThan(0)
        expect(server.splice(withToken as any, sites, server.annotatedRef(frame)), `${def.id} ${variant}`).toEqual(cont.instantiateShot(job, frame))
        checked.push(`${job.def.id}`)
      }
    }
    // The 5B continuation, the 14B image-to-video continuation and its bookend.
    // (No VACE family opens a shot on the one before, so there is none to check.)
    expect(new Set(checked)).toEqual(new Set(['wan22-5b__continue', 'wan22-14b-i2v__continue', 'wan22-14b-i2v__bookend']))
  })
})
