import { afterEach, describe, expect, it, vi } from 'vitest'
import { clipMemory, waitForIdleComfy } from '../src/lib/clipMemory'
import { feasibility, type Hardware, type ModelFile } from '../src/lib/hardware'
import { withVideoLoras } from '../src/lib/refine'
import { FAMILIES } from '../src/lib/workflows'

const GiB = 1024 ** 3
const family = (id: string) => FAMILIES.find((f) => f.id === id)!

function machine(ramGiB: number): Hardware {
  return {
    cpu: { cores: 8, model: 'test' },
    ram: { total: ramGiB * GiB, free: ramGiB * GiB },
    gpu: { name: 'test', vramTotal: 16 * GiB, vramUsed: 0, vramFree: 16 * GiB },
    disk: null,
    platform: 'test',
  }
}

/** Roughly the 32 GB machine the two points were measured on, as the OS reports it. */
const MEASURED_MACHINE = machine(31)
const at = (frames: number, width = 832, height = 480) => ({ width, height, frames })

describe('clipMemory: the 14B pairs against the two measured points', () => {
  const pair = family('wan22-14b-t2v')

  it('lets the size that survived through, and still frees memory first', () => {
    expect(clipMemory(pair, at(49), MEASURED_MACHINE)).toEqual({ level: 'ok', reason: null, release: true })
  })

  it('cautions between the two points, up to and including the edge', () => {
    for (const frames of [53, 81]) {
      const v = clipMemory(pair, at(frames), MEASURED_MACHINE)
      expect(v.level).toBe('caution')
      expect(v.release).toBe(true)
      expect(v.reason).toContain('49 frames at 832 × 480')
    }
  })

  it('refuses above the measured edge, with or without a memory reading', () => {
    for (const hw of [MEASURED_MACHINE, null]) {
      const v = clipMemory(pair, at(85), hw)
      expect(v.level).toBe('refuse')
      expect(v.reason).toMatch(/^Too large for memory\./)
      expect(v.reason).toContain('28.1 GB')
    }
    // Size is frames times pixels, so a larger frame crosses the edge sooner.
    expect(clipMemory(pair, at(81, 1280, 720), MEASURED_MACHINE).level).toBe('refuse')
  })

  it('refuses nothing on a machine with clearly more memory, which nobody measured', () => {
    const roomy = machine(64)
    const v = clipMemory(pair, at(121), roomy)
    expect(v.level).toBe('caution')
    expect(v.reason).toContain('which was not measured')
    expect(clipMemory(pair, at(81, 1280, 720), roomy).level).not.toBe('refuse')
  })

  it('leaves every one-model family alone', () => {
    for (const id of ['wan22-5b', 'hunyuan-video', 'ltxv-0_9_6-gguf']) {
      expect(clipMemory(family(id), at(241, 1280, 720), MEASURED_MACHINE)).toEqual({ level: 'ok', reason: null, release: false })
    }
  })
})

describe('clipMemory: the image-to-video pair against its own record', () => {
  const pair = family('wan22-14b-i2v')

  it('refuses anything larger than the 49 frames that came through once', () => {
    for (const frames of [81, 61]) expect(clipMemory(pair, at(frames), MEASURED_MACHINE).level).toBe('refuse')
  })

  it('never simply lets a clip through, and points to the 5B instead of promising a release will help', () => {
    for (const frames of [49, 33]) {
      const v = clipMemory(pair, at(frames), MEASURED_MACHINE)
      expect(v.level).toBe('caution')
      expect(v.release).toBe(true)
      expect(v.reason).toContain('5B')
      expect(v.reason).not.toContain('released before it runs')
    }
  })

  it('refuses the size that came through once when add-ons are chained on top', () => {
    expect(clipMemory(pair, at(49), MEASURED_MACHINE, 1).level).toBe('refuse')
  })

  it('claims no count of runs the record does not give, and names the size as the pair\'s default', () => {
    // The registry lists one result per setting, by length and add-on count,
    // and gives no size for them.
    for (const frames of [81, 49]) {
      for (const addOns of [0, 1]) {
        for (const ram of [31, 64]) {
          const reason = clipMemory(pair, at(frames), machine(ram), addOns).reason ?? ''
          expect(reason, `${frames} frames, ${addOns} add-ons, ${ram} GB`).not.toMatch(/three times|out of three/)
        }
      }
    }
    const refused = clipMemory(pair, at(81), MEASURED_MACHINE).reason!
    expect(refused).toContain('killed at 81 frames')
    expect(refused).toContain("the pair's default size")
    expect(refused).not.toContain('81 frames at 832 × 480 was killed')
  })
})

describe('clipMemory: add-ons on the text-to-video pair', () => {
  const pair = family('wan22-14b-t2v')

  it('makes the verdict one step stricter, since the pair was measured with none', () => {
    expect(clipMemory(pair, at(49), MEASURED_MACHINE, 1).level).toBe('caution')
    expect(clipMemory(pair, at(61), MEASURED_MACHINE, 1).level).toBe('refuse')
    expect(clipMemory(pair, at(61), MEASURED_MACHINE).level).toBe('caution')
  })

  it('refuses nothing with add-ons on a machine nobody measured', () => {
    const v = clipMemory(pair, at(61), machine(64), 1)
    expect(v.level).toBe('caution')
    expect(v.reason).toContain('which was not measured')
  })

  it('does not claim the release makes room for the clip', () => {
    expect(clipMemory(pair, at(81), MEASURED_MACHINE).reason).not.toContain("so ComfyUI's cached models are released before it runs")
  })

  it('holds a family with no id to the text-to-video points', () => {
    const bare = { dualModel: true }
    for (const [frames, addOns] of [[49, 0], [61, 0], [85, 0], [49, 1], [61, 1]] as const) {
      expect(clipMemory(bare, at(frames), MEASURED_MACHINE, addOns)).toEqual(clipMemory(pair, at(frames), MEASURED_MACHINE, addOns))
    }
  })
})

describe('waiting for an idle ComfyUI before a release', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })
  const queue = (running: number, pending: number) =>
    new Response(JSON.stringify({ queue_running: Array(running).fill([]), queue_pending: Array(pending).fill([]) }), {
      headers: { 'content-type': 'application/json' },
    })

  it('says how many jobs are ahead, and goes once the queue is empty', async () => {
    vi.useFakeTimers()
    const answers = [queue(1, 1), queue(0, 1), queue(0, 0)]
    vi.stubGlobal('fetch', vi.fn(async () => answers.shift()!))
    const heard: number[] = []
    const idle = waitForIdleComfy(undefined, (n) => heard.push(n))
    await vi.advanceTimersByTimeAsync(4000)
    expect(await idle).toBe(true)
    expect(heard).toEqual([2, 1])
  })

  it('gives up when stopped while it waits', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn(async () => queue(1, 0)))
    const stop = new AbortController()
    const idle = waitForIdleComfy(stop.signal)
    await vi.advanceTimersByTimeAsync(100)
    stop.abort()
    expect(await idle).toBe(false)
  })

  it('notices a stop that lands while the queue is being read, without sleeping first', async () => {
    vi.useFakeTimers()
    let answer: (r: Response) => void = () => {}
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((resolve) => { answer = resolve })))
    const stop = new AbortController()
    const heard: number[] = []
    let settled: boolean | null = null
    void waitForIdleComfy(stop.signal, (n) => heard.push(n)).then((v) => { settled = v })
    await vi.advanceTimersByTimeAsync(0)
    stop.abort()
    answer(queue(1, 0))
    await vi.advanceTimersByTimeAsync(10)
    expect(settled).toBe(false)
    expect(heard).toEqual([])
  })
})

describe('feasibility: what a family costs in memory', () => {
  /** Every weight file a graph names, each given `each` bytes. */
  function sizesFor(graph: Record<string, { inputs: Record<string, unknown> }>, each: number): Map<string, ModelFile> {
    const sizes = new Map<string, ModelFile>()
    for (const n of Object.values(graph)) {
      for (const [k, v] of Object.entries(n.inputs)) {
        if (/^(ckpt|unet|clip|vae|lora)_name\d*$/.test(k) && typeof v === 'string') sizes.set(v, { name: v, rel: v, folder: 'x', size: each, mtime: 0 })
      }
    }
    return sizes
  }

  it('counts both of a DualCLIPLoader family\'s text encoders', () => {
    const def = family('hunyuan-video')
    const dual = Object.values(def.graph).find((n) => n.class_type === 'DualCLIPLoader')
    expect(dual).toBeDefined()
    const encoders = [dual!.inputs.clip_name1, dual!.inputs.clip_name2] as string[]
    expect(encoders.every((e) => typeof e === 'string' && e.length > 0)).toBe(true)

    const sizes = sizesFor(def.graph, GiB)
    const v = feasibility(def, sizes, machine(64))
    const counted = v.footprint.files.map((f) => f.name)
    for (const e of encoders) expect(counted).toContain(e)
    expect(v.footprint.weightBytes).toBe(sizes.size * GiB)
    expect(v.footprint.unknown).toEqual([])
  })

  it('prices the graph that will be sent, add-ons included', () => {
    const def = family('wan22-5b')
    const sizes = sizesFor(def.graph, GiB)
    sizes.set('extra.safetensors', { name: 'extra.safetensors', rel: 'Lora/extra.safetensors', folder: 'Lora', size: GiB / 4, mtime: 0 })
    const plain = feasibility(def, sizes, machine(64))
    const chained = withVideoLoras(def, [{ name: 'extra.safetensors', strength: 0.8 }])!
    const withAddOn = feasibility(def, sizes, machine(64), chained.graph)
    expect(withAddOn.footprint.weightBytes - plain.footprint.weightBytes).toBe(GiB / 4)
  })

  it('blocks a family the machine cannot hold at all', () => {
    const def = family('hunyuan-video')
    const v = feasibility(def, sizesFor(def.graph, 8 * GiB), machine(16))
    expect(v.level).toBe('blocked')
    expect(v.selectable).toBe(false)
  })
})
