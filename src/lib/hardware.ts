/**
 * Hardware awareness.
 *
 * Every number here comes from a real measurement — os.totalmem/freemem,
 * nvidia-smi, and stat() on the actual weight files — never from an estimate
 * baked into the registry. A family's footprint changes when you swap a quant,
 * so it has to be computed from what is on disk right now.
 */
import type { FamilyDef } from './registry'

export type Hardware = {
  cpu: { cores: number; model: string }
  ram: { total: number; free: number }
  gpu: { name: string; vramTotal: number; vramUsed: number; vramFree: number } | null
  disk: { free: number; total: number } | null
  platform: string
}

export type ModelFile = { name: string; rel: string; folder: string; size: number; mtime: number }

/** Headroom for the Python process, CUDA context, activations and the OS. */
const OVERHEAD = 2 * 1024 ** 3

export async function probeHardware(): Promise<Hardware> {
  const r = await fetch('/api/hardware')
  if (!r.ok) throw new Error(`hardware probe failed (HTTP ${r.status})`)
  return r.json()
}

export async function modelFiles(): Promise<Map<string, ModelFile>> {
  const r = await fetch('/api/models')
  if (!r.ok) throw new Error(`model inventory failed (HTTP ${r.status})`)
  const { files } = (await r.json()) as { files: ModelFile[] }
  const byName = new Map<string, ModelFile>()
  for (const f of files) {
    byName.set(f.name, f)
    byName.set(f.rel, f)
    byName.set(f.rel.replace(/\\/g, '/'), f)
  }
  return byName
}

export type Footprint = {
  /** Every weight file the graph loads, and its size. */
  files: { name: string; size: number }[]
  /** Total bytes that must be resident in system RAM. */
  weightBytes: number
  /** Largest single file: the peak VRAM a one-model-at-a-time loader needs. */
  largestBytes: number
  /** weightBytes plus runtime overhead. */
  needBytes: number
  unknown: string[]
}

/**
 * Every weight input a loader node can carry, numbered or not.
 *
 * DualCLIPLoader and TripleCLIPLoader name their encoders clip_name1, clip_name2
 * and clip_name3. A fixed list of the unnumbered keys left both of Hunyuan
 * Video's encoders out of its footprint, about 8.7 GB, so the verdict read 'ok'
 * on a machine that could not hold it. sidecarsOf in workflows.ts made the same
 * mistake and reads the inputs by pattern for the same reason.
 */
const WEIGHT_INPUT = /^(ckpt|unet|clip|vae|lora)_name\d*$/

function footprintOf(graph: FamilyDef['graph'], sizes: Map<string, ModelFile>): Footprint {
  const names = new Set<string>()
  for (const node of Object.values(graph)) {
    for (const [k, v] of Object.entries(node.inputs)) {
      if (WEIGHT_INPUT.test(k) && typeof v === 'string') names.add(v)
    }
  }
  const files: { name: string; size: number }[] = []
  const unknown: string[] = []
  for (const n of names) {
    const hit = sizes.get(n) ?? sizes.get(n.split('/').pop() ?? n)
    if (hit) files.push({ name: n, size: hit.size })
    else unknown.push(n)
  }
  const weightBytes = files.reduce((a, f) => a + f.size, 0)
  return {
    files,
    weightBytes,
    largestBytes: files.reduce((a, f) => Math.max(a, f.size), 0),
    needBytes: weightBytes + OVERHEAD,
    unknown,
  }
}

/**
 * The family's graph with this weight file in it, for pricing one file of a
 * family that lists several.
 *
 * The family's own graph names its default file. Priced as it stands, every
 * other file of the family, a second quant or a different checkpoint of the
 * same lineage, was judged at the default's size, so a Q8 read as fitting on
 * the strength of a Q4 and the other way round. A two-model family carries a
 * matched pair and is priced whole, as instantiate() leaves it.
 */
export function modelGraph(def: FamilyDef, model: string): FamilyDef['graph'] {
  const slots = def.bindings.model ?? []
  if (def.dualModel || !model || !slots.length) return def.graph
  const graph = { ...def.graph }
  for (const [id, input] of slots) {
    const node = graph[id]
    if (node) graph[id] = { ...node, inputs: { ...node.inputs, [input]: model } }
  }
  return graph
}

export type Level = 'ok' | 'tight' | 'risky' | 'blocked'

export type Verdict = {
  level: Level
  /** False only when the machine physically cannot hold the weights. */
  selectable: boolean
  /** Model is larger than VRAM, so ComfyUI will stream it from RAM. Slower, not fatal. */
  offloads: boolean
  reason: string
  footprint: Footprint
}

export const gb = (n: number) => `${(n / 1024 ** 3).toFixed(1)} GB`

/**
 * Whether a family fits this machine, priced from the weight files on disk.
 *
 * `graph` is the graph that will be submitted, when the caller has one built:
 * it names the weight file actually chosen, which may be a different quant
 * from the one the family's own graph names, and every add-on chained in.
 * Without it the family's own graph is priced, which is the default file and
 * no add-ons.
 */
export function feasibility(
  def: FamilyDef,
  sizes: Map<string, ModelFile>,
  hw: Hardware,
  graph: FamilyDef['graph'] = def.graph,
): Verdict {
  const fp = footprintOf(graph, sizes)
  const offloads = !!hw.gpu && fp.largestBytes > hw.gpu.vramTotal

  if (fp.unknown.length && !fp.weightBytes) {
    return { level: 'risky', selectable: true, offloads, footprint: fp,
      reason: `Could not measure ${fp.unknown.join(', ')}. Footprint unknown.` }
  }

  let level: Level = 'ok'
  let reason = `Needs about ${gb(fp.needBytes)} of ${gb(hw.ram.total)} RAM.`

  if (fp.needBytes > hw.ram.total) {
    level = 'blocked'
    reason = `Needs about ${gb(fp.needBytes)} but this machine has only ${gb(hw.ram.total)} of RAM in total. It cannot run here.`
  } else if (fp.needBytes > hw.ram.free) {
    // "When last checked", not "right now": a desk holds on to one reading
    // while the reader works, and free memory moves under it. The sentence
    // claims only the reading it was given.
    level = 'risky'
    reason = `Needs about ${gb(fp.needBytes)} but only ${gb(hw.ram.free)} was free when memory was last checked. Close other applications first, or generation may be killed partway.`
  } else if (fp.needBytes > hw.ram.free * 0.85) {
    level = 'tight'
    reason = `Tight fit: about ${gb(fp.needBytes)} against ${gb(hw.ram.free)} free when memory was last checked.`
  }

  if (offloads && level === 'ok') {
    reason += ` Largest file is ${gb(fp.largestBytes)}, above the ${gb(hw.gpu!.vramTotal)} of VRAM, so it streams from RAM and runs slower.`
  }

  return { level, selectable: level !== 'blocked', offloads, reason, footprint: fp }
}

// ---------------------------------------------------------------------------
// Live device status.
//
// probeHardware() above is a one-shot poll, fine for deciding whether a model
// fits. A status readout needs the moving numbers — CPU load, VRAM in use, GPU
// temperature and clock — which only make sense sampled continuously, so they
// arrive over SSE instead.
// ---------------------------------------------------------------------------

export type GpuLive = {
  name: string
  utilGpu: number | null
  utilMem: number | null
  vramTotal: number
  vramUsed: number
  vramFree: number
  tempC: number | null
  powerW: number | null
  powerLimitW: number | null
  clockMhz: number | null
  fanPct: number | null
}

export type DeviceStatus = {
  t: number
  cpu: { overall: number; perCore: number[]; cores: number; model: string; load: number[] }
  ram: { total: number; free: number; used: number }
  gpu: GpuLive | null
  disk: { free: number; total: number } | null
  uptime: number
}

/**
 * Subscribe to live device status. Returns an unsubscribe function.
 * EventSource reconnects on its own; onError is advisory, not terminal.
 */
export function subscribeDeviceStatus(
  onTick: (s: DeviceStatus) => void,
  opts: { everyMs?: number; onError?: (e: Event) => void } = {},
): () => void {
  const ms = opts.everyMs ?? 1000
  let es: EventSource | null = null
  try {
    es = new EventSource(`/api/hardware/stream?ms=${ms}`)
  } catch {
    return () => {}
  }
  es.onmessage = (ev) => {
    try { onTick(JSON.parse(ev.data) as DeviceStatus) } catch { /* partial frame */ }
  }
  es.onerror = (e) => opts.onError?.(e)
  return () => { try { es?.close() } catch { /* already closed */ } }
}

/** Percentage helper that tolerates the nulls nvidia-smi returns on some fields. */
export const pct = (n: number | null | undefined, digits = 0) =>
  n === null || n === undefined ? '—' : `${(n * (n <= 1 ? 100 : 1)).toFixed(digits)}%`
