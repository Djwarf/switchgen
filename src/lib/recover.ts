/**
 * Filing the files no record describes.
 *
 * A generation made by another front end, by ComfyUI's own interface, or by
 * this app before the archive existed, sits in the outputs folder with no
 * record. GET /api/outputs lists every media file there; the difference
 * against the archive is what needs filing. For each such file, ComfyUI's own
 * /history may still hold the exact graph that produced it, and a family's
 * bindings say which node input held the prompt, the seed, the steps, so the
 * settings come back out of the graph rather than being guessed. When history
 * has forgotten the file (it is in-memory and lost on restart), the record is
 * minimal and says so: a file, a time, and `recovered: true`.
 */
import { pastRuns, readBoundParams, relPath, type ApiWorkflow, type FileRef, type PastRun } from './comfy'
import { history, type HistoryEntry, type NewEntry } from './history'
import { familyForGraph } from './workflows'

export type UnfiledFile = {
  rel: string
  name: string
  subfolder: string
  size: number
  mtime: number
  kind: 'image' | 'video'
}

/** Every media file under the outputs root that no record stands for. */
export async function findUnfiled(): Promise<UnfiledFile[]> {
  const res = await fetch('/api/outputs', { headers: { Accept: 'application/json' } })
  const type = res.headers.get('content-type') ?? ''
  if (!res.ok || !type.includes('json')) throw new Error('the outputs listing is not available here')
  const data = (await res.json()) as { files?: UnfiledFile[] }
  const known = new Set<string>()
  for (const e of history.all()) {
    known.add(relPath(e.file))
    for (const f of e.files ?? []) known.add(relPath(f))
  }
  return (data.files ?? []).filter((f) => !known.has(f.rel))
}

const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback)
const num = (v: unknown, fallback = 0): number => (typeof v === 'number' && Number.isFinite(v) ? v : fallback)
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

function lorasOf(graph: ApiWorkflow): HistoryEntry['loras'] {
  const out: NonNullable<HistoryEntry['loras']> = []
  for (const node of Object.values(graph)) {
    if (node.class_type !== 'LoraLoader' && node.class_type !== 'LoraLoaderModelOnly') continue
    const name = node.inputs['lora_name']
    if (typeof name !== 'string') continue
    out.push({
      name,
      strength: num(node.inputs['strength_model'], 1),
      clipStrength: numOrNull(node.inputs['strength_clip']) ?? undefined,
    })
  }
  return out.length ? out : undefined
}

function loadedModel(graph: ApiWorkflow): string {
  for (const node of Object.values(graph)) {
    for (const k of ['ckpt_name', 'unet_name'] as const) {
      const v = node.inputs[k]
      if (typeof v === 'string') return v
    }
  }
  return ''
}

const stem = (file: string) => file.replace(/\.[^.]+$/, '')

function fileRefOf(f: UnfiledFile): FileRef {
  return { filename: f.name, subfolder: f.subfolder, type: 'output' }
}

/** A record read back out of the graph ComfyUI still remembers. */
function fromRun(run: PastRun, f: UnfiledFile): NewEntry {
  const def = familyForGraph(run.graph)
  const bound = def ? readBoundParams(run.graph, def.bindings) : {}
  const hasImage = Object.values(run.graph).some((n) => n.class_type === 'LoadImage')
  const isVideo = def?.mode === 'video' || f.kind === 'video'
  const mode = def?.mode === 'edit' ? 'edit' : isVideo ? (hasImage ? 'i2v' : 't2v') : hasImage ? 'i2i' : 't2i'
  const model = str(bound.model) || loadedModel(run.graph)
  return {
    at: run.finishedAt ?? run.startedAt ?? f.mtime,
    desk: isVideo ? 'video' : 'images',
    kind: f.kind,
    mode,
    file: fileRefOf(f),
    files: run.files.length > 1 ? run.files : undefined,
    familyId: def?.id ?? 'unknown',
    familyLabel: def?.label ?? 'Unknown',
    variant: null,
    model,
    modelLabel: stem(model),
    prompt: str(bound.positive),
    negative: typeof bound.negative === 'string' ? bound.negative : null,
    seed: num(bound.seed),
    steps: num(bound.steps),
    cfg: num(bound.cfg),
    sampler: str(bound.sampler),
    scheduler: str(bound.scheduler),
    width: numOrNull(bound.width),
    height: numOrNull(bound.height),
    length: numOrNull(bound.length) ?? undefined,
    fps: numOrNull(bound.fps) ?? undefined,
    denoise: numOrNull(bound.denoise) ?? undefined,
    loras: lorasOf(run.graph),
    promptId: run.promptId,
    durationMs: run.finishedAt && run.startedAt ? Math.max(0, run.finishedAt - run.startedAt) : 0,
    recovered: true,
  }
}

/** A record for a file ComfyUI no longer remembers. Enough to browse; nothing invented. */
function minimal(f: UnfiledFile): NewEntry {
  return {
    at: f.mtime,
    desk: f.kind === 'video' ? 'video' : 'images',
    kind: f.kind,
    mode: f.kind === 'video' ? 't2v' : 't2i',
    file: fileRefOf(f),
    familyId: 'unknown',
    familyLabel: 'Unknown',
    variant: null,
    model: '',
    modelLabel: '',
    prompt: '',
    negative: null,
    seed: 0,
    steps: 0,
    cfg: 0,
    sampler: '',
    scheduler: '',
    width: null,
    height: null,
    promptId: '',
    durationMs: 0,
    recovered: true,
  }
}

let running: Promise<{ filed: number; fromHistory: number }> | null = null

/**
 * File every unfiled output. Safe to call twice; the second call waits for the
 * first. Returns how many records were made and how many came with settings.
 */
export function recoverUnfiled(): Promise<{ filed: number; fromHistory: number }> {
  if (running) return running
  running = (async () => {
    const unfiled = await findUnfiled()
    if (!unfiled.length) return { filed: 0, fromHistory: 0 }
    let runs: PastRun[] = []
    try { runs = await pastRuns(1000) } catch { /* ComfyUI is down or has forgotten; file minimally */ }
    const byFile = new Map<string, PastRun>()
    for (const run of runs) for (const f of run.files) byFile.set(relPath(f), run)
    let fromHistory = 0
    // Oldest first, so edition numbers follow the order the files were made.
    for (const f of [...unfiled].sort((a, b) => a.mtime - b.mtime)) {
      const run = byFile.get(f.rel)
      history.add(run ? fromRun(run, f) : minimal(f))
      if (run) fromHistory++
    }
    return { filed: unfiled.length, fromHistory }
  })().finally(() => { running = null })
  return running
}
