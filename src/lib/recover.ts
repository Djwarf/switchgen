/**
 * Filing the files no record describes.
 *
 * A generation made by another front end, by ComfyUI's own interface, or by
 * this app before the archive existed, sits in the outputs folder with no
 * record. GET /api/outputs lists every media file there and marks the ones a
 * record on the server already names; what is left, less what this browser
 * holds but has not yet pushed, is what needs filing. The server's mark is the
 * one that matters: this browser keeps only a window of the archive, and a
 * file whose record fell out of that window is filed already, not unfiled.
 *
 * A file whose record the reader removed is not unfiled: removing a record
 * keeps its file on disk by design, and filing it again on the next page load
 * would undo the removal. The server marks those, and only a reader asking for
 * them by name brings them back. A record filed for one of them says so
 * (`refiled`), because the server refuses any other.
 *
 * For each file that really has no record, ComfyUI's own /history may still
 * hold the exact graph that produced it, and a family's bindings say which
 * node input held the prompt, the seed, the steps, so the settings come back
 * out of the graph rather than being guessed. When history has forgotten the
 * file (it is in-memory and lost on restart), the record is minimal and says
 * so: a file, a time, and `recovered: true`.
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
  /** A record on the server names this file. Absent from a server that predates the mark. */
  filed?: boolean
  /** The record that named this file was removed and the file kept. Absent from a server that predates the mark. */
  dismissed?: boolean
}

/**
 * Every media file under the outputs root that no record stands for, and,
 * apart, those whose record was removed.
 */
async function findUnfiled(): Promise<{ unfiled: UnfiledFile[]; removed: UnfiledFile[] }> {
  const res = await fetch('/api/outputs', { headers: { Accept: 'application/json' } })
  const type = res.headers.get('content-type') ?? ''
  if (!res.ok || !type.includes('json')) throw new Error('the outputs listing is not available here')
  const data = (await res.json()) as { files?: UnfiledFile[] }
  const known = new Set<string>()
  for (const e of history.all()) {
    known.add(relPath(e.file))
    for (const f of e.files ?? []) known.add(relPath(f))
  }
  const unfiled: UnfiledFile[] = []
  const removed: UnfiledFile[] = []
  for (const f of data.files ?? []) {
    if (f.filed || known.has(f.rel)) continue
    if (f.dismissed) removed.push(f)
    else unfiled.push(f)
  }
  return { unfiled, removed }
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

/**
 * How far a file's time may fall outside the run that wrote it, in ms. ComfyUI
 * stamps its messages and the server reads the file's time from the same
 * machine's clock, so this only has to cover rounding in the file system.
 */
const SLACK_MS = 5000

/**
 * The run that wrote the file on disk now, out of the runs ComfyUI remembers
 * naming its path, which are given newest first; undefined when none fits.
 *
 * ComfyUI hands a deleted file's name to the next picture made with the same
 * prefix, so one path can be named by several runs, and only one of them
 * wrote this file: the one under way when it was written. A run that ended
 * before then made an earlier file under the name, since deleted, and its
 * prompt and seed would remake that one; a run that began after it found the
 * file already made (a cached run names its outputs without writing them).
 * When none fits, none is taken: a record with no settings is better than one
 * carrying another picture's. A run with no time on one side is not ruled out
 * on that side.
 */
export function runThatWrote(runs: readonly PastRun[], mtime: number): PastRun | undefined {
  return runs.find(
    (r) =>
      (r.startedAt === null || r.startedAt <= mtime + SLACK_MS) &&
      (r.finishedAt === null || r.finishedAt >= mtime - SLACK_MS),
  )
}

export type Recovered = {
  /** Records made. */
  filed: number
  /** Of those, how many came with settings from ComfyUI's history. */
  fromHistory: number
  /** Files left out because their record was removed from the archive. */
  removed: number
}

let running: Promise<Recovered> | null = null

/**
 * File every unfiled output. Safe to call twice; the second call waits for the
 * first. Files whose record was removed are left out and counted, unless
 * `includeRemoved` asks for them, which only the reader does.
 */
export function recoverUnfiled(opts: { includeRemoved?: boolean } = {}): Promise<Recovered> {
  if (running) {
    // A pass already running leaves the removed files alone; one asked to
    // take them goes after it rather than returning its answer.
    return opts.includeRemoved ? running.catch(() => {}).then(() => recoverUnfiled(opts)) : running
  }
  running = (async () => {
    const found = await findUnfiled()
    const unfiled = opts.includeRemoved ? [...found.unfiled, ...found.removed] : found.unfiled
    const removed = opts.includeRemoved ? 0 : found.removed.length
    if (!unfiled.length) return { filed: 0, fromHistory: 0, removed }
    let runs: PastRun[] = []
    try { runs = await pastRuns(1000) } catch { /* ComfyUI is down or has forgotten; file minimally */ }
    // Every run that names each path, kept newest first.
    const byFile = new Map<string, PastRun[]>()
    for (const run of runs) {
      for (const f of run.files) {
        const rel = relPath(f)
        const named = byFile.get(rel)
        if (!named) byFile.set(rel, [run])
        else if (named[named.length - 1] !== run) named.push(run)
      }
    }
    const asked = new Set(opts.includeRemoved ? found.removed : [])
    // Oldest first, so edition numbers follow the order the files were made.
    // One change for the lot: filed one at a time, a folder of thousands held
    // the page for seconds.
    const records = [...unfiled]
      .sort((a, b) => a.mtime - b.mtime)
      .map((f) => {
        const run = runThatWrote(byFile.get(f.rel) ?? [], f.mtime)
        const record = run ? fromRun(run, f) : minimal(f)
        return asked.has(f) ? { ...record, refiled: true } : record
      })
    // Counted from what was filed. A desk can file one of these files while
    // ComfyUI is being asked about them, and that one is not filed twice.
    const made = history.addMany(records)
    return { filed: made.length, fromHistory: made.filter((e) => e.promptId !== '').length, removed }
  })().finally(() => { running = null })
  return running
}
