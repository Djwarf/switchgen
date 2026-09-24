/**
 * The reel draft.
 *
 * A reel is a list of lines and one set of settings that every line shares.
 * Keeping the settings out of the shots is the whole trick: a chained shot has
 * to match the shape of the frame it was handed, so width, height and the frame
 * rate belong to the reel and not to any one shot. What a shot owns is its
 * sentence, its length, and the frames it is pinned to.
 *
 * Persisted to localStorage, debounced, degrading to memory when storage is
 * locked down. Losing a reel of ten written lines to a reload would be worse
 * than losing a prompt, so this is saved the moment typing pauses.
 */
import { REEL_PREFIX } from '../../lib/continuation'
import { useSyncExternalStore } from 'react'
import { store as kv, randomSeed } from '../../lib/session'
import { thumbUrl } from '../../lib/thumbs'
import type { ShotState } from './engine'

export const REEL_KEY = 'switchgen.reel.v1'
const SAVE_DEBOUNCE_MS = 400

/** A frame pinned to a shot: an uploaded file ComfyUI's LoadImage can read. */
export type PinnedFrame = {
  /** The filename in ComfyUI's input folder, or an annotated output path. */
  name: string
  /** What to call it in the margin. */
  label: string
  /** Something the browser can show. Blank when the file came from elsewhere. */
  previewUrl?: string
}

export type ReelShot = {
  /** Stable across reordering, which is why state is keyed by it and not by index. */
  id: string
  prompt: string
  /** Frames. null follows the reel's own length. */
  length: number | null
  /** null follows the reel's seed ladder. A seed somebody typed for this shot. */
  seed: number | null
  /**
   * The seed this shot's rendered take was made with, kept while the reel's
   * seed is fixed. The ladder hands out seeds by position, so without it a
   * cut, an added shot or a move gave every later shot a new seed, and a
   * fixed reel re-rendered takes nobody had touched as different ones. Only
   * a fixed reel reads it, and it is let go when the reel goes back to
   * Random or is given a new seed, so it never pins a take nobody asked to
   * keep. Null when there is nothing to keep.
   */
  keptSeed: number | null
  /** null follows the reel's negative. */
  negative: string | null
  /** A name for the margin, when "Shot 4" is not enough. */
  label: string | null
  /** The frame this shot opens on. Set, it resets the drift count to zero. */
  start: PinnedFrame | null
  /** The frame this shot has to reach. Needs a family that pins both ends. */
  end: PinnedFrame | null
}

export type ReelDraft = {
  familyId: string
  model: string
  width: number
  height: number
  fps: number
  /** Default frames per shot. Longer shots cost less quality than more shots. */
  length: number
  steps: number
  cfg: number
  sampler: string
  scheduler: string
  /** null means the family's house wording. */
  negative: string | null
  seed: number
  seedLocked: boolean
  /** A clean frame the reel can fall back to, to stop the look drifting away. */
  anchor: PinnedFrame | null
  /** Restart from the anchor every N shots. 0 means never. */
  reanchorEvery: number
  /** Output folder under ComfyUI's output directory. */
  prefix: string
  shots: ReelShot[]
}

export function newId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`
}

export function newShot(prompt = ''): ReelShot {
  return {
    id: newId(),
    prompt,
    length: null,
    seed: null,
    keptSeed: null,
    negative: null,
    label: null,
    start: null,
    end: null,
  }
}

/**
 * The seed a shot's job is planned with: its own, else on a fixed reel the one
 * its take was made with, else undefined, which puts it on the reel's ladder.
 */
export function shotSeed(shot: ReelShot, seedLocked: boolean): number | undefined {
  return shot.seed ?? (seedLocked ? shot.keptSeed : null) ?? undefined
}

/**
 * The seeds to keep before the strip is rearranged on a fixed reel: for every
 * shot that follows the ladder and has a rendered take, the seed that take was
 * made with.
 *
 * `reelSeed` is the fixed reel's seed as it stands, and with it a take is kept
 * only when its seed is the one its shot is planned with now: the seed kept
 * from before, or else the ladder's rung at the shot's place (shotPlan gives
 * shot i the reel's seed plus i). Keeping every take undid a new seed. A
 * typed seed, or a shot's own seed cleared to follow the ladder, reads every
 * take made with the old one as changed, and the next cut, move or added shot
 * wrote the old seeds back, so the strip read current again and the new seed
 * was never rendered.
 *
 * Left out, every take keeps its seed whatever the reel's seed is. That is
 * for fixing a Random reel, whose takes were each made with a seed drawn at
 * its own press and which should all stay as they are.
 */
export function seedsToKeep(
  shots: readonly ReelShot[],
  states: Readonly<Record<string, ShotState>>,
  reelSeed?: number,
): Map<string, number> {
  const keep = new Map<string, number>()
  shots.forEach((shot, i) => {
    const made = states[shot.id]?.status === 'done' ? states[shot.id]?.made : null
    if (shot.seed !== null || !made) return
    const planned = reelSeed === undefined ? made.seed : (shot.keptSeed ?? Math.floor(reelSeed) + i)
    if (made.seed === planned) keep.set(shot.id, made.seed)
  })
  return keep
}

export function blankDraft(): ReelDraft {
  return {
    familyId: '',
    model: '',
    width: 1280,
    height: 704,
    fps: 24,
    length: 121,
    steps: 20,
    cfg: 3.5,
    sampler: 'uni_pc',
    scheduler: 'simple',
    negative: null,
    seed: randomSeed(),
    seedLocked: false,
    anchor: null,
    reanchorEvery: 0,
    prefix: REEL_PREFIX,
    // Empty on purpose: the desk prints an invitation, not three blank rows.
    shots: [],
  }
}

// ---------------------------------------------------------------------------
// Reading what was saved
// ---------------------------------------------------------------------------

function str(v: unknown, fallback: string): string {
  return typeof v === 'string' ? v : fallback
}
function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback
}
function maybeStr(v: unknown): string | null {
  return typeof v === 'string' ? v : null
}
/**
 * A pin saved before pins kept thumbnails holds the original render's address,
 * a megabyte or more drawn on a plate 16rem across at most. Read back, it
 * becomes the same file's thumbnail. Any other address is kept as it is.
 */
export function smallPreview(url: string): string {
  const view = '/comfy/view?'
  if (!url.startsWith(view)) return url
  const q = new URLSearchParams(url.slice(view.length))
  const filename = q.get('filename')
  if (!filename) return url
  return thumbUrl({ filename, subfolder: q.get('subfolder') ?? '', type: q.get('type') ?? 'output' }, 512)
}

function readFrame(v: unknown): PinnedFrame | null {
  if (!v || typeof v !== 'object') return null
  const f = v as Record<string, unknown>
  if (typeof f.name !== 'string' || !f.name) return null
  return {
    name: f.name,
    label: str(f.label, f.name),
    // An object URL from a previous page does not survive a reload, so a blob
    // preview is dropped rather than rendered as a broken plate.
    previewUrl:
      typeof f.previewUrl === 'string' && !f.previewUrl.startsWith('blob:') ? smallPreview(f.previewUrl) : undefined,
  }
}

function readShot(v: unknown): ReelShot | null {
  if (!v || typeof v !== 'object') return null
  const s = v as Record<string, unknown>
  return {
    id: str(s.id, newId()),
    prompt: str(s.prompt, ''),
    length: typeof s.length === 'number' ? s.length : null,
    seed: typeof s.seed === 'number' ? s.seed : null,
    keptSeed: typeof s.keptSeed === 'number' && Number.isFinite(s.keptSeed) ? s.keptSeed : null,
    negative: maybeStr(s.negative),
    label: maybeStr(s.label),
    start: readFrame(s.start),
    end: readFrame(s.end),
  }
}

function readDraft(raw: string | null): ReelDraft {
  const base = blankDraft()
  if (!raw) return base
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return base
  }
  if (!parsed || typeof parsed !== 'object') return base
  const d = parsed as Record<string, unknown>
  const rawShots = Array.isArray(d.shots)
    ? d.shots.map(readShot).filter((s): s is ReelShot => s !== null)
    : base.shots
  // A strip of nothing but blank rows is the old default, not a reel someone
  // wrote. Read it as empty so the invitation shows instead of the rows.
  const shots = rawShots.some((s) => s.prompt.trim() || s.start || s.end || s.label) ? rawShots : []
  return {
    familyId: str(d.familyId, base.familyId),
    model: str(d.model, base.model),
    width: num(d.width, base.width),
    height: num(d.height, base.height),
    fps: num(d.fps, base.fps),
    length: num(d.length, base.length),
    steps: num(d.steps, base.steps),
    cfg: num(d.cfg, base.cfg),
    sampler: str(d.sampler, base.sampler),
    scheduler: str(d.scheduler, base.scheduler),
    negative: maybeStr(d.negative),
    seed: num(d.seed, base.seed),
    seedLocked: d.seedLocked === true,
    anchor: readFrame(d.anchor),
    reanchorEvery: Math.max(0, Math.floor(num(d.reanchorEvery, 0))),
    prefix: str(d.prefix, base.prefix),
    shots,
  }
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

let draft: ReelDraft = readDraft(kv.get(REEL_KEY))
const listeners = new Set<() => void>()
let saveTimer: ReturnType<typeof setTimeout> | null = null

function emit(): void {
  for (const fn of [...listeners]) {
    try {
      fn()
    } catch {
      /* one bad subscriber must not strand the rest */
    }
  }
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(save, SAVE_DEBOUNCE_MS)
}

function save(): void {
  saveTimer = null
  try {
    kv.set(REEL_KEY, JSON.stringify(draft))
  } catch {
    // A full quota costs the saved copy, never the reel on screen.
  }
}

function commit(next: ReelDraft): void {
  draft = next
  emit()
}

function mapShots(fn: (shots: ReelShot[]) => ReelShot[]): void {
  commit({ ...draft, shots: fn([...draft.shots]) })
}

export const reel = {
  subscribe(fn: () => void): () => void {
    listeners.add(fn)
    return () => {
      listeners.delete(fn)
    }
  },
  get: (): ReelDraft => draft,

  patch(p: Partial<ReelDraft>): void {
    commit({ ...draft, ...p })
  },

  /** Write the reel's saved copy now, for a page that is about to unload. */
  flush(): void {
    if (saveTimer) clearTimeout(saveTimer)
    save()
  },

  setShot(id: string, p: Partial<ReelShot>): void {
    mapShots((shots) => shots.map((s) => (s.id === id ? { ...s, ...p } : s)))
  },

  add(after?: string): string {
    const shot = newShot()
    mapShots((shots) => {
      const at = after ? shots.findIndex((s) => s.id === after) : -1
      if (at < 0) return [...shots, shot]
      return [...shots.slice(0, at + 1), shot, ...shots.slice(at + 1)]
    })
    return shot.id
  },

  duplicate(id: string): string | null {
    const source = draft.shots.find((s) => s.id === id)
    if (!source) return null
    // The take belongs to the shot it was made for. The copy has none, so it
    // takes its place on the ladder rather than repeat that take.
    const copy: ReelShot = { ...source, id: newId(), keptSeed: null, start: null, end: source.end }
    mapShots((shots) => {
      const at = shots.findIndex((s) => s.id === id)
      return [...shots.slice(0, at + 1), copy, ...shots.slice(at + 1)]
    })
    return copy.id
  },

  /** @returns the removed shot, so the caller can offer it back. */
  remove(id: string): { shot: ReelShot; index: number } | null {
    const index = draft.shots.findIndex((s) => s.id === id)
    const shot = draft.shots[index]
    if (!shot) return null
    mapShots((shots) => shots.filter((s) => s.id !== id))
    return { shot, index }
  },

  restore(shot: ReelShot, index: number): void {
    mapShots((shots) => [...shots.slice(0, index), shot, ...shots.slice(index)])
  },

  /** Keep each listed shot's take seed (see ReelShot.keptSeed). One write for the lot. */
  keepSeeds(seeds: ReadonlyMap<string, number>): void {
    if (!seeds.size) return
    mapShots((shots) => shots.map((s) => (seeds.has(s.id) ? { ...s, keptSeed: seeds.get(s.id) ?? null } : s)))
  },

  /** Let every kept take seed go, so each shot follows the ladder again. */
  releaseSeeds(): void {
    if (!draft.shots.some((s) => s.keptSeed !== null)) return
    mapShots((shots) => shots.map((s) => (s.keptSeed === null ? s : { ...s, keptSeed: null })))
  },

  /** Move one shot by `delta` places. Clamped, so the ends simply hold. */
  move(id: string, delta: number): void {
    mapShots((shots) => {
      const from = shots.findIndex((s) => s.id === id)
      if (from < 0) return shots
      const to = Math.min(shots.length - 1, Math.max(0, from + delta))
      if (to === from) return shots
      const [moved] = shots.splice(from, 1)
      if (!moved) return shots
      shots.splice(to, 0, moved)
      return shots
    })
  },

  clear(): void {
    commit({ ...draft, shots: [] })
  },

  reset(): void {
    commit(blankDraft())
  },
}

/** The draft, live. */
export function useReel(): ReelDraft {
  return useSyncExternalStore(reel.subscribe, reel.get, reel.get)
}

/** Split a block of text into one shot per non-empty line. */
export function shotsFromLines(text: string): ReelShot[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => newShot(line))
}
