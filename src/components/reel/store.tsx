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
  /** null follows the reel's seed ladder. */
  seed: number | null
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
  return { id: newId(), prompt, length: null, seed: null, negative: null, label: null, start: null, end: null }
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
function readFrame(v: unknown): PinnedFrame | null {
  if (!v || typeof v !== 'object') return null
  const f = v as Record<string, unknown>
  if (typeof f.name !== 'string' || !f.name) return null
  return {
    name: f.name,
    label: str(f.label, f.name),
    // An object URL from a previous page does not survive a reload, so a blob
    // preview is dropped rather than rendered as a broken plate.
    previewUrl: typeof f.previewUrl === 'string' && !f.previewUrl.startsWith('blob:') ? f.previewUrl : undefined,
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
    const copy: ReelShot = { ...source, id: newId(), start: null, end: source.end }
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
