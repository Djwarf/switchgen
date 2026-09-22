/**
 * The Reel desk.
 *
 * A clip is five seconds. A scene is not, and neither is anything anybody would
 * call a film. This room exists to turn a list of written lines into a run of
 * shots where each one opens on the last frame of the one before it, so the
 * clips can be laid end to end and read as a single take.
 *
 * WHAT IS DIFFERENT HERE
 *
 * 1. The settings belong to the reel, not to the shot. A chained shot is handed
 *    the previous shot's final frame, and every Wan image node centre crops what
 *    it is handed, so a shape that changes mid reel silently loses the edges of
 *    the picture. Width, height and frame rate therefore sit on the bench and a
 *    shot cannot override them.
 * 2. Nothing runs in parallel. Ten shots is ten generations of several minutes,
 *    and the memory peak that fits on this card was measured one job at a time.
 *    The queue walks in order and the band reports where it has got to.
 * 3. Drift is shown, not hidden. Every hop re encodes the model's own output, so
 *    colour creeps and faces move. The strip prints the hop count on each shot
 *    and the bench prints the deepest one in the reel. A reader who can see the
 *    number can decide for themselves whether to pin a fresh frame.
 * 4. No finish time is ever printed from a guess. The band shows one only when
 *    every remaining shot has three measured runs of the same family at the same
 *    size and length sitting in the archive.
 *
 * The graph work is all in lib/continuation.ts, which derives each shot's
 * workflow from a registry family that was already validated against the live
 * schema. Nothing here builds a graph by hand.
 */
import { ServerDown } from '../components/ServerDown'
import { availabilityOf, inventoryFrom } from '../lib/availability'
import { clipMemory, type ClipMemory } from '../lib/clipMemory'
import { feasibility, modelFiles, probeHardware, type Hardware, type ModelFile } from '../lib/hardware'
import { onPlanLanded } from '../lib/downloads'
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from 'react'

import { connect, fileUrl, objectInfo, type FileRef } from '../lib/comfy'
import {
  annotatedRef,
  checkReel,
  clipFrames,
  deriveContinuation,
  explainUnavailable,
  shotPlan,
  snapLength,
  type ShotJob,
  type ShotPlan,
  type ShotSpec,
} from '../lib/continuation'
import { history } from '../lib/history'
import { newComposition, randomSeed, type Composition } from '../lib/session'
import { FAMILIES, defaultsFor, type FamilyDef, type Params } from '../lib/workflows'
import { go, parseRoute, useExpert } from '../components/shell'

import {
  Assembly,
  Bench,
  KeyframePicker,
  Kicker,
  Quiet,
  ReelProgress,
  Strip,
  currencyOf,
  grouped,
  reel,
  reelRun,
  seconds,
  shotsFromLines,
  shotsToRender,
  useReel,
  type AssemblyClip,
  type Currency,
  type KeyframeTarget,
  type NumSpec,
  type PinnedFrame,
  type ReelDraft,
  type ReelFamily,
  type ReelShot,
  type RunContext,
  type Shape,
  EmptyStrip
} from '../components/reel'

import type { PlayerSlot } from './Video'

export type ReelProps = {
  /** The shell lends the reel the real player, exactly as it does the video desk. */
  renderPlayer?: (slot: PlayerSlot) => ReactNode
  /** Navigate elsewhere in the shell, for example `#/archive?q=is:video`. */
  onNavigate?: (hash: string) => void
}

const EMPTY_PLAN: ShotPlan = { jobs: [], frames: 0, seconds: 0, warnings: [] }

const SPEC_FALLBACK: Record<'size' | 'frames', NumSpec> = {
  size: { min: 32, max: 16384, step: 16 },
  frames: { min: 1, max: 16384, step: 4 },
}

const START_FRAME_NODES = ['Wan22ImageToVideoLatent', 'WanImageToVideo', 'WanFirstLastFrameToVideo']

// ---------------------------------------------------------------------------
// What this machine can run
// ---------------------------------------------------------------------------

type Catalogue = {
  /** Families whose every file is installed. Memory is asked of a fresh reading; see {@link offered}. */
  families: ReelFamily[]
  blocked: { label: string; why: string }[]
  samplers: string[]
  schedulers: string[]
  /** Weight files on disk and their sizes, for pricing a family against a reading. */
  sizes: Map<string, ModelFile>
  /** What the machine had when the catalogue was read. Null when the probe failed. The desk reads it again. */
  hardware: Hardware | null
}

/**
 * The families this machine can hold, priced against one memory reading. One
 * it cannot hold at all goes to the blocked list with the verdict's sentence.
 * Without a reading nothing is priced, as availabilityOf does.
 */
function offered(cat: Catalogue, hardware: Hardware | null): Pick<Catalogue, 'families' | 'blocked'> {
  if (!hardware) return cat
  const families: ReelFamily[] = []
  const blocked = [...cat.blocked]
  for (const family of cat.families) {
    const verdict = feasibility(family.def, cat.sizes, hardware)
    if (verdict.selectable) families.push(family)
    else blocked.push({ label: family.def.label, why: verdict.reason })
  }
  return { families, blocked }
}

function latentClassOf(def: FamilyDef): string | null {
  for (const node of Object.values(def.graph)) {
    if (START_FRAME_NODES.includes(node.class_type)) return node.class_type
    if (/^Empty.*Latent/.test(node.class_type)) return node.class_type
  }
  return null
}

function readSpec(info: Record<string, unknown>, cls: string | null, field: string, fallback: NumSpec): NumSpec {
  if (!cls) return fallback
  const node = info[cls] as { input?: { required?: Record<string, unknown> } } | undefined
  const raw = node?.input?.required?.[field]
  if (!Array.isArray(raw) || raw.length < 2) return fallback
  const opts = raw[1] as { min?: number; max?: number; step?: number }
  return {
    min: typeof opts?.min === 'number' ? opts.min : fallback.min,
    max: typeof opts?.max === 'number' ? opts.max : fallback.max,
    step: typeof opts?.step === 'number' && opts.step > 0 ? opts.step : fallback.step,
  }
}

/**
 * Which video families are installed, and which can chain.
 *
 * A family whose weights, text encoder or VAE are missing is listed as not
 * offered, with the file that is missing, rather than silently dropped. A
 * family that is installed but cannot open a shot on a frame is still offered:
 * it makes perfectly good clips, they simply will not continue from each other,
 * and the bench says so where the choice is made.
 */
async function loadCatalogue(): Promise<Catalogue> {
  const [info, hardware, sizes] = await Promise.all([
    objectInfo(),
    probeHardware().catch(() => null),
    modelFiles().catch(() => new Map<string, ModelFile>()),
  ])

  const inv = inventoryFrom(info)
  const weights = inv.weights

  const families: ReelFamily[] = []
  const blocked: Catalogue['blocked'] = []

  for (const def of FAMILIES) {
    if (def.mode !== 'video') continue

    // Files here, memory in offered(). This desk used to skip the memory
    // verdict, so it could offer a family the machine cannot hold and let the
    // reader find out eight shots in.
    const avail = availabilityOf(def, inv, null, sizes)
    if (!avail.ok) {
      blocked.push({ label: def.label, why: avail.why })
      continue
    }

    const latent = latentClassOf(def)
    families.push({
      def,
      model: def.dualModel ? '' : (def.models.find((m) => weights.has(m)) ?? def.models[0] ?? ''),
      label: def.label,
      chainable: deriveContinuation(def) !== null,
      why: explainUnavailable(def, 'continuation'),
      width: readSpec(info, latent, 'width', SPEC_FALLBACK.size),
      height: readSpec(info, latent, 'height', SPEC_FALLBACK.size),
      frames: readSpec(info, latent, 'length', SPEC_FALLBACK.frames),
    })
  }

  // A family that can carry a reel is offered before one that cannot.
  families.sort((a, b) => Number(b.chainable) - Number(a.chainable))

  return {
    families,
    blocked,
    samplers: inv.samplers,
    schedulers: inv.schedulers,
    sizes,
    hardware,
  }
}

let cataloguePromise: Promise<Catalogue> | null = null

// A family fetched from the catalogue can land while the reader is in another
// room. What is installed is read once and kept, so the kept reading is
// dropped here and the next visit reads what is installed now; the desk on
// screen hears of it through its own listener.
onPlanLanded(() => {
  cataloguePromise = null
})

function catalogue(): Promise<Catalogue> {
  if (!cataloguePromise) {
    cataloguePromise = loadCatalogue().catch((err: unknown) => {
      cataloguePromise = null
      throw err
    })
  }
  return cataloguePromise
}

// ---------------------------------------------------------------------------
// Offered values
// ---------------------------------------------------------------------------

function snapTo(value: number, spec: NumSpec): number {
  const steps = Math.round((value - spec.min) / spec.step)
  return Math.min(spec.max, Math.max(spec.min, spec.min + steps * spec.step))
}

function orientation(w: number, h: number): string {
  if (w > h) return 'Landscape'
  if (h > w) return 'Portrait'
  return 'Square'
}

function shapeChoices(family: ReelFamily): Shape[] {
  const d = defaultsFor(family.def, family.model)
  const out: Shape[] = []
  const push = (w: number, h: number) => {
    const width = snapTo(w, family.width)
    const height = snapTo(h, family.height)
    const label = orientation(width, height)
    if (!out.some((s) => s.width === width && s.height === height)) out.push({ width, height, label })
  }
  push(d.width, d.height)
  if (d.width !== d.height) push(d.height, d.width)
  push(Math.min(d.width, d.height), Math.min(d.width, d.height))
  return out
}

/** A handful of legal shot lengths around the durations people actually cut to. */
function lengthChoices(fps: number, spec: NumSpec): number[] {
  const out: number[] = []
  for (const s of [2, 3, 5, 7]) {
    const n = snapLength(Math.round(s * Math.max(1, fps)))
    if (n >= spec.min && n <= spec.max && !out.includes(n)) out.push(n)
  }
  return out.sort((a, b) => a - b)
}

/**
 * A family's own verified recipe, as bench settings. A reel is one continuous
 * piece, so the whole bench follows the family rather than carrying settings
 * across from a different model.
 */
function recipeFor(f: ReelFamily): Partial<ReelDraft> {
  const d = defaultsFor(f.def, f.model)
  return {
    familyId: f.def.id,
    model: f.model,
    width: d.width,
    height: d.height,
    fps: d.fps || 24,
    length: snapLength(d.length || 81),
    steps: d.steps,
    cfg: d.cfg,
    sampler: d.sampler,
    scheduler: d.scheduler,
    negative: null,
  }
}

/** "Shot 3" or "Shots 2, 4 and 5". */
function shotsWord(numbers: readonly number[]): string {
  if (numbers.length === 1) return `Shot ${numbers[0]}`
  return `Shots ${numbers.slice(0, -1).join(', ')} and ${numbers[numbers.length - 1]}`
}

/**
 * One sentence per distinct memory verdict at `level`, naming the shots it
 * covers unless it covers them all. A reel of ten shots at one size would
 * otherwise print the same paragraph ten times.
 */
function byReason(verdicts: readonly (ClipMemory | null)[], level: ClipMemory['level']): string[] {
  const shots = new Map<string, number[]>()
  verdicts.forEach((v, i) => {
    if (v?.level !== level || !v.reason) return
    shots.set(v.reason, [...(shots.get(v.reason) ?? []), i + 1])
  })
  return [...shots].map(([reason, numbers]) =>
    numbers.length === verdicts.length ? reason : `${shotsWord(numbers)}: ${reason}`,
  )
}

/** A clock that ticks only while something is running. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    setNow(Date.now())
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [active])
  return now
}

// ---------------------------------------------------------------------------
// The desk
// ---------------------------------------------------------------------------

export default function Reel({ renderPlayer, onNavigate }: ReelProps = {}) {
  const draft = useReel()
  const run = useSyncExternalStore(reelRun.subscribe, reelRun.snapshot, reelRun.snapshot)
  const records = useSyncExternalStore(history.subscribe, history.all, history.all)
  const expert = useExpert()

  const [cat, setCat] = useState<Catalogue | null>(null)
  const [catError, setCatError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  const [pinning, setPinning] = useState<KeyframeTarget | null>(null)
  const [watching, setWatching] = useState<string | null>(null)
  const [pasting, setPasting] = useState(false)
  const [pasted, setPasted] = useState('')
  const [undoCut, setUndoCut] = useState<{ shot: ReelShot; index: number } | null>(null)
  const bulk = useRef<HTMLTextAreaElement | null>(null)
  const undoTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  /** A cut shot is offered back for twelve seconds, then it is simply gone. */
  const cut = useCallback((id: string) => {
    const removed = reel.remove(id)
    if (!removed) return
    setUndoCut(removed)
    if (undoTimer.current) clearTimeout(undoTimer.current)
    undoTimer.current = setTimeout(() => setUndoCut(null), 12_000)
  }, [])

  const busy = run.status === 'running'
  const now = useNow(busy)

  /**
   * Leaving the reel. The shell lends a navigator when it has one; without it
   * the hash router is asked directly, which is the same thing one layer down.
   * No router library is installed and none may be added.
   */
  const navigate = useCallback(
    (hash: string) => {
      if (onNavigate) onNavigate(hash)
      else go(parseRoute(hash))
    },
    [onNavigate],
  )

  // --- what is installed ---------------------------------------------------
  useEffect(() => {
    let alive = true
    let retry: ReturnType<typeof setTimeout> | null = null
    connect()
    catalogue().then(
      (c) => {
        if (!alive) return
        setCat(c)
        setCatError(null)
      },
      (err: Error) => {
        if (!alive) return
        setCatError(err.message)
        retry = setTimeout(() => setAttempt((a) => a + 1), 5000)
      },
    )
    return () => {
      alive = false
      if (retry) clearTimeout(retry)
    }
  }, [attempt])

  useEffect(() => onPlanLanded(() => setAttempt((a) => a + 1)), [])

  /**
   * Memory, read again on every visit and whenever the page comes back into
   * view, rather than once per page load with the catalogue. The reel's own
   * checks read the machine's total memory, which does not move, but a probe
   * that failed when the catalogue was read used to leave the desk without a
   * reading until the page was reloaded. A reading that fails keeps the last
   * good one.
   */
  const [freshHardware, setFreshHardware] = useState<Hardware | null>(null)
  const [hardwareAsk, setHardwareAsk] = useState(0)
  useEffect(() => {
    let alive = true
    probeHardware().then(
      (hw) => {
        if (alive) setFreshHardware(hw)
      },
      () => {},
    )
    return () => {
      alive = false
    }
  }, [hardwareAsk])
  useEffect(() => {
    const onShow = () => {
      if (document.visibilityState === 'visible') setHardwareAsk((n) => n + 1)
    }
    document.addEventListener('visibilitychange', onShow)
    return () => document.removeEventListener('visibilitychange', onShow)
  }, [])
  const hardware = freshHardware ?? cat?.hardware ?? null
  const offer = useMemo(() => (cat ? offered(cat, hardware) : null), [cat, hardware])
  const families = useMemo(() => offer?.families ?? [], [offer])

  const family = useMemo<ReelFamily | null>(() => {
    if (!families.length) return null
    return families.find((f) => f.def.id === draft.familyId) ?? families[0] ?? null
  }, [families, draft.familyId])

  // The family's recipe, applied when the draft does not match the family it
  // resolves to: a first load, or a saved family that is no longer installed.
  // A deliberate change of style is handled where the bench patches (see
  // patchBench), because this check alone cannot see it between the two Wan
  // 2.2 14B pairs: both are two-model families with model '' here.
  useEffect(() => {
    if (!family) return
    if (draft.familyId === family.def.id && draft.model === family.model) return
    reel.patch(recipeFor(family))
  }, [family, draft.familyId, draft.model])

  const houseNegative = family ? defaultsFor(family.def, family.model).negative : ''
  const shapes = useMemo(() => (family ? shapeChoices(family) : []), [family])
  const lengthOptions = useMemo(() => {
    if (!family) return []
    const list = lengthChoices(draft.fps, family.frames)
    return list.includes(draft.length) ? list : [...list, draft.length].sort((a, b) => a - b)
  }, [family, draft.fps, draft.length])

  const bookendBlocked = useMemo(
    () => (family ? explainUnavailable(family.def, 'bookend') : 'No video family is installed.'),
    [family],
  )

  // --- the plan ------------------------------------------------------------

  const params = useMemo<Params | null>(() => {
    if (!family) return null
    return {
      model: family.model,
      positive: '',
      negative: draft.negative ?? houseNegative ?? '',
      seed: Math.floor(draft.seed),
      steps: Math.floor(draft.steps),
      cfg: draft.cfg,
      width: Math.floor(draft.width),
      height: Math.floor(draft.height),
      sampler: draft.sampler,
      scheduler: draft.scheduler,
      length: Math.floor(draft.length),
      fps: draft.fps,
    }
  }, [family, draft, houseNegative])

  const plan = useMemo<ShotPlan>(() => {
    if (!family || !params) return EMPTY_PLAN
    const specs: ShotSpec[] = draft.shots.map((s) => ({
      prompt: s.prompt,
      negative: s.negative ?? undefined,
      seed: s.seed ?? undefined,
      length: s.length ?? draft.length,
      label: s.label ?? undefined,
      startImage: s.start?.name,
      endImage: s.end?.name,
    }))
    return shotPlan({
      base: family.def,
      params,
      shots: specs,
      anchorImage: draft.anchor?.name,
      reanchorEvery: draft.reanchorEvery,
      prefix: draft.prefix,
      freshSeeds: !draft.seedLocked,
    })
  }, [family, params, draft.shots, draft.length, draft.anchor, draft.reanchorEvery, draft.prefix, draft.seedLocked])

  const issues = useMemo(() => checkReel(plan.jobs), [plan.jobs])
  const order = useMemo(() => draft.shots.map((s) => s.id), [draft.shots])

  // --- what may be sent ----------------------------------------------------

  const memoryFor = useCallback(
    (job: ShotJob): ClipMemory | null =>
      family
        ? clipMemory(
            family.def,
            { width: job.params.width, height: job.params.height, frames: job.params.length ?? 0 },
            hardware,
          )
        : null,
    [family, hardware],
  )
  const memory = useMemo(() => plan.jobs.map(memoryFor), [plan.jobs, memoryFor])
  /** Per shot: true when the desk will not queue it as it stands. */
  const refused = useMemo(
    () => plan.jobs.map((j, i) => j.blocked !== null || memory[i]?.level === 'refuse'),
    [plan.jobs, memory],
  )
  const refusals = useMemo(
    () => [...plan.jobs.flatMap((j) => (j.blocked ? [j.blocked] : [])), ...byReason(memory, 'refuse')],
    [plan.jobs, memory],
  )
  const cautions = useMemo(() => byReason(memory, 'caution'), [memory])

  // --- what is already on disk ---------------------------------------------

  /** Per shot, in strip order: whether its clip still matches its line. */
  const currency = useMemo<(Currency | null)[]>(
    () => order.map((_, i) => currencyOf(i, order, plan.jobs, run.states)),
    [order, plan.jobs, run.states],
  )
  /** What "Render what is missing" would queue, worked out the way the engine does. */
  const missing = useMemo(() => shotsToRender(order, plan.jobs, run.states), [order, plan.jobs, run.states])
  const blanks = useMemo(
    () => draft.shots.map((s, i) => (s.prompt.trim() ? null : i + 1)).filter((n): n is number => n !== null),
    [draft.shots],
  )

  // --- filing a finished shot ---------------------------------------------

  const context = useMemo<RunContext | null>(() => {
    if (!family) return null
    const label = family.label
    const model = family.model || family.def.label
    const compositionFor = (job: ShotJob): Composition =>
      newComposition('video', {
        mode: job.start.from === 'none' ? 't2v' : 'i2v',
        familyId: family.def.id,
        model: family.model,
        prompt: job.params.positive,
        negative: draft.negative,
        width: Math.floor(draft.width),
        height: Math.floor(draft.height),
        seed: job.params.seed,
        steps: Math.floor(draft.steps),
        cfg: draft.cfg,
        sampler: draft.sampler,
        scheduler: draft.scheduler,
        length: job.params.length ?? draft.length,
        fps: draft.fps,
      })
    const memoryOf = (job: ShotJob): ClipMemory =>
      memoryFor(job) ?? { level: 'ok', reason: null, release: false }
    return { familyLabel: label, modelLabel: model, compositionFor, memory: memoryOf }
  }, [family, draft, memoryFor])

  // --- running -------------------------------------------------------------

  /**
   * The jobs one press hands the engine. On a Random reel every press draws a
   * new seed: with the old one, rendering a shot again queued the identical
   * graph, and ComfyUI answered it from its cache with the clip it had already
   * made. The drawn seed is written back to the bench, so the number shown is
   * the one this press started from. A shot with its own seed keeps it.
   */
  const pressJobs = useCallback((): readonly ShotJob[] => {
    if (draft.seedLocked) return plan.jobs
    const seed = randomSeed()
    reel.patch({ seed })
    return plan.jobs.map((j) => (j.seedFixed ? j : { ...j, params: { ...j.params, seed: seed + j.index } }))
  }, [draft.seedLocked, plan.jobs])

  const renderAll = useCallback(
    (opts: { force?: boolean } = {}) => {
      if (!context || busy || blanks.length || !plan.jobs.length || refusals.length) return
      if (!opts.force && !missing.length) return
      reelRun.renderAll(order, pressJobs(), context, opts)
    },
    [context, busy, blanks.length, plan.jobs.length, refusals.length, missing.length, order, pressJobs],
  )

  const renderOne = useCallback(
    (index: number) => {
      if (!context || busy || refused[index]) return
      reelRun.renderOne(index, order, pressJobs(), context)
    },
    [context, busy, refused, order, pressJobs],
  )

  /**
   * The bench's changes, with two that need more than a patch.
   *
   * A deliberate change of style takes the new family's recipe outright. The
   * effect above cannot see a move between the two Wan 2.2 14B pairs, so that
   * move used to keep the old pair's length: 81 frames on the image-to-video
   * pair, which the registry records being killed for memory at that length.
   *
   * Fixing a Random seed keeps the reel on screen. Each rendered shot that
   * follows the reel's seed takes the seed it was actually made with as its
   * own, so fixing the seed does not turn every earlier take into a change.
   * Typing a number is different: that asks for new seeds, and the shots made
   * with other ones then read as changed.
   */
  const patchBench = useCallback(
    (p: Partial<ReelDraft>) => {
      if (p.familyId !== undefined && p.familyId !== draft.familyId) {
        const next = families.find((f) => f.def.id === p.familyId)
        if (next) {
          reel.patch({ ...p, ...recipeFor(next) })
          return
        }
      }
      if (p.seedLocked === true && p.seed === undefined && !draft.seedLocked) {
        draft.shots.forEach((shot, i) => {
          const state = run.states[shot.id]
          const job = plan.jobs[i]
          if (shot.seed !== null || state?.status !== 'done' || !state.made || !job) return
          if (state.made.seed !== job.params.seed) reel.setShot(shot.id, { seed: state.made.seed })
        })
      }
      reel.patch(p)
    },
    [families, draft.familyId, draft.seedLocked, draft.shots, run.states, plan.jobs],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        renderAll()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [renderAll])

  // The reel is worth keeping even when the tab goes away mid sentence.
  useEffect(() => {
    const flush = () => reel.flush()
    window.addEventListener('pagehide', flush)
    return () => {
      window.removeEventListener('pagehide', flush)
      reel.flush()
    }
  }, [])

  // --- pinning -------------------------------------------------------------

  const pictures = useMemo(() => records.filter((e) => e.kind === 'image').slice(0, 40), [records])
  const reelFrames = useMemo(
    () =>
      order
        .map((id, i) => {
          const frame = run.states[id]?.frame
          if (!frame) return null
          return {
            label: `Shot ${i + 1} last frame`,
            name: annotatedRef(frame),
            previewUrl: fileUrl(frame),
          }
        })
        .filter((f): f is { label: string; name: string; previewUrl: string } => f !== null),
    [order, run.states],
  )

  const pinShot = useCallback((id: string, which: 'start' | 'end') => {
    setPinning({
      title: which === 'start' ? 'The frame this shot opens on' : 'The frame this shot has to reach',
      standfirst:
        which === 'start'
          ? 'Pinning an opening frame breaks the chain here on purpose. The drift count goes back to zero and the shot starts clean.'
          : 'The model fills the span between the opening frame and this one. Storyboarding rather than hoping.',
      onPick: (frame: PinnedFrame) => reel.setShot(id, which === 'start' ? { start: frame } : { end: frame }),
    })
  }, [])

  const pinAnchor = useCallback(() => {
    setPinning({
      title: 'The anchor frame',
      standfirst:
        'A clean frame the whole reel can return to. It sets the look of the opening shot, and the schedule on the bench brings the reel back to it before the drift gets away.',
      onPick: (frame: PinnedFrame) => reel.patch({ anchor: frame }),
    })
  }, [])

  // --- what is on the plate ------------------------------------------------

  const shown = useMemo(() => {
    if (!watching) return null
    const state = run.states[watching]
    const clip = state?.clip
    if (!clip) return null
    const index = order.indexOf(watching)
    const job = index >= 0 ? plan.jobs[index] : null
    const entry = state.entryId ? (records.find((r) => r.id === state.entryId) ?? null) : null
    return {
      file: clip as FileRef,
      entry,
      frames: state.made?.frames ?? (job ? clipFrames(job) : 0),
      fps: state.made?.fps || draft.fps,
      index,
    }
  }, [watching, run.states, order, plan.jobs, records, draft.fps])

  // The cutting room follows the strip, not the last run. Built from the run's
  // own order, a cut shot stayed in the join, a moved one kept its old place,
  // and a pasted reel left the previous reel's clips waiting to be cut.
  const clips = useMemo<AssemblyClip[]>(
    () =>
      order
        .map((id, i) => {
          const state = run.states[id]
          if (!state?.clip || state.status !== 'done') return null
          return {
            index: i,
            label: `Shot ${i + 1}`,
            file: state.clip,
            frames: state.made?.frames ?? state.frames,
            fps: state.made?.fps || draft.fps,
            width: state.made?.width ?? draft.width,
            height: state.made?.height ?? draft.height,
            durationMs: state.durationMs,
            outOfDate: currency[i] !== 'current',
          }
        })
        .filter((c): c is AssemblyClip => c !== null),
    [order, run.states, currency, draft.fps, draft.width, draft.height],
  )

  const changedCount = currency.filter((c) => c === 'changed').length
  const staleCount = currency.filter((c) => c === 'stale').length

  // --- the page ------------------------------------------------------------

  // The other two desks print this page with a Try now link; this one printed
  // one line and left the reader to wait.
  if (catError && !cat) return <ServerDown onRetry={() => setAttempt((a) => a + 1)} detail={catError} />

  return (
    <main className="relative min-h-full px-6 py-6">
      <div className="mb-5 border-b-2 border-burgundy-900 pb-2">
        <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
          <h2 className="text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-burgundy-900">The Reel Desk</h2>
          <p className="text-caption tabular-nums text-grey-700">
            {grouped(draft.shots.length)} shots · {grouped(plan.frames)} frames · {seconds(plan.frames, draft.fps)} of
            screen time
          </p>
        </div>
        <p className="mt-1 text-small italic text-grey-700">
          Each shot opens on the last frame of the one before it. Write a line for each, press once, walk away.
        </p>
      </div>

      {catError ? (
        <p className="notice notice-error mb-5 text-small">
          <strong>ComfyUI is not answering.</strong> {catError}. This desk retries every five seconds by itself.
        </p>
      ) : null}

      <ReelProgress
        run={run}
        jobs={plan.jobs}
        fps={draft.fps}
        width={draft.width}
        height={draft.height}
        familyId={family?.def.id ?? ''}
        records={records}
        now={now}
        onStop={() => reelRun.stop()}
      />

      {shown && renderPlayer ? (
        <section className="mb-6">
          <div className="mb-2 flex items-baseline justify-between gap-3">
            <Kicker>Shot {shown.index + 1}</Kicker>
            <button type="button" className="sg-link text-caption" onClick={() => setWatching(null)}>
              Close the viewer
            </button>
          </div>
          {renderPlayer({
            src: fileUrl(shown.file),
            file: shown.file,
            entry: shown.entry,
            fps: shown.fps,
            frames: shown.frames,
          })}
        </section>
      ) : null}

      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
        {/* the strip -------------------------------------------------------- */}
        <div className="order-2 min-w-0 lg:order-1">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3 border-b-2 border-burgundy-900 pb-1.5">
            <Kicker>The strip</Kicker>
            <div className="flex flex-wrap items-center gap-3">
              <button type="button" className="sg-link text-caption" onClick={() => setPasting((v) => !v)}>
                {pasting ? 'Hide the paste box' : 'Paste a whole reel'}
              </button>
              {clips.length ? (
                <button
                  type="button"
                  className="sg-link text-caption"
                  onClick={() => navigate('#/archive?q=is:video')}
                >
                  Every clip in the archive
                </button>
              ) : null}
            </div>
          </div>

          {pasting ? (
            <div className="mb-4 border border-grey-300 bg-newsprint-aged p-3">
              <p className="mb-2 text-caption italic text-grey-700">
                One line a shot. This replaces the strip, so an unrendered reel is worth copying out first.
              </p>
              <textarea
                ref={bulk}
                className="field h-28 text-small"
                placeholder={'A car pulls up outside a shuttered shop\nThe driver steps out into the rain\nShe looks up at a lit window'}
                value={pasted}
                onChange={(e) => setPasted(e.target.value)}
              />
              <div className="mt-2 flex flex-wrap gap-2">
                <Quiet
                  disabled={!pasted.trim() || busy}
                  onClick={() => {
                    const shots = shotsFromLines(pasted)
                    if (!shots.length) return
                    reel.patch({ shots })
                    setPasted('')
                    setPasting(false)
                  }}
                >
                  Lay these out as shots
                </Quiet>
                <Quiet onClick={() => setPasting(false)}>Cancel</Quiet>
              </div>
            </div>
          ) : null}

          {family && !draft.shots.length ? (
            <EmptyStrip
              onLayOut={(text) => {
                const shots = shotsFromLines(text)
                if (shots.length) reel.patch({ shots })
              }}
              onStartOne={() => reel.add()}
              onPaste={() => setPasting(true)}
            />
          ) : family ? (
            <Strip
              shots={draft.shots}
              jobs={plan.jobs}
              run={run}
              fps={draft.fps}
              reelLength={draft.length}
              lengthOptions={lengthOptions}
              expert={expert}
              busy={busy}
              currency={currency}
              refused={refused}
              bookendBlocked={bookendBlocked}
              onEdit={(id, patch) => reel.setShot(id, patch)}
              onMove={(id, delta) => reel.move(id, delta)}
              onRemove={cut}
              onDuplicate={(id) => reel.duplicate(id)}
              onAdd={(after) => reel.add(after)}
              onRender={renderOne}
              onPin={pinShot}
              onWatch={(id) => setWatching(id)}
            />
          ) : (
            <p className="py-8 text-small italic text-grey-500">
              {catError ? 'Waiting for ComfyUI.' : 'Reading what this machine has installed.'}
            </p>
          )}

          {undoCut ? (
            <p className="notice notice-correction mb-4 text-small">
              <strong>Shot {undoCut.index + 1} cut.</strong> It is not lost yet.{' '}
              <button
                type="button"
                className="sg-link"
                onClick={() => {
                  reel.restore(undoCut.shot, undoCut.index)
                  setUndoCut(null)
                }}
              >
                Put it back
              </button>
            </p>
          ) : null}

          {issues.length || cautions.length ? (
            <div className="mb-4 space-y-1">
              {[...issues, ...cautions].map((issue) => (
                <p key={issue} className="border-l-2 border-warning pl-2 text-caption text-ink-warning">
                  {issue}
                </p>
              ))}
            </div>
          ) : null}

          {refusals.length ? (
            <div className="mb-4 space-y-1">
              {refusals.map((why) => (
                <p key={why} className="border-l-2 border-error pl-2 text-caption text-ink-error">
                  {why}
                </p>
              ))}
            </div>
          ) : null}

          {/* the press ------------------------------------------------------ */}
          <div className="mt-2 border-t-2 border-burgundy-900 pt-4">
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                className="press"
                disabled={
                  busy || blanks.length > 0 || !plan.jobs.length || !family || refusals.length > 0 || !missing.length
                }
                onClick={() => renderAll()}
              >
                {busy ? 'On the press' : clips.length ? 'Render what is missing' : 'Render the reel'}
              </button>

              {clips.length && !busy ? (
                <Quiet onClick={() => renderAll({ force: true })} disabled={refusals.length > 0 || blanks.length > 0}>
                  Render every shot again
                </Quiet>
              ) : null}
              {(run.status !== 'idle' || Object.keys(run.states).length > 0) && !busy ? (
                <Quiet onClick={() => reelRun.clear()}>Clear the run</Quiet>
              ) : null}

              <span className="text-caption italic text-grey-500">
                {blanks.length
                  ? blanks.length === 1
                    ? `Shot ${blanks[0]} has no line yet.`
                    : `Shots ${blanks.join(', ')} have no lines yet.`
                  : busy
                    ? 'One shot at a time, in order.'
                    : refusals.length
                      ? 'Nothing is sent while the note above stands.'
                      : !missing.length
                        ? 'Every shot is rendered and matches its line.'
                        : missing.length === 1
                          ? 'One generation of several minutes. Control and Enter starts it.'
                          : `${missing.length} generations of several minutes each. Control and Enter starts them.`}
              </span>
            </div>

            {changedCount || staleCount ? (
              <p className="mt-3 border-l-2 border-warning pl-2 text-caption text-ink-warning">
                {changedCount
                  ? changedCount === 1
                    ? 'One shot has changed since it was rendered. '
                    : `${changedCount} shots have changed since they were rendered. `
                  : ''}
                {staleCount
                  ? staleCount === 1
                    ? 'One shot was rendered before the shot above it changed. '
                    : `${staleCount} shots were rendered before the shot above them changed. `
                  : ''}
                Rendering what is missing brings them back into line.
              </p>
            ) : null}
          </div>

          <Assembly
            key={clips.map((c) => `${c.file.subfolder}/${c.file.filename}`).join('|')}
            clips={clips}
            shots={draft.shots.length}
            prefix={draft.prefix}
          />
        </div>

        {/* the bench -------------------------------------------------------- */}
        <div className="order-1 lg:order-2">
          <Bench
            draft={draft}
            families={families}
            family={family}
            blocked={offer?.blocked ?? []}
            shapes={shapes}
            lengthOptions={lengthOptions}
            samplers={cat?.samplers ?? []}
            schedulers={cat?.schedulers ?? []}
            houseNegative={houseNegative ?? ''}
            plan={plan}
            expert={expert}
            busy={busy}
            onPatch={patchBench}
            onPinAnchor={pinAnchor}
          />
        </div>
      </div>

      <KeyframePicker
        target={pinning}
        onClose={() => setPinning(null)}
        pictures={pictures}
        reelFrames={reelFrames}
      />
    </main>
  )
}

/** The reel's engine, for the shell's press ledger. */
export { reelRun } from '../components/reel'
