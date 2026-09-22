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
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from 'react'

import { connect, fileUrl, objectInfo, optionsFor, type FileRef, type OutputFile } from '../lib/comfy'
import {
  annotatedRef,
  checkReel,
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
import { FAMILIES, defaultsFor, modelsOf, sidecarsOf, type FamilyDef, type Params } from '../lib/workflows'
import { go, parseRoute, useExpert } from '../components/shell'

import {
  Assembly,
  Bench,
  KeyframePicker,
  Kicker,
  Quiet,
  ReelProgress,
  Strip,
  grouped,
  reel,
  reelRun,
  seconds,
  shotsFromLines,
  useReel,
  type AssemblyClip,
  type KeyframeTarget,
  type NumSpec,
  type PinnedFrame,
  type ReelFamily,
  type ReelShot,
  type RunContext,
  type Shape,
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
  families: ReelFamily[]
  blocked: { label: string; why: string }[]
  samplers: string[]
  schedulers: string[]
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
  const info = await objectInfo()

  const clips = optionsFor(info, 'CLIPLoader', 'clip_name')
  const vaes = optionsFor(info, 'VAELoader', 'vae_name')
  const loras = optionsFor(info, 'LoraLoaderModelOnly', 'lora_name')
  const weights = new Set<string>([
    ...optionsFor(info, 'CheckpointLoaderSimple', 'ckpt_name'),
    ...optionsFor(info, 'UNETLoader', 'unet_name'),
    ...optionsFor(info, 'UnetLoaderGGUF', 'unet_name'),
  ])

  const families: ReelFamily[] = []
  const blocked: Catalogue['blocked'] = []

  for (const def of FAMILIES) {
    if (def.mode !== 'video') continue

    const { clip, vae } = sidecarsOf(def)
    const missing = [
      ...clip.filter((c) => !clips.includes(c)),
      ...(vae && !vaes.includes(vae) ? [vae] : []),
      ...modelsOf(def).filter((m) => !weights.has(m)),
      ...Object.values(def.graph)
        .map((n) => n.inputs['lora_name'])
        .filter((l): l is string => typeof l === 'string' && !loras.includes(l)),
    ]
    if (missing.length) {
      blocked.push({ label: def.label, why: `needs ${[...new Set(missing)].join(', ')}` })
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
    samplers: optionsFor(info, 'KSampler', 'sampler_name'),
    schedulers: optionsFor(info, 'KSampler', 'scheduler'),
  }
}

let cataloguePromise: Promise<Catalogue> | null = null

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

  const family = useMemo<ReelFamily | null>(() => {
    if (!cat || !cat.families.length) return null
    return cat.families.find((f) => f.def.id === draft.familyId) ?? cat.families[0] ?? null
  }, [cat, draft.familyId])

  // The family's own verified recipe, applied when the style changes. A reel is
  // one continuous piece, so the whole bench follows the family rather than
  // carrying settings across from a different model.
  useEffect(() => {
    if (!family) return
    if (draft.familyId === family.def.id && draft.model === family.model) return
    const d = defaultsFor(family.def, family.model)
    reel.patch({
      familyId: family.def.id,
      model: family.model,
      width: d.width,
      height: d.height,
      fps: d.fps || 24,
      length: snapLength(d.length || 81),
      steps: d.steps,
      cfg: d.cfg,
      sampler: d.sampler,
      scheduler: d.scheduler,
      negative: null,
    })
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
    })
  }, [family, params, draft.shots, draft.length, draft.anchor, draft.reanchorEvery, draft.prefix])

  const issues = useMemo(() => checkReel(plan.jobs), [plan.jobs])
  const order = useMemo(() => draft.shots.map((s) => s.id), [draft.shots])
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
    return { familyLabel: label, modelLabel: model, compositionFor }
  }, [family, draft])

  // --- running -------------------------------------------------------------

  const renderAll = useCallback(
    (opts: { force?: boolean } = {}) => {
      if (!context || busy || blanks.length || !plan.jobs.length) return
      reelRun.renderAll(order, plan.jobs, context, opts)
    },
    [context, busy, blanks.length, plan.jobs, order],
  )

  const renderOne = useCallback(
    (index: number) => {
      if (!context || busy) return
      reelRun.renderOne(index, order, plan.jobs, context)
    },
    [context, busy, order, plan.jobs],
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
      run.order
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
    [run],
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
    const index = run.order.indexOf(watching)
    const job = index >= 0 ? plan.jobs[index] : null
    const entry = state.entryId ? (records.find((r) => r.id === state.entryId) ?? null) : null
    return {
      file: clip as FileRef,
      entry,
      frames: job?.params.length ?? 0,
      fps: draft.fps,
      index,
    }
  }, [watching, run, plan.jobs, records, draft.fps])

  const clips = useMemo<AssemblyClip[]>(
    () =>
      run.order
        .map((id, i) => {
          const state = run.states[id]
          if (!state?.clip || state.status !== 'done') return null
          return {
            index: i,
            label: `Shot ${i + 1}`,
            file: state.clip as OutputFile,
            frames: state.frames,
            durationMs: state.durationMs,
          }
        })
        .filter((c): c is AssemblyClip => c !== null),
    [run],
  )

  const stale = clips.length && run.order.some((id) => run.states[id]?.stale)

  // --- the page ------------------------------------------------------------

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
                  disabled={!pasted.trim()}
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

          {family ? (
            <Strip
              shots={draft.shots}
              jobs={plan.jobs}
              run={run}
              fps={draft.fps}
              reelLength={draft.length}
              lengthOptions={lengthOptions}
              expert={expert}
              busy={busy}
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

          {issues.length ? (
            <div className="mb-4 space-y-1">
              {issues.map((issue) => (
                <p key={issue} className="border-l-2 border-warning pl-2 text-caption text-[#78350f]">
                  {issue}
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
                disabled={busy || blanks.length > 0 || !plan.jobs.length || !family}
                onClick={() => renderAll()}
              >
                {busy ? 'On the press' : clips.length ? 'Render what is missing' : 'Render the reel'}
              </button>

              {clips.length && !busy ? (
                <Quiet onClick={() => renderAll({ force: true })}>Render every shot again</Quiet>
              ) : null}
              {run.status !== 'idle' && !busy ? <Quiet onClick={() => reelRun.clear()}>Clear the run</Quiet> : null}

              <span className="text-caption italic text-grey-500">
                {blanks.length
                  ? blanks.length === 1
                    ? `Shot ${blanks[0]} has no line yet.`
                    : `Shots ${blanks.join(', ')} have no lines yet.`
                  : busy
                    ? 'One shot at a time, in order.'
                    : `${plan.jobs.length} generations of several minutes each. Control and Enter starts them.`}
              </span>
            </div>

            {stale ? (
              <p className="mt-3 border-l-2 border-warning pl-2 text-caption text-[#78350f]">
                Some shots were rendered before the shot above them changed. Rendering what is missing brings them back
                into line.
              </p>
            ) : null}
          </div>

          <Assembly clips={clips} fps={draft.fps} shots={draft.shots.length} prefix={draft.prefix} />
        </div>

        {/* the bench -------------------------------------------------------- */}
        <div className="order-1 lg:order-2">
          <Bench
            draft={draft}
            families={cat?.families ?? []}
            family={family}
            blocked={cat?.blocked ?? []}
            shapes={shapes}
            lengthOptions={lengthOptions}
            samplers={cat?.samplers ?? []}
            schedulers={cat?.schedulers ?? []}
            houseNegative={houseNegative ?? ''}
            plan={plan}
            expert={expert}
            busy={busy}
            onPatch={(p) => reel.patch(p)}
            onPinAnchor={pinAnchor}
            onRerollSeed={() => reel.patch({ seed: randomSeed(), seedLocked: false })}
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
