/**
 * The LoRA rack.
 *
 * What a LoRA is for, in one line: refine.ts gives a region enough pixels to
 * draw anatomy correctly, and a LoRA changes what the model thinks correct
 * looks like. Neither substitutes for the other, and the rack says so where the
 * user can read it rather than leaving it in a comment.
 *
 * The rack is two exports used together:
 *
 *   const rack = useLoraRack(family, model)     // state, persistence, fitness
 *   <LoraRack rack={rack} prompt={positive} />  // the stack the user edits
 *
 * and at queue time:
 *
 *   const derived = withLoras(def, rack.selection.specs)
 *
 * The hook is separate from the component because the desk needs the specs at
 * queue time, in a callback, where a component's props are not available. It
 * also means a desk can carry a LoRA stack without rendering the rack at all.
 *
 * WHAT THE RACK REFUSES TO DO. It never hands out a spec for a LoRA whose
 * architecture does not match the loaded checkpoint, whatever the saved stack
 * says. A stack is persisted per family and a family can carry several
 * checkpoints, so the model can change under a stack between sessions. The
 * failure that would cause is silent: ComfyUI patches the few keys that happen
 * to match, reports nothing, and renders noise.
 *
 * WHERE THE STRENGTHS COME FROM. Not from the catalogue's `recommended`, which
 * is the author's own claim on a model card, and not from taste. `measured.tsx`
 * holds Laplacian variance readings taken on this machine, on one base, at one
 * held seed, and the rack opens every anatomy LoRA at the strength those
 * readings support rather than at the figure the card asks for. It is a
 * SHARPNESS reading, and the rack says so beside every number it prints: it
 * settles a strength, it has no opinion about anatomy.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { canTakeLoras } from '../../lib/refine'
import type { FamilyDef } from '../../lib/registry'
import {
  EMPTY_LIBRARY,
  STACK_ADVICE_AT,
  addToStack,
  defaultStrength,
  fitFor,
  loadLoraLibrary,
  loadStack,
  missingTriggers,
  moveInStack,
  patchStack,
  removeFromStack,
  resolveStack,
  saveStack,
  size,
  targetFor,
  type LoraInfo,
  type LoraLibrary,
  type LoraStack,
  type LoraTarget,
  type ResolvedStack,
  type StackEntry,
} from '../../lib/loras'
import { Head, Quiet, RING, Tap } from './bits'
import {
  MeasurementNote,
  RECOMMENDED_RATIO,
  RECOMMENDED_STACK,
  isRecommendedStack,
  measurementsApply,
  openingStrength,
  predictStack,
  ratioText,
  type Prediction,
} from './measured'
import { Picker } from './Picker'
import { Row } from './Row'

export { Picker } from './Picker'
export { Row } from './Row'
export {
  ANATOMY_CAP,
  MEASURED,
  RECOMMENDED_STACK,
  capFor,
  predictStack,
  ratioAt,
  type Prediction,
} from './measured'

export type Rack = {
  family: FamilyDef | null
  model: string
  target: LoraTarget
  /** False when the graph has no single diffusion model loader to patch. */
  supported: boolean
  /** True on checkpoint families, where the text encoder is patched as well. */
  clipPatched: boolean
  lib: LoraLibrary
  loading: boolean
  error: string | null
  reload: () => void
  stack: LoraStack
  /** What withLoras() should be given, plus everything that was left out. */
  selection: ResolvedStack
  add: (info: LoraInfo) => void
  remove: (file: string) => void
  patch: (file: string, patch: Partial<StackEntry>) => void
  move: (from: number, to: number) => void
  clear: () => void
  /**
   * Replace the stack with the one the measurements recommend: anatomy-helper
   * at 0.4 with add-micro-details at 0.6, which measured 1.14x base sharpness.
   */
  useMeasured: () => void
  /** Whether both files of that stack are on disk. */
  measuredReady: { ready: boolean; missing: string[] }
  /**
   * Predicted sharpness of what will actually be sent, as a fraction of the
   * same frame with no LoRA. Null when nothing will be sent, and null on a
   * checkpoint the figures were never taken against.
   */
  prediction: Prediction | null
  /** False when the browser refused to persist, so the stack lasts the session. */
  persisted: boolean
}

/**
 * The library is read once per mount and shared by every rack in the tree
 * through a module level promise. /api/models walks the whole models root, and
 * re walking it on every tab change would cost a directory scan for data that
 * changes only when a download finishes.
 */
let shared: Promise<LoraLibrary> | null = null
const library = (reload = false): Promise<LoraLibrary> => {
  if (reload || !shared) shared = loadLoraLibrary()
  return shared
}

export function useLoraRack(family: FamilyDef | null, model: string): Rack {
  const [lib, setLib] = useState<LoraLibrary>(EMPTY_LIBRARY)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stack, setStack] = useState<LoraStack>([])
  const [persisted, setPersisted] = useState(true)
  const familyId = family?.id ?? ''
  const live = useRef(true)

  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  const pull = useCallback((reload: boolean) => {
    setLoading(true)
    library(reload)
      .then((next) => {
        if (!live.current) return
        setLib(next)
        setError(null)
      })
      .catch((err: unknown) => {
        if (!live.current) return
        setError(String((err as Error)?.message ?? err))
      })
      .finally(() => {
        if (live.current) setLoading(false)
      })
  }, [])

  useEffect(() => pull(false), [pull])

  // The stack belongs to the family, so switching family swaps the stack
  // rather than carrying one model's LoRAs onto another's weights.
  useEffect(() => {
    setStack(loadStack(familyId))
    setPersisted(true)
  }, [familyId])

  const commit = useCallback(
    (next: LoraStack) => {
      setStack(next)
      setPersisted(saveStack(familyId, next))
    },
    [familyId],
  )

  const target = useMemo(() => targetFor(family, model), [family, model])
  const supported = !!family && canTakeLoras(family)
  const clipPatched = useMemo(
    () =>
      !!family &&
      Object.values(family.graph).some((n) => n.class_type === 'CheckpointLoaderSimple'),
    [family],
  )

  const selection = useMemo(() => resolveStack(stack, lib, target), [stack, lib, target])

  const measuredReady = useMemo(() => {
    const missing = RECOMMENDED_STACK.filter(([file]) => !lib.byFile.get(file)?.installed).map(
      ([file]) => lib.byFile.get(file)?.label ?? file,
    )
    return { ready: missing.length === 0, missing }
  }, [lib])

  // Only against a base the readings can speak to. Everywhere else a figure
  // taken on Pony V6 would be a number with no evidence behind it, which is
  // exactly what this module exists to stop.
  const prediction = useMemo(
    () => (measurementsApply(target) ? predictStack(stack, lib.byFile) : null),
    [stack, lib, target],
  )

  return {
    family,
    model,
    target,
    supported,
    clipPatched,
    lib,
    loading,
    error,
    reload: () => pull(true),
    stack,
    selection,
    // A LoRA enters the stack at the strength that was measured here, not at
    // the one printed on its card. anatomy-helper is published at 0.6 and
    // measures 0.718 of base sharpness there; it is added at 0.4.
    add: (info) => {
      const next = addToStack(stack, info)
      if (next === stack) return
      const opening = openingStrength(info, defaultStrength(info))
      commit(patchStack(next, info.file, { strength: opening }))
    },
    remove: (file) => commit(removeFromStack(stack, file)),
    patch: (file, p) => commit(patchStack(stack, file, p)),
    move: (from, to) => commit(moveInStack(stack, from, to)),
    clear: () => commit([]),
    useMeasured: () =>
      commit(RECOMMENDED_STACK.map(([file, strength]) => ({ file, strength, enabled: true }))),
    measuredReady,
    prediction,
    persisted,
  }
}

export function LoraRack({
  rack,
  prompt = '',
  expert,
  onAddTriggers,
}: {
  rack: Rack
  /** The positive prompt, so a missing trigger word can be pointed out. */
  prompt?: string
  expert?: boolean
  /** Offered when the desk can paste trigger words into its prompt field. */
  onAddTriggers?: (tokens: string[]) => void
}) {
  const [picking, setPicking] = useState(false)
  const { lib, stack, selection, target, supported, prediction } = rack

  const inStack = useMemo(() => new Set(stack.map((e) => e.file)), [stack])
  const enabled = stack.filter((e) => e.enabled).length
  const onMeasured = isRecommendedStack(stack)
  const showMeasured = measurementsApply(target)
  const soft = !!prediction && !prediction.measured && prediction.ratio < 1
  const worstLabel = prediction?.worst
    ? (lib.byFile.get(prediction.worst.file)?.label ?? prediction.worst.file)
    : null
  const missing = useMemo(
    () => missingTriggers(stack, lib, prompt, target),
    [stack, lib, prompt, target],
  )

  if (!rack.family) return null

  if (!supported) {
    return (
      <section>
        <Head title="LoRAs" />
        <Quiet>
          This family loads a matched pair of weights, high noise and low noise, which take
          different LoRAs at different strengths. One flat stack cannot express that, so the rack
          is not offered here.
        </Quiet>
      </section>
    )
  }

  return (
    <section>
      <Head
        title="LoRAs"
        figure={stack.length ? `${enabled} of ${stack.length} on` : undefined}
        note="A LoRA changes what the model believes the subject looks like. Region refine changes how much detail fits. Anatomy usually needs both."
      />

      {rack.error ? (
        <p className="mb-2 text-caption text-error">
          The model folder could not be read: {rack.error}. The catalogue is still listed, and
          nothing can be added until the folder answers.
        </p>
      ) : null}

      {stack.length ? (
        <ul className="border-t border-grey-300">
          {stack.map((entry, i) => {
            const info = lib.byFile.get(entry.file)
            return (
              <Row
                key={entry.file}
                entry={entry}
                info={info}
                fit={
                  info
                    ? fitFor(info, target)
                    : { level: 'untested', why: 'This file is no longer in the LoRA folder.' }
                }
                index={i}
                count={stack.length}
                expert={expert}
                clipPatched={rack.clipPatched}
                onPatch={(p) => rack.patch(entry.file, p)}
                onMove={(to) => rack.move(i, to)}
                onRemove={() => rack.remove(entry.file)}
              />
            )
          })}
        </ul>
      ) : (
        <Quiet className="mb-2">
          No LoRAs. The checkpoint decides everything, including the anatomy it was never trained
          to draw.
        </Quiet>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Tap active={picking} onClick={() => setPicking((v) => !v)}>
          {picking ? 'done adding' : 'add a LoRA'}
        </Tap>
        {stack.length ? <Tap onClick={rack.clear}>clear</Tap> : null}
        <span className="ml-auto text-caption tabular-nums text-grey-500">
          {rack.loading ? 'reading the folder' : `${lib.installed} on disk, ${lib.available} to fetch`}
        </span>
      </div>

      {selection.bytes ? (
        <p className="mt-2 text-caption tabular-nums text-grey-700">
          {size(selection.bytes)} of extra weights, patched into the model before the first step.
        </p>
      ) : null}

      {/* ---- what the measurements say ------------------------------------ */}
      {showMeasured ? (
        <div className="mt-2 border-t-2 border-burgundy-900 pt-2">
          {prediction ? (
            <p className="text-caption leading-snug text-grey-700">
              <span className="tabular-nums font-semibold text-ink">
                {ratioText(prediction.ratio)} base sharpness
              </span>{' '}
              {prediction.measured
                ? 'for this exact stack, measured rather than modelled.'
                : 'predicted for this stack, from the measured singles. Expect it within about five percent.'}
              {prediction.unmeasured.length ? (
                <>
                  {' '}
                  {prediction.unmeasured.length} of the {prediction.unmeasured.length +
                    prediction.counted}{' '}
                  have never been measured here, so the figure covers only the rest.
                </>
              ) : null}
            </p>
          ) : null}

          {soft ? (
            <p className="mt-1 text-caption leading-snug text-warning">
              Below the base frame: this stack is predicted to come out softer than the same
              picture with no LoRA at all.{' '}
              {worstLabel
                ? `${worstLabel} is costing the most. Lower it, or raise Add micro details, which is the only file measured here that puts sharpness back.`
                : 'Lower an anatomy LoRA, or add Add micro details, which is the only file measured here that puts sharpness back.'}
            </p>
          ) : null}

          {!onMeasured ? (
            <p className="mt-1.5 text-caption leading-snug text-grey-700">
              The stack these readings recommend is Anatomy helper at 0.4 with Add micro details at
              0.6, which measured {ratioText(RECOMMENDED_RATIO)} base: sharper than no LoRA while
              still carrying the anatomy help.{' '}
              {rack.measuredReady.ready ? (
                <button
                  type="button"
                  className={`sg-link ${RING}`}
                  onClick={rack.useMeasured}
                  title={stack.length ? 'Replaces the stack you have now' : undefined}
                >
                  {stack.length ? 'Replace the stack with it' : 'Use it'}
                </button>
              ) : (
                <span className="italic">
                  Not on disk yet: {rack.measuredReady.missing.join(' and ')}. Add a LoRA below to
                  fetch {rack.measuredReady.missing.length > 1 ? 'them' : 'it'}.
                </span>
              )}
            </p>
          ) : null}

          <MeasurementNote target={target} />
        </div>
      ) : null}

      {enabled >= STACK_ADVICE_AT ? (
        <p className="mt-1 text-caption italic text-warning">
          {enabled} LoRAs at once. They all rewrite the same attention weights, so past four the
          later ones mostly cancel the earlier ones. If one stopped working, this is usually why.
        </p>
      ) : null}

      {missing.length ? (
        <p className="mt-2 text-caption text-grey-700">
          <span className="text-[0.625rem] font-semibold uppercase tracking-[0.18em] text-burgundy-900">
            missing trigger
          </span>{' '}
          {missing.join(', ')} is not in the prompt. Without it the LoRA loads and does close to
          nothing.{' '}
          {onAddTriggers ? (
            <button
              type="button"
              className={`sg-link ${RING}`}
              onClick={() => onAddTriggers(missing)}
            >
              Add to the prompt
            </button>
          ) : null}
        </p>
      ) : null}

      {selection.warnings.map((w) => (
        <p key={w.file} className="mt-2 text-caption leading-snug text-warning">
          <strong className="font-semibold">{w.label}</strong>: {w.why}
        </p>
      ))}

      {selection.dropped.map((d) => (
        <p key={d.file} className="mt-2 text-caption leading-snug text-error">
          <strong className="font-semibold">{d.label}</strong> is not being sent. {d.why}
        </p>
      ))}

      {!rack.persisted ? (
        <Quiet className="mt-2">
          This browser refused to save the stack, so it lasts until the tab closes.
        </Quiet>
      ) : null}

      {picking ? (
        <Picker
          lib={lib}
          target={target}
          inStack={inStack}
          onAdd={(info) => rack.add(info)}
          onFetched={rack.reload}
          onClose={() => setPicking(false)}
        />
      ) : null}
    </section>
  )
}
