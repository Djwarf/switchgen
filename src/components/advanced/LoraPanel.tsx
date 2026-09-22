/**
 * The full LoRA rack, opened on the stack the recipe resolved.
 *
 * The simple screen has one anatomy control with three words on it. This is
 * what that control is shorthand for: every LoRA in the chain, its strength,
 * its order in the chain, whether it is on, and the whole installed library
 * behind an add button. Nothing was removed from the old rack to make the
 * simple control; it was moved here.
 *
 * ORDER IS A CONTROL, NOT DECORATION. Every LoraLoader patches the model the
 * previous one produced, so a style LoRA placed after an anatomy LoRA paints
 * over it. The row keeps its move buttons for that reason, and they are buttons
 * rather than a drag handle because a keyboard has to reach them.
 *
 * WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT. Every ratio printed here is
 * Laplacian variance against the same frame with no LoRA on it, on one
 * checkpoint, at one held seed. That is SHARPNESS. It is not a claim that the
 * anatomy came out right. A LoRA can draw a hand correctly and soften the skin
 * around it, and this reading will call that a loss. It settles a strength; it
 * has no opinion about the picture.
 *
 * THE RACK IS BORROWED WHOLE. ../loras/Row and ../loras/Picker are the existing
 * components, with their measured caps, their fit badges and their download
 * jobs. Reimplementing any of that here would have been a second rack to keep
 * correct.
 */
import { useMemo, useState } from 'react'

import {
  STACK_ADVICE_AT,
  addToStack,
  fitFor,
  moveInStack,
  patchStack,
  removeFromStack,
  size,
  type LoraInfo,
  type LoraLibrary,
  type LoraStack,
  type LoraTarget,
} from '../../lib/loras'
import type { Plan } from '../../lib/recipe'
import { Picker } from '../loras/Picker'
import { Row as LoraRow } from '../loras/Row'
import { Caution, Fault, Head, Kicker, Link, Note, Quiet, Source, ratio } from './bits'
import { decidedStack, type Settled } from './overrides'

export function LoraPanel({
  plan,
  settled,
  lib,
  target,
  overridden,
  onStack,
  onRestore,
  onLibraryReload,
  measuredOn,
}: {
  plan: Plan
  settled: Settled
  lib: LoraLibrary
  target: LoraTarget
  /** True when the stack in force is not the one the recipe resolved. */
  overridden: boolean
  onStack: (next: LoraStack) => void
  onRestore: () => void
  /** A file landed on disk. The desk re reads /api/models. */
  onLibraryReload?: () => void
  measuredOn: string
}) {
  const [picking, setPicking] = useState(false)
  const stack = settled.stack
  const decided = useMemo(() => decidedStack(plan), [plan])
  const inStack = useMemo(() => new Set(stack.map((e) => e.file)), [stack])

  // The checkpoint families bundle their own text encoder, so LoraLoader
  // patches MODEL and CLIP together. The separate encoder families patch only
  // the diffusion model, and the row says which it is.
  const clipPatched = useMemo(
    () => Object.values(plan.base.graph).some((n) => n.class_type === 'CheckpointLoaderSimple'),
    [plan.base],
  )

  const enabled = stack.filter((e) => e.enabled).length
  const bytes = useMemo(
    () =>
      stack.reduce((sum, e) => {
        if (!e.enabled) return sum
        return sum + (lib.byFile.get(e.file)?.bytes ?? 0)
      }, 0),
    [stack, lib],
  )

  if (!plan.capabilities.loras) {
    return (
      <section className="mb-7">
        <Head title="Add-ons" />
        <Note>
          {plan.label} works in two halves that need different settings, and add-ons cannot yet be
          set separately for each half. So they are not offered for this model.
        </Note>
      </section>
    )
  }

  return (
    <section className="mb-7">
      <Head
        title="Add-ons"
        figure={stack.length ? `${enabled} of ${stack.length} on` : 'none'}
        note="An add-on teaches the model a style or a subject it does not draw well on its own. A detail pass just gives it more pixels to draw in. They fix different things."
      />

      {overridden ? (
        <p className="mb-3 flex flex-wrap items-baseline gap-x-2 border-l-2 border-burgundy-900 pl-2.5 text-caption text-grey-700">
          <span className="text-overline font-semibold uppercase tracking-[0.14em] text-burgundy-900">
            Set by hand
          </span>
          <span>
            {decided.length
              ? `${decided.length} add-on${decided.length === 1 ? ' was' : 's were'} picked for you. You have changed that.`
              : 'Nothing was picked for you here. You added these yourself.'}
          </span>
          <Link onClick={onRestore}>Go back to the picks</Link>
        </p>
      ) : null}

      {plan.sharpness && !overridden ? (
        <p className="mb-3 text-caption leading-snug text-grey-700">
          <Kicker>Measured</Kicker>
          These add-ons came out {ratio(plan.sharpness.ratio)} as sharp as the same picture with none
          of them on. {plan.sharpness.verdict}{' '}
          {plan.sharpness.onMeasuredModel
            ? `Measured on this checkpoint.`
            : `Measured on ${measuredOn}, not on this file, so treat it as a starting point.`}
        </p>
      ) : null}
      {overridden ? (
        <Note>
          The stack is no longer the measured one, so no sharpness figure applies to what will
          actually run. The per LoRA readings on each row still do.
        </Note>
      ) : null}

      {stack.length ? (
        <ul className="mt-2 border-t border-grey-300">
          {stack.map((entry, i) => {
            const info = lib.byFile.get(entry.file)
            return (
              <LoraRow
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
                expert
                clipPatched={clipPatched}
                onPatch={(patch) => onStack(patchStack(stack, entry.file, patch))}
                onMove={(to) => onStack(moveInStack(stack, i, to))}
                onRemove={() => onStack(removeFromStack(stack, entry.file))}
              />
            )
          })}
        </ul>
      ) : (
        <Note>No add-ons. The model is drawing this on its own, which is usually fine.</Note>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Quiet onClick={() => setPicking((v) => !v)} pressed={picking}>
          {picking ? 'Close' : 'Browse all add-ons'}
        </Quiet>
        {stack.length ? <Quiet onClick={() => onStack([])}>Remove all</Quiet> : null}
        {bytes ? (
          <span className="text-caption tabular-nums text-grey-500">{size(bytes)} of extra files to load</span>
        ) : null}
      </div>

      {stack.length > STACK_ADVICE_AT ? (
        <div className="mt-2">
          <Caution>
            Past {STACK_ADVICE_AT} add-ons they start fighting each other: they all change the same
              part of the model, so a newer one tends to undo the last. If one seems to have
              stopped working, this is usually why. Nothing is blocked - it is a warning, not a limit.
          </Caution>
        </div>
      ) : null}

      {picking ? (
        <div className="mt-3 border border-grey-300 p-3">
          <Picker
            lib={lib}
            target={target}
            inStack={inStack}
            onAdd={(info: LoraInfo) => onStack(addToStack(stack, info))}
            onFetched={() => onLibraryReload?.()}
            onClose={() => setPicking(false)}
          />
        </div>
      ) : null}

      {settled.dropped.length ? (
        <div className="mt-3">
          <Fault>
            Not sent: {settled.dropped.map((d) => `${d.label}, ${lower(d.why)}`).join('; ')}
          </Fault>
        </div>
      ) : null}

      {settled.cautions.length ? (
        <div className="mt-2">
          <Caution>
            Untested crossing, sent anyway:{' '}
            {settled.cautions.map((w) => `${w.label}, ${lower(w.why)}`).join('; ')}
          </Caution>
        </div>
      ) : null}

      {plan.missingLoras.length ? (
        <div className="mt-3">
          <Kicker>Would help here, not downloaded yet</Kicker>
          <ul className="mt-1 border-t border-grey-300">
            {plan.missingLoras.map((m) => (
              <li key={m.file} className="border-b border-grey-300 py-1.5">
                <span className="block text-caption text-ink">{m.label}</span>
                <span className="block text-caption italic leading-snug text-grey-500">{m.why}</span>
              </li>
            ))}
          </ul>
          <Note>Download them from the list above and they start being used automatically.</Note>
        </div>
      ) : null}

      {plan.refineLoras.length ? (
        <details className="mt-4">
          <summary className="cursor-pointer text-caption text-grey-500">
            Saved for touch-ups ({plan.refineLoras.length})
          </summary>
          <Note>
            These were made from close-up photographs, so they do almost nothing across a whole
              figure. They are offered on the finished picture instead, where you can paint over the
              exact area and see whether it needs one. Adding one here applies it to the whole
              picture, which is allowed and rarely what it was made for.
          </Note>
          <ul className="mt-1 border-t border-grey-300">
            {plan.refineLoras.map((l) => {
              const info = lib.byFile.get(l.file)
              const already = inStack.has(l.file)
              return (
                <li key={l.file} className="border-b border-grey-300 py-1.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="min-w-0 text-caption text-ink">{l.label}</span>
                    <span className="flex shrink-0 items-baseline gap-2">
                      <Source measured={l.measured} />
                      <span className="text-caption tabular-nums text-grey-700">
                        {l.strength.toFixed(2)}
                      </span>
                    </span>
                  </div>
                  <span className="block text-caption italic leading-snug text-grey-500">{l.why}</span>
                  {info && !already ? (
                    <p className="mt-1">
                      <Link onClick={() => onStack(addToStack(stack, info))}>
                        Put it in the first render anyway
                      </Link>
                    </p>
                  ) : null}
                </li>
              )
            })}
          </ul>
        </details>
      ) : null}
    </section>
  )
}

/** Sentence fragments read better lowercased when they follow a comma. */
function lower(why: string): string {
  return why.charAt(0).toLowerCase() + why.slice(1).replace(/\.$/, '')
}
