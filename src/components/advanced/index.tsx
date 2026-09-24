/**
 * The advanced panel: the whole apparatus, behind one link.
 *
 * This is the other half of the bargain the simple screen makes. That screen
 * asks for a prompt, a look and an anatomy level, and decides the other forty
 * things from measurement. It can only be that calm if none of the forty were
 * thrown away, and this is where they went.
 *
 * THE PANEL IS ALLOWED TO BE DENSE. It is a reference page, not a form to fill
 * in. Every row opens on the value the recipe decided, so arriving here is
 * continuous with the simple view rather than landing in a different
 * application with everything blank. Change one and the row marks itself, names
 * what was decided, and offers it back in one press.
 *
 * FIVE SECTIONS, IN THE ORDER A DECISION IS MADE:
 *
 *   Decided     every choice the recipe made, and what it rests on
 *   Model       the ranking, the hardware verdicts, and the pin
 *   LoRAs       the full rack, opened on the resolved stack
 *   Sampling    steps, CFG, sampler, scheduler, size, seed, shift, clip skip,
 *               both prompts, and how many
 *   Passes      face, hands and the two pass render, before the picture
 *   Workflow    the JSON that will be sent
 *
 * HOW THE DESK USES IT:
 *
 *   const ov = useOverrides()
 *   const recipe = decide({ prompt, look, anatomy, ...probes })
 *   const settled = recipe.ok ? settle(recipe, ov.value, lib) : null
 *   // queue: instantiate(settled.def, settled.params), or buildGraph(settled)
 *   <AdvancedPanel recipe={recipe} overrides={ov.value} onOverrides={ov.set} ... />
 *
 * THE MODEL IS THE ONE EXCEPTION and it is documented at length in
 * ModelPanel.tsx: picking a weight file asks the desk to pin it and run
 * decide() again with that file as the only candidate, rather than patching a
 * filename into a plan that was reasoned out for a different one.
 */
import { useCallback, useMemo, useState } from 'react'

import { EMPTY_LIBRARY, type LoraLibrary, type LoraStack } from '../../lib/loras'
import { MEASURED, MEASURED_ON, type Recipe } from '../../lib/recipe'
import { Caution, Fault, Head, Kicker, Link, Note, Source } from './bits'
import { CataloguePanel } from './CataloguePanel'
import { LoraPanel } from './LoraPanel'
import { ModelPanel, type Unloadable } from './ModelPanel'
import {
  NO_OVERRIDES,
  clearOverride,
  overrideKeys,
  settle,
  setOverride,
  targetOf,
  type Overrides,
  type Passes,
} from './overrides'
import { PassPanel } from './PassPanel'
import { SamplingPanel } from './SamplingPanel'
import { WorkflowPeek } from './WorkflowPeek'

export { CataloguePanel } from './CataloguePanel'
export { LoraPanel } from './LoraPanel'
export { ModelPanel, type Unloadable } from './ModelPanel'
export { PassPanel } from './PassPanel'
export { SamplingPanel } from './SamplingPanel'
export { WorkflowPeek } from './WorkflowPeek'
export {
  BOUNDS,
  MEGAPIXELS,
  NO_OVERRIDES,
  NO_PASSES,
  SHAPES,
  anyOverride,
  buildGraph,
  clearOverride,
  decidedStack,
  maxSideOf,
  overrideKeys,
  preLoraDef,
  sanitiseOverrides,
  settle,
  setOverride,
  shapesFor,
  targetOf,
  type OverrideKey,
  type Overrides,
  type Passes,
  type Settled,
} from './overrides'

// ---------------------------------------------------------------------------
// The hook
// ---------------------------------------------------------------------------

export type OverrideStore = {
  value: Overrides
  set: (next: Overrides) => void
  one: <K extends keyof Overrides>(key: K, value: Overrides[K]) => void
  clear: (key: keyof Overrides) => void
  reset: () => void
  any: boolean
  count: number
}

/**
 * Overrides, held wherever the desk wants them.
 *
 * Never kept past the tab. An override is a statement about this picture, and
 * a CFG of 12 restored from last Tuesday into a recipe that decided 5 would be
 * the exact failure this whole redesign exists to end. The desk may keep them
 * for its own tab, so a reload does not drop settings the reader loaded (see
 * sanitiseOverrides), and that store dies with the tab.
 */
export function useOverrides(initial: Overrides = NO_OVERRIDES): OverrideStore {
  const [value, set] = useState<Overrides>(initial)
  const one = useCallback(
    <K extends keyof Overrides>(key: K, v: Overrides[K]) => set((prev) => setOverride(prev, key, v)),
    [],
  )
  const clear = useCallback((key: keyof Overrides) => set((prev) => clearOverride(prev, key)), [])
  const reset = useCallback(() => set(NO_OVERRIDES), [])
  const keys = overrideKeys(value)
  return { value, set, one, clear, reset, any: keys.length > 0, count: keys.length }
}

// ---------------------------------------------------------------------------
// The panel
// ---------------------------------------------------------------------------

export function AdvancedPanel({
  recipe,
  overrides,
  onOverrides,
  lib = EMPTY_LIBRARY,
  onLibraryReload,
  samplers = [],
  schedulers = [],
  pinnedModel = null,
  onPinModel,
  faultNode = null,
  onCatalogueChange,
  onDropAddOn,
  unloadable = [],
  onClose,
}: {
  recipe: Recipe
  overrides: Overrides
  onOverrides: (next: Overrides) => void
  /** From loadLoraLibrary(). Empty until it lands, and the rack says so. */
  lib?: LoraLibrary
  onLibraryReload?: () => void
  /** From ComfyUI's /object_info, through optionsFor(). Empty until it lands. */
  samplers?: readonly string[]
  schedulers?: readonly string[]
  /** The weight file the desk has pinned, when the reader pinned one. */
  pinnedModel?: string | null
  /** Pin a file and decide again with it, or null to drop the pin. */
  onPinModel?: (model: string | null) => void
  faultNode?: string | null
  /** A family's files landed; the desk should re-read what is installed. */
  onCatalogueChange?: () => void
  /**
   * Withdraw an add-on the reader added on the main screen. Removing one here
   * used to set only a stack override, which "Go back to the picks" undid
   * while the add stayed on file, so the add-on came straight back.
   */
  onDropAddOn?: (file: string) => void
  /**
   * Weight files ComfyUI lists that cannot load here, with the reason: a text
   * encoder or VAE the graph names is missing, or the node pack that reads it.
   * The ranking leaves them out, so they are named in the model panel rather
   * than offered and then refused by ComfyUI.
   */
  unloadable?: readonly Unloadable[]
  /** Back to the three questions. */
  onClose: () => void
}) {
  const set = useCallback(
    <K extends keyof Overrides>(key: K, value: Overrides[K]) =>
      onOverrides(setOverride(overrides, key, value)),
    [overrides, onOverrides],
  )
  const clear = useCallback(
    (key: keyof Overrides) => onOverrides(clearOverride(overrides, key)),
    [overrides, onOverrides],
  )

  const plan = recipe.ok ? recipe : null
  const settled = useMemo(() => (plan ? settle(plan, overrides, lib) : null), [plan, overrides, lib])
  const target = useMemo(() => (plan ? targetOf(plan) : null), [plan])
  const loraOverridden = overrides.loras !== undefined
  const touched = overrideKeys(overrides)

  return (
    <div className="text-small">
      <div className="mb-5 flex items-baseline justify-between gap-3 border-b-2 border-burgundy-900 pb-1.5">
        <Kicker tone="burgundy">Everything</Kicker>
        <Link className="text-caption" onClick={onClose}>
          ◂ Back to the three questions
        </Link>
      </div>

      {touched.length ? (
        <p className="mb-5 flex flex-wrap items-baseline gap-x-2 border-l-2 border-burgundy-900 pl-2.5 text-caption text-grey-700">
          <span className="text-overline font-semibold uppercase tracking-[0.14em] text-burgundy-900">
            {touched.length} set by hand
          </span>
          <span>The rest is still the recipe’s.</span>
          <Link onClick={() => onOverrides(NO_OVERRIDES)}>Put all of it back</Link>
        </p>
      ) : null}

      {recipe.warnings.length ? (
        <div className="mb-5 space-y-1.5">
          {recipe.warnings.map((w, i) => (
            <Caution key={i}>{w}</Caution>
          ))}
        </div>
      ) : null}

      {!recipe.ok ? (
        <section className="mb-7">
          <Head title="Nothing can run this" />
          <Fault>{recipe.reason}</Fault>
          <Note>
            The lists below say what is installed, what cannot load here and why, what will not fit
            in this machine’s memory, and what is on disk with no verified graph behind it. The
            catalogue under them fetches a model that can run.
          </Note>
          <div className="mt-4">
            <ModelPanel
              report={recipe.report}
              familyId={null}
              model={null}
              label={null}
              pinned={pinnedModel}
              onPin={(m) => onPinModel?.(m)}
              measured={MEASURED_ON}
              unloadable={unloadable}
            />
          </div>
          {/* The one way to a first model on a machine with none. It used to
              appear only once a recipe could already run, which is exactly
              when it was least needed. */}
          {onCatalogueChange ? <CataloguePanel modes={['image', 'edit']} onInstalled={onCatalogueChange} /> : null}
        </section>
      ) : null}

      {plan && settled && target ? (
        <>
          <Decided recipe={recipe} />

          <ModelPanel
            report={plan.report}
            familyId={plan.familyId}
            model={plan.model}
            label={plan.label}
            pinned={pinnedModel}
            onPin={(m) => onPinModel?.(m)}
            measured={MEASURED_ON}
            unloadable={unloadable}
          />

          {onCatalogueChange ? <CataloguePanel modes={['image', 'edit']} onInstalled={onCatalogueChange} /> : null}

          <LoraPanel
            plan={plan}
            settled={settled}
            lib={lib}
            target={target}
            overridden={loraOverridden}
            onStack={(next: LoraStack) => set('loras', next)}
            onRestore={() => clear('loras')}
            onDropAddOn={onDropAddOn}
            onLibraryReload={onLibraryReload}
            measuredOn={MEASURED_ON}
          />

          <SamplingPanel
            plan={plan}
            settled={settled}
            overrides={overrides}
            set={set}
            clear={clear}
            samplers={samplers}
            schedulers={schedulers}
          />

          <PassPanel
            plan={plan}
            settled={settled}
            onPasses={(next: Passes) => set('passes', next)}
          />

          <WorkflowPeek settled={settled} faultNode={faultNode} />

          <Method />
        </>
      ) : null}
    </div>
  )
}

// ---------------------------------------------------------------------------
// What was decided
// ---------------------------------------------------------------------------

/**
 * The decision log, in the order the decisions were made.
 *
 * Each line carries whether it rests on a number from the measurement table or
 * on a judgement. That badge is the point of the section: it is the difference
 * between "0.4 because a run says 0.5 loses a quarter of the sharpness" and
 * "0.4 because it looked about right", and a reader who cannot tell which they
 * are looking at has no reason to trust either.
 */
function Decided({ recipe }: { recipe: Recipe }) {
  const [open, setOpen] = useState(true)
  const notes = recipe.notes
  const measured = notes.filter((n) => n.measured).length

  return (
    <section className="mb-7">
      <Head
        title="What was decided"
        figure={`${notes.length} decisions`}
        action={
          <Link className="text-caption" onClick={() => setOpen((v) => !v)}>
            {open ? 'hide' : 'show'}
          </Link>
        }
        note={
          measured
            ? `${measured} of them rest on a measured run. The rest are routing: what fits, what attaches to what, what the file was trained with.`
            : 'These are routing decisions: what fits, what attaches to what, what the file was trained with.'
        }
      />
      {open ? (
        <ul className="border-t border-grey-300">
          {notes.map((n, i) => (
            <li key={i} className="flex items-start gap-2 border-b border-grey-300 py-1.5">
              <span className="w-16 shrink-0 pt-px text-overline font-semibold uppercase tracking-[0.14em] text-grey-500">
                {n.kind}
              </span>
              <span className="min-w-0 flex-1 text-caption leading-snug text-grey-700">{n.text}</span>
              {n.measured ? (
                <span className="shrink-0">
                  <Source measured />
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  )
}

// ---------------------------------------------------------------------------
// The method
// ---------------------------------------------------------------------------

/**
 * How every ratio in this panel was taken, and the one thing it does not say.
 *
 * Printed at the foot rather than beside each number, because it is the same
 * caveat every time and repeating it eleven times would train the reader to
 * stop reading it.
 */
function Method() {
  const h = MEASURED.helperCurve
  return (
    <section className="mb-4 border-t border-grey-300 pt-4">
      <Kicker>How the figures were taken</Kicker>
      <p className="mt-1 text-caption leading-snug text-grey-700">{MEASURED.method}</p>
      <p className="mt-1.5 text-caption leading-snug text-grey-700">
        The same picture with no add-ons at all measured {MEASURED.baselineLaplacian}. Every ratio
        printed in this panel is against that number.
      </p>
      <p className="mt-1.5 text-caption italic leading-snug text-grey-700">{MEASURED.metricCaveat}</p>
      <div className="mt-2 border-t border-grey-300 pt-2">
        <Kicker>Why anatomy help is capped at 0.4</Kicker>
        <p className="mt-1 text-caption tabular-nums leading-snug text-grey-700">
          It degrades in one direction only:{' '}
          {h.map((pt) => `${pt.ratio.toFixed(3)} at ${pt.strength}`).join(', ')}. The only restorer
          measured is add micro details, which came out at {MEASURED.microDetailsAlone.ratio}x on its
          own at {MEASURED.microDetailsAlone.strength}. Raise the cap on any row if you want to, and
          the row will keep printing what was read there.
        </p>
      </div>
      <p className="mt-2 text-caption italic leading-snug text-grey-500">{MEASURED.transferNote}</p>
    </section>
  )
}
