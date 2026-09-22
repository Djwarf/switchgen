/**
 * Everything the sampler takes, and the two settings that have no binding.
 *
 * Steps, CFG, sampler, scheduler, shape, width, height, seed and its lock,
 * shift, clip skip, the negative prompt, the positive exactly as it will be
 * sent, and how many pictures to run. Image to image adds its strength and its
 * output budget. Nothing here is new: this is the old expert margin, with one
 * thing added to every row, which is the value the recipe decided and a link
 * back to it.
 *
 * SHIFT AND CLIP SKIP ARE CONDITIONAL and the test is the graph itself rather
 * than a hand kept list of families. instantiate() writes both by class_type,
 * so a family whose graph carries no ModelSampling node has no shift to set and
 * the row is not drawn. Offering a control that is silently discarded is worse
 * than not offering it, because it is then filed in the archive as though it
 * had been applied.
 *
 * WHY THE POSITIVE IS HERE AT ALL. The prompt is the default screen's business
 * and stays there. What is here is the finished string: the quality prefix the
 * file was trained with, the reader's words, and one trigger token per LoRA
 * that needs one. That string is what ComfyUI receives, and the old desk let it
 * be edited through a separate quality words field. It is one field now, and it
 * is the true one.
 */
import { defaultsFor, graphClipSkip, graphShift } from '../../lib/workflows'
import type { Plan } from '../../lib/recipe'
import {
  Caution,
  Chips,
  Choice,
  Head,
  Kicker,
  NumberInput,
  Row,
  times,
} from './bits'
import {
  BOUNDS,
  MEGAPIXELS,
  maxSideOf,
  shapesFor,
  type Overrides,
  type Settled,
} from './overrides'

export function SamplingPanel({
  plan,
  settled,
  overrides,
  set,
  clear,
  samplers,
  schedulers,
}: {
  plan: Plan
  settled: Settled
  overrides: Overrides
  set: <K extends keyof Overrides>(key: K, value: Overrides[K]) => void
  clear: (key: keyof Overrides) => void
  /** From ComfyUI's /object_info. Empty until it lands, and the field says so. */
  samplers: readonly string[]
  schedulers: readonly string[]
}) {
  const p = settled.params
  const decided = plan.params
  const ov = overrides
  const on = (k: keyof Overrides) => ov[k] !== undefined

  const b = plan.base.bindings
  const hasScheduler = !!b.scheduler
  const hasNegative = !!b.negative
  // The same test instantiate() makes, on the family graph rather than on a
  // list of family names that would rot the next time the registry is rebuilt.
  const hasShift = graphShift(plan.base.graph) !== null
  const hasClipSkip = graphClipSkip(plan.base.graph) !== null

  const i2i = !!p.image
  const sized = !i2i
  const maxSide = maxSideOf(plan)
  const shapes = shapesFor(plan, maxSide)
  const shape = shapes.find((s) => s.width === p.width && s.height === p.height) ?? null
  const mp = (p.width * p.height) / 1e6

  const house = defaultsFor(plan.base, plan.model).negative ?? ''
  const alt = altOf(plan)

  return (
    <section className="mb-7">
      <Head
        title="Sampling"
        figure={`${p.steps} steps · CFG ${p.cfg.toFixed(1)}`}
        note="Every value opens on what the recipe decided. Change one and the row says so, and offers it back."
      />

      <Row
        label="Steps"
        id="adv-steps"
        hint="more steps, more time"
        overridden={on('steps')}
        decided={decided.steps}
        onRestore={() => clear('steps')}
      >
        <NumberInput
          id="adv-steps"
          value={p.steps}
          min={BOUNDS.steps.min}
          max={BOUNDS.steps.max}
          round="int"
          onChange={(v) => set('steps', v)}
        />
      </Row>

      <Row
        label="CFG"
        id="adv-cfg"
        hint="how hard it follows the words"
        overridden={on('cfg')}
        decided={decided.cfg.toFixed(1)}
        onRestore={() => clear('cfg')}
      >
        <NumberInput
          id="adv-cfg"
          value={p.cfg}
          min={BOUNDS.cfg.min}
          max={BOUNDS.cfg.max}
          step={BOUNDS.cfg.step}
          onChange={(v) => set('cfg', v)}
        />
      </Row>

      <Row
        label="Sampler"
        id="adv-sampler"
        hint={samplers.length ? undefined : 'ComfyUI has not listed these yet'}
        overridden={on('sampler')}
        decided={decided.sampler}
        onRestore={() => clear('sampler')}
      >
        <Choice
          id="adv-sampler"
          value={p.sampler}
          options={samplers}
          onChange={(v) => set('sampler', v)}
        />
      </Row>

      {hasScheduler ? (
        <Row
          label="Scheduler"
          id="adv-scheduler"
          hint={schedulers.length ? undefined : 'ComfyUI has not listed these yet'}
          overridden={on('scheduler')}
          decided={decided.scheduler}
          onRestore={() => clear('scheduler')}
        >
          <Choice
            id="adv-scheduler"
            value={p.scheduler}
            options={schedulers}
            onChange={(v) => set('scheduler', v)}
          />
        </Row>
      ) : null}

      {alt ? (
        <button
          type="button"
          onClick={() => {
            set('sampler', alt.sampler)
            set('scheduler', alt.scheduler)
            set('steps', alt.steps)
            set('cfg', alt.cfg)
          }}
          className="mb-3 w-full cursor-pointer border border-grey-300 px-2 py-1.5 text-left text-caption text-grey-700 transition-colors hover:border-ink hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
        >
          <Kicker>Use the author’s alternative</Kicker>
          <span className="tabular-nums">
            {alt.sampler} / {alt.scheduler}, {alt.steps} steps, CFG {alt.cfg.toFixed(1)}
          </span>
        </button>
      ) : null}

      {sized ? (
        <>
          <Row
            label="Shape"
            hint={`${times(p.width, p.height)} · ${mp.toFixed(2)} megapixels${
              shape?.maker ? ' · the maker’s shape' : ''
            }`}
            overridden={on('width') || on('height')}
            decided={times(decided.width, decided.height)}
            onRestore={() => {
              clear('width')
              clear('height')
            }}
            note={
              maxSide
                ? `This file’s card caps the long edge at ${maxSide}, so the buckets are scaled to fit it.`
                : undefined
            }
          >
            <Chips
              ariaLabel="Shape"
              value={shape?.key ?? ''}
              options={shapes.map((s) => ({
                value: s.key,
                label: s.label,
                title: `${s.label} · ${times(s.width, s.height)}`,
              }))}
              onChange={(key) => {
                const next = shapes.find((s) => s.key === key)
                if (!next) return
                set('width', next.width)
                set('height', next.height)
              }}
            />
            <div className="mt-2 flex items-end gap-3">
              <label className="block w-24">
                <span className="mb-1 block text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
                  Width
                </span>
                <NumberInput
                  id="adv-width"
                  value={p.width}
                  min={BOUNDS.size.min}
                  max={BOUNDS.size.max}
                  step={BOUNDS.size.step}
                  round="int"
                  onChange={(v) => set('width', v)}
                />
              </label>
              <span className="pb-2 text-grey-400">×</span>
              <label className="block w-24">
                <span className="mb-1 block text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
                  Height
                </span>
                <NumberInput
                  id="adv-height"
                  value={p.height}
                  min={BOUNDS.size.min}
                  max={BOUNDS.size.max}
                  step={BOUNDS.size.step}
                  round="int"
                  onChange={(v) => set('height', v)}
                />
              </label>
            </div>
          </Row>
        </>
      ) : (
        <>
          <Row
            label="Strength"
            id="adv-denoise"
            hint="1 ignores your picture, 0.4 stays close to it"
            overridden={on('denoise')}
            decided={(decided.denoise ?? 0).toFixed(2)}
            onRestore={() => clear('denoise')}
            note={`At ${(p.denoise ?? 0).toFixed(2)} the sampler starts ${Math.round(
              (p.denoise ?? 0) * p.steps,
            )} of ${p.steps} steps from the end, so the first ${
              p.steps - Math.round((p.denoise ?? 0) * p.steps)
            } are your picture rather than noise.`}
          >
            <NumberInput
              id="adv-denoise"
              value={p.denoise ?? 0.65}
              min={BOUNDS.denoise.min}
              max={BOUNDS.denoise.max}
              step={BOUNDS.denoise.step}
              round="two"
              onChange={(v) => set('denoise', v)}
            />
          </Row>

          <Row
            label="Output size"
            hint="aspect ratio is always kept"
            overridden={on('megapixels')}
            decided={`${(decided.megapixels ?? 1).toFixed(1)} MP`}
            onRestore={() => clear('megapixels')}
            note="Your picture is scaled to this budget, aspect kept, edges rounded to 16. Nothing is cropped or stretched."
          >
            <Chips
              ariaLabel="Output size"
              value={nearestMp(p.megapixels ?? 1)}
              options={MEGAPIXELS.map((m) => ({ value: m, label: `${m.toFixed(1)} MP` }))}
              onChange={(m) => set('megapixels', m)}
            />
          </Row>
        </>
      )}

      <Row
        label="Seed"
        id="adv-seed"
        overridden={on('seed') || on('seedLocked')}
        decided={decided.seed}
        onRestore={() => {
          clear('seed')
          clear('seedLocked')
        }}
        note={
          settled.seedLocked
            ? 'The same seed and the same settings make the same picture.'
            : 'A fresh seed every run. The one above is what this recipe rolled.'
        }
      >
        <div className="flex items-center gap-2">
          <NumberInput
            id="adv-seed"
            value={p.seed}
            min={0}
            max={0xffffffff}
            round="int"
            onChange={(v) => {
              set('seed', v)
              set('seedLocked', true)
            }}
          />
          <button
            type="button"
            aria-pressed={settled.seedLocked}
            onClick={() => set('seedLocked', !settled.seedLocked)}
            className={`shrink-0 cursor-pointer border px-2 py-1.5 text-overline font-semibold uppercase tracking-[0.14em] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900 ${
              settled.seedLocked
                ? 'border-ink bg-ink text-newsprint'
                : 'border-grey-300 text-grey-700 hover:border-ink'
            }`}
          >
            {settled.seedLocked ? 'Fixed' : 'Random'}
          </button>
        </div>
      </Row>

      {hasClipSkip ? (
        <Row
          label="Clip skip"
          id="adv-clipskip"
          hint="−2 on Illustrious and Pony checkpoints"
          overridden={on('clipSkip')}
          decided={decided.clipSkip ?? 'not set'}
          onRestore={() => clear('clipSkip')}
        >
          <NumberInput
            id="adv-clipskip"
            value={p.clipSkip ?? -1}
            min={BOUNDS.clipSkip.min}
            max={BOUNDS.clipSkip.max}
            round="int"
            onChange={(v) => set('clipSkip', v)}
          />
        </Row>
      ) : null}

      {hasShift ? (
        <Row
          label="Shift"
          id="adv-shift"
          hint="the sampling curve"
          overridden={on('shift')}
          decided={decided.shift ?? graphShift(plan.base.graph) ?? 'the graph’s own'}
          onRestore={() => clear('shift')}
        >
          <NumberInput
            id="adv-shift"
            value={p.shift ?? graphShift(plan.base.graph) ?? 0}
            min={BOUNDS.shift.min}
            max={BOUNDS.shift.max}
            step={BOUNDS.shift.step}
            onChange={(v) => set('shift', v)}
          />
        </Row>
      ) : null}

      <Row
        label="Prompt as sent"
        id="adv-positive"
        hint="prefix, your words, triggers"
        overridden={on('positive')}
        decided={<span className="not-italic">the recipe’s wording</span>}
        onRestore={() => clear('positive')}
        note="Edit this and the prompt on the simple screen stops driving it, including the trigger tokens a LoRA needs."
      >
        <textarea
          id="adv-positive"
          className="field"
          rows={3}
          value={p.positive}
          spellCheck={false}
          onChange={(e) => set('positive', e.target.value)}
        />
      </Row>

      {hasNegative ? (
        <Row
          label="Negative prompt"
          id="adv-negative"
          hint="what to keep out"
          overridden={on('negative')}
          decided={house ? <span className="not-italic">the house wording</span> : 'empty'}
          onRestore={() => clear('negative')}
        >
          <textarea
            id="adv-negative"
            className="field"
            rows={3}
            value={p.negative}
            spellCheck={false}
            onChange={(e) => set('negative', e.target.value)}
          />
        </Row>
      ) : null}

      <Row
        label="How many"
        hint="one after another, never in one batch"
        overridden={on('runs')}
        decided="1"
        onRestore={() => clear('runs')}
      >
        <Chips
          ariaLabel="How many"
          value={settled.runs}
          options={[1, 2, 4].map((n) => ({ value: n, label: `×${n}` }))}
          onChange={(n) => set('runs', n)}
        />
        {settled.runs > 1 && settled.seedLocked ? (
          <div className="mt-1.5">
            <Caution>
              The seed is fixed and the settings are not changing, so all {settled.runs} runs will
              draw the same picture. Unfix the seed to get variations.
            </Caution>
          </div>
        ) : null}
      </Row>

      {plan.base.notes ? (
        <div className="mb-3 border-l-2 border-burgundy-900 pl-2.5">
          <Kicker>About this family</Kicker>
          <p className="mt-0.5 text-caption italic leading-snug text-grey-700">
            {trim(plan.base.notes, 700)}
          </p>
        </div>
      ) : null}
    </section>
  )
}

/** The author's second set of settings, when the registry carries one. */
function altOf(plan: Plan): { sampler: string; scheduler: string; steps: number; cfg: number } | null {
  const per = (plan.base.perModel?.[plan.model] ?? {}) as Record<string, unknown>
  const raw = per.altSampler as Record<string, unknown> | undefined
  if (!raw) return null
  const sampler = typeof raw.sampler === 'string' ? raw.sampler : null
  const scheduler = typeof raw.scheduler === 'string' ? raw.scheduler : null
  const steps = typeof raw.steps === 'number' ? raw.steps : null
  const cfg = typeof raw.cfg === 'number' ? raw.cfg : null
  if (!sampler || !scheduler || steps === null || cfg === null) return null
  return { sampler, scheduler, steps, cfg }
}

function nearestMp(v: number): number {
  let best = MEGAPIXELS[0]
  for (const m of MEGAPIXELS) if (Math.abs(m - v) < Math.abs(best - v)) best = m
  return best
}

function trim(text: string, n: number): string {
  if (text.length <= n) return text
  return `${text.slice(0, n).replace(/\s+\S*$/, '')}…`
}
