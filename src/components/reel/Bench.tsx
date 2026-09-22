/**
 * The bench: everything the whole reel shares.
 *
 * These settings are deliberately not per shot. A shot that opens on the last
 * frame of the shot before it has to be the same shape as that frame, because
 * every Wan image node centre crops what it is handed. Putting width and height
 * on the reel rather than on the shot makes that impossible to get wrong,
 * instead of merely warned about.
 *
 * The ledger at the foot is the reel's own masthead figures: how many shots,
 * how many frames, how much screen time, and how far the look will have
 * drifted by the end.
 */
import type { FamilyDef } from '../../lib/workflows'
import type { ShotPlan } from '../../lib/continuation'
import { Caution, Chips, Field, Head, Kicker, Leader, NumberField, Quiet, RING, grouped, seconds, times } from './bits'
import type { ReelDraft } from './store'

export type NumSpec = { min: number; max: number; step: number }

/** One installed video family, as the bench offers it. */
export type ReelFamily = {
  def: FamilyDef
  model: string
  label: string
  /** True when a shot of this family can open on a frame. */
  chainable: boolean
  /** One line for a family that cannot chain, so the absence is explicable. */
  why: string | null
  width: NumSpec
  height: NumSpec
  frames: NumSpec
}

export type Shape = { width: number; height: number; label: string }

export type BenchProps = {
  draft: ReelDraft
  families: readonly ReelFamily[]
  family: ReelFamily | null
  blocked: readonly { label: string; why: string }[]
  shapes: readonly Shape[]
  lengthOptions: readonly number[]
  samplers: readonly string[]
  schedulers: readonly string[]
  houseNegative: string
  plan: ShotPlan
  expert: boolean
  busy: boolean
  onPatch: (p: Partial<ReelDraft>) => void
  onPinAnchor: () => void
  onRerollSeed: () => void
}

export function Bench(p: BenchProps) {
  const { draft, family, plan, expert } = p
  const deepest = plan.jobs.reduce((m, j) => Math.max(m, j.hops), 0)

  return (
    <aside className="lg:sticky lg:top-4 lg:self-start">
      <Head title="The bench" note="One set of settings. Every shot follows them." />

      <Field label="Style" hint={family?.chainable ? 'chains' : family ? 'no chaining' : undefined}>
        <select
          className="field text-small"
          value={draft.familyId}
          disabled={p.busy}
          onChange={(e) => p.onPatch({ familyId: e.target.value })}
        >
          {p.families.map((f) => (
            <option key={f.def.id} value={f.def.id}>
              {f.label}
            </option>
          ))}
        </select>
        {family && !family.chainable && family.why ? <Caution>{family.why}</Caution> : null}
      </Field>

      <Field label="Shape" hint={times(draft.width, draft.height)}>
        <Chips
          ariaLabel="The shape of every shot"
          value={`${draft.width}x${draft.height}`}
          options={p.shapes.map((s) => ({
            value: `${s.width}x${s.height}`,
            label: s.label,
            title: times(s.width, s.height),
          }))}
          onChange={(v) => {
            const [w, h] = String(v).split('x').map(Number)
            if (w && h) p.onPatch({ width: w, height: h })
          }}
        />
        <p className="mt-1 text-caption italic text-grey-500">
          Fixed for the whole reel. A chained frame is centre cropped to fit, so a shape that changes mid reel loses
          the edges of the picture it was handed.
        </p>
      </Field>

      <Field label="Shot length" hint={`${draft.length} frames`}>
        <Chips
          ariaLabel="How long each shot runs"
          value={draft.length}
          options={p.lengthOptions.map((n) => ({ value: n, label: seconds(n, draft.fps), title: `${n} frames` }))}
          onChange={(v) => p.onPatch({ length: v })}
        />
        <p className="mt-1 text-caption italic text-grey-500">
          Fewer, longer shots look better than more, shorter ones. Length costs memory once. Every join costs quality
          for the rest of the reel.
        </p>
      </Field>

      <Field label="Anchor frame" hint={draft.anchor ? 'set' : 'none'}>
        {draft.anchor ? (
          <div className="flex items-start gap-2">
            {draft.anchor.previewUrl ? (
              <img
                src={draft.anchor.previewUrl}
                alt=""
                className="h-12 w-20 border border-grey-300 object-cover"
              />
            ) : null}
            <div className="min-w-0 flex-1">
              <p className="truncate text-caption text-ink">{draft.anchor.label}</p>
              <button
                type="button"
                className={`sg-link ${RING} text-caption`}
                onClick={() => p.onPatch({ anchor: null })}
              >
                Remove it
              </button>
            </div>
          </div>
        ) : (
          <Quiet onClick={p.onPinAnchor}>Choose a frame</Quiet>
        )}
        <p className="mt-1 text-caption italic text-grey-500">
          A clean frame the reel can go back to. The opening shot starts from it, and the schedule below returns to it
          when the look has drifted too far.
        </p>
      </Field>

      {draft.anchor ? (
        <Field label="Return to the anchor" hint={draft.reanchorEvery ? `every ${draft.reanchorEvery}` : 'never'}>
          <Chips
            ariaLabel="How often the reel returns to the anchor frame"
            value={draft.reanchorEvery}
            options={[
              { value: 0, label: 'Never' },
              { value: 3, label: 'Every 3' },
              { value: 4, label: 'Every 4' },
              { value: 5, label: 'Every 5' },
            ]}
            onChange={(v) => p.onPatch({ reanchorEvery: v })}
          />
          <p className="mt-1 text-caption italic text-grey-500">
            Restarting from a clean frame resets the drift and buys a cut point. It also shows as a visible jump, which
            is a fair price for a reel that still looks like itself at the end.
          </p>
        </Field>
      ) : null}

      <Field label="Seed" hint={draft.seedLocked ? 'fixed' : 'fresh each reel'}>
        <div className="flex gap-2">
          <NumberField
            label="Seed"
            italic={!draft.seedLocked}
            value={draft.seed}
            min={0}
            max={Number.MAX_SAFE_INTEGER}
            step={1}
            commit={(n) => {
              if (n < 0) return
              p.onPatch({ seed: Math.floor(n), seedLocked: true })
            }}
          />
          <button
            type="button"
            aria-pressed={draft.seedLocked}
            onClick={() => p.onPatch({ seedLocked: !draft.seedLocked })}
            className={`${RING} shrink-0 border px-2 text-[0.625rem] font-semibold uppercase tracking-[0.16em] ${
              draft.seedLocked
                ? 'border-ink bg-ink text-newsprint'
                : 'border-grey-300 text-grey-700 hover:bg-newsprint-aged'
            }`}
          >
            {draft.seedLocked ? 'Fixed' : 'Random'}
          </button>
        </div>
        <p className="mt-1 text-caption italic text-grey-500">
          Shot 1 takes this seed, shot 2 takes the next, and so on up the reel. Fix it and the same reel comes back the
          same way.
        </p>
        {!draft.seedLocked ? (
          <button type="button" className={`sg-link ${RING} mt-1 text-caption`} onClick={p.onRerollSeed}>
            Draw a new one now
          </button>
        ) : null}
      </Field>

      {expert && family ? (
        <div className="mt-5 border-t border-grey-300 pt-4">
          <Kicker tone="quiet" className="mb-3">
            All controls
          </Kicker>

          <div className="grid grid-cols-2 gap-2">
            <Field label="Width">
              <NumberField
                label="Width"
                value={draft.width}
                min={family.width.min}
                max={family.width.max}
                step={family.width.step}
                commit={(n) => p.onPatch({ width: snap(n, family.width) })}
              />
            </Field>
            <Field label="Height">
              <NumberField
                label="Height"
                value={draft.height}
                min={family.height.min}
                max={family.height.max}
                step={family.height.step}
                commit={(n) => p.onPatch({ height: snap(n, family.height) })}
              />
            </Field>
          </div>

          <Field label="Frames per second">
            <NumberField
              label="Frames per second"
              value={draft.fps}
              min={1}
              max={120}
              step={1}
              commit={(n) => p.onPatch({ fps: Math.min(120, Math.max(1, Math.round(n))) })}
            />
          </Field>

          {family.def.dualModel ? (
            <p className="mb-3 text-caption italic text-grey-700">
              This family runs a matched high noise and low noise pass. Its step count comes with the recipe.
            </p>
          ) : (
            <Field label="Steps">
              <NumberField
                label="Steps"
                value={draft.steps}
                min={1}
                max={200}
                step={1}
                commit={(n) => p.onPatch({ steps: Math.min(200, Math.max(1, Math.round(n))) })}
              />
            </Field>
          )}

          <Field label="CFG">
            <NumberField
              label="CFG"
              value={draft.cfg}
              min={0}
              max={30}
              step={0.1}
              commit={(n) => p.onPatch({ cfg: Math.min(30, Math.max(0, n)) })}
            />
          </Field>

          <Field label="Sampler">
            <select
              className="field text-small"
              value={draft.sampler}
              onChange={(e) => p.onPatch({ sampler: e.target.value })}
            >
              {(p.samplers.length ? p.samplers : [draft.sampler]).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Scheduler">
            <select
              className="field text-small"
              value={draft.scheduler}
              onChange={(e) => p.onPatch({ scheduler: e.target.value })}
            >
              {(p.schedulers.length ? p.schedulers : [draft.scheduler]).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Negative">
            <textarea
              className="field h-20 text-caption"
              value={draft.negative ?? p.houseNegative}
              onChange={(e) => p.onPatch({ negative: e.target.value })}
            />
            {draft.negative !== null ? (
              <button
                type="button"
                className={`sg-link ${RING} mt-1 text-caption`}
                onClick={() => p.onPatch({ negative: null })}
              >
                Reset to the house wording
              </button>
            ) : (
              <p className="mt-1 text-caption italic text-grey-500">The family's own wording, unchanged.</p>
            )}
          </Field>

          <Field label="Output folder">
            <input
              className="field text-caption"
              value={draft.prefix}
              onChange={(e) => p.onPatch({ prefix: e.target.value })}
            />
            <p className="mt-1 text-caption italic text-grey-500">
              Under ComfyUI's output directory. Clips are numbered inside it, so they sort into cutting order.
            </p>
          </Field>
        </div>
      ) : null}

      {/* the ledger ---------------------------------------------------------- */}
      <div className="mt-5 border-t-2 border-burgundy-900 pt-2">
        <Kicker className="mb-2">The reel</Kicker>
        <Leader label="Shots" value={grouped(plan.jobs.length)} />
        <Leader label="Frames" value={grouped(plan.frames)} />
        <Leader label="Screen time" value={`${plan.seconds.toFixed(1)} s`} />
        <Leader label="Generations" value={grouped(plan.jobs.length)} />
        <Leader label="Deepest hop" value={deepest === 0 ? 'none' : String(deepest)} />
      </div>

      {plan.warnings.length ? (
        <div className="mt-3 space-y-2">
          {plan.warnings.map((w) => (
            <Caution key={w}>{w}</Caution>
          ))}
        </div>
      ) : null}

      {p.blocked.length ? (
        <div className="mt-4 border-t border-grey-300 pt-2">
          <Kicker tone="quiet" className="mb-1">
            Not offered
          </Kicker>
          {p.blocked.map((b) => (
            <p key={b.label} className="text-caption italic text-grey-500">
              {b.label}: {b.why}.
            </p>
          ))}
        </div>
      ) : null}
    </aside>
  )
}

function snap(value: number, spec: NumSpec): number {
  const steps = Math.round((value - spec.min) / spec.step)
  return Math.min(spec.max, Math.max(spec.min, spec.min + steps * spec.step))
}
