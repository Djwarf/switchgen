/**
 * The detail passes, offered before the picture rather than after it.
 *
 * The desk's rule is that a quality pass is a decision best made looking at the
 * result: you cannot tell whether the hands need rebuilding until you can see
 * the hands. So the finished picture offers all of these, and the default
 * screen offers none of them.
 *
 * They are here as well because sometimes you already know. A batch of four at
 * two in the morning, a checkpoint whose faces always come out soft, a shape
 * the detector reliably finds two faces in: in those cases running the pass in
 * the same job saves a round trip, and the old desk let you say so. Nothing was
 * taken away, it was moved to the place where you have to ask for it.
 *
 * WHAT IS NOT HERE. The region refine pass. It renders a mask you draw over the
 * part you want redrawn, and there is nothing to draw on before the picture
 * exists. It is listed so its absence is stated rather than looking like an
 * omission.
 *
 * THE FACE PASS MEASURED 0.714. That figure is whole frame Laplacian variance,
 * which is sharpness: re rendering a face at 768 and pasting it back softens
 * the frame average while making the face itself right. It is printed because
 * every number in this app is printed, and it is not an argument against the
 * pass. It is the reason the pass is offered rather than run on your behalf.
 */
import { hiresSize, hiresStepsFor } from '../../lib/refine'
import type { Plan } from '../../lib/recipe'
import { Check, Head, Note, times } from './bits'
import type { Passes, Settled } from './overrides'

export function PassPanel({
  plan,
  settled,
  onPasses,
}: {
  plan: Plan
  settled: Settled
  onPasses: (next: Passes) => void
}) {
  const caps = plan.capabilities
  // A pass the graph can carry is still left out when ComfyUI lacks a file it
  // loads: the job would be refused over that file. The sentence naming the
  // file is printed instead, so the missing row has a reason on the page.
  const face = plan.passes.face.available
  const hand = plan.passes.hand.available
  const blocked = [plan.passes.face.blocked, plan.passes.hand.blocked, plan.passes.refine.blocked].filter(
    (b, i, all): b is string => !!b && all.indexOf(b) === i,
  )
  const p = settled.passes
  const steps = settled.params.steps
  const sized = !settled.params.image
  const big = hiresSize({ width: settled.params.width, height: settled.params.height })

  const toggle = (key: keyof Passes) => onPasses({ ...p, [key]: !p[key] })

  const none = !caps.faceDetail && !caps.handDetail && !caps.hires
  const shown = face || hand || caps.hires
  // The region note counts the rows actually drawn above. A blocked detector
  // takes its row away, and the finished picture does not offer that pass
  // either, so a fixed "these three" promised passes that are not coming.
  const listed = [face, hand, caps.hires].filter(Boolean).length
  const alongWith = ['', ', along with the pass above', ', along with these two', ', along with these three'][listed]

  return (
    <section className="mb-7">
      <Head
        title="Detail passes"
        figure={settled.cost > 1 ? `${settled.cost.toFixed(2)}x the time` : undefined}
        note="Each one re renders part of the picture at a higher resolution. That is the only thing that adds real detail, and each costs a full pass of GPU time."
      />

      {none ? (
        <Note>
          {plan.label} samples through a custom schedule with no denoise control, so no detail pass
          can run on it. It is chosen for what it draws, not for what can be done to it afterwards.
        </Note>
      ) : !shown ? null : (
        <ul className="border-t border-grey-300">
          {face ? (
            <Check
              on={p.face}
              onToggle={() => toggle('face')}
              title="Detail every face"
              note="Finds faces with a detector, crops each one, re renders it at up to 1024px and pastes it back. A face 80px across has about a hundred latent cells and cannot hold two eyes and a mouth. At 1024 it has nine thousand."
              cost="about one extra pass per face found"
            />
          ) : null}
          {hand ? (
            <Check
              on={p.hand}
              onToggle={() => toggle('hand')}
              title="Detail every hand"
              note="The same pass on the hand detector, at a higher strength. Hands come out wrong rather than merely soft, so this one is allowed to rebuild rather than sharpen."
              cost="about one extra pass per hand found"
            />
          ) : null}
          {caps.hires ? (
            <Check
              on={p.hires}
              onToggle={() => toggle('hires')}
              title="Two pass render"
              note={`Composes at the size the model was trained on, upscales the latent by 1.5, then redraws at ${hiresStepsFor(
                steps,
              )} steps and 0.45 strength. Every region gets 2.25 times the cells to resolve in.${
                sized ? ` Output ${times(big.width, big.height)}.` : ''
              }`}
              cost="about 1.35 extra passes, and noticeably more VRAM"
            />
          ) : null}
        </ul>
      )}

      {blocked.length ? (
        <div className="mt-2 space-y-1">
          {blocked.map((b) => (
            <Note key={b}>{b}</Note>
          ))}
        </div>
      ) : null}

      {settled.cost > 1 ? (
        <p className="mt-2 text-caption tabular-nums leading-snug text-grey-700">
          Roughly {settled.cost.toFixed(2)}x the time of a plain picture, at least. A detector that
          finds two faces runs the face pass twice.
        </p>
      ) : null}

      <div className="mt-3 border-t border-grey-300 pt-2">
        <Note>
          {plan.passes.refine.available
            ? `A masked region re render is not here because there is nothing to draw a mask on yet. The finished picture offers it${alongWith}.`
            : caps.refine
              ? 'A masked region re render cannot run here until what is named above is installed.'
              : 'This family cannot carry a masked region re render, so the finished picture will not offer one.'}
        </Note>
        {p.face || p.hand || p.hires ? (
          <Note>
            These will run inside the same job. The finished picture offers the same passes, one at a
            time, where you can see first whether they are needed.
          </Note>
        ) : null}
      </div>
    </section>
  )
}
