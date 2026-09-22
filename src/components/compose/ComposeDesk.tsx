/**
 * The compose rail, as a printed page.
 *
 * COUNT THE CONTROLS. That is what this file is for.
 *
 *   1  the prompt            what the picture is of
 *   2  the look              one radio group, one tab stop
 *   3  the anatomy level     one radio group, one tab stop
 *   4  the source well       printed only when there is a picture to work from
 *   5  the button
 *
 * plus one footnote link, at caption size, under a hairline at the foot of the
 * column, which opens everything else.
 *
 * In the state the desk opens in, that is four controls and the footnote. With
 * a picture attached it is five and the footnote, and the well carries a quiet
 * Remove beside it so that capability stays reachable with a mouse.
 *
 * Nothing else is asked. The model, the LoRA stack, every strength, the
 * sampler, the size, the prefix tokens, the trigger words and the detail
 * passes are all decided by recipe.ts from the measured table, and the
 * decisions are printed in prose under the button rather than collected in
 * fields above it. A reader who disagrees opens More, where all of it still
 * lives, unchanged.
 *
 * Quality passes are not here at all. They belong on the finished picture,
 * where the reader can see whether the hands came out wrong before spending a
 * pass on them. The recipe carries the offers; the result prints them.
 *
 * This component holds no state but the disclosure, and even that is
 * controllable. The three answers, the source and the job all belong to the
 * desk.
 */
import type { ReactNode, RefObject } from 'react'
import type { AnatomyLevel, Look, Recipe } from '../../lib/recipe'
import type { SourceRef } from '../../lib/session'
import { AddOnOffers } from './AddOnOffers'
import { AnatomyPicker } from './AnatomyPicker'
import { Deck, EditorialLink, Kicker } from './bits'
import { LookPicker } from './LookPicker'
import { MoreFootnote } from './More'
import { PromptField } from './PromptField'
import { RecipeProse } from './RecipeProse'
import { RunButton, type RunJob } from './RunButton'
import { SourceWell } from './SourceWell'

export type ComposeDeskProps = {
  // --- the three answers ---------------------------------------------------
  prompt: string
  look: Look
  anatomy: AnatomyLevel
  onPrompt: (v: string) => void
  onLook: (v: Look) => void
  onAnatomy: (v: AnatomyLevel) => void

  /** Add-ons the wording matched. Offered, never applied without a decision. */
  onAcceptAddOn?: (file: string) => void
  onDeclineAddOn?: (file: string) => void
  /** Take an added add-on off again. Without it, an added one cannot be removed here. */
  onRemoveAddOn?: (file: string) => void

  /** What decide() returned for those three answers, plus this machine. */
  recipe: Recipe

  // --- the picture being worked from, when there is one --------------------
  source?: SourceRef | null
  /**
   * Print the well even with no source: the desk is in a mode that needs one.
   * Left false, a desk working from words never mentions pictures.
   */
  needsSource?: boolean
  sourceBusy?: boolean
  sourceError?: string | null
  /** Open the picker. It should offer a file and the archive, as it always has. */
  onPickSource?: () => void
  /**
   * Open the region bench on the attached picture. Offered for any picture the
   * desk is holding, whichever way it arrived: uploaded, pasted, dropped, or
   * adopted from the archive. Omit it and nothing is printed.
   */
  onEditRegion?: () => void
  /** A reading of the attached picture, printed under the well. See result/Reading. */
  sourceReading?: ReactNode
  onClearSource?: () => void

  // --- the press -----------------------------------------------------------
  onRun: () => void
  onStop?: () => void
  running?: boolean
  job?: RunJob | null
  queuedAhead?: number
  lastMs?: number | null
  /** The desk's own reason to refuse, on top of the ones counted here. */
  disabled?: boolean
  disabledWhy?: string
  runLabel?: string

  // --- the rest of the apparatus -------------------------------------------
  /** Mounted only while More is open. Everything the desk ever had goes here. */
  advanced?: ReactNode
  moreOpen?: boolean
  onMoreOpenChange?: (open: boolean) => void

  // --- incidentals ---------------------------------------------------------
  title?: string
  deck?: string
  promptRef?: RefObject<HTMLTextAreaElement | null>
  reducedMotion?: boolean
}

export function ComposeDesk({
  prompt,
  look,
  anatomy,
  onPrompt,
  onLook,
  onAnatomy,
  onAcceptAddOn,
  onDeclineAddOn,
  onRemoveAddOn,
  recipe,
  source = null,
  needsSource = false,
  sourceBusy = false,
  sourceError = null,
  onPickSource,
  onClearSource,
  onEditRegion,
  sourceReading,
  onRun,
  onStop,
  running = false,
  job = null,
  queuedAhead = 0,
  lastMs = null,
  disabled = false,
  disabledWhy = '',
  runLabel,
  advanced,
  moreOpen,
  onMoreOpenChange,
  title = 'A new picture',
  deck = 'Say what you want and how it should look. Everything else is worked out from the measurements, and printed under the button.',
  promptRef,
  reducedMotion = false,
}: ComposeDeskProps) {
  const showWell = Boolean(source) || needsSource
  const canPickSource = Boolean(onPickSource)

  const blocked = readiness({
    recipe,
    prompt,
    source,
    needsSource,
    sourceBusy,
    disabled,
    disabledWhy,
  })

  return (
    <section className="max-w-[46rem]">
      <header className="mb-8">
        <Kicker>Compose</Kicker>
        <h2 className="mt-1 text-h2 leading-tight font-semibold text-ink">{title}</h2>
        <div className="mt-2 mb-4 border-b-2 border-burgundy-900" />
        <Deck>{deck}</Deck>
      </header>

      <div className="space-y-8">
        <PromptField
          value={prompt}
          onChange={onPrompt}
          onSubmit={blocked.why ? undefined : onRun}
          textRef={promptRef}
          label={source ? 'The change' : 'The picture'}
          placeholder={
            source
              ? 'Make the jacket red'
              : 'A rain-slicked tram stop at dusk, neon in the puddles'
          }
          rows={source ? 3 : 5}
        />

        <LookPicker value={look} onChange={onLook} />

        <AnatomyPicker value={anatomy} onChange={onAnatomy} />

        {showWell && canPickSource && (
          <div>
            <SourceWell
              source={source}
              busy={sourceBusy}
              error={sourceError}
              onPick={() => onPickSource?.()}
              onClear={() => onClearSource?.()}
            />
            {source && onEditRegion ? (
              <p className="mt-1.5 text-caption text-grey-500">
                <EditorialLink onClick={onEditRegion}>Change part of this picture</EditorialLink>{' '}
                instead, by painting over the area you want redrawn.
              </p>
            ) : null}
            {source && sourceReading ? <div className="mt-2">{sourceReading}</div> : null}
          </div>
        )}

        {recipe.ok && onAcceptAddOn && onDeclineAddOn && (
          <AddOnOffers
            offers={recipe.offers}
            applied={recipe.loras.filter((l) => !l.measured)}
            onAccept={onAcceptAddOn}
            onDecline={onDeclineAddOn}
            onRemove={onRemoveAddOn}
          />
        )}

        <div>
          <RunButton
            label={runLabel ?? (source ? 'Make the change' : 'Make the picture')}
            disabled={Boolean(blocked.why)}
            why={blocked.why}
            running={running}
            queuedAhead={queuedAhead}
            lastMs={lastMs}
            job={job}
            onRun={onRun}
            onStop={onStop}
            reduced={reducedMotion}
          />

          <RecipeProse recipe={recipe} />
        </div>
      </div>

      <MoreFootnote
        open={moreOpen}
        onOpenChange={onMoreOpenChange}
        aside={
          !showWell && canPickSource
            ? 'Drop a picture on this page, or paste one, to work from it instead.'
            : undefined
        }
      >
        {advanced}
      </MoreFootnote>
    </section>
  )
}

/**
 * Why the button is off, in one sentence, or an empty one when it is on.
 *
 * Kept here rather than in the button so the same sentence can be read by the
 * prompt field, which is what Ctrl+Enter goes through.
 */
function readiness({
  recipe,
  prompt,
  source,
  needsSource,
  sourceBusy,
  disabled,
  disabledWhy,
}: {
  recipe: Recipe
  prompt: string
  source: SourceRef | null
  needsSource: boolean
  sourceBusy: boolean
  disabled: boolean
  disabledWhy: string
}): { why: string } {
  if (disabled) return { why: disabledWhy }
  if (!prompt.trim()) return { why: 'Say what the picture is of.' }
  if (!recipe.ok) return { why: recipe.reason }
  if (needsSource && !source) return { why: 'Choose the picture to work from.' }
  if (source && !source.name) return { why: 'The picture is still copying across.' }
  if (sourceBusy) return { why: 'The picture is still copying across.' }
  return { why: '' }
}
