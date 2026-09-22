/**
 * The picture you are working from, when there is one.
 *
 * This is the only control on the default screen that is conditional. It is
 * printed when a source exists, or when the desk has been put into a mode that
 * needs one; the rest of the time the screen does not mention pictures at all,
 * because a screen that asks "or would you like to upload something?" is a
 * screen asking a fourth question.
 *
 * The old well carried four links: replace, from the archive, clear, and a
 * separate mode tab row above it to reach the well in the first place. Three
 * of those collapse into one: {@link onPick} opens the picker, and the picker
 * is where choosing a file and choosing from the archive both live. Remove
 * stays printed, quietly, because a capability that only a mouseless reader
 * can reach is a capability that was deleted.
 */
import type { SourceRef } from '../../lib/session'
import { EditorialLink, Label, Notice } from './bits'

const THIN = ' '

function times(w: number, h: number): string {
  return `${w}${THIN}×${THIN}${h}`
}

function bytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${Math.round(n / 1024)} KB`
  return `${(n / 1024 ** 2).toFixed(1)} MB`
}

export function SourceWell({
  source,
  busy = false,
  error = null,
  onPick,
  onClear,
}: {
  source: SourceRef | null
  /** The upload has not landed yet. */
  busy?: boolean
  error?: string | null
  /** Open the picker. It should offer a file and the archive, as it always has. */
  onPick: () => void
  onClear: () => void
}) {
  const caption = source
    ? [
        source.label ?? source.name,
        source.width && source.height ? times(source.width, source.height) : null,
        source.bytes ? bytes(source.bytes) : null,
      ]
        .filter(Boolean)
        .join(' · ')
    : ''

  return (
    <div>
      <Label>Working from</Label>

      {source ? (
        <div className="flex items-start gap-3">
          <button
            type="button"
            onClick={onPick}
            title="Choose a different picture"
            className="ring sg-tap flex min-w-0 flex-1 cursor-pointer items-start gap-3 border border-grey-300 bg-newsprint-aged p-2 text-left transition-colors hover:border-ink"
          >
            <span className="block h-16 w-16 shrink-0 overflow-hidden border border-grey-300 bg-newsprint">
              {source.previewUrl && (
                <img src={source.previewUrl} alt="" className="h-full w-full object-cover" />
              )}
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-caption italic text-grey-700">{caption}</span>
              {busy && (
                <span className="block text-caption italic text-grey-500">
                  Copying it across{'…'}
                </span>
              )}
              {source.fromEntryId && (
                <span className="block text-caption italic text-grey-500">From your archive.</span>
              )}
              {source.fromFrame != null && (
                <span className="block text-caption italic text-grey-500 tabular-nums">
                  Frame {source.fromFrame} of a clip.
                </span>
              )}
              <span className="mt-1 block text-caption text-grey-500">
                Choose a different one.
              </span>
            </span>
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={onPick}
          className="ring sg-tap w-full cursor-pointer border border-dashed border-grey-400 bg-newsprint-aged px-3 py-5 text-center transition-colors hover:border-burgundy-900"
        >
          <span className="block text-overline font-semibold uppercase tracking-[0.18em] text-grey-700">
            Choose a picture
          </span>
          <span className="mt-1 block text-caption italic text-grey-500">
            Drop one here, or paste it.
          </span>
        </button>
      )}

      {source && (
        <p className="mt-1 text-caption text-grey-500">
          <EditorialLink onClick={onClear}>Remove it</EditorialLink> and the desk goes back to
          working from words alone.
        </p>
      )}

      {error && (
        <div className="mt-2">
          <Notice kind="error" title="We could not use that picture">
            {error}
          </Notice>
        </div>
      )}
    </div>
  )
}
