/**
 * Pinning a frame.
 *
 * Three ways in, in the order a reader reaches for them: a picture already in
 * the archive, a file from disk, or a frame this reel has already produced.
 *
 * An archive picture needs no upload. ComfyUI's LoadImage validates against
 * `folder_paths.exists_annotated_filepath`, so an output can be addressed in
 * place as "subfolder/name.png [output]". A file from disk has never been to
 * the server, so that one is uploaded and comes back as an input filename.
 */
import { useEffect, useRef, useState } from 'react'

import { annotatedRef } from '../../lib/continuation'
import { fileUrl, uploadImage } from '../../lib/comfy'
import type { HistoryEntry } from '../../lib/history'
import { Kicker, Quiet, RING } from './bits'
import type { PinnedFrame } from './store'

export type KeyframeTarget = {
  /** What the pinned frame will do, for the sheet's heading. */
  title: string
  /** One line under it. */
  standfirst: string
  onPick: (frame: PinnedFrame) => void
}

export type KeyframePickerProps = {
  target: KeyframeTarget | null
  onClose: () => void
  /** Pictures from the archive, newest first. */
  pictures: readonly HistoryEntry[]
  /** Frames this reel has already made, newest last. */
  reelFrames: readonly { label: string; name: string; previewUrl: string }[]
}

export function KeyframePicker({ target, onClose, pictures, reelFrames }: KeyframePickerProps) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement | null>(null)
  const sheet = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!target) return
    setError(null)
    setBusy(false)
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    sheet.current?.focus()
    return () => window.removeEventListener('keydown', onKey)
  }, [target, onClose])

  if (!target) return null

  const take = (frame: PinnedFrame) => {
    target.onPick(frame)
    onClose()
  }

  const fromDisk = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('A keyframe has to be a picture. Try a PNG or a JPEG.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const name = await uploadImage(file)
      take({ name, label: file.name, previewUrl: URL.createObjectURL(file) })
    } catch (err) {
      setError(`ComfyUI refused that file. ${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="sg-scrim" onClick={onClose} role="presentation">
      <div
        ref={sheet}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={target.title}
        className="sg-sheet sg-unfold w-full max-w-3xl bg-newsprint p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 border-b-2 border-burgundy-900 pb-2">
          <Kicker>{target.title}</Kicker>
          <p className="mt-1 text-small italic text-grey-700">{target.standfirst}</p>
        </div>

        {error ? <p className="notice notice-error mb-4 text-small">{error}</p> : null}

        <div className="mb-5">
          <Kicker tone="quiet" className="mb-2">
            From this machine
          </Kicker>
          <input
            ref={input}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              e.target.value = ''
              if (file) void fromDisk(file)
            }}
          />
          <Quiet onClick={() => input.current?.click()} disabled={busy}>
            {busy ? 'Uploading' : 'Choose a picture'}
          </Quiet>
        </div>

        {reelFrames.length ? (
          <div className="mb-5">
            <Kicker tone="quiet" className="mb-2">
              Frames this reel has made
            </Kicker>
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-5">
              {reelFrames.map((f) => (
                <Plate
                  key={f.name}
                  src={f.previewUrl}
                  caption={f.label}
                  onClick={() => take({ name: f.name, label: f.label, previewUrl: f.previewUrl })}
                />
              ))}
            </div>
          </div>
        ) : null}

        <div className="mb-5">
          <Kicker tone="quiet" className="mb-2">
            From the archive
          </Kicker>
          {pictures.length ? (
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-5">
              {pictures.slice(0, 20).map((entry) => (
                <Plate
                  key={entry.id}
                  src={fileUrl(entry.file)}
                  caption={`No. ${entry.no}`}
                  onClick={() =>
                    take({
                      name: annotatedRef(entry.file),
                      label: `No. ${entry.no} ${entry.file.filename}`,
                      previewUrl: fileUrl(entry.file),
                    })
                  }
                />
              ))}
            </div>
          ) : (
            <p className="text-caption italic text-grey-500">
              Nothing in the archive yet. Make a picture at the Pictures desk and it will appear here.
            </p>
          )}
        </div>

        <div className="flex justify-end border-t border-grey-300 pt-3">
          <Quiet onClick={onClose}>Close</Quiet>
        </div>
      </div>
    </div>
  )
}

function Plate({ src, caption, onClick }: { src: string; caption: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`${RING} group block w-full border border-grey-300 bg-newsprint-aged text-left transition-colors hover:border-ink`}
    >
      <span className="block aspect-[16/9] overflow-hidden bg-grey-200">
        <img src={src} alt="" loading="lazy" className="h-full w-full object-cover" />
      </span>
      <span className="block truncate px-1 py-0.5 text-caption tabular-nums text-grey-700 group-hover:text-ink">
        {caption}
      </span>
    </button>
  )
}
