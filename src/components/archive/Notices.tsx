/**
 * Notices and the undo bar.
 *
 * A correction is the newspaper's way of saying "that changed, and here is
 * what we did about it". It is quiet, it is italic, and it always offers the
 * way back where there is one.
 */
import { useEffect, useRef, useState, type ReactNode } from 'react'

type Variant = 'info' | 'correction' | 'error' | 'success'

const SKIN: Record<Variant, string> = {
  info: 'bg-burgundy-50 border-burgundy-900 text-ink',
  correction: 'bg-[#F5F5F5] border-ink text-ink italic',
  error: 'bg-[#FEF2F2] border-error text-[#7F1D1D]',
  success: 'bg-[#F5F3ED] border-success text-ink',
}

export function Notice({
  variant = 'info',
  title,
  children,
  onDismiss,
}: {
  variant?: Variant
  title?: string
  children: ReactNode
  onDismiss?: () => void
}) {
  return (
    <div
      role={variant === 'error' ? 'alert' : 'status'}
      className={`flex items-start gap-4 border-l-4 px-4 py-3 text-small leading-relaxed ${SKIN[variant]}`}
    >
      <p className="flex-1">
        {title && (
          <strong className="mr-2 text-[0.75rem] font-bold tracking-[0.05em] uppercase not-italic">
            {title}
          </strong>
        )}
        {children}
      </p>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss this notice"
          className="shrink-0 px-1 text-grey-500 not-italic hover:text-burgundy-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
        >
          ×
        </button>
      )}
    </div>
  )
}

/**
 * The window in which a removal can be taken back, drawn as a rule draining
 * left to right so you can watch it closing rather than guess at it.
 */
export function UndoBar({
  text,
  detail,
  onUndo,
  onDismiss,
  ms = 8000,
}: {
  text: string
  detail?: string
  onUndo: () => void
  onDismiss: () => void
  ms?: number
}) {
  const [draining, setDraining] = useState(false)

  // `onDismiss` is written inline by the caller, so it is a different function
  // on every render of the screen behind this bar. Depending on it would clear
  // and restart the timer each time the archive re-rendered, which on a busy
  // desk means the window never actually closes while the rule on top has
  // already drained to nothing. Hold it in a ref; depend only on the length.
  const dismiss = useRef(onDismiss)
  useEffect(() => {
    dismiss.current = onDismiss
  }, [onDismiss])

  useEffect(() => {
    const raf = requestAnimationFrame(() => setDraining(true))
    const timer = setTimeout(() => dismiss.current(), ms)
    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(timer)
    }
  }, [ms])

  return (
    <div className="fixed bottom-[calc(1.5rem+var(--sg-safe-b))] left-1/2 z-40 w-[min(34rem,calc(100vw-2rem))] -translate-x-1/2 border border-grey-300 bg-newsprint max-xl:bottom-[calc(6.5rem+var(--sg-safe-b))]">
      <div className="h-[2px] w-full bg-grey-200">
        <div
          className="h-full bg-burgundy-900 ease-linear motion-reduce:transition-none"
          style={{
            width: draining ? '0%' : '100%',
            transitionProperty: 'width',
            transitionDuration: `${ms}ms`,
          }}
        />
      </div>
      <div className="flex items-start gap-4 px-4 py-3">
        <p className="flex-1 text-small leading-relaxed italic">
          <strong className="mr-2 text-[0.75rem] font-bold tracking-[0.05em] uppercase not-italic">
            Correction
          </strong>
          {text}
          {detail && <span className="block text-caption text-grey-700 not-italic">{detail}</span>}
        </p>
        <button
          type="button"
          onClick={onUndo}
          className="shrink-0 text-[0.75rem] font-semibold tracking-[0.16em] text-burgundy-900 uppercase underline underline-offset-4 not-italic hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-burgundy-900"
        >
          Undo
        </button>
      </div>
    </div>
  )
}
