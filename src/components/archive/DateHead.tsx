/**
 * The rule that separates one day's work from the next, so the archive reads
 * as a run of back issues rather than an undifferentiated wall.
 */
export function DateHead({ label, count }: { label: string; count: number }) {
  return (
    <div className="sticky top-[calc(var(--sg-bar-h)+var(--sg-safe-t))] z-20 -mx-1 bg-newsprint px-1 pt-5 pb-2">
      <div className="flex items-baseline gap-3">
        <h2 className="text-[0.625rem] font-semibold tracking-[0.18em] text-burgundy-900 uppercase">
          {label}
        </h2>
        <span className="h-px flex-1 bg-grey-300" aria-hidden />
        <span className="text-[0.625rem] font-semibold tracking-[0.18em] text-grey-500 uppercase tabular-nums">
          {count.toLocaleString('en-GB')} {count === 1 ? 'record' : 'records'}
        </span>
      </div>
    </div>
  )
}
