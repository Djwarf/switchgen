/**
 * The page a desk prints when ComfyUI is not answering.
 *
 * Every desk needs the same three things here: what is wrong, the one command
 * that fixes it, and a way to try again now rather than wait for the timer.
 * Two desks had it and the third printed a line with no way back, so it is
 * one component.
 */
import { Link } from './type'

export function ServerDown({
  onRetry,
  retryIn = null,
  detail = null,
}: {
  onRetry: () => void
  /** Seconds until the next automatic attempt, when the desk counts them. */
  retryIn?: number | null
  /** The error as the desk saw it, for the reader who wants it. */
  detail?: string | null
}) {
  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <h1 className="mb-1 text-h3 font-semibold">The server is not answering</h1>
      <p className="mb-4 text-body text-grey-700">
        We could not read the model list from ComfyUI on port 8188. It may not be running.
      </p>
      <pre className="mb-4 border border-grey-300 bg-newsprint-aged px-3 py-2 text-caption">
        systemctl --user start comfyui
      </pre>
      <p className="text-small text-grey-700">
        <Link onClick={onRetry}>Try now</Link>
        <span className="px-2 text-grey-400" aria-hidden>
          ·
        </span>
        <span className="italic text-grey-500">
          {retryIn === null ? (
            'Trying again every 5 seconds.'
          ) : (
            <>
              Trying again in <span className="tabular-nums">{retryIn}</span>s.
            </>
          )}
        </span>
      </p>
      {detail ? <p className="mt-3 text-caption text-grey-500">{detail}</p> : null}
    </main>
  )
}
