/**
 * The server is not answering.
 *
 * ComfyUI being down is not a crash and must not be presented as one: the
 * archive still reads, drafts still save, and the moment the socket comes back
 * the app carries on without a reload. So this is a band under the section bar,
 * not a modal over the page — it tells you what happened, why, and the one
 * command that fixes it.
 */
import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { connect, connectionState, watchConnection, type ConnectionState } from '../../lib/comfy'
import { postNotice } from './Notice'

const subscribe = (fn: () => void): (() => void) => watchConnection(() => fn())

/** The shared socket's state: `connecting`, `open` or `closed`. */
export function useConnection(): ConnectionState {
  return useSyncExternalStore(subscribe, connectionState, connectionState)
}

/** How long a closed socket waits before we admit to it, in ms. */
const GRACE = 1500
const RETRY_SECONDS = 5

export function Offline() {
  const state = useConnection()
  const [shown, setShown] = useState(false)
  const [countdown, setCountdown] = useState(RETRY_SECONDS)
  const wasDown = useRef(false)

  // Open the socket as soon as the shell mounts, rather than at the first job.
  useEffect(() => {
    connect()
  }, [])

  // A blip while the socket reconnects is not news. A second and a half of
  // silence is.
  useEffect(() => {
    if (state === 'open') {
      if (shown) setShown(false)
      if (wasDown.current) {
        wasDown.current = false
        postNotice({
          key: 'connection',
          tone: 'success',
          title: 'Back',
          body: 'The server is answering again. Anything that was running has been picked up.',
          ttl: 5000,
        })
      }
      return
    }
    const t = setTimeout(() => {
      setShown(true)
      wasDown.current = true
    }, GRACE)
    return () => clearTimeout(t)
  }, [state, shown])

  useEffect(() => {
    if (!shown) return
    setCountdown(RETRY_SECONDS)
    const t = setInterval(() => {
      setCountdown((n) => {
        if (n <= 1) {
          connect()
          return RETRY_SECONDS
        }
        return n - 1
      })
    }, 1000)
    return () => clearInterval(t)
  }, [shown])

  if (!shown) return null

  return (
    <div className="border-b border-grey-300 bg-newsprint px-6 py-3" role="alert">
      <div className="mx-auto w-full max-w-[110rem]">
        <div className="notice notice-error text-small">
          <p className="m-0">
            <strong className="block not-italic">THE SERVER IS NOT ANSWERING</strong>
            We could not reach ComfyUI on port 8188. It may not be running.
          </p>
          <p className="mt-2 mb-0">
            Start it with{' '}
            <code className="figures bg-newsprint-aged px-1.5 py-0.5 text-caption">
              systemctl --user start comfyui
            </code>
          </p>
          <p className="mt-2 mb-0 flex items-baseline gap-4 not-italic">
            <span className="text-grey-700 italic">
              Trying again in <span className="figures">{countdown}s</span>…
            </span>
            <button
              type="button"
              className="sg-link ring"
              onClick={() => {
                connect()
                setCountdown(RETRY_SECONDS)
              }}
            >
              Try now
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}
