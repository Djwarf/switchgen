/**
 * SwitchGen.
 *
 * The whole application is four rooms and one frame around them. The frame is
 * `components/shell` — masthead, section bar, running job, notices, undo and the
 * shortcuts card — and it owns every global key. This file does four things and
 * nothing else:
 *
 *   1. probes the machine once, for the line under the wordmark,
 *   2. renders exactly one room for the current hash,
 *   3. lends the video desk and the reel desk the real player,
 *   4. reports every desk's work to the one press ledger the section bar reads.
 *
 * Each room is its own download (point 2). The phone reloads this app often,
 * over Tailscale, and a first paint that waited for all four rooms waited for
 * code it was not about to show. The room on screen loads first; the three
 * desks that own jobs load straight after, whichever room is showing, because
 * each picks up its own saved work when it loads (clips waiting in the video
 * lane, jobs sent before a reload, the reel's run), and a desk nobody visits
 * would otherwise leave that work unfollowed. The Archive loads last.
 *
 * Point 4 is worth a sentence. Each desk keeps its own job engine, because a
 * four-minute clip must survive walking over to the pictures desk. The section
 * bar reads a single ledger. Without the three bridges below, your own picture
 * would be announced as a job this page is not following, with no Stop: the
 * ledger polls ComfyUI's queue and would see a prompt it had never been told
 * about. So the desks' stores are mirrored into the ledger here, at the one
 * place that is mounted for the life of the page.
 *
 * The reel is the third bridge and the odd one. A reel is not one job, it is a
 * queue of them walked in order, so the bridge reports each shot as its own
 * entry in the ledger, numbered, and the slug reads "Shot 3 of 8".
 */
import { Component, Suspense, lazy, useEffect, useState, type ReactNode } from 'react'
import {
  Shell,
  go,
  mirror,
  parseRoute,
  useRoute,
  type Bridge,
  type Reported,
} from './components/shell'
import { Player } from './components/player/Player'
import { posterUrl } from './components/archive/Poster'
import { startArchiveSync } from './lib/archiveSync'
import { forgetObjectInfo } from './lib/comfy'
import { onPlanLanded } from './lib/downloads'
import { gb, probeHardware, type Hardware } from './lib/hardware'
import type { PlayerSlot } from './routes/Video'

// The model list is shared by every desk (objectInfo), and a family landing
// changes it. Registered here, when the app loads and before any desk module
// has, so the shared list is dropped before a desk's own listener reads it
// again.
onPlanLanded(() => forgetObjectInfo())

// ---------------------------------------------------------------------------
// The rooms, each its own download
// ---------------------------------------------------------------------------

/** A room's code did not arrive: the connection dropped, or the app was updated since the page opened. */
class RoomLoadError extends Error {
  constructor(cause: unknown) {
    super('A room did not load.', { cause })
    this.name = 'RoomLoadError'
  }
}

/** One import per room. The browser keeps the module, so every caller shares one download. */
const loadPictures = () => import('./routes/Pictures')
const loadVideo = () => import('./routes/Video')
const loadReel = () => import('./routes/Reel')
const loadArchive = () => import('./routes/ArchivePage')

const room = <M extends { default: unknown }>(load: () => Promise<M>) => () =>
  load().catch((err: unknown) => {
    throw new RoomLoadError(err)
  })

const Pictures = lazy(room(loadPictures))
const Video = lazy(room(loadVideo))
const Reel = lazy(room(loadReel))
const ArchivePage = lazy(room(loadArchive))

/** Each room by name, for the line shown while its code is on its way. */
const ROOM_NAME = {
  pictures: 'the Pictures desk',
  video: 'the Video desk',
  reel: 'the Reel',
  archive: 'the Archive',
} as const

export default function App() {
  const route = useRoute()
  const gpu = useGpuLine()

  useJobBridges()

  // One archive for every device. Started here, the one component mounted
  // for the life of the page, so a route change never restarts it.
  useEffect(() => {
    void startArchiveSync()
  }, [])

  const label = ROOM_NAME[route.name]
  return (
    <Shell gpu={gpu}>
      <RoomBoundary key={route.name} label={label}>
        <Suspense fallback={<RoomWaiting label={label} />}>
          {route.name === 'pictures' && <Pictures />}
          {route.name === 'video' && <Video renderPlayer={renderPlayer} onNavigate={navigate} />}
          {route.name === 'reel' && <Reel renderPlayer={renderPlayer} onNavigate={navigate} />}
          {route.name === 'archive' && <ArchivePage q={route.q} />}
        </Suspense>
      </RoomBoundary>
    </Shell>
  )
}

/** While a room's code is on its way: one quiet line, and room for the page. */
function RoomWaiting({ label }: { label: string }) {
  return (
    <p role="status" className="min-h-[60vh] px-6 pt-10 text-small text-grey-500 italic">
      Opening {label}.
    </p>
  )
}

/**
 * A room whose code did not arrive says so and offers a reload, instead of
 * taking the whole page down with it. Any other error is passed on unchanged,
 * so a fault inside a room is not mistaken for a download that failed.
 */
class RoomBoundary extends Component<{ label: string; children: ReactNode }, { error: unknown }> {
  state: { error: unknown } = { error: null }

  static getDerivedStateFromError(error: unknown) {
    return { error }
  }

  render() {
    const { error } = this.state
    if (error === null) return this.props.children
    if (!(error instanceof RoomLoadError)) throw error
    return (
      <div role="alert" className="px-6 pt-10">
        <p className="text-body text-ink">
          {this.props.label.charAt(0).toUpperCase() + this.props.label.slice(1)} did not load.
        </p>
        <p className="mt-1 text-small text-grey-700">
          The connection may have dropped, or SwitchGen was updated since this page opened.
        </p>
        <button
          type="button"
          className="sg-quiet ring mt-4 [@media(pointer:coarse)]:min-h-11"
          onClick={() => window.location.reload()}
        >
          Load the page again
        </button>
      </div>
    )
  }
}

// ---------------------------------------------------------------------------
// The player, lent to the video desk
// ---------------------------------------------------------------------------

/**
 * The video and reel desks ask for a player rather than importing one, so that
 * each can be read and reviewed on its own. This is the seam. `keyboard="scoped"` keeps the
 * player's frame keys inside the player, where they cannot fight the composer's
 * prompt field two columns to the left.
 */
function renderPlayer(slot: PlayerSlot) {
  return (
    <Player
      src={slot.src}
      file={slot.file}
      entry={slot.entry}
      fps={slot.fps}
      frames={slot.frames}
      poster={posterUrl(slot.file) ?? null}
      keyboard="scoped"
    />
  )
}

/** The video desk hands us a hash; the router decides what it means. */
function navigate(hash: string): void {
  go(parseRoute(hash))
}

// ---------------------------------------------------------------------------
// The dateline's second half
// ---------------------------------------------------------------------------

/**
 * "RTX 5060 Ti · 11.4 GB free of 15.9 GB". Probed once. If the probe fails —
 * there is no local API server, the card is busy, the endpoint moved — the
 * masthead simply carries no hardware line rather than an apology.
 */
function useGpuLine(): string | null {
  const [line, setLine] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    probeHardware()
      .then((hw) => {
        if (live) setLine(describe(hw))
      })
      .catch(() => {
        /* No probe, no line. The section bar still reports the queue. */
      })
    return () => {
      live = false
    }
  }, [])

  return line
}

function describe(hw: Hardware): string | null {
  const gpu = hw.gpu
  if (gpu) {
    const name = gpu.name.replace(/^NVIDIA\s+GeForce\s+/i, '').trim()
    return gpu.vramTotal
      ? `${name} · ${gb(gpu.vramFree)} free of ${gb(gpu.vramTotal)}`
      : name
  }
  if (hw.ram?.total) return `${gb(hw.ram.free)} free of ${gb(hw.ram.total)}`
  return null
}

// ---------------------------------------------------------------------------
// The three bridges
// ---------------------------------------------------------------------------

type PicturesModule = Awaited<ReturnType<typeof loadPictures>>
type VideoModule = Awaited<ReturnType<typeof loadVideo>>
type ReelModule = Awaited<ReturnType<typeof loadReel>>

/** Tries at a desk's code before its bridge gives up; the room itself still says so when visited. */
const BRIDGE_TRIES = 3

/**
 * Mirror every desk into the press ledger, for the whole life of the page.
 *
 * Deliberately not a per-desk effect: the point of a ledger is that it keeps
 * reporting a clip while you are standing at the pictures desk, so the mirror
 * cannot be mounted by the room that started the job.
 *
 * Each bridge attaches when its desk's code arrives, which is asked for here,
 * once the first paint is done, whatever room is showing: loading a desk is
 * what picks up the work it saved, and the bridge is what lets the section
 * bar report and stop that work. The Archive's code is fetched after them,
 * so a later visit does not wait for it.
 */
function useJobBridges(): void {
  useEffect(() => {
    let live = true
    const stops: (() => void)[] = []
    const attach = <M,>(load: () => Promise<M>, bridgeOf: (m: M) => Bridge) => {
      const attempt = (left: number): Promise<void> =>
        load().then(
          (m) => {
            if (live) stops.push(mirror(bridgeOf(m)))
          },
          (): Promise<void> | void => {
            if (!live || left <= 1) return
            return new Promise<void>((resolve) => setTimeout(resolve, 3000)).then(() => attempt(left - 1))
          },
        )
      return attempt(BRIDGE_TRIES)
    }
    // After the first paint, not before it.
    const t = setTimeout(() => {
      void Promise.all([
        attach(loadPictures, pictureBridgeOf),
        attach(loadVideo, videoBridgeOf),
        attach(loadReel, reelBridgeOf),
      ]).then(() => loadArchive().catch(() => undefined))
    }, 0)
    return () => {
      live = false
      clearTimeout(t)
      for (const stop of stops) stop()
    }
  }, [])
}

/**
 * Each bridge is made once, when its desk first arrives, and kept: its `seen`
 * map is what lets a remount (StrictMode's double effect in development)
 * adopt the jobs it already opened instead of announcing them twice.
 *
 * Every bridge passes on the desk's own start, so a job taken up again after
 * a reload is timed from when it was sent, as the desk times it, and not from
 * the reload. It passes on the desk's own stage too, which the slug shows for
 * a job not sent yet. A desk's prompt id is passed on only once the job is
 * past sending: a desk may hold the id it made for a send still on its way,
 * and the ledger takes an id as the queue's word that the job is in it.
 */
let pictureBridge: Bridge | null = null
let videoBridge: Bridge | null = null
let reelBridge: Bridge | null = null

/**
 * The pictures desk. It runs one picture at a time, so it reports one job.
 *
 * Stopping goes through the desk's own Stop, which ends the whole batch. A
 * bare cancel of the picture's prompt found nothing to stop when the picture
 * had already saved, and the desk went on to make the rest of the batch.
 */
function pictureBridgeOf({ pressSnapshot, stopPress, subscribePress }: PicturesModule): Bridge {
  return (pictureBridge ??= {
    desk: 'images',
    kind: 'image',
    subscribe: subscribePress,
    stop: () => stopPress(),
    read: () => {
      const job = pressSnapshot().job
      if (!job) return []
      return [
        {
          key: job.id,
          status: job.status,
          promptId: job.status === 'submitting' ? null : job.promptId,
          label: job.label,
          prompt: '',
          value: job.value,
          max: job.max,
          entryId: null,
          error: null,
          startedAt: job.startedAt,
          stage: job.stage,
        },
      ]
    },
    seen: new Map(),
  })
}

/**
 * The video desk, one entry per clip.
 *
 * Stopping goes through the desk's own Stop, because a heavy clip can wait a
 * long time before it has a prompt to cancel: it waits its turn to release
 * ComfyUI's memory. A bare cancel had nothing to send for it, so the clip went
 * on waiting, released the memory, was sent, and only then was stopped.
 *
 * The sampling pass goes along with the steps: a Wan 2.2 14B clip samples in
 * two passes and counts each from one, and without it the section bar's rule
 * went back to empty half way through every clip.
 */
function videoBridgeOf({ stopVideoJob, videoJobs }: VideoModule): Bridge {
  return (videoBridge ??= {
    desk: 'video',
    kind: 'video',
    subscribe: videoJobs.subscribe,
    stop: (key) => stopVideoJob(key),
    read: () =>
      videoJobs.snapshot().map((job) => ({
        key: job.id,
        status: job.status,
        promptId: job.status === 'submitting' ? null : job.promptId,
        label: job.modelLabel || job.familyLabel,
        prompt: job.prompt,
        value: job.value,
        max: job.max,
        pass: job.pass,
        entryId: job.entryId,
        error: job.error,
        // From when the clip was made, as the desk's own stopwatch counts.
        startedAt: job.startedAt,
        stage: job.stage,
      })),
    seen: new Map(),
  })
}

/**
 * The reel desk.
 *
 * `reelRun` holds one run: an ordered list of shot ids and a state per shot. A
 * shot that is waiting its turn has not been submitted to anything, so it is
 * not reported; the ledger is a record of work in flight, not of intent. The
 * label carries the shot's place in the reel, because "Wan 2.2 5B" three times
 * over tells a reader nothing about how far along the queue is.
 *
 * Stopping a shot from the section bar goes through the reel's own stop,
 * which stops the walk as well as the shot. A bare cancel of the shot's prompt
 * stops only the prompt: land it just as the shot finishes and the reel goes
 * straight on to the next one.
 */
function reelBridgeOf({ reelRun }: ReelModule): Bridge {
  return (reelBridge ??= {
    desk: 'reel',
    kind: 'video',
    subscribe: reelRun.subscribe,
    stop: () => reelRun.stop(),
    read: () => {
      const run = reelRun.snapshot()
      const out: Reported[] = []
      run.order.forEach((shotId, i) => {
        const shot = run.states[shotId]
        if (!shot) return
        if (shot.status === 'waiting') return
        out.push({
          key: `${run.id}:${shotId}`,
          status: shot.status === 'stopped' ? 'cancelled' : shot.status,
          promptId: shot.promptId,
          label: `Shot ${i + 1} of ${run.order.length}`,
          prompt: '',
          value: shot.value,
          max: shot.max,
          pass: shot.pass,
          entryId: shot.entryId,
          error: shot.error,
          startedAt: shot.startedAt ?? undefined,
          stage: shot.stage || undefined,
        })
      })
      return out
    },
    seen: new Map(),
  })
}
