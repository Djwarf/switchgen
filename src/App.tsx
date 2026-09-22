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
 * Point 4 is worth a sentence. Each desk keeps its own job engine, because a
 * four-minute clip must survive walking over to the pictures desk. The section
 * bar reads a single ledger. Without the three bridges below, your own picture
 * would be announced as "a job started outside SwitchGen": the ledger polls
 * ComfyUI's queue and would see a prompt it had never been told about. So the
 * desks' stores are mirrored into the ledger here, at the one place that is
 * mounted for the life of the page.
 *
 * The reel is the third bridge and the odd one. A reel is not one job, it is a
 * queue of them walked in order, so the bridge reports each shot as its own
 * entry in the ledger, numbered, and the slug reads "Shot 3 of 8".
 */
import { useEffect, useState } from 'react'
import { Shell, go, jobs, parseRoute, useRoute } from './components/shell'
import { Player } from './components/player/Player'
import { posterUrl } from './components/archive/Poster'
import { startArchiveSync } from './lib/archiveSync'
import { gb, probeHardware, type Hardware } from './lib/hardware'
import ArchivePage from './routes/ArchivePage'
import Pictures, { pressSnapshot, subscribePress } from './routes/Pictures'
import Reel, { reelRun } from './routes/Reel'
import Video, { videoJobs, type PlayerSlot } from './routes/Video'

export default function App() {
  const route = useRoute()
  const gpu = useGpuLine()

  useJobBridges()

  // One archive for every device. Started here, the one component mounted
  // for the life of the page, so a route change never restarts it.
  useEffect(() => {
    void startArchiveSync()
  }, [])

  return (
    <Shell gpu={gpu}>
      {route.name === 'pictures' && <Pictures />}
      {route.name === 'video' && <Video renderPlayer={renderPlayer} onNavigate={navigate} />}
      {route.name === 'reel' && <Reel renderPlayer={renderPlayer} onNavigate={navigate} />}
      {route.name === 'archive' && <ArchivePage q={route.q} />}
    </Shell>
  )
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
// The two bridges
// ---------------------------------------------------------------------------

/** The shape both desk stores share, once the differences are flattened out. */
type Reported = {
  /** The desk's own id for the job, stable for its lifetime. */
  key: string
  status: 'submitting' | 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  promptId: string | null
  label: string
  prompt: string
  value: number
  max: number
  entryId: string | null
  error: string | null
}

type Bridge = {
  desk: 'images' | 'video'
  kind: 'image' | 'video'
  subscribe: (fn: () => void) => () => void
  read: () => Reported[]
  /**
   * Desk job id → ledger job id, kept on the bridge rather than inside the
   * mirror, so that a remount — StrictMode's double effect in development, or
   * the shell being torn down and rebuilt — adopts the jobs it already opened
   * instead of announcing the same clip twice. A blank value means "seen, and
   * already finished before we looked".
   */
  seen: Map<string, string>
}

/**
 * Mirror every desk into the press ledger, for the whole life of the page.
 *
 * Deliberately not a per-desk effect: the point of a ledger is that it keeps
 * reporting a clip while you are standing at the pictures desk, so the mirror
 * cannot be mounted by the room that started the job.
 */
function useJobBridges(): void {
  useEffect(() => {
    const stops = [mirror(pictureBridge), mirror(videoBridge), mirror(reelBridge)]
    return () => {
      for (const stop of stops) stop()
    }
  }, [])
}

const pictureBridge: Bridge = {
  desk: 'images',
  kind: 'image',
  subscribe: subscribePress,
  read: () => {
    const job = pressSnapshot().job
    if (!job) return []
    return [
      {
        key: job.id,
        status: job.status,
        promptId: job.promptId,
        label: job.label,
        prompt: '',
        value: job.value,
        max: job.max,
        entryId: null,
        error: null,
      },
    ]
  },
  seen: new Map(),
}

const videoBridge: Bridge = {
  desk: 'video',
  kind: 'video',
  subscribe: videoJobs.subscribe,
  read: () =>
    videoJobs.snapshot().map((job) => ({
      key: job.id,
      status: job.status,
      promptId: job.promptId,
      label: job.modelLabel || job.familyLabel,
      prompt: job.prompt,
      value: job.value,
      max: job.max,
      entryId: job.entryId,
      error: job.error,
    })),
  seen: new Map(),
}

/**
 * The reel desk.
 *
 * `reelRun` holds one run: an ordered list of shot ids and a state per shot. A
 * shot that is waiting its turn has not been submitted to anything, so it is
 * not reported; the ledger is a record of work in flight, not of intent. The
 * label carries the shot's place in the reel, because "Wan 2.2 5B" three times
 * over tells a reader nothing about how far along the queue is.
 */
const reelBridge: Bridge = {
  desk: 'video',
  kind: 'video',
  subscribe: reelRun.subscribe,
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
        entryId: shot.entryId,
        error: shot.error,
      })
    })
    return out
  },
  seen: new Map(),
}

const LIVE = new Set(['submitting', 'queued', 'running'])
const SEEN_LIMIT = 200

function mirror(bridge: Bridge): () => void {
  const seen = bridge.seen

  const sync = () => {
    for (const report of bridge.read()) {
      let id = seen.get(report.key)

      if (id === undefined) {
        // Work that was already finished when the bridge first looked belongs
        // to the archive, not to the slug. Note it and leave it alone.
        if (!LIVE.has(report.status)) {
          seen.set(report.key, '')
          continue
        }
        id = jobs.start({
          desk: bridge.desk,
          kind: bridge.kind,
          label: report.label,
          prompt: report.prompt,
          promptId: report.promptId,
          steps: report.max || undefined,
        })
        seen.set(report.key, id)
      }
      if (!id) continue

      const ledgerJob = jobs.get(id)
      if (!ledgerJob) continue

      if (report.promptId && ledgerJob.promptId !== report.promptId) {
        jobs.attach(id, report.promptId)
      }

      switch (report.status) {
        case 'running':
          if (
            ledgerJob.status !== 'running' ||
            ledgerJob.value !== report.value ||
            ledgerJob.max !== report.max
          ) {
            jobs.apply(id, {
              phase: 'running',
              node: null,
              value: report.value,
              max: report.max,
            })
          }
          break
        case 'done':
          if (LIVE.has(ledgerJob.status)) {
            jobs.succeed(id, report.entryId ? { entryId: report.entryId } : undefined)
          }
          break
        case 'error':
          if (LIVE.has(ledgerJob.status)) {
            jobs.fail(id, report.error ?? 'The job stopped short.')
          }
          break
        case 'cancelled':
          if (LIVE.has(ledgerJob.status)) {
            jobs.fail(id, report.error ?? 'Stopped.', { cancelled: true })
          }
          break
        default:
          break
      }
    }

    // The map is a lookup for work in progress, not a second archive. A page
    // left open all day should not accumulate one entry per picture for ever.
    if (seen.size > SEEN_LIMIT) {
      const live = new Set(bridge.read().map((r) => r.key))
      for (const key of [...seen.keys()]) {
        if (seen.size <= SEEN_LIMIT / 2) break
        if (!live.has(key)) seen.delete(key)
      }
    }
  }

  sync()
  return bridge.subscribe(sync)
}
