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
import { Shell, go, mirror, parseRoute, useRoute, type Bridge, type Reported } from './components/shell'
import { Player } from './components/player/Player'
import { posterUrl } from './components/archive/Poster'
import { startArchiveSync } from './lib/archiveSync'
import { gb, probeHardware, type Hardware } from './lib/hardware'
import ArchivePage from './routes/ArchivePage'
import Pictures, { pressSnapshot, stopPress, subscribePress } from './routes/Pictures'
import Reel, { reelRun } from './routes/Reel'
import Video, { stopVideoJob, videoJobs, type PlayerSlot } from './routes/Video'

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
// The three bridges
// ---------------------------------------------------------------------------

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

/**
 * The pictures desk. It runs one picture at a time, so it reports one job.
 *
 * Stopping goes through the desk's own Stop, which ends the whole batch. A
 * bare cancel of the picture's prompt found nothing to stop when the picture
 * had already saved, and the desk went on to make the rest of the batch.
 */
const pictureBridge: Bridge = {
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

/**
 * The video desk, one entry per clip.
 *
 * Stopping goes through the desk's own Stop, because a heavy clip can wait a
 * long time before it has a prompt to cancel: it waits its turn to release
 * ComfyUI's memory. A bare cancel had nothing to send for it, so the clip went
 * on waiting, released the memory, was sent, and only then was stopped.
 */
const videoBridge: Bridge = {
  desk: 'video',
  kind: 'video',
  subscribe: videoJobs.subscribe,
  stop: (key) => stopVideoJob(key),
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
 *
 * Stopping a shot from the section bar goes through the reel's own stop,
 * which stops the walk as well as the shot. A bare cancel of the shot's prompt
 * stops only the prompt: land it just as the shot finishes and the reel goes
 * straight on to the next one.
 */
const reelBridge: Bridge = {
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
        entryId: shot.entryId,
        error: shot.error,
      })
    })
    return out
  },
  seen: new Map(),
}
