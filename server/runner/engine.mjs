/**
 * The queue on the server: the one dispatcher of the desks' waiting work.
 *
 * A page used to hold its own waiting work (the Video lane, the reel's walk,
 * a batch of pictures) and send each job to ComfyUI itself, so the work
 * stopped whenever the phone slept or the tab was thrown away. The pages now
 * hand the whole of it here in one request, and this sends it in turn, files
 * what comes back, and tells every page what happened. It keeps the page's
 * memory rules, now applied in one place for every desk and every device.
 *
 * The rules, in the order a job meets them:
 *
 *   - Depth 1. At most one job of ours is releasing, sending, queued or
 *     running at a time. So "one heavy clip at a time" holds across the Video
 *     desk and the reel, a ComfyUI restart costs at most one job, and nothing
 *     of ours can slip between a release and the prompt it was meant for.
 *   - Turns. Among the jobs that may go, the group served least recently goes
 *     first, then the oldest job, so a picture pressed during a reel goes
 *     after the shot on the press, not after the whole reel. A heavy job that
 *     has been picked stays the pick until it is sent or stopped, and light
 *     work waits behind it rather than keeping ComfyUI's queue from emptying.
 *   - A heavy job waits for ComfyUI's queue to be empty, asks it to release
 *     its memory, waits a moment, checks the queue is still empty, and only
 *     then sends its prompt. The release is a flag ComfyUI's worker reads when
 *     it takes its next prompt, and a prompt put in the queue just before the
 *     worker wakes can take it first; the pause and the second read close
 *     most of that window, though nothing here can promise to close all of it.
 *   - A send is sorted by what is known about it. Refused, and the job fails
 *     with ComfyUI's words. Never reached, and it waits and goes again under a
 *     new id. Anything unclear, and the job is never sent again: ComfyUI is
 *     asked about it by its id until it says.
 *   - A job ComfyUI has lost holds the lane for the heavy work waiting behind
 *     it, until the reader says, because a ComfyUI that just died of memory
 *     would die again on the next heavy clip.
 *
 * Every change of a job's status is committed to disk before the thing it
 * allows (see store.mjs), and every decision is made from the committed state
 * after each wait, never from a copy taken before it: a stop from a page can
 * land in the middle of any of these steps.
 */
import { randomUUID } from 'node:crypto'
import { promises as fs, readFileSync } from 'node:fs'
import { confineReal } from '../guard.mjs'
import { annotatedRef, readPastRun, relOf, samplerPass, splice } from './comfyRecord.mjs'
import { fileJob, filingPatch, relsOf } from './filing.mjs'
import { StoreWriteError, TERMINAL, createStore } from './store.mjs'

/** A job in one of these is ComfyUI's business, or about to be: depth counts them. */
const LIVE = new Set(['releasing', 'sending', 'queued', 'running'])

/** A prompt accepted seconds ago may not be listed yet (comfy.ts LOST_GRACE_MS). */
const GRACE_MS = 20_000
/** Consecutive absences before the job is looked for in /history (comfy.ts LOST_MISSES). */
const MISSES = 2
/** Reads of a history record ComfyUI says exists before giving up on it (comfy.ts LOST_TERMINAL_WAITS). */
const HISTORY_WAITS = 3
/** How often ComfyUI's queue is read while a heavy job waits on it. */
const QUEUE_EVERY_MS = 2000
/** How often the queue looks at its work while anything is unfinished, and while nothing is. */
const LIVE_TICK_MS = 1000
const IDLE_TICK_MS = 10_000
/** Ended jobs are kept this long, and no more than this many, for pages to read. */
const KEEP_ENDED_MS = 48 * 3_600_000
const KEEP_ENDED = 500
/**
 * A reel pass no page has taken in yet is kept far longer, apart from that
 * count: only a page of the browser that pressed it can fold its clips into
 * that strip, and a phone may not open the app again for days.
 */
const KEEP_PASS_MS = 30 * 24 * 3_600_000
const KEEP_PASS_JOBS = 2000
const PRUNE_EVERY_MS = 60_000
/** How often a stop that ComfyUI did not take is asked again while the job is still its. */
const CANCEL_EVERY_MS = 2000
/**
 * How long a stop from a page waits on ComfyUI's answer to its cancel before
 * it answers the page, well inside the 15 s a page waits for it. The stop is
 * on disk before the cancel goes, and a cancel with no answer yet is asked
 * again by the passes that follow the job.
 */
const STOP_WAIT_MS = 3000
/** Passes over the queue that fail in a row before it stops taking work. */
const FAULTS_TO_TRIP = 5
/** Progress events per job, at most, per second: 4. */
const PROGRESS_EVERY_MS = 250

export const REASONS = {
  off: 'Turned off with SWITCHGEN_RUNNER=off.',
  held: 'Another SwitchGen server holds the archive, and the queue with it.',
  folder: 'Another SwitchGen server runs the queue from the same folder.',
  unwritable: (err) => `The server cannot write its list of work: ${String(err?.message ?? err).replace(/\.$/, '')}.`,
  tripped: 'The queue stopped after repeated faults; see the server log.',
  oldComfy: 'This ComfyUI is older than the server queue needs (it has no jobs list); pages send their own work.',
  stopped: 'The queue on the server is not running.',
}

export const DESKS = ['video', 'images', 'reel']

/**
 * This runner has handed over (retired, or lost the archive to another
 * server) while a step was under way. The step ends where it is, as a process
 * killed there would, and the next runner takes the work up from disk.
 */
export class Retired extends Error {
  constructor() {
    super('the queue has handed over')
    this.name = 'Retired'
  }
}

/** A commit could not be written; the change it carried did not happen. */
export class DiskError extends Error {
  constructor(err) {
    super(String(err?.message ?? err), { cause: err })
    this.name = 'DiskError'
    this.code = err?.code ?? null
  }
}

/** This boot of the machine, or null where Linux does not say. */
function readMachineBoot() {
  try {
    return readFileSync('/proc/sys/kernel/random/boot_id', 'utf8').trim() || null
  } catch {
    return null
  }
}

/**
 * Whether the archive's lock file names this process. archiveApi.holds() is
 * also true where no lock could be made at all (a folder that cannot be
 * written), which the archive puts up with for itself; the queue does not,
 * because two servers each sending the same work to ComfyUI is the thing the
 * lock is there to prevent. A fake archive with no file to lock (a test's)
 * is taken at its word.
 */
function lockNamesMe(archive) {
  if (typeof archive?.archiveFile !== 'string' || !archive.archiveFile) return true
  try {
    return JSON.parse(readFileSync(`${archive.archiveFile}.lock`, 'utf8'))?.pid === process.pid
  } catch {
    return false
  }
}

/** Fields a job keeps for the queue's own use, left out of what pages read. */
const INTERNAL = [
  'promptIdInternal',
  'sighted',
  'misses',
  'historyWaits',
  'primaryKind',
  'orFirst',
  'noFile',
  'chain',
  'graphSha',
  'specSha',
  'acceptedAt',
  'sawEndAt',
]

const sameWait = (a, b) => (a ?? null) === (b ?? null) || (!!a && !!b && a.for === b.for && (a.ahead ?? null) === (b.ahead ?? null))

/**
 * Whether a hold on the lane covers this job. A hold after a lost or unsent
 * clip covers every heavy job. A hold on all the work (after the machine
 * restarted, or while the queue was off) covers the work that was waiting
 * when it began. Work made after it did not wait through the restart or the
 * pause, so it goes; but once a heavy clip is lost while it stands
 * (`heavyAfter`), every heavy job waits as well, as after any lost clip.
 */
const covers = (held, job) =>
  !!held &&
  (held.scope === 'all'
    ? (job.createdAt ?? 0) <= held.since || (held.heavyAfter != null && job.heavy === true)
    : job.heavy === true)

/**
 * When a hold on all the work begins: now, or later if the clock says a job
 * it must cover was made later than that (a clock set back after a restart),
 * so it covers every one of them.
 */
const holdSince = (t, jobs) => Math.max(t, ...jobs.map((j) => j.createdAt ?? 0))

/**
 * Whether a list has work that the queue taking it up would send by itself:
 * a waiting job, or one left releasing, which goes back to waiting.
 */
const waitsToGo = (s) => !!s && Object.values(s.jobs).some((j) => j.status === 'waiting' || j.status === 'releasing')

/**
 * Whether this ending of one of its jobs ends a group. A batch of pictures
 * goes on past a picture that wrote no file, as the page's own batch always
 * has; any other failure ends it. A reel pass ends on anything but done,
 * because every shot after may open on the one that did not finish. Clips
 * are independent.
 */
function endsGroup(kind, job) {
  if (kind === 'batch') {
    if (job.status === 'failed') return job.error?.code !== 'no-file'
    return job.status === 'stopped' || job.status === 'lost' || job.status === 'unsent'
  }
  if (kind === 'pass') return TERMINAL.has(job.status) && job.status !== 'done' && job.status !== 'skipped'
  return false
}

function whyEnded(job) {
  if (job.status === 'failed') {
    if (job.error?.code === 'refused') return 'refused'
    if (job.error?.code === 'no-frame') return 'no-frame'
    return 'failed'
  }
  return job.status
}

/** A RunnerError with every field present. */
export function fault(code, extra = {}) {
  return { code, message: null, node: null, nodeType: null, nodeErrors: null, mayExist: false, sent: false, after: null, ...extra }
}

export function createEngine(opts) {
  const comfy = opts.comfy
  const archive = opts.archive
  const outputs = opts.outputs
  const desks = (Array.isArray(opts.desks) ? opts.desks : DESKS).filter((d) => DESKS.includes(d))
  // SWITCHGEN_RUNNER=off wins over anything a caller passes: the queue talks
  // to ComfyUI directly, and the switch is the reader's way to keep it from
  // doing so at all.
  const enabled = process.env.SWITCHGEN_RUNNER !== 'off' && opts.enabled !== false
  const settleMs = typeof opts.settleMs === 'number' && opts.settleMs >= 0 ? opts.settleMs : 1000
  const stopWaitMs = typeof opts.stopWaitMs === 'number' && opts.stopWaitMs >= 0 ? opts.stopWaitMs : STOP_WAIT_MS
  const now = opts.now ?? Date.now
  const sleep = opts.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)))
  const machineBoot = opts.machineBoot ?? readMachineBoot
  const autoTick = opts.autoTick !== false
  /** This runner, as pages tell one from the next: a new one means read everything again. */
  const boot = randomUUID()
  const store = createStore({ dir: opts.dir, now })

  let loaded = false
  let saidUnwritable = false
  let saidFolder = false
  let retired = false
  /**
   * This queue, while not running, saw work waiting on its list with no other
   * queue running the folder: when it runs, it holds that work until the
   * reader says (paused.json keeps the same word for the next process).
   */
  let pausedSeen = false
  let tripped = false
  let faults = 0
  let reason = null
  /** Bumped each time the runner takes the list up or lets it go, so a pass begun before cannot write after. */
  let generation = 0
  let tickGen = -1

  /** The heavy job picked and waiting for ComfyUI's queue to empty; it stays the pick. */
  let head = null
  let nextQueueReadAt = 0
  let lastPrune = 0
  let comfyState = { answering: null, since: now() }
  /**
   * Whether the ComfyUI answering now has the jobs list every job is followed
   * by: null until it has said. Asked again whenever it answers after it did
   * not (`lookAgain`), since the ComfyUI that comes back may not be the one
   * that went. A client with no way to ask (a test's stand-in) has it.
   */
  const canAsk = typeof comfy.hasJobsList === 'function'
  let jobsList = canAsk ? null : true
  let lookAgain = false
  /** Jobs whose last commit could not be written, shown as waiting for room. */
  const diskWait = new Set()
  let diskSaid = false

  const progress = new Map()
  const previews = new Map()
  const graphs = new Map()
  /** The graph a waiting pick will send, made once rather than read from disk on every pass it waits. */
  const toSend = new Map()
  /** execution_start stamps heard on the socket, by prompt id. */
  const starts = new Map()
  /** Prompts running ahead of a queued heavy job that a release has been sent for, by job. */
  const freedFor = new Map()
  /** When a stop was last asked of ComfyUI for a job, so one it did not take is asked again. */
  const lastCancel = new Map()
  const lastForeignRead = new Map()
  const lastProgressEmit = new Map()
  const progressTimers = new Map()
  let executing = null
  let sock = null

  const listeners = new Set()
  let timer = null
  let ticking = null
  let again = false

  // ------------------------------------------------------------- reading --

  const S = () => store.current()
  // Own entries only: an id comes from a URL, and `__proto__` is not a job.
  const job = (id) => {
    const s = S()
    return s && typeof id === 'string' && Object.hasOwn(s.jobs, id) ? s.jobs[id] : null
  }
  const group = (id) => {
    const s = S()
    return s && typeof id === 'string' && Object.hasOwn(s.groups, id) ? s.groups[id] : null
  }
  const jobsWhere = (pred) => Object.values(S()?.jobs ?? {}).filter(pred).sort((a, b) => a.seq - b.seq)

  function view(j) {
    const v = { ...j }
    for (const k of INTERNAL) delete v[k]
    if (diskWait.has(j.id) && !TERMINAL.has(j.status)) v.wait = { for: 'disk' }
    return v
  }

  function emit(event, data) {
    for (const fn of listeners) {
      try { fn(event, data) } catch { /* one watcher must not stop the rest */ }
    }
  }

  function subscribe(fn) {
    listeners.add(fn)
    return () => listeners.delete(fn)
  }

  // ------------------------------------------------------------ commits --

  /**
   * End a job inside a commit. A heavy job lost or never sent holds the lane
   * for the heavy work waiting behind it, in the same commit as the loss, so
   * nothing can be dispatched in between; but only when heavy work is waiting
   * that no hold covers yet, as the page's own lane did.
   *
   * When a hold on all the work already stands, the lane has room for that
   * one hold, so the loss is added to it: from now on it covers every heavy
   * job too (`heavyAfter`) and names the lost one. Its `since` stays, so it
   * still covers no light work made after it began, and a word naming it
   * still answers it.
   */
  function endJob(d, id, status, error, fields = {}) {
    const cur = d.jobs[id]
    if (!cur) return
    const t = now()
    d.jobs[id] = { ...cur, ...fields, status, wait: null, endedAt: t, error }
    if ((status === 'lost' || status === 'unsent') && cur.heavy) {
      const held = d.lane.held
      const behind = Object.values(d.jobs).some((o) => o.id !== id && o.status === 'waiting' && o.heavy && !covers(held, o))
      if (behind) {
        d.lane = { held: held ? { ...held, jobId: id, heavyAfter: t } : { why: status, scope: 'heavy', jobId: id, since: t } }
      }
    }
  }

  /**
   * What every commit settles before it is written: groups that a job's
   * ending ended, with their waiting jobs skipped, groups whose jobs have all
   * ended, a hold with nothing left under it, and a hold on all the work that
   * a lost clip was added to, once only the heavy work behind that loss is
   * left under it: it is then the hold for that loss, and says so.
   */
  function normalize(d) {
    const t = now()
    for (const g of Object.values(d.groups)) {
      if (g.state !== 'active') continue
      const members = g.jobIds.map((id) => d.jobs[id]).filter(Boolean)
      const ender = g.kind === 'clips' ? null : members.find((j) => endsGroup(g.kind, j)) ?? null
      if (ender) {
        for (const j of members) {
          if (j.status !== 'waiting') continue
          d.jobs[j.id] = {
            ...j,
            status: 'skipped',
            wait: null,
            endedAt: t,
            error: fault('skipped', { after: { jobId: ender.id, index: ender.index } }),
          }
        }
        d.groups[g.id] = { ...g, state: 'ended', endedBy: { jobId: ender.id, why: whyEnded(ender) }, endedAt: t }
      } else if (members.every((j) => TERMINAL.has(j.status))) {
        d.groups[g.id] = { ...g, state: 'ended', endedAt: t }
      }
    }
    const held = d.lane.held
    if (!held) return
    const under = Object.values(d.jobs).filter((j) => j.status === 'waiting' && covers(held, j))
    if (!under.length) {
      d.lane = { held: null }
    } else if (held.scope === 'all' && held.heavyAfter != null && !under.some((j) => (j.createdAt ?? 0) <= held.since)) {
      const why = d.jobs[held.jobId]?.status === 'unsent' ? 'unsent' : 'lost'
      d.lane = { held: { why, scope: 'heavy', jobId: held.jobId, since: held.heavyAfter } }
    }
  }

  /** Tell every watcher what a commit changed, each event carrying its rev. */
  function publish(res) {
    const s = S()
    let said = false
    for (const id of res.jobs) {
      const j = s.jobs[id]
      diskWait.delete(id)
      if (TERMINAL.has(j.status)) forgetLive(j)
      emit('job', { rev: res.rev, job: view(j) })
      said = true
    }
    for (const id of res.groups) {
      emit('group', { rev: res.rev, group: s.groups[id] })
      said = true
    }
    if (res.lane) {
      emit('lane', { rev: res.rev, lane: s.lane })
      said = true
    }
    if (res.goneJobs.length || res.goneGroups.length) {
      emit('gone', { rev: res.rev, jobs: res.goneJobs, groups: res.goneGroups })
      said = true
    }
    // Every rev is heard, or a page would take the gap for a missed event
    // and read everything again.
    if (!said) emit('lane', { rev: res.rev, lane: s.lane })
  }

  /** Drop what is kept in memory for a job only while it runs. */
  function forgetLive(j) {
    progress.delete(j.id)
    previews.delete(j.id)
    graphs.delete(j.id)
    toSend.delete(j.id)
    freedFor.delete(j.id)
    lastCancel.delete(j.id)
    lastForeignRead.delete(j.id)
    lastProgressEmit.delete(j.id)
    const pt = progressTimers.get(j.id)
    if (pt) clearTimeout(pt)
    progressTimers.delete(j.id)
    if (j.promptIdInternal) starts.delete(j.promptIdInternal)
    if (head === j.id) head = null
  }

  /**
   * Commit one change: `mutate` edits a draft (see store.mjs), then the
   * groups and the lane are settled, the whole is written, and every watcher
   * hears of it. Throws Retired once this runner has handed over, and
   * DiskError when the write failed, in which case nothing changed.
   */
  function commit(mutate) {
    if (retired || !loaded) throw new Retired()
    let res
    try {
      res = store.commit((d) => {
        const out = mutate(d)
        if (out === false) return false
        normalize(d)
        return out
      })
    } catch (err) {
      if (err instanceof StoreWriteError) throw new DiskError(err)
      throw err
    }
    if (!res) return null
    diskSaid = false
    publish(res)
    return res
  }

  /** Throws Retired when the pass under way belongs to a runner that has handed over. */
  function alive() {
    if (retired || !loaded || tickGen !== generation) throw new Retired()
  }

  /** A commit made by a pass over the queue: refused once the pass is stale. */
  function tcommit(mutate) {
    alive()
    return commit(mutate)
  }

  /** Change one job when it is still as `ok` says, with `patch` or what it returns. */
  function patchJob(id, ok, patch, via = tcommit) {
    return via((d) => {
      const cur = d.jobs[id]
      if (!cur || !ok(cur)) return false
      const p = typeof patch === 'function' ? patch(cur, d) : patch
      if (p === false) return false
      if (p) d.jobs[id] = { ...cur, ...p }
    })
  }

  function setWait(id, status, wait) {
    return patchJob(id, (cur) => cur.status === status && !sameWait(cur.wait, wait), { wait })
  }

  /**
   * Inside a commit: a job that was on its way out (releasing, or sent to no
   * one) goes back to waiting, unless a stop was asked for meanwhile, when it
   * ends there, never sent. Left waiting with its stop asked, it would never
   * be picked again, nor ever end.
   */
  function backToWaiting(d, id, fields) {
    const cur = d.jobs[id]
    if (cur.stopRequested) endJob(d, id, 'stopped', fault('stopped', { sent: false, message: 'Stopped before it was sent.' }))
    else d.jobs[id] = { ...cur, status: 'waiting', ...fields }
  }

  /** Commit backToWaiting for a job still in `status`. */
  function returnToWaiting(id, status, fields) {
    return tcommit((d) => {
      const cur = d.jobs[id]
      if (!cur || cur.status !== status) return false
      backToWaiting(d, id, fields)
    })
  }

  /** A commit for this job could not be written: show it waiting for room, and try again next pass. */
  function markDisk(id, err) {
    if (!diskSaid) console.warn(`[switchgen-runner] could not write the list of work: ${err?.message ?? err}; waiting for room`)
    diskSaid = true
    const j = job(id)
    if (!j || diskWait.has(id)) return
    diskWait.add(id)
    emit('job', { rev: S().rev, job: view(j) })
  }

  /** The archive could not take a job's record now: it stays filing, waiting, and is tried again next pass. */
  function waitForDisk(id, err) {
    const j = job(id)
    if (j && !(j.wait?.for === 'disk')) {
      console.warn(`[switchgen-runner] could not file ${id} yet: ${err?.message ?? err}; trying again`)
    }
    patchJob(id, (cur) => cur.status === 'filing' && cur.wait?.for !== 'disk', { wait: { for: 'disk' } })
  }

  function answering(yes) {
    if (!yes && canAsk) lookAgain = true
    if (comfyState.answering === yes) return
    comfyState = { answering: yes, since: now() }
    emit('comfy', { ...comfyState })
  }

  // ---------------------------------------------------------- lifecycle --

  /**
   * Whether the queue may run now, taking the list up when it newly may. It
   * runs only while this process holds the archive lock and the lock of its
   * own folder: a second server on the same outputs, or pointed at the same
   * folder, stands back, and its pages send their own work.
   */
  function ensureActive() {
    if (retired) { reason = REASONS.stopped; return false }
    if (!enabled) {
      reason = REASONS.off
      notePaused()
      return false
    }
    if (tripped) {
      reason = REASONS.tripped
      notePaused()
      return false
    }
    let holds = false
    try { holds = archive.holds() === true && lockNamesMe(archive) } catch { holds = false }
    if (!holds) {
      reason = REASONS.held
      if (loaded) {
        console.warn('[switchgen-runner] another server holds the archive now; this queue stands back')
        deactivate()
      }
      notePaused()
      return false
    }
    if (loaded && !store.stillLocked(boot)) {
      reason = REASONS.folder
      console.warn(`[switchgen-runner] another server took ${store.dir}; this queue stands back`)
      deactivate()
      notePaused()
      return false
    }
    // Without the jobs list, every job the queue sent would read as gone
    // from ComfyUI while it still ran, and be called lost. The page's own
    // path does without it, so the pages send their work.
    if (jobsList === false) {
      reason = REASONS.oldComfy
      if (loaded) {
        console.warn(`[switchgen-runner] ComfyUI at ${comfy.url} has no jobs list (/api/jobs); this queue stands back and pages send their own work`)
        deactivate()
      }
      notePaused()
      return false
    }
    if (!loaded) {
      let taken
      try {
        taken = store.lock(boot)
      } catch (err) {
        return cannotWrite(err)
      }
      if (taken) {
        if (!saidFolder) console.warn(`[switchgen-runner] another server${taken.pid ? ` (process ${taken.pid})` : ''} runs the queue from ${store.dir}; this one stands back`)
        saidFolder = true
        reason = REASONS.folder
        notePaused()
        return false
      }
      saidFolder = false
      try {
        activate()
      } catch (err) {
        store.close()
        store.unlock(boot)
        loaded = false
        return cannotWrite(err)
      }
      saidUnwritable = false
    }
    reason = null
    return true
  }

  function cannotWrite(err) {
    reason = REASONS.unwritable(err instanceof DiskError ? err.cause ?? err : err)
    if (!saidUnwritable) console.warn(`[switchgen-runner] ${reason} Pages send their own work until it can.`)
    saidUnwritable = true
    notePaused()
    return false
  }

  /**
   * While this queue is not running (turned off, standing back, or stopped
   * after repeated faults): when its list has work waiting and no other queue
   * runs the folder, that work is waiting on a queue that is off. Noted here
   * and in the folder, so that whichever queue next runs it holds it until
   * the reader says, rather than sending it unasked, perhaps after the reader
   * made it again in the page. Another queue running the folder sends that
   * work itself, and nothing is noted.
   *
   * A queue that stopped after repeated faults still holds its folder and its
   * list, so no other queue runs that work, and it notes it itself: its pages
   * send their own work meanwhile, and the queue that takes the folder next,
   * after a restart, must not send the same work unasked.
   */
  function notePaused() {
    if (retired) return
    if (loaded) {
      if (tripped && waitsToGo(S())) store.notePaused({ since: now(), pid: process.pid, reason })
      return
    }
    const saved = store.peek()
    if (!waitsToGo(saved)) return
    if (store.holder()) {
      pausedSeen = false
      return
    }
    pausedSeen = true
    store.notePaused({ since: now(), pid: process.pid, reason })
  }

  function activate() {
    store.open()
    loaded = true
    generation += 1
    head = null
    nextQueueReadAt = 0
    reconcile(pausedSeen || store.pausedNote() !== null)
    store.clearPaused()
    pausedSeen = false
    for (const j of jobsWhere((x) => x.status === 'filing')) archive.claim(relsOf(j))
    openSocket()
    emit('state', snapshot())
  }

  function deactivate() {
    generation += 1
    loaded = false
    store.close()
    store.unlock(boot)
    head = null
    progress.clear()
    previews.clear()
    graphs.clear()
    lastCancel.clear()
    diskWait.clear()
    void closeSocket()
    emit('state', snapshot())
  }

  /**
   * Take up the list as the last runner left it.
   *
   *   - A job left releasing had its release sent, perhaps, and its prompt
   *     not: it waits again and releases again.
   *   - A job left sending may or may not be in ComfyUI's queue. It is never
   *     sent again; the next pass asks ComfyUI about it by its id.
   *   - Queued, running and filing jobs are followed and filed as before.
   *   - A job still to be done whose saved graph has gone cannot be done.
   *   - After the machine itself restarted, hours may have passed and ComfyUI
   *     came back empty, so waiting work is held until the reader says. An
   *     ordinary restart of this server holds nothing.
   *   - Work that waited while the queue was off, or stood back with no other
   *     queue running it (`paused`), is held the same way: its pages showed
   *     it waiting on a queue that was off, and the reader may have made it
   *     again in the page meanwhile.
   *
   * Either hold covers the work waiting now and none made after. One that
   * already stands and covers all of it is left as it is, so the reader's
   * answer to it still counts; a restart takes the place of any other kind.
   */
  function reconcile(paused = false) {
    const bootNow = machineBoot()
    commit((d) => {
      let changed = false
      for (const j of Object.values(d.jobs)) {
        if (TERMINAL.has(j.status)) continue
        if (!store.hasPayload(j.id)) {
          endJob(d, j.id, 'failed', fault('internal', { sent: j.status !== 'waiting' && j.status !== 'releasing', message: 'The saved graph for this job is missing, so it cannot be sent or filed.' }))
          changed = true
        } else if (j.status === 'releasing') {
          backToWaiting(d, j.id, { wait: null })
          changed = true
        }
      }
      const waiting = Object.values(d.jobs).filter((j) => j.status === 'waiting')
      const holdAll = (why, standing) => {
        const h = d.lane.held
        if (!waiting.length || (standing.includes(h?.why) && waiting.every((j) => covers(h, j)))) return
        d.lane = { held: { why, scope: 'all', jobId: null, since: holdSince(now(), waiting) } }
        changed = true
      }
      if (bootNow && d.machineBoot && bootNow !== d.machineBoot) holdAll('restart', ['restart'])
      if (paused) holdAll('paused', ['restart', 'paused'])
      if (bootNow && bootNow !== d.machineBoot) {
        d.machineBoot = bootNow
        changed = true
      }
      return changed ? undefined : false
    })
  }

  function openSocket() {
    if (sock) return
    try {
      sock = comfy.socket(S().clientId, onSocket)
    } catch (err) {
      sock = null
      console.warn(`[switchgen-runner] could not open ComfyUI's socket: ${err?.message ?? err}; progress will not show`)
    }
  }

  async function closeSocket() {
    const s = sock
    sock = null
    if (!s) return
    try { await s.close() } catch { /* closed already */ }
  }

  function trip(err) {
    tripped = true
    reason = REASONS.tripped
    console.warn(`[switchgen-runner] the queue stopped after ${FAULTS_TO_TRIP} failed passes in a row; the last: ${err?.stack ?? err}`)
    notePaused()
    void closeSocket()
    emit('state', snapshot())
  }

  // ------------------------------------------------------------ the tick --

  function schedule() {
    if (!autoTick || retired || tripped) return
    if (timer) clearTimeout(timer)
    const unfinished = Object.values(S()?.jobs ?? {}).some((j) => !TERMINAL.has(j.status))
    timer = setTimeout(() => { timer = null; void tick() }, unfinished ? LIVE_TICK_MS : IDLE_TICK_MS)
    timer.unref?.()
  }

  /** Look again soon: something happened that may let work move. */
  function wake() {
    if (!autoTick || retired || tripped) return
    if (ticking) { again = true; return }
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => { timer = null; void tick() }, 0)
    timer.unref?.()
  }

  /**
   * One pass over the queue, never two at once. A call made while one runs
   * waits for it, and another pass follows, so the caller's pass starts
   * after its call.
   */
  function tick() {
    if (ticking) {
      again = true
      return ticking
    }
    ticking = (async () => {
      try {
        do {
          again = false
          await tickOnce()
        } while (again && !retired && !tripped)
      } finally {
        ticking = null
        schedule()
      }
    })()
    return ticking
  }

  async function tickOnce() {
    let active = ensureActive()
    if ((active || reason === REASONS.oldComfy) && (jobsList !== true || lookAgain)) {
      await lookForJobsList()
      active = ensureActive()
    }
    if (!active) return
    tickGen = generation
    try {
      await pass()
      faults = 0
    } catch (err) {
      if (err instanceof Retired) return
      if (err instanceof DiskError) return
      faults += 1
      console.warn(`[switchgen-runner] a pass over the queue failed: ${err?.stack ?? err}`)
      if (faults >= FAULTS_TO_TRIP) trip(err)
    }
  }

  /**
   * Ask ComfyUI whether it has the jobs list, before a pass that would lean
   * on it. With no answer, what was known stands: the queue does not stand
   * back for a ComfyUI that is only down, and sends nothing until one
   * answers and is seen to have the list.
   */
  async function lookForJobsList() {
    let has
    try {
      has = await comfy.hasJobsList()
    } catch {
      answering(false)
      return
    }
    lookAgain = false
    jobsList = has
    answering(true)
  }

  /** Run `step` for one job, so a write that fails holds that job back and not the pass. */
  async function forJob(id, step) {
    try {
      await step(id)
    } catch (err) {
      if (err instanceof DiskError) markDisk(id, err)
      else throw err
    }
  }

  async function pass() {
    for (const j of jobsWhere((x) => x.status === 'releasing')) {
      // Only a pass that stopped part way leaves one: send it round again.
      await forJob(j.id, (id) => returnToWaiting(id, 'releasing', { wait: null }))
    }
    for (const j of jobsWhere((x) => x.status === 'sending')) await forJob(j.id, resolveSending)
    for (const j of jobsWhere((x) => x.status === 'queued' || x.status === 'running')) await forJob(j.id, follow)
    for (const j of jobsWhere((x) => x.status === 'filing')) await forJob(j.id, (id) => fileJob(filingSide, id))
    await dispatch()
    try {
      settleWaits()
      prune()
    } catch (err) {
      if (!(err instanceof DiskError)) throw err
      if (!diskSaid) console.warn(`[switchgen-runner] could not write the list of work: ${err.message}; waiting for room`)
      diskSaid = true
    }
  }

  // ------------------------------------------------- asking about a send --

  /**
   * A job whose send had no clear answer. ComfyUI is asked about it by its id
   * each pass; it is never sent again. Found, it is queued. Absent twice while
   * ComfyUI answers, and with no record of it having run, it never got there:
   * it is 'unsent', and nothing sends it again by itself.
   */
  async function resolveSending(id) {
    const j = job(id)
    if (!j || j.status !== 'sending') return
    const pid = j.promptIdInternal
    const same = (cur) => cur.status === 'sending' && cur.promptIdInternal === pid
    if (!pid) {
      returnToWaiting(id, 'sending', { wait: null })
      return
    }
    let found
    try {
      found = await comfy.getJob(pid)
      answering(true)
    } catch {
      alive()
      answering(false)
      setWait(id, 'sending', { for: 'comfy' })
      return
    }
    alive()
    if (found) {
      patchJob(id, same, (cur) => ({ status: 'queued', promptId: pid, acceptedAt: cur.acceptedAt ?? now(), sighted: true, misses: 0, wait: null }))
      if (job(id)?.stopRequested && job(id).status === 'queued') await cancelNow(id, tcommit)
      return
    }
    const misses = (j.misses ?? 0) + 1
    if (misses < MISSES) {
      patchJob(id, same, { misses, wait: null })
      return
    }
    let raw
    try {
      raw = await comfy.history(pid)
    } catch {
      alive()
      answering(false)
      setWait(id, 'sending', { for: 'comfy' })
      return
    }
    alive()
    if (raw) {
      await ending(id, pid, raw, same)
      return
    }
    tcommit((d) => {
      const cur = d.jobs[id]
      if (!cur || !same(cur)) return false
      if (cur.stopRequested) endJob(d, id, 'stopped', fault('stopped', { sent: false, message: 'Stopped before it was sent.' }))
      else endJob(d, id, 'unsent', fault('unsent', { sent: false }))
    })
  }

  // ------------------------------------------------------- following one --

  /**
   * A job ComfyUI has in its queue. Its jobs API says pending, running or
   * ended; an ending is read from /history. A job it no longer lists at all,
   * twice in a row and past the grace a new prompt gets, with no history
   * either, was lost: ComfyUI restarted, most likely. The socket never
   * decides any of this; it only makes the next look come sooner.
   */
  async function follow(id) {
    let j = job(id)
    if (!j || (j.status !== 'queued' && j.status !== 'running')) return
    // A stop ComfyUI did not take (no answer, say) is asked again every
    // little while, until it lands or the job ends; asked once only, the
    // prompt would run on and be filed as done.
    if (j.stopRequested && !j.stopLanded && now() - (lastCancel.get(id) ?? -Infinity) >= CANCEL_EVERY_MS) {
      await cancelNow(id, tcommit)
      alive()
      j = job(id)
      if (!j || (j.status !== 'queued' && j.status !== 'running')) return
    }
    const pid = j.promptId
    const same = (cur) => (cur.status === 'queued' || cur.status === 'running') && cur.promptId === pid
    let found
    try {
      found = await comfy.getJob(pid)
      answering(true)
    } catch {
      alive()
      answering(false)
      return
    }
    alive()
    if (found && (found.status === 'pending' || found.status === 'in_progress')) {
      const running = found.status === 'in_progress'
      patchJob(id, same, (cur) => {
        const p = {}
        if (!cur.sighted) p.sighted = true
        if (cur.misses) p.misses = 0
        if (running && cur.status === 'queued') {
          p.status = 'running'
          p.ranAt = found.execution_start_time ?? starts.get(pid) ?? cur.ranAt ?? null
        } else if (cur.status === 'running' && cur.ranAt == null && starts.get(pid) != null) {
          p.ranAt = starts.get(pid)
        }
        return Object.keys(p).length ? p : false
      })
      if (!running && j.heavy) await freeForRunningAhead(id, pid)
      return
    }
    if (found) {
      let raw
      try {
        raw = await comfy.history(pid)
      } catch {
        alive()
        answering(false)
        return
      }
      alive()
      await ending(id, pid, raw, same)
      return
    }
    if (!j.sighted && now() - (j.acceptedAt ?? now()) < GRACE_MS) return
    const misses = (j.misses ?? 0) + 1
    if (misses < MISSES) {
      patchJob(id, same, { misses })
      return
    }
    let raw
    try {
      raw = await comfy.history(pid)
    } catch {
      alive()
      answering(false)
      return
    }
    alive()
    if (raw) {
      await ending(id, pid, raw, same)
      return
    }
    tcommit((d) => {
      const cur = d.jobs[id]
      if (!cur || !same(cur)) return false
      // A prompt taken out of the queue before it ran leaves no record, so a
      // queued job the reader asked to stop ends this way, as stopped, and
      // holds nothing. That is so even when no cancel was heard to land: one
      // whose answer was lost (a ComfyUI too slow to answer in time, which
      // then took it) leaves the prompt just as gone.
      if (cur.stopRequested && cur.status === 'queued') endJob(d, id, 'stopped', fault('stopped', { sent: true }))
      else endJob(d, id, 'lost', fault('lost', { sent: true }))
    })
  }

  /**
   * A job ComfyUI says has ended, read from its history record. A success
   * goes on to be filed, even when a stop was asked for, because the later
   * verdict stands: the file is made, and the archive should say so.
   */
  async function ending(id, pid, raw, same) {
    const run = raw ? readPastRun(pid, raw) : null
    if (!run || run.status === 'unknown') {
      tcommit((d) => {
        const cur = d.jobs[id]
        if (!cur || !same(cur)) return false
        const waits = (cur.historyWaits ?? 0) + 1
        if (waits >= HISTORY_WAITS) endJob(d, id, 'failed', fault('ended-unsent', { sent: true, mayExist: true }))
        else d.jobs[id] = { ...cur, historyWaits: waits }
      })
      return
    }
    const times = {
      ranAt: run.startedAt ?? job(id)?.ranAt ?? null,
      finishedAt: run.finishedAt ?? null,
      promptId: pid,
    }
    if (run.status === 'success') {
      const res = tcommit((d) => {
        const cur = d.jobs[id]
        if (!cur || !same(cur)) return false
        d.jobs[id] = { ...cur, promptId: pid, ...filingPatch(cur, run, now()) }
      })
      if (res) archive.claim(relsOf(job(id)))
      return
    }
    tcommit((d) => {
      const cur = d.jobs[id]
      if (!cur || !same(cur)) return false
      if (run.status === 'error') {
        endJob(d, id, 'failed', fault('failed', {
          sent: true,
          message: run.error?.message || 'ComfyUI reported an error. The job did not finish.',
          node: run.error?.node ?? null,
          nodeType: run.error?.nodeType ?? null,
        }), times)
      } else {
        endJob(d, id, 'stopped', fault('stopped', { sent: true }), times)
      }
    })
  }

  /**
   * While a heavy job of ours waits in ComfyUI's queue, a prompt that is not
   * ours running ahead of it will leave its models behind. ComfyUI applies a
   * release after the job it is running, so one sent now lands between that
   * prompt and ours. Once for each such prompt: a spare one costs only a
   * reload.
   */
  async function freeForRunningAhead(id, pid) {
    const t = now()
    if (t - (lastForeignRead.get(id) ?? -Infinity) < QUEUE_EVERY_MS) return
    lastForeignRead.set(id, t)
    let q
    try {
      q = await comfy.readQueue()
    } catch {
      return
    }
    alive()
    const freed = freedFor.get(id) ?? new Set()
    freedFor.set(id, freed)
    for (const other of q.running) {
      if (other === pid || freed.has(other)) continue
      freed.add(other)
      alive()
      await comfy.free()
    }
  }

  /**
   * Just after a heavy prompt is accepted: anything else ComfyUI lists may
   * have slipped in between the release and the prompt and read the release
   * first. A second release, sent now, is applied after that work and before
   * ours, since ComfyUI's worker reads the flags once it has finished a
   * prompt. It covers every prompt listed now, so none of them is released
   * for again when it is later seen running ahead.
   */
  async function freeIfOthersListed(id, pid) {
    let q
    try {
      q = await comfy.readQueue()
    } catch {
      return
    }
    alive()
    const others = [...q.running, ...q.pending].filter((p) => p !== pid)
    if (!others.length) return
    const freed = freedFor.get(id) ?? new Set()
    freedFor.set(id, freed)
    for (const p of others) freed.add(p)
    await comfy.free()
  }

  // ---------------------------------------------------------- dispatching --

  /** Every job before this one in a batch or a pass has ended. */
  function beforeDone(j, s) {
    const g = s.groups[j.groupId]
    if (!g || g.kind === 'clips') return true
    for (const id of g.jobIds) {
      if (id === j.id) return true
      const o = s.jobs[id]
      if (o && !TERMINAL.has(o.status)) return false
    }
    return true
  }

  function eligible(j, s) {
    if (!j || j.status !== 'waiting' || j.stopRequested) return false
    if (s.groups[j.groupId]?.state !== 'active') return false
    if (covers(s.lane.held, j)) return false
    return beforeDone(j, s)
  }

  /** The next job to go: the group served least recently, then the oldest job. */
  function choose(s) {
    const cands = Object.values(s.jobs).filter((j) => eligible(j, s))
    cands.sort((a, b) => (s.served[a.groupId] ?? 0) - (s.served[b.groupId] ?? 0) || a.seq - b.seq)
    return cands[0] ?? null
  }

  async function dispatch() {
    const s = S()
    if (Object.values(s.jobs).some((j) => LIVE.has(j.status))) return
    let pick = head ? s.jobs[head] : null
    if (!eligible(pick, s)) {
      head = null
      pick = choose(s)
    }
    if (!pick) return
    await forJob(pick.id, async (id) => {
      // Not to a ComfyUI that has not been seen to have the jobs list since it
      // last answered: one older than the list would lose the job.
      if (jobsList !== true || lookAgain) {
        setWait(id, 'waiting', { for: 'comfy' })
        return
      }
      if (job(id).chain && !job(id).openedOn && !(await openChain(id))) return
      const graph = prepareGraph(id)
      if (!graph) return
      if (job(id).heavy) {
        head = id
        await heavySequence(id, graph)
      } else {
        // A light job that found ComfyUI not answering tries again at the
        // pace a heavy one reads the queue, not on every pass.
        if (job(id).wait?.for === 'comfy' && now() < nextQueueReadAt) return
        await sendNow(id, 'waiting', graph)
      }
    })
  }

  /**
   * A reel shot that opens on the last frame of the shot before it, in this
   * pass: now that shot has landed, find its frame on disk and commit it as
   * the one this shot opens on, before anything is released or sent.
   */
  async function openChain(id) {
    const j = job(id)
    const up = job(j.chain.after)
    const frame = up?.status === 'done' ? up.frame : null
    let there = false
    if (frame && (!frame.type || frame.type === 'output')) {
      const full = await confineReal(outputs, relOf(frame))
      if (full) {
        try {
          there = (await fs.stat(full)).isFile()
        } catch {
          there = false
        }
      }
    }
    alive()
    const waiting = (cur) => cur.status === 'waiting' && !cur.openedOn
    if (!there) {
      tcommit((d) => {
        const cur = d.jobs[id]
        if (!cur || !waiting(cur)) return false
        endJob(d, id, 'failed', fault('no-frame', { message: 'The shot before it left no last frame on disk to open on.' }))
      })
      return false
    }
    const res = patchJob(id, waiting, { openedOn: annotatedRef(frame) })
    return !!res && job(id)?.openedOn != null
  }

  /**
   * The graph to send: the saved one, with the frame put in for a chained
   * shot. The saved graph is never rewritten. A graph that cannot be made
   * fails the job, since sending a placeholder would spend the card on the
   * wrong thing.
   */
  function prepareGraph(id) {
    const j = job(id)
    const made = toSend.get(id)
    if (made) return made
    const payload = store.readPayload(id)
    let graph = payload?.graph ?? null
    let why = graph ? null : 'The saved graph for this job is missing, so it cannot be sent.'
    if (graph && j.chain) {
      try {
        graph = splice(graph, j.chain.at, j.openedOn)
      } catch (err) {
        graph = null
        why = String(err?.message ?? err)
      }
    }
    if (!graph) {
      tcommit((d) => {
        const cur = d.jobs[id]
        if (!cur || cur.status !== 'waiting') return false
        endJob(d, id, 'failed', fault('internal', { message: why }))
      })
      return null
    }
    graphs.set(id, payload.graph)
    toSend.set(id, graph)
    return graph
  }

  /** Read how many jobs ComfyUI has, or null when it does not answer. */
  async function queueCount() {
    try {
      const q = await comfy.readQueue()
      answering(true)
      return q.running.length + q.pending.length
    } catch {
      answering(false)
      return null
    }
  }

  const waitFor = (count) => (count === null ? { for: 'comfy' } : { for: 'queue', ahead: count })

  /**
   * Empty queue, release, a pause, empty queue again, prompt. Anything that
   * gets in the way sends the job back to waiting, and the release is made
   * again next time.
   */
  async function heavySequence(id, graph) {
    if (now() < nextQueueReadAt) return
    const first = await queueCount()
    alive()
    if (first !== 0) {
      nextQueueReadAt = now() + QUEUE_EVERY_MS
      setWait(id, 'waiting', waitFor(first))
      return
    }
    const released = patchJob(id, (cur) => cur.status === 'waiting' && !cur.stopRequested && !covers(S().lane.held, cur), { status: 'releasing', wait: null })
    if (!released) return
    alive()
    const freed = await comfy.free()
    alive()
    if (freed === 'unreached') {
      answering(false)
      nextQueueReadAt = now() + QUEUE_EVERY_MS
      returnToWaiting(id, 'releasing', { wait: { for: 'comfy' } })
      return
    }
    await sleep(settleMs)
    alive()
    const second = await queueCount()
    alive()
    if (second !== 0) {
      nextQueueReadAt = now() + QUEUE_EVERY_MS
      returnToWaiting(id, 'releasing', { wait: waitFor(second) })
      return
    }
    await sendNow(id, 'releasing', graph)
  }

  /**
   * Send one job, from `from` ('waiting' for a light job, 'releasing' for a
   * heavy one). Its prompt id is made here and committed with 'sending'
   * before the prompt goes, so whatever happens next, the job can be asked
   * about by that id and is never sent twice.
   */
  async function sendNow(id, from, graph) {
    let promptId = null
    const clientId = S().clientId
    const res = tcommit((d) => {
      const cur = d.jobs[id]
      if (!cur || cur.status !== from) return false
      if (cur.stopRequested) {
        endJob(d, id, 'stopped', fault('stopped', { sent: false, message: 'Stopped before it was sent.' }))
        return 'stopped'
      }
      if (covers(d.lane.held, cur)) {
        d.jobs[id] = { ...cur, status: 'waiting', wait: { for: 'held' } }
        return 'held'
      }
      promptId = randomUUID()
      d.serveCounter = (d.serveCounter ?? 0) + 1
      d.served[cur.groupId] = d.serveCounter
      d.jobs[id] = {
        ...cur,
        status: 'sending',
        wait: null,
        promptIdInternal: promptId,
        attempt: (cur.attempt ?? 0) + 1,
        sentAt: now(),
        sighted: false,
        misses: 0,
        historyWaits: 0,
      }
      return 'sending'
    })
    if (res?.out !== 'sending') {
      if (head === id) head = null
      return
    }
    const heavy = job(id).heavy
    if (head === id) head = null
    // No check for a handover here: 'sending' is on disk with this id, so a
    // runner that takes over while the prompt is on its way (it waits for
    // this pass to end) finds the prompt by its id rather than calling it
    // unsent. Only what comes after the answer is refused to a stale pass.
    const answer = await comfy.submit(graph, promptId, clientId)
    const same = (cur) => cur.status === 'sending' && cur.promptIdInternal === promptId
    if (answer?.accepted || answer?.refused) toSend.delete(id)

    if (answer?.accepted) {
      answering(true)
      patchJob(id, same, { status: 'queued', promptId, acceptedAt: now(), sighted: false, misses: 0, wait: null })
      const cur = job(id)
      if (cur?.status !== 'queued') return
      if (cur.stopRequested) await cancelNow(id, tcommit)
      else if (heavy) await freeIfOthersListed(id, promptId)
      return
    }
    if (answer?.refused) {
      answering(true)
      tcommit((d) => {
        const cur = d.jobs[id]
        if (!cur || !same(cur)) return false
        endJob(d, id, 'failed', fault('refused', {
          message: answer.message ?? null,
          node: answer.node ?? null,
          nodeType: answer.nodeType ?? null,
          nodeErrors: answer.nodeErrors ?? null,
        }))
      })
      return
    }
    if (answer?.unreached) {
      // Nothing reached ComfyUI, so the job goes back to waiting, and a fresh
      // id is made when it goes again; a heavy one releases again first.
      answering(false)
      nextQueueReadAt = now() + QUEUE_EVERY_MS
      tcommit((d) => {
        const cur = d.jobs[id]
        if (!cur || !same(cur)) return false
        backToWaiting(d, id, { promptIdInternal: null, wait: { for: 'comfy' } })
      })
      if (heavy && job(id)?.status === 'waiting') head = id
      return
    }
    // No clear answer: the prompt may be in ComfyUI's queue. It stays
    // 'sending' and is asked about by its id from the next pass on.
    console.warn(`[switchgen-runner] no clear answer sending ${id} (${answer?.reason ?? 'unknown'}); asking ComfyUI about it`)
  }

  /** What each waiting job is waiting for, as pages show it. */
  function settleWaits() {
    const s = S()
    const held = s.lane.held
    const changes = []
    for (const j of Object.values(s.jobs)) {
      if (j.status !== 'waiting' || j.id === head) continue
      let w
      if (!beforeDone(j, s)) w = { for: 'before' }
      else if (covers(held, j)) w = { for: 'held' }
      else if (head && !j.heavy) w = { for: 'heavy' }
      else if (j.wait?.for === 'comfy' && comfyState.answering === false) w = j.wait
      else w = { for: 'turn' }
      if (!sameWait(j.wait, w)) changes.push([j.id, w])
    }
    if (!changes.length) return
    tcommit((d) => {
      let any = false
      for (const [id, w] of changes) {
        const cur = d.jobs[id]
        if (cur?.status !== 'waiting') continue
        d.jobs[id] = { ...cur, wait: w }
        any = true
      }
      return any ? undefined : false
    })
  }

  /**
   * Let ended jobs go 48 hours after they ended, or the oldest of them once
   * there are more than 500, with their saved graphs. A job of a group still
   * running is kept, since the jobs after it may need it.
   *
   * A reel pass that no page has taken in (its group not dismissed) is kept
   * apart from those: 30 days, and no more than 2000 such jobs. Only a page of
   * the browser that pressed it can fold its clips into that strip, and a
   * phone may not open the app again for days, while other devices make
   * hundreds of pictures; dropped, its shots would read as never made and be
   * rendered again. A page that has taken a pass in dismisses it, and it goes
   * as any other ended work does.
   */
  function prune() {
    const t = now()
    if (t - lastPrune < PRUNE_EVERY_MS) return
    lastPrune = t
    // A note that work waited with no queue running, left by a server that
    // looked in the moment before this one took the folder: this queue runs
    // it, so the note is stale, and would hold that work at the next start.
    store.clearPaused()
    const s = S()
    const untaken = (j) => {
      const g = s.groups[j.groupId]
      return g?.kind === 'pass' && g.dismissed !== true
    }
    const ended = Object.values(s.jobs)
      .filter((j) => TERMINAL.has(j.status) && s.groups[j.groupId]?.state !== 'active')
      .sort((a, b) => (a.endedAt ?? 0) - (b.endedAt ?? 0))
    const gone = new Set()
    // Of `list`, oldest first: every job past `keepMs`, then the oldest left
    // until no more than `max` of the `count` ended ones of its kind remain.
    const drop = (list, keepMs, max, count) => {
      const expired = list.filter((j) => t - (j.endedAt ?? t) > keepMs)
      for (const j of expired) gone.add(j.id)
      let left = count - expired.length
      for (const j of list) {
        if (left <= max) break
        if (gone.has(j.id)) continue
        gone.add(j.id)
        left -= 1
      }
    }
    const terminal = Object.values(s.jobs).filter((j) => TERMINAL.has(j.status))
    drop(ended.filter((j) => !untaken(j)), KEEP_ENDED_MS, KEEP_ENDED, terminal.filter((j) => !untaken(j)).length)
    drop(ended.filter(untaken), KEEP_PASS_MS, KEEP_PASS_JOBS, terminal.filter(untaken).length)
    if (!gone.size) return
    tcommit((d) => {
      for (const id of gone) delete d.jobs[id]
      for (const g of Object.values(d.groups)) {
        const left = g.jobIds.filter((id) => !gone.has(id))
        if (left.length === g.jobIds.length) continue
        if (left.length) d.groups[g.id] = { ...g, jobIds: left }
        else delete d.groups[g.id]
      }
    })
    for (const id of gone) store.removePayload(id)
  }

  // --------------------------------------------------------- stopping --

  /**
   * Ask ComfyUI to stop a job it has. A cancel takes a waiting prompt out of
   * the queue or interrupts the running one; a ComfyUI too old for it is
   * interrupted only when its jobs API says this very prompt is running,
   * because an interrupt stops whatever is sampling.
   */
  async function cancelNow(id, via = commit) {
    const j = job(id)
    const pid = j?.promptId
    if (!pid) return
    lastCancel.set(id, now())
    const landed = await comfy.cancel(pid)
    if (landed) {
      patchJob(id, (cur) => cur.promptId === pid && !cur.stopLanded && !TERMINAL.has(cur.status), { stopLanded: true }, via)
      return
    }
    try {
      const found = await comfy.getJob(pid)
      if (found?.status === 'in_progress') await comfy.interrupt(pid)
    } catch {
      /* could not ask; the job is followed as before, and its stop stays asked */
    }
  }

  function stoppedBeforeSent() {
    return fault('stopped', { sent: false, message: 'Stopped before it was sent.' })
  }

  /**
   * For a stop from a page: ask ComfyUI to stop these jobs, all at once, and
   * wait for its answers no longer than stopWaitMs. A ComfyUI that has
   * stalled can take 20 s over a cancel and the read after it; the stop is
   * on disk already, the passes that follow each job ask again until it
   * lands, and the page must hear back well before it gives up waiting.
   */
  async function cancelSoon(ids) {
    if (!ids.length) return
    const all = Promise.all(ids.map((id) => cancelNow(id).catch(() => {})))
    let timer = null
    const cap = new Promise((resolve) => {
      timer = setTimeout(resolve, stopWaitMs)
      timer.unref?.()
    })
    try {
      await Promise.race([all, cap])
    } finally {
      clearTimeout(timer)
    }
  }

  /** Whether a job is ComfyUI's, with a stop asked for that has not been heard to land. */
  const stopToAsk = (j) => !!j && (j.status === 'queued' || j.status === 'running') && j.stopRequested && !j.stopLanded

  /** Stop one job, from any page on any device. */
  async function stopJob(id) {
    const j = job(id)
    if (!j) return null
    if (j.status === 'waiting') {
      commit((d) => {
        if (d.jobs[id]?.status !== 'waiting') return false
        endJob(d, id, 'stopped', stoppedBeforeSent())
      })
      if (head === id) head = null
    } else if (LIVE.has(j.status) && !j.stopRequested) {
      patchJob(id, (cur) => LIVE.has(cur.status) && !cur.stopRequested, { stopRequested: true }, commit)
    }
    if (stopToAsk(job(id))) await cancelSoon([id])
    wake()
    return job(id) ? view(job(id)) : null
  }

  /** Stop every job of a group that has not ended, and end the group. */
  async function stopGroup(id) {
    const g = group(id)
    if (!g) return null
    commit((d) => {
      const cur = d.groups[id]
      if (!cur) return false
      let first = null
      let any = false
      for (const jid of cur.jobIds) {
        const j = d.jobs[jid]
        if (!j) continue
        if (j.status === 'waiting') {
          endJob(d, jid, 'stopped', stoppedBeforeSent())
          first ??= jid
          any = true
        } else if (LIVE.has(j.status)) {
          if (!j.stopRequested) {
            d.jobs[jid] = { ...j, stopRequested: true }
            any = true
          }
          first ??= jid
        }
      }
      if (cur.state === 'active') {
        d.groups[id] = { ...cur, state: 'ended', endedBy: { jobId: first ?? cur.jobIds[0] ?? null, why: 'stopped' }, endedAt: now() }
        any = true
      }
      return any ? undefined : false
    })
    const ask = []
    for (const jid of group(id)?.jobIds ?? []) {
      const j = job(jid)
      if (j && head === jid && j.status !== 'waiting') head = null
      if (stopToAsk(j)) ask.push(jid)
    }
    await cancelSoon(ask)
    wake()
    const after = group(id)
    return after ? { group: after, jobs: after.jobIds.map((jid) => job(jid)).filter(Boolean).map(view) } : null
  }

  /**
   * The reader's word on a held lane. 'send' lets the held work go; 'stop'
   * stops every waiting job the hold covers, none of them ever sent. The word
   * names the hold it answers by when that hold began (`since`, as the page
   * showed it): a word given to a notice that is out of date, on a page that
   * slept, must not send or call off work held since for another reason,
   * which the reader has not seen. Null when nothing is held, false when the
   * hold is not the one named, and nothing changes then.
   */
  function laneWord(action, since = null) {
    const answers = (h) => since === null || h.since === since
    const held = S().lane.held
    if (!held) return null
    if (!answers(held)) return false
    const stopped = []
    let other = false
    const res = commit((d) => {
      const h = d.lane.held
      if (!h) return false
      if (!answers(h)) {
        other = true
        return false
      }
      if (action === 'stop') {
        for (const j of Object.values(d.jobs)) {
          if (j.status !== 'waiting' || !covers(h, j)) continue
          endJob(d, j.id, 'stopped', stoppedBeforeSent())
          stopped.push(j.id)
        }
      }
      d.lane = { held: null }
    })
    if (!res) return other ? false : null
    if (stopped.includes(head)) head = null
    wake()
    return { lane: S().lane, stopped }
  }

  /** Mark ended jobs as seen, so desks and the section bar let them go. */
  function dismiss(ids) {
    const wanted = new Set(ids.filter((x) => typeof x === 'string'))
    const dismissed = []
    commit((d) => {
      let any = false
      for (const id of wanted) {
        const j = Object.hasOwn(d.jobs, id) ? d.jobs[id] : null
        if (!j || !TERMINAL.has(j.status)) continue
        dismissed.push(id)
        if (j.dismissed) continue
        d.jobs[id] = { ...j, dismissed: true }
        any = true
      }
      for (const g of Object.values(d.groups)) {
        if (g.dismissed || g.state !== 'ended') continue
        if (g.jobIds.every((jid) => !d.jobs[jid] || d.jobs[jid].dismissed)) {
          d.groups[g.id] = { ...g, dismissed: true }
          any = true
        }
      }
      return any ? undefined : false
    })
    return dismissed
  }

  // ------------------------------------------------------------ socket --

  /** The job of ours a prompt id belongs to, while it is ComfyUI's. */
  function jobOfPrompt(pid) {
    if (!pid) return null
    for (const j of Object.values(S()?.jobs ?? {})) {
      if ((j.status === 'sending' || j.status === 'queued' || j.status === 'running') && j.promptIdInternal === pid) return j
    }
    return null
  }

  function graphOf(id) {
    let g = graphs.get(id)
    if (g === undefined) {
      g = store.readPayload(id)?.graph ?? null
      graphs.set(id, g)
    }
    return g
  }

  function sendProgress(id) {
    lastProgressEmit.set(id, Date.now())
    const p = progress.get(id)
    if (p) emit('progress', p)
  }

  /** At most four progress events a second for each job, the last always sent. */
  function progressChanged(id) {
    const last = lastProgressEmit.get(id) ?? 0
    const due = last + PROGRESS_EVERY_MS - Date.now()
    if (due <= 0) {
      sendProgress(id)
      return
    }
    if (progressTimers.has(id)) return
    const t = setTimeout(() => {
      progressTimers.delete(id)
      sendProgress(id)
    }, due)
    t.unref?.()
    progressTimers.set(id, t)
  }

  function noteProgress(pid, p) {
    const j = jobOfPrompt(pid)
    if (!j) return
    const graph = graphOf(j.id)
    const node = p.node
    const classType = node && graph && typeof graph[node]?.class_type === 'string' ? graph[node].class_type : null
    const prev = progress.get(j.id)
    progress.set(j.id, {
      id: j.id,
      value: p.value,
      max: p.max,
      node,
      classType,
      pass: (graph && node ? samplerPass(graph, node) : null) ?? prev?.pass ?? null,
      at: now(),
      previewN: prev?.previewN ?? 0,
    })
    progressChanged(j.id)
  }

  /**
   * What ComfyUI says on the socket: progress, previews and a hint to look
   * again. Never how a job ended: a socket that dropped would then decide it
   * wrongly, and a dead socket costs progress only.
   */
  function onSocket(msg) {
    if (retired || !loaded) return
    if (msg?.type === 'preview') {
      const j = jobOfPrompt(msg.promptId ?? executing)
      if (!j) return
      const n = (previews.get(j.id)?.n ?? 0) + 1
      previews.set(j.id, { mime: msg.mime || 'image/jpeg', bytes: msg.bytes, n })
      const p = progress.get(j.id)
      progress.set(j.id, p ? { ...p, previewN: n } : { id: j.id, value: 0, max: 1, node: null, classType: null, pass: null, at: now(), previewN: n })
      emit('preview', { id: j.id, n })
      return
    }
    const d = msg?.data ?? {}
    const pid = typeof d.prompt_id === 'string' ? d.prompt_id : null
    switch (msg?.type) {
      case 'status':
        // ComfyUI greets every new connection with a status carrying its
        // session id. A reconnect can mean a different ComfyUI came up (an
        // older build without the jobs list) while the queue was idle and no
        // read failed, so the list is checked again before the next send.
        if (typeof d.sid === 'string' && canAsk) lookAgain = true
        wake()
        return
      case 'execution_start':
        if (!pid) return
        executing = pid
        if (typeof d.timestamp === 'number') starts.set(pid, d.timestamp)
        noteProgress(pid, { node: null, value: 0, max: 1 })
        wake()
        return
      case 'executing':
        if (!pid) return
        if (d.node == null) {
          if (executing === pid) executing = null
          wake()
          return
        }
        executing = pid
        noteProgress(pid, { node: String(d.node), value: 0, max: 1 })
        return
      case 'progress':
        if (!pid) return
        executing = pid
        noteProgress(pid, {
          node: d.node == null ? null : String(d.node),
          value: Number(d.value ?? 0),
          max: Number(d.max ?? 1) || 1,
        })
        return
      case 'execution_success':
      case 'execution_error':
      case 'execution_interrupted':
        if (pid && executing === pid) executing = null
        wake()
        return
      default:
        return
    }
  }

  // ------------------------------------------------------------ reading --

  function status() {
    const active = ensureActive()
    return { active, desks: active ? [...desks] : [], reason: active ? null : reason }
  }

  /**
   * The list as pages may read it: the one in hand while the queue runs, or,
   * while it is off or stands back, the one on disk, read and never written,
   * so pages show that work as waiting on a queue that is off, not as gone.
   */
  const readableState = () => S() ?? (retired ? null : store.peek())
  const running = () => !retired && enabled && !tripped && loaded

  /**
   * A job as a queue that is not running shows it. Nothing on the list can
   * be sent, stopped or let go until the queue runs again, so a waiting job
   * waits for nothing in particular, and nothing is shown as held.
   */
  function readOnlyView(j) {
    const v = view(j)
    if (v.status === 'waiting') v.wait = null
    return v
  }

  function snapshot() {
    const active = running()
    const s = readableState()
    return {
      v: 1,
      available: active,
      reason: active ? null : reason ?? REASONS.stopped,
      boot,
      rev: s?.rev ?? 0,
      comfy: { ...comfyState },
      // A hold is lifted only by a word to a queue that runs; shown while it
      // cannot be given, it would offer controls that cannot work.
      lane: active && s ? s.lane : { held: null },
      groups: s ? Object.values(s.groups).sort((a, b) => a.createdAt - b.createdAt) : [],
      jobs: s ? Object.values(s.jobs).sort((a, b) => a.seq - b.seq).map(active ? view : readOnlyView) : [],
      progress: active ? Object.fromEntries(progress) : {},
    }
  }

  function jobDetail(id) {
    const s = readableState()
    const j = s && typeof id === 'string' && Object.hasOwn(s.jobs, id) ? s.jobs[id] : null
    if (!j) return null
    const payload = store.readPayload(id)
    return { job: running() ? view(j) : readOnlyView(j), graph: payload?.graph ?? null, record: payload?.record ?? null }
  }

  // --------------------------------------------------------- start, stop --

  function start() {
    ensureActive()
    if (autoTick) {
      timer = setTimeout(() => { timer = null; void tick() }, 0)
      timer.unref?.()
    }
  }

  /**
   * Hand over: no more passes, no more commits, the socket closed. The pass
   * under way, if any, is waited for; it stops at its next step, as a
   * process killed there would, and the next runner takes the work up from
   * the list on disk.
   *
   * With `handover`, the next runner is one of this process (Vite's config
   * reloaded), and the folder's lock is left for it to take over in place:
   * let go, the folder would look free to another server until the next
   * runner had started, and one standing back on the same folder would note
   * the waiting work as waiting on a queue that is off, to be held for no
   * reason at the next start.
   */
  async function retire({ handover = false } = {}) {
    if (retired) return
    retired = true
    // Nothing of this queue commits once it has retired, so the folder is
    // free for the next one now, not only once the pass under way has ended.
    store.unlock(boot, { handover: handover === true })
    if (timer) clearTimeout(timer)
    timer = null
    for (const t of progressTimers.values()) clearTimeout(t)
    progressTimers.clear()
    try { await ticking } catch { /* ended by the handover */ }
    await closeSocket()
    loaded = false
    store.close()
  }

  /** What filing.mjs uses of this engine. */
  const filingSide = {
    job,
    commit: tcommit,
    endJob,
    fault,
    alive,
    archive,
    readPayload: (id) => store.readPayload(id),
    waitForDisk: (id, err) => {
      try {
        waitForDisk(id, err)
      } catch (e) {
        if (e instanceof DiskError) markDisk(id, e)
        else throw e
      }
    },
  }

  return {
    boot,
    desks,
    now,
    store,
    start,
    tick,
    wake,
    retire,
    status,
    snapshot,
    subscribe,
    ensureActive,
    state: S,
    job,
    group,
    view,
    commit,
    endJob,
    stopJob,
    stopGroup,
    laneWord,
    dismiss,
    jobDetail,
    preview: (id) => previews.get(id) ?? null,
    closeSocket,
  }
}
