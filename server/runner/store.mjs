/**
 * The queue's list of work, on disk.
 *
 * Two kinds of file live in the runner's folder. `state.json` is small: every
 * group and job with where each stands, the lane and a few counters, and never
 * a graph. `jobs/<id>.json` holds one job's graph and record template, written
 * once when the job is taken in and never rewritten, so a long reel of heavy
 * graphs costs its bytes once rather than on every change of any job.
 *
 * Every change is a commit, and a commit is write-ahead: the new state is
 * built as a copy, written beside the file, flushed to the disk, renamed over
 * the old one, and only then taken as the state in memory. The engine commits
 * each change of status BEFORE the thing it allows (a release, a send, a
 * cancel, a filing), so a process killed at any point comes back to a state
 * that never claims more than happened. Because the writes are synchronous,
 * commits cannot interleave, and the process needs nothing written at exit.
 *
 * Committed states are frozen, down to each job. A mutator replaces a job, a
 * group or the lane with a new object and never edits one in place, so the
 * state in memory cannot change unless its write succeeded, and what changed
 * in a commit is exactly the objects that are not the ones before it.
 *
 * The folder belongs to one queue at a time. `lock` names the process (and
 * the runner in it) that runs the queue from here; a second server pointed at
 * the same folder finds it held and stands back, so the same work is never
 * sent twice. When Vite reloads its config, the runner of the load before
 * hands the lock to the next in place, so the folder never looks free to
 * another server in between. `paused.json` says a queue that was off, or
 * standing back, saw work waiting here while no queue ran it, so the next
 * queue to run holds that work until the reader says. A queue that is not
 * running reads `state.json` and nothing else, and writes only that note.
 */
import { createHash, randomUUID } from 'node:crypto'
import { closeSync, existsSync, fsyncSync, linkSync, mkdirSync, openSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from 'node:fs'
import path from 'node:path'

/** A job in one of these has ended, and nothing more happens to it. */
export const TERMINAL = new Set(['done', 'failed', 'stopped', 'lost', 'unsent', 'skipped'])

/** A write of the list of work failed: the change it carried did not happen. */
export class StoreWriteError extends Error {
  constructor(err) {
    super(String(err?.message ?? err), { cause: err })
    this.name = 'StoreWriteError'
    this.code = err?.code ?? null
  }
}

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v)

/** A new, empty list of work. */
export function freshState() {
  return {
    v: 1,
    machineBoot: null,
    // The id the runner's socket and every prompt it sends carry, kept across
    // restarts so a prompt sent before one is still recognised after it.
    clientId: randomUUID(),
    rev: 0,
    seq: 0,
    serveCounter: 0,
    served: {},
    lane: { held: null },
    groups: {},
    jobs: {},
  }
}

/** The state as read from disk, or null when it is not one this build wrote. */
function readable(doc) {
  if (!isObject(doc) || doc.v !== 1) return null
  if (typeof doc.clientId !== 'string' || !doc.clientId) return null
  if (!isObject(doc.jobs) || !isObject(doc.groups) || !isObject(doc.lane)) return null
  for (const j of Object.values(doc.jobs)) if (!isObject(j) || typeof j.id !== 'string' || typeof j.status !== 'string') return null
  for (const g of Object.values(doc.groups)) if (!isObject(g) || typeof g.id !== 'string' || !Array.isArray(g.jobIds)) return null
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0)
  return {
    v: 1,
    machineBoot: typeof doc.machineBoot === 'string' ? doc.machineBoot : null,
    clientId: doc.clientId,
    rev: num(doc.rev),
    seq: num(doc.seq),
    serveCounter: num(doc.serveCounter),
    served: isObject(doc.served) ? doc.served : {},
    lane: { held: isObject(doc.lane.held) ? doc.lane.held : null },
    groups: doc.groups,
    jobs: doc.jobs,
  }
}

function deepFreeze(v) {
  if (v === null || typeof v !== 'object' || Object.isFrozen(v)) return v
  Object.freeze(v)
  for (const k of Object.keys(v)) deepFreeze(v[k])
  return v
}

/**
 * Flush a folder's entry for a file just renamed into it. Best effort: some
 * file systems refuse to open a folder for this, and the rename itself is
 * already atomic.
 */
function syncDir(dir) {
  let fd = null
  try {
    fd = openSync(dir, 'r')
    fsyncSync(fd)
  } catch {
    /* not offered here; the rename stands */
  } finally {
    if (fd !== null) try { closeSync(fd) } catch { /* closed */ }
  }
}

/** Write `text` to `file` whole or not at all: beside it, flushed, renamed over. */
function writeDurably(file, text) {
  const tmp = `${file}.tmp`
  let fd = null
  try {
    fd = openSync(tmp, 'w')
    writeFileSync(fd, text, 'utf8')
    fsyncSync(fd)
    closeSync(fd)
    fd = null
    renameSync(tmp, file)
  } catch (err) {
    if (fd !== null) try { closeSync(fd) } catch { /* closed */ }
    try { unlinkSync(tmp) } catch { /* never made */ }
    throw new StoreWriteError(err)
  }
  syncDir(path.dirname(file))
}

const sha256 = (text) => createHash('sha256').update(text).digest('hex')

// ------------------------------------------------------------------- lock --

/** This boot of the machine. A lock written before a reboot names a process that is gone, whatever runs under its number now. */
const MACHINE_BOOT = (() => {
  try { return readFileSync('/proc/sys/kernel/random/boot_id', 'utf8').trim() || null } catch { return null }
})()

/**
 * When a process started, in clock ticks since boot, where Linux says; null
 * elsewhere. With the pid it names one process: a pid is handed out again
 * once its process has gone, the start time is not.
 */
function startOf(pid) {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8')
    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19] ?? null
  } catch {
    return null
  }
}

const SELF_START = startOf(process.pid)

/**
 * The runners of this process that hold a folder, each with its lock file.
 * Kept on the process rather than in this module, because Vite loads the
 * module afresh when its config reloads, and a runner of the load before must
 * still count as holding its folder until it lets it go. Every lock is let go
 * as the process exits.
 */
const HELD = Symbol.for('switchgen.runner.folderLocks')
/**
 * Lock files a runner of this process has handed over, each with the runner
 * it still names, for the next runner of this process to take over in place.
 * Until one does, or the handover is ended, the folder counts as held, here
 * and to every other server, as it did while that runner ran.
 */
const HANDING = Symbol.for('switchgen.runner.folderHandovers')

function heldHere() {
  let held = globalThis[HELD]
  if (!(held instanceof Map)) {
    held = globalThis[HELD] = new Map()
    const all = held
    process.once('exit', () => {
      for (const [owner, file] of all) dropLockFile(file, owner)
      all.clear()
    })
  }
  return held
}

function handingHere() {
  let handing = globalThis[HANDING]
  if (!(handing instanceof Map)) {
    handing = globalThis[HANDING] = new Map()
    const all = handing
    process.once('exit', () => {
      for (const [file, owner] of all) dropLockFile(file, owner)
      all.clear()
    })
  }
  return handing
}

function readLockFile(file) {
  try {
    const doc = JSON.parse(readFileSync(file, 'utf8'))
    return isObject(doc) ? doc : null
  } catch {
    return null
  }
}

/** Remove a lock file only while it still names this process and this runner. */
function dropLockFile(file, owner) {
  const holder = readLockFile(file)
  if (holder?.pid !== process.pid || holder?.owner !== owner) return
  try { unlinkSync(file) } catch { /* gone already */ }
}

/**
 * Whether the runner a lock names still runs. One in this process runs until
 * it lets the folder go, or, having handed it over, until the next takes it
 * or the handover ends; one in another process runs while that process does,
 * told apart from a later process given the same number by the machine's boot
 * and the process's start.
 */
function holderRuns(holder, file) {
  if (!holder || !Number.isInteger(holder.pid) || holder.pid <= 0) return false
  if (holder.pid === process.pid) {
    return typeof holder.owner === 'string' && (heldHere().has(holder.owner) || handingHere().get(file) === holder.owner)
  }
  try {
    process.kill(holder.pid, 0)
  } catch (err) {
    // EPERM: it runs, as another user.
    if (err?.code !== 'EPERM') return false
  }
  if (holder.boot && MACHINE_BOOT && holder.boot !== MACHINE_BOOT) return false
  if (holder.start) {
    const now = startOf(holder.pid)
    if (now !== null && now !== holder.start) return false
  }
  return true
}

/**
 * Let go of every lock a runner of this process handed over that no runner
 * has taken up: the runner meant to take it did not (turned off, standing
 * back, or failed to start), and until now the folder counted as held.
 */
export function endHandovers() {
  const handing = handingHere()
  for (const [file, owner] of handing) dropLockFile(file, owner)
  handing.clear()
}

/** The hash a job's saved graph and record are checked against when read back. */
export function payloadSha(graph, record) {
  return sha256(JSON.stringify({ graph, record }))
}

/**
 * The runner's folder. Nothing is written until the folder is locked and
 * opened, so a runner that is turned off, or stands back for another server,
 * only reads its list (and may leave the note that work waited here).
 */
export function createStore({ dir, now = Date.now }) {
  const file = path.join(dir, 'state.json')
  const jobsDir = path.join(dir, 'jobs')
  const lockFile = path.join(dir, 'lock')
  const pausedFile = path.join(dir, 'paused.json')
  let state = null
  /** The last read of the list made while it was not open, and the file it was read from. */
  let peeked = { key: null, state: null }

  const payloadFile = (id) => path.join(jobsDir, `${id}.json`)

  /** The runner that holds this folder now, or null when none does (a lock left by one that has gone counts as none). */
  function holder() {
    const h = readLockFile(lockFile)
    return h && holderRuns(h, lockFile) ? h : null
  }

  /**
   * Take the folder for the runner `owner`. Returns null once it holds it,
   * or the holder ({pid}) when another runner that still runs does, in this
   * process or another. A lock handed over by the runner of this process
   * before, or left by a runner that has gone, is taken over. The lock is
   * written whole beside it and linked into place, which fails when one is
   * there, so no runner ever reads another's half written.
   * Throws when the folder cannot be made or written.
   */
  function lock(owner) {
    mkdirSync(dir, { recursive: true })
    const mine = { pid: process.pid, boot: MACHINE_BOOT, start: SELF_START, owner }
    const tmp = `${lockFile}.${process.pid}.${owner}.tmp`
    writeFileSync(tmp, JSON.stringify(mine))
    const took = () => {
      handingHere().delete(lockFile)
      heldHere().set(owner, lockFile)
      return null
    }
    try {
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          linkSync(tmp, lockFile)
          return took()
        } catch (err) {
          if (err?.code !== 'EEXIST') throw err
        }
        const h = readLockFile(lockFile)
        if (h?.pid === process.pid && h.owner === owner) return took()
        if (h?.pid === process.pid && typeof h.owner === 'string' && handingHere().get(lockFile) === h.owner) {
          // Handed over by the runner of this process before this one: taken
          // over in place, so the folder is never free between the two.
          renameSync(tmp, lockFile)
          return took()
        }
        if (h && holderRuns(h, lockFile)) return { pid: h.pid }
        if (h?.pid === process.pid) {
          // Left by a runner of this process that has let go of it without
          // removing it: taken over in place.
          renameSync(tmp, lockFile)
          return took()
        }
        try { unlinkSync(lockFile) } catch { /* cleared by someone else first */ }
      }
      return { pid: readLockFile(lockFile)?.pid ?? 0 }
    } finally {
      try { unlinkSync(tmp) } catch { /* renamed into place */ }
    }
  }

  /**
   * Whether `owner` still holds the folder, asked while it runs. A lock
   * deleted by hand, with nobody in its place, is taken again; one that names
   * another runner means that runner took the folder, and this one must stop.
   */
  function stillLocked(owner) {
    const h = readLockFile(lockFile)
    if (h?.pid === process.pid && h.owner === owner) return true
    if (h === null) {
      try { return lock(owner) === null } catch { return false }
    }
    return false
  }

  /**
   * Let the folder go, when `owner` holds it. With `handover`, the lock file
   * stays, still naming `owner`, for the next runner of this process to take
   * over in place (see endHandovers for one that never comes): removed, it
   * would leave the folder looking free to another server for as long as the
   * next runner takes to start, and one standing back would note the work
   * here as waiting on a queue that is off.
   */
  function unlock(owner, { handover = false } = {}) {
    const held = heldHere()
    const had = held.get(owner) === lockFile
    if (had) held.delete(owner)
    if (handover && had) {
      const h = readLockFile(lockFile)
      if (h?.pid === process.pid && h.owner === owner) {
        handingHere().set(lockFile, owner)
        return
      }
    }
    dropLockFile(lockFile, owner)
  }

  /**
   * The list as it stands on disk, read and never written, for a runner that
   * is not running it: turned off, or standing back for another server. Null
   * when there is none or it cannot be read. Read again only when the file
   * has changed.
   */
  function peek() {
    let st
    try {
      st = statSync(file)
    } catch {
      peeked = { key: null, state: null }
      return null
    }
    const key = `${st.ino}:${st.size}:${st.mtimeMs}`
    if (peeked.key === key) return peeked.state
    let loaded = null
    try { loaded = readable(JSON.parse(readFileSync(file, 'utf8'))) } catch { loaded = null }
    peeked = { key, state: loaded ? deepFreeze(loaded) : null }
    return peeked.state
  }

  /** The note that work waited here while no queue ran it, or null. */
  function pausedNote() {
    try {
      const doc = JSON.parse(readFileSync(pausedFile, 'utf8'))
      return isObject(doc) ? doc : null
    } catch {
      return null
    }
  }

  /** Leave that note, best effort: without it the next queue sends the waiting work without asking. */
  function notePaused(note) {
    if (existsSync(pausedFile)) return true
    try {
      writeDurably(pausedFile, JSON.stringify(note))
      return true
    } catch {
      return false
    }
  }

  function clearPaused() {
    try { unlinkSync(pausedFile) } catch { /* none */ }
  }

  /**
   * Make the folder, prove it can be written, and read the list of work.
   * Throws when the folder cannot be made or written; the runner then stays
   * off and says why.
   *
   * A list this build cannot read is set aside beside it, never overwritten,
   * and the runner starts with an empty one.
   */
  function open() {
    mkdirSync(jobsDir, { recursive: true })
    const probe = path.join(dir, `.probe-${process.pid}`)
    writeFileSync(probe, '')
    unlinkSync(probe)
    let raw = null
    try {
      raw = readFileSync(file, 'utf8')
    } catch (err) {
      if (err?.code !== 'ENOENT') throw err
    }
    let loaded = null
    if (raw !== null) {
      try { loaded = readable(JSON.parse(raw)) } catch { loaded = null }
      if (!loaded) {
        const aside = path.join(dir, `state.broken-${now()}.json`)
        try { renameSync(file, aside) } catch { /* gone already */ }
        console.warn(`[switchgen-runner] could not read ${file}; kept as ${aside}, and the queue starts empty`)
      }
    }
    state = deepFreeze(loaded ?? freshState())
    return state
  }

  function close() {
    state = null
  }

  /**
   * Apply `mutate` to a copy of the state and make it the state, on disk
   * first. `mutate` gets a draft whose `jobs`, `groups` and `served` are
   * copies of the maps, holding the same frozen objects: it replaces the ones
   * it changes. Returning false leaves everything as it was, with no write.
   *
   * Returns null for no change, or what changed: the new rev, the ids of the
   * jobs and groups replaced, those removed, whether the lane changed, and
   * what `mutate` returned. Throws StoreWriteError when the write failed, and
   * then nothing changed, in memory or on disk.
   */
  function commit(mutate) {
    if (!state) throw new Error('the list of work is not open')
    const before = state
    const draft = {
      ...before,
      served: { ...before.served },
      groups: { ...before.groups },
      jobs: { ...before.jobs },
    }
    const out = mutate(draft)
    if (out === false) return null
    draft.rev = before.rev + 1
    writeDurably(file, JSON.stringify(draft))
    const jobs = []
    const groups = []
    const goneJobs = []
    const goneGroups = []
    for (const id of Object.keys(draft.jobs)) if (draft.jobs[id] !== before.jobs[id]) jobs.push(id)
    for (const id of Object.keys(before.jobs)) if (!(id in draft.jobs)) goneJobs.push(id)
    for (const id of Object.keys(draft.groups)) if (draft.groups[id] !== before.groups[id]) groups.push(id)
    for (const id of Object.keys(before.groups)) if (!(id in draft.groups)) goneGroups.push(id)
    state = deepFreeze(draft)
    return { rev: draft.rev, jobs, groups, goneJobs, goneGroups, lane: draft.lane !== before.lane, out }
  }

  /**
   * Save one job's graph and record template, durably, before the job is
   * listed. Returns the hash they are checked against when read back.
   */
  function writePayload(id, graph, record) {
    const sha = payloadSha(graph, record)
    writeDurably(payloadFile(id), JSON.stringify({ v: 1, id, sha256: sha, graph, record }))
    return sha
  }

  /**
   * One job's saved graph and record template, or null when the file is
   * missing, unreadable, or not the one that was saved for this job.
   */
  function readPayload(id) {
    let doc
    try { doc = JSON.parse(readFileSync(payloadFile(id), 'utf8')) } catch { return null }
    if (!isObject(doc) || doc.v !== 1 || doc.id !== id || !isObject(doc.graph) || !isObject(doc.record)) return null
    if (doc.sha256 !== payloadSha(doc.graph, doc.record)) return null
    return { graph: doc.graph, record: doc.record, sha256: doc.sha256 }
  }

  function hasPayload(id) {
    return existsSync(payloadFile(id))
  }

  function removePayload(id) {
    try { unlinkSync(payloadFile(id)) } catch { /* gone already */ }
  }

  return {
    dir,
    file,
    lockFile,
    open,
    close,
    commit,
    current: () => state,
    writePayload,
    readPayload,
    hasPayload,
    removePayload,
    lock,
    stillLocked,
    unlock,
    holder,
    peek,
    pausedNote,
    notePaused,
    clearPaused,
  }
}
