/**
 * SwitchGen archive server: one archive for every device.
 *
 * The archive used to live in each browser's localStorage and nowhere else,
 * so a picture made from the phone over Tailscale was invisible on the
 * desktop, and clearing site data lost every record while the files sat
 * untouched. The records now live in one file beside the files they describe,
 * `<outputs>/.switchgen/archive.json`, and every browser keeps localStorage as
 * a cache of it.
 *
 * The protocol is a revision log. Every write bumps `rev` and stamps the
 * record; a client asks for everything after the rev it last saw. Removal is
 * a tombstone, so a device that was offline when a record went learns that it
 * went rather than pushing its stale copy back, and a copy stamped before the
 * removal that arrives anyway is refused. Last write wins per record, which is
 * the right rule for a store one person edits from two rooms.
 *
 * The log carries an `epoch`, made once when the archive is. A client keeps
 * the rev it has read up to, and a rev only means something within one log:
 * when the epoch it remembers is not this one, the archive was started again
 * and the client reads the whole of it rather than the tail, and sends back
 * what it holds that the new log does not.
 *
 * Within one log a restart can still lose the last writes: a change is
 * answered before the debounced save puts it on disk, and a process killed in
 * between comes back behind what it told its clients. So every answer also
 * carries `boot`, made each time this file is loaded, and `base`, the rev this
 * process read from disk. A client that holds revs above `base` from an
 * earlier boot knows the log above `base` was lost and rebuilt, and resends
 * its own copies of what went missing.
 *
 * A removed record's files are remembered as `dismissed`, with the time. The
 * files stay on disk, and without this the recovery pass would find them
 * unnamed and file them again on the next page load. A file written at that
 * path after the removal is a different file and is not dismissed. The mark
 * is kept here, not only in the listing: a record filed after the fact for a
 * dismissed file is refused unless it says it was asked for, so a client that
 * does not know the mark (an older bundle still in a browser's cache, say)
 * cannot file the file again and take the removal back with it.
 *
 * Edition numbers are the server's. A client may propose one; if it collides
 * with a number another device already used, the server hands back the number
 * it assigned instead, and the client corrects its copy.
 *
 * One process serves an archive at a time. Each keeps the whole archive in
 * memory and writes all of it over the file, so a second server on the same
 * file (a dev server started beside the preview, say) would put its copy over
 * the first one's writes, and neither would know. The first to start holds a
 * lock beside the file, `archive.json.lock`, naming its process; another
 * answers every archive route with 503 and a sentence saying so, and takes
 * the archive, read afresh from disk, once the holder has stopped.
 *
 * Local-only, same posture as the other servers: writes pass guardMutation,
 * the body is capped, and nothing here touches a media file. GET /api/outputs
 * lists them, marking the ones a record already names and the ones whose
 * record was removed, so the client can find files no record describes.
 */
import { randomUUID } from 'node:crypto'
import { linkSync, mkdirSync, promises as fs, readFileSync, renameSync, unlinkSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { guardMutation, readBody, reqUrl, safely, send } from './guard.mjs'

const OUTPUTS = process.env.SWITCHGEN_OUTPUTS ?? '/mnt/storage/ai/outputs'
const ARCHIVE = process.env.SWITCHGEN_ARCHIVE ?? path.join(OUTPUTS, '.switchgen', 'archive.json')

const MEDIA = /\.(png|jpe?g|webp|gif|avif|bmp|webm|mp4|mkv|mov|m4v)$/i
const VIDEO = /\.(webm|mp4|mkv|mov|m4v)$/i
/** A record batch. 200 records at about a kilobyte each is a fifth of this. */
const BODY_MAX = 2 * 1048576
/** How long a removal is remembered, so an offline device can learn of it. */
const TOMBSTONE_MS = 30 * 86400000
const SAVE_DEBOUNCE_MS = 300
const OUTPUTS_CACHE_MS = 4000

// ------------------------------------------------------------------ store --

const num = (v, fallback) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback)

/** This load of the file. A client compares it with the one its revs came from. */
const BOOT = randomUUID()

function fresh() {
  return { v: 3, epoch: randomUUID(), rev: 0, base: 0, nextNo: 1, records: new Map(), tombstones: new Map(), dismissed: new Map() }
}

/** `{rel: removedAt}` from the file, keeping only well-formed pairs. */
function readDismissed(raw) {
  const out = new Map()
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out
  for (const [rel, at] of Object.entries(raw)) if (typeof at === 'number' && Number.isFinite(at)) out.set(rel, at)
  return out
}

// ------------------------------------------------------------------- lock --

const LOCK = `${ARCHIVE}.lock`

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
    // The command name, in parentheses, may hold spaces; the fields after it
    // cannot. The start time is the 22nd field, the 20th after the name.
    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19] ?? null
  } catch {
    return null
  }
}

const SELF = { pid: process.pid, boot: MACHINE_BOOT, start: startOf(process.pid) }

/** Whether the process a lock names is still running. */
function holderAlive(holder) {
  if (!Number.isInteger(holder?.pid) || holder.pid <= 0) return false
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

function readHolder() {
  try { return JSON.parse(readFileSync(LOCK, 'utf8')) } catch { return null }
}

/**
 * Whether this process may write the archive. `none` is a folder where no lock
 * could be made at all (read-only, say), which is left as it always was.
 */
let lockState = 'unheld'

/**
 * Take the lock, or return the pid of the live server that holds it (0 when
 * it cannot be told).
 *
 * A lock naming this process is its own: Vite loads this file afresh in the
 * same process when its config reloads, and each load takes the archive over
 * from the one before, as the exit hooks below do. A lock naming a process
 * that has gone was left by a crash or a kill, and is cleared. The lock is
 * written whole beside it and linked into place, which fails if one is there,
 * so no server ever reads another's half written.
 */
function takeLock() {
  if (lockState !== 'unheld') return null
  const tmp = `${LOCK}.${process.pid}.tmp`
  try {
    mkdirSync(path.dirname(LOCK), { recursive: true })
    writeFileSync(tmp, JSON.stringify(SELF))
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        linkSync(tmp, LOCK)
        lockState = 'held'
        return null
      } catch (err) {
        if (err?.code !== 'EEXIST') throw err
      }
      const holder = readHolder()
      if (holder?.pid === process.pid) {
        renameSync(tmp, LOCK)
        lockState = 'held'
        return null
      }
      if (holder !== null && holderAlive(holder)) return holder.pid
      try { unlinkSync(LOCK) } catch { /* cleared by someone else first */ }
    }
    return readHolder()?.pid ?? 0
  } catch (err) {
    console.warn(`[switchgen-archive] could not lock ${ARCHIVE}: ${err?.message ?? err}; a second server on this file would not be noticed`)
    lockState = 'none'
    return null
  } finally {
    try { unlinkSync(tmp) } catch { /* renamed into place, or never written */ }
  }
}

/**
 * Whether the lock still names this process, asked before every write. A lock
 * taken over after this process took it (its file deleted by hand, and
 * another server started) means another server now writes the file.
 */
function stillMine() {
  if (lockState !== 'held') return lockState === 'none'
  const holder = readHolder()
  if (holder?.pid === process.pid) return true
  lockState = 'unheld'
  // Gone with nobody in its place (deleted by hand): taken again, and this
  // process carries on with what it holds.
  if (holder === null && takeLock() === null) return true
  // What this process holds is behind what the other server has written, so
  // it is dropped, and read again from disk if the archive comes back.
  store = null
  loading = null
  console.warn(`[switchgen-archive] ${ARCHIVE} is locked by another server now; this one stops writing it`)
  return false
}

/** Let the file go as the process ends, so the next server does not have to decide the lock is stale. */
function releaseLock() {
  if (lockState !== 'held') return
  if (readHolder()?.pid !== process.pid) return
  try { unlinkSync(LOCK) } catch { /* gone already */ }
  lockState = 'unheld'
}

/** Another server holds the archive. Answered as 503, in its own words. */
class ArchiveBusy extends Error {
  constructor(pid) {
    super(
      `Another SwitchGen server${pid ? ` (process ${pid})` : ''} is using this archive, so this one cannot share it. ` +
        'Stop the other server, or give this one an archive of its own.',
    )
  }
}

/** Say once, where the server was started, that this one is standing back. */
let saidBusy = false
function standBack(pid) {
  if (!saidBusy) {
    saidBusy = true
    console.warn(`[switchgen-archive] ${ARCHIVE} is in use by another SwitchGen server${pid ? ` (process ${pid})` : ''}; this one answers 503 for the archive until that one stops`)
  }
  return new ArchiveBusy(pid)
}

// ------------------------------------------------------------------ store --

let store = null
let loading = null

async function load() {
  if (store) return store
  if (!loading) {
    // Asked again on every request while another server holds the lock, so
    // this one takes the archive, read afresh from disk, once that one stops.
    const holder = takeLock()
    if (holder !== null) throw standBack(holder)
    loading = (async () => {
      try {
        const raw = await fs.readFile(ARCHIVE, 'utf8')
        const doc = JSON.parse(raw)
        if (!doc || typeof doc !== 'object' || !doc.records || typeof doc.records !== 'object') {
          throw new Error('not an archive')
        }
        const records = new Map(Object.entries(doc.records))
        let nextNo = num(doc.nextNo, 1)
        for (const r of records.values()) nextNo = Math.max(nextNo, num(r?.no, 0) + 1)
        const epoch = typeof doc.epoch === 'string' && doc.epoch ? doc.epoch : null
        const rev = num(doc.rev, 0)
        store = {
          v: 3,
          epoch: epoch ?? randomUUID(),
          rev,
          base: rev,
          nextNo,
          records,
          tombstones: new Map(Object.entries(doc.tombstones ?? {})),
          dismissed: readDismissed(doc.dismissed),
        }
        // An archive from before the epoch gets one now and keeps it. Left
        // unwritten, every restart would make a new one and send every client
        // back to reading the whole archive.
        if (!epoch) save()
      } catch (err) {
        if (err?.code !== 'ENOENT') {
          // An archive this build cannot read is set aside, never overwritten.
          const aside = ARCHIVE.replace(/\.json$/, '') + `.broken-${Date.now()}.json`
          try { await fs.rename(ARCHIVE, aside) } catch { /* nothing to set aside */ }
          console.warn(`[switchgen-archive] could not read ${ARCHIVE}: ${err?.message ?? err}; kept as ${aside}`)
        }
        store = fresh()
      }
      return store
    })()
  }
  return loading
}

let saveTimer = null
let writing = Promise.resolve()
/**
 * Changes made, and how many of them a finished write holds. A change is
 * answered before it is on disk, so the two differ until a write that
 * serialised it has been renamed into place, and only then is it safe to exit.
 */
let changes = 0
let persisted = 0

function serialise(s) {
  const cutoff = Date.now() - TOMBSTONE_MS
  for (const [id, t] of s.tombstones) if (num(t?.at, 0) < cutoff) s.tombstones.delete(id)
  return JSON.stringify({
    v: s.v,
    epoch: s.epoch,
    rev: s.rev,
    nextNo: s.nextNo,
    records: Object.fromEntries(s.records),
    tombstones: Object.fromEntries(s.tombstones),
    dismissed: Object.fromEntries(s.dismissed),
  })
}

/** Atomic: write beside, then rename over. A crash mid-write leaves the old file whole. */
function writeNow() {
  saveTimer = null
  if (!store || !stillMine()) return writing
  const upTo = changes
  const body = serialise(store)
  writing = writing.then(async () => {
    await fs.mkdir(path.dirname(ARCHIVE), { recursive: true })
    const tmp = `${ARCHIVE}.tmp`
    await fs.writeFile(tmp, body, 'utf8')
    await fs.rename(tmp, ARCHIVE)
    persisted = Math.max(persisted, upTo)
  }).catch(err => {
    console.warn(`[switchgen-archive] could not write ${ARCHIVE}: ${err?.message ?? err}`)
  })
  return writing
}

function save() {
  changes++
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(writeNow, SAVE_DEBOUNCE_MS)
}

/**
 * Write whatever is not yet on disk, synchronously, as the process ends.
 *
 * An asynchronous write started here never lands: Vite answers SIGTERM by
 * closing the server and calling process.exit(), and the file system work
 * still queued is dropped with the process. It writes through a file of its
 * own beside the archive, so a write already under way cannot interleave
 * with this one.
 */
function flushSync() {
  if (!store || persisted >= changes) return
  if (saveTimer) { clearTimeout(saveTimer); saveTimer = null }
  if (!stillMine()) return
  try {
    mkdirSync(path.dirname(ARCHIVE), { recursive: true })
    const tmp = `${ARCHIVE}.exit.tmp`
    writeFileSync(tmp, serialise(store), 'utf8')
    renameSync(tmp, ARCHIVE)
    persisted = changes
  } catch (err) {
    console.warn(`[switchgen-archive] could not write ${ARCHIVE} on exit: ${err?.message ?? err}`)
  }
}

// -------------------------------------------------------------- mutation --

/** The least a record must carry to be stored: an identity, a time, a file. */
function sane(r) {
  return !!r && typeof r === 'object' && !Array.isArray(r) &&
    typeof r.id === 'string' && r.id.length > 0 && r.id.length <= 128 &&
    typeof r.at === 'number' && Number.isFinite(r.at) &&
    !!r.file && typeof r.file === 'object' && typeof r.file.filename === 'string' && r.file.filename.length > 0
}

/** Every output file a record names, as a path under the outputs root: the same spelling GET /api/outputs uses. */
function relsOf(r) {
  const out = []
  for (const f of [r?.file, ...(Array.isArray(r?.files) ? r.files : [])]) {
    if (!f || typeof f.filename !== 'string' || !f.filename) continue
    if (f.type && f.type !== 'output') continue
    out.push(typeof f.subfolder === 'string' && f.subfolder ? `${f.subfolder}/${f.filename}` : f.filename)
  }
  return out
}

/** Which record names each output file. */
function fileOwners(s) {
  const owners = new Map()
  for (const r of s.records.values()) for (const rel of relsOf(r)) if (!owners.has(rel)) owners.set(rel, r.id)
  return owners
}

/**
 * Whether `rel` is still the file whose record was removed: it was dismissed,
 * and it has not been written since. The listing asks the same question of
 * the same times. A file whose time could not be read is taken to be the one
 * removed, so the removal stands.
 */
function stillDismissed(s, rel, written) {
  const removedAt = s.dismissed.get(rel)
  if (removedAt === undefined) return false
  const at = written.get(rel)
  return at === undefined || at <= removedAt
}

/**
 * When each dismissed file these records would file after the fact was last
 * written, read from disk, for {@link upsert}. Only those are read, and only
 * inside the outputs root.
 */
async function dismissedWritten(s, records) {
  const out = new Map()
  const root = path.resolve(OUTPUTS)
  for (const r of records) {
    if (!sane(r) || r.recovered !== true || r.refiled === true || s.records.has(r.id)) continue
    for (const rel of relsOf(r)) {
      if (!s.dismissed.has(rel) || out.has(rel)) continue
      const full = path.resolve(root, rel)
      if (!full.startsWith(root + path.sep)) continue
      try { out.set(rel, (await fs.stat(full)).mtimeMs) } catch { /* not there: the removal stands */ }
    }
  }
  return out
}

/**
 * A desk's record `r`, taking the place of `held`, a record of the same file
 * filed after the fact. It keeps what the reader added to that one: the star,
 * the note, the tags and rating read from the picture. The first such record
 * it replaces gives it its edition number, so the number the reader has seen
 * on the card stays with the card.
 */
function inPlaceOfRecovered(held, r, takeNumber) {
  return {
    ...r,
    no: takeNumber && num(held.no, 0) > 0 ? held.no : r.no,
    starred: r.starred || held.starred || undefined,
    note: r.note || held.note || undefined,
    tags: Array.isArray(r.tags) && r.tags.length ? r.tags : held.tags,
    rating: r.rating ?? held.rating,
  }
}

/** The fields {@link inPlaceOfRecovered} can fill in from the record it replaces. */
const KEPT_FIELDS = ['starred', 'note', 'tags', 'rating']

/**
 * What a record stored after taking another's place holds that the copy sent
 * did not, for the device that sent it. That device stamps its copy with the
 * rev the record was stored at, and a pull passes over a record whose rev it
 * already holds, so without these it would never show the star or the note;
 * and its next edit, sent whole, would put its copy over the stored one
 * without them, on every device.
 */
function keptOver(sent, stored) {
  let kept = null
  for (const k of KEPT_FIELDS) {
    if (stored[k] !== undefined && stored[k] !== sent[k]) (kept ??= {})[k] = stored[k]
  }
  return kept
}

/**
 * Store what a client sent. Returns what was stamped and the ids refused.
 *
 * Three kinds of record are refused, and the client drops its copy of each:
 *   - A copy stamped before its record was removed here. That is a stale edit
 *     from a device that had not heard of the removal, and the removal is the
 *     newer act. A record with no stamp has never been here, which is how an
 *     undo sends one back, so it passes; so does anything sent to the restore
 *     route, which exists to overrule a removal.
 *   - A record filed after the fact for an output another record already
 *     names. Two tabs, or a browser holding only part of the archive, can each
 *     file the same file; the first record for it is the one kept.
 *   - A record filed after the fact for a file whose record was removed, and
 *     which has not been written again since (`written` says when each was
 *     last written, read from disk). Filing it would take the removal back
 *     with nobody having asked. The reader can ask: a record marked `refiled`
 *     is one the reader filed again on purpose, and the restore route takes
 *     it too. An undo sends back the record that was removed, whose tombstone
 *     is still here, and passes as well.
 *
 * The other way round, a desk's first record of a file that a record filed
 * after the fact already names takes that record's place. The recovery pass
 * on one device can file a clip before the desk that made it on another has
 * filed it (a phone locked while the clip finished, say), and the desk's
 * record, which knows how the clip was made, is the one to keep. It takes the
 * other's edition number and whatever the reader added to it, and the other
 * is removed, so every device drops it on its next pull. Its files are not
 * dismissed: the desk's record names them. What it took that the copy sent
 * did not have comes back in its entry of `assigned`, as `kept`, for the
 * sender to add to its own copy.
 */
function upsert(s, records, { restore = false, written = new Map() } = {}) {
  const assigned = []
  const refused = []
  const noOwner = new Map()
  for (const r of s.records.values()) if (num(r.no, 0) > 0) noOwner.set(r.no, r.id)
  let owners = null
  for (const sent of records) {
    if (!sane(sent)) continue
    let r = sent
    const tomb = s.tombstones.get(r.id)
    if (tomb && !restore && typeof r.rev === 'number' && num(tomb.rev, 0) > r.rev) {
      refused.push(r.id)
      continue
    }
    if (r.recovered === true && !s.records.has(r.id)) {
      owners ??= fileOwners(s)
      const rels = relsOf(r)
      if (rels.some((rel) => { const o = owners.get(rel); return o !== undefined && o !== r.id })) {
        refused.push(r.id)
        continue
      }
      if (!restore && r.refiled !== true && !tomb && rels.some((rel) => stillDismissed(s, rel, written))) {
        refused.push(r.id)
        continue
      }
    } else if (r.recovered !== true && !s.records.has(r.id)) {
      owners ??= fileOwners(s)
      let first = true
      for (const rel of relsOf(r)) {
        const o = owners.get(rel)
        const held = o !== undefined && o !== r.id ? s.records.get(o) : undefined
        if (held?.recovered !== true) continue
        r = inPlaceOfRecovered(held, r, first)
        first = false
        s.records.delete(held.id)
        s.tombstones.set(held.id, { rev: ++s.rev, at: Date.now() })
        if (noOwner.get(held.no) === held.id) noOwner.delete(held.no)
        for (const its of relsOf(held)) if (owners.get(its) === held.id) owners.delete(its)
      }
    }
    const rev = ++s.rev
    let no = num(r.no, 0) > 0 ? Math.floor(r.no) : 0
    const owner = no ? noOwner.get(no) : undefined
    if (owner && owner !== r.id) no = 0
    if (!no) no = s.nextNo++
    else s.nextNo = Math.max(s.nextNo, no + 1)
    noOwner.set(no, r.id)
    // `pending` is a browser's own note that it has a change to send; it
    // means nothing here and must not travel on to other devices.
    const { rev: _oldRev, pending: _pending, ...rest } = r
    s.records.set(r.id, { ...rest, no, rev })
    if (owners) for (const rel of relsOf(r)) if (!owners.has(rel)) owners.set(rel, r.id)
    // A record naming a file again (an undo, a restore) takes back its removal.
    for (const rel of relsOf(r)) s.dismissed.delete(rel)
    s.tombstones.delete(r.id)
    const kept = r === sent ? null : keptOver(sent, r)
    assigned.push(kept ? { id: r.id, no, rev, kept } : { id: r.id, no, rev })
  }
  return { assigned, refused }
}

/**
 * Remove records. Their files stay on disk, and are remembered as dismissed so
 * the recovery pass does not file them again; unlike the tombstone, that is
 * kept for as long as the archive is, because the file is.
 */
function remove(s, ids) {
  let n = 0
  const now = Date.now()
  for (const id of ids) {
    if (typeof id !== 'string' || !s.records.has(id)) continue
    for (const rel of relsOf(s.records.get(id))) s.dismissed.set(rel, now)
    s.records.delete(id)
    s.tombstones.set(id, { rev: ++s.rev, at: now })
    n++
  }
  return n
}

/**
 * What changed after `since`. `removedRev` gives each removal's rev, which a
 * client rebuilding after a lost tail needs: a removal from before the loss is
 * older than a copy it holds from after, and one from since is newer.
 */
function changesSince(s, since) {
  const full = !(since > 0)
  const records = []
  for (const r of s.records.values()) if (full || num(r.rev, 0) > since) records.push(r)
  const removed = []
  const removedRev = Object.create(null)
  for (const [id, t] of s.tombstones) {
    const rev = num(t?.rev, 0)
    if (rev > since) { removed.push(id); removedRev[id] = rev }
  }
  return { epoch: s.epoch, boot: BOOT, base: s.base, rev: s.rev, nextNo: s.nextNo, full, records, removed, removedRev }
}

/** What every stream message says: where the log is, which log, and which load of it. */
const where = (s) => `data: ${JSON.stringify({ rev: s.rev, epoch: s.epoch, boot: BOOT })}\n\n`

// ---------------------------------------------------------------- stream --

const watchers = new Set()

function broadcast(s) {
  const line = where(s)
  for (const res of watchers) {
    try { res.write(line) } catch { watchers.delete(res) }
  }
}

const pinger = setInterval(() => {
  for (const res of watchers) {
    try { res.write(': ping\n\n') } catch { watchers.delete(res) }
  }
}, 25000)
pinger.unref()

// ------------------------------------------------------------------- exit --

// Vite loads this file afresh each time its config reloads, so the process
// hooks are added once, and they call one flush for each archive file: the
// newest load's. A load that has been replaced holds the archive as it stood
// when the new load read the file. Kept on the list, every load's archive
// stayed in memory, one more copy with each reload, and at exit the oldest
// wrote first: one whose last write had failed put its old copy over the
// file, and the newest, with nothing of its own unsaved, left it there. So a
// new load takes the old one's place, and the old one stops its timer and is
// not called again.
//
// SIGTERM, which `switchgen stop` sends, is Vite's: it closes the server and
// calls process.exit(), and the exit hook writes. Nothing ends the process on
// SIGINT but Node's default, which skips exit hooks, and any listener takes
// that default away; the signal-exit handler already in a preview process
// stands down whenever another listener is present. So the SIGINT listener
// here writes and then ends the process itself, with the code the default
// gives, rather than leaving it running.
const FLUSHERS = Symbol.for('switchgen.archive.flush')
let flushers = globalThis[FLUSHERS]
if (!(flushers instanceof Map)) {
  // A build before this one kept every load's flush in a Set under the same
  // name, with hooks that call them all. Emptied, those hooks call nothing,
  // and this list and its own hooks take over.
  flushers?.clear?.()
  flushers = globalThis[FLUSHERS] = new Map()
  const all = flushers
  const flushAll = () => { for (const load of all.values()) load.flush() }
  process.once('exit', flushAll)
  process.once('SIGINT', () => {
    flushAll()
    process.exit(130)
  })
}
flushers.get(ARCHIVE)?.retire()
flushers.set(ARCHIVE, {
  flush: () => {
    flushSync()
    releaseLock()
  },
  retire: () => clearInterval(pinger),
})

// --------------------------------------------------------------- outputs --

let outputsCache = { at: 0, files: [] }

async function walkMedia(dir, root, out) {
  let entries
  try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return out }
  for (const e of entries) {
    if (e.name.startsWith('.')) continue
    const full = path.join(dir, e.name)
    if (e.isDirectory()) { await walkMedia(full, root, out); continue }
    if (!MEDIA.test(e.name)) continue
    try {
      const st = await fs.stat(full)
      const rel = path.relative(root, full).split(path.sep).join('/')
      out.push({
        rel,
        name: e.name,
        subfolder: path.relative(root, dir).split(path.sep).join('/'),
        size: st.size,
        mtime: st.mtimeMs,
        kind: VIDEO.test(e.name) ? 'video' : 'image',
      })
    } catch { /* vanished mid-walk */ }
  }
  return out
}

async function outputs() {
  if (Date.now() - outputsCache.at < OUTPUTS_CACHE_MS) return outputsCache.files
  const files = await walkMedia(OUTPUTS, OUTPUTS, [])
  files.sort((a, b) => b.mtime - a.mtime)
  outputsCache = { at: Date.now(), files }
  return files
}

// ---------------------------------------------------------------- routes --

const METHODS = new Map([
  ['/api/archive', 'GET'],
  ['/api/archive/stream', 'GET'],
  ['/api/archive/upsert', 'POST'],
  ['/api/archive/restore', 'POST'],
  ['/api/archive/remove', 'POST'],
  ['/api/outputs', 'GET'],
])

export function switchgenArchive() {
  const handler = async (req, res, next) => {
    // Parsed the way every other server here parses it. switchgenApi answers
    // an unreadable path before this runs, but only because it is mounted
    // first; this handler must not depend on the order of the plugin list.
    const url = reqUrl(req)
    if (!url) return send(res, 400, { error: 'the request path is not a valid URL' })
    const p = url.pathname
    if (!p.startsWith('/api/archive') && p !== '/api/outputs') return next()

    try {
      if (p === '/api/archive' && req.method === 'GET') {
        const s = await load()
        const since = Number(url.searchParams.get('since')) || 0
        return send(res, 200, { ...changesSince(s, since), file: ARCHIVE })
      }

      if (p === '/api/archive/stream' && req.method === 'GET') {
        const s = await load()
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
          'X-Accel-Buffering': 'no',
        })
        res.write(where(s))
        watchers.add(res)
        const stop = () => { watchers.delete(res); try { res.end() } catch { /* gone */ } }
        res.on('close', stop)
        res.on('error', stop)
        return
      }

      if ((p === '/api/archive/upsert' || p === '/api/archive/restore') && req.method === 'POST') {
        if (!guardMutation(req, res)) return
        const b = await readBody(req, BODY_MAX)
        if (!b || !Array.isArray(b.records)) return send(res, 400, { error: 'body must be JSON {records: [...]} under 2 MB' })
        const s = await load()
        const restore = p === '/api/archive/restore'
        const written = restore ? new Map() : await dismissedWritten(s, b.records)
        const { assigned, refused } = upsert(s, b.records, { restore, written })
        if (assigned.length) { save(); broadcast(s) }
        return send(res, 200, { rev: s.rev, nextNo: s.nextNo, assigned, refused })
      }

      if (p === '/api/archive/remove' && req.method === 'POST') {
        if (!guardMutation(req, res)) return
        const b = await readBody(req, BODY_MAX)
        if (!b || !Array.isArray(b.ids)) return send(res, 400, { error: 'body must be JSON {ids: [...]}' })
        const s = await load()
        const removed = remove(s, b.ids)
        if (removed) { save(); broadcast(s) }
        return send(res, 200, { rev: s.rev, removed })
      }

      if (p === '/api/outputs' && req.method === 'GET') {
        const since = Number(url.searchParams.get('since')) || 0
        const [files, s] = await Promise.all([outputs(), load()])
        // Marked here because only the server holds every record. A browser
        // keeps a window of the archive, so a file whose record fell out of
        // that window would otherwise look unfiled and be filed a second time.
        const named = new Set()
        for (const r of s.records.values()) for (const rel of relsOf(r)) named.add(rel)
        const listed = since ? files.filter(f => f.mtime > since) : files
        const mark = (f) => {
          if (named.has(f.rel)) return { ...f, filed: true }
          // Written no later than its record was removed: the same file, which
          // the reader took out of the archive and kept on disk.
          const removedAt = s.dismissed.get(f.rel)
          return removedAt !== undefined && f.mtime <= removedAt ? { ...f, dismissed: true } : f
        }
        return send(res, 200, { root: OUTPUTS, files: listed.map(mark) })
      }

      const method = METHODS.get(p)
      if (method && method !== req.method) {
        res.setHeader('Allow', method)
        return send(res, 405, { error: `${p} takes ${method}, not ${req.method}` })
      }
      return send(res, 404, { error: `no such endpoint: ${req.method} ${p}` })
    } catch (err) {
      if (res.headersSent) { try { res.end() } catch { /* gone */ } return }
      if (err instanceof ArchiveBusy) return send(res, 503, { error: err.message, busy: 'archive' })
      return send(res, 500, { error: String(err?.message ?? err) })
    }
  }

  // Taken as the server starts, not at its first request, so the server that
  // was running first keeps the archive when a second one is started beside it.
  const serve = (server) => {
    const holder = takeLock()
    if (holder !== null) standBack(holder)
    server.middlewares.use(safely(handler))
  }
  return {
    name: 'switchgen-archive',
    configureServer: serve,
    configurePreviewServer: serve,
  }
}
