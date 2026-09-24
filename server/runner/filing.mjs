/**
 * Filing a finished job in the archive: the one filer of the queue's work.
 *
 * The page used to file what it made, and a page asleep when its clip landed
 * filed nothing until it woke, dated to when it woke. The runner files the
 * moment ComfyUI's record says the job succeeded, from the record template the
 * page built when it pressed Make, so the archive gets the same record the
 * desk would have written, with ComfyUI's own times.
 *
 * It files each job exactly once. The record's id is the job's, fileOnce
 * looks that id up before it adds anything, and the job is called done only
 * after the archive has said its write is on disk. A process killed between
 * any two of those steps comes back to a job still filing, and the next
 * attempt finds the record already there.
 */
import { relOf } from './comfyRecord.mjs'

/**
 * Fields of a record template that are the server's to fill, or that must not
 * travel on a new record: its identity and numbering, a browser's own marks,
 * and the six the finished run decides.
 */
const SERVER_FIELDS = ['id', 'no', 'rev', 'pending', 'recovered', 'refiled', 'file', 'files', 'kind', 'promptId', 'durationMs', 'at']

/**
 * The file a job stands for: the first of the kind the desk asked for, or,
 * when the desk takes any file, the first file of all.
 */
export function pickPrimary(files, kind, orFirst) {
  const list = Array.isArray(files) ? files : []
  return list.find((f) => f?.kind === kind) ?? (orFirst ? list[0] ?? null : null)
}

/**
 * How long ComfyUI spent on the run: its end stamp less its execution_start
 * stamp. 0, which every estimate reads as "not measured", when either is
 * missing or they are out of order. The time the prompt was queued is never
 * used: it counts the wait behind other work as rendering time.
 */
export function durationOf(startedAt, finishedAt) {
  const ok = (v) => typeof v === 'number' && Number.isFinite(v)
  return ok(startedAt) && ok(finishedAt) && finishedAt >= startedAt ? finishedAt - startedAt : 0
}

/** The output files a job names, as paths under the outputs root: what the archive's listing names them by. */
export function relsOf(job) {
  const out = []
  for (const f of [...(Array.isArray(job?.files) ? job.files : []), job?.primary]) {
    if (!f || typeof f.filename !== 'string' || !f.filename) continue
    if (f.type && f.type !== 'output') continue
    const rel = relOf(f)
    if (!out.includes(rel)) out.push(rel)
  }
  return out
}

/**
 * What a job that ComfyUI says succeeded becomes as it enters filing: its
 * files, its handoff frame, the file it stands for, and ComfyUI's times.
 * `sawEndAt` is when the runner learned of the ending, the record's date when
 * ComfyUI kept no end stamp.
 */
export function filingPatch(job, run, now) {
  const files = Array.isArray(run?.files) ? run.files : []
  const ranAt = typeof run?.startedAt === 'number' ? run.startedAt : job.ranAt ?? null
  const finishedAt = typeof run?.finishedAt === 'number' ? run.finishedAt : null
  return {
    status: 'filing',
    wait: null,
    files,
    frame: run?.frame ?? null,
    primary: pickPrimary(files, job.primaryKind, job.orFirst),
    ranAt,
    finishedAt,
    durationMs: durationOf(run?.startedAt ?? null, finishedAt),
    sawEndAt: now,
  }
}

/** The archive record for a job, from the page's template and the finished run. */
export function recordFrom(template, job) {
  const rest = { ...(template ?? {}) }
  for (const k of SERVER_FIELDS) delete rest[k]
  const files = Array.isArray(job.files) ? job.files : []
  return {
    ...rest,
    id: job.id,
    file: job.primary,
    files: files.length > 1 ? files : undefined,
    kind: job.primary.kind,
    promptId: job.promptId ?? job.promptIdInternal ?? '',
    durationMs: job.durationMs ?? 0,
    at: job.finishedAt ?? job.sawEndAt,
  }
}

/**
 * Take one job in 'filing' a step further: to done, to a failure the desk
 * asked for when there is no file, or, when the archive cannot take the
 * record now, to a wait for the next pass.
 *
 * `rt` is the engine's side of this: the job, its commit, its saved payload,
 * the archive, and `alive()`, which throws once the runner has been retired
 * so nothing is filed or committed by a runner that has handed over.
 */
export async function fileJob(rt, id) {
  const job = rt.job(id)
  if (!job || job.status !== 'filing') return
  const rels = relsOf(job)
  const still = (cur) => cur?.status === 'filing'

  if (!job.primary) {
    rt.commit((d) => {
      if (!still(d.jobs[id])) return false
      if (job.noFile === 'fail') {
        rt.endJob(d, id, 'failed', rt.fault('no-file', { sent: true, message: 'The job finished but wrote no file of the kind this desk keeps.' }))
      } else {
        rt.endJob(d, id, 'done', null, { entryId: null, entryNo: null })
      }
    })
    rt.archive.unclaim(rels)
    return
  }

  // Asked again with the same settings, ComfyUI hands back the file it made
  // the first time without drawing anything. Filing it again would put a
  // second record on one file, so the job is done as a repeat of the record
  // that names it, and claims no time.
  if (job.primary.cached === true) {
    let known
    try {
      known = await rt.archive.recordNaming(relOf(job.primary))
    } catch (err) {
      rt.waitForDisk(id, err)
      return
    }
    rt.alive()
    if (known && !known.recovered) {
      rt.commit((d) => {
        if (!still(d.jobs[id])) return false
        rt.endJob(d, id, 'done', null, { entryId: known.id, entryNo: known.no, repeatOf: known.id, durationMs: 0 })
      })
      rt.archive.unclaim(rels)
      return
    }
  }

  const payload = rt.readPayload(id)
  if (!payload) {
    rt.commit((d) => {
      if (!still(d.jobs[id])) return false
      rt.endJob(d, id, 'failed', rt.fault('internal', { sent: true, message: 'The saved record for this job is missing, so it could not be filed. Its file is on disk.' }))
    })
    rt.archive.unclaim(rels)
    return
  }

  let filed
  try {
    filed = await rt.archive.fileOnce(recordFrom(payload.record, job))
  } catch (err) {
    rt.waitForDisk(id, err)
    return
  }
  rt.alive()
  if (filed.removed) {
    // The reader removed this record between two attempts. It stays removed.
    rt.commit((d) => {
      if (!still(d.jobs[id])) return false
      rt.endJob(d, id, 'done', null, { entryId: null, entryNo: null })
    })
    rt.archive.unclaim(rels)
    return
  }

  let durable = false
  try {
    durable = await rt.archive.durable()
  } catch (err) {
    rt.waitForDisk(id, err)
    return
  }
  rt.alive()
  if (!durable) {
    rt.waitForDisk(id, new Error('the archive could not be written'))
    return
  }
  rt.commit((d) => {
    if (!still(d.jobs[id])) return false
    rt.endJob(d, id, 'done', null, { entryId: filed.entryId, entryNo: filed.no })
  })
  rt.archive.unclaim(rels)
}
