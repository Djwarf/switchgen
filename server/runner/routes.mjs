/**
 * The queue's routes under /api/runner: taking in a group of work, the live
 * stream every page watches, and the stops and words a page sends.
 *
 * A page hands over a whole Make, Run or Render in one POST, and the answer
 * comes only once the work is on disk, so a page that gets a 200 can close.
 * The same POST sent again (the phone lost the answer and tried again) is
 * taken as the same work, not new work: ids are the page's, and a group
 * already here with the same jobs and the same contents is answered as it
 * stands. Any answer but 200 means nothing was kept.
 *
 * Every write passes guardMutation before a byte of its body is read, as
 * every other route here does.
 */
import { createHash } from 'node:crypto'
import { guardMutation, readBody, reqUrl, send, sse, sseOpen } from '../guard.mjs'
import { CHAIN_TOKEN, heavyFloor, tokenSites } from './comfyRecord.mjs'
import { DiskError, Retired } from './engine.mjs'

const MiB = 1048576
/** A group of up to 200 jobs, each graph up to 1 MiB. */
const GROUP_BODY_MAX = 16 * MiB
const SMALL_BODY_MAX = 64 * 1024
const GRAPH_MAX = MiB
const RECORD_MAX = 64 * 1024
const META_MAX = 16 * 1024
const JOBS_MAX = 200
const PING_MS = 25_000

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

/**
 * The kind of group each desk sends. A lab set is the model lab's: jobs that
 * go independently, except one chained to another waits for it to end. No
 * busy lock applies to it, so a lab night never refuses the Pictures desk.
 */
const KIND_OF_DESK = { images: 'batch', video: 'clips', reel: 'pass', lab: 'set' }

const DISK_FULL = 'The server’s disk is full, so it cannot save this work. Nothing was taken.'
const BUSY_IMAGES = 'A batch of pictures is already being made, from this page or another.'
const BUSY_REEL = 'A reel is already being rendered, from this page or another. Stop it there or wait for it to finish.'
const HOLD_CHANGED = 'The hold has changed since this page showed it.'

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v)
const bytes = (v) => Buffer.byteLength(JSON.stringify(v) ?? '')
const sha256 = (text) => createHash('sha256').update(text).digest('hex')
const isString = (v, max) => typeof v === 'string' && v.length <= max

/** How often CHAIN_TOKEN appears anywhere in a graph, a substring or a key included. */
function tokenCount(graph) {
  return JSON.stringify(graph).split(CHAIN_TOKEN).length - 1
}

/**
 * Check a POSTed group against the contract, and return it tidied, or
 * `{error}` saying the first thing wrong.
 */
export function validateGroup(body, desks) {
  if (!isObject(body) || body.v !== 1) return { error: 'body must be {v: 1, group, jobs}' }
  const g = body.group
  if (!isObject(g)) return { error: 'group must be an object' }
  if (typeof g.id !== 'string' || !UUID.test(g.id)) return { error: 'group.id must be a lowercase v4 UUID' }
  if (!Object.hasOwn(KIND_OF_DESK, g.desk)) return { error: 'group.desk must be images, video, reel or lab' }
  if (!desks.includes(g.desk)) return { error: `the ${g.desk} desk does not send its work through the queue on this server` }
  if (g.kind !== KIND_OF_DESK[g.desk]) return { error: `a ${g.desk} group must be of kind ${KIND_OF_DESK[g.desk]}` }
  if (!isString(g.label, 200)) return { error: 'group.label must be a string of at most 200 characters' }
  if (!isString(g.device, 64)) return { error: 'group.device must be a string of at most 64 characters' }
  if (!Array.isArray(body.jobs) || body.jobs.length < 1 || body.jobs.length > JOBS_MAX) {
    return { error: `jobs must be a list of 1 to ${JOBS_MAX}` }
  }

  const seen = new Set()
  const jobs = []
  for (let i = 0; i < body.jobs.length; i++) {
    const j = body.jobs[i]
    const at = `jobs[${i}]`
    if (!isObject(j)) return { error: `${at} must be an object` }
    if (typeof j.id !== 'string' || !UUID.test(j.id)) return { error: `${at}.id must be a lowercase v4 UUID` }
    if (seen.has(j.id) || j.id === g.id) return { error: `${at}.id is used twice` }
    if (!isString(j.label, 200)) return { error: `${at}.label must be a string of at most 200 characters` }
    if (!isString(j.prompt, 4000)) return { error: `${at}.prompt must be a string of at most 4000 characters` }
    if (j.kind !== 'image' && j.kind !== 'video') return { error: `${at}.kind must be image or video` }
    if (j.primary !== 'image' && j.primary !== 'video') return { error: `${at}.primary must be image or video` }
    if (typeof j.orFirst !== 'boolean') return { error: `${at}.orFirst must be true or false` }
    if (j.noFile !== 'fail' && j.noFile !== 'done') return { error: `${at}.noFile must be fail or done` }
    if (typeof j.heavy !== 'boolean') return { error: `${at}.heavy must be true or false` }
    if (!isObject(j.graph) || !Object.keys(j.graph).length) return { error: `${at}.graph must be a graph of nodes` }
    for (const [node, n] of Object.entries(j.graph)) {
      if (!isObject(n) || typeof n.class_type !== 'string' || (n.inputs !== undefined && !isObject(n.inputs))) {
        return { error: `${at}.graph node ${node} must have a class_type and inputs` }
      }
    }
    if (bytes(j.graph) > GRAPH_MAX) return { error: `${at}.graph is over 1 MiB` }
    if (!isObject(j.record)) return { error: `${at}.record must be an object` }
    if (typeof j.record.desk !== 'string' || typeof j.record.mode !== 'string') return { error: `${at}.record must name its desk and mode` }
    if (bytes(j.record) > RECORD_MAX) return { error: `${at}.record is over 64 KB` }
    if (j.meta !== undefined && j.meta !== null) {
      if (!isObject(j.meta)) return { error: `${at}.meta must be an object` }
      if (bytes(j.meta) > META_MAX) return { error: `${at}.meta is over 16 KB` }
    }
    if (g.desk === 'reel' && (typeof j.meta?.shotId !== 'string' || !j.meta.shotId)) {
      return { error: `${at}.meta.shotId is required for a reel shot` }
    }

    // The frame a chained shot opens on is put in by the server once the shot
    // before it has landed, at exactly the places the page marked, and a
    // token left anywhere else would reach ComfyUI as a file name.
    let chain = null
    if (j.chain !== undefined && j.chain !== null) {
      const c = j.chain
      if (!isObject(c) || typeof c.after !== 'string' || !UUID.test(c.after)) return { error: `${at}.chain.after must be a job id` }
      if (!seen.has(c.after)) return { error: `${at}.chain.after must name an earlier job of this group` }
      if (!Array.isArray(c.at) || !c.at.length) return { error: `${at}.chain.at must list at least one place` }
      const sites = []
      for (const s of c.at) {
        if (!Array.isArray(s) || s.length !== 2 || typeof s[0] !== 'string' || typeof s[1] !== 'string') {
          return { error: `${at}.chain.at must hold [node, input] pairs` }
        }
        if (j.graph[s[0]]?.inputs?.[s[1]] !== CHAIN_TOKEN) return { error: `${at}.chain.at names ${s[0]}.${s[1]}, which does not hold the frame's place` }
        if (sites.some(([n, k]) => n === s[0] && k === s[1])) return { error: `${at}.chain.at names ${s[0]}.${s[1]} twice` }
        sites.push([s[0], s[1]])
      }
      if (tokenSites(j.graph).length !== sites.length || tokenCount(j.graph) !== sites.length) {
        return { error: `${at}.graph holds the frame's place somewhere chain.at does not name` }
      }
      chain = { after: c.after, at: sites }
    } else if (tokenCount(j.graph) > 0) {
      return { error: `${at}.graph holds the frame's place but the job has no chain` }
    }

    seen.add(j.id)
    jobs.push({
      id: j.id,
      label: j.label,
      prompt: j.prompt,
      kind: j.kind,
      primary: j.primary,
      orFirst: j.orFirst,
      noFile: j.noFile,
      heavy: j.heavy,
      graph: j.graph,
      record: j.record,
      chain,
      meta: j.meta ?? null,
      // What "the same work" means for a request sent again: everything the
      // page sent for this job, under the same group.
      specSha: sha256(JSON.stringify({ group: { id: g.id, desk: g.desk, kind: g.kind, label: g.label, device: g.device }, job: j })),
    })
  }
  return { group: { id: g.id, desk: g.desk, kind: g.kind, label: g.label, device: g.device }, jobs }
}

/** A write to the list of work failed: the sentence the page shows. */
function diskSentence(err) {
  const code = err?.code ?? err?.cause?.code
  if (code === 'ENOSPC' || code === 'EDQUOT') return DISK_FULL
  return `The server cannot save this work: ${String(err?.cause?.message ?? err?.message ?? err).replace(/\.$/, '')}. Nothing was taken.`
}

/**
 * Take a validated group in: its payloads first, each on disk, then the list
 * of work naming it. Answers what the route says.
 */
function intake(rt, spec) {
  const s = rt.state()
  const { group, jobs } = spec

  const held = s.groups[group.id]
  if (held) {
    const same =
      held.jobIds.length === jobs.length &&
      jobs.every((j, i) => held.jobIds[i] === j.id && s.jobs[j.id]?.specSha === j.specSha)
    if (!same) return [409, { error: 'A different group with this id is already here.', conflict: 'id' }]
    return [200, { rev: s.rev, replayed: true, group: held, jobs: held.jobIds.map((id) => s.jobs[id]).filter(Boolean).map(rt.view) }]
  }
  if (jobs.some((j) => s.jobs[j.id])) return [409, { error: 'A job with one of these ids is already here.', conflict: 'id' }]

  const active = Object.values(s.groups).filter((g) => g.state === 'active')
  if (group.kind === 'batch' && active.some((g) => g.kind === 'batch')) {
    return [409, { error: BUSY_IMAGES, busy: 'images' }]
  }
  if (group.kind === 'pass') {
    const live = new Set()
    for (const g of active) {
      if (g.kind !== 'pass') continue
      for (const id of g.jobIds) {
        const shot = s.jobs[id]?.meta?.shotId
        if (typeof shot === 'string') live.add(shot)
      }
    }
    if (jobs.some((j) => live.has(j.meta.shotId))) return [409, { error: BUSY_REEL, busy: 'reel' }]
  }

  const written = []
  const t = rt.now()
  try {
    const shas = new Map()
    for (const j of jobs) {
      rt.store.writePayload(j.id, j.graph, j.record)
      written.push(j.id)
      shas.set(j.id, sha256(JSON.stringify(j.graph)))
    }
    const res = rt.commit((d) => {
      // Asked again inside the commit: nothing can come between the checks
      // above and this, since nothing here waits, but the list is the
      // authority, not a copy read before it.
      if (d.groups[group.id] || jobs.some((j) => d.jobs[j.id])) return false
      d.groups[group.id] = {
        id: group.id,
        desk: group.desk,
        kind: group.kind,
        label: group.label,
        device: group.device,
        createdAt: t,
        state: 'active',
        endedBy: null,
        endedAt: null,
        jobIds: jobs.map((j) => j.id),
        dismissed: false,
      }
      jobs.forEach((j, i) => {
        d.seq = (d.seq ?? 0) + 1
        d.jobs[j.id] = {
          id: j.id,
          groupId: group.id,
          desk: group.desk,
          kind: j.kind,
          seq: d.seq,
          index: i + 1,
          total: jobs.length,
          label: j.label,
          prompt: j.prompt,
          device: group.device,
          // The server never takes a job to be lighter than the page did,
          // and never lighter than a two-sampler graph is.
          heavy: j.heavy || heavyFloor(j.graph),
          status: 'waiting',
          wait: null,
          stopRequested: false,
          stopLanded: false,
          promptId: null,
          attempt: 0,
          createdAt: t,
          sentAt: null,
          ranAt: null,
          finishedAt: null,
          endedAt: null,
          files: [],
          primary: null,
          frame: null,
          openedOn: null,
          entryId: null,
          entryNo: null,
          repeatOf: null,
          durationMs: 0,
          error: null,
          meta: j.meta,
          dismissed: false,
          promptIdInternal: null,
          sighted: false,
          misses: 0,
          historyWaits: 0,
          primaryKind: j.primary,
          orFirst: j.orFirst,
          noFile: j.noFile,
          chain: j.chain,
          graphSha: shas.get(j.id),
          specSha: j.specSha,
          acceptedAt: null,
          sawEndAt: null,
        }
      })
    })
    if (!res) {
      for (const id of written) rt.store.removePayload(id)
      return [409, { error: 'A group or job with one of these ids is already here.', conflict: 'id' }]
    }
  } catch (err) {
    for (const id of written) rt.store.removePayload(id)
    if (err instanceof Retired) return unavailable(rt)
    console.warn(`[switchgen-runner] could not take in group ${group.id}: ${err?.message ?? err}`)
    return [507, { error: diskSentence(err) }]
  }
  rt.wake()
  const after = rt.state()
  return [200, { rev: after.rev, replayed: false, group: after.groups[group.id], jobs: jobs.map((j) => rt.view(after.jobs[j.id])) }]
}

function unavailable(rt) {
  const reason = rt.status().reason ?? 'The queue on the server is not running.'
  return [503, { error: reason, busy: 'runner', reason }]
}

const ROUTES = [
  { re: /^\/api\/runner$/, method: 'GET', name: 'snapshot' },
  { re: /^\/api\/runner\/stream$/, method: 'GET', name: 'stream' },
  { re: /^\/api\/runner\/groups$/, method: 'POST', name: 'submit' },
  { re: /^\/api\/runner\/groups\/([^/]+)\/stop$/, method: 'POST', name: 'stopGroup' },
  { re: /^\/api\/runner\/jobs\/([^/]+)$/, method: 'GET', name: 'job' },
  { re: /^\/api\/runner\/jobs\/([^/]+)\/preview$/, method: 'GET', name: 'preview' },
  { re: /^\/api\/runner\/jobs\/([^/]+)\/stop$/, method: 'POST', name: 'stopJob' },
  { re: /^\/api\/runner\/lane$/, method: 'POST', name: 'lane' },
  { re: /^\/api\/runner\/dismiss$/, method: 'POST', name: 'dismiss' },
]

/** The routes for one runner, and the stream's watchers. */
export function createRoutes(rt) {
  const watchers = new Set()

  const unsubscribe = rt.subscribe((event, data) => {
    for (const res of watchers) sse(res, event, data)
  })

  // A comment line now and then keeps a proxy between here and the phone
  // from closing a stream that is only waiting.
  const pinger = setInterval(() => {
    for (const res of watchers) {
      if (res.destroyed || res.writableEnded) { watchers.delete(res); continue }
      try { res.write(': ping\n\n') } catch { watchers.delete(res) }
    }
  }, PING_MS)
  pinger.unref?.()

  function close() {
    clearInterval(pinger)
    unsubscribe()
    for (const res of watchers) {
      try { res.end() } catch { /* gone */ }
    }
    watchers.clear()
  }

  async function post(req, res, max, act) {
    if (!guardMutation(req, res)) return
    const body = await readBody(req, max)
    if (!isObject(body)) return send(res, 400, { error: `body must be a JSON object under ${max >= MiB ? `${max / MiB} MiB` : `${max / 1024} KB`}` })
    if (!rt.status().active) return send(res, ...unavailable(rt))
    try {
      const [code, answer] = await act(body)
      return send(res, code, answer)
    } catch (err) {
      if (err instanceof Retired) return send(res, ...unavailable(rt))
      if (err instanceof DiskError) return send(res, 507, { error: diskSentence(err) })
      throw err
    }
  }

  const handler = async (req, res, next) => {
    const url = reqUrl(req)
    if (!url) return send(res, 400, { error: 'the request path is not a valid URL' })
    const p = url.pathname
    if (p !== '/api/runner' && !p.startsWith('/api/runner/')) return next()

    try {
      const route = ROUTES.find((r) => r.re.test(p))
      if (!route) return send(res, 404, { error: `no such endpoint: ${req.method} ${p}` })
      if (req.method !== route.method) {
        res.setHeader('Allow', route.method)
        return send(res, 405, { error: `${p} takes ${route.method}, not ${req.method}` })
      }
      let id
      try {
        id = decodeURIComponent(route.re.exec(p)?.[1] ?? '')
      } catch {
        return send(res, 404, { error: `no such endpoint: ${req.method} ${p}` })
      }

      switch (route.name) {
        case 'snapshot':
          rt.ensureActive()
          return send(res, 200, rt.snapshot())

        case 'stream': {
          rt.ensureActive()
          sseOpen(res)
          sse(res, 'state', rt.snapshot())
          watchers.add(res)
          const stop = () => watchers.delete(res)
          res.on('close', stop)
          res.on('error', stop)
          return
        }

        case 'job': {
          const detail = rt.jobDetail(id)
          if (!detail) return send(res, 404, { error: 'no such job' })
          return send(res, 200, detail)
        }

        case 'preview': {
          const shot = rt.preview(id)
          if (!shot) return send(res, 404, { error: 'no preview for this job' })
          res.statusCode = 200
          res.setHeader('Content-Type', shot.mime)
          res.setHeader('Cache-Control', 'no-store')
          res.setHeader('Cross-Origin-Resource-Policy', 'same-origin')
          res.setHeader('Content-Length', shot.bytes.length)
          res.end(shot.bytes)
          return
        }

        case 'submit':
          return post(req, res, GROUP_BODY_MAX, async (body) => {
            const spec = validateGroup(body, rt.desks)
            if (spec.error) return [400, { error: spec.error }]
            return intake(rt, spec)
          })

        case 'stopJob':
          return post(req, res, SMALL_BODY_MAX, async () => {
            const job = await rt.stopJob(id)
            return job ? [200, { job }] : [404, { error: 'no such job' }]
          })

        case 'stopGroup':
          return post(req, res, SMALL_BODY_MAX, async () => {
            const out = await rt.stopGroup(id)
            return out ? [200, out] : [404, { error: 'no such group' }]
          })

        case 'lane':
          return post(req, res, SMALL_BODY_MAX, async (body) => {
            if (body.action !== 'send' && body.action !== 'stop') return [400, { error: 'action must be send or stop' }]
            // The hold the word answers, by when it began, as the page showed
            // it: a word to a hold that has since changed is refused.
            const since = body.since ?? null
            if (since !== null && !(typeof since === 'number' && Number.isFinite(since))) {
              return [400, { error: 'since must be the time the hold began, as the page showed it' }]
            }
            const out = rt.laneWord(body.action, since)
            if (out === false) return [409, { error: HOLD_CHANGED }]
            return out ? [200, out] : [409, { error: 'Nothing is held.' }]
          })

        case 'dismiss':
          return post(req, res, SMALL_BODY_MAX, async (body) => {
            if (!Array.isArray(body.jobIds) || !body.jobIds.every((x) => typeof x === 'string')) {
              return [400, { error: 'jobIds must be a list of job ids' }]
            }
            return [200, { dismissed: rt.dismiss(body.jobIds) }]
          })

        default:
          return send(res, 404, { error: `no such endpoint: ${req.method} ${p}` })
      }
    } catch (err) {
      if (res.headersSent) {
        try { res.end() } catch { /* gone */ }
        return
      }
      return send(res, 500, { error: String(err?.message ?? err) })
    }
  }

  return { handler, close, watchers }
}
