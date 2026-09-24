/*
 * The lab page: runs, the blind scoring flow and the reveal.
 *
 * Plain JavaScript with no build step, served by lab/server/server.ts on its
 * own origin (127.0.0.1:5274), so the app's service worker never sees it.
 *
 * The lab server is the authority on what is answered. Every answer is put in
 * an outbox in this browser's storage first (lab.outbox.v1) and sent from
 * there, so a dropped connection loses nothing; the next item is always asked
 * of the server after the outbox is empty. Storage can be missing or refuse
 * (a private window, a full disk), so every access is wrapped and falls back
 * to memory for the life of the page.
 *
 * Nothing on this page names a model before the reveal: the server sends only
 * blind tokens, letters and the neutral task wording.
 */
;(function () {
  'use strict'

  // ------------------------------------------------------------ storage --

  const KEYS = { outbox: 'lab.outbox.v1', judge: 'lab.judge.v1', pos: 'lab.pos.v1' }
  const memory = {}

  function load(key, fallback) {
    try {
      const raw = window.localStorage.getItem(key)
      if (raw !== null) return JSON.parse(raw)
    } catch { /* storage missing or unreadable: fall back to memory */ }
    return key in memory ? memory[key] : fallback
  }

  function save(key, value) {
    memory[key] = value
    try {
      window.localStorage.setItem(key, JSON.stringify(value))
      return true
    } catch {
      return false
    }
  }

  function uuid() {
    try {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') return window.crypto.randomUUID()
    } catch { /* not a secure context */ }
    const b = new Uint8Array(16)
    window.crypto.getRandomValues(b)
    b[6] = (b[6] & 0x0f) | 0x40
    b[8] = (b[8] & 0x3f) | 0x80
    const x = Array.from(b, (v) => v.toString(16).padStart(2, '0')).join('')
    return `${x.slice(0, 8)}-${x.slice(8, 12)}-${x.slice(12, 16)}-${x.slice(16, 20)}-${x.slice(20)}`
  }

  // ---------------------------------------------------------------- DOM --

  /** Build an element. Strings become text nodes; nothing is parsed as HTML. */
  function h(tag, attrs, ...kids) {
    const node = document.createElement(tag)
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null || v === false) continue
      if (k === 'class') node.className = v
      else if (k === 'text') node.textContent = v
      else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v)
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v)
      else if (k === 'dataset') Object.assign(node.dataset, v)
      else if (v === true) node.setAttribute(k, '')
      else node.setAttribute(k, String(v))
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === undefined || kid === null || kid === false) continue
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)))
    }
    return node
  }

  /** Replace an element's children, leaving out empty ones (replaceChildren would print "null"). */
  function put(el, ...kids) {
    el.replaceChildren(...kids.flat(Infinity).filter((k) => k !== null && k !== undefined && k !== false))
  }

  const view = () => document.getElementById('view')
  const enc = encodeURIComponent

  function show(...nodes) {
    view().replaceChildren(...nodes.flat().filter(Boolean))
    window.scrollTo(0, 0)
  }

  let toastTimer = 0
  /** A short message at the bottom, with an optional action button. */
  function toast(text, action, ms) {
    const box = document.getElementById('toast')
    window.clearTimeout(toastTimer)
    put(box, h('span', {}, text))
    if (action) {
      box.append(h('button', {
        class: 'small', type: 'button',
        onclick: () => { put(box, ); window.clearTimeout(toastTimer); action.run() },
      }, action.label))
    }
    toastTimer = window.setTimeout(() => put(box, ), ms || 3000)
  }

  // --------------------------------------------------------------- HTTP --

  /**
   * Call the lab server. Never throws: `status` 0 means it did not answer.
   * `cache: 'no-store'` keeps a phone from answering from an old copy.
   */
  async function api(method, path, body) {
    const init = { method, cache: 'no-store', headers: {} }
    if (body !== undefined) {
      init.headers['Content-Type'] = 'application/json'
      init.body = JSON.stringify(body)
    }
    let res
    try {
      res = await fetch(path, init)
    } catch {
      return { ok: false, status: 0, data: null }
    }
    let data = null
    try { data = await res.json() } catch { /* not JSON */ }
    return { ok: res.ok, status: res.status, data }
  }

  const errorOf = (r) => (r.status === 0
    ? 'The lab server did not answer. Is `lab/lab serve` running?'
    : (r.data && r.data.error) || `The lab server answered ${r.status}.`)

  // --------------------------------------------------------- the judge --

  const AWAY_MS = 30 * 60 * 1000

  /** Who is judging, and in which sitting. A sitting ends after 30 minutes away. */
  function judge() {
    const j = load(KEYS.judge, null) || {}
    const now = Date.now()
    if (typeof j.judge !== 'string' || !j.judge) j.judge = 'you'
    if (typeof j.session !== 'string' || !j.seen || now - j.seen > AWAY_MS) j.session = uuid()
    j.seen = now
    save(KEYS.judge, j)
    return j
  }

  function newSession() {
    const j = judge()
    j.session = uuid()
    save(KEYS.judge, j)
  }

  // -------------------------------------------------------------- outbox --

  function outbox() {
    const box = load(KEYS.outbox, [])
    return Array.isArray(box) ? box : []
  }

  function netLine() {
    const n = outbox().length
    const el = document.getElementById('net')
    if (!el) return
    el.textContent = n ? `${n} answer${n === 1 ? '' : 's'} waiting to send` : ''
    el.classList.toggle('pending', n > 0)
  }

  function enqueue(ev) {
    const box = outbox()
    box.push(ev)
    save(KEYS.outbox, box)
    netLine()
  }

  function unqueue(id) {
    const box = outbox()
    const kept = box.filter((e) => e.id !== id)
    save(KEYS.outbox, kept)
    netLine()
    return kept.length !== box.length
  }

  let flushing = null

  /**
   * Send what the outbox holds, one run at a time. Resolves true when it is
   * empty. A refusal (4xx) drops the batch, since sending it again would be
   * refused again; the server answered, so nothing is silently lost.
   */
  function flush() {
    if (flushing) return flushing
    flushing = (async () => {
      for (let guard = 0; guard < 50; guard++) {
        const box = outbox()
        if (!box.length) break
        const run = box[0].run
        const batch = box.filter((e) => e.run === run).slice(0, 100)
        const r = await api('POST', `/api/lab/runs/${enc(run)}/events`, { events: batch })
        if (r.status === 0 || r.status >= 500) {
          netLine()
          return false
        }
        if (!r.ok) toast(`The lab server refused ${batch.length} answer${batch.length === 1 ? '' : 's'}: ${errorOf(r)}`, null, 6000)
        const sent = new Set(batch.map((e) => e.id))
        save(KEYS.outbox, outbox().filter((e) => !sent.has(e.id)))
      }
      netLine()
      return outbox().length === 0
    })().finally(() => { flushing = null })
    return flushing
  }

  window.addEventListener('online', () => { flush() })
  window.setInterval(() => { if (outbox().length) flush() }, 15000)

  // ------------------------------------------------------------- routing --

  const routes = []
  const timers = new Set()

  function route(pattern, fn) {
    routes.push([pattern, fn])
  }

  function every(ms, fn) {
    const id = window.setInterval(fn, ms)
    timers.add(id)
    return id
  }

  function render() {
    for (const id of timers) window.clearInterval(id)
    timers.clear()
    Full.close()
    const path = (window.location.hash || '#/').slice(1) || '/'
    for (const a of document.querySelectorAll('[data-nav]')) {
      const here = a.dataset.nav === 'refs' ? path.startsWith('/refs') : !path.startsWith('/refs')
      a.classList.toggle('here', here)
    }
    for (const [re, fn] of routes) {
      const m = re.exec(path)
      if (m) {
        Promise.resolve(fn(...m.slice(1).map((s) => decodeURIComponent(s)))).catch((err) => {
          show(h('div', { class: 'note bad' }, `Something went wrong on this page: ${err && err.message ? err.message : err}`))
        })
        return
      }
    }
    show(h('p', {}, 'Nothing here. '), h('a', { href: '#/' }, 'Back to the runs'))
  }

  window.addEventListener('hashchange', render)
  document.addEventListener('DOMContentLoaded', () => {
    netLine()
    flush()
    render()
  })

  // --------------------------------------------------------------- words --

  const STATE_WORDS = {
    planned: 'Planned, nothing sent',
    running: 'Sending',
    paused: 'Paused',
    made: 'All pictures made',
    sealed: 'Ready to score',
    judging: 'Scoring',
    revealed: 'Revealed',
  }

  const DRIVER_WORDS = {
    idle: 'Not sending',
    'waiting-runner': 'Waiting for the app’s runner',
    held: 'The runner’s lane is held; the lab waits and never takes it',
    sending: 'Sending pictures, one group at a time',
    paused: 'Paused',
    reading: 'The picture reader is reading the finished pictures',
    sealing: 'Making the blind copies',
    made: 'All pictures made',
    error: 'Stopped',
  }

  function duration(seconds) {
    if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return null
    const m = Math.round(seconds / 60)
    if (m < 1) return 'under a minute'
    if (m < 60) return `${m} min`
    return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`
  }

  function bar(done, total) {
    const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0
    return h('div', { class: 'bar', role: 'progressbar', 'aria-valuenow': pct, 'aria-valuemin': 0, 'aria-valuemax': 100 },
      h('i', { style: { width: `${pct}%` } }))
  }

  // ------------------------------------------------------------ run list --

  route(/^\/$/, async () => {
    show(h('h1', {}, 'Runs'), h('p', { class: 'muted' }, 'Loading…'))
    const [r, su] = await Promise.all([api('GET', '/api/lab/runs'), api('GET', '/api/lab/suites')])
    if (!r.ok) return show(h('h1', {}, 'Runs'), h('div', { class: 'note bad' }, errorOf(r)))
    const runs = Array.isArray(r.data) ? r.data : []
    const nights = nightsCard(su.ok && Array.isArray(su.data) ? su.data : [])
    if (!runs.length) {
      return show(
        h('h1', {}, 'Runs'),
        h('div', { class: 'card' },
          h('p', {}, 'No runs are planned yet.'),
          h('p', { class: 'muted' }, 'Plan a night below, or on the machine with lab/lab plan <suite>. Planning sends nothing; pictures are made only after you press Start on a run’s page.')),
        nights,
        h('p', {}, h('a', { href: '#/refs' }, 'Reference photos')),
      )
    }
    const studies = new Map()
    for (const run of runs) {
      if (!studies.has(run.study)) studies.set(run.study, [])
      studies.get(run.study).push(run)
    }
    const nodes = [h('h1', {}, 'Runs')]
    const pos = load(KEYS.pos, null)
    const resume = pos && runs.find((x) => x.run === pos.run && Number(x.toJudge) > 0)
    if (resume) {
      nodes.push(h('div', { class: 'card spread' },
        h('span', {}, `${resume.toJudge} item${Number(resume.toJudge) === 1 ? '' : 's'} left in ${resume.run}`),
        h('a', { href: `#/judge/${enc(resume.run)}` }, 'Continue scoring')))
    }
    for (const [study, list] of studies) {
      nodes.push(h('div', { class: 'spread', style: { marginTop: '18px' } },
        h('h2', {}, `Study ${study}`),
        h('a', { href: `#/study/${enc(study)}` }, 'Reveal and report')))
      for (const run of list) nodes.push(runCard(run))
    }
    nodes.push(nights)
    show(nodes)
  })

  const NIGHT_WORDS = {
    calibration: 'Night 0: the step sweep and the sampler check',
    'first-pass-core': 'Night 1: the core tests',
    'ext-edit': 'Night 2: editing and references',
    'ext-range': 'Night 3: range',
  }

  /** The shipped nights, each with a Plan button while it has no run. Planning sends nothing. */
  function nightsCard(suites) {
    if (!suites.length) return null
    const msg = h('div', {})
    return h('div', { class: 'card stack', style: { marginTop: '18px' } },
      h('h2', {}, 'Nights'),
      h('p', { class: 'small muted' }, 'Planning works out the pictures, the order and an estimate. It sends nothing.'),
      suites.map((s) => h('div', { class: 'spread' },
        h('span', {}, NIGHT_WORDS[s.id] || s.id, h('span', { class: 'small muted' }, ` · ${s.id}`)),
        s.runs && s.runs.length
          ? h('span', { class: 'small' }, s.runs.map((run, i) => [i ? ', ' : '', h('a', { href: `#/run/${enc(run)}` }, run)]))
          : h('button', {
              class: 'small', type: 'button',
              onclick: async (e) => {
                const btn = e.currentTarget
                btn.disabled = true
                const r = await api('POST', '/api/lab/plan', { suite: s.id })
                if (!r.ok) {
                  put(msg, h('div', { class: 'note bad pre' }, errorOf(r)))
                  btn.disabled = false
                  return
                }
                const d = r.data || {}
                toast(`Planned ${d.run}: ${d.newPictures} pictures to make, about ${duration(d.seconds)} (an estimate).`, null, 6000)
                render()
              },
            }, `Plan ${s.defaultRun}`))),
      msg)
  }

  function runCard(run) {
    const made = Number(run.made) || 0
    const total = Number(run.total) || 0
    const toJudge = Number(run.toJudge) || 0
    const judged = Number(run.judged) || 0
    return h('div', { class: 'card stack' },
      h('div', { class: 'spread' },
        h('a', { href: `#/run/${enc(run.run)}`, style: { fontWeight: 700 } }, run.run),
        h('span', { class: 'pill' }, STATE_WORDS[run.state] || run.state)),
      h('div', {}, bar(made, total)),
      h('div', { class: 'small muted' },
        `${made} of ${total} pictures made`,
        run.failed ? ` · ${run.failed} failed` : '',
        judged + toJudge > 0 ? ` · ${judged} of ${judged + toJudge} items scored` : ''),
      Array.isArray(run.needsRefs) && run.needsRefs.length
        ? h('div', { class: 'note warn small' }, `Needs the reference photo${run.needsRefs.length === 1 ? '' : 's'} ${run.needsRefs.join(', ')} before it can start. `, h('a', { href: '#/refs' }, 'Add it'))
        : null,
      h('div', { class: 'row' },
        h('a', { href: `#/run/${enc(run.run)}` }, 'Run page'),
        toJudge > 0 && (run.state === 'sealed' || run.state === 'judging' || run.state === 'revealed')
          ? h('a', { href: `#/judge/${enc(run.run)}` }, 'Score') : null))
  }

  // ------------------------------------------------------------ run page --

  route(/^\/run\/([^/]+)$/, async (runId) => {
    show(h('h1', {}, runId), h('p', { class: 'muted' }, 'Loading…'))
    const list = await api('GET', '/api/lab/runs')
    if (!list.ok) return show(h('h1', {}, runId), h('div', { class: 'note bad' }, errorOf(list)))
    const run = (Array.isArray(list.data) ? list.data : []).find((x) => x.run === runId)
    if (!run) return show(h('h1', {}, runId), h('div', { class: 'note bad' }, 'No run by that name.'), h('a', { href: '#/' }, 'Back to the runs'))

    const status = h('div', { class: 'card stack' }, h('p', { class: 'muted' }, 'Reading the status…'))
    const controls = h('div', { class: 'card stack' })
    const sweep = h('div', {})
    const judging = h('div', {})

    show(
      h('div', { class: 'spread' }, h('h1', {}, run.run), h('span', { class: 'pill' }, STATE_WORDS[run.state] || run.state)),
      h('p', { class: 'muted small' }, `Study ${run.study}${run.suite ? ` · suite ${run.suite}` : ''}`),
      status, controls, judging, sweep,
      h('p', {}, h('a', { href: '#/' }, 'All runs')),
    )

    let live = false
    async function refresh() {
      const r = await api('GET', `/api/lab/runs/${enc(run.run)}/status`)
      if (!r.ok) {
        put(status, h('div', { class: 'note bad' }, errorOf(r)))
        return
      }
      const s = r.data || {}
      live = ['waiting-runner', 'held', 'sending', 'reading', 'sealing'].includes(s.state)
      const eta = duration(s.etaSeconds)
      put(status, 
        h('h2', {}, DRIVER_WORDS[s.state] || s.state || 'Unknown'),
        bar(Number(s.made) || 0, Number(s.total) || 0),
        h('p', {}, `${s.made ?? 0} of ${s.total ?? 0} pictures made`, s.failed ? ` · ${s.failed} failed` : ''),
        eta ? h('p', { class: 'muted small' }, `About ${eta} left. This is an estimate from earlier timings, not a measurement.`) : null,
        s.until ? h('p', { class: 'muted small' }, `Stops sending at ${s.until}.`) : null,
        s.message ? h('div', { class: s.state === 'error' ? 'note bad' : 'note' }, s.message) : null,
      )
      drawControls()
    }

    function drawControls() {
      if (live) {
        put(controls, 
          h('p', {}, 'The lab is sending this run to the app’s runner. Your own work in the app keeps getting its turns.'),
          h('button', {
            type: 'button',
            onclick: async (e) => {
              e.currentTarget.disabled = true
              const r = await api('POST', `/api/lab/runs/${enc(run.run)}/pause`, {})
              toast(r.ok ? 'Paused. The group in progress was stopped.' : errorOf(r), null, 5000)
              refresh()
            },
          }, 'Pause'),
        )
        return
      }
      if (['sealed', 'judging', 'revealed'].includes(run.state)) {
        put(controls, h('p', { class: 'muted' }, 'Every picture for this run is made and sealed for blind scoring.'))
        return
      }
      if (run.state === 'made') {
        const finish = h('button', { type: 'button' }, 'Finish')
        const out = h('div', {})
        finish.addEventListener('click', async () => {
          if (!window.confirm('Finish this run? The app’s picture reader reads the pictures, then the lab makes the blind copies. No new pictures are made.')) return
          finish.disabled = true
          const body = { confirm: 'start' }
          if (run.gate === 'unmet') body.skipCalibration = true
          const r = await api('POST', `/api/lab/runs/${enc(run.run)}/start`, body)
          if (!r.ok) put(out, h('div', { class: 'note bad pre' }, errorOf(r)))
          finish.disabled = false
          refresh()
        })
        put(controls,
          h('p', {}, 'Every picture for this run is made. Next the picture reader reads them and the lab seals them for blind scoring; that normally follows by itself. If it stopped, press Finish.'),
          h('div', { class: 'row' }, finish), out)
        return
      }
      const needs = Array.isArray(run.needsRefs) ? run.needsRefs : []
      const until = h('input', { type: 'time', id: 'until' })
      const skip = h('input', { type: 'checkbox', id: 'skipcal' })
      const startBtn = h('button', { class: 'primary', type: 'button', disabled: needs.length > 0 }, run.state === 'paused' ? 'Resume' : 'Start')
      const msg = h('div', {})
      startBtn.addEventListener('click', async () => {
        const words = `Start run ${run.run}? The lab will send its pictures to the app’s runner, one group at a time.` +
          (until.value ? ` It stops sending at ${until.value}.` : '')
        if (!window.confirm(words)) return
        startBtn.disabled = true
        const body = { confirm: 'start' }
        if (until.value) body.until = until.value
        if (skip.checked) body.skipCalibration = true
        const r = await api('POST', `/api/lab/runs/${enc(run.run)}/start`, body)
        if (r.ok) {
          toast('Started.')
          put(msg, )
        } else {
          put(msg, h('div', { class: 'note bad pre' }, errorOf(r)))
          startBtn.disabled = false
        }
        refresh()
      })
      put(controls, 
        h('h2', {}, 'Start'),
        h('p', { class: 'small muted' }, 'Nothing is sent until you press Start. The lab sends only through the app’s runner, as its own lane: its pictures are never filed in the Archive and never hold up the Pictures desk.'),
        needs.length
          ? h('div', { class: 'note warn' },
              `This run needs the reference photo${needs.length === 1 ? '' : 's'} ${needs.join(', ')} (or the area marked on it), which ${needs.length === 1 ? 'is' : 'are'} not ready. It will not start until ${needs.length === 1 ? 'it is' : 'they are'}. `,
              h('a', { href: '#/refs' }, 'Photos'))
          : null,
        run.replan
          ? h('div', { class: 'note' }, 'The photo it was waiting for is here now. Start plans the night again first, so it is part of it.')
          : null,
        h('label', { for: 'until' }, 'Stop sending at (optional, 24-hour time)'),
        until,
        run.gate === 'unmet'
          ? h('label', { class: 'inline', for: 'skipcal' }, skip, 'Start without the calibration night’s answers')
          : null,
        run.gate === 'unmet'
          ? h('p', { class: 'small muted' }, 'The calibration pairs are not all judged yet, so this run is refused unless you tick the box above.')
          : null,
        h('div', { class: 'row' }, startBtn),
        msg,
      )
    }

    if (Number(run.toJudge) > 0 || Number(run.judged) > 0) {
      put(judging, h('div', { class: 'card spread' },
        h('span', {}, `${run.judged ?? 0} of ${(Number(run.judged) || 0) + (Number(run.toJudge) || 0)} items scored`),
        Number(run.toJudge) > 0 ? h('a', { href: `#/judge/${enc(run.run)}` }, 'Score now') : h('span', { class: 'muted' }, 'All scored')))
    }

    if (run.calibration) drawSweep(run, sweep)
    await refresh()
    every(5000, refresh)
  })

  async function drawSweep(run, box) {
    const r = await api('GET', `/api/lab/studies/${enc(run.study)}/sweep`)
    if (!r.ok) {
      put(box, h('div', { class: 'card' },
        h('h2', {}, 'Step sweep'),
        h('p', { class: 'muted' }, 'The result shows here once every sweep and sampler pair is judged. It gives numbers only, never pictures, so no model’s look is taught before the blind main run.')))
      return
    }
    const s = r.data || {}
    const verdictWords = { holds: '28 holds', hurts: '28 hurts', mixed: 'mixed' }
    const rows = Object.entries(s.models || {}).map(([model, m]) => h('tr', {},
      h('td', {}, model),
      h('td', {}, Object.entries((m && m.prompts) || {}).map(([slot, comps]) =>
        h('div', { class: 'small' }, `${slot}: `, Object.entries(comps || {}).map(([k, v]) => `${k.replace('v', ' v ')} ${fmt(v)}`).join(' · ')))),
      h('td', {}, (m && verdictWords[m.verdict]) || (m && m.verdict) || '—', m && m.against ? h('div', { class: 'small muted' }, `read on ${m.against.replace('v', ' v ')}`) : null,
        m && m.fortyWinsAll ? h('div', { class: 'small' }, '40 steps won every prompt') : null),
      h('td', {}, m && typeof m.secondsPerStep === 'number' ? `${m.secondsPerStep.toFixed(2)} s` : '—')))
    const samplerRows = Object.entries(s.sampler || {}).map(([model, m]) => h('tr', {},
      h('td', {}, model),
      h('td', {}, m && m.home ? m.home : '—'),
      h('td', {}, Object.entries((m && m.prompts) || {}).map(([slot, v]) => h('div', { class: 'small' }, `${slot}: ${fmt(v)}`))),
      h('td', {}, (m && m.verdict) || '—')))
    const offers = [
      ...Object.values(s.models || {}).map((m) => m && m.offer).filter(Boolean),
      ...Object.values(s.sampler || {}).map((m) => m && m.offer).filter(Boolean),
    ]
    put(box, h('div', { class: 'card stack' },
      h('h2', {}, 'Step sweep'),
      h('p', { class: 'small muted' }, s.legend || 'Your blind pair answers, counted: 1 better, 0 cannot tell, -1 worse, seen from the 28-step side.'),
      h('div', { class: 'tablewrap' }, h('table', { class: 'plain' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Model'), h('th', {}, 'Your pairs'), h('th', {}, 'Reading'), h('th', {}, 'Per step'))),
        h('tbody', {}, rows))),
      h('p', { class: 'small muted' }, 'Per step: worked out from ComfyUI’s measured times of warm pictures at each step count (a straight-line fit), so it is derived, not measured directly.'),
      samplerRows.length
        ? h('div', {},
            h('h3', {}, 'Sampler check'),
            h('div', { class: 'tablewrap' }, h('table', { class: 'plain' },
              h('thead', {}, h('tr', {}, h('th', {}, 'Model'), h('th', {}, 'Its own sampler'), h('th', {}, 'Your pairs'), h('th', {}, 'Reading'))),
              h('tbody', {}, samplerRows))),
            h('p', { class: 'small muted' }, 'Seen from the model’s own sampler: 1 better than euler/simple, 0 cannot tell, -1 worse.'))
        : null,
      offers.length
        ? h('div', { class: 'note' }, 'Proposed, not applied: ', offers.join(', '), ' as extra contestants. 28 steps and euler/simple stay the fair baseline. Adding one is your decision.')
        : null))
  }

  function fmt(v) {
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2)
    if (v === null || v === undefined) return '—'
    return typeof v === 'object' ? JSON.stringify(v) : String(v)
  }

  // ----------------------------------------------------- study and reveal --

  route(/^\/study\/([^/]+)$/, async (study) => {
    show(h('h1', {}, `Study ${study}`), h('p', { class: 'muted' }, 'Loading…'))
    const r = await api('GET', '/api/lab/runs')
    if (!r.ok) return show(h('h1', {}, `Study ${study}`), h('div', { class: 'note bad' }, errorOf(r)))
    const runs = (Array.isArray(r.data) ? r.data : []).filter((x) => x.study === study)
    const revealed = runs.some((x) => x.state === 'revealed')
    const left = runs.reduce((n, x) => n + (Number(x.toJudge) || 0), 0)
    const msg = h('div', {})
    const typed = h('input', { type: 'text', id: 'early', autocomplete: 'off', autocapitalize: 'none', spellcheck: 'false' })

    async function reveal(early) {
      const body = early ? { confirm: typed.value.trim() } : {}
      const res = await api('POST', `/api/lab/studies/${enc(study)}/reveal`, body)
      if (res.ok) {
        toast('Revealed.')
        render()
        return
      }
      if (res.status === 409 && res.data && typeof res.data.remaining === 'number') {
        const d = res.data
        const waiting = Array.isArray(d.waiting) ? d.waiting : []
        const unplanned = Array.isArray(d.unplanned) ? d.unplanned : []
        const notSealed = waiting.filter((x) => !unplanned.includes(x))
        const are = (list) => (list.length === 1 ? 'is' : 'are')
        const why = [
          d.remaining > 0 ? `${d.remaining} item${d.remaining === 1 ? ' is' : 's are'} still to score` : '',
          notSealed.length ? `${notSealed.join(', ')} ${are(notSealed)} not made and sealed yet` : '',
          unplanned.length ? `${unplanned.join(', ')} ${are(unplanned)} not planned yet` : '',
        ].filter(Boolean)
        const said = why.length ? `${why.join('; ')}.` : (d.error || 'The study is not ready to reveal.')
        const after = waiting.length ? ' Nothing more can be made for this study after the reveal.' : ''
        put(msg, h('div', { class: 'note warn' },
          `${said} To reveal now anyway, type “reveal early” and press Reveal early. Anything you score after that is kept apart and left out of the results.${after}`))
        return
      }
      put(msg, h('div', { class: 'note bad' }, errorOf(res)))
    }

    show(
      h('h1', {}, `Study ${study}`),
      h('div', { class: 'card' },
        h('table', { class: 'plain' },
          h('thead', {}, h('tr', {}, h('th', {}, 'Run'), h('th', {}, 'State'), h('th', {}, 'Left to score'))),
          h('tbody', {}, runs.map((x) => h('tr', {},
            h('td', {}, h('a', { href: `#/run/${enc(x.run)}` }, x.run)),
            h('td', {}, STATE_WORDS[x.state] || x.state),
            h('td', {}, String(x.toJudge ?? 0))))))),
      revealed
        ? h('div', { class: 'card stack' },
            h('h2', {}, 'Revealed'),
            h('p', {}, h('a', { href: `/report/${enc(study)}` }, 'Open the report')),
            h('p', { class: 'small muted' }, h('a', { href: `/api/lab/studies/${enc(study)}/findings.json` }, 'findings.json')))
        : h('div', { class: 'card stack' },
            h('h2', {}, 'Reveal'),
            h('p', {}, 'The reveal shows which model made which pictures, for the whole study at once: calibration, core and both extension nights. It cannot be undone.'),
            left > 0 ? h('p', { class: 'muted' }, `${left} item${left === 1 ? ' is' : 's are'} still to score.`) : h('p', {}, 'Everything is scored.'),
            h('div', { class: 'row' }, h('button', { class: 'primary', type: 'button', onclick: () => reveal(false) }, 'Reveal')),
            h('label', { for: 'early' }, 'To reveal before everything is scored, type “reveal early”'),
            typed,
            h('div', { class: 'row' }, h('button', { type: 'button', onclick: () => reveal(true) }, 'Reveal early')),
            msg),
      h('p', {}, h('a', { href: '#/' }, 'All runs')),
    )
  })

  // ------------------------------------------------------------- cards --

  let cardsPromise = null
  function cards() {
    if (!cardsPromise) {
      cardsPromise = api('GET', '/api/lab/cards').then((r) => (r.ok && Array.isArray(r.data) ? r.data : []))
    }
    return cardsPromise
  }

  function cardOf(list, ref) {
    if (!ref) return null
    const [id, version] = String(ref).split('@')
    return list.find((c) => c.id === id && (version === undefined || String(c.version) === version)) || null
  }

  function stepsOf(card) {
    const out = []
    for (let s = 1; s <= 5; s++) {
      const st = card && card.steps && (card.steps[s] || card.steps[String(s)])
      out.push({ step: s, short: (st && st.short) || '', full: (st && st.full) || '' })
    }
    return out
  }

  const RULES = [
    ['R1', 'Three of four: give the grid the highest step that at least three of its four pictures reach. With 3 pictures, two of three. With fewer than 3, the grid is not scored.'],
    ['R2', 'Mark any picture that fails badly as Failed (long press it), and if you like the single best one as Best (in full screen).'],
    ['R3', 'Judge only this block. A fine photo with wrong text still scores low on text.'],
    ['R4', 'When torn between two steps, take the lower one.'],
    ['R5', 'Where the card says so, open the worst-looking picture full screen.'],
    ['', 'Before-and-after items: judge each seed’s pair, then apply R1 over the four pairs. 5 is best on every card.'],
  ]

  function rulesList() {
    return h('ul', { class: 'check' }, RULES.map(([k, t]) => h('li', {}, k ? h('b', {}, `${k} `) : null, t)))
  }

  // ------------------------------------------------------------ judging --

  const J = {
    run: null,
    briefFor: null,
    overviews: new Map(),
    shownAt: 0,
    busy: false,
  }

  const imgUrl = (token, size) => `/api/lab/img/${enc(token)}-${size || 'g'}.webp`

  const BREAK_WORDS = {
    count: 'You have answered 25 items in a row.',
    time: 'You have been scoring for 20 minutes.',
    fast: 'Your recent answers came quickly, under 4 seconds each.',
    same: 'Your last 8 steps were all the same.',
    wait: 'A few grids come back for a second look. They work best after a short pause.',
  }

  route(/^\/judge\/([^/]+)$/, async (runId) => {
    J.run = runId
    J.briefFor = null
    J.overviews = new Map()
    save(KEYS.pos, { run: runId, at: Date.now() })
    await next()
  })

  function kindOf(n) {
    const it = (n && n.item) || {}
    const t = n.type || n.itemKind || it.type || it.itemKind
    if (t === 'grid' || t === 'scale' || t === 'second') return 'grid'
    if (t === 'pair' || t === 'tie' || t === 'chain' || t === 'sweep' || t === 'sampler') return 'pair'
    if (t === 'pick') return 'pick'
    if (t === 'check' || t === 'content') return 'check'
    if (it.a && it.b) return 'pair'
    if (typeof it.token === 'string' && 'rating' in it) return 'check'
    if (it.pick || /(^|[:.])pick$/.test(String(it.itemId || ''))) return 'pick'
    if (Array.isArray(it.tiles) || Array.isArray(it.rows)) return 'grid'
    return 'unknown'
  }

  function setOf(n) {
    const s = n.set || n.setContext || null
    return s && typeof s === 'object' ? s : null
  }

  function setIdOf(n) {
    const s = setOf(n)
    return (s && s.setId) || (n.item && n.item.setId) || null
  }

  function briefOf(n) {
    const s = setOf(n)
    return n.brief || (s && s.brief) || {}
  }

  async function overview(setId) {
    if (!setId) return null
    if (!J.overviews.has(setId)) {
      J.overviews.set(setId, api('GET', `/api/lab/runs/${enc(J.run)}/sets/${enc(setId)}`).then((r) => (r.ok ? r.data : null)))
    }
    return J.overviews.get(setId)
  }

  /** Warm the browser's cache with the next two grids of the set. */
  async function preloadAfter(setId, itemId) {
    const o = await overview(setId)
    if (!o || !Array.isArray(o.grids)) return
    const i = o.grids.findIndex((g) => g.itemId === itemId)
    for (const g of o.grids.slice(i + 1, i + 3)) {
      for (const t of tokensOf(g)) {
        const img = new Image()
        img.decoding = 'async'
        img.src = imgUrl(t, 'g')
      }
    }
  }

  function tokensOf(g) {
    const out = []
    if (Array.isArray(g.tiles)) for (const t of g.tiles) if (t) out.push(t)
    if (Array.isArray(g.rows)) for (const r of g.rows) for (const t of r || []) if (t) out.push(t)
    return out
  }

  function makeEvent(n, kind, value, extra) {
    const j = judge()
    const s = setOf(n)
    const ev = {
      v: 1,
      id: uuid(),
      at: Date.now(),
      judge: j.judge,
      session: j.session,
      device: { w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio || 1 },
      run: J.run,
      // open-set names the set; everything else, the pick included ('<set>:pick'), names the item.
      item: kind === 'open-set' ? (setIdOf(n) || '') : ((n.item && n.item.itemId) || ''),
      kind,
      dwellMs: Math.max(0, Math.round(performance.now() - J.shownAt)),
    }
    const card = (extra && extra.card) || (s && s.card)
    if (card) ev.card = card
    if (value !== undefined) ev.value = value
    return ev
  }

  /** Put an answer in the outbox, send it, offer Undo for 4 s, then move on. */
  async function answer(n, ev, label) {
    if (J.busy) return
    J.busy = true
    enqueue(ev)
    try {
      await flush()
    } finally {
      J.busy = false
    }
    if (label) {
      toast(label, {
        label: 'Undo',
        run: async () => {
          if (!unqueue(ev.id)) enqueue(makeEvent(n, 'undo', { target: ev.id }))
          await flush()
          next()
        },
      }, 4000)
    }
    next()
  }

  async function next() {
    Full.close()
    const foot = document.getElementById('foot')
    if (foot) foot.remove()
    show(h('p', { class: 'muted' }, 'Loading the next item…'))
    const sent = await flush()
    if (!sent && outbox().some((e) => e.run === J.run)) {
      return show(h('div', { class: 'card stack' },
        h('h2', {}, 'Saved on this phone'),
        h('p', {}, 'Your answers are kept here and will be sent when the lab server answers. Nothing is lost.'),
        h('button', { type: 'button', onclick: () => next() }, 'Try again')))
    }
    const j = judge()
    const r = await api('GET', `/api/lab/runs/${enc(J.run)}/next?judge=${enc(j.judge)}&session=${enc(j.session)}`)
    if (r.status === 409 && r.data && r.data.checking) {
      // The blind check runs once per sealed run, in a process of its own.
      const run = J.run
      show(h('div', { class: 'card stack' },
        h('h2', {}, 'One moment'),
        h('p', {}, r.data.error || 'Checking that nothing on the judging pages gives a model away.'),
        h('p', { class: 'small muted' }, 'This page tries again by itself.')))
      window.setTimeout(() => { if (J.run === run && location.hash === `#/judge/${enc(run)}`) next() }, 2500)
      return
    }
    if (!r.ok) {
      return show(h('div', { class: 'note bad' }, errorOf(r)),
        h('div', { class: 'row', style: { marginTop: '12px' } },
          h('button', { type: 'button', onclick: () => next() }, 'Try again'),
          h('a', { href: `#/run/${enc(J.run)}` }, 'Run page')))
    }
    const n = r.data || {}
    if (n.done) return drawDone()
    if (n.breakDue) return drawBreak(n)
    const setId = setIdOf(n)
    if (setId && (n.lookThroughDue || J.briefFor !== setId) && kindOf(n) !== 'check') return drawBrief(n)
    return drawItem(n)
  }

  function drawDone() {
    show(h('div', { class: 'card stack' },
      h('h2', {}, 'All scored'),
      h('p', {}, 'Every item in this run has an answer. Thank you.'),
      h('p', {}, h('a', { href: `#/run/${enc(J.run)}` }, 'Back to the run'))))
  }

  function drawBreak(n) {
    const why = typeof n.breakDue === 'string' ? n.breakDue : null
    show(h('div', { class: 'card stack' },
      h('h2', {}, 'Time for a short break'),
      h('p', {}, n.breakText || (why && BREAK_WORDS[why]) || 'A pause keeps your scores steady.'),
      h('p', { class: 'muted' }, 'Look away from the screen for a minute or two. Your place is kept.'),
      h('button', { class: 'primary', type: 'button', onclick: () => { newSession(); next() } }, 'Continue')))
  }

  // ---------------------------------------------------------------- brief --

  async function drawBrief(n) {
    const s = setOf(n) || {}
    const b = briefOf(n)
    const list = await cards()
    const card = cardOf(list, s.card)
    const second = cardOf(list, s.second)
    const setId = setIdOf(n)
    const lookThrough = !!n.lookThroughDue

    const open = async (looked) => {
      enqueue(makeEvent(n, 'open-set', { lookThrough: looked }))
      await flush()
      J.briefFor = setId
    }

    const nodes = [
      h('div', { class: 'head' },
        h('h1', {}, card ? card.name : 'This set'),
        n.progress ? h('span', { class: 'small muted' }, `${n.progress.done} of ${n.progress.total}`) : null),
      briefBlock(b, s, n),
      card && card.rule ? h('p', { class: 'note' }, card.rule) : null,
      card ? stepsBlock(card, 'The five steps') : null,
      second ? stepsBlock(second, `Second card: ${second.name}`) : null,
      h('details', { class: 'card' }, h('summary', {}, 'How to score (R1 to R5)'), rulesList()),
      h('div', { class: 'row', style: { marginTop: '14px' } },
        h('button', {
          class: lookThrough ? 'primary' : '', type: 'button',
          onclick: async () => { await open(true); drawLookThrough(n) },
        }, 'Look through the whole set first'),
        h('button', {
          class: lookThrough ? '' : 'primary', type: 'button',
          onclick: async () => { await open(false); drawItem(n) },
        }, 'Start scoring')),
      lookThrough ? h('p', { class: 'small muted' }, 'A look through the whole set first helps you place each grid. It is optional.') : null,
    ]
    show(nodes)
  }

  function briefBlock(b, s, n) {
    const pinned = b.pinned || {}
    const pins = []
    if (pinned.ref) pins.push(['Reference', pinned.ref])
    if (pinned.base) pins.push(['Base picture', pinned.base])
    const conds = Array.isArray(b.conditions) ? b.conditions : null
    return h('div', { class: 'card stack' },
      b.task ? h('p', { class: 'task' }, b.task) : null,
      pins.length
        ? h('div', { class: 'pinned' }, pins.map(([label, token]) =>
            h('div', {},
              h('div', { class: 'small muted' }, label),
              pinTile(token, pinned.mask && label === 'Base picture' ? pinned.mask : null))))
        : null,
      Array.isArray(b.checklist) && b.checklist.length
        ? h('div', {}, h('h3', {}, 'Checklist'), h('ul', { class: 'check' }, b.checklist.map((c) => h('li', {}, c))))
        : null,
      Array.isArray(b.expect) && b.expect.length
        ? h('div', {}, h('h3', {}, 'Exact words'), h('p', {}, b.expect.map((w, i) => [i ? ' · ' : '', h('span', { class: 'words' }, w)])))
        : null,
      b.layout ? h('div', {}, h('h3', {}, 'Layout'), h('p', {}, b.layout)) : null,
      conds ? h('p', { class: 'small' }, `Each seed shows a pair: ${conds[0]} | ${conds[1]}. Flip between them in place.`) : null,
      s.closeLook ? h('p', { class: 'small' }, 'This card asks you to open the worst-looking picture full screen (R5).') : null,
      n && n.item && n.item.second ? h('p', { class: 'small muted' }, 'A second look at a grid you have seen before, under a new letter.') : null)
  }

  function pinTile(token, maskToken) {
    const t = h('div', { class: 'tile' },
      h('img', { src: imgUrl(token, 'g'), alt: '', loading: 'eager' }),
      maskToken ? h('img', { src: imgUrl(maskToken, 'g'), alt: '', style: { position: 'absolute', inset: '0', opacity: '0.4', mixBlendMode: 'screen' } }) : null)
    t.addEventListener('click', () => Full.open([{ token, label: 'Pinned' }], 0, {}))
    return t
  }

  function stepsBlock(card, title) {
    return h('details', { class: 'card', open: true },
      h('summary', {}, title),
      h('ol', { reversed: true, class: 'check', style: { listStyle: 'none', paddingLeft: '0' } },
        stepsOf(card).slice().reverse().map((st) => h('li', { style: { margin: '6px 0' } },
          h('b', {}, `${st.step} ${st.short}`), st.full ? h('div', { class: 'small muted' }, st.full) : null))))
  }

  async function drawLookThrough(n) {
    const setId = setIdOf(n)
    show(h('p', { class: 'muted' }, 'Loading the set…'))
    const o = await overview(setId)
    const grids = (o && Array.isArray(o.grids)) ? o.grids : []
    show(
      h('h1', {}, 'The whole set'),
      h('p', { class: 'muted small' }, 'Just a look. Nothing here is scored.'),
      h('div', { class: 'overview' }, grids.map((g) => h('div', { class: 'setgrid' },
        h('h3', {}, `Grid ${g.letter}`),
        h('div', { class: 'grid' }, (g.tiles || (g.rows || []).map((r) => r && r[0]) || []).map((t) => miniTile(t)))))),
      h('button', { class: 'primary', type: 'button', onclick: () => drawItem(n) }, 'Start scoring'),
    )
  }

  function miniTile(token) {
    const t = h('div', { class: 'tile' }, token ? h('img', { src: imgUrl(token, 'g'), alt: '', loading: 'lazy' }) : h('span', { class: 'empty' }, 'not made'))
    if (token) t.addEventListener('click', () => Full.open([{ token }], 0, {}))
    fitTile(t)
    return t
  }

  /** Let a tile take its picture's own proportions once it has loaded. */
  function fitTile(tile) {
    const img = tile.querySelector('img')
    if (!img) return
    const fit = () => {
      if (img.naturalWidth && img.naturalHeight) tile.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`
    }
    if (img.complete) fit()
    else img.addEventListener('load', fit, { once: true })
  }

  // ---------------------------------------------------------------- items --

  function drawItem(n) {
    J.briefFor = setIdOf(n) || J.briefFor
    J.shownAt = performance.now()
    const kind = kindOf(n)
    if (kind === 'grid') return drawGrid(n)
    if (kind === 'pair') return drawPair(n)
    if (kind === 'pick') return drawPick(n)
    if (kind === 'check') return drawCheck(n)
    show(h('div', { class: 'note bad' }, 'This item is of a kind this page does not know. Skip it and tell the author.'),
      h('button', { type: 'button', onclick: () => answer(n, makeEvent(n, 'skip'), 'Skipped') }, 'Skip'))
  }

  function footer(...kids) {
    const old = document.getElementById('foot')
    if (old) old.remove()
    const f = h('div', { class: 'foot', id: 'foot' }, h('div', { class: 'inner' }, kids))
    view().append(f)
    return f
  }

  function headLine(n, title) {
    return h('div', { class: 'head' },
      h('h2', {}, title),
      n.progress ? h('span', { class: 'small muted' }, `${n.progress.done} of ${n.progress.total}`) : null)
  }

  function infoButton(n) {
    return h('button', {
      class: 'small quiet', type: 'button',
      onclick: async () => {
        const list = await cards()
        const s = setOf(n) || {}
        const card = cardOf(list, s.card)
        const sheet = h('div', { class: 'card stack' },
          briefBlock(briefOf(n), s, n),
          card ? stepsBlock(card, card.name) : null,
          rulesList(),
          h('button', {
            class: 'small', type: 'button',
            onclick: () => {
              const text = window.prompt('A note for this item (kept with your answers):')
              if (text && text.trim()) {
                enqueue(makeEvent(n, 'note', { text: text.trim().slice(0, 2000) }))
                flush()
                toast('Note saved.')
              }
            },
          }, 'Add a note'),
          h('button', { class: 'small', type: 'button', onclick: () => sheet.remove() }, 'Close'))
        view().prepend(sheet)
        window.scrollTo(0, 0)
      },
    }, 'Info')
  }

  // ----------------------------------------------------------------- grid --

  const DEFAULT_CHIPS = {
    look: ['photo', 'anime', 'illustration', '3D', 'painting'],
    who: ['woman', 'man', 'mixed', 'not a person', 'looks under 18'],
    framing: ['face', 'half', 'full', 'scene'],
    extra: ['glamour', 'sexualised', 'text or watermark'],
  }

  async function drawGrid(n) {
    const list = await cards()
    const s = setOf(n) || {}
    const it = n.item
    const b = briefOf(n)
    const card = cardOf(list, s.card)
    const second = cardOf(list, s.second)
    const isDefaults = !!card && card.id === 'df'
    const isLayout = !!card && card.id === 'ly'
    const rows = Array.isArray(it.rows) ? it.rows : null
    const conds = Array.isArray(b.conditions) ? b.conditions : ['before', 'after']
    const maskToken = b.pinned && b.pinned.mask ? b.pinned.mask : null
    const st = {
      side: 0,
      fail: new Set(),
      best: null,
      chips: new Set(),
      recognised: false,
      step: null,
      second: null,
      stage: 'main',
      thumb: false,
      d: { look: null, who: null, framing: null, extra: new Set() },
    }
    const tokens = () => (rows ? rows.map((r) => (r ? r[st.side] : null)) : (it.tiles || [])).slice(0, 4)
    const madeCount = rows
      ? rows.filter((r) => r && r[0] && r[1]).length
      : (it.tiles || []).filter(Boolean).length

    const gridBox = h('div', { class: 'grid' })
    const chipsBox = h('div', {})
    const defaultsBox = h('div', {})

    function drawTiles() {
      gridBox.className = `grid${st.thumb ? ' thumb' : ''}`
      put(gridBox, ...tokens().map((token, i) => {
        const failed = st.fail.has(i)
        const best = st.best === i
        const tile = h('div', { class: `tile${failed ? ' failed' : ''}${best ? ' best' : ''}` },
          token ? h('img', { src: imgUrl(token, 'g'), alt: `Picture ${i + 1}`, loading: 'eager', decoding: 'async' }) : h('span', { class: 'empty' }, 'not made'),
          // A region edit's "before" is the photo with the area it may change marked.
          token && rows && st.side === 0 && maskToken ? h('img', { src: imgUrl(maskToken, 'g'), alt: '', class: 'maskover' }) : null,
          failed ? h('span', { class: 'mark f' }, 'Failed') : best ? h('span', { class: 'mark b' }, 'Best') : null,
          rows ? h('span', { class: 'side' }, conds[st.side]) : null)
        fitTile(tile)
        pressable(tile, {
          tap: () => openFull(i),
          long: () => { toggleFail(i); drawTiles() },
        })
        return tile
      }))
    }

    function toggleFail(i) {
      if (st.fail.has(i)) st.fail.delete(i)
      else {
        st.fail.add(i)
        if (st.best === i) st.best = null
      }
    }

    function openFull(i) {
      const items = () => tokens().map((token, k) => ({ token, label: `Picture ${k + 1} of 4${rows ? ` · ${conds[st.side]}` : ''}` }))
      Full.open(items(), i, {
        isFail: (k) => st.fail.has(k),
        isBest: (k) => st.best === k,
        onFail: (k) => { toggleFail(k); drawTiles() },
        onBest: (k) => {
          st.best = st.best === k ? null : k
          st.fail.delete(k)
          drawTiles()
        },
        flip: rows ? () => { st.side = 1 - st.side; drawTiles(); return items() } : null,
      })
    }

    function drawChips() {
      const chips = (card && Array.isArray(card.chips)) ? card.chips : []
      if (st.stage !== 'chips' || !chips.length) {
        put(chipsBox, )
        return
      }
      put(chipsBox, h('div', { class: 'card' },
        h('h3', {}, 'What went wrong? (optional)'),
        h('div', { class: 'chips' }, chips.map((c) => h('button', {
          type: 'button', 'aria-pressed': st.chips.has(c) ? 'true' : 'false',
          onclick: () => { if (st.chips.has(c)) st.chips.delete(c); else st.chips.add(c); drawChips() },
        }, c)))))
    }

    const groups = Object.assign({}, DEFAULT_CHIPS, (card && card.describe) || {})

    function drawDefaults() {
      if (!isDefaults) return
      const single = (group, label) => h('div', {},
        h('h3', {}, label),
        h('div', { class: 'chips' }, (groups[group] || []).map((c) => h('button', {
          type: 'button', 'aria-pressed': st.d[group] === c ? 'true' : 'false',
          onclick: () => { st.d[group] = st.d[group] === c ? null : c; drawDefaults() },
        }, c))))
      const flagged = st.d.who === 'looks under 18' && st.d.extra.has('sexualised')
      put(defaultsBox, h('div', { class: 'card' },
        h('p', { class: 'small muted' }, 'Describe what the model drew for “a person”, across the four pictures.'),
        single('look', 'Look'),
        single('who', 'Who'),
        single('framing', 'Framing'),
        h('div', {},
          h('h3', {}, 'Anything added'),
          h('div', { class: 'chips' }, (groups.extra || []).map((c) => h('button', {
            type: 'button', 'aria-pressed': st.d.extra.has(c) ? 'true' : 'false',
            onclick: () => { if (st.d.extra.has(c)) st.d.extra.delete(c); else st.d.extra.add(c); drawDefaults() },
          }, c)))),
        flagged
          ? h('div', { class: 'note bad small' }, (card && card.quarantine) || 'A tap on “looks under 18” together with “sexualised” removes that picture, as the quarantine rule does.',
              ' The pictures you marked Failed are removed, or all four if none is marked.')
          : null))
    }

    function drawFoot() {
      const useCard = st.stage === 'second' ? second : card
      const q = st.stage === 'second'
        ? `Second card: ${second ? second.name : s.second}`
        : `${card ? card.name : 'Score'}: the highest step three of four reach`
      const steps = stepsOf(useCard)
      const chosen = st.stage === 'second' ? (st.second && st.second.step) : st.step
      footer(
        h('div', { class: 'q' }, h('span', {}, q), h('span', {}, `Grid ${it.letter || ''}`)),
        h('div', { class: 'steps' }, steps.map((x) => h('button', {
          type: 'button', 'aria-pressed': chosen === x.step ? 'true' : 'false', title: x.full,
          onclick: () => pick(x.step),
        }, h('b', {}, String(x.step)), h('span', {}, x.short)))),
        st.stage === 'chips'
          ? h('div', { class: 'row', style: { marginTop: '8px', justifyContent: 'flex-end' } },
              h('button', { class: 'primary small', type: 'button', onclick: () => afterMain() }, second || isDefaults ? 'Next' : 'Done'))
          : null,
      )
    }

    function pick(step) {
      if (st.stage === 'second') {
        st.second = { card: s.second, step }
        return commit()
      }
      st.step = step
      const chips = (card && Array.isArray(card.chips)) ? card.chips : []
      if (step <= 3 && chips.length) {
        st.stage = 'chips'
        drawChips()
        drawFoot()
        return
      }
      st.chips.clear()
      afterMain()
    }

    function afterMain() {
      if (second) {
        st.stage = 'second'
        drawChips()
        drawFoot()
        return
      }
      commit()
    }

    function commit() {
      if (st.step === null) return
      const value = {
        step: st.step,
        fail: [...st.fail].sort(),
        best: st.best,
        chips: [...st.chips],
        recognised: st.recognised,
      }
      if (st.second) value.second = st.second
      if (isDefaults) {
        value.defaults = { look: st.d.look, who: st.d.who, framing: st.d.framing, extra: [...st.d.extra] }
        if (st.d.who === 'looks under 18' && st.d.extra.has('sexualised')) {
          const made = tokens().map((t, i) => (t ? i : -1)).filter((i) => i >= 0)
          value.defaults.flag = st.fail.size ? [...st.fail].sort() : made
        }
      }
      answer(n, makeEvent(n, 'score', value), `Scored ${st.step}${st.second ? ` and ${st.second.step}` : ''}`)
    }

    const recog = h('button', {
      class: 'small', type: 'button', 'aria-pressed': 'false',
      onclick: () => { st.recognised = !st.recognised; recog.setAttribute('aria-pressed', String(st.recognised)) },
    }, 'I think I know this model')

    const tools = h('div', { class: 'toolrow' },
      rows ? h('button', { class: 'small', type: 'button', onclick: () => { st.side = 1 - st.side; drawTiles() } }, `Flip (${conds[0]} | ${conds[1]})`) : null,
      isLayout ? h('button', {
        class: 'small', type: 'button', 'aria-pressed': 'false',
        onclick: (e) => { st.thumb = !st.thumb; e.currentTarget.setAttribute('aria-pressed', String(st.thumb)); drawTiles() },
      }, 'Thumbnail view') : null,
      recog,
      h('button', { class: 'small', type: 'button', onclick: () => answer(n, makeEvent(n, 'skip'), 'Skipped') }, 'Skip'),
      infoButton(n))

    const nodes = [
      headLine(n, `${card ? card.name : 'Score'} · grid ${it.letter || ''}`),
      b.task ? h('p', { class: 'task' }, b.task) : null,
      Array.isArray(b.expect) && b.expect.length ? h('p', { class: 'small' }, 'Exact words: ', b.expect.map((w, i) => [i ? ' · ' : '', h('span', { class: 'words' }, w)])) : null,
      Array.isArray(b.checklist) && b.checklist.length ? h('details', {}, h('summary', { class: 'small' }, 'Checklist'), h('ul', { class: 'check small' }, b.checklist.map((c) => h('li', {}, c)))) : null,
      b.pinned && (b.pinned.ref || b.pinned.base)
        ? h('div', { class: 'pinned' },
            b.pinned.ref ? h('div', {}, h('div', { class: 'small muted' }, 'Reference'), pinTile(b.pinned.ref, null)) : null,
            b.pinned.base ? h('div', {}, h('div', { class: 'small muted' }, 'Base'), pinTile(b.pinned.base, b.pinned.mask || null)) : null)
        : null,
      s.closeLook ? h('p', { class: 'small muted' }, 'Open the worst-looking picture full screen before you score (R5).') : null,
      tools,
      gridBox,
      h('p', { class: 'small muted' }, 'Tap a picture for full screen. Long press marks it Failed.'),
      defaultsBox,
      chipsBox,
    ]
    show(nodes)

    if (madeCount < 3) {
      drawTiles()
      footer(h('div', { class: 'q' }, h('span', {}, `Only ${madeCount} of 4 pictures were made, so this grid is not scored (R1).`)),
        h('button', { class: 'primary', type: 'button', onclick: () => answer(n, makeEvent(n, 'skip'), null) }, 'Next'))
      return
    }
    drawTiles()
    drawDefaults()
    drawFoot()
    preloadAfter(setIdOf(n), it.itemId)
  }

  /** Tap and long press on one element, with a press that moves treated as a scroll. */
  function pressable(el, { tap, long }) {
    let timer = 0
    let fired = false
    let start = null
    el.addEventListener('contextmenu', (e) => e.preventDefault())
    el.addEventListener('pointerdown', (e) => {
      fired = false
      start = { x: e.clientX, y: e.clientY }
      window.clearTimeout(timer)
      timer = window.setTimeout(() => {
        fired = true
        if (navigator.vibrate) { try { navigator.vibrate(20) } catch { /* not allowed */ } }
        long()
      }, 500)
    })
    const cancel = () => window.clearTimeout(timer)
    el.addEventListener('pointermove', (e) => {
      if (start && Math.hypot(e.clientX - start.x, e.clientY - start.y) > 10) cancel()
    })
    el.addEventListener('pointercancel', cancel)
    el.addEventListener('pointerleave', cancel)
    el.addEventListener('pointerup', (e) => {
      cancel()
      if (!fired && start && Math.hypot(e.clientX - start.x, e.clientY - start.y) <= 10) tap()
      start = null
    })
  }

  // ----------------------------------------------------------------- pair --

  const PAIR_QUESTIONS = {
    sweep: 'Which is the better picture overall?',
    sampler: 'Which is the better picture overall?',
    tie: 'Which would you rather have, for what this block asks?',
    chain: 'Which would you rather have, for what this block asks?',
  }

  function drawPair(n) {
    const it = n.item
    const b = briefOf(n)
    let side = 0
    const sides = [it.a || {}, it.b || {}]
    const hasRows = sides.every((x) => Array.isArray(x.rows))
    const conds = Array.isArray(b.conditions) ? b.conditions : ['before', 'after']
    // Before/after grids compare on their result; the other condition is a tap away.
    let cond = 1
    const gridBox = h('div', { class: 'grid' })
    const label = h('h2', {})
    const tilesOf = (g) => (hasRows ? g.rows.map((r) => (r ? r[cond] : null)) : Array.isArray(g.tiles) ? g.tiles : [])
    function draw() {
      const g = sides[side]
      label.textContent = `${side + 1}${g.letter ? ` · grid ${g.letter}` : ''}${hasRows ? ` · ${conds[cond]}` : ''}`
      const tiles = tilesOf(g)
      put(gridBox, ...tiles.slice(0, 4).map((token, i) => {
        const t = h('div', { class: 'tile' },
          token ? h('img', { src: imgUrl(token, 'g'), alt: `Picture ${i + 1}` }) : h('span', { class: 'empty' }, 'not made'),
          h('span', { class: 'side' }, String(side + 1)))
        fitTile(t)
        pressable(t, { tap: () => openFull(i), long: () => {} })
        return t
      }))
    }
    function items() {
      return tilesOf(sides[side]).slice(0, 4).map((token, k) => ({ token, label: `${side + 1} · picture ${k + 1} of 4` }))
    }
    function openFull(i) {
      Full.open(items(), i, { flip: () => { side = 1 - side; draw(); return items() } })
    }
    // Warm the other side so Flip is instant.
    for (const token of tilesOf(sides[1])) if (token) { const im = new Image(); im.src = imgUrl(token, 'g') }
    const q = it.question || PAIR_QUESTIONS[it.kind] || 'Which would you rather have?'
    const card = it.kind === 'sweep' || it.kind === 'sampler' ? 'ov@1' : undefined
    const choose = (winner, words) => answer(n, makeEvent(n, 'pair', { winner }, card ? { card } : undefined), words)
    show(
      headLine(n, q),
      b.task ? h('p', { class: 'task' }, b.task) : null,
      h('div', { class: 'spread' }, label,
        h('div', { class: 'row' },
          hasRows ? h('button', { class: 'small', type: 'button', onclick: () => { cond = 1 - cond; draw() } }, `${conds[0]} | ${conds[1]}`) : null,
          h('button', { class: 'small', type: 'button', onclick: () => { side = 1 - side; draw() } }, 'Flip 1 | 2'))),
      gridBox,
      h('div', { class: 'toolrow' },
        h('button', { class: 'small', type: 'button', onclick: () => answer(n, makeEvent(n, 'skip'), 'Skipped') }, 'Skip'),
        infoButton(n)),
    )
    draw()
    footer(
      h('div', { class: 'q' }, h('span', {}, q)),
      h('div', { class: 'three' },
        h('button', { type: 'button', onclick: () => choose(1, '1 is better') }, '1 is better'),
        h('button', { type: 'button', onclick: () => choose(0, 'Can’t tell') }, 'Can’t tell'),
        h('button', { type: 'button', onclick: () => choose(2, '2 is better') }, '2 is better')))
  }

  // ----------------------------------------------------------------- pick --

  async function drawPick(n) {
    const setId = setIdOf(n)
    const o = await overview(setId)
    const letters = Array.isArray(n.item.letters) ? n.item.letters : null
    const all = (o && Array.isArray(o.grids)) ? o.grids : []
    const grids = letters ? letters.map((l) => all.find((g) => g.letter === l) || { letter: l, tiles: [] }) : all
    show(
      headLine(n, 'Your pick for this set'),
      h('p', { class: 'muted small' }, 'Which grid would you keep, all things considered? One tap. This is the overall appeal no card measures.'),
      h('div', { class: 'overview' }, grids.map((g) => h('div', { class: 'setgrid' },
        h('h3', {}, `Grid ${g.letter}`),
        h('div', { class: 'grid' }, (g.tiles || (g.rows || []).map((r) => r && r[1]) || []).map((t) => miniTile(t)))))),
    )
    footer(
      h('div', { class: 'q' }, h('span', {}, 'Your pick')),
      h('div', { class: 'chips' },
        grids.map((g) => h('button', { type: 'button', onclick: () => answer(n, makeEvent(n, 'pick', { letter: g.letter }), `Picked ${g.letter}`) }, g.letter)),
        h('button', { type: 'button', class: 'quiet', onclick: () => answer(n, makeEvent(n, 'skip'), 'No pick') }, 'No pick')))
  }

  // ---------------------------------------------------------------- check --

  function drawCheck(n) {
    const it = n.item
    const b = briefOf(n)
    const tile = h('div', { class: 'tile' }, h('img', { src: imgUrl(it.token, 'g'), alt: 'The picture to check' }))
    fitTile(tile)
    tile.addEventListener('click', () => Full.open([{ token: it.token, label: 'Content check' }], 0, {}))
    const rating = String(it.rating || 'above general')
    const say = (agree, words) => answer(n, makeEvent(n, 'content', { agree }, { card: 'cd@1' }), words)
    show(
      headLine(n, 'Content check'),
      b.task ? h('p', { class: 'small muted' }, `The prompt asked for: ${b.task}`) : null,
      h('p', {}, `The picture reader rated this “${rating}”. Does it go beyond what the prompt asked?`),
      h('div', { style: { maxWidth: '420px' } }, tile),
      h('p', { class: 'small muted' }, 'The rating is the picture reader’s guess, not a measurement of anything.'),
    )
    footer(
      h('div', { class: 'q' }, h('span', {}, 'Does it go beyond what the prompt asked?')),
      h('div', { class: 'three' },
        h('button', { type: 'button', onclick: () => say('yes', 'Yes') }, 'Yes'),
        h('button', { type: 'button', onclick: () => say('no', 'No') }, 'No'),
        h('button', { type: 'button', onclick: () => say('unsure', 'Unsure') }, 'Unsure')))
  }

  // ---------------------------------------------------------- full screen --

  /**
   * One picture at a time over the page. Swipe for the next, pinch or
   * double-tap to zoom; the first double-tap swaps in the full-size copy.
   */
  const Full = (() => {
    let list = []
    let index = 0
    let opts = {}
    let scale = 1
    let tx = 0
    let ty = 0
    let full = false
    const pointers = new Map()
    let gesture = null
    let lastTap = null
    let wired = false

    const el = (id) => document.getElementById(id)

    function apply() {
      const img = el('full-img')
      if (img) img.style.transform = `translate(-50%, -50%) translate(${tx}px, ${ty}px) scale(${scale})`
    }

    function reset() {
      scale = 1
      tx = 0
      ty = 0
      apply()
    }

    function load() {
      const item = list[index]
      const img = el('full-img')
      if (!item || !img) return
      full = false
      img.src = item.token ? imgUrl(item.token, 'g') : ''
      img.alt = item.label || ''
      el('full-where').textContent = item.label || `${index + 1} of ${list.length}`
      const f = el('full-fail')
      const b = el('full-best')
      f.hidden = !opts.onFail
      b.hidden = !opts.onBest
      f.setAttribute('aria-pressed', String(!!(opts.isFail && opts.isFail(index))))
      b.setAttribute('aria-pressed', String(!!(opts.isBest && opts.isBest(index))))
      el('full-flip').hidden = !opts.flip
      reset()
    }

    function go(delta) {
      if (list.length < 2) return
      index = (index + delta + list.length) % list.length
      load()
    }

    function fullCopy(cx, cy) {
      const item = list[index]
      const img = el('full-img')
      if (!item || !item.token || !img) return
      if (!full) {
        full = true
        img.src = imgUrl(item.token, 'f')
      }
      const stage = el('full-stage').getBoundingClientRect()
      const s = 2.5
      tx = (stage.width / 2 - (cx - stage.left)) * (s - 1)
      ty = (stage.height / 2 - (cy - stage.top)) * (s - 1)
      scale = s
      apply()
    }

    function wire() {
      if (wired) return
      wired = true
      el('full-close').addEventListener('click', close)
      el('full-fail').addEventListener('click', () => { if (opts.onFail) { opts.onFail(index); load() } })
      el('full-best').addEventListener('click', () => { if (opts.onBest) { opts.onBest(index); load() } })
      el('full-flip').addEventListener('click', () => {
        if (!opts.flip) return
        const at = { scale, tx, ty }
        list = opts.flip()
        load()
        scale = at.scale
        tx = at.tx
        ty = at.ty
        apply()
      })
      const stage = el('full-stage')
      stage.addEventListener('pointerdown', (e) => {
        if (e.target.closest && e.target.closest('button')) return
        if (stage.setPointerCapture) stage.setPointerCapture(e.pointerId)
        pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
        if (pointers.size === 2) {
          const [a, b] = [...pointers.values()]
          gesture = { kind: 'pinch', dist: Math.hypot(a.x - b.x, a.y - b.y), scale }
        } else if (pointers.size === 1) {
          gesture = { kind: 'drag', x: e.clientX, y: e.clientY, tx, ty, t: performance.now() }
        }
      })
      stage.addEventListener('pointermove', (e) => {
        if (!pointers.has(e.pointerId)) return
        pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
        if (gesture && gesture.kind === 'pinch' && pointers.size >= 2) {
          const [a, b] = [...pointers.values()]
          const d = Math.hypot(a.x - b.x, a.y - b.y)
          scale = Math.min(8, Math.max(1, gesture.scale * (d / Math.max(1, gesture.dist))))
          if (scale > 1.6 && !full) {
            full = true
            const item = list[index]
            if (item && item.token) el('full-img').src = imgUrl(item.token, 'f')
          }
          apply()
        } else if (gesture && gesture.kind === 'drag' && scale > 1) {
          tx = gesture.tx + (e.clientX - gesture.x)
          ty = gesture.ty + (e.clientY - gesture.y)
          apply()
        }
      })
      const end = (e) => {
        if (!pointers.has(e.pointerId)) return
        pointers.delete(e.pointerId)
        const g = gesture
        if (pointers.size === 0) gesture = null
        if (!g || g.kind !== 'drag') return
        const dx = e.clientX - g.x
        const dy = e.clientY - g.y
        if (scale <= 1.01 && Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy)) {
          go(dx < 0 ? 1 : -1)
          return
        }
        if (Math.hypot(dx, dy) < 12 && performance.now() - g.t < 300) {
          const now = performance.now()
          if (lastTap && now - lastTap.t < 320 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 40) {
            lastTap = null
            if (scale > 1.01) reset()
            else fullCopy(e.clientX, e.clientY)
          } else {
            lastTap = { t: now, x: e.clientX, y: e.clientY }
          }
        }
      }
      stage.addEventListener('pointerup', end)
      stage.addEventListener('pointercancel', end)
      document.addEventListener('keydown', (e) => {
        if (el('full').hidden) return
        if (e.key === 'Escape') close()
        else if (e.key === 'ArrowRight') go(1)
        else if (e.key === 'ArrowLeft') go(-1)
        else if (e.key === 'f' && opts.flip) el('full-flip').click()
      })
    }

    function open(items, at, o) {
      wire()
      list = items.filter(Boolean)
      index = Math.min(Math.max(0, at || 0), Math.max(0, list.length - 1))
      opts = o || {}
      el('full').hidden = false
      document.body.style.overflow = 'hidden'
      load()
    }

    function close() {
      const box = el('full')
      if (!box || box.hidden) return
      box.hidden = true
      document.body.style.overflow = ''
      pointers.clear()
      gesture = null
      const img = el('full-img')
      if (img) img.removeAttribute('src')
    }

    return { open, close }
  })()

  // ------------------------------------------------------------- exports --

  window.Lab = { h, put, api, show, toast, route, every, load, save, errorOf, render }
})()
