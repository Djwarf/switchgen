/*
 * The lab page's Photos screen: the reference photos the editing and
 * reference nights use.
 *
 * Upload a photo from the phone, edit the words that describe it, and drag a
 * rectangle for the area a region edit may change. Photos can also be
 * dropped into the lab's refs folder on the machine; the server lists those
 * too. The browser shows each photo the right way up from its EXIF
 * orientation, and the rectangle is measured on the photo as shown, which is
 * the way the lab measures it too.
 */
;(function () {
  'use strict'

  const { h, put, api, show, toast, route, errorOf } = window.Lab
  const enc = encodeURIComponent
  const TYPES = ['image/jpeg', 'image/png', 'image/webp']
  const MAX_BYTES = 20 * 1024 * 1024
  const MAX_EDGE = 4096

  route(/^\/refs$/, async () => {
    show(h('h1', {}, 'Reference photos'), h('p', { class: 'muted' }, 'Loading…'))
    const [refs, needed] = await Promise.all([api('GET', '/api/lab/refs'), api('GET', '/api/lab/refs/needed')])
    if (!refs.ok) return show(h('h1', {}, 'Reference photos'), h('div', { class: 'note bad' }, errorOf(refs)))
    const list = Array.isArray(refs.data) ? refs.data : []
    const need = needed.ok && needed.data && Array.isArray(needed.data.refs) ? needed.data.refs : []
    const dropDir = needed.ok && needed.data ? needed.data.dropDir : null
    const byId = new Map(list.map((r) => [r.id, r]))
    const ids = [...new Set([...need.map((n) => n.id), ...list.map((r) => r.id)])]

    const cardsBox = h('div', {})
    for (const id of ids) cardsBox.append(refCard(id, byId.get(id) || null, need.find((n) => n.id === id) || null))

    show(
      h('h1', {}, 'Reference photos'),
      h('p', { class: 'muted small' }, 'The editing and reference nights start from these photos. They stay on this machine, outside the app’s Archive.'),
      cardsBox,
      uploadCard(need, dropDir),
      h('p', {}, h('a', { href: '#/' }, 'All runs')),
    )
  })

  function refCard(id, info, need) {
    const box = h('div', { class: 'card stack refcard' })
    const title = h('div', { class: 'spread' },
      h('h2', {}, id),
      h('span', { class: 'pill' }, info ? 'Here' : 'Not here yet'))
    const usedBy = need && Array.isArray(need.runs) && need.runs.length ? `Used by ${need.runs.join(', ')}.` : null

    if (!info) {
      put(box, title,
        h('div', { class: 'note warn' },
          `This photo has not arrived. Runs that need it will not start until it is here.`,
          usedBy ? ` ${usedBy}` : ''),
        need && need.describe ? h('p', { class: 'small muted' }, `What it should show: ${need.describe}`) : null,
        need && need.needsMask ? h('p', { class: 'small muted' }, 'Once it is here, drag a rectangle over the area a region edit may change.') : null)
      return box
    }

    const img = h('img', { class: 'refimg', src: `/api/lab/refs/${enc(id)}/image?v=${enc(info.sha12 || '')}`, alt: `Reference photo ${id}` })
    const sizeLine = h('p', { class: 'small muted' },
      info.width && info.height ? `${info.width} × ${info.height} pixels, measured the right way up.` : 'Size not measured.')
    img.addEventListener('load', () => {
      if (info.width && info.height && (img.naturalWidth !== info.width || img.naturalHeight !== info.height)) {
        sizeLine.append(h('span', { class: 'note bad' }, ` This browser shows it as ${img.naturalWidth} × ${img.naturalHeight}. The lab and the browser disagree about which way up it is, so a mask drawn here would not line up.`))
      }
    })

    const describe = h('textarea', { id: `describe-${id}`, rows: 3 })
    describe.value = typeof info.describe === 'string' ? info.describe : (need && need.describe) || ''
    const saveDescribe = h('button', {
      class: 'small', type: 'button',
      onclick: async () => {
        const text = describe.value.trim()
        if (!text) return toast('The description cannot be empty.')
        saveDescribe.disabled = true
        const r = await api('POST', `/api/lab/refs/${enc(id)}/describe`, { describe: text })
        saveDescribe.disabled = false
        if (r.ok) toast('Description saved. A night that has not started yet takes it when you press Start.', null, 5000)
        else toast(`Not saved. ${errorOf(r)}`, null, 12000)
      },
    }, 'Save description')

    put(box,
      title,
      usedBy ? h('p', { class: 'small muted' }, usedBy) : null,
      img,
      sizeLine,
      h('label', { for: `describe-${id}` }, 'What the photo shows (the image-to-image prompts use these words)'),
      describe,
      h('div', { class: 'row' }, saveDescribe),
      h('p', { class: 'small muted' }, 'A night that has not started yet takes new words when you press Start; one already under way keeps the words it started with.'),
    )

    const wantsMask = (need && need.needsMask) || info.mask
    if (wantsMask) box.append(maskEditor(id, info))
    else {
      const open = h('button', { class: 'small quiet', type: 'button' }, 'Draw an area to change')
      open.addEventListener('click', () => open.replaceWith(maskEditor(id, info)))
      box.append(open)
    }
    return box
  }

  // ---------------------------------------------------------------- mask --

  function maskEditor(id, info) {
    const wrap = h('div', { class: 'stack' })
    const img = h('img', { src: `/api/lab/refs/${enc(id)}/image?v=${enc(info.sha12 || '')}`, alt: '', draggable: 'false' })
    const rectEl = h('div', { class: 'rect', hidden: true })
    const overlay = info.mask
      ? h('div', { class: 'overlay' }, h('img', { src: `/api/lab/refs/${enc(id)}/mask.png?v=${Date.now()}`, alt: '' }))
      : null
    const box = h('div', { class: 'maskbox' }, img, overlay, rectEl)
    const readout = h('p', { class: 'small muted' }, info.mask ? 'The white area is the current mask. Drag to draw a new one.' : 'Drag a rectangle over the area a region edit may change.')
    const save = h('button', { class: 'small primary', type: 'button', disabled: true }, 'Save area')
    let rect = null
    let start = null

    function naturalScale() {
      const shown = img.getBoundingClientRect()
      const nw = img.naturalWidth || info.width || 1
      const nh = img.naturalHeight || info.height || 1
      return { shown, sx: nw / Math.max(1, shown.width), sy: nh / Math.max(1, shown.height), nw, nh }
    }

    function at(e) {
      const { shown } = naturalScale()
      return {
        x: Math.min(Math.max(0, e.clientX - shown.left), shown.width),
        y: Math.min(Math.max(0, e.clientY - shown.top), shown.height),
      }
    }

    function drawRect(a, b) {
      const x = Math.min(a.x, b.x)
      const y = Math.min(a.y, b.y)
      const w = Math.abs(a.x - b.x)
      const hh = Math.abs(a.y - b.y)
      rectEl.hidden = false
      Object.assign(rectEl.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${hh}px` })
      const { sx, sy, nw, nh } = naturalScale()
      rect = {
        x: Math.round(x * sx),
        y: Math.round(y * sy),
        w: Math.round(w * sx),
        h: Math.round(hh * sy),
      }
      rect.w = Math.min(rect.w, nw - rect.x)
      rect.h = Math.min(rect.h, nh - rect.y)
      readout.textContent = `Area: ${rect.w} × ${rect.h} pixels, starting ${rect.x} from the left and ${rect.y} from the top (${where(rect, nw, nh)}).`
      save.disabled = rect.w < 16 || rect.h < 16
    }

    box.addEventListener('pointerdown', (e) => {
      if (box.setPointerCapture) box.setPointerCapture(e.pointerId)
      start = at(e)
      e.preventDefault()
    })
    box.addEventListener('pointermove', (e) => {
      if (!start) return
      drawRect(start, at(e))
    })
    const end = (e) => {
      if (!start) return
      drawRect(start, at(e))
      start = null
    }
    box.addEventListener('pointerup', end)
    box.addEventListener('pointercancel', () => { start = null })

    save.addEventListener('click', async () => {
      if (!rect) return
      const scaleToInfo = info.width && info.height && img.naturalWidth && img.naturalHeight &&
        (img.naturalWidth !== info.width || img.naturalHeight !== info.height)
      if (scaleToInfo) {
        toast('The lab and this browser disagree about which way up the photo is, so the area was not saved.', null, 6000)
        return
      }
      save.disabled = true
      const r = await api('POST', `/api/lab/refs/${enc(id)}/mask`, { rect })
      save.disabled = false
      if (r.ok) {
        toast('Area saved.')
        window.Lab.render()
      } else toast(errorOf(r), null, 6000)
    })

    wrap.append(h('h3', {}, 'Area a region edit may change'), box, readout, h('div', { class: 'row' }, save))
    return wrap
  }

  /** Plain position words for a rectangle, e.g. "lower left". */
  function where(r, w, hh) {
    const cx = (r.x + r.w / 2) / w
    const cy = (r.y + r.h / 2) / hh
    const v = cy < 1 / 3 ? 'upper' : cy > 2 / 3 ? 'lower' : 'middle'
    const s = cx < 1 / 3 ? 'left' : cx > 2 / 3 ? 'right' : 'centre'
    return v === 'middle' && s === 'centre' ? 'centre' : `${v} ${s}`
  }

  // -------------------------------------------------------------- upload --

  function uploadCard(need, dropDir) {
    const missing = need.filter((n) => !n.present).map((n) => n.id)
    const names = [...new Set([...missing, ...need.map((n) => n.id)])]
    const file = h('input', { type: 'file', id: 'ref-file', accept: 'image/jpeg,image/png,image/webp,image/heic,image/heif' })
    const name = h('input', { type: 'text', id: 'ref-name', autocomplete: 'off', autocapitalize: 'none', spellcheck: 'false', placeholder: missing[0] || 'scene' })
    if (missing[0]) name.value = missing[0]
    const status = h('div', {})
    const send = h('button', { class: 'primary', type: 'button' }, 'Upload')

    send.addEventListener('click', async () => {
      const f = file.files && file.files[0]
      const label = name.value.trim()
      if (!f) return put(status, h('div', { class: 'note warn' }, 'Choose a photo first.'))
      if (!/^[a-z0-9][a-z0-9-]{0,31}$/i.test(label)) {
        return put(status, h('div', { class: 'note warn' }, 'Name it with letters, digits and dashes (for example scene).'))
      }
      send.disabled = true
      put(status, h('p', { class: 'muted' }, 'Sending…'))
      let blob = f
      let type = f.type
      if (!TYPES.includes(type) || f.size > MAX_BYTES) {
        put(status, h('p', { class: 'muted' }, 'Converting to JPEG on this phone first…'))
        try {
          blob = await toJpeg(f)
          type = 'image/jpeg'
        } catch {
          send.disabled = false
          return put(status, h('div', { class: 'note bad' },
            'This photo could not be read here. The lab takes JPEG, PNG or WebP up to 20 MB; on an iPhone, choose Most Compatible under Settings > Camera > Formats, or share the photo as a JPEG.'))
        }
      }
      let res
      try {
        res = await fetch('/api/lab/refs', { method: 'POST', cache: 'no-store', headers: { 'Content-Type': type, 'X-Lab-Name': label }, body: blob })
      } catch {
        send.disabled = false
        return put(status, h('div', { class: 'note bad' }, 'The lab server did not answer.'))
      }
      let data = null
      try { data = await res.json() } catch { /* not JSON */ }
      send.disabled = false
      if (!res.ok) return put(status, h('div', { class: 'note bad' }, (data && data.error) || `The lab server answered ${res.status}.`))
      toast(`Saved as ${data && data.id ? data.id : label}.`)
      window.Lab.render()
    })

    return h('div', { class: 'card stack' },
      h('h2', {}, 'Add a photo'),
      missing.length ? h('p', {}, `Still needed: ${missing.join(', ')}.`) : null,
      h('label', { for: 'ref-file' }, 'Photo (JPEG, PNG or WebP; other kinds are converted to JPEG on this phone)'),
      file,
      h('label', { for: 'ref-name' }, 'Name'),
      name,
      names.length ? h('div', { class: 'chips' }, names.map((n) => h('button', { class: 'small', type: 'button', onclick: () => { name.value = n } }, n))) : null,
      h('p', { class: 'small muted' }, 'A new photo under a name already in use replaces it for nights that have not started yet.'),
      h('div', { class: 'row' }, send),
      status,
      dropDir ? h('p', { class: 'small muted' }, `Or copy it on the machine into ${dropDir}, named for example scene.jpg.`) : null)
  }

  /** Re-encode a photo as JPEG, the right way up, at most 4096 pixels on its long edge. */
  async function toJpeg(file) {
    let bitmap
    try {
      bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
    } catch {
      bitmap = await new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file)
        const img = new Image()
        img.onload = () => { URL.revokeObjectURL(url); resolve(img) }
        img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('unreadable')) }
        img.src = url
      })
    }
    const w0 = bitmap.width
    const h0 = bitmap.height
    const k = Math.min(1, MAX_EDGE / Math.max(w0, h0))
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(w0 * k)
    canvas.height = Math.round(h0 * k)
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92))
    if (!blob) throw new Error('could not encode')
    if (blob.size > MAX_BYTES) throw new Error('too large')
    return blob
  }
})()
