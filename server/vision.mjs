/**
 * SwitchGen vision: what is actually in a picture, answered locally.
 *
 * WHY THIS EXISTS.
 *
 * src/lib/loraIndex.ts recovers each LoRA's training vocabulary from its own
 * safetensors header, so a prompt can be matched against what a LoRA was really
 * shown. That covers the generate path, where the user has typed something. It
 * does nothing for the edit and refine paths, where the prompt is often a few
 * words or empty and the picture carries the whole intent.
 *
 * This file closes that half. It reads an image and returns booru tags with
 * confidences.
 *
 *
 * WHY WD14 AND NOT THE ALTERNATIVES.
 *
 * The LoRAs installed here were trained on booru captions, and loraIndex.ts
 * normalises tags by lowercasing and turning underscores into spaces. WD14 is
 * the tagger those captions came from, so its output lands in the same
 * vocabulary with no translation step at all: `large_breasts` off the tagger
 * normalises to `large breasts`, which is the exact string in the index. The
 * join is an equality test, not a similarity score.
 *
 * The alternatives were measured against that, not guessed at:
 *
 *   CLIP vision. models/clip_vision is empty, so a model has to be fetched
 *   either way, and what comes back is a 768 or 1024 wide embedding. Nothing
 *   here consumes an embedding. Turning one into tags needs a projection head
 *   or a labelled reference bank, which is a lot of new code to arrive at a
 *   worse version of what WD14 gives directly.
 *
 *   The installed YOLO detectors. Zero download, already on disk, and they
 *   genuinely answer a question WD14 answers badly: where the faces and hands
 *   are and how large they are in frame. They cannot name a style or a concept.
 *   So they are built here too, as a second probe, not as a replacement.
 *
 * Both run on the CPU. onnxruntime 1.30.0 in the ComfyUI venv reports only
 * ['AzureExecutionProvider', 'CPUExecutionProvider'], so there is no CUDA
 * provider to use, and ultralytics is pinned to device 'cpu' below on purpose:
 * the GPU on this machine is contended, and a tagger that waits on a generation
 * or steals memory from the editor would be worse than no tagger.
 *
 *
 * MEASURED ON THIS MACHINE.
 *
 *   wd-vit-tagger-v3, 378,536,310 bytes, CPU, 24 cores:
 *     cold, interpreter + import + session load + one image    816 ms
 *     warm, per further image in the same process              170 ms
 *
 * Cold start is cheap enough that a resident worker is not worth its failure
 * modes, so every request is a one-shot process that cannot outlive its answer.
 * Batching several images into one call amortises the 816 ms down to 170 ms
 * each, which is why /tag and /detect take a list rather than a single path.
 *
 * Peak memory of that process, measured in the review of 2026-09-23 with the
 * ComfyUI queue empty: tagging a batch of 24 outputs peaked at 951 MB, and
 * /inspect (tags plus faces, hands and person) on one picture at 1.81 GB.
 * That is the size of a thing that can tip a render over; see READER_PEAK.
 *
 *
 * POSTURE.
 *
 * Local only, same as api.mjs: every path is confined to a known root, the
 * confinement is re-checked after realpath so a symlink planted in a root
 * cannot reach outside it, and no path carrying a control character is handed
 * to a child. Every child is spawned under an AbortController wired to the
 * response, so a closed tab kills the work rather than leaving a 24 core
 * inference running for nobody.
 */
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import crypto from 'node:crypto'
import { spawn } from 'node:child_process'
import { confineReal, guardMutation, readBody, reqUrl, safely, send } from './guard.mjs'

const MODELS = process.env.SWITCHGEN_MODELS ?? '/mnt/storage/ai/models'
const OUTPUTS = process.env.SWITCHGEN_OUTPUTS ?? '/mnt/storage/ai/outputs'
const COMFY_ROOT = process.env.SWITCHGEN_COMFY ?? '/mnt/storage/repos/ComfyUI'
const COMFY_INPUT = process.env.SWITCHGEN_COMFY_INPUT ?? path.join(COMFY_ROOT, 'input')

/**
 * The ComfyUI venv, because that is where onnxruntime, numpy, pillow and
 * ultralytics already are. Nothing new is installed by this file.
 */
const PYTHON = process.env.SWITCHGEN_PYTHON ?? path.join(COMFY_ROOT, 'venv/bin/python')

/** Where the tagger lives. Two files, both verified by HEAD before download. */
const WD14_DIR = process.env.SWITCHGEN_WD14 ?? path.join(MODELS, 'wd14')
const WD14_MODEL = path.join(WD14_DIR, 'model.onnx')
const WD14_TAGS = path.join(WD14_DIR, 'selected_tags.csv')

/**
 * The exact weights this file was written against, with the byte counts read
 * off a real HEAD request on 2026-09-22. If the tagger is missing, this is what
 * /api/vision/capabilities hands the client so it can offer the download
 * through the existing POST /api/download, which confines to the models root.
 *
 * wd-vit-tagger-v3 rather than the larger swinv2: same v3 vocabulary of 10861
 * tags, 89 MB smaller, and faster per image on a CPU-only runtime.
 */
const WD14_SOURCE = {
  repo: 'SmilingWolf/wd-vit-tagger-v3',
  files: [
    {
      filename: 'model.onnx',
      dest: 'wd14/model.onnx',
      url: 'https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx',
      sizeBytes: 378536310,
    },
    {
      filename: 'selected_tags.csv',
      dest: 'wd14/selected_tags.csv',
      url: 'https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv',
      sizeBytes: 308468,
    },
  ],
}

/** The detectors the Impact Pack already installed. No download for these. */
const YOLO = {
  face: path.join(MODELS, 'ultralytics/bbox/face_yolov8m.pt'),
  hand: path.join(MODELS, 'ultralytics/bbox/hand_yolov8s.pt'),
  person: path.join(MODELS, 'ultralytics/segm/person_yolov8m-seg.pt'),
}

/** Image roots a caller may name by relative path. Nothing else is readable. */
const ROOTS = { output: OUTPUTS, input: COMFY_INPUT, model: MODELS }

const IMAGE = /\.(png|jpe?g|webp|bmp|gif|avif)$/i
/** A batch is an interactive question, not a bulk job. */
const MAX_IMAGES = 24
/** Largest image body accepted on the raw upload path. */
const MAX_UPLOAD = 40 << 20
/** Longest a child may run before it is killed, per batch. */
const RUN_MS = 120000

const GIB = 1073741824
/**
 * What one reader process is expected to peak at, rounded up from the
 * figures measured above: 951 MB to tag, 1.81 GB to inspect. Detection alone
 * was not measured; it loads the same detectors /inspect does, so it is held
 * to the /inspect figure.
 */
const READER_PEAK = { tag: 1.0 * GIB, detect: 1.9 * GIB }
/**
 * The share of RAM below which the machine starts stopping programs to get
 * memory back. earlyoom runs here as `-m 8`, and when it acts it stops the
 * largest process, which while it renders is ComfyUI: its journal shows it
 * doing so ten times between 2026-09-21 and 2026-09-22. It is measured
 * against all of RAM here; earlyoom 1.9 takes zram out of its own total,
 * which puts its line lower, so this errs on the safe side.
 */
const FLOOR_PERCENT = (() => {
  // 0 is a real answer (no earlyoom, only the kernel's own killer at the end).
  const set = Number(process.env.SWITCHGEN_MEMORY_FLOOR_PERCENT?.trim() || NaN)
  return Number.isFinite(set) && set >= 0 && set < 100 ? set : 8
})()
/**
 * Room kept on top of that line, because a render goes on growing while a
 * reading runs: a reading takes a few seconds, and the free figure is read
 * once, before it starts.
 */
const FLOOR_MARGIN = 0.5 * GIB

// ---------------------------------------------------------------------------
// Helpers, matching server/api.mjs
// ---------------------------------------------------------------------------

async function readBytes(req, limit) {
  const chunks = []
  let size = 0
  for await (const c of req) {
    size += c.length
    if (size > limit) return null
    chunks.push(c)
  }
  return Buffer.concat(chunks)
}

async function exists(p) {
  try { await fs.access(p); return true } catch { return false }
}

async function sizeOf(p) {
  try { return (await fs.stat(p)).size } catch { return null }
}

/**
 * Is a fetched file all there, and if not, can a fetch mend it? Being on disk
 * is not enough: aria2c writes under the final name, so a fetch cut short
 * leaves a file that exists and does not load, and nothing in the app would
 * fetch it again. aria2c keeps its control file beside a download until the
 * last byte lands, and with several connections it fills pieces out of order,
 * so a file can reach full size while still a partial. A file cut short some
 * other way is caught by its size, with the same 0.5% of slack the
 * downloader's own check allows.
 *
 * 'partial' and 'short' are told apart because a fetch treats them
 * differently. The downloader resumes a file with aria2c's control file
 * beside it, and refuses to touch any other file already there, so a short
 * file without one (copied in by hand, or its control file deleted) cannot be
 * mended by fetching again: it has to be moved aside first.
 *
 * @returns {Promise<'absent' | 'whole' | 'partial' | 'short'>}
 */
async function fileState(p, bytes) {
  const [size, partial] = await Promise.all([sizeOf(p), exists(p + '.aria2')])
  if (size == null) return 'absent'
  if (partial) return 'partial'
  return size >= bytes * 0.995 ? 'whole' : 'short'
}

/**
 * Turn the caller's image references into absolute paths that are known to
 * exist inside a known root. Anything else is reported by index so the client
 * can say which one it got wrong rather than failing the whole batch blindly.
 */
async function resolveImages(list) {
  const out = []
  const bad = []
  for (let i = 0; i < list.length; i++) {
    const item = list[i]
    const rel = typeof item === 'string' ? item : item?.rel
    const kind = (typeof item === 'object' && item?.kind) || 'output'
    const root = ROOTS[kind]
    if (!root) { bad.push({ index: i, error: `kind must be one of ${Object.keys(ROOTS).join(', ')}` }); continue }
    if (typeof rel !== 'string' || !rel.trim()) { bad.push({ index: i, error: 'rel must be a non-empty string' }); continue }
    // A control character in a path is never legitimate here and is the one
    // thing that could change how a child reads its argument list.
    // oxlint-disable-next-line no-control-regex -- matching them is the point
    if (/[\u0000-\u001f]/.test(rel)) { bad.push({ index: i, error: 'path contains a control character' }); continue }
    if (!IMAGE.test(rel)) { bad.push({ index: i, error: 'not an image extension' }); continue }
    const full = await confineReal(root, rel)
    if (!full) { bad.push({ index: i, error: 'path escapes root' }); continue }
    let st
    try { st = await fs.stat(full) } catch { bad.push({ index: i, error: 'not found' }); continue }
    if (!st.isFile()) { bad.push({ index: i, error: 'not a regular file' }); continue }
    out.push({ index: i, kind, rel, path: full })
  }
  return { images: out, bad }
}

// ---------------------------------------------------------------------------
// The child
// ---------------------------------------------------------------------------

/**
 * The whole Python side, held here rather than in a file of its own so that the
 * server is one unit: there is no second artefact to keep in step with it, and
 * nothing to go stale on disk. It arrives on the child's stdin, which means it
 * is never written anywhere and cannot be edited out from under a running
 * server. The request is argv[1] as JSON; the answer is JSON on stdout.
 *
 * Preprocessing is WD14's own: composite onto white so alpha does not read as
 * black, pad to a square with white, resize to the model's edge, convert RGB to
 * BGR, and leave the values in 0 to 255 with no normalisation. Getting any of
 * that wrong produces tags that look plausible and are wrong, which is the
 * failure mode this project cares most about avoiding.
 */
const PY = String.raw`
import sys, json, csv

def fail(msg):
    sys.stdout.write(json.dumps({"error": msg}))
    sys.exit(0)

try:
    req = json.loads(sys.argv[1])
except Exception as e:
    fail("bad request: %s" % e)

images = req.get("images", [])
out = {"tag": None, "detect": None}

def load_square(p, edge):
    from PIL import Image
    import numpy as np
    im = Image.open(p)
    # Composite onto white. A transparent PNG read straight to RGB gives black
    # where it should give paper, and the tagger answers a different picture.
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im.convert("RGBA"))
    im = bg.convert("RGB")
    w, h = im.size
    m = max(w, h)
    sq = Image.new("RGB", (m, m), (255, 255, 255))
    sq.paste(im, ((m - w) // 2, (m - h) // 2))
    sq = sq.resize((edge, edge), Image.BICUBIC)
    a = np.asarray(sq, dtype=np.float32)[:, :, ::-1]  # RGB to BGR
    return a[None], (w, h)

if req.get("tag"):
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:
        fail("onnxruntime or numpy missing from this interpreter: %s" % e)

    names, cats = [], []
    try:
        with open(req["tagsCsv"], newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names.append(row["name"])
                cats.append(int(row["category"]))
    except Exception as e:
        fail("cannot read selected_tags.csv: %s" % e)

    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(req["model"], opts, providers=["CPUExecutionProvider"])
    except Exception as e:
        fail("cannot open the tagger: %s" % e)

    spec = sess.get_inputs()[0]
    edge = spec.shape[1] if isinstance(spec.shape[1], int) else 448
    if len(names) != sess.get_outputs()[0].shape[-1]:
        fail("tag list and model disagree: %d labels, %d outputs" % (len(names), sess.get_outputs()[0].shape[-1]))

    gen_t = float(req.get("generalThreshold", 0.35))
    chr_t = float(req.get("characterThreshold", 0.75))
    limit = int(req.get("limit", 64))

    # Category codes in selected_tags.csv: 0 general, 4 character, 9 rating.
    gi = [i for i, c in enumerate(cats) if c == 0]
    ci = [i for i, c in enumerate(cats) if c == 4]
    ri = [i for i, c in enumerate(cats) if c == 9]

    rows = []
    for im in images:
        try:
            arr, (w, h) = load_square(im["path"], edge)
            pr = sess.run(None, {spec.name: arr})[0][0]
        except Exception as e:
            rows.append({"index": im["index"], "error": str(e)})
            continue
        general = sorted(
            [{"tag": names[i], "confidence": round(float(pr[i]), 4)} for i in gi if pr[i] >= gen_t],
            key=lambda t: -t["confidence"])[:limit]
        character = sorted(
            [{"tag": names[i], "confidence": round(float(pr[i]), 4)} for i in ci if pr[i] >= chr_t],
            key=lambda t: -t["confidence"])[:8]
        ratings = sorted(
            [{"tag": names[i], "confidence": round(float(pr[i]), 4)} for i in ri],
            key=lambda t: -t["confidence"])
        rows.append({
            "index": im["index"],
            "width": w, "height": h,
            "rating": ratings[0]["tag"] if ratings else None,
            "ratings": ratings,
            "general": general,
            "character": character,
        })
    out["tag"] = {"model": req["model"], "labels": len(names), "edge": edge, "rows": rows}

if req.get("detect"):
    try:
        import os
        os.environ["YOLO_VERBOSE"] = "0"
        from ultralytics import YOLO
    except Exception as e:
        fail("ultralytics missing from this interpreter: %s" % e)

    conf = float(req.get("detectThreshold", 0.35))
    loaded = {}
    rows = []
    for im in images:
        found = {}
        err = None
        for kind, weights in req["detectors"].items():
            try:
                if kind not in loaded:
                    loaded[kind] = YOLO(weights)
                # device is pinned to cpu deliberately: the GPU here is shared
                # with generation and with the editor, and a detector is not
                # worth evicting either of them.
                r = loaded[kind].predict(im["path"], device="cpu", verbose=False, conf=conf)[0]
                iw, ih = r.orig_shape[1], r.orig_shape[0]
                items = []
                for b in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    items.append({
                        "confidence": round(float(b.conf[0]), 4),
                        "box": [round(x1), round(y1), round(x2), round(y2)],
                        # Share of the frame this box covers. This is the number
                        # that decides whether a face is big enough to be worth
                        # a detail pass, so it is returned rather than derived.
                        "areaShare": round(((x2 - x1) * (y2 - y1)) / float(iw * ih), 5),
                    })
                items.sort(key=lambda d: -d["areaShare"])
                found[kind] = items
            except Exception as e:
                err = str(e)
        row = {"index": im["index"], "detections": found}
        if err:
            row["error"] = err
        rows.append(row)
    out["detect"] = {"rows": rows}

sys.stdout.write(json.dumps(out))
`

/**
 * Run one batch in one child.
 *
 * The child is killed when the response closes. The teardown listens on the
 * response and not the request for the reason api.mjs spells out: an
 * IncomingMessage for a bodyless request can be destroyed the moment its
 * message is complete, so its 'close' is not the client leaving. The response
 * stays open until we end it, so its 'close' is.
 */
function runPython(request, signal) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, ['-', JSON.stringify(request)], {
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    let killed = false
    let timedOut = false

    const stop = () => {
      if (killed) return
      killed = true
      try { child.kill('SIGKILL') } catch { /* already gone */ }
    }
    const timer = setTimeout(() => { timedOut = true; stop() }, RUN_MS)
    // A signal can already be aborted when we get here, in which case 'abort'
    // will never fire again and adding a listener would leak the child.
    if (signal?.aborted) stop()
    else signal?.addEventListener('abort', stop, { once: true })

    const done = () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', stop)
    }

    child.stdout.on('data', d => { stdout += d })
    child.stderr.on('data', d => { if (stderr.length < 65536) stderr += d })
    child.stdin.on('error', () => { /* child died before the script landed */ })
    child.stdin.end(PY)

    child.on('error', err => {
      done()
      reject(new Error(`cannot run ${PYTHON}: ${err.message}`))
    })
    child.on('close', code => {
      done()
      if (timedOut) return reject(Object.assign(new Error('vision timed out'), { timeout: true }))
      if (killed) return reject(Object.assign(new Error('cancelled'), { cancelled: true }))
      if (code !== 0) {
        return reject(new Error(stderr.trim().split('\n').slice(-3).join(' ') || `python exited ${code}`))
      }
      try { resolve(JSON.parse(stdout)) } catch {
        reject(new Error(`python returned something that is not JSON: ${stdout.slice(0, 200)}`))
      }
    })
  })
}

// ---------------------------------------------------------------------------
// One reader at a time, and only when it fits
// ---------------------------------------------------------------------------

const gb = n => `${(n / GIB).toFixed(1)} GB`

/**
 * Why a reader that needs `peak` bytes must not start now, as the sentence
 * the page shows, or null when it may.
 *
 * Every request used to start its own process at once, however many were
 * running and however little memory was free. A heavy render can take free
 * memory to within a couple of GB of the line where the machine stops its
 * largest program, and that program is ComfyUI, not the reader, so a 1 to 2
 * GB reading at that moment can cost a render of many minutes, and ComfyUI
 * comes back with an empty queue. So a reading that would leave less than the
 * line plus a margin is not started, and the sentence gives the real figures,
 * so the reader can see why and when to try again.
 */
export function memoryRefusal(peak, free, total, floorPercent = FLOOR_PERCENT, margin = FLOOR_MARGIN) {
  const floor = total * floorPercent / 100
  if (free - peak >= floor + margin) return null
  const line = `${gb(floor)} at which the system stops its largest program (usually ComfyUI) to get memory back`
  // A reading that would go below the line says so; "too close to" is kept
  // for one that would stay above it but inside the margin, where it was the
  // truth and "under" would not be.
  const why = free <= peak
    ? `only ${gb(free)} is free. Running out makes the system stop its largest program (usually ComfyUI) ` +
      'to get memory back'
    : free - peak < floor
      ? `${gb(free)} is free, so reading now would leave about ${gb(free - peak)}, under the ${line}`
      : `${gb(free)} is free, so reading now would leave about ${gb(free - peak)}. That is too close to the ${line}`
  return `The picture reader needs about ${gb(peak)} of memory and ${why}, so nothing was read. ` +
    'Try again once more memory is free, for example when a render has finished.'
}

/**
 * One reader process at a time, across every tab and device. Two tabs
 * reading at once, the phone and the desktop say, used to double the memory
 * taken. A request waits its turn; one whose page goes while it waits leaves
 * the line without ever starting a process.
 */
let reading = false
const waiting = []

function takeTurn(signal) {
  if (!reading) {
    reading = true
    return Promise.resolve()
  }
  return new Promise((resolve, reject) => {
    const turn = () => {
      signal?.removeEventListener('abort', leave)
      resolve()
    }
    const leave = () => {
      const i = waiting.indexOf(turn)
      if (i >= 0) waiting.splice(i, 1)
      reject(Object.assign(new Error('cancelled'), { cancelled: true }))
    }
    if (signal?.aborted) return leave()
    waiting.push(turn)
    signal?.addEventListener('abort', leave, { once: true })
  })
}

function endTurn() {
  const next = waiting.shift()
  // Handed straight to the next in line, so the slot stays taken.
  if (next) next()
  else reading = false
}

/**
 * Run one batch when its turn comes and it fits in memory. Free memory is
 * read inside the turn, just before the process starts, so it is not counted
 * while an earlier reader still holds some of it.
 */
async function read(request, signal) {
  await takeTurn(signal)
  try {
    const peak = request.detect ? READER_PEAK.detect : READER_PEAK.tag
    // os.freemem() is MemAvailable on Linux, the figure earlyoom watches.
    const refusal = memoryRefusal(peak, os.freemem(), os.totalmem())
    if (refusal) throw Object.assign(new Error(refusal), { busy: 'memory' })
    return await runPython(request, signal)
  } finally {
    endTurn()
  }
}

// ---------------------------------------------------------------------------
// What is actually installed
// ---------------------------------------------------------------------------

/**
 * The honest answer to "can this machine see". Every field is a fact checked on
 * disk at call time, not a flag someone set. A client that cannot get `tagger`
 * true has to hide the feature rather than show a control that silently does
 * nothing, which is the mistake api.mjs documents having made once already with
 * deleteFiles.
 */
async function capabilities() {
  const [modelSource, tagsSource] = WD14_SOURCE.files
  const [python, modelState, tagsState] = await Promise.all([
    exists(PYTHON), fileState(WD14_MODEL, modelSource.sizeBytes), fileState(WD14_TAGS, tagsSource.sizeBytes),
  ])
  const model = modelState === 'whole'
  const tagsCsv = tagsState === 'whole'
  const modelStarted = modelState !== 'absent'
  // Short files a fetch would refuse to touch (see fileState).
  const stuck = [[modelState, WD14_MODEL], [tagsState, WD14_TAGS]]
    .filter(([state]) => state === 'short')
    .map(([, file]) => file)
  const [modelBytes, detectors] = await Promise.all([
    sizeOf(WD14_MODEL),
    (async () => {
      const found = {}
      for (const [kind, p] of Object.entries(YOLO)) if (await exists(p)) found[kind] = p
      return found
    })(),
  ])
  const tagger = python && model && tagsCsv
  return {
    server: 'switchgen-vision',
    python: python ? PYTHON : null,
    tagger,
    taggerModel: tagger ? WD14_MODEL : null,
    taggerBytes: modelBytes,
    taggerVocabulary: 'danbooru-v3',
    detect: python && Object.keys(detectors).length > 0,
    detectors: Object.keys(detectors),
    device: 'cpu',
    roots: { ...ROOTS },
    // Everything a client needs to offer the fetch through POST /api/download,
    // which already confines writes to the models root. Nothing is offered
    // while a short file stands in the way: that fetch would be refused, and
    // the reason says what to do instead.
    install: tagger || stuck.length
      ? null
      : { ...WD14_SOURCE, missing: [!model && 'model.onnx', !tagsCsv && 'selected_tags.csv'].filter(Boolean) },
    reason: tagger
      ? null
      : !python
        ? `no interpreter at ${PYTHON}; set SWITCHGEN_PYTHON to one that has onnxruntime`
        : stuck.length === 1
          ? `${stuck[0]} is on disk but short of its full size, and is not a download that stopped part way, ` +
            'so a fetch will not replace it; move it aside, then fetch the tagger'
          : stuck.length
            ? `${stuck.join(' and ')} are on disk but short of their full size, and are not downloads that ` +
              'stopped part way, so a fetch will not replace them; move them aside, then fetch the tagger'
            : modelStarted && !model
              ? 'the tagger download did not finish; fetch it again to complete it'
              : `the tagger is not downloaded yet; it is ${(modelSource.sizeBytes / 1048576).toFixed(0)} MB`,
  }
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

const OWNED = ['/api/vision']
const METHODS = new Map([
  ['/api/vision/capabilities', 'GET'],
  ['/api/vision/tag', 'POST'],
  ['/api/vision/detect', 'POST'],
  ['/api/vision/inspect', 'POST'],
])

/**
 * Accept a raw image body, so the edit path can ask about a picture the user
 * has only just dropped in and which is not under any root yet. It lands in a
 * private temp file that is removed in a finally, including when the tab goes.
 */
async function withUpload(req, fn) {
  const bytes = await readBytes(req, MAX_UPLOAD)
  if (!bytes || bytes.length === 0) return { error: `body must be an image under ${MAX_UPLOAD >> 20} MB` }
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'switchgen-vision-'))
  const file = path.join(dir, `upload-${crypto.randomBytes(6).toString('hex')}`)
  try {
    await fs.writeFile(file, bytes)
    return await fn(file)
  } finally {
    await fs.rm(dir, { recursive: true, force: true }).catch(() => {})
  }
}

export const visionMiddleware = async (req, res, next) => {
  // An unreadable path is refused rather than parsed where it can throw.
  // See reqUrl in guard.mjs for what that throw used to do.
  const url = reqUrl(req)
  if (!url) return send(res, 400, { error: 'the request path is not a valid URL' })
  const p = url.pathname
  if (!p.startsWith('/api/vision')) return next()

  // One controller per request, aborted when the response closes, so a closed
  // tab takes its inference with it instead of leaving 24 cores busy.
  const abort = new AbortController()
  const onClose = () => abort.abort()
  res.on('close', onClose)

  try {
    if (p === '/api/vision/capabilities' && req.method === 'GET') {
      return send(res, 200, await capabilities())
    }

    const wantsTag = p === '/api/vision/tag' || p === '/api/vision/inspect'
    const wantsDetect = p === '/api/vision/detect' || p === '/api/vision/inspect'

    if ((wantsTag || wantsDetect) && req.method === 'POST') {
      if (!guardMutation(req, res, ['application/json', 'image/', 'application/octet-stream'])) return
      const caps = await capabilities()
      if (wantsTag && !caps.tagger) {
        return send(res, 503, { error: caps.reason, install: caps.install, tagger: false })
      }
      if (wantsDetect && !caps.detect) {
        return send(res, 503, { error: 'no detector weights are installed', detect: false })
      }

      const ct = String(req.headers['content-type'] ?? '')
      const raw = ct.startsWith('image/') || ct === 'application/octet-stream'

      /** Build the child's request once, for either input shape. */
      const buildAndRun = async (images) => {
        const request = {
          images,
          tag: wantsTag,
          detect: wantsDetect,
          model: WD14_MODEL,
          tagsCsv: WD14_TAGS,
          detectors: Object.fromEntries(Object.entries(YOLO).filter(([k]) => caps.detectors.includes(k))),
        }
        return request
      }

      if (raw) {
        const result = await withUpload(req, async (file) => {
          const request = await buildAndRun([{ index: 0, path: file }])
          const answer = await read(request, abort.signal)
          return { answer }
        })
        if (result.error) return send(res, 400, { error: result.error })
        const { answer } = result
        if (answer.error) return send(res, 500, { error: answer.error })
        return send(res, 200, { ...answer, uploaded: true })
      }

      const b = await readBody(req)
      if (!b) return send(res, 400, { error: 'body must be JSON under 1 MB, or an image with an image/* content type' })
      const list = Array.isArray(b.images) ? b.images : b.image != null ? [b.image] : null
      if (!list || list.length === 0) {
        return send(res, 400, { error: 'send {images: [{kind, rel}, ...]} or {image: {kind, rel}}, or POST the image bytes directly' })
      }
      if (list.length > MAX_IMAGES) {
        return send(res, 400, { error: `at most ${MAX_IMAGES} images per call, got ${list.length}` })
      }

      const { images, bad } = await resolveImages(list)
      if (images.length === 0) {
        return send(res, 400, { error: 'no readable image in the request', rejected: bad })
      }

      const request = await buildAndRun(images)
      if (typeof b.generalThreshold === 'number') request.generalThreshold = b.generalThreshold
      if (typeof b.characterThreshold === 'number') request.characterThreshold = b.characterThreshold
      if (typeof b.detectThreshold === 'number') request.detectThreshold = b.detectThreshold
      if (typeof b.limit === 'number') request.limit = b.limit

      const answer = await read(request, abort.signal)
      if (answer.error) return send(res, 500, { error: answer.error })
      // Hand back the caller's own reference on each row so a batch can be
      // reassembled without the client trusting array order.
      const refs = new Map(images.map(i => [i.index, { kind: i.kind, rel: i.rel }]))
      for (const section of ['tag', 'detect']) {
        for (const row of answer[section]?.rows ?? []) Object.assign(row, refs.get(row.index) ?? {})
      }
      return send(res, 200, { ...answer, rejected: bad.length ? bad : undefined })
    }

    const method = METHODS.get(p)
    if (method && method !== req.method) {
      res.setHeader('Allow', method)
      return send(res, 405, { error: `${p} takes ${method}, not ${req.method}` })
    }
    if (OWNED.some(o => p === o || p.startsWith(o + '/'))) {
      return send(res, 404, { error: `no such endpoint: ${req.method} ${p}` })
    }
    return next()
  } catch (err) {
    if (err?.cancelled) { try { res.end() } catch { /* gone */ } return }
    if (res.headersSent) { try { res.end() } catch { /* gone */ } return }
    // Busy, not broken, and marked so a caller can tell the two apart: the
    // sentence is meant to be shown as it is.
    if (err?.busy === 'memory') return send(res, 503, { error: err.message, busy: 'memory' })
    return send(res, err?.timeout ? 504 : 500, { error: String(err?.message ?? err) })
  } finally {
    res.off('close', onClose)
  }
}

export function switchgenVision() {
  return {
    name: 'switchgen-vision',
    configureServer(server) { server.middlewares.use(safely(visionMiddleware)) },
    configurePreviewServer(server) { server.middlewares.use(safely(visionMiddleware)) },
  }
}

export default visionMiddleware
