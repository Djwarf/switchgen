/**
 * Validate every registry family against ComfyUI's live /object_info.
 * Instantiates each graph exactly as the app would, then checks node classes,
 * required inputs, enum membership and link integrity. No GPU time.
 *   npm run validate
 *
 * Four layers are checked, not one:
 *
 *   base graphs      the registry's own text-to-image / edit / video graphs
 *   image-to-image   derived at runtime by deriveImg2Img
 *   quality passes   derived at runtime by lib/refine.ts: region refine,
 *                    face and hand detailing, hires fix, LoRA stacks
 *   continuation     derived at runtime by lib/continuation.ts: the chain tap,
 *                    the opening-frame slot, the bookend, and one real two-shot
 *                    seam built through shotPlan and instantiateShot
 *
 * The derived layers matter more than the base one, because nothing about them
 * is hand-verified: they are built by rewiring a graph at run time, and a
 * rewiring mistake shows up as an opaque backend error at queue time rather
 * than as anything a reader could act on. A validator that checks only the
 * base graphs certifies the least interesting third of what actually runs.
 */
import { FAMILIES, IMG2IMG, defaultsFor, instantiate, sidecarsOf, modelsOf, type Params } from '../src/lib/workflows'
import type { FamilyDef } from '../src/lib/registry'
import type { ApiWorkflow } from '../src/lib/comfy'
import {
  CHAIN_PREFIX,
  NODE_IDS,
  PLACEHOLDER_IMAGE,
  deriveBookend,
  deriveChainTap,
  deriveContinuation,
  deriveVaceShot,
  instantiateShot,
  shotPlan,
} from '../src/lib/continuation'
import {
  capabilitiesOf,
  deriveAutoDetail,
  deriveHiresFix,
  deriveRefine,
  hiresStepsFor,
  instantiateDerived,
  instantiateRefine,
  withLoras,
} from '../src/lib/refine'

const COMFY = process.env.COMFY_URL ?? 'http://127.0.0.1:8188'

function optionsOf(spec: any): string[] | null {
  if (Array.isArray(spec?.[0])) return spec[0] as string[]
  if (spec?.[0] === 'COMBO' && Array.isArray(spec?.[1]?.options)) return spec[1].options
  return null
}

const res = await fetch(`${COMFY}/object_info`)
if (!res.ok) { console.error(`Cannot reach ComfyUI at ${COMFY} (HTTP ${res.status})`); process.exit(2) }
const info: Record<string, any> = await res.json()

const have = (node: string, field: string): string[] => {
  const s = info?.[node]?.input?.required?.[field]?.[0]
  return Array.isArray(s) ? s : []
}
const installed = new Set([
  ...have('CheckpointLoaderSimple', 'ckpt_name'),
  ...have('UNETLoader', 'unet_name'),
  ...have('UnetLoaderGGUF', 'unet_name'),
])
const clips = have('CLIPLoader', 'clip_name')
const vaes = have('VAELoader', 'vae_name')
const loras = have('LoraLoaderModelOnly', 'lora_name')

let fail = 0, skip = 0, ok = 0

for (const def of FAMILIES) {
  const model = def.models.find(m => installed.has(m)) ?? def.models[0]
  const { clip, vae } = sidecarsOf(def)
  const missing = [
    ...clip.filter(c => !clips.includes(c)),
    ...(vae && !vaes.includes(vae) ? [vae] : []),
    ...modelsOf(def).filter(m => !installed.has(m)),
    ...Object.values(def.graph)
      .map(n => n.inputs['lora_name'])
      .filter((l): l is string => typeof l === 'string' && !loras.includes(l)),
  ]
  if (missing.length) { console.log(`SKIP  ${def.label}: missing ${missing.join(', ')}`); skip++; continue }

  const d = defaultsFor(def, model)
  const wf = instantiate(def, {
    model, positive: 'a test prompt', negative: d.negative, seed: 1,
    steps: d.steps, cfg: d.cfg, width: d.width, height: d.height,
    sampler: d.sampler, scheduler: d.scheduler,
    length: d.length || undefined, fps: d.fps || undefined,
    image: def.bindings.image ? 'example.png' : undefined,
  })

  const errs: string[] = []
  for (const [id, node] of Object.entries(wf)) {
    const nd = info[node.class_type]
    if (!nd) { errs.push(`node ${id}: unknown class "${node.class_type}"`); continue }
    const req = nd.input?.required ?? {}
    const known = { ...req, ...(nd.input?.optional ?? {}) }
    for (const k of Object.keys(req)) {
      if (!(k in node.inputs)) errs.push(`node ${id} (${node.class_type}): missing required "${k}"`)
    }
    for (const [k, v] of Object.entries(node.inputs)) {
      if (!(k in known)) { errs.push(`node ${id} (${node.class_type}): unexpected input "${k}"`); continue }
      // A link is specifically [nodeId: string, slot: number]. An empty array
      // or a list of links (COMFY_AUTOGROW_V3) is data, not a single link.
      const isLink = Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && typeof v[1] === 'number'
      if (isLink) {
        if (!wf[(v as any)[0]]) errs.push(`node ${id}: "${k}" -> missing node ${(v as any)[0]}`)
        continue
      }
      if (Array.isArray(v)) {
        for (const el of v as any[]) {
          if (Array.isArray(el) && typeof el[0] === 'string' && !wf[el[0]]) {
            errs.push(`node ${id}: "${k}" entry -> missing node ${el[0]}`)
          }
        }
        continue
      }
      const opts = optionsOf(known[k])
      if (opts && !opts.includes(v as string)) {
        const shown = opts.length > 6 ? `${opts.slice(0, 6).join(', ')}, …` : opts.join(', ')
        errs.push(`node ${id} (${node.class_type}): "${k}"="${v}" not in [${shown}]`)
      }
    }
  }

  // every parameter the UI exposes must actually reach the graph
  const need = ['model', 'seed', 'steps', 'cfg', 'positive'] as const
  for (const k of need) {
    if (k === 'model' && def.dualModel) continue
    if (!def.bindings[k]?.length) errs.push(`no "${k}" binding, so the UI control would do nothing`)
  }

  if (errs.length) { fail++; console.log(`FAIL  ${def.label}`); errs.forEach(e => console.log(`        ${e}`)) }
  else { ok++; console.log(`OK    ${def.label}  (${Object.keys(wf).length} nodes, ${Object.keys(def.bindings).length} bindings, model: ${model})`) }
}

// Image-to-image is DERIVED from each text-to-image graph at runtime, so a
// validator that only checks base graphs certifies nothing about it. Validate
// every derived variant too, or "8 OK" overstates what was actually checked.
let i2iOk = 0, i2iFail = 0, i2iSkip = 0
for (const [srcId, def] of Object.entries(IMG2IMG)) {
  const base = FAMILIES.find(f => f.id === srcId)
  if (!base) continue
  const model = base.models.find(m => installed.has(m)) ?? base.models[0]
  const { clip, vae } = sidecarsOf(def)
  const missing = [
    ...clip.filter(c => !clips.includes(c)),
    ...(vae && !vaes.includes(vae) ? [vae] : []),
    ...modelsOf(base).filter(m => !installed.has(m)),
  ]
  if (missing.length) { i2iSkip++; continue }

  const d = defaultsFor(base, model)
  const wf = instantiate(def, {
    model, positive: 'a test prompt', negative: d.negative, seed: 1,
    steps: d.steps, cfg: d.cfg, width: d.width, height: d.height,
    sampler: d.sampler, scheduler: d.scheduler,
    image: 'example.png', denoise: 0.55, megapixels: 1.0,
  })

  const errs: string[] = []
  for (const [id, node] of Object.entries(wf)) {
    const nd = info[node.class_type]
    if (!nd) { errs.push(`node ${id}: unknown class "${node.class_type}"`); continue }
    const req = nd.input?.required ?? {}
    const known = { ...req, ...(nd.input?.optional ?? {}) }
    for (const k of Object.keys(req)) {
      if (!(k in node.inputs)) errs.push(`node ${id} (${node.class_type}): missing required "${k}"`)
    }
    for (const [k, v] of Object.entries(node.inputs)) {
      if (!(k in known)) { errs.push(`node ${id} (${node.class_type}): unexpected input "${k}"`); continue }
      const isLink = Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && typeof v[1] === 'number'
      if (isLink) {
        if (!wf[(v as any)[0]]) errs.push(`node ${id}: "${k}" -> missing node ${(v as any)[0]}`)
        continue
      }
      if (Array.isArray(v)) continue
      const opts = optionsOf(known[k])
      if (opts && !opts.includes(v as string)) {
        errs.push(`node ${id} (${node.class_type}): "${k}"="${v}" not in enum`)
      }
    }
  }
  // the source image and its scaling must actually reach the graph
  if (!def.bindings.image?.length) errs.push('no "image" binding - the source image would be ignored')
  if (!def.bindings.denoise?.length) errs.push('no "denoise" binding - strength would do nothing')
  if (!Object.values(wf).some(n => n.class_type === 'ImageScaleToTotalPixels')) {
    errs.push('no fit node - a full-resolution source would reach the VAE')
  }

  if (errs.length) { i2iFail++; console.log(`FAIL  ${def.label}`); errs.forEach(e => console.log(`        ${e}`)) }
  else { i2iOk++; console.log(`OK    ${def.label}  (${Object.keys(wf).length} nodes)`) }
}

// ---------------------------------------------------------------------------
// Quality derivations.
//
// Region refine, face and hand detailing, the hires fix and LoRA stacks are all
// built by rewiring a family graph at run time. None of them is hand written and
// none is in the registry, so this is the only place their node classes, enum
// values and links are ever checked. Every one of them also has to be checked
// per family, because the derivation succeeds or returns null depending on what
// shape the family's graph happens to be.
// ---------------------------------------------------------------------------

/** The same node-by-node check the two loops above run, as one function. */
function checkGraph(wf: ApiWorkflow): string[] {
  const errs: string[] = []
  for (const [id, node] of Object.entries(wf)) {
    const nd = info[node.class_type]
    if (!nd) { errs.push(`node ${id}: unknown class "${node.class_type}"`); continue }
    const req = nd.input?.required ?? {}
    const known = { ...req, ...(nd.input?.optional ?? {}) }
    for (const k of Object.keys(req)) {
      if (!(k in node.inputs)) errs.push(`node ${id} (${node.class_type}): missing required "${k}"`)
    }
    for (const [k, v] of Object.entries(node.inputs)) {
      if (!(k in known)) { errs.push(`node ${id} (${node.class_type}): unexpected input "${k}"`); continue }
      const isLink = Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && typeof v[1] === 'number'
      if (isLink) {
        if (!wf[(v as any)[0]]) errs.push(`node ${id}: "${k}" -> missing node ${(v as any)[0]}`)
        continue
      }
      if (Array.isArray(v)) {
        for (const el of v as any[]) {
          if (Array.isArray(el) && typeof el[0] === 'string' && !wf[el[0]]) {
            errs.push(`node ${id}: "${k}" entry -> missing node ${el[0]}`)
          }
        }
        continue
      }
      const opts = optionsOf(known[k])
      if (opts && !opts.includes(v as string)) {
        const shown = opts.length > 6 ? `${opts.slice(0, 6).join(', ')}, …` : opts.join(', ')
        errs.push(`node ${id} (${node.class_type}): "${k}"="${v}" not in [${shown}]`)
      }
    }
  }
  return errs
}

/** Every node class a graph uses, for the "did the derivation actually fire" check. */
const classesOf = (wf: ApiWorkflow) => new Set(Object.values(wf).map(n => n.class_type))

const runnable = (def: FamilyDef): boolean => {
  const { clip, vae } = sidecarsOf(def)
  return !(
    clip.some(c => !clips.includes(c)) ||
    (vae && !vaes.includes(vae)) ||
    modelsOf(def).some(m => !installed.has(m))
  )
}

const baseParamsFor = (def: FamilyDef, model: string) => {
  const d = defaultsFor(def, model)
  return {
    model, positive: 'a test prompt', negative: d.negative, seed: 1,
    steps: d.steps, cfg: d.cfg, width: d.width, height: d.height,
    sampler: d.sampler, scheduler: d.scheduler,
  }
}

let qOk = 0, qFail = 0, qSkip = 0

/** Report one derived variant. `expect` are node classes that prove it fired. */
function checkVariant(
  name: string,
  wf: ApiWorkflow | null,
  expect: string[],
  extra: string[] = [],
): void {
  if (!wf) { console.log(`SKIP  ${name}: the family cannot carry this pass`); qSkip++; return }
  const errs = [...checkGraph(wf), ...extra]
  const seen = classesOf(wf)
  for (const cls of expect) {
    if (!seen.has(cls)) errs.push(`derivation did not fire: no ${cls} in the graph`)
  }
  if (errs.length) { qFail++; console.log(`FAIL  ${name}`); errs.forEach(e => console.log(`        ${e}`)) }
  else { qOk++; console.log(`OK    ${name}  (${Object.keys(wf).length} nodes)`) }
}

// A LoRA file that really exists, so the loader enum check is a real check and
// not a tautology. Fit is irrelevant here: this validates wiring, not effect.
const testLora = loras[0] ?? null

for (const def of FAMILIES) {
  if (def.mode !== 'image') continue
  if (!runnable(def)) continue
  const model = def.models.find(m => installed.has(m)) ?? def.models[0]
  const base = baseParamsFor(def, model)
  const caps = capabilitiesOf(def)

  // --- region refine: the anatomy fix -------------------------------------
  const refine = deriveRefine(def)
  if (refine !== null !== caps.refine) {
    console.log(`FAIL  ${def.label}: capabilitiesOf().refine disagrees with deriveRefine()`)
    qFail++
  }
  checkVariant(
    `${def.label}: region refine`,
    refine
      ? instantiateRefine(refine, { ...base, image: 'example.png' }, {
          image: 'example.png',
          mask: 'example.png',
          crop: { x: 64, y: 64, width: 320, height: 448 },
          target: { width: 736, height: 1024 },
          denoise: 0.45, grow: 12, feather: 16,
          prompt: 'the region, described',
        })
      : null,
    // The whole point of the pass, in four nodes: crop it, enlarge it with a
    // real upscaler, denoise only the masked cells, put it back where it came
    // from. If any one of these is missing the pass is not doing its job.
    ['ImageCrop', 'ImageUpscaleWithModel', 'SetLatentNoiseMask', 'ImageCompositeMasked'],
    refine
      ? [
          ...(refine.derived.extra.maskImage?.length ? [] : ['no maskImage binding, so the drawn mask would be ignored']),
          ...(refine.derived.extra.cropX?.length ? [] : ['no cropX binding, so the crop rectangle would stay at its default']),
          ...(refine.derived.extra.targetWidth?.length ? [] : ['no targetWidth binding, so the crop would not be enlarged']),
          ...(refine.bindings.denoise?.length ? [] : ['no denoise binding, so strength would do nothing']),
          ...(Object.values(refine.graph).some(n => n.class_type === 'LoadImageMask' && n.inputs.channel === 'red')
            ? []
            : ['the mask is not read on the red channel; an opaque white-on-black PNG would read back empty']),
        ]
      : [],
  )

  // --- automatic detailing, the two regions a detector covers -------------
  for (const target of ['face', 'hand'] as const) {
    const d = deriveAutoDetail(def, target)
    checkVariant(
      `${def.label}: ${target} detail`,
      d ? instantiateDerived(d, base) : null,
      ['UltralyticsDetectorProvider', 'FaceDetailer'],
      d
        ? Object.values(d.graph).some(
            n => n.class_type === 'UltralyticsDetectorProvider' &&
                 String(n.inputs.model_name).includes(target === 'face' ? 'face_yolov8m' : 'hand_yolov8s'),
          )
          ? []
          : [`the detector is not the ${target} model`]
        : [],
    )
  }

  // --- hires fix -----------------------------------------------------------
  const hires = deriveHiresFix(def)
  checkVariant(
    `${def.label}: hires fix`,
    hires ? instantiateDerived(hires, base, { hiresSteps: hiresStepsFor(base.steps) }) : null,
    ['LatentUpscaleBy'],
    hires
      ? [
          // Two samplers, or the second pass is not there.
          Object.values(hires.graph).filter(n => n.class_type === 'KSampler').length >= 2
            ? ''
            : 'only one KSampler, so there is no second pass',
          hires.derived.extra.hiresDenoise?.length ? '' : 'no hiresDenoise binding',
        ].filter(Boolean)
      : [],
  )

  // --- LoRA stack ----------------------------------------------------------
  if (!testLora) { qSkip++; console.log(`SKIP  ${def.label}: LoRA stack (no LoRA files installed)`) }
  else {
    const stacked = withLoras(def, [{ name: testLora, strength: 0.8 }])
    checkVariant(
      `${def.label}: LoRA stack`,
      stacked ? instantiateDerived(stacked, base) : null,
      [],
      stacked
        ? [
            stacked.derived.loraNodes.length === 1 ? '' : 'the LoRA chain is not one node long',
            Object.values(stacked.graph).some(
              n => n.class_type === 'LoraLoader' || n.class_type === 'LoraLoaderModelOnly',
            )
              ? ''
              : 'no LoRA loader in the graph',
            // The binding must still resolve, or the style picker would write
            // the model name into a node that is no longer there.
            (stacked.bindings.model ?? []).every(([id]) => id in stacked.graph)
              ? ''
              : 'the model binding points at a node the LoRA chain removed',
          ].filter(Boolean)
        : [],
    )
  }

  // --- refine plus LoRAs, which is how the anatomy fix actually ships ------
  if (refine && testLora) {
    const both = withLoras(refine, [{ name: testLora, strength: 0.8 }])
    checkVariant(
      `${def.label}: region refine with a LoRA`,
      both
        ? instantiateRefine(both, { ...base, image: 'example.png' }, {
            image: 'example.png', mask: 'example.png',
            crop: { x: 0, y: 0, width: 256, height: 256 },
            target: { width: 1024, height: 1024 },
            denoise: 0.45, grow: 12, feather: 16, prompt: 'the region',
          })
        : null,
      ['SetLatentNoiseMask', 'ImageCompositeMasked'],
      // Stacking must not lose the refine knobs. This is the exact mistake a
      // rewrite of priorDerivation would make.
      both
        ? [both.derived.extra.cropX?.length ? '' : 'the crop bindings were lost when the LoRA was applied'].filter(Boolean)
        : [],
    )
  }
}

// ---------------------------------------------------------------------------
// Continuation derivations.
//
// The reel desk never queues a registry graph. It queues one lib/continuation.ts
// builds at run time: a tap welded onto the decode so a shot publishes its own
// last frame, and a LoadImage welded onto the next shot's opening slot so that
// shot starts where the last one stopped. None of that wiring is in the
// registry and none of it is hand verified, which is the same argument the
// quality section above makes for itself.
//
// Three things are checked here that no other section can check:
//
//   the tap     ImageFromBatch at batch_index -1, length 1, feeding a SaveImage
//               under CHAIN_PREFIX. Get the index wrong and every handoff is
//               the FIRST frame, so a reel silently walks backwards. Nothing
//               about that failure looks like an error at queue time.
//   the seam    shot 2 built through shotPlan and instantiateShot against shot
//               1's real annotated ref, which is the exact path engine.tsx
//               runs. A plan that type checks and a seam that wires are two
//               different claims.
//   the placeholder
//               no LoadImage may reach the server still holding
//               PLACEHOLDER_IMAGE. A graph that carries its own loader and is
//               run without an image leaves 'example.png' in it, and the only
//               symptom is an opaque node validation error minutes later. This
//               is the shape of review defect A13; it cannot fire while the
//               14B families are skipped for missing weights, and it fires the
//               moment their filenames are corrected.
//
// An annotated ref ("switchgen/chain/x_00001_.png [output]") is deliberately
// NOT in LoadImage's combo list: ComfyUI validates it through
// folder_paths.exists_annotated_filepath instead of against the enum. So the
// enum check is relaxed for exactly that one shape, and for nothing else.
// ---------------------------------------------------------------------------

let cOk = 0, cFail = 0, cSkip = 0

const ANNOTATED = /^.+ \[(output|input|temp)\]$/

/** checkGraph, minus the one enum complaint that is not a defect. See above. */
function checkChainGraph(wf: ApiWorkflow): string[] {
  const annotated = new Set(
    Object.entries(wf)
      .filter(([, n]) => n.class_type === 'LoadImage' && ANNOTATED.test(String(n.inputs.image)))
      .map(([id]) => id),
  )
  return checkGraph(wf).filter(e => !annotated.has(e.split(' ')[1] ?? '') || !e.includes('"image"='))
}

function checkChain(name: string, wf: ApiWorkflow | null, expect: string[], extra: string[] = []): void {
  if (!wf) { console.log(`SKIP  ${name}: the family cannot carry this shot`); cSkip++; return }
  const errs = [...checkChainGraph(wf), ...extra]
  const seen = classesOf(wf)
  for (const cls of expect) if (!seen.has(cls)) errs.push(`derivation did not fire: no ${cls} in the graph`)
  // Nothing may reach the server still holding the inserted loader's stand-in.
  for (const [id, n] of Object.entries(wf)) {
    if (n.class_type === 'LoadImage' && n.inputs.image === PLACEHOLDER_IMAGE) {
      errs.push(`node ${id}: LoadImage still holds the placeholder ${PLACEHOLDER_IMAGE}, so the shot would open on the wrong frame`)
    }
  }
  if (errs.length) { cFail++; console.log(`FAIL  ${name}`); errs.forEach(e => console.log(`        ${e}`)) }
  else { cOk++; console.log(`OK    ${name}  (${Object.keys(wf).length} nodes)`) }
}

/** The tap is the whole handoff. Check the numbers, not just that it is there. */
function tapErrors(graph: Record<string, { class_type: string; inputs: Record<string, unknown> }>): string[] {
  const pick = graph[NODE_IDS.lastFrame]
  const save = graph[NODE_IDS.frameSave]
  const errs: string[] = []
  if (!pick) return [`no ${NODE_IDS.lastFrame} node, so the shot publishes no handoff frame`]
  if (!save) return [`no ${NODE_IDS.frameSave} node, so the handoff frame is never written to disk`]
  if (pick.inputs.batch_index !== -1) {
    errs.push(`the tap reads batch_index ${String(pick.inputs.batch_index)}, not -1: the reel would hand on the wrong frame`)
  }
  if (pick.inputs.length !== 1) errs.push(`the tap takes ${String(pick.inputs.length)} frames, not 1`)
  if (!String(save.inputs.filename_prefix ?? '').startsWith(CHAIN_PREFIX)) {
    errs.push(`the handoff is written under "${String(save.inputs.filename_prefix)}", not ${CHAIN_PREFIX}, so chainFrameOf would not find it`)
  }
  const src = pick.inputs.image
  const from = Array.isArray(src) ? graph[String(src[0])] : undefined
  if (!from || !/^VAEDecode/.test(from.class_type)) {
    errs.push('the tap does not read a VAE decode, so the handoff frame is not a decoded video frame')
  }
  return errs
}

/**
 * A handoff filename in the annotated form ComfyUI resolves through
 * folder_paths.exists_annotated_filepath. Used as the image everywhere below,
 * rather than PLACEHOLDER_IMAGE: feeding the placeholder back in would make the
 * placeholder check tautological, since a binding that wrote it and a binding
 * that never fired leave the graph in the same state.
 */
const HANDOFF = `${CHAIN_PREFIX}/${NODE_IDS.frameSave}_00001_.png [output]`

const videoParamsFor = (def: FamilyDef, model: string): Params => {
  const d = defaultsFor(def, model)
  return { ...baseParamsFor(def, model), length: d.length || undefined, fps: d.fps || undefined }
}

for (const def of FAMILIES) {
  if (def.mode !== 'video') continue
  if (!runnable(def)) {
    console.log(`SKIP  ${def.label}: continued shots (weights not installed)`)
    cSkip++
    continue
  }
  const model = def.models.find(m => installed.has(m)) ?? def.models[0]!
  const params = videoParamsFor(def, model)

  // --- the opening shot: a plain family graph plus the tap -----------------
  const tap = deriveChainTap(def)
  checkChain(
    `${def.label}: opening shot with a handoff tap`,
    tap ? instantiate(tap, params) : null,
    ['ImageFromBatch', 'SaveImage'],
    tap ? tapErrors(tap.graph) : [],
  )

  // --- the continued shot: opens on a supplied frame -----------------------
  const cont = deriveContinuation(def)
  checkChain(
    `${def.label}: continued shot`,
    cont ? instantiate(cont, { ...params, image: HANDOFF }) : null,
    ['LoadImage', 'ImageFromBatch'],
    cont
      ? [
          ...tapErrors(cont.graph),
          // The binding is what carries the handoff filename in. Without it
          // instantiate writes nothing and the loader keeps its stand-in.
          (cont.bindings.image ?? []).length ? '' : 'no "image" binding, so the handoff frame would never reach the loader',
          (cont.bindings.image ?? []).every(([id]) => id in cont.graph)
            ? ''
            : 'the "image" binding points at a node that is not in the graph',
          Object.values(cont.graph).some(
            n => ['Wan22ImageToVideoLatent', 'WanImageToVideo', 'WanFirstLastFrameToVideo'].includes(n.class_type) &&
                 Array.isArray(n.inputs.start_image),
          )
            ? ''
            : 'nothing is wired into a start_image slot, so the opening frame would be ignored',
        ].filter(Boolean)
      : [],
  )

  // --- the bookend: pinned at both ends ------------------------------------
  const book = deriveBookend(def)
  checkChain(
    `${def.label}: first and last frame`,
    book ? instantiate(book, { ...params, image: HANDOFF }) : null,
    ['WanFirstLastFrameToVideo'],
    book
      ? [
          Object.values(book.graph).some(
            n => n.class_type === 'WanFirstLastFrameToVideo' && Array.isArray(n.inputs.end_image),
          )
            ? ''
            : 'no end_image link, so the shot would not be pinned at the end',
          // The end frame travels as a graph write, not a binding, so the
          // bookend's own loader is exempt from the placeholder rule here.
          '',
          // The swap is only safe because the two nodes agree on output slots.
          Object.values(book.graph).some(n => n.class_type === 'WanImageToVideo')
            ? 'a WanImageToVideo survived the swap, so two conditioning paths are live at once'
            : '',
        ].filter(Boolean)
      : [],
  )

  // --- the VACE shot: gated until a VACE family is registered --------------
  const vace = deriveVaceShot(def)
  checkChain(
    `${def.label}: VACE reference shot`,
    vace ? instantiate(vace, { ...params, image: HANDOFF }) : null,
    ['WanVaceToVideo', 'TrimVideoLatent'],
    vace ? tapErrors(vace.graph) : [],
  )

  // --- the seam, through the live path -------------------------------------
  // shotPlan then instantiateShot is exactly what components/reel/engine.tsx
  // runs. Two shots is the smallest reel that has a seam in it.
  const plan = shotPlan({
    base: def,
    params,
    shots: [{ prompt: 'the opening shot' }, { prompt: 'the shot that continues it' }],
  })
  const [first, second] = plan.jobs
  if (!first || !second) {
    console.log(`FAIL  ${def.label}: two-shot seam: shotPlan returned ${plan.jobs.length} jobs, not 2`)
    cFail++
  } else {
    checkChain(`${def.label}: two-shot seam, shot 1`, instantiateShot(first, null), ['ImageFromBatch'])

    // The handoff shot 1 actually publishes, in the annotated form LoadImage
    // resolves through exists_annotated_filepath.
    let shot2: ApiWorkflow | null = null
    const seamErrs: string[] = []
    try { shot2 = instantiateShot(second, HANDOFF) } catch (e) {
      seamErrs.push(`instantiateShot threw for the continued shot: ${(e as Error).message}`)
    }
    checkChain(
      `${def.label}: two-shot seam, shot 2 continues shot 1`,
      shot2,
      ['ImageFromBatch'],
      [
        ...seamErrs,
        ...(shot2
          ? [
              second.start.from === 'previous'
                ? ''
                : `shot 2 starts from "${second.start.from}", so the reel would not continue`,
              Object.values(shot2).some(n => n.class_type === 'LoadImage' && n.inputs.image === HANDOFF)
                ? ''
                : 'the handoff filename never reached a LoadImage',
              // Two shots must not overwrite each other's clip.
              first.outputPrefix !== second.outputPrefix
                ? ''
                : `both shots write to ${first.outputPrefix}, so the second would overwrite the first`,
            ].filter(Boolean)
          : []),
      ],
    )
  }
}

console.log(`\n${ok} ok, ${fail} failed, ${skip} skipped (base graphs)`)
console.log(`${i2iOk} ok, ${i2iFail} failed, ${i2iSkip} skipped (image-to-image variants)`)
console.log(`${qOk} ok, ${qFail} failed, ${qSkip} skipped (quality derivations)`)
console.log(`${cOk} ok, ${cFail} failed, ${cSkip} skipped (continuation derivations)`)
fail += i2iFail + qFail + cFail
process.exit(fail ? 1 : 0)
