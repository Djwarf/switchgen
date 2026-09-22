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
 *
 * Before any of that, every node class those graphs use is looked up, whether
 * or not the family's files are installed, so a missing node pack is named as
 * a missing node pack. The exit code is 1 when anything FAILs, 2 when ComfyUI
 * cannot be reached, and 0 otherwise; a SKIP never fails the run.
 */
import { inventoryFrom, missingFilesFor } from '../src/lib/availability.ts'
import { FAMILIES, IMG2IMG, defaultsFor, instantiate, type Params } from '../src/lib/workflows'
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
  needsOpeningFrame,
  setEndImage,
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
  withVideoLoras,
} from '../src/lib/refine'

const COMFY = process.env.COMFY_URL ?? 'http://127.0.0.1:8188'

function optionsOf(spec: any): string[] | null {
  if (Array.isArray(spec?.[0])) return spec[0] as string[]
  if (spec?.[0] === 'COMBO' && Array.isArray(spec?.[1]?.options)) return spec[1].options
  return null
}

// A refused connection used to end the run with fetch's own stack trace, and
// the exit code 1 that means a graph failed. It is a sentence and exit code 2,
// like the HTTP error below it.
let res: Response
try {
  res = await fetch(`${COMFY}/object_info`)
} catch (e) {
  const cause = (e as { cause?: { code?: string; message?: string } }).cause
  console.error(`Cannot reach ComfyUI at ${COMFY} (${cause?.code ?? cause?.message ?? (e as Error).message}).`)
  console.error('Start it, or set COMFY_URL to where it listens. `bin/switchgen validate` reads COMFY_URL from .env; `npm run validate` reads only the environment.')
  process.exit(2)
}
if (!res.ok) { console.error(`Cannot reach ComfyUI at ${COMFY} (HTTP ${res.status})`); process.exit(2) }
const info: Record<string, any> = await res.json()

// The same inventory the desks read, so what validates here is what is offered
// there: it consults CLIPLoaderGGUF as well as CLIPLoader, which is what keeps a
// family with a quantised encoder from reading as 'missing'.
const inv = inventoryFrom(info)
const installed = inv.weights
const loras = [...inv.loras]

let fail = 0, skip = 0, ok = 0

// ---------------------------------------------------------------------------
// Node classes, and the packs that provide them.
//
// Everything below skips a family whose files ComfyUI does not list, and the
// file list is read off the loader nodes themselves. So a missing pack never
// looked like a missing pack: without ComfyUI-GGUF there is no UnetLoaderGGUF
// to list the .gguf files, every quantised family was skipped as "missing" its
// weights, the class check that would have named the node never ran, and the
// run exited 0. Here every class any graph uses is looked up first, installed
// files or not, including the derived graphs, which need no files to build.
// ---------------------------------------------------------------------------

/** The three packs README asks for, by the classes of theirs the graphs use. */
const PACKS: Record<string, { url: string; classes: string[] }> = {
  'ComfyUI-GGUF': {
    url: 'https://github.com/city96/ComfyUI-GGUF',
    classes: ['UnetLoaderGGUF', 'CLIPLoaderGGUF', 'DualCLIPLoaderGGUF', 'TripleCLIPLoaderGGUF'],
  },
  'ComfyUI-Impact-Pack': {
    url: 'https://github.com/ltdrdata/ComfyUI-Impact-Pack',
    classes: ['FaceDetailer', 'ImpactGaussianBlurMask'],
  },
  'ComfyUI-Impact-Subpack': {
    url: 'https://github.com/ltdrdata/ComfyUI-Impact-Subpack',
    classes: ['UltralyticsDetectorProvider'],
  },
}
const packOf = (cls: string): string | null =>
  Object.entries(PACKS).find(([, p]) => p.classes.includes(cls))?.[0] ?? (/^Impact/.test(cls) ? 'ComfyUI-Impact-Pack' : null)

/** Every graph the app can send for a family, built without any of its files. */
function graphsOf(def: FamilyDef): FamilyDef['graph'][] {
  const out: (FamilyDef['graph'] | undefined)[] = [def.graph, IMG2IMG[def.id]?.graph]
  if (def.mode === 'image') {
    out.push(deriveRefine(def)?.graph, deriveAutoDetail(def, 'face')?.graph, deriveAutoDetail(def, 'hand')?.graph, deriveHiresFix(def)?.graph)
  }
  if (def.mode === 'video') {
    out.push(deriveChainTap(def)?.graph, deriveContinuation(def)?.graph, deriveBookend(def)?.graph, deriveVaceShot(def)?.graph)
  }
  return out.filter((g): g is FamilyDef['graph'] => !!g)
}

const absent = new Map<string, Set<string>>()
for (const def of FAMILIES) {
  for (const graph of graphsOf(def)) {
    for (const node of Object.values(graph)) {
      if (info[node.class_type]) continue
      if (!absent.has(node.class_type)) absent.set(node.class_type, new Set())
      absent.get(node.class_type)!.add(def.label)
    }
  }
}
let nodeFail = 0
const byPack = new Map<string | null, string[]>()
for (const cls of absent.keys()) {
  const pack = packOf(cls)
  byPack.set(pack, [...(byPack.get(pack) ?? []), cls])
}
for (const [pack, classes] of byPack) {
  const needers = [...new Set(classes.flatMap(c => [...absent.get(c)!]))]
  if (pack) {
    console.log(`FAIL  node pack ${pack} is not installed: ${classes.join(', ')} ${classes.length === 1 ? 'is' : 'are'} missing`)
    console.log(`        install it from ${PACKS[pack]?.url ?? 'its repository'} and restart ComfyUI`)
  } else {
    console.log(`FAIL  ${classes.join(', ')} ${classes.length === 1 ? 'is' : 'are'} not in this ComfyUI, which may be older than these graphs need`)
  }
  console.log(`        needed by ${needers.join('; ')}`)
  nodeFail++
}
/** When a family's files read as missing only because the pack that lists them is. */
const unlisted = (missing: string[]) =>
  missing.some(f => /\.gguf$/i.test(f)) && !info.UnetLoaderGGUF
    ? 'the ComfyUI-GGUF node pack is not installed, so ComfyUI cannot list its .gguf files'
    : null

for (const def of FAMILIES) {
  const model = def.models.find(m => installed.has(m)) ?? def.models[0]
  const missing = missingFilesFor(def, inv)
  if (missing.length) { console.log(`SKIP  ${def.label}: ${unlisted(missing) ?? `missing ${missing.join(', ')}`}`); skip++; continue }

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
      // LoadImage.image is written at run time with an uploaded filename or an
      // annotated output path ("sub/name.png [output]"), which ComfyUI resolves
      // but which never appears in the static enum. Checking it here would fail
      // every graph that does the correct thing.
      // LoadImageMask.image is the same: the region bench uploads the mask as it
      // queues the pass, so it is never in the enum either.
      if ((node.class_type === 'LoadImage' || node.class_type === 'LoadImageMask') && k === 'image') continue
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
  // A binding to a node that is not there writes nothing, and one to an input
  // the node does not take is refused at queue time. Either way the control
  // on screen changes nothing, and nothing above notices, because instantiate
  // skips a node it cannot find.
  for (const [key, binds] of Object.entries(def.bindings)) {
    for (const [id, input] of binds ?? []) {
      const node = wf[id]
      if (!node) { errs.push(`the "${key}" binding points at node ${id}, which is not in the graph`); continue }
      const nd = info[node.class_type]
      if (nd && !(input in { ...(nd.input?.required ?? {}), ...(nd.input?.optional ?? {}) })) {
        errs.push(`the "${key}" binding writes "${input}" on node ${id} (${node.class_type}), which takes no such input`)
      }
    }
  }
  // The frame rate a clip is filed and played at is the one its encoder
  // wrote. A video family whose fps binding reaches only the conditioning
  // quotes a rate the file does not have.
  if (def.mode === 'video') {
    const encoder = (def.bindings.fps ?? []).some(([id, input]) => input === 'fps' && /^Save/.test(wf[id]?.class_type ?? ''))
    if (!encoder) errs.push('the "fps" binding does not reach a Save node\'s fps input, so the clip is not written at the rate the desk shows')
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
  const missing = missingFilesFor(def, inv)
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
      // LoadImage.image is written at run time with an uploaded filename or an
      // annotated output path ("sub/name.png [output]"), which ComfyUI resolves
      // but which never appears in the static enum. Checking it here would fail
      // every graph that does the correct thing.
      // LoadImageMask.image is the same: the region bench uploads the mask as it
      // queues the pass, so it is never in the enum either.
      if ((node.class_type === 'LoadImage' || node.class_type === 'LoadImageMask') && k === 'image') continue
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
      // LoadImage.image is written at run time with an uploaded filename or an
      // annotated output path ("sub/name.png [output]"), which ComfyUI resolves
      // but which never appears in the static enum. Checking it here would fail
      // every graph that does the correct thing.
      // LoadImageMask.image is the same: the region bench uploads the mask as it
      // queues the pass, so it is never in the enum either.
      if ((node.class_type === 'LoadImage' || node.class_type === 'LoadImageMask') && k === 'image') continue
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

const runnable = (def: FamilyDef): boolean => missingFilesFor(def, inv).length === 0

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
  const refines = refine !== null
  if (refines !== caps.refine) {
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

/**
 * `carries` says whether the family's own graph has what the shot needs, read
 * independently of the derivation under test. A null derivation for a family
 * that carries the shot is a regression, not a skip: it is how the reel would
 * quietly lose "continue from the last frame" with every check still green.
 */
function checkChain(name: string, wf: ApiWorkflow | null, expect: string[], extra: string[] = [], carries = false): void {
  if (!wf && carries) {
    cFail++
    console.log(`FAIL  ${name}`)
    console.log('        the derivation returned nothing, but the family\'s graph has what this shot needs')
    return
  }
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

/**
 * Nodes with a start_image slot, as the live schema has them. Stated here
 * rather than imported from lib/continuation.ts, so a change to that module's
 * own list cannot also change what this check expects.
 */
const FRAME_HOSTS = ['Wan22ImageToVideoLatent', 'WanImageToVideo', 'WanFirstLastFrameToVideo']

const videoParamsFor = (def: FamilyDef, model: string): Params => {
  const d = defaultsFor(def, model)
  return { ...baseParamsFor(def, model), length: d.length || undefined, fps: d.fps || undefined }
}

for (const def of FAMILIES) {
  if (def.mode !== 'video') continue
  if (!runnable(def)) {
    console.log(`SKIP  ${def.label}: continued shots (${unlisted(missingFilesFor(def, inv)) ?? 'weights not installed'})`)
    cSkip++
    continue
  }
  const model = def.models.find(m => installed.has(m)) ?? def.models[0]!
  const params = videoParamsFor(def, model)
  const has = classesOf(def.graph)
  const carries = {
    // Every video graph decodes frames, so every one can publish its last.
    tap: [...has].some(c => /^VAEDecode/.test(c)),
    continuation: FRAME_HOSTS.some(c => has.has(c)),
    bookend: has.has('WanImageToVideo') || has.has('WanFirstLastFrameToVideo'),
    vace: has.has('WanVaceToVideo') || def.models.some(m => /vace/i.test(m)),
  }

  // --- the opening shot: a plain family graph plus the tap -----------------
  const tap = deriveChainTap(def)
  checkChain(
    `${def.label}: opening shot with a handoff tap`,
    tap ? instantiate(tap, needsOpeningFrame(def) ? { ...params, image: HANDOFF } : params) : null,
    ['ImageFromBatch', 'SaveImage'],
    tap ? tapErrors(tap.graph) : [],
    carries.tap,
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
    carries.continuation,
  )

  // --- the bookend: pinned at both ends ------------------------------------
  const book = deriveBookend(def)
  // instantiate() writes the opening frame through the image binding; the closing
  // frame is a separate node, so build it the way the reel UI does.
  const bookendWf = book ? instantiate(book, { ...params, image: HANDOFF }) : null
  if (bookendWf) setEndImage(bookendWf, HANDOFF)
  checkChain(
    `${def.label}: first and last frame`,
    book ? bookendWf : null,
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
    carries.bookend,
  )

  // --- the VACE shot: gated until a VACE family is registered --------------
  const vace = deriveVaceShot(def)
  checkChain(
    `${def.label}: VACE reference shot`,
    vace ? instantiate(vace, { ...params, image: HANDOFF }) : null,
    ['WanVaceToVideo', 'TrimVideoLatent'],
    vace ? tapErrors(vace.graph) : [],
    carries.vace,
  )

  // --- the seam, through the live path -------------------------------------
  // shotPlan then instantiateShot is exactly what components/reel/engine.tsx
  // runs. Two shots is the smallest reel that has a seam in it.
  const plan = shotPlan({
    base: def,
    params,
    // an image-to-video family cannot open cold, so give shot 1 a frame the way the UI must
      shots: [{ prompt: 'the opening shot', ...(needsOpeningFrame(def) ? { startImage: 'opening-frame.png' } : {}) }, { prompt: 'the shot that continues it' }],
  })
  const [first, second] = plan.jobs
  if (!first || !second) {
    console.log(`FAIL  ${def.label}: two-shot seam: shotPlan returned ${plan.jobs.length} jobs, not 2`)
    cFail++
  } else if (second.start.from !== 'previous') {
    // A family with no slot for a frame cannot continue one, and wan22-14b-t2v
    // is exactly that: correct, so a skip. A family that has the slot and
    // still plans shot 2 to start elsewhere has lost the reel's whole promise.
    if (carries.continuation) {
      console.log(`FAIL  ${def.label}: two-shot seam`)
      console.log(`        shot 2 starts from "${second.start.from}", so the reel would not continue`)
      cFail++
    } else {
      console.log(`SKIP  ${def.label}: two-shot seam: this family cannot continue from a frame`)
      cSkip += 2
    }
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

// ---------------------------------------------------------------------------
// Add-on chains on video families.
//
// withVideoLoras() chains one loader on a one-model family and one chain per
// half on the Wan 2.2 14B pairs, wired in front of any add-on the family
// already carries. Every runnable video family is chained with one installed
// Wan add-on and the result checked node by node, plus the two things the
// derivation promises: the right number of loaders appeared, and every
// sampler reaches its weights through one of them. A count alone passes a
// loader added and left dangling, which ComfyUI prunes without a word, so the
// add-on the reader chose would do nothing.
// ---------------------------------------------------------------------------

let vOk = 0, vFail = 0, vSkip = 0
const videoLora = loras.find(l => /wan/i.test(l)) ?? loras[0]
for (const def of FAMILIES) {
  if (def.mode !== 'video') continue
  if (!runnable(def)) { vSkip++; continue }
  if (!videoLora) { console.log(`SKIP  ${def.label}: add-on chain: no add-on installed to chain`); vSkip++; continue }
  const before = Object.values(def.graph).filter(n => n.class_type === 'LoraLoaderModelOnly').length
  const derived = withVideoLoras(def, [{ name: videoLora, strength: 0.8 }])
  if (!derived) { console.log(`SKIP  ${def.label}: add-on chain: the family cannot carry one`); vSkip++; continue }
  const model = def.models.find(m => installed.has(m)) ?? def.models[0]
  const d = defaultsFor(def, model)
  const wf = instantiate(derived, {
    ...baseParamsFor(def, model),
    length: d.length || undefined,
    fps: d.fps || undefined,
  })
  const errs = checkGraph(wf)
  const after = Object.values(wf).filter(n => n.class_type === 'LoraLoaderModelOnly').length
  const want = before + (def.dualModel ? 2 : 1)
  if (after !== want) errs.push(`expected ${want} LoraLoaderModelOnly nodes after chaining, found ${after}`)
  const added = new Set(derived.derived.loraNodes)
  for (const [id, n] of Object.entries(wf)) {
    if (!/Sampler|Guider|Scheduler/.test(n.class_type) || !('model' in n.inputs)) continue
    let cur = n.inputs.model as unknown
    let hops = 0
    let reached = false
    let through = false
    while (Array.isArray(cur) && hops++ < 12) {
      const up = wf[cur[0] as string]
      if (!up) { errs.push(`node ${id}: model chain reaches missing node ${cur[0]}`); break }
      if (added.has(cur[0] as string)) through = true
      if (/^(CheckpointLoaderSimple|UNETLoader|UnetLoaderGGUF)$/.test(up.class_type)) { reached = true; break }
      cur = up.inputs.model
    }
    if (!reached && !errs.some(e => e.startsWith(`node ${id}:`))) {
      errs.push(`node ${id} (${n.class_type}): its model chain never reaches a weights loader`)
    }
    // A scheduler reads the model for its sigmas only; the add-on has to be on
    // what samples.
    if (/Sampler|Guider/.test(n.class_type) && !through) {
      errs.push(`node ${id} (${n.class_type}): samples without the add-on, which sits on no path to it`)
    }
  }
  if (errs.length) {
    console.log(`FAIL  ${def.label}: add-on chain (${videoLora})`)
    for (const e of errs) console.log(`        ${e}`)
    vFail++
  } else {
    console.log(`OK    ${def.label}: add-on chain, ${want} loader${want === 1 ? '' : 's'} (${Object.keys(wf).length} nodes)`)
    vOk++
  }
}

console.log('')
if (nodeFail) console.log(`${nodeFail} failed (node classes)`)
console.log(`${ok} ok, ${fail} failed, ${skip} skipped (base graphs)`)
console.log(`${i2iOk} ok, ${i2iFail} failed, ${i2iSkip} skipped (image-to-image variants)`)
console.log(`${qOk} ok, ${qFail} failed, ${qSkip} skipped (quality derivations)`)
console.log(`${cOk} ok, ${cFail} failed, ${cSkip} skipped (continuation derivations)`)
console.log(`${vOk} ok, ${vFail} failed, ${vSkip} skipped (video add-on chains)`)
// Every tally, the video add-on chains included: they were printed and then
// left out of this sum, so a broken chain exited 0.
const failed = nodeFail + fail + i2iFail + qFail + cFail + vFail
process.exit(failed ? 1 : 0)
