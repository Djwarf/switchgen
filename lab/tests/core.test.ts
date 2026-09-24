/**
 * Core (TEST PLAN, B): parameters, graphs, N/A, cell ids, finalizing, the lab-only
 * shape, the suite checker, plans, the object_info checker and the lab folder,
 * for all 13 generation models plus the editing model. Nothing is sent: the
 * graphs are built in memory and checked against a stand-in object_info.
 */
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { BY_ID, familyOwning } from '../../src/lib/workflows.ts'
import { CHAIN_TOKEN as SERVER_TOKEN } from '../../server/runner/comfyRecord.mjs'
import { DEFAULTS, labEnv, cellPrefix, refRel, maskRel, runDir, REPO_ROOT } from '../core/env.ts'
import { labParams } from '../core/params.ts'
import { buildGraph, shapeFor } from '../core/graphs.ts'
import { detailOnPicture } from '../core/labShapes.ts'
import { naReason, REASONS } from '../core/na.ts'
import {
  CHAIN_TOKEN, SHIPPED_SUITES, cellIdOf, expand, finalizeGraph, maskKey, positionWords, refBindings, verifyCell,
} from '../core/cells.ts'
import { CHILD_CODED, checkSuite, defineSuite, descriptionProblem, mentionsPerson } from '../core/suite.ts'
import { checkGraph } from '../core/validate.ts'
import { SEED_TIMINGS, describePlan, makePlan, measuredTimings, planFrom, savePlan, sendOrder, startRefusal, suiteById } from '../core/plan.ts'
import type { DoneCell, RefInfo, Slot, Suite } from '../core/types.ts'
import calibration from '../suites/calibration.ts'
import core, { CORE, MODELS } from '../suites/first-pass.ts'
import extEdit from '../suites/ext-edit.ts'
import extRange from '../suites/ext-range.ts'
import { objectInfoFrom } from './fixtures/objectInfo.ts'
import { removeTemp, tempDir } from './helpers.ts'

afterAll(removeTemp)

const CAT: RefInfo = { id: 'cat', sha12: 'abcdef012345', ext: 'jpg', width: 3060, height: 4080, mask: false }
const SCENE: RefInfo = { id: 'scene', sha12: '0123456789ab', ext: 'png', width: 1600, height: 1200, mask: true, rect: { x: 100, y: 800, w: 300, h: 250 } }
const done = (ids: Iterable<string>, run = 'x'): Map<string, DoneCell> =>
  new Map([...ids].map((id) => [id, { cellId: id, rel: `.lab/cells/${id}_00001_.png`, durationMs: 1000, cold: false, cached: false, finishedAt: 1, run }]))

const slot = (over: Partial<Slot> = {}): Slot => ({ id: 's', block: 'following', shape: 'square', op: 't2i', text: 'A red apple on a table.', models: 'core', ...over })
const suite = (slots: Slot[], over: Partial<Suite> = {}): Suite => ({ ...core, id: 'test', slots, chains: [], ...over })

describe('env', () => {
  it('refuses a lab folder inside the repo, or inside outputs', () => {
    expect(() => labEnv({ SWITCHGEN_LAB_DIR: path.join(REPO_ROOT, 'lab-data') })).toThrow(/inside the repo/)
    expect(() => labEnv({ SWITCHGEN_LAB_DIR: '/tmp/x/outputs/lab', SWITCHGEN_OUTPUTS: '/tmp/x/outputs' })).toThrow(/inside the outputs/)
    expect(() => labEnv({ SWITCHGEN_LAB_DIR: '/tmp/lab', SWITCHGEN_OUTPUTS: path.join(REPO_ROOT, 'out') })).toThrow(/inside the repo/)
    const e = labEnv({ SWITCHGEN_LAB_DIR: '/tmp/lab-x', SWITCHGEN_OUTPUTS: '/tmp/out-x' })
    expect(e).toMatchObject({ labDir: '/tmp/lab-x', outputs: '/tmp/out-x', port: 5274, host: '127.0.0.1', appUrl: 'http://127.0.0.1:5273' })
    expect(cellPrefix('0123456789abcdef')).toBe('.lab/cells/0123456789abcdef')
    expect(refRel('abc', 'jpg')).toBe('.lab/refs/abc.jpg')
    expect(maskRel('abc')).toBe('.lab/refs/abc.mask.png')
    expect(() => runDir(e, '../x')).toThrow()
  })
})

describe('labParams', () => {
  const cfg: Record<string, number> = { noobai: 5, semireal: 5, wai: 6, pony: 7, oneObsession: 4, miaomiaoHarem: 4, miaomiaoRealskin: 4, chroma: 3.5, zbase: 4, krea2: 1, klein: 1, qwen21: 1, zturbo: 1, qwenEdit: 4 }
  for (const key of [...CORE, 'qwenEdit']) {
    it(`${key}: 28 steps, its own cfg, euler/simple, its prefix, clip skip and shift`, () => {
      const file = MODELS[key].file
      const def = familyOwning(file)!
      const p = labParams(def, file, slot(), core, 1001)
      expect(p).toMatchObject({ model: file, steps: 28, cfg: cfg[key], sampler: 'euler', scheduler: 'simple', seed: 1001, width: 1024, height: 1024 })
      const per = (def.perModel[file] ?? {}) as Record<string, unknown>
      if (typeof per.positivePrefix === 'string') expect(p.positive).toBe(per.positivePrefix + 'A red apple on a table.')
      else expect(p.positive).toBe('A red apple on a table.')
      expect(p.clipSkip).toBe(def.id === 'sdxl-illustrious' ? -2 : undefined)
      expect(p.shift).toBe(def.id === 'z-image' ? 3 : undefined)
    })
  }
  it('gives Pony its score tags and the home sampler arm', () => {
    const def = familyOwning(MODELS.pony.file)!
    const p = labParams(def, MODELS.pony.file, slot(), core, 7, { sampler: 'home' })
    expect(p.positive.startsWith('score_9, score_8_up, score_7_up, ')).toBe(true)
    expect([p.sampler, p.scheduler]).toEqual(['dpmpp_2m', 'karras'])
  })
  it('adds a negative after the house one, or alone when there is none', () => {
    const noob = familyOwning(MODELS.noobai.file)!
    expect(labParams(noob, MODELS.noobai.file, slot({ negativeAdd: 'balloons' }), core, 1).negative.endsWith(', balloons')).toBe(true)
    const z = familyOwning(MODELS.zbase.file)!
    expect(labParams(z, MODELS.zbase.file, slot({ negativeAdd: 'balloons' }), core, 1).negative).toBe('balloons')
  })
})

describe('buildGraph readback', () => {
  for (const key of CORE) {
    it(`${key}: the sampler reads the params`, () => {
      const file = MODELS[key].file
      const def = familyOwning(file)!
      const p = labParams(def, file, slot({ shape: 'wide' }), core, 3003)
      const b = buildGraph({ file, op: 't2i', params: p })
      if (!b.graph) throw new Error(b.reason)
      const g = b.graph
      if (key === 'klein') {
        const sch = Object.values(g).find((n) => n.class_type === 'Flux2Scheduler')!
        expect(sch.inputs).toMatchObject({ steps: 28, width: 1344, height: 768 })
        expect(Object.values(g).find((n) => n.class_type === 'CFGGuider')!.inputs.cfg).toBe(1)
        expect(Object.values(g).find((n) => n.class_type === 'RandomNoise')!.inputs.noise_seed).toBe(3003)
      } else {
        const ks = Object.values(g).find((n) => n.class_type === 'KSampler')!
        expect(ks.inputs).toMatchObject({ steps: 28, cfg: p.cfg, seed: 3003, sampler_name: 'euler', scheduler: 'simple' })
        const lat = Object.values(g).find((n) => /^Empty.*Latent/.test(n.class_type))!
        expect(lat.inputs).toMatchObject({ width: 1344, height: 768 })
      }
    })
  }
  it('runs hires with a 17-step second pass', () => {
    const file = MODELS.chroma.file
    const b = buildGraph({ file, op: 'hires', params: labParams(familyOwning(file)!, file, slot(), core, 1) })
    if (!b.graph) throw new Error(b.reason)
    const samplers = Object.values(b.graph).filter((n) => n.class_type === 'KSampler').map((n) => n.inputs.steps)
    expect(samplers.sort()).toEqual([17, 28])
  })
})

describe('N/A', () => {
  const klein = MODELS.klein.file
  it('Klein has no image-to-image, region, face or hires pass', () => {
    expect(naReason(klein, 'i2i', slot())).toBe(REASONS.noI2i)
    expect(naReason(klein, 'region', slot())).toBe(REASONS.noRegion)
    expect(naReason(klein, 'face', slot())).toBe(REASONS.noFace)
    expect(naReason(klein, 'hires', slot())).toBe(REASONS.noHires)
    expect(buildGraph({ file: klein, op: 'i2i', params: labParams(familyOwning(klein)!, klein, slot(), core, 1), image: 'ref:abcdef012345' })).toEqual({ graph: null, reason: REASONS.noI2i })
  })
  it('negative tests: no input on Klein and Krea2, guidance off on Turbo and Qwen 2.1', () => {
    const neg = slot({ block: 'negative' })
    expect(naReason(klein, 't2i', neg)).toBe('no negative prompt input')
    expect(naReason(MODELS.krea2.file, 't2i', neg)).toBe('no negative prompt input')
    expect(naReason(MODELS.zturbo.file, 't2i', neg)).toMatch(/^guidance off/)
    expect(naReason(MODELS.qwen21.file, 't2i', neg)).toMatch(/^guidance off/)
    for (const k of ['noobai', 'semireal', 'pony', 'wai', 'oneObsession', 'miaomiaoHarem', 'miaomiaoRealskin', 'chroma', 'zbase']) expect(naReason(MODELS[k].file, 't2i', neg)).toBeNull()
  })
  it('the editing model takes only edits', () => {
    const f = MODELS.qwenEdit.file
    for (const op of ['t2i', 'i2i', 'region', 'face', 'hires'] as const) expect(naReason(f, op, slot())).toBe(REASONS.editOnly)
    expect(naReason(f, 'edit', slot())).toBeNull()
  })
})

describe('cellIdOf and expansion', () => {
  it('is stable, and follows seed, steps and model but not the prefix', () => {
    const a = expand([core], [CAT]), b = expand([core], [CAT])
    expect(a.cells.map((c) => c.cellId)).toEqual(b.cells.map((c) => c.cellId))
    expect(new Set(a.cells.map((c) => c.cellId)).size).toBe(676)
    const file = MODELS.noobai.file
    const def = familyOwning(file)!
    const id = (seed: number, steps?: number, f = file) => {
      const g = buildGraph({ file: f, op: 't2i', params: labParams(familyOwning(f)!, f, slot(), core, seed, { steps }) })
      return cellIdOf(g.graph!)
    }
    expect(id(1)).toBe(id(1))
    expect(id(1)).not.toBe(id(2))
    expect(id(1)).not.toBe(id(1, 20))
    expect(id(1)).not.toBe(id(1, undefined, MODELS.pony.file))
    const g = buildGraph({ file, op: 't2i', params: labParams(def, file, slot(), core, 1) }).graph!
    const renamed = JSON.parse(JSON.stringify(g))
    for (const n of Object.values(renamed) as any[]) if (n.class_type === 'SaveImage') n.inputs.filename_prefix = 'elsewhere/x'
    expect(cellIdOf(renamed)).toBe(cellIdOf(g))
  })
  it('the calibration reuses 88 core pictures and the prompt-style sentences 56', () => {
    const coreIds = new Set(expand([core], [CAT]).cells.map((c) => c.cellId))
    const cal = expand([calibration], [CAT])
    expect(new Set(cal.cells.map((c) => c.cellId)).size).toBe(272)
    expect(cal.cells.filter((c) => c.steps === 28 && c.set === 'sweep').every((c) => coreIds.has(c.cellId))).toBe(true)
    expect(cal.cells.filter((c) => coreIds.has(c.cellId)).length).toBe(88)
    expect(expand([extRange], [CAT]).cells.filter((c) => coreIds.has(c.cellId)).length).toBe(56)
  })
  it('records N/A from the registry, never as cells', () => {
    const x = expand([extEdit], [CAT, SCENE])
    expect(x.na.filter((n) => n.model === 'klein').map((n) => n.slot)).toEqual(expect.arrayContaining(['ref.cat.ghibli', 'region.scene', 'detail.group.face', 'detail.group.hires']))
    expect(x.cells.some((c) => c.model === 'klein' && c.slot === 'ref.cat.ghibli')).toBe(false)
    expect(x.cells.filter((c) => c.set === 'negative.party').length).toBe(9 * 2 * 4)
  })
  it('uses the user\'s description, the upright size and the mask\'s position', () => {
    const x = expand([extEdit], [{ ...CAT, describe: 'a grey cat' }, SCENE])
    const noob = x.cells.find((c) => c.model === 'noobai' && c.slot === 'ref.cat.ghibli')!
    expect(x.prompts.get(noob.cellId)).toBe('A Studio Ghibli style illustration of a grey cat.')
    expect(noob.height).toBeGreaterThan(noob.width)
    const qe = x.cells.find((c) => c.model === 'qwenEdit' && c.slot === 'region.scene')!
    expect(qe.op).toBe('edit')
    expect(x.prompts.get(qe.cellId)).toContain('in the lower left of the picture')
    expect(positionWords({ x: 0, y: 0, w: 10, h: 10 }, { width: 300, height: 300 })).toBe('upper left')
  })
  it('refuses a description with a child-coded word', () => {
    const x = expand([extEdit], [{ ...CAT, describe: 'a young cat' }, SCENE])
    expect(x.blocked.some((b) => b.ref === 'cat' && b.kind === 'description' && /young/.test(b.why))).toBe(true)
    expect(x.cells.some((c) => c.slot.startsWith('ref.cat') && c.op === 'i2i')).toBe(false)
    expect(startRefusal(planFrom('exta-1', [extEdit], x, new Map()))).toMatch(/description of the photo of your cat/)
    const d = expand([extEdit], [{ ...CAT, describe: "a cat on my daughter's lap" }, SCENE])
    expect(d.blocked.some((b) => b.kind === 'description')).toBe(true)
  })
})

describe('a region cell goes by its own mask', () => {
  // Two near-full-frame rectangles on one photo: both crops clamp to the whole frame.
  const scene = (rect: { x: number; y: number; w: number; h: number }, maskSha12: string): RefInfo => ({ ...SCENE, width: 1024, height: 768, rect, maskSha12 })
  const A = scene({ x: 10, y: 10, w: 1000, h: 740 }, '111111111111')
  const B = scene({ x: 20, y: 20, w: 990, h: 730 }, '222222222222')
  const regions = (r: RefInfo) => expand([extEdit], [CAT, r]).cells.filter((c) => c.slot === 'region.scene' && c.op === 'region')
  it('a rectangle drawn again gives new region ids, even when both crop to the whole frame', () => {
    const a = regions(A), b = regions(B)
    expect(a.length).toBeGreaterThan(0)
    const ids = new Set(a.map((c) => c.cellId))
    expect(b.filter((c) => ids.has(c.cellId))).toEqual([])
  })
  it('the placeholder, the binding and the copy all name the mask\'s own hash', () => {
    const x = expand([extEdit], [CAT, A])
    const c = x.cells.find((y) => y.model === 'zbase' && y.slot === 'region.scene')!
    expect(c.placeholders.map(([, , v]) => v)).toContain('mask:111111111111')
    expect(c.placeholders.map(([, , v]) => v)).not.toContain('mask:0123456789ab')
    const refs = refBindings([CAT, A])
    expect(refs['mask:111111111111']).toBe('.lab/refs/111111111111.mask.png')
    const f = finalizeGraph(x.graphs.get(c.cellId)!, c, { refs })
    expect(f.graph.__rf_mask.inputs.image).toBe('.lab/refs/111111111111.mask.png [output]')
    expect(verifyCell(f.graph, c)).toEqual([])
    // A binding keyed by the photo's hash, as before, is not the mask this cell was planned with.
    const photoKeyed = { 'ref:0123456789ab': refs['ref:0123456789ab'], 'mask:0123456789ab': '.lab/refs/0123456789ab.mask.png' }
    expect(() => finalizeGraph(x.graphs.get(c.cellId)!, c, { refs: photoKeyed })).toThrow('No file was given for mask:111111111111')
  })
  it('maskKey is the mask\'s hash, or the photo\'s when the index gives none', () => {
    expect(maskKey({ sha12: 'aaa', maskSha12: null })).toBe('aaa')
    expect(maskKey({ sha12: 'aaa', maskSha12: 'bbb' })).toBe('bbb')
  })
})

describe('a reference description names only the cat or the room: no person at all', () => {
  // Each with the word the refusal names.
  const PERSON: [string, string][] = [
    ['a cat held by someone', 'someone'],
    ['a cat and its owner', 'owner'],
    ["a cat on someone's lap", 'someone'],
    ['a grey cat held by its owner', 'owner'],
    ['a cat in her arms', 'arms'],
    ["a cat on a woman's lap", 'woman'],
    ['a cat held by a woman in her thirties', 'woman'],
  ]
  for (const [d, word] of PERSON) {
    it(`refuses "${d}", naming "${word}", and makes no image-to-image cell of the cat`, () => {
      const x = expand([extEdit], [{ ...CAT, describe: d }, SCENE])
      const b = x.blocked.filter((y) => y.ref === 'cat' && y.kind === 'description')
      expect(b.length, d).toBeGreaterThan(0)
      for (const y of b) {
        // "her arms" gets the plainer sentence ("…describe only the cat."), the rest the person one.
        expect(y.why).toMatch(/describe only the cat/i)
        expect(y.why).toContain(`"${word}"`)
      }
      expect(x.cells.some((c) => c.slot.startsWith('ref.cat') && c.op === 'i2i')).toBe(false)
    })
  }
  it('keeps a description of the cat alone, the arm of a sofa included', () => {
    for (const d of ['a grey cat', extEdit.refs.cat.describe, 'a cat on the arm of the sofa']) {
      expect(expand([extEdit], [{ ...CAT, describe: d }, SCENE]).blocked, d).toEqual([])
    }
  })
  it('the start refusal names the photo, the word and what to write instead', () => {
    const x = expand([extEdit], [{ ...CAT, describe: 'a cat held by its owner' }, SCENE])
    const why = startRefusal(planFrom('exta-1', [extEdit], x, new Map()))
    expect(why).toMatch(/description of the photo of your cat/)
    expect(why).toMatch(/"owner"/)
    expect(why).toMatch(/Describe only the cat/)
  })
  it('a suite\'s own starting description is held to the same rule', () => {
    const s = { ...extEdit, refs: { ...extEdit.refs, cat: { ...extEdit.refs.cat, describe: 'a cat held by someone' } } }
    expect(checkSuite(s, SHIPPED_SUITES.filter((o) => o.id !== extEdit.id)).join(' ')).toMatch(/brings a person into the picture/)
  })
  it('the scene\'s description asks for the room or table only', () => {
    const why = descriptionProblem('scene', 'a kitchen with my wife cooking')
    expect(why).toMatch(/"wife"/)
    expect(why).toMatch(/the room or table/)
    expect(descriptionProblem('cat', 'a grey cat')).toBeNull()
  })
  it('refuses children in the plural and by every common word, and people by plural, nationality and nickname', () => {
    for (const t of [
      'a cat with two babies', 'a cat with toddlers', 'a cat with infants', 'a cat with teenagers', 'a cat with schoolgirls',
      'a cat with minors', 'a cat with a youngster', 'a cat with germans', 'a cat with a German', 'a cat with a buddy',
      'a cat with a pal', 'a cat and its mate', 'a cat being brushed', 'a cat scooped up', 'a cat sniffing toes',
      'a cat with postmen',
    ]) expect(descriptionProblem('cat', t), t).not.toBeNull()
    for (const t of ['a cat with two babies', 'a cat with toddlers', 'a cat with a youngster']) {
      expect(descriptionProblem('cat', t), t).toMatch(/which no lab prompt may use/)
    }
  })
  it('reads the words as they are saved, so spacing or a line break hides nothing', () => {
    expect(descriptionProblem('cat', 'a cat being picked\n   up')).not.toBeNull()
  })
  it('leaves alone the things that only share a word with a person', () => {
    for (const t of [
      'a tortoiseshell tabby cat with a white belly and white paws, wearing a dark collar', 'a cat at the foot of the bed',
      'a German shepherd plush toy beside a cat', 'a cat under a Roman blind', 'a room with a ceiling fan and a record player',
      'a cat on the arm of the sofa', 'a cat by a chest of drawers', 'a cat on an ottoman', 'a Birman cat',
    ]) expect(descriptionProblem('cat', t), t).toBeNull()
  })
  it('a slot that says "someone" is a person too, and must be marked humans', () => {
    expect(mentionsPerson('someone reading on a bench')).toBe(true)
    expect(checkSuite(suite([slot({ text: 'Someone reading on a bench.' })])).join(' ')).toMatch(/mark it humans: true/)
  })
})

describe('the person rule for a description is broad: anything that brings in a person is refused', () => {
  // Each with the word the refusal quotes. The phone photos of the checks
  // (items 4 and 14), plurals and -man words, and a sample of each kind of
  // word the rule names: people, family and pet names, roles and titles,
  // "my" and "your", and being held.
  const PERSON: [string, string][] = [
    ['a cat curled at my feet', 'my'],
    ['a cat on my chest', 'my'],
    ['a cat in a sling on my back', 'my'],
    ['a cat being picked up', 'picked up'],
    ['a cat getting a cuddle', 'cuddle'],
    ['a cat with its keeper', 'keeper'],
    ['a cat and its caretaker', 'caretaker'],
    ['a cat with the postman', 'postman'],
    ['a cat with a bride', 'bride'],
    ['a cat with a teacher', 'teacher'],
    ['a cat with two nurses', 'nurses'],
    ['a cat with a soldier', 'soldier'],
    ['a cat with a policewoman', 'policewoman'],
    ['a cat on a sleeping grandad', 'grandad'],
    ['a cat with mommy', 'mommy'],
    ['a cat and a king', 'king'],
    ['a cat getting a hug', 'hug'],
    ['a cat in a tight embrace', 'embrace'],
    ['a cat and two policemen', 'policemen'],
    ['a cat with the salespeople', 'salespeople'],
    ['a cat next to the wives', 'wives'],
    ['a cat with someone', 'someone'],
    ['a cat waiting for somebody', 'somebody'],
    ['a cat ignoring everyone', 'everyone'],
    ['a cat among people', 'people'],
    ['a cat and a friend', 'friend'],
    ['a cat asleep next to mum', 'mum'],
    ['a cat on papa', 'papa'],
    ['a cat on granny', 'granny'],
    ['a cat with a police officer', 'police'],
    ['a cat with a doctor', 'doctor'],
    ['a cat and a queen', 'queen'],
    ['a cat looking at me', 'me'],
    ['a cat of mine', 'mine'],
    ['our cat asleep on a cushion', 'our'],
    ['your cat asleep on a cushion', 'your'],
    ['a cat we adopted', 'we'],
    ['a cat waiting for us', 'us'],
    ['a cat being held', 'held'],
    ['a cat being carried', 'carried'],
    ['a cat being stroked', 'stroked'],
    ['a cat being petted', 'petted'],
    ['a cat asleep on a lap', 'lap'],
  ]
  for (const [d, word] of PERSON) {
    it(`refuses "${d}" as bringing in a person, naming "${word}"`, () => {
      const why = descriptionProblem('cat', d)
      expect(why, d).not.toBeNull()
      expect(why).toContain(`"${word}"`)
      expect(why).toMatch(/brings a person into the picture/)
      expect(why).toMatch(/Describe only the cat/)
    })
  }

  // Words that may well be about the cat itself: refused, with a sentence that says why.
  const PLAIN = "Words like 'her', 'his' or 'hand' read as a person. Call the cat 'it' and describe only the cat."
  const READS_AS_PERSON: [string, string][] = [
    ['a tabby cat licking her paw', 'her'],
    ['a ginger cat asleep on his bed', 'his'],
    ['a cat in a hand-knitted jumper', 'hand'],
    ['a cat with white feet', 'feet'],
    ['a cat that sleeps where he likes', 'he'],
    ['a cat that sleeps where she likes', 'she'],
    ['a cat looking at him', 'him'],
    ['a cat asleep on their sofa', 'their'],
    ['a cat batting at two hands', 'hands'],
    ['a cat asleep on an arm', 'arm'],
    ['a cat curled in two arms', 'arms'],
    ['a cat rubbing a leg', 'leg'],
    ['a cat weaving between legs', 'legs'],
    ['a cat asleep on a chest', 'chest'],
    ['a cat perched on a shoulder', 'shoulder'],
    ['a cat with a white face', 'face'],
    ['a cat biting a finger', 'finger'],
  ]
  for (const [d, word] of READS_AS_PERSON) {
    it(`refuses "${d}", naming "${word}", and says to call the cat "it"`, () => {
      const why = descriptionProblem('cat', d)
      expect(why, d).not.toBeNull()
      expect(why).toContain(`"${word}"`)
      expect(why).toContain(PLAIN)
      expect(why).not.toMatch(/brings a person into the picture/)
    })
  }

  it('a description with both kinds of word is refused as bringing in a person, and quotes every word', () => {
    const lap = descriptionProblem('cat', 'a cat on his lap')!
    expect(lap).toMatch(/brings a person into the picture/)
    expect(lap).toMatch(/Describe only the cat/)
    expect(lap).toContain('"his"')
    expect(lap).toContain('"lap"')
    const owner = descriptionProblem('cat', 'a cat held by its owner')!
    expect(owner).toMatch(/brings a person into the picture/)
    expect(owner).toContain('"held"')
    expect(owner).toContain('"owner"')
  })
  it('the scene\'s plainer sentence asks for the room or table with no one in it', () => {
    const why = descriptionProblem('scene', 'a kitchen table with her keys')!
    expect(why).toContain('"her"')
    expect(why.endsWith('Describe only the room or table, with no one in it.')).toBe(true)
  })
  it('keeps words that only look like a person: -man words that are not, furniture with arms and legs, the shipped description', () => {
    for (const d of [
      'a Birman cat on an ottoman',
      'a bowl of ramen on a table',
      'a cat with a white abdomen',
      'a cat on a chest of drawers',
      'a cat under the table legs',
      'a cat on the arm of the sofa',
      'a cat that uses a litter box',
      extEdit.refs.cat.describe,
    ]) {
      expect(descriptionProblem('cat', d), d).toBeNull()
    }
    expect(extEdit.refs.cat.describe).toMatch(/white belly/)
  })
  it('a description the broad rule refuses makes no image-to-image cell of the cat', () => {
    const x = expand([extEdit], [{ ...CAT, describe: 'a cat with two nurses' }, SCENE])
    expect(x.blocked.some((b) => b.ref === 'cat' && b.kind === 'description' && /"nurses"/.test(b.why))).toBe(true)
    expect(x.cells.filter((c) => c.slot.startsWith('ref.cat') && c.op === 'i2i')).toEqual([])
    // The same night with the cat described alone has them.
    expect(expand([extEdit], [CAT, SCENE]).cells.some((c) => c.slot.startsWith('ref.cat') && c.op === 'i2i')).toBe(true)
  })
  it('a suite prompt keeps its own rule: "a cowboy hat" or "a hand-painted sign" brings in no one', () => {
    expect(checkSuite(suite([slot({ text: 'A cowboy hat on a hook.' })]))).toEqual([])
    expect(checkSuite(suite([slot({ text: 'A hand-painted sign on a door.' })]))).toEqual([])
  })
})

describe('finalizeGraph and verifyCell', () => {
  const x = expand([extEdit], [CAT, SCENE])
  const refs = refBindings([CAT, SCENE])
  const byChain = (id: string, step = 0) => x.cells.find((c) => c.chain === id && c.chainStep === step)!
  it('puts the token only at chainAt, or the exact annotated upstream', () => {
    const c = byChain('chain.klein-faces')
    const t = finalizeGraph(x.graphs.get(c.cellId)!, c, { upstream: 'token', refs })
    expect(t.chainAt).toEqual([['__i2i_load', 'image']])
    expect(JSON.stringify(t.graph).split(CHAIN_TOKEN).length - 1).toBe(1)
    expect(verifyCell(t.graph, c)).toEqual([])
    const l = finalizeGraph(x.graphs.get(c.cellId)!, c, { upstream: { rel: `.lab/cells/${c.upstream}_00001_.png` }, refs })
    expect(l.chainAt).toBeNull()
    expect(l.graph.__i2i_load.inputs.image).toBe(`.lab/cells/${c.upstream}_00001_.png [output]`)
    expect(verifyCell(l.graph, c)).toEqual([])
    expect(CHAIN_TOKEN).toBe(SERVER_TOKEN)
  })
  it('names the prefix, the refs and the mask', () => {
    const c = x.cells.find((y) => y.model === 'zbase' && y.slot === 'region.scene')!
    const f = finalizeGraph(x.graphs.get(c.cellId)!, c, { refs })
    expect(f.graph.__rf_src.inputs.image).toBe('.lab/refs/0123456789ab.png [output]')
    expect(f.graph.__rf_mask.inputs.image).toBe('.lab/refs/0123456789ab.mask.png [output]')
    const save = Object.values(f.graph).find((n) => n.class_type === 'SaveImage')!
    expect(save.inputs.filename_prefix).toBe(`.lab/cells/${c.cellId}`)
    expect(verifyCell(f.graph, c)).toEqual([])
  })
  it('refuses paths that are not the upstream\'s or the refs\'', () => {
    const c = byChain('chain.klein-faces')
    expect(() => finalizeGraph(x.graphs.get(c.cellId)!, c, { upstream: { rel: '.lab/cells/../../etc/passwd' }, refs })).toThrow()
    expect(() => finalizeGraph(x.graphs.get(c.cellId)!, c, { upstream: { rel: `.lab/cells/ffffffffffffffff_00001_.png` }, refs })).toThrow()
    expect(() => finalizeGraph(x.graphs.get(c.cellId)!, c, { refs })).toThrow(/not made yet/)
    const r = x.cells.find((y) => y.model === 'zbase' && y.slot === 'region.scene')!
    expect(() => finalizeGraph(x.graphs.get(r.cellId)!, r, { refs: { ...refs, 'ref:0123456789ab': 'switchgen/other.png' } })).toThrow()
  })
  it('catches a changed setting, a second save, a stray token and a big graph', () => {
    const c = x.cells.find((y) => y.model === 'wai' && y.slot === 'detail.group.face')!
    const f = finalizeGraph(x.graphs.get(c.cellId)!, c, { refs }).graph
    expect(verifyCell(f, c)).toEqual([])
    const seed = JSON.parse(JSON.stringify(f)); seed['6'].inputs.seed = 5
    // Caught twice: the seed reads back wrong, and the graph is not the one hashed.
    expect(verifyCell(seed, c).join(' ')).toMatch(/seed reads 5, not the cell's seed/)
    expect(verifyCell(seed, c).join(' ')).toMatch(/not the one this cell was hashed from/)
    const two = JSON.parse(JSON.stringify(f)); two.extra = { class_type: 'SaveImage', inputs: { images: ['7', 0], filename_prefix: 'x' } }
    expect(verifyCell(two, c).join(' ')).toMatch(/exactly one picture/)
    const tok = JSON.parse(JSON.stringify(f)); tok['3'].inputs.text = CHAIN_TOKEN
    expect(verifyCell(tok, c).join(' ')).toMatch(/token/)
    const big = JSON.parse(JSON.stringify(f)); big['3'].inputs.text = 'x'.repeat(1048577)
    expect(verifyCell(big, c).join(' ')).toMatch(/1 MiB/)
  })
  it('every cell of every night finalizes and verifies', () => {
    const all = expand([...SHIPPED_SUITES], [CAT, SCENE])
    for (const c of all.cells) {
      const f = finalizeGraph(all.graphs.get(c.cellId)!, c, { upstream: c.upstream ? { rel: `.lab/cells/${c.upstream}_00001_.png` } : undefined, refs })
      expect(verifyCell(f.graph, c), `${c.slot} ${c.model} ${c.op}`).toEqual([])
    }
  })
})

describe('detailOnPicture', () => {
  it('is 10 nodes, with the pass reading the loaded picture and one token site', () => {
    const shape = detailOnPicture(BY_ID['z-image'], 'face')!
    expect(Object.keys(shape.def.graph)).toHaveLength(10)
    const fd = Object.values(shape.def.graph).find((n) => n.class_type === 'FaceDetailer')!
    const load = Object.entries(shape.def.graph).find(([, n]) => n.class_type === 'LoadImage')![0]
    expect(fd.inputs.image).toEqual([load, 0])
    expect(shape.labOnly).toBe(true)
    const x = expand([extEdit], [CAT, SCENE])
    const c = x.cells.find((y) => y.chain === 'chain.klein-faces')!
    expect(c.labOnly).toBe(true)
    expect(c.placeholders).toHaveLength(1)
    expect(detailOnPicture(BY_ID['flux2-klein'], 'face')).toBeNull()
    expect('reason' in shapeFor(MODELS.klein.file, 'detailOnPicture')).toBe(true)
  })
})

describe('checkSuite', () => {
  it('rejects child-coded words anywhere', () => {
    for (const w of ['girl', 'young', 'teen', '1girl', 'female_child']) {
      expect(checkSuite(suite([slot({ text: `A ${w} on a beach.` })])).join(' ')).toMatch(/no lab prompt may use/)
    }
    expect(CHILD_CODED).toContain('youthful')
    expect(checkSuite(suite([slot({ text: 'A cowboy hat on a hook.' })]))).toEqual([])
  })
  it('requires an adult for a person, and a humans mark', () => {
    expect(checkSuite(suite([slot({ text: 'A person reading.', humans: true })])).join(' ')).toMatch(/name an adult/)
    expect(checkSuite(suite([slot({ text: 'A woman in her thirties reading.' })])).join(' ')).toMatch(/humans: true/)
    expect(checkSuite(suite([slot({ text: 'A woman in her thirties reading.', humans: true })]))).toEqual([])
    expect(checkSuite(suite([slot({ text: 'An old fisherman mending a net.', humans: true })])).join(' ')).toMatch(/name an adult/)
    expect(checkSuite(suite([slot({ text: 'a person', defaultsProbe: true })]))).toEqual([])
  })
  it('requires expect words in the text, consistent sets and a neutral task', () => {
    expect(checkSuite(suite([slot({ text: 'A sign.', expect: ['OPEN'] })])).join(' ')).toMatch(/OPEN/)
    const a = slot({ id: 'a', set: 'x', condition: 'one' }), b = slot({ id: 'b', set: 'x', condition: 'one', block: 'style' })
    expect(checkSuite(suite([a, b])).join(' ')).toMatch(/mixes blocks|condition label twice/)
    expect(checkSuite(suite([slot({ op: 'i2i', source: { ref: 'cat' }, perModelText: { qwenEdit: 'Do it.' }, models: ['noobai', 'qwenEdit'] })])).join(' ')).toMatch(/neutral task/)
  })
  it('passes all four shipped suites', () => {
    for (const s of SHIPPED_SUITES) expect(checkSuite(s, SHIPPED_SUITES.filter((o) => o !== s)), s.id).toEqual([])
    expect(() => expand([defineSuite(suite([slot({ text: 'A girl.' })]))])).toThrow(/cannot be used/)
  })
})

describe('plan', () => {
  it('core-1 is 676 pictures, 88 reused after cal-1, with a finite estimate', () => {
    const p0 = planFrom('core-1', [core], expand([core], [CAT]), new Map())
    expect(p0.estimate.pictures).toBe(676)
    expect(p0.estimate.newPictures).toBe(676)
    const calIds = new Set(expand([calibration], [CAT]).cells.map((c) => c.cellId))
    const p = planFrom('core-1', [core], expand([core], [CAT]), done(calIds))
    expect(p.reused).toHaveLength(88)
    expect(p.order).toHaveLength(588)
    expect(Number.isFinite(p.estimate.seconds) && p.estimate.seconds > 0).toBe(true)
    expect(p.na).toEqual([])
    expect(describePlan(p)).toMatch(/Nothing is sent until you press Start/)
    expect(Object.values(p.prompts).some((t) => t.startsWith('score_9'))).toBe(false)
  })
  it('ext-edit waits for the scene photo, then its rectangle, and names them', () => {
    const p = planFrom('exta-1', [extEdit], expand([extEdit], [CAT]), new Map())
    expect(startRefusal(p, { labDir: DEFAULTS.labDir })).toMatch(/room or table photo[\s\S]*\/mnt\/storage\/ai\/lab\/refs/)
    const noRect = planFrom('exta-1', [extEdit], expand([extEdit], [CAT, { ...SCENE, mask: false, rect: null }]), new Map())
    expect(startRefusal(noRect)).toMatch(/rectangle/)
    expect(noRect.blocked.some((b) => b.slot === 'edit.winter')).toBe(false)
    const ok = planFrom('exta-1', [extEdit], expand([extEdit], [CAT, SCENE]), new Map())
    expect(startRefusal(ok)).toBeNull()
    expect(ok.estimate.pictures).toBe(664)
    const noCat = planFrom('exta-1', [extEdit], expand([extEdit], [SCENE]), new Map())
    expect(noCat.blocked.some((b) => b.chain === 'chain.cat-twice')).toBe(true)
  })
  it('sends each cell after the picture it starts from, model by model', () => {
    const x = expand([extEdit], [CAT, SCENE])
    const order = sendOrder(x.cells, [extEdit])
    const pos = new Map(order.map((id, i) => [id, i]))
    for (const c of x.cells) if (c.upstream) expect(pos.get(c.upstream)!).toBeLessThan(pos.get(c.cellId)!)
    expect(() => makePlan('x', [core, { ...core, id: 'y', study: 'other' }], [], [], new Map())).toThrow(/one study/)
    expect(suiteById('core')?.id).toBe('first-pass-core')
  })
  it('saves the plan and the cells only under the lab folder', () => {
    const dir = tempDir('lab-core-')
    {
      const env = labEnv({ SWITCHGEN_LAB_DIR: path.join(dir, 'lab'), SWITCHGEN_OUTPUTS: path.join(dir, 'out') })
      const x = expand([calibration], [CAT])
      const p = planFrom('cal-1', [calibration], x, new Map())
      const file = savePlan(env, p, x.graphs)
      expect(file).toBe(path.join(dir, 'lab', 'runs', 'cal-1', 'plan.json'))
      expect(JSON.parse(readFileSync(file, 'utf8')).order).toHaveLength(272)
      expect(existsSync(path.join(dir, 'lab', 'cells', `${p.order[0]}.json`))).toBe(true)
      expect(existsSync(path.join(dir, 'out'))).toBe(false)
    }
  })
})

describe('checkGraph', () => {
  const x = expand([...SHIPPED_SUITES], [CAT, SCENE])
  const info = objectInfoFrom(x.graphs.values())
  const any = x.cells.find((c) => c.model === 'noobai' && c.slot === 'ref.cat.ghibli')!
  const g = finalizeGraph(x.graphs.get(any.cellId)!, any, { refs: refBindings([CAT]) }).graph
  it('passes every lab graph, with annotated paths on LoadImage', () => {
    for (const graph of x.graphs.values()) expect(checkGraph(graph, info)).toEqual([])
    expect(checkGraph(g, info)).toEqual([])
  })
  it('reports a missing class, a bad combo, a dangling link and a bad file', () => {
    const miss = JSON.parse(JSON.stringify(g)); miss['6'].class_type = 'NoSuchNode'
    expect(checkGraph(miss, info).join(' ')).toMatch(/no node class "NoSuchNode"/)
    const combo = JSON.parse(JSON.stringify(g)); combo['6'].inputs.sampler_name = 'nope'
    expect(checkGraph(combo, info).join(' ')).toMatch(/not one of/)
    const link = JSON.parse(JSON.stringify(g)); link['7'].inputs.samples = ['99', 0]
    expect(checkGraph(link, info).join(' ')).toMatch(/not in the graph/)
    const file = JSON.parse(JSON.stringify(g)); file.__i2i_load.inputs.image = '../../etc/passwd [output]'
    expect(checkGraph(file, info).join(' ')).toMatch(/not a file ComfyUI can load/)
  })
})

// ---------------------------------------------------------------------------
// Beyond B's own checks: what the plan and the builders said was missing
// ---------------------------------------------------------------------------

describe('the user\'s description of a reference photo', () => {
  it('changes the image-to-image words and so the cell id, but not the editing model\'s', () => {
    const a = expand([extEdit], [{ ...CAT, describe: 'a tortoiseshell tabby cat with a white belly and white paws, wearing a dark collar' }, SCENE])
    const b = expand([extEdit], [{ ...CAT, describe: 'a grey cat' }, SCENE])
    const pick = (x: typeof a, model: string) => x.cells.find((c) => c.model === model && c.slot === 'ref.cat.ghibli' && c.seed === 1001)!
    expect(a.prompts.get(pick(a, 'noobai').cellId)).toBe('A Studio Ghibli style illustration of a tortoiseshell tabby cat with a white belly and white paws, wearing a dark collar.')
    expect(pick(a, 'noobai').cellId).not.toBe(pick(b, 'noobai').cellId)
    // The editing model is given its instruction, which does not use the description.
    expect(pick(a, 'qwenEdit').cellId).toBe(pick(b, 'qwenEdit').cellId)
    expect(a.prompts.get(pick(a, 'qwenEdit').cellId)).toMatch(/^Turn this photo into a Studio Ghibli style illustration/)
  })
  it('falls back to the suite\'s starting description when the user has none', () => {
    const x = expand([extEdit], [CAT, SCENE])
    const c = x.cells.find((y) => y.model === 'zbase' && y.slot === 'ref.cat.oil')!
    expect(x.prompts.get(c.cellId)).toContain(extEdit.refs.cat.describe)
  })
})

describe('reference photos are measured upright', () => {
  it('the cat, stored sideways, is 3060 x 4080 upright, so image-to-image comes out portrait', () => {
    const x = expand([extEdit], [CAT, SCENE])
    for (const c of x.cells.filter((y) => y.slot.startsWith('ref.cat') && y.op === 'i2i')) {
      expect(c.height, `${c.model}`).toBeGreaterThan(c.width)
      expect(c.refs).toEqual(['cat'])
    }
    const wide = expand([extEdit], [{ ...CAT, width: 4080, height: 3060 }, SCENE])
    const w = wide.cells.find((y) => y.model === 'noobai' && y.slot === 'ref.cat.ghibli')!
    expect(w.width).toBeGreaterThan(w.height)
  })
  it('the rectangle sets the editing model\'s position words, and the region cells keep the source size', () => {
    const x = expand([extEdit], [CAT, { ...SCENE, rect: { x: 1300, y: 50, w: 200, h: 200 } }])
    const qe = x.cells.find((c) => c.model === 'qwenEdit' && c.slot === 'region.scene')!
    expect(qe.op).toBe('edit')
    expect(x.prompts.get(qe.cellId)).toContain('in the upper right of the picture')
    const zb = x.cells.find((c) => c.model === 'zbase' && c.slot === 'region.scene')!
    expect([zb.width, zb.height]).toEqual([1600, 1200])
    expect(zb.denoise).toBe(0.55)
    const moved = expand([extEdit], [CAT, { ...SCENE, rect: { x: 100, y: 50, w: 200, h: 200 } }])
    expect(moved.cells.find((c) => c.model === 'zbase' && c.slot === 'region.scene' && c.seed === zb.seed)!.cellId).not.toBe(zb.cellId)
  })
})

describe('the ext-edit night', () => {
  const x = expand([extEdit], [CAT, SCENE])
  it('holds 9 negative-capable models in each negative set: 9 x 2 x 4 = 72 cells', () => {
    for (const set of ['negative.party', 'negative.beach']) {
      const cells = x.cells.filter((c) => c.set === set)
      expect(cells).toHaveLength(72)
      expect(new Set(cells.map((c) => c.model))).toEqual(new Set(['noobai', 'semireal', 'pony', 'wai', 'oneObsession', 'miaomiaoHarem', 'miaomiaoRealskin', 'chroma', 'zbase']))
    }
    for (const m of ['krea2', 'klein', 'qwen21', 'zturbo']) expect(x.na.some((n) => n.model === m && n.slot === 'negative.party.with')).toBe(true)
  })
  it('lists the 16 core pictures its chains start from as context', () => {
    expect(x.context.size).toBe(16)
    const ctx = x.cells.filter((c) => x.context.has(c.cellId))
    expect(new Set(ctx.map((c) => `${c.slot}:${c.model}`))).toEqual(new Set(['text.bakery:noobai', 'photo.kitchen:qwen21', 'following.fruit:zturbo', 'layout.thumbnail:qwen21']))
    const coreIds = new Set(expand([core], [CAT]).cells.map((c) => c.cellId))
    for (const id of x.context) expect(coreIds.has(id)).toBe(true)
    const p = planFrom('exta-1', [extEdit], x, done(coreIds))
    expect(p.context).toHaveLength(16)
    expect([p.estimate.pictures, p.estimate.newPictures, p.reused.length]).toEqual([664, 648, 16])
  })
  it('leaves Klein out of image-to-image, region, face and hires with the registry\'s reasons, and never counts N/A', () => {
    const k = x.na.filter((n) => n.model === 'klein')
    expect(k.map((n) => n.slot)).toEqual(expect.arrayContaining(['ref.cat.ghibli', 'ref.cat.oil', 'region.scene', 'detail.group.face', 'detail.group.hires']))
    const p = planFrom('exta-1', [extEdit], x, new Map())
    const naKeys = new Set(p.na.map((n) => `${n.slot}:${n.model}`))
    for (const c of p.cells) expect(naKeys.has(`${c.slot}:${c.model}`)).toBe(false)
    expect(p.estimate.pictures).toBe(new Set(p.cells.map((c) => c.cellId)).size)
  })
})

describe('the ext-range night', () => {
  it('is 944 pictures, 56 of them the prompt-style sentences already made on the core night', () => {
    const coreIds = expand([core], [CAT]).cells.map((c) => c.cellId)
    const calIds = expand([calibration], [CAT]).cells.map((c) => c.cellId)
    const p = planFrom('extb-1', [extRange], expand([extRange], [CAT, SCENE]), done([...coreIds, ...calIds]))
    expect([p.estimate.pictures, p.reused.length, p.order.length]).toEqual([944, 56, 888])
  })
})

describe('the calibration night', () => {
  it('is 272 pictures, and its every sweep and sampler model is one of the 13', () => {
    const p = planFrom('cal-1', [calibration], expand([calibration], [CAT]), new Map())
    expect(p.estimate.pictures).toBe(272)
    for (const m of [...calibration.sweep!.models, ...calibration.samplerCheck!.models]) expect(CORE).toContain(m)
    expect(calibration.steps).toBe(28)
  })
})

describe('timings are labelled for what they are', () => {
  it('every seed row says where it came from, and a guess never says archive or measured', () => {
    const files = new Set(Object.values(MODELS).map((m) => m.file))
    for (const f of files) expect(SEED_TIMINGS[f], f).toBeDefined()
    for (const [file, t] of Object.entries(SEED_TIMINGS)) {
      if (t.source === 'archive') expect(t.note, file).toMatch(/^\d+ pictures?, /)
      else {
        expect(t.source, file).toBe('guess')
        expect(t.note, file).toMatch(/^no archive picture/)
      }
    }
    for (const k of ['krea2', 'zbase', 'qwenEdit']) expect(SEED_TIMINGS[MODELS[k].file].source).toBe('guess')
  })
  it('a plan marks guessed rows and calls its total an estimate', () => {
    const p = planFrom('exta-1', [extEdit], expand([extEdit], [CAT, SCENE]), new Map())
    expect(p.timingSource.krea2).toBe('guess')
    expect(p.timingSource.noobai).toBe('archive')
    const text = describePlan(p)
    expect(text).toMatch(/an estimate from the archive and guesses, not a measurement/)
    for (const line of text.split('\n')) {
      if (/^ {2}(krea2|zbase|qwenEdit): \d+ pictures/.test(line)) expect(line).toMatch(/guessed: no archive figure/)
      if (/^ {2}noobai: \d+ pictures/.test(line)) expect(line).not.toMatch(/guessed/)
    }
    expect(text.match(/guessed: no archive figure/g)).toHaveLength(3)
    expect(text).not.toMatch(/measured/)
  })
  it('only warm, uncached, finished pictures become a measured figure', () => {
    const x = expand([core], [CAT])
    const noob = x.cells.filter((c) => c.model === 'noobai' && c.op === 't2i' && c.width === 1024 && c.height === 1024).slice(0, 4)
    expect(noob).toHaveLength(4)
    const d = new Map<string, DoneCell>([
      [noob[0].cellId, { cellId: noob[0].cellId, rel: 'x', durationMs: 28_000, cold: false, cached: false, finishedAt: 1, run: 'r' }],
      [noob[1].cellId, { cellId: noob[1].cellId, rel: 'x', durationMs: 99_000, cold: true, cached: false, finishedAt: 1, run: 'r' }],
      [noob[2].cellId, { cellId: noob[2].cellId, rel: 'x', durationMs: 1_000, cold: false, cached: true, finishedAt: 1, run: 'r' }],
      [noob[3].cellId, { cellId: noob[3].cellId, rel: 'x', durationMs: 1_000, cold: false, cached: true, finishedAt: 1, run: 'r' }],
    ])
    const t = measuredTimings(x.cells, d)
    expect(t[MODELS.noobai.file]).toMatchObject({ msPerStep: 1000, source: 'measured' })
    expect(t[MODELS.krea2.file].source).toBe('guess')
  })
  // An opt-in check against the author's own archive, read only: LAB_ARCHIVE=<outputs>/.switchgen/archive.json.
  // It never runs by default, because the tests must not read the author's folders.
  it.skipIf(!process.env.LAB_ARCHIVE)('each archive row has text-to-image pictures in the archive, and each guess has none', () => {
    const a = JSON.parse(readFileSync(process.env.LAB_ARCHIVE!, 'utf8')) as { records: Record<string, { mode?: string; model?: string; durationMs?: number; desk?: string }> }
    const count = new Map<string, number>()
    for (const r of Object.values(a.records)) if (r.mode === 't2i' && (r.durationMs ?? 0) > 0 && r.model) count.set(r.model, (count.get(r.model) ?? 0) + 1)
    for (const [file, t] of Object.entries(SEED_TIMINGS)) {
      if (t.source === 'archive') expect(count.get(file) ?? 0, file).toBeGreaterThan(0)
      else expect(count.get(file) ?? 0, file).toBe(0)
    }
  })
})

describe('the house rules on words', () => {
  it('reject every child-coded word, in a slot, a chain step or a description', () => {
    for (const w of CHILD_CODED) {
      expect(checkSuite(suite([slot({ text: `A photo of a ${w} by a lake.` })])).join(' '), w).toMatch(/no lab prompt may use/)
    }
    const chained = suite([slot({ id: 'base' })], { chains: [{ id: 'c', name: 'c', block: 'following', compose: { slot: 'base', model: 'noobai' }, steps: [{ model: 'zbase', op: 'i2i', text: 'a schoolgirl' }] }] })
    expect(checkSuite(chained).join(' ')).toMatch(/no lab prompt may use/)
  })
  it('every shipped prompt that shows a person names an adult', () => {
    for (const s of SHIPPED_SUITES) for (const sl of s.slots) if (sl.humans) expect(checkSuite(suite([sl])).join(' '), `${s.id} ${sl.id}`).not.toMatch(/name an adult/)
  })
})
