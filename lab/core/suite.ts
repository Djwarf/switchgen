/**
 * Suites as data, and the checker every suite passes before it is expanded.
 *
 * The checker is the lab's content rule in code: no prompt may use a word that
 * points at a child, and every prompt that shows a person names an adult. The
 * one exception is the defaults probe ("a person"), which asks what a model
 * draws unprompted; the picture reader's quarantine backs it up. A reference
 * photo's description (the suite's, or the user's own from the lab page)
 * names only the cat or object and brings in no person at all.
 */
import { familyOwning } from '../../src/lib/workflows.ts'
import type { ChainSpec, Op, Slot, Suite } from './types.ts'

export const OPS: readonly Op[] = ['t2i', 'i2i', 'edit', 'region', 'face', 'hand', 'hires', 'detailOnPicture']

/** Words no lab prompt may use, matched as whole words (so "1girl" and "female_child" count). */
export const CHILD_CODED: readonly string[] = [
  'child',
  'children',
  'kid',
  'kids',
  'girl',
  'girls',
  'boy',
  'boys',
  'teen',
  'teens',
  'teenage',
  'teenager',
  'loli',
  'shota',
  'schoolgirl',
  'schoolboy',
  'baby',
  'babies',
  'toddler',
  'toddlers',
  'infant',
  'infants',
  'newborn',
  'newborns',
  'minor',
  'minors',
  'underage',
  'preteen',
  'preteens',
  'tween',
  'tweens',
  'teenagers',
  'adolescent',
  'adolescents',
  'youth',
  'youths',
  'youngster',
  'youngsters',
  'juvenile',
  'juveniles',
  'pupil',
  'pupils',
  'kiddie',
  'kiddies',
  'kiddo',
  'kiddos',
  'schoolgirls',
  'schoolboys',
  'schoolchild',
  'schoolchildren',
  'lolis',
  'shotas',
  'young',
  'youthful',
]

/** A whole word: not preceded or followed by a letter. Digits and underscores do not join words here. */
const wordRe = (words: readonly string[]) => new RegExp(`(?:^|[^a-z])(${words.join('|')})(?=$|[^a-z])`, 'gi')

const CHILD_RE = wordRe(CHILD_CODED)

/** Words that name an adult, or an adult age. */
const ADULT_WORDS = ['adult', 'adults', 'woman', 'women', 'man', 'men', 'mature', 'elderly', 'gentleman', 'gentlemen', 'lady', 'ladies']
const ADULT_RE = new RegExp(
  `${wordRe(ADULT_WORDS).source}|\\bin (?:her|his|their) (?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\\b|\\b[2-9]0s\\b`,
  'i',
)

/** Words that name a person. */
const PERSON_WORDS = [
  'person',
  'persons',
  'people',
  'someone',
  'somebody',
  'anyone',
  'anybody',
  'everyone',
  'everybody',
  'human',
  'humans',
  'man',
  'men',
  'woman',
  'women',
  'adult',
  'adults',
  'lady',
  'ladies',
  'gentleman',
  'gentlemen',
  'guy',
  'guys',
  'couple',
  'crowd',
  'family',
  'friend',
  'friends',
  // Relationships.
  'owner',
  'owners',
  'mother',
  'mum',
  'mom',
  'father',
  'dad',
  'parent',
  'parents',
  'grandmother',
  'grandma',
  'grandfather',
  'grandpa',
  'wife',
  'husband',
  'girlfriend',
  'boyfriend',
  'partner',
  'brother',
  'brothers',
  'sister',
  'sisters',
  'daughter',
  'daughters',
  'son',
  'sons',
  'aunt',
  'uncle',
  'cousin',
  'neighbour',
  'neighbor',
  // Roles.
  'pianist',
  'dancer',
  'dancers',
  'fisherman',
  'instructor',
  'astronaut',
  'student',
  'students',
  'vet',
  'doctor',
  'nurse',
  'chef',
  'baker',
  'farmer',
  'rider',
  'waiter',
  'waitress',
]
const PERSON_RE = wordRe(PERSON_WORDS)

/**
 * A reference description is held to a much broader rule than a suite prompt:
 * it names only the cat or the room, so anything that could bring a person in
 * is refused, and a description that says too much is simply reworded. A suite
 * prompt may use these words ("a hand-painted sign", "a woman in her
 * thirties", "a cowboy hat"), so only a description is refused for them.
 */

/** More words for a person: in general, in the family, and by role or title. Each counts in the plural too ("nurses"). */
const MORE_PEOPLE = [
  // In general.
  'stranger', 'visitor', 'guest', 'passerby', 'passersby', 'bystander', 'onlooker', 'spectator', 'audience', 'tourist',
  'folk', 'gal', 'dude', 'bloke', 'chap', 'lad', 'lass', 'grownup', 'grown-up', 'lover', 'ladies', 'families',
  // Family, and the pet names for them.
  'mommy', 'mommies', 'mummy', 'mummies', 'mama', 'mamma', 'daddy', 'daddies', 'papa', 'grandad', 'granddad', 'gramps',
  'granny', 'grannies', 'gran', 'nan', 'nana', 'nanna', 'grandparent', 'grandson', 'granddaughter', 'grandchild',
  'grandchildren', 'wives', 'spouse', 'fiance', 'fiancee', 'bride', 'groom', 'bridegroom', 'bridesmaid', 'widow',
  'widower', 'sibling', 'niece', 'nephew', 'auntie', 'aunty', 'stepmother', 'stepfather', 'stepmum', 'stepmom',
  'stepdad', 'mistress', 'roommate', 'flatmate', 'housemate', 'classmate', 'teammate', 'colleague', 'coworker',
  // The people a pet is left with.
  'keeper', 'zookeeper', 'caretaker', 'carer', 'caregiver', 'sitter', 'petsitter', 'babysitter', 'nanny', 'nannies',
  'handler', 'groomer', 'breeder', 'rescuer', 'volunteer',
  // Everyday words for someone. Words that are more often a thing in a room
  // ("fan", "player", "patient") are left out, so a room is not refused for them.
  'buddy', 'buddies', 'pal', 'mate', 'companion', 'boss', 'host', 'hostess', 'bro', 'sis', 'santa',
  // Roles and titles.
  'soldier', 'officer', 'police', 'cop', 'sheriff', 'detective', 'bodyguard', 'lifeguard', 'firefighter', 'paramedic',
  'medic', 'surgeon', 'dentist', 'priest', 'nun', 'monk', 'vicar', 'pastor', 'rabbi', 'imam', 'pope', 'king', 'queen',
  'prince', 'princess', 'knight', 'emperor', 'empress', 'pharaoh', 'lord', 'duke', 'duchess', 'witch', 'wizard',
  'sorcerer', 'magician', 'clown', 'pirate', 'cowboy', 'cowgirl', 'ninja', 'samurai', 'viking', 'hiker', 'climber',
  'cyclist', 'jogger', 'surfer', 'skater', 'swimmer', 'athlete', 'gymnast', 'wrestler', 'singer', 'musician',
  'guitarist', 'drummer', 'violinist', 'artist', 'painter', 'sculptor', 'photographer', 'actor', 'actress', 'model',
  'celebrity', 'teacher', 'professor', 'tutor', 'scientist', 'engineer', 'builder', 'plumber', 'electrician',
  'mechanic', 'driver', 'pilot', 'sailor', 'captain', 'hunter', 'gardener', 'butcher', 'barista', 'bartender',
  'cashier', 'clerk', 'maid', 'butler', 'servant', 'worker', 'employee', 'customer', 'shopper', 'passenger', 'mayor',
  'president', 'judge', 'lawyer', 'explorer', 'traveller', 'traveler', 'writer',
]

/** Words that end in -man or -men and are not a person: furniture, food, cat and dog breeds, and a few more. */
const NOT_A_PERSON = ['ottoman', 'ramen', 'abdomen', 'omen', 'specimen', 'stamen', 'amen', 'birman', 'doberman']

/** Any word ending in -man, -men, -woman, -women, -person or -people ("postman", "policewomen", "salespeople"). */
const ANY_MAN = `(?!(?:${NOT_A_PERSON.join('|')})s?(?![a-z]))[a-z]*(?:m[ae]ns?|persons?|peoples?)`

/** The one taking the photo or looking at it, and a pet being held: someone is in the picture or at its edge. */
const PERSON_ANCHORS = [
  'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
  'you', 'your', 'yours', 'yourself', 'yourselves',
  'held', 'hold', 'holds', 'holding', 'hug', 'hugs', 'hugged', 'hugging', 'cuddle', 'cuddles', 'cuddled', 'cuddling',
  'embrace', 'embraces', 'embraced', 'embracing', 'carry', 'carries', 'carried', 'carrying', 'cradled', 'cradling',
  'pick(?:s|ed|ing)? up', 'stroked', 'stroking', 'petted', 'petting', 'patted', 'patting', 'kissed', 'kissing',
  'tickled', 'tickling', 'lap', 'laps', 'selfie', 'selfies',
  'being (?:brushed|lifted|fed|groomed|washed|bathed|walked|trained|dressed)', 'scooped up', 'handed over',
]

/**
 * Words that read as a person even when they are about a cat: "he", "she"
 * and their kin, and the parts of a person a pet sits on or is held in.
 * Each part counts in the plural too ("hands").
 */
const PRONOUNS = ['he', 'she', 'him', 'his', 'her', 'hers', 'himself', 'herself', 'they', 'them', 'their', 'theirs', 'themselves']
const BODY_PARTS = [
  'hand', 'arm', 'forearm', 'elbow', 'wrist', 'fist', 'palm', 'finger', 'fingertip', 'thumb', 'shoulder', 'chest',
  'face', 'leg', 'thigh', 'knee', 'feet', 'foot', 'toe',
]

/** A list of words, each also in the plural. */
const plural = (words: readonly string[]) => `(?:${words.join('|')})(?:e?s)?`
const READS_AS_PERSON = `${PRONOUNS.join('|')}|${plural(BODY_PARTS)}`
const READS_AS_PERSON_RE = new RegExp(`^(?:${READS_AS_PERSON})$`, 'i')
const DESCRIPTION_PERSON_RE = new RegExp(
  `(?:^|[^a-z])(${ANY_MAN}|${plural([...PERSON_WORDS, ...MORE_PEOPLE])}|${PERSON_ANCHORS.join('|')}|${READS_AS_PERSON})(?=$|[^a-z])`,
  'gi',
)

/**
 * Things that share a word with a person and bring in no one: furniture has
 * arms, legs, a foot and chests ("the arm of the sofa", "at the foot of the
 * bed", "a chest of drawers"), and a German shepherd or a Roman blind is a dog
 * or a window.
 */
const FURNITURE = 'sofa|couch|settee|chair|armchair|bench|table|stool|bed|desk'
const FURNITURE_PARTS_RE = new RegExp(
  `(?:^|[^a-z])(?:(?:arms?|legs?|foot) of (?:the |a |an )?(?:${FURNITURE})|(?:${FURNITURE}) (?:arms?|legs?)|chests? of drawers|german shepherds?|roman blinds?)(?=$|[^a-z])`,
  'gi',
)

/** What each reference's description may name. */
const REF_SUBJECT: Readonly<Record<string, string>> = { cat: 'the cat', scene: 'the room or table' }

/** Every child-coded word in `text`, lower-cased, in order. */
export function childWords(text: string): string[] {
  return [...text.matchAll(CHILD_RE)].map((m) => m[1].toLowerCase())
}

/** True when `text` names an adult or an adult age. */
export function namesAdult(text: string): boolean {
  return ADULT_RE.test(text)
}

/** True when `text` puts a person in the picture. */
export function mentionsPerson(text: string): boolean {
  return new RegExp(PERSON_RE.source, 'i').test(text)
}

const quoted = (words: string[]) => [...new Set(words)].map((w) => `"${w}"`).join(', ')

/**
 * Why a reference photo's description cannot go into a prompt, in plain
 * sentences, or null when it can. A description names only the reference
 * itself (the cat, the room), so it may not bring in a person at all, adult
 * or not: no people, no owner, no "my", no one holding the cat, no hands or
 * lap. It goes into prompts that otherwise show no one, and refusing every
 * person is simpler and safer than asking for an adult's age in a pet's
 * description. When the only words found are "her", "his" or a part of the
 * body, which may well be about the cat, the refusal says just that.
 */
export function descriptionProblem(id: string, raw: string): string | null {
  // Read the words as they will be saved: runs of spaces and line breaks
  // become one space, so "picked  up" split over two lines is still caught.
  const text = raw.replace(/\s+/g, ' ').trim()
  const out: string[] = []
  const child = childWords(text)
  if (child.length) out.push(`The ${id} description uses ${quoted(child)}, which no lab prompt may use.`)
  const found = [...text.replace(FURNITURE_PARTS_RE, ' ').matchAll(DESCRIPTION_PERSON_RE)].map((m) => m[1].toLowerCase())
  const subject = REF_SUBJECT[id] ?? 'the cat or object in the photo'
  if (found.some((w) => !READS_AS_PERSON_RE.test(w))) {
    out.push(`The ${id} description brings a person into the picture (${quoted(found)}). Describe only ${subject}, with no one in it, not even a hand or a lap.`)
  } else if (found.length) {
    const instead = id === 'cat' ? "Call the cat 'it' and describe only the cat." : `Describe only ${subject}, with no one in it.`
    out.push(`The ${id} description uses ${quoted(found)}. Words like 'her', 'his' or 'hand' read as a person. ${instead}`)
  }
  return out.length ? out.join(' ') : null
}

/**
 * A suite, typed and frozen. It is not checked here, because a chain may name
 * a slot of another suite in the same study; `checkSuite` with that context,
 * and `expand`, which refuses a suite that fails it, do the checking.
 */
export function defineSuite(s: Suite): Suite {
  return deepFreeze(s)
}

function deepFreeze<T>(v: T): T {
  if (v && typeof v === 'object' && !Object.isFrozen(v)) {
    Object.freeze(v)
    for (const k of Object.keys(v)) deepFreeze((v as Record<string, unknown>)[k])
  }
  return v
}

/** The set a slot belongs to: its own id unless it names one. */
export function setOf(slot: Slot): string {
  return slot.set ?? slot.id
}

/**
 * The slot a chain's compose names, found in the suite itself or in the other
 * suites of its study.
 */
export function findSlot(id: string, s: Suite, context: readonly Suite[] = []): { slot: Slot; suite: Suite } | null {
  const found: { slot: Slot; suite: Suite }[] = []
  for (const suite of [s, ...context]) {
    if (suite !== s && suite.study !== s.study) continue
    const slot = suite.slots.find((x) => x.id === id)
    if (slot) found.push({ slot, suite })
  }
  // A slot copied only for its words (the calibration's, with no models of
  // its own) is the last choice: the picture belongs to the suite that draws it.
  return found.find((f) => !(Array.isArray(f.slot.models) && f.slot.models.length === 0)) ?? found[0] ?? null
}

/** The text a chain step sends, with {text} standing for the compose slot's words. */
export function chainStepText(step: ChainSpec['steps'][number], composeText: string): string {
  return step.text.split('{text}').join(composeText)
}

/**
 * Every problem with a suite, in plain words; empty when it is fine.
 * `context` is the other suites of the study, where a chain's compose slot may
 * live. Without it, a compose slot that is not in this suite is not reported.
 */
export function checkSuite(s: Suite, context: readonly Suite[] = []): string[] {
  const out: string[] = []
  const say = (m: string) => out.push(m)

  if (!s.id) say('The suite has no id.')
  if (!s.study) say(`Suite ${s.id} names no study.`)
  if (!Number.isInteger(s.version) || s.version < 1) say(`Suite ${s.id} needs a whole version number of 1 or more.`)
  if (!s.seeds.length) say(`Suite ${s.id} has no seeds.`)
  if (new Set(s.seeds).size !== s.seeds.length) say(`Suite ${s.id} lists a seed twice.`)
  for (const seed of s.seeds) if (!Number.isInteger(seed) || seed < 0) say(`Seed ${seed} is not a whole number of 0 or more.`)
  if (!Number.isInteger(s.steps) || s.steps < 1) say(`Suite ${s.id} needs a whole step count of 1 or more.`)

  // Models are the app's own files.
  for (const [key, m] of Object.entries(s.models)) {
    const def = familyOwning(m.file)
    if (!def) {
      say(`Model ${key} (${m.file}) is not a file the app's registry knows.`)
      continue
    }
    if ((m.role === 'edit') !== (def.mode === 'edit')) {
      say(`Model ${key} is marked ${m.role ?? 'generate'} but the app runs it as ${def.mode === 'edit' ? 'an editing model' : 'a picture model'}.`)
    }
    if (def.mode === 'video') say(`Model ${key} is a video model, which the lab does not test.`)
  }
  for (const key of s.core) {
    if (!s.models[key]) say(`Core model ${key} is not in the suite's models.`)
    else if (s.models[key].role === 'edit') say(`Core model ${key} is the editing model; the core holds picture models only.`)
  }
  if (new Set(s.core).size !== s.core.length) say(`Suite ${s.id} lists a core model twice.`)

  for (const [id, r] of Object.entries(s.refs)) {
    const problem = descriptionProblem(id, r.describe)
    if (problem) say(problem)
  }

  // Ids.
  const ids = new Set<string>()
  for (const x of [...s.slots.map((sl) => sl.id), ...s.chains.map((c) => c.id)]) {
    if (ids.has(x)) say(`The id ${x} is used twice.`)
    ids.add(x)
  }

  const modelKnown = (k: string) => k in s.models
  const listed = (models: Slot['models']) => (Array.isArray(models) ? models : [])

  for (const slot of s.slots) {
    const where = `Slot ${slot.id}`
    if (!(slot.shape in s.shapes)) say(`${where} asks for shape ${slot.shape}, which the suite does not define.`)
    if (!OPS.includes(slot.op)) say(`${where} asks for operation ${slot.op}, which the lab does not know.`)
    for (const k of listed(slot.models)) if (!modelKnown(k)) say(`${where} names model ${k}, which the suite does not list.`)
    for (const k of Object.keys(slot.perModelText ?? {})) if (!modelKnown(k)) say(`${where} has wording for model ${k}, which the suite does not list.`)
    if (slot.perModelText && Object.keys(slot.perModelText).length && !slot.task) {
      say(`${where} gives models their own wording, so it needs a neutral task for the phone.`)
    }
    const src = slot.source
    if (src && 'ref' in src && !(src.ref in s.refs)) say(`${where} uses reference ${src.ref}, which the suite does not list.`)
    if (src && 'cell' in src) {
      if (!findSlot(src.cell.slot, s, context) && context.length) say(`${where} starts from slot ${src.cell.slot}, which does not exist in study ${s.study}.`)
      if (!modelKnown(src.cell.model)) say(`${where} starts from model ${src.cell.model}, which the suite does not list.`)
    }
    if (['i2i', 'edit', 'region', 'detailOnPicture'].includes(slot.op) && !src) say(`${where} is ${slot.op} but has no source picture.`)
    if (slot.op === 'region') {
      const ref = src && 'ref' in src ? s.refs[src.ref] : null
      if (!ref?.needsMask) say(`${where} is a region edit, so its reference must need a mask.`)
    }
    if (src && 'chainStep' in src) say(`${where} starts from a chain step, which only a chain can do.`)
    const texts = [slot.text, ...Object.values(slot.perModelText ?? {})]
    if (texts.some((t) => t.includes('{describe}')) && !(src && 'ref' in src)) say(`${where} uses {describe} without a reference.`)
    if (texts.some((t) => t.includes('{position}')) && !(src && 'ref' in src && s.refs[src.ref]?.needsMask)) {
      say(`${where} uses {position}, which needs a reference with a drawn rectangle.`)
    }
    if (slot.after && !s.slots.some((x) => x.id === slot.after)) say(`${where} is compared with slot ${slot.after}, which does not exist.`)
    if ((slot.op === 'face' || slot.op === 'hand') && slot.target && slot.target !== slot.op) {
      say(`${where} is a ${slot.op} pass but names target ${slot.target}.`)
    }
    for (const w of slot.expect ?? []) if (!slot.text.includes(w)) say(`${where} expects the words "${w}", which are not in its prompt.`)
    for (const t of [...texts, slot.task ?? '', slot.negativeAdd ?? '', slot.condition ?? '', slot.layout ?? '', ...(slot.checklist ?? [])]) {
      for (const w of childWords(t)) say(`${where} uses "${w}", which no lab prompt may use.`)
    }
    if (slot.denoise !== undefined && !(slot.denoise > 0 && slot.denoise <= 1)) say(`${where} has a denoise outside 0 to 1.`)
    for (const t of texts) checkPeople(where, slot, t, say)
  }

  // Sets: members share a block, an operation and a model list, and each has its own label.
  const sets = new Map<string, Slot[]>()
  for (const slot of s.slots) sets.set(setOf(slot), [...(sets.get(setOf(slot)) ?? []), slot])
  for (const [id, members] of sets) {
    if (members.length < 2) continue
    const first = members[0]
    for (const m of members.slice(1)) {
      if (m.block !== first.block) say(`Set ${id} mixes blocks ${first.block} and ${m.block}.`)
      if (m.op !== first.op) say(`Set ${id} mixes operations ${first.op} and ${m.op}.`)
      if (JSON.stringify(m.models) !== JSON.stringify(first.models)) say(`Set ${id} gives its members different models.`)
      if (m.second !== first.second) say(`Set ${id} gives its members different second cards.`)
    }
    const labels = members.map((m) => m.condition)
    if (labels.some((l) => !l)) say(`Set ${id} has a member with no condition label.`)
    else if (new Set(labels).size !== labels.length) say(`Set ${id} uses a condition label twice.`)
  }

  // Chains.
  for (const c of s.chains) {
    const where = `Chain ${c.id}`
    const found = findSlot(c.compose.slot, s, context)
    // A compose slot of another suite can only be looked for with that suite in hand.
    if (!found && context.length) say(`${where} starts from slot ${c.compose.slot}, which does not exist in study ${s.study}.`)
    if (!modelKnown(c.compose.model)) say(`${where} starts from model ${c.compose.model}, which the suite does not list.`)
    if (!c.steps.length) say(`${where} has no steps.`)
    for (const [i, step] of c.steps.entries()) {
      if (!modelKnown(step.model)) say(`${where} step ${i + 1} names model ${step.model}, which the suite does not list.`)
      if (!OPS.includes(step.op) || step.op === 't2i') say(`${where} step ${i + 1} must work on a picture, not ${step.op}.`)
      for (const w of childWords(step.text)) say(`${where} step ${i + 1} uses "${w}", which no lab prompt may use.`)
      // A step works on a picture its compose prompt made, where the adult
      // rule already applied. A step whose own words bring in a person must
      // name an adult as well.
      const text = found ? chainStepText(step, found.slot.text) : step.text
      if (mentionsPerson(text) && !namesAdult(text)) say(`${where} step ${i + 1} puts a person in its prompt, so it must name an adult: "${text}".`)
    }
  }

  if (s.sweep) {
    for (const k of s.sweep.models) {
      if (!modelKnown(k)) say(`The step sweep names model ${k}, which the suite does not list.`)
      if (!(k in s.sweep.homeFor)) say(`The step sweep has no home step count for ${k}.`)
    }
    for (const id of s.sweep.slots) if (!s.slots.some((x) => x.id === id)) say(`The step sweep uses slot ${id}, which the suite does not define.`)
    for (const n of s.sweep.steps) if (!Number.isInteger(n) || n < 1) say(`The step sweep has a step count ${n} that is not a whole number of 1 or more.`)
  }
  if (s.samplerCheck) {
    for (const k of s.samplerCheck.models) if (!modelKnown(k)) say(`The sampler check names model ${k}, which the suite does not list.`)
    for (const id of s.samplerCheck.slots) if (!s.slots.some((x) => x.id === id)) say(`The sampler check uses slot ${id}, which the suite does not define.`)
  }

  return out
}

/** The adult rule for one prompt of a slot. */
function checkPeople(where: string, slot: Slot, text: string, say: (m: string) => void) {
  if (slot.defaultsProbe) return
  if (slot.humans === true) {
    if (!namesAdult(text)) say(`${where} shows a person, so its prompt must name an adult: "${text}".`)
    return
  }
  if (slot.humans === undefined && mentionsPerson(text)) {
    say(`${where} puts a person in its prompt, so mark it humans: true and name an adult (or humans: false if it asks for no one).`)
  }
}
