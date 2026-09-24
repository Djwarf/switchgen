/**
 * The reel desk's public surface.
 *
 * Everything the route uses is re-exported here, so the route imports from one
 * place and the file layout inside this folder stays ours to change.
 */
export {
  Caution,
  Chips,
  Field,
  Head,
  Kicker,
  Leader,
  Link,
  Mark,
  NumberField,
  Quiet,
  Rail,
  RING,
  clamp,
  duration,
  grouped,
  seconds,
  times,
  type ChipOption,
} from './bits'

export { Bench, type BenchProps, type NumSpec, type ReelFamily, type Shape } from './Bench'
export { lengthsFor, type LengthChoice } from './lengths'
export { Strip, type StripProps } from './Strip'
export { EmptyStrip } from './EmptyStrip'
export { ReelProgress, measuredFor, type Measured, type ReelProgressProps } from './Progress'
export { Assembly, type AssemblyClip, type AssemblyProps } from './Assembly'
export { KeyframePicker, type KeyframePickerProps, type KeyframeTarget } from './Keyframe'
export { recipeFor } from './recipe'

export {
  REEL_KEY,
  blankDraft,
  newId,
  newShot,
  reel,
  seedsToKeep,
  shotSeed,
  shotsFromLines,
  useReel,
  type PinnedFrame,
  type ReelDraft,
  type ReelShot,
} from './store'

export {
  currencyOf,
  reelRun,
  shotsToRender,
  useReelRun,
  waitingInPage,
  type Currency,
  type Elsewhere,
  type Made,
  type RunContext,
  type RunState,
  type RunStatus,
  type ShotState,
  type ShotStatus,
} from './engine'
