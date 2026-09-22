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
export { Strip, type StripProps } from './Strip'
export { EmptyStrip } from './EmptyStrip'
export { ReelProgress, measuredFor, type Measured, type ReelProgressProps } from './Progress'
export { Assembly, type AssemblyClip, type AssemblyProps } from './Assembly'
export { KeyframePicker, type KeyframePickerProps, type KeyframeTarget } from './Keyframe'

export {
  REEL_KEY,
  blankDraft,
  newId,
  newShot,
  reel,
  shotsFromLines,
  useReel,
  type PinnedFrame,
  type ReelDraft,
  type ReelShot,
} from './store'

export {
  reelRun,
  useReelRun,
  type RunContext,
  type RunState,
  type RunStatus,
  type ShotState,
  type ShotStatus,
} from './engine'
