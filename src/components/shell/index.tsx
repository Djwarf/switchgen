/**
 * The shell's public surface.
 *
 * Everything another part of the app may use is re-exported here, so imports
 * read `from './components/shell'` and the internal file layout stays ours to
 * change.
 */

// The frame
export { Shell, type ShellProps } from './Shell'
export { Masthead, britishDate, useEntryCount, type MastheadProps } from './Masthead'
export { SectionBar } from './SectionBar'
export { RunningSlug, type RunningSlugProps } from './RunningSlug'
export { Offline, useConnection } from './Offline'

// Where we are
export {
  SECTIONS,
  SECTION_LABEL,
  SECTION_STANDFIRST,
  currentRoute,
  deskForSection,
  go,
  goToSection,
  normaliseHash,
  parseRoute,
  requestSearchFocus,
  routeHref,
  sectionForDesk,
  sectionHref,
  setArchiveQuery,
  useRoute,
  useSection,
  FOCUS_SEARCH_EVENT,
  SEARCH_INPUT_ID,
  type Route,
  type Section,
} from './route'

// Simple or all controls
export {
  SettingsToggle,
  ShowAllControlsLink,
  toggleExpert,
  useExpert,
  useSettings,
  type SettingsToggleProps,
} from './SettingsToggle'

// The press ledger
export {
  RECENT_MS,
  elapsedText,
  headline,
  jobs,
  progressOf,
  remainingOf,
  roughText,
  stageFor,
  useJob,
  useJobs,
  type Job,
  type JobDesk,
  type JobInit,
  type JobStatus,
  type JobsSnapshot,
  type ServerQueue,
} from './jobs'
export { mirror, type Bridge, type Reported } from './mirror'

// Telling the reader something
export {
  Notice,
  NoticeRail,
  clearNotices,
  dismissNotice,
  postNotice,
  useNotices,
  type NoticeAction,
  type NoticeInput,
  type NoticeProps,
  type NoticeTone,
  type PostedNotice,
} from './Notice'

// Taking it back
export {
  UNDO_MS,
  UndoBar,
  commitUndo,
  offerUndo,
  undoLast,
  useUndo,
  type UndoInput,
  type UndoOffer,
} from './UndoBar'

// Keys
export {
  isTyping,
  useChord,
  useHoldToConfirm,
  useHotkeys,
  type HoldHandlers,
  type Hotkey,
} from './hotkeys'
export {
  SHORTCUTS,
  Shortcuts,
  closeShortcuts,
  openShortcuts,
  shortcutsOpen,
  toggleShortcuts,
  useShortcutsOpen,
  type Shortcut,
  type ShortcutGroup,
} from './Shortcuts'
