/**
 * The compose rail.
 *
 * One import for the desk: `ComposeDesk` is the whole default screen, four
 * controls and a footnote, with the advanced apparatus passed in as a child so
 * that nothing the desk ever did has been removed, only moved.
 *
 *   import { ComposeDesk } from '../components/compose'
 *
 *   <ComposeDesk
 *     prompt={c.prompt} look={look} anatomy={anatomy}
 *     onPrompt={setPrompt} onLook={setLook} onAnatomy={setAnatomy}
 *     recipe={decide({ prompt: c.prompt, look, anatomy, hardware, sizes, installed, loras })}
 *     source={c.source} onPickSource={pick} onClearSource={clear}
 *     onRun={run} onStop={stop} running={busy} job={job}
 *     advanced={<EverythingElse />}
 *   />
 *
 * The pieces are exported individually too, for a desk that wants to set its
 * own page: they hold no state between them and no context.
 */
export { ComposeDesk, type ComposeDeskProps } from './ComposeDesk'
export { PromptField } from './PromptField'
export { LookPicker } from './LookPicker'
export { AnatomyPicker } from './AnatomyPicker'
export { SourceWell } from './SourceWell'
export { RunButton, type RunJob } from './RunButton'
export { RecipeProse } from './RecipeProse'
export { MoreFootnote } from './More'
export {
  Choice,
  Deck,
  EditorialLink,
  Hairline,
  Kicker,
  Label,
  Notice,
  type ChoiceOption,
  type NoticeKind,
} from './bits'
