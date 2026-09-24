/**
 * The first of the three answers: what the picture is of.
 *
 * One textarea, set at a printed measure. It is the largest thing on the
 * screen after the headline because it is the only field whose content the
 * reader actually has to invent. Everything the old rail asked underneath it
 * is now derived from these words.
 *
 * It does not print the quality prefix, the trigger tokens or the model's tag
 * style. Those are decisions, and decisions are printed once, in prose, under
 * the button. A caption per field is how the old screen got to forty two.
 */
import type { KeyboardEvent as ReactKeyboardEvent, RefObject } from 'react'
import { Label } from './bits'

export function PromptField({
  value,
  onChange,
  onSubmit,
  textRef,
  label = 'The picture',
  placeholder = 'A rain-slicked tram stop at dusk, neon in the puddles',
  rows = 5,
  disabled = false,
}: {
  value: string
  onChange: (v: string) => void
  /** Ctrl+Enter, the only keyboard shortcut this field owns. */
  onSubmit?: () => void
  textRef?: RefObject<HTMLTextAreaElement | null>
  label?: string
  placeholder?: string
  rows?: number
  disabled?: boolean
}) {
  const onKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (!onSubmit) return
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      onSubmit()
    }
  }

  return (
    <label className="block">
      <Label>{label}</Label>
      <textarea
        ref={textRef}
        value={value}
        rows={rows}
        spellCheck
        disabled={disabled}
        onChange={e => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        className="field"
        style={{ fontSize: '1.125rem', lineHeight: 1.6, maxWidth: '62ch' }}
      />
      {/* A key combination, so it is not offered where there is no keyboard:
          on a phone it named a key the reader does not have. */}
      <span className="mt-1 block text-caption italic text-grey-500 [@media(hover:none)]:hidden">
        Ctrl+Enter runs it.
      </span>
    </label>
  )
}
