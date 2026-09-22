/**
 * The footnote that opens the rest of the apparatus.
 *
 * It is printed at caption size, under a hairline, at the foot of the column,
 * and it is a link rather than a button because it must not read as a second
 * thing to press. The one control on this page that does something is the
 * burgundy one above it.
 *
 * What it opens is not written here and is not owned here. The advanced panel
 * is passed in as a child: every control the desk ever had, in full, relocated
 * rather than removed. This component's only job is that the default screen
 * does not have to carry it.
 *
 * The children are mounted only while it is open, so a closed panel costs no
 * tab stops, no ranking work and no LoRA rows.
 */
import { useId, useState, type ReactNode } from 'react'
import { EditorialLink, Hairline } from './bits'

export function MoreFootnote({
  children,
  open: controlled,
  onOpenChange,
  summary = 'Every model, every add-on, every setting, and why each one was chosen. Nothing was taken away - it was moved here.',
  aside,
}: {
  children?: ReactNode
  /** Controlled when provided. Wire it to the expert flag if you want it sticky. */
  open?: boolean
  onOpenChange?: (open: boolean) => void
  summary?: string
  /** One more quiet sentence on the footnote line, when the desk has one to say. */
  aside?: ReactNode
}) {
  const panelId = useId()
  const [local, setLocal] = useState(false)
  const open = controlled ?? local

  const toggle = () => {
    const next = !open
    if (controlled === undefined) setLocal(next)
    onOpenChange?.(next)
  }

  return (
    <div className="mt-10">
      <Hairline className="mb-2" />
      <p className="max-w-[62ch] text-caption text-grey-500">
        <EditorialLink onClick={toggle} expanded={open} controls={panelId}>
          {open ? 'Less' : 'More'}
        </EditorialLink>{' '}
        {summary}
      </p>
      {aside && <p className="mt-1 max-w-[62ch] text-caption text-grey-500">{aside}</p>}
      <div id={panelId} className="mt-6">
        {open && children}
      </div>
    </div>
  )
}
