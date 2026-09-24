/**
 * Copying text, on a page that is often not a secure context.
 *
 * navigator.clipboard exists only in a secure context (https, or localhost).
 * The phone reaches this app over plain http on a Tailscale address, where
 * it is undefined, and every copy button there did nothing. The older way
 * still works there: select the text in a field and ask the document to copy
 * it, inside the same tap.
 *
 * Resolves true when the text was copied, false when neither way worked, so
 * the caller can say "Could not copy" instead of implying it did.
 */
export async function copyText(text: string): Promise<boolean> {
  const clip = (globalThis as { navigator?: { clipboard?: { writeText?: (t: string) => Promise<void> } } }).navigator
    ?.clipboard
  if (clip && typeof clip.writeText === 'function') {
    try {
      await clip.writeText(text)
      return true
    } catch {
      // Refused (no permission, or the document lost focus): try the old way.
    }
  }
  return copyBySelection(text)
}

/**
 * The old way: a field off screen, its text selected, and execCommand('copy').
 * Read-only so a phone does not raise its keyboard for it, and put back where
 * it was so the page's selection and focus are not lost.
 */
function copyBySelection(text: string): boolean {
  if (typeof document === 'undefined' || !document.body) return false
  const area = document.createElement('textarea')
  area.value = text
  area.setAttribute('readonly', '')
  area.setAttribute('aria-hidden', 'true')
  area.tabIndex = -1
  // Off screen but still laid out: a field with display none cannot be selected.
  area.style.position = 'fixed'
  area.style.top = '0'
  area.style.left = '-9999px'
  area.style.opacity = '0'
  // 16px, or iOS zooms the page to the field for the moment it is focused.
  area.style.fontSize = '16px'

  const before = document.activeElement as HTMLElement | null
  const selection = document.getSelection()
  const range = selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null

  document.body.appendChild(area)
  let ok = false
  try {
    area.focus({ preventScroll: true })
    area.select()
    // iOS Safari ignores select() on a read-only field without an explicit range.
    area.setSelectionRange(0, text.length)
    ok = document.execCommand('copy')
  } catch {
    ok = false
  } finally {
    area.remove()
    if (range && selection) {
      selection.removeAllRanges()
      selection.addRange(range)
    }
    before?.focus?.({ preventScroll: true })
  }
  return ok
}
