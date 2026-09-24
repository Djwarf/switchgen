/**
 * What the archive says once files are deleted, kept apart from the page so
 * the wording can be checked without one.
 */

/** One record's deletion, as the banner needs it. */
export type Deleted = { gone: number; left: readonly string[]; kept: readonly string[] }

/**
 * How many files went and with how many records, then any the records named
 * that are still on disk: those that would not go, and last frames the reel
 * still opens a shot on. A record's run can write more than one file, so the
 * files are counted, not the records.
 */
export function deletedText(results: readonly Deleted[]): string {
  const records = results.length
  const files = results.reduce((n, r) => n + r.gone, 0)
  const left = results.flatMap((r) => r.left)
  const kept = results.flatMap((r) => r.kept)
  const said =
    records === 1 && files === 1
      ? ['One file has gone from the outputs folder, and its record with it.']
      : [
          `${files} files have gone from the outputs folder, and their ${
            records === 1 ? 'record' : `${records} records`
          } with them.`,
        ]
  if (left.length === 1) said.push(`One more, ${left[0]}, could not be deleted and is still on disk.`)
  else if (left.length) said.push(`${left.length} more could not be deleted and are still on disk: ${left.join(', ')}.`)
  if (kept.length === 1) said.push(`${kept[0]} stays on disk, because the reel still opens a shot on it.`)
  else if (kept.length) said.push(`${kept.length} last frames stay on disk, because the reel still opens shots on them.`)
  return said.join(' ')
}
