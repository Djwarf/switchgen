/**
 * The add-on rack on the Video desk.
 *
 * The Pictures desk decides a stack from measurements and offers the rest;
 * nothing on a clip has been measured that way, so this rack is the reader's
 * alone: the same rows and the same picker as the picture rack, at the
 * author's strengths, and every figure it prints says which author.
 *
 * On a two-half family a pair of files is one row, and the picker offers it
 * by its HIGH half. A HIGH row can give its LOW half a row of its own, which
 * is how the two halves get two strengths. See lib/videoLoras.ts.
 *
 * The files a family already loads in its own graph are shown as in use and
 * never chained a second time, because chaining one again in front of the
 * family's own copy would patch each half with the same weights twice.
 */
import { useMemo, useState } from 'react'
import {
  addToStack,
  fitFor,
  moveInStack,
  patchStack,
  removeFromStack,
  size,
  targetFor,
  type LoraInfo,
  type LoraLibrary,
  type LoraStack,
} from '../../lib/loras'
import { canTakeVideoLoras } from '../../lib/refine'
import { builtInLoras, familyLoras, pairedHalf, partnerOf } from '../../lib/videoLoras'
import type { FamilyDef } from '../../lib/workflows'
import { Caution, Head, Note, Quiet } from '../advanced/bits'
import { Picker } from '../loras/Picker'
import { Row as LoraRow } from '../loras/Row'

export function LoraRack({
  def,
  model,
  lib,
  stack,
  onStack,
  onLibraryReload,
}: {
  def: FamilyDef
  model: string
  lib: LoraLibrary
  stack: LoraStack
  onStack: (next: LoraStack) => void
  onLibraryReload?: () => void
}) {
  const [picking, setPicking] = useState(false)
  const target = useMemo(() => targetFor(def, model), [def, model])
  const own = useMemo(() => familyLoras(def), [def])
  const builtIn = useMemo(() => builtInLoras(def), [def])
  const onRack = useMemo(() => new Map(stack.map((e) => [e.file, e])), [stack])
  // Present means the picker offers it as in use rather than as a second copy:
  // a row on the rack, the other half of a pair that has a row, or a file the
  // family's own graph already loads.
  const inStack = useMemo(() => {
    const out = new Set<string>(builtIn)
    for (const e of stack) {
      out.add(e.file)
      const partner = def.dualModel ? partnerOf(e.file) : null
      if (partner) out.add(partner)
    }
    return out
  }, [stack, builtIn, def.dualModel])
  // A pair is offered once, by its HIGH half, which brings the LOW with it.
  const pickerLib = useMemo<LoraLibrary>(() => {
    if (!def.dualModel) return lib
    const all = lib.all.filter((i) => {
      const half = pairedHalf(i.file)
      const partner = half?.half === 'low' ? partnerOf(i.file) : null
      return !(partner && lib.byFile.get(partner)?.installed)
    })
    return all.length === lib.all.length ? lib : { ...lib, all }
  }, [lib, def.dualModel])
  const carries = canTakeVideoLoras(def)
  const enabled = stack.filter((e) => e.enabled).length
  const bytes = useMemo(
    () =>
      stack.reduce((sum, e) => {
        // A row at 0 chains nothing, and neither does the partner it pulls in.
        if (!e.enabled || e.strength === 0 || builtIn.has(e.file)) return sum
        const info = lib.byFile.get(e.file)
        const partner = def.dualModel ? partnerOf(e.file) : null
        const twin = partner ? lib.byFile.get(partner) : undefined
        // The partner is only pulled in when it has no row of its own. With a
        // row it is counted there, and switched off it is not loaded at all.
        const pulledIn = !!partner && !!twin?.installed && !onRack.has(partner)
        return sum + (info?.bytes ?? 0) + (pulledIn && twin ? twin.bytes : 0)
      }, 0),
    [stack, lib, def.dualModel, builtIn, onRack],
  )
  const ownNames = useMemo(() => {
    const labelOf = (file: string) =>
      (lib.byFile.get(file)?.label ?? file.replace(/\.[^.]+$/, '')).replace(/[\s_-]+(HIGH|LOW)$/i, '')
    const out: string[] = []
    const done = new Set<string>()
    for (const l of own) {
      if (done.has(l.name)) continue
      done.add(l.name)
      const partner = partnerOf(l.name)
      const twin = partner ? own.find((o) => o.name === partner) : undefined
      if (twin && twin.strength === l.strength) {
        done.add(twin.name)
        out.push(`${labelOf(l.name)}, HIGH and LOW, at ${l.strength}`)
      } else {
        out.push(`${lib.byFile.get(l.name)?.label ?? l.name} at ${l.strength}`)
      }
    }
    return out
  }, [own, lib])

  if (!carries) {
    return (
      <section className="mt-4 border-t border-grey-300 pt-3">
        <Head title="Add-ons" />
        <Note>{def.label} loads its model in a shape this desk cannot chain an add-on into, so none are offered.</Note>
      </section>
    )
  }

  return (
    <section className="mt-4 border-t border-grey-300 pt-3">
      <Head
        title="Add-ons"
        figure={stack.length ? `${enabled} of ${stack.length} on` : 'none'}
        note={
          def.dualModel
            ? 'This family runs in two halves. A pair of files with the same stem, one HIGH and one LOW, goes one to each half: one row at one strength, or a row for each half at two.'
            : 'At the author’s strengths. Nothing on a clip has been measured the way the picture stacks were.'
        }
      />

      {own.length ? (
        <Caution>
          {def.label} already loads {ownNames.join('; ')}, so {own.length === 1 ? 'that file is' : 'those files are'}{' '}
          not offered again. Anything added here goes in front of them and costs memory on top; the model's card
          below says what this machine survived.
        </Caution>
      ) : null}

      {stack.length ? (
        <ul className="mt-2 border-t border-grey-300">
          {stack.map((entry, i) => {
            const info = lib.byFile.get(entry.file)
            const half = def.dualModel ? pairedHalf(entry.file) : null
            const partner = half ? partnerOf(entry.file) : null
            const twin = partner ? lib.byFile.get(partner) : undefined
            const twinRow = partner ? onRack.get(partner) : undefined
            // Whether the partner's own row reaches its half. A row that is
            // on but cannot load is left out, and this file then goes to both
            // halves, the same as when the partner's row is switched off.
            const twinRuns =
              !!twinRow?.enabled && !!twin?.installed && fitFor(twin, target).level !== 'mismatch'
            const HALF = half?.half.toUpperCase()
            const OTHER = half?.half === 'high' ? 'LOW' : 'HIGH'
            // A HIGH row whose LOW half is installed and has no row can hand
            // that half a row of its own, at this row's strength to start.
            const canSplit = half?.half === 'high' && !!partner && !twinRow && !!twin?.installed
            return (
              <li key={entry.file} className="list-none">
                <LoraRow
                  entry={entry}
                  info={info}
                  fit={info ? fitFor(info, target) : { level: 'untested', why: 'This file is no longer in the add-ons folder.' }}
                  index={i}
                  count={stack.length}
                  expert
                  clipPatched={false}
                  onPatch={(patch) => onStack(patchStack(stack, entry.file, patch))}
                  onMove={(to) => onStack(moveInStack(stack, i, to))}
                  onRemove={() => onStack(removeFromStack(stack, entry.file))}
                />
                {builtIn.has(entry.file) ? (
                  <p className="mb-2 text-caption italic text-grey-500">
                    {def.label} already loads this file itself, so this row is left out rather than loaded twice.
                  </p>
                ) : half ? (
                  <div className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <p className="text-caption italic text-grey-500">
                      {twinRuns && twinRow?.strength === 0
                        ? `The ${HALF} half, at this row’s strength. Its ${OTHER} partner’s row is at 0, so the other half gets neither file.`
                        : twinRuns
                          ? `The ${HALF} half, at this row’s strength. Its ${OTHER} partner has a row of its own and goes to the other half at that row’s strength.`
                          : twinRow?.enabled
                            ? `The ${HALF} half. Its ${OTHER} partner’s row cannot be loaded, so this goes to both halves at the same strength, which nobody has measured.`
                            : twinRow
                              ? `The ${HALF} half. Its ${OTHER} partner is switched off, so this goes to both halves at the same strength, which nobody has measured.`
                              : twin?.installed
                                ? `The ${HALF} half. Its ${OTHER} partner is installed and goes to the other half at the same strength.`
                                : `The ${HALF} half only. No partner file is installed, so this goes to both halves at the same strength, which nobody has measured.`}
                    </p>
                    {canSplit && twin ? (
                      <button
                        type="button"
                        className="text-caption text-burgundy-900 underline"
                        onClick={() => {
                          const next = [...stack]
                          next.splice(i + 1, 0, {
                            file: twin.file,
                            strength: entry.strength,
                            clipStrength: entry.clipStrength,
                            enabled: entry.enabled,
                          })
                          onStack(next)
                        }}
                      >
                        Give the {OTHER} half its own strength
                      </button>
                    ) : null}
                  </div>
                ) : def.dualModel ? (
                  <p className="mb-2 text-caption italic text-grey-500">
                    Not a HIGH/LOW pair, so it goes to both halves at the same strength. Not measured.
                  </p>
                ) : null}
              </li>
            )
          })}
        </ul>
      ) : (
        <Note>No add-ons. The model is making this on its own.</Note>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Quiet onClick={() => setPicking((v) => !v)} pressed={picking}>
          {picking ? 'Close' : 'Browse all add-ons'}
        </Quiet>
        {stack.length ? <Quiet onClick={() => onStack([])}>Remove all</Quiet> : null}
        {bytes ? <span className="text-caption tabular-nums text-grey-500">{size(bytes)} of extra files to load</span> : null}
      </div>

      {picking ? (
        <div className="mt-3 border border-grey-300 p-3">
          <Picker
            lib={pickerLib}
            target={target}
            inStack={inStack}
            onAdd={(info: LoraInfo) => onStack(addToStack(stack, info))}
            onFetched={() => onLibraryReload?.()}
            onClose={() => setPicking(false)}
          />
        </div>
      ) : null}
    </section>
  )
}
