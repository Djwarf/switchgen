/**
 * The strip before it has a shot on it.
 *
 * The desk used to open on three blank rows, which reads as a form to fill in
 * rather than an invitation. This is the invitation: what a reel is, three
 * that can be laid out with one press, and the two other ways in.
 */
import { Head, Quiet } from './bits'

const EXAMPLES: readonly { title: string; lines: readonly string[] }[] = [
  {
    title: 'A late arrival',
    lines: [
      'A car pulls up outside a shuttered shop, rain on the windscreen',
      'The driver steps out into the rain and looks up',
      'A light comes on in the window above the shop',
    ],
  },
  {
    title: 'Out past the breakwater',
    lines: [
      'Dawn over a small harbour, gulls wheeling',
      'A fishing boat noses out past the breakwater',
      'The town shrinks to a line of lights on the water',
    ],
  },
  {
    title: 'The kettle',
    lines: [
      'A kettle comes to the boil in an empty kitchen',
      'Steam fogs the window over the sink',
      'Outside, a bicycle leans on the garden gate',
    ],
  },
]

export function EmptyStrip({
  onLayOut,
  onStartOne,
  onPaste,
}: {
  /** Lay the given lines out as shots. */
  onLayOut: (text: string) => void
  /** Add a single blank shot to write into. */
  onStartOne: () => void
  /** Open the paste box. */
  onPaste: () => void
}) {
  return (
    <section>
      <Head title="Nothing on the strip yet" note="A reel is a list of shots. Each opens on the last frame of the one before." />
      <p className="dropcap max-w-[62ch] text-body leading-relaxed text-ink">
        Write one line for each shot, in order, and press once. The desk renders them in turn,
        hands the last frame of each to the next, and joins the clips at the end. Three reels to
        try, or start with a line of your own.
      </p>

      <ol className="mt-5 grid gap-4 sm:grid-cols-3">
        {EXAMPLES.map((ex) => (
          <li key={ex.title} className="border-t border-grey-300 pt-2">
            <p className="text-small font-semibold text-ink">{ex.title}</p>
            <ol className="mt-1 space-y-0.5 text-caption text-grey-700">
              {ex.lines.map((l, i) => (
                <li key={i}>
                  <span className="mr-1 tabular-nums text-grey-500">{i + 1}.</span>
                  {l}
                </li>
              ))}
            </ol>
            <div className="mt-2">
              <Quiet onClick={() => onLayOut(ex.lines.join('\n'))}>Lay it out</Quiet>
            </div>
          </li>
        ))}
      </ol>

      <div className="mt-6 flex flex-wrap items-baseline gap-4 border-t border-grey-300 pt-3">
        <Quiet onClick={onStartOne}>Start with one shot</Quiet>
        <Quiet onClick={onPaste}>Paste a whole reel</Quiet>
      </div>
    </section>
  )
}
