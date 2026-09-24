/**
 * The graph, as JSON, with a copy button.
 *
 * This is the derived graph the run button would send, not the bare family one.
 * With a detail pass, a two pass render or a LoRA chain switched on, the two
 * differ by several nodes and a rewired decode, and printing the bare graph
 * under a Copy button would put JSON on screen that reproduces none of the
 * picture. That was a real bug in the old desk and it is worth not repeating.
 *
 * It is collapsed by default. Six hundred lines of JSON is the single densest
 * thing in the application and it earns its place only when somebody is
 * debugging a node, which is exactly when a fault node id is worth naming.
 */
import { useMemo, useState } from 'react'

import { copyText } from '../../lib/clipboard'
import { Head, Link, Note } from './bits'
import { buildGraph, type Settled } from './overrides'

export function WorkflowPeek({
  settled,
  faultNode,
}: {
  settled: Settled
  /** The node ComfyUI complained about, when it complained about one. */
  faultNode?: string | null
}) {
  const [open, setOpen] = useState(false)
  /** What the Copy link last came to, said in its place for a moment. */
  const [copied, setCopied] = useState<'copied' | 'failed' | null>(null)

  const json = useMemo(() => {
    if (!open) return ''
    try {
      return JSON.stringify(buildGraph(settled), null, 2)
    } catch (err) {
      return `Could not build the graph: ${err instanceof Error ? err.message : String(err)}`
    }
  }, [open, settled])

  const nodes = Object.keys(settled.def.graph).length

  return (
    <section className="mb-7">
      <Head
        title="The workflow"
        figure={`${nodes} nodes`}
        note={
          settled.rebuilt
            ? 'Rebuilt from the family graph because a pass or the add-on chain changed it.'
            : 'The graph the recipe built, unchanged.'
        }
      />

      <p className="text-caption">
        <Link onClick={() => setOpen((v) => !v)}>
          {open ? 'Hide the JSON' : 'Show the JSON as it will be sent'}
        </Link>
        {open ? (
          <>
            <span className="text-grey-400"> · </span>
            {/* navigator.clipboard exists only on a secure page, and this one
                is usually reached over plain http, where the old handler did
                nothing at all and said nothing either. copyText falls back to
                the older copy command, and a copy that fails is said. */}
            <Link
              onClick={() => {
                void copyText(json).then((ok) => {
                  setCopied(ok ? 'copied' : 'failed')
                  setTimeout(() => setCopied(null), ok ? 1500 : 3000)
                })
              }}
            >
              {copied === 'copied' ? 'Copied' : copied === 'failed' ? 'Could not copy' : 'Copy'}
            </Link>
          </>
        ) : null}
      </p>

      {faultNode ? (
        <Note>Node {faultNode} is the one ComfyUI complained about on the last run.</Note>
      ) : null}

      {open ? (
        <pre className="mt-2 max-h-80 overflow-auto border border-grey-300 bg-newsprint-aged p-2 text-[11px] leading-snug">
          {json}
        </pre>
      ) : null}
    </section>
  )
}
