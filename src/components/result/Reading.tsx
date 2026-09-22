/**
 * WHAT IS IN THE PICTURE, read by a machine that looks at pixels.
 *
 * Everything else on this desk reads the prompt. This reads the picture: the
 * WD14 tagger says what is in it and how explicit it is, the YOLO detectors
 * say where the faces and hands are and how big, and the add-on index ranks
 * the installed add-ons against what was seen. It runs on the CPU, on the
 * server, and never touches the card.
 *
 * It is offered, never automatic. A reading costs a couple of seconds and
 * makes a claim about content, and both are the reader's to ask for. Nothing
 * here changes a setting by itself: the rating offers a switch, the tags offer
 * words, the suggestions offer add-ons, and each is one link.
 */
import { useEffect, useState } from 'react'
import { installFiles, type FetchProgress } from '../../lib/downloads'
import type { AnatomyLevel } from '../../lib/recipe'
import type { LoraArch } from '../../lib/loras'
import type { IndexedBase } from '../../lib/loraIndex'
import { size } from '../../lib/loras'
import {
  inspectImage,
  refreshCapabilities,
  watchCapabilities,
  type ImageFacts,
  type ImageRating,
  type ImageSource,
  type VisionCapabilities,
  type VisionReport,
} from '../../lib/vision'
import { Kicker, Link } from '../type'
import { Meter } from '../loras/bits'

/**
 * The base the add-on index files each architecture under. An architecture
 * outside it gets no ranking, not a wrong one. The index knows Wan only as
 * `wan`, so every Wan size reads that one base.
 */
const INDEXED: Readonly<Partial<Record<LoraArch, IndexedBase>>> = {
  pony: 'pony',
  illustrious: 'illustrious',
  sdxl: 'sdxl',
  flux1d: 'flux1d',
  wan: 'wan',
  'wan-14b': 'wan',
  'wan-5b': 'wan',
  'wan-1.3b': 'wan',
}

const RATING_WORD: Record<ImageRating, string> = {
  general: 'general, nothing suggestive',
  sensitive: 'sensitive: suggestive, not explicit',
  questionable: 'questionable: partial nudity or suggestive framing',
  explicit: 'explicit',
}

/** Readings already made this session, so a second look costs nothing. */
const readings = new Map<string, VisionReport>()

export type ReadingProps = {
  /** The picture, as the server can reach it. Null prints nothing. */
  source: ImageSource | null
  /** Identity for the cache: an archive id, or the input filename. */
  cacheKey: string
  /** The architecture the next render will use, for ranking the add-ons. */
  arch?: LoraArch | null
  anatomy?: AnatomyLevel
  /** Offered when the rating says the picture is more explicit than the setting. */
  onAnatomy?: (level: AnatomyLevel) => void
  /** Accept a suggested add-on for the next render. */
  onAddOn?: (file: string) => void
  /** Put the tagger's words into the prompt. */
  onUseWords?: (words: string) => void
  /** The reading landed. For persisting tags, or feeding the detector facts onward. */
  onRead?: (report: VisionReport) => void
  /** Tags already on the record, printed without a read. */
  known?: { tags: readonly string[]; rating: string | null } | null
  /** One line under a source well rather than a section. */
  compact?: boolean
}

function factsSentence(facts: ImageFacts | null): string | null {
  if (!facts) return null
  const pct = (n: number) => `${(n * 100).toFixed(1)}%`
  const part = (list: { areaShare: number }[], word: string) => {
    if (!list.length) return `no ${word}s`
    const largest = Math.max(...list.map((d) => d.areaShare))
    return `${list.length} ${word}${list.length === 1 ? '' : 's'}, the largest ${pct(largest)} of the frame`
  }
  return `The detector found ${part(facts.face, 'face')}; ${part(facts.hand, 'hand')}; ${facts.person.length} ${facts.person.length === 1 ? 'person' : 'people'}.`
}

export function Reading(props: ReadingProps) {
  const { source, cacheKey, compact = false } = props
  const [caps, setCaps] = useState<VisionCapabilities | null>(null)
  const [installing, setInstalling] = useState<FetchProgress | null>(null)
  const [installError, setInstallError] = useState<string | null>(null)

  // Asked again while the server has not answered, so a reading that mounted
  // during a restart offers itself, or the tagger fetch, once the server is back.
  useEffect(() => watchCapabilities(setCaps), [])

  if (!source || !caps) return null

  const install = () => {
    if (!caps.install) return
    const files = caps.install.files.filter((f) => caps.install!.missing.includes(f.filename))
    setInstallError(null)
    setInstalling({ state: 'starting', pct: 0, done: 0, total: 0, speed: 0, etaSec: null, error: null })
    void installFiles(files.length ? files : caps.install.files, setInstalling)
      .then(() => refreshCapabilities())
      .then((c) => setCaps(c))
      .catch((e: unknown) => setInstallError(e instanceof Error ? e.message : String(e)))
      .finally(() => setInstalling(null))
  }

  const text = compact ? 'text-caption' : 'text-small'

  // The tagger is missing and fetching it would fix that: there is an
  // interpreter to run it and the server says what to fetch. This is offered
  // whether or not the detectors are there, because with only them a reading
  // finds faces and hands and no tags. A download that stopped part way is
  // mended the same way: the server lists the unfinished file as missing.
  const offer = !caps.tagger && caps.python && caps.install ? caps.install : null
  const fetchLine = offer
    ? (() => {
        const wanted = offer.files.filter((f) => offer.missing.includes(f.filename))
        const bytes = (wanted.length ? wanted : offer.files).reduce((n, f) => n + f.sizeBytes, 0)
        const unfinished = offer.missing.includes('model.onnx') && (caps.taggerBytes ?? 0) > 0
        return (
          <div className={caps.detect ? `${text} mt-1` : text}>
            {installing ? (
              <>
                <p className="italic text-grey-700">Fetching the tagger, {size(installing.done)} of {size(installing.total || bytes)}.</p>
                <Meter pct={installing.pct} label="Fetching the tagger" />
              </>
            ) : (
              <p className="text-grey-700">
                {unfinished ? "The tagger's download did not finish." : 'The tagger is not installed.'}{' '}
                <Link onClick={install}>{unfinished ? 'Fetch it again' : 'Fetch it'}</Link>, {size(bytes)} from
                HuggingFace.
                {caps.detect
                  ? ' Until then a reading finds the faces and hands but not what is in the picture.'
                  : ' It reads pictures on the CPU and does not touch the card.'}
              </p>
            )}
            {installError ? <p className="mt-1 text-ink-error">{installError}</p> : null}
          </div>
        )
      })()
    : null

  // Nothing installed. Say what is missing and, when it can be fetched, offer it.
  if (!caps.tagger && !caps.detect) {
    return fetchLine ?? <p className={`${text} italic text-grey-500`}>{caps.reason ?? 'No image reading is installed on the server.'}</p>
  }

  // One picture's reading lives and dies with that picture. Keyed, so a read
  // still in flight when the picture changes lands in an instance that is no
  // longer shown, never under the next picture. It still reaches the cache,
  // and the onRead of the picture it was asked for.
  return (
    <>
      <PictureReading key={cacheKey} {...props} source={source} text={text} />
      {fetchLine}
    </>
  )
}

function PictureReading({
  source,
  cacheKey,
  arch = null,
  anatomy,
  onAnatomy,
  onAddOn,
  onUseWords,
  onRead,
  known = null,
  compact = false,
  text,
}: ReadingProps & { source: ImageSource; text: string }) {
  const [report, setReport] = useState<VisionReport | null>(() => readings.get(cacheKey) ?? null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [added, setAdded] = useState<ReadonlySet<string>>(() => new Set())

  const read = () => {
    setBusy(true)
    setError(null)
    const base = arch ? INDEXED[arch] : undefined
    const bases = base ? [base] : undefined
    void inspectImage(source, { bases, limit: 4 })
      .then((r) => {
        readings.set(cacheKey, r)
        setReport(r)
        if (!r.unavailable) onRead?.(r)
        else setError(r.unavailable)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false))
  }

  if (!report) {
    return (
      <div className={text}>
        {known?.tags.length ? (
          <p className="text-grey-700">
            <Kicker className="mr-2">Tagged</Kicker>
            {known.tags.slice(0, 24).map((t) => t.replace(/_/g, ' ')).join(', ')}
            {known.rating ? <span className="text-grey-500"> · reads as {known.rating}</span> : null}
          </p>
        ) : null}
        <p className={known?.tags.length ? 'mt-1 text-grey-500' : 'text-grey-500'}>
          {busy ? (
            <span className="italic">Reading the picture on the CPU.</span>
          ) : (
            <>
              <Link onClick={read}>{known?.tags.length ? 'Read it again' : 'Read this picture'}</Link>
              {compact ? ' for its tags, faces and hands.' : ' for what the tagger sees, where the faces and hands are, and which add-ons suit it.'}
            </>
          )}
        </p>
        {error ? <p className="mt-1 text-ink-error">{error}</p> : null}
      </div>
    )
  }

  const tags = report.tags?.general ?? []
  const rating = report.rating
  const wantsMore = (rating === 'questionable' || rating === 'explicit') && anatomy === 'off' && onAnatomy
  const suggestions = report.suggestions.filter((s) => s.confidence !== 'incidental')

  return (
    <div className={text}>
      {!compact ? <Kicker className="block">What is in it</Kicker> : null}
      {rating ? (
        <p className={`${compact ? '' : 'mt-1 '}text-grey-700`}>
          Reads as <span className="text-ink">{RATING_WORD[rating]}</span>.
          {wantsMore ? (
            <>
              {' '}
              The anatomy setting is Standard.{' '}
              <Link onClick={() => onAnatomy('natural')}>Switch to Sharper faces and hands</Link> for the
              next render.
            </>
          ) : null}
        </p>
      ) : null}
      {tags.length ? (
        <p className="mt-1 text-grey-700">
          {tags.slice(0, 24).map((t) => t.tag.replace(/_/g, ' ')).join(', ')}
          {tags.length > 24 ? ` and ${tags.length - 24} more` : ''}.
          {onUseWords ? (
            <>
              {' '}
              <Link onClick={() => onUseWords(report.promptFromImage)}>Use these words</Link>
            </>
          ) : null}
        </p>
      ) : null}
      {factsSentence(report.facts) ? <p className="mt-1 text-grey-700">{factsSentence(report.facts)}</p> : null}
      {suggestions.length ? (
        <ul className="mt-2 space-y-1 border-t border-grey-300 pt-2">
          {suggestions.map((s) => (
            <li key={s.entry.file} className="text-grey-700">
              <span className="text-ink">{s.entry.stem}</span>
              <span className="text-grey-500"> · {s.confidence}</span>. {s.why}
              {onAddOn ? (
                <>
                  {' '}
                  {added.has(s.entry.file) ? (
                    <span className="italic text-grey-500">Added for the next one.</span>
                  ) : (
                    <Link
                      onClick={() => {
                        onAddOn(s.entry.file)
                        setAdded((prev) => new Set([...prev, s.entry.file]))
                      }}
                    >
                      Add for the next one
                    </Link>
                  )}
                </>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
      <p className="mt-1 text-caption italic text-grey-500">
        Ranked against {report.considered.ranked} add-ons
        {report.considered.noTagData ? `; ${report.considered.noTagData} more carry no tag data and were not weighed` : ''}.
        {' '}
        <Link onClick={read}>Read it again</Link>
      </p>
      {error ? <p className="mt-1 text-ink-error">{error}</p> : null}
    </div>
  )
}
