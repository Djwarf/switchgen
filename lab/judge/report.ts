/**
 * The report: one self-contained page (its own styles and a small sorting
 * script, nothing fetched), and the capability map as CSV.
 *
 * Every number says what it is: your score, the picture reader's rating (a
 * guide), measured by ComfyUI, or estimated. N/A is shown hatched with the
 * registry's reason and is never a score.
 */
import type { Block } from '../core/types.ts'
import { MAP_GROUPS, type FindingCell, type Findings } from './aggregate.ts'

const BLOCK_NAMES: Record<Block, string> = {
  following: 'Following',
  style: 'Style',
  photo: 'Photo',
  variation: 'Variation',
  content: 'Content',
  text: 'Text',
  reference: 'Reference',
  layout: 'Layout',
  anatomy: 'Anatomy',
  detail: 'Detail',
  cost: 'Cost',
  shapes: 'Shapes',
  character: 'Same character',
  sensitivity: 'Sensitivity',
  negative: 'Negative',
  defaults: 'Defaults',
  region: 'Region edit',
  edit: 'Instruction edit',
  overall: 'Overall',
}
export const blockName = (b: Block): string => BLOCK_NAMES[b] ?? b

const esc = (v: unknown): string =>
  String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string)

const fmt = (x: number | null | undefined, d = 1): string => (x === null || x === undefined || !Number.isFinite(x) ? '—' : x.toFixed(d))
const pct = (x: number | null | undefined): string => (x === null || x === undefined ? '—' : `${Math.round(x * 100)}%`)
const when = (t: number | null): string => (t ? new Date(t).toISOString().replace('T', ' ').slice(0, 16) + ' UTC' : 'not yet')

function nameOf(f: Findings, id: string): string {
  const c = f.contestants.find(x => x.id === id)
  if (!c) return id
  if (c.kind === 'chain') return c.name ?? id
  return c.label ? `${c.label}` : id
}

function sub(f: Findings, id: string): string {
  const c = f.contestants.find(x => x.id === id)
  if (!c) return ''
  if (c.kind === 'chain') return `chain: ${(c.links ?? []).join(' → ')}`
  return [c.id, c.file].filter(Boolean).join(' · ')
}

function sourceWords(c: FindingCell): string {
  switch (c.source) {
    case 'judge': return 'your score'
    case 'reader': return 'the picture reader rated (a guide)'
    case 'comfy': return 'measured by ComfyUI'
    case 'estimated': return 'estimated'
    default: return ''
  }
}

function cellHtml(c: FindingCell | undefined): string {
  if (!c) return '<td class="nt">—</td>'
  if (c.score === null && c.na) return `<td class="na" data-v="-1" title="${esc(c.na)}"><span class="naw">N/A</span><span class="why">${esc(c.na)}</span></td>`
  if (c.score === null) return '<td class="nt" data-v="-2" title="not tested">—</td>'
  const bits: string[] = []
  if (c.source === 'judge') {
    bits.push(`n ${c.n}`)
    if (c.failShare) bits.push(`fails ${fmt(c.failShare * 4, 1)}/4`)
    const p = c.pairs
    if (p.won + p.lost + p.tied) bits.push(`pairs ${p.won}–${p.lost}–${p.tied}`)
    if (c.recognisedShare) bits.push(`known ${pct(c.recognisedShare)}`)
  } else if (c.note) bits.push(c.note)
  const band = Math.max(1, Math.min(5, Math.round(c.score)))
  const tier = c.tier ? `<span class="tier" title="tier ${c.tier}: within one step of the one above and not clearly beaten in pairs counts as about the same">T${c.tier}</span>` : ''
  return `<td class="s${band}" data-v="${c.score}" title="${esc(sourceWords(c))}"><span class="sc">${fmt(c.score, c.source === 'judge' ? 1 : 0)}</span>${tier}<span class="mini">${esc(bits.join(' · '))}</span>${c.source !== 'judge' ? `<span class="src">${esc(sourceWords(c))}</span>` : ''}</td>`
}

function reliabilityText(r: Findings['reliability'][string] | undefined): string {
  if (!r) return '—'
  const tries: string[] = []
  if (r.failed) tries.push(`${r.failed} failed`)
  if (r.lost) tries.push(`${r.lost} lost`)
  if (r.noFile) tries.push(`${r.noFile} with no file`)
  if (r.refused) tries.push(`${r.refused} refused by ComfyUI`)
  const out = [`${r.made} of ${r.pictures} pictures made`]
  if (tries.length) out.push(`tries that ended ${tries.join(', ')}`)
  if (r.removed) out.push(`${r.removed} removed by the quarantine rule or your “looks under 18” with “sexualised” taps`)
  if (r.notMade) out.push(`${r.notMade} grid${r.notMade === 1 ? '' : 's'} not made`)
  return out.join(' · ')
}

function mapTable(f: Findings): string {
  const groups = MAP_GROUPS.map(g => ({ name: g.name, blocks: g.blocks.filter(b => f.blocks.includes(b)) })).filter(g => g.blocks.length)
  const cols = groups.flatMap(g => g.blocks)
  const byKey = new Map(f.cells.map(c => [`${c.contestant}\u0000${c.block}`, c]))
  const head1 = `<tr><th class="stick" rowspan="2">Contestant</th>${groups.map(g => `<th colspan="${g.blocks.length}" class="grp">${esc(g.name)}</th>`).join('')}<th rowspan="2">Reliability</th></tr>`
  const head2 = `<tr>${cols.map((b, i) => `<th class="sort" data-col="${i + 1}" tabindex="0" role="button" title="Sort by ${esc(blockName(b))}">${esc(blockName(b))}</th>`).join('')}</tr>`
  const body = f.contestants.map(c => {
    const tds = cols.map(b => cellHtml(byKey.get(`${c.id}\u0000${b}`))).join('')
    return `<tr><th class="stick" scope="row"><span class="nm">${esc(nameOf(f, c.id))}</span><span class="fl">${esc(sub(f, c.id))}</span></th>${tds}<td class="rel">${esc(reliabilityText(f.reliability[c.id]))}</td></tr>`
  }).join('\n')
  return `<div class="scroll"><table id="map" class="map"><thead>${head1}${head2}</thead><tbody>${body}</tbody></table></div>`
}

function cards(f: Findings): string {
  return `<div class="cards">${f.contestants.map(c => {
    const rows = f.blocks.map(b => {
      const x = f.cells.find(y => y.contestant === c.id && y.block === b)
      if (!x || (x.score === null && !x.na)) return ''
      const v = x.score === null ? `<span class="naw">N/A</span> <span class="why">${esc(x.na)}</span>` : `<b>${fmt(x.score, x.source === 'judge' ? 1 : 0)}</b>${x.tier ? ` <span class="tier">T${x.tier}</span>` : ''} <span class="mini">${esc(x.source === 'judge' ? `n ${x.n}` : sourceWords(x))}</span>`
      return `<li><span>${esc(blockName(b))}</span><span>${v}</span></li>`
    }).join('')
    return `<section class="card"><h3>${esc(nameOf(f, c.id))}</h3><p class="fl">${esc(sub(f, c.id))}</p><ul class="kv">${rows}</ul><p class="mini">${esc(reliabilityText(f.reliability[c.id]))}</p></section>`
  }).join('')}</div>`
}

function usecases(f: Findings): string {
  if (!f.usecases.length) return ''
  return f.usecases.map(u => {
    const ranked = u.ranking.filter(r => r.score !== null || r.vetoed)
    const list = ranked.slice(0, 8).map((r, i) => {
      const drivers = r.drivers.map(d => `${blockName(d.block)} ${d.effect >= 0 ? '+' : ''}${fmt(d.effect, 1)}`).join(', ')
      const status = r.vetoed
        ? `<span class="veto">left out: ${esc(r.vetoed)}</span>`
        : r.notTested.length
          ? `<span class="muted">not ranked: ${esc(r.notTested.map(blockName).join(', '))} not tested</span>`
          : `<b>${fmt(r.score, 2)}</b> <span class="mini">(${pct(r.coverage)} of the weight scored)</span>`
      const isRanked = !r.vetoed && !r.notTested.length
      return `<li><span class="rk">${isRanked ? i + 1 : '·'}</span> ${esc(nameOf(f, r.contestant))} ${status}${drivers && isRanked ? `<div class="mini">what drove the rank: ${esc(drivers)}</div>` : ''}</li>`
    }).join('')
    return `<section class="card"><h3>${esc(u.name)}</h3>${u.note ? `<p class="muted">${esc(u.note)}</p>` : ''}<p class="mini">Must have: ${esc(u.mustHave.map(blockName).join(', '))}. A must-have that is N/A or scored 2 or less leaves a contestant out.</p><ol class="rank">${list || '<li class="muted">Nothing scored yet.</li>'}</ol></section>`
  }).join('')
}

function suggestions(f: Findings): string {
  const s = f.suggestions
  const best = s.bestPerUseCase.map(b => {
    const u = f.usecases.find(x => x.id === b.usecase)
    return `<li><b>${esc(u?.name ?? b.usecase)}</b>: ${b.contestant ? `${esc(nameOf(f, b.contestant))} (${fmt(b.score, 2)})` : 'nothing qualifies yet'}</li>`
  }).join('')
  const gaps = s.gaps.length
    ? `<p>Where nothing reaches 4 on your scores: ${esc(s.gaps.map(blockName).join(', '))}. These are gaps no single model fills today.</p>`
    : '<p>Every scored block has at least one contestant at 4 or above.</p>'
  const byBlock = Object.entries(s.bestByBlock).map(([b, list]) => `<li>${esc(blockName(b as Block))}: ${esc(list.map(id => nameOf(f, id)).join(', '))}</li>`).join('')
  return `<ul>${best}</ul>${gaps}<p>${s.splitAcrossModels
    ? 'The best contestant differs from block to block. That argues for building the app around tasks and chains rather than one model name.'
    : 'One contestant leads every scored block.'}</p><details><summary>Best by block</summary><ul>${byBlock}</ul></details>`
}

function chainsTable(f: Findings): string {
  if (!f.chains.length) return '<p class="muted">No chain was made in this study.</p>'
  const rows = f.chains.map(c => {
    const change = c.change.map(x => `${blockName(x.block)}: ${fmt(x.chain)} against ${fmt(x.compose)} (${x.delta === null ? '—' : `${x.delta >= 0 ? '+' : ''}${fmt(x.delta, 2)}`})`).join('<br>')
    const extra = c.extraSeconds === null ? '—' : `${fmt(c.extraSeconds, 1)} s <span class="mini">${c.extraSource === 'comfy' ? 'measured by ComfyUI' : c.extraSource === 'estimated' ? 'estimated (includes model loads)' : ''}</span>`
    const value = c.valuePerSecond === null ? '—' : `${fmt(c.valuePerSecond * 60, 2)} steps per extra minute`
    return `<tr><th scope="row">${esc(c.name)}</th><td>${c.compose ? esc(nameOf(f, c.compose.model)) : '—'}</td><td>${c.pair.won}–${c.pair.lost}–${c.pair.tied}</td><td>${change}</td><td>${extra}</td><td>${esc(value)}</td></tr>`
  }).join('')
  return `<div class="scroll"><table class="plain"><thead><tr><th>Chain</th><th>Starts from</th><th>Pairs against it (won–lost–tied)</th><th>Your score, chain against start</th><th>Extra time</th><th>Value</th></tr></thead><tbody>${rows}</tbody></table></div>`
}

function contentTable(f: Findings): string {
  const rows = Object.entries(f.content.byModel).map(([m, r]) => {
    const counts = `${r.raw.general} general · ${r.raw.sensitive} sensitive · ${r.raw.questionable} questionable · ${r.raw.explicit} explicit`
    return `<tr><th scope="row">${esc(nameOf(f, m))}</th><td>${r.raw.band ?? '—'}</td><td>${r.checked.band ?? '—'}</td><td>${r.raw.n}</td><td class="mini">${esc(counts)}${r.raw.unread ? ` · ${r.raw.unread} not read` : ''}</td><td>${r.removed}</td><td class="mini">${r.checks.yes} yes · ${r.checks.no} no · ${r.checks.unsure} unsure</td></tr>`
  }).join('')
  return `<p class="muted">${esc(f.content.note)}</p><div class="scroll"><table class="plain"><thead><tr><th>Model</th><th>Band as the picture reader rated</th><th>Band after your checks</th><th>Pictures</th><th>What the picture reader rated</th><th>Removed by the quarantine rule or your “looks under 18” with “sexualised” taps</th><th>Your checks: goes beyond the prompt?</th></tr></thead><tbody>${rows || '<tr><td colspan="7" class="muted">No readings.</td></tr>'}</tbody></table></div>`
}

function costTable(f: Findings): string {
  const src = (s: string) => (s === 'comfy' ? 'measured by ComfyUI' : s === 'estimated' ? 'estimated' : '')
  const rows = Object.entries(f.cost.byModel).map(([m, r]) =>
    `<tr><th scope="row">${esc(nameOf(f, m))}</th><td>${fmt(r.warmMedianS)} s <span class="mini">n ${r.warmN}</span></td><td>${fmt(r.coldMedianS)} s <span class="mini">n ${r.coldN}</span></td><td>${r.cachedLeftOut}</td><td>${fmt(r.secondsPerStep, 2)} s <span class="mini">${src(r.perStepSource)}</span></td><td>${r.homeSteps ?? '—'}</td><td>${fmt(r.homeS)} s <span class="mini">${src(r.homeSource)}</span></td><td>${r.band ?? '—'}</td></tr>`).join('')
  return `<p class="muted">${esc(f.cost.note)}</p><div class="scroll"><table class="plain"><thead><tr><th>Model</th><th>Warm median at 28 steps</th><th>After a model load</th><th>Cached, left out</th><th>Per step</th><th>Its own steps</th><th>At its own steps</th><th>Band</th></tr></thead><tbody>${rows || '<tr><td colspan="8" class="muted">No timings.</td></tr>'}</tbody></table></div>`
}

function promptStyleTable(f: Findings): string {
  if (!f.promptStyle.length) return ''
  const rows = f.promptStyle.map(p => `<tr><th scope="row">${esc(nameOf(f, p.contestant))}</th><td>${p.sentence}</td><td>${p.tags}</td><td>${p.tied}</td></tr>`).join('')
  return `<h2>Tags against sentences</h2><p class="muted">For the models trained on booru tags: which followed the brief better, the core's sentence or the same brief as tags. Your pairs.</p><div class="scroll"><table class="plain"><thead><tr><th>Model</th><th>Sentence better</th><th>Tags better</th><th>Can't tell</th></tr></thead><tbody>${rows}</tbody></table></div>`
}

function judgeBox(f: Findings): string {
  const j = f.judge
  const line = j.n
    ? `Second looks: ${j.n}. The same step ${pct(j.agreeExact)} of the time, within one step ${pct(j.agreeWithin1)}. Drift ${j.drift === null ? '—' : `${j.drift >= 0 ? '+' : ''}${fmt(j.drift, 2)}`} steps (positive: kinder the second time).`
    : 'No second looks were answered.'
  return `<div class="note${j.warning ? ' warn' : ''}"><b>How steady your scores were.</b> ${esc(line)} Sessions: ${j.sessions}.${j.warning ? `<br>${esc(j.warning)}` : ''}</div>`
}

const CSS = `
:root{--bg:#1e1e1e;--fg:#e8e6e3;--muted:#a09c96;--line:#3a3a3a;--card:#262626;--accent:#e0b04a;--bad:#e06c5a;--s1:#5a2a26;--s2:#5a4026;--s3:#4a4a2a;--s4:#2f4a2c;--s5:#23533a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1400px;margin:0 auto;padding:16px}
h1{font-size:1.6rem;margin:.2em 0}
h2{font-size:1.25rem;margin:1.6em 0 .4em;border-bottom:1px solid var(--line);padding-bottom:.2em}
h3{font-size:1.05rem;margin:.2em 0}
.muted,.fl,.why{color:var(--muted)}
.mini{display:block;font-size:.75rem;color:var(--muted)}
.fl{display:block;font-size:.75rem;word-break:break-all}
.note{background:var(--card);border-left:3px solid var(--accent);padding:10px 12px;margin:12px 0;border-radius:4px}
.note.warn{border-left-color:var(--bad)}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line);border-radius:6px}
table{border-collapse:separate;border-spacing:0;min-width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{background:#2b2b2b;position:sticky;top:0;z-index:1;font-weight:600}
.map td{min-width:92px}
.map .stick{position:sticky;left:0;background:#242424;z-index:2;min-width:170px;max-width:220px}
.map thead .stick{z-index:3}
th.grp{text-align:center;color:var(--accent)}
th.sort{cursor:pointer;white-space:nowrap}
th.sort[aria-sort="descending"]::after{content:" ▼"}
th.sort[aria-sort="ascending"]::after{content:" ▲"}
.sc{font-size:1.15rem;font-weight:700}
.tier{font-size:.7rem;border:1px solid var(--muted);border-radius:8px;padding:0 5px;margin-left:5px;color:var(--muted)}
.src{display:block;font-size:.7rem;color:var(--accent)}
td.s1{background:var(--s1)}td.s2{background:var(--s2)}td.s3{background:var(--s3)}td.s4{background:var(--s4)}td.s5{background:var(--s5)}
td.na{background:repeating-linear-gradient(45deg,#2a2a2a 0 6px,#333 6px 12px);color:var(--muted)}
.naw{font-weight:700;color:var(--fg)}
td.na .why{display:block;font-size:.7rem}
td.nt{color:var(--muted)}
td.rel{font-size:.8rem;color:var(--muted);min-width:160px}
.cards{display:none}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:10px 0}
.kv{list-style:none;padding:0;margin:6px 0}
.kv li{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dashed var(--line);padding:3px 0}
.rank{padding-left:0;list-style:none}
.rank li{margin:6px 0}
.rk{display:inline-block;min-width:1.4em;color:var(--accent);font-weight:700}
.veto{color:var(--bad)}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
table.plain td,table.plain th{font-size:.9rem}
details summary{cursor:pointer;color:var(--accent)}
@media (max-width:700px){.wide{display:none}.cards{display:block}main{padding:12px}}
`

const SCRIPT = `
(function(){var t=document.getElementById('map');if(!t)return;var hs=t.querySelectorAll('th.sort');
function sort(h){var col=+h.getAttribute('data-col');var dir=h.getAttribute('aria-sort')==='descending'?1:-1;
hs.forEach(function(x){x.removeAttribute('aria-sort')});h.setAttribute('aria-sort',dir<0?'descending':'ascending');
var body=t.tBodies[0];var rows=[].slice.call(body.rows);
rows.sort(function(a,b){var va=a.cells[col]?parseFloat(a.cells[col].getAttribute('data-v')):NaN;var vb=b.cells[col]?parseFloat(b.cells[col].getAttribute('data-v')):NaN;
if(isNaN(va))va=-9;if(isNaN(vb))vb=-9;return dir<0?vb-va:va-vb});rows.forEach(function(r){body.appendChild(r)})}
hs.forEach(function(h){h.addEventListener('click',function(){sort(h)});h.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();sort(h)}})})})();
`

/** The report page for a study's findings. */
export function renderReport(f: Findings): string {
  const runs = f.runs.join(', ')
  const notMade = f.notMade.length
    ? `<details><summary>${f.notMade.length} grid${f.notMade.length === 1 ? '' : 's'} not made (fewer than 3 of 4 pictures)</summary><ul>${f.notMade.map(n => `<li>${esc(nameOf(f, n.contestant))}: ${esc(n.slot)} (${n.have} made)</li>`).join('')}</ul></details>`
    : ''
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>Lab findings</title><style>${CSS}</style></head>
<body><main>
<h1>What the study found</h1>
<p class="muted">Study ${esc(f.study)} · runs ${esc(runs)} · revealed ${esc(when(f.revealedAt))} · key ${esc(f.sealedSha.split(',').map(s => s.slice(0, 12)).join(', '))}</p>
<div class="note"><b>How to read this.</b> A score is <b>your score</b>: the mean of the steps you gave each grid (1 to 5, 5 best), with n grids. "fails k/4" is how many of a grid's four pictures you marked Failed, on average. "pairs" is won–lost–tied in your tie-break and chain pairs. "known" is how often you thought you knew the model. Content is what <b>the picture reader rated</b>, a guide and nothing more. Cost is <b>measured by ComfyUI</b>; anything worked out rather than measured says <b>estimated</b>. N/A (hatched) means the app has no way to run that test for that model, with the reason; it is never counted as a score. One test per block: a first look, not a verdict.</div>
${judgeBox(f)}
<h2>Capability map</h2>
<p class="muted">Tap a block's name to sort by it. T1 is the top tier of a block: contestants within one step of the one above, and not clearly beaten in pairs, are about the same.</p>
<div class="wide">${mapTable(f)}</div>
${cards(f)}
${notMade}
<h2>Use cases</h2>
<div class="grid2">${usecases(f)}</div>
<h2>What this suggests for the app</h2>
${suggestions(f)}
<h2>Chains</h2>
${chainsTable(f)}
${promptStyleTable(f)}
<h2>Content default</h2>
${contentTable(f)}
<h2>Cost</h2>
${costTable(f)}
<h2>Method notes</h2>
<ul>
<li>Every graph was built by the app's own functions, so a finding carries over to the app. The one lab-only graph (detail on an existing picture) is marked as not in the app.</li>
<li>The same seed does not give the same starting noise across architectures, so seed-for-seed comparisons across models mean less than they look.</li>
<li>Denoise means different things for the older (eps) and newer (flow) models; image-to-image strengths are not strictly comparable.</li>
<li>28 steps for everyone charges the distilled models (home 8 steps) more time than they need; the cost table shows their own step count apart.</li>
<li>Sentence prompts favour models with language-model text encoders; the tags-against-sentences pairs show how much.</li>
<li>Everyone ran the common sampler (euler, simple scheduler) except where a model has only its own; the sampler check shows where that cost a model.</li>
<li>Guidance stayed at each model's own value, so the distilled models ran with guidance off (CFG 1), where a negative prompt does nothing.</li>
${f.blocks.includes('shapes') ? '<li>Where the lighthouse set ran, each grid got one step for wide and tall together, set by the weaker shape, so this report does not say which of the two shapes was weaker.</li>\n' : ''}<li>Blindness had limits: a house look (Pony, the anime mixes) can be recognised, and ComfyUI's own interface and history name the weights if browsed during judging. "known" in the map shows how often you thought you recognised one.</li>
<li>${f.skipped} answer${f.skipped === 1 ? ' was a skip' : 's were skips'}; ${f.afterReveal} event${f.afterReveal === 1 ? '' : 's'} came after the reveal and ${f.afterReveal === 1 ? 'is' : 'are'} left out.</li>
</ul>
</main><script>${SCRIPT}</script></body></html>
`
}

const csvCell = (v: unknown): string => {
  if (v === null || v === undefined) return ''
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : ''
  let s = String(v)
  // A spreadsheet runs a cell that starts like a formula; start it with a quote instead.
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

/** The capability map as CSV: one row per contestant and block. */
export function toCsv(f: Findings): string {
  const head = ['contestant', 'kind', 'block', 'score', 'n', 'min', 'max', 'fail_share', 'best_share', 'pairs_won', 'pairs_lost', 'pairs_tied', 'recognised_share', 'tier', 'source', 'na', 'note']
  const kind = new Map(f.contestants.map(c => [c.id, c.kind]))
  const lines = [head.join(',')]
  for (const c of f.cells) {
    lines.push([
      c.contestant, kind.get(c.contestant) ?? '', c.block, c.score, c.n, c.min, c.max, c.failShare, c.bestShare,
      c.pairs.won, c.pairs.lost, c.pairs.tied, c.recognisedShare, c.tier ?? null, c.source, c.na ?? null, c.note ?? null,
    ].map(csvCell).join(','))
  }
  return lines.join('\n') + '\n'
}
