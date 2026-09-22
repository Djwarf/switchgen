/**
 * THE RESULT SURFACE.
 *
 * Requirement 3 of the simplification, in one folder: quality passes are
 * offered AFTER the picture, not before it.
 *
 * The compose screen asks for three things. Everything that used to sit beside
 * those three asking to be decided in advance, a face pass, a hand pass, a
 * hires pass, a region refine, lives here instead, printed under the finished
 * picture where the reader can see whether it is needed. Nothing was deleted to
 * get there. Every pass is the same derivation in lib/refine.ts that the old
 * checkboxes drove, reached at the moment it can be judged.
 *
 * WIRING IT UP, IN FULL:
 *
 *   import { ResultActions, type ResultActionId } from '../components/result'
 *
 *   <ResultActions
 *     picture={{ url: fileUrl(entry.file), width: entry.width, height: entry.height }}
 *     def={plan.def}                  // the recipe's derived def, LoRAs and all
 *     canSource={tabI2I}
 *     busy={running}
 *     blocked={pressFault}
 *     onAction={(id) => {
 *       if (id === 'refine') return void openRefine(entry)   // the refine bench
 *       if (id === 'face')   return queue(deriveAutoDetail(plan.def, 'face'))
 *       if (id === 'hand')   return queue(deriveAutoDetail(plan.def, 'hand'))
 *       if (id === 'hires')  return queue(deriveHiresFix(plan.def))
 *       if (id === 'again')  return generate({ seed: randomSeed() })
 *       if (id === 'source') return takeRecord(entry)        // into the source slot
 *     }}
 *   />
 *
 * `def` is the graph the picture was actually made with, not the family it came
 * from. Hand it `plan.def` and the rows match what can really be derived from
 * there; hand it the bare family and a picture made through image to image will
 * be offered passes against the wrong graph.
 *
 * The desk owns the queue. This folder decides what to offer and says what it
 * costs, and then gets out of the way.
 */
export { ResultActions, default, type ResultActionsProps } from './ResultActions'
export {
  SHARPNESS_CAVEAT,
  offerFigures,
  offersFor,
  type OfferOptions,
  type ResultActionId,
  type ResultOffer,
} from './offers'
