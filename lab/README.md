# SwitchGen model lab

A lab, not a feature. It puts every installed picture model through the same
tests, lets you score the results blind on your phone, and turns the scores
into a map of what each model is good at, by building block (prompt
following, text, photorealism, reference keeping and so on), plus rankings
for real uses such as a thumbnail or a pet portrait. The point is to design
the app around tasks rather than model names.

It lives in `lab/`, outside the app. The app never imports it, the app's
build and `npm test` never see it, and it has its own type check and tests.

**Nothing is sent anywhere until you press Start on the lab page or run
`lab/lab start <run>`.** Planning, checking and reading the plan send
nothing. The lab never sends ComfyUI anything itself: its only request to
ComfyUI is a read of `/object_info` in `lab/lab check`, and all its pictures
go through the app's server-side runner as their own lane (the 'lab' desk).
Lab pictures are never filed in the archive, never appear in History and
never lock the Pictures desk.

## The tests

Four nights make up the first study. Every model runs the same prompts with
the same four seeds (1001, 2002, 3003, 4004), the same shapes and 28 steps,
with the common euler/simple sampler. Guidance (CFG) stays at each model's
own value from the app's registry, so the distilled models run at CFG 1, as
they do in the app. Each model also keeps its own quality prefix, house
negative, clip skip and shift. Every graph is built by the app's own
functions, so what the lab finds carries over to what the app sends.

All thirteen picture models are in the first pass: NoobAI, semiReal, Pony,
WAI, the three Anima mixes (oneObsession, miaomiaoHarem, miaomiaoRealskin),
Krea2, Chroma, Klein, Qwen-Image 2.1, Z-Image Base and Z-Image Turbo.
Qwen-Image-Edit 2511 joins the editing, reference and chain tests.

| Night | Suite | Run | Pictures | New |
| --- | --- | --- | --- | --- |
| 0, calibration | `calibration` | `cal-1` | 272 | 272 |
| 1, core | `first-pass-core` | `core-1` | 676 | 588 (88 made by the calibration) |
| 2, editing and references | `ext-edit` | `exta-1` | 664 | 648 (16 are core pictures the chains start from) |
| 3, range | `ext-range` | `extb-1` | 944 | 888 (56 are core pictures) |

`lab/lab plan` prints the real figures and an estimated time. The estimate
comes from your archive's own timings where it has them and from guesses
where it does not, and it always says which.

The calibration comes first. It checks two hunches before the main run: the
step sweep (do the fast distilled models suffer at 28 steps?) and the
sampler check (does the common sampler hurt a model whose home sampler is
different?). The core run will not start until its pairs are judged, unless
you skip it on purpose.

Some tests do not apply to some models, and the reason always comes from
the app's registry: Klein has no image-to-image graph, Klein and Krea2 have
no negative prompt input, and Z-Image Turbo and Qwen-Image 2.1 run with
guidance off, so a negative prompt does nothing there. Those are shown as
"not applicable" with the reason, never scored and never counted as a 1.

Every prompt that shows a person names an adult. The suite checker refuses
any prompt with a child-coded word, and refuses a prompt with a person that
does not name an adult; the one exception is the defaults probe, "a person",
and the picture reader's quarantine backs that up.

## Your photos

Night 2 uses your own photos:

- **cat**: your cat. Its starting description is "a tortoiseshell tabby cat
  with a white belly and white paws, wearing a dark collar"; change it on
  the lab page if you like.
- **scene**: an everyday room or table, with a rectangle drawn on it for the
  region test. Add it from your phone on the lab page (Photos), or drop the
  file into the lab's `refs` folder (`/mnt/storage/ai/lab/refs` unless you
  set `SWITCHGEN_LAB_DIR`), then draw the rectangle on the lab page.

A description names only what is in the photo: the cat, or the room or
table. Anything that brings a person in is refused when you save it: people
and family words, "my" or "your", a hand or a lap, someone holding or
cuddling the cat, and "he", "she", "his" or "her". Call the cat "it".

A photo put in the `refs` folder by hand is known by its file name. Name
the cat photo `cat.jpg` and the scene photo `scene.jpg` (or `.png`, `.webp`);
`room.jpg` and `table.jpg` also count as the scene photo. A file with the
camera's own name, such as `IMG_2034.jpg`, is the "img-2034" photo, which
no test asks for: rename the file in the `refs` folder to `scene.jpg`.
Rename it before you draw the rectangle: the rectangle belongs to the old
name. A photo kept anywhere else is copied in under the right name with
`lab/lab refs add <path to the photo> --name scene`.

Until a photo (or its rectangle) is there, the plan names the tests that
wait for it, and the night refuses to start. Photos are read the way you
see them, with the camera's orientation flag applied. The lab page shows
each photo without its camera metadata (place, time, camera).

## Where things live

In the repo, committed: the code, the suites, the scorecards and whatever
findings you choose to copy into `lab/findings/`.

Outside the repo, private, in `SWITCHGEN_LAB_DIR` (default
`/mnt/storage/ai/lab`): your photos (`refs/`), each picture's graph
(`cells/`), the list of finished pictures (`cells.jsonl`), the picture
reader's results, one folder per run under `runs/` with its plan, ledger,
sealed key, blind copies and your scores, and one folder per study under
`studies/` with its reveal, its report and the reader's results for the
pictures it judged, kept for the report after prune. The lab
refuses to start if this folder is inside the repo, or inside the outputs
folder, which ComfyUI serves to the whole tailnet.

Under the outputs folder, only what ComfyUI itself must read or write:
the pictures (`.lab/cells/<id>_00001_.png`, named by an id that says
nothing about the model) and the copies of your photos it loads
(`.lab/refs/`). The leading dot keeps them out of the app's History.

## Commands

Everything runs from the repo root through `lab/lab`.

| Command | What it does |
| --- | --- |
| `lab/lab check` | Checks every planned graph against ComfyUI's `/object_info` (a read) and that nothing in `src/` or `server/` imports the lab. |
| `lab/lab plan <suite…> --run <id>` | Plans a night: counts, what is reused, what does not apply and why, what waits for a photo, and an estimate. Sends nothing. |
| `lab/lab smoke` | Plans one picture per operation on the fastest model that can do it, at 8 steps. It is sent only after you press Start. |
| `lab/lab start <run> [--until 07:30] [--skip-calibration]` | Starts a run: sends its pictures through the app's runner, one lab group at a time. `--until` stops sending at that time. A run that cannot start (a photo missing, the calibration pairs not judged, its study already revealed) is refused there and then, with the reason. |
| `lab/lab run <run>` | The same, in the foreground. |
| `lab/lab pause <run>` | Stops the current group and marks the run paused. |
| `lab/lab status` | Where each run is. Names no model before the reveal. |
| `lab/lab refs add <file> [--name cat]` | Copies a reference photo into the lab's `refs` folder, under that name. |
| `lab/lab refs mask <id> x,y,w,h` | Draws the region rectangle, in the photo's upright pixels. |
| `lab/lab refs describe <id> <words…>` | Changes what the photo is described as. A description that brings in a person is refused. |
| `lab/lab read <run>` | Runs the picture reader over a finished run. |
| `lab/lab seal <run>` | Makes the blind copies and the sealed key. |
| `lab/lab check-blind <run>` | Scans everything the phone can see for anything that names a model. Judging waits until it passes. |
| `lab/lab serve` | Starts the lab page on 127.0.0.1:5274. |
| `lab/lab phone` | Prints the one command that puts the lab page on your tailnet. It never runs it. |
| `lab/lab report <run>` / `lab/lab findings <run>` | The report after the reveal; `findings` copies findings.json into `lab/findings/`, only when you ask. |
| `lab/lab prune <run>` | Deletes a revealed run's original pictures, when you ask (`--yes`). A picture a run not yet revealed or not yet sealed still needs is kept. A later run that needs the same pictures makes them again, and the picture reader reads them again. |
| `lab/lab test` | The lab's own tests: `npx vitest run --config lab/vitest.config.ts`. |
| `lab/lab typecheck` | The lab's own type check: `npx tsc -p lab/tsconfig.json`. |

## Scoring on the phone

The lab page runs on its own port, 5274, a separate origin from the app, so
the app's service worker never touches it. To reach it from your phone, run
this once yourself (the lab prints it and never runs it; never use funnel):

```
sudo tailscale serve --bg --https=8443 http://127.0.0.1:5274
```

Scoring is blind. Each set shows every applicable model's four pictures under
a fresh letter, in a shuffled order, with the same neutral task for all of
them. The blind copies carry no metadata, and timings, the picture reader's
ratings, file names, sizes and arrival order stay hidden until the reveal.

The reveal covers the whole study at once, all four nights. Revealing
before every night is made, sealed and scored needs "reveal early", and
after the reveal nothing more can be planned or started in that study (the
smoke run is the exception). A run that is being made has to be paused
before the reveal.

Marking pictures "looks under 18" together with "sexualised" on the
defaults card removes them. Undo on the phone takes that back: the blind
copies are deleted once 30 seconds have passed since the score arrived, if
it still stands then, and the originals at the reveal.

## How the lab stays out of the app

- Build: nothing in `src/` or `server/` imports `lab/`, and the app's build
  starts at `index.html`.
- Tests: the app's `vitest.config.ts` includes only `tests/**/*.test.ts`;
  `lab/vitest.config.ts` includes only `lab/tests/**/*.test.ts`, points
  ComfyUI and the app at port 9 where nothing listens, and keeps every
  folder temporary.
- Types: `tsc -b` covers the app, its scripts and its Vite config;
  `lab/tsconfig.json` is run only by `lab/lab typecheck` and is not
  referenced from the root.
- The launcher and `package.json` are untouched, and the app never rebuilds
  because of a lab edit.
