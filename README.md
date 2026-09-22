# SwitchGen

An editorial front end for local AI image and video generation, built on the
SwitchSides classical-newspaper brand (burgundy `#2e0000` on newsprint,
Crimson Text, two rule weights, no rounded corners). It talks to ComfyUI over
its HTTP and WebSocket API, so a ComfyUI update cannot break it, and it is
built to be used from a phone, a TV and a laptop on the same tailnet.

Two rules govern everything in it. The desk asks a few questions and decides
the rest from measured runs, printing what it decided and why under the
button. And nothing claims what was not measured: every strength, ratio and
cost on screen is read off a run, a graph or a probe, and the copy says which.

## Getting started

You need a machine with an NVIDIA card, Node 22 or later, and ComfyUI. The app
was built and measured on a 16 GB RTX 5060 Ti with 32 GB of RAM; smaller cards
run the smaller families, and the desk says which fit before you press.

### 1. Install ComfyUI

ComfyUI 0.37 or later, in its own Python venv, with three node packs the
graphs depend on:

| Pack | Provides |
|---|---|
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | `UnetLoaderGGUF`, `CLIPLoaderGGUF`: every quantised family |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | `FaceDetailer`: the face and hand passes |
| [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack) | `UltralyticsDetectorProvider`: the detectors those passes use |

Everything else the graphs use ships with ComfyUI. `npm run validate` names
any node that is missing.

Point ComfyUI at a model library with the folder layout the app expects:
copy `contrib/extra_model_paths.yaml` into ComfyUI's root and set `base_path`.
The folders are `Stable-Diffusion`, `diffusion_models`, `text_encoders`,
`VAE`, `Lora`, `ultralytics/bbox`, `ultralytics/segm` and, for the picture
reader, `wd14`. Start ComfyUI with `--output-directory` pointing at the folder
the app should file outputs from. `contrib/comfyui.service` is a systemd user
unit that does this, which lets the launcher start ComfyUI on demand:

```bash
cp contrib/comfyui.service ~/.config/systemd/user/   # then edit the two paths
systemctl --user daemon-reload && systemctl --user enable --now comfyui
```

Optional, each probed and reported rather than assumed: `aria2c` for
downloads from the catalogue, `ffmpeg` and `ffprobe` for joining a reel, and
`onnxruntime` plus `ultralytics` in the ComfyUI venv for the picture reader.

### 2. Install the app

```bash
git clone git@github.com:Djwarf/switchgen.git
cd switchgen
cp .env.example .env            # set the models and outputs folders, and COMFY_URL
npm install
npm run validate                # every graph against your ComfyUI's live schema
bin/switchgen install           # links the launcher into ~/.local/bin
switchgen                       # builds, starts ComfyUI if it can, serves on :5273
```

The desk offers only families whose files are installed and fit in memory.
Fetch the rest from the catalogue behind More, which shows each family's
missing files, their size and the server's fit verdict. Add-ons are indexed
from your own LoRA folder with `npm run index-loras`, which reads each file's
header and rewrites `src/lib/loraIndex.ts`; the checked-in index describes
the folder this was built against.

### 3. Use it from a phone

The app binds every interface. `localhost`, this machine's hostname and
addresses, and any `*.ts.net` name are allowed; add anything else to
`SWITCHGEN_ALLOWED_HOSTS`. Open it once on the phone and add it to the home
screen: it installs as an app, and the archive is the same on every device.

## The four rooms

| Room | Hash | What it does |
|---|---|---|
| Pictures | `#/pictures` | Three questions (prompt, look, anatomy), a recipe decided from measurements, add-ons offered rather than applied, region editing on a mask, and "What next" on every finished picture with the real cost of each pass. Also the instruction-edit desk and the picture reader. |
| Video | `#/video` | One clip at a time from words or a start frame, an add-on rack per family, and "What next" on a finished clip. |
| Reel | `#/reel` | A list of shots rendered in order, each opening on the last frame of the one before, joined by the server into one file. |
| Archive | `#/archive` | Every picture and clip, searchable, with the settings that made it. One archive for every device. |

## Architecture

```
Browser (React 19 + Tailwind 4, hash router, no state library)
   |  /comfy      -> HTTP       (proxied by Vite, single origin)
   |  /comfy-ws   -> WebSocket  (live progress, reconnecting)
   |  /api/*      -> the SwitchGen middleware below, in the same Vite process
   v
ComfyUI :8188                      systemctl --user status comfyui
/mnt/storage/ai/models             the shared model library
/mnt/storage/ai/outputs            generated files, and .switchgen/archive.json
```

Nothing imports ComfyUI's Python. The only coupling is node names and input
schemas, which `npm run validate` checks against the live server.

### The local server

Five Vite middlewares in `server/` mount under `/api` in both `vite dev` and
`vite preview`. Every mutating route runs the same-origin guard in
`server/guard.mjs`: the browser's `Sec-Fetch-Site` verdict when present, else
`Origin` must match `Host`, and the body must be the type the route reads. A
request with neither header (curl, a script) is allowed. `GET /api/capabilities`
probes the binaries below and reports what actually runs.

| Route | Server | Purpose |
|---|---|---|
| `GET /api/capabilities` | api | What this server can do, probed not assumed |
| `GET /api/hardware`, `GET /api/hardware/stream` | api | CPU, RAM, GPU, disk; one-shot and SSE |
| `GET /api/models` | api | Every weight file under the models root |
| `POST /api/delete` | api | Unlink one output or model, confined to its root |
| `GET /api/archive`, `GET /api/archive/stream` | archive | The shared archive as a revision log, and its event stream |
| `POST /api/archive/upsert`, `/remove`, `/restore` | archive | Write records; the server assigns revisions and edition numbers |
| `GET /api/outputs` | archive | Every media file under the outputs root |
| `GET /api/catalog`, `GET /api/catalog/plan` | downloads | The 104-family catalogue annotated with what is on disk; a fit verdict and download plan per family |
| `POST /api/download`, `/download/cancel`, `GET /download/status` | downloads | Fetch through aria2c, resumable, as an event stream |
| `GET /api/reel/probe`, `POST /api/reel/stitch` | reel | Probe clips with ffprobe; join them with ffmpeg |
| `GET /api/vision/capabilities`, `POST /api/vision/tag`, `/detect`, `/inspect` | vision | The WD14 tagger and YOLO detectors, through the ComfyUI venv's Python, on the CPU |

### Environment

Every path has a default and an override. The launcher reads them from a
`.env` file beside `package.json`; `.env.example` lists them all.

| Variable | Default | Used by |
|---|---|---|
| `COMFY_URL` | `http://127.0.0.1:8188` | the Vite proxy, `validate`, `chain-e2e` |
| `SWITCHGEN_PORT` | `5273` | where the app listens |
| `SWITCHGEN_ALLOWED_HOSTS` | empty | extra hostnames the app may be reached by, comma separated |
| `SWITCHGEN_MODELS` | `/mnt/storage/ai/models` | api, downloads, vision |
| `SWITCHGEN_OUTPUTS` | `/mnt/storage/ai/outputs` | api, archive, reel, vision |
| `SWITCHGEN_ARCHIVE` | `<outputs>/.switchgen/archive.json` | archive |
| `SWITCHGEN_CATALOG` | `server/catalog.json` | downloads |
| `SWITCHGEN_ARIA2C`, `SWITCHGEN_FFMPEG`, `SWITCHGEN_FFPROBE` | `/usr/bin/...` | downloads, reel, capabilities |
| `HF_TOKEN_FILE` | `~/.cache/huggingface/token` | downloads, for gated files, only ever sent to HuggingFace |
| `SWITCHGEN_COMFY`, `SWITCHGEN_COMFY_INPUT` | `/mnt/storage/repos/ComfyUI`, `<comfy>/input` | vision |
| `SWITCHGEN_PYTHON` | `<comfy>/venv/bin/python` | vision |
| `SWITCHGEN_WD14` | `<models>/wd14` | vision |
| `SWITCHGEN_LORA_DIR` | `<models>/Lora` | `index-loras` |
| `SWITCHGEN_TRUSTED_ORIGINS` | empty | the guard, for a TLS terminator that rewrites Host |

## Usage

```bash
switchgen            # start ComfyUI if needed, build if anything changed, serve on :5273, open a browser
switchgen --no-open  # the same without opening a browser
switchgen dev        # hot-reloading dev server
switchgen validate   # check every graph and derivation against ComfyUI's live schema
switchgen stop
```

The launcher lives at `bin/switchgen`, reads `.env` beside `package.json`,
and rebuilds whenever a source file is newer than the last build. Changes to
`server/*.mjs` need a restart (`switchgen stop && switchgen`), because the
middlewares are loaded when the server starts.

## Model families

`src/lib/registry.ts` holds fifteen families, each a validated ComfyUI API
graph with bindings for the inputs the desk sets. What the picker offers is
decided at run time by `src/lib/availability.ts`: every file the graph names
must be listed by ComfyUI (consulting the GGUF and dual encoder loaders as
well as the plain one), and the weights must fit in this machine's memory.
A family that fails either is listed as blocked with the reason, never
offered and then failed.

| Id | Family | Mode | Model loader | Text encoder | VAE | Verified |
|---|---|---|---|---|---|---|
| `sdxl-illustrious` | SDXL booru bases (Illustrious, Pony, NoobAI) | image | CheckpointLoaderSimple | bundled | bundled | no |
| `z-image` | Z-Image (Turbo + Base) | image | UNETLoader | CLIPLoader: `qwen_3_4b` | `Flux/ae` | yes |
| `flux2-klein` | Flux.2 Klein 4B (distilled, fp8) | image | UNETLoader | CLIPLoader: `qwen_3_4b` | `Flux/flux2-vae` | yes |
| `anima` | Anima (Cosmos-based 2B/2.9B anime DiT) | image | UNETLoader | CLIPLoader: `oneObsession_anima29BV1_txt` | `qwen_image_vae` | yes |
| `krea2` | Krea 2 | image | UNETLoader | CLIPLoader: `qwen3vl_4b` | `qwen_image_vae` | yes |
| `qwen-image-21` | Qwen Image 2.1 | image | UNETLoader | CLIPLoader: `qwen3vl_8b_int8_convrot` | `qwen_image_2.1_vae_bf16` | yes |
| `qwen-image-edit` | Qwen Image Edit Plus 2511 (GGUF Q4_K_M) | edit | UnetLoaderGGUF | CLIPLoader: `qwen_2.5_vl_7b_fp8_scaled` | `qwen_image_vae` | yes |
| `chroma` | Chroma1-HD (uncensored photoreal) | image | UNETLoader | CLIPLoader: `t5xxl_fp8_e4m3fn_scaled` | `Flux/ae` | yes |
| `wan22-5b` | Wan 2.2 TI2V 5B (text+image to video) | video | UNETLoader | CLIPLoader: `umt5_xxl_fp8_e4m3fn_scaled` | `wan2.2_vae` | yes |
| `wan22-14b-t2v` | Wan 2.2 Text to Video 14B (High/Low Noise pair) | video | UnetLoaderGGUF ×2 | CLIPLoaderGGUF: `umt5-xxl-encoder-Q4_K_M` | `wan_2.1_vae` | yes |
| `wan22-14b-i2v` | Wan 2.2 I2V A14B (High/Low GGUF, Lightning 4-step) | video | UnetLoaderGGUF ×2 | CLIPLoaderGGUF: `umt5-xxl-encoder-Q4_K_M` | `wan_2.1_vae` | yes |
| `wan21-vace-14b-gguf` | Wan 2.1 VACE 14B (video to video) | video | UnetLoaderGGUF | CLIPLoader: `umt5_xxl_fp8_e4m3fn_scaled` | `wan_2.1_vae` | yes |
| `wan21-vace-1_3b-gguf` | Wan 2.1 VACE 1.3B (fast video to video) | video | UnetLoaderGGUF | CLIPLoader: `umt5_xxl_fp8_e4m3fn_scaled` | `wan_2.1_vae` | yes |
| `hunyuan-video` | HunyuanVideo (uncensored text to video) | video | UnetLoaderGGUF | DualCLIPLoader: `clip_l`, `llava_llama3_fp8_scaled` | `hunyuan_video_vae_bf16` | yes |
| `ltxv-0_9_6-gguf` | LTX-Video 0.9.6 distilled (fast video) | video | UnetLoaderGGUF | CLIPLoader: `t5xxl_fp8_e4m3fn_scaled` | `LTX-Video-0.9.6-VAE-BF16` | yes |

The same `qwen_3_4b` file serves both Z-Image and Flux.2 Klein: ComfyUI picks
the encoder from the loader's `type`, not the file.

Derived from those, never hand-written: image-to-image for every image family
whose graph can sample from an encoded picture; region refine, face and hand
detailing, hires fix and add-on chains in `src/lib/refine.ts`; shot chaining,
bookends and VACE reference shots in `src/lib/continuation.ts`; and per-half
add-on chains for the two-model Wan pairs. All of it is validated by
`npm run validate`.

### The catalogue

`server/catalog.json` knows 104 families; six of the fifteen above have an
entry in it, matched by the files they load rather than by name. The
catalogue panel behind More lists the ones not installed, what each is
missing, how big that is, and the server's fit verdict, with one link to
fetch. The nine with no catalogue entry (the SDXL, Anima and Krea finetunes,
Z-Image, the Wan 2.2 families, VACE 1.3B and LTX 0.9.6) are named with the
file to place by hand. The 98 catalogue families with no graph here are not
offered.

## The recipe

`src/lib/recipe.ts` takes three answers and returns everything the old form
collected by hand. Every add-on strength it applies is either measured
(Laplacian variance on this machine, one seed, one prompt, only the stack
changed) or the author's, and each row says which. Add-ons matched from the
prompt's vocabulary are offers the reader accepts or dismisses; an accepted
one is chained into the graph and its trigger words follow it into the
prompt. The explicit-content controls are two separate questions: is there a
person in the picture, and did the brief ask for explicit content. The
measurement is sharpness. It is not anatomical correctness, and the copy never
says it is.

## The archive

Records live beside the files, in `<outputs>/.switchgen/archive.json`, served
as a revision log. Every browser keeps localStorage as a cache: it pulls on
start, pushes what the server has never seen (which is how an archive that
predates the server migrates), and follows the server's event stream. A
removal is a tombstone, so a device that was offline learns of it. Files no
record describes are filed from the outputs folder, with settings read back
out of ComfyUI's own history where the graph survives.

## Reading a picture

`server/vision.mjs` runs the WD14 `wd-vit-tagger-v3` tagger and the Impact
Pack YOLO detectors through the ComfyUI venv's Python, on the CPU. A reading
is one link, never automatic: under a finished picture, under an attached
source, on every archive record, and in bulk from the archive rail. Tags go on
the record and are searchable as `tag:red_hair`. If the tagger is missing the
page says so and offers to fetch it.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` / `build` / `preview` | Vite |
| `npm run lint` | oxlint |
| `npm run validate` | Every base graph, image-to-image variant, quality derivation, continuation derivation and video add-on chain, checked node by node against ComfyUI's live `/object_info` |
| `npm run index-loras` | Reads every safetensors header in the LoRA folder and writes `src/lib/loraIndex.ts`: bases, triggers and training vocabulary, with no network calls |
| `npx tsx scripts/chain-e2e.ts` | A live two-shot reel on the 5B family, end to end, against the GPU |

## Runtime dependencies

Node 22 or later. ComfyUI 0.37 or later on `:8188` with the three node packs
under Getting started; `nvidia-smi` and GNU `df` for the hardware line;
`aria2c` for downloads; `ffmpeg` and `ffprobe` for the reel; the ComfyUI
venv's Python with `onnxruntime`, `numpy`, `Pillow` and `ultralytics` for the
reader. Each is probed, and a feature whose binary is missing stands down with
a sentence rather than failing later.

## Network posture

Vite binds every interface and allows this machine's hostname and addresses
and the tailnet, so the app is reachable from a phone. There is no login. The
same-origin guard stops a page on another origin from driving the write
routes from a browser, and every file path is confined to its root after
`realpath`. That is the whole of it: keep the server on a private network.

## License

MIT. See `LICENSE`.

## Adding a family

Add a `FamilyDef` to `FAMILY_DEFS` in `src/lib/registry.ts` with its graph
and bindings, then run `npm run validate`. The picker, the availability check,
the derivations and the validator all read the registry; nothing else needs
to know.

## Verifying a change

```bash
npx tsc -b && npx oxlint && npm run validate && npm run build
```
