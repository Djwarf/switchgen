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

### 1. Get the app

```bash
git clone https://github.com/Djwarf/switchgen.git
cd switchgen
```

With an SSH key registered on GitHub, `git clone
git@github.com:Djwarf/switchgen.git` works as well. Every command below runs
from this checkout.

### 2. Install ComfyUI

ComfyUI 0.37 or later, in its own Python venv, with three node packs the
graphs depend on:

| Pack | Provides |
|---|---|
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | `UnetLoaderGGUF`, `CLIPLoaderGGUF`: every quantised family |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | `FaceDetailer`, `ImpactGaussianBlurMask`: the face and hand passes, and the soft edge of region refine |
| [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack) | `UltralyticsDetectorProvider`: the detectors those passes use |

Everything else the graphs use ships with ComfyUI. `bin/switchgen validate`
(step 3) looks up every node class the graphs use before it looks at any
weights, and names a missing pack as a missing pack.

Point ComfyUI at a model library with the folder layout the app expects:
copy `contrib/extra_model_paths.yaml` into ComfyUI's root and set `base_path`.
The folders are `Stable-Diffusion`, `diffusion_models`, `text_encoders`,
`VAE`, `Lora`, `upscale_models`, `ultralytics/bbox`, `ultralytics/segm` and,
for the picture reader, `wd14`.

The passes offered on a finished picture need four files of their own, named
in `src/lib/refine.ts` and `server/vision.mjs`. The catalogue does not fetch
them:

| File | Folder | Used by |
|---|---|---|
| `4x-UltraSharp.pth` | `upscale_models` | region refine, which enlarges the masked region before drawing it again |
| `face_yolov8m.pt` | `ultralytics/bbox` | the face pass, and the picture reader |
| `hand_yolov8s.pt` | `ultralytics/bbox` | the hand pass, and the picture reader |
| `person_yolov8m-seg.pt` | `ultralytics/segm` | the picture reader |

The upscaler is at
[huggingface.co/Kim2091/UltraSharp](https://huggingface.co/Kim2091/UltraSharp)
(`4x-UltraSharp.pth`). The three detectors are the ones the Impact Subpack's
`install.py` downloads from
[huggingface.co/Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer).
ComfyUI-Manager runs that script when it installs the pack; a pack cloned by
hand has not run it. Run it yourself with the ComfyUI venv's Python, with
`COMFYUI_MODEL_PATH` set to the model library, so the files land where the
picture reader looks for them (it reads the library, not ComfyUI's own
`models` folder):

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-Impact-Subpack
COMFYUI_MODEL_PATH=/path/to/models /path/to/ComfyUI/venv/bin/python install.py
```

Without the upscaler ComfyUI refuses region refine, and `switchgen validate`
fails every region refine check with `"model_name"="4x-UltraSharp.pth" not in
[...]`. Without a detector it does the same for the face or hand pass.

Start ComfyUI with `--output-directory` pointing at the folder the app should
file outputs from. `contrib/comfyui.service` is a systemd user unit that does
this, which lets the launcher start ComfyUI on demand. First copy it:

```bash
mkdir -p ~/.config/systemd/user
cp contrib/comfyui.service ~/.config/systemd/user/
```

Then, in `~/.config/systemd/user/comfyui.service`, change `%h/ComfyUI` to the
folder ComfyUI is in and `%h/ai/outputs` to the folder `SWITCHGEN_OUTPUTS`
names. Only after that, enable it. Enabled with paths that are wrong for this
machine, the unit fails to start, and `Restart=always` tries again every five
seconds for as long as it is enabled.

```bash
systemctl --user daemon-reload && systemctl --user enable --now comfyui
```

The sample unit also passes `--enable-compress-response-body`, which has
ComfyUI gzip its JSON answers; pictures and clips are sent as they are. The
app's proxy does not compress what it passes on, and ComfyUI's model list,
which every desk reads when it opens, measured 2,154,283 bytes here as plain
text and 244,641 gzipped. A unit copied from an older checkout needs the flag
added by hand, then `systemctl --user daemon-reload && systemctl --user
restart comfyui` at a moment when nothing is rendering: a restart empties
ComfyUI's queue.

Optional, each probed and reported rather than assumed: `aria2c` for
downloads from the catalogue, `ffmpeg` and `ffprobe` for joining a reel and
for thumbnails, and `onnxruntime` plus `ultralytics` in the ComfyUI venv for
the picture reader.

### 3. Install the app

```bash
cp .env.example .env            # set the models and outputs folders, and COMFY_URL
npm install
bin/switchgen validate          # every graph against your ComfyUI's live schema
bin/switchgen install           # links the launcher into ~/.local/bin
switchgen                       # builds, starts ComfyUI if it can, serves on :5273
```

The launcher reads `.env`; npm scripts do not. `npm run validate` works as
well, with `COMFY_URL` exported in the shell when ComfyUI is not on
`127.0.0.1:8188`.

The desk offers only families whose files are installed and fit in memory.
Fetch the rest from the catalogue behind More, which shows each family's
missing files, their size and the server's fit verdict. Once a fetch has
begun it runs on the server, not in the page: a reload, a locked phone or a
dropped connection leaves it going. Before that, while the server is still
checking the disk and memory, closing the page calls it off. When the page comes back, the catalogue finds a family's
fetch again, with its Stop. An add-on or the picture reader's tagger is not
shown again by itself: pressing its fetch once more takes up the fetch still
running, with its progress and Stop, rather than starting a second. Add-ons
are indexed from your own LoRA folder with `npm run index-loras`, which reads
each file's header and rewrites `src/lib/loraIndex.ts`; the checked-in index
describes the folder this was built against. It reads `SWITCHGEN_LORA_DIR`, or
the `Lora` folder under `SWITCHGEN_MODELS`, from the environment, so export
them first. The tests carry their own index rows, so a regenerated index does
not change what `npm test` checks.

### 4. Use it from a phone

The app binds every interface. `localhost`, this machine's hostname and
addresses, and any `*.ts.net` name are allowed; add anything else to
`SWITCHGEN_ALLOWED_HOSTS`. The archive is the same on every device.

Open it on the phone over https. A browser treats a plain `http://` address
on the tailnet as insecure, and there it runs no service worker (so no kept
copy of the app for a weak signal), gives the page no clipboard and no way
to keep the screen on, and "Add to Home screen" makes a bookmark rather than
an installed app. Tailscale Serve puts https in front, with a certificate for
the machine's tailnet name. HTTPS certificates have to be turned on for the
tailnet first, on the DNS page of the Tailscale admin console. Then, once, on
the machine that runs SwitchGen:

```bash
tailscale serve --bg 5273
```

(or the port `SWITCHGEN_PORT` names). Open
`https://<machine>.<tailnet>.ts.net` on the phone and add it to the
home screen. `switchgen` prints that address when Serve is set up for its
port, and the command above when it is not. The setting stays until
`tailscale serve reset`.

To the browser the https address is a different site from the http one,
with storage of its own. The archive lives on the server and is the same at
either address; what the phone kept only in the browser under the old
address stays there, clips waiting in the Video lane and a reel in progress
included, so let those finish first. Current browsers mark the app's own
requests as same-origin, which the write guard accepts behind Serve with no
setting. If writes are refused as cross-site (an older browser, or a proxy
that rewrites the Host header), set `SWITCHGEN_TRUSTED_ORIGINS` in `.env` to
the https address.

The phone's icon finds the app only while it is running. To have it back
after the machine restarts, with nobody at the desk, install
`contrib/switchgen.service` as a user unit. First copy it:

```bash
mkdir -p ~/.config/systemd/user
cp contrib/switchgen.service ~/.config/systemd/user/
```

Then, in `~/.config/systemd/user/switchgen.service`, change `%h/switchgen`
to the folder this checkout is in. Only after that, enable it, and let your
user's units (ComfyUI's too) start when the machine does rather than when
you log in:

```bash
systemctl --user daemon-reload && systemctl --user enable --now switchgen
loginctl enable-linger "$USER"
```

The unit runs `switchgen serve`: a build when anything changed, then the
server in the foreground, restarted if it crashes. After a change, `systemctl
--user restart switchgen` serves it.

## The four rooms

| Room | Hash | What it does |
|---|---|---|
| Pictures | `#/pictures` | Three questions (prompt, look, anatomy), a recipe decided from measurements, add-ons offered rather than applied, region editing on a mask, and "What next" on every finished picture with the real cost of each pass. Also the instruction-edit desk and the picture reader. |
| Video | `#/video` | One clip at a time from words or a start frame, an add-on rack per family, and "What next" on a finished clip. |
| Reel | `#/reel` | A list of shots rendered in order, each opening on the last frame of the one before, joined by the server into one file. |
| Archive | `#/archive` | Every picture and clip, searchable, with the settings that made it. One archive for every device. |

The 14B text-to-video pair has been killed for memory when models from
earlier runs were still loaded, so a clip on either 14B pair waits until
ComfyUI has nothing queued or running, has ComfyUI release its cached models,
and only then is sent.
Until it is sent, nothing on the server knows about it. The clips waiting in
that lane are kept in the tab's own session storage: a reload of the tab
picks them up again in order, and closing the tab loses them, which the desk
says while any are waiting. A copied tab, or a page that crashed, does not
send what it finds on a guess; the desk lists those clips and asks whether to
send them from here or forget them. Stop, on the desk or in the section bar,
calls a waiting clip off before it is ever sent, and on the Pictures desk it
stops the rest of the batch as well as the picture in hand.

Three kinds of work wait in the page, not in ComfyUI: clips in that lane,
the shots of a reel after the one on the press, and the pictures of a batch
after the one being made. The page sends each when its turn comes, so while
the page is closed or hidden, or the phone is locked, nothing more is sent;
what ComfyUI already has carries on. The desk says so while anything waits,
and over https (step 4) the page keeps the screen on until the last one has
gone, except while the Video desk holds its lane after a lost clip: then
nothing goes until you send the waiting clips or call them off, and the
screen may sleep. A browser does not allow the wake lock over plain http.

A clip or picture already sent is kept in the tab's session storage with its
job number until it settles. When the page is reloaded, or the phone threw
the background tab away and opens it again, the desk follows it again, with
its progress and Stop, and files it when it finishes. If ComfyUI has
restarted in the meantime and has no record of it, the desk says the job was
lost rather than waiting on it.

One tab renders a reel at a time. Another tab open on the same reel shows the
shots as they land and which one is on the press, and sends nothing of its
own until that tab is done, so no shot is rendered twice. A reload hands the
shot on the press to the next page at once; a tab that closed or crashed is
taken over once it has been quiet for a few minutes.

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

Six Vite middlewares in `server/` mount under `/api` in both `vite dev` and
`vite preview`. Every mutating route runs the same-origin guard in
`server/guard.mjs`: the browser's `Sec-Fetch-Site` verdict when present, else
`Origin` must match `Host`, and the body must be the type the route reads. A
request with neither header (curl, a script) is allowed. Writes bound for
ComfyUI through `/comfy` (queueing a prompt, freeing memory, stopping a job,
uploading a picture) are held to the same origin rule before the proxy
forwards them, because the proxy rewrites `Origin` to ComfyUI's own and
ComfyUI's check can no longer see which page sent them. The WebSocket
handshake on `/comfy-ws` is checked for host and origin the same way. CORS
is off in both servers, so a page on another port of this machine cannot
read what either answers. `GET /api/capabilities` probes the binaries below
and reports what actually runs. A JSON answer of 1 KB or more is gzipped
when the browser accepts it.

ComfyUI sends the files behind `/view` with no `Cache-Control`, and it gives a
deleted file's name to the next render, so the proxy marks those answers
`no-cache`: the browser keeps its copy but asks before it plays or shows it
again, and an unchanged file costs a 304.

| Route | Server | Purpose |
|---|---|---|
| `GET /api/capabilities` | api | What this server can do, probed not assumed |
| `GET /api/hardware`, `GET /api/hardware/stream` | api | CPU, RAM, GPU, disk; one-shot, and SSE from one sampler shared by every viewer |
| `GET /api/models` | api | Every weight file under the models root, leaving out any still being fetched (an `.aria2` file beside it) |
| `POST /api/delete` | api | Unlink one output or model, confined to its root, and an output's thumbnails with it |
| `GET /api/thumb?rel=&w=` | thumbs | A WebP of one output, 256, 512 or 1024 pixels wide, made with ffmpeg and kept under `<outputs>/.switchgen/thumbs`; a picture it cannot shrink is redirected to the full file |
| `GET /api/archive`, `GET /api/archive/stream` | archive | The shared archive as a revision log, and its event stream |
| `POST /api/archive/upsert`, `/remove`, `/restore` | archive | Write records; the server assigns revisions and edition numbers |
| `GET /api/outputs` | archive | Every media file under the outputs root |
| `GET /api/catalog`, `GET /api/catalog/plan` | downloads | The 104-family catalogue annotated with what is on disk; a fit verdict and download plan per family |
| `POST /api/download`, `/download/cancel`, `GET /download/status` | downloads | Fetch through aria2c, resumable, as an event stream; a fetch goes on when its page goes, and the status lists each family fetch that is running or ended in the last ten minutes |
| `GET /api/reel/probe`, `POST /api/reel/stitch` | reel | Probe clips with ffprobe; join them with ffmpeg |
| `GET /api/vision/capabilities`, `POST /api/vision/tag`, `/detect`, `/inspect` | vision | The WD14 tagger and YOLO detectors, through the ComfyUI venv's Python, on the CPU, one reading at a time; a 503 with `busy: 'memory'` when free memory is too short to start one |

### Environment

Every path has a default and an override. The launcher reads them from a
`.env` file beside `package.json`; `.env.example` lists them all. npm scripts
(`npm run dev`, `validate`, `index-loras`) read only the environment, so
export them first, or use `switchgen dev` and `switchgen validate`.

| Variable | Default | Used by |
|---|---|---|
| `COMFY_URL` | `http://127.0.0.1:8188` | the Vite proxy, the launcher, `validate`, `chain-e2e` |
| `SWITCHGEN_PORT` | `5273` | where the app listens, and where the launcher looks for it |
| `SWITCHGEN_NO_OPEN` | empty | the launcher: any value starts without opening a browser |
| `SWITCHGEN_ALLOWED_HOSTS` | empty | extra hostnames the app may be reached by, comma separated |
| `SWITCHGEN_MODELS` | `/mnt/storage/ai/models` | api, downloads, vision, `index-loras` |
| `SWITCHGEN_OUTPUTS` | `/mnt/storage/ai/outputs` | api, archive, thumbs, reel, vision |
| `SWITCHGEN_ARCHIVE` | `<outputs>/.switchgen/archive.json` | archive |
| `SWITCHGEN_THUMBS` | `<outputs>/.switchgen/thumbs` | thumbs |
| `SWITCHGEN_CATALOG` | `server/catalog.json` | downloads |
| `SWITCHGEN_ARIA2C`, `SWITCHGEN_FFMPEG`, `SWITCHGEN_FFPROBE` | `/usr/bin/...` | downloads, reel, thumbs, capabilities |
| `HF_TOKEN_FILE` | `~/.cache/huggingface/token` | downloads, for gated files, only ever sent to HuggingFace |
| `SWITCHGEN_COMFY`, `SWITCHGEN_COMFY_INPUT` | `/mnt/storage/repos/ComfyUI`, `<comfy>/input` | vision |
| `SWITCHGEN_PYTHON` | `<comfy>/venv/bin/python` | vision |
| `SWITCHGEN_WD14` | `<models>/wd14` | vision |
| `SWITCHGEN_MEMORY_FLOOR_PERCENT` | `8` | vision: the share of RAM below which the system stops programs to get memory back (earlyoom's `-m`) |
| `SWITCHGEN_LORA_DIR` | `<models>/Lora` | `index-loras` |
| `SWITCHGEN_TRUSTED_ORIGINS` | empty | the guard, for a TLS terminator that rewrites Host |

## Usage

```bash
switchgen            # start ComfyUI if needed, build if anything changed, serve on :5273, open a browser
switchgen --no-open  # the same without opening a browser (also: switchgen start --no-open)
switchgen serve      # build if anything changed, then serve in the foreground (what the user unit runs)
switchgen dev        # hot-reloading dev server; refuses while the app is serving, since both would write one archive
switchgen phone      # the https address to open on a phone, or how to set one up
switchgen validate   # check every graph and derivation against ComfyUI's live schema
switchgen build      # build without serving
switchgen stop       # stop this checkout's server; ComfyUI is left running
switchgen install    # link the launcher into ~/.local/bin
```

The launcher lives at `bin/switchgen`, reads `.env` beside `package.json`,
and rebuilds whenever a source file is newer than the last build. Changes to
`server/*.mjs` need a restart (`switchgen stop && switchgen`, or `systemctl
--user restart switchgen` under the user unit), because the middlewares are
loaded when the server starts.

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
fetch. A file where the catalogue would put one, under its name but shorter
than the catalogue lists, and not a fetch that stopped part way, is taken to
be another build of it: it is kept and used as it is, never fetched over, and
the verdict says the family's files are on disk rather than that it is ready
to run. The nine with no catalogue entry (the SDXL, Anima and Krea finetunes,
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
removal is a tombstone, so a device that was offline learns of it. The log
names itself and each start of the server, so when the archive is started
again, or the server stopped before it saved changes it had already answered
for, a browser notices and sends back what the new log lacks. One server holds
the archive at a time, through `archive.json.lock` beside it; a second one
started on the same file (a dev server beside the running app, say) answers
503 for the archive, with a sentence the page quotes, until the first stops.

Files no record describes are filed from the outputs folder, with settings
read back out of ComfyUI's own history where the graph survives. One job is
filed once: two tabs following the same shot, or the recovery pass finding a
file its desk is about to file, make one record, and the desk's account
replaces the bare one the recovery pass made. That holds across devices too,
because the server itself folds a record the recovery pass filed into the
desk's record for the same file, which keeps its edition number and what the
reader added to it. A removed record stays removed. Its file stays on disk and
is remembered as dismissed; the recovery pass leaves it out and says so, and
files it again only when the reader asks. The server holds that line itself: a
record filed after the fact for a dismissed file is refused unless it says it
was asked for, so an older copy of the app still cached in some browser cannot
bring it back. A file written at that path after the removal is a new file,
and is filed as usual.

Delete the file removes every output the record's run wrote, a reel shot's
last frame as well as its clip, except a last frame the reel in this browser
still opens a shot on and a file another record names. There is no trash folder to take it
back from.

## Reading a picture

`server/vision.mjs` runs the WD14 `wd-vit-tagger-v3` tagger and the Impact
Pack YOLO detectors through the ComfyUI venv's Python, on the CPU. A reading
is one link, never automatic: under a finished picture, under an attached
source, on every archive record, and in bulk from the archive rail. Tags go on
the record and are searchable as `tag:red_hair`. If the tagger is missing the
page says so and offers to fetch it.

A reading is a process of its own that peaked, measured here, at 951 MB to
tag 24 pictures and 1.81 GB to read one with the detectors as well. A heavy
render can take free memory close to the line where the system stops its
largest program, usually ComfyUI, to get memory back, so one reading runs at
a time, whichever device asked, and a reading that would leave less than
that line (8% of RAM, earlyoom's setting here, `SWITCHGEN_MEMORY_FLOOR_PERCENT`)
plus half a GB is not started. The page gives both figures, what the reading
needs and what is free, and says to try again later.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` / `build` / `preview` | Vite |
| `npm start` | `bin/switchgen start` |
| `npm run lint` | oxlint |
| `npm test` | Vitest: the pure modules under `src/lib`, the desks' job engines and the press ledger with ComfyUI stood in, a few components rendered to a string, the service worker, the Vite config, the launcher's install advice, and the server middlewares against temporary folders. Needs no ComfyUI, no models, no running app and no LoRA index of your own |
| `npm run validate` | Every node class the graphs use, naming a missing node pack, then every base graph, image-to-image variant, quality derivation, continuation derivation and video add-on chain, checked node by node against ComfyUI's live `/object_info`. Exits 1 when anything fails and 2 when ComfyUI cannot be reached |
| `npm run index-loras` | Reads every safetensors header in the LoRA folder and writes `src/lib/loraIndex.ts`: bases, triggers and training vocabulary, with no network calls |
| `npx tsx scripts/chain-e2e.ts` | A live two-shot reel on the 5B family, end to end, against the GPU |

## Runtime dependencies

Node 22 or later. ComfyUI 0.37 or later on `:8188` with the three node packs
and the four pass files under Getting started; `nvidia-smi` and GNU `df` for
the hardware line; `aria2c` for downloads; `ffmpeg` and `ffprobe` for the reel
and thumbnails; the ComfyUI venv's Python with `onnxruntime`, `numpy`, `Pillow`
and `ultralytics` for the reader. Each is probed, and a feature whose binary is
missing stands down with a sentence rather than failing later.

## Network posture

Vite binds every interface and allows this machine's hostname and addresses
and the tailnet, so the app is reachable from a phone. It speaks plain http
itself; Tailscale Serve gives the tailnet an https address in front of it
(step 4). There is no login. The
same-origin guard stops a page on another origin from driving the write
routes from a browser, its own and ComfyUI's through the proxy alike, CORS is
off so such a page cannot read the answers either, and every file path is
confined to its root after `realpath`. That is the whole of it: keep the
server on a private network.

## License

MIT. See `LICENSE`.

## Adding a family

Add a `FamilyDef` to `FAMILY_DEFS` in `src/lib/registry.ts` with its graph
and bindings, then run `npm run validate`. The picker, the availability check,
the derivations and the validator all read the registry; nothing else needs
to know.

## Verifying a change

```bash
npx tsc -b && npx oxlint && npm test && npm run validate && npm run build
```

CI runs all of it except `validate`, which needs a live ComfyUI.
