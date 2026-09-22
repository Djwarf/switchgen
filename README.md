# SwitchGen

An editorial frontend for local AI image and video generation, built on the
SwitchSides classical-newspaper brand (burgundy `#2e0000` on newsprint,
Crimson Text, restraint palette).

Replaces the GTK4 [switchgen](https://github.com/Djwarf/switchgen), which
vendored ComfyUI as a submodule and imported it as a library. This talks to
ComfyUI over its HTTP + WebSocket API instead, so `comfy update` cannot break it.

## Architecture

```
Browser (React 19 + Tailwind 4)
   |  /comfy      -> HTTP   (proxied by Vite, single origin)
   |  /comfy-ws   -> WebSocket (live progress)
   v
ComfyUI service :8188        systemctl --user status comfyui
   |
/mnt/storage/ai/models       shared model library
/mnt/storage/ai/outputs      generated files
```

Nothing imports ComfyUI's Python internals. The only coupling is node names
and input schemas, which `npm run validate` checks against the live server.

## Usage

```bash
switchgen          # build if needed, start ComfyUI + serve the app, open browser
switchgen dev      # hot-reloading dev server
switchgen validate # check every workflow against ComfyUI's live schema
switchgen stop
```

## Supported model families

| Family | Mode | Text encoder | VAE |
|---|---|---|---|
| Stable Diffusion XL | image | bundled in checkpoint | bundled |
| Z-Image (Turbo / Base) | image | `qwen_3_4b` as `stable_diffusion` | `Flux/ae` |
| Flux.2 Klein | image | `qwen_3_4b` as `flux2` | `Flux/flux2-vae` |
| Anima | image | `oneObsession_anima29BV1_txt` | `qwen_image_vae` |
| Wan 2.2 TI2V 5B | video | `umt5_xxl_fp8_e4m3fn_scaled` | `wan2.2_vae` |

Note the same `qwen_3_4b.safetensors` serves both Z-Image and Flux.2 Klein —
ComfyUI selects the encoder from the CLIPLoader `type`, not the file.

A family whose text encoder or VAE is missing is hidden from the model picker
rather than offered and then failing at generation time.

## Adding a family

Add a `Recipe` to `RECIPES` in `src/lib/workflows.ts`, extend `familyOf()` so
its filenames map to it, then run `npm run validate`.
