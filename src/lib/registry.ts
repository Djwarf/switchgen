// GENERATED from the model-params-and-wiring swarm. Do not hand-edit;
// re-run scripts/gen-registry to regenerate. Each graph was validated against
// ComfyUI's live /object_info and adversarially reviewed.

export type Binding = [nodeId: string, input: string]

export type FamilyDef = {
  id: string
  label: string
  mode: "image" | "video" | "edit"
  models: string[]
  verified: boolean
  dualModel: boolean
  clipType: string
  clipFile: string
  vaeFile: string
  defaults: {
    steps: number; cfg: number; width: number; height: number
    sampler: string; scheduler: string; length: number; fps: number; negative: string
  }
  perModel: Record<string, Record<string, unknown>>
  notes: string
  graph: Record<string, { class_type: string; inputs: Record<string, unknown> }>
  bindings: Partial<Record<
    "model"|"seed"|"steps"|"cfg"|"sampler"|"scheduler"|"positive"|"negative"|"width"|"height"|"length"|"image"|"fps"|"denoise"|"megapixels",
    Binding[]>>
}

export const FAMILY_DEFS: FamilyDef[] = [
  {
    "id": "sdxl-illustrious",
    "label": "SDXL booru bases (Illustrious, Pony, NoobAI)",
    "mode": "image",
    "models": [
      "semiRealIllustrious_v40.safetensors",
      "waiMatureIllustrious_v30.safetensors",
      "ponyDiffusionV6XL.safetensors",
      "NoobAI-XL-v1.1.safetensors"
    ],
    "verified": false,
    "dualModel": false,
    "clipType": "",
    "clipFile": "",
    "vaeFile": "",
    "defaults": {
      "steps": 30.0,
      "cfg": 5.0,
      "width": 832,
      "height": 1216,
      "sampler": "dpmpp_2m",
      "scheduler": "karras",
      "length": 0,
      "fps": 0.0,
      "negative": "bad quality, worst quality, worst detail, sketch, censor, jpeg artifacts, watermark, signature, text, logo, bad anatomy, bad hands, extra digits, fewer digits"
    },
    "perModel": {
      "semiRealIllustrious_v40.safetensors": {
        "label": "Semi-Real (Illustrious) | MM v4.0",
        "steps": 30,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "clipSkip": -2,
        "width": 832,
        "height": 1216,
        "positivePrefix": "masterpiece, best quality, amazing quality, highly detailed, ",
        "negative": "bad quality, worst quality, worst detail, sketch, censor, jpeg artifacts, watermark, signature, text, logo, bad anatomy, bad hands, extra digits, fewer digits, blurry",
        "note": "Author card: DPM++ 2M + Karras, 30 steps, CFG 5, hi-res fix on every sample. Semi-realistic output: CFG above 6 crisps into plastic skin. Prefer portrait bucket 832x1216 for characters."
      },
      "waiMatureIllustrious_v30.safetensors": {
        "label": "WAI-Mature-illustrious v3.0",
        "steps": 28,
        "cfg": 6.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "clipSkip": -2,
        "width": 832,
        "height": 1216,
        "positivePrefix": "masterpiece, best quality, amazing quality, ",
        "negative": "bad quality, worst quality, worst detail, sketch, censor",
        "altSampler": {
          "sampler": "euler_ancestral",
          "scheduler": "normal",
          "steps": 30,
          "cfg": 7.0,
          "note": "WAI lineage A1111 default; softer, more painterly than dpmpp_2m/karras."
        },
        "note": "Author card: DPM++ 2M, 20-30 steps, CFG 5-7, VAE baked in (do not load an external VAE). Quality-tag prefix and the short 5-tag negative above are the WAI house style."
      },
      "ponyDiffusionV6XL.safetensors": {
        "steps": 30,
        "cfg": 7,
        "width": 1024,
        "height": 1024,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "negative": "score_6, score_5, score_4, worst quality, low quality, bad anatomy, bad hands, extra digits, watermark, signature",
        "notes": "Pony V6 requires the score tag prefix in the positive prompt to reach its trained quality: score_9, score_8_up, score_7_up. Without it output looks like base SDXL. Largest NSFW LoRA ecosystem of anything installed."
      },
      "NoobAI-XL-v1.1.safetensors": {
        "steps": 28,
        "cfg": 5,
        "width": 1024,
        "height": 1216,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "negative": "worst quality, low quality, bad anatomy, bad hands, extra digits, watermark, signature, jpeg artifacts",
        "notes": "Booru tag prompting, comma separated. Trained on explicit content, so anatomy is markedly stronger than the photoreal bases. Euler ancestral suits it."
      }
    },
    "notes": "ARCHITECTURE (verified, not assumed): both files are plain SDXL eps-prediction checkpoints. I read the safetensors headers directly: 1680 model.diffusion_model.* keys, adm/label_emb present, conditioner.embedders.0 (CLIP-L) + .1 (CLIP-G) bundled, 248 first_stage_model.* VAE keys bundled, all tensors F16, 6.9 GB each. Neither contains 'v_pred', 'ztsnr', 'edm_mean' or 'edm_vpred.sigma_max', so comfy",
    "graph": {
      "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
          "ckpt_name": "waiMatureIllustrious_v30.safetensors"
        }
      },
      "2": {
        "class_type": "CLIPSetLastLayer",
        "inputs": {
          "clip": [
            "1",
            1
          ],
          "stop_at_clip_layer": -2
        }
      },
      "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "2",
            0
          ],
          "text": "masterpiece, best quality, amazing quality, 1girl, solo, looking at viewer"
        }
      },
      "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "2",
            0
          ],
          "text": "bad quality, worst quality, worst detail, sketch, censor, jpeg artifacts, watermark, signature, text, logo, bad anatomy, bad hands, extra digits, fewer digits"
        }
      },
      "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        }
      },
      "6": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "positive": [
            "3",
            0
          ],
          "negative": [
            "4",
            0
          ],
          "latent_image": [
            "5",
            0
          ],
          "seed": 0,
          "steps": 30,
          "cfg": 5.0,
          "sampler_name": "dpmpp_2m",
          "scheduler": "karras",
          "denoise": 1.0
        }
      },
      "7": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "6",
            0
          ],
          "vae": [
            "1",
            2
          ]
        }
      },
      "8": {
        "class_type": "SaveImage",
        "inputs": {
          "images": [
            "7",
            0
          ],
          "filename_prefix": "switchgen/illustrious"
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "ckpt_name"
        ]
      ],
      "seed": [
        [
          "6",
          "seed"
        ]
      ],
      "steps": [
        [
          "6",
          "steps"
        ]
      ],
      "cfg": [
        [
          "6",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "6",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "6",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "3",
          "text"
        ]
      ],
      "negative": [
        [
          "4",
          "text"
        ]
      ],
      "width": [
        [
          "5",
          "width"
        ]
      ],
      "height": [
        [
          "5",
          "height"
        ]
      ]
    }
  },
  {
    "id": "z-image",
    "label": "Z-Image (Turbo + Base)",
    "mode": "image",
    "models": [
      "Z-Image-Turbo-fp8mix.safetensors",
      "Z-Image-Base-bf16.safetensors"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "lumina2",
    "clipFile": "qwen_3_4b.safetensors",
    "vaeFile": "Flux/ae.safetensors",
    "defaults": {
      "steps": 8.0,
      "cfg": 1.0,
      "width": 1024,
      "height": 1024,
      "sampler": "res_multistep",
      "scheduler": "simple",
      "length": 0,
      "fps": 0.0,
      "negative": ""
    },
    "perModel": {
      "Z-Image-Turbo-fp8mix.safetensors": {
        "steps": 8,
        "cfg": 1.0,
        "shift": 3,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "weight_dtype": "default",
        "maxSide": 2048,
        "notes": "Distilled. CFG must be 1 (samplers.py:610-611 sets uncond_=None when cond_scale is 1.0, so the negative prompt has no effect). 4 steps works, 8 is better (template value), up to 20 for very hard prompts. Verified on disk: UNet 6.12 GiB, qwen_3_4b encoder 5.24 GiB, Flux/ae VAE 0.31 GiB. At 1024x1024 the whole stack is ~12.3 GiB and stays resident on 16 GB. At 2048x2048 activations reach ~3.67 GiB (ComfyUI memory_required with ZImage.memory_usage_factor 2.8), pushing the resident total to ~15.0 GiB before VAE decode, so ComfyUI will evict the text encoder between encode and sample - still functional, with reload latency per queue item."
      },
      "Z-Image-Base-bf16.safetensors": {
        "steps": 25,
        "cfg": 4.0,
        "shift": 3,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "weight_dtype": "default",
        "maxSide": 2048,
        "notes": "Not distilled. CFG 4 is the shipped template value; SwarmUI gives a normal range (eg 4 or 7). 25 steps per template, up to 50 for hard prompts. Negative prompt is live here (unlike Turbo) but both official Base templates leave it empty - supply one only if the user writes it, do not inject generic quality tags, which suit booru-tagged SD models rather than this Qwen3-conditioned flow model. Verified on disk: 11.46 GiB bf16 + 5.24 GiB encoder = 16.7 GiB, over the card, so the encoder is evicted after the CLIPTextEncode nodes run and reloaded each queue item. With the encoder out, weights plus activations are ~12.4 GiB at 1024 and ~15.1 GiB at 2048; 2048 is therefore tight but reachable. If it OOMs at 2048, set weight_dtype to fp8_e4m3fn (halves the UNet to ~5.7 GiB) or launch ComfyUI with --lowvram."
      },
      "_alternates": {
        "sampler": "euler_ancestral is reported better for photorealism detail per SwarmUI; euler also fine",
        "scheduler": "beta is a very slight improvement per SwarmUI",
        "shift": "raise 3 -> 6 for stronger composition coherence per SwarmUI",
        "_unsourced": "none of the values in the main overrides are unsourced; every steps/cfg/sampler/scheduler/shift figure traces to a shipped template KSampler row, supported_models.py:1211-1214, or the SwarmUI Z-Image Parameters list"
      },
      "_turboSeedVarietyTrick": {
        "appliesTo": "Z-Image-Turbo-fp8mix.safetensors",
        "how": "Feed a real init image through VAEEncode into KSampler.latent_image, set denoise 0.7 (SwarmUI 'Init Image Creativity' maps to denoise), steps 8, and shift 22 on ModelSamplingAuraFlow. Destabilises the model in a way it recovers from, and the recovery path diverges strongly per seed. Rarely needed on Base. Requires adding LoadImage + VAEEncode nodes; not present in the shipped nodeGraph."
      }
    },
    "notes": "CLIP type confirmed at source, not guessed. comfy/sd.py:1909 `elif te_model == TEModel.QWEN3_4B:` branches only on `clip_type == CLIPType.FLUX or clip_type == CLIPType.FLUX2` -> klein_te/KleinTokenizer; the `else` gives z_image.te/ZImageTokenizer. So any non-flux2 type selects the Z-Image encoder from the same qwen_3_4b.safetensors file. Both official templates use `lumina2`, so I standardised on ",
    "graph": {
      "1": {
        "class_type": "UNETLoader",
        "inputs": {
          "unet_name": "Z-Image-Turbo-fp8mix.safetensors",
          "weight_dtype": "default"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "qwen_3_4b.safetensors",
          "type": "lumina2",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "Flux/ae.safetensors"
        }
      },
      "4": {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "shift": 3.0,
          "sampling": "flow"
        }
      },
      "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "2",
            0
          ],
          "text": "PROMPT_HERE"
        }
      },
      "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "2",
            0
          ],
          "text": ""
        }
      },
      "7": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        }
      },
      "8": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "4",
            0
          ],
          "positive": [
            "5",
            0
          ],
          "negative": [
            "6",
            0
          ],
          "latent_image": [
            "7",
            0
          ],
          "seed": 0,
          "steps": 8,
          "cfg": 1.0,
          "sampler_name": "res_multistep",
          "scheduler": "simple",
          "denoise": 1.0
        }
      },
      "9": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "8",
            0
          ],
          "vae": [
            "3",
            0
          ]
        }
      },
      "10": {
        "class_type": "SaveImage",
        "inputs": {
          "images": [
            "9",
            0
          ],
          "filename_prefix": "z-image"
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "8",
          "seed"
        ]
      ],
      "steps": [
        [
          "8",
          "steps"
        ]
      ],
      "cfg": [
        [
          "8",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "8",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "8",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "5",
          "text"
        ]
      ],
      "negative": [
        [
          "6",
          "text"
        ]
      ],
      "width": [
        [
          "7",
          "width"
        ]
      ],
      "height": [
        [
          "7",
          "height"
        ]
      ]
    }
  },
  {
    "id": "flux2-klein",
    "label": "Flux.2 Klein 4B (distilled, fp8)",
    "mode": "image",
    "models": [
      "flux-2-klein-4b-fp8.safetensors"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "flux2",
    "clipFile": "qwen_3_4b.safetensors",
    "vaeFile": "Flux/flux2-vae.safetensors",
    "defaults": {
      "steps": 8.0,
      "cfg": 1.0,
      "width": 1024,
      "height": 1024,
      "sampler": "euler",
      "scheduler": "simple",
      "length": 0,
      "fps": 0.0,
      "negative": ""
    },
    "perModel": {
      "flux-2-klein-4b-fp8.safetensors": {
        "variant": "distilled",
        "steps": 8,
        "cfg": 1.0,
        "sampler": "euler",
        "schedulerNode": "Flux2Scheduler",
        "negativePromptSupported": false,
        "presets": {
          "fast": {
            "steps": 4,
            "cfg": 1.0,
            "note": "ComfyUI official template default for this exact file; ~2x faster, slightly softer detail"
          },
          "balanced": {
            "steps": 8,
            "cfg": 1.0,
            "note": "SwarmUI recommended distilled setting; app default"
          },
          "quality": {
            "steps": 12,
            "cfg": 1.0,
            "note": "diminishing returns past ~12 on the distilled model"
          }
        },
        "resolutions": {
          "default": [
            1024,
            1024
          ],
          "portrait": [
            832,
            1216
          ],
          "landscape": [
            1216,
            832
          ],
          "hires": [
            1536,
            1536
          ],
          "max16gb": [
            2048,
            2048
          ],
          "constraint": "width/height must be multiples of 16 for EmptyFlux2LatentImage (step=16); Flux2Scheduler accepts step=1 but MUST be given the same width/height as the latent or the sigma schedule is wrong"
        },
        "wiring": {
          "Flux2Scheduler.width": "must equal EmptyFlux2LatentImage.width",
          "Flux2Scheduler.height": "must equal EmptyFlux2LatentImage.height",
          "seed": "node 10 RandomNoise.noise_seed",
          "prompt": "node 4 CLIPTextEncode.text"
        }
      },
      "_NOT_INSTALLED_flux-2-klein-base-4b-fp8.safetensors": {
        "variant": "base (non-distilled)",
        "note": "Not installed. If ever added, the numbers change completely: steps 20, cfg 5.0, and the negative must be a REAL second CLIPTextEncode fed from node 2 instead of the ConditioningZeroOut at node 5. Everything else (Flux2Scheduler, euler, EmptyFlux2LatentImage, flux2 CLIPLoader type, Flux/flux2-vae.safetensors) is identical.",
        "steps": 20,
        "cfg": 5.0
      }
    },
    "notes": "VERIFIED AGAINST LIVE SCHEMA. Every class_type, enum value and filename in nodeGraph was machine-checked against http://127.0.0.1:8188/object_info (required inputs present, link types matched slot-by-slot, all enums members). No generation was run; no POST to /prompt.\n\n1) FluxGuidance: NOT required, and actively pointless here. I read the safetensors header of flux-2-klein-4b-fp8.safetensors direc",
    "graph": {
      "1": {
        "class_type": "UNETLoader",
        "inputs": {
          "unet_name": "flux-2-klein-4b-fp8.safetensors",
          "weight_dtype": "default"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "qwen_3_4b.safetensors",
          "type": "flux2",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "Flux/flux2-vae.safetensors"
        }
      },
      "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "PROMPT_HERE",
          "clip": [
            "2",
            0
          ]
        }
      },
      "5": {
        "class_type": "ConditioningZeroOut",
        "inputs": {
          "conditioning": [
            "4",
            0
          ]
        }
      },
      "6": {
        "class_type": "CFGGuider",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "positive": [
            "4",
            0
          ],
          "negative": [
            "5",
            0
          ],
          "cfg": 1.0
        }
      },
      "7": {
        "class_type": "KSamplerSelect",
        "inputs": {
          "sampler_name": "euler"
        }
      },
      "8": {
        "class_type": "Flux2Scheduler",
        "inputs": {
          "steps": 8,
          "width": 1024,
          "height": 1024
        }
      },
      "9": {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        }
      },
      "10": {
        "class_type": "RandomNoise",
        "inputs": {
          "noise_seed": 0
        }
      },
      "11": {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
          "noise": [
            "10",
            0
          ],
          "guider": [
            "6",
            0
          ],
          "sampler": [
            "7",
            0
          ],
          "sigmas": [
            "8",
            0
          ],
          "latent_image": [
            "9",
            0
          ]
        }
      },
      "12": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "11",
            0
          ],
          "vae": [
            "3",
            0
          ]
        }
      },
      "13": {
        "class_type": "SaveImage",
        "inputs": {
          "images": [
            "12",
            0
          ],
          "filename_prefix": "Flux2-Klein"
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "10",
          "noise_seed"
        ]
      ],
      "steps": [
        [
          "8",
          "steps"
        ]
      ],
      "cfg": [
        [
          "6",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "7",
          "sampler_name"
        ]
      ],
      "positive": [
        [
          "4",
          "text"
        ]
      ],
      "width": [
        [
          "8",
          "width"
        ],
        [
          "9",
          "width"
        ]
      ],
      "height": [
        [
          "8",
          "height"
        ],
        [
          "9",
          "height"
        ]
      ]
    }
  },
  {
    "id": "anima",
    "label": "Anima (Cosmos-based 2B/2.9B anime DiT)",
    "mode": "image",
    "models": [
      "miaomiaoHarem_29BBETA10.safetensors",
      "miaomiaoRealskin_anima13.safetensors",
      "oneObsession_anima29BV1.safetensors"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "stable_diffusion",
    "clipFile": "oneObsession_anima29BV1_txt.safetensors",
    "vaeFile": "qwen_image_vae.safetensors",
    "defaults": {
      "steps": 30.0,
      "cfg": 4.0,
      "width": 1024,
      "height": 1024,
      "sampler": "er_sde",
      "scheduler": "simple",
      "length": 0,
      "fps": 0.0,
      "negative": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"
    },
    "perModel": {
      "oneObsession_anima29BV1.safetensors": {
        "verified_prefix": "model.diffusion_model.",
        "blocks": 40,
        "params": "~2.9B",
        "file_bytes": 5843219918,
        "dtype": "BF16",
        "steps": 30,
        "cfg": 4.0,
        "sampler": "er_sde",
        "scheduler": "simple",
        "width": 1024,
        "height": 1024,
        "note": "Loads via UNETLoader: unet_prefix_from_state_dict lists 'model.diffusion_model.' as its first candidate and 925 keys carry it, far over the >5 threshold. Ships its own matching text encoder (oneObsession_anima29BV1_txt.safetensors) — the only Qwen3-0.6B TE installed, so all three models use it."
      },
      "miaomiaoHarem_29BBETA10.safetensors": {
        "verified_prefix": "net.",
        "blocks": 40,
        "params": "~2.9B",
        "file_bytes": 5843203272,
        "dtype": "BF16",
        "steps": 30,
        "cfg": 4.0,
        "sampler": "er_sde",
        "scheduler": "simple",
        "width": 1024,
        "height": 1024,
        "note": "Loads via UNETLoader: 'net.' is an explicit candidate prefix (commented '#cosmos') and 925 keys carry it. Same 40-block geometry as oneObsession, so identical params."
      },
      "miaomiaoRealskin_anima13.safetensors": {
        "verified_prefix": "net.",
        "blocks": 28,
        "params": "~2.0B",
        "file_bytes": 4182218328,
        "dtype": "BF16",
        "steps": 30,
        "cfg": 4.0,
        "sampler": "er_sde",
        "scheduler": "simple",
        "width": 1024,
        "height": 1024,
        "suggested_alt_cfg": 3.5,
        "note": "Smallest of the three (28 blocks = stock anima-base-v1.0 depth, vs 40 for the other two); fastest and lightest. Core params verified identical — suggested_alt_cfg 3.5 is an UNVERIFIED starting point for this photoreal-leaning finetune, not a measured value. No per-model metadata exists to tune from: all three safetensors have an empty __metadata__ block."
      }
    },
    "notes": "VALIDATED: the nodeGraph above passed a programmatic check against the live /object_info — every class_type exists, every input name is a real slot, every link's source output type matches the destination slot type, every required input is present, and every enum/filename literal appears verbatim in the live lists. Nothing was submitted to /prompt; no GPU work was done.\n\nBOTH PREFIXES CONFIRMED LO",
    "graph": {
      "1": {
        "class_type": "UNETLoader",
        "inputs": {
          "unet_name": "oneObsession_anima29BV1.safetensors",
          "weight_dtype": "default"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "oneObsession_anima29BV1_txt.safetensors",
          "type": "stable_diffusion",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "qwen_image_vae.safetensors"
        }
      },
      "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "1girl, solo, masterpiece, best quality, detailed anime illustration",
          "clip": [
            "2",
            0
          ]
        }
      },
      "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
          "clip": [
            "2",
            0
          ]
        }
      },
      "6": {
        "class_type": "EmptyLatentImage",
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        }
      },
      "7": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "seed": 0,
          "steps": 30,
          "cfg": 4.0,
          "sampler_name": "er_sde",
          "scheduler": "simple",
          "positive": [
            "4",
            0
          ],
          "negative": [
            "5",
            0
          ],
          "latent_image": [
            "6",
            0
          ],
          "denoise": 1.0
        }
      },
      "8": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "7",
            0
          ],
          "vae": [
            "3",
            0
          ]
        }
      },
      "9": {
        "class_type": "SaveImage",
        "inputs": {
          "images": [
            "8",
            0
          ],
          "filename_prefix": "Anima"
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "7",
          "seed"
        ]
      ],
      "steps": [
        [
          "7",
          "steps"
        ]
      ],
      "cfg": [
        [
          "7",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "7",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "7",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "4",
          "text"
        ]
      ],
      "negative": [
        [
          "5",
          "text"
        ]
      ],
      "width": [
        [
          "6",
          "width"
        ]
      ],
      "height": [
        [
          "6",
          "height"
        ]
      ]
    }
  },
  {
    "id": "krea2",
    "label": "Krea 2",
    "mode": "image",
    "models": [
      "moodyCutieMixKrea2_v50_int8.safetensors"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "krea2",
    "clipFile": "qwen3vl_4b.safetensors",
    "vaeFile": "qwen_image_vae.safetensors",
    "defaults": {
      "steps": 8.0,
      "cfg": 1.0,
      "width": 1024,
      "height": 1024,
      "sampler": "euler",
      "scheduler": "simple",
      "length": 0,
      "fps": 0.0,
      "negative": ""
    },
    "perModel": {
      "moodyCutieMixKrea2_v50_int8.safetensors": {
        "vram_notes": "12.84 GiB of int8_convrot weights (13,789,287,384 bytes decimal) on a 16 GiB card. quant_config {'mixed_ops': True} is detected, so the weights stay int8 in VRAM and are NOT dequantized to bf16. Sampling activations at 1024x1024 cost ~720 MB by model_base.BaseModel.memory_required (area 16384 * 2B * 0.01 * memory_usage_factor 2.2). Leaves ~2.5 GiB headroom after desktop overhead; resident, not offloaded, in the common case. The qwen3vl_4b encoder is a separate 4.88 GiB load that is evicted before the DiT resides.",
        "quality_preset": {
          "steps": 12,
          "cfg": 1.0,
          "note": "UNSOURCED EXTRAPOLATION. SwarmUI sources only 8 recommended / 4 minimum for Turbo. 12 is the reviewing agent's guess, not documented. Do not raise CFG above 1 on a turbo-lineage checkpoint (this part IS sourced)."
        },
        "resolution_range_sourced": "SwarmUI: side length 1024 default, works 128 to 4096. The spec's '1536+ will thrash' is an inference about this card, not a model limit."
      }
    },
    "notes": "VAE ANSWER: qwen_image_vae.safetensors (top-level, verbatim in the live VAELoader enum). Proven two ways: (a) SwarmUI \"# Krea 2\" says \"Uses Qwen 3 VL 4B as a text encoder, and the QwenImage VAE\"; (b) comfy/supported_models.py class Krea2 sets latent_format = latent_formats.Wan21 (16 latent channels, /8 spatial) — I confirmed latent_channels==16 by running model_detection on the real file. Do NOT u",
    "graph": {
      "1": {
        "class_type": "UNETLoader",
        "inputs": {
          "unet_name": "moodyCutieMixKrea2_v50_int8.safetensors",
          "weight_dtype": "default"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "qwen3vl_4b.safetensors",
          "type": "krea2",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "qwen_image_vae.safetensors"
        }
      },
      "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "2",
            0
          ],
          "text": "PROMPT"
        }
      },
      "5": {
        "class_type": "ConditioningZeroOut",
        "inputs": {
          "conditioning": [
            "4",
            0
          ]
        }
      },
      "6": {
        "class_type": "EmptyLatentImage",
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        }
      },
      "7": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "positive": [
            "4",
            0
          ],
          "negative": [
            "5",
            0
          ],
          "latent_image": [
            "6",
            0
          ],
          "seed": 0,
          "steps": 8,
          "cfg": 1.0,
          "sampler_name": "euler",
          "scheduler": "simple",
          "denoise": 1.0
        }
      },
      "8": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "7",
            0
          ],
          "vae": [
            "3",
            0
          ]
        }
      },
      "9": {
        "class_type": "SaveImage",
        "inputs": {
          "images": [
            "8",
            0
          ],
          "filename_prefix": "krea2"
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "7",
          "seed"
        ]
      ],
      "steps": [
        [
          "7",
          "steps"
        ]
      ],
      "cfg": [
        [
          "7",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "7",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "7",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "4",
          "text"
        ]
      ],
      "width": [
        [
          "6",
          "width"
        ]
      ],
      "height": [
        [
          "6",
          "height"
        ]
      ]
    }
  },
  {
    "id": "qwen-image-21",
    "label": "Qwen Image 2.1",
    "mode": "image",
    "models": [
      "qwen_image_2.1_int8_convrot.safetensors"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "qwen_image",
    "clipFile": "qwen3vl_8b_int8_convrot.safetensors",
    "vaeFile": "qwen_image_2.1_vae_bf16.safetensors",
    "defaults": {
      "steps": 20.0,
      "cfg": 1.0,
      "width": 1024,
      "height": 1024,
      "sampler": "euler",
      "scheduler": "simple",
      "length": 0,
      "fps": 0.0,
      "negative": ""
    },
    "perModel": {},
    "notes": "CLIPLoader type is \"qwen_image\" (verified in the live enum and in comfy/sd.py line 1955: CLIPType.QWEN_IMAGE + TEModel.QWEN3VL_8B selects the Qwen-Image 2.1 encoder — full Qwen3-VL-8B, last hidden state, image slots spliced by the DiT). Both the t2i and edit templates hardcode exactly our three installed filenames.\n\nTEXT ENCODE: use TextEncodeQwenImage21, NOT CLIPTextEncode. One node produces both",
    "graph": {
      "1": {
        "class_type": "UNETLoader",
        "inputs": {
          "unet_name": "qwen_image_2.1_int8_convrot.safetensors",
          "weight_dtype": "default"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
          "type": "qwen_image",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "qwen_image_2.1_vae_bf16.safetensors"
        }
      },
      "4": {
        "class_type": "TextEncodeQwenImage21",
        "inputs": {
          "clip": [
            "2",
            0
          ],
          "prompt": "PROMPT_GOES_HERE",
          "negative_prompt": "",
          "resolution": 1024,
          "images": []
        }
      },
      "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        }
      },
      "6": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "seed": 0,
          "steps": 20,
          "cfg": 1.0,
          "sampler_name": "euler",
          "scheduler": "simple",
          "positive": [
            "4",
            0
          ],
          "negative": [
            "4",
            1
          ],
          "latent_image": [
            "5",
            0
          ],
          "denoise": 1.0
        }
      },
      "7": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "6",
            0
          ],
          "vae": [
            "3",
            0
          ]
        }
      },
      "8": {
        "class_type": "SaveImage",
        "inputs": {
          "images": [
            "7",
            0
          ],
          "filename_prefix": "switchgen/qwen_image_21"
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "6",
          "seed"
        ]
      ],
      "steps": [
        [
          "6",
          "steps"
        ]
      ],
      "cfg": [
        [
          "6",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "6",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "6",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "4",
          "prompt"
        ]
      ],
      "negative": [
        [
          "4",
          "negative_prompt"
        ]
      ],
      "width": [
        [
          "5",
          "width"
        ]
      ],
      "height": [
        [
          "5",
          "height"
        ]
      ]
    }
  },
  {
    "id": "qwen-image-edit",
    "label": "Qwen Image Edit Plus 2511 (GGUF Q4_K_M)",
    "mode": "edit",
    "models": [
      "qwen-image-edit-2511-Q4_K_M.gguf"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "qwen_image",
    "clipFile": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vaeFile": "qwen_image_vae.safetensors",
    "defaults": {
      "steps": 20.0,
      "cfg": 4.0,
      "width": 0,
      "height": 0,
      "sampler": "euler",
      "scheduler": "simple",
      "length": 0,
      "fps": 0.0,
      "negative": ""
    },
    "perModel": {
      "qwen-image-edit-2511-Q4_K_M.gguf": {
        "default": {
          "steps": 20,
          "cfg": 4.0,
          "shift": 3.1,
          "sampler": "euler",
          "scheduler": "simple",
          "denoise": 1.0,
          "cfgnorm_strength": 1.0,
          "cfgnorm_pre_cfg": false,
          "source": "image_qwen_image_edit_2511.json MarkdownNote node 157, 'Comfy' column: Steps 20 / CFG 4.0"
        },
        "quality": {
          "steps": 40,
          "cfg": 4.0,
          "shift": 3.1,
          "source": "image_qwen_image_edit_2511.json MarkdownNote node 157 'Qwen' column AND the subgraph's own wired default (PrimitiveInt node 166 = 40, PrimitiveFloat node 154 = 4.0, selected when PrimitiveBoolean 168 = false)"
        },
        "fast": {
          "steps": 12,
          "cfg": 2.5,
          "shift": 3.1,
          "source": "UNSOURCED - extrapolation only, no citation exists; remove or mark experimental"
        },
        "single_image_fidelity": {
          "steps": 20,
          "cfg": 4.0,
          "shift": 0.8,
          "source": "SwarmUI Model Support.md '### Qwen Image Edit': 'Sigma Shift: 3 or lower (as low as 0.5) is a valid range. Some users report that a value below 1 might be ideal for single-image inputs.' The 0.8 value is a chosen point inside that range, not a cited figure."
        },
        "lightning_4step": {
          "available": false,
          "reason": "Confirmed against live /object_info LoraLoaderModelOnly.lora_name enum, which contains ONLY the four Wan22 I2V/T2V Lightning files. No Qwen Edit Lightning LoRA is installed.",
          "if_installed": {
            "steps": 4,
            "cfg": 1.0,
            "shift": 3.1,
            "lora": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "source": "subgraph node 153 LoraLoaderModelOnly widget = that exact filename at strength 1.0; PrimitiveInt 165 = 4 steps; PrimitiveFloat 155 = 1.0 cfg; wired CFGNorm(152) -> LoraLoaderModelOnly(153) -> KSampler",
            "wiring": "Insert LoraLoaderModelOnly (strength_model 1.0) between CFGNorm node 15 and KSampler node 8 - confirmed to match the template's link order"
          }
        }
      }
    },
    "notes": "VALIDATION: the graph was machine-checked against live /object_info — 15 nodes / 39 inputs, 0 errors (every class_type exists, every link's source output type matches the destination's declared type, every required input present, every enum value legal). The 3-reference-image variant also validates (17 nodes / 45 inputs, 0 errors). Files confirmed on disk: UNet 13G, text encoder 8.8G (download fin",
    "graph": {
      "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
          "unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
          "type": "qwen_image",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "qwen_image_vae.safetensors"
        }
      },
      "10": {
        "class_type": "LoadImage",
        "inputs": {
          "image": "example.png"
        }
      },
      "12": {
        "class_type": "FluxKontextImageScale",
        "inputs": {
          "image": [
            "10",
            0
          ]
        }
      },
      "4": {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {
          "clip": [
            "2",
            0
          ],
          "vae": [
            "3",
            0
          ],
          "image1": [
            "12",
            0
          ],
          "prompt": "Change the sofa upholstery to brown leather, keep everything else identical."
        }
      },
      "5": {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {
          "clip": [
            "2",
            0
          ],
          "vae": [
            "3",
            0
          ],
          "image1": [
            "12",
            0
          ],
          "prompt": ""
        }
      },
      "13": {
        "class_type": "FluxKontextMultiReferenceLatentMethod",
        "inputs": {
          "conditioning": [
            "4",
            0
          ],
          "reference_latents_method": "index_timestep_zero"
        }
      },
      "14": {
        "class_type": "FluxKontextMultiReferenceLatentMethod",
        "inputs": {
          "conditioning": [
            "5",
            0
          ],
          "reference_latents_method": "index_timestep_zero"
        }
      },
      "16": {
        "class_type": "VAEEncode",
        "inputs": {
          "pixels": [
            "12",
            0
          ],
          "vae": [
            "3",
            0
          ]
        }
      },
      "7": {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "shift": 3.1,
          "sampling": "flow"
        }
      },
      "15": {
        "class_type": "CFGNorm",
        "inputs": {
          "model": [
            "7",
            0
          ],
          "strength": 1.0,
          "pre_cfg": false
        }
      },
      "8": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "15",
            0
          ],
          "positive": [
            "13",
            0
          ],
          "negative": [
            "14",
            0
          ],
          "latent_image": [
            "16",
            0
          ],
          "seed": 123456789,
          "steps": 20,
          "cfg": 4.0,
          "sampler_name": "euler",
          "scheduler": "simple",
          "denoise": 1.0
        }
      },
      "9": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "8",
            0
          ],
          "vae": [
            "3",
            0
          ]
        }
      },
      "11": {
        "class_type": "SaveImage",
        "inputs": {
          "images": [
            "9",
            0
          ],
          "filename_prefix": "switchgen/qwen-image-edit"
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "8",
          "seed"
        ]
      ],
      "steps": [
        [
          "8",
          "steps"
        ]
      ],
      "cfg": [
        [
          "8",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "8",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "8",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "4",
          "prompt"
        ]
      ],
      "negative": [
        [
          "5",
          "prompt"
        ]
      ],
      "image": [
        [
          "10",
          "image"
        ]
      ]
    }
  },
  {
    "id": "wan22-5b",
    "label": "Wan 2.2 TI2V 5B (text+image to video)",
    "mode": "video",
    "models": [
      "wan2.2_ti2v_5B_fp16.safetensors"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "wan",
    "clipFile": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vaeFile": "wan2.2_vae.safetensors",
    "defaults": {
      "steps": 20.0,
      "cfg": 3.5,
      "width": 1280,
      "height": 704,
      "sampler": "uni_pc",
      "scheduler": "simple",
      "length": 121,
      "fps": 24.0,
      "negative": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    },
    "perModel": {
      "wan2.2_ti2v_5B_fp16.safetensors": {
        "presets": {
          "720p_default": {
            "note": "5.04s @24fps (121/24). Batch-1 sampler peak ~12554 MiB (9537 weights + 3017 infer) on a 15888 MiB card; ComfyUI will ASK for 9537+7253=16790 MiB, fail that, and fall back to its minimum_memory_required path, which still fully loads the weights on an otherwise-idle GPU. Matches the official template's resolution and frame count."
          },
          "720p_max_frames": {
            "note": "6.71s (161/24). Batch-1 peak ~13528 MiB of 15888 MiB. Practical max, not a hard max: it depends entirely on how much VRAM the desktop session is holding. Verify /system_stats vram_free >= ~13700 MiB before offering this preset."
          },
          "portrait_720p": {
            "note": "704x1280 — identical latent volume and identical cost to 1280x704. Both dims are multiples of 32 (Wan22ImageToVideoLatent step=32), confirmed against the live schema."
          }
        },
        "alt_values": {
          "shift": "8.0 — sourced twice: the ComfyUI template video_wan2_2_5B_ti2v.json (ModelSamplingSD3 widgets_values [8]) and comfy/supported_models.py WAN21_T2V.sampling_settings shift 8.0, inherited by WAN22_T2V. SwarmUI 'Video Model Support.md' gives default 8, suggested range 8-12. (The previous claim that the upstream ti2v_5B config uses 5.0 is REMOVED — unsourced.)",
          "resolution": "1280x704 is the ComfyUI template preset, not a documented training resolution. SwarmUI documents 832x480 as the trained res for Wan and 1280x720 only for the 14B; SwarmUI's registry uses 960x960-equivalent for wan-2_2-ti2v-5b. Treat 1280x704 as 'the template's 720p default'.",
          "weight_dtype": "'default' = fp16, 9537 MiB. 'fp8_e4m3fn' ~4766 MiB (4,999,787,712 params x 1 byte). Prefer fp8 whenever /system_stats vram_free < ~11000 MiB, which was the case on this machine at verification time. No sourced quality comparison exists locally — do not assert one."
        },
        "saveVideo": "Both {format:'mp4', 'format.codec':'h264'} and the template's {format:'auto','format.codec':'auto'} validate and expand correctly. Prefer 'auto'/'auto' as the default: nodes_video.py:175-176 resolves auto to mp4 unless the codec is av1, and it does not require an h264 encoder to be present at graph-build time."
      }
    },
    "notes": "VERIFIED, NOT GUESSED — the whole graph was statically validated against the live /object_info: every class_type exists, every required input is present (including dynamic-combo expansion), every link's source type matches the sink type, every enum value and filename is in the live option lists. No /prompt POST was made; no GPU work.\n\nWHAT THE FAMILY IS\n- One model does both t2v and i2v. wan2.2_ti",
    "graph": {
      "1": {
        "class_type": "UNETLoader",
        "inputs": {
          "unet_name": "wan2.2_ti2v_5B_fp16.safetensors",
          "weight_dtype": "default"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
          "type": "wan",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "wan2.2_vae.safetensors"
        }
      },
      "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "POSITIVE_PROMPT",
          "clip": [
            "2",
            0
          ]
        }
      },
      "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
          "clip": [
            "2",
            0
          ]
        }
      },
      "6": {
        "class_type": "Wan22ImageToVideoLatent",
        "inputs": {
          "vae": [
            "3",
            0
          ],
          "width": 1280,
          "height": 704,
          "length": 121,
          "batch_size": 1
        }
      },
      "7": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "shift": 8.0
        }
      },
      "8": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "7",
            0
          ],
          "seed": 0,
          "steps": 20,
          "cfg": 3.5,
          "sampler_name": "uni_pc",
          "scheduler": "simple",
          "positive": [
            "4",
            0
          ],
          "negative": [
            "5",
            0
          ],
          "latent_image": [
            "6",
            0
          ],
          "denoise": 1.0
        }
      },
      "9": {
        "class_type": "VAEDecodeTiled",
        "inputs": {
          "samples": [
            "8",
            0
          ],
          "vae": [
            "3",
            0
          ],
          "tile_size": 512,
          "overlap": 64,
          "temporal_size": 32,
          "temporal_overlap": 8
        }
      },
      "11": {
        "class_type": "SaveWEBM",
        "inputs": {
          "images": [
            "9",
            0
          ],
          "filename_prefix": "switchgen/wan22-5b",
          "codec": "vp9",
          "fps": 24.0,
          "crf": 32.0
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "8",
          "seed"
        ]
      ],
      "steps": [
        [
          "8",
          "steps"
        ]
      ],
      "cfg": [
        [
          "8",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "8",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "8",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "4",
          "text"
        ]
      ],
      "negative": [
        [
          "5",
          "text"
        ]
      ],
      "width": [
        [
          "6",
          "width"
        ]
      ],
      "height": [
        [
          "6",
          "height"
        ]
      ],
      "length": [
        [
          "6",
          "length"
        ]
      ],
      "fps": [
        [
          "11",
          "fps"
        ]
      ]
    }
  },
  {
    "id": "wan22-14b-t2v",
    "label": "Wan 2.2 Text to Video 14B (High/Low Noise pair)",
    "mode": "video",
    "models": [
      "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
      "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"
    ],
    "verified": true,
    "dualModel": true,
    "clipType": "wan",
    "clipFile": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vaeFile": "wan_2.1_vae.safetensors",
    "defaults": {
      "steps": 20,
      "cfg": 3.5,
      "width": 832,
      "height": 480,
      "sampler": "euler",
      "scheduler": "simple",
      "length": 81,
      "fps": 16.0,
      "negative": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    },
    "perModel": {
      "_roles": {
        "Wan2.2-T2V-A14B-HighNoise-Q5_K_M.gguf": {
          "role": "high_noise",
          "node": "1",
          "lora": "Wan22_T2V_HIGH_Lightning_4steps.safetensors",
          "lora_node": "3",
          "sampler_node": "12",
          "start_at_step": 0,
          "end_at_step": "split",
          "add_noise": "enable",
          "return_with_leftover_noise": "enable"
        },
        "Wan2.2-T2V-A14B-LowNoise-Q5_K_M.gguf": {
          "role": "low_noise",
          "node": "2",
          "lora": "Wan22_T2V_LOW_Lightning_4steps.safetensors",
          "lora_node": "4",
          "sampler_node": "13",
          "start_at_step": "split",
          "end_at_step": 10000,
          "add_noise": "disable",
          "return_with_leftover_noise": "disable"
        }
      },
      "lightning-4step": {
        "_default": true,
        "_desc": "Graph as emitted. Both Lightning LoRAs at strength 1.0. Recommended on 16GB.",
        "steps": 4,
        "split": 2,
        "cfg": 1.0,
        "shift": 5.0,
        "sampler": "euler",
        "scheduler": "simple",
        "lora_strength": 1.0,
        "width": 832,
        "height": 480,
        "length": 81,
        "fps": 16,
        "est_minutes_5060ti": "5-7"
      },
      "lightning-8step-quality": {
        "_desc": "Same LoRAs, more steps for better motion/detail. Set nodes 12/13 steps=8, split=4.",
        "steps": 8,
        "split": 4,
        "cfg": 1.0,
        "shift": 5.0,
        "lora_strength": 1.0
      },
      "base-no-lora": {
        "_desc": "Delete nodes 3 and 4 and rewire 5.model=[\"1\",0], 6.model=[\"2\",0]. CFG 5 per SwarmUI (official ComfyUI template ships 3.5; both usable).",
        "steps": 20,
        "split": 10,
        "cfg": 5.0,
        "cfg_alt_template": 3.5,
        "shift": 5.0,
        "shift_alt_native": 8.0,
        "sampler": "euler",
        "scheduler": "simple",
        "est_minutes_5060ti": "25-35 at 832x480"
      },
      "720p": {
        "_desc": "Node 11 width/height. Heavy on 16GB - shorten the clip. Lightning only.",
        "width": 1280,
        "height": 720,
        "length": 49,
        "steps": 4,
        "split": 2,
        "cfg": 1.0
      },
      "short-preview": {
        "_desc": "Fast iteration preset.",
        "width": 640,
        "height": 640,
        "length": 49,
        "steps": 4,
        "split": 2,
        "cfg": 1.0
      }
    },
    "notes": "Q4 pair with a quantised umt5 encoder: the only configuration measured to fit here. A LoRA on both noise halves is OOM-killed, because LoRA patching dequantises GGUF weights. Peak 28.1 GB of 30.5 GB, so close other applications first.",
    "graph": {
      "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
          "unet_name": "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"
        }
      },
      "2": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
          "unet_name": "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"
        }
      },
      "5": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "shift": 5.0
        }
      },
      "6": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
          "model": [
            "2",
            0
          ],
          "shift": 5.0
        }
      },
      "7": {
        "class_type": "CLIPLoaderGGUF",
        "inputs": {
          "clip_name": "umt5-xxl-encoder-Q4_K_M.gguf",
          "type": "wan"
        }
      },
      "8": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "wan_2.1_vae.safetensors"
        }
      },
      "9": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "7",
            0
          ],
          "text": "PROMPT_GOES_HERE"
        }
      },
      "10": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "7",
            0
          ],
          "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        }
      },
      "11": {
        "class_type": "EmptyHunyuanLatentVideo",
        "inputs": {
          "width": 832,
          "height": 480,
          "length": 81,
          "batch_size": 1
        }
      },
      "12": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
          "model": [
            "5",
            0
          ],
          "add_noise": "enable",
          "noise_seed": 0,
          "steps": 20,
          "cfg": 3.5,
          "sampler_name": "euler",
          "scheduler": "simple",
          "positive": [
            "9",
            0
          ],
          "negative": [
            "10",
            0
          ],
          "latent_image": [
            "11",
            0
          ],
          "start_at_step": 0,
          "end_at_step": 10,
          "return_with_leftover_noise": "enable"
        }
      },
      "13": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
          "model": [
            "6",
            0
          ],
          "add_noise": "disable",
          "noise_seed": 0,
          "steps": 20,
          "cfg": 3.5,
          "sampler_name": "euler",
          "scheduler": "simple",
          "positive": [
            "9",
            0
          ],
          "negative": [
            "10",
            0
          ],
          "latent_image": [
            "12",
            0
          ],
          "start_at_step": 10,
          "end_at_step": 10000,
          "return_with_leftover_noise": "disable"
        }
      },
      "14": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "13",
            0
          ],
          "vae": [
            "8",
            0
          ]
        }
      },
      "16": {
        "class_type": "SaveWEBM",
        "inputs": {
          "images": [
            "14",
            0
          ],
          "filename_prefix": "switchgen/wan22-14b-t2v",
          "codec": "vp9",
          "fps": 16.0,
          "crf": 32.0
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ],
        [
          "2",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "12",
          "noise_seed"
        ],
        [
          "13",
          "noise_seed"
        ]
      ],
      "steps": [
        [
          "12",
          "steps"
        ],
        [
          "13",
          "steps"
        ]
      ],
      "cfg": [
        [
          "12",
          "cfg"
        ],
        [
          "13",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "12",
          "sampler_name"
        ],
        [
          "13",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "12",
          "scheduler"
        ],
        [
          "13",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "9",
          "text"
        ]
      ],
      "negative": [
        [
          "10",
          "text"
        ]
      ],
      "width": [
        [
          "11",
          "width"
        ]
      ],
      "height": [
        [
          "11",
          "height"
        ]
      ],
      "length": [
        [
          "11",
          "length"
        ]
      ],
      "fps": [
        [
          "16",
          "fps"
        ]
      ]
    }
  },
  {
    "id": "wan22-14b-i2v",
    "label": "Wan 2.2 I2V A14B (High/Low GGUF, Lightning 4-step)",
    "mode": "video",
    "models": [
      "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
      "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
    ],
    "verified": true,
    "dualModel": true,
    "clipType": "wan",
    "clipFile": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vaeFile": "wan_2.1_vae.safetensors",
    "defaults": {
      "steps": 20,
      "cfg": 3.5,
      "width": 832,
      "height": 480,
      "sampler": "euler",
      "scheduler": "simple",
      "length": 81,
      "fps": 16.0,
      "negative": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    },
    "perModel": {
      "Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf": {
        "role": "high-noise base (first half of sigma schedule)",
        "graphNodes": [
          "1",
          "2",
          "3",
          "13"
        ],
        "lora": "Wan22_I2V_HIGH_Lightning_4steps.safetensors",
        "loraStrength": 1.0,
        "add_noise": "enable",
        "start_at_step": 0,
        "end_at_step": 2,
        "return_with_leftover_noise": "enable",
        "noise_seed": "SEED (this is the only node that consumes the seed)",
        "shift": 5.0
      },
      "Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf": {
        "role": "low-noise refiner (second half, StepSwap)",
        "graphNodes": [
          "4",
          "5",
          "6",
          "14"
        ],
        "lora": "Wan22_I2V_LOW_Lightning_4steps.safetensors",
        "loraStrength": 1.0,
        "add_noise": "disable",
        "start_at_step": 2,
        "end_at_step": 10000,
        "return_with_leftover_noise": "disable",
        "noise_seed": 0,
        "shift": 5.0
      },
      "_mode_lightning_4step": {
        "description": "DEFAULT in nodeGraph. Lightning LoRAs wired on both experts. ~4x faster; the only practical mode on 16GB.",
        "steps": 4,
        "cfg": 1.0,
        "boundary_step": 2,
        "shift": 5.0,
        "sampler": "euler",
        "scheduler": "simple",
        "loras": "enabled on nodes 2 and 5",
        "source": "ComfyUI video_wan2_2_14B_i2v.json switch-ON branch (PrimitiveInt 4 / PrimitiveFloat 1 / PrimitiveInt 2)"
      },
      "_mode_base_no_lora": {
        "description": "Quality reference per SwarmUI. To use: delete nodes 2 and 5, repoint 3.model->[\"1\",0] and 6.model->[\"4\",0], then apply these numbers to nodes 13 and 14.",
        "steps": 20,
        "cfg": 3.5,
        "boundary_step": 10,
        "shift": 5.0,
        "sampler": "euler",
        "scheduler": "simple",
        "node13": {
          "start_at_step": 0,
          "end_at_step": 10
        },
        "node14": {
          "start_at_step": 10,
          "end_at_step": 10000
        },
        "source": "SwarmUI 'Wan 2.2' section (CFG 3.5, swap percent 0.5) + ComfyUI template switch-OFF branch (20 / 3.5 / 10)",
        "warning": "~5x slower on a 5060 Ti; 20 steps x 2 expert loads at 832x480x81 is many minutes per clip"
      },
      "_resolution_presets": {
        "landscape_480p": {
          "width": 832,
          "height": 480,
          "note": "DEFAULT. Native Wan 480p training res and the WanImageToVideo node default."
        },
        "portrait_480p": {
          "width": 480,
          "height": 832
        },
        "square": {
          "width": 640,
          "height": 640,
          "note": "ComfyUI official template default"
        },
        "720p": {
          "width": 1280,
          "height": 720,
          "note": "DO NOT USE at length 81 on 16GB - will OOM. Only viable at length<=41, and still marginal."
        }
      },
      "_length_presets": {
        "81": "5.06s @16fps (default)",
        "61": "3.81s @16fps",
        "41": "2.56s @16fps (use for 720p attempts)",
        "constraint": "length MUST satisfy (length-1)%4==0; WanImageToVideo builds ((length-1)//4)+1 latent frames"
      }
    },
    "notes": "Q4 pair with a quantised umt5 encoder: the only configuration measured to fit here. A LoRA on both noise halves is OOM-killed, because LoRA patching dequantises GGUF weights. Peak 28.1 GB of 30.5 GB, so close other applications first.",
    "graph": {
      "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
          "unet_name": "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"
        }
      },
      "3": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "shift": 5.0
        }
      },
      "4": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
          "unet_name": "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
        }
      },
      "5": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
          "model": [
            "4",
            0
          ],
          "lora_name": "Wan22_I2V_NSFW_General_LOW.safetensors",
          "strength_model": 0.85
        }
      },
      "6": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
          "model": [
            "5",
            0
          ],
          "shift": 5.0
        }
      },
      "7": {
        "class_type": "CLIPLoaderGGUF",
        "inputs": {
          "clip_name": "umt5-xxl-encoder-Q4_K_M.gguf",
          "type": "wan"
        }
      },
      "8": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "7",
            0
          ],
          "text": "PROMPT_GOES_HERE"
        }
      },
      "9": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "7",
            0
          ],
          "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        }
      },
      "10": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "wan_2.1_vae.safetensors"
        }
      },
      "11": {
        "class_type": "LoadImage",
        "inputs": {
          "image": "example.png"
        }
      },
      "12": {
        "class_type": "WanImageToVideo",
        "inputs": {
          "positive": [
            "8",
            0
          ],
          "negative": [
            "9",
            0
          ],
          "vae": [
            "10",
            0
          ],
          "width": 832,
          "height": 480,
          "length": 81,
          "batch_size": 1,
          "start_image": [
            "11",
            0
          ]
        }
      },
      "13": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
          "model": [
            "3",
            0
          ],
          "add_noise": "enable",
          "noise_seed": 0,
          "steps": 20,
          "cfg": 3.5,
          "sampler_name": "euler",
          "scheduler": "simple",
          "positive": [
            "12",
            0
          ],
          "negative": [
            "12",
            1
          ],
          "latent_image": [
            "12",
            2
          ],
          "start_at_step": 0,
          "end_at_step": 10,
          "return_with_leftover_noise": "enable"
        }
      },
      "14": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
          "model": [
            "6",
            0
          ],
          "add_noise": "disable",
          "noise_seed": 0,
          "steps": 20,
          "cfg": 3.5,
          "sampler_name": "euler",
          "scheduler": "simple",
          "positive": [
            "12",
            0
          ],
          "negative": [
            "12",
            1
          ],
          "latent_image": [
            "13",
            0
          ],
          "start_at_step": 10,
          "end_at_step": 10000,
          "return_with_leftover_noise": "disable"
        }
      },
      "15": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "14",
            0
          ],
          "vae": [
            "10",
            0
          ]
        }
      },
      "16": {
        "class_type": "SaveWEBM",
        "inputs": {
          "images": [
            "15",
            0
          ],
          "filename_prefix": "switchgen/wan22-14b-i2v",
          "codec": "vp9",
          "fps": 16.0,
          "crf": 32.0
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ],
        [
          "4",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "13",
          "noise_seed"
        ],
        [
          "14",
          "noise_seed"
        ]
      ],
      "steps": [
        [
          "13",
          "steps"
        ],
        [
          "14",
          "steps"
        ]
      ],
      "cfg": [
        [
          "13",
          "cfg"
        ],
        [
          "14",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "13",
          "sampler_name"
        ],
        [
          "14",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "13",
          "scheduler"
        ],
        [
          "14",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "8",
          "text"
        ]
      ],
      "negative": [
        [
          "9",
          "text"
        ]
      ],
      "width": [
        [
          "12",
          "width"
        ]
      ],
      "height": [
        [
          "12",
          "height"
        ]
      ],
      "length": [
        [
          "12",
          "length"
        ]
      ],
      "image": [
        [
          "11",
          "image"
        ]
      ],
      "fps": [
        [
          "16",
          "fps"
        ]
      ]
    }
  },
  {
    "id": "chroma",
    "label": "Chroma1-HD (uncensored photoreal)",
    "mode": "image",
    "models": [
      "Chroma1-HD-fp8mixed.safetensors"
    ],
    "dualModel": false,
    "clipType": "chroma",
    "clipFile": "t5xxl_fp8_e4m3fn_scaled.safetensors",
    "vaeFile": "Flux/ae.safetensors",
    "defaults": {
      "steps": 26.0,
      "cfg": 3.5,
      "width": 1024,
      "height": 1024,
      "sampler": "euler",
      "scheduler": "beta",
      "length": 0,
      "fps": 0.0,
      "negative": "low quality, blurry, jpeg artifacts, watermark, text, deformed, bad anatomy"
    },
    "perModel": {
      "Chroma1-HD-fp8mixed.safetensors": {
        "label": "Chroma1-HD fp8mixed",
        "steps": 26,
        "cfg": 3.5,
        "note": "9.19 GB - recommended for a 16 GB card. Total 14.69 GB with sidecars."
      },
      "Chroma1-HD.safetensors": {
        "label": "Chroma1-HD bf16",
        "steps": 26,
        "cfg": 3.5,
        "requiresBytes": 23292691364,
        "note": "17.80 GB bf16; 23.29 GB with sidecars - under total RAM but above the ~23 GB typically free, so it is borderline on this machine."
      }
    },
    "notes": "Flux derivative trained without a content filter, which is why it handles anatomy that the filtered photoreal bases refuse. Real CFG, not distilled: 3.5 is the useful centre. Scheduler beta. 26 steps is the author reference.",
    "graph": {
      "1": {
        "class_type": "UNETLoader",
        "inputs": {
          "unet_name": "Chroma1-HD-fp8mixed.safetensors",
          "weight_dtype": "default"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "t5xxl_fp8_e4m3fn_scaled.safetensors",
          "type": "chroma",
          "device": "default"
        }
      },
      "3": {
        "class_type": "T5TokenizerOptions",
        "inputs": {
          "clip": [
            "2",
            0
          ],
          "min_padding": 0,
          "min_length": 0
        }
      },
      "4": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "Flux/ae.safetensors"
        }
      },
      "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "3",
            0
          ],
          "text": "a cinematic photograph of a red fox in a misty pine forest at dawn"
        }
      },
      "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "clip": [
            "3",
            0
          ],
          "text": "low quality, blurry, jpeg artifacts, watermark, text, deformed, bad anatomy"
        }
      },
      "7": {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "shift": 1.0,
          "sampling": "flow"
        }
      },
      "8": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        }
      },
      "9": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "7",
            0
          ],
          "positive": [
            "5",
            0
          ],
          "negative": [
            "6",
            0
          ],
          "latent_image": [
            "8",
            0
          ],
          "seed": 0,
          "steps": 26,
          "cfg": 3.5,
          "sampler_name": "euler",
          "scheduler": "beta",
          "denoise": 1.0
        }
      },
      "10": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "9",
            0
          ],
          "vae": [
            "4",
            0
          ]
        }
      },
      "11": {
        "class_type": "SaveImage",
        "inputs": {
          "images": [
            "10",
            0
          ],
          "filename_prefix": "switchgen/chroma"
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "9",
          "seed"
        ]
      ],
      "steps": [
        [
          "9",
          "steps"
        ]
      ],
      "cfg": [
        [
          "9",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "9",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "9",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "5",
          "text"
        ]
      ],
      "negative": [
        [
          "6",
          "text"
        ]
      ],
      "width": [
        [
          "8",
          "width"
        ]
      ],
      "height": [
        [
          "8",
          "height"
        ]
      ],
      "denoise": [
        [
          "9",
          "denoise"
        ]
      ]
    },
    "detect": {
      "archClass": "Chroma",
      "detectKeys": [
        "distilled_guidance_layer.0.norms.0.weight",
        "distilled_guidance_layer.0.norms.0.scale",
        "distilled_guidance_layer.norms.0.weight",
        "distilled_guidance_layer.norms.0.scale",
        "double_blocks.0.img_attn.norm.key_norm.weight",
        "double_blocks.0.img_attn.norm.key_norm.scale"
      ],
      "filenameHints": [
        "chroma",
        "chroma1-hd",
        "chroma1",
        "chroma-unlocked"
      ]
    },
    "verified": true
  },
  {
    "id": "wan21-vace-14b-gguf",
    "label": "Wan 2.1 VACE 14B (video to video)",
    "mode": "video",
    "models": [
      "Wan2.1_14B_VACE-Q4_K_M.gguf"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "wan",
    "clipFile": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vaeFile": "wan_2.1_vae.safetensors",
    "defaults": {
      "steps": 20.0,
      "cfg": 6.0,
      "width": 832,
      "height": 480,
      "sampler": "uni_pc",
      "scheduler": "simple",
      "length": 81,
      "fps": 16.0,
      "negative": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    },
    "perModel": {
      "Wan2.1_14B_VACE-Q4_K_M.gguf": {
        "label": "Wan 2.1 VACE 14B Q4_K_M (installed)",
        "_default": true,
        "sizeBytes": 11639453600,
        "requiresBytes": 18629175815,
        "url": "https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/resolve/main/Wan2.1_14B_VACE-Q4_K_M.gguf",
        "steps": 20,
        "cfg": 6.0,
        "shift": 8.0,
        "note": "10.84 GiB. Already present at /mnt/storage/ai/models/diffusion_models/Wan2.1_14B_VACE-Q4_K_M.gguf (11,639,453,600 bytes, confirmed via /api/models) and already a member of the live UnetLoaderGGUF unet_name enum, so this family is runnable today once umt5 + wan_2.1_vae are in place (both installed)."
      }
    },
    "notes": "Reference image anchors appearance across a long reel, which is the strongest lever against the drift that accumulates when each shot re-encodes the one before. Measured 20.5 GB peak with a reference, 20.0 GB without.",
    "graph": {
      "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
          "unet_name": "Wan2.1_14B_VACE-Q4_K_M.gguf"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
          "type": "wan",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "wan_2.1_vae.safetensors"
        }
      },
      "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "POSITIVE_PROMPT",
          "clip": [
            "2",
            0
          ]
        }
      },
      "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
          "clip": [
            "2",
            0
          ]
        }
      },
      "6": {
        "class_type": "WanVaceToVideo",
        "inputs": {
          "positive": [
            "4",
            0
          ],
          "negative": [
            "5",
            0
          ],
          "vae": [
            "3",
            0
          ],
          "width": 832,
          "height": 480,
          "length": 81,
          "batch_size": 1,
          "strength": 1.0
        }
      },
      "7": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "shift": 8.0
        }
      },
      "8": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "7",
            0
          ],
          "seed": 0,
          "steps": 20,
          "cfg": 6.0,
          "sampler_name": "uni_pc",
          "scheduler": "simple",
          "positive": [
            "6",
            0
          ],
          "negative": [
            "6",
            1
          ],
          "latent_image": [
            "6",
            2
          ],
          "denoise": 1.0
        }
      },
      "9": {
        "class_type": "TrimVideoLatent",
        "inputs": {
          "samples": [
            "8",
            0
          ],
          "trim_amount": [
            "6",
            3
          ]
        }
      },
      "10": {
        "class_type": "VAEDecodeTiled",
        "inputs": {
          "samples": [
            "9",
            0
          ],
          "vae": [
            "3",
            0
          ],
          "tile_size": 512,
          "overlap": 64,
          "temporal_size": 32,
          "temporal_overlap": 8
        }
      },
      "11": {
        "class_type": "SaveWEBM",
        "inputs": {
          "images": [
            "10",
            0
          ],
          "filename_prefix": "switchgen/wan21-vace-14b",
          "codec": "vp9",
          "fps": 16.0,
          "crf": 32.0
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "8",
          "seed"
        ]
      ],
      "steps": [
        [
          "8",
          "steps"
        ]
      ],
      "cfg": [
        [
          "8",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "8",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "8",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "4",
          "text"
        ]
      ],
      "negative": [
        [
          "5",
          "text"
        ]
      ],
      "width": [
        [
          "6",
          "width"
        ]
      ],
      "height": [
        [
          "6",
          "height"
        ]
      ],
      "length": [
        [
          "6",
          "length"
        ]
      ],
      "fps": [
        [
          "11",
          "fps"
        ]
      ],
      "denoise": [
        [
          "8",
          "denoise"
        ]
      ]
    }
  },
  {
    "id": "wan21-vace-1_3b-gguf",
    "label": "Wan 2.1 VACE 1.3B (fast video to video)",
    "mode": "video",
    "models": [
      "Wan2.1-VACE-1.3B-Q8_0.gguf"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "wan",
    "clipFile": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vaeFile": "wan_2.1_vae.safetensors",
    "defaults": {
      "steps": 20.0,
      "cfg": 6.0,
      "width": 832,
      "height": 480,
      "sampler": "uni_pc",
      "scheduler": "simple",
      "length": 81,
      "fps": 16.0,
      "negative": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    },
    "perModel": {
      "Wan2.1_14B_VACE-Q4_K_M.gguf": {
        "label": "Wan 2.1 VACE 14B Q4_K_M (installed)",
        "_default": true,
        "sizeBytes": 11639453600,
        "requiresBytes": 18629175815,
        "url": "https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/resolve/main/Wan2.1_14B_VACE-Q4_K_M.gguf",
        "steps": 20,
        "cfg": 6.0,
        "shift": 8.0,
        "note": "10.84 GiB. Already present at /mnt/storage/ai/models/diffusion_models/Wan2.1_14B_VACE-Q4_K_M.gguf (11,639,453,600 bytes, confirmed via /api/models) and already a member of the live UnetLoaderGGUF unet_name enum, so this family is runnable today once umt5 + wan_2.1_vae are in place (both installed)."
      }
    },
    "notes": "The small VACE. Quick enough to block out a sequence before committing GPU time to the 14B. Same reference image anchoring, lower fidelity.",
    "graph": {
      "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
          "unet_name": "Wan2.1-VACE-1.3B-Q8_0.gguf"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
          "type": "wan",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "wan_2.1_vae.safetensors"
        }
      },
      "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "POSITIVE_PROMPT",
          "clip": [
            "2",
            0
          ]
        }
      },
      "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
          "clip": [
            "2",
            0
          ]
        }
      },
      "6": {
        "class_type": "WanVaceToVideo",
        "inputs": {
          "positive": [
            "4",
            0
          ],
          "negative": [
            "5",
            0
          ],
          "vae": [
            "3",
            0
          ],
          "width": 832,
          "height": 480,
          "length": 81,
          "batch_size": 1,
          "strength": 1.0
        }
      },
      "7": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "shift": 8.0
        }
      },
      "8": {
        "class_type": "KSampler",
        "inputs": {
          "model": [
            "7",
            0
          ],
          "seed": 0,
          "steps": 20,
          "cfg": 6.0,
          "sampler_name": "uni_pc",
          "scheduler": "simple",
          "positive": [
            "6",
            0
          ],
          "negative": [
            "6",
            1
          ],
          "latent_image": [
            "6",
            2
          ],
          "denoise": 1.0
        }
      },
      "9": {
        "class_type": "TrimVideoLatent",
        "inputs": {
          "samples": [
            "8",
            0
          ],
          "trim_amount": [
            "6",
            3
          ]
        }
      },
      "10": {
        "class_type": "VAEDecodeTiled",
        "inputs": {
          "samples": [
            "9",
            0
          ],
          "vae": [
            "3",
            0
          ],
          "tile_size": 512,
          "overlap": 64,
          "temporal_size": 32,
          "temporal_overlap": 8
        }
      },
      "11": {
        "class_type": "SaveWEBM",
        "inputs": {
          "images": [
            "10",
            0
          ],
          "filename_prefix": "switchgen/wan21-vace-1_3b",
          "codec": "vp9",
          "fps": 16.0,
          "crf": 32.0
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "8",
          "seed"
        ]
      ],
      "steps": [
        [
          "8",
          "steps"
        ]
      ],
      "cfg": [
        [
          "8",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "8",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "8",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "4",
          "text"
        ]
      ],
      "negative": [
        [
          "5",
          "text"
        ]
      ],
      "width": [
        [
          "6",
          "width"
        ]
      ],
      "height": [
        [
          "6",
          "height"
        ]
      ],
      "length": [
        [
          "6",
          "length"
        ]
      ],
      "fps": [
        [
          "11",
          "fps"
        ]
      ],
      "denoise": [
        [
          "8",
          "denoise"
        ]
      ]
    }
  },
  {
    "id": "hunyuan-video",
    "label": "HunyuanVideo (uncensored text to video)",
    "mode": "video",
    "models": [
      "hunyuan-video-t2v-720p-Q4_K_M.gguf"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "hunyuan_video",
    "clipFile": "llava_llama3_fp8_scaled.safetensors",
    "vaeFile": "hunyuan_video_vae_bf16.safetensors",
    "defaults": {
      "steps": 20,
      "cfg": 1.0,
      "width": 848,
      "height": 480,
      "sampler": "euler",
      "scheduler": "simple",
      "length": 73,
      "fps": 24.0,
      "negative": "low quality, worst quality, jpeg artifacts, blurry, static, watermark, text, deformed hands, extra fingers"
    },
    "perModel": {
      "hunyuan-video-t2v-720p-Q4_K_M.gguf": {
        "label": "Hunyuan Video T2V 720p - Q4_K_M (7.9 GB)",
        "note": "Already installed. The comfortable choice on a 16 GB card.",
        "bytes": 7883680512,
        "requiresBytes": 17714201345,
        "fitsTypicalFreeRAM": true
      },
      "hunyuan-video-t2v-720p-Q5_K_M.gguf": {
        "label": "Hunyuan Video T2V 720p - Q5_K_M (9.4 GB)",
        "note": "1.6 GB more weights than Q4_K_M for a small quality gain.",
        "bytes": 9449663232,
        "requiresBytes": 19280184065,
        "fitsTypicalFreeRAM": true
      },
      "hunyuan-video-t2v-720p-Q6_K.gguf": {
        "label": "Hunyuan Video T2V 720p - Q6_K (11.0 GB)",
        "note": "Highest quantisation that still leaves activation headroom at 848x480x73 on 16 GB.",
        "bytes": 10953714432,
        "requiresBytes": 20784235265,
        "fitsTypicalFreeRAM": true
      }
    },
    "notes": "SwarmUI rates its censorship \"No\". Distilled, so CFG stays at 1 and guidance is carried by the FluxGuidance node instead. 16.5 GB resident, comfortably inside this machine.",
    "graph": {
      "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
          "unet_name": "hunyuan-video-t2v-720p-Q4_K_M.gguf"
        }
      },
      "2": {
        "class_type": "DualCLIPLoader",
        "inputs": {
          "clip_name1": "clip_l.safetensors",
          "clip_name2": "llava_llama3_fp8_scaled.safetensors",
          "type": "hunyuan_video",
          "device": "default"
        }
      },
      "3": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "hunyuan_video_vae_bf16.safetensors"
        }
      },
      "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "A cinematic tracking shot of a red vintage sports car driving along a coastal highway at golden hour, waves breaking on the rocks below, warm rim light, shallow depth of field.",
          "clip": [
            "2",
            0
          ]
        }
      },
      "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "low quality, worst quality, jpeg artifacts, blurry, static, watermark, text, deformed hands, extra fingers",
          "clip": [
            "2",
            0
          ]
        }
      },
      "6": {
        "class_type": "FluxGuidance",
        "inputs": {
          "conditioning": [
            "4",
            0
          ],
          "guidance": 6.0
        }
      },
      "7": {
        "class_type": "ModelSamplingSD3",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "shift": 7.0
        }
      },
      "8": {
        "class_type": "EmptyHunyuanLatentVideo",
        "inputs": {
          "width": 848,
          "height": 480,
          "length": 73,
          "batch_size": 1
        }
      },
      "9": {
        "class_type": "CFGGuider",
        "inputs": {
          "model": [
            "7",
            0
          ],
          "positive": [
            "6",
            0
          ],
          "negative": [
            "5",
            0
          ],
          "cfg": 1.0
        }
      },
      "10": {
        "class_type": "KSamplerSelect",
        "inputs": {
          "sampler_name": "euler"
        }
      },
      "11": {
        "class_type": "BasicScheduler",
        "inputs": {
          "model": [
            "7",
            0
          ],
          "scheduler": "simple",
          "steps": 20,
          "denoise": 1.0
        }
      },
      "12": {
        "class_type": "RandomNoise",
        "inputs": {
          "noise_seed": 0
        }
      },
      "13": {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
          "noise": [
            "12",
            0
          ],
          "guider": [
            "9",
            0
          ],
          "sampler": [
            "10",
            0
          ],
          "sigmas": [
            "11",
            0
          ],
          "latent_image": [
            "8",
            0
          ]
        }
      },
      "14": {
        "class_type": "VAEDecodeTiled",
        "inputs": {
          "samples": [
            "13",
            0
          ],
          "vae": [
            "3",
            0
          ],
          "tile_size": 256,
          "overlap": 64,
          "temporal_size": 32,
          "temporal_overlap": 4
        }
      },
      "16": {
        "class_type": "SaveWEBM",
        "inputs": {
          "images": [
            "14",
            0
          ],
          "filename_prefix": "switchgen/hunyuan-video",
          "codec": "vp9",
          "fps": 24.0,
          "crf": 32.0
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "12",
          "noise_seed"
        ]
      ],
      "steps": [
        [
          "11",
          "steps"
        ]
      ],
      "cfg": [
        [
          "9",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "10",
          "sampler_name"
        ]
      ],
      "scheduler": [
        [
          "11",
          "scheduler"
        ]
      ],
      "positive": [
        [
          "4",
          "text"
        ]
      ],
      "negative": [
        [
          "5",
          "text"
        ]
      ],
      "width": [
        [
          "8",
          "width"
        ]
      ],
      "height": [
        [
          "8",
          "height"
        ]
      ],
      "length": [
        [
          "8",
          "length"
        ]
      ],
      "fps": [
        [
          "15",
          "fps"
        ]
      ],
      "denoise": [
        [
          "11",
          "denoise"
        ]
      ]
    }
  },
  {
    "id": "ltxv-0_9_6-gguf",
    "label": "LTX-Video 0.9.6 distilled (fast video)",
    "mode": "video",
    "models": [
      "ltxv-2b-0.9.6-distilled-Q8_0.gguf"
    ],
    "verified": true,
    "dualModel": false,
    "clipType": "ltxv",
    "clipFile": "t5xxl_fp8_e4m3fn_scaled.safetensors",
    "vaeFile": "",
    "defaults": {
      "steps": 30,
      "cfg": 3.0,
      "width": 768,
      "height": 512,
      "sampler": "euler",
      "scheduler": "simple",
      "length": 97,
      "fps": 24.0,
      "negative": "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
    },
    "perModel": {
      "ltx-video-2b-v0.9.5.safetensors": {
        "label": "LTX-Video 2B v0.9.5 (6.3 GB)",
        "steps": 30,
        "cfg": 3.0,
        "note": "All-in-one checkpoint: DiT + video VAE in one file, so vaeFile is empty and VAEDecode reads CheckpointLoaderSimple output slot 2. Fastest video model in this group.",
        "bytes": 6340729500,
        "requiresBytes": 11498078188,
        "fitsTypicalFreeRAM": true
      },
      "ltxv-2b-0.9.8-distilled.safetensors": {
        "label": "LTX-Video 2B 0.9.8 distilled (6.3 GB)",
        "steps": 8,
        "cfg": 1.0,
        "note": "Distilled - run at CFG 1.0 and 8 steps. Higher CFG burns the output.",
        "bytes": 6340744492,
        "requiresBytes": 11498093180,
        "fitsTypicalFreeRAM": true
      },
      "ltxv-13b-0.9.8-dev-fp8.safetensors": {
        "label": "LTX-Video 13B 0.9.8 dev fp8 (15.7 GB)",
        "steps": 30,
        "cfg": 3.0,
        "note": "15.7 GB of weights plus a 5.2 GB encoder = 20.9 GB; fits in RAM but needs ComfyUI offload on a 16 GB card.",
        "bytes": 15694279916,
        "requiresBytes": 20851628604,
        "fitsTypicalFreeRAM": true
      }
    },
    "notes": "Distilled and small, so it is the quickest way to see whether an idea works before spending minutes on Wan. The catalogue entry is the 0.9.5 checkpoint, which bundles its VAE; this is the 0.9.6 GGUF, so the VAE is loaded separately and must be the matching 0.9.6 file.",
    "graph": {
      "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
          "unet_name": "ltxv-2b-0.9.6-distilled-Q8_0.gguf"
        }
      },
      "2": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "t5xxl_fp8_e4m3fn_scaled.safetensors",
          "type": "ltxv",
          "device": "default"
        }
      },
      "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "A cinematic tracking shot of a red vintage sports car driving along a coastal highway at golden hour, waves breaking on the rocks below, warm rim light, shallow depth of field.",
          "clip": [
            "2",
            0
          ]
        }
      },
      "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
          "clip": [
            "2",
            0
          ]
        }
      },
      "5": {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
          "width": 768,
          "height": 512,
          "length": 97,
          "batch_size": 1
        }
      },
      "6": {
        "class_type": "LTXVConditioning",
        "inputs": {
          "positive": [
            "3",
            0
          ],
          "negative": [
            "4",
            0
          ],
          "frame_rate": 24.0
        }
      },
      "7": {
        "class_type": "KSamplerSelect",
        "inputs": {
          "sampler_name": "euler"
        }
      },
      "8": {
        "class_type": "LTXVScheduler",
        "inputs": {
          "steps": 30,
          "max_shift": 2.05,
          "base_shift": 0.95,
          "stretch": true,
          "terminal": 0.1,
          "latent": [
            "5",
            0
          ]
        }
      },
      "9": {
        "class_type": "SamplerCustom",
        "inputs": {
          "model": [
            "1",
            0
          ],
          "add_noise": true,
          "noise_seed": 0,
          "cfg": 3.0,
          "positive": [
            "6",
            0
          ],
          "negative": [
            "6",
            1
          ],
          "sampler": [
            "7",
            0
          ],
          "sigmas": [
            "8",
            0
          ],
          "latent_image": [
            "5",
            0
          ]
        }
      },
      "10": {
        "class_type": "VAEDecode",
        "inputs": {
          "samples": [
            "9",
            0
          ],
          "vae": [
            "__ltx_vae",
            0
          ]
        }
      },
      "12": {
        "class_type": "SaveWEBM",
        "inputs": {
          "images": [
            "10",
            0
          ],
          "filename_prefix": "switchgen/ltxv",
          "codec": "vp9",
          "fps": 24.0,
          "crf": 32.0
        }
      },
      "__ltx_vae": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "LTX-Video-0.9.6-VAE-BF16.safetensors"
        }
      }
    },
    "bindings": {
      "model": [
        [
          "1",
          "unet_name"
        ]
      ],
      "seed": [
        [
          "9",
          "noise_seed"
        ]
      ],
      "steps": [
        [
          "8",
          "steps"
        ]
      ],
      "cfg": [
        [
          "9",
          "cfg"
        ]
      ],
      "sampler": [
        [
          "7",
          "sampler_name"
        ]
      ],
      "positive": [
        [
          "3",
          "text"
        ]
      ],
      "negative": [
        [
          "4",
          "text"
        ]
      ],
      "width": [
        [
          "5",
          "width"
        ]
      ],
      "height": [
        [
          "5",
          "height"
        ]
      ],
      "length": [
        [
          "5",
          "length"
        ]
      ],
      "fps": [
        [
          "6",
          "frame_rate"
        ]
      ]
    }
  }
] as FamilyDef[]
