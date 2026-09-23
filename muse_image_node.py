import os
import io
import json
import base64
import torch
import numpy as np
from PIL import Image, ImageOps
import urllib.request
import urllib.error
from pathlib import Path


def _is_valid_key(k: str) -> bool:
    if not k:
        return False
    k = str(k).strip()
    if k.lower() in ("false", "true", "none", "null", "undefined", ""):
        return False
    if "your_meta_model_api_key" in k or "your_api_key" in k:
        return False
    if len(k) < 15:
        return False
    return True


def _get_api_key(override: str = "") -> str:
    if _is_valid_key(override):
        return override.strip()

    cfg_file = Path(__file__).parent / "config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                key = json.load(f).get("MODEL_API_KEY", "").strip()
                if _is_valid_key(key):
                    return key
        except Exception:
            pass

    metabot_env = Path("/home/phaulty/Work/metabot/.env")
    if metabot_env.exists():
        try:
            with open(metabot_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MODEL_API_KEY=") or line.startswith("META_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if _is_valid_key(key):
                            return key
        except Exception:
            pass

    for var in ["MODEL_API_KEY", "META_API_KEY"]:
        env_key = os.getenv(var, "").strip()
        if _is_valid_key(env_key):
            return env_key

    for p in [Path(__file__).parent / ".env", Path(__file__).resolve().parents[2] / ".env"]:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("MODEL_API_KEY=") or line.startswith("META_API_KEY="):
                            key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if _is_valid_key(key):
                                return key
            except Exception:
                pass

    return ""


def _get_base_url() -> str:
    cfg_file = Path(__file__).parent / "config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                base_url = json.load(f).get("MODEL_BASE_URL", "").strip()
                if base_url:
                    return base_url.rstrip("/")
        except Exception:
            pass

    base_url = os.getenv("MODEL_BASE_URL", "").strip()
    return (base_url or "https://api.meta.ai/v1").rstrip("/")


def _clean_size(size_str: str) -> str:
    if not size_str:
        return "auto"
    raw = size_str.split()[0].strip().lower()
    valid_sizes = {"1024x1024", "1024x1536", "1536x1024", "auto"}
    if raw in valid_sizes:
        return raw
    if "x" in raw:
        try:
            w, h = map(int, raw.split("x"))
            if w > h:
                return "1536x1024"
            elif h > w:
                return "1024x1536"
            else:
                return "1024x1024"
        except Exception:
            pass
    return "auto"


def _tensor_to_data_url(image_tensor: torch.Tensor) -> str:
    if len(image_tensor.shape) == 4:
        np_img = 255.0 * image_tensor[0].cpu().numpy()
    else:
        np_img = 255.0 * image_tensor.cpu().numpy()
    pil_img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"


def _extract_images(reference_image: torch.Tensor = None, reference_images: list = None) -> list:
    collected = []
    if reference_image is not None:
        if isinstance(reference_image, torch.Tensor):
            if len(reference_image.shape) == 4:
                for i in range(reference_image.shape[0]):
                    collected.append(reference_image[i : i + 1])
            else:
                collected.append(reference_image)
        elif isinstance(reference_image, (list, tuple)):
            collected.extend(reference_image)

    if reference_images is not None:
        if isinstance(reference_images, (list, tuple)):
            for img in reference_images:
                if isinstance(img, torch.Tensor):
                    if len(img.shape) == 4 and img.shape[0] > 1:
                        for i in range(img.shape[0]):
                            collected.append(img[i : i + 1])
                    else:
                        collected.append(img)
                elif img is not None:
                    collected.append(img)
        elif isinstance(reference_images, torch.Tensor):
            if len(reference_images.shape) == 4:
                for i in range(reference_images.shape[0]):
                    collected.append(reference_images[i : i + 1])
            else:
                collected.append(reference_images)

    return collected


def _bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img)[None,]


def _call_muse_api(api_key: str, payload: dict) -> tuple:
    base_url = _get_base_url()
    endpoint = f"{base_url}/responses"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ComfyUI-MuseImage",
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Muse Image API error ({e.code}): {err_body}")

    new_response_id = resp_data.get("id", "")
    output_items = resp_data.get("output", [])

    img_bytes = None
    reasoning_summary = ""

    for item in output_items:
        item_type = item.get("type")
        if item_type == "image_generation_call":
            b64_result = item.get("result")
            if b64_result:
                img_bytes = base64.b64decode(b64_result)
        elif item_type == "reasoning":
            summaries = item.get("summary", [])
            reasoning_summary = " ".join(
                s.get("text", "") for s in summaries if s.get("type") == "summary_text"
            )

    if img_bytes is None:
        raise RuntimeError("Muse Image API did not return an image.")

    img_tensor = _bytes_to_tensor(img_bytes)
    return img_tensor, new_response_id, reasoning_summary


class MuseImageNode:
    """
    Primary generator node: Generates an initial image from scratch or reference image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a cinematic photo of an astronaut on Mars during sunset, ultra-detailed",
                    },
                ),
                "size": (
                    [
                        "auto",
                        "1024x1024 (Square 1:1)",
                        "1024x1536 (Portrait 2:3)",
                        "1536x1024 (Landscape 3:2)",
                    ],
                    {"default": "auto"},
                ),
                "reasoning_strength": (
                    ["high", "medium", "low"],
                    {"default": "high"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": "randomize",
                    },
                ),
                "model": (["muse-image-1.0"], {"default": "muse-image-1.0"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_images": ("MUSE_IMAGES",),
                "api_key_override": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "response_id", "reasoning_summary")
    FUNCTION = "generate"
    CATEGORY = "phaulty nodes/Muse"

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return seed

    def generate(
        self,
        prompt: str,
        size: str,
        reasoning_strength: str,
        seed: int = 0,
        model: str = "muse-image-1.0",
        reference_image: torch.Tensor = None,
        reference_images: list = None,
        api_key_override: str = "",
    ):
        api_key = _get_api_key(api_key_override)
        if not api_key:
            raise ValueError(
                "MODEL_API_KEY is not set or invalid. Please check ComfyUI/custom_nodes/ComfyUI-MuseImage/config.json."
            )

        api_size = _clean_size(size)

        payload = {
            "model": model,
            "store": True,
            "tools": [
                {
                    "type": "image_generation",
                    "reasoning_strength": reasoning_strength,
                    "size": api_size,
                    "output_format": "png",
                }
            ],
        }

        images = _extract_images(reference_image, reference_images)
        if images:
            content = [{"type": "input_text", "text": prompt}]
            for img in images:
                image_data_url = _tensor_to_data_url(img)
                content.append({"type": "input_image", "image_url": image_data_url})
            payload["input"] = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        else:
            payload["input"] = prompt

        return _call_muse_api(api_key, payload)


_EDITOR_SESSION_STORE = {}


class MuseImageEditorNode:
    """
    Iterative Refiner/Editor node:
    Receives an initial response_id from an upstream node, then auto-accumulates edits
    on every subsequent run so you can progressively refine turn-by-turn.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "add soft smoke rising and vibrant northern lights in the sky",
                    },
                ),
                "reasoning_strength": (
                    ["high", "medium", "low"],
                    {"default": "high"},
                ),
                "mode": (
                    ["accumulate_edits", "reset_to_incoming_id"],
                    {"default": "accumulate_edits"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": "randomize",
                    },
                ),
            },
            "optional": {
                "previous_response_id": ("STRING", {"forceInput": True}),
                "override_response_id": ("STRING", {"default": "", "multiline": False}),
                "reference_image": ("IMAGE",),
                "reference_images": ("MUSE_IMAGES",),
                "model": (["muse-image-1.0"], {"default": "muse-image-1.0"}),
                "api_key_override": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("IMAGE", "response_id", "reasoning_summary", "turn_number")
    FUNCTION = "edit"
    CATEGORY = "phaulty nodes/Muse"

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return seed

    def edit(
        self,
        prompt: str,
        reasoning_strength: str,
        mode: str,
        seed: int = 0,
        previous_response_id: str = None,
        override_response_id: str = "",
        reference_image: torch.Tensor = None,
        reference_images: list = None,
        model: str = "muse-image-1.0",
        api_key_override: str = "",
        unique_id: str = "default_editor",
    ):
        api_key = _get_api_key(api_key_override)
        if not api_key:
            raise ValueError(
                "MODEL_API_KEY is not set or invalid. Please check ComfyUI/custom_nodes/ComfyUI-MuseImage/config.json."
            )

        base_id = (override_response_id.strip() if override_response_id else "") or (previous_response_id.strip() if previous_response_id else "")

        session = _EDITOR_SESSION_STORE.get(unique_id, {})
        last_turn_id = session.get("last_response_id")
        last_incoming_base = session.get("incoming_base_id")
        turn_number = session.get("turn_count", 0)

        if mode == "reset_to_incoming_id" or not last_turn_id or (base_id and base_id != last_incoming_base):
            effective_response_id = base_id
            turn_number = 1
        else:
            effective_response_id = last_turn_id
            turn_number += 1

        if not effective_response_id:
            raise ValueError("No previous response ID provided! Connect an upstream Muse Image node or enter an override_response_id.")

        payload = {
            "model": model,
            "store": True,
            "previous_response_id": effective_response_id,
            "tools": [
                {
                    "type": "image_generation",
                    "reasoning_strength": reasoning_strength,
                    "output_format": "png",
                }
            ],
        }

        images = _extract_images(reference_image, reference_images)
        if images:
            content = [{"type": "input_text", "text": prompt}]
            for img in images:
                image_data_url = _tensor_to_data_url(img)
                content.append({"type": "input_image", "image_url": image_data_url})
            payload["input"] = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        else:
            payload["input"] = prompt

        img_tensor, new_response_id, reasoning = _call_muse_api(api_key, payload)

        _EDITOR_SESSION_STORE[unique_id] = {
            "last_response_id": new_response_id,
            "incoming_base_id": base_id,
            "turn_count": turn_number,
        }

        return (img_tensor, new_response_id, reasoning, turn_number)


class MuseShowTextNode:
    """
    Built-in lightweight text display node for viewing reasoning summary,
    interaction IDs, or prompts directly on the ComfyUI canvas.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "show_text"
    CATEGORY = "phaulty nodes/Muse"

    def show_text(self, text: str = ""):
        display_str = str(text) if text is not None else ""
        return {"ui": {"text": [display_str]}, "result": (display_str,)}


class MuseSwitchNode:
    """
    Switches between initial generation and iterative editor outputs.
    Allows using a single SaveImage and ShowText node, eliminating duplicate saved images,
    and forwards the active response_id for downstream file naming or reference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["generate", "edit"], {"default": "generate"}),
            },
            "optional": {
                "generate_image": ("IMAGE", {"lazy": True}),
                "generate_reasoning": ("STRING", {"lazy": True, "forceInput": True}),
                "edit_image": ("IMAGE", {"lazy": True}),
                "edit_reasoning": ("STRING", {"lazy": True, "forceInput": True}),
                "generate_response_id": ("STRING", {"lazy": True, "forceInput": True}),
                "edit_response_id": ("STRING", {"lazy": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "reasoning_summary", "response_id")
    FUNCTION = "route"
    CATEGORY = "phaulty nodes/Muse"

    def check_lazy_status(self, mode: str, **kwargs):
        prefix = "edit" if mode == "edit" else "generate"
        needed = []
        for key in (f"{prefix}_image", f"{prefix}_reasoning", f"{prefix}_response_id"):
            if key in kwargs:
                needed.append(key)
        return needed

    def route(
        self,
        mode: str,
        generate_image: torch.Tensor = None,
        generate_response_id: str = "",
        generate_reasoning: str = "",
        edit_image: torch.Tensor = None,
        edit_response_id: str = "",
        edit_reasoning: str = "",
    ):
        if mode == "edit":
            img = edit_image if edit_image is not None else generate_image
            reasoning = edit_reasoning if edit_reasoning else generate_reasoning
            resp_id = edit_response_id if edit_response_id else generate_response_id
        else:
            img = generate_image if generate_image is not None else edit_image
            reasoning = generate_reasoning if generate_reasoning else edit_reasoning
            resp_id = generate_response_id if generate_response_id else edit_response_id

        if img is None:
            img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (img, reasoning or "", resp_id or "")


class MuseImageArrayNode:
    """
    Combines multiple reference images of different resolutions or aspect ratios
    into a unified image array/bundle for Meta Muse generation and editing nodes.
    Supports chaining with other Muse Image Array nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_array": ("MUSE_IMAGES",),
            },
        }

    RETURN_TYPES = ("MUSE_IMAGES",)
    RETURN_NAMES = ("image_array",)
    FUNCTION = "collect_images"
    CATEGORY = "phaulty nodes/Muse"

    def collect_images(
        self,
        image_1: torch.Tensor = None,
        image_2: torch.Tensor = None,
        image_3: torch.Tensor = None,
        image_4: torch.Tensor = None,
        image_array: list = None,
    ):
        images = []
        if image_array is not None:
            if isinstance(image_array, (list, tuple)):
                images.extend(image_array)
            elif isinstance(image_array, torch.Tensor):
                images.append(image_array)

        for img in (image_1, image_2, image_3, image_4):
            if img is not None:
                if isinstance(img, torch.Tensor) and len(img.shape) == 4 and img.shape[0] > 1:
                    for i in range(img.shape[0]):
                        images.append(img[i : i + 1])
                else:
                    images.append(img)

        return (images,)


def _call_muse_spark_api(api_key: str, payload: dict) -> tuple:
    base_url = _get_base_url()
    endpoint = f"{base_url}/responses"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ComfyUI-MuseImage",
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Muse Spark API error ({e.code}): {err_body}")

    new_response_id = resp_data.get("id", "")
    output_items = resp_data.get("output", [])

    text_content = ""
    reasoning_summary = ""

    for item in output_items:
        item_type = item.get("type")
        if item_type == "message":
            parts = item.get("content", [])
            text_content = "".join(
                p.get("text", "") for p in parts if p.get("type") == "output_text"
            )
        elif item_type == "reasoning":
            summaries = item.get("summary", [])
            reasoning_summary = "\n".join(
                s.get("text", "") for s in summaries if s.get("type") == "summary_text"
            )

    usage = resp_data.get("usage", {})
    output_tokens = usage.get("output_tokens", 0)
    reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)

    if not reasoning_summary and (output_tokens or reasoning_tokens):
        reasoning_summary = f"Tokens: {output_tokens} output ({reasoning_tokens} reasoning)"

    return text_content, new_response_id, reasoning_summary


def _extract_tag_content(text: str, tag: str) -> str:
    for t in (tag, tag.lower(), tag.upper()):
        start_tag = f"[{t}]"
        end_tag = f"[/{t}]"
        if start_tag in text and end_tag in text:
            return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
        elif start_tag in text:
            after = text.split(start_tag, 1)[1]
            next_tag_idx = after.find("[")
            if next_tag_idx != -1:
                return after[:next_tag_idx].strip()
            return after.strip()
    return ""


def _parse_spark_output(raw_text: str, include_negative: bool) -> tuple:
    expanded_prompt = ""
    negative_prompt = ""
    prompt_g = ""
    prompt_l = ""

    prompt_g = (
        _extract_tag_content(raw_text, "PROMPT_G")
        or _extract_tag_content(raw_text, "CLIP_G")
        or _extract_tag_content(raw_text, "TEXT_G")
    )
    prompt_l = (
        _extract_tag_content(raw_text, "PROMPT_L")
        or _extract_tag_content(raw_text, "CLIP_L")
        or _extract_tag_content(raw_text, "TEXT_L")
    )

    explicit_prompt = _extract_tag_content(raw_text, "PROMPT")

    if explicit_prompt:
        expanded_prompt = explicit_prompt
    elif prompt_g and prompt_l:
        expanded_prompt = f"{prompt_g}\n\n{prompt_l}"
    elif prompt_g:
        expanded_prompt = prompt_g
    elif prompt_l:
        expanded_prompt = prompt_l

    if include_negative:
        negative_prompt = (
            _extract_tag_content(raw_text, "NEGATIVE")
            or _extract_tag_content(raw_text, "NEGATIVE_PROMPT")
        )

    if not expanded_prompt and not prompt_g:
        clean = raw_text.strip()
        if clean.startswith("```") and clean.endswith("```"):
            lines = clean.splitlines()
            if len(lines) >= 2:
                clean = "\n".join(lines[1:-1]).strip()
        expanded_prompt = clean

    if not prompt_g:
        prompt_g = expanded_prompt
    if not prompt_l:
        prompt_l = expanded_prompt

    return expanded_prompt, negative_prompt, prompt_g, prompt_l


class MuseSparkPromptExpander:
    """
    Prompt Expansion node powered by Meta's Muse Spark reasoning model.
    Expands base ideas into high-detail prompts with controllable architecture styles
    (modern natural language vs old-school CLIP tags vs SDXL dual encoders), optional
    negative prompt generation, custom instruction overrides, and seed-based caching.
    """

    PRESET_GUIDES = {
        "photorealistic": (
            "Focus on authentic realism and camera photography: natural skin/surface micro-textures, "
            "lens specifications (e.g. 35mm or 85mm f/1.4), accurate real-world lighting, authentic materials, "
            "depth of field, and lifelike physical detail without artificial gloss."
        ),
        "cinematic": (
            "Focus on cinematic film aesthetics: dramatic anamorphic composition, widescreen framing, "
            "atmospheric haze or volumetric light, rich color grading, directional rim lighting, and emotional mood."
        ),
        "pony_realism": (
            "Focus on Pony SDXL photorealism with strict dual-anchoring: Both text encoders (CLIP-G and CLIP-L) "
            "must begin with 'score_9, score_8_up, score_7_up, source_photo, rating_safe'. "
            "After anchors, address CLIP-G with photographic scene/lighting tags and CLIP-L with Danbooru subject/attire tags, "
            "paired with Pony's low-score reject negative anchors."
        ),
        "digital_art / anime": (
            "Focus on high-end stylized digital art, concept art, or anime illustration: expressive line work, "
            "vibrant color palettes, painterly textures or clean cel shading, dynamic angles, and striking highlights."
        ),
        "general_expansion": (
            "Provide a balanced, versatile enhancement: enrich subject anatomy, setting, atmospheric lighting, "
            "and composition without over-constraining the visual medium."
        ),
        "minimax_h3_fl2va (first frame + audio guide)": (
            "MiniMax H3 FL2VA / I2VA / T2VA director mode: keyframe alignment instruction header, "
            "3-section timeline (integrated_multimodal_description, overall_soundscape, non_diegetic_music), "
            "diegetic dialogue/singing with speaker (S1) and <d>[Language] lyrics</d> brackets, and acoustic feature "
            "description for latent-guided audio without <Audio 1> tags."
        ),
        "minimax_h3_ref2va (multi-reference r2v)": (
            "MiniMax H3 Ref2VA multi-reference mode: full 6-section structure (subject_definitions, summary, "
            "retention_analysis, detailed_description, overall_soundscape, non_diegetic_music) using 1-indexed "
            "<Picture 1>..<Picture 9> and <Audio 1>..<Audio 3> token bindings."
        ),
        "minimax_h3 (video + audio director)": (
            "MiniMax H3 omni-modal text-to-video director mode (alias for minimax_h3_fl2va)."
        ),
        "custom": "Follow the user's custom instructions precisely.",
    }

    FORMAT_GUIDES = {
        "natural_language (modern / flux / muse)": (
            "Format as rich, descriptive natural language prose. Write coherent English sentences "
            "describing subject, actions, clothing, composition, camera angles, lighting conditions, and textures. "
            "Avoid keyword stuffing or comma-separated tag spam."
        ),
        "sdxl (dual clip_g + clip_l)": (
            "Format specifically for SDXL dual text encoders: [PROMPT_G] with rich, descriptive natural language "
            "prose describing scene composition, atmosphere, lighting, and visual narrative for OpenCLIP ViT-bigG, "
            "and [PROMPT_L] with clean comma-separated tokens, subject details, clothing, and quality enhancers for OpenAI CLIP ViT-L."
        ),
        "clip_l_tags (sd1.5 / booru)": (
            "Format as clean, comma-separated tokens, Danbooru/Booru tags, and quality enhancers suitable "
            "for CLIP text encoders (e.g., SD 1.5). Prioritize core subjects, clothing, poses, background "
            "elements, lighting tags, and quality tokens (e.g. masterpiece, sharp focus). Do NOT write full conversational sentences."
        ),
        "clip_l_tags (sd1.5 / sdxl / booru)": (
            "Format as clean, comma-separated tokens, Danbooru/Booru tags, and quality enhancers suitable "
            "for CLIP text encoders (e.g., SD 1.5, SDXL base). Prioritize core subjects, clothing, poses, background "
            "elements, lighting tags, and quality tokens (e.g. masterpiece, sharp focus). Do NOT write full conversational sentences."
        ),
        "minimax_h3_fl2va (first frame + audio timeline)": (
            "Format as MiniMax H3 FL2VA/I2VA/T2VA: optional keyframe alignment header, followed by the 3 core sections: "
            "integrated_multimodal_description, overall_soundscape, and non_diegetic_music."
        ),
        "minimax_h3_ref2va (6-section multi-reference)": (
            "Format strictly as MiniMax H3 Ref2VA six-section screenplay: "
            "subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, and non_diegetic_music."
        ),
        "minimax_h3 (video + audio timeline)": (
            "Format strictly as a MiniMax H3 three-section video/audio screenplay (alias for fl2va)."
        ),
    }

    SDXL_DUAL_INSTRUCTIONS = (
        "You are an expert AI prompt engineer specializing in Stable Diffusion XL (SDXL).\n"
        "IMPORTANT ARCHITECTURAL CALIBRATION - DO NOT PROMPT LIKE T5-XXL OR FLUX:\n"
        "SDXL uses contrastive CLIP text encoders with a 77-token context limit, NOT large autoregressive language models (like T5-XXL).\n"
        "Avoid long, novelistic, poetic paragraphs, flowery metaphors, or abstract storytelling filler (e.g. 'a sense of wonder', 'whispers of time'). "
        "Every single word must directly describe visible elements in the image to prevent cross-attention dilution.\n"
        "Also avoid collapsing into a single sparse sentence. Aim for the calibrated 'sweet spot':\n\n"
        "SDXL employs a dual text-encoder architecture:\n"
        "1. OpenCLIP ViT-bigG (CLIP-G / text_g): High-capacity encoder (1280 dim). It excels at coherent natural prose for overall scene layout. "
        "The ideal length is a balanced 2 to 3 concise, visually dense sentences (~35 to 55 words):\n"
        "   - Sentence 1: Core subject, action, posture, and immediate physical presence.\n"
        "   - Sentence 2: Environment, architectural/natural surroundings, framing, and depth of field.\n"
        "   - Sentence 3: Lighting quality, atmospheric conditions, color palette, and photographic or artistic medium.\n"
        "2. OpenAI CLIP ViT-L (CLIP-L / text_l): Sensitive to specific concept tokens (768 dim). "
        "Format as 15 to 25 clean, comma-separated keywords and Danbooru-style detail tags emphasizing subject micro-details, "
        "clothing, materials, facial features, camera/lens specs, and quality boosters (e.g. 35mm photo, sharp focus, intricate textures).\n\n"
        "Your task is to expand the user's idea into two synergistic prompts:\n"
        "- [PROMPT_G]: Exactly 2 to 3 concise, visually dense English sentences for CLIP-G (35-55 words). Concrete visual nouns and lighting; zero fluff.\n"
        "- [PROMPT_L]: 15 to 25 clean comma-separated tokens and detail tags for CLIP-L.\n"
    )

    PONY_SDXL_INSTRUCTIONS = (
        "You are an expert AI prompt engineer specializing in Pony Diffusion SDXL (v6, v6.5, and realism fine-tunes).\n"
        "CRITICAL ARCHITECTURAL REQUIREMENT - DUAL ANCHORING:\n"
        "Pony Diffusion's text encoders (OpenCLIP ViT-bigG and OpenAI CLIP ViT-L) were extensively fine-tuned on "
        "curated tag datasets using special aesthetic score tags. To activate the high-quality photorealistic latent space, "
        "BOTH [PROMPT_G] and [PROMPT_L] MUST begin with the mandatory anchor prefix:\n"
        "  `score_9, score_8_up, score_7_up, source_photo, rating_safe`\n"
        "(Note: Set rating_safe to rating_questionable or rating_explicit ONLY if the user prompt explicitly requests suggestive/mature/NSFW content).\n\n"
        "DUAL-ENCODER SPECIALIZATION AFTER THE ANCHORS:\n"
        "- [PROMPT_G] (OpenCLIP ViT-bigG / text_g): Drives the global pooled scene vector, atmosphere, and composition. "
        "After the anchor prefix, include photographic medium and scene descriptors, environment, camera/lens specifications, and lighting tags "
        "(e.g., `realistic, photo, 35mm photograph, soft natural lighting, shallow depth of field, outdoor, city street, volumetric light`).\n"
        "- [PROMPT_L] (OpenAI CLIP ViT-L / text_l): Drives fine token-by-token cross-attention for character, attire, and micro-details. "
        "After the anchor prefix, format primarily as clean, comma-separated Danbooru/Booru tags for subject, hair, eye color, attire, posture, expressions, and physical micro-textures "
        "(e.g., `1girl, solo, black hair, hazel eyes, trenchcoat, looking at viewer, detailed skin texture, micro details`).\n\n"
        "NEGATIVE PROMPT DUAL ANCHORING:\n"
        "Pony requires low-score and style-reject anchors in the negative prompt to discard illustrated and low-quality data clusters. "
        "The negative prompt MUST start with:\n"
        "  `score_4, score_5, score_6, score_1, score_2, score_3, source_anime, source_cartoon, source_furry, source_pony, 3d, render, illustration, drawing, painting`\n"
        "Followed by common photographic defects: `bad anatomy, bad hands, missing fingers, extra digits, blurry, low quality, watermark`."
    )

    MINIMAX_H3_FL2VA_INSTRUCTIONS = (
        "You are an expert AI director and screenplay prompt engineer for MiniMax H3 FL2VA / I2VA / T2VA "
        "(an omni-modal video generation model that natively co-generates synchronized stereo audio, dialogue, and music in a single pass).\n"
        "Your task is to transform the user's idea into a complete, professional MiniMax H3 screenplay prompt.\n\n"
        "Strict Architectural Guidelines:\n"
        "1. Keyframe Alignment Header (when reference image(s) are supplied):\n"
        "   - Exactly 1 Image (I2VA / First Frame): MUST begin with the exact header line:\n"
        "     For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n"
        "     Followed by a blank line before the core sections.\n"
        "   - Exactly 2 Images (FL2VA / First & Last Frame): MUST begin with the exact header line:\n"
        "     How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.\n"
        "     Followed by a blank line before the core sections.\n"
        "   - 0 Images (T2VA / Text-to-Video): Omit any alignment header and start directly with integrated_multimodal_description:.\n\n"
        "2. Section 1 - 'integrated_multimodal_description:':\n"
        "   - Opening Shot: [Shot 1] MUST begin with visual style (e.g. 'Live-action, cinematic,' or '3D CG,' or 'Vintage 35mm film,') "
        "and camera framing. [Shot 1] MUST NEVER have a timestamp. If keyframe images are used, ground [Shot 1] in the visual features, "
        "lighting, and subjects of <Picture 1> before developing forward.\n"
        "   - Shot Cuts: Subsequent shots must use increasing timestamps formatted strictly as 'At MM:SS.mmm' "
        "(e.g. '[Shot 2] At 00:03.500, the camera cuts to...'). For smooth FL2VA interpolation, single-shot progression is preferred unless cuts are explicitly requested.\n"
        "   - Camera Movement: Embed camera motion naturally into action descriptions using Motion Type + Amplitude + Speed "
        "(e.g. 'The camera pushes in with small amplitude at slow speed toward the subject'). Valid motions include: "
        "Push In, Pull Out, Zoom In/Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot.\n"
        "   - Dialogue, Singing, & Lip-Sync: Assign speaker IDs e.g. '(S1)', '(S2)'. Character posture, facial expressions, vocal delivery, "
        "and actions belong OUTSIDE `<d>`. Spoken language code and exact lyrics or dialogue belong INSIDE `<d>[Language] spoken or sung words</d>` "
        "(e.g. `The singer with an intense expression (S1) mouths: <d>[English] My head is a flame</d>`). All diegetic vocalization belongs here.\n"
        "   - On-Screen Text: Enclose visible signage/text in double quotes.\n\n"
        "3. Section 2 - 'overall_soundscape:':\n"
        "   - 1 to 4 sentences describing ambient environmental audio, room tone, footsteps, wind, weather, and physical action Foley. "
        "Non-verbal human sounds (breathing, panting, gasps) belong here. NEVER repeat spoken/sung lyrics or background score in this section.\n\n"
        "4. Section 3 - 'non_diegetic_music:':\n"
        "   - 1 to 3 sentences describing audience-only background musical score: instrumentation, tempo/BPM, rhythm dynamics, and acoustic timbre "
        "(e.g. 'Heavy distorted analog synth bass pulsing at 115 BPM, joined by rhythmic kick drums and airy synthesizer pads that build in intensity').\n"
        "   - IMPORTANT LATENT AUDIO GUIDE RULE: In FL2VA pipelines where audio is conditioned directly in latent space (via latent guides), "
        "do NOT use `<Audio 1>` tags in the prompt text. Instead, meticulously describe the acoustic characteristics (tempo, rhythm, instruments, timbre) "
        "so the model's cross-attention aligns with the latent audio conditioning! Write 'N/A' only if completely silent."
    )

    MINIMAX_H3_REF2VA_INSTRUCTIONS = (
        "You are an expert AI director and multi-modal screenplay prompt engineer for MiniMax H3 Ref2VA "
        "(multi-reference video generation model supporting tokenized reference images <Picture 1>..<Picture 9> "
        "and reference audio tracks <Audio 1>..<Audio 3>).\n"
        "Your task is to transform the user's idea and reference inputs into the official MiniMax H3 6-section rewrite format.\n\n"
        "Strict Architectural Guidelines:\n"
        "Your output MUST contain exactly six structured sections in this exact order:\n\n"
        "1. 'subject_definitions:':\n"
        "   Define each referenced entity on its own line using standardized tags:\n"
        "   - `<Subject N>`: Reusable visible subject (person, costume, environment, prop, or style). E.g.:\n"
        "     `<Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue cardigan, and silver necklace.`\n"
        "   - `<Picture N>`: Standalone image anchor if used as first frame, keyframe, or composition anchor. E.g.:\n"
        "     `<Picture 1> is the first frame of [Shot 1], establishing the subject and lighting.`\n"
        "   - `<Video N>`: Reference video for motion, camera work, or structural edit.\n"
        "   - `<Audio N>`: Reference audio track. E.g.:\n"
        "     `<Audio 1> is the background music track providing tempo, beat, and melody.` or\n"
        "     `<Audio 1> is the voice-timbre reference for <Subject 1> (S1).`\n\n"
        "2. 'summary:':\n"
        "   One concise English paragraph summarizing the target video and reference relationships. "
        "MUST begin with a square-bracketed task type prefix, such as:\n"
        "   `[reference generation] ...`\n"
        "   `[keyframe completion + audio reference] ...`\n"
        "   `[audio reuse + reference generation] ...`\n\n"
        "3. 'retention_analysis:':\n"
        "   Explicitly describe how referenced content is preserved, modified, or transferred across the video. "
        "Document what aspects of <Picture N> or <Audio N> are retained (e.g. facial features, costume, musical rhythm) "
        "and what aspects change or evolve.\n\n"
        "4. 'detailed_description:':\n"
        "   The shot-by-shot visual and diegetic timeline in playback order:\n"
        "   - [Shot 1] begins with style and composition framing (no timestamp). Subsequent shots use '[Shot N] At MM:SS.mmm, ...'.\n"
        "   - Explicitly cite where referenced entities appear (e.g. '<Subject 1> turns toward the window...').\n"
        "   - Camera movement: Motion Type + Amplitude + Speed (e.g. 'The camera pushes in with small amplitude at slow speed...').\n"
        "   - Dialogue, singing, & lip-sync: Speaker IDs e.g. '(S1)' with delivery outside and spoken words inside `<d>[Language] lyrics or dialogue</d>`.\n"
        "   - Visible text in double quotes.\n\n"
        "5. 'overall_soundscape:':\n"
        "   1 to 4 sentences summarizing ambient sound, room tone, environmental audio, and physical action Foley. "
        "No dialogue or background score repetition.\n\n"
        "6. 'non_diegetic_music:':\n"
        "   1 to 3 sentences describing the audience-only musical score. When reference audio is supplied, explicitly bind "
        "and describe it using `<Audio 1>` (e.g. 'Driven by <Audio 1>, energetic synthwave bass at 120 BPM with driving percussion...'). "
        "Write 'N/A' if completely silent."
    )

    MINIMAX_H3_INSTRUCTIONS = MINIMAX_H3_FL2VA_INSTRUCTIONS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a cybernetic samurai in rain",
                    },
                ),
                "prompt_format": (
                    [
                        "natural_language (modern / flux / muse)",
                        "sdxl (dual clip_g + clip_l)",
                        "clip_l_tags (sd1.5 / booru)",
                        "clip_l_tags (sd1.5 / sdxl / booru)",
                        "minimax_h3_fl2va (first frame + audio timeline)",
                        "minimax_h3_ref2va (6-section multi-reference)",
                        "minimax_h3 (video + audio timeline)",
                    ],
                    {"default": "natural_language (modern / flux / muse)"},
                ),
                "preset": (
                    [
                        "photorealistic",
                        "cinematic",
                        "pony_realism",
                        "digital_art / anime",
                        "general_expansion",
                        "minimax_h3_fl2va (first frame + audio guide)",
                        "minimax_h3_ref2va (multi-reference r2v)",
                        "minimax_h3 (video + audio director)",
                        "custom",
                    ],
                    {"default": "photorealistic"},
                ),
                "include_negative": ("BOOLEAN", {"default": False}),
                "model": (
                    [
                        "muse-spark-1.3",
                        "muse-spark-1.3-contributor",
                        "muse-spark-1.2",
                        "muse-spark-1.2-contributor",
                        "muse-spark-1.1",
                    ],
                    {"default": "muse-spark-1.3"},
                ),
                "reasoning_effort": (
                    ["low", "medium", "high"],
                    {"default": "low"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": "fixed",
                    },
                ),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_images": ("MUSE_IMAGES",),
                "custom_instructions": ("STRING", {"forceInput": True}),
                "api_key_override": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("expanded_prompt", "negative_prompt", "reasoning_summary", "prompt_g", "prompt_l")
    FUNCTION = "expand"
    CATEGORY = "phaulty nodes/Muse"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

    def expand(
        self,
        prompt: str,
        prompt_format: str,
        preset: str,
        include_negative: bool,
        model: str,
        reasoning_effort: str,
        seed: int,
        reference_image: torch.Tensor = None,
        reference_images: list = None,
        custom_instructions: str = None,
        api_key_override: str = "",
    ):
        api_key = _get_api_key(api_key_override)
        if not api_key:
            raise ValueError(
                "MODEL_API_KEY is not set or invalid. Please check ComfyUI/custom_nodes/ComfyUI-MuseImage/config.json."
            )

        images = _extract_images(reference_image, reference_images)

        is_ref2va = (
            preset == "minimax_h3_ref2va (multi-reference r2v)"
            or prompt_format == "minimax_h3_ref2va (6-section multi-reference)"
        )
        is_fl2va = (
            preset in ("minimax_h3_fl2va (first frame + audio guide)", "minimax_h3 (video + audio director)")
            or prompt_format in ("minimax_h3_fl2va (first frame + audio timeline)", "minimax_h3 (video + audio timeline)")
        )
        is_sdxl_dual = (
            prompt_format in ("sdxl (dual clip_g + clip_l)", "sdxl_dual (clip_g prose + clip_l tags)")
        )

        if is_ref2va:
            format_structure = (
                "Strict Output Format:\n"
                "[PROMPT]\n"
                "subject_definitions:\n"
                "<Subject 1> is ...\n\n"
                "summary:\n"
                "[reference generation] ...\n\n"
                "retention_analysis:\n"
                "...\n\n"
                "detailed_description:\n"
                "[Shot 1] ...\n\n"
                "overall_soundscape:\n"
                "...\n\n"
                "non_diegetic_music:\n"
                "...\n"
                "[/PROMPT]\n"
            )
            if include_negative:
                format_structure += (
                    "[NEGATIVE]\n"
                    "your tailored negative prompt here\n"
                    "[/NEGATIVE]\n"
                )
            format_structure += "Do NOT include any conversational filler, notes, or markdown fences outside these tags."

            instructions_parts = [self.MINIMAX_H3_REF2VA_INSTRUCTIONS]
            if images:
                num_imgs = len(images)
                instructions_parts.append(
                    f"Reference Image(s) Attached: You have received {num_imgs} reference image(s). "
                    f"Bind them under subject_definitions as <Picture 1>"
                    + (f" through <Picture {num_imgs}>" if num_imgs > 1 else "")
                    + " and/or assign them to <Subject 1>, etc. Reference them consistently in retention_analysis and detailed_description."
                )
            if custom_instructions and custom_instructions.strip():
                instructions_parts.append(f"Director / User custom requirements: {custom_instructions.strip()}")
            if not include_negative:
                instructions_parts.append(
                    "Negative prompt guidance: MiniMax H3 does NOT use negative prompts. Incorporate all visual and acoustic "
                    "quality directives directly into the description. Do NOT output a negative prompt."
                )
            else:
                instructions_parts.append(
                    "Negative prompt guidance: Generate both the MiniMax H3 prompt and a targeted negative prompt."
                )
            instructions_parts.append(format_structure)
            instructions = "\n\n".join(instructions_parts)

        elif is_fl2va:
            format_structure = "Strict Output Format:\n[PROMPT]\n"
            if images:
                if len(images) == 1:
                    format_structure += (
                        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
                    )
                else:
                    format_structure += (
                        "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the [duration]-second mark of the target video.\n\n"
                    )
            format_structure += (
                "integrated_multimodal_description:\n"
                "[Shot 1] ...\n\n"
                "overall_soundscape:\n"
                "...\n\n"
                "non_diegetic_music:\n"
                "...\n"
                "[/PROMPT]\n"
            )
            if include_negative:
                format_structure += (
                    "[NEGATIVE]\n"
                    "your tailored negative prompt here\n"
                    "[/NEGATIVE]\n"
                )
            format_structure += "Do NOT include any conversational filler, notes, or markdown fences outside these tags."

            instructions_parts = [self.MINIMAX_H3_FL2VA_INSTRUCTIONS]
            if images:
                if len(images) == 1:
                    instructions_parts.append(
                        "Reference Image Attached: Exactly 1 image is provided as the first-frame anchor (<Picture 1>). "
                        "You MUST begin the prompt with the exact header:\n"
                        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n"
                        "Then ground [Shot 1] in the visual style, subject, clothing, and scene of <Picture 1> before developing forward."
                    )
                elif len(images) >= 2:
                    instructions_parts.append(
                        f"Reference Images Attached: {len(images)} images provided (Picture 1 as first frame, Picture 2 as last frame). "
                        "You MUST begin the prompt with the exact keyframe alignment header:\n"
                        "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the [duration]-second mark of the target video.\n"
                        "Describe the continuous motion and transformation path connecting Picture 1 to Picture 2."
                    )
            else:
                instructions_parts.append(
                    "No reference images provided (Text-to-Video / T2VA mode). Do NOT include any keyframe alignment instruction; "
                    "begin directly with integrated_multimodal_description:."
                )

            if custom_instructions and custom_instructions.strip():
                instructions_parts.append(f"Director / User custom requirements: {custom_instructions.strip()}")
            if not include_negative:
                instructions_parts.append(
                    "Negative prompt guidance: MiniMax H3 does NOT use negative prompts. Incorporate all visual and acoustic "
                    "quality directives directly into the description. Do NOT output a negative prompt."
                )
            else:
                instructions_parts.append(
                    "Negative prompt guidance: Generate both the MiniMax H3 prompt and a targeted negative prompt."
                )
            instructions_parts.append(format_structure)
            instructions = "\n\n".join(instructions_parts)

        elif is_sdxl_dual:
            is_pony = (preset == "pony_realism")
            if is_pony:
                format_structure = (
                    "Strict Output Format:\n"
                    "[PROMPT_G]\n"
                    "score_9, score_8_up, score_7_up, source_photo, rating_safe, your descriptive scene composition, environment, camera, and photographic lighting here\n"
                    "[/PROMPT_G]\n"
                    "[PROMPT_L]\n"
                    "score_9, score_8_up, score_7_up, source_photo, rating_safe, your clean comma-separated Danbooru tags for subject, hair, attire, and micro-details here\n"
                    "[/PROMPT_L]\n"
                )
            else:
                format_structure = (
                    "Strict Output Format:\n"
                    "[PROMPT_G]\n"
                    "your descriptive natural language prose for CLIP-G (OpenCLIP ViT-bigG) here\n"
                    "[/PROMPT_G]\n"
                    "[PROMPT_L]\n"
                    "your clean comma-separated tokens and detail tags for CLIP-L (OpenAI CLIP ViT-L) here\n"
                    "[/PROMPT_L]\n"
                )
            if include_negative:
                if is_pony:
                    format_structure += (
                        "[NEGATIVE]\n"
                        "score_4, score_5, score_6, score_1, score_2, score_3, source_anime, source_cartoon, source_furry, source_pony, 3d, render, illustration, drawing, painting, bad anatomy, bad hands, blurry, ...\n"
                        "[/NEGATIVE]\n"
                    )
                else:
                    format_structure += (
                        "[NEGATIVE]\n"
                        "your tailored SDXL negative prompt here\n"
                        "[/NEGATIVE]\n"
                    )
            format_structure += "Do NOT include any conversational filler, notes, or markdown fences outside these tags."

            preset_guide = self.PRESET_GUIDES.get(
                preset, self.PRESET_GUIDES["photorealistic"]
            )
            if is_pony:
                instructions_parts = [
                    self.PONY_SDXL_INSTRUCTIONS,
                    "Format requirement: Both [PROMPT_G] and [PROMPT_L] MUST begin with the mandatory anchor prefix: 'score_9, score_8_up, score_7_up, source_photo, rating_safe'. "
                    "Follow with specialized content: [PROMPT_G] for scene atmosphere, camera, and photographic lighting; [PROMPT_L] for clean Danbooru character, clothing, and micro-detail tags.",
                ]
            else:
                instructions_parts = [
                    self.SDXL_DUAL_INSTRUCTIONS,
                    "Format requirement: [PROMPT_G] MUST be exactly 2 to 3 concise, visually dense sentences (35-55 words) describing subject, scene, and lighting for OpenCLIP bigG (avoid long novelistic paragraphs or single sparse sentences). [PROMPT_L] MUST be 15 to 25 clean comma-separated tokens and detail tags for CLIP-L.",
                ]
            if images:
                instructions_parts.append(
                    f"Reference Image(s) Attached: You have received {len(images)} reference image(s). "
                    "Carefully inspect their visual elements (subjects, appearance, clothing, lighting, textures, "
                    "color palette, composition, environment). Use these visual cues to ground, inspire, and enrich "
                    "both PROMPT_G and PROMPT_L, incorporating specific details from the images while executing the user's concept."
                )

            if preset == "custom" and custom_instructions and custom_instructions.strip():
                instructions_parts.append(f"Instructions: {custom_instructions.strip()}")
            else:
                instructions_parts.append(f"Aesthetic guidance: {preset_guide}")
                if custom_instructions and custom_instructions.strip():
                    instructions_parts.append(f"Additional instructions: {custom_instructions.strip()}")

            if include_negative:
                if is_pony:
                    instructions_parts.append(
                        "Negative prompt guidance: For Pony realism, you MUST begin the negative prompt with the low-score and style-reject anchors: "
                        "'score_4, score_5, score_6, score_1, score_2, score_3, source_anime, source_cartoon, source_furry, source_pony, 3d, render, illustration, drawing, painting', "
                        "followed by photographic defect tags like 'bad anatomy, bad hands, blurry, low quality, watermark'."
                    )
                else:
                    instructions_parts.append(
                        "Negative prompt guidance: Generate a tailored negative prompt for SDXL to suppress common artifacts, "
                        "anatomical deformities, bad hands, blurriness, pixelation, watermarks, and unwanted styling."
                    )
            else:
                instructions_parts.append(
                    "Negative prompt guidance: Do NOT output a negative prompt."
                )

            instructions_parts.append(format_structure)
            instructions = "\n\n".join(instructions_parts)
        else:
            format_guide = self.FORMAT_GUIDES.get(
                prompt_format, self.FORMAT_GUIDES["natural_language (modern / flux / muse)"]
            )
            preset_guide = self.PRESET_GUIDES.get(
                preset, self.PRESET_GUIDES["photorealistic"]
            )

            if include_negative:
                neg_instruction = (
                    "Generate both an expanded positive prompt and a targeted negative prompt to eliminate common artifacts "
                    "or unwanted elements for this subject and style."
                )
                format_structure = (
                    "Strict Output Format:\n"
                    "[PROMPT]\n"
                    "your expanded positive prompt here\n"
                    "[/PROMPT]\n"
                    "[NEGATIVE]\n"
                    "your tailored negative prompt here\n"
                    "[/NEGATIVE]\n"
                    "Do NOT include any conversational filler, notes, or markdown fences outside these tags."
                )
            else:
                neg_instruction = (
                    "The target image model does NOT support or use negative prompts (e.g. distilled or modern architecture). "
                    "Incorporate all quality, cleanliness, lighting, and detail guidance directly into the positive prompt. "
                    "Do NOT output or mention a negative prompt."
                )
                format_structure = (
                    "Strict Output Format:\n"
                    "[PROMPT]\n"
                    "your expanded positive prompt here\n"
                    "[/PROMPT]\n"
                    "Do NOT include any conversational filler, notes, or markdown fences outside these tags."
                )

            instructions_parts = [
                "You are an expert AI prompt engineer for image generation.",
                f"Format requirement: {format_guide}",
            ]

            if images:
                instructions_parts.append(
                    f"Reference Image(s) Attached: You have received {len(images)} reference image(s). "
                    "Carefully inspect their visual elements (subjects, appearance, clothing, lighting, textures, "
                    "color palette, composition, environment). Use these visual cues to ground, inspire, and enrich "
                    "your expanded prompt, incorporating specific details from the images while executing the user's concept."
                )

            if preset == "custom" and custom_instructions and custom_instructions.strip():
                instructions_parts.append(f"Instructions: {custom_instructions.strip()}")
            else:
                instructions_parts.append(f"Aesthetic guidance: {preset_guide}")
                if custom_instructions and custom_instructions.strip():
                    instructions_parts.append(f"Additional instructions: {custom_instructions.strip()}")

            instructions_parts.append(f"Negative prompt guidance: {neg_instruction}")
            instructions_parts.append(format_structure)

            instructions = "\n\n".join(instructions_parts)

        payload = {
            "model": model,
            "instructions": instructions,
            "reasoning": {
                "effort": reasoning_effort,
                "summary": "detailed",
            },
            "store": True,
        }

        if images:
            input_text = prompt if prompt and prompt.strip() else "Analyze the provided reference image(s) and expand into a detailed prompt."
            content = [{"type": "input_text", "text": input_text}]
            for img in images:
                image_data_url = _tensor_to_data_url(img)
                content.append({"type": "input_image", "image_url": image_data_url})
            payload["input"] = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        else:
            payload["input"] = prompt

        raw_text, response_id, reasoning_summary = _call_muse_spark_api(api_key, payload)
        expanded_prompt, negative_prompt, prompt_g, prompt_l = _parse_spark_output(raw_text, include_negative)

        return (expanded_prompt, negative_prompt, reasoning_summary, prompt_g, prompt_l)


class MuseSparkSDXLExpander:
    """
    Dedicated SDXL Prompt Expansion node powered by Meta's Muse Spark reasoning model.
    Specifically architects prompts for SDXL's dual text encoders:
    - prompt_g (OpenCLIP ViT-bigG / text_g): Rich natural language prose describing scene composition, atmosphere, lighting, camera angles, and visual narrative.
    - prompt_l (OpenAI CLIP ViT-L / text_l): Clean comma-separated tokens, Danbooru/booru tags, subject/clothing micro-details, and quality boosters.
    - negative_prompt: Tailored negative prompt designed for SDXL to eliminate deformities, artifacts, and blur.
    - expanded_prompt: Combined prompt (prompt_g + prompt_l) for single-input pipelines.
    - reasoning_summary: Model reasoning tokens or summary.
    """

    SDXL_DUAL_INSTRUCTIONS = MuseSparkPromptExpander.SDXL_DUAL_INSTRUCTIONS
    PONY_SDXL_INSTRUCTIONS = MuseSparkPromptExpander.PONY_SDXL_INSTRUCTIONS

    PRESET_GUIDES = {
        "photorealistic": (
            "Focus on authentic realism and camera photography: natural skin/surface micro-textures, "
            "lens specifications (e.g. 35mm or 85mm f/1.4), accurate real-world lighting, authentic materials, "
            "depth of field, and lifelike physical detail without artificial gloss."
        ),
        "cinematic": (
            "Focus on cinematic film aesthetics: dramatic anamorphic composition, widescreen framing, "
            "atmospheric haze or volumetric light, rich color grading, directional rim lighting, and emotional mood."
        ),
        "pony_realism": (
            "Focus on Pony SDXL photorealism with strict dual-anchoring: Both text encoders (CLIP-G and CLIP-L) "
            "must begin with 'score_9, score_8_up, score_7_up, source_photo, rating_safe'. "
            "After anchors, address CLIP-G with photographic scene/lighting tags and CLIP-L with Danbooru subject/attire tags, "
            "paired with Pony's low-score reject negative anchors."
        ),
        "anime / manga": (
            "Focus on anime and manga illustration aesthetics: expressive character design, dynamic framing, "
            "vibrant or cel-shaded palette, crisp line work, beautiful anime eyes, hair highlights, and clean composition."
        ),
        "digital_art / concept_art": (
            "Focus on high-end stylized digital art and concept illustration: dramatic fantasy or sci-fi mood, "
            "rich painterly textures, intricate environmental details, stylized color keys, and striking visual storytelling."
        ),
        "general_expansion": (
            "Provide a balanced, versatile enhancement: enrich subject anatomy, setting, atmospheric lighting, "
            "and composition without over-constraining the visual medium."
        ),
        "custom": "Follow the user's custom instructions precisely.",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a cybernetic samurai in rain",
                    },
                ),
                "preset": (
                    [
                        "photorealistic",
                        "cinematic",
                        "pony_realism",
                        "anime / manga",
                        "digital_art / concept_art",
                        "general_expansion",
                        "custom",
                    ],
                    {"default": "photorealistic"},
                ),
                "dual_format": (
                    [
                        "clip_g prose + clip_l tags (recommended)",
                        "clip_g prose + clip_l prose",
                        "clip_g tags + clip_l tags",
                    ],
                    {"default": "clip_g prose + clip_l tags (recommended)"},
                ),
                "include_negative": ("BOOLEAN", {"default": True}),
                "model": (
                    [
                        "muse-spark-1.3",
                        "muse-spark-1.3-contributor",
                        "muse-spark-1.2",
                        "muse-spark-1.2-contributor",
                        "muse-spark-1.1",
                    ],
                    {"default": "muse-spark-1.3"},
                ),
                "reasoning_effort": (
                    ["low", "medium", "high"],
                    {"default": "low"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": "fixed",
                    },
                ),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_images": ("MUSE_IMAGES",),
                "custom_instructions": ("STRING", {"forceInput": True}),
                "api_key_override": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_g", "prompt_l", "negative_prompt", "expanded_prompt", "reasoning_summary")
    FUNCTION = "expand"
    CATEGORY = "phaulty nodes/Muse"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

    def expand(
        self,
        prompt: str,
        preset: str,
        dual_format: str,
        include_negative: bool,
        model: str,
        reasoning_effort: str,
        seed: int,
        reference_image: torch.Tensor = None,
        reference_images: list = None,
        custom_instructions: str = None,
        api_key_override: str = "",
    ):
        api_key = _get_api_key(api_key_override)
        if not api_key:
            raise ValueError(
                "MODEL_API_KEY is not set or invalid. Please check ComfyUI/custom_nodes/ComfyUI-MuseImage/config.json."
            )

        images = _extract_images(reference_image, reference_images)
        preset_guide = self.PRESET_GUIDES.get(preset, self.PRESET_GUIDES["photorealistic"])
        is_pony = (preset == "pony_realism")

        if is_pony:
            if dual_format == "clip_g prose + clip_l prose":
                format_rule = (
                    "PONY DUAL ANCHORING REQUIREMENT: Both [PROMPT_G] and [PROMPT_L] MUST begin with the mandatory anchor prefix: "
                    "'score_9, score_8_up, score_7_up, source_photo, rating_safe'. "
                    "Follow the prefix with coherent photographic scene and subject descriptions for both encoders."
                )
            elif dual_format == "clip_g tags + clip_l tags":
                format_rule = (
                    "PONY DUAL ANCHORING REQUIREMENT: Both [PROMPT_G] and [PROMPT_L] MUST begin with the mandatory anchor prefix: "
                    "'score_9, score_8_up, score_7_up, source_photo, rating_safe'. "
                    "Follow the prefix with clean comma-separated Danbooru tags for both encoders."
                )
            else:
                format_rule = (
                    "PONY DUAL ANCHORING REQUIREMENT: Both [PROMPT_G] and [PROMPT_L] MUST begin with the mandatory anchor prefix: "
                    "'score_9, score_8_up, score_7_up, source_photo, rating_safe'. "
                    "Then, address the two text encoders separately:\n"
                    "- [PROMPT_G] (OpenCLIP ViT-bigG): Follow the anchor prefix with scene composition, environment, camera, and photographic lighting tags (e.g. realistic, 35mm photo, cinematic lighting, outdoor, bokeh).\n"
                    "- [PROMPT_L] (OpenAI CLIP ViT-L): Follow the anchor prefix with clean comma-separated Danbooru/Booru tags for subject, hair, eyes, clothing, and micro-details."
                )
        else:
            if dual_format == "clip_g prose + clip_l prose":
                format_rule = (
                    "Format requirement: Both [PROMPT_G] and [PROMPT_L] should be 2 to 3 concise, visually dense sentences (35-55 words). "
                    "[PROMPT_G] focuses on scene context, lighting, and composition; [PROMPT_L] focuses on detailed subject appearance and action. Avoid novelistic or poetic filler."
                )
            elif dual_format == "clip_g tags + clip_l tags":
                format_rule = (
                    "Format requirement: Both [PROMPT_G] and [PROMPT_L] should be clean, comma-separated tokens and quality tags (15-25 tokens each). "
                    "[PROMPT_G] for high-level scene/style tags, [PROMPT_L] for subject, attire, and detail tags."
                )
            else:
                format_rule = (
                    "Format requirement: [PROMPT_G] MUST be exactly 2 to 3 concise, visually dense sentences (~35-55 words) describing subject, scene, and lighting (avoid long novelistic paragraphs or single sparse sentences). "
                    "[PROMPT_L] MUST be 15 to 25 clean comma-separated tokens, booru tags, and detail keywords for OpenAI CLIP-L (no full sentences)."
                )

        if is_pony:
            format_structure = (
                "Strict Output Format:\n"
                "[PROMPT_G]\n"
                "score_9, score_8_up, score_7_up, source_photo, rating_safe, your scene and photographic lighting descriptors for CLIP-G (OpenCLIP ViT-bigG) here\n"
                "[/PROMPT_G]\n"
                "[PROMPT_L]\n"
                "score_9, score_8_up, score_7_up, source_photo, rating_safe, your clean Danbooru tags and subject details for CLIP-L (OpenAI CLIP ViT-L) here\n"
                "[/PROMPT_L]\n"
            )
            if include_negative:
                format_structure += (
                    "[NEGATIVE]\n"
                    "score_4, score_5, score_6, score_1, score_2, score_3, source_anime, source_cartoon, source_furry, source_pony, 3d, render, illustration, drawing, painting, bad hands, blurry, ...\n"
                    "[/NEGATIVE]\n"
                )
        else:
            format_structure = (
                "Strict Output Format:\n"
                "[PROMPT_G]\n"
                "your text for CLIP-G (OpenCLIP ViT-bigG) here\n"
                "[/PROMPT_G]\n"
                "[PROMPT_L]\n"
                "your text for CLIP-L (OpenAI CLIP ViT-L) here\n"
                "[/PROMPT_L]\n"
            )
            if include_negative:
                format_structure += (
                    "[NEGATIVE]\n"
                    "your tailored SDXL negative prompt here\n"
                    "[/NEGATIVE]\n"
                )
        format_structure += "Do NOT include any conversational filler, notes, or markdown fences outside these tags."

        instructions_parts = [
            self.PONY_SDXL_INSTRUCTIONS if is_pony else self.SDXL_DUAL_INSTRUCTIONS,
            format_rule,
        ]

        if images:
            instructions_parts.append(
                f"Reference Image(s) Attached: You have received {len(images)} reference image(s). "
                "Carefully inspect their visual elements (subjects, appearance, clothing, lighting, textures, "
                "color palette, composition, environment). Use these visual cues to ground, inspire, and enrich "
                "both PROMPT_G and PROMPT_L, incorporating specific details from the images while executing the user's concept."
            )

        if preset == "custom" and custom_instructions and custom_instructions.strip():
            instructions_parts.append(f"Instructions: {custom_instructions.strip()}")
        else:
            instructions_parts.append(f"Aesthetic guidance: {preset_guide}")
            if custom_instructions and custom_instructions.strip():
                instructions_parts.append(f"Additional instructions: {custom_instructions.strip()}")

        if include_negative:
            if is_pony:
                instructions_parts.append(
                    "Negative prompt guidance: For Pony realism, you MUST begin the negative prompt with the low-score and style-reject anchors: "
                    "'score_4, score_5, score_6, score_1, score_2, score_3, source_anime, source_cartoon, source_furry, source_pony, 3d, render, illustration, drawing, painting', "
                    "followed by photographic defect tags like 'bad anatomy, bad hands, blurry, low quality, watermark'."
                )
            else:
                instructions_parts.append(
                    "Negative prompt guidance: Generate a tailored negative prompt for SDXL to suppress common artifacts, "
                    "anatomical deformities, bad hands, extra limbs, blurriness, pixelation, watermarks, and signature tokens, "
                    "harmonized with the selected aesthetic."
                )
        else:
            instructions_parts.append(
                "Negative prompt guidance: Do NOT output a negative prompt."
            )

        instructions_parts.append(format_structure)
        instructions = "\n\n".join(instructions_parts)

        payload = {
            "model": model,
            "instructions": instructions,
            "reasoning": {
                "effort": reasoning_effort,
                "summary": "detailed",
            },
            "store": True,
        }

        if images:
            input_text = prompt if prompt and prompt.strip() else "Analyze the provided reference image(s) and expand into an SDXL dual prompt."
            content = [{"type": "input_text", "text": input_text}]
            for img in images:
                image_data_url = _tensor_to_data_url(img)
                content.append({"type": "input_image", "image_url": image_data_url})
            payload["input"] = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        else:
            payload["input"] = prompt

        raw_text, response_id, reasoning_summary = _call_muse_spark_api(api_key, payload)
        expanded_prompt, negative_prompt, prompt_g, prompt_l = _parse_spark_output(raw_text, include_negative)

        return (prompt_g, prompt_l, negative_prompt, expanded_prompt, reasoning_summary)


NODE_CLASS_MAPPINGS = {
    "MuseImageNode": MuseImageNode,
    "MuseImageEditorNode": MuseImageEditorNode,
    "MuseShowTextNode": MuseShowTextNode,
    "MuseSwitchNode": MuseSwitchNode,
    "MuseImageArrayNode": MuseImageArrayNode,
    "MuseSparkPromptExpander": MuseSparkPromptExpander,
    "MuseSparkSDXLExpander": MuseSparkSDXLExpander,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseImageNode": "Meta Muse Image",
    "MuseImageEditorNode": "Meta Muse Image Editor / Refiner",
    "MuseShowTextNode": "Meta Muse Show Text / Reasoning",
    "MuseSwitchNode": "Meta Muse Mode Switch",
    "MuseImageArrayNode": "Meta Muse Image Array",
    "MuseSparkPromptExpander": "Meta Muse Spark Prompt Expander",
    "MuseSparkSDXLExpander": "Meta Muse Spark SDXL Prompt Expander",
}

