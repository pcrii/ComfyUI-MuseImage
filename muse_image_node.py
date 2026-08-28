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

    def generate(
        self,
        prompt: str,
        size: str,
        reasoning_strength: str,
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

    def edit(
        self,
        prompt: str,
        reasoning_strength: str,
        mode: str,
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


NODE_CLASS_MAPPINGS = {
    "MuseImageNode": MuseImageNode,
    "MuseImageEditorNode": MuseImageEditorNode,
    "MuseShowTextNode": MuseShowTextNode,
    "MuseSwitchNode": MuseSwitchNode,
    "MuseImageArrayNode": MuseImageArrayNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseImageNode": "Meta Muse Image",
    "MuseImageEditorNode": "Meta Muse Image Editor / Refiner",
    "MuseShowTextNode": "Meta Muse Show Text / Reasoning",
    "MuseSwitchNode": "Meta Muse Mode Switch",
    "MuseImageArrayNode": "Meta Muse Image Array",
}

