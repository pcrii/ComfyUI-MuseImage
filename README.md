# ComfyUI-MuseImage

Custom ComfyUI nodes for generating and iteratively editing images using Meta's [Muse Image](https://developer.meta.com/ai/models/muse-image/) model.

---

## Setup & Configuration

1. Clone this repository into your `ComfyUI/custom_nodes/` directory.
2. Copy `config.json.example` to `config.json`:
   ```bash
   cp config.json.example config.json
   ```
3. Open `config.json` and insert your API key:
   ```json
   {
     "MODEL_API_KEY": "your_api_key_here",
     "MODEL_BASE_URL": "https://api.meta.ai/v1"
   }
   ```
*(You can also set the `MODEL_API_KEY` or `META_API_KEY` environment variable, or provide an inline API key override in the node itself).*

---

## Nodes

Nodes are located under the **`phaulty nodes` &rsaquo; `Muse`** category:

- **Meta Muse Image (`MuseImageNode`)**: Text-to-image and image-to-image generation with seed-based cache control. Outputs the generated `IMAGE`, `response_id`, and `reasoning_summary`.
- **Meta Muse Image Editor / Refiner (`MuseImageEditorNode`)**: Iterative multi-turn image editing with seed-based cache control.
- **Meta Muse Spark Prompt Expander (`MuseSparkPromptExpander`)**: Prompt expansion powered by Meta's Muse Spark reasoning models. Supports modern natural language, dedicated SDXL dual encoders (`prompt_g` and `prompt_l`), or classic CLIP tags, optional negative prompt generation, aesthetic presets, custom instruction overrides, and fake seed caching.
- **Meta Muse Spark SDXL Prompt Expander (`MuseSparkSDXLExpander`)**: Dedicated prompt expansion engineered specifically for Stable Diffusion XL's dual text-encoder architecture. Outputs `prompt_g` (for OpenCLIP ViT-bigG / `text_g`) and `prompt_l` (for OpenAI CLIP ViT-L / `text_l`) directly, plus tailored SDXL negative prompts.
- **Meta Muse Show Text / Reasoning (`MuseShowTextNode`)**: Lightweight canvas display node to view reasoning logs, prompts, and IDs.
- **Meta Muse Mode Switch (`MuseSwitchNode`)**: Routes between initial generation and iterative editor outputs (`IMAGE`, `reasoning_summary`, and `response_id`) to drive a single `SaveImage` / `MuseShowTextNode`, avoiding duplicate saved images and allowing dynamic response ID file naming.
- **Meta Muse Image Array (`MuseImageArrayNode`)**: Combines multiple reference images of different resolutions or aspect ratios into an image bundle (`MUSE_IMAGES`) without requiring resizing or cropping. Supports chaining for unlimited reference images.

---

## Prompt Expansion (`MuseSparkPromptExpander` & `MuseSparkSDXLExpander`)

### 1. Dedicated SDXL Expander (`Meta Muse Spark SDXL Prompt Expander`)
Engineered from the ground up for SDXL's dual text-encoder architecture:
- **`dual_format`**:
  - `clip_g prose + clip_l tags (recommended)`: Generates rich, coherent natural language prose for **OpenCLIP ViT-bigG** (`text_g` focusing on overall scene, lighting, composition, and mood) while generating clean comma-separated tokens, booru tags, and detail keywords for **OpenAI CLIP ViT-L** (`text_l` focusing on specific subjects, clothing, textures, and quality boosters).
  - `clip_g prose + clip_l prose`: Coherent descriptive prose for both encoders.
  - `clip_g tags + clip_l tags`: Comma-separated tokens and quality tags for both encoders.
- **`preset`**: `photorealistic`, `cinematic`, `anime / manga`, `digital_art / concept_art`, `general_expansion`, and `custom`.
- **`include_negative`**: Enabled by default (`True`). Produces a tailored negative prompt suppressing SDXL artifacts, bad anatomy, blur, and distortion.
- **Outputs**:
  - `prompt_g`: Connects directly to `text_g` on `CLIPTextEncodeSDXL`.
  - `prompt_l`: Connects directly to `text_l` on `CLIPTextEncodeSDXL`.
  - `negative_prompt`: Connects to negative `CLIPTextEncodeSDXL` (`text_g` & `text_l`).
  - `expanded_prompt`: Combined prompt (`prompt_g` + `prompt_l`) for single-input workflows.
  - `reasoning_summary`: Token usage / reasoning details for `MuseShowTextNode`.

### 2. General Expander (`Meta Muse Spark Prompt Expander`)
The multi-architecture expander transforms brief ideas into rich prompts across diverse models:
- **`prompt_format`**:
  - `natural_language (modern / flux / muse)`: Descriptive prose covering subject, lighting, composition, and texture.
  - `sdxl (dual clip_g + clip_l)`: Dual-encoder expansion splitting into `prompt_g` prose and `prompt_l` tags.
  - `clip_l_tags (sd1.5 / booru)`: Comma-separated CLIP tokens, quality tags, and booru keywords tailored for SD 1.5.
  - `clip_l_tags (sd1.5 / sdxl / booru)`: Backwards-compatible alias for existing workflows.
  - `minimax_h3_fl2va (first frame + audio timeline)`: Keyframe alignment header + 3-section video/audio timeline (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`).
  - `minimax_h3_ref2va (6-section multi-reference)`: MiniMax H3 6-section rewrite structure with `<Picture 1>`..`<Picture 9>` and `<Audio 1>`..`<Audio 3>`.
  - `minimax_h3 (video + audio timeline)`: Backwards-compatible alias for `fl2va`.
- **`preset`**: Includes `photorealistic`, `cinematic`, `digital_art / anime`, `general_expansion`, `minimax_h3_fl2va (first frame + audio guide)`, `minimax_h3_ref2va (multi-reference r2v)`, `minimax_h3 (video + audio director)`, and `custom`.
- **`include_negative`**: Optional negative prompt generation (`False` by default for distilled/modern models).
- **`seed`**: Standard ComfyUI seed widget with cache locking.
- **`reference_image` / `reference_images`**: Ground prompt expansion in reference image features.
- **Outputs**:
  - `expanded_prompt`: Primary positive prompt.
  - `negative_prompt`: Tailored negative prompt (when enabled).
  - `reasoning_summary`: Reasoning statistics for canvas display.
  - `prompt_g`: CLIP-G prompt (populated for SDXL dual mode, falls back to expanded prompt).
  - `prompt_l`: CLIP-L prompt (populated for SDXL dual mode, falls back to expanded prompt).

---

## Multiple Reference Images

Meta Muse Image supports multiple reference images to guide generation, style, character consistency, and composition.

You can provide multiple reference images in two ways:
1. **Using `Meta Muse Image Array`**:
   - Connect up to 4 images of **any different sizes/aspect ratios** directly to `image_1` through `image_4`.
   - Wire the `image_array` output into `reference_images` on `Meta Muse Image` or `Meta Muse Image Editor`.
   - To use more than 4 images, chain multiple array nodes using the `image_array` input!
2. **Standard ComfyUI Batches**:
   - You can also plug a standard batched tensor (`[B, H, W, C]`) directly into `reference_image`. The node will automatically unpack all images in the batch.

## Iterative Editing (`accumulate_edits`)

The **Muse Image Editor** node enables stateful, conversational image refinement:

- Connect `response_id` from an upstream Muse node to `previous_response_id`.
- Set **`mode`** to:
  - **`accumulate_edits`** *(default)*: Chains subsequent prompt edits across executions in the current session without manual ID re-wiring.
  - **`reset_to_incoming_id`**: Discards the session history and restarts refinement from the incoming `previous_response_id`.

---

## Example Workflows

An example workflow is included in the [`example_workflows/`](example_workflows/) folder:
- **`metamuse-full-workflow.json`**: Complete pipeline showing initial generation (`MuseImageNode`), multi-turn editing (`MuseImageEditorNode`), output switching (`MuseSwitchNode`), reasoning visualization (`MuseShowTextNode`), and dynamic file naming using the active `response_id`.

