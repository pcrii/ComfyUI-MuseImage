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
- **Meta Muse Spark Prompt Expander (`MuseSparkPromptExpander`)**: Prompt expansion powered by Meta's Muse Spark reasoning models. Supports modern natural language or old-school CLIP tags, optional negative prompt generation, aesthetic presets, custom instruction overrides, and fake seed caching.
- **Meta Muse Show Text / Reasoning (`MuseShowTextNode`)**: Lightweight canvas display node to view reasoning logs, prompts, and IDs.
- **Meta Muse Mode Switch (`MuseSwitchNode`)**: Routes between initial generation and iterative editor outputs (`IMAGE`, `reasoning_summary`, and `response_id`) to drive a single `SaveImage` / `MuseShowTextNode`, avoiding duplicate saved images and allowing dynamic response ID file naming.
- **Meta Muse Image Array (`MuseImageArrayNode`)**: Combines multiple reference images of different resolutions or aspect ratios into an image bundle (`MUSE_IMAGES`) without requiring resizing or cropping. Supports chaining for unlimited reference images.

---

## Prompt Expansion (`MuseSparkPromptExpander`)

The **Muse Spark Prompt Expander** node transforms brief ideas into rich, high-fidelity prompts:

- **`prompt_format`**:
  - `natural_language (modern / flux / muse)`: Descriptive prose covering subject, lighting, composition, and texture.
  - `clip_l_tags (sd1.5 / sdxl / booru)`: Comma-separated CLIP tokens, quality tags, and booru keywords tailored for older or tag-based models.
  - `minimax_h3_fl2va (first frame + audio timeline)`: Formats into keyframe alignment header + 3-section video/audio timeline (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`). Diegetic singing/dialogue is tagged with speaker `(S1)` and `<d>[Language] lyrics</d>`, and latent-guided audio is guided via acoustic features (no `<Audio 1>` prompt tags).
  - `minimax_h3_ref2va (6-section multi-reference)`: Formats strictly into the official MiniMax H3 6-section rewrite structure (`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`) with tokenized `<Picture 1>`..`<Picture 9>` and `<Audio 1>`..`<Audio 3>`.
  - `minimax_h3 (video + audio timeline)`: Backwards-compatible alias for `fl2va`.
- **`preset`**: Includes `photorealistic`, `cinematic`, `digital_art / anime`, `general_expansion`, `minimax_h3_fl2va (first frame + audio guide)`, `minimax_h3_ref2va (multi-reference r2v)`, `minimax_h3 (video + audio director)` (alias), and `custom`.
- **`include_negative`**:
  - `False` *(default)*: Optimized for distilled / modern models. Instructs the model not to rely on negative prompts and embeds all quality directives directly into the positive prompt. Negative prompt output is empty (`""`).
  - `True`: Generates both an expanded positive prompt and a tailored negative prompt to eliminate common artifacts.
- **`seed`**: Standard ComfyUI seed widget (defaults to `fixed` so generated prompts stay locked across queues until you manually re-roll or randomize).
- **`reference_image` / `reference_images`**: Optional single image or array from `Meta Muse Image Array` (`MUSE_IMAGES`). Muse Spark's multimodal vision encoder directly inspects the visual features, composition, colors, and characters of your references to ground and enrich the expanded prompt.
- **`custom_instructions`**: Optional string input socket to override or supplement the prompt generation instructions.
- **Outputs**:
  - `expanded_prompt`: The expanded positive prompt.
  - `negative_prompt`: The negative prompt (when enabled).
  - `reasoning_summary`: Token statistics or reasoning summary (can be plugged into **Muse Show Text** to view on canvas).

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

