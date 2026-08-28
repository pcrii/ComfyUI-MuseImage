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

- **Meta Muse Image (`MuseImageNode`)**: Text-to-image and image-to-image generation. Outputs the generated `IMAGE`, `response_id`, and `reasoning_summary`.
- **Meta Muse Image Editor / Refiner (`MuseImageEditorNode`)**: Iterative multi-turn image editing.
- **Meta Muse Show Text / Reasoning (`MuseShowTextNode`)**: Lightweight canvas display node to view reasoning logs and IDs.
- **Meta Muse Mode Switch (`MuseSwitchNode`)**: Routes between initial generation and iterative editor outputs (`IMAGE`, `reasoning_summary`, and `response_id`) to drive a single `SaveImage` / `MuseShowTextNode`, avoiding duplicate saved images and allowing dynamic response ID file naming.
- **Meta Muse Image Array (`MuseImageArrayNode`)**: Combines multiple reference images of different resolutions or aspect ratios into an image bundle (`MUSE_IMAGES`) without requiring resizing or cropping. Supports chaining for unlimited reference images.

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

