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

---

## Iterative Editing (`accumulate_edits`)

The **Muse Image Editor** node enables stateful, conversational image refinement:

- Connect `response_id` from an upstream Muse node to `previous_response_id`.
- Set **`mode`** to:
  - **`accumulate_edits`** *(default)*: Chains subsequent prompt edits across executions in the current session without manual ID re-wiring.
  - **`reset_to_incoming_id`**: Discards the session history and restarts refinement from the incoming `previous_response_id`.
