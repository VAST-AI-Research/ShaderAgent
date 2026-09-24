# Repository map

Python 3.10+. `src/` layout. Run commands from the repository root. GPU is required (Cycles CUDA).

| Path | Role |
|---|---|
| `src/shader_agent/pipeline.py` | Designer → DSL → render → param search → Critic |
| `src/shader_agent/llm/` | Chat Completions and Anthropic Messages |
| `src/shader_agent/dsl/graph.py` | DSL parse / validate / `to_bpy` |
| `src/shader_agent/render/` | Cycles subprocess renderer and tuning daemon |
| `src/shader_agent/tools.py` | `validate_dsl`, `render_dsl` |
| `src/shader_agent/mcp.py` | stdio MCP wrapping those two functions |
| `src/shader_agent/agents/` | Designer, Classifier, Critic, Tuner |
| `config/default.yaml` | gpt-6-astra over OpenAI-compatible `/v1` |
| `config/default-claude.yaml` | claude-opus-5 over Anthropic Messages |
| `data/` | `xmas.png`, `index.jsonl`, `agent_ball.blend`, `agent_plane.blend` |

Copy a config, then edit `model_id` or `api_base`. Keys stay in the environment: `openai_server` reads `OPENAI_API_KEY`, `anthropic` reads `ANTHROPIC_API_KEY`, `gemini` reads `GOOGLE_API_KEY`. Chat default is gpt-6-astra with gpt-image-2.5-flare for texture maps; the Claude config pairs claude-opus-5 with Gemini image generation. Paper results used Gemini 3.1 Pro Preview.

YAML uses a single `models:` map (designer, classifier, critic, inspector, …). `max_trials` = failed-trial retries; `max_steps` = structure rewrites; `param_tuning.max_iter` = numeric search per structure step. `scene.render_mode: ball` is the saved hero view; the tuner also renders plane / dark_ball for MatScorer, while the Critic sees only bright_ball.

## Commands

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m shader_agent --config config/default.yaml --image data/xmas.png --text "The wrapper of the present box." --output outputs/xmas
python -m shader_agent.batch --config config/default.yaml --tag outputs/demo --limit 1
```

MCP: `python -m shader_agent.mcp`. Tools are `validate_dsl` (`dsl: str`) and `render_dsl` (`dsl`, `output_path`, optional `render_mode`, `samples`, `resolution`, `compute_device_type`, `blender`).

```python
from shader_agent.tools import validate_dsl, render_dsl
```

`render_mode`: `ball`, `plane`, `bright_ball`, `dark_ball`.

The procedural path uses the chat client. `texture_generator` is built only on the TEXTURE route: `openai_server` goes through the OpenAI Images API, `gemini` through `google-genai`.

## Config fields that matter

- `pipeline.max_trials`, `max_steps`, `param_tuning_enabled`, `dataset_indices_path`
- `blender.executable_path`, `compute_device_type` (`CUDA` / `OPTIX`)
- `models.<role>.provider` / `model_id` / `api_base`
- `image_metrics.model_path` (HF id `Qwen/Qwen3-VL-Reranker-8B`), `adapter_path` (MatScorer LoRA, defaults to `yuanze1024/ShaderAgent-MatScorer-Qwen3VLReranker-LoRA`; `null` runs the base reranker)
- `param_tuning.max_iter` (numeric search steps per structure step; paper: 50)

Add a sample: drop a PNG next to `data/index.jsonl` and append a line `{"img_name":"...","textual_prompt":"...","sub_dir":".","unique_name":"..."}`.

Outputs go under the `--output` / `--tag` directory (`*_best.py`, `*_best.dsl`, renders, `usage.json`). Longer runs can use `outputs/<name>/`.

Match surrounding Python: four-space indent, `snake_case` functions, `PascalCase` classes. Environment defaults use `${VAR-default}` when a shell default is needed.
