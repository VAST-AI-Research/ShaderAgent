import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, Dict, List, Optional, Tuple

import yaml
from omegaconf import DictConfig, OmegaConf

from .render.blender import (
    _BLENDER_PRESETS,
    CameraConfig,
    Preset,
    RENDER_MODE_CONFIGS,
    RenderConfig,
    SceneConfig,
)


def resolve_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    """Pick the env var that matches the provider's wire format.

    ``openai_server`` → ``OPENAI_API_KEY``, ``anthropic`` → ``ANTHROPIC_API_KEY``,
    ``gemini`` → ``GOOGLE_API_KEY``. An explicit ``api_key`` in YAML wins over the
    environment.
    """
    if api_key:
        return api_key
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "gemini":
        return os.environ.get("GOOGLE_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


@dataclass
class ModelConfig:
    provider: str = "openai_server"
    model_id: str = "gpt-6-astra"
    api_base: Optional[str] = None
    api_key: Optional[str] = None


# Names kept so existing imports continue to work.
AgentConfig = ModelConfig
ModelServiceConfig = ModelConfig


@dataclass
class PipelineConfig:
    max_trials: int = 3
    max_steps: int = 3
    dataset_indices_path: Optional[str] = None
    dataset_limit: Optional[int] = None
    dataset_offset: Optional[int] = None
    param_tuning_enabled: bool = True
    dataset_workers: int = 1

@dataclass
class BlenderConfig:
    render: RenderConfig
    camera: CameraConfig
    scene: SceneConfig
    executable_path: str = "blender"
    assets_root: Optional[str] = None


def load_blender_config(yaml_path: Optional[str], preset: Preset = Preset.SCENE) -> BlenderConfig:
    """Load BlenderConfig from YAML, overlaying the named preset defaults."""
    base_cfg = OmegaConf.structured(BlenderConfig)
    
    preset_cfg = OmegaConf.create(_BLENDER_PRESETS[preset])
    base_cfg = OmegaConf.merge(base_cfg, preset_cfg)
    
    OmegaConf.set_struct(base_cfg, True)

    if yaml_path and Path(yaml_path).exists():
        file_cfg = OmegaConf.load(yaml_path)
        
        if isinstance(file_cfg, DictConfig):
            if "blender" in file_cfg:
                base_cfg = OmegaConf.merge(base_cfg, file_cfg["blender"])
    
            section_key = f"blender_{preset.value}"
            if section_key in file_cfg:
                base_cfg = OmegaConf.merge(base_cfg, file_cfg[section_key])

    if preset == Preset.SCENE:
        mode = base_cfg.scene.render_mode
        if mode in RENDER_MODE_CONFIGS and base_cfg.scene.blend_file is None:
            base_cfg.scene.blend_file = RENDER_MODE_CONFIGS[mode]["blend_file"]

    result = cast(BlenderConfig, OmegaConf.to_object(base_cfg))
    config_dir = Path(yaml_path).resolve().parent if yaml_path else Path.cwd()
    asset_root = Path(os.environ.get("SHADER_AGENT_ASSET_ROOT", "")) if os.environ.get("SHADER_AGENT_ASSET_ROOT") else None
    project_root = Path(__file__).resolve().parents[2]

    def resolve_asset(path: Optional[str]) -> Optional[str]:
        if not path:
            return path
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        roots = [config_dir]
        if result.assets_root:
            root = Path(result.assets_root).expanduser()
            roots.insert(0, root if root.is_absolute() else config_dir / root)
        if asset_root:
            roots.insert(0, asset_root)
        roots.append(project_root)
        for root in roots:
            relative = candidate
            if relative.parts and relative.parts[0] == root.name:
                relative = Path(*relative.parts[1:])
            resolved = root / relative
            if resolved.exists():
                return str(resolved.resolve())
        return str((roots[0] / candidate).resolve())

    result.scene.blend_file = resolve_asset(result.scene.blend_file)
    result.scene.hdri_path = resolve_asset(result.scene.hdri_path)
    return result

@dataclass
class ParamTuningConfig:
    max_iter: int = 50
    max_params: int = 5
    accept_prob: float = 0.05  # annealing: chance of accepting a worse-scoring step
    seed: int = 42
    # Ablation switch: sample tunable params at random instead of asking the LLM inspector.
    random_inspector: bool = False
    # Fast-renderer overrides used ONLY during tuning iterations: close enough to the
    # scorer's training domain while keeping per-iter wall time low.
    render_samples: int = 256
    render_resolution: int = 256
    # Perturbation velocity decay: 0.0 = memoryless, 0.9 = strong memory.
    momentum: float = 0.0
    # Log-uniform perturbation: exp(±log_range) is the scale factor range.
    # 2.3 ≈ ln(10) → 0.1x ~ 10x; 3.0 → 0.05x ~ 20x
    log_range: float = 2.3
    color_sv_perturb: float = 0.2
    hue_perturb: float = 0.15
    bool_flip_prob: float = 0.2


@dataclass
class CriticConfig:
    good_enough_min: float = 0.8

@dataclass
class ImageSimilarityMetricConfig:
    model_path: str = "Qwen/Qwen3-VL-Reranker-8B"
    adapter_path: Optional[str] = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    merge_lora: bool = True

@dataclass
class AppConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    blender_scene: BlenderConfig = field(default_factory=lambda: load_blender_config(None, Preset.SCENE))
    blender_part: BlenderConfig = field(default_factory=lambda: load_blender_config(None, Preset.PART))
    models: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "designer": ModelConfig(),
        "classifier": ModelConfig(),
        "critic": ModelConfig(),
        "inspector": ModelConfig(),
        "texture_generator": ModelConfig(),
        "prompt_enricher": ModelConfig(),
        "image_generator": ModelConfig(),
    })
    image_metrics: ImageSimilarityMetricConfig = field(default_factory=ImageSimilarityMetricConfig)
    param_tuning: ParamTuningConfig = field(default_factory=ParamTuningConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)

class Config:
    def __init__(self, config_path: str):
        self._config = self._load_config(config_path)

    def _load_config(self, path: str) -> AppConfig:
        config = AppConfig()
        if not path:
            raise ValueError("config path is empty")
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"config file not found: {config_path}")
        with open(config_path, 'r') as f:
                user_config_dict = yaml.safe_load(f) or {}
                
                if "pipeline" in user_config_dict:
                    for k, v in user_config_dict["pipeline"].items():
                        if hasattr(config.pipeline, k):
                            setattr(config.pipeline, k, v)

                config.blender_scene = load_blender_config(str(config_path), Preset.SCENE)
                config.blender_part = load_blender_config(str(config_path), Preset.PART)

                model_block: Dict[str, dict] = {}
                for section in ("agents", "services", "models"):
                    block = user_config_dict.get(section) or {}
                    if isinstance(block, dict):
                        model_block.update(block)
                for name, data in model_block.items():
                    if not isinstance(data, dict):
                        continue
                    if name not in config.models:
                        config.models[name] = ModelConfig()
                    model_conf = config.models[name]
                    for k, v in data.items():
                        if hasattr(model_conf, k):
                            setattr(model_conf, k, v)
                                
                if "image_metrics" in user_config_dict:
                    for k, v in user_config_dict["image_metrics"].items():
                        if hasattr(config.image_metrics, k):
                            setattr(config.image_metrics, k, v)

                if "param_tuning" in user_config_dict:
                    for k, v in user_config_dict["param_tuning"].items():
                        if hasattr(config.param_tuning, k):
                            setattr(config.param_tuning, k, v)

                if "critic" in user_config_dict:
                    for k, v in user_config_dict["critic"].items():
                        if hasattr(config.critic, k):
                            setattr(config.critic, k, v)

        self._validate_paths(config, config_path)

        for model in config.models.values():
            model.api_key = resolve_api_key(model.provider, model.api_key)
        return config

    @staticmethod
    def _validate_paths(config: AppConfig, config_path: Path) -> None:
        for label, value in (("scene.blend_file", config.blender_scene.scene.blend_file),
                             ("scene.hdri_path", config.blender_scene.scene.hdri_path),
                             ("part.blend_file", config.blender_part.scene.blend_file)):
            if value and not Path(value).is_file():
                raise FileNotFoundError(f"missing asset for {label} in {config_path}: {value}")

    @property
    def pipeline(self) -> PipelineConfig:
        return self._config.pipeline

    @property
    def blender_scene(self) -> BlenderConfig:
        return self._config.blender_scene

    @property
    def blender_part(self) -> BlenderConfig:
        return self._config.blender_part

    @property
    def models(self) -> Dict[str, ModelConfig]:
        return self._config.models

    @property
    def agents(self) -> Dict[str, ModelConfig]:
        return self._config.models

    @property
    def services(self) -> Dict[str, ModelConfig]:
        return self._config.models
    
    @property
    def image_metrics(self) -> ImageSimilarityMetricConfig:
        return self._config.image_metrics

    @property
    def param_tuning(self) -> ParamTuningConfig:
        return self._config.param_tuning

    @property
    def critic(self) -> CriticConfig:
        return self._config.critic
