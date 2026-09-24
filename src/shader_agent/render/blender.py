import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class LoopException(Exception):
    """Recoverable Blender-side error (parse / render / validate)."""
    pass


class RepairExhausted(Exception):
    """In-step repair sub-loop used up its budget (default 2 attempts) without
    producing a renderable graph. Caught by the trial loop, which moves on to
    the next trial.
    """
    pass


# Cap concurrent one-shot SceneRenderer subprocess launches across the process.
# Param-tuning daemon renders are gated separately in agents/tuner.py.
_SCENE_RENDER_SEMAPHORE = threading.Semaphore(
    int(os.environ.get("SHADER_PIPELINE_SCENE_MAX_CONCURRENT", "1"))
)

def _bootstrap_import_path() -> None:
    here = Path(__file__).resolve().parent       # .../shader_agent/render
    pkg = here.parent                            # .../shader_agent
    src = pkg.parent                             # .../src
    for p in (src, pkg, here):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

_bootstrap_import_path()

try:
    from shader_agent.io import get_logger
except ImportError:
    from io import get_logger  # type: ignore

try:
    import bpy
    import mathutils # type: ignore
    IN_BLENDER = True
except ImportError:
    bpy: Any = None
    mathutils: Any = None
    IN_BLENDER = False


logger = get_logger(__name__)

class Preset(Enum):
    SCENE = "scene"
    PART = "part"


# Maps render_mode → default scene overrides (blend_file, part_name).
# Paths are relative to the project root, matching the YAML config convention.
RENDER_MODE_CONFIGS = {
    "ball": {
        "blend_file": "data/agent_ball.blend",
        "part_name": "SolidModel",
        "hidden_objects": [],
    },
    "plane": {
        "blend_file": "data/agent_plane.blend",
        "part_name": "SolidModel",
        "hidden_objects": [],
    },
    # VLMScorer training-aligned variants: same ball blend, but `dark_ball`
    # hides the main light to expose dark-side PBR response (spec/metallic).
    # The blend must contain an object literally named "MainLight" for the
    # dark variant to differ from the bright one — override in YAML if your
    # blend uses a different name.
    "bright_ball": {
        "blend_file": "data/agent_ball.blend",
        "part_name": "SolidModel",
        "hidden_objects": [],
    },
    "dark_ball": {
        "blend_file": "data/agent_ball.blend",
        "part_name": "SolidModel",
        "hidden_objects": ["MainLight"],
    },
}


_BLENDER_PRESETS = {
    Preset.SCENE: {
        "render": {
            "engine": "CYCLES",
            "samples": 128,
            "resolution": [256, 256],
            "device": "GPU",
            "use_denoising": True,
            "compute_device_type": "CUDA",
            "film_transparent": True,
        },
        "camera": {
            "azimuth_angles": [45],
            "elevation": 20,
            "distance": None,
            "front_axis": "-Y",
            "fov": 50.0,
        },
        "scene": {
            "normalize_size": 2.0,
            "environment_strength": 1.0,
            "hdri_path": None,
            "blend_file": None,
            "part_name": "SolidModel",
            "render_mode": "ball",
        },
    },
    Preset.PART: {
        "render": {
            "engine": "BLENDER_EEVEE",
            "samples": 1,
            "resolution": [256, 256],
            "device": "GPU",
            "use_denoising": False,
            "compute_device_type": "CUDA",
            "film_transparent": False,
        },
        "camera": {
            "azimuth_angles": [45],
            "elevation": 20,
            "distance": None,
            "front_axis": "-Y",
            "fov": 50.0,
        },
        "scene": {
            "normalize_size": 2.0,
            "environment_strength": 0.0,
            "hdri_path": None,
            "blend_file": None,
            "part_name": "SolidModel",
        },
    },
}


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


WANTED_COLOR = ("Green", "#008000")
UNWANTED_COLOR = ("Red", "#FF0000")
HIDDEN_COLOR = ("Black", "#000000")

# Colors that are too similar to Red/Green/Black and must be excluded from
# the unchecked palette to avoid VLM confusion.
_RESERVED_COLOR_NAMES = frozenset({"Red", "Green", "Black", "Lime", "Crimson", "Olive"})


def get_safe_vlm_palette() -> List[Tuple[str, str]]:
    """Full palette of distinctive colors (some may overlap with reserved)."""
    return [
        ("Red", "#FF0000"), ("Blue", "#0000FF"), ("White", "#FFFFFF"),
        ("Yellow", "#FFFF00"), ("Green", "#008000"), ("Magenta", "#FF00FF"),
        ("Pink", "#FF9EC4"), ("Cyan", "#00FFFF"), ("Orange", "#FF8500"),
        ("Lime", "#00FF00"), ("Purple", "#702c8c"), ("Navy", "#1B2E57"),
        ("Teal", "#008080"), ("Crimson", "#8f121d"), ("Olive", "#708238"),
        ("Grey", "#808080"), ("Dirt", "#5c4033"), ("Beige", "#f8e1c0"),
    ]


class RenderEngine(Enum):
    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE"


@dataclass
class RenderConfig:
    engine: str  # CYCLES, BLENDER_EEVEE
    samples: int
    use_denoising: bool
    film_transparent: bool
    resolution: Tuple[int, int] = (512, 512)
    device: str = "GPU"
    compute_device_type: str = "CUDA"  # CUDA, OPTIX, or NONE (CPU)
    threads: Optional[int] = None  # CPU thread limit, None = unlimited

    def get_engine(self) -> RenderEngine:
        return RenderEngine.CYCLES if self.engine == "CYCLES" else RenderEngine.EEVEE


def resolve_compute_device_type(device: str, requested: Optional[str] = None) -> str:
    """CPU rendering needs no compute backend; otherwise honour the request, else the platform default."""
    if device.upper() == "CPU":
        return "NONE"
    if requested:
        return requested.upper()
    return "CUDA"


@dataclass
class CameraConfig:
    azimuth_angles: List[int] = field(default_factory=lambda: [45])
    elevation: int = 20
    distance: Optional[float] = None
    front_axis: str = "-Y"
    fov: float = 50.0


@dataclass
class SceneConfig:
    environment_strength: float
    normalize_size: float = 2.0
    hdri_path: Optional[str] = None
    blend_file: Optional[str] = None
    part_name: str = "SolidModel"
    render_mode: str = "ball"
    hidden_objects: List[str] = field(default_factory=list)


class BlenderOps(ABC):
    
    def __init__(self, executable_path: str = "blender"):
        self.executable_path = executable_path

    @staticmethod
    def _build_configs_from_preset(preset_cfg) -> Tuple[RenderConfig, SceneConfig, CameraConfig]:
        render_config = RenderConfig(**preset_cfg["render"])
        scene_config = SceneConfig(**preset_cfg["scene"])
        camera_config = CameraConfig(**preset_cfg["camera"])
        return render_config, scene_config, camera_config
    
    @staticmethod
    def _require_bpy() -> None:
        if not IN_BLENDER:
            raise RuntimeError("This function requires running inside Blender (bpy available).")
    
    def _clear_scene(self) -> None:
        self._require_bpy()
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    
    def _set_frame_zero(self) -> None:
        self._require_bpy()
        scene = bpy.context.scene
        scene.frame_set(0)
        scene.frame_start = 0
        if scene.frame_end < 0:
            scene.frame_end = 0
    
    def _get_mesh_objects(self) -> List:
        self._require_bpy()
        return [obj for obj in bpy.data.objects if obj.type == "MESH"]
    
    def _find_root_objects(self) -> List:
        self._require_bpy()
        return [obj for obj in bpy.context.scene.objects if obj.parent is None]
    
    def _load_model(self, model_path: str) -> bool:
        if not IN_BLENDER:
            return False

        if not os.path.exists(model_path):
            logger.error("Model file not found: %s", model_path)
            return False
            
        ext = os.path.splitext(model_path)[1].lower()
        try:
            if ext == '.blend':
                bpy.ops.wm.open_mainfile(filepath=model_path)
            elif ext in ['.glb', '.gltf']:
                view_layer = bpy.context.view_layer
                window = bpy.context.window_manager.windows[0] if bpy.context.window_manager.windows else None
                override_context = {
                    'scene': bpy.context.scene,
                    'view_layer': view_layer,
                    'window': window,
                }
                with bpy.context.temp_override(**override_context):
                    bpy.ops.import_scene.gltf(filepath=model_path)
            elif ext == '.obj':
                bpy.ops.import_scene.obj(filepath=model_path)
            elif ext == '.fbx':
                bpy.ops.import_scene.fbx(filepath=model_path)
            else:
                logger.error("Unsupported file format: %s", ext)
                return False
            return True
        except Exception as e:
            logger.error("Error loading model: %s", e)
            return False
    
    @staticmethod
    def _compute_world_bbox(objects):
        BlenderOps._require_bpy()
        
        all_corners = []
        for obj in objects:
            if obj.type != 'MESH':
                continue
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            all_corners.extend(bbox_corners)
        
        if not all_corners:
            return mathutils.Vector((0, 0, 0)), mathutils.Vector((0, 0, 0))
        
        min_coord = mathutils.Vector((
            min(corner.x for corner in all_corners),
            min(corner.y for corner in all_corners),
            min(corner.z for corner in all_corners)
        ))
        max_coord = mathutils.Vector((
            max(corner.x for corner in all_corners),
            max(corner.y for corner in all_corners),
            max(corner.z for corner in all_corners)
        ))
        return min_coord, max_coord
    
    def _normalize_objects(self, objects, target_size: float = 2.0, reference_objects=None):
        # Scale via a parent Empty so animated objects normalize too; scaling
        # meshes directly can distort them.
        self._require_bpy()
        
        if not objects:
            return 1.0, mathutils.Vector((0, 0, 0)), None
        
        objects_for_bounds = reference_objects if reference_objects else objects
        min_coord, max_coord = self._compute_world_bbox(objects_for_bounds)
        center = (min_coord + max_coord) / 2
        dimensions = max_coord - min_coord
        max_dim = max(dimensions)
        
        logger.info("Original bounds: min=%s, max=%s, max_dim=%.4f", min_coord, max_coord, max_dim)
        
        scale_factor = (target_size * 0.95) / max_dim if max_dim > 0 else 1.0
        
        parent_empty = bpy.data.objects.new("GlobalScaler", None)
        bpy.context.scene.collection.objects.link(parent_empty)
        parent_empty.scale = (scale_factor, scale_factor, scale_factor)
        parent_empty.location = -center * scale_factor
        
        bpy.context.view_layer.update()
        
        def get_root(o):
            while o.parent is not None:
                o = o.parent
            return o
        
        root_objects = {get_root(obj) for obj in objects}
        for root in root_objects:
            root.parent = parent_empty
            root.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        
        bpy.context.view_layer.update()
        
        min_after, max_after = self._compute_world_bbox(objects_for_bounds)
        dims_after = max_after - min_after
        logger.info("Normalized: scale=%.4f, bounds min=%s, max=%s, max_dim=%.4f",
                    scale_factor, min_after, max_after, max(dims_after))
        
        return scale_factor, center, parent_empty
    
    def _ensure_camera(self, name: str = "Camera"):
        self._require_bpy()
        
        cam_obj = bpy.data.objects.get(name)
        if cam_obj and cam_obj.type == "CAMERA":
            return cam_obj

        cam_data = bpy.data.cameras.new(name)
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        return cam_obj
    
    def _setup_camera_spherical(self, azimuth: float, elevation: float, distance: float, lookat=None):
        self._require_bpy()
        
        if lookat is None:
            lookat = mathutils.Vector((0, 0, 0))
        
        camera = self._ensure_camera()
        
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)
        
        x = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = distance * math.sin(elevation_rad)
        
        camera.location = lookat + mathutils.Vector((x, y, z))
        
        direction = lookat - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        
        bpy.context.scene.camera = camera
        return camera
    
    def _setup_camera_axis(self, front_axis: str, distance: float, lookat=None):
        self._require_bpy()
        
        if lookat is None:
            lookat = mathutils.Vector((0, 0, 0))
        
        camera = self._ensure_camera()
        
        axis_mapping = {
            "+X": mathutils.Vector((1, 0, 0)),
            "-X": mathutils.Vector((-1, 0, 0)),
            "+Y": mathutils.Vector((0, 1, 0)),
            "-Y": mathutils.Vector((0, -1, 0)),
            "+Z": mathutils.Vector((0, 0, 1)),
            "-Z": mathutils.Vector((0, 0, -1)),
        }
        direction = axis_mapping.get(front_axis.upper(), axis_mapping["-Y"]).normalized()
        camera.location = lookat + direction * distance
        
        rot = (lookat - camera.location).to_track_quat("-Z", "Y")
        camera.rotation_euler = rot.to_euler()
        
        bpy.context.scene.camera = camera
        return camera
    
    @staticmethod
    def _calculate_camera_distance(target_size: float = 2.0, fov: float = 50.0) -> float:
        # sqrt(3) accounts for the bounding-cube diagonal; 1.1 is slack.
        fov_rad = math.radians(fov)
        return (target_size / 2 * math.sqrt(3)) / math.tan(fov_rad / 2) * 1.1
    
    @staticmethod
    def _compute_camera_distance_for_bounds(reference_objects, fov: float, azimuth: float, elevation: float, lookat) -> float:
        BlenderOps._require_bpy()
        
        if not reference_objects:
            return BlenderOps._calculate_camera_distance(fov=fov)
            
        fov_rad = math.radians(fov)
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)
        
        # Camera sits on a sphere around lookat and looks back toward it.
        direction = mathutils.Vector((
            -math.cos(elevation_rad) * math.cos(azimuth_rad),
            -math.cos(elevation_rad) * math.sin(azimuth_rad),
            -math.sin(elevation_rad)
        )).normalized()
        
        # Use Blender's built-in tracking rotation to ensure consistency with _setup_camera_spherical
        rot_mat = direction.to_track_quat('-Z', 'Y').to_matrix()
        
        # Camera local axes transformed to world space: Right (X), Up (Y), Forward (-Z)
        right = rot_mat @ mathutils.Vector((1, 0, 0))
        cam_up = rot_mat @ mathutils.Vector((0, 1, 0))
        cam_forward = rot_mat @ mathutils.Vector((0, 0, -1))
        
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        max_z_towards_cam = float('-inf')
        
        scene = bpy.context.scene
        render = scene.render
        aspect_ratio = render.resolution_x / render.resolution_y
        
        # Blender's Auto sensor fit applies the configured FOV to the longer
        # axis, so derive the other axis from the aspect ratio.
        if aspect_ratio >= 1.0:
            fov_x = fov_rad
            fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect_ratio)
        else:
            fov_y = fov_rad
            fov_x = 2.0 * math.atan(math.tan(fov_y / 2.0) * aspect_ratio)

        for obj in reference_objects:
            if obj.type != 'MESH':
                continue
                
            matrix_world = obj.matrix_world
            mesh = obj.data
            
            # All vertices, not obj.bound_box: the 2D fit must be tight.
            for vertex in mesh.vertices:
                vertex_world = matrix_world @ vertex.co
                v = vertex_world - lookat
                
                x = v.dot(right)
                y = v.dot(cam_up)
                z = v.dot(cam_forward) # positive z is extending towards the camera
                
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                max_z_towards_cam = max(max_z_towards_cam, z)
        
        width = max_x - min_x
        height = max_y - min_y
        
        dist_x = (width / 2.0) / math.tan(fov_x / 2.0) if width > 0 else 0
        dist_y = (height / 2.0) / math.tan(fov_y / 2.0) if height > 0 else 0
        
        # Take the maximum required distance, adding the max depth so nothing clips the near plane
        target_dist = max(dist_x, dist_y) + max(0, max_z_towards_cam)
        
        # Floor the distance so a very flat object can't swallow the camera.
        min_dist = max(width, height) * 0.1
        return max(target_dist, min_dist)
    
    def _create_emission_material(self, name: str, color: Tuple[float, float, float], strength: float = 1.0):
        self._require_bpy()
        
        mat = bpy.data.materials.get(name)
        if mat is not None:
            return mat
        
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = (*color, 1.0)
        emission.inputs['Strength'].default_value = strength
        emission.location = (0, 0)
        
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
        return mat
    
    def _assign_material(self, obj, material) -> bool:
        self._require_bpy()
        
        if isinstance(obj, str):
            obj = bpy.data.objects.get(obj)
        
        if obj is None or obj.type != "MESH":
            return False

        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
        return True
    
    def _setup_render_output(self, config: RenderConfig) -> None:
        self._require_bpy()
        scene = bpy.context.scene
        scene.render.resolution_x = config.resolution[0]
        scene.render.resolution_y = config.resolution[1]
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.film_transparent = config.film_transparent
    
    def _setup_cycles(self, config: RenderConfig) -> None:
        self._require_bpy()
        scene = bpy.context.scene
        scene.render.engine = RenderEngine.CYCLES.value
        
        cycles = scene.cycles
        cycles.device = config.device
        cycles.samples = config.samples
        cycles.use_denoising = config.use_denoising
        
        if config.threads is not None:
            scene.render.threads_mode = 'FIXED'
            scene.render.threads = config.threads
            logger.info("CPU thread limit set to: %d", config.threads)
        else:
            scene.render.threads_mode = 'AUTO'
        
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons['cycles'].preferences
        cycles_prefs.compute_device_type = config.compute_device_type
        
        cycles_prefs.get_devices()
        logger.info("Available compute devices: %s", [d.name for d in cycles_prefs.devices])
        
        enabled_devices = []
        for device in cycles_prefs.devices:
            if config.device == 'GPU' and device.type == 'CPU':
                device.use = False
                continue
            device.use = True
            enabled_devices.append(device.name)
            
        logger.info("Enabled compute devices: %s", enabled_devices)
    
    def _setup_eevee(self, config: RenderConfig) -> None:
        self._require_bpy()
        scene = bpy.context.scene
        scene.render.engine = RenderEngine.EEVEE.value
        
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = config.samples
    
    def _setup_renderer(self, config: RenderConfig) -> None:
        self._require_bpy()
        self._setup_render_output(config)
        
        engine = config.get_engine()
        if engine == RenderEngine.CYCLES:
            self._setup_cycles(config)
        elif engine == RenderEngine.EEVEE:
            self._setup_eevee(config)
        
        logger.info("%s renderer configured", engine.value)
    
    def _setup_world_background(
        self,
        color: Tuple[float, float, float, float] = (0, 0, 0, 1),
        strength: float = 0.0,
        hdri_path: Optional[str] = None
    ) -> None:
        self._require_bpy()
        
        # Ensure absolute path for HDRI to avoid missing texture (pink render)
        if hdri_path:
            hdri_path = os.path.abspath(hdri_path)
        
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        
        bg_node = nodes.new(type='ShaderNodeBackground')
        out_node = nodes.new(type='ShaderNodeOutputWorld')
        
        if hdri_path and os.path.exists(hdri_path):
            env_tex = nodes.new(type="ShaderNodeTexEnvironment")
            env_tex.image = bpy.data.images.load(hdri_path)
            links.new(env_tex.outputs["Color"], bg_node.inputs["Color"])
            bg_node.inputs["Strength"].default_value = strength
            logger.info("Loaded HDRI: %s", hdri_path)
        else:
            bg_node.inputs['Color'].default_value = color
            bg_node.inputs['Strength'].default_value = strength
        
        links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])
    
    def _render_to_file(self, output_path: str, resolution: Tuple[int, int] = (512, 512)) -> float:
        self._require_bpy()
        
        scene = bpy.context.scene
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.filepath = output_path
        
        start_time = time.time()
        bpy.ops.render.render(write_still=True)
        return time.time() - start_time
    
    @abstractmethod
    def _execute_from_args(self, args: argparse.Namespace) -> None:
        pass


class SceneRenderer(BlenderOps):
    """High-quality scene renderer built on Cycles."""

    @classmethod
    def default_configs(cls) -> Tuple[RenderConfig, SceneConfig, CameraConfig]:
        return cls._build_configs_from_preset(_BLENDER_PRESETS[Preset.SCENE])
    
    def __init__(
        self, 
        executable_path: str = "blender",
        render_config: Optional[RenderConfig] = None,
        scene_config: Optional[SceneConfig] = None,
        camera_config: Optional[CameraConfig] = None,
    ):
        super().__init__(executable_path)
        default_render, default_scene, default_camera = self.default_configs()
        self.render_config = render_config or default_render
        self.scene_config = scene_config or default_scene
        self.camera_config = camera_config or default_camera
    
    def _create_material_from_script(self, script: str):
        # The script must define create_material() returning a bpy.types.Material;
        # a top-level `mat` is accepted as a fallback.
        self._require_bpy()
        
        if os.path.isfile(script) and script.endswith(".py"):
            with open(script, "r", encoding="utf-8") as f:
                script = f.read()

        script_locals = {}
        try:
            exec(script, globals(), script_locals)
        except Exception as e:
            raise LoopException(f"Blender raised an exception: \n{e}")
        
        mat = None
        if "create_material" in script_locals and callable(script_locals["create_material"]):
            mat = script_locals["create_material"]()
        elif "mat" in script_locals:
            mat = script_locals["mat"]
        
        if mat is None:
            raise LoopException("Material not found. Script must define 'create_material' function.")
        if not isinstance(mat, bpy.types.Material):
            raise LoopException(f"Expected bpy.types.Material, got {type(mat)}.")

        return mat

    # Shared per-render core for both the one-shot CLI path (_execute_from_args)
    # and the persistent daemon (daemon.py); one implementation keeps the two
    # invocation modes from drifting apart.

    def apply_material_and_render(
        self,
        bpy_script: str,
        part_name: str,
        output_path: str,
        hidden_objects: Optional[List[str]] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Hide objects, build the material from the script, assign it, render → PNG.

        Returns wall-clock render seconds. Camera / world / renderer config must
        already be set up: one-shot does that in ``_execute_from_args``, the daemon
        once at startup.
        """
        self._require_bpy()

        for name in hidden_objects or []:
            obj = bpy.data.objects.get(name)
            if obj is None:
                logger.warning("hidden_objects: %s not found in scene", name)
                continue
            obj.hide_render = True
            try:
                obj.hide_set(True)
            except Exception:
                pass

        material = self._create_material_from_script(bpy_script)

        logger.info("Assigning material to: %s", part_name)
        if not self._assign_material(part_name, material):
            raise LoopException(f"Failed to assign material to '{part_name}'")

        res = resolution or self.render_config.resolution
        logger.info("Rendering to: %s", output_path)
        duration = self._render_to_file(output_path, res)
        logger.info("Render complete in %.2fs", duration)
        return duration

    def reset_state_for_new_render(self, material_name: Optional[str] = None) -> None:
        """Clear state that would otherwise leak between daemon renders.

        Removes the same-named material, so ``bpy.data.materials.new(name=...)``
        keeps the exact name instead of getting a ".001" suffix, and clears every
        hide_render flag, so a render never inherits a hidden MainLight.
        No-op in one-shot mode, where each subprocess starts fresh.
        """
        self._require_bpy()
        if material_name:
            old = bpy.data.materials.get(material_name)
            if old is not None:
                bpy.data.materials.remove(old, do_unlink=True)
        for obj in bpy.data.objects:
            obj.hide_render = False
            try:
                obj.hide_set(False)
            except Exception:
                pass

    def render(
        self,
        bpy_script: str,
        output_path: str,
        blend_file: Optional[str] = None,
        part_name: str = "SolidModel",
        model_path: Optional[str] = None,
        debug: bool = False,
        render_mode: Optional[str] = None,
        focus_target: Optional[str] = "all",
    ) -> str:
        render_mode = render_mode or self.scene_config.render_mode

        hidden_objects = list(self.scene_config.hidden_objects)
        if render_mode in RENDER_MODE_CONFIGS:
            mode_cfg = RENDER_MODE_CONFIGS[render_mode]
            if not blend_file:
                blend_file = mode_cfg.get("blend_file") or self.scene_config.blend_file
            if not hidden_objects:
                hidden_objects = list(mode_cfg.get("hidden_objects", []))

        if blend_file and not os.path.isabs(blend_file) and not os.path.isfile(blend_file):
            rooted = Path(__file__).resolve().parents[3] / blend_file
            if rooted.is_file():
                blend_file = str(rooted)

        samples = self.render_config.samples
        resolution = self.render_config.resolution
        device = self.render_config.device
        hdri_path = self.scene_config.hdri_path
        
        azimuth_angles = self.camera_config.azimuth_angles
        elevation = self.camera_config.elevation
        azimuth = azimuth_angles[0] if azimuth_angles else None
        
        cmd = [self.executable_path, "-b", "-noaudio"]
        
        if blend_file:
            cmd.append(blend_file)
        
        cmd.extend(["-P", __file__, "--"])
        cmd.extend(["--renderer", "scene"])
        cmd.extend(["--bpy_script", bpy_script])
        cmd.extend(["--part_name", part_name])
        cmd.extend(["--output_path", output_path])
        cmd.extend(["--render_mode", render_mode])
        if hidden_objects:
            cmd.extend(["--hidden_objects", ",".join(hidden_objects)])
        cmd.extend(["--samples", str(samples)])
        cmd.extend(["--resolution", f"{resolution[0]},{resolution[1]}"])
        cmd.extend(["--device", device])
        cmd.extend(["--compute_device_type", self.render_config.compute_device_type])
        cmd.extend(["--focus_target", str(focus_target)])

        if azimuth is not None:
            cmd.extend(["--azimuth_angles", str(azimuth)])
        if elevation is not None:
            cmd.extend(["--elevation", str(elevation)])

        if render_mode == "new_scene":
            assert model_path is not None, "model_path required for new_scene mode"
            cmd.extend(["--model_path", model_path])
        
        if hdri_path:
            cmd.extend(["--hdri", hdri_path])
        
        if debug:
            cmd.append("--debug")
        
        logger.info("Launching Blender: %s", " ".join(cmd))

        try:
            with _SCENE_RENDER_SEMAPHORE:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Blender stdout:\n%s", result.stdout)
            if result.stderr:
                logger.warning("Blender stderr:\n%s", result.stderr)
            return output_path
        except subprocess.CalledProcessError as e:
            e.cmd = [arg[:200] + '... [truncated]' if isinstance(arg, str) and len(arg) > 200 else arg for arg in e.cmd]
            if e.stdout:
                logger.error("Blender stdout:\n%s", e.stdout[-200:])
            if e.stderr:
                logger.error("Blender stderr:\n%s", e.stderr[-200:])
            if e.returncode == 2:
                logger.error("Recoverable Blender error (exit code 2)")
                raise LoopException(e.stderr)
            logger.error("Blender execution failed (exit code %d)", e.returncode)
            raise RuntimeError(f"Blender execution failed (exit code {e.returncode})") from e
    
    def _execute_from_args(self, args: argparse.Namespace) -> None:
        self._require_bpy()
        
        resolution = args.resolution
        
        if args.render_mode not in ("ball", "plane", "bright_ball", "dark_ball"):
            if not args.model_path:
                logger.error("--model_path required for non-ball mode.")
                sys.exit(1)
            if not args.hdri:
                logger.error("--hdri required for non-ball mode.")
                sys.exit(1)

            self._clear_scene()
            self._set_frame_zero()
            
            logger.info("Loading model: %s", args.model_path)
            if not self._load_model(args.model_path):
                logger.error("Failed to load model.")
                sys.exit(1)

            mesh_objects = self._get_mesh_objects()
            if not mesh_objects:
                logger.error("No mesh objects found.")
                sys.exit(1)

            reference_objects = None
            if args.focus_target == "part":
                reference_objects = [obj for obj in mesh_objects if obj.name == args.part_name]
                if not reference_objects:
                    logger.error("Target part '%s' not found for focus", args.part_name)
                    sys.exit(2) # This problem is recoverable by rerunning the part merger
            elif args.focus_target == "all":
                reference_objects = mesh_objects

            self._normalize_objects(mesh_objects, target_size=self.scene_config.normalize_size, reference_objects=reference_objects)
            
            # After _normalize_objects, the center of reference_objects is exactly at the origin.
            lookat = mathutils.Vector((0, 0, 0))
                
            azimuth = float(args.azimuth_angles.split(",")[0]) if args.azimuth_angles else self.camera_config.azimuth_angles[0]
            elevation = args.elevation if args.elevation is not None else self.camera_config.elevation
            fov = self.camera_config.fov
            
            camera_distance = self.camera_config.distance
            
            if camera_distance is None:
                if reference_objects:
                    camera_distance = self._compute_camera_distance_for_bounds(reference_objects, fov, azimuth, elevation, lookat)
                else:
                    camera_distance = self._calculate_camera_distance(target_size=self.scene_config.normalize_size, fov=fov)

            if args.azimuth_angles:
                self._setup_camera_spherical(azimuth, elevation, camera_distance, lookat=lookat)
            else:
                self._setup_camera_axis(self.camera_config.front_axis, camera_distance, lookat=lookat)
                
            self._setup_world_background(
                hdri_path=args.hdri,
                strength=self.scene_config.environment_strength
            )
        else:
            self._ensure_camera()

        config = RenderConfig(
            engine="CYCLES",
            samples=args.samples,
            resolution=resolution,
            device=args.device,
            use_denoising=self.render_config.use_denoising,
            compute_device_type=resolve_compute_device_type(args.device, args.compute_device_type),
            film_transparent=True,
        )
        self._setup_renderer(config)

        if args.debug:
            debug_path = os.path.join("/tmp", f"debug_{time.strftime('%Y%m%d_%H%M%S')}.blend")
            bpy.ops.wm.save_as_mainfile(filepath=debug_path)
            logger.debug("Saved debug blend: %s", debug_path)

        hidden_names = [n.strip() for n in (args.hidden_objects or "").split(",") if n.strip()]
        try:
            self.apply_material_and_render(
                bpy_script=args.bpy_script,
                part_name=args.part_name,
                output_path=args.output_path,
                hidden_objects=hidden_names,
                resolution=resolution,
            )
        except LoopException:
            raise
        except Exception as e:
            logger.error("Render failed: %s", e)
            sys.exit(1)


class PartRenderer(BlenderOps):
    """Fast flat-color part preview renderer built on EEVEE."""

    @classmethod
    def default_configs(cls) -> Tuple[RenderConfig, SceneConfig, CameraConfig]:
        return cls._build_configs_from_preset(_BLENDER_PRESETS[Preset.PART])
    
    def __init__(
        self, 
        executable_path: str = "blender",
        render_config: Optional[RenderConfig] = None,
        scene_config: Optional[SceneConfig] = None,
        camera_config: Optional[CameraConfig] = None,
    ):
        super().__init__(executable_path)
        default_render, default_scene, default_camera = self.default_configs()
        self.render_config = render_config or default_render
        self.scene_config = scene_config or default_scene
        self.camera_config = camera_config or default_camera
    
    def _setup_flat_color_output(self) -> None:
        self._require_bpy()
        scene = bpy.context.scene
        
        if hasattr(scene, "eevee"):
            scene.eevee.use_bloom = False
            scene.eevee.use_gtao = False
            scene.eevee.use_ssr = False
            scene.eevee.use_volumetric_lights = False
        
        if hasattr(scene, "view_settings"):
            scene.view_settings.view_transform = 'Standard'
            scene.view_settings.look = 'None'
            scene.view_settings.exposure = 0.0
            scene.view_settings.gamma = 1.0
        
        if hasattr(scene.render, "dither_intensity"):
            scene.render.dither_intensity = 0.0
    
    def _apply_color_assignments(self, mesh_objects, color_assignments: Dict[str, Tuple[str, str]]) -> None:
        """Apply color assignments to mesh objects. Unassigned objects get HIDDEN_COLOR."""
        self._require_bpy()
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.data.materials:
                obj.data.materials.clear()

        for obj in mesh_objects:
            if obj.name in color_assignments:
                color_name, hex_color = color_assignments[obj.name]
                rgb = hex_to_rgb(hex_color)
                mat = self._create_emission_material(f"VLM_{color_name}_{obj.name}", rgb, strength=1.0)
            else:
                # Unassigned parts render black, blending into the background.
                rgb = hex_to_rgb(HIDDEN_COLOR[1])
                mat = self._create_emission_material(f"VLM_Hidden_{obj.name}", rgb, strength=1.0)
            self._assign_material(obj, mat)

    @staticmethod
    def auto_assign_colors(part_names: List[str]) -> Dict[str, Tuple[str, str]]:
        """Assign distinct palette colors to parts, skipping colors a VLM could
        confuse with Red/Green/Black. Parts beyond palette capacity are left out
        of the result, which renders them hidden (black).
        """
        palette = get_safe_vlm_palette()
        palette = [p for p in palette if p[0] not in _RESERVED_COLOR_NAMES]
        assignments = {}
        for i, name in enumerate(part_names):
            if i < len(palette):
                assignments[name] = palette[i]
        return assignments
    
    def render(
        self,
        model_path: str,
        output_dir: str,
        color_assignments: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        resolution = self.render_config.resolution
        azimuth_angles = self.camera_config.azimuth_angles
        elevation = self.camera_config.elevation

        cmd = [self.executable_path, "-b", "-noaudio"]
        cmd.extend(["-P", __file__, "--"])
        cmd.extend(["--renderer", "part"])
        cmd.extend(["--model_path", model_path])
        cmd.extend(["--output_dir", output_dir])
        cmd.extend(["--resolution", f"{resolution[0]},{resolution[1]}"])
        cmd.extend(["--elevation", str(elevation)])

        if azimuth_angles:
            cmd.extend(["--azimuth_angles", ",".join(map(str, azimuth_angles))])

        if color_assignments:
            serializable = {k: list(v) for k, v in color_assignments.items()}
            cmd.extend(["--color_assignments", json.dumps(serializable)])

        logger.info("Launching Blender: %s", " ".join(cmd))

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Blender stdout:\n%s", result.stdout)
            return self._parse_render_output(result.stdout)
        except subprocess.CalledProcessError as e:
            e.cmd = [arg[:200] + '... [truncated]' if isinstance(arg, str) and len(arg) > 200 else arg for arg in e.cmd]
            logger.error("Blender execution failed: %s", e)
            raise RuntimeError(f"Blender execution failed: {e}") from e
    
    def _parse_render_output(self, stdout: str) -> Dict[str, Any]:
        result = {'color_mapping': {}, 'rendered_images': []}
        for line in stdout.splitlines():
            if line.startswith("COLOR_MAP:"):
                parts = line[len("COLOR_MAP:"):].split("|")
                if len(parts) == 3:
                    result['color_mapping'][parts[0]] = {'color_name': parts[1], 'hex_color': parts[2]}
            elif line.startswith("RENDERED:"):
                result['rendered_images'].append(line[len("RENDERED:"):])
        return result
    
    def _execute_from_args(self, args: argparse.Namespace) -> None:
        self._require_bpy()

        input_path = Path(args.model_path)
        if not input_path.exists():
            logger.error("File not found: %s", input_path)
            sys.exit(1)

        if input_path.suffix.lower() not in ['.glb', '.gltf']:
            logger.error("Unsupported format: %s", input_path.suffix)
            sys.exit(1)

        bpy.ops.wm.read_factory_settings(use_empty=True)

        resolution = args.resolution
        config = RenderConfig(
            engine="BLENDER_EEVEE",
            samples=1,
            resolution=resolution,
            use_denoising=self.render_config.use_denoising,
            film_transparent=False,
        )
        self._setup_renderer(config)
        self._setup_flat_color_output()
        self._setup_world_background(color=(0, 0, 0, 1), strength=0.0)

        if not self._load_model(str(input_path)):
            logger.error("Failed to load model")
            sys.exit(1)

        mesh_objects = self._get_mesh_objects()
        if not mesh_objects:
            logger.error("No mesh objects found")
            sys.exit(1)

        logger.info("Found %d mesh objects", len(mesh_objects))
        self._normalize_objects(mesh_objects, target_size=self.scene_config.normalize_size)

        if args.color_assignments:
            raw = json.loads(args.color_assignments)
            color_assignments = {k: tuple(v) for k, v in raw.items()}
        else:
            part_names = [obj.name for obj in mesh_objects]
            color_assignments = self.auto_assign_colors(part_names)

        self._apply_color_assignments(mesh_objects, color_assignments)

        model_name = input_path.stem
        model_output_dir = Path(args.output_dir) / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        azimuth_angles = args.azimuth_angles if args.azimuth_angles else self.camera_config.azimuth_angles
        elevation = args.elevation if args.elevation is not None else self.camera_config.elevation
        camera_distance = self.camera_config.distance or self._calculate_camera_distance(
            target_size=self.scene_config.normalize_size,
            fov=self.camera_config.fov
        )

        rendered_images = []
        for azimuth in azimuth_angles:
            self._setup_camera_spherical(azimuth, elevation, camera_distance)
            filename = f"render_az{azimuth}_el{elevation}.png"
            output_path = str(model_output_dir / filename)
            logger.info("Rendering: %s", filename)
            self._render_to_file(output_path, resolution)
            rendered_images.append(output_path)

        # stdout protocol parsed by _parse_render_output; unassigned parts report
        # HIDDEN_COLOR.
        for obj in mesh_objects:
            if obj.name in color_assignments:
                color_name, hex_color = color_assignments[obj.name]
            else:
                color_name, hex_color = HIDDEN_COLOR
            print(f"COLOR_MAP:{obj.name}|{color_name}|{hex_color}")
        for img_path in rendered_images:
            print(f"RENDERED:{img_path}")

        logger.info("Rendering complete. Output: %s", model_output_dir)


def _parse_resolution(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) == 1:
        size = int(parts[0])
        return (size, size)
    return (int(parts[0]), int(parts[1]))


def _unpack_textures_to_disk(material, texture_entries, unpack_dir):
    """Ensure each entry's image exists on disk; return a new entry list with
    absolute filepaths. Entries that cannot be written (image missing, node gone)
    are passed through unchanged so the caller still sees them in the sidecar.

    Blender-only; imports bpy lazily so this module still imports outside Blender.
    """
    import bpy  # type: ignore

    # Rebuild the DSL node_id → image map here instead of reaching into
    # extract_texture_paths' internals, to keep the coupling loose.
    from shader_agent.dsl.graph import _BpyExtractor  # type: ignore
    extractor = _BpyExtractor()
    extractor.run(material)
    dsl_id_to_image = {}

    def _walk(tree):
        for bn in tree.nodes:
            if bn.bl_idname == "ShaderNodeTexImage":
                dsl_id = extractor._id_map.get(bn.as_pointer())
                if dsl_id and getattr(bn, "image", None) is not None:
                    dsl_id_to_image[dsl_id] = bn.image
            elif bn.bl_idname == "ShaderNodeGroup" and bn.node_tree is not None:
                _walk(bn.node_tree)
    _walk(getattr(material, "node_tree", material))

    os.makedirs(unpack_dir, exist_ok=True)
    rewritten = []
    for entry in texture_entries:
        node_id = entry.get("node_id")
        img = dsl_id_to_image.get(node_id)
        if img is None:
            rewritten.append(entry)
            continue

        orig_fp = entry.get("filepath", "") or ""
        ext = os.path.splitext(orig_fp)[1].lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp"):
            ext = ".png"
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in node_id)
        out_path = os.path.abspath(os.path.join(unpack_dir, f"{safe_id}{ext}"))

        wrote = False
        # Dump the ORIGINAL packed bytes: byte-identical to what the author packed,
        # no color management, no re-encoding. save_render() (used here before)
        # applies the scene view transform and silently alters pixel values, which
        # ruins normal / roughness / displacement maps.
        pf = getattr(img, "packed_file", None)
        if pf is not None:
            try:
                raw = bytes(pf.data)
                with open(out_path, "wb") as f:
                    f.write(raw)
                wrote = os.path.exists(out_path) and os.path.getsize(out_path) > 0
            except Exception as exc:
                logger.warning(
                    "packed_file.data dump failed for %s (%s): %s",
                    node_id, img.name, exc,
                )

        # Not packed (linked to a missing file, or generated): image.save() respects
        # the image's own colorspace and file_format without applying the scene
        # view transform.
        if not wrote:
            try:
                prev_fp = img.filepath_raw
                img.filepath_raw = out_path
                if not img.file_format or img.file_format == "":
                    img.file_format = "PNG"
                img.save()
                img.filepath_raw = prev_fp
                wrote = os.path.exists(out_path) and os.path.getsize(out_path) > 0
            except Exception as exc:
                logger.warning("image.save fallback failed for %s: %s", node_id, exc)

        if wrote:
            # Keep colorspace / alpha_mode / file_format; only filepath changes.
            new_entry = dict(entry)
            new_entry["node_id"] = node_id
            new_entry["filepath"] = out_path
            rewritten.append(new_entry)
        else:
            rewritten.append(entry)  # keep stale path; sidecar still records it
    return rewritten


class ShaderExtractor(BlenderOps):
    """Extract a shader DSL from a material in a .blend file.

    Launches Blender as a subprocess, walks the chosen material's node_tree,
    flattens ShaderNodeGroup instances inline, and writes:
      - the DSL text to ``output_path``
      - the texture-paths sidecar to ``output_path + '.textures.json'``
    """

    def extract(
        self,
        blend_file: str,
        material_name: Optional[str],
        output_path: str,
    ) -> Tuple[str, List[Dict[str, str]]]:
        cmd = [self.executable_path, "-b", "-noaudio", blend_file,
               "-P", __file__, "--",
               "--renderer", "extract",
               "--output_path", output_path]
        if material_name:
            cmd += ["--material_name", material_name]

        logger.info("Launching Blender (extract): %s", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info("Blender stdout:\n%s", result.stdout[-2000:])
        if result.stderr:
            logger.warning("Blender stderr:\n%s", result.stderr[-2000:])

        with open(output_path, "r", encoding="utf-8") as f:
            dsl = f.read()
        sidecar_path = output_path + ".textures.json"
        textures: List[Dict[str, str]] = []
        if os.path.exists(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as f:
                textures = json.load(f)
        return dsl, textures

    def _execute_from_args(self, args: argparse.Namespace) -> None:
        self._require_bpy()
        if not args.output_path:
            raise RuntimeError("--output_path is required for extract renderer")

        if args.material_name:
            mat = bpy.data.materials.get(args.material_name)
            if mat is None:
                available = [m.name for m in bpy.data.materials]
                raise RuntimeError(
                    f"Material '{args.material_name}' not found in .blend. "
                    f"Available: {available}"
                )
            logger.info("Extracting material: %s", mat.name)
        else:
            candidates = [m for m in bpy.data.materials if m.use_nodes]
            if not candidates:
                raise RuntimeError("No node-based material in .blend")
            mat = candidates[0]
            logger.info("Extracting first material: %s", mat.name)

        # Imports deferred so this module still imports outside Blender.
        from shader_agent.dsl.graph import (
            ShaderGraph, extract_texture_paths,
        )

        graph = ShaderGraph.from_bpy(mat)
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(graph.to_dsl())

        textures = extract_texture_paths(mat)

        # Blenderkit and friends pack image pixels into the .blend and leave
        # filepath pointing at a stale path that never existed on this machine,
        # so unpack every texture for the DSL → to_bpy → render flow.
        unpack_dir = args.output_path + ".textures"
        textures = _unpack_textures_to_disk(mat, textures, unpack_dir)

        with open(args.output_path + ".textures.json", "w", encoding="utf-8") as f:
            json.dump(textures, f, indent=2)

        logger.info(
            "Extracted %d nodes, %d links, %d textures → %s",
            len(graph.nodes), len(graph.links), len(textures), args.output_path,
        )


class MaterialSaver(BlenderOps):
    """Run a bpy material-creation script, then save the .blend to disk."""

    def save_blend(
        self,
        bpy_script: str,
        output_blend: str,
        material_name: str = "GeneratedMaterial",
    ) -> str:
        cmd = [self.executable_path, "-b", "-noaudio",
               "-P", __file__, "--",
               "--renderer", "save_blend",
               "--bpy_script", bpy_script,
               "--output_path", output_blend,
               "--material_name", material_name]
        logger.info("Launching Blender (save_blend): %s", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info("Blender stdout:\n%s", result.stdout[-2000:])
        if result.stderr:
            logger.warning("Blender stderr:\n%s", result.stderr[-2000:])
        return output_blend

    def _execute_from_args(self, args: argparse.Namespace) -> None:
        self._require_bpy()
        if not args.bpy_script or not args.output_path:
            raise RuntimeError(
                "--bpy_script and --output_path are required for save_blend"
            )
        # Empty file first, otherwise the default "Material" from startup.blend
        # holds args.material_name and Blender appends a ".001" suffix.
        bpy.ops.wm.read_factory_settings(use_empty=True)

        renderer = SceneRenderer()
        mat = renderer._create_material_from_script(args.bpy_script)
        if args.material_name and mat.name != args.material_name:
            # Free the target name if another material somehow grabbed it first.
            conflict = bpy.data.materials.get(args.material_name)
            if conflict is not None and conflict is not mat:
                bpy.data.materials.remove(conflict, do_unlink=True)
            mat.name = args.material_name
        if args.material_name and mat.name != args.material_name:
            raise RuntimeError(
                f"Failed to rename material to '{args.material_name}' "
                f"(got '{mat.name}'); another material with that name exists."
            )
        # Orphan datablocks (users == 0) are pruned on save. Set fake user so
        # the material persists even though nothing references it.
        mat.use_fake_user = True
        bpy.ops.wm.save_as_mainfile(filepath=args.output_path)
        logger.info("Saved material '%s' → %s", mat.name, args.output_path)


def main():
    parser = argparse.ArgumentParser(description="Blender Render Script")
    parser.add_argument("--renderer", type=str,
                        choices=["scene", "part", "extract", "save_blend"],
                        default="scene",
                        help="Renderer type: 'scene', 'part' (EEVEE), 'extract' (.blend→DSL), or 'save_blend'")
    parser.add_argument("--material_name", type=str, default=None,
                        help="Target material name (extract/save_blend)")
    
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument("--output_path", type=str, help="Path to save output")
    parser.add_argument("--output_dir", type=str, help="Output directory (part renderer)")
    parser.add_argument("--resolution", type=str, default="512,512", 
                        help="Resolution as 'W,H' or single value for square")
    parser.add_argument("--device", type=str, default='GPU', help="Render device (CPU/GPU)")
    parser.add_argument("--compute_device_type", type=str, default=None,
                        help="Cycles backend (CUDA/OPTIX/NONE); defaults to CUDA")
    
    parser.add_argument("--part_name", type=str, help="Name of the part to select")
    parser.add_argument("--samples", type=int, default=128, help="Render samples")
    parser.add_argument("--render_mode", type=str,
                        choices=["ball", "plane", "bright_ball", "dark_ball", "new_scene"],
                        default="ball")
    parser.add_argument("--hidden_objects", type=str, default="",
                        help="Comma-separated object names to hide before rendering")
    parser.add_argument("--debug", action="store_true", help="Save debug blend file")
    parser.add_argument("--hdri", type=str, help="Path to HDRI environment file")
    parser.add_argument("--bpy_script", type=str, help="The bpy script to execute inside Blender")
    parser.add_argument("--focus_target", type=str, choices=["all", "part"], default="all",
                        help="What to focus the camera on: 'all' (entire scene) or 'part' (specific part)")
    
    parser.add_argument("--azimuth_angles", type=str, help="Comma-separated azimuth angles")
    parser.add_argument("--elevation", type=int, default=20, help="Elevation angle")

    parser.add_argument("--color_assignments", type=str, help="JSON dict mapping part names to [color_name, hex] pairs")

    if not IN_BLENDER:
        args = parser.parse_args()
        args.resolution = _parse_resolution(args.resolution)
        
        if args.renderer == "scene":
            if not args.part_name or not args.output_path:
                parser.print_help()
                sys.exit(1)
            
            azimuth_angles = [int(float(args.azimuth_angles.split(",")[0]))] if args.azimuth_angles else []
            elevation = int(args.elevation) if args.elevation is not None else 20
            
            render_config, scene_config, camera_config = SceneRenderer.default_configs()
            render_config.samples = args.samples
            render_config.resolution = args.resolution
            render_config.device = args.device
            render_config.compute_device_type = resolve_compute_device_type(args.device, args.compute_device_type)
            scene_config.render_mode = args.render_mode
            scene_config.hdri_path = args.hdri
            camera_config.azimuth_angles = azimuth_angles
            camera_config.elevation = elevation
            
            renderer = SceneRenderer(
                render_config=render_config,
                scene_config=scene_config,
                camera_config=camera_config,
            )
            renderer.render(
                bpy_script=args.bpy_script or "",
                output_path=args.output_path,
                part_name=args.part_name,
                model_path=args.model_path,
                debug=args.debug,
            )
        elif args.renderer == "part":
            if not args.model_path or not args.output_dir:
                parser.print_help()
                sys.exit(1)
            
            azimuth_angles = [int(x) for x in args.azimuth_angles.split(",")] if args.azimuth_angles else [45, 135, 225, 315]
            elevation = int(args.elevation) if args.elevation is not None else 20
            
            render_config, scene_config, camera_config = PartRenderer.default_configs()
            render_config.resolution = args.resolution
            camera_config.azimuth_angles = azimuth_angles
            camera_config.elevation = elevation
            
            renderer = PartRenderer(
                render_config=render_config,
                scene_config=scene_config,
                camera_config=camera_config,
            )
            renderer.render(
                model_path=args.model_path,
                output_dir=args.output_dir,
            )
    else:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1:]
        else:
            argv = []
            
        if not argv:
            return

        args = parser.parse_args(argv)
        args.resolution = _parse_resolution(args.resolution)
        
        if args.renderer == "scene":
            if not args.part_name or not args.output_path:
                logger.error("Missing required arguments for scene renderer.")
                return
            ops = SceneRenderer()
        elif args.renderer == "part":
            if not args.model_path or not args.output_dir:
                logger.error("Missing required arguments for part renderer.")
                return
            if args.azimuth_angles:
                args.azimuth_angles = [int(x) for x in args.azimuth_angles.split(",")]
            ops = PartRenderer()
        elif args.renderer == "extract":
            if not args.output_path:
                logger.error("--output_path required for extract renderer.")
                return
            ops = ShaderExtractor()
        elif args.renderer == "save_blend":
            if not args.bpy_script or not args.output_path:
                logger.error("--bpy_script and --output_path required for save_blend.")
                return
            ops = MaterialSaver()
        else:
             logger.error("Unknown renderer: %s", args.renderer)
             return
        
        try:
            ops._execute_from_args(args)
        except LoopException as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
