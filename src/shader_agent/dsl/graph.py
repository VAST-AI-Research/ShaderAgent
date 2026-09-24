from copy import deepcopy
import ast
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import ROOT_DIR
from ..io import get_logger

logger = get_logger(__name__)

def _strip_comment(line: str) -> str:
    """Return *line* with any trailing ``# …`` comment removed.

    A ``#`` inside quotes is kept; an unterminated quote yields no comment.

    Hand-written scan rather than a regex: the previous pattern nested a
    quantifier over an alternation, so a non-matching line (an unbalanced
    quote, which models emit routinely) sent it into catastrophic
    backtracking — ~16x runtime per 4 extra characters, once hanging a real
    run for 42 minutes inside the parser.
    """
    quote: str | None = None
    for i, ch in enumerate(line):
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def _remove_comments(lines: List[str]) -> List[str]:
    """Strip full-line and inline ``# …`` comments, keeping ``#`` inside quotes."""
    cleaned: List[str] = []
    for line in lines:
        content = _strip_comment(line).rstrip()
        if content:
            cleaned.append(content)
    return cleaned


def _is_tunable(val) -> bool:
    """Check if a value is tunable for MC parameter search (numeric or bool)."""
    if isinstance(val, bool):
        return True
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, tuple) and 0 < len(val) <= 4:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in val)
    return False

@dataclass
class NodeInput:
    default: Optional[Any] = None


@dataclass
class ShaderNode:
    id: str
    node_type: str
    props: Dict[str, Any]
    inputs: Dict[str, NodeInput]
    input_slots: List[Tuple[str, NodeInput]] = field(default_factory=list)
    ramp_props: Dict[str, Any] = field(default_factory=dict)
    ramp_elements: List[Tuple[Any, float]] = field(default_factory=list)
    curve_points: List[Tuple[str, float, float]] = field(default_factory=list)

    def __post_init__(self):
        self._validate_key(self.id, "node id")
        for k in self.props:
            self._validate_key(k, "property key")
        for k in self.inputs:
            self._validate_key(k, "input key")

    def _validate_key(self, key: str, context: str):
        if re.match(r"^[a-zA-Z0-9_ ]+$", key):
            return
        if len(key) >= 2 and (
            (key.startswith("'") and key.endswith("'")) or (key.startswith('"') and key.endswith('"'))
        ):
            return
        raise ValueError(
            f"Invalid {context} '{key}': must consist only of letters, numbers, and underscores, or be enclosed in quotes."
        )

    def to_dsl(self) -> str:
        lines = [f"node {self.id} {self.node_type}"]
        for k, v in self.props.items():
            val_str = f"'{v}'" if isinstance(v, str) else str(v)
            lines.append(f"prop {k}={val_str}")

        slots = self.input_slots if self.input_slots else list(self.inputs.items())
        for k, v in slots:
            if v.default is not None:
                val_str = f"'{v.default}'" if isinstance(v.default, str) else str(v.default)
                lines.append(f"in {k}={val_str}")
            else:
                lines.append(f"in {k}")
        for channels, x, y in self.curve_points:
            lines.append(f"point '{channels}' {x} {y}")

        for k, v in self.ramp_props.items():
            val_str = f"'{v}'" if isinstance(v, str) else str(v)
            lines.append(f"ramp_prop {k}={val_str}")

        for color, pos in self.ramp_elements:
            c_str = f"'{color}'" if isinstance(color, str) else str(color)
            lines.append(f"ramp {c_str} {pos}")

        return "\n".join(lines)

    def set_input_value(self, name: str, value) -> None:
        """Set an input default, keeping ``inputs`` and ``input_slots`` in sync."""
        if name in self.inputs:
            self.inputs[name].default = value
        for slot_name, inp in self.input_slots:
            if slot_name == name:
                inp.default = value

    def set_prop_value(self, name: str, value) -> None:
        self.props[name] = value

    def input_slot_count(self, name: str) -> int:
        if self.input_slots:
            return sum(1 for slot_name, _ in self.input_slots if slot_name == name)
        return 1 if name in self.inputs else 0

    @staticmethod
    def from_block(lines: List[str]) -> "ShaderNode":
        cleaned = _remove_comments(lines)
        if not cleaned:
            raise ValueError("Empty data block for node parsing")

        header_parts = cleaned[0].split()
        if len(header_parts) < 3 or header_parts[0] != "node":
            raise ValueError(f"Invalid node header: '{cleaned[0]}'. Expected: node <id> <node_type>")

        node_id = header_parts[1]
        node_type = header_parts[2]
        props: Dict[str, Any] = {}
        inputs: Dict[str, NodeInput] = {}
        input_slots: List[Tuple[str, NodeInput]] = []
        ramp_props: Dict[str, Any] = {}
        ramp_elements: List[Tuple[Any, float]] = []
        curve_points: List[Tuple[str, float, float]] = []

        for line in cleaned[1:]:
            if line.startswith("prop "):
                key, val = ShaderNode._parse_assignment(line[5:])
                props[key] = ShaderNode._parse_literal(val)
            elif line.startswith("in "):
                body = line[3:]
                if "=" in body:
                    name, expr = ShaderNode._parse_assignment(body)
                    node_input = NodeInput(default=ShaderNode._parse_literal(expr))
                else:
                    name = body.strip()
                    if not name:
                        raise ValueError(f"Input declaration missing name in node '{node_id}'")
                    node_input = NodeInput(default=None)
                input_slots.append((name, node_input))
                inputs.setdefault(name, node_input)
            elif line.startswith("ramp "):
                # ramp <color> <position>
                content = line[5:].strip()
                last_space_idx = content.rfind(" ")
                if last_space_idx == -1:
                     raise ValueError(f"Invalid ramp definition in node '{node_id}': {line}")
                
                color_expr = content[:last_space_idx].strip()
                pos_expr = content[last_space_idx+1:].strip()
                
                color = ShaderNode._parse_literal(color_expr)
                pos = float(ShaderNode._parse_literal(pos_expr))
                ramp_elements.append((color, pos))

            elif line.startswith("ramp_prop "):
                key, val = ShaderNode._parse_assignment(line[len("ramp_prop "):])
                ramp_props[key] = ShaderNode._parse_literal(val)

            elif line.startswith("point "):
                # point <channel> <x> <y>
                parts = line[6:].strip().split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid point definition in node '{node_id}': {line}")
                
                channel = ShaderNode._parse_literal(parts[0])
                x = ShaderNode._parse_literal(parts[1])
                y = ShaderNode._parse_literal(parts[2])
                curve_points.append((channel, x, y))

            else:
                raise ValueError(f"Unrecognized line in node '{node_id}': {line}")

        return ShaderNode(
            id=node_id, 
            node_type=node_type, 
            props=props, 
            inputs=inputs, 
            input_slots=input_slots,
            ramp_props=ramp_props,
            ramp_elements=ramp_elements,
            curve_points=curve_points
        )

    @staticmethod
    def _parse_assignment(text: str) -> Tuple[str, str]:
        if "=" not in text:
            raise ValueError(f"Missing assignment operator '=' in: {text}")
        key, val = text.split("=", 1)
        return key.strip(), val.strip()

    @staticmethod
    def _parse_literal(val: str) -> Any:
        val = val.strip().strip("'\"")
        val_lower = val.lower()

        if val.endswith(";"):
             val = val[:-1].strip().strip("'\"")
             val_lower = val.lower()

        if val_lower == "true": return True
        if val_lower == "false": return False

        if val.startswith("#") and len(val) in (4, 7, 9):
            return ShaderNode._hex_to_rgba(val)

        # Unwrap "Vector(1, 2, 3)" / "Color[1, 0, 0]" into a plain tuple literal.
        match = re.search(r'(?:Vector|Color|RGB|vec3)?[\(\[]\s*(.*?)\s*[\)\]]', val, flags=re.IGNORECASE)
        if match:
            val = f"({match.group(1)})"

        try:
            parsed = ast.literal_eval(val)
            # bpy default_value assignment wants tuples, not lists.
            if isinstance(parsed, list):
                return tuple(float(x) for x in parsed)
            if isinstance(parsed, tuple):
                return tuple(float(x) for x in parsed)
            return parsed  # int or float
        except (ValueError, SyntaxError):
            pass

        # Fallback: bare "1.0, 2.0, 3.0"
        if "," in val:
            parts = [p.strip() for p in val.split(",") if p.strip()]
            try:
                return tuple(float(p) for p in parts)
            except ValueError:
                pass

        # Fallback: bare "1.0 1.0 1.0 1.0"
        parts = val.split()
        if len(parts) >= 2:
            try:
                return tuple(float(p) for p in parts)
            except ValueError:
                pass

        # Enum, texture path, or any other bare string.
        return val

    @staticmethod
    def _srgb_to_linear(c: float) -> float:
        """Convert one sRGB channel [0, 1] to linear light (IEC 61966-2-1)."""
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    @staticmethod
    def _hex_to_rgba(hex_str: str) -> Tuple[float, float, float, float]:
        """#RRGGBB / #RRGGBBAA → (R, G, B, A). R/G/B go sRGB→linear; alpha does not."""
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 3:
            hex_str = ''.join(c + c for c in hex_str)

        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = int(hex_str[6:8], 16) / 255.0 if len(hex_str) == 8 else 1.0

        return (
            ShaderNode._srgb_to_linear(r),
            ShaderNode._srgb_to_linear(g),
            ShaderNode._srgb_to_linear(b),
            a,
        )


@dataclass
class ShaderLink:
    source_node: str
    source_output: str
    target_node: str
    target_input: str

    @staticmethod
    def from_str(line: str) -> "ShaderLink":
        if not line.startswith("link "):
            raise ValueError(f"Invalid link line: {line}")
        payload = line[5:].strip()
        if "->" not in payload:
            raise ValueError(f"Missing '->' in link definition: {line}")
        src_text, dst_text = payload.split("->", 1)
        src_node, src_output = ShaderLink._parse_endpoint(src_text.strip(), "source")
        tgt_node, tgt_input = ShaderLink._parse_endpoint(dst_text.strip(), "target")
        return ShaderLink(source_node=src_node, source_output=src_output, target_node=tgt_node, target_input=tgt_input)

    @staticmethod
    def _parse_endpoint(text: str, endpoint_name: str) -> Tuple[str, str]:
        if "." not in text:
            raise ValueError(f"Invalid {endpoint_name} endpoint '{text}', expected <node>.<socket>")
        node, socket = text.split(".", 1)
        node = node.strip()
        socket = socket.strip()
        if not node or not socket:
            raise ValueError(f"Invalid {endpoint_name} endpoint '{text}', expected <node>.<socket>")
        return node, socket


@dataclass
class ParamRef:
    """Reference to a single tunable parameter on a node."""
    node_id: str
    param_name: str
    kind: str  # "input" or "prop"
    value: object  # float, int, bool, or tuple


class ShaderGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, ShaderNode] = {}
        self.links: List[ShaderLink] = []
        self.errors: List[str] = []
        self.subgraphs: Optional[Dict[str, "ShaderGraph"]] = None
        self.material_props: Dict[str, Any] = {}

    def get_linked_inputs(self) -> Set[Tuple[str, str]]:
        """Return the set of ``(node_id, input_name)`` pairs driven by links."""
        return {(lk.target_node, lk.target_input) for lk in self.links}

    def collect_tunable_params(self) -> List[ParamRef]:
        """Return every tunable (numeric / bool) parameter that is NOT linked.

        A linked input's default value has no effect, so it is excluded.
        """
        linked = self.get_linked_inputs()
        params: List[ParamRef] = []

        for node_id, node in self.nodes.items():
            slots = node.input_slots if node.input_slots else list(node.inputs.items())
            for name, inp in slots:
                if (node_id, name) in linked:
                    continue
                if inp.default is not None and _is_tunable(inp.default):
                    params.append(ParamRef(node_id, name, "input", inp.default))
            for pname, pval in node.props.items():
                if _is_tunable(pval):
                    params.append(ParamRef(node_id, pname, "prop", pval))

        return params

    def to_dsl(self) -> str:
        blocks = []
        for k, v in self.material_props.items():
            val_str = f"'{v}'" if isinstance(v, str) else str(v)
            blocks.append(f"material_prop {k}={val_str}")

        for node in self.nodes.values():
            blocks.append(node.to_dsl())
        
        for link in self.links:
            blocks.append(f"link {link.source_node}.{link.source_output} -> {link.target_node}.{link.target_input}")
            
        return "\n\n".join(blocks)

    def register_node(self, node: ShaderNode) -> None:
        if node.id in self.nodes:
            self.record_error(f"Duplicate node ID '{node.id}' detected.")
        self.nodes[node.id] = node

    def add_link(self, link: ShaderLink) -> None:
        if link.source_node not in self.nodes:
            self.record_error(f"Link references unknown source node '{link.source_node}'")
        elif link.target_node not in self.nodes:
            self.record_error(f"Link references unknown target node '{link.target_node}'")
        else:
            target_inputs = self.nodes[link.target_node].inputs
            if link.target_input not in target_inputs:
                # An undeclared input may still exist in bpy; verification is deferred to to_bpy.
                logger.info(f"Link targets input '{link.target_input}' on node '{link.target_node}' not predefined, but it may be ok because it may exist in bpy.")
        self.links.append(link)

    def rename(self, prefix: str) -> "ShaderGraph":
        new_graph = ShaderGraph()
        new_graph.material_props = self.material_props.copy()
        id_map = {old_id: f"{prefix}{old_id}" for old_id in self.nodes}

        for node in self.nodes.values():
            new_node = ShaderNode(
                id=id_map[node.id],
                node_type=node.node_type,
                props=node.props.copy(),
                inputs=node.inputs.copy(),
                input_slots=list(node.input_slots),
                ramp_props=node.ramp_props.copy(),
                ramp_elements=list(node.ramp_elements),
                curve_points=list(node.curve_points),
            )
            new_graph.register_node(new_node)

        for link in self.links:
            new_graph.add_link(ShaderLink(
                source_node=id_map[link.source_node],
                source_output=link.source_output,
                target_node=id_map[link.target_node],
                target_input=link.target_input
            ))
            
        return new_graph

    def _validate_acyclic(self) -> None:
        adjacency: Dict[str, List[str]] = {}
        for link in self.links:
            adjacency.setdefault(link.source_node, []).append(link.target_node)

        visiting: Dict[str, bool] = {}
        visited: Dict[str, bool] = {}

        def dfs(node_id: str) -> None:
            if visiting.get(node_id):
                raise ValueError(f"Cycle detected involving node '{node_id}'")
            if visited.get(node_id):
                return
            visiting[node_id] = True
            for neighbor in adjacency.get(node_id, []):
                dfs(neighbor)
            visiting.pop(node_id, None)
            visited[node_id] = True

        for node_id in self.nodes:
            if not visited.get(node_id):
                dfs(node_id)

    def _validate_dangling_nodes(self) -> None:
        ALLOWED_OUTPUTS = {
            "ShaderNodeOutputMaterial",
            "ShaderNodeOutputWorld",
            "ShaderNodeOutputLight",
            "NodeGroupOutput"
        }
        
        out_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        for link in self.links:
            if link.source_node in out_degree:
                out_degree[link.source_node] += 1
        
        for node_id, degree in out_degree.items():
            if degree == 0:
                node = self.nodes[node_id]
                if node.node_type not in ALLOWED_OUTPUTS:
                    self.record_error(f"Node '{node_id}' ({node.node_type}) has no outgoing links and is not a recognized output node.")

    @classmethod
    def from_dsl(cls, dsl_text: str, *, validate: bool = True) -> "ShaderGraph":
        graph = cls()

        lines = [line.strip() for line in dsl_text.strip().splitlines()]
        current_block: List[str] = []
        link_lines: List[str] = []

        def flush_block() -> None:
            nonlocal current_block
            if current_block:
                try:
                    node = ShaderNode.from_block(current_block)
                    graph.register_node(node)
                except ValueError as e:
                    graph.record_error(str(e))
                current_block = []

        lines = _remove_comments(lines)

        for raw_line in lines:
            if not raw_line:
                continue
            if raw_line.startswith("node "):
                flush_block()
                current_block.append(raw_line)
            elif raw_line.startswith("link "):
                flush_block()
                link_lines.append(raw_line)
            elif raw_line.startswith("material_prop "):
                flush_block()
                key, val = ShaderNode._parse_assignment(raw_line[len("material_prop "):])
                graph.material_props[key] = ShaderNode._parse_literal(val)
            else:
                if current_block:
                    current_block.append(raw_line)
                else:
                    graph.record_error(f"Line outside of a node or link definition: {raw_line}")

        flush_block()

        for link_line in link_lines:
            try:
                link = ShaderLink.from_str(link_line)
                if validate:
                    graph.add_link(link)
                else:
                    graph.links.append(link)
            except ValueError as e:
                graph.record_error(str(e))

        # Lexical and grammatical checks only, not semantics.
        if validate:
            try:
                graph._validate_acyclic()
            except ValueError as e:
                graph.record_error(str(e))

            graph._validate_dangling_nodes()
            
        if graph.errors:
            logger.warning(f"ShaderGraph parsing completed with errors: {graph.errors}")
        return graph

    @classmethod
    def from_bpy(cls, material_or_tree, *, flatten_groups: bool = True) -> "ShaderGraph":
        """Build a ShaderGraph by walking a bpy.types.Material (or ShaderNodeTree).

        With ``flatten_groups=True`` (the only supported path) every
        ``ShaderNodeGroup`` is inlined: inner nodes are prefixed with the group
        label and merged into the parent graph, while ``NodeGroupInput`` /
        ``NodeGroupOutput`` / the group instance are elided and links through
        them are rewired producer-to-consumer directly.

        Must be called inside Blender.
        """
        if not flatten_groups:
            raise NotImplementedError("Non-flatten extraction is not implemented.")
        return _BpyExtractor().run(material_or_tree)

    def structurally_equal(
        self, other: "ShaderGraph", tol: float = 1e-5,
    ) -> Tuple[bool, List[str]]:
        """Structural equality with float tolerance. Not implemented."""
        raise NotImplementedError(
            "ShaderGraph.structurally_equal is not implemented yet; "
            "verify round-trip correctness by comparing rendered PNGs."
        )

    def record_error(self, message: str) -> None:
        self.errors.append(message)

    def remove_nodes(self, node_ids: set) -> list:
        """Remove nodes by ID plus any links touching them.

        Returns the removed links so ``replace_nodes`` can restore bridge links
        whose other endpoint survives the patch.
        """
        for nid in node_ids:
            self.nodes.pop(nid, None)
        removed_links = [
            link for link in self.links
            if link.source_node in node_ids or link.target_node in node_ids
        ]
        self.links = [
            link for link in self.links
            if link.source_node not in node_ids and link.target_node not in node_ids
        ]
        return removed_links

    def replace_nodes(
        self,
        patch_graph: "ShaderGraph",
        removed_links: Optional[list] = None,
    ) -> None:
        """Merge replacement nodes/links from *patch_graph* into this graph.

        Same-ID nodes are overwritten and patch links appended. Given
        *removed_links* from a prior ``remove_nodes``, any link whose both
        endpoints survive and that is not already present is restored — without
        this, a patch DSL that omits links between surviving nodes silently
        drops them.
        """
        for node in patch_graph.nodes.values():
            self.nodes[node.id] = node
        self.links.extend(patch_graph.links)

        if removed_links:
            existing_link_set = {
                (l.source_node, l.source_output, l.target_node, l.target_input)
                for l in self.links
            }
            # Track occupancy per (node, input) so restores respect slot capacity.
            occupied_targets: Dict[Tuple[str, str], int] = {}
            for l in self.links:
                tgt_key = (l.target_node, l.target_input)
                occupied_targets[tgt_key] = occupied_targets.get(tgt_key, 0) + 1

            for link in removed_links:
                key = (link.source_node, link.source_output,
                       link.target_node, link.target_input)
                if (link.source_node in self.nodes
                        and link.target_node in self.nodes
                        and key not in existing_link_set):
                    tgt_key = (link.target_node, link.target_input)
                    target_node_obj = self.nodes.get(link.target_node)
                    max_slots = max(
                        1,
                        target_node_obj.input_slot_count(link.target_input)
                        if target_node_obj else 1,
                    )
                    current_count = occupied_targets.get(tgt_key, 0)
                    if current_count >= max_slots:
                        continue
                    self.links.append(link)
                    existing_link_set.add(key)
                    occupied_targets[tgt_key] = current_count + 1
    

class ShaderGraphParser:
    """Converts the ShaderGraph IR into executable Blender Python (bpy) code."""
    @staticmethod
    def _generate_nodes_code(
        graph: ShaderGraph,
        texture_map_path: Optional[str],
        texture_map_paths: Optional[Dict[str, str]] = None,
        texture_colorspaces: Optional[Dict[str, str]] = None,
        texture_alpha_modes: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        lines = []
        lines.append("    # Create nodes")
        for node_id, node in graph.nodes.items():
            lines.append(f"    # Node: {node_id} ({node.node_type})")
            lines.append("    try:")
            lines.append(f"        node = nodes.new(type='{node.node_type}')")
            lines.append(f"        node.label = '{node_id}'")
            lines.append(f"        node_map['{node_id}'] = node")

            # Per-node map wins over the legacy single-path fallback, so a
            # multi-texture material binds each TexImage to its own file.
            current_texture_path = None
            if texture_map_paths is not None:
                current_texture_path = texture_map_paths.get(node_id)
            if current_texture_path is None:
                current_texture_path = texture_map_path
            if node.node_type == "ShaderNodeTexImage":
                assert current_texture_path, (
                    f"Texture map path must be provided for ShaderNodeTexImage node '{node_id}'."
                )
                if not os.path.isabs(current_texture_path):
                    current_texture_path = str(pathlib.Path(ROOT_DIR) / current_texture_path)
                assert os.path.exists(current_texture_path), f"Texture map file not found: {current_texture_path}"
                cs = None
                if texture_colorspaces:
                    cs = texture_colorspaces.get(node_id)
                am = None
                if texture_alpha_modes:
                    am = texture_alpha_modes.get(node_id)
                lines.append(f"        if hasattr(node, 'image'):")
                lines.append(f"            img = bpy.data.images.load(r'{current_texture_path}')")
                if cs:
                    # Colorspace must be set BEFORE node.image is assigned:
                    # Blender samples pixels through the colorspace active at bind time.
                    lines.append(
                        f"            try: img.colorspace_settings.name = {cs!r}\n"
                        f"            except Exception: pass"
                    )
                if am:
                    lines.append(
                        f"            try: img.alpha_mode = {am!r}\n"
                        f"            except Exception: pass"
                    )
                lines.append(f"            node.image = img")

            if node.node_type == "ShaderNodeGroup" and "name" in node.props:
                group_name = node.props["name"]
                lines.append(f"        assert '{group_name}' in bpy.data.node_groups, 'Node group {group_name} not found in bpy.data.node_groups'")
                lines.append(f"        node.node_tree = bpy.data.node_groups['{group_name}']")

            # bpy keeps a minimum of 1 ramp element, so clearing leaves the
            # default white stop and each round-trip accumulates a spurious
            # entry. Overwrite in place, trim the tail, append the rest.
            if node.node_type == "ShaderNodeValToRGB" and node.ramp_elements:
                lines.append(f"        if hasattr(node, 'color_ramp') and node.color_ramp:")
                for prop, val in node.ramp_props.items():
                    value_literal = f"'{val}'" if isinstance(val, str) else str(val)
                    lines.append(f"            if hasattr(node.color_ramp, '{prop}'):")
                    lines.append(f"                node.color_ramp.{prop} = {value_literal}")
                lines.append(f"            _ramp = node.color_ramp.elements")
                lines.append(f"            _new_ramp = {list(node.ramp_elements)!r}")
                lines.append(f"            while len(_ramp) > max(1, len(_new_ramp)):")
                lines.append(f"                _ramp.remove(_ramp[-1])")
                lines.append(f"            for _i, (_c, _p) in enumerate(_new_ramp):")
                lines.append(f"                if _i < len(_ramp):")
                lines.append(f"                    _ramp[_i].position = _p")
                lines.append(f"                    _ramp[_i].color = _c")
                lines.append(f"                else:")
                lines.append(f"                    _e = _ramp.new(_p)")
                lines.append(f"                    _e.color = _c")

            if node.node_type in ("ShaderNodeRGBCurve", "ShaderNodeVectorCurve", "ShaderNodeFloatCurve") and node.curve_points:
                lines.append(f"        if hasattr(node, 'mapping') and hasattr(node.mapping, 'curves'):")
                # Group by channel here to keep the generated code flat.
                points_by_channel = {}
                for ch, x, y in node.curve_points:
                    points_by_channel.setdefault(ch, []).append((x, y))
                
                channel_map = {
                    'R': 0, 'X': 0,
                    'G': 1, 'Y': 1,
                    'B': 2, 'Z': 2,
                    'C': 3
                }

                for ch, points in points_by_channel.items():
                    idx = channel_map.get(ch, 3)
                    if node.node_type == "ShaderNodeFloatCurve": 
                        idx = 0
                    
                    # bpy.CurveMap.points has a minimum of 2: removing below
                    # that raises "Unable to remove curve point". Overwrite the
                    # first 2 in place, drop the tail, append the rest.
                    lines.append(f"            curve = node.mapping.curves[{idx}]")
                    lines.append(f"            _new_pts = {list(points)!r}")
                    lines.append(f"            while len(curve.points) > max(2, len(_new_pts)):")
                    lines.append(f"                curve.points.remove(curve.points[-1])")
                    lines.append(f"            for _i, (_x, _y) in enumerate(_new_pts):")
                    lines.append(f"                if _i < len(curve.points):")
                    lines.append(f"                    curve.points[_i].location = (_x, _y)")
                    lines.append(f"                else:")
                    lines.append(f"                    curve.points.new(_x, _y)")
                    lines.append(f"            node.mapping.update()")

            for prop, val in node.props.items():
                # For ShaderNodeGroup, 'name' selects the tree to link, not node.name.
                if node.node_type == "ShaderNodeGroup" and prop == "name":
                    continue
                
                if isinstance(val, str):
                    lines.append(f"        if hasattr(node, '{prop}'):")
                    lines.append(f"            node.{prop} = '{val}'")
                else:
                    lines.append(f"        if hasattr(node, '{prop}'):")
                    lines.append(f"            node.{prop} = {val}")

            slots = node.input_slots if node.input_slots else list(node.inputs.items())
            # Value/RGB hold their data on outputs[0].default_value, not on any
            # input, so DSL ``in X=`` redirects to the output (float for Value,
            # RGBA tuple for RGB). Without this the value vanishes on round-trip
            # and downstream sees 0/black — e.g. Mapping fed by Value gets scale=0.
            _OUTPUT_DEFAULT_NODES = {"ShaderNodeValue", "ShaderNodeRGB"}
            is_output_default_node = node.node_type in _OUTPUT_DEFAULT_NODES
            lines.append("        input_slot_usage = {}")
            for input_name, input_def in slots:
                lines.append(f"        input_slot_key = '{input_name}'")
                lines.append("        input_slot_idx = input_slot_usage.get(input_slot_key, 0)")
                lines.append("        input_slot_usage[input_slot_key] = input_slot_idx + 1")
                if input_def.default is None:
                    continue

                val = input_def.default
                value_literal = f"'{val}'" if isinstance(val, str) else str(val)
                if is_output_default_node:
                    lines.append(f"        if len(node.outputs) > 0:")
                    lines.append(f"            node.outputs[0].default_value = {value_literal}")
                    continue
                lines.append(f"        input_candidates = [sock for sock in node.inputs if sock.name == '{input_name}' and sock.enabled]")
                lines.append(f"        if input_slot_idx < len(input_candidates):")
                lines.append(f"            input_candidates[input_slot_idx].default_value = {value_literal}")
                lines.append(f"        else:")
                lines.append(f"            raise IndexError('No matching input socket slot for input {input_name}')")
            lines.append("    except Exception as exc:")
            lines.append(f"        errors.append(f'Node {node_id} ({node.node_type}) failed: {{exc}}')")
            lines.append("")
        return lines

    @staticmethod
    def _generate_links_code(graph: ShaderGraph) -> List[str]:
        lines = []
        lines.append("    # Create links")
        link_slot_map = ShaderGraphParser._link_slot_map(graph)
        lines.append(f"    link_slot_map = {link_slot_map!r}")
        lines.append("    link_socket_usage = {}")
        for link in graph.links:
            source_node_type = graph.nodes[link.source_node].node_type if link.source_node in graph.nodes else None
            lines.extend([
                "    try:",
                f"        src = node_map['{link.source_node}']",
                f"        dst = node_map['{link.target_node}']",
                f"        target_sockets = [sock for sock in dst.inputs if sock.name == '{link.target_input}' and sock.enabled]",
                f"        usage_key = ('{link.target_node}', '{link.target_input}')",
                "        idx = link_socket_usage.get(usage_key, 0)",
                "        explicit_slots = link_slot_map.get(usage_key)",
                "        if explicit_slots is not None:",
                "            if idx >= len(explicit_slots):",
                "                raise IndexError('No matching explicit socket slot for link')",
                "            target_idx = explicit_slots[idx]",
                "        else:",
                "            target_idx = idx",
                "        if target_idx >= len(target_sockets):",
                "            raise IndexError('No matching socket slot for link (index out of range)')",
                "        target_socket = target_sockets[target_idx]",
            ])

            if source_node_type == "ShaderNodeGroup":
                lines.extend([
                    f"        # ShaderNodeGroup outputs are defined by the group's interface; default to BSDF if needed",
                    f"        try:",
                    f"            src_socket = src.outputs['{link.source_output}']",
                    f"        except Exception:",
                    f"            try:",
                    f"                src_socket = src.outputs['BSDF']",
                    f"            except Exception:",
                    f"                src_socket = src.outputs[0]",
                    f"        links.new(src_socket, target_socket)",
                ])
            else:
                lines.append(f"        links.new(src.outputs['{link.source_output}'], target_socket)")

            lines.extend([
                "        if not target_socket.enabled:",
                "            raise RuntimeError(f'Linked to a disabled socket of type {target_socket.type}')",
                "        link_socket_usage[usage_key] = idx + 1",
                "    except Exception as exc:",
                f"        errors.append(f'Link {link.source_node}.{link.source_output} -> {link.target_node}.{link.target_input} failed: {{exc}}')",
            ])
        return lines

    @staticmethod
    def _link_slot_map(graph: ShaderGraph) -> Dict[Tuple[str, str], List[int]]:
        """Return explicit target slot indices for links on duplicate sockets.

        Extraction emits a bare ``in <name>`` placeholder for each linked slot
        of a repeated socket; this map turns those placeholders into exact
        socket indices during ``to_bpy`` link creation.
        """
        out: Dict[Tuple[str, str], List[int]] = {}
        for node_id, node in graph.nodes.items():
            slots = node.input_slots or []
            if not slots:
                continue
            per_name_counts: Dict[str, int] = {}
            for name, _ in slots:
                per_name_counts[name] = per_name_counts.get(name, 0) + 1
            if not any(count > 1 for count in per_name_counts.values()):
                continue
            per_name_index: Dict[str, int] = {}
            for name, input_def in slots:
                slot_idx = per_name_index.get(name, 0)
                per_name_index[name] = slot_idx + 1
                if per_name_counts[name] <= 1:
                    continue
                if input_def.default is None:
                    out.setdefault((node_id, name), []).append(slot_idx)
        return out

    @staticmethod
    def _generate_group_code(
        name: str,
        graph: ShaderGraph,
        texture_map_path: Optional[str],
        texture_map_paths: Optional[Dict[str, str]] = None,
        texture_colorspaces: Optional[Dict[str, str]] = None,
        texture_alpha_modes: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        lines = []
        lines.append(f"    # --- Group: {name} ---")
        lines.append(f"    # If the group already exists, replace it (idempotent generation)")
        lines.append(f"    if '{name}' in bpy.data.node_groups:")
        lines.append(f"        bpy.data.node_groups.remove(bpy.data.node_groups['{name}'], do_unlink=True)")
        lines.append(f"    group = bpy.data.node_groups.new(name='{name}', type='ShaderNodeTree')")

        # Blender 4.1+ interface API. Every subgraph gets a fixed shape: no
        # inputs, one BSDF output — the IR carries no per-node output metadata.
        lines.append("    # Fix group interface: no inputs, single BSDF output")
        lines.append("    iface = group.interface")
        lines.append("    for item in list(iface.items_tree):")
        lines.append("        iface.remove(item)")
        lines.append("    iface.new_socket(name='BSDF', in_out='OUTPUT', socket_type='NodeSocketShader')")

        lines.append(f"    nodes = group.nodes")
        lines.append(f"    links = group.links")
        lines.append(f"    node_map = {{}}")
        lines.append(f"    socket_usage = {{}}")
        lines.append("")

        # rename("") clones so the Output→GroupOutput rewrite below is local.
        group_graph = graph.rename("")

        for node in group_graph.nodes.values():
            if node.node_type == "ShaderNodeOutputMaterial":
                node.node_type = "NodeGroupOutput"

        for link in group_graph.links:
            target_node = group_graph.nodes.get(link.target_node)
            if target_node and target_node.node_type == "NodeGroupOutput":
                if link.target_input == "Surface":
                    link.target_input = "BSDF"
        
        lines.extend(ShaderGraphParser._generate_nodes_code(
            group_graph, texture_map_path, texture_map_paths,
            texture_colorspaces, texture_alpha_modes,
        ))
        lines.extend(ShaderGraphParser._generate_links_code(group_graph))
        
        lines.append(f"    # End Group: {name}")
        lines.append("")
        return lines

    @staticmethod
    def to_bpy(
        graph: ShaderGraph,
        material_name: str = "GeneratedMaterial",
        texture_map_path: Optional[str] = None,
        subgraphs: Optional[Dict[str, ShaderGraph]] = None,
        texture_map_paths: Optional[Dict[str, str]] = None,
        texture_colorspaces: Optional[Dict[str, str]] = None,
        texture_alpha_modes: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate a Python script that recreates this material inside Blender.

        ``texture_map_paths`` maps DSL node_id → file path and takes precedence
        per-node over ``texture_map_path``, which remains the legacy
        single-texture default.
        """
        lines = [
            "import bpy",
            "",
            "def create_material():",
            "    errors = []",
            "",
        ]

        subgraphs = subgraphs or getattr(graph, "subgraphs", None)
        if subgraphs:
            lines.append("    # --- Subgraphs support ---")
            for name, sg_graph in subgraphs.items():
                lines.extend(ShaderGraphParser._generate_group_code(
                    name, sg_graph, texture_map_path, texture_map_paths,
                    texture_colorspaces, texture_alpha_modes,
                ))

        lines.extend([
            f"    # --- Main Material: {material_name} ---",
            f"    mat = bpy.data.materials.new(name='{material_name}')",
            "    mat.use_nodes = True",
            "    nodes = mat.node_tree.nodes",
            "    links = mat.node_tree.links",
            "",
            "    # Clear default nodes",
            "    nodes.clear()",
            "",
            "    node_map = {}",
            "    socket_usage = {}",
            ""
        ])

        if graph.material_props:
            lines.append("    # Material-level render settings")
            for prop, val in graph.material_props.items():
                value_literal = f"'{val}'" if isinstance(val, str) else str(val)
                lines.append(f"    if hasattr(mat, '{prop}'):")
                lines.append(f"        mat.{prop} = {value_literal}")
            lines.append("")

        lines.extend(ShaderGraphParser._generate_nodes_code(
            graph, texture_map_path, texture_map_paths,
            texture_colorspaces, texture_alpha_modes,
        ))
        lines.extend(ShaderGraphParser._generate_links_code(graph))

        lines.extend([
            "",
            "    if errors:",
            "        raise RuntimeError(' ; '.join(errors))",
            "",
            "    return mat",
            "",
            "if __name__ == '__main__':",
            "    create_material()"
        ])
        return "\n".join(lines)


# Per-node-type allowlist of properties to extract into DSL. `bl_rna.properties`
# also exposes generic UI fields (name/location/width/hide/mute/color/parent/...)
# that would pollute the DSL, so only the properties `to_bpy` actually sets are
# listed here.
NODE_PROP_ALLOWLIST: Dict[str, Tuple[str, ...]] = {
    "ShaderNodeMath":           ("operation", "use_clamp"),
    "ShaderNodeVectorMath":     ("operation",),
    "ShaderNodeMixRGB":         ("blend_type", "use_clamp"),
    "ShaderNodeMix":            ("data_type", "blend_type", "clamp_result", "clamp_factor", "factor_mode"),
    "ShaderNodeMapRange":       ("interpolation_type", "data_type", "clamp"),
    "ShaderNodeTexNoise":       ("noise_dimensions", "noise_type", "normalize"),
    # noise_dimensions drives which inputs exist (only 4D has "W"). Without it a
    # 4D node round-trips to the 3D default and to_bpy fails on the missing W.
    "ShaderNodeTexWhiteNoise":  ("noise_dimensions",),
    "ShaderNodeTexVoronoi":     ("voronoi_dimensions", "feature", "distance", "normalize"),
    "ShaderNodeTexBrick":       ("offset", "offset_frequency", "squash", "squash_frequency"),
    "ShaderNodeTexGradient":    ("gradient_type",),
    "ShaderNodeTexWave":        ("wave_type", "bands_direction", "rings_direction", "wave_profile"),
    "ShaderNodeTexMusgrave":    ("musgrave_dimensions", "musgrave_type", "normalize"),
    "ShaderNodeTexImage":       ("projection", "interpolation", "extension"),
    "ShaderNodeTexEnvironment": ("projection", "interpolation"),
    "ShaderNodeBsdfPrincipled": ("distribution", "subsurface_method"),
    "ShaderNodeBump":           ("invert",),
    "ShaderNodeNormalMap":      ("space", "uv_map"),
    "ShaderNodeDisplacement":   ("space",),
    "ShaderNodeVectorDisplacement": ("space",),
    "ShaderNodeSeparateColor":  ("mode",),
    "ShaderNodeCombineColor":   ("mode",),
    "ShaderNodeMapping":        ("vector_type",),
    "ShaderNodeClamp":          ("clamp_type",),
    "ShaderNodeBsdfGlass":      ("distribution",),
    "ShaderNodeBsdfRefraction": ("distribution",),
    "ShaderNodeBsdfGlossy":     ("distribution",),
    "ShaderNodeBsdfAnisotropic":("distribution",),
    "ShaderNodeAttribute":      ("attribute_type", "attribute_name"),
    "ShaderNodeUVMap":          ("uv_map", "from_instancer"),
    "ShaderNodeTangent":        ("direction_type", "axis"),
    # rotation_type drives which sockets exist (EULER_XYZ → "Rotation";
    # AXIS_ANGLE → "Axis" + "Angle"). Without it a EULER node round-trips to the
    # AXIS_ANGLE default and "Rotation" vanishes, failing in to_bpy.
    "ShaderNodeVectorRotate":   ("rotation_type", "invert"),
}

# Props written to DSL even when they equal Blender's RNA default: some drive
# socket layout, others behave differently between nodes loaded from an old file
# and nodes freshly created in the current Blender runtime.
_NODE_PROPS_ALWAYS_WRITE: Dict[str, frozenset] = {
    "ShaderNodeTexNoise":       frozenset(("noise_dimensions", "noise_type", "normalize")),
    "ShaderNodeTexVoronoi":     frozenset(("voronoi_dimensions",)),
    "ShaderNodeTexWhiteNoise":  frozenset(("noise_dimensions",)),
    "ShaderNodeTexMusgrave":    frozenset(("musgrave_dimensions", "musgrave_type", "normalize")),
    "ShaderNodeTexGradient":    frozenset(("gradient_type",)),
    "ShaderNodeVectorRotate":   frozenset(("rotation_type",)),
    "ShaderNodeMix":            frozenset(("data_type",)),
    "ShaderNodeMapRange":       frozenset(("data_type",)),
    "ShaderNodeCombineColor":   frozenset(("mode",)),
    "ShaderNodeSeparateColor":  frozenset(("mode",)),
    "ShaderNodeMapping":        frozenset(("vector_type",)),
    "ShaderNodeNormalMap":      frozenset(("space",)),
    "ShaderNodeDisplacement":   frozenset(("space",)),
    "ShaderNodeVectorDisplacement": frozenset(("space",)),
    "ShaderNodeBump":           frozenset(("invert",)),
    "ShaderNodeTangent":        frozenset(("direction_type",)),
    "ShaderNodeBsdfPrincipled": frozenset(("distribution", "subsurface_method")),
    "ShaderNodeBsdfGlass":      frozenset(("distribution",)),
    "ShaderNodeBsdfRefraction": frozenset(("distribution",)),
    "ShaderNodeBsdfGlossy":     frozenset(("distribution",)),
    "ShaderNodeBsdfAnisotropic": frozenset(("distribution",)),
}

# Curve node index → channel letter (reverse of the channel_map in to_bpy)
_CURVE_CHANNEL_MAP: Dict[str, Dict[int, str]] = {
    "ShaderNodeRGBCurve":    {0: "R", 1: "G", 2: "B", 3: "C"},
    "ShaderNodeVectorCurve": {0: "X", 1: "Y", 2: "Z"},
    "ShaderNodeFloatCurve":  {0: "C"},  # to_bpy forces idx=0 regardless
}

# Nodes never emitted to the DSL: virtual Group I/O, the group instance being
# flattened away, and NodeFrame (UI-only container, owns no sockets).
# NodeReroute is deliberately absent — skipping it would drop the signal, so it
# round-trips through to_bpy's generic path.
_SKIPPED_BL_IDNAMES = {
    "ShaderNodeGroup", "NodeGroupInput", "NodeGroupOutput",
    "NodeFrame",
}

_FLOAT_ROUND_DP = 6

MATERIAL_PROP_ALLOWLIST: Tuple[str, ...] = (
    "blend_method",
    "surface_render_method",
    "displacement_method",
    "shadow_method",
    "use_screen_refraction",
    "show_transparent_back",
    "use_backface_culling",
    "use_backface_culling_shadow",
    "alpha_threshold",
    "use_raytrace_refraction",
    "refraction_depth",
    "use_sss_translucency",
    "volume_intersection_method",
    "use_transparent_shadow",
    "max_vertex_displacement",
)


def _sanitize_node_id(raw: str, existing: Set[str]) -> str:
    """Produce a DSL-legal node id (matches ShaderNode._validate_key rules)."""
    if not raw:
        raw = "node"
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw).strip("_")
    if not cleaned:
        cleaned = "node"
    candidate = cleaned
    i = 2
    while candidate in existing:
        candidate = f"{cleaned}_{i}"
        i += 1
    return candidate


def _coerce_value(v: Any) -> Any:
    """Convert a bpy/mathutils value into a DSL-friendly Python literal.

    Sequences become tuples of floats rounded to 6 dp, which suppresses f32 repr
    noise. bool is tested before int because bool is an int subclass.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return round(v, _FLOAT_ROUND_DP)
    # mathutils Vector/Color and bpy_prop_array behave like sequences
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        try:
            return tuple(round(float(x), _FLOAT_ROUND_DP) for x in v)
        except (TypeError, ValueError):
            return v
    return v


class _BpyExtractor:
    """Stateful walker that converts a bpy ShaderNodeTree to a flat ShaderGraph.

    Must run inside Blender. Entrypoint: ``run(material_or_tree)``.
    """

    def __init__(self) -> None:
        # Ids used across the whole flattened graph, so group-label prefixes
        # cannot collide.
        self._used_ids: Set[str] = set()
        # bpy pointer → DSL id, for emitted nodes only (no group virtuals).
        self._id_map: Dict[int, str] = {}
        # Child ShaderNodeGroup pointer → its inner group_output_provider
        # (written in Pass 3b, read in Pass 3c).
        self._group_output_cache: Dict[int, Dict[str, Tuple[str, str]]] = {}
        self._graph = ShaderGraph()
        # Parallel to ``self._graph.links``: each link's target slot index among
        # same-name sockets. ``_reorder_links_by_slot`` uses it so to_bpy's
        # socket_usage counter lands on the right socket for duplicate-name
        # inputs (MixShader's two "Shader", Math's several "Value").
        self._link_slot_idx: List[int] = []

    def run(self, material_or_tree) -> ShaderGraph:
        try:
            import bpy  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ShaderGraph.from_bpy must be called inside Blender"
            ) from exc

        tree = getattr(material_or_tree, "node_tree", material_or_tree)
        if tree is None or not hasattr(tree, "nodes"):
            raise ValueError(
                "from_bpy requires a material with node_tree or a ShaderNodeTree"
            )

        if hasattr(material_or_tree, "node_tree"):
            self._graph.material_props = self._read_material_props(material_or_tree)

        self._walk(tree, prefix="",
                   outer_input_src=None, outer_output_sink=None,
                   outer_input_default=None)
        self._reorder_links_by_slot()
        return self._graph

    @staticmethod
    def _read_material_props(mat) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for prop in MATERIAL_PROP_ALLOWLIST:
            if not hasattr(mat, prop):
                continue
            try:
                out[prop] = _coerce_value(getattr(mat, prop))
            except Exception:
                continue
        return out

    def _reorder_links_by_slot(self) -> None:
        """Order links sharing a ``(target_node, target_input)`` by slot index.

        ``to_bpy`` dispatches successive links on same-name inputs to successive
        sockets via a ``socket_usage`` counter, so emitting them out of order
        scrambles the wiring (MixShader's two "Shader" inputs swap). Reordering
        happens within each bucket, preserving each link's global position.
        """
        from collections import defaultdict
        buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for i, lk in enumerate(self._graph.links):
            buckets[(lk.target_node, lk.target_input)].append(i)

        for key, positions in buckets.items():
            if len(positions) < 2:
                continue
            # Stable: an identical slot_idx keeps its intra-bucket order.
            sorted_positions = sorted(
                positions, key=lambda p: self._link_slot_idx[p]
            )
            if sorted_positions == positions:
                continue
            resorted = [self._graph.links[p] for p in sorted_positions]
            resorted_idx = [self._link_slot_idx[p] for p in sorted_positions]
            for dst, lk, si in zip(positions, resorted, resorted_idx):
                self._graph.links[dst] = lk
                self._link_slot_idx[dst] = si

    def _walk(
        self,
        tree,
        prefix: str,
        outer_input_src: Optional[Dict[str, Tuple[str, str]]],
        outer_output_sink: Optional[Dict[str, List[Tuple[str, str]]]],
        outer_input_default: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tuple[str, str]]:
        """Walk one tree; return this tree's group_output_provider map.

        ``outer_input_src`` (when inside a group): maps NodeGroupInput output
            socket name → (external_dsl_id, external_output_socket).
        ``outer_output_sink`` (when inside a group): maps NodeGroupOutput
            input socket name → list of (external_dsl_id, external_input_socket)
            pairs that consume the group output externally.
        ``outer_input_default`` (when inside a group): maps NodeGroupInput
            output socket name → the group-instance's interface default for that
            socket, used when the outer wired no link into this input. Without
            it an inner consumer loses its signal on round-trip and falls back
            to its type default (0/black), rendering wrongly.

        Returns a ``group_output_provider`` for THIS tree, meaningful only when
        this tree is a group's inner tree.
        """
        # Pass 1: assign DSL ids for all nodes we'll emit (skip virtuals + groups)
        local_nodes = []  # insertion order keeps ids deterministic
        for bn in tree.nodes:
            if bn.bl_idname in _SKIPPED_BL_IDNAMES:
                continue
            raw = (bn.label.strip() if bn.label else "") or bn.name
            full = f"{prefix}{raw}"
            nid = _sanitize_node_id(full, self._used_ids)
            self._used_ids.add(nid)
            self._id_map[bn.as_pointer()] = nid
            local_nodes.append(bn)

        # Pass 2: emit a ShaderNode for each retained node.
        for bn in local_nodes:
            self._graph.register_node(
                self._extract_node(bn, self._id_map[bn.as_pointer()])
            )

        # Pass 3a: for each child ShaderNodeGroup, record which external source
        # feeds each of its inputs (resolving NodeGroupInput / nested groups),
        # and where its outputs are consumed, so the inner walk can map both.
        child_group_input_src: Dict[int, Dict[str, Tuple[str, str]]] = {}
        child_group_output_sink: Dict[int, Dict[str, List[Tuple[str, str]]]] = {}
        # Per child group: interface default_value for each input with no
        # external link here. Seeds ``outer_input_default`` for the inner walk
        # so NodeGroupInput fallbacks still render.
        child_group_input_default: Dict[int, Dict[str, Any]] = {}

        this_tree_output_provider: Dict[str, Tuple[str, str]] = {}

        for link in tree.links:
            src_bn = link.from_node
            tgt_bn = link.to_node
            src_sock = link.from_socket.name
            tgt_sock = link.to_socket.name

            # The real source for the DSL link's src side, virtuals stripped.
            resolved_src = self._resolve_source(
                src_bn, src_sock, outer_input_src,
                owning_tree_provider=this_tree_output_provider,
            )
            if resolved_src is None:
                # Either an unlinked NodeGroupInput (skip it) or a
                # ShaderNodeGroup whose inner walk hasn't run yet — the latter
                # resolves in Pass 3c once recursion has filled the provider.
                if src_bn.bl_idname == "NodeGroupInput":
                    continue

            if tgt_bn.bl_idname == "ShaderNodeGroup":
                if resolved_src is not None:
                    child_group_input_src.setdefault(
                        tgt_bn.as_pointer(), {})[tgt_sock] = resolved_src
                continue

            # NodeGroupOutput consumers propagate outward via outer_output_sink.
            if tgt_bn.bl_idname == "NodeGroupOutput":
                if resolved_src is not None:
                    this_tree_output_provider[tgt_sock] = resolved_src
                continue

        # Pass 3a.5: wire up interface defaults for child-group inputs with no
        # external link here. Preferred route is synthesis — emit a constant
        # node and register it in ``child_group_input_src`` as if it were an
        # external link, so NodeGroupInput.<socket> resolves like any other link
        # and reroute chains propagate the default without type-aware injection
        # per consumer. ``child_group_input_default`` is the fallback for socket
        # types we cannot synthesize.
        for bn in tree.nodes:
            if bn.bl_idname != "ShaderNodeGroup" or bn.node_tree is None:
                continue
            ptr = bn.as_pointer()
            linked_srcs = child_group_input_src.setdefault(ptr, {})
            # Only synthesize for inputs actually consumed inside; otherwise the
            # DSL fills up with dead constant nodes.
            consumed: set = set()
            for inner in bn.node_tree.nodes:
                if inner.bl_idname != "NodeGroupInput":
                    continue
                for out in inner.outputs:
                    if getattr(out, "is_linked", False):
                        consumed.add(out.name)
            defaults: Dict[str, Any] = {}
            for in_sock in bn.inputs:
                if not getattr(in_sock, "enabled", True):
                    continue
                if in_sock.name in linked_srcs:
                    continue  # externally linked; handled by outer_input_src
                if in_sock.name not in consumed:
                    continue
                if getattr(in_sock, "type", "") == "SHADER":
                    continue
                dval = getattr(in_sock, "default_value", None)
                if dval is None:
                    continue
                synth = self._synthesize_constant_source(bn, in_sock, prefix)
                if synth is not None:
                    linked_srcs[in_sock.name] = synth
                else:
                    defaults[in_sock.name] = _coerce_value(dval)
            if defaults:
                child_group_input_default[ptr] = defaults

        # Pass 3b: recurse into each child ShaderNodeGroup.
        for bn in tree.nodes:
            if bn.bl_idname != "ShaderNodeGroup" or bn.node_tree is None:
                continue
            inner_prefix = f"{prefix}{(bn.label or bn.name).strip()}_"
            inner_input_src = child_group_input_src.get(bn.as_pointer(), {})
            inner_input_default = child_group_input_default.get(
                bn.as_pointer(), {}
            )

            # Which external links consume each of bn.outputs[socket_name].
            # Entries are (dsl_id, sock_name, slot_idx); slot_idx is the position
            # among same-name sockets on the consumer, for socket_usage ordering.
            inner_output_sink: Dict[str, List[Tuple[str, str, int]]] = {}
            for link in tree.links:
                if link.from_node is bn:
                    consumer = self._resolve_target(
                        link.to_node, link.to_socket,
                        outer_output_sink=outer_output_sink,
                    )
                    if consumer is not None:
                        inner_output_sink.setdefault(
                            link.from_socket.name, []).extend(consumer)

            inner_provider = self._walk(
                bn.node_tree, inner_prefix,
                outer_input_src=inner_input_src,
                outer_output_sink=inner_output_sink,
                outer_input_default=inner_input_default,
            )
            # Cached so Pass 3c can resolve src_bn == ShaderNodeGroup in this tree.
            self._group_output_cache[bn.as_pointer()] = inner_provider

        # Pass 3c: emit links for this tree.
        for link in tree.links:
            src_bn = link.from_node
            tgt_bn = link.to_node
            src_sock = link.from_socket.name
            tgt_sock = link.to_socket.name

            # bpy keeps links to disabled sockets in tree.links after a
            # data_type toggle, but they contribute no shading. Emitting them
            # makes to_bpy attach to an invisible slot and IndexError, losing the
            # link and leaving the real active slot at its default.
            if not getattr(link.to_socket, "enabled", True):
                continue
            if not getattr(link.from_socket, "enabled", True):
                continue

            # Already captured in group_input_src / output_sink.
            if tgt_bn.bl_idname == "ShaderNodeGroup":
                continue
            if tgt_bn.bl_idname == "NodeGroupOutput":
                continue

            resolved_src = self._resolve_source(
                src_bn, src_sock, outer_input_src,
                owning_tree_provider=this_tree_output_provider,
            )
            if resolved_src is None:
                # NodeGroupInput with no external link: push the group instance's
                # interface default onto the inner consumer as a static value, so
                # the signal survives flattening without new DSL primitives.
                if (src_bn.bl_idname == "NodeGroupInput"
                        and outer_input_default is not None):
                    dval = outer_input_default.get(src_sock)
                    if dval is not None:
                        self._push_default_to_node(
                            tgt_bn, link.to_socket, dval
                        )
                continue

            tgt_id = self._id_map.get(tgt_bn.as_pointer())
            if tgt_id is None:
                continue

            src_id, src_out = resolved_src
            self._graph.links.append(ShaderLink(
                source_node=src_id,
                source_output=src_out,
                target_node=tgt_id,
                target_input=tgt_sock,
            ))
            self._link_slot_idx.append(self._slot_index(tgt_bn, link.to_socket))

        # Pass 3d: for a group tree, fan each NodeGroupOutput provider out to its
        # external consumers, which carry slot_idx so slot order is restorable.
        if outer_output_sink is not None:
            for out_sock, consumers in outer_output_sink.items():
                provider = this_tree_output_provider.get(out_sock)
                if provider is None:
                    continue
                src_id, src_out = provider
                for tgt_id, tgt_in, slot_idx in consumers:
                    self._graph.links.append(ShaderLink(
                        source_node=src_id,
                        source_output=src_out,
                        target_node=tgt_id,
                        target_input=tgt_in,
                    ))
                    self._link_slot_idx.append(slot_idx)

        return this_tree_output_provider

    def _resolve_source(
        self,
        src_bn,
        src_sock: str,
        outer_input_src: Optional[Dict[str, Tuple[str, str]]],
        owning_tree_provider: Dict[str, Tuple[str, str]],
    ) -> Optional[Tuple[str, str]]:
        """Resolve a link's source side, stripping group virtuals.

        Returns ``(dsl_id, output_socket_name)`` or ``None`` if unresolvable.
        """
        if src_bn.bl_idname == "NodeGroupInput":
            if outer_input_src is None:
                return None
            return outer_input_src.get(src_sock)
        if src_bn.bl_idname == "ShaderNodeGroup":
            provider = self._group_output_cache.get(src_bn.as_pointer())
            if provider is None:
                return None
            return provider.get(src_sock)
        dsl_id = self._id_map.get(src_bn.as_pointer())
        if dsl_id is None:
            return None
        return (dsl_id, src_sock)

    def _resolve_target(
        self,
        tgt_bn,
        tgt_sock_obj,
        outer_output_sink: Optional[Dict[str, List[Tuple[str, str, int]]]],
    ) -> Optional[List[Tuple[str, str, int]]]:
        """Resolve a link's target side into a list of concrete consumers.

        Each consumer is ``(dsl_id, socket_name, slot_idx)``, slot_idx being the
        socket's position among same-name sockets on the target node. MixShader,
        Math and MapRange expose several sockets under one name, and to_bpy's
        ``socket_usage`` counter needs those links emitted in slot-index order.
        """
        if tgt_bn.bl_idname == "NodeGroupOutput":
            if outer_output_sink is None:
                return None
            # Fan-out entries already carry slot_idx from the outer level, where
            # the group's outputs are consumed.
            return outer_output_sink.get(tgt_sock_obj.name, [])
        if tgt_bn.bl_idname == "ShaderNodeGroup":
            return None
        dsl_id = self._id_map.get(tgt_bn.as_pointer())
        if dsl_id is None:
            return None
        slot_idx = self._slot_index(tgt_bn, tgt_sock_obj)
        return [(dsl_id, tgt_sock_obj.name, slot_idx)]

    def _synthesize_constant_source(self, group_bn, in_sock, prefix
                                     ) -> Optional[Tuple[str, str]]:
        """Emit a constant-source DSL node carrying ``in_sock``'s default value.

        Returns ``(dsl_id, output_socket_name)`` to register as a link source in
        ``child_group_input_src``, or ``None`` for socket types with no constant
        equivalent (those fall back to the default-push path).

            VALUE / INT / BOOLEAN → ShaderNodeValue
            RGBA                  → ShaderNodeRGB
            VECTOR                → ShaderNodeCombineXYZ
        """
        sock_type = getattr(in_sock, "type", "")
        dval = getattr(in_sock, "default_value", None)
        if dval is None:
            return None

        group_raw = (group_bn.label or group_bn.name).strip() or "Group"
        sock_raw = in_sock.name or "Input"
        base = f"{prefix}{group_raw}_in_{sock_raw}"
        dsl_id = _sanitize_node_id(base, self._used_ids)
        self._used_ids.add(dsl_id)

        if sock_type in ("VALUE", "INT", "BOOLEAN"):
            try:
                val = float(dval)
            except (TypeError, ValueError):
                if hasattr(dval, "__len__") and len(dval) > 0:
                    val = float(dval[0])
                else:
                    return None
            ni = NodeInput(default=round(val, _FLOAT_ROUND_DP))
            self._graph.register_node(ShaderNode(
                id=dsl_id, node_type="ShaderNodeValue",
                props={}, inputs={"Value": ni},
                input_slots=[("Value", ni)],
            ))
            return (dsl_id, "Value")

        if sock_type == "RGBA":
            coerced = _coerce_value(dval)
            if not isinstance(coerced, (list, tuple)) or len(coerced) < 3:
                return None
            if len(coerced) == 3:
                color = tuple(coerced) + (1.0,)
            else:
                color = tuple(coerced[:4])
            ni = NodeInput(default=color)
            self._graph.register_node(ShaderNode(
                id=dsl_id, node_type="ShaderNodeRGB",
                props={}, inputs={"Color": ni},
                input_slots=[("Color", ni)],
            ))
            return (dsl_id, "Color")

        if sock_type == "VECTOR":
            coerced = _coerce_value(dval)
            if not isinstance(coerced, (list, tuple)) or len(coerced) < 3:
                return None
            x = round(float(coerced[0]), _FLOAT_ROUND_DP)
            y = round(float(coerced[1]), _FLOAT_ROUND_DP)
            z = round(float(coerced[2]), _FLOAT_ROUND_DP)
            ni_x = NodeInput(default=x)
            ni_y = NodeInput(default=y)
            ni_z = NodeInput(default=z)
            self._graph.register_node(ShaderNode(
                id=dsl_id, node_type="ShaderNodeCombineXYZ",
                props={},
                inputs={"X": ni_x, "Y": ni_y, "Z": ni_z},
                input_slots=[("X", ni_x), ("Y", ni_y), ("Z", ni_z)],
            ))
            return (dsl_id, "Vector")

        return None

    def _push_default_to_node(self, bn, socket_obj, default_val) -> None:
        """Inject ``default_val`` onto an already-registered DSL node's input.

        Called from Pass 3c for a ``NodeGroupInput`` with no external feed but
        with an interface default on the parent group instance. ``_read_inputs``
        skipped the socket in Pass 2 because bpy reported ``is_linked=True``, so
        the default is added retroactively here. First writer wins.

        The interface socket type may differ from the inner consumer's: Blender
        auto-converts across a live link, but ``sock.default_value = val`` is
        strict, so the value is reshaped to the consumer's type and dropped when
        unreshapable rather than raising ``sequence expected`` at render.
        """
        sock_type = getattr(socket_obj, "type", "")
        if sock_type == "SHADER":
            return
        # A NodeReroute socket's type is inferred from what is downstream at
        # render time, so a default written here is brittle: ``sock.type`` may
        # read VALUE at extract time while the rebuilt reroute resolves to
        # VECTOR, and Blender rejects the scalar. Skipping lets the consumer
        # fall back to its own default, which beats a type crash.
        if getattr(bn, "bl_idname", "") == "NodeReroute":
            return
        tgt_id = self._id_map.get(bn.as_pointer())
        if tgt_id is None:
            return
        node = self._graph.nodes.get(tgt_id)
        if node is None:
            return
        name = socket_obj.name
        if name in node.inputs:
            return
        coerced = _coerce_value(default_val)
        reshaped = self._reshape_default_for_socket(coerced, sock_type)
        if reshaped is None:
            return
        ni = NodeInput(default=reshaped)
        node.inputs[name] = ni
        node.input_slots.append((name, ni))

    @staticmethod
    def _reshape_default_for_socket(val, sock_type: str):
        """Coerce ``val`` to a shape valid for ``sock_type``'s default_value.

        Returns ``None`` if the value cannot be meaningfully reshaped.
        """
        is_seq = isinstance(val, (list, tuple)) and not isinstance(val, (str, bytes))
        if sock_type in ("VALUE", "INT", "BOOLEAN"):
            if is_seq:
                # First channel beats dropping the value entirely.
                return _coerce_value(val[0]) if len(val) > 0 else None
            return val
        if sock_type == "VECTOR":
            if not is_seq:
                f = float(val)
                return (round(f, _FLOAT_ROUND_DP),) * 3
            if len(val) == 3:
                return tuple(val)
            if len(val) == 4:
                return tuple(val[:3])
            if len(val) == 1:
                return (val[0], val[0], val[0])
            return None
        if sock_type == "RGBA":
            if not is_seq:
                f = round(float(val), _FLOAT_ROUND_DP)
                return (f, f, f, 1.0)
            if len(val) == 4:
                return tuple(val)
            if len(val) == 3:
                return (val[0], val[1], val[2], 1.0)
            if len(val) == 1:
                return (val[0], val[0], val[0], 1.0)
            return None
        return None

    @staticmethod
    def _slot_index(bn, socket_obj) -> int:
        """Index of ``socket_obj`` among ENABLED same-name entries in bn.inputs.

        Must count exactly as ``to_bpy`` enumerates target sockets, i.e. only
        enabled ones. ``ShaderNodeMix`` (Blender 3.4+) exposes 10 inputs with
        2-3 enabled depending on ``data_type``; counting disabled siblings gives
        an out-of-range slot_idx, and to_bpy's socket_usage dispatcher then
        IndexErrors, dropping the link and leaving the input at its default
        (symptom: a Mix/MixShader slot reads "white" in the rebuilt graph).

        Compares ``as_pointer()`` rather than ``is`` because bpy returns a fresh
        Python wrapper on each attribute access.
        """
        name = socket_obj.name
        target_ptr = socket_obj.as_pointer()
        idx = 0
        for s in bn.inputs:
            if s.as_pointer() == target_ptr:
                return idx
            if s.name == name and getattr(s, "enabled", True):
                idx += 1
        return 0

    def _extract_node(self, bn, nid: str) -> ShaderNode:
        node_type = bn.bl_idname
        props = self._read_props(bn, node_type)
        input_slots, inputs = self._read_inputs(bn)
        # Value/RGB carry data on outputs[0].default_value, not on any input.
        # Emit it as a DSL ``in <name>=<val>`` line so the value round-trips;
        # to_bpy redirects it back to the output for these node types.
        if node_type == "ShaderNodeValue":
            self._inject_output_default(
                bn, "Value", input_slots, inputs, scalar=True
            )
        elif node_type == "ShaderNodeRGB":
            self._inject_output_default(
                bn, "Color", input_slots, inputs, scalar=False
            )
        ramp_props = (
            self._read_ramp_props(bn)
            if node_type == "ShaderNodeValToRGB" else {}
        )
        ramp = self._read_ramp(bn) if node_type == "ShaderNodeValToRGB" else []
        curves = (self._read_curves(bn)
                  if node_type in _CURVE_CHANNEL_MAP else [])
        return ShaderNode(
            id=nid,
            node_type=node_type,
            props=props,
            inputs=inputs,
            input_slots=input_slots,
            ramp_props=ramp_props,
            ramp_elements=ramp,
            curve_points=curves,
        )

    @staticmethod
    def _inject_output_default(bn, sock_name: str, input_slots, inputs,
                               scalar: bool) -> None:
        """Read bn.outputs[0].default_value and inject as a DSL ``in`` slot."""
        outs = getattr(bn, "outputs", None)
        if not outs or len(outs) == 0:
            return
        raw = getattr(outs[0], "default_value", None)
        if raw is None:
            return
        coerced = float(raw) if scalar else _coerce_value(raw)
        ni = NodeInput(default=round(coerced, _FLOAT_ROUND_DP)
                        if isinstance(coerced, float) else coerced)
        input_slots.append((sock_name, ni))
        inputs.setdefault(sock_name, ni)

    @staticmethod
    def _read_props(bn, node_type: str) -> Dict[str, Any]:
        allow = NODE_PROP_ALLOWLIST.get(node_type, ())
        always_write = _NODE_PROPS_ALWAYS_WRITE.get(node_type, frozenset())
        out: Dict[str, Any] = {}
        for p in allow:
            if not hasattr(bn, p):
                continue
            val = getattr(bn, p)
            # Values equal to the RNA default are skipped to keep the DSL
            # clean, except socket-layout props: a rebuild's freshly-created
            # node may carry a different Blender-version default, leaving DSL
            # links pointing at sockets that do not exist (``.W`` on a 4D
            # WhiteNoise defaulting to 3D) and raising IndexError in to_bpy.
            if p not in always_write:
                try:
                    rna_default = bn.bl_rna.properties[p].default
                    if val == rna_default:
                        continue
                except (KeyError, AttributeError):
                    pass
            out[p] = _coerce_value(val)
        return out

    @staticmethod
    def _read_inputs(bn) -> Tuple[List[Tuple[str, NodeInput]], Dict[str, NodeInput]]:
        slots: List[Tuple[str, NodeInput]] = []
        inputs_dict: Dict[str, NodeInput] = {}
        # NodeReroute's socket type is set by downstream consumers at
        # evaluation time, so a default captured as VALUE here crashes on
        # rebuild if the chain resolves it to VECTOR. Reroutes propagate signal
        # only; never emit their defaults.
        if getattr(bn, "bl_idname", "") == "NodeReroute":
            return slots, inputs_dict
        enabled_non_shader = [
            sock for sock in bn.inputs
            if getattr(sock, "enabled", True)
            and getattr(sock, "type", "") != "SHADER"
        ]
        name_counts: Dict[str, int] = {}
        for sock in enabled_non_shader:
            name_counts[sock.name] = name_counts.get(sock.name, 0) + 1
        for sock in bn.inputs:
            if not getattr(sock, "enabled", True):
                continue
            # Shader sockets have no default_value.
            if getattr(sock, "type", "") == "SHADER":
                continue
            if sock.is_linked:
                # A linked socket carries its wiring on the DSL link line, but
                # repeated names (Math.Value, MapRange.Value) still need a
                # positional placeholder, or a sibling's default slides into the
                # linked slot when to_bpy rebuilds.
                if name_counts.get(sock.name, 0) > 1:
                    ni = NodeInput(default=None)
                    slots.append((sock.name, ni))
                    inputs_dict.setdefault(sock.name, ni)
                continue
            raw = getattr(sock, "default_value", None)
            if raw is None:
                continue
            ni = NodeInput(default=_coerce_value(raw))
            slots.append((sock.name, ni))
            inputs_dict.setdefault(sock.name, ni)
        return slots, inputs_dict

    @staticmethod
    def _read_ramp(bn) -> List[Tuple[Any, float]]:
        if not (hasattr(bn, "color_ramp") and bn.color_ramp):
            return []
        out = []
        for elt in bn.color_ramp.elements:
            color = tuple(round(float(c), _FLOAT_ROUND_DP) for c in elt.color)
            out.append((color, round(float(elt.position), _FLOAT_ROUND_DP)))
        return out

    @staticmethod
    def _read_ramp_props(bn) -> Dict[str, Any]:
        if not (hasattr(bn, "color_ramp") and bn.color_ramp):
            return {}
        out: Dict[str, Any] = {}
        for prop in ("interpolation", "color_mode", "hue_interpolation"):
            if hasattr(bn.color_ramp, prop):
                out[prop] = _coerce_value(getattr(bn.color_ramp, prop))
        return out

    @staticmethod
    def _read_curves(bn) -> List[Tuple[str, float, float]]:
        if not (hasattr(bn, "mapping") and hasattr(bn.mapping, "curves")):
            return []
        ch_map = _CURVE_CHANNEL_MAP.get(bn.bl_idname, {})
        out: List[Tuple[str, float, float]] = []
        for idx, curve in enumerate(bn.mapping.curves):
            ch = ch_map.get(idx, "C")
            for pt in curve.points:
                out.append((
                    ch,
                    round(float(pt.location[0]), _FLOAT_ROUND_DP),
                    round(float(pt.location[1]), _FLOAT_ROUND_DP),
                ))
        return out


def extract_texture_paths(material_or_tree) -> List[Dict[str, str]]:
    """Return texture filepaths and colorspace for every ShaderNodeTexImage
    reachable after flattening. Walks the same tree as ``ShaderGraph.from_bpy``,
    so node ids match what the DSL will contain.

    ``colorspace`` matters for PBR correctness: normal/roughness/displacement
    maps must load as ``Non-Color`` or their values get gamma-corrected.
    Must run inside Blender.
    """
    extractor = _BpyExtractor()
    graph = extractor.run(material_or_tree)

    # Re-walk to map each bpy node to its filepath, then reuse the DSL ids the
    # extractor already assigned.
    entries: List[Dict[str, str]] = []

    def _visit(tree):
        for bn in tree.nodes:
            if bn.bl_idname == "ShaderNodeTexImage":
                dsl_id = extractor._id_map.get(bn.as_pointer())
                if dsl_id is None:
                    continue
                img = getattr(bn, "image", None)
                fp = ""
                colorspace = ""
                alpha_mode = ""
                file_format = ""
                if img is not None:
                    fp = getattr(img, "filepath", "") or ""
                    if fp.startswith("//"):
                        try:
                            import bpy  # type: ignore
                            fp = bpy.path.abspath(fp)
                        except Exception:
                            pass
                    cs = getattr(img, "colorspace_settings", None)
                    if cs is not None:
                        colorspace = getattr(cs, "name", "") or ""
                    alpha_mode = getattr(img, "alpha_mode", "") or ""
                    file_format = getattr(img, "file_format", "") or ""
                entries.append({
                    "node_id": dsl_id,
                    "filepath": fp,
                    "colorspace": colorspace,
                    "alpha_mode": alpha_mode,
                    "file_format": file_format,
                })
            elif bn.bl_idname == "ShaderNodeGroup" and bn.node_tree is not None:
                _visit(bn.node_tree)

    root = getattr(material_or_tree, "node_tree", material_or_tree)
    _visit(root)

    del graph
    return entries


class ReplaceShaderNodesTool:
    """Apply a local replacement to an independent, fully validated graph."""

    name = "replace_shader_nodes"
    description = "Replace specified ShaderGraph nodes with validated replacement DSL."
    inputs = {
        "graph": {"type": "object", "description": "Existing graph."},
        "node_ids_to_remove": {"type": "array", "description": "Node IDs to remove."},
        "replacement_dsl": {"type": "string", "description": "Replacement DSL."},
    }
    output_type = "object"

    def forward(self, graph: ShaderGraph, node_ids_to_remove: list[str], replacement_dsl: str) -> ShaderGraph:
        if not isinstance(node_ids_to_remove, list) or any(
            not isinstance(node_id, str) or not node_id.strip()
            for node_id in node_ids_to_remove
        ):
            raise ValueError("node_ids_to_remove must be an array of nonempty string IDs.")
        if not isinstance(replacement_dsl, str):
            raise ValueError("replacement_dsl must be a string.")
        ids = set(node_ids_to_remove)
        if len(ids) != len(node_ids_to_remove):
            raise ValueError("Duplicate removal node IDs.")
        missing = ids - graph.nodes.keys()
        if missing:
            raise ValueError(f"Unknown removal node IDs: {sorted(missing)}")
        patch = ShaderGraph.from_dsl(replacement_dsl, validate=False)
        if patch.errors:
            raise ValueError("Replacement DSL has syntax errors:\n- " + "\n- ".join(patch.errors))
        conflicts = patch.nodes.keys() & (graph.nodes.keys() - ids)
        if conflicts:
            raise ValueError(f"Replacement node IDs conflict with surviving nodes: {sorted(conflicts)}")
        if patch.material_props:
            raise ValueError("Replacement DSL must contain only node definitions and links.")
        candidate = deepcopy(graph)
        removed_links = candidate.remove_nodes(ids)
        candidate.replace_nodes(patch, removed_links=removed_links)
        full_dsl = candidate.to_dsl()
        validated = ShaderGraph.from_dsl(full_dsl)
        if validated.errors:
            raise ValueError("Patched graph has structural errors:\n- " + "\n- ".join(validated.errors))
        candidate.errors = []
        candidate._raw_dsl = full_dsl
        return candidate
