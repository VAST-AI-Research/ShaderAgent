PROMPT_ENRICH_TEMPLATE = """You are a professional prompt enhancer specialized in material visualization and texture generation.

Your task is to rewrite the user's input into a precise image prompt that generates a clear, complete, and technically analyzable representation of a material or surface.

Core Objectives:
1.  **Visual Completeness:** The target object or surface carrier must be fully contained within the frame with visible margins on all sides.
2.  **Material Homogeneity:** The material must be strictly homogeneous, spatially invariant, or repetitive (tileable).

Material Constraints (Strict Enforcement):
-   **Definition of Homogeneity:** Treat the material as if it will be baked into a texture map. A random crop from any part of the surface should look consistent with any other part.
-   **No Unique Features:** Strictly remove or rewrite any descriptions implying a central logo, specific placement, gradient meant for a specific shape, or non-repeating decals.
-   **Repetition:** If there are patterns (embroidery, scratches, grain), describe them as an "all-over pattern," "seamless tiling," or "uniform distribution."

Composition and Framing Rules:
-   **Containment:** Describe the subject as "isolated" or "floating" in the center with "wide margins" or "ample negative space" around it.
-   **Avoid Cropping:** Explicitly command the view to be "zoomed out slightly" or "full shot" to prevent edges from touching the image border.
-   **Carrier Object:** If no object is specified, use a "flat square fabric swatch," "sphere," or "draped cloth" that is fully visible against a neutral background.

Enhancement Rules:
-   Focus on physical attributes: roughness, specularity, normal map details (bumps/relief), and fiber structure.
-   Lighting: Use "studio lighting" or "soft ambient occlusion" to show texture depth without harsh shadows that obscure the pattern.

Output Constraints:
-   Output ONLY the final enhanced image prompt.
-   Do NOT include explanations or prefixes.

User input:
{description}
"""

IMAGE_GENERATE_TEMPLATE = """You are an image generation model specialized in producing high-fidelity material reference images.

Your task is to generate an image that strictly adheres to the prompt, prioritizing material consistency and perfect framing.

CRITICAL Constraint 1: COMPOSITION & FRAMING (Zero Tolerance for Cropping)
* **Full Visibility:** The subject must be 100% visible. No part of the object, swatch, or shape usually touches the edge of the canvas.
* **Safety Margin:** Ensure there is a visible gap (padding/negative space) between the subject and the four borders of the image.
* **Center Bias:** Place the subject in the visual center, surrounded by a neutral background.

CRITICAL Constraint 2: MATERIAL HOMOGENEITY (Texture Consistency)
* **Spatial Invariance:** The material surface must be uniform. Do not generate unique variations, fading, or specific localized wear (e.g., worn edges vs. clean center). The texture must appear consistent across the entire surface.
* **Repetitive Logic:** If the prompt implies a pattern, render it as a seamless, tiling, all-over pattern.
* **No Central Focal Points:** Avoid placing a unique design or logo in the middle. The surface information should be distributed equally.

Rendering Priorities:
* **Physical Plausibility:** Accurately simulate light interaction (PBR-like qualities), surface relief, and tactile details.
* **Clarity:** Use even, neutral studio lighting. Avoid artistic vignettes or dramatic shadows that hide the material edges.

Output Requirements:
* Produce a single, coherent image.
* The subject should occupy roughly 70-80% of the canvas to ensure details are visible but edges are NOT cut off.

The prompt is:
{prompt}
"""

TEXTURE_GENERATION_TEMPLATE = """You are an expert texture artist specialized in creating seamless, high-quality PBR texture maps.

Your task is to generate a base color (albedo) texture map based on the provided reference image(s) and description.

Input Description:
{description}

Requirements:
1.  **Seamlessness:** The texture must be perfectly tileable (seamless) in both X and Y directions.
2.  **Flatness:** The texture should represent the surface color (albedo) without baked-in lighting, strong shadows, or specular highlights. It should be a flat scan of the material.
3.  **Consistency:** The texture must match the visual characteristics (color, pattern, scale) of the material shown in the reference image(s). If multiple images are provided, compare them to distinguish true base color from lighting effects.
4.  **Resolution:** High detail and sharpness.

Output:
-   A single square image representing the base color texture.
"""

CLASSIFICATION_TEMPLATE = """You are a vision-language material classifier for 3D shader/texture workflows.

You are given:
- Reference image(s) showing a material on a carrier object, such as a swatch,
  sphere, cloth, object part, or full object.
- An optional text description naming the part or material of interest.

Produce, in one shot:
1. One label describing how that material should be authored.
2. A tight bounding box around the target material region in the FIRST reference image.

There are only two labels, distinguished by where the visible variation lives:

- PROCEDURAL — the look is a property of the surface STRUCTURE: micro-geometry,
  fiber direction, layering, anisotropy, sub-surface variation. Shader nodes
  (noise / voronoi / wave / bump / displacement / anisotropy / coordinate-driven
  masks) model exactly this.

- TEXTURE — the look is 2D IMAGE CONTENT painted, printed, or projected onto an
  otherwise plain surface. The defining identity is the picture itself, and is
  authored as a basecolor (and companion) image map.

### Input Contract

Each item is one target material. The pipeline has already filtered out
multi-material grouping and UV problems — do not reason about them. If the
description names a material, classify that one; otherwise classify the
dominant single material in the FIRST image. Text is a pointer; if it
conflicts with the image, trust the image. Multiple images show the same
material from different aspects — cross-reference them.

### STEP 1 — Classify

One question:

  Is the defining variation produced by the surface STRUCTURE, or is it
  IMAGE CONTENT laid on top of an otherwise plain surface?

Concrete check — the lighting test:

  Imagine the material under a different lighting direction. Do the pattern's
  highlights, shadows, or anisotropic streaks shift with the light?

  - YES, the pattern reacts to light → STRUCTURE → PROCEDURAL.
  - NO, the pattern is a flat picture and only its overall brightness changes
    → IMAGE CONTENT → TEXTURE.

Guards against common mis-classification:

- Periodicity alone decides nothing. Ribbed knit is periodic structure
  (PROCEDURAL); a wallpaper motif is periodic image content (TEXTURE).
- "Natural / organic / complex-looking" is not a reason to pick TEXTURE.
  Wood grain, stone veining, weathering, scales, foam, brushed metal are all
  STRUCTURE — they pass the lighting test.
- Per-pixel reproduction is not required for PROCEDURAL. A semantically
  faithful, editable shader is enough.
- "A photo could also represent this" is not a reason to pick TEXTURE.
- When the surface is plain except for a depicted picture or graphic, it is
  TEXTURE regardless of how natural-looking the picture is.

### STEP 2 — Ground

Return an axis-aligned bounding box around the target material region in the
FIRST reference image.

- The box must enclose a region authorable by a SINGLE ShaderGraph at the
  chosen label. Composite materials are fine as long as one graph (with
  masks / sub-shaders) can capture them.
- Exclude background, unrelated parts, and neighboring regions that would
  require a separate ShaderGraph.
- Tighten the box; do not pad.

### Output Format

Return a single JSON object, nothing else.

The bbox follows the model's native convention:
[ymin, xmin, ymax, xmax]

Use integers in [0, 1000], where:
- 0 = top or left edge
- 1000 = bottom or right edge
- ymin < ymax
- xmin < xmax

{
  "reasoning": "<one short sentence: which authoring route was chosen and why, and what region the bbox encloses>",
  "bbox": [ymin, xmin, ymax, xmax],
  "label": "PROCEDURAL" | "TEXTURE"
}

No markdown fences, no prose outside the JSON.
"""

BLENDER_SHADER_NODE_SIGNATURES = """ShaderNodeTexCoord
  out Generated (VECTOR)
  out Normal (VECTOR)
  out Object (VECTOR)
  out Camera (VECTOR)
  out Window (VECTOR)
  out Reflection (VECTOR)
  # NOTE: UV and Generated output is NOT available (meshes lack proper UV unwrapping, and may have different scale).
  # Use ONLY the Object output for spatial coordinates.

ShaderNodeBsdfPrincipled
  prop distribution='GGX'
  in Base Color (RGBA)
  in Metallic (VALUE)
  in Roughness (VALUE)
  in IOR (VALUE)
  in Alpha (VALUE)
  in Normal (VECTOR)
  in Subsurface Weight (VALUE)
  in Subsurface Radius (VECTOR)
  in Subsurface Scale (VALUE)
  in Subsurface Anisotropy (VALUE)
  in Specular IOR Level (VALUE)
  in Specular Tint (RGBA)
  in Anisotropic (VALUE)
  in Anisotropic Rotation (VALUE)
  in Tangent (VECTOR)
  in Transmission Weight (VALUE)
  in Coat Weight (VALUE)
  in Coat Roughness (VALUE)
  in Coat IOR (VALUE)
  in Coat Tint (RGBA)
  in Coat Normal (VECTOR)
  in Sheen Weight (VALUE)
  in Sheen Roughness (VALUE)
  in Sheen Tint (RGBA)
  in Emission Color (RGBA)
  in Emission Strength (VALUE)
  out BSDF (SHADER)
  # Dynamic Rules:
  # * IF subsurface_method=='RANDOM_WALK_SKIN': ADD Input 'Subsurface IOR'
  # * IF subsurface_method=='BURLEY': REMOVE Input 'Subsurface Anisotropy'

ShaderNodeDisplacement
  prop space='OBJECT'
  in Height (VALUE)
  in Midlevel (VALUE)
  in Scale (VALUE)
  in Normal (VECTOR)
  out Displacement (VECTOR)

ShaderNodeOutputMaterial
  prop target='ALL'
  in Surface (SHADER)
  in Volume (SHADER)
  in Displacement (VECTOR)

ShaderNodeCombineXYZ
  in X (VALUE)
  in Y (VALUE)
  in Z (VALUE)
  out Vector (VECTOR)

ShaderNodeMath
  in Value (VALUE)
  in Value (VALUE)
  out Value (VALUE)
  # Dynamic Rules:
  # * IF operation=='RADIANS': REMOVE Input 'Value'
  # * IF operation=='MULTIPLY_ADD': ADD Input 'Value'

ShaderNodeVectorMath
  in Vector (VECTOR)
  in Vector (VECTOR)
  out Vector (VECTOR)
  # Dynamic Rules:
  # * IF operation=='NORMALIZE': REMOVE Input 'Vector'
  # * IF operation=='DOT_PRODUCT': ADD Output 'Value', REMOVE Output 'Vector'
  # * IF operation=='SCALE': ADD Input 'Scale', REMOVE Input 'Vector'

ShaderNodeVectorRotate
  in Vector (VECTOR)
  in Center (VECTOR)
  in Axis (VECTOR)
  in Angle (VALUE)
  out Vector (VECTOR)
  # Dynamic Rules:
  # * IF rotation_type=='EULER_XYZ': ADD Input 'Rotation', REMOVE Input 'Axis', REMOVE Input 'Angle'

ShaderNodeTexImage
  prop projection='FLAT'
  prop interpolation='Linear'
  prop extension='REPEAT'
  in Vector (VECTOR)
  out Color (RGBA)
  out Alpha (VALUE)

ShaderNodeMix
  prop data_type='RGBA'
  prop factor_mode='UNIFORM'
  prop blend_type='MULTIPLY'
  in Factor (VALUE)
  in A (VALUE)
  in B (VALUE)
  out Result (VALUE)

ShaderNodeBrightContrast
  in Color (RGBA)
  in Bright (VALUE)
  in Contrast (VALUE)
  out Color (RGBA)

ShaderNodeHueSaturation
  in Hue (VALUE)
  in Saturation (VALUE)
  in Value (VALUE)
  in Fac (VALUE)
  in Color (RGBA)
  out Color (RGBA)

ShaderNodeInvert
  in Fac (VALUE)
  in Color (RGBA)
  out Color (RGBA)

ShaderNodeNormalMap
  prop space='TANGENT'
  in Strength (VALUE)
  in Color (RGBA)
  out Normal (VECTOR)

ShaderNodeRGB
  out Color (RGBA)

ShaderNodeBump
  in Strength (VALUE)
  in Distance (VALUE)
  in Height (VALUE)
  in Normal (VECTOR)
  out Normal (VECTOR)

ShaderNodeLayerWeight
  in Blend (VALUE)
  in Normal (VECTOR)
  out Fresnel (VALUE)
  out Facing (VALUE)

ShaderNodeTexVoronoi
  in Vector (VECTOR)
  in Scale (VALUE)
  in Detail (VALUE)
  in Roughness (VALUE)
  in Lacunarity (VALUE)
  in Randomness (VALUE)
  out Distance (VALUE)
  out Color (RGBA)
  out Position (VECTOR)
  # Dynamic Rules:
  # * IF voronoi_dimensions=='4D' AND distance=='EUCLIDEAN' AND feature=='F1': ADD Input 'W', ADD Output 'W'
  # * IF voronoi_dimensions=='4D' AND distance=='EUCLIDEAN' AND feature=='SMOOTH_F1': ADD Input 'W', ADD Input 'Smoothness', ADD Output 'W'
  # * IF voronoi_dimensions=='3D' AND distance=='EUCLIDEAN' AND feature=='SMOOTH_F1': ADD Input 'Smoothness'
  # * IF voronoi_dimensions=='3D' AND distance=='EUCLIDEAN' AND feature=='DISTANCE_TO_EDGE': REMOVE Output 'Color', REMOVE Output 'Position'
  # * IF voronoi_dimensions=='3D' AND distance=='MINKOWSKI' AND feature=='F1': ADD Input 'Exponent'

ShaderNodeMapping
  prop vector_type='POINT'
  in Vector (VECTOR)
  in Location (VECTOR)
  in Rotation (VECTOR)
  in Scale (VECTOR)
  out Vector (VECTOR)

ShaderNodeMixShader
  in Fac (VALUE)
  in Shader (SHADER)
  in Shader (SHADER)
  out Shader (SHADER)

ShaderNodeValToRGB
  in Fac (VALUE)
  out Color (RGBA)
  out Alpha (VALUE)

ShaderNodeAddShader
  in Shader (SHADER)
  in Shader (SHADER)
  out Shader (SHADER)

ShaderNodeFresnel
  in IOR (VALUE)
  in Normal (VECTOR)
  out Fac (VALUE)

ShaderNodeTexNoise
  in Vector (VECTOR)
  in Scale (VALUE)
  in Detail (VALUE)
  in Roughness (VALUE)
  in Lacunarity (VALUE)
  in Distortion (VALUE)
  out Fac (VALUE)
  out Color (RGBA)
  # Dynamic Rules:
  # * IF noise_dimensions=='4D' AND noise_type=='FBM': ADD Input 'W'
  # * IF noise_dimensions=='3D' AND noise_type=='HYBRID_MULTIFRACTAL': ADD Input 'Offset', ADD Input 'Gain'

ShaderNodeNormal
  in Normal (VECTOR)
  out Normal (VECTOR)
  out Dot (VALUE)

ShaderNodeNewGeometry
  out Position (VECTOR)
  out Normal (VECTOR)
  out Tangent (VECTOR)
  out True Normal (VECTOR)
  out Incoming (VECTOR)
  out Parametric (VECTOR)
  out Backfacing (VALUE)
  out Pointiness (VALUE)
  out Random Per Island (VALUE)

ShaderNodeUVMap
  out UV (VECTOR)
  # WARNING: Do NOT use this node. Meshes lack proper UV unwrapping and may have different scale.
  # Use ShaderNodeTexCoord with the Object output instead.

ShaderNodeSeparateXYZ
  in Vector (VECTOR)
  out X (VALUE)
  out Y (VALUE)
  out Z (VALUE)

ShaderNodeSeparateColor
  prop mode='RGB'
  in Color (RGBA)
  out Red (VALUE)
  out Green (VALUE)
  out Blue (VALUE)

ShaderNodeCombineColor
  prop mode='RGB'
  in Red (VALUE)
  in Green (VALUE)
  in Blue (VALUE)
  out Color (RGBA)

ShaderNodeObjectInfo
  out Location (VECTOR)
  out Color (RGBA)
  out Alpha (VALUE)
  out Object Index (VALUE)
  out Material Index (VALUE)
  out Random (VALUE)

ShaderNodeVectorTransform
  prop vector_type='NORMAL'
  prop convert_from='OBJECT'
  prop convert_to='WORLD'
  in Vector (VECTOR)
  out Vector (VECTOR)

ShaderNodeValue
  out Value (VALUE)

ShaderNodeTexChecker
  in Vector (VECTOR)
  in Color1 (RGBA)
  in Color2 (RGBA)
  in Scale (VALUE)
  out Color (RGBA)
  out Fac (VALUE)

ShaderNodeClamp
  prop clamp_type='MINMAX'
  in Value (VALUE)
  in Min (VALUE)
  in Max (VALUE)
  out Result (VALUE)

ShaderNodeEmission
  in Color (RGBA)
  in Strength (VALUE)
  out Emission (SHADER)

ShaderNodeRGBCurve
  in Fac (VALUE)
  in Color (RGBA)
  out Color (RGBA)

ShaderNodeTexWave
  prop wave_type='BANDS'
  prop bands_direction='X'
  prop rings_direction='X'
  prop wave_profile='SIN'
  in Vector (VECTOR)
  in Scale (VALUE)
  in Distortion (VALUE)
  in Detail (VALUE)
  in Detail Scale (VALUE)
  in Detail Roughness (VALUE)
  in Phase Offset (VALUE)
  out Color (RGBA)
  out Fac (VALUE)

ShaderNodeMapRange
  prop interpolation_type='LINEAR'
  prop data_type='FLOAT'
  in Value (VALUE)
  in From Min (VALUE)
  in From Max (VALUE)
  in To Min (VALUE)
  in To Max (VALUE)
  out Result (VALUE)

ShaderNodeVolumeAbsorption
  in Color (RGBA)
  in Density (VALUE)
  out Volume (SHADER)

ShaderNodeTexGradient
  prop gradient_type='LINEAR'
  in Vector (VECTOR)
  out Color (RGBA)
  out Fac (VALUE)

ShaderNodeTexWhiteNoise
  prop noise_dimensions='3D'
  in Vector (VECTOR)
  out Value (VALUE)
  out Color (RGBA)

ShaderNodeBevel
  in Radius (VALUE)
  in Normal (VECTOR)
  out Normal (VECTOR)

ShaderNodeTexMagic
  in Vector (VECTOR)
  in Scale (VALUE)
  in Distortion (VALUE)
  out Color (RGBA)
  out Fac (VALUE)

ShaderNodeTexBrick
  in Vector (VECTOR)
  in Color1 (RGBA)
  in Color2 (RGBA)
  in Mortar (RGBA)
  in Scale (VALUE)
  in Mortar Size (VALUE)
  in Mortar Smooth (VALUE)
  in Bias (VALUE)
  in Brick Width (VALUE)
  in Row Height (VALUE)
  out Color (RGBA)
  out Fac (VALUE)

ShaderNodeFloatCurve
  in Factor (VALUE)
  in Value (VALUE)
  out Value (VALUE)

ShaderNodeVectorCurve
  in Fac (VALUE)
  in Vector (VECTOR)
  out Vector (VECTOR)

ShaderNodeAmbientOcclusion
  in Color (RGBA)
  in Distance (VALUE)
  in Normal (VECTOR)
  out Color (RGBA)
  out AO (VALUE)

ShaderNodeCombineHSV
  in H (VALUE)
  in S (VALUE)
  in V (VALUE)
  out Color (RGBA)

ShaderNodeSeparateHSV
  in Color (RGBA)
  out H (VALUE)
  out S (VALUE)
  out V (VALUE)

ShaderNodeVectorDisplacement
  prop space='TANGENT'
  in Vector (RGBA)
  in Midlevel (VALUE)
  in Scale (VALUE)
  out Displacement (VECTOR)
  
ShaderNodeGroup
  prop name=<name>
  out BSDF (Shader)
  out Displacement (Vector)
  out Volume (Shader)
  # placeholder for subgraph node groups"""

COMMON_DSL_BLOCK = f"""----------------------------------------------------------------
### DSL Syntax

1. Define a Node:
node <node_id> <NodeType>
  prop <property_name>=<value>
  in <input_socket_name>=<literal_default_value>

2. Define a Link (node-to-node connection):
link <source_node_id>.<output_socket> -> <target_node_id>.<input_socket>

3. Special Node Commands:
- For ShaderNodeValToRGB (Color Ramp):
    ramp <color> <position>
    Example: ramp '#FFFFFF' 0.5
    * Note: The node usage needs to clear default elements first. The `ramp` commands will add elements.
- For ShaderNodeRGBCurve, ShaderNodeVectorCurve, ShaderNodeFloatCurve:
    point <channel_C_R_G_B_or_X_Y_Z> <x> <y>
    Example: point 'C' 0.0 0.2
    * Note: 'C' is the combined channel for RGB curves.

**CRITICAL RULES:**
- `in` values MUST be literal constants (numbers, tuples, color hex strings).
  NEVER write node references like `SomeNode.Output` in an `in` field.
- All node-to-node connections MUST use `link` statements.
- If an input receives its value from another node, do NOT set a default for it
  in the `in` line; just write the `link` instead.

Correct Example:
```
node Noise ShaderNodeTexNoise
  in Scale=50.0

node Disp ShaderNodeDisplacement
  in Scale=0.02
  in Midlevel=0.5

node BSDF ShaderNodeBsdfPrincipled
  in Base Color=(0.8, 0.2, 0.1, 1.0)
  in Roughness=0.5

node Output ShaderNodeOutputMaterial

link Noise.Fac -> Disp.Height
link BSDF.BSDF -> Output.Surface
link Disp.Displacement -> Output.Displacement
```

----------------------------------------------------------------
### Available Nodes & Signatures

Use ONLY the nodes, properties, and sockets defined below.
You MUST NOT invent nodes, sockets, or properties that are not listed.

{BLENDER_SHADER_NODE_SIGNATURES}

----------------------------------------------------------------
### Surface Micro-Structure Guideline

For surface micro-details (cracks, pores, grain, relief, bumps, engravings,
ridges, scratches, etc.):

- Always prefer ShaderNodeDisplacement connected to
  ShaderNodeOutputMaterial.Displacement to produce real geometry.
  Performance is NOT a concern (we always render with Cycles).
- Use ShaderNodeBump / ShaderNodeNormalMap ONLY for extremely subtle,
  purely visual detail that does not warrant geometric displacement.
- If an existing graph uses Bump/NormalMap for noticeable surface structure,
  prefer replacing it with Displacement.
"""

SHADER_REFINE_FULL_TEMPLATE = """You are an expert technical artist refining a Blender shader graph.

You will be given:
- The target material description.
- Reference image(s) — physical evidence of the target.
- The previous DSL (the current best graph for this task).
- Critic feedback describing the remaining visual mismatches.

Your job: emit a COMPLETE new DSL that closes those gaps. Keep, modify, or
replace any part of the previous structure — change as little or as much as
the visual gap warrants.

{role_context}
""" + COMMON_DSL_BLOCK + """
----------------------------------------------------------------
### Previous DSL

```
{previous_dsl}
```

----------------------------------------------------------------
### Critic Feedback

```
{feedback}
```

----------------------------------------------------------------
### Task

{description}

----------------------------------------------------------------
### Output

Write Python code that:
1. Builds a string holding the FULL new DSL.
2. Calls `ShaderGraph_DSL_Parser(dsl_string=...)`.
3. Returns the resulting ShaderGraph as the final answer.

Do NOT `import` any module. Plain string + tool call is all you need.
"""

SHADER_GENERATION_TEMPLATE = """You are an expert technical artist specializing in Blender shader nodes.
Your task is to generate a shader graph definition for the target material,
faithfully reproducing its material behavior as observed in the provided reference image(s).

You must use the following Domain Specific Language (DSL) to define the graph.

""" + COMMON_DSL_BLOCK + """
----------------------------------------------------------------
### Task Context

Task:
{description}

Reference Image(s):
Use all provided reference images to infer the physical and visual properties
of the target material. Treat the images as physical evidence, not as style
inspiration.

----------------------------------------------------------------
### Required Physical Reasoning Order

You MUST internally reason about the material using the following fixed order.
You may decide that certain steps are NOT APPLICABLE for this material, but
you must not violate the order.

0. Spatial-variation analysis:
   - For the visible surface, decide whether each appearance aspect (color,
     roughness, normal/relief, metallic, etc.) is:
       (a) spatially uniform,
       (b) driven by procedural noise / patterns (no geometry correlation), or
       (c) correlated with the mesh's own geometry — i.e. with curvature,
           surface normal direction, ambient occlusion / cavity depth,
           view-angle (Fresnel), object-space height/axis, or distance to
           nearest edge.
   - This decision determines, for each subsequent step, what feeds the
     corresponding BSDF input: a constant, a procedural texture, or a
     geometry-derived signal (Pointiness / AmbientOcclusion / Fresnel /
     LayerWeight / Bevel / object coordinate). Plan the masks/inputs
     accordingly before filling channel values.

1. Metalness classification:
   - Determine whether the material is metallic (0 or 1).
   - This decision constrains all subsequent reasoning.

2. Surface structure:
   - Follow the "Surface Micro-Structure Guideline" above: always use
     ShaderNodeDisplacement for surface micro-details (cracks, pores, grain,
     relief). Bump/NormalMap only for negligible visual-only detail.

3. Specular response:
   - Roughness
   - IOR
   - Anisotropy
   - Clear coat / sheen (if applicable)

4. Base appearance:
   - Base Color (diffuse for non-metals, specular color for metals)
   - Emission (if present)

5. Subsurface effects:
   - Subsurface scattering (only if physically plausible)

6. Internal / volumetric effects:
   - Transmission
   - Volume absorption or scattering (only if physically plausible)

You MUST NOT introduce effects that contradict earlier decisions
(e.g. subsurface effects on a clearly metallic surface).

----------------------------------------------------------------
### Graph Design Constraints

- Every node in the graph MUST contribute to the final output.
- Do NOT include unused or decorative nodes.
- Avoid relying on default input values to hide missing logic.
- The graph structure should directly correspond to observable visual features
  in the reference image(s).

----------------------------------------------------------------
### Output Instructions

1. Briefly explain your understanding of the material
   and which physical properties you aim to reproduce.

2. Describe the planned node structure at a high level
   (key nodes and their roles).

3. Generate Python code that:
   a. Constructs a string variable containing ONLY valid DSL code.
   b. Calls the tool `ShaderGraph_DSL_Parser(dsl_string=...)` with that string.
   c. Ends with the tool call so that the ShaderGraph object is returned.

IMPORTANT:
- The DSL must strictly follow the syntax and available signatures.
- The final output MUST result in a valid ShaderGraph object.
- Do NOT `import` any module. Build the DSL as a plain string and call the
  tool — that's all the runtime needs.
"""

DERIVE_PBR_TEMPLATE = """You are an expert technical artist specializing in PBR material creation and Blender shader nodes.
Your task is to generate a shader graph that derives a full PBR material (Roughness, Metallic, etc.) from a provided Base Color texture map.

You will be provided with:
1. Reference image(s) of the material.
2. A user prompt that points at the wanted material on a specific part.
3. A generated Base Color texture map.

Your Goal:
Create a shader graph using the provided DSL that:
1.  Starts with an `ShaderNodeTexImage` node representing the Base Color.
    *   **CRITICAL:** For the `image` in this Image Texture node, just use a python-style comment placeholder `# Placeholder for Base Color texture` instead of actual image loading code. I will replace it later.
2.  Derives other PBR channels (Roughness, Metallic, Normal, Bump, etc.) based on the visual appearance in the reference image(s) and the semantic properties of the material.
    *   You can use constant values if the property is uniform.
    *   Based on the semantic information, you can also derive values from the Base Color using math nodes (e.g., `ShaderNodeMath`, `ShaderNodeValToRGB`, `ShaderNodeSeparateColor`, `ShaderNodeInvert`) if the surface details align with the color pattern (e.g., embroidery may bump out of the background cloth).
3.  There should be a `ShaderNodeOutputMaterial` in the end.

""" + COMMON_DSL_BLOCK + """
### Task
**User Description:** {description}

### Instructions
1. **Analyze the Material:** Look at the reference image(s). If multiple images are provided, they may describe the same material from different aspects (e.g., different lighting directions, angles, or close-ups). Contrast these views to distinguish between temporary lighting effects (like specular highlights) and fixed material properties (like albedo and roughness). How does the material react to light? Is it metallic? Is it rough or shiny? Does the surface relief (bump/normal) match the color pattern? Any other extended attributes, e.g. Subsurface Scattering, emission, transparency?
2. **Design the Graph:**
    *   Create the `ShaderNodeTexImage` for the Base Color (remember the placeholder).
    *   Decide how to generate Roughness, Metallic, Normal, and e.t.c.
    *   Example: If it's a gold thread embroidery on cloth:
        *   Separate the gold color from the background (maybe using `ShaderNodeSeparateColor` or `ShaderNodeValToRGB`).
        *   Use this mask to drive Metallic (1 for gold, 0 for cloth).
        *   Use this mask to drive Roughness (shiny gold, rough cloth).
        *   Use this mask to drive Bump/Normal (embroidery is raised).
3. **Generate Code:** Write a Python script that:
    a. Constructs a string variable containing the DSL code.
    b. Calls the tool `ShaderGraph_DSL_Parser(dsl_string=...)` with that string.
    c. The tool will validate the DSL and return the `ShaderGraph` object.

**Important:**
- Ensure your DSL string is valid.
- The Python code must end with the tool call.
- Do NOT `import` any module — build the DSL string and call the tool.
"""

CRITIC_PAIRWISE_PROMPT = """You are an expert Technical Artist and Material Supervisor.
Pick which of two procedurally generated material candidates more faithfully
reproduces the reference image(s).

References: {ref_indices}.
Candidate A ({a_label}): {a_view_indices}
Candidate B ({b_label}): {b_view_indices}

User Description: {description}

### Comparison rules
- Cross-reference all reference images to fix the material's intrinsic ground truth.
- Each candidate is shown as a single bright_ball render (color/specular/normal under direct light).
- Decouple lighting: ignore differences in shadow direction, reflected colors, or illumination intensity. Judge only intrinsic properties (Albedo, Surface Structure, Roughness, Metallic, Sheen, …).
- The reference list may include a base-color texture map alongside photographs; treat every reference image as ground truth.

### Evaluation protocol

Phase 1 — Decompose the description into 3–6 scoring points. Assign each a positive `weight`; weights MUST sum to 1.0.
Weight sourcing: ≥ 70 %% from what the Reference Image(s) actually show, ≤ 30 %% from the User Description (description picks *which* traits matter; never let its adjectives override the image).

> Example — "Weathered red clay brick with light mortar, surface pitting":
> | # | Question                                                  | max |
> |---|-----------------------------------------------------------|-----|
> | 1 | Muted red/clay base color matching the reference?         | 0.30|
> | 2 | Light mortar at correct scale/width?                      | 0.25|
> | 3 | Visible surface pores/pitting?                            | 0.25|
> | 4 | Subtle roughness variation across the surface?            | 0.20|

Phase 2 — Score A and B independently on the SAME rubric. For each point, assign normalized scores in [0, 1]:
- Full marks: trait clearly present, matches the reference.
- Partial: trait present but noticeably off (wrong scale, intensity, hue).
- Zero: trait missing or contradicting the reference.

Phase 3 — Decide:
- `scores.A` = sum of A's per-point scores (in [0, 1]).
- `scores.B` = sum of B's per-point scores (in [0, 1]).
- `winner` = "A" or "B" from the weighted totals; ties must choose "A".
- `good_enough` = true iff the winner scores >= {good_enough_min} x max on EVERY scoring point.
- `winner_remaining_issues` = structural advice (see Feedback constraint) for whatever the winner still gets less than full marks on. Empty string if `good_enough`.

### Feedback constraint
`winner_remaining_issues` drives a downstream rewrite agent that controls graph STRUCTURE only.
- Speak in terms of: nodes to add / remove / rewire, enum switches (`wave_type`,
  `bands_direction`, `blend_type`, `noise_type`, `feature` …), where in the
  graph the missing channel should plug in (e.g. "the rib pattern needs to
  drive the Displacement input, not just Base Color").
- Do NOT prescribe numeric magnitudes — a parameter tuner explores those.
  "Add a high-frequency noise channel feeding Bump.Height" is good;
  "set Noise.Scale=500" is not.

### Output
Return one `final_answer(...)` call:

final_answer({{
    "winner": "A" | "B",
    "rubric": [{{"dimension": "...", "weight": <float>, "score_a": <float in [0,1]>, "score_b": <float in [0,1]>, "evidence": "..."}}],
    "scores": {{"A": <weighted total in [0,1]>, "B": <weighted total in [0,1]}},
    "reason": "<one-sentence why winner beat the other>",
    "good_enough": <bool>,
    "winner_remaining_issues": "<structural advice, or '' if good_enough>"
}})
"""


INSPECTOR_TEMPLATE = """You are a shader technical artist assistant.
Pick the numeric parameters most worth tuning to close the gap between the
current render and the target material.

You are given (in order):
  Image 1: the reference image showing the target material.
  Image 2: the current rendered output of the graph below.

Compare Image 2 to the target material in Image 1 — that comparison drives
the selection. The textual description below disambiguates which material in
Image 1 to focus on (the reference may show multiple regions).

### Target Material Description

{description}

### Current Shader Graph (DSL)

```
{graph_dsl}
```

### Tunable Parameters

`node_id.param_name = current_value`. Link-driven inputs are already excluded.

{param_list}

### Instructions

1. Identify the largest visual gaps between the target material in the
   reference and the current render (color, displacement depth, pattern
   scale, roughness, anisotropy, …).
2. Map each gap to specific parameters in the tunable list. Prefer params
   that directly control the mismatched dimension (e.g. pattern-scale gap →
   a `Scale` on a procedural texture; displacement-depth gap → a `Disp.Scale`
   or `Bump.Strength`).
3. Pick at most {max_params}.

Return ONLY this JSON object:

{{
  "reasoning": "<≤2 sentences: main visual gaps and why these params target them>",
  "params": [{{"node": "noise1", "param": "Scale"}}, ...]
}}
"""
