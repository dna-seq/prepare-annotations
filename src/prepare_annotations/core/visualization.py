"""
Dagster asset lineage graph visualization.

Generates graphviz diagrams from Dagster asset definitions
for programmatic documentation and pipeline visualization.
"""

from pathlib import Path
from typing import Optional

import graphviz
from dagster import Definitions


# Color scheme for different pipeline groups
ASSET_COLORS = {
    "ensembl": {"fill": "#e3f2fd", "border": "#1976d2", "font": "#0d47a1"},  # Blue
    "longevitymap": {"fill": "#e8f5e9", "border": "#388e3c", "font": "#1b5e20"},  # Green
    "lipidmetabolism": {"fill": "#fff3e0", "border": "#f57c00", "font": "#e65100"},  # Orange
    "vo2max": {"fill": "#fce4ec", "border": "#c2185b", "font": "#880e4f"},  # Pink
    "superhuman": {"fill": "#f3e5f5", "border": "#7b1fa2", "font": "#4a148c"},  # Purple
    "coronary": {"fill": "#ffebee", "border": "#d32f2f", "font": "#b71c1c"},  # Red
    "default": {"fill": "#f5f5f5", "border": "#757575", "font": "#212121"},  # Grey
}

# Node shapes for different asset types
ASSET_SHAPES = {
    "source": "cylinder",  # External data sources
    "upload": "folder",  # Upload/output assets
    "sqlite": "box3d",  # SQLite database files
    "default": "box",  # Regular processing assets
}


def get_asset_group(asset_name: str) -> str:
    """Determine which pipeline group an asset belongs to."""
    name_lower = asset_name.lower()
    for group in ["ensembl", "longevitymap", "lipidmetabolism", "vo2max", "superhuman", "coronary"]:
        if name_lower.startswith(group):
            return group
    return "default"


def get_asset_shape(asset_name: str) -> str:
    """Determine the shape for an asset based on its type."""
    name_lower = asset_name.lower()
    if "source" in name_lower or "ftp" in name_lower:
        return ASSET_SHAPES["source"]
    if "upload" in name_lower:
        return ASSET_SHAPES["upload"]
    if "sqlite" in name_lower:
        return ASSET_SHAPES["sqlite"]
    return ASSET_SHAPES["default"]


def extract_asset_graph(defs: Definitions) -> tuple[set[str], list[tuple[str, str]]]:
    """
    Extract nodes (assets) and edges (dependencies) from Dagster definitions.
    
    Returns:
        Tuple of (set of asset names, list of (upstream, downstream) edges)
    """
    nodes: set[str] = set()
    edges: list[tuple[str, str]] = []
    
    # Get all asset specs from definitions (the new API)
    specs = defs.resolve_all_asset_specs()
    
    for spec in specs:
        asset_name = spec.key.to_user_string()
        nodes.add(asset_name)
        
        # Get dependencies from the spec
        if spec.deps:
            for dep in spec.deps:
                dep_name = dep.asset_key.to_user_string()
                nodes.add(dep_name)
                edges.append((dep_name, asset_name))
    
    return nodes, edges


def generate_lineage_graph(
    defs: Definitions,
    output_path: Optional[Path] = None,
    output_format: str = "jpg",
    title: str = "Prepare Annotations Pipeline",
    rankdir: str = "LR",  # Left to right
    dpi: int = 150,
) -> Path:
    """
    Generate a lineage graph from Dagster definitions.
    
    Args:
        defs: Dagster Definitions object
        output_path: Path for output file (without extension)
        output_format: Output format (jpg, png, svg, pdf)
        title: Graph title
        rankdir: Graph direction (LR=left-right, TB=top-bottom)
        dpi: Resolution for raster formats
        
    Returns:
        Path to the generated graph file
    """
    if output_path is None:
        output_path = Path("images/pipelines")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract graph structure
    nodes, edges = extract_asset_graph(defs)
    
    # Create graphviz digraph
    dot = graphviz.Digraph(
        name="dagster_lineage",
        comment=title,
        format=output_format,
        engine="dot",
    )
    
    # Graph attributes
    dot.attr(
        rankdir=rankdir,
        label=title,
        labelloc="t",
        fontsize="20",
        fontname="Helvetica-Bold",
        dpi=str(dpi),
        bgcolor="white",
        pad="0.5",
        nodesep="0.4",
        ranksep="0.8",
    )
    
    # Default node attributes
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Helvetica",
        fontsize="11",
        margin="0.2,0.1",
    )
    
    # Default edge attributes
    dot.attr(
        "edge",
        arrowsize="0.8",
        color="#666666",
    )
    
    # Group assets by pipeline for subgraph clustering
    groups: dict[str, list[str]] = {}
    for node in nodes:
        group = get_asset_group(node)
        if group not in groups:
            groups[group] = []
        groups[group].append(node)
    
    # Create subgraphs for each group
    for group_name, group_nodes in groups.items():
        colors = ASSET_COLORS.get(group_name, ASSET_COLORS["default"])
        
        with dot.subgraph(name=f"cluster_{group_name}") as sub:
            sub.attr(
                label=group_name.replace("_", " ").title() if group_name != "default" else "",
                style="rounded,dashed",
                color=colors["border"],
                fontcolor=colors["font"],
                fontsize="14",
                fontname="Helvetica-Bold",
            )
            
            for node in group_nodes:
                shape = get_asset_shape(node)
                # Format label with line breaks for long names
                label = node.replace("_", "\\n")
                
                sub.node(
                    node,
                    label=label,
                    fillcolor=colors["fill"],
                    color=colors["border"],
                    fontcolor=colors["font"],
                    shape=shape,
                )
    
    # Add edges
    for upstream, downstream in edges:
        upstream_group = get_asset_group(upstream)
        downstream_group = get_asset_group(downstream)
        
        # Use darker color for cross-group edges
        if upstream_group != downstream_group:
            edge_color = "#333333"
            edge_style = "bold"
        else:
            edge_color = ASSET_COLORS.get(upstream_group, ASSET_COLORS["default"])["border"]
            edge_style = "solid"
        
        dot.edge(upstream, downstream, color=edge_color, style=edge_style)
    
    # Render the graph
    output_file = dot.render(
        filename=str(output_path),
        cleanup=True,  # Remove the .gv source file
    )
    
    return Path(output_file)


def generate_pipeline_graph_from_module(
    module_path: str = "prepare_annotations.definitions",
    output_path: Optional[Path] = None,
    output_format: str = "jpg",
    **kwargs,
) -> Path:
    """
    Generate a lineage graph by importing Dagster definitions from a module.
    
    Args:
        module_path: Dotted module path containing 'defs' object
        output_path: Output file path (without extension)
        output_format: Output format (jpg, png, svg, pdf)
        **kwargs: Additional arguments passed to generate_lineage_graph
        
    Returns:
        Path to the generated graph file
    """
    import importlib
    
    module = importlib.import_module(module_path)
    defs = getattr(module, "defs")
    
    return generate_lineage_graph(
        defs=defs,
        output_path=output_path,
        output_format=output_format,
        **kwargs,
    )
