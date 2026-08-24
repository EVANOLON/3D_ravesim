"""Replicate shockwave-260122test-Copy4.ipynb simulation setup (cell 5, 'classical 3d').

Extracts the exact config_dict literal from the notebook, applies the final
materials override, and calls multisim.setup_simulation exactly like the
notebook does. Prints the created simulation directory.
"""
import ast
import json
import sys
from pathlib import Path

repo = Path("/mnt/d/rave-sim-main/rave-sim-main")
# Drop the script's own directory from sys.path so the editable install of
# nist_lookup (not the repo's outer nist_lookup/ dir) is used.
sys.path.pop(0)
sys.path.insert(0, str(repo / "big-wave"))

import multisim  # noqa: E402
import propagation  # noqa: E402

notebook_path = repo / "notebooks/shockwavetest_251225/shockwave-260122test-Copy4.ipynb"
nb = json.loads(notebook_path.read_text(encoding="utf-8"))

# Find the first code cell that defines config_dict and calls setup_simulation.
src = None
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = "".join(cell["source"])
    if "config_dict = {" in s and "setup_simulation" in s:
        src = s
        break
assert src is not None, "no setup cell found"

tree = ast.parse(src)

config_dict = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "config_dict":
                config_dict = eval(compile(ast.Expression(node.value), "<cfg>", "eval"),
                                   {"__builtins__": {}, "propagation": propagation, "ii": 0}, {})

assert config_dict is not None, "config_dict literal not found"

# Apply the final un-commented materials override (the shockwave profile list).
def subscript_chain(node):
    """Return (base_name, [slice values]) for a (nested) subscript target."""
    slices = []
    while isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant):
            slices.append(sl.value)
        else:
            return None
        node = node.value
    if isinstance(node, ast.Name):
        return node.id, list(reversed(slices))
    return None

materials_override = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            chain = subscript_chain(t)
            if chain == ("config_dict", ["elements", 0, "materials"]):
                try:
                    materials_override = eval(compile(ast.Expression(node.value), "<mat>", "eval"),
                                              {"__builtins__": {}}, {})
                except (ValueError, SyntaxError):
                    pass

assert materials_override is not None, "materials override not found"
config_dict["elements"][0]["materials"] = materials_override

n_materials = len(config_dict["elements"][0]["materials"])
print(f"materials: {n_materials} layers")
print(f"N: {config_dict['sim_params']['N']}")
print(f"dx: {config_dict['sim_params']['dx']}")
print(f"nr_source_points: {config_dict['multisource']['nr_source_points']}")

simulations_dir = repo / "output"
sim_path = multisim.setup_simulation(config_dict, repo, simulations_dir)
print(f"SIM_PATH={sim_path}")
