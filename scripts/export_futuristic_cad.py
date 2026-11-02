"""Run with FreeCADCmd; export the actual FCStd surfaces, in metres/link frames.

flatpak run --command=FreeCADCmd org.freecad.FreeCAD scripts/export_futuristic_cad.py
The CAD is an appearance concept: the explicit rigid-part mapping is provisional.
"""

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import FreeCAD as App
import Mesh
import MeshPart

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/robot/source/cad/MicroDuck_Futuristic.FCStd"
OUT = ROOT / "assets/robot/meshes/futuristic"
OUT.mkdir(parents=True, exist_ok=True)
doc = App.openDocument(str(SOURCE))
with zipfile.ZipFile(SOURCE) as archive:
    gui = ET.fromstring(archive.read("GuiDocument.xml"))
colors = {}
for obj in gui.find("ViewProviderData"):
    color = obj.find("Properties/Property[@name='LineColor']/PropertyColor")
    if color is not None:
        packed = int(color.get("value"))
        colors[obj.get("name")] = [((packed >> shift) & 255) / 255 for shift in (24, 16, 8)] + [1]
hip_angle = math.atan2(17, 37)
shin_angle = -math.atan2(21, 44)
parts = []
for obj in doc.Objects:
    if not obj.isDerivedFrom("Part::Feature") or obj.Shape.isNull():
        continue
    name = obj.Name
    if name.startswith(("Left_", "Right_")):
        side = "l" if name.startswith("Left_") else "r"
        y = 0.034 if side == "l" else -0.034
        z = obj.Shape.BoundBox.Center.z
        if (
            any(
                word in name
                for word in ("sole", "foot_deck", "ankle_bracket", "toe_grip", "deck_fastener")
            )
            or z < 31
        ):
            link, pivot, angle = side + "_foot", (0.005, y, 0.021), 0
        elif "hip_roll_servo" in name:
            link, pivot, angle = side + "_hip_roll_link", (0.001, y, 0.102), 0
        elif (
            any(word in name for word in ("shin_", "link_hole"))
            or ("joint_" in name or "axle_bolt" in name)
            and z < 80
        ):
            link, pivot, angle = side + "_knee_link", (-0.016, y, 0.065), shin_angle
        else:
            link, pivot, angle = side + "_thigh", (0.001, y, 0.102), hip_angle
    elif obj in doc.getObject("Head").Group:
        link, pivot, angle = "head", (0, 0, 0.190), 0
    else:
        link, pivot, angle = "torso", (0, 0, 0.116), 0
    mesh = MeshPart.meshFromShape(
        Shape=obj.Shape, LinearDeflection=0.25, AngularDeflection=0.25, Relative=False
    )
    vertices, faces = mesh.Topology
    c, s = math.cos(angle), math.sin(angle)
    transformed = []
    for v in vertices:
        x, y, z = -v.x * 0.001 - pivot[0], -v.y * 0.001 - pivot[1], v.z * 0.001 - pivot[2]
        transformed.append((c * x - s * z, y, s * x + c * z))
    target = OUT / (name + ".stl")
    Mesh.Mesh([tuple(transformed[i] for i in face) for face in faces]).write(str(target))
    parts.append(
        dict(
            name=name,
            link=link,
            file=str(target.relative_to(ROOT)),
            rgba=colors.get(name, [0.2, 0.2, 0.2, 1]),
            triangles=len(faces),
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )
    )
manifest = dict(
    source=str(SOURCE.relative_to(ROOT)),
    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    units="metres",
    cad_to_sim="rotate 180 degrees about z; scale mm to m",
    articulation="explicit provisional rigid mapping; cables do not deform",
    parts=parts,
)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Exported", len(parts), "actual CAD parts", flush=True)
App.closeDocument(doc.Name)
