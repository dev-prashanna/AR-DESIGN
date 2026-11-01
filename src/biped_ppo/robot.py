"""Explicit URDF primitive/inertia import; never infer joints from visual CAD."""

import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .config import ROOT, read


def vec(text, default="0 0 0"):
    return np.fromstring(text or default, sep=" ")


def s(values):
    return " ".join(f"{x:.12g}" for x in values)


def origin(element):
    node = element.find("origin")
    if node is None:
        return {"pos": "0 0 0", "euler": "0 0 0"}
    return {"pos": node.get("xyz", "0 0 0"), "euler": node.get("rpy", "0 0 0")}


def generate(robot_path="configs/robot/custom.yaml", output=None, ground=False, motors=False):
    cfg = read(robot_path)
    tree = ET.parse(ROOT / cfg["urdf"]).getroot()
    mj = ET.Element("mujoco", model=cfg["name"])
    ET.SubElement(mj, "compiler", angle="radian", eulerseq="xyz", fusestatic="false")
    ET.SubElement(
        mj,
        "option",
        timestep=str(cfg["physics_dt"]),
        gravity="0 0 -9.81",
        integrator="implicitfast",
        iterations="30",
    )
    default = ET.SubElement(mj, "default")
    ET.SubElement(
        default, "geom", friction=f"{cfg['friction']} 0.005 0.0001", condim="3", solref="0.008 1"
    )
    world = ET.SubElement(mj, "worldbody")
    if ground:
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            size="2 2 .1",
            contype="2",
            conaffinity="1",
            rgba=".85 .87 .89 1",
        )
        ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
    links = {link_entry.get("name"): link_entry for link_entry in tree.findall("link")}
    joints = tree.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    roots = set(links) - children
    if len(roots) != 1:
        raise ValueError(f"Expected one root: {roots}")
    mats = {m.get("name"): m.find("color").get("rgba") for m in tree.findall("material")}
    audit_links, audit_joints = [], []
    visited = set()

    def build(name, parent, joint=None):
        if name in visited:
            raise ValueError("Robot tree contains cycle or duplicate child")
        visited.add(name)
        link = links[name]
        body = ET.SubElement(
            parent, "body", name=name, **(origin(joint) if joint is not None else {"pos": "0 0 0"})
        )
        if joint is None:
            ET.SubElement(body, "freejoint", name="floating_base")
        elif joint.get("type") != "fixed":
            if joint.get("type") != "revolute":
                raise ValueError("Only fixed/revolute joints supported by this importer")
            lim = joint.find("limit")
            dyn = joint.find("dynamics")
            ET.SubElement(
                body,
                "joint",
                name=joint.get("name"),
                type="hinge",
                axis=joint.find("axis").get("xyz"),
                range=f"{lim.get('lower')} {lim.get('upper')}",
                damping=dyn.get("damping", "0") if dyn is not None else "0",
                frictionloss=dyn.get("friction", "0") if dyn is not None else "0",
            )
        inert = link.find("inertial")
        mass = float(inert.find("mass").get("value"))
        vals = inert.find("inertia").attrib
        inertia = np.array(
            [
                [float(vals["ixx"]), float(vals["ixy"]), float(vals["ixz"])],
                [float(vals["ixy"]), float(vals["iyy"]), float(vals["iyz"])],
                [float(vals["ixz"]), float(vals["iyz"]), float(vals["izz"])],
            ]
        )
        io = origin(inert)
        R = Rotation.from_euler("xyz", vec(io["euler"])).as_matrix()
        inertia = R @ inertia @ R.T
        eigen = np.linalg.eigvalsh(inertia)
        valid = bool(mass > 0 and min(eigen) > 0 and eigen[-1] <= sum(eigen[:2]) + 1e-12)
        if not valid:
            raise ValueError(f"Unphysical inertia: {name}: {eigen}")
        ET.SubElement(
            body,
            "inertial",
            pos=io["pos"],
            mass=str(mass),
            fullinertia=s(
                [
                    inertia[0, 0],
                    inertia[1, 1],
                    inertia[2, 2],
                    inertia[0, 1],
                    inertia[0, 2],
                    inertia[1, 2],
                ]
            ),
        )
        audit_links.append(
            dict(
                name=name,
                mass_kg=mass,
                com_local_m=vec(io["pos"]).tolist(),
                inertia_kg_m2=inertia.tolist(),
                principal_moments=eigen.tolist(),
                valid=valid,
            )
        )
        for kind in ["visual", "collision"]:
            for idx, g in enumerate(link.findall(kind)):
                box = g.find("geometry/box")
                if box is None:
                    raise ValueError("This importer currently accepts supplied box geometry only")
                geom = dict(
                    name=f"{name}_{kind}_{idx}",
                    type="box",
                    size=s(vec(box.get("size")) / 2),
                    **origin(g),
                )
                if kind == "visual":
                    m = g.find("material")
                    geom.update(
                        contype="0",
                        conaffinity="0",
                        group="2",
                        mass="0",
                        rgba=mats.get(m.get("name"), ".3 .3 .3 1"),
                    )
                else:
                    geom.update(
                        priority="1",
                        contype="1",
                        conaffinity="2" if not cfg["self_collision"] else "3",
                        group="3",
                        rgba=".7 .1 .1 0",
                    )
                ET.SubElement(body, "geom", **geom)
        if name in ("l_foot", "r_foot"):
            ET.SubElement(body, "site", name=name + "_contact", pos="0 0 -.008", size=".002")
        for j in joints:
            if j.find("parent").get("link") == name:
                audit_joints.append(
                    dict(
                        name=j.get("name"),
                        type=j.get("type"),
                        parent=name,
                        child=j.find("child").get("link"),
                        origin=origin(j),
                        axis=j.find("axis").attrib if j.find("axis") is not None else None,
                        limits=j.find("limit").attrib if j.find("limit") is not None else None,
                    )
                )
                build(j.find("child").get("link"), body, j)

    build(next(iter(roots)), world)
    if visited != set(links):
        raise ValueError("Disconnected robot links")
    if motors:
        act = ET.SubElement(mj, "actuator")
        for name in cfg["joint_names"]:
            limit = next(j.find("limit") for j in joints if j.get("name") == name)
            effort = float(limit.get("effort"))
            ET.SubElement(
                act,
                "motor",
                name=name + "_motor",
                joint=name,
                gear="1",
                ctrllimited="true",
                ctrlrange=s([-effort, effort]),
            )
    ET.indent(mj)
    xml = ET.tostring(mj, encoding="unicode")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(xml)
    return xml, dict(
        robot=cfg,
        root=next(iter(roots)),
        links=audit_links,
        joints=audit_joints,
        mass_kg=sum(link_entry["mass_kg"] for link_entry in audit_links),
        actuation_order=cfg["joint_names"],
        self_collision=cfg["self_collision"],
    )


def validate(robot_path="configs/robot/custom.yaml"):
    import mujoco
    from .diagnostics import physics_checks, render_model

    xml, report = generate(
        robot_path, ROOT / "assets/robot/models/microduck.xml", ground=True, motors=True
    )
    model = mujoco.MjModel.from_xml_string(xml)
    report.update(physics_checks(model, report["robot"]))
    (ROOT / "docs/model_audit.json").write_text(json.dumps(report, indent=2))
    render_model(model, report["robot"], ROOT / "results/figures/imported_robot.png")
    return report
