"""Build a six-joint simulation assembly from explicit CAD landmarks and meshes."""

import json
import math
from pathlib import Path
import xml.etree.ElementTree as E
import yaml

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "assets/robot/meshes/futuristic/manifest.json").read_text())
robot = E.Element("robot", name="microduck_futuristic")
hip = math.atan2(17, 37)
shin = -math.atan2(21, 44)
length1, length2 = math.hypot(0.017, 0.037), math.hypot(0.021, 0.044)


def numbers(values):
    return " ".join(str(x) for x in values)


def link(name, mass, size, com=(0, 0, 0), collision=None):
    body = E.SubElement(robot, "link", name=name)
    inert = E.SubElement(body, "inertial")
    E.SubElement(inert, "origin", xyz=numbers(com))
    E.SubElement(inert, "mass", value=str(mass))
    x, y, z = size
    E.SubElement(
        inert,
        "inertia",
        ixx=str(mass * (y * y + z * z) / 12),
        iyy=str(mass * (x * x + z * z) / 12),
        izz=str(mass * (x * x + y * y) / 12),
        ixy="0",
        ixz="0",
        iyz="0",
    )
    if collision:
        pos, dimensions = collision
        geom = E.SubElement(body, "collision")
        E.SubElement(geom, "origin", xyz=numbers(pos))
        E.SubElement(E.SubElement(geom, "geometry"), "box", size=numbers(dimensions))
    for part in manifest["parts"]:
        if part["link"] != name:
            continue
        visual = E.SubElement(body, "visual")
        E.SubElement(E.SubElement(visual, "geometry"), "mesh", filename=part["file"])
        E.SubElement(
            E.SubElement(visual, "material", name=part["name"]), "color", rgba=numbers(part["rgba"])
        )


def joint(name, parent, child, pos=(0, 0, 0), angle=0, axis=None, limits=None):
    j = E.SubElement(robot, "joint", name=name, type="revolute" if axis else "fixed")
    E.SubElement(j, "parent", link=parent)
    E.SubElement(j, "child", link=child)
    E.SubElement(j, "origin", xyz=numbers(pos), rpy=f"0 {angle} 0")
    if axis:
        E.SubElement(j, "axis", xyz=axis)
        E.SubElement(
            j,
            "limit",
            lower=str(limits[0]),
            upper=str(limits[1]),
            effort=str(limits[2]),
            velocity="6",
        )
        E.SubElement(j, "dynamics", damping="0.015", friction="0.002")


link("torso", 0.12, (0.05, 0.07, 0.04), collision=((0, 0, 0), (0.05, 0.07, 0.03)))
link(
    "head",
    0.12,
    (0.065, 0.088, 0.055),
    com=(0.002, 0, 0.004),
    collision=((0.004, 0, 0.005), (0.060, 0.086, 0.052)),
)
joint("head_joint", "torso", "head", (0, 0, 0.074))
for side, y in [("l", 0.034), ("r", -0.034)]:
    link(side + "_hip_roll_link", 0.035, (0.024, 0.022, 0.016))
    joint(
        side + "_hip_roll",
        "torso",
        side + "_hip_roll_link",
        (0.001, y, -0.014),
        axis="1 0 0",
        limits=(-0.52, 0.52, 1.2),
    )
    link(side + "_hip_pitch_link", 0.035, (0.021, 0.023, 0.020))
    joint(
        side + "_hip_pitch",
        side + "_hip_roll_link",
        side + "_hip_pitch_link",
        axis="0 1 0",
        limits=(-1.05, 1.05, 1.2),
    )
    link(
        side + "_thigh",
        0.020,
        (0.020, 0.020, length1),
        com=(0, 0, -length1 / 2),
        collision=((0, 0, -length1 / 2), (0.020, 0.022, length1)),
    )
    joint(side + "_thigh_joint", side + "_hip_pitch_link", side + "_thigh")
    link(
        side + "_knee_link",
        0.025,
        (0.018, 0.022, length2),
        com=(0, 0, -length2 / 2),
        collision=((0, 0, -length2 / 2), (0.014, 0.022, length2)),
    )
    joint(
        side + "_knee",
        side + "_thigh",
        side + "_knee_link",
        (0, 0, -length1),
        axis="0 1 0",
        limits=(-1.57, 0, 0.8),
    )
    link(side + "_shin", 0.005, (0.014, 0.014, 0.02))
    joint(side + "_shin_joint", side + "_knee_link", side + "_shin")
    link(
        side + "_foot",
        0.010,
        (0.052, 0.034, 0.012),
        com=(0, 0, -0.015),
        collision=((0, 0, -0.015), (0.052, 0.034, 0.012)),
    )
    joint(side + "_foot_joint", side + "_shin", side + "_foot", (0, 0, -length2), angle=-shin)
E.indent(robot)
(ROOT / "assets/robot/source/microduck_futuristic.urdf").write_text(
    E.tostring(robot, encoding="unicode") + "\n"
)
spec = yaml.safe_load((ROOT / "configs/robot/custom.yaml").read_text())
spec.update(
    name="microduck_futuristic_v1",
    urdf="assets/robot/source/microduck_futuristic.urdf",
    nominal=[0, hip, shin - hip, 0, hip, shin - hip],
    base_height=0.1165,
    action_scale=[0.30, 0.65, 0.65, 0.30, 0.65, 0.65],
    kp=[2, 2, 1.5, 2, 2, 1.5],
    kd=[0.06, 0.06, 0.04, 0.06, 0.06, 0.04],
    fall_height=0.065,
    standing_gate_min_height=0.095,
    foot_site_pos="0 0 -.021",
    model_path="assets/robot/models/futuristic.xml",
    audit_path="docs/futuristic_model_audit.json",
    figure_path="results/figures/futuristic_robot.png",
)
spec["provenance"].update(
    geometry="Actual futuristic FCStd surfaces; explicit CAD hip/knee/ankle landmarks; simplified collision proxies",
    mass_inertia="Estimated 0.500 kg assembly; box inertias, not material-derived CAD masses",
    nominal_pose="CAD bent-leg stance, feet level; six assumed actuated axes; fixed ankles and head",
)
(ROOT / "configs/robot/futuristic.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
stable = dict(spec)
stable.update(
    name="microduck_futuristic_stable_v1",
    armature=0.0001,
    model_path="assets/robot/models/futuristic_stable.xml",
    audit_path="docs/futuristic_stable_model_audit.json",
)
stable["provenance"] = dict(
    spec["provenance"],
    armature="1e-4 kg m^2 assumed reflected rotor inertia per joint; stabilizes explicit 500 Hz PD; uncalibrated",
)
(ROOT / "configs/robot/futuristic_stable.yaml").write_text(yaml.safe_dump(stable, sort_keys=False))
task = yaml.safe_load((ROOT / "configs/task/flat.yaml").read_text())
task.update(
    final_forward_range=[-0.06, 0.06],
    final_yaw_range=[-0.3, 0.3],
    tracking_bandwidth_m_s=0.06,
    yaw_bandwidth_rad_s=0.3,
    upright_bandwidth=0.35,
    height_bandwidth_m=0.025,
)
task["rewards"]["yaw"]["weight"] = 1.5
task["rewards"]["termination"] = dict(
    weight=-100.0, equation="true_fall", units="indicator; -2 reward per fall at dt=.02"
)
(ROOT / "configs/task/futuristic.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
run = dict(
    robot_path="configs/robot/futuristic_stable.yaml",
    task_path="configs/task/futuristic.yaml",
    profile="futuristic",
    init_std=0.15,
    num_envs=256,
    horizon=24,
    iterations=5000,
    max_wall_time=14400,
    max_transitions=30720000,
    checkpoint_interval=250,
    episode_seconds=10,
    curriculum=False,
    randomization=False,
    seed=42,
)
(ROOT / "configs/experiments/futuristic.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
for split, seeds in [("validation", [1101, 1102, 1103]), ("test", [2101, 2102, 2103])]:
    suite = dict(
        split=split,
        seeds=seeds,
        commands=[
            [0, 0, 0],
            [0.05, 0, 0],
            [-0.05, 0, 0],
            [0, 0, 0.25],
            [0, 0, -0.25],
            [0.05, 0, 0.25],
            [0.05, 0, -0.25],
        ],
        episode_seconds=10,
        friction=[0.8],
        success_survival_fraction=0.95,
        success_velocity_rmse=0.025,
        success_yaw_rmse=0.1,
    )
    (ROOT / f"configs/evaluation/futuristic_{split}.yaml").write_text(
        yaml.safe_dump(suite, sort_keys=False)
    )
print("Built futuristic assembly, task, training config and frozen evaluation suites")

# Separate, reproducible revisions; never overwrite the original factorial task.
gait_task = dict(task, gait_frequency_hz=1.5, foot_clearance_m=0.008, tracking_bandwidth_m_s=0.03)
gait_task["rewards"] = dict(task["rewards"])
gait_task["rewards"]["clearance"] = dict(
    weight=1.5, equation="mean(exp(-((foot_z-phase_target)/.004)^2))", units="dimensionless"
)
gait_task["rewards"]["contact_phase"] = dict(
    weight=0.5, equation="fraction of feet matching desired phase contacts", units="dimensionless"
)
(ROOT / "configs/task/futuristic_gait.yaml").write_text(yaml.safe_dump(gait_task, sort_keys=False))
gait_run = dict(
    run,
    task_path="configs/task/futuristic_gait.yaml",
    profile="futuristic_gait",
    iterations=10000,
    max_transitions=61440000,
)
(ROOT / "configs/experiments/futuristic_gait.yaml").write_text(
    yaml.safe_dump(gait_run, sort_keys=False)
)
residual_spec = dict(
    stable,
    name="microduck_futuristic_residual_v1",
    action_scale=[0.2, 0.5, 0.5, 0.2, 0.5, 0.5],
    model_path="assets/robot/models/futuristic_residual.xml",
    audit_path="docs/futuristic_residual_model_audit.json",
)
(ROOT / "configs/robot/futuristic_residual.yaml").write_text(
    yaml.safe_dump(residual_spec, sort_keys=False)
)
residual_task = dict(gait_task, reference_gait=True, gait_frequency_hz=2.0, foot_clearance_m=0.012)
(ROOT / "configs/task/futuristic_residual.yaml").write_text(
    yaml.safe_dump(residual_task, sort_keys=False)
)
residual_run = dict(
    gait_run,
    profile="futuristic_residual",
    robot_path="configs/robot/futuristic_residual.yaml",
    task_path="configs/task/futuristic_residual.yaml",
    num_envs=1024,
    iterations=5000,
    max_transitions=122880000,
    init_std=0.1,
)
(ROOT / "configs/experiments/futuristic_residual.yaml").write_text(
    yaml.safe_dump(residual_run, sort_keys=False)
)

progress_task = dict(residual_task)
progress_task["rewards"] = dict(residual_task["rewards"])
progress_task["rewards"]["linear_progress"] = dict(
    weight=1.0, equation="clip(dot(v_xy,cmd_xy)/.05^2,-2,2)", units="dimensionless"
)
progress_task["rewards"]["yaw_progress"] = dict(
    weight=0.5, equation="clip(yaw_rate*cmd_yaw/.25^2,-2,2)", units="dimensionless"
)
(ROOT / "configs/task/futuristic_progress.yaml").write_text(
    yaml.safe_dump(progress_task, sort_keys=False)
)
progress_run = dict(
    residual_run, profile="futuristic_progress", task_path="configs/task/futuristic_progress.yaml"
)
(ROOT / "configs/experiments/futuristic_progress.yaml").write_text(
    yaml.safe_dump(progress_run, sort_keys=False)
)

walk_task = dict(progress_task, left_swing_phase_sign=-1)
walk_task["rewards"] = {name: dict(term) for name, term in progress_task["rewards"].items()}
walk_task["rewards"]["clearance"]["weight"] = 3.0
walk_task["rewards"]["contact_phase"]["weight"] = 1.0
walk_task["rewards"]["slip"]["weight"] = -30.0
(ROOT / "configs/task/futuristic_walk.yaml").write_text(yaml.safe_dump(walk_task, sort_keys=False))
walk_run = dict(
    progress_run, profile="futuristic_walk", task_path="configs/task/futuristic_walk.yaml"
)
(ROOT / "configs/experiments/futuristic_walk.yaml").write_text(
    yaml.safe_dump(walk_run, sort_keys=False)
)

for split in ("validation", "test"):
    path = ROOT / f"configs/evaluation/futuristic_{split}.yaml"
    walking_suite = dict(yaml.safe_load(path.read_text()), selection_metric="walking_success_rate")
    (ROOT / f"configs/evaluation/futuristic_walking_{split}.yaml").write_text(
        yaml.safe_dump(walking_suite, sort_keys=False)
    )

long_run = dict(walk_run, iterations=20000, max_transitions=491520000, max_wall_time=14400.0)
(ROOT / "configs/experiments/futuristic_long.yaml").write_text(
    yaml.safe_dump(long_run, sort_keys=False)
)
