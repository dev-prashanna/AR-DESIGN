import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from biped_ppo.robot import generate
from biped_ppo.diagnostics import physics_checks
from biped_ppo.config import read


def test_import_preserves_mass_tree_and_dofs():
    xml, audit = generate(ground=True, motors=True)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nu == 6 and model.nv == 12 and model.nq == 13
    assert np.isclose(model.body_mass.sum(), audit["mass_kg"])
    assert len(audit["links"]) == 14
    assert sum(j["type"] == "revolute" for j in audit["joints"]) == 6
    assert all(link_entry["valid"] for link_entry in audit["links"])
    assert [model.actuator(i).trnid[0] for i in range(6)] == [
        model.joint(n).id for n in read("configs/robot/custom.yaml")["joint_names"]
    ]
    assert ET.fromstring(xml).find("worldbody/body/freejoint") is not None


def test_standing_and_actuator_direction():
    xml, _ = generate(ground=True, motors=True)
    result = physics_checks(mujoco.MjModel.from_xml_string(xml), read("configs/robot/custom.yaml"))
    assert result["standing"]["finite"]
    assert all(j["passed"] for j in result["joint_command_checks"])
    assert result["standing"]["passed"]
