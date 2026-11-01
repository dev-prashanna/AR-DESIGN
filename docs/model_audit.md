# Model validation — provisional Microduck

Source: unchanged user URDF, copied under assets/robot/source. Appearance CAD is
not used for dynamics. Every supplied mass/inertia/limit is an unverified estimate.

- Root: torso, connected to world with a free joint (not a fixed pelvis).
- 14 links, 6 actuated revolute joints; remaining joints fixed.
- Total mass: 0.500 kg. Neutral collision height: 0.1935 m.
- Neutral COM in world metres: [0.0006, 0.0, 0.09718000000000002].
- All link inertia matrices: positive definite and triangle inequalities pass.
- Five-second standing: minimum height 0.135764 m,
  finite throughout; saturation fraction 0.0000.
- Both positive/negative actuator acceleration checks pass for all six joints.
- Gravity drop remains finite. Standing contacts: 8.
- Initial collision contact count: 0 (feet start slightly above floor).
- Self-collision is disabled. No clearance or dynamic stepping certification.

SI units; x forward, y left, z up. Hip roll axis x, hip pitch and knee axis y.
Left is positive y. Action order: l_hip_roll, l_hip_pitch, l_knee, r_hip_roll, r_hip_pitch, r_knee.
Foot sites: l_foot_contact and r_foot_contact. Each foot is 38x26x8 mm.
The support hull spans x=+-19 mm and y=+-51 mm in the neutral pose.
No ankle actuator; sagittal foot orientation is coupled to hip/knee motion.
Gearing, servo manufacturer, backlash and calibrated torque-speed curves unknown.

The complete parent/child tree, fixed offsets, joint axes/ranges, inertial origins,
inertia matrices and principal moments are in model_audit.json. Conservative
standing passes only for this nominal model; locomotion remains to be measured.
