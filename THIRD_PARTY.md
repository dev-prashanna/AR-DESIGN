# Ownership and dependencies

The supplied URDF is copied from the user's workspace. Its provenance and
redistribution permission are not established. Do not publish these assets or
CAD by applying the software license to them. The original CAD and draft PPO
script remain outside this repository, unchanged.

mjlab 1.6.0: Apache-2.0; selected inherited utilities BSD-3-Clause.
RSL-RL 5.4.2: BSD-3-Clause. MuJoCo and MuJoCo Warp: Apache-2.0.
PyTorch: BSD-style; NVIDIA redistributable runtime wheels have their own terms.
Warp: Apache-2.0 with dependency notices. These dependencies retain upstream
licenses; installed distribution metadata and the generated inventory record
exact installed versions. No dependency code is relicensed by this project.

Verified sources: https://github.com/mujocolab/mjlab/blob/main/LICENSE,
https://github.com/leggedrobotics/rsl_rl/blob/main/LICENSE,
https://github.com/google-deepmind/mujoco_warp/blob/main/LICENSE.

The downloaded PyPI wheel and its SHA256 in uv.lock identify the installed
release. A separately inspected upstream main revision is NOT asserted to be
the source revision of that wheel; unavailable release-to-commit provenance
must remain explicitly unknown.
