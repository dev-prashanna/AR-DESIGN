# Decisions and explicit limits

- One training backend: mjlab 1.6.0 + MuJoCo Warp + RSL-RL 5.4.2. CPU MuJoCo is
  used for physical diagnostics of the same generated model, not a second PPO stack.
- Isolated .venv, exact direct pins and uv.lock. Existing venv, CUDA, ROS untouched.
- Existing URDF is preserved; inertial and servo values are unverified estimates.
- Robot collision geoms have priority 1 so their friction overrides the ground's
  default (otherwise MuJoCo max-combination could mask low-friction randomization).
- Self-collision disabled explicitly in the first baseline. This limits claims.
- Actor/critic share 27 named proprioceptive fields, no simulator base velocity.
- Gaussian latent u is stored and scored by PPO; tanh(u) is applied only inside
  the deterministic action transform. Joint targets clipped to mechanical limits.
- No guessed torque-speed envelope. Ideal PD plus effort clamp is provisional.
- mjlab auto_reset=False allows exact post-action terminal observations. The
  project wrapper adds gamma*V(terminal) only for pure truncations, resets done
  worlds, and omits upstream time_outs to avoid its V(s_t) bootstrap.
- Checkpoints restore learning state and RNG, but reset simulation state. This
  continuation is not bitwise reproduction; GPU kernels can be nondeterministic.
- The repository is local until a destination and asset publication permissions
  are resolved. No fabricated history, external logging or automatic upload.
