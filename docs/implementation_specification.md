# Coding-agent prompt: custom bipedal robot, parallel PPO, reproducible experiments, and technical paper

Copy the instructions below into your coding agent. Supply your robot design files when available. This is an implementation specification; it does not claim that training has already been performed.

---

You are my robotics software engineer and research collaborator. Build a complete, executable project for learning locomotion on my custom bipedal robot using Proximal Policy Optimization (PPO). I already have the robot's 3D design. Your work must cover model preparation, physics validation, parallel simulation, training, evaluation, reproducible logging, policy export, a technical paper, and an organized GitHub repository.

Implement and verify the project. Do not stop after writing a plan or generating a skeleton. Work through the milestones below and report concrete blockers precisely. Never claim a run, result, paper experiment, or GitHub push happened unless it actually happened.

## 1. Project context and engineering decisions

Target development hardware: Ubuntu 24.04, Ryzen 7 8845HS, 16 GB system RAM, NVIDIA RTX 4060 laptop GPU with 8 GB VRAM. Inspect the actual machine first; this hardware description may differ from the execution environment. Do not overwrite my existing CUDA, PyTorch, ROS, or Python installations. Create an isolated environment with pinned dependencies.

Default stack: MuJoCo, MuJoCo Warp, mjlab, RSL-RL, PyTorch, typed Python configuration, TensorBoard, JSONL/CSV/Parquet logging, Matplotlib, pytest, and LaTeX. Verify current official documentation, compatible releases, platform requirements, API signatures, and dependency licenses before installing. Pin a tested release or exact commit, not a moving main branch. Record the installed package versions and upstream commit SHAs.

Use mjlab's maintained RSL-RL integration and vectorized physics rather than unnecessarily writing an entire GPU simulator or PPO implementation. Add project-specific task configuration, diagnostics, training orchestration, evaluation, and reporting around it. Prefer one working backend. If this stack cannot support an essential robot feature, document the specific incompatibility and choose a compatible alternative such as Isaac Lab with RSL-RL. Do not silently substitute one physics engine while reporting another, and do not build two full training stacks without a demonstrated need.

Start with proprioceptive observations and simple flat terrain; camera rendering is unnecessary for the first controller. Headless simulation is the default. Benchmark memory and throughput before selecting environment count; do not assume thousands of robots will fit on an 8 GB GPU.

The parallel-training design must be explicit: N independent copies of the same robot collect experience for one shared policy and value function. Each copy has its own state, commands, reset conditions, and randomization. Aggregate their rollouts into PPO minibatches. Independent research seeds are separate training runs with separate weights; parallel environments are not independent trained models. Run seeds sequentially by default on this laptop.

## 2. Inspect inputs and validate the robot model

Inspect the repository and supplied assets before changing anything. Read applicable AGENTS.md instructions. Preserve the original CAD and existing work. Support URDF or MJCF input directly when practical; convert STEP/STL/OBJ assemblies through documented tools if necessary. A visual mesh does not specify an articulated physical robot. Do not invent a working mechanism from disconnected shapes.

Create a model audit covering:

- Link names, parent-child tree, root/free joint, movable and fixed joints, axes, origins, limits, and joint ordering.
- SI units, coordinate frames, scale, total height, total mass, center of mass, inertia tensors and their frames.
- Physical validity of inertia tensors, including positive definiteness and principal-moment triangle inequalities where applicable.
- Visual versus collision geometry, simplified foot and body colliders, collision exclusions, and initial interpenetration.
- Left/right mapping, actuated degrees of freedom, gearing, motor torque and speed limits, damping, friction, and nominal pose.
- Foot contact names, support geometry, mechanical range, and whether the mechanism can plausibly balance and step.

Put all hardware-dependent values in one versioned robot specification. Distinguish measured values, CAD-derived values, manufacturer values, and provisional estimates. Use CAD mass properties when density/material is known; otherwise expose assumptions. If essential joint structure or actuator information is missing, ask one concise bundled question. Continue implementing input-independent infrastructure while waiting. A reference robot may validate the pipeline, but its results must remain separate from results on my custom design.

Create a viewer and diagnostic scripts for the robot. Verify gravity drop, nominal standing geometry, joint-axis direction, positive/negative actuator commands, foot contact, mass distribution, and actuator saturation. Inspect behavior visually and numerically. Produce a model-validation report and render of the actual imported robot.

Before an expensive run, pass these gates: load succeeds; physics state remains finite; resets are valid; joint commands affect the intended joints; contacts are plausible; a conservative standing controller is tested; and a short batch of parallel environments runs without invalid states. Standing failure is a diagnostic result, not permission to hide instability by fixing the pelvis to the world during locomotion training.

## 3. Environment, observations, and control

Implement a Gymnasium-compatible task where practical and the adapter required by the pinned RSL-RL version. Clearly distinguish task termination from time-limit truncation and respect simulator auto-reset semantics. Use terminal observations for appropriate timeout bootstrapping; never bootstrap a true fall as if it were a time limit.

Use named, configurable observation fields with documented shapes, units, frames, clipping, and scaling:

- Body-frame base angular velocity and projected gravity.
- Joint positions relative to the nominal pose and joint velocities.
- Previous policy action.
- Desired forward velocity, lateral velocity, and yaw rate.
- Optional short observation history if needed for latency or partial observability.

Do not expose simulator-only ground-truth quantities to a deployment actor without documenting the required real-world estimator. Base linear velocity and terrain/contact information may be critic-only privileged observations in a clearly labeled asymmetric actor-critic extension. Keep the initial baseline actor and critic observation choices simple and explicit. A gait-phase signal is optional and must be documented; do not hide hand-engineered phase control inside claims of unassisted emergence.

Use bounded joint-position targets around a safe nominal pose as the first action representation. Implement a documented PD actuator model:

q_target = q_nominal + action_scale * action

tau = clip(Kp * (q_target - q) - Kd * q_dot, -tau_limit, tau_limit)

Use per-joint action scales, gains, target limits, and motor limits. Decide whether clipping, squashing, or another bounded-action convention is used, and keep PPO log-probabilities mathematically consistent with that convention. Model servo limits and delay if the actual robot uses position servos. Add an optional torque-speed envelope where hardware data support it. Do not imply an ideal torque actuator matches a cheap hobby servo.

Start by testing a physics timestep of 0.002–0.005 seconds and a policy timestep of 0.02 seconds, using an integer control decimation. These are candidate settings, not assumed validated values. Check stability, contact behavior, joint stiffness, and throughput before freezing them. Apply PD control at the appropriate physics rate. Express rewards, delays, and episode lengths in physical time.

Implement vectorized reset, observation generation, reward calculation, termination, and randomization. Minimize per-environment Python loops and host-device synchronization. Ensure one environment's contacts and reset state cannot corrupt another.

## 4. Task, reward, and curriculum

First achieve standing and low-speed forward motion on flat ground, then add turning, stopping, and modest terrain variation. Start with conservative, mechanically feasible command ranges. Specify episode duration, command resampling interval, reset distribution, warm-up behavior, fall definition, invalid-state handling, and joint-limit violations.

Implement a reward registry; each term has an equation, unit convention, weight, timestep scaling, and separate log. Candidate terms:

- Exponential tracking rewards for commanded planar velocity and yaw rate, with explicit error bandwidths.
- Upright orientation and configurable base-height tracking.
- Penalties for excessive vertical motion, roll/pitch rate, joint limit violation, actuator saturation, and unintended body contacts.
- Joint-speed, torque-squared, and action-rate penalties.
- Foot slip penalties conditioned on contact and appropriate support/contact behavior.
- Optional foot clearance and swing-duration shaping if needed, introduced only after diagnosing a specific failure.

Avoid rewarding standing when motion is commanded, hopping in place, dragging the feet, exploiting collision geometry, or sacrificing command tracking to collect survival reward. Demonstrate that action smoothing does not erase essential fast balance corrections. Use separate fall termination and logging; a non-finite physics state is an error to diagnose.

Implement a curriculum with observable promotion/demotion criteria. Progress from flat-ground balance to slow walking, command changes, and robustness. Log every curriculum change and its trigger. Keep evaluation difficulty fixed so changing training difficulty does not create misleading performance curves.

Add staged, configurable domain randomization: friction, masses/inertias using physically valid perturbations, center-of-mass offsets, actuator strength, gains, sensor noise/bias, control delay, initial pose, and external disturbances. Initial bounds are provisional and documented; replace them with measured uncertainty when possible. Never use physically impossible independent parameter draws. Record distributions, units, cadence, and held-out ranges.

## 5. PPO configuration and implementation correctness

Use a maintained PPO implementation. Explain and record its clipped surrogate objective, GAE, value loss, entropy term, minibatching, learning-rate schedule, advantage normalization, observation normalization, gradient clipping, and timeout handling. Check the installed implementation rather than assuming a paper and library share every default.

Initial tuning configuration:

| Parameter | Candidate starting value |
|---|---|
| Actor / critic | Separate MLPs, hidden layers [256, 256, 128], ELU |
| Policy distribution | Diagonal Gaussian with documented bounded-action handling |
| Initial action std | 0.5, configurable with units/convention documented |
| Discount gamma | 0.99 |
| GAE lambda | 0.95 |
| PPO clipping epsilon | 0.2 |
| Learning rate | 3e-4 |
| PPO epochs per rollout | 5 |
| Minibatches per epoch | 4, subject to valid batch divisibility |
| Value loss coefficient | 1.0 |
| Entropy coefficient | 0.01, tune if supported by evidence |
| Maximum gradient norm | 1.0 |
| Rollout horizon per environment | 24 policy steps initially |
| Environment counts to benchmark | 32, 64, 128, 256; larger only if feasible |
| Planned independent seeds | 0, 1, 2; five if compute permits |

These are starting hypotheses, not proven optimal hyperparameters. Compute transitions per update as num_envs * rollout_horizon. Distinguish physics substeps, policy steps, aggregate environment transitions, episodes, PPO updates, gradient steps, and epochs. Never label every quantity “epochs.”

Add meaningful tests for reward signs, joint/action ordering, finite observations, reset isolation, termination versus truncation, GAE on a small hand-computable example or adapter-level bootstrap behavior, normalization save/load, checkpoint resume, and exported-policy parity. Test interfaces and failure-prone mathematical semantics instead of duplicating upstream library tests.

## 6. Memory, performance, and run orchestration

Provide a hardware diagnostic and benchmark command. Measure peak VRAM, RAM, aggregate transitions/second, physics steps/second with its definition, rollout time, optimizer time, initialization/compilation time, and wall-clock time. Use a warm-up phase and GPU synchronization at timing boundaries. Report batch-size and precision settings. Do not present simulator-only throughput as end-to-end training throughput.

Start small and increase environment count until measured throughput stops improving or memory headroom is inadequate. Account for collision complexity, contacts, simulator buffers, rollout buffers, and network memory. Recover cleanly from OOM by checkpointing where possible and reporting the failure. A change in environment count changes PPO batch size; log it as a new configuration instead of silently resuming a supposedly identical experiment.

Use explicit resource profiles: smoke test, pilot, and research. Default to smoke tests and a bounded pilot, for example at most 20 minutes, before estimating research-run cost. Add max-wall-time and max-transitions limits, clean interruption handling, and configurable checkpoint cadence. Full training and experiment matrices must be resumable commands, not an unbounded process hidden inside setup. Use only local resources unless I separately authorize paid compute. If research runs exceed available execution time, save the working pipeline and list exactly which runs remain; never fill gaps with synthetic results.

## 7. Experiment tracking and artifacts

Each run gets a unique actual timestamp/UUID, immutable run manifest, resolved configuration, seed, code SHA, working-tree status/diff, robot-asset checksums, dependency lockfile, hardware/driver versions, and command line. Use JSONL/Parquet or CSV as the portable source of truth and TensorBoard as a viewer. External experiment services are optional; no account should be required.

Save per-update metrics: transition count, wall time, reward components, episode duration, falls, command tracking, actor/critic losses, entropy, approximate KL, clip fraction, explained variance where defined, gradient norms, action std, saturation, curriculum level, throughput, and memory. Aggregate completed episodes correctly across vector environments. Save detailed trajectories for a small selected evaluation subset; do not record every joint of every environment at every physics step by default.

Save complete resumable checkpoints containing actor, critic, optimizer, learning-rate scheduler, normalization statistics, RNG states, configuration, curriculum state, and iteration/transition counts. Distinguish training-state resume from exact simulator-state resume. Record when physics state cannot be restored bit-for-bit. Test interrupted continuation and disclose GPU nondeterminism.

Retain periodic checkpoints, last checkpoint, best validation checkpoint, final checkpoint, and optionally EMA weights as a separate labeled variant. Serialize architecture and parameter count. Export an inference policy with normalization, joint names/order, action scaling, nominal pose, frame conventions, control frequency, and history-buffer requirements. Validate ONNX or another supported export against PyTorch within a declared tolerance and save the comparison.

## 8. Evaluation and a defensible research question

The honest initial research question is: “For this custom biped and a fixed local compute budget, how do curriculum learning and physically grounded domain randomization affect locomotion reliability and robustness?” Treat this as an empirical study. Standard PPO walking on a new CAD model is not automatically an algorithmic novelty or publication-worthy research.

Implement these experiments as a 2x2 design:

1. PPO without curriculum or domain randomization.
2. PPO with curriculum only.
3. PPO with domain randomization only.
4. PPO with both.

Keep architecture, reward definition, simulator, training-transition budget, seed set, and evaluation protocol matched except for intended changes. Define the fixed command distribution used without curriculum and explain the comparison. Report any remaining confound. Separate performance benchmarking over environment counts from the locomotion ablation. Report both samples and wall time; distinguish equal-sample comparisons from equal-time comparisons.

Use at least three actual independent training seeds per condition for an initial study; increase if feasible. Start with one pilot per condition before launching the full matrix. The number of evaluation episodes does not replace independent training seeds. If compute supports fewer runs, mark the work exploratory and show the actual count.

Freeze validation/test splits before selecting a checkpoint. Use validation for tuning and checkpoint selection; evaluate the selected checkpoint on untouched test conditions without retuning. Use identical held-out command sequences, terrain cases, perturbation schedules, and episode seeds across methods where possible. Specify success and failure criteria before inspecting final results.

Evaluate motion tracking, stopping and turning, standing at zero command, survival duration, fall rate, distance traveled, foot slip, actuator saturation, joint limit violations, and robustness across held-out friction/mass/delay settings. Use forward distance or path length appropriate to the command; do not divide energy by near-zero distance. Report mechanical cost of transport using a declared convention such as integral(sum(abs(tau*q_dot))) / (mass*g*distance), units, and exclusions. This is mechanical work, not measured battery energy.

For pushes, report force, duration, impulse, direction, application point, and normalized scale where useful. Compare recovery over a predefined disturbance grid rather than claiming a precise maximum from one successful trial. For terrain and delay, remain within mechanically plausible limits and report failure boundaries honestly.

Save episode-level results linked to condition, training seed, checkpoint hash, and evaluation seed. Aggregate at the independent-run level, show individual seeds and mean/spread, and state exactly how uncertainty intervals were computed. Do not treat thousands of episodes from one policy as thousands of independent learning experiments. Show raw and smoothed learning curves with the smoothing method and x-axis units disclosed. Report failed runs, evaluation exclusions, and all seeds.

## 9. Automated technical paper

Create an editable LaTeX paper, bibliography, compiled PDF, supplementary configuration tables, and figure-generation code. Target an approximately 6–8-page technical report initially, extending appendices as needed. Use plain, precise scientific English with defined symbols and explicit units. Choose a modest title that describes the actual completed experiments.

Include: abstract; introduction and research question; related work; robot morphology and physical modeling; task and MDP; observations/actions/PD control; PPO equations and implementation; reward and randomization tables; training/compute protocol; results; ablations; failure cases; limitations; conclusion; reproducibility statement; references.

Generate every numerical result, main table, and plot directly from versioned evaluation/training artifacts. Create a result manifest mapping paper claims to run IDs, checkpoint hashes, configuration hashes, and analysis code. Store raw aggregates in machine-readable files. Make paper build fail or visibly mark “not measured” when required real data are absent. Never generate plausible-looking learning curves, invented uncertainties, fabricated citations, fake hardware tests, or unrun comparisons.

Include robot/model renders, architecture and control-loop diagrams, reward/observation tables, learning curves, tracking plots, robustness plots, and representative success/failure snapshots when real data exist. Select videos under a documented rule and include failures, not only the best rollout. A paper with incomplete experiments must be labeled a preliminary methods report; do not write completed-results prose around planned work.

Retrieve and verify original references. Start with Schulman et al., “Proximal Policy Optimization Algorithms” (2017), GAE, relevant parallel legged-locomotion and sim-to-real research, the actual simulator/framework publications, and any method implemented. Verify authors, year, title, and DOI/arXiv identifier. Cite dependencies and distinguish their contribution from my custom task, model, experiment design, and findings. Include an AI-assistance disclosure if required by the eventual venue. Do not guarantee publication.

## 10. Repository structure and usable commands

Use a clear layout, adapting names when justified:

```text
biped-ppo/
  README.md
  pyproject.toml
  <dependency lockfile>
  LICENSE
  CITATION.cff
  assets/robot/{source,meshes,collision,models}/
  configs/{robot,task,ppo,experiments}/
  src/biped_ppo/{envs,robots,control,training,evaluation,tracking}/
  scripts/
  tests/
  docs/{model_audit,design_decisions,experiment_protocol}/
  experiments/manifests/
  results/{aggregates,figures}/
  paper/{main.tex,references.bib,sections}/
  .github/workflows/
```

Keep bulky training runs in a configurable ignored output directory; preserve their manifests and compact aggregates in the repository. Use GitHub Releases or appropriately configured Git LFS for selected large models and videos, with checksums and retrieval instructions. Check ownership and licenses before distributing CAD or upstream assets. Do not apply a blanket project license to third-party assets.

Implement and document stable project commands covering:

```text
biped doctor
biped validate-model --robot configs/robot/custom.yaml
biped view --robot configs/robot/custom.yaml
biped benchmark --num-envs 32 64 128 256
biped train --config configs/experiments/pilot.yaml --seed 0 --max-wall-time 1200
biped train --config configs/experiments/full.yaml --seed 0
biped train --resume <checkpoint> --max-wall-time <seconds>
biped evaluate --checkpoint <checkpoint> --suite <suite.yaml>
biped sweep --config <ablation-matrix.yaml> --max-wall-time <seconds>
biped export --checkpoint <checkpoint> --format onnx
biped analyze --runs <run-directory>
biped paper --results <validated-results-directory>
```

These are interfaces you must implement and test, not claims that upstream packages already provide these exact commands. Provide exact setup commands for the pinned stack and a short “first successful run” guide. CPU smoke tests should not require an NVIDIA GPU. GPU integration tests should be clearly tagged and should report skipped status honestly when hardware is absent. Standard CI should run lint, configuration/unit tests, and inexpensive model checks; do not launch long training on every push. Support an explicitly triggered training workflow only when compatible GPU execution is available.

## 11. GitHub and truthful continuous development

Organize the work into genuine milestones with meaningful commits made when work is performed: model audit; simulator task; baseline PPO; logging/resume; evaluation; ablations; analysis; paper; release. Maintain issues, a roadmap, dated experiment notes, CHANGELOG, and version tags tied to tested milestones. Do not generate empty commits to simulate activity or automate meaningless contribution farming.

Use actual timestamps. Do not backdate new work into January–December 2025, fabricate authorship or experiment dates, or imply development occurred before it did. If authentic historical source or an existing Git repository is supplied, preserve its real metadata and clearly document any import. Clearly label retrospective notes as written now about earlier work.

Use the configured authenticated account and user-specified repository. If no repository is specified, prepare the complete local repository and propose a descriptive name; ask for the target or required missing authentication only when publication is ready. Do not expose credentials, rewrite existing public history, force-push, delete remote work, or change repository visibility without explicit authorization. Commit only intended files. Perform a secret/large-file check before pushing. If pushing is unavailable, leave a tested local repository and precise commands, with the limitation clearly stated.

## 12. Completion gates and final handoff

Work in dependency order: inspect and isolate environment; audit/import robot; validate mechanics; run CPU and small GPU smoke tests; benchmark vectorization; train a bounded pilot; verify resume and evaluation; freeze protocol; run feasible experiments; regenerate results and paper; prepare/push the authorized repository.

Do not turn missing custom CAD into a claim that the custom robot is finished. Do not treat scripts that have never executed as verified. Do not treat a short successful rollout as reliable walking. Define acceptance thresholds in configuration and report measured results against them, including failures.

Final handoff must state: what works; exact commands executed; robot model status; tests and any skips; completed run IDs and budgets; measured policy performance; remaining experiments; paper build status; exported policy checks; repository URL and pushed commit if applicable; and known limitations. Include installation and reproduction commands plus links to actual artifacts. Clearly separate simulation-only results from any future hardware deployment. Exporting a policy does not authorize running it on a physical robot; prepare a separate staged hardware validation procedure with joint, torque, temperature, fall-protection, and emergency-stop checks for later review.

Proceed autonomously on reversible implementation decisions. Ask concise questions only for essential information that cannot be inferred safely, such as joint structure, actuator limits, missing design files, or the intended remote repository.

---

## Official starting references for the coding agent

- PPO paper: https://arxiv.org/abs/1707.06347
- MuJoCo Warp documentation: https://mujoco.readthedocs.io/en/latest/mjwarp/
- MuJoCo Warp repository: https://github.com/google-deepmind/mujoco_warp
- mjlab repository: https://github.com/mujocolab/mjlab
- mjlab RSL-RL integration: https://github.com/mujocolab/mjlab/blob/main/docs/source/training/rsl_rl.rst
- RSL-RL repository: https://github.com/leggedrobotics/rsl_rl
- Isaac Lab alternative: https://isaac-sim.github.io/IsaacLab/main/source/concepts/reinforcement_learning.html

The agent must resolve references against the release it actually installs; these moving documentation URLs are starting points, not a dependency lockfile.
