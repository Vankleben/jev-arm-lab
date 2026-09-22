# jev-arm-lab — A task-level decision lab for a robot arm

> **Results first: [docs/FINDINGS.md](docs/FINDINGS.md)** (all numbers, the failure map, limitations, how to reproduce)
> In one sentence: hand "what to do next" to a model that only makes judgments (Jev) and keep the veto in
> code — 20/20 scenes completed, grasp-state judgments correct on all 79 samples (Brier 0.030), 0 dangerous
> false positives; and with the wrong measurement protocol, the same model would be scored at 77%.

![20 scenes](docs/media/reel.gif)

Pick-and-place ("grasp the cube, put it in the target zone") with the xArm7 in MuJoCo, where a
**typed-judgment model** (TypeSafe's Jev, or a local rule-based stand-in) decides **which skill to run at
each step**. The point is to test one hypothesis:

> Give the judgment (should I move, is the grasp secure, how far along is the task) to the model; keep the
> workflow, the geometry and the safety veto in code — does that division of labor actually hold up on a
> robot arm?

The answer is **yes, but the thresholds must be calibrated yourself**. Measured numbers are in
"Measured results" below.

## Quick start

```powershell
cd D:\AI\jev-arm-lab

# 1) Rule-based stand-in, no API key needed — see the whole loop first
.\.venv\Scripts\python.exe -m jev_arm.main --jev-mode fake --cycles 20 --log logs/fake_run.jsonl

# 2) Real Jev decisions (needs the TYPESAFE_API_KEY environment variable)
.\.venv\Scripts\python.exe -m jev_arm.main --jev-mode live

# 3) Open the MuJoCo window and watch in real time (throttled to near real time; it pops up as its own window)
.\.venv\Scripts\python.exe -m jev_arm.main --jev-mode live --viewer

# Default gates are 0.30 / 0.45, calibrated from measurements. To reproduce "gates too high -> frozen robot":
#   ... --gate-confidence 0.55 --gate-grasp 0.70

# 4) Export frames + score
.\.venv\Scripts\python.exe -m jev_arm.main --jev-mode live --frames out --frame-every 1
.\.venv\Scripts\python.exe tools\summarize_log.py logs\jev_run_loose.jsonl
```

The only dependencies are `mujoco` and `numpy` (versions in `requirements.txt`, installed in `.venv`).
Regression tests (no API key needed): `.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`

## Directory

```
jev_arm/
  sim.py        MuJoCo wrapper: IK (pure solver), servoing (ramp interpolation + gravity-sag correction), gripper, contacts/ground truth
  skills.py     9 skills: approach/grasp/lift/carry/lower/release/retreat/hold/finish
  judge.py      The decision layer. QUESTIONS (all three questions live here) + FakeJudge + JevJudge + state construction
  main.py       Main loop: state -> judgment -> code veto -> execute skill -> write JSONL; also viewer + PNG export
models/menagerie_xarm7/
  xarm7_lab.xml   The arm used in this experiment (see "Simplification 1" below)
  lab_pick_place.xml  The scene: table, cube, target zone
  (everything else — xarm7.xml / hand.xml / assets — is original from MuJoCo Menagerie, Apache-2.0)
tools/
  summarize_log.py  Read the JSONL: accuracy, Brier score, confidence reliability table, veto count
  diag_sequence.py  Run a fixed skill sequence without the decision layer (for regression tests)
logs/, out/         Run logs and rendered frames
```

## How the decision layer is asked

Each cycle is **one call with three questions** (`QUESTIONS` in `judge.py` — change a question in that one
place only):

| Question | Type | Purpose |
|---|---|---|
| `intent` | choice (9 options) | which skill to execute next |
| `grasp_secure` | noul | is the object held securely right now (will it drop) |
| `task_progress` | score (4 levels) | how far along the task is |

The state deliberately **contains no simulation ground truth**: only the commanded gripper gap, the measured
gap, one tactile "a pad touched something" bit, the object position / height above the table / distance to
target, and the last three actions. So "is the grasp secure" has to be inferred by the model itself.

Code keeps the veto (`main.py: enforce()`):

```
intent ∈ {lift,carry,lower} and grasp_secure < gate_grasp  → may not carry (falls back to grasp/hold)
intent_confidence < gate_confidence                        → stay put
progress = 3 and intent not in {retreat,finish}            → retreat right away
```

## Measured results (2026-09-21, this machine)

| Run | Result |
|---|---|
| `--jev-mode fake` (20 random scenes, **real contact carrying**) | 20/20 completed, 6 cycles per scene, horizontal placement error **median 2.3 mm / worst 3.1 mm** |
| `live`, gates 0.55 / 0.70 | **15 of 16 proposals vetoed, frozen in place**. The model's proposals were sensible every time (grasp/lift/carry), but intent confidence was only 0.43–0.58 |
| `live`, gates 0.30 / 0.45 | completed in 6 cycles, **every judgment correct**, placement error 1.9 mm |
| Calibration (from the 6-cycle live log) | `grasp_secure` accuracy 5/6, Brier 0.199; `task_progress` mean absolute error 0.57 levels |

The two `live` rows were measured during the kinematic-carry era (earlier on 2026-09-21); after carrying
became real contact physics, the fake baseline was re-measured (first row), and the live numbers wait for
the next batch calibration to be updated.

Two takeaways:

1. **The gates are everything.** Same model, same environment: raising the gate from 0.30 to 0.55 turns
   "task completed" into "does not move a single step". Numbers like these cannot be guessed — they must be
   labeled from logs that carry ground truth (that is exactly what `tools/summarize_log.py` does).
2. **More options naturally lower confidence**: confidence = (n·peak−1)/(n−1), so with 9 options even a peak
   probability of 0.6 yields only 0.55. Giving the model few, well-chosen candidates is design work on par
   with tuning the gates.
   (The samples are small: 6 and 16 runs — enough to indicate an order of magnitude, not enough to call it
   calibration. Doing it properly would take a few hundred runs.)

## Failure map (fault injection, 2026-09-21)

The same task, run under 8 single-point faults with 5–10 random scenes each (70 scenes, ~400 decision
cycles), injected via `--stress <name>` — see `jev_arm/stress.py`. Except for "grasp 22 mm off" (real
physics), all of them **only corrupt what the model is told** (sensor lies / stale data / noise / external
force); the world itself stays real, so the ground truth remains trustworthy.

| Stressor | Complete | Placement mm | Grasp acc. | Brier | Danger FP | Missed FN |
|---|---|---|---|---|---|---|
| none (baseline) | 100% | 1.7 | 100% | 0.030 | 0 | 0 |
| grasp_off (grasp 22 mm off) | 100% | **16.9** | 100% | 0.029 | 0 | 0 |
| tactile_dead (tactile always says "no contact") | 100% | 1.9 | 100% | 0.012 | 0 | 0 |
| tactile_stuck (tactile always says "contact") | 100% | 1.7 | 100% | 0.040 | 0 | 0 |
| occlusion (vision updates every 3 cycles) | 100% | 2.0 | 100% | 0.036 | 0 | 0 |
| camera_frozen (vision frozen solid) | 100% | 1.7 | 100% | 0.037 | 0 | 0 |
| noise (10 mm position jitter) | 100% | 2.0 | 100% | 0.034 | 0 | 0 |
| shove (external push at cycle 1) | 100% | 1.0 | 100% | 0.033 | 0 | 0 |

Danger FP = "claims ≥60% confident the object is held while it is actually not" (would make the code take a
dangerous action). All zero.

Three takeaways:

1. **A single failing sensor, it survives — and beats hand-written rules.** In `tactile_dead` the bump bit
   is False the whole run, yet Jev judges "held" (0.64→0.85) from nothing but "gripper gap 87→59 mm +
   object height above the table + recent actions", and the task completes 100%; the rule-based stand-in in
   the same scene falls into a "re-grasp forever, never lift" loop, 0% completion. **Redundant channels in
   the state are the source of robustness.**
2. **The truly dangerous sensor is not the one that lies — it is the one that freezes.** Under
   `camera_frozen` the actions are still all correct, but `task_progress` tops out at **2.65** and never
   reaches 3 ("placed in the target zone") — because vision is stuck on the first frame, it never learns the
   job is done. This is the failure class hardest to catch on a real robot: no crash, no error, it just never
   finishes. The engineering answer is **monitor data freshness (timestamps/heartbeats) and escalate to a
   human**, not swap in a smarter model.
3. **Reliable judgment ≠ physical precision.** With the grasp 22 mm off, judgment is 100% correct, but
   placement error grows from 1.7 mm to 16.9 mm. Only calibration fixes that; the judgment layer cannot, and
   the two must be measured separately.

Limitations: all single-point faults, the scene is still simple (one object, flat table, nobody around).
The table above was measured during the kinematic-carry era; failures like "slips out of the hand after
grasping" could not even be expressed under that simplification. After carrying was switched to real
contact friction on 2026-09-21 they became expressible (ground truth gained a `slipping` flag — see
"Physics notes" below). Round 2 of the failure map follows.

Re-scoring needs no re-simulation: `tools/stress_map.py` recomputes straight from `logs/stress_*/` (zero API
cost).

### Round 2: failure map under real contact physics (2026-09-21)

**Same seeds 300–304, same gates as the round above** — the only variable is the physics (real contact
carrying + three new stressors). This round ran with `--mode fake` (rule-based stand-in), because its
purpose is to validate the physics, not to evaluate the judgment model:

| Stressor | Complete | Escalated | Drop rate | Placement mm | Cycles |
|---|---|---|---|---|---|
| none | 100% | 0% | 0% | 0.5 | 6.0 |
| grasp_off (22 mm off) | 100% | 0% | **40%** | 4.2 | 7.6 |
| tactile_dead (rule stand-in) | 0% | 100% | 0% | — | 4.0 |
| tactile_stuck | 100% | 0% | 0% | 0.5 | 6.0 |
| occlusion | 0% | 100% | 0% | — | 3.2 |
| camera_frozen | 0% | 100% | 0% | — | 2.0 |
| noise | 100% | 0% | 0% | 0.5 | 6.0 |
| shove | 100% | 0% | 0% | 5.6 | 6.0 |
| **low_friction** (μ=0.05) | 0% | 20% | **100%** | — | 19.0 |
| **marginal_grip** (μ=0.08) | 60% | 0% | **100%** | 49.7 | 14.4 |
| **heavy_object** (1.5 kg) | 0% | 0% | **100%** | — | 20.0 |

Three new takeaways:

1. **Drops are measurable for the first time — and they rewrite an old conclusion.** `grasp_off` used to be
   just "placement off by 16.9 mm"; now 2/5 scenes actually drop the object (cycles 3–4), **and both
   completed after re-grasping** (placement 1.6 / 5.2 mm) — the loop can recover from a drop, but no
   previous metric recorded that it happened.
2. **The success criterion itself can lie.** In `marginal_grip` the cube slides out of the fingers
   mid-carry and happens to fall straight into the target zone, which "object in target + gripper open"
   scores as success (3/5 scenes "completed", median placement 49.7 mm — just inside the 5 cm radius). Only
   the new drop rate (100%) exposed it — the same lesson as FINDINGS §3: **criteria lie just like labels do**.
3. **The watchdog has a blind spot (not yet fixed).** Under `heavy_object` every cycle goes
   grasp→lift→cube slides back to the table→grasp, and none of the 20 cycles escalate, because the trigger
   is "the same skill for a third consecutive time" while here two skills alternate. Candidate fix: also
   trigger on "a period-2 repeating pattern + unchanged progress estimate".

The sensor stressors (both tactile cases, noise, shove, both camera cases) behave exactly as in the round
above — the physics change did not contaminate the sensor-layer conclusions. Judgment metrics
(accuracy/Brier/slip-FP) need a live re-run, see the next section.

### Round 3: real physics + real judgment (live, 2026-09-21)

Same seeds 300–304, same gates (0.30/0.45), 11 stressors × 5 scenes = **55 scenes / 571 decision cycles,
all judged by Jev (zero fallback)**, 428 scoreable grasp-state samples:

| Stressor | Complete | Escalated | Drop rate | Placement mm | Cycles | Acc. | Brier | Progress MAE | Danger FP |
|---|---|---|---|---|---|---|---|---|---|
| none | 100% | 0% | 0% | 1.8 | 6.8 | 100% | 0.085 | 0.29 | 0 |
| grasp_off | 100% | 0% | 40% | 4.3 | 10.4 | 97% | 0.075 | 0.67 | 0 |
| tactile_dead | 100% | 0% | 0% | 0.5 | 6.0 | 100% | 0.073 | 0.33 | 0 |
| tactile_stuck | 60% | 20% | 60% | 11.5 | 13.0 | 93% | 0.067 | 0.85 | 1 |
| occlusion | 0% | 100% | 0% | — | 2.8 | 100% | 0.030 | 0.05 | 0 |
| camera_frozen | 0% | 100% | 0% | — | 2.0 | 100% | 0.003 | 0.06 | 0 |
| noise | 100% | 0% | 0% | 1.8 | 6.6 | 95% | 0.108 | 0.29 | 0 |
| shove | 100% | 0% | 40% | 5.6 | 10.6 | 94% | 0.084 | 0.65 | 1 |
| low_friction | 0% | 0% | 100% | — | 20.0 | 75% | 0.108 | 1.37 | **12** |
| marginal_grip | 20% | 0% | 100% | — | 19.4 | 88% | 0.073 | 1.24 | 2 |
| heavy_object | 0% | 20% | 100% | — | 19.6 | 85% | 0.075 | 1.31 | 2 |

Compared with round 1 (same seeds, same Jev, **the only variable is the physics**): baseline placement 0.8 →
1.8 mm with judgment still 100% correct; `grasp_off` changed from "placement off by 15.4 mm" to "40% drop
rate"; `shove` changed from 1.4 mm to "40% drop rate".

**The most important new finding: the object dropped, and the judgment layer still says it is holding it.**
Under `low_friction`, danger FP = 12 cycles — after the cube slipped and fell back onto the table, Jev
claims 0.60–0.67 confidence and proposes `carry`. What it saw at the time: commanded gap = measured gap =
42.2 mm (the pads fully closed on nothing), tactile False, **object height ≈ 0 (clearly on the table)**,
recent actions grasp/grasp/lift. That is, in the state "fingers closed + just attempted a grasp", it
**ignores the mutually contradictory redundant channels** and confidently carries an empty hand — a failure
impossible during the kinematic-carry era, and exactly the expensive kind on a real robot. (The safety gate
cannot stop it either: 0.62 > the 0.45 gate.)

Two more:

- **Tactile "always says contact" escalates to a dangerous fault under real physics**: 60% completion / 60%
  drops / 1 danger FP. Under the old physics it had "zero false positives" — because the cube could not
  drop; now the judgment layer trusts the tactile bit and still believes it is holding on after the slip.
- **The new slip-FP metric never triggered once (honest record)**: a slip is transient — at the decision
  instant the object is either still in the hand or already lost contact entirely — so what actually catches
  this physical danger is the old danger FP. The metric stays, but its contribution in this round is zero.
- The watchdog blind spot reproduced live: `heavy_object` ran the full 20 cycles without escalating in 4/5
  scenes (grasp↔lift alternating, which never satisfies "same skill three times in a row").


### From finding to fix: the freshness gate + the stuck watchdog

The fix for the "frozen camera" finding above is implemented; both mechanisms live in `main.py`:

1. **Freshness gate**: every reading in the state carries `age_s` (published by the sensor layer in
   `jev_arm/sensors.py` — camera at 4 Hz, tactile at 20 Hz). Each skill declares which channels it depends
   on (`SKILL_NEEDS`), checked before execution: if a required channel is older than `--stale-limit`
   (default 1.0 sim seconds), the arm may not move — **blocked and escalated to a human**.
2. **Stuck watchdog**: the same action requested for a third consecutive time *and* the judgment layer's own
   progress estimate unchanged — meaning it cannot see the world changing — is blocked and escalated the
   same way. Because a channel that *lies* can be caught by cross-validation, while one that *stops
   updating* can only be caught with a clock and repetition.

Before vs. after (same 8 stressors × 5 random scenes):

| Stressor | Before | After |
|---|---|---|
| Camera frozen | "100% complete", but progress estimate capped at 2.65/3 (silent failure) | **0% complete / 100% escalated to a human** |
| Camera one frame per 4 s | 100% complete | **0% complete / 100% escalated to a human** |
| The other 6 (tactile lies / jitter / shove / miscalibration) | 100% complete | 100% complete, **zero false alarms** (including the noise scene) |

`grasp_secure` accuracy and Brier stay unchanged in every row — the gate is a code-level addition and does
not touch the judgment itself.

Re-scoring needs no re-simulation: `tools/stress_map.py` recomputes straight from `logs/stress_*/` (zero API
cost).


## Physics notes (honest disclosure)

1. **Gripper**: Menagerie's original is a linkage gripper, which in simulation can be back-driven by the
   object — squeezing a 6 cm cube produced contact forces of 85–265 N while the actuator only delivers
   14–16 N, so the fingers get pried open and nothing can ever be held. It was therefore replaced with this
   project's own parallel-jaw gripper (`lab_hand` in `xarm7_lab.xml`): two slide joints + position actuators
   with a force limit (±30 N per finger), and the cube's mass explicitly set to 50 g (the default density
   of 1000 kg/m³ would give 216 g).
2. **Carrying is real contact friction** (fixed 2026-09-21). Previously "holding" was kinematic carrying
   (each step snapped the cube back to its grasp-time relative pose), which the README recorded as an
   "cause unknown" piece of technical debt: a 30 N grip + friction 1.2 on a 50 g cube slipped the moment it
   was lifted. Frame-by-frame measurement with `tools/diagnose_slip.py` showed **the cause was not friction**
   — at grip time the normal force was 15 N, friction-cone utilization only 0.38, the payload needed 0.49 N,
   a 30× margin; the real cause was `ik()` writing its solution straight into `d.qpos`, so every Cartesian
   move was a "teleport + servo" and **the pads never physically traveled** — on lift, the hand materialized
   above the cube. Fixing that solver side effect (`ik()` became a pure function) had the baseline parameters
   genuinely lifting the cube 140 mm; then `move_tcp` got ramped interpolation (a step command breaks the
   friction cone instantly, sliding the cube 10 mm between the fingers), and placement error returned to
   millimeter level. Remaining physics limits: quasi-static tabletop, single object.
3. **Ground truth for "grasped securely"**: contact on both pads + relative velocity to detect slipping
   (`grasp_flags().slipping`) — real contact carrying makes "slipped after grasping" class of contact
   failures expressible and scorable for the first time.

## Next steps

- **Slip stressors**: real contact unlocked fault injection for "slips mid-grasp / object too slippery" —
  add it to the failure map.
- **Calibration**: build a batch of fixed scenes (varying positions/masses), run them in bulk, produce
  reliability curves with `summarize_log.py`, then set the gates.
- **Question phrasing**: narrow the 9-option `intent` down to "3 candidates per stage" and see how
  confidence and accuracy move.
- **Real robot**: this decision layer ports directly to small arms like the SO-101 (~$122) — swap the
  Cartesian targets in `skills.py` for real-arm SDK calls; `judge.py` and the gates stay untouched.

## Attribution

Everything under `models/menagerie_xarm7/` except `xarm7_lab.xml` and `lab_pick_place.xml` comes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0, `LICENSE` included).
