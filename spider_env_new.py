"""
SpiderEnv — Gymnasium environment for the SpdrBot 4-arm / 12-servo robot.

URDF: SpdrBot_description copy/urdf/SpdrBot.urdf
All 12 controlled joints are type="continuous" (no hard limits in URDF).
Software limits of ±JOINT_LIMIT rad are enforced by the env.

Elbow/wrist joint axes are diagonal (±0.707, ±0.707, 0) in the XY plane,
arising from the physical servo mounting angle in the CAD model.

  Leg │ Servo    │ URDF joint   │ Axis    │ PyBullet idx │ Motion
  ────┼──────────┼──────────────┼─────────┼──────────────┼──────────────────
  1   │ shoulder │ Revolute 77  │ Z       │ 0            │ hip swing (yaw)
  1   │ elbow    │ Revolute 90  │ XY-diag │ 1            │ arm raise/lower
  1   │ wrist    │ Revolute 104 │ XY-diag │ 2            │ forearm curl
  2   │ shoulder │ Revolute 78  │ Z       │ 3            │ hip swing
  2   │ elbow    │ Revolute 93  │ XY-diag │ 4            │ arm raise/lower
  2   │ wrist    │ Revolute 103 │ XY-diag │ 5            │ forearm curl
  3   │ shoulder │ Revolute 79  │ Z       │ 6            │ hip swing
  3   │ elbow    │ Revolute 92  │ XY-diag │ 7            │ arm raise/lower
  3   │ wrist    │ Revolute 102 │ XY-diag │ 8            │ forearm curl
  4   │ shoulder │ Revolute 80  │ Z       │ 9            │ hip swing
  4   │ elbow    │ Revolute 91  │ XY-diag │ 10           │ arm raise/lower
  4   │ wrist    │ Revolute 105 │ XY-diag │ 11           │ forearm curl

Physical leg positions (shoulder servo XY in base frame):
  Leg 1 (Rev 77): rear  +Y  →  (-0.0752,  0.0301)
  Leg 2 (Rev 78): rear  −Y  →  (-0.0752, -0.0797)
  Leg 3 (Rev 79): front −Y  →  ( 0.0752, -0.0301)
  Leg 4 (Rev 80): front +Y  →  ( 0.0752,  0.0301)

Standing pose:
  STAND_ANGLES are currently set to zero (neutral / flat pose).
  TODO: run test_env(render=True), use introspect_joints(), then tune
  STAND_ANGLES and SPAWN_Z so the feet just contact the ground.

CPG gait (mirrored trot):
  Diagonal pair A (leg0=rear+Y, leg2=front-Y):  phase offset = 0
  Diagonal pair B (leg1=rear-Y, leg3=front+Y):  phase offset = π
  Explicit stance/swing separation (duty cycle = 0.65):
    Stance (65%): shoulder sweeps dir*(+AMP→-AMP), foot backward, elbow down.
    Swing  (35%): shoulder sweeps dir*(-AMP→+AMP), foot forward, elbow lifted.
  GAIT_SHOULDER_DIRS = [+1, -1, -1, +1]:
    +Y-side legs (0, 3): positive sweep = foot forward.
    -Y-side legs (1, 2): multiplied by -1 so foot also sweeps forward→backward.
    All four legs produce symmetric +X thrust; lateral forces cancel → no spin.
  RL action is treated as a residual correction on top of CPG targets.

Action  (12,) float32 — residual joint-angle offsets (rad), clipped to ±1.2
Obs    (372,) float32 — 360 LiDAR distances (m)  +  12 joint positions (rad)

Usage
-----
    env = SpiderEnv(render_mode="human")
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    # Sanity check with pure CPG gait:
    from spider_env_new import test_env
    test_env(render=True)
"""

import math
import os
import random

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


# ── Controlled joint names, in action-vector order ────────────────────────────
# Derived by tracing each kinematic chain in the new URDF:
#   shoulder → servo_arm → arm-a → elbow servo → servo_arm → arm-b → wrist servo → foot
JOINT_NAMES = [
    "Revolute 77",   # leg 1 – shoulder  (Z-axis,    PyBullet idx 0)
    "Revolute 90",   # leg 1 – elbow     (XY-diag,   PyBullet idx 1)
    "Revolute 104",  # leg 1 – wrist     (XY-diag,   PyBullet idx 2)
    "Revolute 78",   # leg 2 – shoulder  (Z-axis,    PyBullet idx 3)
    "Revolute 93",   # leg 2 – elbow     (XY-diag,   PyBullet idx 4)
    "Revolute 103",  # leg 2 – wrist     (XY-diag,   PyBullet idx 5)
    "Revolute 79",   # leg 3 – shoulder  (Z-axis,    PyBullet idx 6)
    "Revolute 92",   # leg 3 – elbow     (XY-diag,   PyBullet idx 7)
    "Revolute 102",  # leg 3 – wrist     (XY-diag,   PyBullet idx 8)
    "Revolute 80",   # leg 4 – shoulder  (Z-axis,    PyBullet idx 9)
    "Revolute 91",   # leg 4 – elbow     (XY-diag,   PyBullet idx 10)
    "Revolute 105",  # leg 4 – wrist     (XY-diag,   PyBullet idx 11)
]

_DEFAULT_URDF = os.path.join("SpdrBot_description copy", "urdf", "SpdrBot.urdf")
URDF_PATH = os.environ.get("SPDRBOT_URDF", _DEFAULT_URDF)

# ── Standing-pose target angles (radians) ─────────────────────────────────────
# Layout: [shoulder, elbow, wrist] × 4 legs  (same order as JOINT_NAMES).
#
# TODO: these are placeholder zeros for the new CAD-accurate URDF.
# The elbow/wrist joints now have diagonal XY axes, so the old angle values
# no longer correspond to the same physical poses.  Calibrate by:
#   1. Run test_env(render=True, n_steps=0) to see the neutral pose.
#   2. Incrementally adjust elbow and wrist values until the feet just
#      contact the ground and the robot is stable.
#   3. Update SPAWN_Z in _load_robot() to match the new standing height.
#
STAND_ANGLES = np.array([
    0.0, 0.0, 0.0,   # leg 1 – rear  +Y
    0.0, 0.0, 0.0,   # leg 2 – rear  −Y
    0.0, 0.0, 0.0,   # leg 3 – front −Y
    0.0, 0.0, 0.0,   # leg 4 – front +Y
], dtype=np.float64)

# ── CPG gait parameters ───────────────────────────────────────────────────────
GAIT_FREQUENCY  = 2.0     # Hz — stride cycles per second (higher = faster walk)
GAIT_SIM_HZ     = 240.0   # PyBullet default physics rate
GAIT_PHASE_STEP = 2.0 * math.pi * GAIT_FREQUENCY / GAIT_SIM_HZ

GAIT_AMP_SHOULDER = 0.35  # rad — reduced from 0.55 for stable debugging (raise once gait confirmed)
GAIT_AMP_LIFT     = 0.45  # rad — how high the elbow pulls up during swing

# Phase offsets — perfectly symmetric trot (no deliberate asymmetry at this stage):
#   Pair A (leg0=rear+Y, leg2=front-Y) — true left diagonal, phase 0.0
#   Pair B (leg1=rear-Y, leg3=front+Y) — true right diagonal, phase π
# Both legs in a pair are in swing simultaneously; the two pairs alternate.
# No asymmetric offset: any residual spin is now caused by geometry, not phase.
GAIT_PHASE_OFFSETS = np.array([0.0, math.pi, 0.0, math.pi])

# Shoulder direction multiplier per leg.
#
# The shoulder servo rotates around the Z-axis.  Positive rotation sweeps the
# arm tip counterclockwise when viewed from above.  The foot sweep direction in
# X depends on which side the arm extends from the body:
#
#   +Y side (legs 0 and 3): arm points outward in +Y direction.
#          Positive shoulder → foot sweeps toward +X (forward).
#          So during stance (power stroke), sweep is +AMP → -AMP: correct.
#
#   -Y side (legs 1 and 2): arm points outward in -Y direction.
#          Positive shoulder → foot sweeps toward -X (backward — WRONG!).
#          Without correction, the stance stroke pushes the foot forward
#          instead of backward, producing a backward ground force → net torque.
#
# Multiplying by -1 for -Y legs inverts their shoulder sweep so every leg
# performs the same physical motion: foot starts forward, sweeps backward
# during stance, then lifts and recovers during swing.  Left/right forces
# now cancel laterally and sum in +X → forward motion without rotation.
# Signs are negated from the original geometric prediction because the URDF
# arm geometry means positive Z-rotation on +Y-side legs actually sweeps the
# foot toward -X (confirmed by running the sim).  Negating corrects this.
GAIT_SHOULDER_DIRS = np.array([-1.0, +1.0, +1.0, -1.0])

# ── Substeps and episode length ───────────────────────────────────────────────
# Each call to env.step() runs PHYSICS_SUBSTEPS physics steps before returning.
# The agent makes one decision, then the robot acts on it for that many ticks.
#
#   PHYSICS_SUBSTEPS = 8  →  each agent step = 8/240 ≈ 33 ms of simulation
#   MAX_STEPS        = 1000 →  episode = 1000 × 33 ms ≈ 33 seconds
#                           = ~66 full gait cycles at 2.0 Hz
#
# Without substeps a 500-step episode was only ~2 seconds — far too short for
# the robot to move any meaningful distance before the episode ended.
PHYSICS_SUBSTEPS = 8
MAX_STEPS        = 1000

# ── Stability threshold ───────────────────────────────────────────────────────
FALL_ANGLE_RAD = 1.05    # ~60° in pitch or roll — robot considered fallen

# ── Reward weights ────────────────────────────────────────────────────────────
# Target scale: most steps should produce a total reward in roughly [-3, +3].
FORWARD_WEIGHT       = 6.0    # reduced — collision avoidance now takes priority
VELOCITY_WEIGHT      = 1.5    # reduced — fades to zero automatically near obstacles
HEADING_WEIGHT       = 2.0    # reward for facing +X: cos(yaw) × weight
SURVIVAL_BONUS       = 0.01   # tiny — must not compete with forward progress
SAFE_RADIUS          = 1.2    # m — wider warning zone gives agent more time to react
OBSTACLE_WEIGHT      = 2.0    # strong proximity gradient — cost grows quickly as robot closes in
COLLISION_PENALTY    = -15.0  # hard terminal penalty; 2.5× the goal bonus to make contact clearly fatal
FALL_PENALTY         = -5.0   # terminal penalty for tipping over
ENERGY_WEIGHT        = 0.001  # scales sum of |joint torques|; kept very small
SMOOTH_WEIGHT        = 0.02   # smoothness nudge on action changes
STAGNATION_PENALTY   = 0.2    # gentle nudge to keep moving; low to avoid jitter
STAGNATION_THRESHOLD = 0.0002 # min metres per step to avoid stagnation penalty

# Lateral evasion reward.
# Active only when an obstacle is within SAFE_RADIUS.  Rewards sideways
# movement (|vy|) scaled by how close the nearest obstacle is:
#   closeness = (SAFE_RADIUS - dist) / SAFE_RADIUS   → 0 at edge, 1 at contact
#   bonus = LATERAL_EVASION_WEIGHT × closeness × |vy|
# Far from obstacles this is exactly zero — no incentive to swerve on open ground.
# The forward velocity reward is simultaneously suppressed by (1 - closeness) so
# the agent cannot gain by charging straight into an obstacle.
LATERAL_EVASION_WEIGHT = 3.0

# ── IMU-based stability penalties ─────────────────────────────────────────────
# These directly address the spinning-in-place failure mode.
#
# YAW_RATE_WEIGHT: penalty per rad/s of yaw (Z-axis) angular velocity.
# TILT_WEIGHT: penalty on (roll² + pitch²) in radians (quadratic).
# YAW_RATE_CLIP: normalisation ceiling for yaw_rate in the observation.
YAW_RATE_WEIGHT = 1.0   # penalty per rad/s of yaw angular velocity
TILT_WEIGHT     = 0.5   # penalty weight on roll² + pitch² (rad²)
YAW_RATE_CLIP   = 5.0   # rad/s ceiling for obs normalisation

# ── Residual action scale and smoothing ───────────────────────────────────────
# RESIDUAL_SCALE multiplies the agent's raw action before it is added to the
# CPG target.  It must be small enough that the CPG dominates.
#
#   CPG shoulder amplitude : ±0.55 rad  (range = 1.10 rad)
#   Max residual (raw)     : ±(1.2 × 0.15) = ±0.18 rad   ≈ 33% of CPG swing
#
# This ensures the CPG provides ≥67% of the control signal at all times, so
# random RL exploration cannot flip joint directions or override the gait.
RESIDUAL_SCALE = 0.15

# EMA weight applied to the residual each step before it reaches the joints.
# Acts as a first-order IIR low-pass filter: only ACTION_SMOOTH_ALPHA of the
# new action bleeds through per step; the rest carries over from the previous.
#   α = 0.3  →  ~3-step time constant  →  ~100 ms lag at 33 ms/step
# Effect: PPO cannot inject a full-magnitude impulse in a single step.
# The policy can still learn sustained corrections — just not high-frequency noise.
ACTION_SMOOTH_ALPHA = 0.3


class SpiderEnv(gym.Env):
    """
    Gymnasium env: SpdrBot 4-leg spider walks forward (+X) and avoids obstacles.

    Episode structure
    -----------------
    Each episode lasts at most MAX_STEPS steps (truncated=True at the limit).
    It ends early (terminated=True) only if the robot falls over.
    Reaching x >= 5.0 m ends the episode with a large bonus.
    Hitting an obstacle is penalised but does NOT end the episode — the robot
    must learn to walk around things, not just avoid the first one.
    """
    metadata = {"render_modes": ["human"]}

    # Joints are type="continuous" in the URDF (no hard limits).
    # JOINT_LIMIT is enforced purely in software via np.clip.
    JOINT_LIMIT  = 1.2     # ±68.7° software limit
    MAX_FORCE    = 20.0    # N·m  — tune to match physical servo stall torque
    MAX_VELOCITY = 5.0     # rad/s — tune to match physical servo speed

    MAX_LIDAR = 5.0
    NUM_RAYS  = 360

    # Mirror module-level reward constants as class attributes so they can be
    # overridden per-instance if needed.
    FORWARD_WEIGHT       = FORWARD_WEIGHT
    VELOCITY_WEIGHT      = VELOCITY_WEIGHT
    HEADING_WEIGHT       = HEADING_WEIGHT
    SURVIVAL_BONUS       = SURVIVAL_BONUS
    SAFE_RADIUS          = SAFE_RADIUS
    OBSTACLE_WEIGHT      = OBSTACLE_WEIGHT
    COLLISION_PENALTY    = COLLISION_PENALTY
    FALL_PENALTY         = FALL_PENALTY
    ENERGY_WEIGHT        = ENERGY_WEIGHT
    SMOOTH_WEIGHT        = SMOOTH_WEIGHT
    STAGNATION_PENALTY   = STAGNATION_PENALTY
    STAGNATION_THRESHOLD = STAGNATION_THRESHOLD
    RESIDUAL_SCALE       = RESIDUAL_SCALE
    ACTION_SMOOTH_ALPHA  = ACTION_SMOOTH_ALPHA
    YAW_RATE_WEIGHT        = YAW_RATE_WEIGHT
    TILT_WEIGHT            = TILT_WEIGHT
    YAW_RATE_CLIP          = YAW_RATE_CLIP
    LATERAL_EVASION_WEIGHT = LATERAL_EVASION_WEIGHT

    def __init__(self, render_mode: str = "human", urdf_path: str = URDF_PATH):
        super().__init__()
        self.render_mode = render_mode
        self.urdf_path   = urdf_path

        # Observation: 360 LiDAR + 12 joint angles + 1 yaw + 3 IMU → 376-dim vector
        #   [0:360]   LiDAR distances (m)
        #   [360:372] joint positions (rad)
        #   [372]     yaw (rad) — absolute heading, for aligning with +X
        #   [373]     roll_n    — normalised to [-1, 1] via /π
        #   [374]     pitch_n   — normalised to [-1, 1] via /π
        #   [375]     yaw_rate_n — normalised to [-1, 1] via clip/YAW_RATE_CLIP
        #
        # roll/pitch let the agent detect and correct tilt before falling.
        # yaw_rate gives the agent direct feedback on spinning so it can
        # learn to cancel rotation — something heading (yaw) alone cannot do
        # because the agent would need to differentiate yaw over time itself.
        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.zeros(self.NUM_RAYS, dtype=np.float32),
                np.full(12,   -np.pi,  dtype=np.float32),
                np.array([-np.pi],     dtype=np.float32),  # yaw
                np.full(3,    -1.0,    dtype=np.float32),  # roll_n, pitch_n, yaw_rate_n
            ]),
            high=np.concatenate([
                np.full(self.NUM_RAYS, self.MAX_LIDAR, dtype=np.float32),
                np.full(12,  np.pi,                   dtype=np.float32),
                np.array([np.pi],                     dtype=np.float32),  # yaw
                np.full(3,   1.0,                     dtype=np.float32),  # roll_n, pitch_n, yaw_rate_n
            ]),
            dtype=np.float32,
        )
        # Action: residual joint-angle offsets added on top of CPG targets
        self.action_space = spaces.Box(
            low=-self.JOINT_LIMIT, high=self.JOINT_LIMIT,
            shape=(12,), dtype=np.float32,
        )

        # PyBullet
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.robot_id      = None
        self.joint_ids     = []
        self.wall_ids      = []   # arena boundary walls — penalise but don't terminate
        self.obstacle_ids  = []   # red box obstacles — penalise and track proximity
        self.position      = [0.0, 0.0, 0.0]
        self._prev_x       = 0.0
        self.gait_phase    = 0.0
        self.joint_torques = np.zeros(12, dtype=np.float64)
        self.step_count         = 0    # counts steps within the current episode
        self._prev_action       = np.zeros(12, dtype=np.float32)  # for smoothness penalty
        self._smoothed_residual = np.zeros(12, dtype=np.float32)  # EMA state for action filter

        self.reset()

    # ── URDF introspection ────────────────────────────────────────────────────

    def introspect_joints(self, verbose: bool = True) -> dict:
        """
        Print all joint info from the loaded URDF and the current joint states.
        Call after reset() to verify joint indices, axes and limits.
        """
        TYPE_NAMES = {
            p.JOINT_REVOLUTE:  "REVOLUTE",
            p.JOINT_PRISMATIC: "PRISMATIC",
            p.JOINT_SPHERICAL: "SPHERICAL",
            p.JOINT_PLANAR:    "PLANAR",
            p.JOINT_FIXED:     "FIXED",
        }
        n = p.getNumJoints(self.robot_id)
        info_dict = {}

        if verbose:
            print(f"\n{'─'*72}")
            print(f"  Robot ID: {self.robot_id}   Total joints: {n}")
            print(f"{'─'*72}")
            print(f"  {'Idx':>3}  {'Name':<20}  {'Type':<10}  {'Lower':>7}  {'Upper':>7}")
            print(f"{'─'*72}")

        for i in range(n):
            ji    = p.getJointInfo(self.robot_id, i)
            name  = ji[1].decode()
            jtype = TYPE_NAMES.get(ji[2], str(ji[2]))
            lo, hi = ji[8], ji[9]
            info_dict[i] = {"name": name, "type": jtype, "lower": lo, "upper": hi}
            if verbose:
                print(f"  {i:>3}  {name:<20}  {jtype:<10}  {lo:>7.4f}  {hi:>7.4f}")

        if verbose:
            print(f"{'─'*72}")
            print("\nControlled joint states (after settle):")
            print(f"  {'Idx':>3}  {'Name':<20}  {'Pos (rad)':>10}  {'Vel (rad/s)':>12}")
            print(f"{'─'*72}")
            for jid in self.joint_ids:
                js   = p.getJointState(self.robot_id, jid)
                name = p.getJointInfo(self.robot_id, jid)[1].decode()
                print(f"  {jid:>3}  {name:<20}  {js[0]:>10.4f}  {js[1]:>12.4f}")
            print(f"{'─'*72}\n")

        return info_dict

    # ── Robot loading ─────────────────────────────────────────────────────────

    def _load_robot(self):
        # SPAWN_Z: height so feet just contact z=0 in the standing pose.
        # TODO: calibrate once STAND_ANGLES are set.
        SPAWN_Z = 0.20
        robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, SPAWN_Z],
            useFixedBase=False,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        p.changeDynamics(robot_id, -1, linearDamping=0.3, angularDamping=0.3)

        # Build a name→index map for all joints in the URDF
        name_to_idx = {
            p.getJointInfo(robot_id, i)[1].decode(): i
            for i in range(p.getNumJoints(robot_id))
        }

        missing = [n for n in JOINT_NAMES if n not in name_to_idx]
        if missing:
            raise RuntimeError(
                f"URDF missing joints: {missing}\n"
                f"Available: {sorted(name_to_idx)}"
            )

        joint_ids = [name_to_idx[n] for n in JOINT_NAMES]

        # Enable torque sensors so we can measure energy usage in the reward
        for jid in joint_ids:
            p.enableJointForceTorqueSensor(robot_id, jid, enableSensor=True)

        # Move joints to the standing pose and hold them there
        for i, jid in enumerate(joint_ids):
            angle = float(np.clip(STAND_ANGLES[i], -self.JOINT_LIMIT, self.JOINT_LIMIT))
            p.resetJointState(robot_id, jid, targetValue=angle, targetVelocity=0.0)
            p.setJointMotorControl2(
                robot_id, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=self.MAX_FORCE,
                maxVelocity=self.MAX_VELOCITY,
            )

        return robot_id, joint_ids

    # ── Arena construction ────────────────────────────────────────────────────

    def _make_box(self, half_extents, position, orientation=None, color=None):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=color or [0.6, 0.6, 0.6, 1],
        )
        kw = dict(baseMass=0,
                  baseCollisionShapeIndex=col,
                  baseVisualShapeIndex=vis,
                  basePosition=position)
        if orientation is not None:
            kw["baseOrientation"] = orientation
        return p.createMultiBody(**kw)

    def _create_wall(self, pos, length=5.0):
        # Wall running along the X axis
        return self._make_box([length / 2, 0.05, 0.5], pos)

    def _create_wall90(self, pos, length=5.0):
        # Wall running along the Y axis (rotated 90°)
        orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        return self._make_box([length / 2, 0.05, 0.5], pos, orn)

    def _create_obstacle(self, pos):
        # Red box — the things the robot should steer around
        return self._make_box([0.2, 0.2, 0.5], pos, color=[1, 0, 0, 1])

    # ── Sensors ───────────────────────────────────────────────────────────────

    def _get_lidar(self) -> np.ndarray:
        """Cast 360 rays horizontally from the robot centre; return hit distances."""
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        angles    = np.linspace(0, 2 * np.pi, self.NUM_RAYS, endpoint=False)
        rays_from = [base_pos] * self.NUM_RAYS
        rays_to   = [
            [base_pos[0] + self.MAX_LIDAR * np.cos(a),
             base_pos[1] + self.MAX_LIDAR * np.sin(a),
             base_pos[2]]
            for a in angles
        ]
        results = p.rayTestBatch(rays_from, rays_to)
        # r[2] is the fraction of the ray that hit something (0–1); scale to metres
        return np.array([r[2] * self.MAX_LIDAR for r in results], dtype=np.float32)

    def _get_joint_positions(self) -> np.ndarray:
        return np.array(
            [p.getJointState(self.robot_id, jid)[0] for jid in self.joint_ids],
            dtype=np.float32,
        )

    def _get_yaw(self) -> float:
        """Return the robot's yaw angle in [-π, π] (rotation around Z axis)."""
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        return float(yaw)

    def _get_imu(self) -> tuple[float, float, float]:
        """
        Return normalised IMU signals (roll_n, pitch_n, yaw_rate_n), each in [-1, 1].

        Normalisation:
          roll_n      = roll    / π          — full-circle normalisation
          pitch_n     = pitch   / π          — full-circle normalisation
          yaw_rate_n  = clip(ω_z, ±YAW_RATE_CLIP) / YAW_RATE_CLIP

        Why yaw_rate and not just yaw:
          Yaw (absolute heading) tells the agent *where* it is pointing.
          Yaw_rate tells the agent *how fast it is spinning right now*.
          A robot that has already spun 90° and is still accelerating needs
          both signals to correct itself; rate alone enables reactive damping.
        """
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        _, ang_vel = p.getBaseVelocity(self.robot_id)
        yaw_rate = float(ang_vel[2])   # world-Z angular velocity ≈ yaw rate

        roll_n     = float(roll)   / math.pi
        pitch_n    = float(pitch)  / math.pi
        yaw_rate_n = float(np.clip(yaw_rate, -self.YAW_RATE_CLIP, self.YAW_RATE_CLIP)) / self.YAW_RATE_CLIP
        return roll_n, pitch_n, yaw_rate_n

    def _get_observation(self) -> np.ndarray:
        # 360 LiDAR + 12 joint positions + 1 yaw + 3 IMU → 376-dim obs vector
        roll_n, pitch_n, yaw_rate_n = self._get_imu()
        return np.concatenate([
            self._get_lidar(),
            self._get_joint_positions(),
            np.array([self._get_yaw()],                     dtype=np.float32),
            np.array([roll_n, pitch_n, yaw_rate_n],         dtype=np.float32),
        ])

    # ── Stability check ───────────────────────────────────────────────────────

    def _has_fallen(self) -> bool:
        """True if roll or pitch exceeds FALL_ANGLE_RAD (~60°)."""
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        return abs(roll) > FALL_ANGLE_RAD or abs(pitch) > FALL_ANGLE_RAD

    # ── CPG gait ──────────────────────────────────────────────────────────────

    def _compute_gait_targets(self) -> np.ndarray:
        """
        Compute 12 joint targets using an asymmetric stance/swing CPG with
        per-side shoulder mirroring.

        Each leg cycle has two phases:

        STANCE (DUTY_CYCLE = 65% of cycle):
          - Foot on ground.  Shoulder sweeps dir * (+AMP → -AMP).
          - The direction multiplier (GAIT_SHOULDER_DIRS) ensures the foot
            always moves backward relative to the body regardless of which
            side the leg is on.  Friction converts this into forward thrust.
          - Elbow held at stand angle to keep the foot in ground contact.

        SWING (remaining 35%):
          - Foot lifted.  Shoulder sweeps dir * (-AMP → +AMP) to recover.
          - Bell-curve elbow lift (sin profile) — smooth liftoff and touchdown.

        Symmetry guarantee:
          GAIT_SHOULDER_DIRS = [+1, -1, -1, +1]
          Legs 0 and 3 are on the +Y side:  positive shoulder = foot forward.
          Legs 1 and 2 are on the -Y side:  positive shoulder = foot backward.
          Negating -Y legs' sweep inverts this, so all four feet sweep forward
          then backward identically.  Left/right lateral forces cancel and only
          the +X thrust remains → no net yaw torque from the CPG.
        """
        DUTY_CYCLE = 0.65
        TWO_PI = 2.0 * math.pi

        targets = np.empty(12, dtype=np.float64)

        for leg in range(4):
            phi_norm = ((self.gait_phase + GAIT_PHASE_OFFSETS[leg]) % TWO_PI) / TWO_PI
            direction = GAIT_SHOULDER_DIRS[leg]   # +1 for +Y side, -1 for -Y side

            s_idx = leg * 3
            e_idx = leg * 3 + 1
            w_idx = leg * 3 + 2

            if phi_norm < DUTY_CYCLE:
                # ── STANCE: power stroke ──────────────────────────────────────
                t = phi_norm / DUTY_CYCLE          # 0 → 1 during stance
                # dir*(+AMP → -AMP): foot starts forward, sweeps backward.
                # For -Y legs, direction=-1 flips this to (-AMP → +AMP) in raw
                # joint space, which is still foot-forward → foot-backward in
                # Cartesian space because the arm extends in -Y.
                shoulder = STAND_ANGLES[s_idx] + direction * GAIT_AMP_SHOULDER * (1.0 - 2.0 * t)
                elbow    = STAND_ANGLES[e_idx]   # foot stays low, maintains ground contact
                wrist    = STAND_ANGLES[w_idx]
            else:
                # ── SWING: recovery stroke ────────────────────────────────────
                t = (phi_norm - DUTY_CYCLE) / (1.0 - DUTY_CYCLE)  # 0 → 1 during swing
                # dir*(-AMP → +AMP): foot returns from backward to forward.
                shoulder = STAND_ANGLES[s_idx] + direction * GAIT_AMP_SHOULDER * (2.0 * t - 1.0)
                lift     = math.sin(t * math.pi)   # bell curve: 0 at liftoff/touchdown, 1 at peak
                elbow    = STAND_ANGLES[e_idx] - GAIT_AMP_LIFT * lift
                wrist    = STAND_ANGLES[w_idx]

            targets[s_idx] = shoulder
            targets[e_idx] = elbow
            targets[w_idx] = wrist

        np.clip(targets, -self.JOINT_LIMIT, self.JOINT_LIMIT, out=targets)
        self.gait_phase = (self.gait_phase + GAIT_PHASE_STEP * PHYSICS_SUBSTEPS) % TWO_PI
        return targets.astype(np.float32)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, robot_pos, robot_vel_x: float, action: np.ndarray) -> tuple[float, dict]:
        """
        Compute reward and return (total, component_dict).

        Components
        ----------
        forward      : reward for +X displacement this step (position progress)
        velocity     : reward for current global forward speed
        heading      : reward for facing +X  (cos yaw × weight)
        yaw_rate     : penalty for spinning  (|ω_z| × weight)
        tilt         : penalty for roll/pitch tilt  ((roll² + pitch²) × weight)
        survival     : small bonus for each step still alive
        proximity    : penalty for being close to a red obstacle
        collision    : penalty for touching a red obstacle
        fall         : penalty for tipping over
        energy       : penalty proportional to motor effort
        smooth       : penalty for abrupt action changes
        stagnation   : penalty for barely moving forward

        Spinning fix rationale
        ----------------------
        The heading reward (cos yaw) rewards *being* aligned with +X but gives
        no signal about *rate of rotation* — the agent cannot distinguish a
        stable heading from a momentary crossing of the +X direction during a spin.
        yaw_rate_penalty is a direct, instantaneous cost on angular velocity that
        makes every degree of unproductive rotation immediately expensive.
        Together they complement each other: heading teaches direction, yaw_rate
        teaches to stop rotating.
        """
        # ── Position / velocity signals ───────────────────────────────────────
        dx = robot_pos[0] - self._prev_x
        forward_reward  = dx * self.FORWARD_WEIGHT
        velocity_reward = robot_vel_x * self.VELOCITY_WEIGHT

        # ── IMU + lateral velocity ────────────────────────────────────────────
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        yaw_rate = float(ang_vel[2])   # world-Z angular velocity ≈ yaw rate (rad/s)
        vy       = float(lin_vel[1])   # global Y velocity — used for lateral evasion

        # Heading: cos(yaw)==1 facing +X, -1 facing -X.
        heading_reward = math.cos(yaw) * self.HEADING_WEIGHT

        # Yaw rate penalty: immediate cost for spinning regardless of heading.
        # Makes every rad/s of unproductive rotation strictly expensive.
        yaw_rate_penalty = self.YAW_RATE_WEIGHT * abs(yaw_rate)

        # Tilt penalty: quadratic cost on roll and pitch.
        # Near-zero when upright; grows with lean to keep the body level.
        tilt_penalty = self.TILT_WEIGHT * (float(roll) ** 2 + float(pitch) ** 2)

        # ── Other penalties ───────────────────────────────────────────────────
        stagnation_penalty = (
            -self.STAGNATION_PENALTY if dx < self.STAGNATION_THRESHOLD else 0.0
        )

        survival = self.SURVIVAL_BONUS

        # ── Obstacle proximity, lateral evasion, and forward suppression ─────
        # For each obstacle inside SAFE_RADIUS, compute a normalised closeness
        # score in [0, 1] (0 = at the safe-radius boundary, 1 = touching).
        # This single value drives three coupled effects:
        #
        #   1. proximity_penalty : grows linearly with closeness — strong
        #      deterrent that activates well before contact.
        #
        #   2. lateral_bonus : rewards sideways speed (|vy|) proportional to
        #      closeness — incentive to swerve rather than brake.
        #
        #   3. velocity_reward suppression : scales down the forward-velocity
        #      reward by (1 - max_closeness) — charging into an obstacle can
        #      never be profitable even at high speed.
        proximity_penalty = 0.0
        lateral_bonus     = 0.0
        max_closeness     = 0.0   # track the nearest obstacle for suppression

        for obs_id in self.obstacle_ids:
            obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
            dist = float(np.linalg.norm(
                np.array(robot_pos[:2]) - np.array(obs_pos[:2])
            ))
            if dist < self.SAFE_RADIUS:
                closeness = (self.SAFE_RADIUS - dist) / self.SAFE_RADIUS  # [0, 1]
                proximity_penalty += closeness * self.OBSTACLE_WEIGHT
                lateral_bonus     += closeness * abs(vy) * self.LATERAL_EVASION_WEIGHT
                max_closeness      = max(max_closeness, closeness)

        # Suppress forward velocity reward as obstacles get closer.
        # At closeness=0 (edge of safe zone): full velocity reward.
        # At closeness=1 (contact): velocity reward = 0.
        velocity_reward *= (1.0 - max_closeness)

        collision_penalty = 0.0
        for obs_id in self.obstacle_ids:
            if p.getContactPoints(self.robot_id, obs_id):
                collision_penalty += self.COLLISION_PENALTY

        fall_penalty    = self.FALL_PENALTY if self._has_fallen() else 0.0
        energy_penalty  = -self.ENERGY_WEIGHT * float(np.sum(np.abs(self.joint_torques)))
        smooth_penalty  = -self.SMOOTH_WEIGHT * float(np.mean(np.abs(action - self._prev_action)))

        total = (
            forward_reward
            + velocity_reward       # already suppressed near obstacles
            + heading_reward
            - yaw_rate_penalty
            - tilt_penalty
            + survival
            - proximity_penalty
            + lateral_bonus         # reward for swerving when close
            + collision_penalty     # already negative
            + fall_penalty          # already negative
            + energy_penalty        # already negative
            + smooth_penalty        # already negative
            + stagnation_penalty    # already negative
        )

        components = {
            "forward":    round(forward_reward,      4),
            "velocity":   round(velocity_reward,     4),
            "heading":    round(heading_reward,       4),
            "yaw_rate":   round(-yaw_rate_penalty,   4),
            "tilt":       round(-tilt_penalty,        4),
            "survival":   round(survival,             4),
            "proximity":  round(-proximity_penalty,   4),
            "lateral":    round(lateral_bonus,        4),
            "collision":  round(collision_penalty,    4),
            "fall":       round(fall_penalty,         4),
            "energy":     round(energy_penalty,       4),
            "smooth":     round(smooth_penalty,       4),
            "stagnation": round(stagnation_penalty,   4),
            "total":      round(total,                4),
            # Raw IMU values (debug only — not part of the reward total)
            "_yaw_rate_rads": round(yaw_rate,             3),
            "_roll_deg":      round(math.degrees(roll),   1),
            "_pitch_deg":     round(math.degrees(pitch),  1),
        }
        return float(total), components

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.robot_id, self.joint_ids = self._load_robot()

        # Let gravity settle the robot into its standing pose (~0.5 s at 240 Hz)
        for _ in range(120):
            p.stepSimulation()

        self.gait_phase         = 0.0
        self.joint_torques      = np.zeros(12, dtype=np.float64)
        self.step_count         = 0
        self._prev_action       = np.zeros(12, dtype=np.float32)
        self._smoothed_residual = np.zeros(12, dtype=np.float32)  # reset filter memory each episode

        # Arena walls — stored separately so they don't affect the reward signal
        self.wall_ids = [
            self._create_wall   ([ 2.5,  3.0, 0.5], length=7),  # top wall
            self._create_wall90 ([-1.0,  0.0, 0.5], length=6),  # back wall
            self._create_wall   ([ 2.5, -3.0, 0.5], length=7),  # bottom wall
            self._create_wall90 ([ 6.0,  0.0, 0.5], length=6),  # front wall (goal end)
        ]

        # Red obstacles — these affect proximity/collision reward
        self.obstacle_ids = []
        for _ in range(5):
            self.obstacle_ids.append(self._create_obstacle([
                random.uniform(1.0, 5.0),
                random.uniform(-2.5, 2.5),
                0.5,
            ]))

        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.position = list(pos)
        self._prev_x  = pos[0]

        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        # 1. Get CPG baseline joint targets for this phase
        gait_targets = self._compute_gait_targets()

        # 2. Combine RL residual with CPG baseline.
        #
        #    Two-stage protection against RL destabilising the gait:
        #
        #    a) EMA low-pass filter — update the running smoothed residual with
        #       only ACTION_SMOOTH_ALPHA of the new action each step.  This prevents
        #       PPO from flipping joint targets in a single 33 ms step; sustained
        #       corrections bleed through fully over ~3 steps, but high-frequency
        #       impulses are heavily attenuated.
        #
        #    b) RESIDUAL_SCALE = 0.15 — caps the smoothed residual at ±0.18 rad,
        #       keeping it ≤33% of the CPG shoulder swing amplitude (±0.55 rad).
        #       CPG remains the dominant signal at all times.
        #
        #    The smoothness *penalty* (SMOOTH_WEIGHT) is computed on the raw action
        #    delta — this trains the policy itself to prefer gradual changes, not just
        #    filters them away silently.
        action = np.asarray(action, dtype=np.float32)
        self._smoothed_residual = (
            self.ACTION_SMOOTH_ALPHA * action
            + (1.0 - self.ACTION_SMOOTH_ALPHA) * self._smoothed_residual
        )
        residual = self._smoothed_residual * self.RESIDUAL_SCALE  # max ±0.18 rad
        joint_targets = np.clip(gait_targets + residual,
                                -self.JOINT_LIMIT, self.JOINT_LIMIT).astype(np.float32)

        # 3. Send targets to all 12 actuators
        #    PyBullet holds these targets across all substeps automatically.
        for i, jid in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                self.robot_id, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(joint_targets[i]),
                force=self.MAX_FORCE,
                maxVelocity=self.MAX_VELOCITY,
            )

        # 4. Advance physics PHYSICS_SUBSTEPS times.
        #    This makes each agent decision cover ~33 ms of real simulation
        #    instead of just 4 ms, so the robot moves meaningful distance
        #    per step and MAX_STEPS episodes last ~33 seconds.
        for _ in range(PHYSICS_SUBSTEPS):
            p.stepSimulation()
        self.step_count += 1

        # 5. Read motor torques after the last substep (used in energy penalty)
        self.joint_torques = np.array(
            [p.getJointState(self.robot_id, jid)[3] for jid in self.joint_ids],
            dtype=np.float64,
        )

        # 6. Gather new position, velocity, and observation
        new_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        new_vel, _ = p.getBaseVelocity(self.robot_id)
        obs = self._get_observation()

        # 7. Compute reward (pass forward velocity and current action for new terms)
        reward, components = self._compute_reward(new_pos, new_vel[0], action)

        # 8. Check termination / truncation
        terminated = False
        truncated  = False

        collided = any(
            p.getContactPoints(self.robot_id, obs_id)
            for obs_id in self.obstacle_ids
        )

        if new_pos[0] >= 5.0:
            # Robot reached the goal — bonus scaled to ~10 steps of fast travel
            reward    += 5.0
            terminated = True
        elif collided:
            # Hit an obstacle — apply terminal penalty and end episode.
            # COLLISION_PENALTY is already applied per-step inside _compute_reward,
            # but we also terminate here so the agent cannot recover and linger.
            terminated = True
        elif self._has_fallen():
            # Robot tipped over — end episode
            terminated = True
        elif self.step_count >= MAX_STEPS:
            # Ran out of time — truncate (not a failure, just a timeout)
            truncated = True

        # 9. Update bookkeeping
        self.position    = list(new_pos)
        self._prev_x     = new_pos[0]
        self._prev_action = action.copy()

        return obs, float(reward), terminated, truncated, {"reward_components": components}

    def render(self):
        pass  # GUI is opened in __init__ when render_mode == "human"

    def close(self):
        p.disconnect(self.client)


# ── Sanity-check / demo ───────────────────────────────────────────────────────

def test_env(
    urdf_path: str = URDF_PATH,
    n_steps:   int = 100,
    render:    bool = False,
):
    """
    Run n_steps with zero RL action (pure CPG gait) and print reward breakdown
    plus forward-motion and IMU diagnostics.

    What to look for:
      • 'forward' positive and X increasing → robot is moving in +X
      • net_displacement > 0               → CPG alone produces forward bias
      • 'yaw_rate' near 0                  → robot not spinning
      • 'tilt' near 0                      → robot staying upright
      • 'fall' = 0.0                       → stable
      • ω_z column small                   → low angular velocity (no spin)
    """
    import time

    env = SpiderEnv(render_mode="human" if render else "direct", urdf_path=urdf_path)
    env.reset()
    env.introspect_joints(verbose=True)

    # Record start position and initial IMU for baseline
    start_pos, orn = p.getBasePositionAndOrientation(env.robot_id)
    roll, pitch, _ = p.getEulerFromQuaternion(orn)
    print(f"Start position : {[round(v, 3) for v in start_pos]}")
    print(f"Roll={math.degrees(roll):.1f}°  Pitch={math.degrees(pitch):.1f}°  "
          f"Fallen={env._has_fallen()}\n")

    header = (f"{'Step':>5}  {'X':>7}  {'dX':>+7}  "
              f"{'fwd':>7}  {'ω_z':>6}  {'roll°':>6}  {'ptch°':>6}  "
              f"{'yr_pen':>7}  {'tlt_pen':>7}  {'fall':>5}  {'total':>8}")
    print(f"Running {n_steps} steps — pure CPG (action = zeros)")
    print(header)
    print("─" * len(header))

    zero_action = np.zeros(12, dtype=np.float32)
    cumulative  = {}
    x_history   = [start_pos[0]]
    done        = False
    steps_run   = 0

    for step in range(n_steps):
        _, _, terminated, truncated, info = env.step(zero_action)
        done = terminated or truncated
        c = info["reward_components"]
        # Accumulate only reward keys (skip debug keys starting with '_')
        for k, v in c.items():
            if not k.startswith("_"):
                cumulative[k] = cumulative.get(k, 0.0) + v

        x_now = env.position[0]
        x_history.append(x_now)
        steps_run = step + 1

        if step % 10 == 0 or done:
            window_start = max(0, len(x_history) - 11)
            dx_window    = x_history[-1] - x_history[window_start]
            print(
                f"{step:>5}  {x_now:>7.4f}  {dx_window:>+7.4f}"
                f"  {c.get('forward', 0):>7.4f}"
                f"  {c.get('_yaw_rate_rads', 0):>+6.2f}"
                f"  {c.get('_roll_deg',      0):>+6.1f}"
                f"  {c.get('_pitch_deg',     0):>+6.1f}"
                f"  {c.get('yaw_rate',       0):>7.4f}"
                f"  {c.get('tilt',           0):>7.4f}"
                f"  {c.get('fall',           0):>5.2f}"
                f"  {c.get('total',          0):>8.4f}"
            )
        if render:
            time.sleep(1.0 / 60.0)
        if done:
            reason = "terminated" if terminated else "truncated (time limit)"
            print(f"\n  Episode ended at step {step}: {reason}")
            break

    print("─" * len(header))

    # ── Forward-motion and stability summary ──────────────────────────────────
    net_disp  = x_history[-1] - x_history[0]
    sim_time  = steps_run * PHYSICS_SUBSTEPS / GAIT_SIM_HZ
    avg_speed = net_disp / sim_time if sim_time > 0 else 0.0
    fwd_steps = sum(1 for i in range(1, len(x_history)) if x_history[i] > x_history[i-1])
    bwd_steps = sum(1 for i in range(1, len(x_history)) if x_history[i] < x_history[i-1])
    avg_yr_pen = cumulative.get("yaw_rate", 0.0) / max(steps_run, 1)

    print(f"\nForward-motion summary ({steps_run} steps, {sim_time:.2f} s sim):")
    print(f"  Start X          : {x_history[0]:>+.4f} m")
    print(f"  End X            : {x_history[-1]:>+.4f} m")
    print(f"  Net displacement : {net_disp:>+.4f} m  "
          f"({'forward ✓' if net_disp > 0 else 'backward/no motion ✗'})")
    print(f"  Average speed    : {avg_speed:>+.4f} m/s")
    print(f"  Steps with +X    : {fwd_steps}  /  Steps with -X : {bwd_steps}")

    print(f"\nStability summary:")
    print(f"  Avg yaw_rate penalty/step : {avg_yr_pen:>+.4f}  "
          f"({'low spin ✓' if avg_yr_pen > -1.0 else 'spinning ✗'})")
    print(f"  Avg tilt penalty/step     : {cumulative.get('tilt', 0) / max(steps_run, 1):>+.4f}")

    print("\nCumulative reward totals:")
    for k, v in cumulative.items():
        print(f"  {k:<14}: {v:>10.4f}")

    env.close()
    return cumulative


if __name__ == "__main__":
    test_env(render=True, n_steps=500)
