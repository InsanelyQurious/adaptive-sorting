"""Section 30: environment init, reset, randomization, action validation, reward, joint limits,
object spawning, bin detection, grasp detection, placement detection."""
import numpy as np
import pytest
from simulation.environments.sorting_env import IMAGE_KEY, STATE_KEY, VolvoSortingEnv, make_env
from simulation.environments.workcell_sim import wrap_angle


def test_env_init_spaces(env):
    assert STATE_KEY in env.observation_space.spaces and IMAGE_KEY in env.observation_space.spaces
    assert env.observation_space[IMAGE_KEY].shape == (64, 64, 3)
    assert env.observation_space[STATE_KEY].shape == (len(env.state_layout),)
    assert env.action_space.shape == (7,)


def test_scene_contents(sim):
    m = sim.model
    names = [m.body(i).name for i in range(m.nbody)]
    for b in ("robot_base", "wrist_3_link", "gripper_base", "left_finger", "right_finger", "bin_a", "bin_b", "bracket", "bolt"):
        assert b in names, f"missing body {b}"
    assert m.geom("table_top").id >= 0
    cams = [m.camera(i).name for i in range(m.ncam)]
    assert "overhead" in cams and "workcell" in cams
    assert sim.robot_name == "UR10e"


def test_reset_returns_valid_obs(env):
    obs, info = env.reset(seed=1)
    assert obs[IMAGE_KEY].dtype == np.uint8 and obs[IMAGE_KEY].shape == (64, 64, 3)
    assert np.all(np.isfinite(obs[STATE_KEY]))
    assert obs[IMAGE_KEY].std() > 5, "camera image should not be blank"
    assert info["target"] in env.objects


def test_reset_is_deterministic_per_seed(env):
    o1, _ = env.reset(seed=7)
    o2, _ = env.reset(seed=7)
    np.testing.assert_allclose(o1[STATE_KEY][:6], o2[STATE_KEY][:6], atol=1e-6)
    p1 = env.sim.episode_config.object_poses
    env.reset(seed=7)
    assert env.sim.episode_config.object_poses == p1


def test_randomization_changes_layout(env):
    poses = []
    for s in range(5):
        env.reset(seed=s)
        c = env.sim.episode_config
        poses.append((c.object_poses["bracket"]["x"], c.object_poses["bracket"]["y"], c.object_poses["bracket"]["yaw"],
                      c.object_poses["bolt"]["x"], c.object_scales["bolt"], c.frictions["bracket"]))
    assert len(set(poses)) == 5
    # objects respect spawn zone and separation
    zone = env.env_cfg["randomization"]["spawn_zone"]
    for s in range(5):
        env.reset(seed=s)
        c = env.sim.episode_config.object_poses
        for o in ("bracket", "bolt"):
            assert zone["x"][0] - 1e-6 <= c[o]["x"] <= zone["x"][1] + 1e-6
            assert zone["y"][0] - 1e-6 <= c[o]["y"] <= zone["y"][1] + 1e-6
        d = np.hypot(c["bracket"]["x"] - c["bolt"]["x"], c["bracket"]["y"] - c["bolt"]["y"])
        assert d >= zone["min_separation"] - 1e-6


def test_randomization_level_zero_is_fixed(env):
    env.reset(seed=1, options={"randomization_level": 0.0})
    c1 = env.sim.episode_config.object_poses
    env.reset(seed=2, options={"randomization_level": 0.0})
    c2 = env.sim.episode_config.object_poses
    assert c1 == c2
    env.reset(seed=1, options={"randomization_level": 1.0})


def test_object_spawning_rests_on_table(env):
    env.reset(seed=3)
    for o in env.objects:
        p = env.sim.obj_pos(o)
        assert 0.80 < p[2] < 0.86, f"{o} not resting on the table: {p}"
        assert not env.sim.is_dropped(o)


def test_action_validation_and_clipping(env):
    env.reset(seed=1)
    obs, r, term, trunc, info = env.step(np.array([5.0, -5.0, 5.0, 0, 0, 5.0, 5.0]))  # out of range -> clipped
    assert np.isfinite(r) and not term
    ee = env.sim.ee_pos()
    assert np.all(ee >= env.sim.ws_lo - 0.06) and np.all(ee <= env.sim.ws_hi + 0.06)


def test_joint_limits_and_velocity_limits(sim):
    sim.reset(seed=1)
    prev = sim.q_target.copy()
    for _ in range(30):
        sim.apply_action(np.array([1, 1, -1, 0, 0, 1, 0]))
        q = sim.q_target
        assert np.all(q >= sim.joint_lo - 1e-9) and np.all(q <= sim.joint_hi + 1e-9)
        dq = np.abs(q - prev)
        assert np.all(dq <= sim.vel_limits * sim.control_dt + 1e-6), f"velocity limit exceeded {dq}"
        prev = q.copy()
    assert sim.is_stable()


def test_workspace_bounds(sim):
    sim.reset(seed=1)
    for _ in range(40):
        sim.apply_action(np.array([0, 0, -1, 0, 0, 0, -1]))
    assert sim.ee_pos()[2] >= sim.ws_lo[2] - 0.01
    assert sim.is_stable()


def test_gripper_opens_and_closes(sim):
    sim.reset(seed=1)
    sim.set_gripper(1.0); sim.step_physics(300)
    closed = sim.gripper_opening()
    sim.set_gripper(0.0); sim.step_physics(300)
    opened = sim.gripper_opening()
    assert closed < 0.3 and opened > 0.95


def test_bin_detection(sim):
    sim.reset(seed=1)
    c = sim.bin_pos("bin_a")
    adr = sim.obj_qpos_adr["bracket"]
    sim.data.qpos[adr:adr + 3] = [c[0], c[1], c[2] - 0.03]
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.in_bin("bracket", "bin_a") and not sim.in_bin("bracket", "bin_b")
    assert sim.which_bin("bracket") == "bin_a"
    assert sim.which_bin("bolt") is None


def _scripted_grasp(sim, obj):
    sim.track_target = obj
    p = sim.obj_pos(obj); yaw = wrap_angle(sim.obj_yaw(obj))
    sim.set_gripper(0.0)
    sim.move_to(p + [0, 0, 0.2], yaw, max_steps=60)
    sim.move_to([p[0], p[1], sim.ws_lo[2]], yaw, max_steps=40, tol=0.004)
    for _ in range(8):
        sim.apply_action(np.array([0, 0, 0, 0, 0, 0, 1.0]))


def test_grasp_detection_uses_real_contacts(sim):
    sim.reset(seed=0)
    assert not sim.is_grasped("bracket")
    _scripted_grasp(sim, "bracket")
    cs = sim.contacts("bracket")
    assert cs.left_pad_target and cs.right_pad_target
    assert sim.is_grasped("bracket", cs)
    for _ in range(12):
        sim.apply_action(np.array([0, 0, 0.6, 0, 0, 0, 1.0]))
    assert sim.is_lifted("bracket"), "object should be physically lifted by friction contacts"


def test_placement_detection_and_reward(env):
    env.reset(seed=0)
    sim = env.sim
    tgt = env.target
    _scripted_grasp(sim, tgt)
    # step through the env so stage flags update
    for _ in range(3):
        obs, r, term, trunc, info = env.step(np.array([0, 0, 0.6, 0, 0, 0, 1.0]))
    assert info["grasped_now"], "env should detect the grasp"
    assert info["reward_components"]["grasp_reward"] > 0
    b = sim.bin_pos(env.bin_mapping[tgt])
    for _ in range(80):
        d = np.array([b[0], b[1], 1.02]) - sim.target_pos
        a = np.zeros(7); a[:3] = np.clip(d / sim.max_dpos, -1, 1); a[6] = 1.0
        obs, r, term, trunc, info = env.step(a)
        if np.linalg.norm(d[:2]) < 0.01 or term or trunc:
            break
    assert info["lifted_now"]
    total = 0.0
    for _ in range(15):
        obs, r, term, trunc, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0])); total += r
        if info["placed_count"] >= 1 or term or trunc:
            break
    assert info["placed_count"] >= 1, "placement into the correct bin must be detected"
    assert total > env.env_cfg["reward"]["placement_reward"] * 0.9


def test_reward_components_are_configured_not_hardcoded(env):
    w = env.env_cfg["reward"]
    for k in ("approach_reward", "grasp_reward", "lift_reward", "transport_reward", "placement_reward",
              "collision_penalty", "wrong_bin_penalty", "control_effort_penalty", "timeout_penalty"):
        assert k in w
    assert env.reward_fn.w["placement_reward"] == w["placement_reward"]


def test_timeout_truncates(env):
    env.reset(seed=2)
    for i in range(env.max_steps + 2):
        obs, r, term, trunc, info = env.step(np.zeros(7))
        if term or trunc:
            break
    assert trunc and not term and info["step"] == env.max_steps


def test_sb3_alias_wrapper():
    e = make_env(sb3=True)
    obs, _ = e.reset(seed=0)
    assert set(obs.keys()) == {"image", "state"}
    e.close()
