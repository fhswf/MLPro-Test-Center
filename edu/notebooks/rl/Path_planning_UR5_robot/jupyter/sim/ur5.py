import os
import numpy as np
import mujoco

from gymnasium.envs.mujoco import mujoco_env
from gymnasium import utils
from gymnasium.spaces import Box


class UR5Env(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        model_path = os.path.join(os.path.dirname(__file__), "assets", "ur5.xml")
        observation_space = Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(self, model_path, 1, observation_space, **kwargs)

        # self.cam2 = self.setup_camera("test")

    def step(self, action):
        terminated = False
        info = {}
        reward = 0

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Do the action
        # print("Sending Action", action)
        self.do_simulation(action, self.frame_skip)

        # Get observation
        obs = self.get_obs()

        if self.render_mode == "human":
            self.render()

        reward = 0

        return obs, reward, terminated, False, info

    def setup_camera(self, camera_name):
        cam = mujoco.MjvCamera()
        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = camera_id

        return cam

    def get_camera_data(self, cam):
        rgb_arr = np.zeros(
            3 * self.mujoco_renderer.viewer.viewport.width * self.mujoco_renderer.viewer.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            self.mujoco_renderer.viewer.viewport.width * self.mujoco_renderer.viewer.viewport.height, dtype=np.float32
        )

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.mujoco_renderer.viewer.vopt,
            self.mujoco_renderer.viewer.pert,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.mujoco_renderer.viewer.scn,
        )

        mujoco.mjr_render(
            self.mujoco_renderer.viewer.viewport, self.mujoco_renderer.viewer.scn, self.mujoco_renderer.viewer.con
        )

        # Read Pixel
        mujoco.mjr_readPixels(rgb_arr, depth_arr, self.mujoco_renderer.viewer.viewport, self.mujoco_renderer.viewer.con)
        rgb_img = rgb_arr.reshape(self.mujoco_renderer.viewer.viewport.height, self.mujoco_renderer.viewer.viewport.width, 3)
        return rgb_img[::-1, :, :]

    def reset_model(self):
        # Init qpos and qvel
        qpos = np.array([1.54, -1.54, 1.54, -1.54, -1.54, 0.0])
        qvel = np.zeros(shape=(self.model.nv,))

        # Set position and velocity robot to 0
        self.set_state(qpos, qvel)

        return self.get_obs()
    
    def get_sample_data(self):
        return {
            "joint_angle": self.data.qpos.flat[:6],
            "joint_velocity": self.data.qvel.flat[:6],
            "eef_position": self.data.body("eef").xpos,
            "eef_orientation": self.data.body("eef").xquat,
            "eef_speed": self.data.body("eef").cvel
        }

    def get_obs(self):
        d = self.unwrapped.data
        collide = 0
        if d.ncon > 0:
            collide = 1

        return np.concatenate(
            [
                self.data.qpos.flat[:6],  # 0 Joint Angle
                self.data.qvel.flat[:6],  # 6 Joint Velocity
                self.data.body("eef").xpos,  # 12 EEF Position
                self.data.body("eef").xquat,  # 15 EEF Orientation
                self.data.body("eef").cvel, # 19 EEF Speed
            ]
        )
