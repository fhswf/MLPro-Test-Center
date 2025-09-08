from mlpro.rl.models import *
import gymnasium as gym
import numpy as np
      

class UR5_Mujoco(Environment):


## -------------------------------------------------------------------------------------------------
    def __init__(
            self,
            p_allowance_eef_pose=0.05, #meter
            p_allowance_eef_orient=0.05, #meter
            p_target_orientation=False,
            p_env_boundaries=True,
            p_velocity_max=0.5,
            p_logging=Log.C_LOG_ALL
            ):
        
        self._ori_env               = gym.make("sim:UR5Env-v0", render_mode="human")
        self._allowance_eef_pose    = p_allowance_eef_pose
        self._allowance_eef_orient  = p_allowance_eef_orient
        self._velocity_max          = p_velocity_max
        
        self._target_orientation    = p_target_orientation
        if self._target_orientation:
            self._target_eef        = [0, -0.33, 0.25, 0, 1, 0, 0]
        else:
            self._target_eef        = [0, -0.33, 0.25]
        
        self._env_boundaries        = p_env_boundaries
        if self._env_boundaries:
            # self._envb              = [[-0.25, 0.45], [0.17, -0.60], [0, 1]] #[x, y, z] in meters
            self._envb              = [[-0.25, 0.15], [-0.20, -0.75], [0.20, 0.85]] #[x, y, z] in meters
        
        super().__init__(p_logging=p_logging)
        
        self._state_space, self._action_space = self._setup_spaces()
        self.reset()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        
        return None, None

        
## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):

        state_space     = ESpace()
        action_space    = ESpace()

        for x in range(6):
            state_space.add_dim(
                Dimension(
                    p_name_short='JA_'+str(x+1),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='Joint Angle '+str(x+1),
                    p_boundaries=[-np.inf, np.inf]
                    )
                )

        for x in range(6):
            state_space.add_dim(
                Dimension(
                    p_name_short='JV_'+str(x+1),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='Joint Velocity '+str(x+1),
                    p_boundaries=[-np.inf, np.inf]
                    )
                )

        for x in range(3):
            state_space.add_dim(
                Dimension(
                    p_name_short='EEF_Pos_'+str(x+1),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='EEF Position '+str(x+1),
                    p_boundaries=[-np.inf, np.inf]
                    )
                )

        for x in range(4):
            state_space.add_dim(
                Dimension(
                    p_name_short='EEF_Ori_'+str(x+1),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='EEF Orientation '+str(x+1),
                    p_boundaries=[-np.inf, np.inf]
                    )
                )

        for x in range(6):
            state_space.add_dim(
                Dimension(
                    p_name_short='EEF_V_'+str(x+1),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='EEF Velocity '+str(x+1),
                    p_boundaries=[-np.inf, np.inf]
                    )
                )

        for x in range(6):
            action_space.add_dim(
                Dimension(
                    p_name_short='JV_'+str(x+1),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='Joint Velocity '+str(x+1),
                    p_boundaries=[-self._velocity_max,self._velocity_max]
                    )
                )

        return state_space, action_space        
        

## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
            
        reward = Reward()
        reward.set_overall_reward(0)
        
        return reward


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        
        success         = True
        states          = p_state.get_values()
        cur_eef_pose    = states[12:15]
        cur_eef_orient  = states[15:19]
        
        for x in range(3):
            if abs(cur_eef_pose[x]-self._target_eef[x]) > self._allowance_eef_pose:
                success = False
        
        if self._target_orientation:
            for x in range(4):
                if abs(cur_eef_orient[x]-self._target_eef[x+3]) > self._allowance_eef_orient:
                    success = False
            
        if success:
            self._state.set_success(True)
            self._state.set_terminal(True)
            return True
        else:
            self._state.set_success(False)
            return False


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        
        broken          = False
        states          = p_state.get_values()
        cur_eef_pose    = states[12:15]

        if self._env_boundaries:
            for x in range(3):
                if x != 1:
                    if cur_eef_pose[x] < self._envb[x][0] or cur_eef_pose[x] > self._envb[x][1]:
                        broken = True
                else:
                    if cur_eef_pose[x] > self._envb[x][0] or cur_eef_pose[x] < self._envb[x][1]:
                        broken = True
        
        if broken:
            self._state.set_broken(True)
            self._state.set_terminal(True)
            return True
        else:
            self._state.set_broken(False)
            return False


## -------------------------------------------------------------------------------------------------
    def _get_init_obs(self) -> bool:

        return np.concatenate(
            [
                self._ori_env.data.qpos.flat[:6],  # 0 Joint Angle
                self._ori_env.data.qvel.flat[:6],  # 6 Joint Velocity
                self._ori_env.data.body("eef").xpos,  # 12 EEF Position
                self._ori_env.data.body("eef").xquat,  # 15 EEF Orientation
                self._ori_env.data.body("eef").cvel, # 19 EEF Speed
            ]
        )
            
    
## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        
        # try:
        #     self._ori_env.close()
        # except:
        #     pass
        
        self._ori_env.reset()
        
        obs = self._get_init_obs()
        self._state = State(self._state_space)
        for i in range(len(obs)):
            self._state.set_value(self._state.get_dim_ids()[i], obs[i])
        

## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action) -> State:
        
        action = p_action.get_sorted_values()
        obs, _, _, _, _ = self._ori_env.step(action)
        
        self._state = State(self._state_space)

        for i in range(len(obs)):
            self._state.set_value(self._state.get_dim_ids()[i], obs[i])
        
        return self._state
      