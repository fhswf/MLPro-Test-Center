import time
import glfw
import mujoco
import numpy as np
from threading import Lock
from lxml import etree

class CallbacksViewer():
    """
    All callbacks function for the viewer
    """

    def __init__(self) -> None:
        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._render_every_frame = True
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return
        # Pause
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # Quit
        if key == glfw.KEY_ESCAPE:
            glfw.destroy_window(self.window)
            glfw.terminate()

    def _scroll_callback(self, window, x_offset, y_offset):
            with self._gui_lock:
                mujoco.mjv_moveCamera(
                    self.model, 
                    mujoco.mjtMouse.mjMOUSE_ZOOM, 
                    0, 
                    -0.05 * y_offset, 
                    self.scn, 
                    self.cam)

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        shift_pressed = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right_pressed:
            if shift_pressed:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            if shift_pressed:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, action, dx / height, dy / height, self.scn, self.cam
            )

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)    

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._button_right_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)


class BaseViewer():
    def __init__(self, model, data, width, height) -> None:
        self.model = model
        self.data = data

        self._markers = []
        self._overlays = {}

        self.viewport = mujoco.MjrRect(0, 0, width, height)

        # This goes to specific visualizer
        self.scn = mujoco.MjvScene(self.model, 1000)
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        self._make_context_current()

        # Keep in Mujoco Context
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._set_mujoco_buffer()
        
    def _init_camera(self, xyz_pos=None, elevation=None, distance=None):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        
        # X Y Z Position of the camera
        if xyz_pos is None:
            self.cam.lookat = np.zeros(3)
        else:
            for i in range(3):
                self.cam.lookat[i] = xyz_pos[i]

        # Camera Distance
        if distance is None:
            self.cam.distance = self.model.stat.extent * 3.0
        else:
            self.cam.distance = distance

        if elevation is None:
            self.cam.elevation = -20
        else:
            self.cam.elevation = elevation

    def add_overlay(self, gridpos: int, text1: str, text2: str):
        if gridpos not in self._overlays:
            self._overlays[gridpos] = ["", ""]
        self._overlays[gridpos][0] += text1 + "\n"
        self._overlays[gridpos][1] += text2 + "\n"
            
    def _set_mujoco_buffer(self):
        raise NotImplementedError
    
    def _make_context_current(self):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError


class OffRenderViewer(BaseViewer):
    def __init__(self, model, data, xyz_pos=None, elevation=None, distance=None) -> None:
        width = model.vis.global_.offwidth
        height = model.vis.global_.offheight
        
        self._get_opengl_backend(width, height)
        BaseViewer.__init__(self, model, data, width, height)
        
        self._init_camera(xyz_pos, elevation, distance)
        
    def _set_mujoco_buffer(self):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

    def _make_context_current(self):
        self.opengl_context.make_current()

    def _get_opengl_backend(self, width: int, height: int):
        try:
            from mujoco.glfw import GLContext
            self.opengl_context = GLContext(width, height)
        except:
            raise RuntimeError("Runtime Error OpenGL Context")
        
    def render(self):
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        for marker_params in self._markers:
            self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(self.viewport, self.scn, self.con)

        for gridpos, (text1, text2) in self._overlays.items():
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                gridpos,
                self.viewport,
                text1.encode(),
                text2.encode(),
                self.con,
            )

        rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )

        mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.con)

        rgb_img = rgb_arr.reshape(self.viewport.height, self.viewport.width, 3)
        # original image is upside-down, so flip i
        return rgb_img[::-1, :, :]

    def close(self):
        self.free()
        glfw.terminate()

    def free(self):
        self.opengl_context.free()

    def __del__(self):
        self.free()


class RenderViewer(BaseViewer, CallbacksViewer):
    def __init__(self, model, data, xyz_pos=None, elevation=None, distance=None) -> None:
        # Init GLFW
        glfw.init()

        # Get Width and Height of monitor
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size

        # Create Window
        self.window = glfw.create_window(
            width, height, "MuJoCo in MLPRo Viewer", None, None)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width
        
        BaseViewer.__init__(self, model, data, framebuffer_width, framebuffer_height)
        glfw.swap_interval(1)
        
        # Set Callbacks
        CallbacksViewer.__init__(self)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        
        self._init_camera(xyz_pos, elevation, distance)

    def _create_overlays(self):
        """
        Should be user customizeable
        """
        pass
    
    def _set_mujoco_buffer(self):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)

    def _make_context_current(self):
        glfw.make_context_current(self.window)

    def render(self):
        def update():
            self._create_overlays()

            render_start = time.time()

            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()

            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)

            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)

                # render
                mujoco.mjr_render(self.viewport, self.scn, self.con)

                # overlays
                for gridpos, [t1, t2] in self._overlays.items():
                        mujoco.mjr_overlay(
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            gridpos,
                            self.viewport,
                            t1,
                            t2,
                            self.con,
                        )   

                glfw.swap_buffers(self.window)

            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

            self._overlays.clear()

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

    def close(self):
        self.free()
        glfw.terminate()

    def free(self):
        if self.window:
            if glfw.get_current_context() == self.window:
                glfw.make_context_current(None)
        glfw.destroy_window(self.window)
        self.window = None
        
    def __del__(self):
        self.free()


class MujocoHandler:
    """
    Module provides the functionality of MuJoCo.

    Parameters
    ----------
        p_mujoco_file : str
            String path points the MuJoCo file.
        p_frame_skip : int
            Frame skips for each simulation step.
        p_state_mapping : list
            State mapping for customized state. Defaults to None.
        p_action_mapping : list
            Action mapping for customized action. Defaults to None.
        p_camera_conf : tuple
            Camera configuration (xyz position, elevation, distance). Defaults to (None, None, None).
        p_visualize : bool
            Visualize the MuJoCo Simulation. Defaults to False.
        p_logging : bool
            Logging. Defaults to Log.C_LOG_ALL.
    """

    def __init__(self, 
                p_mujoco_file, 
                p_frame_skip=1,
                p_camera_conf=(None, None, None),
                p_visualize=True,
                ):
        
        self._viewer = None
        self._frame_skip = p_frame_skip
        self._xyz_camera, self._distance_camera, self._elavation_camera = p_camera_conf
        self._model_path = p_mujoco_file
        self._visualize = p_visualize

        self._camera_list = {}

        self._initialize_simulation()

        self.init_qpos = self._data.qpos.ravel().copy()
        self.init_qvel = self._data.qvel.ravel().copy()
        self._init_qpos_space = None
        self._init_qvel_space = None
        
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(self._model_path, parser=parser)
        self._xml_root = tree.getroot()
        etree.strip_tags(self._xml_root, etree.Comment)
        etree.cleanup_namespaces(self._xml_root)

    def __del__(self):
        try:
            self._close()
            self.log(self.C_LOG_TYPE_I, 'Closed')
        except:
            pass

    def setup_spaces(self):
        """
        Setup state and action spaces.

        Returns:
            ESpace, ESpace: State Space and Action Space
        """
        state_space = self._get_state_space()
        action_space = self._get_action_space()
        return state_space, action_space

    def get_init_qpos_space(self):
        """
        Get Initial State Space
        """
        return self._init_qpos_space

    def get_init_qvel_space(self):
        """
        Get Initial State Space
        """
        return self._init_qvel_space
    
    def _get_state_space(self):
        """
        Generate the state space based on MJCF file.

        Returns
        -------
            ESpace: 
                State Space
        """
        return None

    def _get_action_space(self):
        """
        Generate the action space based on MJCF file.

        Returns
        -------
            ESpace: 
                Action Space
        """
        return None

    def _get_obs(self):
        """
        Get the current observation of MuJoCo Simulation.

        Returns
        -------
            list: 
                List of state values.
        """
        return None

    def _reset_model(self, reset_state=None):
        """
        Reset the model simulation. If resete_state is None, then the default state from MuJoCo
        will be taken for the intial value. Otherwise, a customized state can be defined in 
        _reset() function.

        Parameters
        ----------
            reset_state : list 
                qpos data and qvel data. Defaults to None.

        Returns
        -------
            list: 
                List of state values
        """
        if reset_state is None:
            qpos = self.init_qpos
            qvel = self.init_qvel
            self._set_state(qpos, qvel)
        else:
            qpos = reset_state[0].get_values()
            qvel = reset_state[1].get_values()
            self._set_state(qpos, qvel)
            
        return self._get_obs()
  
    def _set_state(self, *args):
        """
        Set the current state of MuJoCo Simulation.
        """
        if len(args) == 2:
            self._data.qpos[:] = np.copy(args[0])
            self._data.qvel[:] = np.copy(args[1])
        
        if len(args) == 3:
            self._data.qpos[:] = np.copy(args[0])
            self._data.qvel[:] = np.copy(args[1])
            self._data.qacc[:] = np.copy(args[2])

        if self._model.na == 0:
            self._data.act[:] = None
        mujoco.mj_forward(self._model, self._data)
   
    def _initialize_simulation(self):
        """
        Initialize Simulation.
        """
        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._model.vis.global_.offwidth = 480
        self._model.vis.global_.offheight = 480
        self._data = mujoco.MjData(self._model)
   
    def _setup_camera(self, camera_name):
        """
        Setup MuJoCo Camera Object.

        Parameters
        ----------
            camera_name : str 
                Camera name in MJCF file.

        Returns
        -------
            mujoco.MjvCamera: 
                MuJoCo Camera Object.
        """
        cam = mujoco.MjvCamera()
        camera_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = camera_id

        return cam
 
    def _get_camera_data(self, cam):
        """
        Get camera data from MuJoCo camera.

        Parameters
        ----------
            cam : mujoco.MjvCamera
                MuJoCo Camera Object.

        Returns
        -------
            ndarray: 
                RGB Image or Depth Image.
        """
        if self._viewer is None:
            self._get_viewer()
        
        cam_viewport = mujoco.MjrRect(0, 0, 1024, 768)
        rgb_arr = np.zeros(
            3 * cam_viewport.width * cam_viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            cam_viewport.width * cam_viewport.height, dtype=np.float32
        )

        mujoco.mjv_updateScene(
            self._model,
            self._data,
            self._viewer.vopt,
            self._viewer.pert,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._viewer.scn,
        )

        mujoco.mjr_render(
            cam_viewport, self._viewer.scn, self._viewer.con
        )

        # Read Pixel
        mujoco.mjr_readPixels(rgb_arr, depth_arr, cam_viewport, self._viewer.con)
        rgb_img = rgb_arr.reshape(cam_viewport.height, cam_viewport.width, 3)
        depth_img = depth_arr.reshape(cam_viewport.height, cam_viewport.width)
        
        return rgb_img[::-1, :, :], depth_img[::-1, :]
   
    def _get_viewer(self):
        """
        Get the MuJoCo viewer.

        Returns
        -------
            BaseViewer:
                MuJoCo Viewer
        """
        if self._viewer is None:
            if self._visualize:
                self._viewer = RenderViewer(self._model, self._data, self._xyz_camera, self._distance_camera, self._elavation_camera)
            else:
                self._viewer = OffRenderViewer(self._model, self._data, self._xyz_camera, self._distance_camera, self._elavation_camera)
        
        return self._viewer

    def _reset_simulation(self, reset_state=None):
        """
        Reset the simulation. If resete_state is None, then the default state from MuJoCo
        will be taken for the intial value. Otherwise, a customized state can be defined in 
        _reset() function.

        Parameters
        ----------
            reset_state : list
                qpos data and qvel data. Defaults to None.

        Returns
        -------
            list: 
                List of state values
        """
        mujoco.mj_resetData(self._model, self._data)
        ob = self._reset_model(reset_state)
        return ob

    def _step_simulation(self, p_action):
        """
        Pass the action to the simulation.

        Parameters
        ----------
            p_action : Action
                Action
        """
        self._data.ctrl[:] = p_action
        mujoco.mj_step(self._model, self._data, nstep=self._frame_skip)
        mujoco.mj_rnePostConstraint(self._model, self._data)
        if self._visualize:
            self.render()

    def render(self):
        """
        Render the MuJoCo Viewer.
        """
        self._get_viewer().render()

    def get_latency(self):
        """
        Get latency from MJCF file.

        Returns
        -------
            float: 
                None or Timestep
        """
        for option_elem in self._xml_root.iter("option"):
            try:
                timestep = option_elem.attrib["timestep"]
                return float(timestep)
            except KeyError as e:
                return None

    def _close(self):
        """
        Close MuJoCo Viewer
        """
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
