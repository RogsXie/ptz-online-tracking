"""Fresh P-DQN profile; RGB/own-camera inputs, corrected duration and motion state.

No legacy policy/replay is read by create_agent. Original author's four-network
architecture and update equations remain unchanged; extensions are explicit.
"""
import speed_bootstrap
import copy, math, random, types
import gymnasium as gym
import numpy as np
import torch
from author_aligned.source import load_core
from author_aligned.run import NETWORKS, OPTIMIZERS
from unified_perception import bearing, project
import unified_core as core

PROFILE = 'precision15_fresh_rgb_kalman_duration_v2'
LOW = (.3, .25, .25, .225, .225, .3, .3)
HIGH = (8., 25., 25., 22.5, 22.5, 8., 8.)
SECONDS = core.SECONDS
SEED = 20261102
STATE_FIELDS = ['predicted_x', 'predicted_y', 'actual_area_percent', 'confidence',
    'last_executed_action_div6', 'last_executed_seconds_div_point1',
    'observed_delta_x', 'observed_delta_y', 'rgb_kf_yaw_rate_div10',
    'rgb_kf_pitch_rate_div10', 'own_fov_div66', 'image_age_div_point1',
    'own_yaw_delta_div10', 'own_pitch_delta_div10', 'own_fov_delta_div10']


def create_agent(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cls, _, _ = load_core()
    space = gym.spaces.Tuple((gym.spaces.Discrete(7), *[
        gym.spaces.Box(np.array([lo], np.float32), np.array([hi], np.float32))
        for lo, hi in zip(LOW, HIGH)]))
    a = cls(gym.spaces.Discrete(15), space, batch_size=128, gamma=.9, seed=seed,
        actor_lr=3e-4, param_net_lr=1e-4,
        actor_kwargs={'hidden_layers': [128,256,128,64]},
        param_net_kwargs={'hidden_layers': [128,256,128,64]}, device='cuda')
    # Declared physical-duration initialization, not pretrained weights.
    with torch.no_grad():
        torch.nn.init.normal_(a.param_net.layers[-1].weight, std=.005)
        a.param_net.layers[-1].bias.copy_(torch.tensor([1.,2.,2.,2.,2.,1.5,1.5], device=a.device))
    a.param_net_target.load_state_dict(a.param_net.state_dict())
    original = a.choose_action
    def choose(self, state, train=True):
        action, selected, params = original(state, train=train)
        params = np.clip(params, LOW, HIGH).astype(np.float32)
        if hasattr(self, 'camera_diagnostics'):
            valid = core.valid_actions(self.camera_diagnostics, self.camera_limits)
            if not valid[int(action)]:
                with torch.inference_mode():
                    q = self.actor_net(torch.as_tensor(state, device=self.device).unsqueeze(0),
                        torch.as_tensor(params, device=self.device).unsqueeze(0))[0].cpu().numpy()
                q[~valid] = -np.inf; action = int(q.argmax())
        return int(action), np.array([params[int(action)]], np.float32), params
    a.choose_action = types.MethodType(choose, a)
    a.precision_motion_budget=True
    assert len(a.memory) == 0 and a.update_count == 0
    assert all(not getattr(a,n).state for n in OPTIMIZERS)
    return a


def load_for_evaluation(path):
    saved = torch.load(path, map_location='cuda', weights_only=False)
    if saved['metadata']['motion_profile'] != PROFILE:
        raise ValueError('Incompatible precision observation profile')
    a = create_agent()
    for name in NETWORKS:
        getattr(a, name).load_state_dict(saved[name])
    a.precision_control_v3=saved['metadata'].get('camera_control_profile')=='precision_v3_zoom_lookahead_edge_priority'
    a.epsilon = 0.
    return a


class StateBuilder:
    def __init__(self):
        self.previous = None
        self.generation = None

    def build(self, row, diag, association, action=0, duration=0., age=0.):
        generation = association.get('target_generation', 0)
        if generation != self.generation:
            self.previous = None; self.generation = generation
        tracked_velocity = association.get('locked_bearing_velocity_deg_s')
        velocity = np.clip(tracked_velocity if tracked_velocity is not None else [0.,0.], -30.,30.)
        age = float(np.clip(age, 0., .3))
        camera = np.array([diag['rotation'][1],diag['rotation'][0],diag['fov']], float)
        delta_camera = camera-self.previous[2] if self.previous is not None else np.zeros(3)
        if row['visible']:
            xy = np.array([row['x'],row['y']], float)
            delta = xy-self.previous[1] if self.previous is not None and self.previous[1] is not None else np.zeros(2)
            angles = bearing((xy+1)/2, diag)
            predicted = np.clip(2*np.asarray(project(angles+velocity*age,diag))-1, -1.5,1.5)
            head = [*predicted, row['area']*100.,row['confidence'],action/6.,duration/.1,*delta]
        else:
            xy = None; head = [-2.,-2.,-1.,0.,action/6.,duration/.1,0.,0.]
        state = np.r_[head, velocity/10.,camera[2]/66.,age/.1,np.clip(delta_camera/10.,-1.,1.)].astype(np.float32)
        assert state.shape == (15,) and np.isfinite(state).all()
        self.previous = (float(diag['sim_time']),xy,camera)
        row.update(precision_observation_profile=PROFILE, image_motion_bearing_rate_deg_s=velocity.tolist(),
            measured_perception_age_s=age, precision_state=state.tolist())
        return state


def observation_class(base):
    class PrecisionObservation(base):
        def reset(self,*args,**kwargs):
            self.precision_builder=StateBuilder()
            return super().reset(*args,**kwargs)
        def observe(self, action=0, duration=0., **kwargs):
            _,row=super().observe(action,duration,**kwargs)
            raw=getattr(self.t,'raw',None); frame=getattr(self.t,'frame',None)
            if raw is not None and frame is not None and getattr(raw,'clock',None) is not None:
                import time
                age=max(0.,time.perf_counter()-raw.clock-frame.wall_s)
            else: age=float(getattr(self,'precision_training_age',0.))
            state=self.precision_builder.build(row,self.t.diagnostics,self.cfff.info,action,duration,age)
            row.update(policy_area_feature=float(state[2]),area_mapping='Actual RGB whole-box area *100; precision profile has no legacy 4%-to-2% remap.')
            return state,row
    return PrecisionObservation
