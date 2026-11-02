"""Wide camera limits, full-image state and current-RGB temporal objective."""
import math
import numpy as np
from control_learning.environment import ControlEnvironment
from control_v5.environment import ContinuingState,transition_terminal
from control_v6.temporal import temporal_reward
from control_coverage_v8.scenes import load_protocol

STATE_FEATURES=['4*normalized_x_no_half_frame_clipping','4*normalized_y_no_half_frame_clipping','log(area/.04)/4','confidence_or0','reserved0','reserved0','reserved0','reserved0']


class CoverageState(ContinuingState):
    def build(self,detection,action=(0,0.)):
        state=super().build(detection,action)
        state[:2]=4*self.last_seen[:2]
        return state


class CoverageEnvironment(ControlEnvironment):
    def __init__(self,*args,legacy_observation=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.legacy_observation=legacy_observation
        self.camera_contract=load_protocol()['control_execution']

    def reset(self,scenario,seed):
        super().reset(scenario,seed)
        self.fixed_location=tuple(self.transport.diagnostics['camera_location'])
        self.fixed_fov=float(self.transport.diagnostics['fov'])
        if not 41.25-1e-4 <= self.fixed_fov <= 66.+1e-4:
            raise RuntimeError('Initial FOV outside declared coverage')
        self.transport.mode('COVERAGE_STATIC_PDQN_V8')
        self.builder=(ContinuingState if self.legacy_observation else CoverageState)(self.spec['horizon_seconds'])
        return self.builder.build(self.selected)

    def step(self,action,duration):
        before=self.rows[-1]
        state,_,done,row=super().step(action,duration)
        elapsed=row['sim_time_s']-before['sim_time_s']
        expected=list(before['rotation'])
        pmin,pmax=self.camera_contract['pitch_limits_deg'];ymin,ymax=self.camera_contract['yaw_limits_deg']
        if action in (1,2):expected[0]=min(pmax,max(pmin,expected[0]+(1 if action==2 else -1)*8*duration))
        if action in (3,4):expected[1]=min(ymax,max(ymin,expected[1]+(1 if action==3 else -1)*8*duration))
        error=max(abs(a-b) for a,b in zip(expected,row['rotation']))
        reward,parts=temporal_reward(before,row,action,duration)
        if error>1e-3 or abs(elapsed-duration)>1e-5 or abs(row['fov']-self.fixed_fov)>1e-4 or not math.isfinite(reward):
            raise RuntimeError('Wide-limit action/time/FOV/finite-value contract failed')
        if tuple(self.transport.diagnostics['camera_location'])!=self.fixed_location or not pmin-1e-4<=row['rotation'][0]<=pmax+1e-4:
            raise RuntimeError('Fixed camera installation or downward pitch contract failed')
        unchanged=max(abs(a-b) for a,b in zip(before['rotation'],row['rotation']))<1e-5
        row.update(reward=float(reward),reward_parts=parts,actuator_error_deg=error,
            saturated_nonstay=bool(action!=0 and unchanged),terminated=False,truncated=bool(done),
            bootstrap_from_actual_next_observation=True)
        return state,float(reward),done,row
