"""Reuse verified actuation and RGB frontend, enforce fixed FOV and center reward."""
from control_zoom_v11.environment import ZoomEnvironment,STATE_FEATURES
from control_v6.temporal import temporal_reward


class CenterEnvironment(ZoomEnvironment):
    def reset(self,scenario,seed):
        state=super().reset(scenario,seed)
        self.fixed_fov=float(self.rows[0]['fov'])
        return state

    def step(self,action,duration):
        if action not in (0,1,2,3,4):raise ValueError('Stage A prohibits zoom actions')
        before=self.rows[-1]
        state,_,done,row=super().step(action,duration)
        if abs(row['fov']-self.fixed_fov)>.0001:raise RuntimeError('Stage A FOV moved')
        reward,parts=temporal_reward(before,row,action,duration)
        row.update(reward=float(reward),reward_parts=parts,objective='center_only_v10')
        return state,float(reward),done,row
