"""Declared image/history-only training objective; no direction teacher."""
import speed_bootstrap
import math
from speed_environment_v3 import Environment as BaseEnvironment,image_cost,hold_assessment

def objective(before,row,rows,previous_axis):
    action=row['action'];reversal=False
    if action:
        axis=0 if action in (1,2) else 1 if action in (3,4) else 2
        reversal=axis in previous_axis and previous_axis[axis]!=action;previous_axis[axis]=action
    cost=image_cost(row);inner=bool(row['visible'] and row['center_error']<=.012 and abs(math.log(row['area']/.04))<=.18)
    reward=20*(image_cost(before)-cost)-2*cost+.75*inner-.02*(action!=0)-.5*reversal
    hold=hold_assessment(rows)
    strict=bool(hold['success'] and hold['hold_center_mean']<=.015 and hold['hold_center_p95']<=.03)
    truncated=bool(row['truncated']);done=strict or truncated
    reward+=10. if strict and not truncated else -10. if truncated else 0.
    return float(reward),bool(done),dict(success=bool(strict and not truncated),training_axis_reversal=bool(reversal),training_hold_margin=.015)

class StabilityEnvironment(BaseEnvironment):
    def reset(self,*args,**kwargs):
        self.training_previous_axis={}
        return super().reset(*args,**kwargs)

    def step(self,action,raw_parameter):
        before=self.previous.copy();state,_,_,row=super().step(action,raw_parameter)
        reward,done,detail=objective(before,row,self.rows,self.training_previous_axis)
        row.update(reward=reward,done=done,**detail)
        return state,reward,done,row
