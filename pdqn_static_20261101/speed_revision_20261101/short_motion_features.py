"""Observation-only shorter RGB bearing fit; unchanged neural motor policy."""
import speed_bootstrap
from motion_learning15 import Features

class ShortFeatures(Features):
    def __init__(self,window):super().__init__();self.window=window
    def extend(self,state,row,diag):
        now=float(diag['sim_time'])
        while self.samples and now-self.samples[0][0]>self.window:self.samples.popleft()
        result,row=super().extend(state,row,diag)
        row['motion_estimator_window_s']=self.window
        return result,row
