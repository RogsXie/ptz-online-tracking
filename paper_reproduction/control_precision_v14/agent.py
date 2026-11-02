"""Learn action responses from RGB transitions; plan with the unchanged joint reward.

No action-direction table, camera/FOV/GT input or manually selected control action.
"""
from contextlib import contextmanager
import hashlib,pickle,json
from pathlib import Path
import numpy as np
from control_v6.temporal import discount,discounted_seconds
from .scenes import load_protocol

def features(s):
    s=np.asarray(s,dtype=np.float64);x,y,z=np.moveaxis(s,-1,0)
    return np.stack([np.ones_like(x),x,y,z,x*x,y*y,z*z,x*y,x*z,y*z],axis=-1)

def observation(row):
    if not row['visible']:return None
    return np.array([row['x'],row['y'],np.log(row['area_ratio']/.04)],dtype=np.float64)

def predicted_reward(before,after,actions):
    center=np.linalg.norm(after[:,:2],axis=1);area=.04*np.exp(after[:,2])
    cost0=np.maximum(0,np.linalg.norm(before[:,:2],axis=1)-.015)+.2*np.abs(before[:,2]);cost1=np.maximum(0,center-.015)+.2*np.abs(after[:,2])
    good=(center<=.025)&(area>=.025)&(area<=.055)
    rate=np.where(good,1.,-.2)-6*cost1
    return 20*(cost0-discount(.04)*cost1)+rate*discounted_seconds(.04)-.02*(actions!=0)

class ResponseAgent:
    FORMAT='precision_band_rgb_planning_v14'
    def __init__(self):
        self.cfg=load_protocol()['learning'];self.rng=np.random.default_rng(self.cfg['seed'])
        self.transitions=[];self.coefficients=np.zeros((7,10,3),dtype=np.float64)
        self.environment_steps=0;self.update_count=0;self.completed_episodes=0
        self.counts=np.zeros(7,dtype=int);self.ready=False;self.frozen=False;self.last_selection={}
        self.last_fit=None
    def digest(self):
        return hashlib.sha256(self.coefficients.tobytes()).hexdigest()
    def state_digest(self):
        # Canonical contents, independent of pickle memoization/array ownership.
        payload=dict(transitions=self.transitions,rng=self.rng.bit_generator.state,
            steps=self.environment_steps,updates=self.update_count,episodes=self.completed_episodes,
            counts=self.counts.tolist(),ready=self.ready,model=self.digest())
        return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    def fit(self):
        if self.frozen:raise RuntimeError('Fit forbidden during frozen evaluation')
        valid=[t for t in self.transitions if t['state'] is not None and t['next_state'] is not None]
        s=np.asarray([t['state'] for t in valid]);a=np.asarray([t['action'] for t in valid]);y=np.asarray([t['next_state'] for t in valid])-s
        counts=np.bincount(a,minlength=7);self.counts=counts
        if np.min(counts)<self.cfg['min_samples_per_action']:return None
        f=features(s);reg=np.eye(f.shape[1])*self.cfg['ridge'];reg[0,0]*=.01
        coef=[];errors=[]
        for k in range(7):
            xk=f[a==k];yk=y[a==k]
            ck=np.linalg.solve(xk.T@xk+reg,xk.T@yk)
            coef.append(ck);errors.append(np.mean((xk@ck-yk)**2,axis=0).tolist())
        self.coefficients=np.asarray(coef);self.ready=True;self.update_count+=1
        if not np.isfinite(self.coefficients).all():raise RuntimeError('Nonfinite dynamics model')
        mse=np.mean((np.einsum('ni,nij->nj',f,self.coefficients[a])-y)**2,axis=0)
        self.last_fit=dict(update=self.update_count,real_interactions=self.environment_steps,
            fitted_transitions=len(valid),action_counts=counts.tolist(),mse_xyz=mse.tolist(),
            per_action_mse_xyz=errors,model_sha256=self.digest())
        return self.last_fit
    def learn(self,state,action,next_state,reward,terminated,truncated,seed,decision):
        if self.frozen:raise RuntimeError('Training transition forbidden during evaluation')
        self.transitions.append(dict(state=state.tolist() if state is not None else None,action=int(action),
            next_state=next_state.tolist() if next_state is not None else None,reward=float(reward),
            done=bool(terminated),terminated=bool(terminated),truncated=bool(truncated),seed=seed,decision=decision))
        self.environment_steps+=1
        if self.environment_steps>=self.cfg['warmup'] and self.environment_steps%self.cfg['update_interval']==0:return self.fit()
        return None
    def predict(self,states,actions):
        return np.asarray(states)+np.einsum('ni,nij->nj',features(states),self.coefficients[np.asarray(actions)])
    def plan(self,state):
        if not self.ready:raise RuntimeError('No fitted response model')
        states=np.asarray(state).reshape(1,3);scores=np.zeros(1);first=np.array([-1]);factor=discount(.04)
        diagnostics=None
        for depth in range(self.cfg['horizon']):
            parent=np.repeat(states,7,axis=0);actions=np.tile(np.arange(7),len(states))
            ns=self.predict(parent,actions)
            if not np.isfinite(ns).all() or np.max(np.abs(ns))>100:raise RuntimeError('Nonfinite or extreme model extrapolation')
            values=np.repeat(scores,7)+factor**depth*predicted_reward(parent,ns,actions)
            # This is an image-space prediction-validity penalty, not an action rule.
            outside=(np.abs(ns[:,:2])>1).any(axis=1)|(ns[:,2]>np.log(1/.04))|(ns[:,2]<-12)
            values[outside]=-1e6
            roots=actions if depth==0 else np.repeat(first,7)
            if depth==0:diagnostics=dict(predicted_next=ns.tolist(),one_step_rewards=values.tolist())
            order=np.argsort(-values,kind='stable')[:self.cfg['beam_width']]
            states,scores,first=ns[order],values[order],roots[order]
        best=int(np.argmax(scores));return int(first[best]),dict(**diagnostics,planned_return=float(scores[best]),horizon=self.cfg['horizon'])
    def select(self,state,training):
        if state is None:raise RuntimeError('Missing RGB observation; no fabricated state or expert recovery')
        random=bool(training and (not self.ready or self.rng.random()<self.cfg['epsilon']))
        if random:action=int(self.rng.integers(7));details={}
        else:action,details=self.plan(state)
        self.last_selection=dict(random=random,model_ready=self.ready,action=action,**details)
        return action
    def save(self,path,metadata):
        p=Path(path);tmp=p.with_suffix('.tmp')
        tmp.write_bytes(pickle.dumps(dict(format=self.FORMAT,metadata=metadata,agent=self.__dict__),protocol=5));tmp.replace(p)
    @classmethod
    def load(cls,path):
        payload=pickle.loads(Path(path).read_bytes());assert payload['format']==cls.FORMAT
        self=cls();self.__dict__.update(payload['agent']);return self,payload['metadata']

@contextmanager
def frozen(agent,receipt):
    if agent.frozen:raise RuntimeError('Already frozen')
    before=agent.state_digest();model=agent.digest();counts=(agent.environment_steps,agent.update_count,len(agent.transitions))
    receipt.update(kind='actual_response_model_freeze_v14',before=before,model_before=model,
        counters_before=counts,neural_networks='not_applicable',learning_disabled=True)
    agent.frozen=True;agent.coefficients.flags.writeable=False
    try:yield
    finally:
        after=agent.state_digest();receipt.update(after=after,model_after=agent.digest(),
            counters_after=(agent.environment_steps,agent.update_count,len(agent.transitions)),
            model_data_rng_counters_unchanged=before==after,model_unchanged=model==agent.digest())
        agent.frozen=False;agent.coefficients.flags.writeable=True
        if before!=after:raise RuntimeError('Frozen development changed learning data/model/RNG/counters')
