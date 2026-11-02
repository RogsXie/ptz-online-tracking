"""One declared reset-sampling curriculum; author learner/controller unchanged."""
import bootstrap
from bootstrap import HERE
import json,random
import train as runner
from scenes import layout as original_layout,SIGNS

DECLARATION=json.loads((HERE/'coverage_support_declaration.json').read_text(encoding='utf-8'))
original_write=runner.write
original_evaluate=runner.evaluate

def coverage_layout(split,index):
    if split!='train':return original_layout(split,index)
    ordinal=index-320
    if not 0<=ordinal<160:raise ValueError('This declared sampling block only covers episodes321..480')
    sequence=ordinal//2
    focused=ordinal%2==0
    source_index=(60000 if focused else 40000)+sequence
    item=original_layout('train',source_index)
    item.update(source_case_index=source_index,index=index,sampling_profile=DECLARATION['profile'],focused_support=focused)
    if focused:
        rng=random.Random(item['seed']);q=sequence%4;mode=sequence//4%3
        sx,sy=SIGNS[q]
        if mode==0:ox,oy=rng.uniform(.30,.38),rng.uniform(.30,.38)
        elif mode==1:ox,oy=rng.uniform(.30,.38),rng.uniform(.04,.25)
        else:ox,oy=rng.uniform(.04,.25),rng.uniform(.30,.38)
        item.update(quadrant=q,layer=4,scale_group=0,fov_deg=rng.uniform(56.,63.),
            planned_center=[.5+sx*ox,.5+sy*oy],support_mode=mode)
    return item

def declared_write(path,value):
    if path.name=='configuration.json':value['sampling_declaration']=DECLARATION
    return original_write(path,value)

def full_development(agent,env,root,completed,count=20,split='development'):
    return original_evaluate(agent,env,root,completed,count=80,split=split)

if __name__=='__main__':
    runner.layout=coverage_layout
    runner.write=declared_write
    runner.evaluate=full_development
    runner.main()
