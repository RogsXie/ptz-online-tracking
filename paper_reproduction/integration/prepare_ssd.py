from pathlib import Path
import torch,time,json
from torchvision.models.detection import ssd300_vgg16,SSD300_VGG16_Weights
root=Path(r'D:\EI\ptz_ue\paper_reproduction')
torch.hub.set_dir(str(root/'weights'/'torch_hub'))
torch.set_num_threads(4)
t=time.monotonic()
m=ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).eval()
x=torch.rand(3,540,960)
with torch.inference_mode():
 y=m([x])[0]
print('loaded SSD300',time.monotonic()-t, 'output',y['boxes'].shape)
