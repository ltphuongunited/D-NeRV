from torchvision import models
import torch
from model import ConvNeXtVideo
import torch.nn as nn
from backbone.convnext import ConvNeXt
from backbone.convnextv2 import ConvNeXtV2
# from models.raftcore.raft_nerv import raft_ready
from models.utils import *
# from models.CURE import Fuser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = torch.rand(1,3,256,320).to(device)
model = ConvNeXtV2(depths=[3, 3, 9, 3,3],dims=[64, 64, 64, 64, 64]).to(device)
output = model(image)
for i in output:
    print(i.shape)
# OF = raft_ready().to(device)
# orig_ckt = torch.load('models/raft-things.pth')
# if 'module' in list(orig_ckt.keys())[0] and not hasattr(OF, 'module'):
#     new_ckt={k.replace('module.',''):v for k,v in orig_ckt.items()}
#     OF.load_state_dict(new_ckt)

# image1 = torch.randn(2, 3, 256, 320).to(device)
# image2 = torch.randn(2, 3, 256, 320).to(device)
# cell = 3
# fuse = Fuser()
# # Measure GPU memory usage before running the model
# mem_before = torch.cuda.memory_allocated(device)
# # Run the modelH

# B, C, H, W = image1.shape
# T = 8
# padder = InputPadder(image1.shape)
# image1, image2 = padder.pad(image1, image2)
# # print(image1.shape, image2.shape)
# _, flow21 = OF(image1, image2, iters=20, test_mode=True)
# _, flow12 = OF(image2, image1, iters=20, test_mode=True)
# flow21, flow12 = padder.unpad(flow21), padder.unpad(flow12)
# # print(flow21.shape, flow12.shape)

# flow12 = torch.cat([flow12[:, 0: 1, :, :] / ((flow12.shape[3] - 1.0) / 2.0),
#                             flow12[:, 1: 2, :, :] / ((flow12.shape[2] - 1.0) / 2.0)], 1)
# flow21 = torch.cat([flow21[:, 0: 1, :, :] / ((flow21.shape[3] - 1.0) / 2.0),
#                             flow21[:, 1: 2, :, :] / ((flow21.shape[2] - 1.0) / 2.0)], 1)
# # print(flow12.shape, flow21.shape)
# # >> torch.Size([2, 2, 256, 320]) torch.Size([2, 2, 256, 320])

# flowf1t, flowf2t, flowb1t, flowb2t = generate_double_flow_time_t(flow12, flow21, time_stamp=torch.randn(2, 8).to(device))
# # flowf1t, flowf2t, flowb1t, flowb2t = generate_double_flow_time(flow12, flow21, time_stamp=0.5)
# # print(flowf1t.shape, flowb1t.shape)
# # >> torch.Size([2, 2, 8, 256, 320]) torch.Size([2, 2, 8, 256, 320])

# coorMap = generate_coorMap((H, W), scale=True, flatten=False).type_as(image1)
# # print(coorMap.shape) >>  torch.Size([256, 320, 2])


# # coorMapf1_t, coorMapb1_t, coorMapf2_t, coorMapb2_t = generate_double_coormap(flowf1t, flowf2t, flowb1t, flowb2t, coorMap)
# coorMapf1_t, coorMapb1_t, coorMapf2_t, coorMapb2_t = generate_double_coormap_t(flowf1t, flowf2t, flowb1t, flowb2t, coorMap)

# # print(coorMapf1_t.shape)
# # print(image1.shape)
# image1 = image1.unsqueeze(2).expand(-1, -1, T, -1, -1).contiguous().view(B*T, -1, H, W)
# # image1 = image1.unsqueeze(2).expand(-1, T, -1, -1, -1).contiguous().view(B*T, -1, H, W)
# # print(image1.shape)
# fef1 = nn.functional.grid_sample(image1, coorMapf1_t, mode='bilinear', padding_mode='reflection')
# print(fef1.shape)
# # feb1 = nn.functional.grid_sample(image1, coorMapb1_t, mode='bilinear', padding_mode='reflection')
# # fef2 = nn.functional.grid_sample(image2, coorMapf2_t, mode='bilinear', padding_mode='reflection')
# # feb2 = nn.functional.grid_sample(image2, coorMapb2_t, mode='bilinear', padding_mode='reflection')

# # print(feb1.shape, feb1.shape, fef2.shape, feb2.shape)

# # Measure GPU memory usage after running the model
# mem_after = torch.cuda.memory_allocated(device)

# # Calculate the difference in GPU memory usage
# mem_diff = mem_after - mem_before

# # Convert bytes to MB
# mem_diff_mb = mem_diff / (1024 * 1024)

# print(f'GPU memory usage: {mem_diff_mb:.2f} MB')



