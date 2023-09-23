import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import pi, sqrt
from utils import WarpKeyframe
from timm.models.layers import trunc_normal_, DropPath
from backbone.convnext import ConvNeXt
from backbone.convnextv2 import ConvNeXtV2, ConvNeXtV2_temp
from models.raftcore.raft_nerv import raft_ready
from models.utils import *

class MLP(nn.Module):
    def __init__(self, in_chan, out_chan, hidden_chan=512, act='GELU', bias=True, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_chan, hidden_chan, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv1d(hidden_chan, out_chan, 1, 1, 0, bias=bias),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self, kernel_size=3, stride=1, stride_list=[], bias=True):
        super().__init__()
        n_resblocks = len(stride_list)

        # define head module
        m_head = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, kernel_size//2, kernel_size//2), bias=bias),
            nn.GELU(),
        )
        m_body = []
        for i in range(n_resblocks):
            m_body.append(nn.Sequential(
                            nn.Conv3d(64, 64, kernel_size=(1, stride_list[i], stride_list[i]), stride=(1, stride_list[i], stride_list[i]), padding=(0, 0, 0), bias=bias),
                            nn.GELU(),
                            )
                        )
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)

    def forward(self, x):
        key_feature_list = [x]
        x = self.head(x)
        for stage in self.body:
           x = stage(x)
           key_feature_list.append(x)
        return key_feature_list[::-1]

class Head(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv3d(in_chan, in_chan // 4, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
                        nn.GELU(),
                        nn.Conv3d(in_chan // 4, out_chan, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
                    )

    def forward(self, x):
        x = self.conv(x)
        return x

class ToStyle(nn.Module):
    def __init__(self, in_chan, out_chan, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_chan, out_chan*2, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)

    def forward(self, x, style):
        B, C, T, H, W = x.shape
        style = self.conv(style)  # style -> [B, 2*C, 1, H, W]
        style = style.view(B, 2, C, -1, H, W)  # [B, 2, C, 1, H, W]
        x = x * (style[:, 0] + 1.) + style[:, 1] # [B, C, T, H, W] -- Function (6)
        return x

class PixelShuffle(nn.Module):
    def __init__(self, scale=(1, 2, 2)):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        B, C, T, H, W = x.size()
        C_out = C // (self.scale[0] * self.scale[1] * self.scale[2])
        x = x.view(B, C_out, self.scale[0], self.scale[1], self.scale[2], T, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(B, C_out, T * self.scale[0], H * self.scale[1], W * self.scale[2])
        return x
    
class DNeRVBlock(nn.Module):
    def __init__(self, kernel=3, bias=True, **kwargs):
        super().__init__()
        in_chan = kwargs['ngf']
        out_chan = kwargs['new_ngf'] * kwargs['stride'] * kwargs['stride']
        dim = kwargs['dim']
        # Spatially-adaptive Fusion
        # self.to_style = ToStyle(64, in_chan, bias=bias)
        self.to_style = ToStyle(dim, in_chan, bias=bias)
        # 3x3 Convolution-> PixelShuffle -> Activation, same as NeRVBlock
        self.conv = nn.Conv3d(in_chan, out_chan, kernel_size=(1, kernel, kernel), stride=(1, 1, 1), padding=(0, kernel//2, kernel//2), bias=bias)     

        self.upsample = PixelShuffle(scale=(1, kwargs['stride'], kwargs['stride']))
        self.act = nn.GELU()
        # Global Temporal MLP module
        self.tfc = nn.Conv2d(kwargs['new_ngf']*kwargs['clip_size'], kwargs['new_ngf']*kwargs['clip_size'], 1, 1, 0, bias=True, groups=kwargs['new_ngf'])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv3d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def forward(self, x, style_appearance):
        x = self.to_style(x, style_appearance)
        x = self.act(self.upsample(self.conv(x)))               # (7)
        # x = self.act(self.upsample(self.conv2(self.conv1(x))))
        B, C, D, H, W = x.shape
        x = x + self.tfc(x.view(B, C*D, H, W)).view(B, C, D, H, W)   # (8)
        return x

class RAFT_Block(nn.Module):
    def __init__(self, kernel=3, bias=True, **kwargs):
        super().__init__()
        in_chan = kwargs['ngf']
        out_chan = kwargs['new_ngf'] * kwargs['stride'] * kwargs['stride']
        # 3x3 Convolution-> PixelShuffle -> Activation, same as NeRVBlock
        self.conv = nn.Conv3d(in_chan, out_chan, kernel_size=(1, kernel, kernel), stride=(1, 1, 1), padding=(0, kernel//2, kernel//2), bias=bias)     
        self.upsample = PixelShuffle(scale=(1, kwargs['stride'], kwargs['stride']))
        self.act = nn.GELU()
        # Global Temporal MLP module
        self.tfc = nn.Conv2d(kwargs['new_ngf']*kwargs['clip_size'], kwargs['new_ngf']*kwargs['clip_size'], 1, 1, 0, bias=True, groups=kwargs['new_ngf'])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv3d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.act(self.upsample(self.conv(x)))               # (7)
        B, C, D, H, W = x.shape
        x = x + self.tfc(x.view(B, C*D, H, W)).view(B, C, D, H, W)   # (8)
        return x
    
class DNeRV(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        ngf = kwargs['fc_dim']
        self.stem = MLP(in_chan=kwargs['embed_length'], out_chan=self.fc_h * self.fc_w * ngf)

        self.stride_list = kwargs['stride_list']
        self.num_stages = len(self.stride_list)

        encoder_dim = 64
        self.encoder = Encoder(stride_list=self.stride_list[::-1])
        self.norm = nn.InstanceNorm3d(ngf + encoder_dim)

        self.decoder_list = nn.ModuleList()
        self.flow_pred_list = nn.ModuleList([Head(ngf + encoder_dim, 4)])

        height = self.fc_h
        width = self.fc_w
        self.wk_list = nn.ModuleList([WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device'])])
        for i, stride in enumerate(self.stride_list):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                new_ngf = max(round(ngf / stride), kwargs['lower_width'])

            self.decoder_list.append(DNeRVBlock(ngf=ngf + encoder_dim if i == 0 else ngf, new_ngf=new_ngf, 
                                                stride=stride, clip_size=kwargs['clip_size'], dim=64))
            self.flow_pred_list.append(Head(new_ngf, 4))
            height = height * stride
            width = width * stride
            self.wk_list.append(WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device']))

            ngf = new_ngf
        
        self.rgb_head_layer = Head(new_ngf + 3, 3)

        self.dataset_mean = torch.tensor(kwargs['dataset_mean']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.dataset_std = torch.tensor(kwargs['dataset_std']).view(1, 3, 1, 1, 1).to(kwargs['device'])

    def forward(self, embed_input, keyframe, backward_distance):
        B, C, D = embed_input.size()
        backward_distance = backward_distance.view(B, 1, -1, 1, 1)
        forward_distance = 1 - backward_distance

        key_feature_list = self.encoder(keyframe) # [B, encoder_dim, 2, H, W]
        output = self.stem(embed_input)  # [B, C*fc_h*fc_w, D]
        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 1, 4, 2, 3)  # [B, C, D, fc_h, fc_w]
        # I^1_t
        content_feature = F.interpolate(key_feature_list[0], scale_factor=(D/2, 1, 1), mode='trilinear') # [B, encoder_dim, D, fc_h, fc_w]
        # output = self.norm(torch.concat([output, content_feature], dim=1))
        # M^1_t
        output = self.norm(torch.cat([output, content_feature], dim=1))         # (1)
        for i in range(self.num_stages + 1):
            # generate flow at the decoder input stage
            flow = self.flow_pred_list[i](output) # [B, 4, D, fc_h, fc_w]
            forward_flow, backward_flow = torch.split(flow, [2, 2], dim=1)      # (2)
            start_key_feature, end_key_feature = torch.split(key_feature_list[i], [1, 1], dim=2)
            # warp the keyframe features with predicted forward and backward flow
            forward_warp = self.wk_list[i](start_key_feature, forward_flow)     # (3)
            backward_warp = self.wk_list[i](end_key_feature, backward_flow)     # (3)
            # distance-aware weighted sum
            fused_warp = forward_warp * forward_distance + backward_warp * backward_distance # (1 - t) * forward_warp + t * backward_warp (4)

            if i < self.num_stages:
                output = self.decoder_list[i](output, fused_warp)
            else:
                output = self.rgb_head_layer(torch.cat([output, fused_warp], dim=1))

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)

class HDNeRV2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        
        # ngf = kwargs['fc_dim']
        ngf = kwargs['enc_dim'][-1]
        dims = kwargs['enc_dim'][::-1]
        self.stride_list = kwargs['stride_list'][::-1]
        self.encoder = ConvNeXtVideo(
                depths=kwargs['enc_block'], dims=kwargs['enc_dim'], type=kwargs['ver']
            )
        self.num_stages = len(self.stride_list)
        # encoder_dim = 64
        # self.norm = nn.InstanceNorm3d(encoder_dim)
        self.norm = nn.InstanceNorm3d(ngf)

        self.decoder_list = nn.ModuleList()
        # self.flow_pred = Head(encoder_dim, 4)
        self.flow_pred = Head(ngf, 4)
        self.wk_list = WarpKeyframe(self.fc_h, self.fc_w, kwargs['clip_size'], device=kwargs['device'])

        for i, stride in enumerate(self.stride_list):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
                self.decoder_list.append(DNeRVBlock(ngf=ngf, new_ngf=new_ngf, 
                                                stride=stride, clip_size=kwargs['clip_size'],dim=dims[i])
                                                )
            else:
                new_ngf = max(round(ngf / stride), kwargs['lower_width'])
                self.decoder_list.append(RAFT_Block(ngf=ngf, new_ngf=new_ngf, stride=stride, clip_size=kwargs['clip_size']))
            
            ngf = new_ngf
        
        self.rgb_head_layer = Head(new_ngf, 3)

        self.dataset_mean = torch.tensor(kwargs['dataset_mean']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.dataset_std = torch.tensor(kwargs['dataset_std']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        
    def forward(self, video, keyframe, backward_distance):
        B, D = backward_distance.size()
        backward_distance = backward_distance.view(B, 1, -1, 1, 1)
        forward_distance = 1 - backward_distance

        output = self.encoder(video)

        # M^1_t
        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 1, 4, 2, 3)  # [B, C, D, fc_h, fc_w]
        output = self.norm(output)      # (1)

        for i in range(self.num_stages + 1):
            if i == 0:
                # generate flow at the decoder input stage
                flow = self.flow_pred(output) # [B, 4, D, fc_h, fc_w]
                forward_flow, backward_flow = torch.split(flow, [2, 2], dim=1)      # (2)
                start_key_feature = output[:, :, 0, :, :].unsqueeze(2)
                end_key_feature = output[:, :, -1, :, :].unsqueeze(2)
                # warp the keyframe features with predicted forward and backward flow
                forward_warp = self.wk_list(start_key_feature, forward_flow)     # (3)
                backward_warp = self.wk_list(end_key_feature, backward_flow)     # (3)
                # distance-aware weighted sum
                print(backward_warp.shape)
                print(backward_distance.shape)
                fused_warp = forward_warp * forward_distance + backward_warp * backward_distance    # (1 - t) * forward_warp + t * backward_warp (4)
                output = self.decoder_list[i](output, fused_warp)
            elif i < self.num_stages:
                output = self.decoder_list[i](output)
            else:
                output = self.rgb_head_layer(output)

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)

class HDNeRV2Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder 
        self.norm = model.norm
        self.fc_h = model.fc_h
        self.fc_w = model.fc_w

    def forward(self, x):
        B, _, D, _, _ = x.size()
        output = self.encoder(x)
        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 1, 4, 2, 3)
        output = self.norm(output)
        return output

class HDNeRV2Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.num_stages = model.num_stages
        self.flow_pred = model.flow_pred
        self.wk_list = model.wk_list
        self.decoder_list = model.decoder_list
        self.rgb_head_layer = model.rgb_head_layer
        self.dataset_std = model.dataset_std
        self.dataset_mean = model.dataset_mean

    def forward(self, output, backward_distance):
        B, D = backward_distance.size()
        backward_distance = backward_distance.view(B, 1, -1, 1, 1)
        forward_distance = 1 - backward_distance
        for i in range(self.num_stages + 1):
            if i == 0:
                # generate flow at the decoder input stage
                flow = self.flow_pred(output) # [B, 4, D, fc_h, fc_w]
                forward_flow, backward_flow = torch.split(flow, [2, 2], dim=1)      # (2)
                start_key_feature = output[:, :, 0, :, :].unsqueeze(2)
                end_key_feature = output[:, :, -1, :, :].unsqueeze(2)
                # warp the keyframe features with predicted forward and backward flow
                forward_warp = self.wk_list(start_key_feature, forward_flow)     # (3)
                backward_warp = self.wk_list(end_key_feature, backward_flow)     # (3)
                fused_warp = forward_warp * forward_distance + backward_warp * backward_distance    # (1 - t) * forward_warp + t * backward_warp (4)
                output = self.decoder_list[i](output, fused_warp)
            elif i < self.num_stages:
                output = self.decoder_list[i](output)
            else:
                output = self.rgb_head_layer(output)

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)

class HDNeRV3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        
        ngf = kwargs['enc_dim'][-1]
        dims = kwargs['enc_dim'][::-1]
        self.stride_list = kwargs['stride_list'][::-1]
        self.encoder_frame = ConvNeXtVideo(
                depths=kwargs['enc_block'], dims=kwargs['enc_dim'], type=kwargs['ver']
            )
        self.encoder_keyframe = ConvNeXtVideo_temp(
                depths=kwargs['enc_block'], dims=kwargs['enc_dim'], type=kwargs['ver']
            )
        self.num_stages = len(self.stride_list)

        self.norm = nn.InstanceNorm3d(ngf)

        self.decoder_list = nn.ModuleList()
        self.flow_pred_list = nn.ModuleList([Head(ngf, 4)])

        height = self.fc_h
        width = self.fc_w
        self.wk_list = nn.ModuleList([WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device'])])
        for i, stride in enumerate(self.stride_list):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                new_ngf = max(round(ngf / stride), kwargs['lower_width'])

            self.decoder_list.append(DNeRVBlock(ngf=ngf, new_ngf=new_ngf, 
                                                stride=stride, clip_size=kwargs['clip_size'],dim=dims[i])
                                                )
            self.flow_pred_list.append(Head(new_ngf, 4))
            height = height * stride
            width = width * stride
            self.wk_list.append(WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device']))

            ngf = new_ngf
        
        self.rgb_head_layer = Head(new_ngf + 3, 3)

        self.dataset_mean = torch.tensor(kwargs['dataset_mean']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.dataset_std = torch.tensor(kwargs['dataset_std']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        
    def forward(self, video, keyframe, backward_distance):
        B, D = backward_distance.size()
        backward_distance = backward_distance.view(B, 1, -1, 1, 1)
        forward_distance = 1 - backward_distance

        frame_feature = self.encoder_frame(video[:,:,1:-1,:,:])
        key_feature_list = self.encoder_keyframe(keyframe)
        # key_feature_list = self.encoder_keyframe(video[:,:,::video.size(2)-1,:,:])
        # print(frame_feature.shape)
        frame_feature = frame_feature.view(B, -1, self.fc_h, self.fc_w, D - 2).permute(0, 1, 4, 2, 3)  # [B, C, D, fc_h, fc_w]

        start = key_feature_list[0][:, :, 0, :, :].unsqueeze(2)
        end = key_feature_list[0][:, :, -1, :, :].unsqueeze(2)
        output = self.norm(torch.cat([start, frame_feature, end], dim=2))
        # M^1_t
        # output = self.norm(key_feature_list[0])      # (1)
        # print(output.shape)
        for i in range(self.num_stages + 1):
            # generate flow at the decoder input stage
            flow = self.flow_pred_list[i](output) # [B, 4, D, fc_h, fc_w]
            forward_flow, backward_flow = torch.split(flow, [2, 2], dim=1)      # (2)
            # start_key_feature, end_key_feature = torch.split(key_feature_list[i], [1, 1], dim=2) # I
            start_key_feature = key_feature_list[i][:, :, 0, :, :].unsqueeze(2)
            end_key_feature = key_feature_list[i][:, :, -1, :, :].unsqueeze(2)
            # warp the keyframe features with predicted forward and backward flow
            # torch.Size([1, 64, 1, 4, 5]) torch.Size([1, 2, 8, 4, 5])
            forward_warp = self.wk_list[i](start_key_feature, forward_flow)     # (3)
            backward_warp = self.wk_list[i](end_key_feature, backward_flow)     # (3)
            # distance-aware weighted sum
            fused_warp = forward_warp * forward_distance + backward_warp * backward_distance # (1 - t) * forward_warp + t * backward_warp (4)
            if i < self.num_stages:
                output = self.decoder_list[i](output, fused_warp)
            else:
                output = self.rgb_head_layer(torch.cat([output, fused_warp], dim=1))

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)

class HDNeRV3Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder_frame = model.encoder_frame 
        self.fc_w = model.fc_w
        self.fc_h = model.fc_h
    def forward(self, x):
        B, _, D, _, _ = x.size()
        output = self.encoder_frame(x)
        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 1, 4, 2, 3)  # [B, C, D, fc_h, fc_w]
        return output
    
class HDNeRV3Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder_keyframe = model.encoder_keyframe
        self.norm = model.norm
        self.num_stages = model.num_stages
        self.flow_pred_list = model.flow_pred_list
        self.wk_list = model.wk_list
        self.decoder_list = model.decoder_list
        self.rgb_head_layer = model.rgb_head_layer
        self.dataset_std = model.dataset_std
        self.dataset_mean = model.dataset_mean

    def forward(self, embedding, keyframe, backward_distance):
        B, D = backward_distance.size()
        backward_distance = backward_distance.view(B, 1, -1, 1, 1)
        forward_distance = 1 - backward_distance

        key_feature_list = self.encoder_keyframe(keyframe)

        start = key_feature_list[0][:, :, 0, :, :].unsqueeze(2)
        end = key_feature_list[0][:, :, -1, :, :].unsqueeze(2)
        output = self.norm(torch.cat([start, embedding, end], dim=2))
        # M^1_t
        # output = self.norm(key_feature_list[0])      # (1)
        # print(output.shape)
        for i in range(self.num_stages + 1):
            # generate flow at the decoder input stage
            flow = self.flow_pred_list[i](output) # [B, 4, D, fc_h, fc_w]
            forward_flow, backward_flow = torch.split(flow, [2, 2], dim=1)      # (2)
            # start_key_feature, end_key_feature = torch.split(key_feature_list[i], [1, 1], dim=2) # I
            start_key_feature = key_feature_list[i][:, :, 0, :, :].unsqueeze(2)
            end_key_feature = key_feature_list[i][:, :, -1, :, :].unsqueeze(2)
            # warp the keyframe features with predicted forward and backward flow
            # torch.Size([1, 64, 1, 4, 5]) torch.Size([1, 2, 8, 4, 5])
            forward_warp = self.wk_list[i](start_key_feature, forward_flow)     # (3)
            backward_warp = self.wk_list[i](end_key_feature, backward_flow)     # (3)
            # distance-aware weighted sum
            fused_warp = forward_warp * forward_distance + backward_warp * backward_distance # (1 - t) * forward_warp + t * backward_warp (4)
            if i < self.num_stages:
                output = self.decoder_list[i](output, fused_warp)
            else:
                output = self.rgb_head_layer(torch.cat([output, fused_warp], dim=1))

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)


class NeRVBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(kwargs['ngf'], kwargs['new_ngf']*kwargs['stride']*kwargs['stride'], 3, 1, 1)
        self.up_scale = nn.PixelShuffle(kwargs['stride'])
        self.act = nn.GELU()

    def forward(self, x, sty=None):
        x = self.act(self.up_scale(self.conv(x)))
        return x

class NeRV(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        ngf = kwargs['fc_dim']
        self.stem = MLP(in_chan=kwargs['embed_length'], out_chan=self.fc_h * self.fc_w * ngf)
        
        self.layers = nn.ModuleList()
        for i, stride in enumerate(kwargs['stride_list']):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                reduction = stride
                new_ngf = max(round(ngf / reduction), kwargs['lower_width'])

            self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=stride))
            ngf = new_ngf

        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1) 

    def forward(self, embed_input):
        B, C, D = embed_input.size()

        output = self.stem(embed_input) # [B, C*fc_h*fc_w, D]
        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 4, 1, 2, 3)  # [B, D, C, fc_h, fc_w]
        output = output.reshape(B*D, -1, self.fc_h, self.fc_w)
        out_list = []
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        output = self.head_layer(output)
        output = (torch.tanh(output) + 1) * 0.5

        BD, C, H, W = output.size()
        output = output.view(B, D, C, H, W).permute(0, 2, 1, 3, 4) # [B, C, D, H, W]
        return  output


###################################  Code for ConvNeXt   ###################################
class ConvNeXtVideo(nn.Module):
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], type=True):
        super(ConvNeXtVideo, self).__init__()
        if type:
            self.convnext = ConvNeXtV2(depths=depths,dims=dims)
        else:
            self.convnext = ConvNeXt(depths=depths, dims=dims)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
        # x = x.view(-1, C, H, W)
        # Forward each frame through the ConvNeXt model
        x = self.convnext(x)
        # Reshape the output back to shape (batchsize, depth, features)
        x = x.view(B, D, -1).permute(0, 2, 1)
        return x

class ConvNeXtVideo_temp(nn.Module):
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], type=True):
        super(ConvNeXtVideo_temp, self).__init__()
        if type:
            self.convnext = ConvNeXtV2_temp(depths=depths,dims=dims)
        else:
            self.convnext = ConvNeXt(depths=depths, dims=dims)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
        # x = x.view(-1, C, H, W)s
        # Forward each frame through the ConvNeXt model
        x = self.convnext(x)

        for i in range(len(x)):
            _, c, h, w = x[i].size()
            x[i] = x[i].view(B, D, c, h, w).permute(0, 2, 1, 3, 4)
        return x[::-1]


class Encoder2(nn.Module):
    def __init__(self, kernel_size=3, stride=1, stride_list=[], bias=True):
        super().__init__()
        n_resblocks = len(stride_list)

        # define head module
        m_head = nn.Sequential(
            nn.Conv3d(12, 64, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, kernel_size//2, kernel_size//2), bias=bias),
            nn.GELU(),
        )
        m_body = []
        for i in range(n_resblocks):
            m_body.append(nn.Sequential(
                            nn.Conv3d(64, 64, kernel_size=(1, stride_list[i], stride_list[i]), stride=(1, stride_list[i], stride_list[i]), padding=(0, 0, 0), bias=bias),
                            nn.GELU(),
                            )
                        )
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)

    def forward(self, x):
        x = self.head(x)
        for stage in self.body:
           x = stage(x)
        return x
    
class HDNeRV(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        
        ngf = kwargs['fc_dim']

        self.stride_list = kwargs['stride_list']
        self.cnn = ConvNeXtVideo(
                depths=kwargs['enc_block'], dims=kwargs['enc_dim'], type=kwargs['ver']
            )
        self.num_stages = len(self.stride_list)

        encoder_dim = 64
        self.encoder = Encoder(stride_list=self.stride_list[::-1])
        self.norm = nn.InstanceNorm3d(ngf + encoder_dim)

        self.decoder_list = nn.ModuleList()
        self.flow_pred_list = nn.ModuleList([Head(ngf + encoder_dim, 4)])

        height = self.fc_h
        width = self.fc_w
        self.wk_list = nn.ModuleList([WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device'])])
        for i, stride in enumerate(self.stride_list):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                new_ngf = max(round(ngf / stride), kwargs['lower_width'])

            self.decoder_list.append(DNeRVBlock(ngf=ngf + encoder_dim if i == 0 else ngf, new_ngf=new_ngf, 
                                                stride=stride, clip_size=kwargs['clip_size']))
            self.flow_pred_list.append(Head(new_ngf, 4))
            height = height * stride
            width = width * stride
            self.wk_list.append(WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device']))

            ngf = new_ngf
        
        self.rgb_head_layer = Head(new_ngf + 3, 3)

        self.dataset_mean = torch.tensor(kwargs['dataset_mean']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.dataset_std = torch.tensor(kwargs['dataset_std']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        
    def forward(self, video, keyframe, backward_distance):
        B, D = backward_distance.size()
        backward_distance = backward_distance.view(B, 1, -1, 1, 1)
        forward_distance = 1 - backward_distance

        key_feature_list = self.encoder(keyframe) # [B, encoder_dim, 2, H, W]
        output = self.cnn(video)
        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 1, 4, 2, 3)  # [B, C, D, fc_h, fc_w]
        # I^1_t
        content_feature = F.interpolate(key_feature_list[0], scale_factor=(D/2, 1, 1), mode='trilinear') # [B, encoder_dim, D, fc_h, fc_w]
        # output = self.norm(torch.concat([output, content_feature], dim=1))
        # M^1_t
        output = self.norm(torch.cat([output, content_feature], dim=1))      # (1)

        for i in range(self.num_stages + 1):
            # generate flow at the decoder input stage
            flow = self.flow_pred_list[i](output) # [B, 4, D, fc_h, fc_w]
            forward_flow, backward_flow = torch.split(flow, [2, 2], dim=1)      # (2)
            start_key_feature, end_key_feature = torch.split(key_feature_list[i], [1, 1], dim=2) # I
            # warp the keyframe features with predicted forward and backward flow
            # print(start_key_feature.shape, forward_flow.shape)
            # torch.Size([1, 64, 1, 4, 5]) torch.Size([1, 2, 8, 4, 5])
            forward_warp = self.wk_list[i](start_key_feature, forward_flow)     # (3)
            backward_warp = self.wk_list[i](end_key_feature, backward_flow)     # (3)
            # distance-aware weighted sum
            fused_warp = forward_warp * forward_distance + backward_warp * backward_distance # (1 - t) * forward_warp + t * backward_warp (4)
            
            if i < self.num_stages:
                output = self.decoder_list[i](output, fused_warp)
            else:
                output = self.rgb_head_layer(torch.cat([output, fused_warp], dim=1))

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)


class RAFT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        
        ngf = kwargs['fc_dim']

        self.stride_list = kwargs['stride_list']
        self.cnn = ConvNeXtVideo(
                depths=kwargs['enc_block'], dims=kwargs['enc_dim'], type=kwargs['ver']
            )
        self.num_stages = len(self.stride_list)

        encoder_dim = 64
        self.encoder = Encoder2(stride_list=self.stride_list[::-1])
        self.norm = nn.InstanceNorm3d(ngf + encoder_dim)

        self.decoder_list = nn.ModuleList()

        height = self.fc_h
        width = self.fc_w
        for i, stride in enumerate(self.stride_list):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                new_ngf = max(round(ngf / stride), kwargs['lower_width'])

            self.decoder_list.append(RAFT_Block(ngf=ngf + encoder_dim if i == 0 else ngf, new_ngf=new_ngf, 
                                                stride=stride, clip_size=kwargs['clip_size']))

            height = height * stride
            width = width * stride

            ngf = new_ngf
        
        self.rgb_head_layer = Head(new_ngf, 3)

        self.dataset_mean = torch.tensor(kwargs['dataset_mean']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.dataset_std = torch.tensor(kwargs['dataset_std']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.OF = raft_ready()

    def forward(self, video,  keyframe, backward_distance):
        B, C, D, H, W = video.size()

        image1 = keyframe[:,:,0,:,:]
        image2 = keyframe[:,:,1,:,:]

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, flow21 = self.OF(image1, image2, iters=30, test_mode=True)
        _, flow12 = self.OF(image2, image1, iters=30, test_mode=True)
        flow21, flow12 = padder.unpad(flow21), padder.unpad(flow12)
        
        flow12 = torch.cat([flow12[:, 0: 1, :, :] / ((flow12.shape[3] - 1.0) / 2.0),
                                    flow12[:, 1: 2, :, :] / ((flow12.shape[2] - 1.0) / 2.0)], 1)
        flow21 = torch.cat([flow21[:, 0: 1, :, :] / ((flow21.shape[3] - 1.0) / 2.0),
                                    flow21[:, 1: 2, :, :] / ((flow21.shape[2] - 1.0) / 2.0)], 1)
        
        flowf1t, flowf2t, flowb1t, flowb2t = generate_double_flow_time_t(flow12, flow21, time_stamp=backward_distance)
        coorMap = generate_coorMap((H, W), scale=True, flatten=False).type_as(image1)
        coorMapf1_t, coorMapb1_t, coorMapf2_t, coorMapb2_t = generate_double_coormap_t(flowf1t, flowf2t, flowb1t, flowb2t, coorMap)

        image1 = image1.unsqueeze(2).expand(-1, -1, D, -1, -1).contiguous().view(B*D, -1, H, W)
        image2 = image2.unsqueeze(2).expand(-1, -1, D, -1, -1).contiguous().view(B*D, -1, H, W)
        fef1 = nn.functional.grid_sample(image1, coorMapf1_t, mode='bilinear', padding_mode='reflection')
        feb1 = nn.functional.grid_sample(image1, coorMapb1_t, mode='bilinear', padding_mode='reflection')
        fef2 = nn.functional.grid_sample(image2, coorMapf2_t, mode='bilinear', padding_mode='reflection')
        feb2 = nn.functional.grid_sample(image2, coorMapb2_t, mode='bilinear', padding_mode='reflection')
        fef1 = fef1.view(B,D,C,H,W).permute(0,2,1,3,4)
        feb1 = feb1.view(B,D,C,H,W).permute(0,2,1,3,4)
        fef2 = fef2.view(B,D,C,H,W).permute(0,2,1,3,4)
        feb2 = feb2.view(B,D,C,H,W).permute(0,2,1,3,4)

        optical_flow = torch.cat([fef1, feb1, fef2, feb2], 1)
        key_feature = self.encoder(optical_flow)

        output = self.cnn(video)

        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 1, 4, 2, 3)  # [B, C, D, fc_h, fc_w]

        output = self.norm(torch.cat([output, key_feature], dim=1))       # (1)
        
        for i in range(self.num_stages + 1):
            if i < self.num_stages:
                output = self.decoder_list[i](output)
            else:
                output = self.rgb_head_layer(output)

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)

class RAFT_t(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        
        ngf = kwargs['fc_dim']
        self.stem = MLP(in_chan=kwargs['embed_length'], out_chan=self.fc_h * self.fc_w * ngf)

        self.stride_list = kwargs['stride_list']
        self.num_stages = len(self.stride_list)

        encoder_dim = 64
        self.encoder = Encoder2(stride_list=self.stride_list[::-1])
        self.norm = nn.InstanceNorm3d(ngf + encoder_dim)

        self.decoder_list = nn.ModuleList()

        height = self.fc_h
        width = self.fc_w
        for i, stride in enumerate(self.stride_list):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                new_ngf = max(round(ngf / stride), kwargs['lower_width'])

            self.decoder_list.append(RAFT_Block(ngf=ngf + encoder_dim if i == 0 else ngf, new_ngf=new_ngf, 
                                                stride=stride, clip_size=kwargs['clip_size']))

            height = height * stride
            width = width * stride

            ngf = new_ngf
        
        self.rgb_head_layer = Head(new_ngf, 3)

        self.dataset_mean = torch.tensor(kwargs['dataset_mean']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.dataset_std = torch.tensor(kwargs['dataset_std']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.OF = raft_ready()

    def forward(self, embed_input, keyframe, backward_distance):
        B, _, D = embed_input.size()
        _, C, _,H, W = keyframe.size()

        image1 = keyframe[:,:,0,:,:]
        image2 = keyframe[:,:,1,:,:]

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, flow21 = self.OF(image1, image2, iters=30, test_mode=True)
        _, flow12 = self.OF(image2, image1, iters=30, test_mode=True)
        flow21, flow12 = padder.unpad(flow21), padder.unpad(flow12)
        
        flow12 = torch.cat([flow12[:, 0: 1, :, :] / ((flow12.shape[3] - 1.0) / 2.0),
                                    flow12[:, 1: 2, :, :] / ((flow12.shape[2] - 1.0) / 2.0)], 1)
        flow21 = torch.cat([flow21[:, 0: 1, :, :] / ((flow21.shape[3] - 1.0) / 2.0),
                                    flow21[:, 1: 2, :, :] / ((flow21.shape[2] - 1.0) / 2.0)], 1)
        
        flowf1t, flowf2t, flowb1t, flowb2t = generate_double_flow_time_t(flow12, flow21, time_stamp=backward_distance)
        coorMap = generate_coorMap((H, W), scale=True, flatten=False).type_as(image1)
        coorMapf1_t, coorMapb1_t, coorMapf2_t, coorMapb2_t = generate_double_coormap_t(flowf1t, flowf2t, flowb1t, flowb2t, coorMap)

        image1 = image1.unsqueeze(2).expand(-1, -1, D, -1, -1).contiguous().view(B*D, -1, H, W)
        image2 = image2.unsqueeze(2).expand(-1, -1, D, -1, -1).contiguous().view(B*D, -1, H, W)
        fef1 = nn.functional.grid_sample(image1, coorMapf1_t, mode='bilinear', padding_mode='reflection')
        feb1 = nn.functional.grid_sample(image1, coorMapb1_t, mode='bilinear', padding_mode='reflection')
        fef2 = nn.functional.grid_sample(image2, coorMapf2_t, mode='bilinear', padding_mode='reflection')
        feb2 = nn.functional.grid_sample(image2, coorMapb2_t, mode='bilinear', padding_mode='reflection')
        fef1 = fef1.view(B,D,C,H,W).permute(0,2,1,3,4)
        feb1 = feb1.view(B,D,C,H,W).permute(0,2,1,3,4)
        fef2 = fef2.view(B,D,C,H,W).permute(0,2,1,3,4)
        feb2 = feb2.view(B,D,C,H,W).permute(0,2,1,3,4)
        # print(fef1.shape, feb1.shape, fef2.shape, feb2.shape)
        fuse = torch.cat([fef1, feb1, fef2, feb2], 1)
        key_feature_list = self.encoder(fuse)

        output = self.stem(embed_input)  # [B, C*fc_h*fc_w, D]
        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 1, 4, 2, 3)  # [B, C, D, fc_h, fc_w]
        
        output = self.norm(torch.cat([output, key_feature_list[0]], dim=1))      # (1)
        # print(output.shape)
        
        for i in range(self.num_stages + 1):
            if i < self.num_stages:
                output = self.decoder_list[i](output)
            else:
                output = self.rgb_head_layer(output)

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)