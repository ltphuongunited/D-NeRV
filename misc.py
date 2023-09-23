    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='UVG', help='dataset')
    # parser.add_argument('--keyframe_quality', type=int, default=3, help='keyframe quality, control flag used for keyframe image compression')
    # parser.add_argument('--clip_size', type=int, default=8, help='clip_size to sample at a single time')

    # dataset_str = 'Dataset_HDNeRV_UVG'
    # args = parser.parse_args()
    # args.num_classes = 7
    # args.dataset_mean = [0.4519, 0.4505, 0.4519]
    # args.dataset_std = [0.2434, 0.2547, 0.2958]
    
    # transform_rgb = transforms.Compose([transforms.ToTensor()])
    # transform_keyframe = transforms.Compose([transforms.ToTensor(), transforms.Normalize(args.dataset_mean, args.dataset_std)])
    # train_dataset = eval(dataset_str)(args, transform_rgb, transform_keyframe)
    # # train_dataset[0]
    # video, input_index, keyframe, backward_distance, frame_mask = train_dataset[len(train_dataset)-1]
    # # print(len(train_dataset))
    # # print(video.shape, input_index.shape, keyframe.shape,backward_distance.shape, frame_mask.shape)
    # # print(input_index.shape)
    # # print(keyframe.shape)
    # # print(backward_distance.shape)
    # # print(frame_mask)
    # print(video.shape)
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
    
class HDNeRV3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        
        ngf = kwargs['fc_dim']

        self.stride_list = [2, 2, 2, 2, 4]
        self.encoder = ConvNeXtVideo(
                depths=kwargs['enc_block'], dims=kwargs['enc_dim'], type=kwargs['ver']
            )
        self.num_stages = len(self.stride_list)
        encoder_dim = 64
        self.norm = nn.InstanceNorm3d(encoder_dim)

        self.decoder_list = nn.ModuleList()
        self.flow_pred_list = nn.ModuleList([Head(encoder_dim, 4)])

        height = self.fc_h
        width = self.fc_w
        self.wk_list = nn.ModuleList([WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device'])])
        for i, stride in enumerate(self.stride_list):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                new_ngf = max(round(ngf / stride), kwargs['lower_width'])

            self.decoder_list.append(DNeRVBlock(ngf=ngf, new_ngf=new_ngf, 
                                                stride=stride, clip_size=kwargs['clip_size']))
            self.flow_pred_list.append(Head(new_ngf, 4))
            height = height * stride
            width = width * stride
            self.wk_list.append(WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device']))

            ngf = new_ngf
        
        self.ignore = kwargs['ignore']
        if self.ignore:
            self.rgb_head_layer = Head(new_ngf, 3)
        else: 
            self.rgb_head_layer = Head(new_ngf + 3, 3)

        self.dataset_mean = torch.tensor(kwargs['dataset_mean']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.dataset_std = torch.tensor(kwargs['dataset_std']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        
    def forward(self, video, keyframe, backward_distance):
        B, D = backward_distance.size()
        backward_distance = backward_distance.view(B, 1, -1, 1, 1)
        forward_distance = 1 - backward_distance

        key_feature_list = self.encoder(video)

        # output = self.norm(torch.concat([output, content_feature], dim=1))
        # M^1_t
        output = self.norm(key_feature_list[0])      # (1)

        for i in range(self.num_stages + 1):
            # generate flow at the decoder input stage
            flow = self.flow_pred_list[i](output) # [B, 4, D, fc_h, fc_w]
            forward_flow, backward_flow = torch.split(flow, [2, 2], dim=1)      # (2)
            # start_key_feature, end_key_feature = torch.split(key_feature_list[i], [1, 1], dim=2) # I
            start_key_feature = key_feature_list[i][:, :, 0, :, :].unsqueeze(2)
            end_key_feature = key_feature_list[i][:, :, -1, :, :].unsqueeze(2)
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
                if self.ignore:
                    output = self.rgb_head_layer(output)
                else:
                    output = self.rgb_head_layer(torch.cat([output, fused_warp], dim=1))

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)
