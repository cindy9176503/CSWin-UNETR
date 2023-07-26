from monai.networks.nets import SwinUNETR, UNETR, UNet, AttentionUnet, VNet, SegResNet, HighResNet, RegUNet, DynUNet, UNETR, AHNet

from networks.networkx.unetcnx_x3_2_2 import UNETCNX_X3_2_2
from networks.CoTr.network_architecture.ResTranUnet import ResTranUnet as CoTr

from networks.CSWinTransformer.models.cswinunetr_96_3 import CSwinUNETR as cswinunetr_96_3
from networks.CSWinTransformer.models.cswinunetr_128_32_8 import CSwinUNETR as cswinunetr_128_32_8
from networks.CSWinTransformer.models.cswinunetr_128_32_8_4 import CSwinUNETR as cswinunetr_128_32_8_4
from networks.CSWinTransformer.models.cswinunetr_112_7 import CSwinUNETR as cswinunetr_112_7
from networks.multitask.MultiTaskCSWinUNETR import MultiTaskCSWinUNETR


import torch
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def network(model_name, args):
    print(f'model: {model_name}')

    if model_name == 'Swinunetr':
        return SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            use_checkpoint=True,
        ).to(args.device)

    elif model_name == 'Unetr':
        return UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(args.device)

    elif model_name == 'Unet3d':
        return UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(64, 128, 256, 256),
            strides=(2, 2, 2),
            num_res_units=0,
            act='RELU',
            norm='BATCH'
        ).to(args.device)
    
    elif model_name == 'AttentionUnet':
        return AttentionUnet(
          spatial_dims=3,
          in_channels=args.in_channels,
          out_channels=args.out_channels,
          channels=(32, 64, 128, 256),
          strides=(2, 2, 2),
        ).to(args.device)
    
    elif model_name == 'cotr':
        '''
        CAUTION: if deep_supervision is True mean network output will be 
        a list e.x. [result, ds0, ds1, ds2], so loss func 
        should be use CoTr deep supervision loss.
        '''
        # TODO: deep_supervision 
        return CoTr(
            norm_cfg='IN',
            activation_cfg='LeakyReLU',
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            num_classes=args.out_channels,
            weight_std=False,
            deep_supervision=False
        ).to(args.device)
    
    elif model_name == 'Vnet3d':
        return VNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            act=('elu', {'inplace': True}),
            dropout_prob=0.5, 
            dropout_dim=3, 
            bias=False
        ).to(args.device)
    
    elif model_name == 'DynUNet':
        return DynUNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
            strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
            upsample_kernel_size=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]],
            filters=[16, 32, 64, 128, 256]
        ).to(args.device)
    
    elif model_name == 'Unetr':
        return UNETR(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z)
        ).to(args.device)
    
    elif model_name == 'MultiTaskCSWinUNETR':
        # 128, 32, 16, 8
        # 8,8
        return MultiTaskCSWinUNETR(
            num_classes=2
        ).to(args.device)
    

    elif model_name == 'cswinunetr_96_3_mask':
        return cswinunetr_96_3(
            img_size=(96, 96, 96), 
            split_size=(1,2,3,3),
            cswt_depths=[2,4,32,2],
            cwst_first_stage_divide=2,
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=False,
            drop_rate=0.,
            attn_drop_rate=0.,
            dropout_path_rate=0.1
        ).to(args.device)
    
    elif model_name == 'cswinunetr_96_3_center_edge':
        return cswinunetr_96_3(
            img_size=(96, 96, 96), 
            split_size=(1,2,3,3),
            cswt_depths=[2,4,32,2],
            cwst_first_stage_divide=2,
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=False,
            drop_rate=0.,
            attn_drop_rate=0.,
            dropout_path_rate=0.1
        ).to(args.device)
    
    elif model_name == 'cswinunetr_112_7':
        return cswinunetr_112_7(
            img_size=(112, 112, 112), 
            split_size=(1,2,7,7),
            cswt_depths=[2,4,32,2],
            cwst_first_stage_divide=2,
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            dropout_path_rate=0.1
        ).to(args.device)

    
    elif model_name == 'cswinunetr_128_32_8':
        return cswinunetr_128_32_8(
            img_size=(128, 128, 128), 
            split_size=(1,2,8,8),
            cswt_depths=[2,4,32,2],
            cwst_first_stage_divide=4,
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            dropout_path_rate=0.1
        ).to(args.device)
    
    elif model_name == 'cswinunetr_128_32_8_4':
        return cswinunetr_128_32_8_4(
            img_size=(128, 128, 128), 
            split_size=(1,2,4,4),
            cswt_depths=[2,4,32,2],
            cwst_first_stage_divide=4,
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            dropout_path_rate=0.1
        ).to(args.device)
    
    else:
        raise ValueError(f'not found model name: {model_name}')