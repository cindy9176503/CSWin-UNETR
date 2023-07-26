import os
from pathlib import PurePath

import torch
import torch.nn.functional as F
import numpy as np

from monai.data import decollate_batch
from monai.transforms import (
    LoadImaged,
    AddChannel,
    SqueezeDimd,
    AsDiscrete
)
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from data_utils.io import save_img
from transforms.transform_utils import LabelToBinaryLabeld
import matplotlib.pyplot as plt

from sklearn import preprocessing


def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data['image'].to(device))
        output_lbl = torch.argmax(output, dim=1)
        
        temp_prob = output[0,-1,:,:,:]      
        prob = np.squeeze(output_lbl * temp_prob)
        scaler = preprocessing.MinMaxScaler()      
        # norm prob
        min_value = np.min(prob)
        max_value = np.max(prob)
        norm_prob = (prob - min_value) / (max_value - min_value)
        norm_prob = np.rot90(norm_prob, k=-2) #RAS
    return output_lbl, norm_prob


def check_channel(inp):
    # check shape is 5
    add_ch = AddChannel()
    len_inp_shape = len(inp.shape)
    if len_inp_shape == 4:
        inp = add_ch(inp)
    if len_inp_shape == 3:
        inp = add_ch(inp)
        inp = add_ch(inp)
    return inp


def eval_label_pred(data, cls_num, device):
    # post transform
    post_label = AsDiscrete(to_onehot=cls_num, threshold=0.7)
    
    # metric
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )
    
    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean",
        get_not_nans=False
    )
    
    # batch data
    val_label, val_pred = (data["label"].to(device), data["pred"].to(device))   
   
    
    # check shape is 5
    val_label = check_channel(val_label)
    val_pred = check_channel(val_pred)
    
    # deallocate batch data
    val_labels_convert = [
        post_label(val_label_tensor) for val_label_tensor in val_label
    ]
    val_output_convert = [
        post_label(val_pred_tensor) for val_pred_tensor in val_pred
    ]
    
    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    hd95_metric(y_pred=val_output_convert, y=val_labels_convert)

    dc_vals = dice_metric.get_buffer().detach().cpu().numpy().squeeze()
    hd95_vals = hd95_metric.get_buffer().detach().cpu().numpy().squeeze()

    return dc_vals, hd95_vals


def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def run_infering(
        model,
        data,
        model_inferer,
        post_transform,
        args
    ):

    # get label cls
    lbl_cls = data['label'].flatten().unique()
    
    # test
    data['pred'], prob = infer(model, data, model_inferer, args.device)
    
    # eval infer tta
    if 'label' in data.keys():
        dc_vals, hd95_vals = eval_label_pred(data, args.out_channels, args.device)
        print('\ninfer test time aug:')
        print('dice:', dc_vals)
        print('hd95:', hd95_vals)
        print('avg dice:', dc_vals.mean())
        print('avg hd95:', hd95_vals.mean())
        
        # post label transform 
        sqz_transform = SqueezeDimd(keys=['label'])
        data = sqz_transform(data)
    
    # post transform
    data = post_transform(data)

    # eval infer origin
    if 'label' in data.keys():
        # get orginal label
        lbl_dict = {'label': data['label_meta_dict']['filename_or_obj']}
        lbl_data = LoadImaged(keys='label')(lbl_dict)
            
        # lbl_cls = data['pred'].flatten().unique()
        
        if len(lbl_cls) == 2:
            cvt = LabelToBinaryLabeld(keys=["label"])
            lbl_data = cvt(lbl_data)
        
        data['label'] = lbl_data['label']
        data['label_meta_dict'] = lbl_data['label']
        
        dc_vals, hd95_vals = eval_label_pred(data, args.out_channels, args.device)
        print('\ninfer test original:')
        print('dice:', dc_vals)
        print('hd95:', hd95_vals)
        print('avg dice:', dc_vals.mean())
        print('avg hd95:', hd95_vals.mean())
    
    # save pred result
    filename = get_filename(data)
    porb_filename = filename.replace(".nii.gz", "_prob.nii.gz")
    infer_img_pth = os.path.join(args.infer_dir, filename)
    porb_pth = os.path.join(args.infer_dir, porb_filename) 
    save_img(
      data['pred'], 
      data['pred_meta_dict'], 
      infer_img_pth
    )
    
    save_img(
      prob, 
      data['pred_meta_dict'], 
      porb_pth
    )