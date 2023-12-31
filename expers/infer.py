import sys

import os
from functools import partial

import torch

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    Orientationd,
    ToNumpyd,
)
from monailabel.transform.post import Restored

from data_utils.data_loader_utils import load_data_dict_json
from data_utils.dataset import get_infer_data
from data_utils.io import load_json
from runners.inferer import run_infering
from networks.network import network

from expers.args import get_parser


def main():
    args = get_parser(sys.argv[1:])
    main_worker(args)
    

def main_worker(args):
    # make dir
    os.makedirs(args.infer_dir, exist_ok=True)

    # device
    if torch.cuda.is_available():
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda is not available")
        args.device = torch.device("cpu")

    # model
    model = network(args.model_name, args)

    # check point
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # load model
        model.load_state_dict(checkpoint["state_dict"])
        # load check point epoch and best acc
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "early_stop_count" in checkpoint:
            early_stop_count = checkpoint["early_stop_count"]
        print(
          "=> loaded checkpoint '{}' (epoch {}) (bestacc {}) (early stop count {})"\
          .format(args.checkpoint, start_epoch, best_acc, early_stop_count)
        )        


    # inferer jack
    keys = ['pred']
    post_transform = Compose([
        Orientationd(keys=keys, axcodes="LPS"),
        # ToNumpyd(keys=keys),
        Restored(keys=keys, ref_image="image")
    ])
    
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    # prepare data_dict
    if args.data_dicts_json:
        data_dicts = load_data_dict_json(args.data_dir, args.data_dicts_json)
    else:
        data_dicts = [{
            'image': args.img_pth,
            'label': args.lbl_pth
        }]

    # run infer
    for data_dict in data_dicts:
        print('infer data:', data_dict)
      
        # load infer data
        data = get_infer_data(data_dict, args)

        # infer
        run_infering(
            model,
            data,
            model_inferer,
            post_transform,
            args
        )


if __name__ == "__main__":
    main()