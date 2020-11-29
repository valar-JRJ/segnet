"""
Infer segmentation results from a trained SegNet model


Usage:
python inference.py --data_root /home/SharedData/intern_sayan/PascalVOC2012/data/VOCdevkit/VOC2012/ \
                    --val_path ImageSets/Segmentation/val.txt \
                    --img_dir JPEGImages \
                    --mask_dir SegmentationClass \
                    --model_path /home/SharedData/intern_sayan/PascalVOC2012/model_best.pth \
                    --output_dir /home/SharedData/intern_sayan/PascalVOC2012/predictions \
                    --gpu 1
"""

from __future__ import print_function

import argparse
import time
import tqdm
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from dataset import PascalVOCDataset, NUM_CLASSES
from model import SegNet
from score import Score, AverageMeter


def validate(model, val_path, img_path, mask_path, batch_size):
    running_metrics_val = Score(NUM_CLASSES)
    val_loss_meter = AverageMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    val_dataset = PascalVOCDataset(list_file=val_path,
                                   img_dir=img_path,
                                   mask_dir=mask_path,
                                   img_size=512,
                                   is_transform=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)

    class_weights = 1.0 / val_dataset.get_class_probability().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    with torch.no_grad():
        for batch_idx, (image, mask) in tqdm.tqdm(enumerate(val_dataloader), desc='Evaluating...'):
            input_tensor = torch.autograd.Variable(image.to(device))
            target_tensor = torch.autograd.Variable(mask.to(device))

            output = model(input_tensor)
            loss = criterion(output, target_tensor)

            pred = output.data.max(1)[1].cpu().numpy()
            gt = target_tensor.data.cpu().numpy()

            running_metrics_val.update(gt, pred)
            val_loss_meter.update(loss.item())
    score, class_iou = running_metrics_val.get_scores()
    return val_loss_meter.avg, score, class_iou


if __name__ == "__main__":
    # Constants
    NUM_INPUT_CHANNELS = 3
    NUM_OUTPUT_CHANNELS = NUM_CLASSES

    # Arguments
    parser = argparse.ArgumentParser(description='Validate a SegNet model')

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data_root', type=str, default='/data/pascal/VOCdevkit/VOC2012')
    parser.add_argument('--val_path', type=str, default='ImageSets/Segmentation/val.txt')
    parser.add_argument('--img_dir', type=str, default='JPEGImages')
    parser.add_argument('--mask_dir', type=str, default='SegmentationClass')
    parser.add_argument('--model_path', type=str, required=True)

    args = parser.parse_args()
    print(args)

    data_root = args.data_root
    val_dir = os.path.join(data_root, args.val_path)
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegNet().to(device)
    model.load_state_dict(torch.load(args.model_path))

    val_loss, val_score, val_class_iou = validate(
        model=model, val_path=val_dir, img_path=img_dir, mask_path=mask_dir, batch_size=args.batch_size
    )

    for k, v in val_score.items():
        print(k, v)

    for k, v in val_class_iou.items():
        print(k, v)
    print('\nval_loss:'+val_loss)

