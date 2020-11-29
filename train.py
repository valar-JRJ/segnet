"""
Train a SegNet model


Usage:
python train.py --data_root /home/SharedData/intern_sayan/PascalVOC2012/data/VOCdevkit/VOC2012/ \
                --train_path ImageSets/Segmentation/train.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --save_dir /home/SharedData/intern_sayan/PascalVOC2012/ \
                --checkpoint /home/SharedData/intern_sayan/PascalVOC2012/model_best.pth \
                --gpu 1
"""

from __future__ import print_function

import argparse
from dataset import PascalVOCDataset, NUM_CLASSES
from model import SegNet
from logger import Writer, log
from test import validate
from score import AverageMeter
import os
import time
import datetime
import torch
import torch.optim
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # Constants
    NUM_INPUT_CHANNELS = 3
    NUM_OUTPUT_CHANNELS = NUM_CLASSES

    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9

    # Arguments
    parser = argparse.ArgumentParser(description='Train a SegNet model')

    parser.add_argument('--epochs', type =int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='data/pascal/VOCdevkit/VOC2012')
    parser.add_argument('--train_path', type=str, default='ImageSets/Segmentation/train.txt')
    parser.add_argument('--val_path', type=str, default='ImageSets/Segmentation/val.txt')
    parser.add_argument('--img_dir', type=str, default='JPEGImages')
    parser.add_argument('--mask_dir', type=str, default='SegmentationClass')
    parser.add_argument('--checkpoint')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    data_root = args.data_root
    train_path = os.path.join(data_root, args.train_path)
    train_img_path = os.path.join(data_root, args.img_dir)
    train_mask_path = os.path.join(data_root, args.mask_dir)
    val_path = os.path.join(data_root, args.val_path)
    val_img_path = os.path.join(data_root, args.img_dir)
    val_mask_path = os.path.join(data_root, args.mask_dir)

    writer = Writer('logs')
    logger = log('')

    train_dataset = PascalVOCDataset(list_file=train_path,
                                     img_dir=train_img_path,
                                     mask_dir=train_mask_path,
                                     img_size=512,
                                     is_transform=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    model = SegNet().to(device)
    class_weights = 1.0/train_dataset.get_class_probability().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    # start from checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # training
    is_better = True
    prev_loss = float('inf')
    epoch_loss = AverageMeter()

    model.train()

    for epoch in range(args.epochs):
        t_start = time.time()

        for index, (image, mask) in enumerate(train_dataloader):
            batches_done = len(train_dataloader) * epoch + index

            input_tensor = torch.autograd.Variable(image.to(device))
            target_tensor = torch.autograd.Variable(mask.to(device))
            output = model(input_tensor)

            optimizer.zero_grad()
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss.update(loss.item())
            writer.scalar_summary('train_loss', loss, batches_done)

            # Determine approximate time left for epoch
            epoch_batches_left = len(train_dataloader) - (index + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time()-t_start) / (index + 1))
            print('epoch: {}\tbatches: {}\tloss: {:.8f}\tremaining time: {}'
                  .format(epoch, batches_done, loss, time_left))

        writer.scalar_summary('train_loss_epoch', epoch_loss.avg, epoch+1)
        is_better = epoch_loss.avg < prev_loss
        if is_better:
            prev_loss = epoch_loss.avg
            torch.save(model.state_dict(), f"checkpoints/best_ckpt_%d.pth" % epoch)
        else:
            torch.save(model.state_dict(), f"checkpoints/segnet_ckpt_%d.pth" % epoch)

        if epoch % args.eval_interval == 0:
            val_loss, score, class_iou = validate(
                model=model, val_path=val_path, img_path=val_img_path, mask_path=val_mask_path, batch_size=8
            )
            for k, v in score.items():
                print(k, v)
                logger.info("{}: {}".format(k, v))
                writer.scalar_summary("val_metrics/{}".format(k), v, epoch + 1)

            for k, v in class_iou.items():
                logger.info("{}: {}".format(k, v))
                writer.scalar_summary("val_metrics/cls_{}".format(k), v, epoch + 1)


