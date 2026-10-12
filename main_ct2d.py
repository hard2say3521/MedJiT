"""Train the conditional 2D NCCT-to-CTA model."""

import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import copy

from dataset_ct_2d import CT2DPairDataset, CT2DTransform
from dataset_ct_2d_preprocessed import CT2DPairDatasetPreprocessed
import util.misc as misc
from engine_ct2d import train_one_epoch, evaluate
from denoiser_ct2d import Denoiser_CT2D


def get_args_parser():
    parser = argparse.ArgumentParser('JiT-CT2D', add_help=False)

    # Model
    parser.add_argument('--model', default='JiT-CT2D-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=512, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument('--condition_fusion', type=str, default='add', 
                        choices=['concat', 'add', 'cross_attn'],
                        help='Condition fusion method')
    parser.add_argument('--mask_dir', type=str, default='', help='Training mask directory')
    parser.add_argument('--eval_mask_dir', type=str, default='', help='Validation mask directory')

    # Training
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=6, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='cosine',  
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE sampling method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--eval_freq', type=int, default=20,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=16,
                        help='Generation batch size')

    # Data
    parser.add_argument('--non_contrast_dir', type=str, required=True)
    parser.add_argument('--contrast_dir', type=str, required=True)
    parser.add_argument('--eval_non_contrast_dir', type=str, default='')
    parser.add_argument('--eval_contrast_dir', type=str, default='')
    parser.add_argument('--slice_axis', type=int, default=2)
    parser.add_argument('--hu_min', type=float, default=-1000)
    parser.add_argument('--hu_max', type=float, default=1000)
    parser.add_argument('--window_width', type=float, default=500)
    parser.add_argument('--window_level', type=float, default=70)
    parser.add_argument('--use_window', action='store_true', default=True)
    parser.add_argument('--no_use_window', dest='use_window', action='store_false')
    parser.add_argument('--cache_slices', action='store_true')
    parser.add_argument('--use_preprocessed', action='store_true')

    # Checkpointing
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--resume', default='', help='Directory containing checkpoint-last.pth')
    parser.add_argument('--save_last_freq', type=int, default=5)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda')

    # Distributed training
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    torch.set_float32_matmul_precision('high') 
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    transform_train = CT2DTransform(flip_prob=0.5, rotate_prob=0.5)

    if args.use_preprocessed:
        dataset_train = CT2DPairDatasetPreprocessed(
            non_contrast_dir=args.non_contrast_dir, contrast_dir=args.contrast_dir,
            img_size=args.img_size, normalize=True, transform=transform_train, mask_dir=args.mask_dir,
        )
    else:
        dataset_train = CT2DPairDataset(
            non_contrast_dir=args.non_contrast_dir, contrast_dir=args.contrast_dir,
            img_size=args.img_size, normalize=True, hu_min=args.hu_min, hu_max=args.hu_max,
            slice_axis=args.slice_axis, transform=transform_train, cache_slices=args.cache_slices,
            window_width=args.window_width, window_level=args.window_level, use_window=args.use_window,
        )

    dataset_eval = dataset_train
    if args.eval_non_contrast_dir and args.eval_contrast_dir:
        if args.use_preprocessed:
            dataset_eval = CT2DPairDatasetPreprocessed(
                non_contrast_dir=args.eval_non_contrast_dir, contrast_dir=args.eval_contrast_dir,
                img_size=args.img_size, normalize=True, transform=None, mask_dir=args.eval_mask_dir,
            )
        else:
            dataset_eval = CT2DPairDataset(
                non_contrast_dir=args.eval_non_contrast_dir, contrast_dir=args.eval_contrast_dir,
                img_size=args.img_size, normalize=True, hu_min=args.hu_min, hu_max=args.hu_max,
                slice_axis=args.slice_axis, transform=None, cache_slices=False,
                window_width=args.window_width, window_level=args.window_level, use_window=args.use_window,
            )

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )

    model = Denoiser_CT2D(args)
    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        parameter_names = [name for name, _ in model_without_ddp.named_parameters()]
        ema_state1 = checkpoint.get('model_ema1', checkpoint['model'])
        ema_state2 = checkpoint.get('model_ema2', checkpoint['model'])
        model_without_ddp.ema_params1 = [
            ema_state1[name].detach().clone().to(device) for name in parameter_names
        ]
        model_without_ddp.ema_params2 = [
            ema_state2[name].detach().clone().to(device) for name in parameter_names
        ]
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))

    if args.evaluate_gen:
        with torch.no_grad():
            evaluate(model_without_ddp, args, dataset_eval, 0, batch_size=args.gen_bsz, log_writer=log_writer, num_samples=100)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch, epoch_name="last")

        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, dataset_eval, epoch, batch_size=args.gen_bsz, log_writer=log_writer, num_samples=100)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    print('Training time:', str(datetime.timedelta(seconds=int(total_time))))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
