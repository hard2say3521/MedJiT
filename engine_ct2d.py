"""Training and evaluation loops for 2D CT translation."""

import math
import sys
import os

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (non_contrast, contrast) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Update the learning rate per iteration.
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        non_contrast = non_contrast.to(device, non_blocking=True).to(torch.float32)
        contrast = contrast.to(device, non_blocking=True).to(torch.float32)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(contrast, non_contrast)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, dataset, epoch, batch_size=16, log_writer=None, num_samples=100):
    """Generate samples, save images, and report reconstruction metrics."""
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    
    save_folder = os.path.join(
        args.output_dir,
        "eval_epoch{}_steps{}_res{}".format(
            epoch, model_without_ddp.steps, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_folder_gt = os.path.join(save_folder, "ground_truth")
    save_folder_input = os.path.join(save_folder, "input")
    save_folder_generated = os.path.join(save_folder, "generated")
    if misc.get_rank() == 0:
        os.makedirs(save_folder_gt, exist_ok=True)
        os.makedirs(save_folder_input, exist_ok=True)
        os.makedirs(save_folder_generated, exist_ok=True)
    
    # Wait until rank zero has created the output folders.
    torch.distributed.barrier()
    
    # Evaluate with the first EMA snapshot.
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)
    
    num_steps = (num_samples + batch_size * world_size - 1) // (batch_size * world_size)
    for i in range(num_steps):
        print("Generation step {}/{}".format(i + 1, num_steps))
        
        batch_indices = []
        for j in range(batch_size):
            idx = (i * batch_size * world_size + local_rank * batch_size + j) % len(dataset)
            batch_indices.append(idx)
        
        batch_non_contrast = []
        batch_contrast = []
        for idx in batch_indices:
            non_contrast, contrast = dataset[idx]
            batch_non_contrast.append(non_contrast)
            batch_contrast.append(contrast)
        
        device = next(model_without_ddp.parameters()).device
        non_contrast_batch = torch.stack(batch_non_contrast).to(device)
        contrast_batch = torch.stack(batch_contrast).to(device)
        
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
            generated_contrast = model_without_ddp.generate(non_contrast_batch)
        
        torch.distributed.barrier()
        
        # Map model values from [-1, 1] to [0, 1].
        non_contrast_batch = (non_contrast_batch + 1) / 2
        contrast_batch = (contrast_batch + 1) / 2
        generated_contrast = (generated_contrast + 1) / 2
        
        non_contrast_batch = non_contrast_batch.detach().cpu().numpy()
        contrast_batch = contrast_batch.detach().cpu().numpy()
        generated_contrast = generated_contrast.detach().cpu().numpy()
        
        for b_id in range(non_contrast_batch.shape[0]):
            # Use a global index so ranks never overwrite each other's output.
            sample_idx = i * batch_size * world_size + local_rank * batch_size + b_id
            if sample_idx >= num_samples:
                continue
            
            try:
                input_img = non_contrast_batch[b_id, 0]
                input_img = np.clip(input_img * 255, 0, 255).astype(np.uint8)
                input_path = os.path.join(save_folder_input, '{:05d}.png'.format(sample_idx))
                success = cv2.imwrite(input_path, input_img)
                if not success:
                    print(f"Warning: Failed to save input image {sample_idx}")
                
                gt_img = contrast_batch[b_id, 0]
                gt_img = np.clip(gt_img * 255, 0, 255).astype(np.uint8)
                gt_path = os.path.join(save_folder_gt, '{:05d}.png'.format(sample_idx))
                success = cv2.imwrite(gt_path, gt_img)
                if not success:
                    print(f"Warning: Failed to save ground truth image {sample_idx}")
                
                gen_img = generated_contrast[b_id, 0]
                gen_img = np.clip(gen_img * 255, 0, 255).astype(np.uint8)
                gen_path = os.path.join(save_folder_generated, '{:05d}.png'.format(sample_idx))
                success = cv2.imwrite(gen_path, gen_img)
                if not success:
                    print(f"Warning: Failed to save generated image {sample_idx}")
            except Exception as e:
                print(f"Error saving image {sample_idx}: {e}")
                continue
    
    # Wait until all ranks have finished writing before metric calculation.
    torch.distributed.barrier()
    
    if misc.is_main_process():
        mse_list = []
        mae_list = []
        psnr_list = []
        ssim_list = []
        
        for idx in range(num_samples):
            gt_path = os.path.join(save_folder_gt, '{:05d}.png'.format(idx))
            gen_path = os.path.join(save_folder_generated, '{:05d}.png'.format(idx))
            
            if not os.path.exists(gt_path) or not os.path.exists(gen_path):
                continue
            
            try:
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gen_img = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
                
                if gt_img is None or gen_img is None:
                    print(f"Warning: Failed to read images for sample {idx}, skipping")
                    continue
                
                gt_img = gt_img.astype(np.float32) / 255.0
                gen_img = gen_img.astype(np.float32) / 255.0
                
                if gt_img.shape != gen_img.shape:
                    print(f"Warning: Shape mismatch for sample {idx}, skipping")
                    continue
                
                mse = np.mean((gt_img - gen_img) ** 2)
                mse_list.append(mse)
                
                mae = np.mean(np.abs(gt_img - gen_img))
                mae_list.append(mae)
                
                try:
                    from skimage.metrics import structural_similarity as ssim
                    from skimage.metrics import peak_signal_noise_ratio as psnr
                    
                    if mse > 0:
                        psnr_val = psnr(gt_img, gen_img, data_range=1.0)
                        psnr_list.append(psnr_val)
                    
                    ssim_val = ssim(gt_img, gen_img, data_range=1.0)
                    ssim_list.append(ssim_val)
                except ImportError:
                    pass
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        if len(mse_list) > 0:
            avg_mse = np.mean(mse_list)
            avg_mae = np.mean(mae_list)
            
            print(f"Evaluation metrics (epoch {epoch}):")
            print(f"  Valid samples: {len(mse_list)}/{num_samples}")
            print(f"  MSE: {avg_mse:.6f}")
            print(f"  MAE: {avg_mae:.6f}")
            
            if log_writer is not None:
                log_writer.add_scalar('eval/mse', avg_mse, epoch)
                log_writer.add_scalar('eval/mae', avg_mae, epoch)
            
            if len(psnr_list) > 0:
                avg_psnr = np.mean(psnr_list)
                print(f"  PSNR: {avg_psnr:.4f} dB")
                if log_writer is not None:
                    log_writer.add_scalar('eval/psnr', avg_psnr, epoch)
            
            if len(ssim_list) > 0:
                avg_ssim = np.mean(ssim_list)
                print(f"  SSIM: {avg_ssim:.4f}")
                if log_writer is not None:
                    log_writer.add_scalar('eval/ssim', avg_ssim, epoch)
        else:
            print(f"Warning: No valid samples for evaluation at epoch {epoch}")
    
    # Restore the training parameters.
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
    
    torch.distributed.barrier()
