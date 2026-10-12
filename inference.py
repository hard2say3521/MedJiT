import os
import argparse
import random
import numpy as np
import torch
import cv2
from scipy import stats

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
import lpips

from dataset_ct_2d_preprocessed import CT2DPairDatasetPreprocessed
from denoiser_ct2d import Denoiser_CT2D


def safe_nrmse(gt, gen, normalization='min-max', eps=1e-8):
    """Calculate NRMSE safely when the reference image is constant."""
    if normalization == 'min-max':
        denom = gt.max() - gt.min()
        if denom < eps:
            return 0.0 if np.allclose(gt, gen) else 1.0
    try:
        return nrmse(gt, gen, normalization=normalization)
    except Exception:
        return np.nan


def get_stats_result(data):
    """Format a metric using a normality-aware summary."""
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return "No valid data"
    if len(valid_data) < 3:
        return f"{np.mean(valid_data):.4f} ± {np.std(valid_data):.4f} (p=N/A)"

    _, p_value = stats.shapiro(valid_data)
    p_str = f"{p_value:.3f}".replace("0.0000", "0.000")

    if p_value > 0.05:
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        return f"{mean_val:.4f} ± {std_val:.4f} (normal, p={p_str})"
    else:
        median_val = np.median(valid_data)
        q1, q3 = np.percentile(valid_data, [25, 75])
        return f"{median_val:.4f} ({q1:.4f}-{q3:.4f}) (non-normal, p={p_str})"


def auto_soft_tissue_mask(img_01, low_thr=0.05, high_thr=0.92):
    """Build a binary soft-tissue mask from a CT slice in [0, 1]."""
    mask = np.logical_and(img_01 > low_thr, img_01 < high_thr).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def apply_auto_mask(img, mask):
    return img * mask


def get_args():
    parser = argparse.ArgumentParser("JiT-CT2D inference for val set", add_help=True)

    # Model settings must match training.
    parser.add_argument("--model", type=str, default="JiT-CT2D-B/16")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument(
        "--condition_fusion",
        type=str,
        default="add",
        choices=["concat", "add", "cross_attn"],
    )

    # Flow-matching settings must match training.
    parser.add_argument("--P_mean", type=float, default=-0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)
    parser.add_argument("--sampling_method", type=str, default="heun")
    parser.add_argument("--num_sampling_steps", type=int, default=50)

    parser.add_argument(
        "--non_contrast_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed NCCT slices",
    )
    parser.add_argument(
        "--contrast_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed CTA slices",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to checkpoint-last.pth",
    )
    parser.add_argument(
        "--save_root_dir",
        type=str,
        required=True,
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mask_low", type=float, default=0.05, help="Lower automatic-mask threshold for excluding air")
    parser.add_argument("--mask_high", type=float, default=0.92, help="Upper automatic-mask threshold for excluding dense bone")

    return parser.parse_args()


def build_model(args):
    device = torch.device(args.device)

    class DummyArgs:
        pass

    dargs = DummyArgs()
    dargs.model = args.model
    dargs.img_size = args.img_size
    dargs.attn_dropout = args.attn_dropout
    dargs.proj_dropout = args.proj_dropout
    dargs.condition_fusion = args.condition_fusion

    dargs.P_mean = args.P_mean
    dargs.P_std = args.P_std
    dargs.noise_scale = args.noise_scale
    dargs.t_eps = args.t_eps
    dargs.ema_decay1 = args.ema_decay1
    dargs.ema_decay2 = args.ema_decay2
    dargs.sampling_method = args.sampling_method
    dargs.num_sampling_steps = args.num_sampling_steps

    model = Denoiser_CT2D(dargs).to(device)

    ckpt_path = args.ckpt_dir
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model"], strict=True)
    print("Loaded main model weights")

    if "model_ema1" in checkpoint:
        ema_state = checkpoint["model_ema1"]
        missing, unexpected = model.load_state_dict(ema_state, strict=False)
        print("Switched to EMA1 weights for inference")
        if missing:
            print("Missing keys in EMA state_dict:", missing)
        if unexpected:
            print("Unexpected keys in EMA state_dict:", unexpected)

    model.eval()
    return model


def main():
    args = get_args()

    os.makedirs(args.save_root_dir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model = build_model(args)

    device = torch.device(args.device)
    lpips_model = lpips.LPIPS(net='alex').to(device)

    dataset = CT2DPairDatasetPreprocessed(
        non_contrast_dir=args.non_contrast_dir,
        contrast_dir=args.contrast_dir,
        mask_dir=None,
        img_size=args.img_size,
        normalize=True,
        transform=None,
    )

    patient_ids = dataset.patient_pairs
    print(f"Found {len(patient_ids)} patients in val set")
    print(f"Automatic soft-tissue mask: low={args.mask_low}, high={args.mask_high}; excluding air and dense bone")

    all_psnr = []
    all_ssim = []
    all_nrmse = []
    all_mae = []
    all_lpips = []
    all_patients = []

    total_patients = len(patient_ids)
    for patient_idx, patient_id in enumerate(patient_ids, 1):
        print(f"Processing patient {patient_idx}/{total_patients}: {patient_id}")
        patient_save_dir = os.path.join(args.save_root_dir, patient_id)
        os.makedirs(patient_save_dir, exist_ok=True)

        plain_dir = os.path.join(patient_save_dir, "plain")
        gt_dir = os.path.join(patient_save_dir, "ground_truth")
        gen_dir = os.path.join(patient_save_dir, "generated")
        os.makedirs(plain_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)

        patient_slice_indices = []
        for idx, (pid, slice_name) in enumerate(dataset.slices):
            if pid == patient_id:
                patient_slice_indices.append(idx)

        total_slices = len(patient_slice_indices)
        print(f"Patient {patient_id} has {total_slices} slices")

        patient_psnr = []
        patient_ssim = []
        patient_nrmse = []
        patient_mae = []
        patient_lpips = []

        num_batches = (total_slices + args.batch_size - 1) // args.batch_size
        with torch.no_grad():
            for batch_idx in range(num_batches):
                print(f"Batch {batch_idx+1}/{num_batches}")
                start = batch_idx * args.batch_size
                end = start + args.batch_size
                batch_indices = patient_slice_indices[start:end]

                batch_plain = []
                batch_gt = []
                slice_names = []
                for idx in batch_indices:
                    plain, gt = dataset[idx]
                    batch_plain.append(plain)
                    batch_gt.append(gt)
                    slice_names.append(dataset.slices[idx][1])

                batch_plain = torch.stack(batch_plain).to(device)
                batch_gt = torch.stack(batch_gt).to(device)
                batch_gen = model.generate(batch_plain)

                batch_plain = (batch_plain + 1) / 2.0
                batch_gt = (batch_gt + 1) / 2.0
                batch_gen = (batch_gen + 1) / 2.0

                batch_plain = batch_plain.clamp(0, 1)
                batch_gt = batch_gt.clamp(0, 1)
                batch_gen = batch_gen.clamp(0, 1)

                for i in range(len(batch_plain)):
                    plain_np = batch_plain[i].squeeze().cpu().numpy()
                    gt_np = batch_gt[i].squeeze().cpu().numpy()
                    gen_np = batch_gen[i].squeeze().cpu().numpy()
                    slice_name = slice_names[i]

                    cv2.imwrite(os.path.join(plain_dir, slice_name), (plain_np*255).astype(np.uint8))
                    cv2.imwrite(os.path.join(gt_dir, slice_name), (gt_np*255).astype(np.uint8))
                    cv2.imwrite(os.path.join(gen_dir, slice_name), (gen_np*255).astype(np.uint8))

                    tissue_mask = auto_soft_tissue_mask(gt_np, low_thr=args.mask_low, high_thr=args.mask_high)
                    if np.sum(tissue_mask) < 100:
                        continue
                    gt_calc = apply_auto_mask(gt_np, tissue_mask)
                    gen_calc = apply_auto_mask(gen_np, tissue_mask)

                    psnr_val = psnr(gt_calc, gen_calc, data_range=1.0)
                    patient_psnr.append(psnr_val)
                    all_psnr.append(psnr_val)

                    ssim_val = ssim(gt_calc, gen_calc, data_range=1.0)
                    patient_ssim.append(ssim_val)
                    all_ssim.append(ssim_val)

                    nrmse_val = safe_nrmse(gt_calc, gen_calc)
                    patient_nrmse.append(nrmse_val)
                    all_nrmse.append(nrmse_val)

                    mae_val = np.mean(np.abs(gt_calc - gen_calc))
                    patient_mae.append(mae_val)
                    all_mae.append(mae_val)

                    gt_tensor = (batch_gt[i:i+1] * 2 - 1).repeat(1, 3, 1, 1)
                    gen_tensor = (batch_gen[i:i+1] * 2 - 1).repeat(1, 3, 1, 1)
                    lpips_val = lpips_model(gt_tensor, gen_tensor).item()
                    patient_lpips.append(lpips_val)
                    all_lpips.append(lpips_val)

        metric_path = os.path.join(patient_save_dir, "metrics.txt")
        with open(metric_path, 'w', encoding='utf-8') as f:
            f.write(f"Patient ID: {patient_id}\n")
            f.write(f"Total slices: {total_slices}\n")
            f.write(f"Automatic soft-tissue mask thresholds: low={args.mask_low}, high={args.mask_high}\n")
            f.write("="*60 + "\n")
            f.write("[Generated CTA vs reference CTA metrics within the soft-tissue region]\n")
            f.write(f"PSNR:   {get_stats_result(np.array(patient_psnr))}\n")
            f.write(f"SSIM:   {get_stats_result(np.array(patient_ssim))}\n")
            f.write(f"NRMSE:  {get_stats_result(np.array(patient_nrmse))}\n")
            f.write(f"MAE:    {get_stats_result(np.array(patient_mae))}\n")
            f.write(f"LPIPS:  {get_stats_result(np.array(patient_lpips))}\n")
            f.write("="*60 + "\n")

        all_patients.append({
            "id": patient_id,
            "slices": total_slices,
            "psnr": patient_psnr,
            "ssim": patient_ssim,
            "nrmse": patient_nrmse,
            "mae": patient_mae,
            "lpips": patient_lpips
        })

    total_metric_path = os.path.join(args.save_root_dir, "total_metrics.txt")
    with open(total_metric_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Metrics for all patients: generated CTA vs reference CTA\n")
        f.write("Evaluation region: automatically thresholded soft-tissue ROI excluding air and dense bone\n")
        f.write(f"Mask thresholds: low={args.mask_low}, high={args.mask_high}\n")
        f.write("="*70 + "\n\n")

        for p in all_patients:
            f.write(f"Patient: {p['id']} | Slices: {p['slices']}\n")
            f.write("-"*50 + "\n")
            f.write(f"PSNR:   {get_stats_result(np.array(p['psnr']))}\n")
            f.write(f"SSIM:   {get_stats_result(np.array(p['ssim']))}\n")
            f.write(f"NRMSE:  {get_stats_result(np.array(p['nrmse']))}\n")
            f.write(f"MAE:    {get_stats_result(np.array(p['mae']))}\n")
            f.write(f"LPIPS:  {get_stats_result(np.array(p['lpips']))}\n\n")

        f.write("="*70 + "\n")
        f.write("Global summary across valid soft-tissue slices\n")
        f.write("="*70 + "\n\n")
        f.write(f"PSNR:   {get_stats_result(np.array(all_psnr))}\n")
        f.write(f"SSIM:   {get_stats_result(np.array(all_ssim))}\n")
        f.write(f"NRMSE:  {get_stats_result(np.array(all_nrmse))}\n")
        f.write(f"MAE:    {get_stats_result(np.array(all_mae))}\n")
        f.write(f"LPIPS:  {get_stats_result(np.array(all_lpips))}\n")
        f.write("="*70 + "\n")

    print(f"\nInference complete. Metrics saved to:\n{total_metric_path}")


if __name__ == "__main__":
    main()
