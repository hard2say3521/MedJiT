import os
import argparse
import json
import random
import shutil
import nibabel as nib
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def list_nii_files(directory: str) -> List[str]:
    """List NIfTI files in a directory."""
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return []
    files = sorted([f for f in os.listdir(directory) if f.endswith(('.nii', '.nii.gz'))])
    logger.info(f"Found {len(files)} NIfTI files in {directory}")
    return files

def extract_patient_id(filename: str) -> str:
    """Extract a patient ID from a NIfTI filename."""
    name = filename.replace('.nii.gz', '').replace('.nii', '')
    if '_' in name:
        return name.split('_')[0]
    return name

def find_exact_file(directory: str, patient_id: str) -> Optional[str]:
    """Find an exact patient match without prefix collisions."""
    if not os.path.exists(directory):
        return None
    
    files = os.listdir(directory)
    candidates = []
    
    for f in files:
        if not f.endswith(('.nii', '.nii.gz')):
            continue
        file_pid = extract_patient_id(f)
        if file_pid == patient_id:
            candidates.append(f)
    
    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return candidates[0]
    else:
        logger.warning(f"Multiple files match patient {patient_id} in {directory}: {candidates}")
        for c in candidates:
            if '_' in c:
                return c
        return candidates[0]

def split_dataset_randomly(
    patient_ids: List[str],
    seed: int,
    train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    """Randomly split patient IDs into training and validation sets."""
    rng = random.Random(seed)
    
    shuffled_ids = list(patient_ids)
    rng.shuffle(shuffled_ids)
    
    split_idx = int(len(shuffled_ids) * train_ratio)
    
    train_ids = shuffled_ids[:split_idx]
    val_ids = shuffled_ids[split_idx:]
    
    logger.info(f"Random {train_ratio*10:g}:{(1-train_ratio)*10:g} split complete:")
    logger.info(f"  Train ({train_ratio*100:.1f}%) = {len(train_ids)} patients")
    logger.info(f"  Val   ({(1-train_ratio)*100:.1f}%) = {len(val_ids)} patients")
    
    return train_ids, val_ids

def process_nii_to_slices(
    nii_path: str, 
    output_dir: str, 
    patient_id: str, 
    slice_axis: int
) -> Tuple[int, Tuple[int, int, int]]:
    """Convert one NIfTI volume to PNG slices."""
    if not os.path.exists(nii_path):
        logger.error(f"File not found: {nii_path}")
        return 0, (0, 0, 0)
    
    try:
        nii = nib.load(nii_path)
        volume = nii.get_fdata().astype(np.float32)
        original_shape = volume.shape
        num_slices = volume.shape[slice_axis]

        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)

        v_min, v_max = np.min(volume), np.max(volume)
        if v_max - v_min > 1e-6:
            volume = (volume - v_min) / (v_max - v_min)
        else:
            logger.warning(f"Volume {nii_path} is constant: {v_min}")
        volume = (np.clip(volume, 0, 1) * 255).astype(np.uint8)

        for slice_idx in range(num_slices):
            if slice_axis == 0: 
                slice_data = volume[slice_idx, :, :]
            elif slice_axis == 1: 
                slice_data = volume[:, slice_idx, :]
            else: 
                slice_data = volume[:, :, slice_idx]

            if slice_data.ndim == 3: 
                slice_data = slice_data[:, :, 0]
                logger.warning(f"Slice {slice_idx} is 3D; using the first channel")
            
            img = Image.fromarray(slice_data.astype(np.uint8), mode='L')
            img.save(os.path.join(patient_output_dir, f"slice_{slice_idx:04d}.png"))
        
        return num_slices, original_shape
        
    except Exception as e:
        logger.error(f"Failed to process {nii_path}: {e}")
        return 0, (0, 0, 0)

def validate_sequences_consistency(
    src_dir: str,
    patient_id: str,
    sequences: List[str],
    slice_axis: int
) -> Dict[str, Any]:
    """Validate shape and slice-count consistency across sequences."""
    result = {
        'valid': True,
        'shapes': {},
        'num_slices': {},
        'missing': []
    }
    
    for seq in sequences:
        seq_dir = os.path.join(src_dir, seq)
        filename = find_exact_file(seq_dir, patient_id)
        
        if filename is None:
            result['valid'] = False
            result['missing'].append(seq)
            continue
        
        nii_path = os.path.join(seq_dir, filename)
        try:
            nii = nib.load(nii_path)
            shape = nii.shape
            result['shapes'][seq] = shape
            result['num_slices'][seq] = shape[slice_axis]
        except Exception as e:
            logger.error(f"Failed to read {nii_path}: {e}")
            result['valid'] = False
    
    if len(set(result['num_slices'].values())) > 1:
        logger.warning(f"Slice-count mismatch for patient {patient_id}: {result['num_slices']}")
        result['valid'] = False
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Convert paired volumes to slices with a random 8:2 split')
    parser.add_argument('--src', type=str, required=True, help='Input root containing enhance and plain directories')
    parser.add_argument('--out', type=str, required=True, help='Output root directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--slice_axis', type=int, default=2, choices=[0, 1, 2], help='Slice axis')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output directory')
    parser.add_argument('--check_consistency', action='store_true', help='Check sequence consistency before conversion')

    args = parser.parse_args()

    sequences = ['enhance', 'plain']
    output_names = {'enhance': 'enhance_2d', 'plain': 'plain_2d'}

    if os.path.exists(args.out):
        if args.overwrite: 
            logger.warning(f"Removing existing output directory: {args.out}")
            shutil.rmtree(args.out)
        else: 
            raise SystemExit("Output directory already exists; use --overwrite to replace it")
    
    enhance_dir = os.path.join(args.src, 'enhance')
    if not os.path.exists(enhance_dir):
        raise SystemExit(f"Enhance directory not found: {enhance_dir}")
    
    nii_files = list_nii_files(enhance_dir)
    if not nii_files:
        raise SystemExit(f"No NIfTI files found in the enhance directory: {enhance_dir}")
    
    enhance_ids = {extract_patient_id(f) for f in nii_files}
    plain_dir = os.path.join(args.src, 'plain')
    plain_ids = {extract_patient_id(f) for f in list_nii_files(plain_dir)}
    patient_ids = sorted(enhance_ids & plain_ids)
    if not patient_ids:
        raise SystemExit("No matched patient IDs found in enhance and plain directories")
    unmatched = len(enhance_ids ^ plain_ids)
    if unmatched:
        logger.warning(f"Skipping {unmatched} unmatched patient IDs")
    logger.info(f"Found {len(patient_ids)} matched patient IDs")
    
    if args.check_consistency:
        logger.info("Checking sequence consistency...")
        inconsistent = []
        
        for pid in tqdm(patient_ids, desc="Checking consistency"):
            validation = validate_sequences_consistency(args.src, pid, sequences, args.slice_axis)
            if not validation['valid']:
                inconsistent.append((pid, validation))
        
        if inconsistent:
            logger.warning(f"Found {len(inconsistent)} inconsistent patients")
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, 'inconsistent_patients.json'), 'w') as f:
                json.dump(inconsistent, f, indent=2)
    
    logger.info("\n" + "="*50)
    logger.info("Randomly splitting the dataset into training and validation sets at an 8:2 ratio")
    logger.info("="*50)
    
    train_ids, val_ids = split_dataset_randomly(patient_ids, args.seed, train_ratio=0.8)
    splits = {'train': train_ids, 'val': val_ids}
    
    manifest = {
        'num_patients': {
            'train': len(train_ids),
            'val': len(val_ids)
        },
        'num_slices': {
            'train': 0,
            'val': 0
        },
        'missing_files': defaultdict(list),
        'processing_errors': defaultdict(list)
    }

    for split_name, ids in splits.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {split_name}: {len(ids)} patients")
        logger.info(f"{'='*50}")
        
        for pid in tqdm(ids, desc=f"Processing {split_name}"):
            patient_slices = {}
            
            for seq in sequences:
                seq_src_dir = os.path.join(args.src, seq)
                filename = find_exact_file(seq_src_dir, pid)
                
                if filename is None:
                    logger.warning(f"Patient {pid} is missing the {seq} sequence")
                    manifest['missing_files'][split_name].append(f"{pid}/{seq}")
                    continue
                
                nii_path = os.path.join(seq_src_dir, filename)
                output_dir = os.path.join(args.out, split_name, output_names[seq])
                
                num_slices, shape = process_nii_to_slices(
                    nii_path, output_dir, pid, args.slice_axis
                )
                
                if num_slices == 0:
                    manifest['processing_errors'][split_name].append(f"{pid}/{seq}")
                
                patient_slices[seq] = num_slices
            
            if len(set(patient_slices.values())) > 1:
                logger.warning(f"Sequence slice-count mismatch for patient {pid}: {patient_slices}")
            
            if 'enhance' in patient_slices:
                manifest['num_slices'][split_name] += patient_slices['enhance']

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, 'split_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "="*70)
    logger.info("Processing complete")
    logger.info("="*70)
    logger.info(f"Output layout: {args.out}/[train|val]/[enhance_2d|plain_2d]/[PatientID]/*.png")
    logger.info("-"*70)
    logger.info("Dataset split summary:")
    logger.info(f"  train: {len(train_ids):4d} patients, {manifest['num_slices']['train']} slices")
    logger.info(f"  val:   {len(val_ids):4d} patients, {manifest['num_slices']['val']} slices")
    
    total_missing = sum(len(v) for v in manifest['missing_files'].values())
    total_errors = sum(len(v) for v in manifest['processing_errors'].values())
    
    if total_missing > 0:
        logger.warning(f"\n{total_missing} paired sequence files are missing")
    if total_errors > 0:
        logger.warning(f"Failed to process {total_errors} volume files")
    
    logger.info(f"\nManifest saved to: {manifest_path}")
    logger.info("="*70)

if __name__ == '__main__':
    main()
