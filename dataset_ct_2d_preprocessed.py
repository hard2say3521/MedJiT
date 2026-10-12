"""Paired preprocessed PNG dataset for 2D NCCT-to-CTA training."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from scipy.ndimage import zoom


class CT2DPairDatasetPreprocessed(Dataset):
    """Load aligned NCCT and CTA slices with optional masks."""
    def __init__(
        self,
        non_contrast_dir,
        contrast_dir,
        mask_dir=None,
        img_size=256,
        normalize=True,
        transform=None,
    ):
        """Initialize paths, image size, normalization, and augmentation."""
        self.non_contrast_dir = non_contrast_dir
        self.contrast_dir = contrast_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.normalize = normalize
        self.transform = transform
        
        def get_patient_dirs(directory):
            return {d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))}
        
        non_contrast_patients = get_patient_dirs(non_contrast_dir)
        contrast_patients = get_patient_dirs(contrast_dir)
        
        if mask_dir is not None and os.path.exists(mask_dir):
            mask_patients = get_patient_dirs(mask_dir)
            self.patient_pairs = sorted(list(non_contrast_patients & contrast_patients & mask_patients))
            self.use_mask = True
            print(f"Mask enabled: found {len(self.patient_pairs)} patients shared by plain, enhance, and mask")
        else:
            self.patient_pairs = sorted(list(non_contrast_patients & contrast_patients))
            self.use_mask = False
            print(f"Mask disabled: found {len(self.patient_pairs)} patients shared by plain and enhance")
        
        assert len(self.patient_pairs) > 0, "No matched patient directories found"
        
        self.slices = []
        for patient_id in self.patient_pairs:
            nc_patient_dir = os.path.join(non_contrast_dir, patient_id)
            c_patient_dir = os.path.join(contrast_dir, patient_id)
            
            def get_png_slices(directory):
                return {f for f in os.listdir(directory) if f.endswith('.png')}
            
            nc_slices = get_png_slices(nc_patient_dir)
            c_slices = get_png_slices(c_patient_dir)
            
            if self.use_mask:
                m_patient_dir = os.path.join(mask_dir, patient_id)
                m_slices = get_png_slices(m_patient_dir)
                common_slices = sorted(list(nc_slices & c_slices & m_slices))
            else:
                common_slices = sorted(list(nc_slices & c_slices))
            
            for slice_file in common_slices:
                self.slices.append((patient_id, slice_file))
        
        print(f"Collected {len(self.slices)} matched slice pairs")
    
    def resize_slice(self, slice_data, target_size):
        """Resize a slice while preserving its aspect ratio."""
        h, w = slice_data.shape
        target_h, target_w = target_size
        
        scale_h = target_h / h
        scale_w = target_w / w
        scale = min(scale_h, scale_w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        slice_data = zoom(slice_data, (new_h / h, new_w / w), order=1)
        
        return slice_data
    
    def pad_slice(self, slice_data, target_size, pad_mode='edge'):
        """Pad a slice to the target size."""
        h, w = slice_data.shape
        target_h, target_w = target_size
        
        pad_top = max(0, (target_h - h) // 2)
        pad_bottom = max(0, target_h - h - pad_top)
        pad_left = max(0, (target_w - w) // 2)
        pad_right = max(0, target_w - w - pad_left)
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            slice_data = np.pad(
                slice_data,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode=pad_mode
            )
        
        return slice_data
    
    def resize_and_pad(self, slice_data, target_size):
        """Resize and pad a slice to the target size."""
        h, w = slice_data.shape
        if (h, w) != target_size:
            slice_data = self.resize_slice(slice_data, target_size)
            slice_data = self.pad_slice(slice_data, target_size, pad_mode='edge')
        return slice_data
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        patient_id, slice_filename = self.slices[idx]
        
        nc_path = os.path.join(self.non_contrast_dir, patient_id, slice_filename)
        c_path = os.path.join(self.contrast_dir, patient_id, slice_filename)
        
        non_contrast_img = cv2.imread(nc_path, cv2.IMREAD_GRAYSCALE)
        contrast_img = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        if non_contrast_img is None or contrast_img is None:
            raise OSError(f"Failed to read paired slices: {nc_path}, {c_path}")
        non_contrast_img = non_contrast_img.astype(np.float32)
        contrast_img = contrast_img.astype(np.float32)
        
        target_size = (self.img_size, self.img_size)
        non_contrast_img = self.resize_and_pad(non_contrast_img, target_size)
        contrast_img = self.resize_and_pad(contrast_img, target_size)
        
        if self.use_mask:
            m_path = os.path.join(self.mask_dir, patient_id, slice_filename)
            mask_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise OSError(f"Failed to read mask: {m_path}")
            mask_img = self.resize_and_pad(mask_img, target_size)
            
            binary_mask = (mask_img > 127).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary_mask)
            
            centers = []
            for i in range(1, min(num_labels, 3)):
                component_coords = np.column_stack(np.where(labels == i))
                if len(component_coords) > 0:
                    center = np.mean(component_coords, axis=0)
                    center = tuple(map(int, center[::-1]))
                    centers.append(center)
            
            if centers:
                dilated_mask = np.zeros_like(mask_img)
                for center in centers:
                    cv2.circle(dilated_mask, center, 30, 255, -1)
                mask_img = np.maximum(mask_img, dilated_mask)
            
            mask_weight = (mask_img > 127).astype(np.float32)
            non_contrast_img = non_contrast_img * mask_weight
            contrast_img = contrast_img * mask_weight
        
        non_contrast_img = non_contrast_img / 255.0
        contrast_img = contrast_img / 255.0
        
        non_contrast_img = torch.from_numpy(non_contrast_img).float()
        contrast_img = torch.from_numpy(contrast_img).float()
        
        if self.normalize:
            non_contrast_img = non_contrast_img * 2.0 - 1.0
            contrast_img = contrast_img * 2.0 - 1.0
        
        non_contrast_img = non_contrast_img.unsqueeze(0)
        contrast_img = contrast_img.unsqueeze(0)
        
        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            non_contrast_img = self.transform(non_contrast_img)
            torch.manual_seed(seed)
            contrast_img = self.transform(contrast_img)
        
        return non_contrast_img, contrast_img


class CT2DTransform:
    """Apply paired flips and right-angle rotations."""
    def __init__(self, flip_prob=0.5, rotate_prob=0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
    
    def __call__(self, slice_data):
        """Transform a slice with shape (1, H, W)."""
        if torch.rand(1).item() < self.flip_prob:
            slice_data = torch.flip(slice_data, dims=[2])
        
        if torch.rand(1).item() < self.flip_prob:
            slice_data = torch.flip(slice_data, dims=[1])
        
        if torch.rand(1).item() < self.rotate_prob:
            k = torch.randint(0, 4, (1,)).item()
            slice_data = torch.rot90(slice_data, k=k, dims=[1, 2])
        
        return slice_data

