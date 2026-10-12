"""Paired 2D slice dataset backed by 3D NIfTI volumes."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


class CT2DPairDataset(Dataset):
    """Extract aligned NCCT and CTA slices from paired volumes."""
    def __init__(
        self,
        non_contrast_dir,
        contrast_dir,
        img_size=256,
        normalize=True,
        hu_min=-1000,
        hu_max=1000,
        slice_axis=2,
        transform=None,
        cache_slices=True,
        window_width=500,
        window_level=70,
        use_window=True,
    ):
        """Configure volume paths, normalization, slicing, and caching."""
        self.non_contrast_dir = non_contrast_dir
        self.contrast_dir = contrast_dir
        self.img_size = img_size
        self.normalize = normalize
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.slice_axis = slice_axis
        self.transform = transform
        self.cache_slices = cache_slices
        self.window_width = window_width
        self.window_level = window_level
        self.use_window = use_window
        
        if self.use_window:
            self.window_min = self.window_level - self.window_width / 2.0
            self.window_max = self.window_level + self.window_width / 2.0
            print(f"Using CT window: width={window_width}, level={window_level}, range=[{self.window_min:.1f}, {self.window_max:.1f}]")
        else:
            print(f"Using HU range: [{hu_min}, {hu_max}]")
        
        non_contrast_all = sorted([
            f for f in os.listdir(non_contrast_dir) 
            if f.endswith(('.nii', '.nii.gz'))
        ])
        contrast_all = sorted([
            f for f in os.listdir(contrast_dir) 
            if f.endswith(('.nii', '.nii.gz'))
        ])
        
        def strip_suffix(filename, suffixes):
            stem = filename.removesuffix('.nii.gz').removesuffix('.nii')
            for suffix in suffixes:
                if stem.endswith(suffix):
                    return stem[:-len(suffix)]
            return None

        contrast_by_patient = {}
        for contrast_file in contrast_all:
            patient_id = strip_suffix(contrast_file, ('_enhance', '_contrast'))
            if patient_id is not None:
                contrast_by_patient[patient_id] = contrast_file

        self.non_contrast_files = []
        self.contrast_files = []
        
        for nc_file in non_contrast_all:
            patient_id = strip_suffix(nc_file, ('_plain', '_non_contrast'))
            if patient_id in contrast_by_patient:
                self.non_contrast_files.append(nc_file)
                self.contrast_files.append(contrast_by_patient[patient_id])
        
        assert len(self.non_contrast_files) == len(self.contrast_files), \
            f"NCCT/CTA file-count mismatch: {len(self.non_contrast_files)} vs {len(self.contrast_files)}"
        assert len(self.non_contrast_files) > 0, "No matched volume pairs found"
        
        self.slices = []
        for file_idx, filename in enumerate(self.non_contrast_files):
            non_contrast_path = os.path.join(non_contrast_dir, filename)
            contrast_path = os.path.join(contrast_dir, self.contrast_files[file_idx])
            non_contrast_shape = nib.load(non_contrast_path).shape
            contrast_shape = nib.load(contrast_path).shape
            if non_contrast_shape[slice_axis] != contrast_shape[slice_axis]:
                raise ValueError(
                    f"Slice-count mismatch for {filename}: "
                    f"{non_contrast_shape[slice_axis]} vs {contrast_shape[slice_axis]}"
                )

            num_slices = non_contrast_shape[slice_axis]
            
            for slice_idx in range(num_slices):
                self.slices.append((file_idx, slice_idx))
        
        print(f"Found {len(self.non_contrast_files)} CT pairs and {len(self.slices)} slices")
        
        self.cached_slices = {}
        if cache_slices:
            print("Caching slices in memory...")
            self._cache_all_slices()
    
    def _cache_all_slices(self):
        """Cache all paired slices in memory."""
        for idx in range(len(self.slices)):
            file_idx, slice_idx = self.slices[idx]
            non_contrast_path = os.path.join(
                self.non_contrast_dir, 
                self.non_contrast_files[file_idx]
            )
            contrast_path = os.path.join(
                self.contrast_dir,
                self.contrast_files[file_idx]
            )
            
            non_contrast_slice, contrast_slice = self._load_slice_pair(
                non_contrast_path, contrast_path, slice_idx
            )
            self.cached_slices[idx] = (non_contrast_slice, contrast_slice)
        
        print(f"Cached {len(self.cached_slices)} slice pairs")
    
    def _load_slice_pair(self, non_contrast_path, contrast_path, slice_idx):
        """Load one aligned slice pair."""
        non_contrast_nii = nib.load(non_contrast_path)
        contrast_nii = nib.load(contrast_path)
        
        non_contrast_vol = non_contrast_nii.get_fdata().astype(np.float32)
        contrast_vol = contrast_nii.get_fdata().astype(np.float32)
        
        if self.slice_axis == 0:
            non_contrast_slice = non_contrast_vol[slice_idx, :, :]
            contrast_slice = contrast_vol[slice_idx, :, :]
        elif self.slice_axis == 1:
            non_contrast_slice = non_contrast_vol[:, slice_idx, :]
            contrast_slice = contrast_vol[:, slice_idx, :]
        else:
            non_contrast_slice = non_contrast_vol[:, :, slice_idx]
            contrast_slice = contrast_vol[:, :, slice_idx]
        
        return non_contrast_slice, contrast_slice
    
    def apply_window(self, slice_data):
        """Clip HU values to the configured intensity window."""
        if self.use_window:
            slice_data = np.clip(slice_data, self.window_min, self.window_max)
        else:
            slice_data = np.clip(slice_data, self.hu_min, self.hu_max)
        
        return slice_data
    
    def normalize_hu(self, slice_data):
        """Window and normalize HU values."""
        slice_data = self.apply_window(slice_data)
        
        if self.use_window:
            data_min = self.window_min
            data_max = self.window_max
        else:
            data_min = self.hu_min
            data_max = self.hu_max
        
        slice_data = (slice_data - data_min) / (data_max - data_min)
        
        if self.normalize:
            slice_data = slice_data * 2.0 - 1.0
        
        return slice_data
    
    def resize_slice(self, slice_data, target_size):
        """Resize a slice while preserving its aspect ratio."""
        from scipy.ndimage import zoom
        
        h, w = slice_data.shape
        target_h, target_w = target_size
        
        scale_h = target_h / h
        scale_w = target_w / w
        scale = min(scale_h, scale_w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        zoom_factors = [new_h / h, new_w / w]
        slice_data = zoom(slice_data, zoom_factors, order=1)
        
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
    
    def preprocess_slice(self, slice_data):
        """Normalize, resize, and pad a slice without cropping."""
        slice_data = self.normalize_hu(slice_data)
        
        h, w = slice_data.shape
        target_size = (self.img_size, self.img_size)
        
        if (h, w) != target_size:
            slice_data = self.resize_slice(slice_data, target_size)
            slice_data = self.pad_slice(slice_data, target_size, pad_mode='edge')
        
        assert slice_data.shape == target_size, \
            f"Slice-size mismatch: {slice_data.shape} vs {target_size}"
        
        return slice_data
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        if self.cache_slices and idx in self.cached_slices:
            non_contrast_slice, contrast_slice = self.cached_slices[idx]
        else:
            file_idx, slice_idx = self.slices[idx]
            non_contrast_path = os.path.join(
                self.non_contrast_dir, 
                self.non_contrast_files[file_idx]
            )
            contrast_path = os.path.join(
                self.contrast_dir,
                self.contrast_files[file_idx]
            )
            non_contrast_slice, contrast_slice = self._load_slice_pair(
                non_contrast_path, contrast_path, slice_idx
            )
        
        non_contrast_slice = self.preprocess_slice(non_contrast_slice)
        contrast_slice = self.preprocess_slice(contrast_slice)
        
        non_contrast_slice = torch.from_numpy(non_contrast_slice).float()
        contrast_slice = torch.from_numpy(contrast_slice).float()
        
        non_contrast_slice = non_contrast_slice.unsqueeze(0)
        contrast_slice = contrast_slice.unsqueeze(0)
        
        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            non_contrast_slice = self.transform(non_contrast_slice)
            torch.manual_seed(seed)
            contrast_slice = self.transform(contrast_slice)
        
        return non_contrast_slice, contrast_slice


class CT2DTransform:
    """Apply paired flips and right-angle rotations."""
    def __init__(self, flip_prob=0, rotate_prob=0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
    
    def __call__(self, slice_data):
        """Transform a slice with shape (1, H, W)."""
        if np.random.rand() < self.flip_prob:
            slice_data = torch.flip(slice_data, dims=[2])
        
        if np.random.rand() < self.flip_prob:
            slice_data = torch.flip(slice_data, dims=[1])
        
        if np.random.rand() < self.rotate_prob:
            k = np.random.randint(0, 4)
            slice_data = torch.rot90(slice_data, k=k, dims=[1, 2])
        
        return slice_data

