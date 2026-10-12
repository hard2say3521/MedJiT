"""Conditional flow-matching denoiser for NCCT-to-CTA generation."""

import torch
import torch.nn as nn
from model_jit_ct2d import JiT_CT2D_models


class Denoiser_CT2D(nn.Module):
    """Generate contrast-enhanced CT slices from non-contrast CT slices."""
    def __init__(self, args):
        super().__init__()
        self.net = JiT_CT2D_models[args.model](
            input_size=args.img_size,
            in_channels=1,
            condition_channels=1,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            condition_fusion=getattr(args, 'condition_fusion', 'add'),
        )
        self.img_size = args.img_size
        
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
    
    def sample_t(self, n: int, device=None):
        """Sample flow-matching time steps."""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)
    
    def forward(self, x_target, x_condition):
        """Compute the loss for target CTA and conditioning NCCT."""
        t = self.sample_t(x_target.size(0), device=x_target.device)
        t = t.view(-1, *([1] * (x_target.ndim - 1)))
        
        e = torch.randn_like(x_target) * self.noise_scale
        z = t * x_target + (1 - t) * e
        
        v = (x_target - z) / (1 - t).clamp_min(self.t_eps)
        
        x_pred = self.net(z, t.flatten(), x_condition)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        
        loss = (v - v_pred) ** 2
        loss = loss.sum(dim=(1, 2, 3)).mean()
        
        return loss
    
    @torch.no_grad()
    def generate(self, x_condition):
        """Generate CTA slices conditioned on an NCCT batch."""
        device = x_condition.device
        bsz = x_condition.size(0)
        
        z = self.noise_scale * torch.randn(
            bsz, 1, self.img_size, self.img_size, 
            device=device
        )
        
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown sampling method: {self.method}")
        
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, x_condition)
        
        z = self._euler_step(z, timesteps[-2], timesteps[-1], x_condition)
        
        return z
    
    @torch.no_grad()
    def _forward_sample(self, z, t, x_condition):
        """Predict the conditional velocity for one sampling step."""
        t_tensor = torch.full((z.size(0),), t, device=z.device)
        x_cond = self.net(z, t_tensor, x_condition)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        
        return v_cond
    
    @torch.no_grad()
    def _euler_step(self, z, t, t_next, x_condition):
        """Take one Euler step."""
        v_pred = self._forward_sample(z, t, x_condition)
        z_next = z + (t_next - t) * v_pred
        return z_next
    
    @torch.no_grad()
    def _heun_step(self, z, t, t_next, x_condition):
        """Take one Heun step."""
        v_pred_t = self._forward_sample(z, t, x_condition)
        
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, x_condition)
        
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        
        return z_next
    
    @torch.no_grad()
    def update_ema(self):
        """Update both exponential moving averages."""
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
