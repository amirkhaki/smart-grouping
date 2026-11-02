"""Multi-head container to evaluate all 36 probe configurations concurrently."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.config import (
    LATENT_HEAD_STEM_CHANNELS,
    NUM_CLASSES,
    PROBE_ARCHITECTURES,
    QUANTIZATION_MODES,
    SLICE_SPECS,
    SliceSpec,
)
from benchmarks.models.latent_head import LatentHead
from benchmarks.models.linear_probe import LinearProbe


class MultiHeadClassifierContainer(nn.Module):
    """Container holding classification heads across slice views, architectures,

    and quantization states with support for dynamic per-head early stopping and freezing.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        architectures: Optional[List[str]] = None,
        views: Optional[List[str]] = None,
        quant_modes: Optional[List[str]] = None,
        stem_channels: Optional[Union[int, List[int]]] = None,
        custom_channel_subsets: Optional[Dict[str, List[int]]] = None,
        native_width: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.native_width = native_width
        self.heads = nn.ModuleDict()
        self.head_metadata: Dict[str, Dict[str, Any]] = {}

        selected_archs = architectures or PROBE_ARCHITECTURES
        selected_quants = quant_modes or QUANTIZATION_MODES

        if stem_channels is None:
            stem_widths = [LATENT_HEAD_STEM_CHANNELS]
        elif isinstance(stem_channels, int):
            stem_widths = [stem_channels]
        else:
            stem_widths = list(stem_channels)

        # 1. Custom channel subsets (e.g. least-sparse channel selections)
        if custom_channel_subsets is not None:
            for subset_name, indices in custom_channel_subsets.items():
                in_ch = len(indices)
                for arch in selected_archs:
                    for quant in selected_quants:
                        if arch == "linear_probe":
                            head_key = f"{subset_name}_{arch}_{quant}"
                            head = LinearProbe(in_channels=in_ch, num_classes=num_classes)
                            self.heads[head_key] = head
                            self.head_metadata[head_key] = {
                                "slice_name": subset_name,
                                "slice_spec": None,
                                "channel_indices": indices,
                                "arch": arch,
                                "base_arch": "linear_probe",
                                "quant": quant,
                                "in_channels": in_ch,
                                "stem_channels": None,
                                "native_width": False,
                                "num_params": sum(p.numel() for p in head.parameters()),
                            }
                        elif arch.startswith("latent_head_w"):
                            stem_w = int(arch.split("_w")[-1])
                            head_key = f"{subset_name}_{arch}_{quant}"
                            head = LatentHead(in_channels=in_ch, stem_channels=stem_w, num_classes=num_classes)
                            self.heads[head_key] = head
                            self.head_metadata[head_key] = {
                                "slice_name": subset_name,
                                "slice_spec": None,
                                "channel_indices": indices,
                                "arch": arch,
                                "base_arch": "latent_head",
                                "quant": quant,
                                "in_channels": in_ch,
                                "stem_channels": stem_w,
                                "native_width": False,
                                "num_params": sum(p.numel() for p in head.parameters()),
                            }
                        elif arch == "latent_head":
                            if native_width:
                                stem_w = in_ch
                                head_key = f"{subset_name}_{arch}_native_{quant}"
                                arch_label = "latent_head_native"
                                head = LatentHead(in_channels=in_ch, stem_channels=stem_w, num_classes=num_classes)
                                self.heads[head_key] = head
                                self.head_metadata[head_key] = {
                                    "slice_name": subset_name,
                                    "slice_spec": None,
                                    "channel_indices": indices,
                                    "arch": arch_label,
                                    "base_arch": "latent_head",
                                    "quant": quant,
                                    "in_channels": in_ch,
                                    "stem_channels": stem_w,
                                    "native_width": True,
                                    "num_params": sum(p.numel() for p in head.parameters()),
                                }
                            else:
                                for stem_w in stem_widths:
                                    if stem_widths == [LATENT_HEAD_STEM_CHANNELS]:
                                        head_key = f"{subset_name}_{arch}_{quant}"
                                        arch_label = arch
                                    else:
                                        head_key = f"{subset_name}_{arch}_w{stem_w}_{quant}"
                                        arch_label = f"latent_head_w{stem_w}"

                                    head = LatentHead(in_channels=in_ch, stem_channels=stem_w, num_classes=num_classes)
                                    self.heads[head_key] = head
                                    self.head_metadata[head_key] = {
                                        "slice_name": subset_name,
                                        "slice_spec": None,
                                        "channel_indices": indices,
                                        "arch": arch_label,
                                        "base_arch": "latent_head",
                                        "quant": quant,
                                        "in_channels": in_ch,
                                        "stem_channels": stem_w,
                                        "native_width": False,
                                        "num_params": sum(p.numel() for p in head.parameters()),
                                    }

        # 2. Standard contiguous slice views
        if views is not None:
            selected_views = views
        elif custom_channel_subsets is None:
            selected_views = list(SLICE_SPECS.keys())
        else:
            selected_views = []

        for slice_name in selected_views:
            if slice_name not in SLICE_SPECS:
                raise ValueError(f"Unknown slice view: {slice_name}. Valid views: {list(SLICE_SPECS.keys())}")
            spec = SLICE_SPECS[slice_name]
            in_ch = spec.num_channels

            for arch in selected_archs:
                for quant in selected_quants:
                    if arch == "linear_probe":
                        head_key = f"{slice_name}_{arch}_{quant}"
                        head = LinearProbe(in_channels=in_ch, num_classes=num_classes)
                        self.heads[head_key] = head
                        self.head_metadata[head_key] = {
                            "slice_name": slice_name,
                            "slice_spec": spec,
                            "channel_indices": None,
                            "arch": arch,
                            "base_arch": "linear_probe",
                            "quant": quant,
                            "in_channels": in_ch,
                            "stem_channels": None,
                            "native_width": False,
                            "num_params": sum(p.numel() for p in head.parameters()),
                        }
                    elif arch.startswith("latent_head_w"):
                        # Specific stem width requested via architecture name
                        stem_w = int(arch.split("_w")[-1])
                        head_key = f"{slice_name}_{arch}_{quant}"
                        head = LatentHead(in_channels=in_ch, stem_channels=stem_w, num_classes=num_classes)
                        self.heads[head_key] = head
                        self.head_metadata[head_key] = {
                            "slice_name": slice_name,
                            "slice_spec": spec,
                            "channel_indices": None,
                            "arch": arch,
                            "base_arch": "latent_head",
                            "quant": quant,
                            "in_channels": in_ch,
                            "stem_channels": stem_w,
                            "native_width": False,
                            "num_params": sum(p.numel() for p in head.parameters()),
                        }
                    elif arch == "latent_head":
                        if native_width:
                            stem_w = in_ch
                            head_key = f"{slice_name}_{arch}_native_{quant}"
                            arch_label = "latent_head_native"
                            head = LatentHead(in_channels=in_ch, stem_channels=stem_w, num_classes=num_classes)
                            self.heads[head_key] = head
                            self.head_metadata[head_key] = {
                                "slice_name": slice_name,
                                "slice_spec": spec,
                                "channel_indices": None,
                                "arch": arch_label,
                                "base_arch": "latent_head",
                                "quant": quant,
                                "in_channels": in_ch,
                                "stem_channels": stem_w,
                                "native_width": True,
                                "num_params": sum(p.numel() for p in head.parameters()),
                            }
                        else:
                            for stem_w in stem_widths:
                                if stem_widths == [LATENT_HEAD_STEM_CHANNELS]:
                                    head_key = f"{slice_name}_{arch}_{quant}"
                                    arch_label = arch
                                else:
                                    head_key = f"{slice_name}_{arch}_w{stem_w}_{quant}"
                                    arch_label = f"latent_head_w{stem_w}"

                                head = LatentHead(in_channels=in_ch, stem_channels=stem_w, num_classes=num_classes)
                                self.heads[head_key] = head
                                self.head_metadata[head_key] = {
                                    "slice_name": slice_name,
                                    "slice_spec": spec,
                                    "channel_indices": None,
                                    "arch": arch_label,
                                    "base_arch": "latent_head",
                                    "quant": quant,
                                    "in_channels": in_ch,
                                    "stem_channels": stem_w,
                                    "native_width": False,
                                    "num_params": sum(p.numel() for p in head.parameters()),
                                }
                    else:
                        raise ValueError(f"Unknown architecture: {arch}")

        # Dynamic early stopping tracking: active head keys
        self.active_head_keys: List[str] = list(self.heads.keys())

    @property
    def head_keys(self) -> List[str]:
        return list(self.heads.keys())

    def freeze_head(self, head_key: str):
        """Freeze a head once it has converged, removing it from active training."""
        if head_key in self.active_head_keys:
            self.active_head_keys.remove(head_key)
            # Freeze parameters so PyTorch does not track gradients
            for p in self.heads[head_key].parameters():
                p.requires_grad = False

    def is_active(self, head_key: str) -> bool:
        return head_key in self.active_head_keys

    def all_frozen(self) -> bool:
        """Return True if all heads have converged and been frozen."""
        return len(self.active_head_keys) == 0

    def forward(
        self,
        y_quantized: torch.Tensor,
        y_continuous: torch.Tensor,
        all_heads: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass across active heads (or all heads if all_heads=True).

        Args:
            y_quantized: Quantized latent tensor (B, 320, 16, 16)
            y_continuous: Continuous latent tensor (B, 320, 16, 16)
            all_heads: If True, evaluates all heads regardless of frozen status.

        Returns:
            Dictionary mapping head_key -> logits (B, num_classes)
        """
        logits_dict: Dict[str, torch.Tensor] = {}
        target_keys = list(self.heads.keys()) if all_heads else self.active_head_keys

        for head_key in target_keys:
            head = self.heads[head_key]
            meta = self.head_metadata[head_key]
            quant: str = meta["quant"]

            # Select representation source
            source = y_quantized if quant == "quantized" else y_continuous

            # Channel selection (custom indices or contiguous slice)
            if meta.get("channel_indices") is not None:
                indices = meta["channel_indices"]
                x_slice = source[:, indices, :, :]
            else:
                spec: SliceSpec = meta["slice_spec"]
                x_slice = source[:, spec.start_ch:spec.end_ch, :, :]

            # Forward through designated head
            logits_dict[head_key] = head(x_slice)

        return logits_dict

    def compute_loss(
        self,
        logits_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute summed cross-entropy loss across active heads.

        Args:
            logits_dict: Dictionary mapping head_key -> logits (B, num_classes)
            targets: Class index tensor of shape (B,)
            label_smoothing: Float in [0.0, 1.0) specifying label smoothing regularizer.

        Returns:
            total_loss: Scalar PyTorch tensor for backpropagation
            per_head_loss: Dict of float loss values for monitoring
        """
        if not logits_dict:
            return torch.tensor(0.0, device=targets.device, requires_grad=True), {}

        total_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        per_head_loss: Dict[str, float] = {}

        for head_key, logits in logits_dict.items():
            if label_smoothing > 0.0:
                loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
            else:
                loss = F.cross_entropy(logits, targets)
            total_loss = total_loss + loss
            per_head_loss[head_key] = loss.item()

        return total_loss, per_head_loss

    @staticmethod
    def compute_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        topk: Tuple[int, ...] = (1, 5),
    ) -> Dict[str, float]:
        """Compute top-k accuracy percentages for a single set of logits."""
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res: Dict[str, float] = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f"top{k}"] = (correct_k.item() / batch_size) * 100.0
        return res
