"""Multi-head container to evaluate all 36 probe configurations concurrently."""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.config import (
    NUM_CLASSES,
    PROBE_ARCHITECTURES,
    QUANTIZATION_MODES,
    SLICE_SPECS,
    SliceSpec,
)
from benchmarks.models.latent_head import LatentHead
from benchmarks.models.linear_probe import LinearProbe


class MultiHeadClassifierContainer(nn.Module):
    """Container holding all 36 classification heads (9 slice views x 2 probe

    architectures x 2 quantization states) to enable single-pass parallel training.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.heads = nn.ModuleDict()
        self.head_metadata: Dict[str, Dict[str, Union[str, int, SliceSpec]]] = {}

        # Instantiate all 36 heads
        for slice_name, spec in SLICE_SPECS.items():
            in_ch = spec.num_channels
            for arch in PROBE_ARCHITECTURES:
                for quant in QUANTIZATION_MODES:
                    head_key = f"{slice_name}_{arch}_{quant}"
                    if arch == "linear_probe":
                        head = LinearProbe(in_channels=in_ch, num_classes=num_classes)
                    elif arch == "latent_head":
                        head = LatentHead(in_channels=in_ch, num_classes=num_classes)
                    else:
                        raise ValueError(f"Unknown architecture: {arch}")

                    self.heads[head_key] = head
                    self.head_metadata[head_key] = {
                        "slice_name": slice_name,
                        "slice_spec": spec,
                        "arch": arch,
                        "quant": quant,
                        "in_channels": in_ch,
                    }

    @property
    def head_keys(self) -> List[str]:
        return list(self.heads.keys())

    def forward(
        self,
        y_quantized: torch.Tensor,
        y_continuous: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Parallel forward pass across all 36 heads.

        Args:
            y_quantized: Quantized latent tensor (B, 320, 16, 16)
            y_continuous: Continuous latent tensor (B, 320, 16, 16)

        Returns:
            Dictionary mapping head_key -> logits (B, num_classes)
        """
        logits_dict: Dict[str, torch.Tensor] = {}

        for head_key, head in self.heads.items():
            meta = self.head_metadata[head_key]
            spec: SliceSpec = meta["slice_spec"]
            quant: str = meta["quant"]

            # Select representation source
            source = y_quantized if quant == "quantized" else y_continuous

            # Channel slice
            x_slice = source[:, spec.start_ch:spec.end_ch, :, :]

            # Forward through designated head
            logits_dict[head_key] = head(x_slice)

        return logits_dict

    def compute_loss(
        self,
        logits_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute summed cross-entropy loss across all 36 heads.

        Args:
            logits_dict: Dictionary mapping head_key -> logits (B, 257)
            targets: Class index tensor of shape (B,)

        Returns:
            total_loss: Scalar PyTorch tensor for backpropagation
            per_head_loss: Dict of float loss values for monitoring
        """
        total_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        per_head_loss: Dict[str, float] = {}

        for head_key, logits in logits_dict.items():
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
