"""Gated multi-signal fusion probe."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.utils.reproducibility import set_seed


def prepare_hidden_features(
    hidden_states_dict: Dict[int, torch.Tensor],
    layers: Optional[List[int]] = None,
) -> torch.Tensor:
    """Average a few hidden-state layers into one feature matrix."""
    if layers is None:
        layers = [8, 12, 15]
    chosen_layers = [hidden_states_dict[li].float() for li in layers]
    return sum(chosen_layers) / float(len(layers))


class GatedFusionDataset(Dataset):
    """One row: pooled hidden, flat attention eigs, 3 logit scalars, label."""

    def __init__(
        self,
        hidden: torch.Tensor,
        attn: torch.Tensor,
        logit: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.h = hidden.float()
        self.attn = attn.float()
        self.logit = logit.float()
        self.label = labels.float()

    def __len__(self) -> int:
        return self.h.shape[0]

    def __getitem__(self, i: int):
        return self.h[i], self.attn[i], self.logit[i], self.label[i]


class GatedFusionProbe(nn.Module):
    """Encoders for three streams + input-dependent softmax gates + head."""

    def __init__(
        self,
        hidden_dim: int = 2048,
        attn_dim: int = 80,
        logit_dim: int = 3,
        dropout: float = 0.0,
        enc_hidden: int = 128,
    ) -> None:
        super().__init__()
        d_h, d_a, d_l = hidden_dim, attn_dim, logit_dim
        self.h_enc = nn.Sequential(
            nn.Linear(d_h, enc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, 64),
        )
        self.a_enc = nn.Sequential(
            nn.Linear(d_a, enc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, 64),
        )
        self.l_enc = nn.Sequential(
            nn.Linear(d_l, enc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, 64),
        )
        self.gate = nn.Linear(d_h + d_a + d_l, 3)
        self.clf = nn.Linear(64, 1)  # sigmoid in forward

    def forward(
        self,
        hidden: torch.Tensor,
        attn: torch.Tensor,
        logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e_h = self.h_enc(hidden)
        e_a = self.a_enc(attn)
        e_l = self.l_enc(logit)
        x = torch.cat([hidden, attn, logit], dim=-1)
        g = F.softmax(self.gate(x), dim=-1)
        fused = g[:, 0:1] * e_h + g[:, 1:2] * e_a + g[:, 2:3] * e_l
        logit_b = self.clf(fused).squeeze(-1)
        p = torch.sigmoid(logit_b)
        return p, g


def train_gated_fusion(
    model: GatedFusionProbe,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    device: Optional[torch.device] = None,
    seed: int = 42,
    val_loss_history: Optional[List[float]] = None,
    weight_decay: float = 0.0,
) -> GatedFusionProbe:
    """Train with BCE and stop when validation loss stalls."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.BCELoss()
    best_loss = float("inf")
    stale_epochs = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for _ in range(n_epochs):
        model.train()
        for hidden, attn, logit, labels in train_loader:
            hidden = hidden.to(device)
            attn = attn.to(device)
            logit = logit.to(device)
            labels = labels.to(device)
            prob, _ = model(hidden, attn, logit)
            loss = loss_fn(prob, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss_total = 0.0
        val_rows = 0
        with torch.no_grad():
            for hidden, attn, logit, labels in val_loader:
                hidden = hidden.to(device)
                attn = attn.to(device)
                logit = logit.to(device)
                labels = labels.to(device)
                prob, _ = model(hidden, attn, logit)
                loss = loss_fn(prob, labels)
                val_loss_total += float(loss) * hidden.size(0)
                val_rows += hidden.size(0)
        val_loss = val_loss_total / max(val_rows, 1)
        if val_loss_history is not None:
            val_loss_history.append(float(val_loss))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            stale_epochs = 0
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def get_gate_weights(
    model: GatedFusionProbe,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """All gate rows ``(N, 3)`` in order, for later plots / tables."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    gates: List[torch.Tensor] = []
    for hidden, attn, logit, _ in dataloader:
        hidden = hidden.to(device)
        attn = attn.to(device)
        logit = logit.to(device)
        _, gate = model(hidden, attn, logit)
        gates.append(gate.cpu())
    if not gates:
        return torch.zeros(0, 3)
    return torch.cat(gates, dim=0)
