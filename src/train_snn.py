"""
train_snn.py  —  SNN: Siamese Noise-Aware Network
Diyagramdaki mimarinin tam implementasyonu.

Mimari:
  1. Paylaşımlı Encoder  : BN → Linear(9→64) → GELU → Dropout(0.2)
                           → Linear(64→128) → GELU → LayerNorm → z ∈ R^128
  2. NT-Xent Contrastive Loss  : z_clean ile z_aug arasında
  3. Noise Detector            : ||z_clean − z_aug|| → güvenilirlik skoru
  4. Sample Reweighting        : gürültülü örnek → düşük ağırlık
  5. Projection Head           : Linear(128→64) → GELU → Linear(64→32) → L2-norm → e ∈ R^32
  6. Mod head                  : Linear(32→1) + sigmoid, Weighted Focal Loss
  7. Base head                 : Linear(32→4), CrossEntropy
  8. Toplam kayıp              : L = w × Focal(mod) + λ₁ × CE(base) + λ₂ × NT-Xent(z)

Kullanım:
    python train_snn.py \
        --dataset    data/processed/clean_dataset_cl.pt \
        --epochs     60 \
        --batch_size 1024 \
        --out_weights data/weights/snn_best.pth
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# ──────────────────────────────────────────────────────────────
# 1.  Paylaşımlı Encoder
# ──────────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    """
    BN → Linear(9→64) → GELU → Dropout(0.2)
    → Linear(64→128) → GELU → LayerNorm → z ∈ R^128
    """
    def __init__(self, n_features: int = 9, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        return self.net(x)   # (B, 128)


# ──────────────────────────────────────────────────────────────
# 2.  Projection Head
# ──────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Linear(128→64) → GELU → Linear(64→32) → L2-norm → e ∈ R^32
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

    def forward(self, z):
        e = self.net(z)
        return F.normalize(e, p=2, dim=-1)   # L2-norm → e ∈ R^32


# ──────────────────────────────────────────────────────────────
# 3.  Tam Model
# ──────────────────────────────────────────────────────────────

class SNN(nn.Module):
    """
    Siamese Noise-Aware Network

    forward(x_clean, x_aug) → (mod_logit, base_logit, z_clean, z_aug, weights)
    İnference: forward(x_clean, x_clean) veya sadece predict(x_clean)
    """
    def __init__(
        self,
        n_features: int = 9,
        n_base_classes: int = 4,
        dropout: float = 0.2,
        noise_threshold: float = 1.0,  # Noise Detector eşiği
    ):
        super().__init__()
        self.noise_threshold = noise_threshold

        # Paylaşımlı encoder (ağırlıklar aynı)
        self.encoder = SharedEncoder(n_features, dropout)

        # Projection head
        self.proj_head = ProjectionHead()

        # Mod head: Linear(32→1) + sigmoid (loss'ta BCEWithLogits)
        self.mod_head = nn.Linear(32, 1)

        # Base head: Linear(32→4) + CrossEntropy
        self.base_head = nn.Linear(32, n_base_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        """Encoder + Projection: x → e ∈ R^32"""
        z = self.encoder(x)          # (B, 128)
        e = self.proj_head(z)        # (B, 32), L2-normalized
        return z, e

    def forward(self, x_clean, x_aug):
        """
        x_clean: (B, 9) — orijinal sinyal
        x_aug  : (B, 9) — augmented sinyal (x + ε)
        """
        # ── Encoder (ağırlıklar paylaşımlı) ──────────────────────────
        z_clean, e_clean = self.encode(x_clean)   # (B,128), (B,32)
        z_aug,   _       = self.encode(x_aug)     # (B,128)

        # ── Noise Detector → Sample Reweighting ──────────────────────
        # ||z_clean − z_aug||₂ → güvenilirlik skoru
        noise_score = torch.norm(z_clean - z_aug, p=2, dim=1)  # (B,)
        # Gürültülü örnek → düşük ağırlık; temiz → yüksek ağırlık
        # sigmoid(-score) ile normalize
        weights = torch.sigmoid(-noise_score / self.noise_threshold)  # (B,) ∈ (0,1)
        weights = weights / (weights.mean() + 1e-8)   # ortalaması ≈ 1

        # ── Projection → Kafalar ──────────────────────────────────────
        mod_logit  = self.mod_head(e_clean)     # (B, 1)
        base_logit = self.base_head(e_clean)    # (B, 4)

        return mod_logit, base_logit, z_clean, z_aug, weights

    @torch.no_grad()
    def predict(self, x):
        """İnference: sadece x_clean gönder."""
        z, e = self.encode(x)
        mod_prob   = torch.sigmoid(self.mod_head(e)).squeeze(1)   # (B,)
        base_class = self.base_head(e).argmax(dim=1)               # (B,)
        return mod_prob, base_class


# ──────────────────────────────────────────────────────────────
# 4.  Loss Fonksiyonları
# ──────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    NT-Xent Contrastive Loss
    Aynı örnek (z_clean, z_aug) → yakın; farklı örnekler → uzak
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_clean, z_aug):
        """
        z_clean, z_aug: (B, 128)  — normalize edilmemiş embeddings
        """
        B = z_clean.size(0)

        # L2 normalize
        z1 = F.normalize(z_clean, p=2, dim=1)
        z2 = F.normalize(z_aug,   p=2, dim=1)

        # (2B, 128) — clean ve aug birleştir
        z  = torch.cat([z1, z2], dim=0)

        # Cosine similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.T) / self.temperature

        # Diyagonali maskele (kendisiyle benzerlik)
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Pozitif çiftler: (i, i+B) ve (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device)
        ])  # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for mod head
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, sample_weights=None):
        """
        logits        : (B, 1) veya (B,)
        targets       : (B,) float — 0 veya 1
        sample_weights: (B,) — noise detector'dan gelen ağırlıklar
        """
        logits = logits.squeeze(1)
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob   = torch.sigmoid(logits)
        p_t    = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal  = alpha_t * (1 - p_t) ** self.gamma * bce

        if sample_weights is not None:
            focal = focal * sample_weights

        return focal.mean()


class SNNLoss(nn.Module):
    """
    Toplam kayıp:
    L = w × Focal(mod) + λ₁ × CE(base) + λ₂ × NT-Xent(z)
    """
    def __init__(
        self,
        lambda1: float = 0.5,   # CE(base) ağırlığı
        lambda2: float = 0.3,   # NT-Xent ağırlığı
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.focal   = WeightedFocalLoss(focal_alpha, focal_gamma)
        self.ce      = nn.CrossEntropyLoss(reduction="none")
        self.ntxent  = NTXentLoss(temperature)

    def forward(self, mod_logit, base_logit, z_clean, z_aug,
                mod_labels, base_labels, weights):
        """
        mod_logit   : (B, 1)
        base_logit  : (B, 4)
        z_clean     : (B, 128)
        z_aug       : (B, 128)
        mod_labels  : (B,) float 0/1
        base_labels : (B,) long  0-3
        weights     : (B,) float — noise reweighting
        """
        # Weighted Focal Loss (mod)
        focal_loss = self.focal(mod_logit, mod_labels, sample_weights=weights)

        # Weighted CrossEntropy (base)
        ce_loss = (self.ce(base_logit, base_labels) * weights).mean()

        # NT-Xent Contrastive Loss
        ntxent_loss = self.ntxent(z_clean, z_aug)

        total = focal_loss + self.lambda1 * ce_loss + self.lambda2 * ntxent_loss
        return total, focal_loss.item(), ce_loss.item(), ntxent_loss.item()


# ──────────────────────────────────────────────────────────────
# 5.  Augmentation
# ──────────────────────────────────────────────────────────────

def gaussian_augment(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """x_aug = x + ε,  ε ~ N(0, std²)"""
    return x + torch.randn_like(x) * std


# ──────────────────────────────────────────────────────────────
# 6.  Veri yükleme
# ──────────────────────────────────────────────────────────────

def load_dataset(path: str, val_ratio: float = 0.1):
    """
    Beklenen format: {"signals": Tensor[N, 9], "labels": Tensor[N]}
    labels: modifikasyon etiketi (0/1) — base label olarak da kullanılır.
    """
    data   = torch.load(path, map_location="cpu")
    X      = data["signals"].float()   # (N, 9)
    y_mod  = data["labels"].float()    # (N,)  — mod etiketi (0/1)

    # Base label: eğer veri setinde "base_labels" varsa kullan,
    # yoksa mod label'dan 2-sınıf türet (genişletilebilir)
    if "base_labels" in data:
        y_base = data["base_labels"].long()
    else:
        # Fallback: mod label'ı 0/1 → base label olarak kullan (2 sınıf)
        # Gerçek kullanımda: A/C/G/T sınıfları için 0-3 arası etiket gerekmeli
        y_base = y_mod.long()

    dataset = TensorDataset(X, y_mod, y_base)
    n_val   = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[data] Train: {n_train:,}  |  Val: {n_val:,}")
    return train_ds, val_ds


# ──────────────────────────────────────────────────────────────
# 7.  Eğitim döngüsü
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    aug_std=0.1, scaler=None):
    model.train()
    total_loss = focal_sum = ce_sum = ntx_sum = 0.0
    correct = total = 0

    for X_batch, y_mod, y_base in loader:
        X_batch = X_batch.to(device)
        y_mod   = y_mod.to(device)
        y_base  = y_base.to(device)

        # Gaussian augmentation (online)
        x_aug = gaussian_augment(X_batch, std=aug_std)

        optimizer.zero_grad()

        if scaler:   # AMP
            with torch.autocast(device_type="cuda"):
                mod_logit, base_logit, z_clean, z_aug, weights = model(X_batch, x_aug)
                loss, fl, cel, ntxl = criterion(
                    mod_logit, base_logit, z_clean, z_aug, y_mod, y_base, weights
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            mod_logit, base_logit, z_clean, z_aug, weights = model(X_batch, x_aug)
            loss, fl, cel, ntxl = criterion(
                mod_logit, base_logit, z_clean, z_aug, y_mod, y_base, weights
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        focal_sum  += fl
        ce_sum     += cel
        ntx_sum    += ntxl

        preds   = (torch.sigmoid(mod_logit.squeeze(1)) > 0.5).long()
        correct += (preds == y_mod.long()).sum().item()
        total   += len(y_mod)

    n = len(loader)
    return total_loss/n, focal_sum/n, ce_sum/n, ntx_sum/n, correct/total


@torch.no_grad()
def evaluate(model, loader, criterion, device, aug_std=0.1):
    model.eval()
    total_loss = correct = total = 0

    for X_batch, y_mod, y_base in loader:
        X_batch = X_batch.to(device)
        y_mod   = y_mod.to(device)
        y_base  = y_base.to(device)
        x_aug   = gaussian_augment(X_batch, std=aug_std)

        mod_logit, base_logit, z_clean, z_aug, weights = model(X_batch, x_aug)
        loss, _, _, _ = criterion(
            mod_logit, base_logit, z_clean, z_aug, y_mod, y_base, weights
        )
        total_loss += loss.item()

        preds   = (torch.sigmoid(mod_logit.squeeze(1)) > 0.5).long()
        correct += (preds == y_mod.long()).sum().item()
        total   += len(y_mod)

    return total_loss / len(loader), correct / total


# ──────────────────────────────────────────────────────────────
# 8.  Ana eğitim fonksiyonu
# ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_ds, val_ds = load_dataset(args.dataset)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Sınıf sayısını belirle
    n_base = args.n_base_classes

    model = SNN(
        n_features     = 9,
        n_base_classes = n_base,
        dropout        = args.dropout,
        noise_threshold= args.noise_threshold,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] SNN — {n_params:,} parametre")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = SNNLoss(
        lambda1     = args.lambda1,
        lambda2     = args.lambda2,
        focal_alpha = args.focal_alpha,
        focal_gamma = args.focal_gamma,
        temperature = args.temperature,
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    Path(args.out_weights).parent.mkdir(parents=True, exist_ok=True)

    best_val_acc  = 0.0
    best_val_loss = float("inf")

    header = (f"{'Epoch':>5}  {'TrLoss':>8}  {'Focal':>7}  {'CE':>7}  "
              f"{'NTXent':>7}  {'TrAcc':>7}  {'VlLoss':>8}  {'VlAcc':>7}  {'LR':>8}")
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, fl, cel, ntxl, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            aug_std=args.aug_std, scaler=scaler
        )
        vl_loss, vl_acc = evaluate(
            model, val_loader, criterion, device, aug_std=args.aug_std
        )
        scheduler.step()

        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:5d}  {tr_loss:8.4f}  {fl:7.4f}  {cel:7.4f}  "
              f"{ntxl:7.4f}  {tr_acc:7.4f}  {vl_loss:8.4f}  "
              f"{vl_acc:7.4f}  {lr_now:.2e}  ({elapsed:.0f}s)")

        # En iyi modeli val_acc'e göre kaydet
        if vl_acc > best_val_acc or (vl_acc == best_val_acc and vl_loss < best_val_loss):
            best_val_acc  = vl_acc
            best_val_loss = vl_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    vl_loss,
                "val_acc":     vl_acc,
                "args":        vars(args),
            }, args.out_weights)
            print(f"         ↑ en iyi model kaydedildi  "
                  f"(val_acc={vl_acc:.4f}, val_loss={vl_loss:.4f})")

    print(f"\n[✓] Eğitim tamamlandı. En iyi ağırlıklar → {args.out_weights}")
    print(f"    Best val_acc  : {best_val_acc:.4f}")
    print(f"    Best val_loss : {best_val_loss:.4f}")


# ──────────────────────────────────────────────────────────────
# 9.  CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SNN: Siamese Noise-Aware Network")

    # ── Zorunlu ────────────────────────────────────────────────
    p.add_argument("--dataset",     required=True)
    p.add_argument("--out_weights", required=True)

    # ── Eğitim ─────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=1024)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--dropout",     type=float, default=0.2)

    # ── Augmentation ───────────────────────────────────────────
    p.add_argument("--aug_std",     type=float, default=0.1,
                   help="Gaussian gürültü std (x_aug = x + ε)")

    # ── Model ──────────────────────────────────────────────────
    p.add_argument("--n_base_classes", type=int, default=4,
                   help="Base head sınıf sayısı (varsayılan: 4 = A/C/G/T)")
    p.add_argument("--noise_threshold", type=float, default=1.0,
                   help="Noise detector eşiği (sigmoid scaling)")

    # ── Loss ağırlıkları ───────────────────────────────────────
    p.add_argument("--lambda1",     type=float, default=0.5,
                   help="CE(base) ağırlığı λ₁")
    p.add_argument("--lambda2",     type=float, default=0.3,
                   help="NT-Xent ağırlığı λ₂")
    p.add_argument("--focal_alpha", type=float, default=0.25,
                   help="Focal Loss α (sınıf dengesi)")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal Loss γ (zor örnek odağı)")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="NT-Xent sıcaklık parametresi")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 65)
    print("  SNN — Siamese Noise-Aware Network")
    print(f"  Veri        : {args.dataset}")
    print(f"  Epoklar     : {args.epochs}   Batch: {args.batch_size}")
    print(f"  λ₁(CE)      : {args.lambda1}   λ₂(NT-Xent): {args.lambda2}")
    print(f"  Focal α/γ   : {args.focal_alpha}/{args.focal_gamma}")
    print(f"  Aug std     : {args.aug_std}   Temp: {args.temperature}")
    print("=" * 65)
    train(args)
