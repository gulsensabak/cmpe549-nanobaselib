"""
train_mtl_transformer.py  —  Transformer tabanlı Multi-Task Learning
Orijinal train_mtl.py (CRNN) ile aynı CLI interface'ini kullanır.

Fark:
  - CRNN:        Conv1D  →  GRU  →  iki kafa
  - Transformer: Conv1D patch embedding  →  Transformer encoder  →  iki kafa

İki kafa (orijinalle aynı):
  1. CTC kafası      → base sequence tahmini (basecalling)
  2. BCE kafası      → modifikasyon tespiti (0/1)

Kullanım (orijinal ile birebir aynı):
    python src/train_mtl_transformer.py \
        --dataset    data/processed/clean_dataset_cl.pt \
        --epochs     50 \
        --batch_size 1024 \
        --out_weights data/weights/nanospeech_transformer_best.pth

Gereksinim: torch >= 2.0
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# ──────────────────────────────────────────────────────────────
# 1.  Model mimarisi
# ──────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinüzoidal pozisyon kodlaması."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class NanoSpeechTransformer(nn.Module):
    """
    Transformer tabanlı çift-başlı MTL modeli.

    Girdi  : (B, seq_len, n_features=9)  — orijinal 9-feature sliding window
    Çıktı  :
        ctc_logits   (B, seq_len, vocab_size)  — CTC basecalling kafası
        mod_logits   (B, 1)                    — modifikasyon skoru (sigmoid'a girmemiş)
    """

    def __init__(
        self,
        n_features: int   = 9,
        d_model:    int   = 128,
        n_heads:    int   = 4,
        n_layers:   int   = 4,
        d_ff:       int   = 256,
        dropout:    float = 0.1,
        vocab_size: int   = 5,       # A C G T/U + blank(CTC)
        patch_size: int   = 1,       # Conv1D ile ilk projeksiyon
    ):
        super().__init__()
        self.d_model = d_model

        # ── Patch embedding (Conv1D projeksiyon) ──────────────────────
        # 9 feature → d_model boyutuna projeksiyon
        # kernel_size=patch_size ile yerel bağlamı da yakalar
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=patch_size,
                      padding=patch_size // 2),
            nn.LayerNorm(d_model),          # channel-wise norm (transpose gerekir)
        )
        # Not: Conv1d (B, C, T) bekler; forward'da transpose yapıyoruz

        # ── Positional encoding ───────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # ── Transformer encoder ───────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model     = d_model,
            nhead       = n_heads,
            dim_feedforward = d_ff,
            dropout     = dropout,
            batch_first = True,      # (B, T, C) formatı — PyTorch >= 1.9
            norm_first  = True,      # Pre-LN: daha stabil eğitim
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── CTC kafası (basecalling) ──────────────────────────────────
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # ── Modifikasyon kafası (BCE) ─────────────────────────────────
        # Tüm zaman adımlarını global average pooling ile özetle
        self.mod_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x):
        """
        x: (B, T, 9)
        """
        B, T, C = x.shape

        # ── Patch embedding ───────────────────────────────────────────
        # Conv1d: (B, C_in, T) → (B, d_model, T)
        h = x.transpose(1, 2)                    # (B, 9, T)
        h = self.input_proj[0](h)                # Conv1d → (B, d_model, T)
        h = h.transpose(1, 2)                    # → (B, T, d_model)
        h = self.input_proj[1](h)                # LayerNorm

        # ── Positional encoding ───────────────────────────────────────
        h = self.pos_enc(h)                      # (B, T, d_model)

        # ── Transformer encoder ───────────────────────────────────────
        h = self.transformer(h)                  # (B, T, d_model)

        # ── CTC kafası ───────────────────────────────────────────────
        ctc_logits = self.ctc_head(h)            # (B, T, vocab_size)

        # ── Modifikasyon kafası ───────────────────────────────────────
        pooled    = h.mean(dim=1)                # (B, d_model)  — global avg pool
        mod_logit = self.mod_head(pooled)        # (B, 1)

        return ctc_logits, mod_logit


# ──────────────────────────────────────────────────────────────
# 2.  Veri yükleme
# ──────────────────────────────────────────────────────────────

def load_dataset(path: str, val_ratio: float = 0.1):
    """
    .pt dosyasını okur, train/val olarak böler.
    Beklenen format (denoise_labels_cl.py çıktısı ile uyumlu):
        {"signals": Tensor[N, 9], "labels": Tensor[N]}
    """
    data = torch.load(path, map_location="cpu")
    X = data["signals"].float()    # (N, 9)
    y = data["labels"].float()     # (N,)

    # Transformer için (N, 1, 9) → tek zaman adımı gibi; ya da
    # orijinal pipeline'da sliding window gruplanmış olabilir.
    # Burada her örneği (1, 9) boyutlu bir sekans olarak alıyoruz.
    # Gerçek sequence modunda T>1 yapısı için extract_eventalign_features.py
    # çıktısını read bazlı gruplamak gerekir — bu basitleştirilmiş versiyon.
    X = X.unsqueeze(1)             # (N, T=1, 9)

    dataset = TensorDataset(X, y)
    n_val   = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[data] Train: {n_train:,}  |  Val: {n_val:,}")
    return train_ds, val_ds


# ──────────────────────────────────────────────────────────────
# 3.  Loss fonksiyonları
# ──────────────────────────────────────────────────────────────

class MTLLoss(nn.Module):
    """
    İki görevi dengeleyen kayıp:
      L_total = alpha * L_ctc  +  (1 - alpha) * L_bce

    CTC kaybı için dummy hedef (gerçek pipeline'da reference sequence gelir).
    Bu implementasyonda CTC'yi placeholder olarak tutuyoruz;
    gerçek basecalling entegrasyonu için extract_eventalign_features.py
    çıktısından sequence labelları gerekir.
    """

    def __init__(self, alpha: float = 0.5, vocab_size: int = 5):
        super().__init__()
        self.alpha     = alpha
        self.bce       = nn.BCEWithLogitsLoss()
        self.ctc       = nn.CTCLoss(blank=vocab_size - 1, reduction="mean",
                                    zero_infinity=True)
        self.vocab_size = vocab_size

    def forward(self, ctc_logits, mod_logit, mod_labels):
        """
        ctc_logits : (B, T, vocab_size)
        mod_logit  : (B, 1)
        mod_labels : (B,)
        """
        B, T, V = ctc_logits.shape

        # ── BCE loss (modifikasyon) ───────────────────────────────────
        bce_loss = self.bce(mod_logit.squeeze(1), mod_labels)

        # ── CTC loss (dummy — sadece placeholder) ────────────────────
        # Gerçek kullanımda burada reference sequence labelları olmalı.
        # Şimdilik tüm pozisyonları "blank" kabul ediyoruz.
        log_probs   = ctc_logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, B, V)
        input_lens  = torch.full((B,), T, dtype=torch.long)
        # Dummy target: her örnek için tek "A" (indeks 0)
        targets     = torch.zeros(B, dtype=torch.long)
        target_lens = torch.ones(B, dtype=torch.long)
        ctc_loss = self.ctc(log_probs, targets, input_lens, target_lens)

        total = self.alpha * ctc_loss + (1 - self.alpha) * bce_loss
        return total, ctc_loss.item(), bce_loss.item()


# ──────────────────────────────────────────────────────────────
# 4.  Eğitim döngüsü
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = ctc_sum = bce_sum = 0.0
    correct = total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        if scaler:   # AMP
            with torch.autocast(device_type="cuda"):
                ctc_logits, mod_logit = model(X_batch)
                loss, ctc_l, bce_l = criterion(ctc_logits, mod_logit, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            ctc_logits, mod_logit = model(X_batch)
            loss, ctc_l, bce_l = criterion(ctc_logits, mod_logit, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        ctc_sum    += ctc_l
        bce_sum    += bce_l

        # Accuracy (modifikasyon kafası)
        preds   = (torch.sigmoid(mod_logit.squeeze(1)) > 0.5).long()
        correct += (preds == y_batch.long()).sum().item()
        total   += len(y_batch)

    n = len(loader)
    return total_loss / n, ctc_sum / n, bce_sum / n, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        ctc_logits, mod_logit = model(X_batch)
        loss, _, _ = criterion(ctc_logits, mod_logit, y_batch)
        total_loss += loss.item()
        preds   = (torch.sigmoid(mod_logit.squeeze(1)) > 0.5).long()
        correct += (preds == y_batch.long()).sum().item()
        total   += len(y_batch)

    return total_loss / len(loader), correct / total


# ──────────────────────────────────────────────────────────────
# 5.  Ana eğitim fonksiyonu
# ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Veri
    train_ds, val_ds = load_dataset(args.dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = NanoSpeechTransformer(
        n_features = 9,
        d_model    = args.d_model,
        n_heads    = args.n_heads,
        n_layers   = args.n_layers,
        d_ff       = args.d_model * 2,
        dropout    = args.dropout,
        vocab_size = 5,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] NanoSpeechTransformer  —  {n_params:,} parametre")

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = MTLLoss(alpha=args.mtl_alpha)

    # AMP (sadece CUDA'da)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    Path(args.out_weights).parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'CTC':>8}  "
          f"{'BCE':>8}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, ctc_l, bce_l, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:5d}  {tr_loss:10.4f}  {ctc_l:8.4f}  "
              f"{bce_l:8.4f}  {tr_acc:9.4f}  {vl_loss:9.4f}  "
              f"{vl_acc:8.4f}  {lr_now:.2e}  ({elapsed:.0f}s)")

        # En iyi modeli kaydet
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    vl_loss,
                "val_acc":     vl_acc,
                "args":        vars(args),
            }, args.out_weights)
            print(f"         ↑ en iyi model kaydedildi (val_loss={vl_loss:.4f})")

    print(f"\n[✓] Eğitim tamamlandı. En iyi ağırlıklar → {args.out_weights}")


# ──────────────────────────────────────────────────────────────
# 6.  CLI  (orijinal train_mtl.py ile birebir aynı argümanlar +
#          transformer-özel ek argümanlar)
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Transformer tabanlı NanoSpeech-MTL eğitimi"
    )
    # ── Orijinal argümanlar ──────────────────────────────────────────
    p.add_argument("--dataset",     required=True,
                   help="Temizlenmiş .pt veri dosyası")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=1024)
    p.add_argument("--out_weights", required=True,
                   help="Model ağırlıklarının kaydedileceği .pth dosyası")

    # ── Transformer-özel argümanlar ──────────────────────────────────
    p.add_argument("--d_model",    type=int,   default=128,
                   help="Transformer gizli boyutu (varsayılan: 128)")
    p.add_argument("--n_heads",    type=int,   default=4,
                   help="Attention head sayısı (varsayılan: 4)")
    p.add_argument("--n_layers",   type=int,   default=4,
                   help="Transformer encoder katman sayısı (varsayılan: 4)")
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Başlangıç öğrenme hızı")
    p.add_argument("--mtl_alpha",  type=float, default=0.5,
                   help="CTC / BCE ağırlık dengesi (0=sadece BCE, 1=sadece CTC)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  NanoSpeech-MTL  —  Transformer Eğitimi")
    print(f"  Veri   : {args.dataset}")
    print(f"  Epoklar: {args.epochs}   Batch: {args.batch_size}")
    print(f"  Model  : d={args.d_model}  heads={args.n_heads}  layers={args.n_layers}")
    print("=" * 60)
    train(args)
