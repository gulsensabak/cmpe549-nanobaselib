"""
train_mscan.py  —  Multi-Scale Contextual Attention Network (MSCAN)
====================================================================
Projenin en yüksek modification detection accuracy'sini hedefleyen mimari.

Mevcut CRNN ve Transformer'dan üç temel farkı:

  1. Multi-Scale Feature Extractor
     Aynı 9-feature girdiyi üç farklı genişlikte MLP ile işler (k=1,3,5),
     sonuçları concat eder. Dar, orta ve geniş bağlamı aynı anda yakalar.

  2. Cross-Position Attention
     prev / cur / next eventları arası dikkat mekanizması.
     cur = query, [prev,cur,next] = key/value.
     "Komşu sinyal, merkez base'in modification durumunu ne kadar açıklıyor?"
     sorusunu dinamik olarak öğrenir.

  3. Physics-Guided Gating
     denoise_labels.py'daki GMM mantığını (pU→std spike, m6A→mean drop)
     öğrenilebilir bir kapı mekanizmasına dönüştürür.
     Domain bilgisi, modeli sıfırdan öğrenmek yerine doğru yöne iter.

  4. Kalibrasyon (Temperature Scaling)
     Çıkış olasılıkları kalibre edilir; eşik 0.5 yerine
     validation seti üzerinde optimal threshold seçilir.
     Label smoothing + Focal loss ile gürültüye dayanıklılık.

Kullanım:
    python src/train_mscan.py \\
        --dataset  data/processed/clean_dataset_cl.pt \\
        --mod_type pU \\
        --epochs   80 \\
        --batch_size 512 \\
        --out_weights nanospeech_mscan_best.pth

Gereksinim: torch >= 2.0  (torch-geometric gerekmez)
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# ─────────────────────────────────────────────────────────────
# 1.  ÇOK ÖLÇEKLİ ÖZELLİK ÇIKARICI
# ─────────────────────────────────────────────────────────────

class MultiScaleExtractor(nn.Module):
    """
    9-feature girdiyi üç farklı genişlikte MLP ile işler.
    Dar  (32 nöron) : sadece merkez event sinyali
    Orta (32 nöron) : komşuluk etkisi
    Geniş(32 nöron) : tüm trituple ilişkisi
    → Concat → LayerNorm → h ∈ R^96
    """
    def __init__(self, in_features: int = 9, hidden: int = 32):
        super().__init__()
        # Dar: sadece cur event (indeks 3,4,5)
        self.narrow = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # Orta: prev+cur veya cur+next (6 feature)
        self.medium = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # Geniş: tüm 9 feature
        self.wide = nn.Sequential(
            nn.Linear(9, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 9)
        cur      = x[:, 3:6]          # merkez event
        prev_cur = x[:, 0:6]          # prev + cur
        h_n = self.narrow(cur)
        h_m = self.medium(prev_cur)
        h_w = self.wide(x)
        return self.norm(torch.cat([h_n, h_m, h_w], dim=-1))  # (B, 96)


# ─────────────────────────────────────────────────────────────
# 2.  CROSS-POSITION ATTENTION
# ─────────────────────────────────────────────────────────────

class CrossPositionAttention(nn.Module):
    """
    prev / cur / next eventlarını ayrı token olarak görür,
    cur token'ı query olarak kullanarak komşulardan bilgi toplar.

    Girdi: x (B, 9)  →  3 token × 3 feature
    Çıktı: (B, d_model) — cur token'ın zenginleştirilmiş temsili
    """
    def __init__(self, token_dim: int = 3, d_model: int = 96, n_heads: int = 4):
        super().__init__()
        self.proj_in  = nn.Linear(token_dim, d_model)
        self.attn     = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=0.1, batch_first=True)
        self.norm     = nn.LayerNorm(d_model)
        self.ffn      = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 9) → (B, 3, 3) → (B, 3, d_model)
        tokens = x.view(x.size(0), 3, 3)      # (B, 3 token, 3 feature)
        tokens = self.proj_in(tokens)           # (B, 3, d_model)

        # cur = token[1] → query; tüm tokenlar → key/value
        query = tokens[:, 1:2, :]              # (B, 1, d_model)
        attn_out, _ = self.attn(query, tokens, tokens)  # (B, 1, d_model)
        attn_out = attn_out.squeeze(1)         # (B, d_model)

        # Residual + FFN
        h = self.norm(attn_out + tokens[:, 1, :])
        h = self.norm2(h + self.ffn(h))
        return h                               # (B, d_model=96)


# ─────────────────────────────────────────────────────────────
# 3.  PHYSICS-GUIDED GATE
# ─────────────────────────────────────────────────────────────

class PhysicsGuidedGate(nn.Module):
    """
    Domain bilgisini öğrenilebilir kapı mekanizmasına dönüştürür.

    pU  modu: std spike (indeks 4) yüksekse kapı açılır
    m6A modu: mean drop (indeks 3) düşükse kapı açılır

    Girdi: h (B, d_model), x_raw (B, 9)
    Çıktı: h * gate  — fizik bilgisiyle ağırlıklandırılmış özellik
    """
    def __init__(self, d_model: int = 96, mod_type: str = "pU"):
        super().__init__()
        self.mod_type  = mod_type
        # Fizik özelliği (skaler) → kapı vektörü
        self.gate_proj = nn.Sequential(
            nn.Linear(1, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        if self.mod_type == "pU":
            phys = x_raw[:, 4:5]       # cur std — turbulance
        else:  # m6A
            phys = -x_raw[:, 3:4]      # -cur mean — current drop (negatif: düşük mean = yüksek sinyal)

        gate = self.gate_proj(phys)    # (B, d_model)
        return h * gate


# ─────────────────────────────────────────────────────────────
# 4.  ANA MİMARİ: MSCAN
# ─────────────────────────────────────────────────────────────

class MSCAN(nn.Module):
    """
    Multi-Scale Contextual Attention Network

    Akış:
        x (B, 9)
        → MultiScaleExtractor    → ms_feat (B, 96)
        → CrossPositionAttention → attn_feat (B, 96)
        → Fusion (ms + attn)     → fused (B, 96)
        → PhysicsGuidedGate      → gated (B, 96)
        → Mod head               → p(mod) ∈ [0,1]
        → Base head              → logits (B, 4)
    """
    def __init__(self, mod_type: str = "pU",
                 d_model: int = 96,
                 n_heads: int = 4,
                 num_base_classes: int = 4):
        super().__init__()

        self.ms_extractor  = MultiScaleExtractor(in_features=9, hidden=d_model // 3)
        self.cross_attn    = CrossPositionAttention(token_dim=3, d_model=d_model, n_heads=n_heads)
        self.fusion_norm   = nn.LayerNorm(d_model)
        self.physics_gate  = PhysicsGuidedGate(d_model=d_model, mod_type=mod_type)

        # Dropout + projeksiyon
        self.dropout = nn.Dropout(0.3)
        self.proj    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Modifikasyon kafası — temperature scaling ile
        self.mod_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
        )
        # Temperature scaling parametresi (kalibrasyon)
        self.temperature = nn.Parameter(torch.ones(1))

        # Base calling kafası
        self.base_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, num_base_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        ms_feat   = self.ms_extractor(x)           # (B, 96)
        attn_feat = self.cross_attn(x)              # (B, 96)

        # Füzyon: iki stream'i topla + normalize
        fused = self.fusion_norm(ms_feat + attn_feat)
        fused = self.physics_gate(fused, x)         # fizik kapısı
        fused = self.dropout(self.proj(fused))

        mod_logits  = self.mod_head(fused) / self.temperature
        base_logits = self.base_head(fused)
        return base_logits, mod_logits

    def get_mod_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Inference için kalibrasyonlu olasılık döndürür."""
        _, mod_logits = self.forward(x)
        return torch.sigmoid(mod_logits).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# 5.  FOCAL LOSS — gürültüye dayanıklı sınıflandırma
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Kolay örnekleri cezalandırmaz, zor örneklere odaklanır.
    Özellikle imbalanced veri ve gürültülü etiketler için etkili.
    gamma=2 standart, alpha=pos_weight ile imbalance düzeltmesi.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        focal = (1 - p_t) ** self.gamma * bce
        return focal.mean()


# ─────────────────────────────────────────────────────────────
# 6.  OPTIMAL THRESHOLD — validation seti üzerinde
# ─────────────────────────────────────────────────────────────

def find_optimal_threshold(model, val_loader, device) -> float:
    """
    F1 skorunu maksimize eden eşik değerini bulur.
    0.5 sabit eşik yerine veriye özgü optimal eşik kullanır.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x_val, _, y_val in val_loader:
            x_val = x_val.to(device)
            probs = model.get_mod_probability(x_val).cpu()
            all_probs.append(probs)
            all_labels.append(y_val)

    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels).long()

    best_f1, best_thr = 0.0, 0.5
    for thr in torch.linspace(0.1, 0.9, 81):
        preds = (probs >= thr).long()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
        if f1 > best_f1:
            best_f1, best_thr = f1, thr.item()

    return best_thr


# ─────────────────────────────────────────────────────────────
# 7.  EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────────────────────

def compute_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    sorted_idx  = torch.argsort(probs, descending=True)
    n_pos       = labels.sum().item()
    n_neg       = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp_cum = labels[sorted_idx].cumsum(0).float()
    fp_cum = (1 - labels[sorted_idx]).cumsum(0).float()
    tpr    = tp_cum / n_pos
    fpr    = fp_cum / n_neg
    return torch.trapz(tpr, fpr).abs().item()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MSCAN] Cihaz: {device.type.upper()}")

    print(f"[MSCAN] Dataset yükleniyor: {args.dataset}")
    loaded = torch.load(args.dataset, weights_only=False)

    if isinstance(loaded, dict):
        X_all  = loaded["signals"].float()
        y_mod  = loaded["labels"].float()
        y_base = torch.zeros(len(y_mod), dtype=torch.long)
    else:
        X_all, y_base, y_mod = loaded
        X_all  = X_all.float()
        y_mod  = y_mod.float()
        y_base = y_base.long()

    print(f"  Toplam : {len(y_mod):,}  |  Pozitif: {int(y_mod.sum()):,}  |  Negatif: {int((y_mod==0).sum()):,}")

    dataset    = TensorDataset(X_all, y_base, y_mod)
    train_size = int(0.85 * len(dataset))
    val_size   = len(dataset) - train_size
    generator  = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MSCAN(
        mod_type=args.mod_type,
        d_model=96,
        n_heads=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametre sayısı: {n_params:,}")

    # Optimizer: AdamW + Cosine annealing with warm restarts
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=2e-3, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Imbalance düzeltme
    n_pos = y_mod.sum().item()
    n_neg = len(y_mod) - n_pos
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)]).to(device)

    # Kayıp fonksiyonları
    focal_loss    = FocalLoss(gamma=2.0, pos_weight=pos_w)
    ce_loss       = nn.CrossEntropyLoss(label_smoothing=0.05)
    lambda_base   = 0.3  # base calling ikincil görev

    best_val_auc  = 0.0
    best_threshold = 0.5
    print(f"\n[MSCAN] Eğitim başlıyor — mod_type={args.mod_type}, epochs={args.epochs}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = mod_loss_sum = 0.0
        train_probs, train_labels = [], []

        for x_b, y_base_b, y_mod_b in train_loader:
            x_b      = x_b.to(device)
            y_base_b = y_base_b.to(device)
            y_mod_b  = y_mod_b.to(device)

            optimizer.zero_grad()
            base_logits, mod_logits = model(x_b)

            loss_mod  = focal_loss(mod_logits.squeeze(), y_mod_b)
            loss_base = ce_loss(base_logits, y_base_b)
            loss      = loss_mod + lambda_base * loss_base

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss    += loss.item()
            mod_loss_sum  += loss_mod.item()
            probs = torch.sigmoid(mod_logits.squeeze()).detach().cpu()
            train_probs.append(probs)
            train_labels.append(y_mod_b.cpu())

        scheduler.step()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for x_v, _, y_v in val_loader:
                x_v = x_v.to(device)
                p   = model.get_mod_probability(x_v).cpu()
                val_probs.append(p)
                val_labels.append(y_v)

        vp = torch.cat(val_probs)
        vl = torch.cat(val_labels).long()
        val_auc  = compute_auc(vp, vl)
        tr_auc   = compute_auc(torch.cat(train_probs), torch.cat(train_labels).long())

        # Her 10 epoch'ta threshold güncelle
        if epoch % 10 == 0:
            best_threshold = find_optimal_threshold(model, val_loader, device)

        val_acc = ((vp >= best_threshold).long() == vl).float().mean().item() * 100
        n_b     = len(train_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {total_loss/n_b:.4f} (mod {mod_loss_sum/n_b:.4f}) | "
            f"Tr AUC {tr_auc:.4f} | "
            f"Val AUC {val_auc:.4f} | "
            f"Val Acc {val_acc:.2f}% | "
            f"Thr {best_threshold:.2f} | "
            f"T {model.temperature.item():.3f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "model_state": model.state_dict(),
                "threshold":   best_threshold,
                "mod_type":    args.mod_type,
                "val_auc":     val_auc,
                "epoch":       epoch,
            }, args.out_weights)
            print(f"  --> Yeni en iyi kaydedildi (AUC={val_auc:.4f}, Thr={best_threshold:.2f})")

    print(f"\n[MSCAN] Tamamlandı. En iyi Val AUC: {best_val_auc:.4f}")
    print(f"         Optimal threshold: {best_threshold:.3f}")
    print(f"         Model: {args.out_weights}")


# ─────────────────────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MSCAN — Multi-Scale Contextual Attention Network"
    )
    p.add_argument("--dataset",     required=True)
    p.add_argument("--mod_type",    default="pU", choices=["pU", "m6A"],
                   help="Fizik kapısını ayarlar: pU veya m6A")
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--out_weights", default="nanospeech_mscan_best.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 65)
    print("  NanoSpeech-MTL — MSCAN (Multi-Scale Contextual Attention)")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Mod type  : {args.mod_type}")
    print(f"  Epochs    : {args.epochs}  |  LR: {args.lr}")
    print(f"  Kayıp     : Focal Loss (gamma=2) + Label Smoothing")
    print(f"  Kalibrasyon: Temperature Scaling + Optimal Threshold")
    print("=" * 65)
    train(args)
