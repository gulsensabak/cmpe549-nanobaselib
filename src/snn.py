"""
train_snn.py  —  Siamese Noise-Aware Network (SNN)
===================================================
95%+ modification detection accuracy hedefli mimari.

Neden mevcut modellerden farklı:

  Mevcut CRNN / Transformer / MSCAN:
    - Gürültülü etiketlere körü körüne güvenir
    - Her örneğe eşit ağırlık verir
    - Hangi örneğin "zor" olduğunu bilmez

  SNN'in çözümü:
    1. Siamese çift kol: aynı encoder ağırlıklarını paylaşan
       iki kol, orijinal ve augmented (gürültü eklenmiş) sinyali
       paralel işler.

    2. NT-Xent Contrastive Loss: aynı örneğin temiz/gürültülü
       versiyonlarını yakın, farklı örnekleri uzak tutar.
       Encoder, modification'ın gerçek izini öğrenmek zorunda kalır.

    3. Noise Detector: ||z_clean - z_aug|| mesafesini ölçer.
       Bu mesafe büyükse örnek gürültülüdür → düşük ağırlık.
       Bu sayede kötü etiketli örneklerin etkisi azaltılır.

    4. Sample Reweighting: gürültü skoru loss hesabına
       örnek ağırlığı olarak girer. Temiz örnekler daha fazla
       öğretir, gürültülü örnekler daha az zarar verir.

    5. Toplam kayıp:
       L = w × Focal(mod) + λ₁ × CE(base) + λ₂ × NT-Xent(z)

Kullanım:
    python src/train_snn.py \\
        --dataset  data/processed/clean_dataset_cl.pt \\
        --mod_type pU \\
        --epochs   80 \\
        --batch_size 512 \\
        --out_weights nanospeech_snn_best.pth

Gereksinim: sadece PyTorch (torch-geometric gerekmez)
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# ─────────────────────────────────────────────────────────────
# 1.  AUGMENTATION — eğitim sırasında gürültü ekleme
# ─────────────────────────────────────────────────────────────

def augment(x: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    Orijinal sinyale Gaussian gürültü ekler.
    Siamese kolun augmented versiyonunu üretir.
    noise_std: gürültü şiddeti — çok yüksek olursa
               encoder ayrımı öğrenemez.
    """
    return x + torch.randn_like(x) * noise_std


# ─────────────────────────────────────────────────────────────
# 2.  PAYLAŞIMLI ENCODER
# ─────────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    """
    Her iki siamese kol da bu encoder'ı paylaşır
    (aynı ağırlık nesnesi, iki farklı girdi).

    Mimari:
        BatchNorm(9) → Linear(9→64) → GELU
        → Dropout(0.2) → Linear(64→128)
        → GELU → LayerNorm(128)
        → z ∈ R^128
    """
    def __init__(self, in_features: int = 9, hidden: int = 64,
                 out_features: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_features),
            nn.GELU(),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 128)


# ─────────────────────────────────────────────────────────────
# 3.  NT-Xent CONTRASTIVE LOSS
# ─────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    SimCLR'dan uyarlanmıştır.

    Aynı örneğin temiz/augmented çifti pozitif,
    batch içindeki diğer tüm örnekler negatif kabul edilir.

    temperature: düşük → daha keskin ayrım, yüksek → daha yumuşak.
                 0.1-0.5 arası tipik değerler.
    """
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)

        # L2 normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Tüm çiftlerin benzerlik matrisi (2B × 2B)
        z  = torch.cat([z1, z2], dim=0)          # (2B, 128)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Diyagonali -inf yap (kendi kendine benzerlik)
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))

        # Pozitif çiftler: (i, i+B) ve (i+B, i)
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])   # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────
# 4.  NOISE DETECTOR — örnek güvenilirlik skoru
# ─────────────────────────────────────────────────────────────

def compute_noise_score(z_clean: torch.Tensor,
                        z_aug:   torch.Tensor) -> torch.Tensor:
    """
    İki temsil arasındaki L2 mesafesi.
    Büyük mesafe → encoder bu örneği gürültüye duyarlı buluyor
                  → etiket güvenilmez olabilir.

    Çıktı: (B,) — her örnek için 0-1 arası güvenilirlik ağırlığı.
           güvenilirlik = 1 / (1 + distance)
    """
    dist = torch.norm(z_clean - z_aug, dim=-1)    # (B,)
    reliability = 1.0 / (1.0 + dist)              # (B,) ∈ (0,1]
    return reliability


# ─────────────────────────────────────────────────────────────
# 5.  FOCAL LOSS — ağırlıklı versiyon
# ─────────────────────────────────────────────────────────────

def weighted_focal_loss(logits:     torch.Tensor,
                        targets:    torch.Tensor,
                        weights:    torch.Tensor,
                        pos_weight: torch.Tensor,
                        gamma:      float = 2.0) -> torch.Tensor:
    """
    Her örnek için ayrı ağırlık uygulayan Focal Loss.
    weights: noise detector'dan gelen güvenilirlik skoru (B,)
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='none'
    )
    probs = torch.sigmoid(logits)
    p_t   = probs * targets + (1 - probs) * (1 - targets)
    focal = (1 - p_t) ** gamma * bce              # (B,)
    return (focal * weights).mean()


# ─────────────────────────────────────────────────────────────
# 6.  ANA MİMARİ: SNN
# ─────────────────────────────────────────────────────────────

class SNN(nn.Module):
    """
    Siamese Noise-Aware Network

    Eğitim akışı:
        x → augment → x_aug
        x     → SharedEncoder → z_clean
        x_aug → SharedEncoder → z_aug   (aynı ağırlıklar)
        reliability = noise_score(z_clean, z_aug)
        z_clean → ProjectionHead → e ∈ R^32
        e → mod_head → p(mod)
        e → base_head → logits(4)

    Kayıp:
        L = reliability × Focal(mod)
          + λ₁ × CE(base)
          + λ₂ × NTXent(z_clean, z_aug)
    """

    def __init__(self, in_features: int = 9,
                 hidden: int = 64, z_dim: int = 128,
                 proj_dim: int = 32, num_base_classes: int = 4):
        super().__init__()

        # Paylaşımlı encoder — her iki kol bunu kullanır
        self.encoder = SharedEncoder(in_features, hidden, z_dim)

        # Projection head: contrastive öğrenme sonrası
        # daha kompakt temsil
        self.proj_head = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, proj_dim),
        )

        # Modification kafası
        self.mod_head = nn.Sequential(
            nn.Linear(proj_dim, 16),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(16, 1),
        )

        # Base calling kafası
        self.base_head = nn.Sequential(
            nn.Linear(proj_dim, 16),
            nn.GELU(),
            nn.Linear(16, num_base_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sadece z temsilini döndürür (contrastive loss için)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor, x_aug: torch.Tensor = None):
        """
        Eğitim: x ve x_aug verilir → z_clean, z_aug, mod_logits, base_logits, reliability
        Inference: sadece x verilir → mod_logits, base_logits
        """
        z_clean = self.encoder(x)

        if x_aug is not None:
            z_aug       = self.encoder(x_aug)
            reliability = compute_noise_score(z_clean, z_aug)
        else:
            z_aug       = None
            reliability = torch.ones(x.size(0), device=x.device)

        e           = self.proj_head(z_clean)
        mod_logits  = self.mod_head(e)
        base_logits = self.base_head(e)

        return mod_logits, base_logits, z_clean, z_aug, reliability

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: kalibrasyonlu modifikasyon olasılığı."""
        mod_logits, _, _, _, _ = self.forward(x)
        return torch.sigmoid(mod_logits).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# 7.  YARDIMCI: AUC & OPTIMAL THRESHOLD
# ─────────────────────────────────────────────────────────────

def compute_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    idx   = torch.argsort(probs, descending=True)
    tpr   = labels[idx].cumsum(0).float() / n_pos
    fpr   = (1 - labels[idx]).cumsum(0).float() / n_neg
    return torch.trapz(tpr, fpr).abs().item()


def find_best_threshold(probs: torch.Tensor,
                        labels: torch.Tensor) -> float:
    """F1'i maksimize eden eşik değeri."""
    best_f1, best_thr = 0.0, 0.5
    for thr in torch.linspace(0.05, 0.95, 91):
        preds = (probs >= thr).long()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
        if f1 > best_f1:
            best_f1, best_thr = f1, thr.item()
    return best_thr


# ─────────────────────────────────────────────────────────────
# 8.  EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SNN] Cihaz: {device.type.upper()}")

    print(f"[SNN] Dataset yükleniyor: {args.dataset}")
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

    n_pos = int(y_mod.sum().item())
    n_neg = int((y_mod == 0).sum().item())
    print(f"  Toplam: {len(y_mod):,}  |  Pozitif: {n_pos:,}  |  Negatif: {n_neg:,}")

    dataset    = TensorDataset(X_all, y_base, y_mod)
    train_size = int(0.85 * len(dataset))
    val_size   = len(dataset) - train_size
    generator  = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model = SNN(
        in_features=9,
        hidden=64,
        z_dim=128,
        proj_dim=32,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametre sayısı: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=1e-3, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Imbalance düzeltme
    pos_weight    = torch.tensor([n_neg / (n_pos + 1e-6)]).to(device)
    ntxent_loss   = NTXentLoss(temperature=args.temperature)
    ce_loss       = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_auc  = 0.0
    best_threshold = 0.5

    print(f"\n[SNN] Eğitim başlıyor — "
          f"mod_type={args.mod_type}, epochs={args.epochs}, "
          f"noise_std={args.noise_std}, T={args.temperature}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_sum = focal_sum = ntx_sum = 0.0
        all_tr_probs, all_tr_labels = [], []

        for x_b, y_base_b, y_mod_b in train_loader:
            x_b      = x_b.to(device)
            y_base_b = y_base_b.to(device)
            y_mod_b  = y_mod_b.to(device)

            # Augmented kopya üret
            x_aug_b  = augment(x_b, noise_std=args.noise_std)

            optimizer.zero_grad()

            mod_logits, base_logits, z_clean, z_aug, reliability = \
                model(x_b, x_aug_b)

            # Ağırlıklı Focal Loss
            loss_mod  = weighted_focal_loss(
                mod_logits.squeeze(), y_mod_b,
                weights=reliability,
                pos_weight=pos_weight,
                gamma=args.focal_gamma,
            )
            # NT-Xent contrastive loss
            loss_ntx  = ntxent_loss(z_clean, z_aug)
            # Base calling kaybı (ikincil)
            loss_base = ce_loss(base_logits, y_base_b)

            loss = loss_mod + args.lambda_base * loss_base \
                            + args.lambda_ntx  * loss_ntx

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            focal_sum      += loss_mod.item()
            ntx_sum        += loss_ntx.item()

            probs = torch.sigmoid(mod_logits.squeeze()).detach().cpu()
            all_tr_probs.append(probs)
            all_tr_labels.append(y_mod_b.cpu())

        scheduler.step()

        # ── Validation ──
        model.eval()
        all_val_probs, all_val_labels = [], []
        with torch.no_grad():
            for x_v, _, y_v in val_loader:
                x_v = x_v.to(device)
                p   = model.predict(x_v).cpu()
                all_val_probs.append(p)
                all_val_labels.append(y_v)

        vp = torch.cat(all_val_probs)
        vl = torch.cat(all_val_labels).long()
        tp = torch.cat(all_tr_probs)
        tl = torch.cat(all_tr_labels).long()

        val_auc = compute_auc(vp, vl)
        tr_auc  = compute_auc(tp, tl)

        # Her 10 epoch'ta optimal threshold güncelle
        if epoch % 10 == 0 or epoch == 1:
            best_threshold = find_best_threshold(vp, vl)

        val_acc = ((vp >= best_threshold).long() == vl).float().mean().item() * 100
        n_b     = len(train_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {total_loss_sum/n_b:.4f} "
            f"(focal {focal_sum/n_b:.4f} ntx {ntx_sum/n_b:.4f}) | "
            f"Tr AUC {tr_auc:.4f} | "
            f"Val AUC {val_auc:.4f} | "
            f"Val Acc {val_acc:.2f}% | "
            f"Thr {best_threshold:.2f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "model_state": model.state_dict(),
                "threshold":   best_threshold,
                "mod_type":    args.mod_type,
                "val_auc":     val_auc,
                "epoch":       epoch,
                "args":        vars(args),
            }, args.out_weights)
            print(f"  --> Yeni en iyi kaydedildi "
                  f"(AUC={val_auc:.4f}, Thr={best_threshold:.2f}): "
                  f"{args.out_weights}")

    print(f"\n[SNN] Tamamlandı.")
    print(f"  En iyi Val AUC  : {best_val_auc:.4f}")
    print(f"  Optimal threshold: {best_threshold:.3f}")
    print(f"  Model           : {args.out_weights}")


# ─────────────────────────────────────────────────────────────
# 9.  INFERENCE YARDIMCISI
# ─────────────────────────────────────────────────────────────

def load_and_predict(weights_path: str, dataset_path: str,
                     num_samples: int = 5000):
    """
    Eğitilmiş SNN modelini yükleyip tahmin yapar.
    inference_demo.py ile entegrasyon için kullanılabilir.

    Örnek:
        probs = load_and_predict("snn_best.pth", "clean_dataset.pt")
    """
    checkpoint = torch.load(weights_path, map_location="cpu",
                            weights_only=False)
    threshold  = checkpoint.get("threshold", 0.5)

    model = SNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loaded = torch.load(dataset_path, map_location="cpu",
                        weights_only=False)
    if isinstance(loaded, dict):
        X = loaded["signals"][:num_samples].float()
    else:
        X = loaded[0][:num_samples].float()

    with torch.no_grad():
        probs = model.predict(X).numpy()

    preds = (probs >= threshold).astype(int)
    print(f"[SNN Inference] {num_samples} örnek işlendi")
    print(f"  Modified    : {preds.sum()}")
    print(f"  Unmodified  : {(preds == 0).sum()}")
    return probs, preds


# ─────────────────────────────────────────────────────────────
# 10.  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SNN — Siamese Noise-Aware Network"
    )
    p.add_argument("--dataset",      required=True,
                   help="clean_dataset.pt (denoise çıktısı)")
    p.add_argument("--mod_type",     default="pU",
                   choices=["pU", "m6A"])
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--noise_std",    type=float, default=0.05,
                   help="Augmentation gürültü şiddeti (varsayılan: 0.05)")
    p.add_argument("--temperature",  type=float, default=0.2,
                   help="NT-Xent sıcaklık parametresi (varsayılan: 0.2)")
    p.add_argument("--focal_gamma",  type=float, default=2.0,
                   help="Focal Loss gamma (varsayılan: 2.0)")
    p.add_argument("--lambda_base",  type=float, default=0.3,
                   help="Base calling kayıp ağırlığı (varsayılan: 0.3)")
    p.add_argument("--lambda_ntx",   type=float, default=0.5,
                   help="NT-Xent kayıp ağırlığı (varsayılan: 0.5)")
    p.add_argument("--out_weights",  default="nanospeech_snn_best.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 65)
    print("  NanoSpeech-MTL — SNN (Siamese Noise-Aware Network)")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Mod type     : {args.mod_type}")
    print(f"  Epochs       : {args.epochs}  |  LR: {args.lr}")
    print(f"  Noise std    : {args.noise_std}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Focal gamma  : {args.focal_gamma}")
    print(f"  λ_base       : {args.lambda_base}")
    print(f"  λ_ntx        : {args.lambda_ntx}")
    print(f"  Kayıp        : w×Focal + λ₁×CE + λ₂×NT-Xent")
    print("=" * 65)
    train(args)
