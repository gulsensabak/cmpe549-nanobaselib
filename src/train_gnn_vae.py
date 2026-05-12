"""
train_gnn_vae.py  —  GNN-VAE Hibrit Mimari
===========================================
Mevcut CRNN ve Transformer mimarilerine alternatif olarak geliştirilen
tamamen farklı bir yaklaşım.

Temel fikir:
  - Her okuma (read), 3 düğümlü bir k-mer grafı olarak temsil edilir:
      prev_event → cur_event → next_event
  - GNN (Graph Attention Network), bu komşuluk ilişkilerini öğrenir
  - VAE, GNN çıktısını sıkıştırılmış latent uzaya (z ∈ R^32) taşır
  - Modification tahmini doğrudan z üzerinden yapılır
  - VAE'nin rekonstrüksiyon kaybı + KL ıraksaması ek düzenleme sağlar

Mevcut pipeline ile uyum:
  - Girdi: clean_dataset.pt (denoise_labels.py veya denoise_labels_cl.py çıktısı)
  - Çıktı: nanospeech_gnn_vae_best.pth ağırlık dosyası

Avantajlar:
  - Latent space doğrudan yorumlanabilir (modified vs unmodified ayrışır)
  - KL kaybı overfitting'i azaltır
  - GNN k-mer komşuluk bilgisini yapısal olarak kodlar
  - VAE'nin üretken yapısı veri artırımına (augmentation) imkân tanır

Gereksinimler:
  pip install torch torch-geometric

Kullanım:
  python src/train_gnn_vae.py \\
      --dataset data/processed/clean_dataset_cl.pt \\
      --epochs 60 \\
      --batch_size 512 \\
      --latent_dim 32 \\
      --out_weights nanospeech_gnn_vae_best.pth
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[UYARI] torch-geometric bulunamadı. Basit MLP fallback kullanılacak.")
    print("        Kurulum: pip install torch-geometric")


# ─────────────────────────────────────────────────────────────
# 1. GRAF OLUŞTURMA
#    Her örnek 3 düğüm (prev, cur, next), 9 özellikten oluşur.
#    Kenarlar: prev→cur, cur→next (yönlü)
# ─────────────────────────────────────────────────────────────

EDGE_INDEX = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()


def features_to_graph_batch(x_batch: torch.Tensor) -> "Batch":
    """
    (B, 9) tensörünü PyG Batch nesnesine dönüştürür.
    Her örnek: [prev_mean, prev_std, prev_dwell,
                cur_mean,  cur_std,  cur_dwell,
                next_mean, next_std, next_dwell]
    → 3 düğüm × 3 özellik
    """
    graphs = []
    edge_index = EDGE_INDEX.to(x_batch.device)
    for i in range(x_batch.size(0)):
        node_features = x_batch[i].view(3, 3)   # (3 düğüm, 3 özellik)
        graphs.append(Data(x=node_features, edge_index=edge_index))
    return Batch.from_data_list(graphs)


# ─────────────────────────────────────────────────────────────
# 2. GNN ENCODER
#    2 katmanlı Graph Attention Network
#    Global mean pool → h ∈ R^hidden_dim
# ─────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    def __init__(self, node_features: int = 3, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(node_features, hidden_dim // heads, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        return global_mean_pool(x, batch)   # (B, hidden_dim)


# ─────────────────────────────────────────────────────────────
# 3. GNN-VAE ANA MİMARİ
# ─────────────────────────────────────────────────────────────

class GNNVAEModDetector(nn.Module):
    """
    GNN-VAE hibrit modifikasyon dedektörü.

    Akış:
        Girdi (B, 9)
        → GNN Encoder → h ∈ R^64
        → mu head, logvar head → z ∈ R^latent_dim  (reparameterization)
        → Decoder → x_hat ∈ R^9          (rekonstrüksiyon)
        → Mod head → p(mod) ∈ [0,1]      (modifikasyon tahmini)
    """

    def __init__(self, node_features: int = 3, hidden_dim: int = 64,
                 latent_dim: int = 32, gnn_heads: int = 4):
        super().__init__()

        # GNN encoder (PyG gerektirir)
        if HAS_PYG:
            self.gnn_encoder = GNNEncoder(node_features, hidden_dim, gnn_heads)
        else:
            # Fallback: düz MLP (PyG yoksa)
            self.gnn_encoder = nn.Sequential(
                nn.Linear(9, hidden_dim), nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
            )

        # VAE kafaları
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Decoder: sinyal rekonstrüksiyonu
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 9),   # 9 orijinal özelliği yeniden üret
        )

        # Modifikasyon kafası: latent z → olasılık
        self.mod_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(16, 1),
        )

        self.latent_dim = latent_dim

    def encode(self, x_batch: torch.Tensor):
        """Girdiyi latent parametrelerine dönüştür."""
        if HAS_PYG:
            graph_batch = features_to_graph_batch(x_batch)
            h = self.gnn_encoder(
                graph_batch.x, graph_batch.edge_index, graph_batch.batch
            )
        else:
            h = self.gnn_encoder(x_batch)

        mu     = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Eğitimde stokastik örnekleme, değerlendirmede mu kullan."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x_batch: torch.Tensor):
        mu, logvar = self.encode(x_batch)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decoder(z)
        mod_logits = self.mod_head(z)
        return x_hat, mod_logits, mu, logvar


# ─────────────────────────────────────────────────────────────
# 4. KAYIP FONKSİYONU
#    L = MSE(x, x_hat) + beta * KL(q||p) + lambda * BCE(mod)
# ─────────────────────────────────────────────────────────────

def gnn_vae_loss(x, x_hat, mod_logits, mod_labels,
                 mu, logvar,
                 criterion_mod,
                 beta: float = 1.0,
                 lambda_mod: float = 2.0):
    """
    Toplam kayıp:
      recon_loss : Sinyal rekonstrüksiyon hatası (MSE)
      kl_loss    : KL ıraksaması — latent uzayı N(0,I)'ye zorlar
      mod_loss   : Modifikasyon sınıflandırma kaybı (BCE)
    """
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    mod_loss   = criterion_mod(mod_logits, mod_labels.unsqueeze(1))
    total      = recon_loss + beta * kl_loss + lambda_mod * mod_loss
    return total, recon_loss, kl_loss, mod_loss


# ─────────────────────────────────────────────────────────────
# 5. EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GNN-VAE] Cihaz: {device.type.upper()}")

    print(f"[GNN-VAE] Dataset yükleniyor: {args.dataset}")
    loaded = torch.load(args.dataset, weights_only=False)

    # Mevcut pipeline formatlarını destekle
    if isinstance(loaded, dict):
        X_all    = loaded["signals"]                  # (N, 9)
        y_mod    = loaded["labels"].float()
        y_base   = torch.zeros(len(y_mod), dtype=torch.long)  # GNN-VAE base calling yapmaz
    else:
        X_all, y_base, y_mod = loaded
        X_all  = X_all.float()
        y_mod  = y_mod.float()

    print(f"  Toplam örnek : {len(y_mod):,}")
    print(f"  Pozitif (mod): {int(y_mod.sum()):,}")
    print(f"  Negatif      : {int((y_mod==0).sum()):,}")

    dataset    = TensorDataset(X_all, y_mod)
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    generator  = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model = GNNVAEModDetector(
        node_features=3,
        hidden_dim=64,
        latent_dim=args.latent_dim,
        gnn_heads=4,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Imbalance düzeltme
    n_pos = y_mod.sum().item()
    n_neg = len(y_mod) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)]).to(device)
    criterion_mod = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    print("\n[GNN-VAE] Eğitim başlıyor...\n")

    for epoch in range(1, args.epochs + 1):
        # — Eğitim —
        model.train()
        total_loss_sum = recon_sum = kl_sum = mod_sum = 0.0
        train_correct = train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            x_hat, mod_logits, mu, logvar = model(x_batch)

            loss, recon_l, kl_l, mod_l = gnn_vae_loss(
                x_batch, x_hat, mod_logits, y_batch,
                mu, logvar, criterion_mod,
                beta=args.beta, lambda_mod=args.lambda_mod,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            recon_sum      += recon_l.item()
            kl_sum         += kl_l.item()
            mod_sum        += mod_l.item()

            preds = (torch.sigmoid(mod_logits.squeeze()) > 0.5).long()
            train_correct += (preds == y_batch.long()).sum().item()
            train_total   += len(y_batch)

        scheduler.step()

        # — Doğrulama —
        model.eval()
        val_correct = val_total = 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                _, mod_logits_v, _, _ = model(x_val)
                probs = torch.sigmoid(mod_logits_v.squeeze())
                preds = (probs > 0.5).long()
                val_correct += (preds == y_val.long()).sum().item()
                val_total   += len(y_val)
                all_probs.append(probs.cpu())
                all_labels.append(y_val.cpu())

        val_acc  = 100 * val_correct / val_total
        tr_acc   = 100 * train_correct / train_total
        n_batches = len(train_loader)

        # Basit AUC tahmini (ROC eğrisi gerektirmez)
        all_probs_t  = torch.cat(all_probs)
        all_labels_t = torch.cat(all_labels).long()
        sorted_idx   = torch.argsort(all_probs_t, descending=True)
        n_pos_v      = all_labels_t.sum().item()
        n_neg_v      = len(all_labels_t) - n_pos_v
        tp_cumsum    = all_labels_t[sorted_idx].cumsum(0).float()
        fp_cumsum    = (1 - all_labels_t[sorted_idx]).cumsum(0).float()
        tpr = tp_cumsum / (n_pos_v + 1e-8)
        fpr = fp_cumsum / (n_neg_v + 1e-8)
        auc = torch.trapz(tpr, fpr).abs().item()

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {total_loss_sum/n_batches:.3f} "
            f"(recon {recon_sum/n_batches:.3f} kl {kl_sum/n_batches:.3f} mod {mod_sum/n_batches:.3f}) | "
            f"Tr Acc {tr_acc:.1f}% | "
            f"Val Acc {val_acc:.1f}% | "
            f"Val AUC {auc:.4f}"
        )

        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), args.out_weights)
            print(f"  --> Yeni en iyi model kaydedildi (AUC={auc:.4f}): {args.out_weights}")

    print(f"\n[GNN-VAE] Eğitim tamamlandı. En iyi Val AUC: {best_val_auc:.4f}")


# ─────────────────────────────────────────────────────────────
# 6. LATENT SPACE GÖRSELLEŞTIRME YARDIMCISI
#    (isteğe bağlı — eğitim sonrası çağrılabilir)
# ─────────────────────────────────────────────────────────────

def extract_latents(model, dataset_path: str, device, max_samples: int = 5000):
    """
    Eğitilmiş modelin latent z vektörlerini ve etiketleri döndürür.
    t-SNE veya UMAP ile görselleştirme için kullanılabilir.

    Örnek kullanım:
        z_all, y_all = extract_latents(model, "clean_dataset.pt", device)
        # sklearn.manifold.TSNE ile 2D görselleştirme yap
    """
    model.eval()
    loaded = torch.load(dataset_path, weights_only=False)
    if isinstance(loaded, dict):
        X = loaded["signals"][:max_samples]
        y = loaded["labels"][:max_samples]
    else:
        X, _, y = loaded
        X = X[:max_samples]
        y = y[:max_samples]

    with torch.no_grad():
        mu, _ = model.encode(X.to(device))
    return mu.cpu().numpy(), y.numpy()


# ─────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GNN-VAE Hibrit Modification Detector")
    p.add_argument("--dataset",     required=True,
                   help="clean_dataset.pt (denoise çıktısı)")
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--latent_dim",  type=int,   default=32,
                   help="VAE latent boyutu (varsayılan: 32)")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--beta",        type=float, default=1.0,
                   help="KL ağırlığı — artırılırsa latent uzay daha düzgün olur")
    p.add_argument("--lambda_mod",  type=float, default=2.0,
                   help="Mod kayıp ağırlığı — artırılırsa sınıflandırma odaklanır")
    p.add_argument("--out_weights", default="nanospeech_gnn_vae_best.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 65)
    print("  NanoSpeech-MTL — GNN-VAE Hibrit Modification Detector")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Latent dim : {args.latent_dim}")
    print(f"  Beta (KL)  : {args.beta}")
    print(f"  Lambda mod : {args.lambda_mod}")
    print(f"  PyG        : {'var' if HAS_PYG else 'YOK — MLP fallback'}")
    print("=" * 65)
    train(args)
