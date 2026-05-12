"""
denoise_labels_cl.py  —  Confident Learning tabanlı gürültü temizleyici
Orijinal denoise_labels.py (GMM) ile aynı CLI interface'ini kullanır.

Fark:
  - GMM: fiziksel kurala göre (turbulence / current drop) etiket atar
  - Confident Learning: modelin kendi güven skorlarına bakarak çelişkili
    etiketleri otomatik tespit eder ve temizler (cleanlab kütüphanesi)

Kullanım (orijinal ile birebir aynı):
    python src/denoise_labels_cl.py \
        --input  data/raw/full_production_dataset.pt \
        --mod_type pU \
        --output data/processed/clean_dataset_cl.pt

Gereksinim:
    pip install cleanlab
"""

import argparse
import numpy as np
import torch
from pathlib import Path

# cleanlab import — yoksa anlamlı hata ver
try:
    from cleanlab.filter import find_label_issues
    from cleanlab.count import estimate_cv_predicted_probabilities as estimate_cv_predicted_probs
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise ImportError(
        "cleanlab veya sklearn bulunamadı.\n"
        "Kur: pip install cleanlab scikit-learn"
    )


# ──────────────────────────────────────────────
# 1.  Veri yükleme
# ──────────────────────────────────────────────

def load_dataset(path: str):
    """
    .pt dosyasını yükler.
    Beklenen format (orijinal pipeline ile uyumlu):
        {
            "signals":  Tensor[N, 9]   — 9-feature sliding window (mean, std, dwell × 3 pencere)
            "labels":   Tensor[N]      — 0 = unmodified, 1 = modified
            "read_ids": list[str]      — opsiyonel, read başına id
        }
    """
    data = torch.load(path, map_location="cpu")
    signals = data["signals"]          # (N, 9)
    labels  = data["labels"].long()    # (N,)
    read_ids = data.get("read_ids", None)
    print(f"[load] {len(labels):,} örnek yüklendi  —  "
          f"pozitif: {labels.sum().item():,}  "
          f"negatif: {(labels==0).sum().item():,}")
    return signals, labels, read_ids, data


# ──────────────────────────────────────────────
# 2.  Confident Learning  —  çekirdek fonksiyon
# ──────────────────────────────────────────────

def find_noisy_labels_cl(signals: torch.Tensor,
                         labels:  torch.Tensor,
                         mod_type: str,
                         n_folds: int = 5) -> np.ndarray:
    """
    Confident Learning ile gürültülü etiketleri tespit eder.

    Adımlar:
      1. Hafif bir sklearn sınıflandırıcı (LogReg) ile cross-val
         olasılık tahminleri üret  →  pred_probs[N, 2]
      2. cleanlab.filter.find_label_issues ile çelişkili indeksleri bul
      3. Gürültülü olarak işaretlenen örnekleri döndür (bool mask)

    mod_type parametresi GMM sürümüyle uyum için alınır;
    Confident Learning fizik bilgisine ihtiyaç duymaz —
    ama mod_type'a göre küçük bir ağırlık ayarı yapılır
    (m6A genellikle mean-shift, pU std-spike → farklı feature ağırlığı).
    """
    X = signals.numpy().astype(np.float32)   # (N, 9)
    y = labels.numpy().astype(int)           # (N,)

    # mod_type'a göre feature ağırlığı (GMM mantığını yansıtır)
    # mean features: indeks 0,3,6  |  std features: 1,4,7
    feature_weights = np.ones(X.shape[1])
    if mod_type == "m6A":
        feature_weights[[0, 3, 6]] *= 2.0   # mean shift önemli
    elif mod_type == "pU":
        feature_weights[[1, 4, 7]] *= 2.0   # std (turbulence) önemli

    X_weighted = X * feature_weights

    print(f"[CL] {n_folds}-fold cross-val başlıyor  "
          f"(mod_type={mod_type}, N={len(y):,}) ...")

    # Basit ama güvenilir sınıflandırıcı
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",   # dengesiz sınıf durumuna karşı
            solver="lbfgs",
            n_jobs=-1,
        )
    )

    # Cross-val tahmin olasılıkları  →  cleanlab buna bakacak
    pred_probs = estimate_cv_predicted_probs(
        X_weighted, y, clf, cv_n_folds=n_folds
    )                                         # (N, 2)

    # Cleanlab: hangi örneklerin etiketi güvenilmez?
    noise_mask = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
        filter_by="both",                     # FP ve FN her ikisini de yakala
    )
    # noise_mask: gürültülü olduğu düşünülen indeksler listesi

    # Bool mask'e çevir
    is_noisy = np.zeros(len(y), dtype=bool)
    is_noisy[noise_mask] = True

    n_noisy = is_noisy.sum()
    pct = 100 * n_noisy / len(y)
    print(f"[CL] Gürültülü etiket tespiti tamamlandı: "
          f"{n_noisy:,} / {len(y):,}  ({pct:.1f}%) örnek kaldırılacak")

    return is_noisy


# ──────────────────────────────────────────────
# 3.  Temizleme & kaydetme
# ──────────────────────────────────────────────

def clean_and_save(signals, labels, read_ids, raw_data,
                   is_noisy: np.ndarray, output_path: str):
    keep = ~is_noisy

    clean_signals = signals[keep]
    clean_labels  = labels[keep]

    # read_ids varsa filtrele
    if read_ids is not None:
        if isinstance(read_ids, (list, np.ndarray)):
            clean_read_ids = [r for r, k in zip(read_ids, keep) if k]
        else:
            clean_read_ids = read_ids[keep]
    else:
        clean_read_ids = None

    # Orijinal veri sözlüğünü kopyala, üzerine temizlenmiş değerleri yaz
    out = dict(raw_data)
    out["signals"]  = clean_signals
    out["labels"]   = clean_labels
    if clean_read_ids is not None:
        out["read_ids"] = clean_read_ids
    out["denoising_method"] = "confident_learning"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)

    pos = clean_labels.sum().item()
    neg = (clean_labels == 0).sum().item()
    print(f"[save] Temiz veri kaydedildi → {output_path}")
    print(f"       Toplam: {len(clean_labels):,}  |  "
          f"pozitif: {pos:,}  |  negatif: {neg:,}")


# ──────────────────────────────────────────────
# 4.  CLI  (orijinal denoise_labels.py ile birebir aynı argümanlar)
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Confident Learning tabanlı gürültü temizleyici"
    )
    p.add_argument("--input",    required=True,
                   help="Ham .pt veri dosyası (orijinal pipeline çıktısı)")
    p.add_argument("--mod_type", required=True, choices=["pU", "m6A"],
                   help="Modifikasyon türü: pU veya m6A")
    p.add_argument("--output",   required=True,
                   help="Temizlenmiş veri için çıktı .pt dosyası")
    p.add_argument("--n_folds",  type=int, default=5,
                   help="Cross-val fold sayısı (varsayılan: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  NanoSpeech-MTL  —  Confident Learning Denoiser")
    print(f"  Girdi  : {args.input}")
    print(f"  Mod    : {args.mod_type}")
    print(f"  Çıktı  : {args.output}")
    print("=" * 60)

    signals, labels, read_ids, raw_data = load_dataset(args.input)
    is_noisy = find_noisy_labels_cl(
        signals, labels, args.mod_type, n_folds=args.n_folds
    )
    clean_and_save(signals, labels, read_ids, raw_data,
                   is_noisy, args.output)
    print("[✓] Confident Learning denoising tamamlandı.")


if __name__ == "__main__":
    main()
