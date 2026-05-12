"""
fast5_to_dataset.py
====================
demo_0.fast5 dosyasından direkt olarak pipeline'ın beklediği
dataset.pt dosyasını üretir.

eventalign.txt veya nanopolish gerekmez.

Yapılan işlem:
  1. Her read'in ham elektrik sinyalini oku
  2. pA (picoampere) birimine normalize et
  3. Kayan pencere (sliding window, boyut=3 event) ile
     9-feature vektörler üret:
       [prev_mean, prev_std, prev_dwell,
        cur_mean,  cur_std,  cur_dwell,
        next_mean, next_std, next_dwell]
  4. Basit eşik tabanlı geçici etiket ata
     (CL denoiser bunları zaten düzeltecek)
  5. torch.save ile .pt olarak kaydet

Kullanım:
    python src/fast5_to_dataset.py \
        --fast5  NanoBaseLib/demo_dataset/1_raw_signal/multi_fast5/demo_0.fast5 \
        --output data/raw/full_production_dataset.pt \
        --window 100 \
        --step   50
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch


# ─────────────────────────────────────────────
# Yardımcı: ham sinyal → pA normalize
# ─────────────────────────────────────────────
def raw_to_pA(signal: np.ndarray, digitisation: float,
              offset: float, rng: float) -> np.ndarray:
    """Oxford Nanopore ham ADC değerlerini picoampere'e çevirir."""
    return (signal + offset) * (rng / digitisation)


# ─────────────────────────────────────────────
# Kayan pencere ile 9-feature vektör üret
# ─────────────────────────────────────────────
def extract_features(signal_pA: np.ndarray,
                     window: int = 100,
                     step: int   = 50) -> np.ndarray:
    """
    Sinyali örtüşen pencerelere böler, her pencereden
    mean/std/dwell üretir ve 3'lü sliding window ile
    9-feature vektörler oluşturur.

    Çıktı: (N, 9) float32
    """
    n = len(signal_pA)
    events = []
    pos = 0
    while pos + window <= n:
        chunk = signal_pA[pos: pos + window]
        events.append({
            "mean":  float(np.mean(chunk)),
            "std":   float(np.std(chunk)),
            "dwell": float(window),
        })
        pos += step

    if len(events) < 3:
        return np.empty((0, 9), dtype=np.float32)

    rows = []
    for i in range(1, len(events) - 1):
        prev, cur, nxt = events[i-1], events[i], events[i+1]
        rows.append([
            prev["mean"], prev["std"], prev["dwell"],
            cur["mean"],  cur["std"],  cur["dwell"],
            nxt["mean"],  nxt["std"],  nxt["dwell"],
        ])
    return np.array(rows, dtype=np.float32)


# ─────────────────────────────────────────────
# Geçici etiket: std spike → pU tahmini
# CL denoiser bunları zaten temizleyecek
# ─────────────────────────────────────────────
def assign_temp_labels(features: np.ndarray,
                       mod_type: str = "pU") -> np.ndarray:
    """
    Fizik tabanlı basit eşik:
      pU  → merkez std (indeks 4) yüksekse modified
      m6A → merkez mean (indeks 3) düşükse modified
    Bu etiketler gürültülüdür; CL denoiser temizleyecek.
    """
    if mod_type == "pU":
        col = features[:, 4]          # merkez std
        threshold = np.percentile(col, 75)
        labels = (col > threshold).astype(np.int64)
    else:  # m6A
        col = features[:, 3]          # merkez mean
        threshold = np.percentile(col, 25)
        labels = (col < threshold).astype(np.int64)
    return labels


# ─────────────────────────────────────────────
# Ana fonksiyon
# ─────────────────────────────────────────────
def fast5_to_dataset(fast5_path: str, output_path: str,
                     window: int = 100, step: int = 50,
                     mod_type: str = "pU",
                     max_reads: int = None):

    all_features = []
    all_labels   = []
    all_read_ids = []

    print(f"[1/3] Fast5 okunuyor: {fast5_path}")
    with h5py.File(fast5_path, "r") as f:
        read_ids = list(f.keys())
        if max_reads:
            read_ids = read_ids[:max_reads]

        total = len(read_ids)
        print(f"      Toplam {total} read işlenecek...")

        for i, rid in enumerate(read_ids):
            if i % 500 == 0:
                print(f"      [{i}/{total}] işleniyor...")
            try:
                raw = f[rid]["Raw/Signal"][:]
                ch  = f[rid]["channel_id"].attrs
                pA  = raw_to_pA(
                    raw.astype(np.float32),
                    float(ch["digitisation"]),
                    float(ch["offset"]),
                    float(ch["range"]),
                )
                feats = extract_features(pA, window=window, step=step)
                if len(feats) == 0:
                    continue
                labels = assign_temp_labels(feats, mod_type=mod_type)
                all_features.append(feats)
                all_labels.append(labels)
                all_read_ids.extend([rid] * len(feats))

            except Exception as e:
                print(f"      UYARI: {rid} atlandı — {e}")
                continue

    if len(all_features) == 0:
        raise RuntimeError("Hiç feature üretilemedi!")

    print(f"[2/3] Tensor'a çevriliyor...")
    X = torch.tensor(np.vstack(all_features), dtype=torch.float32)
    y = torch.tensor(np.concatenate(all_labels), dtype=torch.float32)

    pos = int(y.sum().item())
    neg = int((y == 0).sum().item())
    print(f"      Toplam: {len(y):,} örnek")
    print(f"      Pozitif (modified): {pos:,}  |  Negatif: {neg:,}")

    print(f"[3/3] Kaydediliyor: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "signals":  X,
        "labels":   y,
        "read_ids": all_read_ids,
        "source":   fast5_path,
        "mod_type": mod_type,
        "window":   window,
        "step":     step,
        "note":     "Geçici etiketler — CL denoiser ile temizlenecek",
    }, output_path)

    print(f"[✓] dataset.pt oluşturuldu → {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="fast5 → dataset.pt dönüştürücü"
    )
    p.add_argument("--fast5",      required=True,
                   help="demo_0.fast5 dosya yolu")
    p.add_argument("--output",     required=True,
                   help="Çıktı .pt dosyası")
    p.add_argument("--window",     type=int, default=100,
                   help="Sinyal pencere boyutu (varsayılan: 100)")
    p.add_argument("--step",       type=int, default=50,
                   help="Pencere adım boyutu (varsayılan: 50)")
    p.add_argument("--mod_type",   default="pU",
                   choices=["pU", "m6A"])
    p.add_argument("--max_reads",  type=int, default=None,
                   help="Test için sadece ilk N read'i işle")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  fast5 → dataset.pt dönüştürücü")
    print(f"  Girdi  : {args.fast5}")
    print(f"  Çıktı  : {args.output}")
    print(f"  Window : {args.window}  Step: {args.step}")
    print("=" * 60)
    fast5_to_dataset(
        fast5_path = args.fast5,
        output_path= args.output,
        window     = args.window,
        step       = args.step,
        mod_type   = args.mod_type,
        max_reads  = args.max_reads,
    )
