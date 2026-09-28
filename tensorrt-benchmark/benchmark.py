"""
Benchmark comparatif PyTorch FP32 vs TensorRT FP16 sur YOLOv8 (n/s/m).

Mesure la latence d'inférence et le FPS de chaque modèle, dans les deux
formats, sur 100 prédictions avec warmup GPU rigoureux (10 runs jetés).

Sortie :
    - results/benchmark.csv : tableau brut des mesures
    - results/benchmark.png : graphique comparatif (latence + FPS)
    - Console : résumé formaté avec speedup

"""

from pathlib import Path
from typing import Dict, List
import time
import statistics

import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd


# ===================== Configuration =====================

MODELS_DIR = Path("./models")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZES = ["yolov8n", "yolov8s", "yolov8m"]
FORMATS = ["pt", "engine"]  # PyTorch vs TensorRT FP16

# Paramètres du benchmark
WARMUP_RUNS = 10       # Runs de chauffe (jetés) — cache CUDA, compilation
BENCHMARK_RUNS = 100   # Runs mesurés
IMG_SIZE = 640

# Classes COCO filtrées (identique au projet drone : personne + voiture)
CLASSES = [0, 2]


# ===================== Fonctions =====================

def create_test_image() -> np.ndarray:
    """
    Crée une image de test synthétique (640x640, BGR, valeurs aléatoires).
    Pour un benchmark de latence pure, une image aléatoire suffit :
    le temps d'inférence dépend de la taille du réseau, pas du contenu.
    """
    return np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


def benchmark_model(
    model_path: Path,
    test_image: np.ndarray,
    warmup: int,
    n_runs: int,
) -> Dict[str, float]:
    """
    Benchmark d'un modèle sur `n_runs` prédictions, avec warmup GPU.

    Retour :
        latency_mean_ms, latency_std_ms, latency_min_ms, latency_max_ms, fps
    """
    model = YOLO(model_path)

    # Warmup : réchauffe le GPU (cache CUDA, compilation JIT, etc.)
    for _ in range(warmup):
        _ = model.predict(test_image, classes=CLASSES, verbose=False)
    torch.cuda.synchronize()

    # Mesures : sur n_runs prédictions, avec sync GPU avant/après
    latencies_ms: List[float] = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict(test_image, classes=CLASSES, verbose=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    return {
        "latency_mean_ms": mean_ms,
        "latency_std_ms": statistics.stdev(latencies_ms),
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "fps": 1000.0 / mean_ms,
    }


def format_row(name: str, fmt: str, stats: Dict[str, float]) -> str:
    """Formate une ligne du tableau console."""
    return (
        f"{name:<10} {fmt:<8} "
        f"{stats['latency_mean_ms']:>7.2f} ± {stats['latency_std_ms']:>4.2f}   "
        f"{stats['latency_min_ms']:>6.2f}   {stats['latency_max_ms']:>6.2f}   "
        f"{stats['fps']:>6.1f}"
    )


# ===================== Programme principal =====================

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible.")

    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"CUDA : {torch.version.cuda}")
    print(f"Warmup : {WARMUP_RUNS} runs | Mesures : {BENCHMARK_RUNS} runs\n")

    test_image = create_test_image()
    results = []

    header = (
        f"{'Modèle':<10} {'Format':<8} "
        f"{'Latence moy (ms)':>16}   {'Min':>6}   {'Max':>6}   {'FPS':>6}"
    )
    print(header)
    print("-" * len(header))

    for size in MODEL_SIZES:
        for fmt in FORMATS:
            model_path = MODELS_DIR / f"{size}.{fmt}"
            if not model_path.exists():
                print(f" {model_path} manquant, skip.")
                continue

            stats = benchmark_model(
                model_path, test_image, WARMUP_RUNS, BENCHMARK_RUNS
            )
            print(format_row(size, fmt, stats))
            results.append({"model": size, "format": fmt, **stats})

    # ===================== Sauvegarde CSV =====================
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "benchmark.csv"
    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"\nCSV sauvegardé : {csv_path.absolute()}")

    # ===================== Graphique =====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(MODEL_SIZES))
    width = 0.35

    pt_lat = [
        df[(df["model"] == m) & (df["format"] == "pt")]["latency_mean_ms"].iloc[0]
        for m in MODEL_SIZES
    ]
    trt_lat = [
        df[(df["model"] == m) & (df["format"] == "engine")]["latency_mean_ms"].iloc[0]
        for m in MODEL_SIZES
    ]

    # Sous-graphique 1 : latence
    ax1.bar(x - width/2, pt_lat, width, label="PyTorch FP32", color="#4C72B0")
    ax1.bar(x + width/2, trt_lat, width, label="TensorRT FP16", color="#55A868")
    ax1.set_xlabel("Modèle")
    ax1.set_ylabel("Latence (ms)")
    ax1.set_title("Latence d'inférence (plus bas = mieux)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_SIZES)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for i, (pt, trt) in enumerate(zip(pt_lat, trt_lat)):
        ax1.text(i - width/2, pt + 0.5, f"{pt:.1f}", ha="center", fontsize=9)
        ax1.text(i + width/2, trt + 0.5, f"{trt:.1f}", ha="center", fontsize=9)

    # Sous-graphique 2 : FPS
    pt_fps = [1000 / lat for lat in pt_lat]
    trt_fps = [1000 / lat for lat in trt_lat]
    ax2.bar(x - width/2, pt_fps, width, label="PyTorch FP32", color="#4C72B0")
    ax2.bar(x + width/2, trt_fps, width, label="TensorRT FP16", color="#55A868")
    ax2.set_xlabel("Modèle")
    ax2.set_ylabel("FPS")
    ax2.set_title("Débit (FPS) — plus haut = mieux")
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_SIZES)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for i, (pt, trt) in enumerate(zip(pt_fps, trt_fps)):
        ax2.text(i - width/2, pt + 2, f"{pt:.0f}", ha="center", fontsize=9)
        ax2.text(i + width/2, trt + 2, f"{trt:.0f}", ha="center", fontsize=9)

    plt.suptitle(
        "Benchmark YOLOv8 : PyTorch vs TensorRT FP16 (RTX 4060 Laptop)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    png_path = RESULTS_DIR / "benchmark.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Graphique sauvegardé : {png_path.absolute()}")

    # ===================== Résumé speedup =====================
    print("\n" + "=" * 60)
    print("Speedup TensorRT vs PyTorch")
    print("=" * 60)
    for m in MODEL_SIZES:
        pt = df[(df["model"] == m) & (df["format"] == "pt")]["latency_mean_ms"].iloc[0]
        trt = df[(df["model"] == m) & (df["format"] == "engine")]["latency_mean_ms"].iloc[0]
        print(f"  {m}: {pt:.1f} ms → {trt:.1f} ms  ({pt/trt:.2f}x plus rapide)")


if __name__ == "__main__":
    main()