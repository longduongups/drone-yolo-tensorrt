"""
Conversion YOLOv8 PyTorch (.pt) → TensorRT (.engine) en FP16.

"""

from pathlib import Path
from ultralytics import YOLO
import time
import torch


# Répertoire où les modèles seront téléchargés puis convertis
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

# Tailles de modèles à benchmarker : nano, small, medium
# On teste les trois pour montrer le compromis vitesse/précision
MODEL_SIZES = ["yolov8n", "yolov8s", "yolov8m"]

IMG_SIZE = 640


def convert_model(model_name: str) -> None:
    """
    Convertit un modèle YOLOv8 .pt en TensorRT .engine (FP16).

    Le fichier .pt est téléchargé automatiquement par Ultralytics
    s'il n'est pas déjà présent. La conversion passe par ONNX puis
    par TensorRT via l'API Ultralytics `export()`.
    """
    pt_path = MODELS_DIR / f"{model_name}.pt"
    engine_path = MODELS_DIR / f"{model_name}.engine"

    print(f"\n{'=' * 60}")
    print(f"Conversion de {model_name}")
    print(f"{'=' * 60}")

    # Chargement (télécharge le .pt s'il n'existe pas)
    model = YOLO(pt_path if pt_path.exists() else f"{model_name}.pt")

    # Si Ultralytics a téléchargé dans le dossier courant, on déplace
    default_download = Path(f"{model_name}.pt")
    if default_download.exists() and not pt_path.exists():
        default_download.rename(pt_path)

    # Export vers TensorRT en FP16
    # imgsz : résolution d'entrée du réseau (doit matcher l'usage réel)
    # half=True : active la précision FP16 (2x plus rapide sur RTX 4060)
    # device=0 : GPU 0 (votre RTX 4060)
    # workspace : mémoire max allouée pour l'optimisation (Go)
    print(f"Export vers TensorRT FP16 en cours (peut prendre 3-5 min)...")
    t0 = time.time()
    model.export(
        format="engine",
        imgsz=IMG_SIZE,
        half=True,
        device=0,
        workspace=4,
    )
    dt = time.time() - t0

    # Le fichier .engine est créé à côté du .pt d'origine, on le déplace
    engine_created = Path(f"{model_name}.engine")
    if engine_created.exists():
        engine_created.rename(engine_path)

    print(f"✅ {model_name}.engine créé en {dt:.1f}s")
    print(f"   Taille : {engine_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    """
    Point d'entrée : vérifie l'environnement et convertit chaque modèle.
    """
    # Vérification CUDA obligatoire pour TensorRT
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA n'est pas disponible. TensorRT nécessite un GPU NVIDIA."
        )

    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
    print(f"CUDA        : {torch.version.cuda}")
    print(f"PyTorch     : {torch.__version__}")

    # Conversion de chaque taille
    for model_name in MODEL_SIZES:
        convert_model(model_name)

    print(f"\n{'=' * 60}")
    print("Toutes les conversions terminées.")
    print(f"Fichiers .engine dans : {MODELS_DIR.absolute()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()