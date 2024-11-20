import os
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from RealESRGAN.model import RealESRGAN


def setup_model():
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"

    device = torch.device(device_name)
    print(f"Device: {device}")
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    return model


def upscale_images(input_dir, model):
    model = setup_model()
    output_dir = Path(input_dir) / "upscaled"
    output_dir.mkdir(exist_ok=True)

    image_files = list(Path(input_dir).glob("*.png"))

    for img_path in tqdm(image_files, desc="Verarbeite Bilder"):
        try:
            image = Image.open(img_path).convert('RGB')

            # Garbage Collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            upscaled_image = model.predict(image)

            output_path = output_dir / f"{img_path.stem}_upscaled.png"
            upscaled_image.save(output_path, "PNG")

        except Exception as e:
            print(f"Fehler bei {img_path.name}: {e}")
            continue


def process_input_directories():
    # Hauptverzeichnis
    base_dir = Path("./input")

    # Finde alle Unterverzeichnisse
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]

    # Filtere Verzeichnisse, die noch keinen 'upscaled' Ordner haben
    dirs_to_process = [
        d for d in subdirs
        if not (d / "upscaled").exists()
    ]

    if not dirs_to_process:
        print("Keine neuen Verzeichnisse zum Verarbeiten gefunden.")
        return

    total_dirs = len(dirs_to_process)
    print(f"Gefundene Verzeichnisse zum Verarbeiten ({total_dirs}):")
    for dir_path in dirs_to_process:
        print(f"- {dir_path.name}")

    model = setup_model()

    # Verarbeite jedes Verzeichnis
    for dir_idx, dir_path in enumerate(tqdm(dirs_to_process, desc="Verzeichnisse", unit="Dir")):
        print(f"\nVerzeichnis {dir_idx + 1}/{total_dirs}")
        upscale_images(dir_path, model)


if __name__ == "__main__":
    # supress warnings
    process_input_directories()
