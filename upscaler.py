from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
from tqdm import tqdm


def setup_pipeline():
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    dtype = torch.float16 # if device == "cuda" else torch.float32
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    )
    pipeline = pipeline.to(device)
    # Optional: Speicheroptimierung
    if True: # device == "cuda":
        pipeline.enable_attention_slicing()
    return pipeline


def upscale_images(input_dir):
    pipeline = setup_pipeline()

    # Erstelle Upscaled Verzeichnis
    output_dir = Path(input_dir) / "upscaled"
    output_dir.mkdir(exist_ok=True)

    # Verarbeite alle PNG Dateien
    image_files = list(Path(input_dir).glob("*.png"))
    total_images = len(image_files)

    print(f"\nVerarbeite Verzeichnis: {Path(input_dir).name}")

    for idx, img_path in enumerate(tqdm(image_files, desc="Bilder", unit="Bild")):
        print(f"Verarbeite Bild {idx + 1}/{total_images} ({img_path.name})")

        # Lade und verarbeite Bild
        image = Image.open(img_path).convert("RGB")

        # Generiere einen passenden Prompt basierend auf dem Dateinamen
        prompt = "high quality, detailed, high resolution upscale"

        # Upscale durchführen
        upscaled_image = pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]

        # Speichere upscaled Bild
        output_path = output_dir / f"{img_path.stem}_upscaled.png"
        upscaled_image.save(output_path, "PNG")


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

    # Verarbeite jedes Verzeichnis
    for dir_idx, dir_path in enumerate(tqdm(dirs_to_process, desc="Verzeichnisse", unit="Dir")):
        print(f"\nVerzeichnis {dir_idx + 1}/{total_dirs}")
        upscale_images(dir_path)


if __name__ == "__main__":
    process_input_directories()
