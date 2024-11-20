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

    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    )
    pipeline = pipeline.to(device)
    # Optional: Speicheroptimierung
    if device == "cuda":
        pipeline.enable_memory_efficient_fp16()
        pipeline.enable_attention_slicing()
    return pipeline


def upscale_images(input_dir):
    pipeline = setup_pipeline()

    # Erstelle Upscaled Verzeichnis
    output_dir = Path(input_dir) / "Upscaled"
    output_dir.mkdir(exist_ok=True)

    # Verarbeite alle PNG Dateien
    image_files = list(Path(input_dir).glob("*.png"))

    for img_path in tqdm(image_files):
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


if __name__ == "__main__":
    # Pfad zum Ordner mit den Bildern
    input_directory = "./input"
    upscale_images(input_directory)