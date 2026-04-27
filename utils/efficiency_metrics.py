import time
import torch

def measure_inference_zero_shot(model, img_path, central_point):
    """Mide la latencia y el consumo de VRAM de una inferencia zero-shot
    usando un punto central como prompt."""
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1024**2 # Pasamos de Bytes a MB
    
    start   = time.time()
    results = model(img_path, points=central_point, labels=[1], device="cuda", verbose=False)
    latency = (time.time() - start) * 1000  # Pasamos de segundos a ms

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**2 - vram_before
    else:
        vram = 0

    return results, latency, vram


def measure_inference_refcocog(model, img_path, bbox):
    """Esta función se diferencia de la anterior únicamente en el hecho de que usa
    una caja delimitadora como prompt en lugar de un punto central."""
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1024**2

    start = time.time()
    results = model(img_path, bboxes=[bbox], device="cuda", verbose=False)
    latency = (time.time() - start) * 1000

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**2 - vram_before
    else:
        vram = 0

    return results, latency, vram


def measure_inference_sam3_prompt_zero_shot(predictor, img_path, text_prompt):
    """Mide la latencia y el consumo de VRAM de una inferencia zero-shot
    con SAM3 usando un prompt textual."""
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1024**2

    start = time.time()
    predictor.set_image(img_path)
    predictor.model.set_classes(text=[text_prompt])
    predictor.prompts["text"] = [text_prompt]
    results = predictor()
    
    latency = (time.time() - start) * 1000

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**2 - vram_before
    else:
        vram = 0

    return results, latency, vram
