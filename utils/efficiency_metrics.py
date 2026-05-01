import time
import torch

torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda:0")

def measure_inference_central_point(model, img_path, central_point):
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

def measure_inference_fine_tuning(predictor, image, point_coords, point_labels):
    """Esta función mide la latencia y el consumo de VRAM de una inferencia sobre
    un modelo fine-tuneado usando coordenadas de puntos como prompt."""
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1024**2
    
    start   = time.time()
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)
    latency = (time.time() - start) * 1000  # Está en ms

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**2 - vram_before
    else:
        vram = 0

    return masks, scores, latency, vram

def measure_inference_sam3_prompt_refcocog(sam_wrapper, img_path, text_prompt):
    """Esta función mide la latencia y el consumo de VRAM de una inferencia con SAM 3
    usando un prompt textual. Primero localiza el objeto mediante Grounding DINO
    y luego segmenta la región detectada con SAM 3."""
    from groundingdino.util.inference import load_model, load_image, predict as gd_predict

    GROUNDING_DINO_CONFIG = "C:\\Users\\DanielTalavera\\Desktop\\Trabajo_Fin_de_Grado\\Grounding_Dino\\GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_WEIGHTS = "C:\\Users\\DanielTalavera\\Desktop\\Trabajo_Fin_de_Grado\\Grounding_Dino\\groundingdino_swint_ogc.pth"
    gd_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_WEIGHTS).to(device)

    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1024**2
    start = time.time()

    image_source, image_tensor = load_image(img_path)
    boxes, logits, _ = gd_predict(
        model=gd_model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=0.3,
        text_threshold=0.25,
        device=device
    )

    if boxes is None or len(boxes) == 0:
        latency = (time.time() - start) * 1000
        vram = torch.cuda.max_memory_allocated() / 1024**2 - vram_before
        return None, latency, vram

    H, W, _ = image_source.shape
    boxes_xyxy = boxes * torch.tensor([W, H, W, H], dtype=torch.float32).to(boxes.device)
    cx, cy, bw, bh = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    boxes_xyxy = torch.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], dim=1)

    best_box = boxes_xyxy[logits.argmax()].unsqueeze(0).cpu().numpy().tolist()

    results = sam_wrapper(img_path, bboxes=best_box)

    latency = (time.time() - start) * 1000
    vram = torch.cuda.max_memory_allocated() / 1024**2 - vram_before
    return results, latency, vram

def measure_inference_fine_tuning_refcocog(predictor, image, bbox):
    """Mide la latencia y el consumo de VRAM de una inferencia sobre
    un modelo fine-tuneado usando una caja delimitadora como prompt."""
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1024**2
    start = time.time()
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=bbox, multimask_output=True)
    latency = (time.time() - start) * 1000
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**2 - vram_before
    else:
        vram = 0
    return masks, scores, latency, vram
