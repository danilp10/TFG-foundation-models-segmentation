import time
import torch

torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda:0")


def measure_inference_central_point(model, img_path, central_point):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    results = model(img_path, points=central_point, labels=[1], device="cuda", verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.time() - start) * 1000

    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1024**2
    else:
        vram = 0

    return results, latency, vram


def measure_inference_refcocog(model, img_path, bbox):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    results = model(img_path, bboxes=[bbox], device="cuda", verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.time() - start) * 1000

    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1024**2
    else:
        vram = 0

    return results, latency, vram


def measure_inference_sam3_prompt_zero_shot(predictor, img_path, text_prompt):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    predictor.set_image(img_path)
    predictor.model.set_classes(text=[text_prompt])
    predictor.prompts["text"] = [text_prompt]
    predictor.args.save = False
    predictor.args.show = False
    predictor.args.save_txt = False
    predictor.args.save_crop = False
    results = predictor()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.time() - start) * 1000

    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1024**2
    else:
        vram = 0

    return results, latency, vram


def measure_inference_fine_tuning(predictor, image, point_coords, point_labels):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.time() - start) * 1000

    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1024**2
    else:
        vram = 0

    return masks, scores, latency, vram


def measure_inference_fine_tuning_refcocog(predictor, image, bbox):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=bbox, multimask_output=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.time() - start) * 1000

    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1024**2
    else:
        vram = 0

    return masks, scores, latency, vram


class GroundingDinoSAM3Inference:
    def __init__(self, sam_wrapper, gd_config_path, gd_weights_path, box_threshold=0.3, text_threshold=0.25):
        from groundingdino.util.inference import load_model

        self.sam_wrapper = sam_wrapper
        self.gd_model = load_model(gd_config_path, gd_weights_path).to(device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def __call__(self, _unused_first_arg, img_path, text_prompt):
        return self.run(img_path, text_prompt)

    def run(self, img_path, text_prompt):
        from groundingdino.util.inference import load_image, predict as gd_predict

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        image_source, image_tensor = load_image(img_path)
        boxes, logits, _ = gd_predict(
            model=self.gd_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=device,
        )

        if boxes is None or len(boxes) == 0:
            torch.cuda.synchronize()
            latency = (time.time() - start) * 1000
            vram = torch.cuda.max_memory_allocated() / 1024 ** 2
            return None, latency, vram

        H, W, _ = image_source.shape
        boxes_xyxy = boxes * torch.tensor([W, H, W, H], dtype=torch.float32).to(boxes.device)
        cx, cy, bw, bh = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        boxes_xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)

        best_box = boxes_xyxy[logits.argmax()].unsqueeze(0).cpu().numpy().tolist()
        results = self.sam_wrapper(img_path, bboxes=best_box)

        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000
        vram = torch.cuda.max_memory_allocated() / 1024 ** 2
        return results, latency, vram
    