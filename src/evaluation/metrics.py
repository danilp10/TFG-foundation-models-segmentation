from scipy.ndimage import binary_erosion
import cv2
from scipy.spatial.distance import directed_hausdorff
import numpy as np

def get_boundary(mask, width=2):
    """Extrae el contorno de una máscara binaria mediante erosión morfológica."""
    eroded = binary_erosion(mask, iterations=width)
    return mask & ~eroded

def boundary_iou(pred_mask, gt_mask, width=2):
    """Variante del IoU clásico que opera sobre los contornos de ambas máscaras,
    siendo más sensible a la precisión de los bordes que al solapamiento global."""
    pred_boundary = get_boundary(pred_mask, width)
    gt_boundary   = get_boundary(gt_mask, width)
    intersection  = np.logical_and(pred_boundary, gt_boundary).sum()
    union         = np.logical_or(pred_boundary, gt_boundary).sum()
    return intersection / union if union > 0 else 0

def resize_for_hausdorff(pred, gt, max_size=512):
    """Redimensiona las máscaras proporcionalmente si superan max_size para
    reducir el coste computacional del cálculo de la distancia de Hausdorff."""
    if max(pred.shape) > max_size:
        scale = max_size / max(pred.shape)
        new_h = int(pred.shape[0] * scale)
        new_w = int(pred.shape[1] * scale)
        pred = cv2.resize(pred.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        gt   = cv2.resize(gt.astype(np.uint8),   (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return pred, gt

def hausdorff_95(pred_mask, gt_mask):
    """Calcula el percentil 95 de la distancia de Hausdorff entre dos máscaras,
    descartando el 5% de valores extremos para mayor robustez ante outliers."""
    pred_points = np.argwhere(pred_mask)
    gt_points   = np.argwhere(gt_mask)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0
    d1 = directed_hausdorff(pred_points, gt_points)[0]
    d2 = directed_hausdorff(gt_points, pred_points)[0]
    return np.percentile([d1, d2], 95)

def compute_all_metrics(pred_mask, gt_mask):
    """Calcula el conjunto completo de métricas de segmentación binaria
    (IoU, precisión, recall, F1, Dice, especificidad, F2 y pixel accuracy)
    a partir de la matriz de confusión."""
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    tn = np.logical_and(~pred_mask, ~gt_mask).sum()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f2 = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    pixel_accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    return iou, precision, recall, f1, dice, specificity, f2, pixel_accuracy
