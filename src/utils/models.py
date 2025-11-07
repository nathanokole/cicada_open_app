import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
import numpy as np
import cv2

class ModifiedInception(nn.Module):
    def __init__(self, model_name='inceptionv3', backbone_out=32768, num_classes=13, pretrained_path=None):
        super().__init__()
        self.backbone = pretrainedmodels.__dict__[model_name](pretrained=None)
        self.backbone.last_linear = nn.Linear(backbone_out, num_classes)
        if pretrained_path:
            self.load_weights(pretrained_path)
    
    def new_logits(self, features):
        x = F.avg_pool2d(features, kernel_size=9, stride = 8)# output_size=(2,2))
        # x = F.adaptive_avg_pool2d(features, output_size=(2,2))
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        return self.backbone.last_linear(x)
    
    def load_weights(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)       # unwrap if needed
        self.backbone.load_state_dict(state_dict, strict=False)
        
    def forward(self, x):
        features = self.backbone.features(x)
        return self.new_logits(features)


def get_predictions(model, images, labels, batch_size=8, device='cpu'):
    labels_out, idx_out, probs_out = [], [], []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([
            torch.from_numpy(img).permute(2,0,1) for img in images[i:i+batch_size]
        ]).to(device)
        with torch.no_grad():
            probs = F.softmax(model(batch), dim=1).cpu().numpy()   # (N, C)
        idx = probs.argmax(1)

        labels_out += [labels[j] for j in idx]
        idx_out    += idx.tolist()
        probs_out  += [
            {lab: round(p, 3) for lab, p in zip(labels, row)}
            for row in probs
        ]
    return labels_out, idx_out, probs_out


def _to_nchw_tensor(imgs: list[np.ndarray]) -> torch.Tensor:
    """Convierte lista de HWC uint8/float a tensor NCHW float32 [0,1]."""
    arr = np.stack([
        (im if im.flags['C_CONTIGUOUS'] else np.ascontiguousarray(im))
        for im in imgs
    ], axis=0)  # (N,H,W,C)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (N,H,W,3), got {arr.shape}")
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # NCHW
    if t.dtype != torch.float32:
        t = t.float()
    # if t.max() > 1.5:  # normaliza a [0,1] si vienen en 0..255
    #     t = t / 255.0
    return t.contiguous()


def get_prediction_TTA(
    model,
    images: list[np.ndarray],
    labels,
    batch_size: int = 32,
    device: str = "cpu",
    empty_cache: bool = False,
):
    """
    TTA por lote: original, flip horizontal, rot90. Devuelve la predicción promedio.
    Optimizada para una sola GPU con menor overhead de memoria.
    """
    model.eval()

    # Usa el device real del modelo para evitar mismatches
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device if torch.cuda.is_available() else "cpu")
    dev = model_device
    use_cuda = (dev.type == "cuda")

    N = len(images)
    all_idx = []
    all_probs = []

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_cuda):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_np = images[start:end]  # lista de np.ndarrays

            x = _to_nchw_tensor(batch_np)
            if use_cuda:
                x = x.pin_memory().to(dev, non_blocking=True)
            else:
                x = x.to(dev)

            # Augmentaciones tensoriales
            x0 = x
            x1 = torch.flip(x, dims=[3])            # flip horizontal
            x2 = torch.rot90(x, k=1, dims=(2, 3))   # rot90

            # Combinar por LOGITS (más estable que promediar softmax)
            logits_sum = 0
            for xb in (x0, x1, x2):
                logits = model(xb)                      # (B,C)
                # defensa ante NaN/Inf en modelos raros
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.zeros_like(logits)
                logits_sum = logits_sum + logits

            probs_mean = F.softmax(logits_sum / 3.0, dim=1).float()  # (B,C)

            idx = torch.argmax(probs_mean, dim=1)
            all_idx.append(idx.detach().cpu())
            all_probs.append(probs_mean.detach().cpu())

            # liberar referencias del batch
            del x, x0, x1, x2, logits_sum, probs_mean, idx
            if empty_cache and use_cuda:
                torch.cuda.empty_cache()

    # concatenar resultados
    idx_cat = torch.cat(all_idx, dim=0).tolist()
    probs_cat = torch.cat(all_probs, dim=0).numpy()

    labels_out = [labels[i] for i in idx_cat]
    probs_out = [
        {lab: round(float(p), 3) for lab, p in zip(labels, row)}
        for row in probs_cat
    ]

    if use_cuda:
        torch.cuda.synchronize()

    return labels_out, idx_cat, probs_out




def exact_boundingbox(roi_img):
    gray = cv2.bitwise_not(cv2.cvtColor(roi_img, cv2.COLOR_RGB2LAB))[..., 0]
    gray = cv2.dilate(cv2.convertScaleAbs(gray, alpha=1.0, beta=-50), kernel=np.ones((3,3)))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    area_thresh = 1000
    filtered = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_thresh:
            filtered[labels == i] = 255

    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel=np.ones((5,5)), iterations=1)
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(filtered)
    for i, cnt in enumerate(contours, start=1):
        cv2.drawContours(filled_mask, [cnt], -1, i, thickness=cv2.FILLED)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered, connectivity=8)
    stats = stats[1:]
    centroids = centroids[1:]
    
    MIN_SIDE_LENGTH = 224
    new_boxes = []
    scale_factor = 1.1
    for s, c in zip(stats, centroids):
        x, y, w, h = s[:4]
        cx, cy = c
        side = max(w, h) * scale_factor
        side = side if side >= MIN_SIDE_LENGTH else MIN_SIDE_LENGTH
        new_x = int(cx - side / 2)
        new_y = int(cy - side / 2)
        new_boxes.append((new_x, new_y, int(side), int(side)))
    
    return filled_mask, new_boxes
