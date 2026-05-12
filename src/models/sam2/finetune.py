import numpy as np
import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_sam2(train_dataset, weights_path, config_path, output_weights, epochs=50, batch_size=4, lr=1e-4):
    """Hace fine tuning de SAM2 congelando el image encoder y el prompt encoder
    y entrenando solo el mask decoder. A diferencia de SAM, SAM2 obtiene los
    embeddings de imagen a través del SAM2ImagePredictor (que internamente
    extrae image_embed y high_res_feats). El config_path apunta al .yaml de
    arquitectura del modelo. Cada muestra del dataset debe devolver
    (imagen, máscara, punto, label). Guarda los pesos en output_weights."""
    sam2 = build_sam2(config_path, weights_path)
    sam2.to(device)

    for param in sam2.image_encoder.parameters():
        param.requires_grad = False
    for param in sam2.sam_prompt_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(sam2.sam_mask_decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    sam2.train()
    predictor = SAM2ImagePredictor(sam2)

    for epoch in range(epochs):
        total_loss = 0
        for images, masks, points, labels in train_loader:
            masks = masks.to(device)
            points = points.to(device)
            labels = labels.to(device)

            loss_batch = 0
            for i in range(images.shape[0]):
                image_np = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                with torch.no_grad():
                    predictor.set_image(image_np)

                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam2.sam_prompt_encoder(
                        points=(points[i].unsqueeze(0), labels[i].unsqueeze(0)),
                        boxes=None,
                        masks=None,
                    )

                image_embedding = predictor._features["image_embed"]
                image_pe = sam2.sam_prompt_encoder.get_dense_pe()
                high_res_features = predictor._features["high_res_feats"]

                low_res_masks, _, _, _ = sam2.sam_mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )

                loss_batch += loss_fn(low_res_masks, masks[i].unsqueeze(0))

            loss = loss_batch / images.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

    torch.save({"model": sam2.state_dict()}, output_weights)
    return output_weights
