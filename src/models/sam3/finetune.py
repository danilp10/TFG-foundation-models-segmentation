import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import SAM
from ultralytics.models.sam import SAM3Predictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_sam3(train_dataset, weights_path, output_weights, epochs=50, batch_size=4, lr=1e-4,
               bb_feat_sizes=((288, 288), (144, 144), (72, 72))):
    sam3_wrapper = SAM(weights_path)
    sam3 = sam3_wrapper.model
    for param in sam3.parameters():
        param.data = param.data.to(device)
    for buffer in sam3.buffers():
        buffer.data = buffer.data.to(device)
    sam3.to(device)

    for param in sam3.image_encoder.parameters():
        param.requires_grad = False
    for param in sam3.sam_prompt_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(sam3.sam_mask_decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    predictor = SAM3Predictor(overrides={"model": weights_path, "task": "segment", "mode": "predict", "verbose": False})
    predictor.setup_model(sam3)
    predictor._bb_feat_sizes = list(bb_feat_sizes)

    sam3.train()

    for epoch in range(epochs):
        total_loss = 0
        for images, masks, points, labels in train_loader:
            masks = masks.to(device)
            points = points.to(device)
            labels = labels.to(device)

            loss_batch = 0
            for i in range(images.shape[0]):

                with torch.no_grad():
                    image_tensor = images[i].unsqueeze(0).to(device)
                    backbone_out = sam3.forward_image(image_tensor)
                    _, vision_feats, _, _ = sam3._prepare_backbone_features(backbone_out)
                    if sam3.directly_add_no_mem_embed:
                        vision_feats[-1] = vision_feats[-1] + sam3.no_mem_embed
                    feats = [
                        feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
                        for feat, feat_size in zip(vision_feats, predictor._bb_feat_sizes)
                    ]
                    features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam3.sam_prompt_encoder(
                        points=(points[i].unsqueeze(0), labels[i].unsqueeze(0)),
                        boxes=None,
                        masks=None,
                    )

                image_embedding = features["image_embed"]
                image_pe = sam3.sam_prompt_encoder.get_dense_pe()
                high_res_features = features["high_res_feats"]

                low_res_masks, _, _, _ = sam3.sam_mask_decoder(
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

    torch.save({"model": sam3.state_dict()}, output_weights)
    return output_weights
