import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_sam(train_dataset, weights_path, output_weights, vit, epochs=50, batch_size=4, lr=1e-4):
    sam = sam_model_registry[vit](checkpoint=weights_path)
    sam.to(device)

    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    sam.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, masks, points, labels in train_loader:
            images, masks = images.to(device), masks.to(device)
            points, labels = points.to(device), labels.to(device)

            loss_batch = 0
            with torch.no_grad():
                image_embeddings = sam.image_encoder(images)

            for i in range(images.shape[0]):
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=(points[i].unsqueeze(0), labels[i].unsqueeze(0)),
                        boxes=None,
                        masks=None,
                    )

                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                loss_batch += loss_fn(low_res_masks, masks[i].unsqueeze(0))

            loss = loss_batch / images.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

    torch.save(sam.state_dict(), output_weights)
    return output_weights
