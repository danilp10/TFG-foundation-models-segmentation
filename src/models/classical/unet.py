import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_unet(train_dataset, output_weights, epochs=50, batch_size=4, lr=1e-4,
               encoder_name="resnet34", encoder_weights="imagenet"):
    """Entrena un modelo UNet con el encoder indicado sobre el dataset dado.
    Usa Adam, scheduler StepLR (gamma 0.5 cada 20 épocas) y BCEWithLogitsLoss.
    Guarda los pesos finales en output_weights y devuelve esa misma ruta."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), output_weights)
    return output_weights
