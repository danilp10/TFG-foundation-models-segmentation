from src.models.sam2.finetune import train_sam2 as _train_sam2


def train_sam2_1(train_dataset, weights_path, config_path, output_weights, epochs=50, batch_size=4, lr=1e-4):
    return _train_sam2(
        train_dataset=train_dataset,
        weights_path=weights_path,
        config_path=config_path,
        output_weights=output_weights,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
