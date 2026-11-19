def freeze_model(model, name):
    """Properly freeze the 2D model"""
    for param in model.parameters():
        param.requires_grad = False

    # Set to evaluation mode to disable dropout/batch norm updates
    model.eval()

    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{name} - Frozen parameters: {frozen_params:,}")
    print(f"{name} - Trainable parameters: {trainable_params:,}")