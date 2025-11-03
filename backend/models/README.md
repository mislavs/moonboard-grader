# Model Files

## Adding the Inference Model

Place your trained model file in this directory with the name `model_for_inference.pth`.

### Expected File

- `model_for_inference.pth` - The trained PyTorch model checkpoint

### Model Format

The model file should be a PyTorch checkpoint (.pth file) created using the classifier training pipeline. It must contain:
- `model_state_dict` - The model weights and architecture information
- Standard PyTorch checkpoint format compatible with the Predictor class

### Example

```bash
# Copy your best model to the inference directory
cp ../classifier/models/best_model.pth ./model_for_inference.pth
```

Or on Windows PowerShell:

```powershell
Copy-Item ..\classifier\models\best_model.pth -Destination .\model_for_inference.pth
```

### Note

Without this file, the backend will fail to start. Make sure to add your trained model before running the server.

