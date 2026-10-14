# MNIST Digit Recognition with PyTorch

## Overview
A Convolutional Neural Network (CNN) that classifies grayscale handwritten digits into ten classes (0–9) using MNIST.

## Requirements
- Python 3.9+
- PyTorch
- torchvision
- NumPy
- Matplotlib

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python train.py
```

MNIST is downloaded automatically on the first run. The script selects CUDA when available, Apple MPS when available, or CPU otherwise.

## Preprocessing
`torchvision.datasets.MNIST` loads the images and labels. `transforms.ToTensor()` converts each grayscale image into a tensor of shape `(1, 28, 28)` and scales pixel values from `[0, 255]` to `[0, 1]`. The original 60,000 training examples are split into 54,000 training and 6,000 validation examples; the 10,000 test examples are held out for final evaluation.

## Model architecture

| Layer | Configuration |
|---|---|
| Input | 1 × 28 × 28 |
| Convolution | 32 filters, 3 × 3, ReLU |
| Max pooling | 2 × 2 |
| Convolution | 64 filters, 3 × 3, ReLU |
| Max pooling | 2 × 2 |
| Flatten | 64 × 5 × 5 = 1,600 features |
| Fully connected | 128 neurons, ReLU |
| Dropout | 0.3 |
| Output | 10 logits, one per digit class |

The model is trained with Adam and `nn.CrossEntropyLoss()`. The final layer returns raw logits; `CrossEntropyLoss` internally applies the appropriate log-softmax operation. Softmax is applied only when probabilities are needed for sample predictions.

## Training and evaluation
The default settings are 5 epochs, batch size 128, and learning rate 0.001. Each epoch reports training and validation loss and accuracy. At the end, the model is evaluated on the separate test set.

## Generated outputs

The `outputs/` folder contains:
- `mnist_cnn.pth` — saved model weights
- `results.json` — test loss, test accuracy, and device
- `training_history.json` — metrics for each epoch
- `training_curves.png` — training and validation plots
- `sample_predictions.png` — ten test images labelled with actual and predicted digits
- `sample_predictions.json` — predicted labels and confidence values

## Results
Run `python train.py` to generate the actual metrics for your environment. This architecture commonly reaches around 98–99% test accuracy, but the measured result may vary.

## Notes
- The test set is not used to update model weights.
- Increase `EPOCHS` for additional training if desired.
- The `.pth` file stores the model's state dictionary; recreate `MNISTCNN` before loading it.
