# kaf_act 🚀

PyTorch implementation of a learnable activation function combining base activation and Random Fourier Features (RFF). This package provides a flexible and powerful activation function that can be easily integrated into your neural networks.

## Features ✨
- **Learnable Activation**: Combines base activation with RFF for enhanced expressiveness
- **CUDA Support**: Fully compatible with GPU acceleration
- **Flexible Configuration**: Customizable number of Fourier feature grids, dropout, and base activation
- **Auto Dimension Inference**: Automatically detects input dimension during first forward pass

## Installation 📦
```bash
cd kaf_act
pip install .
```

## Usage 🚀

```python
from kaf_act import RFFActivation
import torch.nn as nn
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(128, 256),
    #RFFActivation(num_grids=16, dropout=0.1, activation_expectation=1.64, use_layernorm=False, base_activation=F.gelu),
    RFFActivation(base_activation=F.silu),
    nn.Linear(256, 10)
)
```
## RFFActivation Parameters
| Parameter               | Default      | Description                          |
|-------------------------|--------------|--------------------------------------|
| `num_grids`             | 9            | Number of RFF grid points           |
| `dropout`               | 0.0          | Dropout probability for RFF features|
| `activation_expectation`| 1.64         | Rff initial variance factor         |
| `use_layernorm`         | False        | Enable LayerNorm before RFF         |
| `base_activation`       | F.gelu       | Base activation function            |

## Requirements 📋
- Python >= 3.7
- PyTorch >= 1.10.0

## License 📄
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing 🤝
Contributions are welcome! Please open an issue or submit a pull request.

## Support 💬
For any questions or issues, please open an issue on GitHub.
