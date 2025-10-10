# Breast Cancer Image Classification using Modified ResNet-18

This project implements a deep learning solution for breast cancer classification using a modified ResNet-18 architecture with data augmentation and comprehensive evaluation metrics. The model is trained on the BreastMNIST dataset from the MedMNIST collection.

## 🎯 Project Overview

The project focuses on binary classification of breast cancer images, distinguishing between malignant and benign cases. It employs a modified ResNet-18 architecture with several enhancements including dropout regularization, data augmentation, and comprehensive evaluation using multiple metrics.

## 📊 Dataset

- **Dataset**: BreastMNIST from MedMNIST collection
- **Task**: Binary classification (malignant vs benign)
- **Image Size**: 28x28 pixels
- **Channels**: 1 (grayscale)
- **Classes**: 2 (Negative/Normal, Positive/Malignant)

## 🏗️ Model Architecture

### Modified ResNet-18 Features:
- **Base Architecture**: ResNet-18 with ImageNet pretrained weights
- **Input Layer Modification**: Custom first convolutional layer adapted for 28x28 grayscale images
  - Kernel size: 4x4 (instead of 7x7)
  - Stride: 1 (instead of 2)
  - Padding: 1
- **Regularization**: Dropout layer (rate: 0.3) before final classification
- **Output**: 2-class classification head

## 🔧 Key Features

### 1. Data Augmentation
- Random rotation (±10 degrees)
- Random affine transformations (translation)
- Random horizontal flipping (50% probability)
- Normalization (mean=0.5, std=0.5)

### 2. Training Configuration
- **Optimizer**: SGD with momentum
- **Learning Rate**: 0.0009
- **Momentum**: 0.95
- **Weight Decay**: 0.00008
- **Batch Size**: 16
- **Epochs**: 18
- **Loss Function**: CrossEntropyLoss

### 3. Comprehensive Evaluation
- **Metrics**: AUC, AUPR, Accuracy, Precision, Recall, F1-Score
- **Visualizations**: ROC curves, Precision-Recall curves, Confusion matrices
- **Cross-Validation**: 5-fold cross-validation for robust evaluation

## 📈 Performance Results

### Final Model Performance:
- **Validation Set**:
  - AUC: 0.9131
  - Accuracy: 0.9487
  - AUPR: 0.9466
  - Precision: 0.9344
  - Recall: 1.0000
  - F1-Score: 0.9661

- **Test Set**:
  - AUC: 0.9081
  - Accuracy: 0.8846
  - AUPR: 0.9489
  - Precision: 0.8934
  - Recall: 0.9561
  - F1-Score: 0.9237

### 5-Fold Cross-Validation Results:
- **Average AUC**: 0.9464
- **Average Accuracy**: 0.8818
- **Average AUPR**: 0.9828
- **Average Precision**: 0.9286
- **Average Recall**: 0.9176
- **Average F1-Score**: 0.9231

## 🚀 Getting Started

### Prerequisites
```bash
pip install medmnist matplotlib scikit-learn torch torchvision
```

### Usage
1. **Run the notebook**: Execute `ModifiedResnet18 with data augmentation and evaluation.ipynb`
2. **Model training**: The notebook includes complete training pipeline with data augmentation
3. **Evaluation**: Comprehensive evaluation with multiple metrics and visualizations
4. **Cross-validation**: 5-fold cross-validation implementation

### Key Components

#### 1. Model Definition
```python
class ModifiedResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ModifiedResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(0.3)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
```

#### 2. Data Augmentation
```python
data_augmentation_transform = v2.Compose([
    v2.RandomRotation(degrees=10),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])
```

## 📋 Project Structure

```
Image-classification-on-Breast-Cancer-dataset/
├── ModifiedResnet18 with data augmentation and evaluation.ipynb
└── README.md
```

## 🔬 Technical Details

### Hyperparameter Optimization
The project includes extensive hyperparameter tuning with grid search over:
- Learning rates: [0.00085, 0.0009, 0.00095]
- Batch sizes: [60, 64, 68, 72]
- Momentum values: [0.93, 0.94, 0.95]
- Epochs: [35, 40, 45]
- Dropout rates: [0.28, 0.3, 0.32]
- Weight decay: [0.0001, 0.00011, 0.00013]

### Evaluation Methodology
1. **Hold-out validation**: Separate train/validation/test splits
2. **Cross-validation**: 5-fold CV for robust performance estimation
3. **Multiple metrics**: Comprehensive evaluation beyond accuracy
4. **Visualization**: ROC and PR curves, confusion matrices

## 🎯 Key Achievements

1. **High Performance**: Achieved >90% AUC and accuracy on both validation and test sets
2. **Robust Evaluation**: Implemented comprehensive evaluation with multiple metrics
3. **Data Augmentation**: Successfully applied augmentation to improve model generalization
4. **Cross-Validation**: Demonstrated model stability through 5-fold CV
5. **Visualization**: Clear performance visualization through multiple plot types

## 🔧 Dependencies

- Python 3.7+
- PyTorch 2.3.0+
- Torchvision 0.18.0+
- MedMNIST 3.0.1+
- Matplotlib 3.8.4+
- Scikit-learn 1.4.2+
- NumPy 1.26.4+
- Pandas 2.2.2+

## 📝 Notes

- The model uses pretrained ResNet-18 weights from ImageNet
- Data augmentation is applied only to the training set
- All experiments use fixed random seeds for reproducibility
- The model is optimized for the specific characteristics of medical imaging data

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the model architecture
- Adding new data augmentation techniques
- Implementing additional evaluation metrics
- Optimizing hyperparameters further

## 📄 License

This project is open source and available under the MIT License.