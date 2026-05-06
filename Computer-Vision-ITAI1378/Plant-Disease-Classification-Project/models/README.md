# Model Documentation

## 1. Pretrained Model
This project uses ResNet-50 pretrained on ImageNet.  
Loaded using:
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

## 2. Training Configuration
- Architecture: ResNet-50
- Modified final FC layer for 38 classes
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Image size: 224×224
- Augmentations: RandomResizedCrop, RandomHorizontalFlip
- Normalization: ImageNet mean/std
- Batch size: 32

## 3. Training Output
The training process produced a best model checkpoint (`best_simple_model.pth`), 
but the file was lost because the Colab session was not saved to Google Drive.

## 4. Model Performance
Accuracy (test set): 95.80%
Precision: 95.00%
Recall: 95.00%
F1-Score: 95.00%

## 5. How to Reproduce the Training
1. Mount Google Drive  
2. Load dataset from /data/raw  
3. Run the training notebook  
4. The model will save itself to:  
   /content/drive/MyDrive/CV_PlantDiseaseClassification_Project/best_simple_model.pth
