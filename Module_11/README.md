# DeepFER: Facial Emotion Recognition Using Deep Learning

**Project Type:** Classification  
**Contribution:** Individual  
**Team Member:** Karthik Doguparthi  
**GitHub Link:** [Module_11](https://github.com/karthikdoguparthi/AlmaBetter/tree/main/Module_11)
**Dataset:** [Face Emotion Recognition Dataset](https://drive.google.com/file/d/1WxFwPgUTPHAIgXHVuPntH9pXQGiKda6F/view)

## Project Summary

This project is a deep learning system designed for recognizing human emotions from facial images. The system classifies faces into one of seven distinct emotion categories: **angry, disgust, fear, happy, neutral, sad, and surprise** utilizing Convolutional Neural Networks (CNNs) and Transfer Learning.

### Dataset
The dataset comprises 28,821 training images and 6,066 validation images, organized into class-wise subdirectories. The classes exhibit imbalance, with the 'happy' class having the most samples (7,164 training images) and the 'disgust' class having the fewest (436 training images).

### Approach
Three different models were developed and compared to determine the best performance:
1. **Custom CNN:** A baseline model consisting of 4 convolutional blocks with batch normalization and dropout, built from scratch.
2. **VGG16 with Transfer Learning:** Leveraging ImageNet weights, the top layers were replaced, and the last two convolutional blocks were fine-tuned.
3. **EfficientNetB0 with Transfer Learning:** A more parameter-efficient architecture that achieved the best overall performance.

### Preprocessing & Augmentation
- Images were resized to 48x48 pixels (grayscale) for the Custom CNN and 96x96 pixels (RGB) for the transfer learning models.
- Data augmentation techniques (rotation ±20°, horizontal flip, zoom, brightness shift) were applied to the training generator to improve model generalization.
- Class weights were computed and applied to handle the dataset's class imbalance.

### Results
The models achieved the following validation accuracies:
- **Custom CNN:** 62%
- **VGG16:** 65%
- **EfficientNetB0:** 67%

The final best-performing model (EfficientNetB0) is saved as `deepfer_best_model.keras`.

### Applications
The developed system can be applied in various domains, including:
- Real-time Human-Computer Interaction (HCI) systems
- Mental health monitoring
- Customer sentiment analysis
- Educational engagement tracking

## Problem Statement

The goal is to accurately classify a given facial image into one of seven discrete emotion categories in real-time, effectively handling challenges such as:
- Class imbalance (e.g., 'disgust' has significantly fewer samples than 'happy')
- Intra-class variation (different intensities of the same emotion)
- Inter-class confusion (e.g., 'fear' vs. 'surprise' often share similar facial features like raised eyebrows)
- Variable lighting, occlusions, and varied poses

Traditional handcrafted approaches often fail to generalize well on these tasks. This project overcomes these limitations by leveraging the power of CNNs and Transfer Learning to learn robust, hierarchical features directly from the image pixels.
