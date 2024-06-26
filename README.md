# Image-Text-Match-Deep-Learning-Models

This is the Task 1 of the Advanced Machine Learning Module Assesment Assignment. This code is tested on modified flickr8k data set. There are 4 different classifier models for the task. 

## Model 1 (ResNet50 and CNN)
Model 1 utilizes a pretrained ResNet50 model for vision encoding and a custom CNN structure for text encoding. The model achieves a test accuracy of 86.30%.

## Model 2 (ITM Updated Baseline)
This model is a modified version of the baseline ITM code provided in the assignment. It achieves a test accuracy of 78.73%.

## Model 3 (ResNet50 and BERT Language Model)
Model 3 employs a pretrained ResNet50 model for vision encoding and Google's pretrained BERT language model for text encoding. It demonstrates a test accuracy of 80.28%.

## Model 4 (ResNet50V2 and GPT2 Language Model)
For Model 4, a pretrained ResNet50V2 model is utilized for vision encoding, while OpenAI's pretrained GPT2 language model serves as the text encoder. This model achieves a test accuracy of 76.74%.

# Model Result Table

| Model                       | Training Time (seconds) | Test Accuracy   | TensorFlow Loss | TensorFlow Accuracy   |
|----------------------------|--------------------------:|:----------------|------------------:|:----------------------|
| ResNet50 & CNN Text Encoder |                   2772.3  | 86.30%          |            0.4495 | 84.75%                |
| ITM-Updated Baseline        |                   1009.26 | 78.73%          |            0.9181 | 78.73%                |
| ResNet50 & Finetuned BERT   |                   5383.6  | 80.28%          |            0.9971 | 79.67%                |
| ResNet50V2 & Finetuned GPT2 |                   4534.32 | 76.74%          |            1.0106 | 75.71%                |

# Performance Metrics

| Model | Balanced Accuracy | AUC | Recall | Precision | F1-score |
|---|---|---|---|---|---|
| ResNet50 & CNN Text Encoder | 86.07% | 92.68% | 85.18% | 95.28% | 89.95% |
| ITM-Updated Baseline | 72.82% | 82.18% | 84.38% | 87.06% | 85.70% |
| ResNet50 & Finetuned BERT | 78.03% | 86.87% | 81.41% | 90.84% | 85.87% |
| ResNet50V2 & Finetuned GPT2 | 74.46% | 83.16%  | 76.74% | 89.49% | 82.63% | 
