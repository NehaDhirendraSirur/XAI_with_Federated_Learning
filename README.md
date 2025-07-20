# XAI with Federated Learning for Plant Disease Detection 🌿🤖🔍

This project explores the integration of Explainable Artificial Intelligence (XAI) and Federated Learning to enhance the transparency, privacy, and accuracy of plant disease detection models.

## Overview

In agriculture, particularly in plant disease detection, the need for interpretable and privacy-preserving AI models is growing rapidly. 
This project aims to:

- Leverage XAI techniques to provide transparent explanations behind model predictions.
- Implement Federated Learning (FedAvg) to address data privacy concerns.
- Improve prediction accuracy using a collaborative yet decentralized learning approach.

## Methodology

- A custom CNN model was developed and trained on a dataset of 1530 plant leaf samples.
- Federated Learning (FedAvg) was used to train models locally across decentralized nodes without sharing raw data.
- XAI methods were applied post-training to visualize and interpret model decisions.

### Tech Stack

- Python
- TensorFlow / Keras and Pytorch
- Federated Learning framework (FedAvg)
- XAI Libraries: LIME, SHAP, or similar
- Matplotlib for visualizations

## Results

- Centralized CNN Accuracy: 91.3%
- Federated Learning Accuracy (FedAvg): 96.6%
- Added Value: Increased model trust through explainability + maintained data privacy

##  Publication

This work is published as part of a Springer book chapter:

📘 [Read the full paper](https://link.springer.com/chapter/10.1007/978-981-96-2179-8_7)
📖 Free Access (Google Books): Read on Google Books

---

## How to Run

Clone this repository:
   git clone https://github.com/NehaDhirendraSirur/XAI_with_Federated_Learning.git
   cd XAI_with_Federated_Learning
