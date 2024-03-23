# Sequence to Sequence Learning with Attention Mechanism for Language Translation

This project focuses on implementing a neural machine translation model using PyTorch, with a specific emphasis on the attention mechanism.

## General Information

This repository provides a detailed step-by-step guide to implementing a sequence-to-sequence model for language translation tasks. The model architecture is designed to capture dependencies between sequences and learn similarities using context vectors, facilitated by the attention mechanism.

### Training Environment  

The training for this model was conducted on High-Performance Computing (HPC) environments to ensure efficient computation and resource utilization.

### Dataset

The training dataset utilized in this project is an English to French translation dataset. Despite encountering overfitting issues during training, the primary objective remains to showcase the attention mechanism's effectiveness in capturing sequence dependencies.

## Model Overview

The sequence-to-sequence model architecture employed in this project consists of an encoder-decoder framework with an attention mechanism. The encoder processes the input sequence, while the decoder generates the output sequence. The attention mechanism allows the model to focus on relevant parts of the input sequence at each decoding step, thereby enhancing translation performance.

## Usage

The repository includes pre-trained models (encoder and decoder) that can be readily applied for English to French translation tasks. Moreover, the framework is highly adaptable and can be extended to accommodate other datasets, languages, and tasks. Users can experiment with increasing model complexity by adding additional encoder and decoder layers or incorporating advanced techniques.

### Note

While this project primarily focuses on English to French translation, the underlying principles and architecture can be generalized to a wide range of language translation tasks.

## Dataset Source
The dataset used in this project is available [here](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset).

## Reference 
https://medium.com/@eugenesh4work/attention-mechanism-for-lstm-used-in-a-sequence-to-sequence-task-be1d54919876
https://www.kaggle.com/code/asemsaber/english2french-nmt-tf-seq2seq-attention

## Requirements

- Python (>=3.6)
- PyTorch
  
