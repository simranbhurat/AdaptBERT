# AdaptBERT: Domain Adaptation for Learning Technological Classes of Defensive Publications

## Overview
AdaptBERT is a transfer learning model designed to predict Cooperative Patent Classification (CPC) classes for research disclosures by leveraging information from patent data, even in the absence of labelled data for research disclosures.

## Model Architecture
- We employ a pre-trained BERT (Bidirectional Encoder Representations from Transformers) base uncased model provided by Hugging Face to encode textual information effectively.
- The BERT model generates contextualized word embeddings, capturing intricate meanings and relationships among words.
- A pooling operation is applied in BERT to obtain a condensed representation of the entire input sequence, using the [CLS] token as the representative token for pooling.
- The pooled output is concatenated with metadata(year of publication) to create a fused representation, which undergoes normalization to standardize the input for subsequent layers.
- The data flows through two hidden layers, allowing the model to learn and represent the underlying patterns and relationships in the patent data.

<img width="1200" alt="Screenshot 2023-07-26 at 19 36 10" src="https://github.com/simranbhurat/Thesis/assets/44201011/a6c7a318-23f6-43d7-a73b-3879b7f2df4d">

- The pre-trained BERT classifier processes the input text and generates a hidden state representation for the research disclosures and Patents.
- We perform a maximum mean discrepancy (MMD) calculation to align the hidden state representations from both the patent and research disclosure data.
- By computing gradients and applying backpropagation, we update the weights of the hidden layers, allowing the model to learn from both labelled source data and unlabeled target data.

Note: We named our model "AdaptBERT" to reflect its primary purpose and capability, which is domain adaptation using the BERT architecture.
