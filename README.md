# Prompt Agnostic Essay Scorer (PAES)

This repository contains the implementation of the paper [Prompt Agnostic Essay Scorer: A Domain Generalization Approach to Cross-prompt Automated Essay Scoring](https://arxiv.org/abs/2008.01441v1) using pytorch. 
## Overview

The Prompt Agnostic Essay Scorer (PAES) is a neural network-based approach designed to score essays accurately across different prompts without needing prompt-specific training data. It combines syntactic features and non prompt-specific features for automated essay scoring (AES).

## Dataset

- 24000 student-written argumentative essays. Each essay is scored on a scale of 1 to 6
- Provided by Kaggle's [Learning Agency Lab - Automated Essay Scoring 2.0 competition](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data)


## Key Features

- **AES:** Scores essays from new prompts using models trained on essays from other prompts.
- **Syntactic Feature:** Uses part-of-speech (POS) embeddings to represent essay text syntactically.
- **Non Prompt-specific Features:** Extracts various linguistic features including length, readability, text complexity, text variation, and sentiment.
- **Model Architecture:** Combines convolutional neural networks (CNN) and long short-term memory (LSTM) layers with attention mechanisms.

## Technical Details

### Model Architecture

The PAES model architecture consists of:

1. **POS Embeddings:** Essays are tokenized and tagged with part-of-speech tags. These tags are then mapped to dense vectors using an embedding matrix.
    - `POS Embedding Dimension: 50`

2. **Convolutional Layer:** Applies a 1D convolutional layer on top of the POS embeddings, followed by a dropout layer.
    - `Number of Filters: 100`
    - `Filter Length: 5`

3. **Attention Pooling Layer:** Captures sentence-level representations by applying attention pooling on the output of the convolutional layer.

4. **Recurrent Layer (LSTM):** Uses LSTMs to obtain a sequential representation of the essay, followed by another attention pooling layer to capture the overall essay representation, followed by a dropout layer.
    - `LSTM Output Dimension: 100`

5. **Non Prompt-specific Features:** Extracts various features that are not specific to any prompt to represent essay quality.
    - **Length-based Features:** e.g., word count, sentence count
    - **Readability Scores:** e.g., Flesch reading ease score, Gunning fog index
    - **Text Complexity:** e.g., number of clauses, parse tree depth
    - **Text Variation:** e.g., unique word count, POS tag counts
    - **Sentiment Analysis:** e.g., proportion of positive, negative, neutral sentences

6. **Concatenation:** Concatenates the essay representation from the recurrent layer with the non prompt-specific feature vector.


7. **Linear Layer:** Two linear layer + relu on the concatenated vector.

8. **Sigmoid Layer:** Finally, a sigmoid activation function to predict the essay score.

### Loss Function

- **Mean Squared Error (MSE):** The objective function used to optimize the model, calculating the average squared difference between the predicted scores and the gold standard scores.

### Optimizer

- **RMSprop:** Used to optimize the model with a learning rate of 0.001.

### Training
- Trained for 100 epochs with batch size 64 on training dataeset 
    ```
    Epoch 100
    Train Loss: 0.27685916087319773
    Valid Loss: 0.38885924436829306
    ```
- Implemented best model saving using validation dataset
    ```
    Epoch 52
    Train Loss: 0.3234565282060254
    Valid Loss: 0.37755104194987904
    saved best model
    ```

### Evaluation Metric

- **Quadratic Weighted Kappa (QWK):** Measures the agreement between the predicted scores and the human-annotated scores, considering the severity of disagreements.



### Performance on validation datset
- 0.76 QWK score out of 1
    ```
    Quadratic Weighted Kappa: 0.7659923925088844
    ```

### Performance on test datset
- 0.76 QWK score out of 1
- Outperformed original paper's qwk score of 0.686
    - possibly due to addition of relevant features (feature enginnering)
    - possibly due to difference in dataset used for training and testing.
