# Machine Learning Glossary (Sample)

Short definitions used to demonstrate retrieval-augmented generation.

## Overfitting

Overfitting happens when a model learns noise in the training data instead of the underlying pattern. It shows up as high training accuracy but poor accuracy on held-out data. Common remedies are regularization, more data, simpler models, and early stopping.

## Data Leakage

Data leakage occurs when information that would not be available at prediction time is used during training, such as fitting a scaler on the full dataset before splitting. It produces overly optimistic validation scores.

## Embedding

An embedding is a dense vector that represents text, images, or other items so that similar items are close together. Retrieval-augmented generation uses embeddings to find passages related to a question.

## Precision and Recall

Precision is the share of predicted positives that are truly positive. Recall is the share of actual positives the model finds. In retrieval, recall@k measures how often the relevant document appears in the top k results.

## Temperature

Temperature controls randomness in language model sampling. A temperature of 0 makes output nearly deterministic, which suits extraction and classification. Higher values produce more varied text, which suits brainstorming.
