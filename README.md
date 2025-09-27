

# BERT for News Classification

This university project explores the application of **BERT** (Bidirectional Encoder Representations from Transformers) for the multi-class classification of news articles. The goal is to analyze the model's performance in categorizing texts from the well-known **AG News** dataset.

## 📌 Objective

The main objective of the project was to train and evaluate a **BERT-based model** to classify news articles into four distinct categories:
* **World** 🌍
* **Sports** 🏅
* **Business** 💼
* **Sci/Tech** 🔬

We implemented a complete pipeline, from data preparation to model tuning, to test the effectiveness of a pre-trained Transformer architecture on a Natural Language Processing (NLP) task.

## 🗂️ Dataset

The **AG News Dataset** was used for this project. It is a public collection of news articles aggregated from more than 2,000 sources.

* **Total Size**: ~127,600 articles.
* **Structure**: The dataset is perfectly balanced, with 30,000 samples per class in the training set and 1,900 in the test set.

## ⚙️ Development Pipeline

The project was structured following a standard Data Science pipeline.

### 1. ETL (Extract, Transform, Load)
In this phase, the raw data was prepared for processing.
* **Text Cleaning**: Removal of stopwords, special characters, emojis, HTML tags, and links.
* **Normalization**: Conversion of all text to lowercase for uniformity.
* **Sampling**: To optimize training time, a balanced subset of 20,000 articles (5,000 per class) was extracted.

### 2. Preprocessing
The cleaned text was transformed into a BERT-compatible format.
* **Tokenization**: Used the pre-trained `bert-base-uncased` tokenizer from the HuggingFace library.
* **Padding & Truncation**: Token sequences were standardized to a fixed length of 64 tokens to ensure consistent input dimensions.

### 3. Modeling and Training
A pre-trained model was fine-tuned for the classification task.
* **Model**: `BertForSequenceClassification`, a BERT architecture with an added classification layer.
* **Framework**: PyTorch and the `Transformers` library by HuggingFace.
* **Optimizer**: AdamW, a variant of the Adam optimizer particularly suited for Transformer models.
* **Training**: The dataset was split into 90% for training and 10% for testing. A *stratified split* was used for the validation set to maintain class proportions. Two experiments were conducted, training the model for **2 epochs (M2)** and **4 epochs (M4)**.

### 4. Evaluation
The performance of the two models was measured using standard metrics.
* **Metrics**: Accuracy and confusion matrices were used to analyze classification errors.
* **Comparison**: The results of the M2 and M4 models were compared to determine the impact of the number of epochs.

## 📊 Results

The results demonstrate the high effectiveness of BERT for text classification tasks.

| Model | Epochs | Accuracy |
| :--- | :---: | :---: |
| **M2** | 2 | `92.55%` |
| **M4** | 4 | `92.75%` |

The **M4 model** (trained for 4 epochs) achieved a slightly higher accuracy, suggesting that longer training allowed it to learn more detailed representations of the data. However, the minimal difference indicates that the model had already reached near-optimal performance after just 2 epochs, with a lower risk of overfitting.

## 🚀 Technologies Used

* **Language**: Python
* **Core Libraries**:
    * `PyTorch`: Deep learning framework.
    * `HuggingFace Transformers`: For accessing pre-trained models and tokenizers.
    * `scikit-learn`: For evaluation metrics and data splitting.
    * `pandas`: For data manipulation.
    * `seaborn` & `matplotlib`: For results visualization.

## 👥 Authors

This project was created by:
* Walter Di Sabatino
* Agnese Bruglia
* Alessandra D’Anna