# Hate Speech Detection Across Social Media Platforms

This project analyzes and compares the effectiveness of various machine learning models, from traditional approaches to state-of-the-art transformers, in detecting hate speech across different social media platforms.

## Project Overview

The proliferation of online social media has amplified the need for robust and accurate hate speech detection systems. This research investigates the performance of different classification models on a diverse dataset aggregated from Facebook, Reddit, Twitter, and YouTube. By standardizing and cleaning data from these varied sources, this study provides a comprehensive benchmark of model performance, highlighting the strengths of modern transformer architectures like BERT, RoBERTa, and ELECTRA against traditional methods such as TF-IDF and Word2Vec.

The central research question is: **What, if any, is the difference between textual hate-speech detection across various social media platforms?**

* **[Final Project Poster](./AAIB%20G10%20Report.pdf)**: A high-resolution PDF of the project summary poster.


## Abstract
The rise of social media platforms has led to an increased need for effective hate speech detection to maintain a safe online environment. This study explores the performance of various machine learning models, comparing traditional approaches with state-of-the-art transformer-based models in the context of hate speech detection across different social media platforms. Using datasets from Reddit, Twitter, Facebook, and YouTube, we standardized and cleaned the data to ensure consistency and robustness. We evaluated traditional models including TF-IDF, Word2Vec trained from scratch, and Word2Vec pre-trained on Google News, as well as advanced transformer models like BERT, RoBERTa, ELECTRA, and GPT-3.5. Our findings reveal that pre-trained transformer models, particularly ELECTRA, outperform traditional methods, indicating the importance of advanced context-aware techniques in hate speech detection. This research underscores the necessity of tailored moderation strategies that can adapt to the unique linguistic styles of different platforms.

## Full Report 

### Key Objectives

  * To understand the nuances and challenges of hate speech detection.
  * To evaluate if combining datasets from multiple platforms can improve model generalization and accuracy.
  * To develop an effective and generalizable pipeline for hate speech detection.

## Datasets

The study utilizes text data from four major social media platforms:

  * Facebook
  * Reddit
  * Twitter
  * YouTube

The raw data is located in the `/data/source` directory. It undergoes a rigorous cleaning and balancing process, with the final prepared datasets available in `/data/cleaned` and `/data/balanced`.

## Project Structure

```
.
├── data/
│   ├── source/         # Original, raw datasets from various platforms
│   ├── intermediary/   # Datasets after initial processing steps
│   ├── cleaned/        # Fully cleaned and standardized datasets
│   └── balanced/       # Datasets balanced for model training
├── dpt/                # Jupyter notebooks and scripts for data processing
├── ml/
│   ├── gpt/            # Scripts related to GPT model experiments
│   ├── results/        # Model performance metrics, plots, and CSVs
│   ├── Bert_modelandeval.ipynb # Notebook for BERT, RoBERTa, ELECTRA
│   ├── tf-idf.ipynb    # Notebook for TF-IDF model
│   └── w2v.ipynb       # Notebook for Word2Vec models
├── AAIB G10 Report.pdf # The final research paper
├── AAIB G10 Presentation.pptx # The project presentation slides
└── requirements.txt    # Required Python libraries for the project
```

## Models Evaluated

A wide range of models were trained and evaluated to provide a thorough comparison.

#### 1\. Traditional Machine Learning Models

  * **TF-IDF**: A baseline model using Term Frequency-Inverse Document Frequency features with a Random Forest classifier.
  * **Word2Vec (Custom Trained)**: Word embeddings trained from scratch on the project's combined dataset, used with a Random Forest classifier.
  * **Word2Vec (Pre-trained)**: Utilizes the Google News pre-trained Word2Vec model with a Random Forest classifier.

#### 2\. Transformer-based Models

  * **BERT**: Bidirectional Encoder Representations from Transformers.
  * **RoBERTa**: A Robustly Optimized BERT Pretraining Approach.
  * **ELECTRA**: Pre-training Text Encoders as Discriminators Rather Than Generators.
  * **GPT-3.5**: Evaluated using zero-shot and few-shot prompting techniques for classification.

## Key Findings

  * **Transformers Outperform Traditional Methods**: Pre-trained transformer models, particularly **ELECTRA**, consistently achieved the highest accuracy and F1-scores, demonstrating the power of contextual embeddings for this task.
  * **Combined Datasets Yield Better Generalization**: Models trained on a combined dataset from all four platforms generally performed better on individual platform tests, suggesting that exposure to diverse linguistic styles improves robustness.
  * **Platform-Specific Difficulty**: The models found it most challenging to detect hate speech on **Reddit**. This is attributed to the longer text format, use of niche slang, and complex, community-specific contexts.
  * **Pre-trained Models vs. Zero-Shot LLMs**: Finely-tuned, pre-trained models like ELECTRA outperformed the newer, larger GPT-3.5 model when it was used in a zero-shot or few-shot setting. This indicates that for specialized tasks, a well-tuned smaller model can be more effective than a generic large language model without specific fine-tuning.

## How to Run

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://your-repository-url/hate-speech-classification.git
    cd hate-speech-classification
    ```

2.  **Set up a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Data Processing**: The `dpt/dpt.ipynb` notebook contains the code for cleaning, standardizing, and balancing the datasets.
2.  **Model Training**: The Jupyter notebooks in the `ml/` directory are used to train and evaluate the different models.
      * For **BERT, RoBERTa, and ELECTRA**, use `ml/Bert_modelandeval.ipynb`.
      * For **TF-IDF and Word2Vec**, use the respective notebooks (`tf-idf.ipynb`, `w2v.ipynb`, etc.).
3.  **View Results**: All evaluation metrics, confusion matrices, and performance plots are generated and saved in the `ml/results/` directory.
