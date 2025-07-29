# Hugging Face Transformers: Leveraging Pre-trained Models for NLP Tasks

This project demonstrates the power and simplicity of using the Hugging Face `transformers` library for various Natural Language Processing (NLP) tasks. It specifically showcases two common use cases:

1.  **Sentiment Analysis with BERT**: (Commented out in the provided code, but included for conceptual understanding) Demonstrates fine-tuning a pre-trained BERT model for sequence classification (e.g., IMDB movie review sentiment).
2.  **Text Generation with GPT-2**: Illustrates how to use a pre-trained GPT-2 model for open-ended text generation.

The Hugging Face `transformers` library provides thousands of pre-trained models to perform tasks on texts such as classification, information extraction, summarization, translation, and text generation.

## Project Overview

The Python script highlights:

* **Loading Pre-trained Models and Tokenizers**: How to easily load popular models like `bert-base-uncased` and `gpt2` along with their corresponding tokenizers.
* **Dataset Preparation**: (For BERT example) How to load datasets from the `datasets` library and prepare them for model input using tokenizers (padding, truncation).
* **Model Fine-tuning**: (For BERT example) The basic setup for fine-tuning a pre-trained model on a specific task using `Trainer`.
* **Text Generation**: How to use a causal language model (like GPT-2) to generate new text based on a given prompt.

## Part 1: Sentiment Analysis with BERT (Commented Out)

This section, though commented out, provides a blueprint for how one would typically fine-tune a pre-trained BERT model for a classification task like sentiment analysis on the IMDB dataset.

### Key Steps:

1.  **Load Dataset**: `load_dataset("imdb")` fetches the IMDB movie reviews.
2.  **Load Tokenizer**: `AutoTokenizer.from_pretrained("bert-base-uncased")` loads the BERT tokenizer. This tokenizer handles converting raw text into numerical input IDs, attention masks, and token type IDs that BERT understands.
3.  **Tokenize Dataset**: A `tokenize_function` is applied to the dataset using `dataset.map()`. This function tokenizes the text, applies `padding="max_length"` to ensure all sequences have the same length, and `truncation=True` to cut off longer sequences.
4.  **Prepare Data Format**: Columns irrelevant to training are removed, the label column is renamed to `labels` (as expected by `Trainer`), and the dataset format is set to "torch" for PyTorch compatibility.
5.  **Load Model**: `AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)` loads a BERT model pre-trained on a vast corpus and adapts its final layer for a 2-class (positive/negative) classification task.
6.  **Training Arguments**: `TrainingArguments` defines various training parameters like `output_dir`, `learning_rate`, `batch_size`, `num_train_epochs`, etc.
7.  **Trainer Initialization**: The `Trainer` class orchestrates the fine-tuning process, taking the model, training arguments, datasets, and tokenizer.
8.  **Training and Evaluation**: `trainer.train()` starts the fine-tuning. `trainer.evaluate()` assesses the model's performance on the test set.

**Why Fine-tuning?**
Fine-tuning allows us to leverage the powerful general language understanding capabilities that models like BERT have learned from massive text corpora. By training the model for a few epochs on a smaller, task-specific dataset (like IMDB reviews), it can adapt its learned representations to excel at that specific task (sentiment analysis) with far less data and training time than training from scratch.

## Part 2: Text Generation with GPT-2

This section of the code focuses on using a pre-trained GPT-2 model for generative tasks, demonstrating its ability to complete a given text prompt.

### Key Steps:

1.  **Load GPT Model and Tokenizer**:
    * `AutoModelForCausalLM.from_pretrained("gpt2")` loads the GPT-2 model, which is designed for causal language modeling (predicting the next token).
    * `AutoTokenizer.from_pretrained("gpt2")` loads the corresponding GPT-2 tokenizer.
2.  **Encode Input Text**: `tokenizer.encode(input_text, return_tensors="pt")` converts the input prompt ("Once upon a time") into numerical input IDs, formatted as a PyTorch tensor.
3.  **Create Attention Mask**: `input_ids.ne(tokenizer.pad_token_id)` creates an attention mask. This mask tells the model which tokens are actual input and which are padding (if any), ensuring that the model doesn't attend to padding tokens. For GPT-2, which is typically not trained with padding, `tokenizer.eos_token_id` is often used as `pad_token_id` during generation if `pad_token_id` is not explicitly set or available.
4.  **Generate Text**:
    * `gpt_model.generate(...)` is the core function for text generation.
    * `input_ids`: The starting prompt.
    * `attention_mask`: Specifies actual input tokens.
    * `max_length=50`: The maximum length of the generated sequence (including the prompt).
    * `num_return_sequences=1`: Generates one sequence.
    * `pad_token_id=pad_token_id`: Handles padding during generation. GPT-2, by default, doesn't have a dedicated padding token, so `eos_token_id` is often used to ensure the generation process terminates correctly when padding is involved internally. The code handles cases where `eos_token_id` might be `None`.
5.  **Decode Output**: `tokenizer.decode(output[0], skip_special_tokens=True)` converts the generated numerical output IDs back into human-readable text, skipping any special tokens like `[CLS]`, `[SEP]`, `<PAD>`.

**How GPT-2 Generates Text:**
GPT-2 is a decoder-only Transformer model trained to predict the next word in a sequence. During generation, it takes the input prompt, predicts the most likely next word, appends it to the sequence, and then repeats the process until a maximum length is reached or an end-of-sequence token is generated.

## Setup and Usage

### Prerequisites

* Python 3.x
* `transformers` library
* `datasets` library (for BERT example)

You can install these libraries using pip:

```bash
pip install transformers datasets