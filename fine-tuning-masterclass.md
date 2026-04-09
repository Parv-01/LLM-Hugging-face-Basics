# 🤗 The Complete Engineer's Guide to HuggingFace Transformers
### From Zero to Production-Grade Fine-Tuning — Offline & Google Colab

> **Author's Note:** This document is written from the perspective of 14+ years in AI/ML engineering. Every section reflects real-world lessons from production deployments, failed experiments, and hard-won debugging sessions. This is not a "getting started" fluff guide — this is a field manual.

---

## 📋 Table of Contents

1. [Mental Model — How Transformers & HuggingFace Actually Work](#1-mental-model)
2. [Environment Setup — Local, Offline & Google Colab](#2-environment-setup)
3. [The HuggingFace Ecosystem — What Lives Where](#3-the-huggingface-ecosystem)
4. [Tokenizers — The Most Underestimated Component](#4-tokenizers)
5. [Loading & Running Pretrained Models](#5-loading--running-pretrained-models)
6. [The `pipeline` API — Fast Prototyping](#6-the-pipeline-api)
7. [Datasets — Loading, Processing & Custom Data](#7-datasets)
8. [Fine-Tuning Strategy — Choosing Your Approach](#8-fine-tuning-strategy)
9. [Full Fine-Tuning with the Trainer API](#9-full-fine-tuning-with-trainer)
10. [Parameter-Efficient Fine-Tuning (PEFT / LoRA / QLoRA)](#10-peft--lora--qlora)
11. [Fine-Tuning for Specific Tasks](#11-fine-tuning-for-specific-tasks)
12. [Evaluation & Metrics](#12-evaluation--metrics)
13. [Model Saving, Export & Offline Deployment](#13-model-saving-export--offline-deployment)
14. [Google Colab — The Right Way](#14-google-colab--the-right-way)
15. [Memory Optimization & Quantization](#15-memory-optimization--quantization)
16. [Troubleshooting & Common Pitfalls](#16-troubleshooting--common-pitfalls)
17. [Reference Architectures & Quick Templates](#17-reference-architectures--quick-templates)

---

## 1. Mental Model

Before writing a single line of code, you must have the correct mental model. Most engineers get confused because they treat HuggingFace as a black box.

### 1.1 The Transformer Architecture in One Paragraph

A Transformer model is a neural network built on the **self-attention mechanism**. Unlike RNNs that process tokens sequentially, Transformers process all tokens in parallel, learning contextual relationships between them. Every modern LLM (BERT, GPT, T5, LLaMA, Mistral) is a variant of this architecture.

```
Raw Text → Tokenizer → Token IDs → Embeddings → [Attention Layers × N] → Logits → Output
```

The key insight: **the model never sees text**. It sees integer token IDs, which are then converted to floating-point vectors. This is why tokenizer choice is critical.

### 1.2 The Three Types of Transformer Models You'll Work With

| Type | Architecture | Pretrained On | Best For | Examples |
|------|-------------|---------------|----------|---------|
| **Encoder-only** | Bidirectional attention | Masked Language Modeling (MLM) | Classification, NER, embeddings | BERT, RoBERTa, DeBERTa |
| **Decoder-only** | Causal (left-to-right) attention | Next Token Prediction | Text generation, chat, completion | GPT-2, LLaMA, Mistral, Falcon |
| **Encoder-Decoder** | Both | Seq2Seq objectives | Translation, summarization, Q&A | T5, BART, mT5 |

> ⚠️ **Critical Rule:** If you pick the wrong architecture for your task, fine-tuning will fail silently or produce terrible results. Match the architecture to the task FIRST.

### 1.3 What HuggingFace Actually Is

HuggingFace is **three things bundled together**, and conflating them causes confusion:

1. **`transformers` library** — The Python SDK for loading/running/fine-tuning models
2. **Hub** — A model/dataset/space registry (like GitHub for ML models)
3. **`datasets` library** — A separate library for loading and processing datasets efficiently

They are **independently installable** and **independently usable**. You do not need Hub access to use the `transformers` library offline.

---

## 2. Environment Setup

### 2.1 Local Machine Setup (Full Offline Capability)

#### Step 1: Create an Isolated Environment

```bash
# Using conda (strongly recommended for GPU-heavy work)
conda create -n transformers-env python=3.11 -y
conda activate transformers-env

# OR using venv
python -m venv transformers-env
source transformers-env/bin/activate  # Linux/macOS
# transformers-env\Scripts\activate   # Windows
```

#### Step 2: Install PyTorch First (Always Before Transformers)

```bash
# Check your CUDA version first
nvidia-smi  # Look for "CUDA Version: XX.X"

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### Step 3: Install the HuggingFace Stack

```bash
# Core libraries
pip install transformers datasets tokenizers

# For fine-tuning (Trainer API + optimizations)
pip install accelerate evaluate peft trl

# For quantization (bitsandbytes for QLoRA)
pip install bitsandbytes  # Linux/CUDA only
# pip install bitsandbytes-windows  # Windows

# For faster tokenizers and optimized inference
pip install sentencepiece protobuf

# Optional but recommended
pip install wandb      # Experiment tracking
pip install scipy      # Required by some models
pip install einops     # Required by some architectures (e.g., Falcon)
```

#### Step 4: Verify Your Installation

```python
import transformers
import torch
import datasets
import peft

print(f"Transformers: {transformers.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 2.2 Configuring Offline Mode

This is critical for air-gapped environments, servers without internet, or when you've already downloaded models and want reproducible behavior.

```python
import os

# Method 1: Environment variables (set before importing transformers)
os.environ["TRANSFORMERS_OFFLINE"] = "1"          # Block all Hub requests
os.environ["HF_DATASETS_OFFLINE"] = "1"           # Block dataset Hub requests
os.environ["HF_HUB_OFFLINE"] = "1"                # Block huggingface_hub requests

# Method 2: Set cache directory to a custom location (for portable setups)
os.environ["HF_HOME"] = "/path/to/your/models"    # Master cache location
os.environ["TRANSFORMERS_CACHE"] = "/path/to/your/models/hub"
os.environ["HF_DATASETS_CACHE"] = "/path/to/your/datasets"

# For a shared server, set this in your .bashrc or .zshrc
# export HF_HOME=/mnt/shared/hf_cache
```

#### Pre-Downloading Models for Offline Use

```python
# Download while online — then switch to offline mode
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"

# This downloads to cache (~/.cache/huggingface/hub by default)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save to an explicit local directory for portability
tokenizer.save_pretrained("./models/bert-base-uncased")
model.save_pretrained("./models/bert-base-uncased")

# Later, load from local path — no internet required
tokenizer = AutoTokenizer.from_pretrained("./models/bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./models/bert-base-uncased")
```

#### Using huggingface-cli for Bulk Downloads

```bash
# Login once
huggingface-cli login

# Download entire model repository to local cache
huggingface-cli download meta-llama/Llama-2-7b-hf \
  --local-dir ./models/llama2-7b \
  --local-dir-use-symlinks False

# Download specific files only (e.g., just the config and tokenizer)
huggingface-cli download meta-llama/Llama-2-7b-hf \
  --include "tokenizer*" "config.json" "*.model"
```

---

## 3. The HuggingFace Ecosystem

### 3.1 Hub Model Naming Convention

```
{organization_or_user}/{model-name}
Examples:
  bert-base-uncased           → Legacy, top-level model
  google/flan-t5-xl           → Google's Flan-T5 XL
  meta-llama/Llama-2-7b-hf   → Meta's LLaMA 2 7B (HF format)
  mistralai/Mistral-7B-v0.1  → Mistral 7B
  your-username/my-finetuned → Your own pushed model
```

### 3.2 Navigating the Hub — What to Look For

Before downloading any model, examine these on the model card:

1. **Model type** — Is it base, instruct, chat, or RLHF-tuned? 
   - `base` = raw pretrained, best for further fine-tuning
   - `instruct` / `chat` = instruction-tuned, best for direct inference
2. **License** — `llama2` license ≠ Apache 2.0. Check before production use
3. **Tags** — `text-classification`, `token-classification`, `text-generation` etc.
4. **Files** — Look for `config.json`, `tokenizer.json`, and weight files (`.safetensors` preferred over `.bin`)
5. **Model size** — Rule of thumb: 1B params ≈ 2GB in fp16, 4GB in fp32

### 3.3 Key Model Families and When to Use Them

| Task | Recommended Model | Reason |
|------|------------------|--------|
| Text classification | `deberta-v3-base` | Best encoder for classification benchmarks |
| NER/Token classification | `roberta-base` | Strong contextual embeddings |
| Sentence embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Balanced speed/quality |
| Summarization | `facebook/bart-large-cnn` | SOTA for news summarization |
| Translation | `Helsinki-NLP/opus-mt-*` | Specialized per language pair |
| Text generation (small) | `mistralai/Mistral-7B-Instruct-v0.2` | Excellent quality, fits on single GPU |
| Text generation (large) | `meta-llama/Llama-2-13b-chat-hf` | Strong instruction following |
| Multilingual | `xlm-roberta-base` | 100 languages, robust |
| Code | `Salesforce/codegen2-7B` / `bigcode/starcoder2-7b` | Code completion |

---

## 4. Tokenizers

### 4.1 Why Tokenizers Matter More Than Most Engineers Think

The tokenizer converts text to integer IDs. **A wrong or mismatched tokenizer will destroy your model's performance silently.** Each model was pretrained with a specific tokenizer — you must use the same one.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Basic encoding
text = "The quick brown fox jumps over the lazy dog."
encoding = tokenizer(text)

print(encoding.input_ids)      # [101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 1012, 102]
print(encoding.attention_mask) # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(tokenizer.decode(encoding.input_ids))  # [CLS] the quick brown fox jumps over the lazy dog. [SEP]
```

### 4.2 Essential Tokenizer Parameters

```python
# For model input — always use these parameters
encoding = tokenizer(
    text,
    max_length=512,           # Truncate to model's max context length
    truncation=True,          # Enable truncation
    padding="max_length",     # Pad to max_length (for batching)
    padding="longest",        # Alternative: pad to longest in batch (more efficient)
    return_tensors="pt",      # Return PyTorch tensors; "tf" for TF, "np" for numpy
    return_attention_mask=True,
    return_token_type_ids=True,  # Needed for BERT-style models
    add_special_tokens=True,     # Add [CLS], [SEP], <s>, </s> etc.
)

# Batch encoding
texts = ["First sentence.", "Second sentence.", "Third sentence."]
batch_encoding = tokenizer(
    texts,
    max_length=128,
    truncation=True,
    padding=True,
    return_tensors="pt",
)
# Shape: input_ids = [3, 128]
```

### 4.3 Understanding Special Tokens

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Inspect special tokens
print(tokenizer.cls_token)     # [CLS]
print(tokenizer.sep_token)     # [SEP]
print(tokenizer.pad_token)     # [PAD]
print(tokenizer.unk_token)     # [UNK]
print(tokenizer.mask_token)    # [MASK] — for MLM

print(tokenizer.cls_token_id)  # 101
print(tokenizer.pad_token_id)  # 0

# GPT-style tokenizers (decoder-only) often have no pad token by default
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(gpt_tokenizer.pad_token)  # None ← This will cause issues in batching!

# Fix: Set pad token to eos token (standard practice for GPT-style)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
# Optionally add a new pad token
# gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
```

### 4.4 Handling Long Texts — Chunking Strategy

```python
def tokenize_with_overlap(text, tokenizer, max_length=512, overlap=50):
    """
    Splits long text into overlapping chunks.
    Useful for document classification, long-context NER.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Account for [CLS] and [SEP]
    effective_length = max_length - 2
    step = effective_length - overlap
    
    chunks = []
    for i in range(0, max(1, len(tokens) - overlap), step):
        chunk = tokens[i:i + effective_length]
        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        # Pad if necessary
        padding_length = max_length - len(chunk)
        attention_mask = [1] * len(chunk) + [0] * padding_length
        chunk = chunk + [tokenizer.pad_token_id] * padding_length
        chunks.append({
            "input_ids": chunk,
            "attention_mask": attention_mask
        })
        if i + effective_length >= len(tokens):
            break
    
    return chunks
```

---

## 5. Loading & Running Pretrained Models

### 5.1 The Auto Classes — Your Primary Interface

```python
from transformers import (
    AutoTokenizer,
    AutoModel,                           # Base model (no task head)
    AutoModelForSequenceClassification,  # Text classification
    AutoModelForTokenClassification,     # NER, POS tagging
    AutoModelForQuestionAnswering,       # Extractive QA
    AutoModelForCausalLM,               # Decoder-only text generation
    AutoModelForSeq2SeqLM,              # Encoder-decoder tasks
    AutoModelForMaskedLM,               # Masked language modeling
    AutoModelForMultipleChoice,         # Multiple choice tasks
)
```

> **Rule:** Always use `Auto*` classes unless you have a specific reason to use the concrete class (e.g., `BertForSequenceClassification`). Auto classes are forward-compatible.

### 5.2 Loading Models with Precision Control

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# FP32 (default) — Full precision, highest memory usage
model = AutoModelForCausalLM.from_pretrained(model_name)

# FP16 — Half precision, ~2x memory reduction, minor accuracy loss
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # Automatically distributes across available GPUs/CPU
)

# BF16 — Better numerical stability than FP16 for large models (requires Ampere GPU+)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# INT8 quantization (load_in_8bit) — requires bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# INT4 quantization (load_in_4bit) — QLoRA setup
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # Nested quantization for extra memory savings
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — best for normal distributions
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 5.3 Running Basic Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()  # Always set to eval mode for inference

# Prepare input
prompt = "Explain the attention mechanism in transformers."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,        # Creativity (0 = deterministic, 1 = sampling)
        top_p=0.95,             # Nucleus sampling
        top_k=50,               # Top-k sampling
        repetition_penalty=1.1, # Reduce repetition
        do_sample=True,         # Enable sampling (vs greedy)
        pad_token_id=tokenizer.eos_token_id  # Suppress padding warning
    )

# Decode — only the newly generated tokens
new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response)
```

### 5.4 Chat Templates — The Correct Way for Instruction Models

Instruction-tuned models expect prompts in a specific format. Using the wrong format severely degrades performance.

```python
# The modern way — use the model's built-in chat template
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is its population?"}
]

# Apply chat template (handles all formatting automatically)
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,         # Return string instead of tokens
    add_generation_prompt=True  # Add the prompt that triggers model generation
)

print(formatted_prompt)
# Output: <s>[INST] What is the capital of France? [/INST] The capital of France is Paris. </s>[INST] What is its population? [/INST]

# Tokenize and run
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
```

---

## 6. The `pipeline` API — Fast Prototyping

### 6.1 When to Use Pipeline (and When NOT To)

**Use `pipeline` when:**
- Rapid prototyping or experimentation
- You don't need fine-grained control over the model
- You're running a standard task (classification, summarization, etc.)

**Do NOT use `pipeline` when:**
- Fine-tuning (it hides training infrastructure)
- Custom preprocessing/postprocessing is needed
- Production systems where you need full control

### 6.2 All Important Pipeline Types

```python
from transformers import pipeline

# --- TEXT CLASSIFICATION ---
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0  # GPU 0; -1 for CPU
)
result = classifier("I absolutely loved this movie!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# --- ZERO-SHOT CLASSIFICATION (no fine-tuning needed) ---
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = zero_shot(
    "This email is about a meeting tomorrow at 3pm.",
    candidate_labels=["calendar", "urgent", "spam", "finance", "HR"]
)
# Ranks labels by relevance without any training

# --- NAMED ENTITY RECOGNITION ---
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy="simple")
result = ner("Elon Musk founded SpaceX in Hawthorne, California.")
# [{'entity_group': 'PER', 'word': 'Elon Musk', ...}, ...]

# --- QUESTION ANSWERING (Extractive) ---
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
result = qa(
    question="What is the boiling point of water?",
    context="Water boils at 100 degrees Celsius at standard atmospheric pressure."
)
# {'answer': '100 degrees Celsius', 'score': 0.98, 'start': 15, 'end': 34}

# --- SUMMARIZATION ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn",
                      min_length=30, max_length=150, do_sample=False)

# --- TEXT GENERATION ---
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=100,
    num_return_sequences=3,  # Generate 3 variations
    temperature=0.8
)

# --- TRANSLATION ---
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

# --- FILL MASK ---
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
result = fill_mask("The capital of France is [MASK].")

# --- FEATURE EXTRACTION (embeddings) ---
embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder(["Hello world", "Hi there"])

# --- BATCH PROCESSING (for efficiency) ---
texts = ["Text 1", "Text 2", "Text 3", ...]  # Large list
results = classifier(texts, batch_size=32)    # Process in batches
```

---

## 7. Datasets

### 7.1 Loading Datasets

```python
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

# Load from Hub
dataset = load_dataset("imdb")
# DatasetDict({'train': Dataset(3000 rows), 'test': Dataset(...)})

# Load specific split
train_dataset = load_dataset("imdb", split="train")

# Load with subset/configuration
dataset = load_dataset("glue", "mrpc")  # GLUE MRPC task

# Load from local files
# CSV
dataset = load_dataset("csv", data_files="./data/my_data.csv")
dataset = load_dataset("csv", data_files={
    "train": "./data/train.csv",
    "validation": "./data/val.csv",
    "test": "./data/test.csv"
})

# JSON Lines
dataset = load_dataset("json", data_files="./data/my_data.jsonl")

# Text file
dataset = load_dataset("text", data_files="./data/corpus.txt")

# From a pandas DataFrame
df = pd.read_csv("my_data.csv")
dataset = Dataset.from_pandas(df)

# From a Python dict
data_dict = {
    "text": ["I love this.", "I hate this.", "This is okay."],
    "label": [1, 0, 1]
}
dataset = Dataset.from_dict(data_dict)
```

### 7.2 Exploring and Inspecting Datasets

```python
# Basic inspection
print(dataset)
print(dataset.features)     # Column types
print(dataset.column_names)
print(dataset[0])           # First row
print(dataset[:5])          # First 5 rows
print(dataset["text"][:3])  # First 3 items in a column

# Statistics
print(dataset.num_rows)
print(len(dataset))
```

### 7.3 Preprocessing Datasets — The `.map()` Function

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    """
    This function is called on BATCHES of examples.
    'examples' is a dict of lists, not a single item.
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Apply preprocessing
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,          # Process in batches (MUCH faster than row-by-row)
    batch_size=1000,       # Batch size for mapping
    num_proc=4,            # Parallelize with 4 CPU cores
    remove_columns=["text"],  # Remove original text column after tokenizing
    desc="Tokenizing"      # Progress bar label
)

# Set the format for PyTorch
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# Cache — .map() results are automatically cached
# Force recompute with:
tokenized_dataset = dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
```

### 7.4 Splitting and Shuffling

```python
# Train/test split from a single dataset
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
test_dataset = split["test"]

# Stratified split (important for imbalanced classification)
split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")

# Shuffle
shuffled = dataset.shuffle(seed=42)

# Select a subset
small_dataset = dataset.select(range(1000))  # First 1000 examples
```

### 7.5 Building a Custom Dataset Class (PyTorch Style)

```python
from torch.utils.data import Dataset as TorchDataset
import torch

class CustomTextDataset(TorchDataset):
    """
    Use this when your data doesn't fit neatly into HF Dataset format,
    or when you need complex preprocessing logic.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None  # Return lists, not tensors (we'll convert per item)
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Usage
train_texts = ["Positive review!", "Negative review!", "Neutral review."]
train_labels = [1, 0, 1]
train_dataset = CustomTextDataset(train_texts, train_labels, tokenizer)
```

---

## 8. Fine-Tuning Strategy

### 8.1 Decision Framework — Which Approach to Choose

```
Your Task
    │
    ├─ Do you have labeled data?
    │       │
    │       ├─ YES → How much?
    │       │           ├─ < 1,000 examples → Zero/Few-Shot or PEFT (LoRA)
    │       │           ├─ 1,000–50,000 → Fine-tune smaller model OR PEFT on larger
    │       │           └─ > 50,000 → Full fine-tune OR PEFT on larger model
    │       │
    │       └─ NO → Zero-Shot with instruction model
    │
    ├─ What's your hardware?
    │       ├─ < 16GB VRAM → QLoRA (4-bit) or LoRA on small models
    │       ├─ 16–40GB VRAM → LoRA or full fine-tune of 1-7B models
    │       └─ 40GB+ VRAM → Full fine-tune up to 13B, or larger with DeepSpeed
    │
    └─ What's your task?
            ├─ Classification → Encoder model (BERT, DeBERTa, RoBERTa)
            ├─ Generation/Chat → Decoder model (LLaMA, Mistral) + instruction tuning
            ├─ Summarization/Translation → Encoder-Decoder (T5, BART)
            └─ Embeddings → Sentence transformers with contrastive learning
```

### 8.2 The Learning Rate is the Most Important Hyperparameter

| Model Size | Fine-tuning Type | Recommended LR |
|-----------|-----------------|----------------|
| BERT-base (110M) | Full fine-tune | 2e-5 to 5e-5 |
| RoBERTa-large (355M) | Full fine-tune | 1e-5 to 3e-5 |
| T5-base (250M) | Full fine-tune | 3e-4 to 1e-3 |
| 7B models | LoRA | 1e-4 to 3e-4 |
| 7B models | QLoRA | 2e-4 to 2e-4 |
| 13B models | LoRA/QLoRA | 1e-4 to 2e-4 |

> **Golden Rule:** When in doubt, start with a lower learning rate and use a scheduler with warmup.

---

## 9. Full Fine-Tuning with Trainer

### 9.1 Classification Fine-Tuning — Complete Example

```python
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
import evaluate

# ─── 1. Load Data ───────────────────────────────────────────────────────────
dataset = load_dataset("imdb")  # Replace with your dataset

# ─── 2. Setup Tokenizer ─────────────────────────────────────────────────────
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        # No padding here — DataCollatorWithPadding handles it dynamically
    )

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"],
)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# ─── 3. Load Model ──────────────────────────────────────────────────────────
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# ─── 4. Data Collator ───────────────────────────────────────────────────────
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ─── 5. Metrics ─────────────────────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

# ─── 6. Training Arguments ──────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results/my-classifier",
    
    # Training schedule
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,   # Effective batch size = batch_size × grad_accum
    
    # Learning rate
    learning_rate=2e-5,
    weight_decay=0.01,               # L2 regularization
    warmup_ratio=0.06,               # 6% of steps for LR warmup
    lr_scheduler_type="linear",      # linear, cosine, cosine_with_restarts
    
    # Evaluation & saving
    eval_strategy="epoch",           # Evaluate every epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    
    # Optimization
    fp16=True,                       # Mixed precision (if GPU supports)
    dataloader_num_workers=4,        # Parallel data loading
    
    # Logging
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",                # "wandb", "tensorboard", or "none"
    
    # Misc
    seed=42,
    push_to_hub=False,
)

# ─── 7. Trainer ─────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ─── 8. Train ───────────────────────────────────────────────────────────────
trainer.train()

# ─── 9. Evaluate ────────────────────────────────────────────────────────────
eval_results = trainer.evaluate()
print(f"Eval Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Eval F1: {eval_results['eval_f1']:.4f}")

# ─── 10. Save ───────────────────────────────────────────────────────────────
trainer.save_model("./models/my-classifier-final")
tokenizer.save_pretrained("./models/my-classifier-final")
```

### 9.2 Sequence-to-Sequence Fine-Tuning (Summarization)

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np

model_checkpoint = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load summarization dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
# Features: 'article' (input), 'highlights' (target summary)

max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
    )
    
    # Tokenize targets with text_target
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["highlights"],
            max_length=max_target_length,
            truncation=True,
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Seq2Seq specific data collator (handles label shifting)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in labels (padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="./results/summarizer",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    predict_with_generate=True,  # CRITICAL for Seq2Seq evaluation
    generation_max_length=max_target_length,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rouge2",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

## 10. PEFT / LoRA / QLoRA

### 10.1 Understanding LoRA — Why It Works

**LoRA (Low-Rank Adaptation)** freezes the original model weights and adds small trainable matrices to specific layers. Instead of updating a W matrix of shape (d × k), it learns two smaller matrices A (d × r) and B (r × k), where r << d.

```
Original: W ∈ ℝ^(d×k)           — e.g., 4096 × 4096 = 16.7M parameters
LoRA:     W + BA where:
              B ∈ ℝ^(d×r)        — e.g., 4096 × 8 = 32,768 parameters  
              A ∈ ℝ^(r×k)        — e.g., 8 × 4096 = 32,768 parameters
              Total new: 65,536 parameters (0.4% of original!)
```

**Why this matters:** You only update ~1% of parameters, so you need ~100x less memory and training is ~10x faster, while retaining 95%+ of full fine-tuning quality.

### 10.2 LoRA Fine-Tuning — Complete Template

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

# ─── 1. Model & Quantization Config ────────────────────────────────────────
model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Important for causal LM

# ─── 2. Prepare Model for QLoRA ─────────────────────────────────────────────
# This enables gradient checkpointing and casts layer norms to fp32
model = prepare_model_for_kbit_training(model)

# ─── 3. LoRA Config ─────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                           # LoRA rank — higher = more capacity but more memory
    lora_alpha=32,                  # Scaling factor (typically 2×r)
    target_modules=[                # Modules to apply LoRA to
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",                    # Don't train biases
    inference_mode=False,
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Prints: trainable params: 39,976,960 || all params: 3,791,374,336 || trainable%: 1.054

# ─── 4. Dataset Preparation ─────────────────────────────────────────────────
dataset = load_dataset("your_dataset", split="train")

def format_instruction(sample):
    """Format for instruction fine-tuning."""
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample.get('input', '')}

### Response:
{sample['output']}"""

# ─── 5. Training Arguments ──────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results/mistral-7b-lora",
    
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # Effective batch = 4 × 4 = 16
    
    learning_rate=2e-4,
    weight_decay=0.001,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    optim="paged_adamw_32bit",      # Memory-efficient optimizer for QLoRA
    fp16=False,
    bf16=True,                       # Use bf16 for Ampere+ GPUs
    
    gradient_checkpointing=True,     # Trade compute for memory
    max_grad_norm=0.3,              # Gradient clipping
    
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=25,
    save_total_limit=3,
    load_best_model_at_end=True,
    
    group_by_length=True,           # Group samples by length for efficiency
    report_to="none",
    seed=42,
)

# ─── 6. SFT Trainer (from TRL) ─────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=format_instruction,
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=False,              # True = pack multiple short samples per sequence
)

trainer.train()

# ─── 7. Save LoRA Adapters ──────────────────────────────────────────────────
trainer.model.save_pretrained("./models/mistral-lora-adapters")
tokenizer.save_pretrained("./models/mistral-lora-adapters")
```

### 10.3 Target Modules by Model Architecture

```python
# Find the correct target modules for any model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-model")

# Print all linear layer names
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append(name)
        
print(linear_layers[:20])

# Common target modules by architecture:
# Mistral / LLaMA:    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Falcon:             ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
# GPT-2:              ["c_attn", "c_proj", "c_fc"]
# BERT/RoBERTa:       ["query", "key", "value", "dense"]
# T5:                 ["q", "k", "v", "o", "wi", "wo"]
# Bloom:              ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
```

### 10.4 Loading and Merging LoRA Weights

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters on top
model_with_lora = PeftModel.from_pretrained(
    base_model,
    "./models/mistral-lora-adapters",
    device_map="auto"
)

# Option A: Keep separate (smaller files, can swap adapters)
# Use model_with_lora for inference directly

# Option B: Merge for deployment (single model file, faster inference)
merged_model = model_with_lora.merge_and_unload()
merged_model.save_pretrained("./models/mistral-merged")
tokenizer.save_pretrained("./models/mistral-merged")

# Option C: Multiple LoRA adapters on same base model
from peft import load_peft_weights, set_peft_model_state_dict

# You can hot-swap adapters without reloading the base model
model_with_lora.load_adapter("./models/adapter-v2", adapter_name="v2")
model_with_lora.set_adapter("v2")  # Switch to v2
model_with_lora.set_adapter("default")  # Switch back
```

---

## 11. Fine-Tuning for Specific Tasks

### 11.1 Instruction Fine-Tuning / Supervised Fine-Tuning (SFT)

```python
# Standard instruction format — adjust to match your base model's convention
def format_alpaca(sample):
    if sample.get("input"):
        return f"""Below is an instruction that describes a task, paired with an input.
Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    return f"""Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

# For training ONLY on the response (not the instruction)
# Use DataCollatorForCompletionOnlyLM from TRL
from trl import DataCollatorForCompletionOnlyLM

response_template = "### Response:"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)
# This ensures loss is only computed on the response tokens, not the instruction
```

### 11.2 Text Classification — Custom Labels

```python
# Multi-class classification (>2 classes)
from transformers import AutoModelForSequenceClassification

categories = ["sports", "politics", "technology", "entertainment", "business"]
id2label = {i: label for i, label in enumerate(categories)}
label2id = {label: i for i, label in enumerate(categories)}

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(categories),
    id2label=id2label,
    label2id=label2id
)

# Multi-LABEL classification (multiple labels per example)
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(categories),
    problem_type="multi_label_classification"  # Changes loss from CE to BCE
)

# For multi-label, your labels should be a float tensor of shape [batch, num_labels]
# Example: [0, 1, 0, 1, 0] means labels "politics" and "entertainment" are active
```

### 11.3 Named Entity Recognition (NER)

```python
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

# Define your entity types
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",  # Use cased for NER — capitalization matters!
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,  # CRITICAL: tokens are already split
        max_length=512,
        truncation=True,
    )
    
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens get -100 (ignored in loss)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                # Subword tokens after the first get -100 or the label with I- prefix
                label_ids.append(-100)  # Simple strategy
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

### 11.4 Question Answering (Extractive)

```python
from transformers import AutoModelForQuestionAnswering, DefaultDataCollator

model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking")

def preprocess_qa(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",  # Only truncate context, not the question
        stride=128,                # Overlap between windows
        return_overflowing_tokens=True,  # Handle long contexts
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Map tokens back to character positions for answer extraction
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = examples["answers"][sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        # Token positions
        sequence_ids = inputs.sequence_ids(i)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # Clamp to context window
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
```

---

## 12. Evaluation & Metrics

### 12.1 Using the `evaluate` Library

```python
import evaluate
import numpy as np

# Classification metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

# NLP generation metrics
rouge = evaluate.load("rouge")      # Summarization
bleu = evaluate.load("bleu")        # Translation
bertscore = evaluate.load("bertscore")  # Semantic similarity

# NER
seqeval = evaluate.load("seqeval")

# Usage
predictions = [0, 1, 1, 0, 1]
references  = [0, 1, 0, 0, 1]

acc_result = accuracy.compute(predictions=predictions, references=references)
print(acc_result)  # {'accuracy': 0.8}

# For text generation
preds = ["the cat sat on the mat", "it is a nice day today"]
refs  = ["the cat sat on the mat", "today is a nice day"]
rouge_result = rouge.compute(predictions=preds, references=refs)
print(rouge_result)  # {'rouge1': ..., 'rouge2': ..., 'rougeL': ...}
```

### 12.2 Custom Evaluation During Training

```python
def make_compute_metrics(threshold=0.5):
    """Factory function for configurable metrics."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        
        # For multi-label classification
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.tensor(logits))
        predictions = (probs >= threshold).int().numpy()
        
        # For multi-class
        # predictions = np.argmax(logits, axis=-1)
        
        # Flatten for multi-label metrics
        flat_preds = predictions.flatten()
        flat_labels = labels.flatten()
        
        return {
            "accuracy": accuracy_metric.compute(predictions=flat_preds, references=flat_labels)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=flat_preds, references=flat_labels, average="macro")["f1"],
            "f1_weighted": f1_metric.compute(predictions=flat_preds, references=flat_labels, average="weighted")["f1"],
        }
    
    return compute_metrics
```

---

## 13. Model Saving, Export & Offline Deployment

### 13.1 Saving Formats

```python
# Standard HuggingFace format (recommended)
model.save_pretrained("./models/my_model")
tokenizer.save_pretrained("./models/my_model")
# Creates: config.json, model.safetensors (or pytorch_model.bin), tokenizer files

# Force safetensors format (safer, faster loading)
model.save_pretrained("./models/my_model", safe_serialization=True)

# Push to Hub
model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")
```

### 13.2 ONNX Export for Production Inference

```python
from transformers.onnx import export, FeaturesManager
from pathlib import Path

model_checkpoint = "./models/my_model"
save_path = Path("./models/my_model_onnx")
save_path.mkdir(exist_ok=True)

onnx_config = FeaturesManager.get_supported_features_for_model_type(
    "bert",
    "feature-extraction"  # or "sequence-classification", etc.
)

export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=13,
    output=save_path / "model.onnx"
)

# For inference with ONNX Runtime
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("./models/my_model_onnx/model.onnx")

inputs = tokenizer("Hello, world!", return_tensors="np")
outputs = session.run(None, dict(inputs))
```

### 13.3 Fully Offline Inference Pipeline

```python
# This pattern works 100% offline after initial download
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

LOCAL_MODEL_PATH = "./models/my-classifier-final"

# Load everything from local path
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Inference
results = classifier(["This is great!", "This is terrible."])
```

---

## 14. Google Colab — The Right Way

### 14.1 The Hard Truths About Colab

| Issue | Free Tier | Pro | Pro+ |
|-------|-----------|-----|------|
| GPU | T4 (15GB), random availability | T4/V100 | A100 (40GB) |
| RAM | ~12GB | ~25GB | ~52GB |
| Session timeout | ~12 hours | ~24 hours | ~24 hours |
| Storage | 15GB Google Drive | Same | Same |
| Disk (temporary) | ~78GB | ~166GB | ~225GB |

> **Critical Colab Rule:** All files in `/content/` are **wiped when the session ends**. Always save to Google Drive or download to local.

### 14.2 The Gold-Standard Colab Setup Cell

```python
# ═══════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP — Run this first, every session
# ═══════════════════════════════════════════════════════════════

# Check hardware
import subprocess
gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(gpu_info.stdout if gpu_info.returncode == 0 else "No GPU detected")

import os, sys
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "")

# Install dependencies
!pip install -q transformers datasets peft trl accelerate evaluate bitsandbytes sentencepiece

# Mount Google Drive (for persistent storage)
from google.colab import drive
drive.mount('/content/drive')

# Set cache to Google Drive (survives session restarts!)
GDRIVE_BASE = "/content/drive/MyDrive/ML/huggingface_cache"
os.makedirs(GDRIVE_BASE, exist_ok=True)

os.environ["HF_HOME"] = GDRIVE_BASE
os.environ["TRANSFORMERS_CACHE"] = f"{GDRIVE_BASE}/hub"
os.environ["HF_DATASETS_CACHE"] = f"{GDRIVE_BASE}/datasets"

print(f"\n✅ HF cache → {GDRIVE_BASE}")
print("✅ Environment ready!")
```

### 14.3 HuggingFace Authentication in Colab

```python
# ═══════════════════════════════════════════════════════════════
# CELL 2: AUTHENTICATION (for gated models like LLaMA)
# ═══════════════════════════════════════════════════════════════

# Method 1: Interactive login (opens browser)
from huggingface_hub import notebook_login
notebook_login()

# Method 2: Using Colab Secrets (RECOMMENDED — no token in notebook)
from google.colab import userdata
from huggingface_hub import login

HF_TOKEN = userdata.get('HF_TOKEN')  # Set in Colab Secrets panel (🔑 icon on left)
login(token=HF_TOKEN)
```

### 14.4 Memory Management in Colab

```python
import gc
import torch

def free_memory():
    """Call this between experiments to avoid OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Delete model and free memory
def delete_model(model):
    del model
    free_memory()

# Check available memory before loading
def check_memory():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        free = total - reserved
        print(f"Total: {total:.1f}GB | Reserved: {reserved:.1f}GB | Allocated: {allocated:.1f}GB | Free: {free:.1f}GB")
    
check_memory()
```

### 14.5 The Anti-Crash Training Pattern for Colab

```python
# ═══════════════════════════════════════════════════════════════
# SAFE TRAINING PATTERN FOR COLAB
# Handles session crashes, saves checkpoints to Drive
# ═══════════════════════════════════════════════════════════════

import os
from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback

CHECKPOINT_DIR = "/content/drive/MyDrive/ML/checkpoints/my-experiment"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class ColabSafeCallback(TrainerCallback):
    """
    Saves a checkpoint to Drive after every eval.
    Protects against Colab session timeouts.
    """
    def __init__(self, drive_checkpoint_dir):
        self.drive_dir = drive_checkpoint_dir
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # Save current state to Drive
        checkpoint_path = os.path.join(self.drive_dir, f"step-{state.global_step}")
        model.save_pretrained(checkpoint_path)
        print(f"\n✅ Saved checkpoint to Drive: {checkpoint_path}")

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    
    # For Colab T4 (15GB VRAM) with 7B model:
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Simulate batch size of 16
    
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",        # Most memory-efficient optimizer
    
    # Aggressive checkpointing for crash recovery
    save_strategy="steps",
    save_steps=50,
    eval_steps=50,
    save_total_limit=2,
    
    # Resume from checkpoint if session crashed
    # Trainer auto-detects latest checkpoint in output_dir
    
    num_train_epochs=3,
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    # ... other args
    callbacks=[ColabSafeCallback(CHECKPOINT_DIR)]
)

# Resume training from checkpoint if it exists
import glob
checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*"))
resume_from = max(checkpoints, key=os.path.getctime) if checkpoints else None

if resume_from:
    print(f"⚡ Resuming from {resume_from}")

trainer.train(resume_from_checkpoint=resume_from)
```

### 14.6 Colab Hardware-Specific Model Size Guide

```
T4 GPU (15GB VRAM) — Free Tier
├── Full fine-tune: up to BERT-large (340M params in fp32) or GPT-2-medium
├── LoRA: up to 7B models in fp16 (barely, with gradient checkpointing)
└── QLoRA (4-bit): up to 13B models comfortably, 30B with tricks

V100 GPU (16GB VRAM) — Pro
├── Full fine-tune: up to 1-2B models in fp16
└── QLoRA: up to 30B models

A100 GPU (40GB VRAM) — Pro+
├── Full fine-tune: up to 13B models in fp16
├── LoRA: up to 70B models in fp16
└── QLoRA: up to 70B+ models
```

---

## 15. Memory Optimization & Quantization

### 15.1 Gradient Checkpointing

Gradient checkpointing trades compute for memory by recomputing activations during the backward pass instead of storing them.

```python
# Enable before training
model.gradient_checkpointing_enable()

# In TrainingArguments
training_args = TrainingArguments(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Newer recommended setting
    ...
)
```

### 15.2 Mixed Precision Training

```python
# FP16 (most common, supported on all modern GPUs)
training_args = TrainingArguments(fp16=True, ...)

# BF16 (better range, requires Ampere GPU — A100, RTX 3090, etc.)
training_args = TrainingArguments(bf16=True, ...)

# For inference only
with torch.autocast("cuda", dtype=torch.float16):
    outputs = model(**inputs)

# Or using torch.cuda.amp
from torch.cuda.amp import autocast
with autocast():
    outputs = model(**inputs)
```

### 15.3 Dynamic Quantization (Post-Training, CPU-Focused)

```python
import torch

# Dynamic quantization — fastest, works on CPU, INT8 weights
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # Quantize all linear layers
    dtype=torch.qint8
)

# Check size reduction
def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

print(f"Original: {get_model_size_mb(model):.1f} MB")
print(f"Quantized: {get_model_size_mb(quantized_model):.1f} MB")
```

### 15.4 Accelerate for Multi-GPU Training

```python
# accelerate config — run this once from CLI
# accelerate config

# In your training script
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)

# Wrap model, optimizer, dataloaders
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Training loop becomes device-agnostic
for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

# Launch with: accelerate launch train.py
# Or for Colab: accelerator = Accelerator() # auto-detects
```

---

## 16. Troubleshooting & Common Pitfalls

### 16.1 The Most Common Errors and Their Fixes

```
ERROR: RuntimeError: CUDA out of memory
FIXES:
  1. Reduce batch size → per_device_train_batch_size=1
  2. Increase gradient_accumulation_steps to compensate
  3. Enable gradient_checkpointing=True
  4. Use load_in_8bit=True or load_in_4bit=True
  5. Use fp16=True if not already
  6. Call torch.cuda.empty_cache() before training
  7. Set PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

ERROR: ValueError: You should supply an encoding or a list of encodings
FIXES:
  - Make sure return_tensors="pt" is set in tokenizer call
  - If using DataLoader, ensure collate_fn returns tensors

ERROR: IndexError: index out of range in self
FIXES:
  - Tokenizer sequence length exceeds model's max position embeddings
  - Set max_length=model.config.max_position_embeddings in tokenizer call
  - Or set truncation=True (but also set max_length)

ERROR: The size of tensor a must match the size of tensor b
FIXES:
  - Label shape mismatch — check num_labels in model vs dataset
  - For classification: labels should be [batch_size], not [batch_size, 1]
  - Squeeze labels: labels = labels.squeeze(-1)

ERROR: Loss is NaN from the first step
FIXES:
  - Learning rate too high → reduce by 10x
  - Input IDs contain out-of-vocabulary tokens
  - Labels contain -100 everywhere (all tokens masked)
  - fp16 overflow → switch to bf16 or reduce lr

ERROR: Model generates garbage / repetitive text
FIXES:
  - Repetition_penalty=1.1 to 1.3
  - Use temperature < 1.0 for less randomness
  - Model is not instruction-tuned → use base model for generation
  - Wrong chat template applied

ERROR: fine-tuned model performs worse than base model
FIXES:
  - Learning rate too high → lower by 5-10x
  - Overfitting → add dropout, reduce epochs, use smaller model
  - Catastrophic forgetting → use LoRA instead of full fine-tuning
  - Label noise in your dataset → audit training data
  - Insufficient data for the task complexity
```

### 16.2 Debugging Tools

```python
# Check for NaN in gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in: {name}")

# Monitor GPU memory during training
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def print_gpu_memory():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Memory: {info.used/1e9:.2f}GB / {info.total/1e9:.2f}GB")

# Check token distribution in your dataset
from collections import Counter
lengths = [len(tokenizer.encode(text)) for text in your_texts]
print(f"Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.0f}")
# If max >> mean, you have outliers that will waste compute
```

### 16.3 Reproducibility Checklist

```python
import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# Also set seed in TrainingArguments: seed=42
```

---

## 17. Reference Architectures & Quick Templates

### 17.1 Quick-Start Templates by Task

#### Template A: Binary/Multi-Class Text Classification
```python
# TEMPLATE A: Classification
# Models: bert-base-uncased, roberta-base, deberta-v3-base
# Dataset format: {"text": str, "label": int}

TEMPLATE_CONFIG = {
    "model": "microsoft/deberta-v3-base",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_ratio": 0.06,
}
```

#### Template B: Instruction Fine-Tuning (Small Model, Free GPU)
```python
# TEMPLATE B: Instruction Tuning on Colab T4
# Models: TinyLlama/TinyLlama-1.1B-Chat-v1.0, Qwen/Qwen2-1.5B-Instruct
# Use for: Custom chatbots, domain-specific assistants
# VRAM: ~4GB (fits T4 with headroom for QLoRA)

TEMPLATE_CONFIG = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "epochs": 3,
}
```

#### Template C: QLoRA on 7B Model (A100 or Pro+ Colab)
```python
# TEMPLATE C: QLoRA on 7B Model
# Models: mistralai/Mistral-7B-v0.1, meta-llama/Llama-2-7b-hf
# VRAM required: ~12GB (T4 barely), ~8GB (with aggressive optimization)

TEMPLATE_CONFIG = {
    "model": "mistralai/Mistral-7B-v0.1",
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "lora_r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "batch_size": 1,
    "gradient_accumulation": 16,
    "learning_rate": 2e-4,
    "optimizer": "paged_adamw_32bit",
}
```

### 17.2 The Full Production Pipeline (End-to-End)

```python
"""
PRODUCTION PIPELINE CHECKLIST

Phase 1: Data Preparation
  □ Audit raw data for quality (duplicates, noise, label errors)
  □ Establish train/val/test split BEFORE any preprocessing
  □ Compute class distribution — handle imbalance if >10:1 ratio
  □ Determine max token length from data (not arbitrarily)
  □ Write tokenization function, verify on edge cases

Phase 2: Baseline
  □ Run a simple baseline (sklearn, rule-based) to know floor
  □ Test fine-tuning with 1 epoch on 10% of data first
  □ Verify loss decreases and metrics are sensible

Phase 3: Fine-Tuning
  □ Choose architecture matching the task
  □ Start with smaller model, scale up only if needed
  □ Use PEFT/LoRA for models >1B params
  □ Track experiments (wandb, mlflow, or even just a log file)
  □ Hyperparameter search: LR first, then batch size, then other

Phase 4: Evaluation
  □ Evaluate on HELD-OUT test set (not val set)
  □ Per-class metrics, not just overall accuracy
  □ Error analysis: examine worst-performing examples
  □ Test on out-of-domain data if relevant

Phase 5: Deployment
  □ Save model in safetensors format
  □ Test loaded model matches training-time performance
  □ Consider quantization for inference speed
  □ Document model: task, data, limitations, metrics
"""
```

### 17.3 Hardware Requirements Quick Reference

```
Task                           | Minimum VRAM | Recommended
───────────────────────────────────────────────────────────
BERT-base classification       | 4GB          | 8GB
BERT-large classification      | 8GB          | 16GB
T5-small/base summarization    | 4GB          | 8GB
GPT-2 fine-tuning              | 4GB          | 8GB
LLaMA-2-7B QLoRA               | 10GB         | 16GB
LLaMA-2-7B LoRA (fp16)         | 20GB         | 24GB
LLaMA-2-7B Full fine-tune      | 60GB         | 80GB+ (A100×2)
LLaMA-2-13B QLoRA              | 14GB         | 24GB
Mistral-7B QLoRA               | 10GB         | 16GB
Inference (7B, fp16)           | 14GB         | 16GB
Inference (7B, int4)           | 5GB          | 8GB
Inference (13B, int4)          | 9GB          | 12GB
```

---

## Appendix: Essential Commands Cheat Sheet

```bash
# ── Hub CLI ──────────────────────────────────────────────────────────────────
huggingface-cli login                         # Authenticate
huggingface-cli whoami                        # Check current user
huggingface-cli download {org}/{model}        # Download model to cache
huggingface-cli upload {org}/{model} ./dir    # Upload local model to Hub
huggingface-cli repo create {model-name}      # Create new Hub repo

# ── Cache Management ─────────────────────────────────────────────────────────
huggingface-cli cache scan                    # List cached models
huggingface-cli cache delete                  # Interactive cache cleanup
du -sh ~/.cache/huggingface/                  # Check total cache size

# ── Accelerate ───────────────────────────────────────────────────────────────
accelerate config                             # Interactive multi-GPU setup
accelerate launch train.py                    # Launch distributed training
accelerate launch --num_processes 2 train.py  # Specify GPU count

# ── System ───────────────────────────────────────────────────────────────────
nvidia-smi                                    # GPU status
nvidia-smi --loop=1                           # Live GPU monitoring
watch -n 1 nvidia-smi                         # Continuous monitoring
nvtop                                         # Interactive GPU monitor (install separately)
```

```python
# ── Python Snippets ──────────────────────────────────────────────────────────

# Count model parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.2f}%)")

# Move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Freeze all layers except classifier head
for param in model.base_model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

# Get model config
print(model.config)

# List all layer names
for name, _ in model.named_modules():
    print(name)
```

---

*Document Version: 1.0 | Validated against: transformers==4.40.x, peft==0.10.x, trl==0.8.x, datasets==2.19.x*

*Always pin your dependency versions in production: `pip freeze > requirements.txt`*