# 🧠 The Complete Beginner's Field Guide to LLMs, Transformers, Ollama & GPU AI
### From "What is a Token?" to Running LLaMA & Gemma Locally — With Every Example You Need

> **Who This Is For:** Complete beginners in AI/ML who want to go from zero to running real models locally, on GPU, and understanding *why* everything works the way it does. No fluff, no hand-waving — every concept is explained and every example is runnable.

---

## 📋 Table of Contents

### PART I — FOUNDATIONS (Start Here, Skip Nothing)
1. [What Is a Language Model? The Real Explanation](#1-what-is-a-language-model)
2. [Tokens, Embeddings & Vocabulary — The ABCs](#2-tokens-embeddings--vocabulary)
3. [The Transformer Architecture — Explained Simply](#3-the-transformer-architecture)
4. [Types of Models & What They're Good For](#4-types-of-models)
5. [Parameters, Model Sizes & What They Mean](#5-parameters-model-sizes--what-they-mean)

### PART II — HARDWARE (Your GPU Is the Engine)
6. [GPU Basics for AI — What You Actually Need to Know](#6-gpu-basics-for-ai)
7. [VRAM Requirements — Calculating Before You Download](#7-vram-requirements)
8. [CPU vs GPU vs Cloud — When to Use What](#8-cpu-vs-gpu-vs-cloud)
9. [Setting Up CUDA — The Right Way](#9-setting-up-cuda)

### PART III — OLLAMA (Run LLMs Locally, The Easy Way)
10. [What Is Ollama & Why It's a Game-Changer](#10-what-is-ollama)
11. [Installing Ollama on Windows, macOS & Linux](#11-installing-ollama)
12. [Running Your First Model (LLaMA, Gemma, Mistral)](#12-running-your-first-model)
13. [Ollama Model Library — What's Available](#13-ollama-model-library)
14. [Ollama REST API — Using It From Python](#14-ollama-rest-api)
15. [Ollama + LangChain & Custom Pipelines](#15-ollama--langchain--custom-pipelines)
16. [Modelfiles — Customizing Models in Ollama](#16-modelfiles--customizing-models)
17. [Ollama Performance Tuning & GPU Optimization](#17-ollama-performance-tuning)

### PART IV — TRANSFORMERS FOR BEGINNERS (The HuggingFace Way)
18. [Your First HuggingFace Model in 10 Lines](#18-your-first-huggingface-model)
19. [Understanding the Pipeline API Deeply](#19-understanding-the-pipeline-api)
20. [Tokenization — Step by Step With Real Examples](#20-tokenization-step-by-step)
21. [Running LLaMA & Gemma with Transformers](#21-running-llama--gemma-with-transformers)
22. [Prompt Engineering for Beginners](#22-prompt-engineering-for-beginners)
23. [Building Your First Fine-Tuned Classifier](#23-building-your-first-fine-tuned-classifier)

### PART V — PUTTING IT ALL TOGETHER
24. [Project: Local AI Chatbot with Ollama + Python](#24-project-local-ai-chatbot)
25. [Project: Document Q&A System](#25-project-document-qa-system)
26. [Troubleshooting Master Reference](#26-troubleshooting-master-reference)
27. [Learning Roadmap — What to Study Next](#27-learning-roadmap)

---

# PART I — FOUNDATIONS

---

## 1. What Is a Language Model?

### 1.1 The Simple, Honest Explanation

A language model is a program that has learned to **predict what comes next in a sequence of text** by reading billions of documents. That's it. The "intelligence" you see is an emergent property of doing this one thing at scale.

Think of it like this:

```
You type:    "The sky is "
The model:   Has read 500 billion words.
             Learned that "blue" follows this phrase 60% of the time,
             "cloudy" 15%, "clear" 10%, "dark" 8%, etc.
Result:      Outputs "blue" (or samples from the distribution)
```

### 1.2 How Training Works (Conceptually)

```
TRAINING PHASE:
┌─────────────────────────────────────────────────────────┐
│  Input text: "The cat sat on the"                       │
│  Target:     "mat"                                      │
│                                                         │
│  Model predicts: "floor" (wrong)                        │
│  Loss computed: How wrong was it?                       │
│  Weights updated: Slightly more likely to say "mat"     │
│                                                         │
│  Repeat this 10 TRILLION times across all of human text │
└─────────────────────────────────────────────────────────┘

INFERENCE PHASE (after training):
┌─────────────────────────────────────────────────────────┐
│  Your input:  "Explain gravity to a 5-year-old:"        │
│  Model:       Generates one token at a time             │
│  Token 1:     "Gravity"                                 │
│  Token 2:     "is"                                      │
│  Token 3:     "like"                                    │
│  ...continues until it generates an end token           │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Pre-trained vs Fine-tuned vs Instruction-tuned

This is the most important distinction for a beginner:

| Type | What It Is | What It Does | Example |
|------|-----------|--------------|---------|
| **Base/Pre-trained** | Trained on raw text only | Continues text (not good for chat) | LLaMA-2-7b |
| **Instruction-tuned** | Base + trained on instruction examples | Follows instructions, answers questions | LLaMA-2-7b-chat |
| **RLHF-tuned** | Instruction + human feedback reinforcement | Safer, more aligned | ChatGPT, Claude |
| **Fine-tuned** | Any of the above + your custom data | Specialized for your domain | Your model |

> ⚠️ **Beginner Mistake #1:** Downloading a base model and wondering why it doesn't chat. Always use the `-chat`, `-instruct`, or `-it` variant for conversation.

### 1.4 The Auto-Regressive Loop — Why Generation Is Slow

```python
# This is conceptually what happens during generation
def generate_text(prompt, max_tokens=100):
    tokens = tokenize(prompt)       # Convert text to numbers
    
    for _ in range(max_tokens):
        # Run entire model on ALL tokens so far
        next_token_probs = model.forward(tokens)
        
        # Pick the next token (sampling or greedy)
        next_token = sample(next_token_probs)
        
        tokens.append(next_token)   # Append to sequence
        
        if next_token == END_TOKEN:
            break
    
    return detokenize(tokens)       # Convert numbers back to text

# Why is it slow? The model runs ONCE PER TOKEN.
# For a 200-token response: model runs 200 times.
# Each run on a 7B model: millions of multiply-add operations.
```

---

## 2. Tokens, Embeddings & Vocabulary

### 2.1 What Is a Token?

A token is NOT a word. A token is a **chunk of text** determined by a tokenization algorithm. Most modern models use **Byte-Pair Encoding (BPE)** or a variant.

```python
# Install and run this yourself to see tokenization
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Example 1: Common words = 1 token
text = "cat"
print(tokenizer.encode(text))   # [9246] — one token

# Example 2: Uncommon words = multiple tokens
text = "photosynthesis"
print(tokenizer.encode(text))   # [34431, 428, 25361] — three tokens!

# Example 3: A full sentence
text = "The quick brown fox"
tokens = tokenizer.encode(text)
print(tokens)                   # [464, 2068, 7586, 21831]
print(len(tokens))              # 4 tokens

# See the actual token strings
for token_id in tokens:
    print(f"ID {token_id} → '{tokenizer.decode([token_id])}'")
# ID 464 → 'The'
# ID 2068 → ' quick'   ← note the space is part of the token
# ID 7586 → ' brown'
# ID 21831 → ' fox'
```

### 2.2 Token Counts vs Word Counts

A rough rule-of-thumb: **1 word ≈ 1.3 tokens in English**. But this varies:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

examples = [
    "Hello world",                              # Simple English
    "The mitochondria is the powerhouse",       # Scientific
    "नमस्ते दुनिया",                            # Hindi
    "def fibonacci(n): return n",              # Code
    "!@#$%^&*()",                              # Special chars
    "supercalifragilisticexpialidocious",       # Long word
]

for text in examples:
    tokens = tokenizer.encode(text)
    words = len(text.split())
    print(f"'{text[:30]}...' → {len(tokens)} tokens, {words} words")

# Lesson: Non-English text uses MANY more tokens (more expensive, slower)
# Code is relatively efficient (code tokens were in training data)
```

### 2.3 Context Window — The Model's Working Memory

```
┌──────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW                            │
│                                                              │
│  GPT-2:         1,024 tokens  (~750 words)                  │
│  LLaMA-2:       4,096 tokens  (~3,000 words)                │
│  Mistral-7B:    8,192 tokens  (~6,000 words)                │
│  LLaMA-3:      128,000 tokens (~96,000 words)               │
│  Gemma-2:       8,192 tokens  (~6,000 words)                │
│                                                              │
│  [Your Input Tokens] + [Generated Output Tokens]            │
│        must BOTH fit within the context window              │
│                                                              │
│  If you exceed it: older context is LOST (sliding window)   │
└──────────────────────────────────────────────────────────────┘
```

```python
# Practical example: Check if your text fits
def check_fits_in_context(text, model_name, max_context=4096):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_count = len(tokenizer.encode(text))
    
    print(f"Token count: {token_count}")
    print(f"Context window: {max_context}")
    print(f"Remaining for output: {max_context - token_count}")
    
    if token_count > max_context * 0.8:
        print("⚠️  WARNING: Using >80% of context. Leave room for output!")
    elif token_count > max_context:
        print("❌ ERROR: Text exceeds context window. Will be truncated!")
    else:
        print("✅ Fits comfortably")
    
    return token_count
```

### 2.4 Embeddings — How Words Become Numbers

Before tokens enter the model's attention layers, they're converted to **embedding vectors** — dense floating-point arrays that capture semantic meaning.

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Get embeddings for words
words = ["king", "queen", "man", "woman", "cat", "dog"]
embeddings = {}

for word in words:
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the [CLS] token embedding as the word representation
    embedding = outputs.last_hidden_state[0, 0, :].numpy()
    embeddings[word] = embedding

# The famous analogy: king - man + woman ≈ queen
def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

import numpy as np
analogy = embeddings["king"] - embeddings["man"] + embeddings["woman"]
similarity_to_queen = cosine_similarity(analogy, embeddings["queen"])
print(f"king - man + woman similarity to queen: {similarity_to_queen:.4f}")
# Should be high (~0.6-0.8) — this is WHY embeddings are powerful
```

---

## 3. The Transformer Architecture

### 3.1 The Big Picture

```
INPUT TEXT: "What is 2+2?"
     │
     ▼
┌─────────────┐
│  Tokenizer  │  Text → Token IDs [1, 1724, 338, 29871, 29906, 29974, 29906, 29973]
└─────────────┘
     │
     ▼
┌─────────────┐
│ Embedding   │  Token IDs → Vectors (shape: [8, 4096] for LLaMA-7B)
│   Layer     │  Each token becomes a 4096-dimensional vector
└─────────────┘
     │
     ▼
┌─────────────────────────────────┐
│   Transformer Block × 32        │  (32 layers for 7B model)
│  ┌─────────────────────────┐   │
│  │   Self-Attention         │   │  "How does each token relate to others?"
│  └─────────────────────────┘   │
│  ┌─────────────────────────┐   │
│  │   Feed-Forward Network  │   │  "Process the attended information"
│  └─────────────────────────┘   │
└─────────────────────────────────┘
     │
     ▼
┌─────────────┐
│  LM Head    │  Vectors → Probability over vocabulary
└─────────────┘
     │
     ▼
OUTPUT: Most likely next token → "4"
```

### 3.2 Self-Attention — The Core Innovation

Self-attention lets every token "look at" every other token when forming its representation. This is why Transformers beat RNNs — there's no distance penalty.

```python
import torch
import torch.nn.functional as F

# Simplified self-attention (educational — not production code)
def simple_self_attention(x, W_q, W_k, W_v):
    """
    x:    Input embeddings, shape [seq_len, embed_dim]
    W_q:  Query weight matrix
    W_k:  Key weight matrix  
    W_v:  Value weight matrix
    """
    # Step 1: Compute Queries, Keys, Values
    Q = x @ W_q   # "What am I looking for?"
    K = x @ W_k   # "What do I contain?"
    V = x @ W_v   # "What information do I provide?"
    
    # Step 2: Compute attention scores
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    # scores[i][j] = "How much should token i attend to token j?"
    
    # Step 3: Softmax to get probabilities
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 4: Weighted sum of values
    output = attention_weights @ V
    # Each token is now a weighted combination of all token values
    
    return output, attention_weights

# Visualizing attention
print("""
EXAMPLE: "The cat sat on the mat because it was tired"

Attention for "it" might look like:
Token:      The  cat  sat  on  the  mat  because  it  was  tired
Attention:  0.01 0.45 0.05 0.01 0.01 0.35  0.05   0.05 0.01 0.01

→ "it" strongly attends to "cat" (0.45) and "mat" (0.35)
→ Model learns that "it" refers to "cat" through attention
""")
```

### 3.3 Why "Large" Language Models Are Large

```
Model Size  │ Layers │ Hidden Dim │ Attention Heads │ Parameters
────────────┼────────┼────────────┼─────────────────┼────────────
BERT-base   │   12   │    768     │       12        │   110M
BERT-large  │   24   │  1,024     │       16        │   340M
GPT-2       │   12   │    768     │       12        │   117M
GPT-2-XL    │   48   │  1,600     │       25        │   1.5B
LLaMA-2-7B  │   32   │  4,096     │       32        │   6.7B
LLaMA-2-13B │   40   │  5,120     │       40        │  13.0B
LLaMA-2-70B │   80   │  8,192     │       64        │  69.0B
GPT-4       │  ???   │   ???      │      ???        │  ~1.8T (est)

More parameters = More "memory" to store knowledge
More layers = More "depth" of reasoning
More heads = More "perspectives" on each relationship
```

---

## 4. Types of Models

### 4.1 Encoder-Only Models (BERT Family)

**Best for:** Understanding and classifying text, not generating it.

```python
from transformers import pipeline

# Sentiment Analysis — understands the full sentence bidirectionally
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")

examples = [
    "I absolutely loved this movie, it was fantastic!",
    "The service was terrible and the food was cold.",
    "The movie was okay, nothing special.",
]

for text in examples:
    result = classifier(text)
    print(f"'{text[:40]}...' → {result[0]['label']} ({result[0]['score']:.2%})")

# Output:
# 'I absolutely loved this movie...' → POSITIVE (99.87%)
# 'The service was terrible...'      → NEGATIVE (99.92%)
# 'The movie was okay...'            → NEGATIVE (53.21%)  ← uncertain!
```

```python
# Named Entity Recognition — find people, places, organizations
ner = pipeline("ner",
               model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy="simple")

text = "Barack Obama was born in Honolulu, Hawaii, and served as the 44th President of the United States."
entities = ner(text)

for entity in entities:
    print(f"  {entity['word']:20} → {entity['entity_group']:5} (confidence: {entity['score']:.2%})")

# Output:
#   Barack Obama         → PER   (confidence: 99.82%)
#   Honolulu             → LOC   (confidence: 99.71%)
#   Hawaii               → LOC   (confidence: 98.45%)
#   United States        → LOC   (confidence: 99.54%)
```

### 4.2 Decoder-Only Models (GPT / LLaMA Family)

**Best for:** Generating text, chatting, coding, creative writing.

```python
from transformers import pipeline
import torch

# Text Generation with GPT-2 (small, downloadable without auth)
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0 if torch.cuda.is_available() else -1
)

prompt = "Once upon a time in a land far away, there was a"

output = generator(
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_p=0.9,
    num_return_sequences=2,   # Generate 2 different continuations
    do_sample=True,
    pad_token_id=50256        # GPT-2's EOS token ID
)

print("=== Continuation 1 ===")
print(output[0]["generated_text"])
print("\n=== Continuation 2 ===")
print(output[1]["generated_text"])
```

### 4.3 Encoder-Decoder Models (T5 / BART Family)

**Best for:** Translation, summarization, question answering with answers outside the context.

```python
from transformers import pipeline

# Summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
natural intelligence displayed by animals including humans. AI research has been defined 
as the field of study of intelligent agents, which refers to any system that perceives 
its environment and takes actions that maximize its chance of achieving its goals.
The term "artificial intelligence" had previously been used to describe machines that 
mimic and display human cognitive skills associated with the human mind, such as learning 
and problem-solving. This definition has since been rejected by major AI researchers who 
now describe AI in terms of rationality and acting rationally, which does not limit how 
intelligence can be articulated.
"""

summary = summarizer(long_text, max_length=60, min_length=20, do_sample=False)
print("SUMMARY:", summary[0]["summary_text"])

# Translation
translator = pipeline("translation_en_to_fr",
                       model="Helsinki-NLP/opus-mt-en-fr")

english_texts = [
    "Hello, how are you today?",
    "The weather is beautiful.",
    "I love learning about artificial intelligence.",
]

for text in english_texts:
    translation = translator(text)
    print(f"EN: {text}")
    print(f"FR: {translation[0]['translation_text']}\n")
```

---

## 5. Parameters, Model Sizes & What They Mean

### 5.1 Parameters Are the Model's "Weights"

```
A "parameter" is a single floating-point number that the model
learns during training. More parameters = more capacity to
store knowledge and recognize patterns.

7 billion parameters (7B model):
  7,000,000,000 numbers × 2 bytes (fp16) = 14 GB
  7,000,000,000 numbers × 4 bytes (fp32) = 28 GB
  7,000,000,000 numbers × 1 byte  (int8) = 7 GB
  7,000,000,000 numbers × 0.5 bytes(int4) = 3.5 GB
```

### 5.2 Model Size Calculator

```python
def calculate_model_memory(
    params_billions,
    precision="fp16",
    include_overhead=True
):
    """
    Calculate approximate VRAM needed to LOAD a model.
    
    Note: Training requires 3-10× more than inference!
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    bpp = bytes_per_param[precision]
    base_memory_gb = (params_billions * 1e9 * bpp) / (1024**3)
    
    # Add ~20% overhead for activations, KV cache, etc.
    if include_overhead:
        total_memory_gb = base_memory_gb * 1.2
    else:
        total_memory_gb = base_memory_gb
    
    print(f"\nModel: {params_billions}B parameters in {precision}")
    print(f"Base memory:  {base_memory_gb:.1f} GB")
    print(f"With overhead: {total_memory_gb:.1f} GB")
    
    # GPU recommendations
    print("\nGPU Options:")
    gpus = [
        ("RTX 3060", 12),
        ("RTX 3080/4070", 16),
        ("RTX 3090/4090", 24),
        ("A100 40GB", 40),
        ("A100 80GB", 80),
    ]
    for gpu_name, vram in gpus:
        fits = "✅ Fits" if vram >= total_memory_gb else "❌ Too small"
        print(f"  {gpu_name:20} ({vram:2}GB): {fits}")
    
    return total_memory_gb

# Examples
calculate_model_memory(7, "fp16")    # LLaMA 7B in FP16
calculate_model_memory(7, "int4")    # LLaMA 7B quantized
calculate_model_memory(13, "int4")   # LLaMA 13B quantized
calculate_model_memory(70, "int4")   # LLaMA 70B quantized
```

---

# PART II — HARDWARE

---

## 6. GPU Basics for AI

### 6.1 Why GPUs and Not CPUs?

```
CPU (Central Processing Unit):
  ┌────────────────────────────────────────┐
  │  Core 1  │  Core 2  │  Core 3  │ ...  │
  │ (Complex logic, handles 1 task well)   │
  │ 8-64 cores, ~3-5 GHz                  │
  │ Sequential operations: FAST            │
  │ Parallel operations: SLOW              │
  └────────────────────────────────────────┘

GPU (Graphics Processing Unit):
  ┌────────────────────────────────────────┐
  │  10,000+ simple cores                  │
  │  Designed for parallel math            │
  │  Each core: slower than CPU            │
  │  Together: 100-1000× faster for AI    │
  └────────────────────────────────────────┘

Matrix multiplication (the core of neural networks):
  CPU:  1 second  for a 4096×4096 matrix multiply
  GPU:  0.001 sec for the SAME operation (1000× faster)
```

### 6.2 GPU Specifications That Matter for AI

```python
# Run this to see your GPU specs
import subprocess

def get_gpu_info():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,compute_cap',
             '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                print(f"\n=== GPU {i} ===")
                print(f"  Name:            {parts[0]}")
                print(f"  Total VRAM:      {parts[1]}")
                print(f"  Free VRAM:       {parts[2]}")
                print(f"  Driver Version:  {parts[3]}")
                print(f"  Compute Capability: {parts[4]}")
        else:
            print("No NVIDIA GPU found (or nvidia-smi not installed)")
    except FileNotFoundError:
        print("nvidia-smi not found. Install NVIDIA drivers first.")

import torch
print(f"PyTorch version:    {torch.__version__}")
print(f"CUDA available:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:       {torch.version.cuda}")
    print(f"GPU count:          {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  VRAM:         {props.total_memory / 1e9:.1f} GB")
        print(f"  CUDA Cores:   {props.multi_processor_count * 128}")  # approx
        print(f"  Compute Cap:  {props.major}.{props.minor}")

get_gpu_info()
```

### 6.3 GPU Compute Capability — What It Means

| Compute Capability | GPU Generation | Key AI Feature |
|-------------------|---------------|----------------|
| 6.x | Pascal (GTX 1000) | FP16 (half precision) |
| 7.x | Volta/Turing (RTX 2000, Tesla V100) | Tensor Cores (fast matrix math) |
| 8.x | Ampere (RTX 3000, A100) | BF16, TF32, sparsity |
| 8.9 | Ada (RTX 4000) | FP8, improved transformer ops |
| 9.x | Hopper (H100) | FP8, transformer engine |

> **Beginner Rule:** Compute Capability ≥ 7.0 for serious AI work. Below that, you can train/infer but it'll be noticeably slower and some quantization methods won't work.

---

## 7. VRAM Requirements

### 7.1 The Master Table

| Model | Size | FP32 | FP16/BF16 | INT8 | INT4 (QLoRA) |
|-------|------|------|-----------|------|--------------|
| GPT-2 | 117M | 0.5 GB | 0.3 GB | 0.15 GB | 0.07 GB |
| BERT-base | 110M | 0.5 GB | 0.3 GB | 0.15 GB | — |
| DistilBERT | 66M | 0.3 GB | 0.2 GB | 0.1 GB | — |
| T5-base | 250M | 1 GB | 0.5 GB | 0.3 GB | — |
| LLaMA-2-7B | 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| LLaMA-2-13B | 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| Mistral-7B | 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| Gemma-2B | 2B | 8 GB | 4 GB | 2 GB | 1 GB |
| Gemma-7B | 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| LLaMA-3-8B | 8B | 32 GB | 16 GB | 8 GB | 4 GB |
| LLaMA-2-70B | 70B | 280 GB | 140 GB | 70 GB | 35 GB |

**Add 20-30% for inference overhead (KV cache, activations)**

### 7.2 What Happens When You Run Out of VRAM

```python
# This is what an OOM error looks like
# RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
# (GPU 0; 15.90 GiB total capacity; 13.50 GiB already allocated)

# Prevention strategy — check BEFORE loading
import torch

def can_fit_model(required_gb, safety_margin=0.15):
    if not torch.cuda.is_available():
        print("No GPU — will use CPU (slow)")
        return False
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    current_used = torch.cuda.memory_allocated(0) / 1e9
    available = total_vram - current_used
    needed = required_gb * (1 + safety_margin)
    
    print(f"Total VRAM:    {total_vram:.1f} GB")
    print(f"Currently used: {current_used:.1f} GB")
    print(f"Available:     {available:.1f} GB")
    print(f"Required:      {needed:.1f} GB (with {safety_margin:.0%} margin)")
    
    if available >= needed:
        print("✅ Should fit!")
        return True
    else:
        shortage = needed - available
        print(f"❌ Need {shortage:.1f} GB more. Options:")
        print("   1. Use INT8: load_in_8bit=True (halves memory)")
        print("   2. Use INT4: load_in_4bit=True (quarters memory)")
        print("   3. Use CPU offloading: device_map='auto'")
        print("   4. Use a smaller model")
        return False

# Check if LLaMA-7B FP16 will fit
can_fit_model(14)  # 14 GB for 7B in FP16
```

---

## 8. CPU vs GPU vs Cloud

### 8.1 Decision Matrix

```
                    CPU                 GPU (Local)         Cloud GPU
────────────────────────────────────────────────────────────────────
Cost per query:     Free                Free (after HW)     $0.001–0.1
Speed:              VERY SLOW           Fast–Very Fast      Fast
Privacy:            ✅ Full             ✅ Full             ❌ Data leaves
Setup complexity:   Easy                Medium              Easy
Model size limit:   RAM (32–128 GB)     VRAM (4–80 GB)      Unlimited
Good for:           Testing, <1B models 7B–70B models       Any size

RULES OF THUMB:
- Testing/prototyping: CPU is fine for BERT-size models
- Development with 7B models: You NEED a GPU
- Privacy-sensitive data: Local GPU only
- One-off large model inference: Cloud (Colab, vast.ai, RunPod)
- Production at scale: Cloud GPU cluster
```

### 8.2 Running on CPU (When You Must)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = "gpt2"  # Small model for CPU testing

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Force CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # CPU doesn't benefit from fp16
)
model.eval()

# Benchmark CPU speed
prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")

start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
elapsed = time.time() - start

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
tokens_per_second = 50 / elapsed

print(f"Generated: {generated}")
print(f"Time: {elapsed:.1f}s for 50 tokens")
print(f"Speed: {tokens_per_second:.1f} tokens/sec")
# CPU: ~1-5 tokens/sec for GPT-2
# GPU: ~50-200 tokens/sec for GPT-2
```

---

## 9. Setting Up CUDA

### 9.1 The CUDA Stack (What You're Actually Installing)

```
Your Python Code
      │
      ▼
  PyTorch (pip install torch)
      │ calls
      ▼
  CUDA Runtime (libcuda.so / cuda.dll)
      │ calls
      ▼
  cuDNN (Deep Neural Network library)
      │ calls
      ▼
  NVIDIA GPU Driver
      │ controls
      ▼
  Physical GPU Hardware
```

### 9.2 Step-by-Step CUDA Setup

```bash
# STEP 1: Install NVIDIA GPU Driver
# Download from: https://www.nvidia.com/drivers
# On Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-driver-535  # Use latest available version
sudo reboot

# Verify driver installation
nvidia-smi
# Should show GPU info and "CUDA Version: 12.x"

# STEP 2: Check which CUDA version your driver supports
# The number in nvidia-smi "CUDA Version" is the MAXIMUM you can use
# You can install PyTorch with any CUDA version UP TO that number

# STEP 3: Install PyTorch with correct CUDA
# Go to: https://pytorch.org/get-started/locally/
# Select your OS, CUDA version, etc.

# For CUDA 12.1 (most common in 2024)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# STEP 4: Verify everything works
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000).cuda()
    y = x @ x.T  # Matrix multiply on GPU
    print('GPU computation: OK', y.shape)
"
```

### 9.3 Common CUDA Installation Errors and Fixes

```bash
# ERROR: "CUDA driver version is insufficient for CUDA runtime version"
# CAUSE: Your driver is too old for the CUDA version in PyTorch
# FIX: Update NVIDIA driver OR install older PyTorch CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118  # Try older CUDA

# ERROR: "libcuda.so.1: cannot open shared object file"
# CAUSE: CUDA library path not set
# FIX:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Add this to ~/.bashrc to make permanent

# ERROR: torch.cuda.is_available() returns False even with GPU
# DEBUGGING STEPS:
nvidia-smi                           # Is the driver installed?
python -c "import torch; print(torch.version.cuda)"  # What CUDA version?
ls /usr/local/cuda                   # Is CUDA toolkit installed?
nvcc --version                       # Is NVCC available?

# WINDOWS SPECIFIC:
# Make sure to install "CUDA Toolkit" from developer.nvidia.com
# Restart after installation
# Check Device Manager → Display Adapters → NVIDIA GPU shows "working"
```

---

# PART III — OLLAMA

---

## 10. What Is Ollama?

### 10.1 The Problem Ollama Solves

Before Ollama, running an LLM locally looked like this:
```
1. Download model (complicated format)
2. Install dependencies (often conflicting)
3. Write inference code
4. Handle model loading, memory, quantization
5. Write a server if you want an API
6. Debug for days
```

With Ollama:
```bash
ollama run llama3
# That's it. The model downloads and runs.
```

### 10.2 What Ollama Does Under the Hood

```
┌─────────────────────────────────────────────────────────┐
│                      OLLAMA                             │
│                                                         │
│  Model Management:  Downloads, stores, versions models  │
│  Quantization:      Runs GGUF quantized models          │
│  Server:            Exposes OpenAI-compatible REST API  │
│  GPU Detection:     Auto-uses GPU if available          │
│  Memory Mgmt:       Handles model loading/unloading     │
│                                                         │
│  Underlying engine: llama.cpp (C++ inference engine)    │
└─────────────────────────────────────────────────────────┘
```

**GGUF Format:** Ollama uses the GGUF format (successor to GGML), which packages model weights + metadata into a single file and supports various quantization levels.

---

## 11. Installing Ollama

### 11.1 Linux Installation

```bash
# Official one-liner install
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Ollama runs as a system service on Linux
# Start/stop the service
sudo systemctl start ollama
sudo systemctl stop ollama
sudo systemctl status ollama
sudo systemctl enable ollama  # Auto-start on boot

# Check logs if something's wrong
journalctl -u ollama -f

# Manual start (for debugging, shows all output)
ollama serve
```

### 11.2 macOS Installation

```bash
# Download the installer from https://ollama.com/download
# Or via Homebrew:
brew install ollama

# Start Ollama (runs in background as a menu bar app)
ollama serve  # Or just open the app

# Ollama on macOS uses Metal (Apple Silicon GPU)
# M1/M2/M3 Macs run 7B models at ~30-60 tokens/sec
```

### 11.3 Windows Installation

```powershell
# Download installer from https://ollama.com/download/windows
# Run the installer (OllamaSetup.exe)
# Ollama starts automatically in the system tray

# Verify in PowerShell or Command Prompt
ollama --version
ollama list
```

### 11.4 Configure GPU Usage

```bash
# Ollama auto-detects GPU. To verify:
ollama run llama3 "Hello"
# Look for "GPU layers: XX" in the output — higher = more GPU usage

# Force specific GPU (multi-GPU systems)
CUDA_VISIBLE_DEVICES=0 ollama serve   # Use GPU 0 only

# Set GPU memory limit (in MB)
OLLAMA_GPU_MEMORY=8000 ollama serve   # Use max 8GB VRAM

# Force CPU-only (for testing)
OLLAMA_NOPRUNE=1 CUDA_VISIBLE_DEVICES="" ollama serve

# Environment variables for Ollama
# Add to /etc/systemd/system/ollama.service or ~/.bashrc
export OLLAMA_HOST=0.0.0.0:11434     # Allow external connections
export OLLAMA_MODELS=/custom/path     # Custom model storage location
export OLLAMA_NUM_PARALLEL=4          # Parallel request processing
export OLLAMA_MAX_LOADED_MODELS=2     # Keep 2 models loaded at once
```

---

## 12. Running Your First Model

### 12.1 Interactive Chat

```bash
# Pull and run LLaMA 3 (8B, recommended for 8GB+ VRAM)
ollama run llama3

# Pull and run Gemma (Google's model)
ollama run gemma:7b
ollama run gemma:2b    # Smaller, runs on 4GB VRAM

# Pull and run Mistral
ollama run mistral

# Run with specific quantization
ollama run llama3:8b-instruct-q4_0    # 4-bit quantized
ollama run llama3:8b-instruct-q8_0    # 8-bit quantized (better quality)
ollama run llama3:8b-instruct-fp16    # Full precision (needs 16GB+ VRAM)

# One-shot query (no interactive mode)
ollama run llama3 "What is the capital of France?"

# List all downloaded models
ollama list

# Remove a model
ollama rm llama3

# Show model info
ollama show llama3
```

### 12.2 Understanding Quantization Tags

```
Model name format: ollama run {name}:{size}-{variant}-{quantization}

Quantization options (quality vs size tradeoff):
  q2_K  → Most compressed, lowest quality, very fast
  q3_K  → 3-bit, poor quality, use only on tiny hardware
  q4_0  → 4-bit, good quality, most popular ← START HERE
  q4_K  → 4-bit with K-means, slightly better than q4_0
  q4_K_M → 4-bit K-means medium, best quality/size 4-bit
  q5_K  → 5-bit, very good quality, needs more VRAM
  q6_K  → 6-bit, excellent quality, approaches fp16
  q8_0  → 8-bit, near fp16 quality, 2× size of q4
  fp16  → Full half-precision, maximum quality, needs most VRAM

Rule: Use q4_K_M as your default. It's the sweet spot.
      Use q8_0 if you have VRAM to spare.
      Use q5_K_M for a quality boost with moderate VRAM.
```

---

## 13. Ollama Model Library

### 13.1 The Most Important Models

```bash
# ── GENERAL PURPOSE CHAT ──────────────────────────────────────

# LLaMA 3 (Meta) — Current best open-source for most tasks
ollama pull llama3                  # 8B — best all-rounder
ollama pull llama3:70b              # 70B — near-GPT4 quality (needs 48GB+ VRAM)

# Mistral (Mistral AI) — Fast, efficient, great for coding
ollama pull mistral                 # 7B
ollama pull mistral:7b-instruct     # Instruction-tuned

# Gemma (Google) — Efficient, great on smaller hardware
ollama pull gemma:2b                # Runs on 4GB VRAM (laptop GPU!)
ollama pull gemma:7b                # Better quality, needs 8GB

# Phi-3 (Microsoft) — Tiny but surprisingly capable
ollama pull phi3                    # 3.8B — great for 4GB VRAM

# ── CODE GENERATION ──────────────────────────────────────────

# CodeLlama (Meta) — Specialized for code
ollama pull codellama               # 7B general code
ollama pull codellama:34b           # 34B — very capable, needs 24GB+
ollama pull codellama:7b-code       # Pure code completion (no chat)
ollama pull codellama:7b-instruct   # Code with instructions

# DeepSeek Coder — Excellent coding model
ollama pull deepseek-coder:6.7b
ollama pull deepseek-coder:33b

# Starcoder2
ollama pull starcoder2:7b

# ── EMBEDDINGS ────────────────────────────────────────────────

# For RAG (Retrieval Augmented Generation)
ollama pull nomic-embed-text        # General embeddings
ollama pull mxbai-embed-large       # High quality embeddings

# ── SPECIALIZED ───────────────────────────────────────────────

# Medical
ollama pull medllama2               # Medical fine-tune
# Math
ollama pull mathstral               # Math reasoning
# Long context
ollama pull llama3:8b-instruct-q4_0  # 8K context window
```

### 13.2 Comparing Models Side-by-Side

```bash
# Test the same prompt on multiple models
for model in llama3 mistral gemma:7b; do
    echo "=== $model ==="
    ollama run $model "Explain quantum entanglement in one sentence." --verbose 2>/dev/null
    echo ""
done
```

---

## 14. Ollama REST API

### 14.1 The API Basics

Ollama runs a local server at `http://localhost:11434` with an OpenAI-compatible API. This is the key feature for integration.

```python
import requests
import json

# ── BASIC GENERATION ─────────────────────────────────────────

def ollama_generate(prompt, model="llama3", stream=False):
    """Simple generation with Ollama API."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 256,  # Max tokens to generate
                "num_ctx": 4096,     # Context window size
            }
        }
    )
    
    if not stream:
        result = response.json()
        return result["response"]
    else:
        # Handle streaming response
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                full_response += chunk.get("response", "")
                if chunk.get("done"):
                    break
        return full_response

# Usage
answer = ollama_generate("What is a transformer neural network?")
print(answer)
```

```python
# ── CHAT API (OpenAI Compatible) ──────────────────────────────

def ollama_chat(messages, model="llama3", stream=False):
    """
    Chat with conversation history.
    messages format: [{"role": "user/assistant/system", "content": "..."}]
    """
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "num_ctx": 4096,
            }
        }
    )
    
    if not stream:
        result = response.json()
        return result["message"]["content"]
    else:
        full_content = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk:
                    full_content += chunk["message"].get("content", "")
                if chunk.get("done"):
                    break
        return full_content

# Multi-turn conversation
conversation = [
    {"role": "system", "content": "You are a helpful Python tutor. Be concise."},
    {"role": "user",   "content": "What is a list comprehension?"},
]

response1 = ollama_chat(conversation)
print("Assistant:", response1)

# Continue the conversation
conversation.append({"role": "assistant", "content": response1})
conversation.append({"role": "user", "content": "Give me an example."})

response2 = ollama_chat(conversation)
print("Assistant:", response2)
```

```python
# ── STREAMING WITH REAL-TIME OUTPUT ──────────────────────────

def ollama_stream_chat(messages, model="llama3"):
    """Stream response tokens as they're generated."""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True
    )
    
    print("Assistant: ", end="", flush=True)
    full_response = ""
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if "message" in chunk:
                token = chunk["message"].get("content", "")
                print(token, end="", flush=True)  # Print each token as it arrives
                full_response += token
            if chunk.get("done"):
                print()  # New line when done
                # Print timing stats
                if "eval_count" in chunk:
                    tokens = chunk["eval_count"]
                    duration = chunk["eval_duration"] / 1e9  # ns to seconds
                    print(f"\n[{tokens} tokens | {tokens/duration:.1f} tok/sec]")
                break
    
    return full_response

# Usage
ollama_stream_chat([{"role": "user", "content": "Write a haiku about Python."}])
```

```python
# ── EMBEDDINGS ────────────────────────────────────────────────

def get_embeddings(texts, model="nomic-embed-text"):
    """Get vector embeddings for a list of texts."""
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = []
    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text}
        )
        embedding = response.json()["embedding"]
        embeddings.append(embedding)
    
    return embeddings

# Semantic similarity
import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

texts = [
    "The cat sat on the mat",
    "A feline rested on a rug",  # Semantically similar
    "JavaScript is a programming language",  # Unrelated
]

embeddings = get_embeddings(texts)

print("Similarity between sentence 1 and 2:", 
      cosine_similarity(embeddings[0], embeddings[1]))  # High ~0.9
print("Similarity between sentence 1 and 3:", 
      cosine_similarity(embeddings[0], embeddings[2]))  # Low ~0.2
```

```python
# ── USING OPENAI SDK WITH OLLAMA (Drop-in Replacement) ────────

# pip install openai
from openai import OpenAI

# Point to local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Required but not used by Ollama
)

# Now use exactly the same code as OpenAI API!
response = client.chat.completions.create(
    model="llama3",  # Your local Ollama model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ],
    temperature=0.7,
    max_tokens=200,
)

print(response.choices[0].message.content)

# Streaming with OpenAI SDK
stream = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Count from 1 to 10 slowly."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 15. Ollama + LangChain & Custom Pipelines

```python
# pip install langchain langchain-community

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_core.output_parsers import StrOutputParser

# ── BASIC LLM CHAIN ──────────────────────────────────────────

llm = Ollama(model="llama3", temperature=0.7)

# Simple chain
prompt = PromptTemplate.from_template(
    "You are an expert in {topic}. Answer this question: {question}"
)
chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "topic": "Python programming",
    "question": "What is the difference between a list and a tuple?"
})
print(response)

# ── CHAT MODEL CHAIN ─────────────────────────────────────────

chat_model = ChatOllama(model="llama3")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Be {style}."),
    ("human", "{question}")
])

chat_chain = chat_prompt | chat_model | StrOutputParser()

response = chat_chain.invoke({
    "role": "pirate",
    "style": "dramatic and theatrical",
    "question": "What is the meaning of life?"
})
print(response)
```

```python
# ── RAG (Retrieval Augmented Generation) WITH OLLAMA ─────────

# pip install langchain faiss-cpu pypdf

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Step 1: Load your documents
# loader = PyPDFLoader("your_document.pdf")
# OR create test data:
from langchain.schema import Document

documents = [
    Document(page_content="Python was created by Guido van Rossum in 1991.", metadata={"source": "python_facts"}),
    Document(page_content="Python is named after Monty Python, not the snake.", metadata={"source": "python_facts"}),
    Document(page_content="Python uses indentation for code blocks instead of curly braces.", metadata={"source": "python_facts"}),
    Document(page_content="The latest Python version as of 2024 is Python 3.12.", metadata={"source": "python_facts"}),
]

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(splits, embeddings)

# Step 4: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 5: Build RAG chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context:

Context:
{context}

Question: {question}

If the answer is not in the context, say "I don't have information about that."
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | ChatOllama(model="llama3")
    | StrOutputParser()
)

# Ask questions
questions = [
    "Who created Python?",
    "Why is Python named Python?",
    "What year was Python released?",
    "What is Python's market share?",  # Not in our documents
]

for question in questions:
    print(f"Q: {question}")
    answer = rag_chain.invoke(question)
    print(f"A: {answer}\n")
```

---

## 16. Modelfiles — Customizing Models in Ollama

A Modelfile is like a Dockerfile — it defines how a model behaves in Ollama.

```dockerfile
# ── BASIC MODELFILE ──────────────────────────────────────────
# Save as: Modelfile

FROM llama3

# System prompt — defines the model's personality/role
SYSTEM """
You are Alex, a friendly and knowledgeable Python tutor with 10 years of experience.
You explain concepts clearly with concrete examples.
You always check if the student understood before moving on.
Never give answers without explanation — teaching is more important than speed.
"""

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096       # Context window
PARAMETER num_predict 512    # Max response length
PARAMETER repeat_penalty 1.1

# Template format (how prompts are structured)
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""
```

```bash
# Build and run your custom model
ollama create python-tutor -f Modelfile
ollama run python-tutor

# Now "Alex" will always respond as a Python tutor
# You can share this model with others
```

```dockerfile
# ── ADVANCED MODELFILE: Domain Expert ────────────────────────
# Save as: MedicalModelfile

FROM mistral

SYSTEM """
You are a medical information assistant. You provide accurate health information 
to help people understand medical topics. 

Important guidelines:
- Always recommend consulting a doctor for personal medical advice
- Cite general medical consensus, not personal opinions
- Use clear, accessible language for non-medical professionals
- Never diagnose conditions or prescribe treatments
"""

PARAMETER temperature 0.3    # Lower temperature = more consistent/factual
PARAMETER top_p 0.8
PARAMETER num_ctx 8192       # Larger context for detailed medical questions
```

```python
# Using custom models from Python
import requests

# Your custom model is just another model name
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "python-tutor",  # Your custom Modelfile model
        "messages": [
            {"role": "user", "content": "Explain list comprehensions"}
        ]
    }
)
print(response.json()["message"]["content"])
```

---

## 17. Ollama Performance Tuning

### 17.1 Benchmarking Your Setup

```python
import requests
import json
import time

def benchmark_ollama(model, prompt, num_runs=3):
    """Benchmark a model's generation speed."""
    results = []
    
    for i in range(num_runs):
        start = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 100}  # Fixed output length
            }
        )
        elapsed = time.time() - start
        data = response.json()
        
        metrics = {
            "total_time": elapsed,
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0),
            "prompt_eval_duration_ms": data.get("prompt_eval_duration", 0) / 1e6,
            "eval_duration_ms": data.get("eval_duration", 0) / 1e6,
        }
        
        # Calculate speeds
        if metrics["eval_duration_ms"] > 0:
            metrics["tokens_per_sec"] = (
                metrics["eval_count"] / (metrics["eval_duration_ms"] / 1000)
            )
        
        results.append(metrics)
        print(f"Run {i+1}: {metrics.get('tokens_per_sec', 0):.1f} tok/sec")
    
    avg_speed = sum(r.get("tokens_per_sec", 0) for r in results) / len(results)
    print(f"\nAverage: {avg_speed:.1f} tokens/second")
    return results

benchmark_ollama("llama3", "Tell me about machine learning.")
```

### 17.2 Performance Optimization Settings

```bash
# Set number of GPU layers (higher = more GPU, less CPU)
# -1 means ALL layers on GPU (default if VRAM allows)
OLLAMA_GPU_LAYERS=32 ollama serve

# Parallel requests (serve multiple users at once)
OLLAMA_NUM_PARALLEL=2 ollama serve

# Keep model in memory (don't unload between requests)
# Set to -1 for indefinite, or "5m" for 5 minutes
OLLAMA_KEEP_ALIVE=5m ollama serve

# Thread count for CPU inference
OLLAMA_NUM_THREAD=8 ollama serve  # Set to number of physical cores

# Context size (affects KV cache memory)
# Set in API call:
requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3",
    "prompt": "...",
    "options": {
        "num_ctx": 4096,    # Default. Reduce to 2048 to save memory
        "num_batch": 512,   # Prompt processing batch size (higher=faster prompt processing)
        "num_gpu": 1,       # Number of GPUs to use
        "num_thread": 8,    # CPU threads
        "use_mlock": True,  # Lock model in RAM (prevents swapping)
        "use_mmap": True,   # Memory-map the model file (faster loading)
        "f16_kv": True,     # FP16 KV cache (saves memory)
    }
})
```

# PART IV — TRANSFORMERS FOR BEGINNERS

---

## 18. Your First HuggingFace Model in 10 Lines

### 18.1 The Absolute Minimum

```python
# Step 1: Install
# pip install transformers torch

# Step 2: Your first model — 10 lines total
from transformers import pipeline

# Load a pre-built pipeline (this downloads the model automatically)
sentiment_analyzer = pipeline("sentiment-analysis")

# Run it
texts = [
    "I love this so much!",
    "This is absolutely terrible.",
    "It was fine, nothing special.",
]

for text in texts:
    result = sentiment_analyzer(text)
    print(f"Text:  {text}")
    print(f"Label: {result[0]['label']}, Confidence: {result[0]['score']:.1%}")
    print()
```

### 18.2 What Just Happened?

```python
# Let's slow down and understand each step
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# When you run pipeline("sentiment-analysis"), here's what actually happens:

# 1. It picks a default model (distilbert-base-uncased-finetuned-sst-2-english)
# 2. Downloads the model weights (~260MB) to ~/.cache/huggingface
# 3. Downloads the tokenizer files
# 4. Loads everything into memory

# Here's the expanded, explicit version:
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Tokenizer: converts text to numbers the model understands
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Model: the neural network with pre-trained weights
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Let's trace one example through manually
import torch

text = "I love transformers!"

# STEP 1: Tokenize
tokens = tokenizer(text, return_tensors="pt")
print("Token IDs:", tokens["input_ids"])
print("Decoded:", [tokenizer.decode([id]) for id in tokens["input_ids"][0]])

# STEP 2: Forward pass through model
with torch.no_grad():  # No gradient computation needed for inference
    outputs = model(**tokens)
    
print("Raw logits:", outputs.logits)  # tensor([[-3.4, 3.6]])

# STEP 3: Convert logits to probabilities
import torch.nn.functional as F
probabilities = F.softmax(outputs.logits, dim=-1)
print("Probabilities:", probabilities)  # tensor([[0.003, 0.997]])

# STEP 4: Get the label
predicted_class = probabilities.argmax().item()
label = model.config.id2label[predicted_class]
confidence = probabilities[0][predicted_class].item()

print(f"Prediction: {label} ({confidence:.1%})")
```

---

## 19. Understanding the Pipeline API Deeply

### 19.1 Every Major Pipeline with Examples

```python
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

# ── 1. SENTIMENT ANALYSIS ────────────────────────────────────
print("=" * 60)
print("SENTIMENT ANALYSIS")
print("=" * 60)

sentiment = pipeline("sentiment-analysis",
                     model="distilbert-base-uncased-finetuned-sst-2-english",
                     device=device)

reviews = [
    "Best pizza I've ever had. Will definitely come back!",
    "Waited 45 minutes for lukewarm food. Never again.",
    "The food was okay. Nothing extraordinary.",
]

for review in reviews:
    result = sentiment(review)
    icon = "👍" if result[0]["label"] == "POSITIVE" else "👎"
    print(f"{icon} {result[0]['label']:8} ({result[0]['score']:.1%}) | {review[:50]}")
```

```python
# ── 2. ZERO-SHOT CLASSIFICATION (No Training Required!) ──────
print("\nZERO-SHOT CLASSIFICATION")
print("=" * 60)

zero_shot = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli",
                     device=device)

emails = [
    "Hi, I need help resetting my password for the account.",
    "URGENT: Your account has been compromised! Click here now!",
    "Monthly newsletter: Top tech news from October 2024",
    "Please find attached the invoice for services rendered.",
]

labels = ["technical support", "spam/phishing", "newsletter", "billing", "general inquiry"]

for email in emails:
    result = zero_shot(email, candidate_labels=labels, multi_label=False)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    print(f"\nEmail: {email[:60]}...")
    print(f"Category: {top_label} ({top_score:.1%})")
    # Show all scores
    for label, score in zip(result["labels"][:3], result["scores"][:3]):
        bar = "█" * int(score * 20)
        print(f"  {label:20} {score:.1%} {bar}")
```

```python
# ── 3. NAMED ENTITY RECOGNITION ──────────────────────────────
print("\nNAMED ENTITY RECOGNITION")
print("=" * 60)

ner = pipeline("ner",
               model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy="simple",
               device=device)

texts = [
    "Elon Musk founded SpaceX in Hawthorne, California in 2002.",
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino.",
    "The Eiffel Tower in Paris, France attracts millions of visitors annually.",
]

entity_icons = {"PER": "👤", "ORG": "🏢", "LOC": "📍", "MISC": "🔷"}

for text in texts:
    print(f"\nText: {text}")
    entities = ner(text)
    for e in entities:
        icon = entity_icons.get(e["entity_group"], "❓")
        print(f"  {icon} {e['word']:20} → {e['entity_group']} ({e['score']:.1%})")
```

```python
# ── 4. QUESTION ANSWERING ─────────────────────────────────────
print("\nQUESTION ANSWERING")
print("=" * 60)

qa = pipeline("question-answering",
              model="deepset/roberta-base-squad2",
              device=device)

context = """
Python is a high-level, general-purpose programming language. Its design philosophy 
emphasizes code readability. Python was created by Guido van Rossum and first released 
in 1991. Python consistently ranks as one of the most popular programming languages.
Python 3.0 was released in December 2008. The latest version, Python 3.12, was released 
in October 2023. Python is widely used in data science, machine learning, and web development.
"""

questions = [
    "Who created Python?",
    "When was Python first released?",
    "When was Python 3.0 released?",
    "What is Python used for?",
]

for question in questions:
    result = qa(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']} (confidence: {result['score']:.1%})")
    print()
```

```python
# ── 5. TEXT GENERATION ────────────────────────────────────────
print("TEXT GENERATION")
print("=" * 60)

generator = pipeline("text-generation",
                     model="gpt2",
                     device=device)

prompts = [
    "The most important thing about machine learning is",
    "Once upon a time, in a world powered by AI,",
    "def calculate_fibonacci(n):",
]

for prompt in prompts:
    result = generator(
        prompt,
        max_new_tokens=60,
        temperature=0.8,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=50256
    )
    generated = result[0]["generated_text"]
    new_text = generated[len(prompt):]  # Only show new tokens
    print(f"Prompt: '{prompt}'")
    print(f"Generated: {new_text}")
    print()
```

```python
# ── 6. SUMMARIZATION ─────────────────────────────────────────
print("SUMMARIZATION")
print("=" * 60)

summarizer = pipeline("summarization",
                      model="sshleifer/distilbart-cnn-12-6",
                      device=device)

article = """
The James Webb Space Telescope (JWST) has revolutionized our understanding of the 
early universe. Since its launch in December 2021 and becoming fully operational in 
2022, it has captured images of galaxies from the universe's infancy, just a few 
hundred million years after the Big Bang. The telescope's infrared capabilities 
allow it to see through cosmic dust clouds that were opaque to its predecessor, 
the Hubble Space Telescope. Scientists have used Webb to study exoplanet atmospheres, 
searching for signs of water vapor and carbon dioxide — potential indicators of 
habitable conditions. The telescope has also provided unprecedented detail of 
nebulae, the stellar nurseries where new stars are born. One of its most remarkable 
achievements has been confirming theoretical predictions about galaxy formation while 
simultaneously revealing unexpected structures that challenge existing models.
"""

summary = summarizer(article, max_length=80, min_length=30, do_sample=False)
print("Original length:", len(article.split()), "words")
print("Summary:", summary[0]["summary_text"])
print("Summary length:", len(summary[0]["summary_text"].split()), "words")
```

---

## 20. Tokenization — Step by Step With Real Examples

### 20.1 The Full Tokenization Process

```python
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Use BERT's tokenizer for this exploration
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("=" * 60)
print("DEEP DIVE: BERT TOKENIZER")
print("=" * 60)

# ── How WordPiece Tokenization Works ────────────────────────
words_to_explore = [
    "cat",            # Simple, common word — likely 1 token
    "cats",           # Plural — might be 2 tokens (cat + ##s)
    "running",        # Gerund
    "unfortunately",  # Long word — multiple tokens
    "transformer",    # Technical word
    "transformers",   # Plural of technical word
    "BERT",           # Acronym — case matters?
    "bert",           # Lowercase version
    "3.14159",        # Number
    "hello@world.com",# Email address
    "C++",            # Programming term
    "नमस्ते",          # Hindi — watch the token explosion!
]

print(f"\n{'Word':<20} {'Tokens':<40} {'Count'}")
print("-" * 70)

for word in words_to_explore:
    tokens = tokenizer.tokenize(word)
    print(f"{word:<20} {str(tokens):<40} {len(tokens)}")
```

```python
# ── Understanding ## (continuation tokens) ───────────────────
print("\n" + "=" * 60)
print("UNDERSTANDING ## TOKENS")
print("=" * 60)

text = "The transformer architecture revolutionized natural language processing."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"\nText: '{text}'")
print(f"\nTokens ({len(tokens)}):")
for i, (token, tid) in enumerate(zip(tokens, token_ids[1:-1])):  # Skip CLS/SEP
    prefix = "├─" if i < len(tokens)-1 else "└─"
    note = " ← continuation of previous" if token.startswith("##") else ""
    print(f"  {prefix} [{tid:6}] '{token}'{note}")

print(f"\n[CLS] = {token_ids[0]}  (start token)")
print(f"[SEP] = {token_ids[-1]}  (end token)")
```

```python
# ── Padding and Truncation Visualized ────────────────────────
print("\n" + "=" * 60)
print("PADDING AND TRUNCATION")
print("=" * 60)

sentences = [
    "Hi!",
    "The quick brown fox.",
    "Artificial intelligence is transforming every industry and changing how humans work, live, and interact with technology.",
]

# Without padding (variable length)
print("\nWithout padding:")
for sent in sentences:
    ids = tokenizer.encode(sent)
    print(f"  '{sent[:30]}...' → {len(ids)} tokens")

# With padding to longest
print("\nWith padding='longest':")
encoded = tokenizer(sentences, padding=True, truncation=True, max_length=30)
for i, sent in enumerate(sentences):
    ids = encoded["input_ids"][i]
    mask = encoded["attention_mask"][i]
    pad_count = ids.count(0)
    real_count = sum(mask)
    print(f"  Sentence {i+1}: {len(ids)} total, {real_count} real, {pad_count} padding")

# Attention mask explanation
print("\nAttention Mask Explained:")
print("  1 = Real token (model pays attention)")
print("  0 = Padding token (model IGNORES)")
sample = encoded["attention_mask"][0]
print(f"  Example: {sample}")
```

---

## 21. Running LLaMA & Gemma with Transformers

### 21.1 Running Gemma (No Auth Needed for Some Versions)

```python
# Gemma 2B — Works on 6GB VRAM, good for learning
# Note: You need to accept terms on HuggingFace for Gemma
# Visit: https://huggingface.co/google/gemma-2b-it

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Option A: Gemma in 4-bit (works on 4GB VRAM) ─────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "google/gemma-2b-it"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model (this may take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.eval()
print(f"Model loaded! Device: {next(model.parameters()).device}")

# ── Chat Function ────────────────────────────────────────────
def chat_with_gemma(user_message, history=None, max_new_tokens=256):
    """
    Chat with Gemma using its native format.
    history: list of (user, assistant) tuples
    """
    if history is None:
        history = []
    
    # Build conversation using Gemma's chat template
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_message})
    
    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only new tokens
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response.strip()

# ── Test It ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("CHATTING WITH GEMMA")
print("=" * 50)

# Single turn
response = chat_with_gemma("What is gradient descent in simple terms?")
print(f"User: What is gradient descent in simple terms?")
print(f"Gemma: {response}")

# Multi-turn conversation
print("\n--- Multi-turn conversation ---")
history = []
turns = [
    "What is a neural network?",
    "How does it learn?",
    "Can you give a real-world example?",
]

for user_msg in turns:
    response = chat_with_gemma(user_msg, history)
    history.append((user_msg, response))
    print(f"\nUser: {user_msg}")
    print(f"Gemma: {response}")
```

### 21.2 Running LLaMA 3 with Transformers

```python
# LLaMA 3 requires HuggingFace token
# 1. Create account at huggingface.co
# 2. Accept Meta's license at huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# 3. Create token at huggingface.co/settings/tokens
# 4. Run: huggingface-cli login

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load in 4-bit for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

model.eval()

# TextStreamer for real-time output
streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

def chat_llama3(
    user_message,
    system_prompt="You are a helpful assistant.",
    max_new_tokens=512,
    temperature=0.7,
    stream=True
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    
    # LLaMA 3 specific chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        pad_token_id=tokenizer.eos_token_id,
    )
    
    if stream:
        generate_kwargs["streamer"] = streamer
    
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    
    if not stream:
        new_tokens = outputs[0][input_length:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

# Test with different system prompts
print("=== LLaMA 3 as Python Expert ===")
chat_llama3(
    user_message="Write a function to find all prime numbers up to n.",
    system_prompt="You are an expert Python developer. Write clean, commented, efficient code.",
)

print("\n=== LLaMA 3 as Teacher ===")
chat_llama3(
    user_message="What is attention in transformers?",
    system_prompt="You are a patient teacher explaining ML to a complete beginner. Use analogies.",
)
```

---

## 22. Prompt Engineering for Beginners

### 22.1 The Anatomy of a Good Prompt

```python
from transformers import pipeline
import torch

# We'll use a smaller model for these examples
generator = pipeline("text-generation", model="gpt2", device=-1)

def generate(prompt, max_tokens=150):
    result = generator(prompt, max_new_tokens=max_tokens, 
                      temperature=0.7, do_sample=True,
                      pad_token_id=50256)
    return result[0]["generated_text"][len(prompt):]

print("=" * 60)
print("PROMPT ENGINEERING TECHNIQUES")
print("=" * 60)

# ── Technique 1: Zero-Shot ────────────────────────────────────
print("\n1. ZERO-SHOT (just ask)")
print("-" * 40)
prompt = "Classify the sentiment of this review: 'The food was amazing, best restaurant in town!'\nSentiment:"
print(f"Prompt: {prompt}")
print(f"Output: {generate(prompt, 20)}")

# ── Technique 2: Few-Shot ─────────────────────────────────────
print("\n2. FEW-SHOT (show examples)")
print("-" * 40)
prompt = """Classify sentiment as POSITIVE or NEGATIVE.

Text: "I loved this movie!"
Sentiment: POSITIVE

Text: "Terrible waste of time."
Sentiment: NEGATIVE

Text: "Best meal I've had in years!"
Sentiment:"""
print(f"Output: {generate(prompt, 10)}")

# ── Technique 3: Chain of Thought ─────────────────────────────
print("\n3. CHAIN OF THOUGHT (think step by step)")
print("-" * 40)
prompt = """Solve this problem step by step:
If a train travels at 60 mph for 2.5 hours, how far does it travel?

Step 1: Identify what we know
- Speed = 60 mph
- Time = 2.5 hours

Step 2: Apply the formula
Distance = Speed × Time
Distance ="""
print(f"Output: {generate(prompt, 30)}")

# ── Technique 4: Role Assignment ─────────────────────────────
print("\n4. ROLE ASSIGNMENT (become an expert)")
print("-" * 40)
prompt = """You are a senior software engineer with 15 years of Python experience.
A junior developer asks: "Should I use a list or a tuple?"
Your expert answer:"""
print(f"Output: {generate(prompt, 100)}")
```

### 22.2 Prompt Templates for Common Tasks

```python
# These are ready-to-use templates for Ollama/LLaMA/Gemma

PROMPT_TEMPLATES = {
    
    "summarize": """Please summarize the following text in {num_sentences} sentences.
Focus on the main points and key takeaways.

Text to summarize:
{text}

Summary:""",

    "classify": """Classify the following text into exactly one of these categories: {categories}

Rules:
- Output only the category name, nothing else
- If unsure, pick the closest match

Text: {text}
Category:""",

    "extract_info": """Extract the following information from the text below.
If a piece of information is not found, write "Not found".

Information to extract:
{fields}

Text:
{text}

Extracted Information:""",

    "rewrite": """Rewrite the following text to be {style}.
Keep the same meaning but change the tone and language.

Original text:
{text}

Rewritten version:""",

    "code_explain": """You are an expert programmer. Explain this code to a {audience}.
Be clear and use analogies where helpful.

Code:
```{language}
{code}
```

Explanation:""",

    "qa": """Answer the following question using ONLY the information provided in the context.
If the answer cannot be found in the context, say "I cannot answer based on the provided information."

Context:
{context}

Question: {question}

Answer:""",

    "brainstorm": """Generate {num_ideas} creative ideas for {topic}.
Format each idea as a numbered list.
Be specific and practical.

Ideas:""",
}

# Usage example with Ollama
def fill_template(template_name, **kwargs):
    template = PROMPT_TEMPLATES[template_name]
    return template.format(**kwargs)

# Example 1: Summarize
prompt = fill_template(
    "summarize",
    num_sentences=2,
    text="Artificial intelligence is changing every industry. Machine learning models can now process images, text, and audio. Companies are investing billions in AI research. The technology is becoming more accessible to developers and businesses of all sizes."
)
print("SUMMARIZE PROMPT:")
print(prompt)
print()

# Example 2: Extract info
prompt = fill_template(
    "extract_info",
    fields="- Person name\n- Company\n- Date\n- Location",
    text="John Smith, CEO of TechCorp, announced yesterday at their San Francisco headquarters that they will release a new product next quarter."
)
print("EXTRACT INFO PROMPT:")
print(prompt)
```

---

## 23. Building Your First Fine-Tuned Classifier

### 23.1 Complete Beginner Fine-Tuning Example

```python
"""
COMPLETE EXAMPLE: Fine-tune DistilBERT for Email Classification
Task: Classify emails as "spam" or "not_spam"
Dataset: We'll create a small example dataset

This is a complete, runnable example that teaches every step.
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import evaluate

# ── STEP 1: CREATE YOUR DATASET ──────────────────────────────
print("STEP 1: Creating dataset...")

# In real life, you'd load this from a CSV file
# For learning, we create it directly in Python
spam_emails = [
    "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize NOW!",
    "URGENT: Your account has been compromised. Verify immediately at fake-bank.com",
    "FREE iPhone! Limited time offer. Just pay shipping. Act now!!!",
    "You have been selected for a special offer. Reply with your SSN to claim.",
    "MAKE MONEY FAST! Work from home. $5000/week guaranteed. No experience needed!",
    "Dear Winner, Your email address won our lottery. Send $50 processing fee.",
    "HOT SINGLES in your area want to meet you! Click here!",
    "Lose 30 pounds in 30 days with this ONE WEIRD TRICK doctors don't want you to know!",
    "FINAL WARNING: Your computer has a virus. Call 1-800-FAKE now!",
    "You qualify for a pre-approved loan of $50,000. No credit check!",
    "Re: Your Order - There's a problem. Update payment info immediately.",
    "Earn $500 daily just by clicking links. Guaranteed income!",
]

not_spam_emails = [
    "Hi, just following up on our meeting from yesterday. Can we schedule a call this week?",
    "Your package has been shipped and will arrive by Thursday.",
    "Please find attached the Q3 financial report for your review.",
    "Don't forget — team standup is at 10am tomorrow.",
    "Thanks for your purchase! Your order #12345 is confirmed.",
    "Reminder: Your dentist appointment is on Friday at 2pm.",
    "The project proposal looks great. I have a few minor suggestions.",
    "Monthly newsletter: New features and updates for October.",
    "Hi there! I saw your profile and would love to connect about potential opportunities.",
    "Your subscription renewal is coming up. Review your plan at account.example.com.",
    "Happy Birthday! The team wanted to wish you a wonderful day.",
    "Meeting notes from yesterday's product review are now available in Confluence.",
]

# Create labels (0 = not_spam, 1 = spam)
texts  = not_spam_emails + spam_emails
labels = [0] * len(not_spam_emails) + [1] * len(spam_emails)

# Create HuggingFace Dataset
data = {"text": texts, "label": labels}
dataset = Dataset.from_dict(data)

# Split into train and test
split = dataset.train_test_split(test_size=0.25, seed=42, stratify_by_column="label")
train_dataset = split["train"]
test_dataset  = split["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Test size:  {len(test_dataset)}")
print(f"Label distribution: {sum(labels)} spam, {len(labels)-sum(labels)} not_spam")

# ── STEP 2: LOAD TOKENIZER ────────────────────────────────────
print("\nSTEP 2: Loading tokenizer...")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_examples(examples):
    """Tokenize a batch of examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        # No padding here — DataCollatorWithPadding handles it
    )

# Apply tokenization to entire dataset
print("Tokenizing dataset...")
tokenized_train = train_dataset.map(tokenize_examples, batched=True)
tokenized_test  = test_dataset.map(tokenize_examples, batched=True)

print(f"Features after tokenization: {tokenized_train.column_names}")

# ── STEP 3: LOAD MODEL ────────────────────────────────────────
print("\nSTEP 3: Loading model...")

id2label = {0: "NOT_SPAM", 1: "SPAM"}
label2id = {"NOT_SPAM": 0, "SPAM": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ── STEP 4: DEFINE METRICS ────────────────────────────────────
print("\nSTEP 4: Setting up metrics...")

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(
        predictions=predictions, references=labels
    )["accuracy"]
    
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="binary"
    )["f1"]
    
    return {"accuracy": accuracy, "f1": f1}

# ── STEP 5: TRAINING ARGUMENTS ────────────────────────────────
print("\nSTEP 5: Configuring training...")

training_args = TrainingArguments(
    output_dir="./results/spam-classifier",
    
    # How long to train
    num_train_epochs=5,
    
    # Batch sizes (reduce if out of memory)
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    
    # Learning rate schedule
    learning_rate=2e-5,        # Typical for BERT fine-tuning
    warmup_steps=50,           # Gradually increase LR at start
    weight_decay=0.01,         # L2 regularization
    
    # Evaluation settings
    eval_strategy="epoch",     # Evaluate at end of each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    
    # Logging
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",           # Disable wandb/tensorboard for this example
    
    # Speed optimization
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
)

# ── STEP 6: CREATE TRAINER ────────────────────────────────────
print("\nSTEP 6: Creating Trainer...")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ── STEP 7: TRAIN ─────────────────────────────────────────────
print("\nSTEP 7: Training...")
print("(This may take 1-5 minutes depending on your hardware)\n")

train_result = trainer.train()

print(f"\nTraining complete!")
print(f"Training time: {train_result.metrics['train_runtime']:.1f} seconds")

# ── STEP 8: EVALUATE ──────────────────────────────────────────
print("\nSTEP 8: Evaluating on test set...")

eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.1%}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")

# ── STEP 9: TEST ON NEW EXAMPLES ──────────────────────────────
print("\nSTEP 9: Testing on new examples...")

new_emails = [
    "Hi John, can we reschedule our meeting to 3pm tomorrow?",
    "AMAZING DEAL! Buy 2 get 5 FREE! Limited time only! Click NOW!",
    "Your Amazon order has been delivered. Rate your experience.",
    "You've been pre-selected for a credit card with 0% interest!!!",
    "The report you requested is attached. Let me know if you need anything else.",
]

# Make predictions
from transformers import pipeline as hf_pipeline

classifier = hf_pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

print(f"\n{'Email':<50} {'Label':10} {'Confidence'}")
print("-" * 75)
for email in new_emails:
    result = classifier(email)[0]
    icon = "🚫" if result["label"] == "SPAM" else "✅"
    print(f"{icon} {email[:48]:<48} {result['label']:10} {result['score']:.1%}")

# ── STEP 10: SAVE THE MODEL ───────────────────────────────────
print("\nSTEP 10: Saving model...")

save_path = "./models/spam-classifier"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model saved to {save_path}")
print("Files created:")
import os
for f in os.listdir(save_path):
    size = os.path.getsize(os.path.join(save_path, f)) / 1e6
    print(f"  {f:30} {size:.1f} MB")

# ── STEP 11: RELOAD AND USE OFFLINE ───────────────────────────
print("\nSTEP 11: Reloading model from disk...")

loaded_classifier = hf_pipeline(
    "text-classification",
    model=save_path,   # Load from local path!
    tokenizer=save_path,
)

# Test it works after reloading
test = loaded_classifier("You've won a prize!")
print(f"Test prediction: {test[0]}")
print("✅ Model works offline!")
```

---

# PART V — PUTTING IT ALL TOGETHER

---

## 24. Project: Local AI Chatbot with Ollama + Python

```python
"""
PROJECT: Full-Featured Local AI Chatbot
Features:
  - Multi-turn conversation memory
  - Switchable personalities
  - Command system
  - Streaming output
  - Conversation save/load
"""

import requests
import json
import os
import datetime
from pathlib import Path

class LocalAIChatbot:
    
    PERSONALITIES = {
        "assistant": "You are a helpful, accurate, and concise AI assistant.",
        "teacher": """You are a patient and encouraging teacher. You explain concepts 
                    step-by-step, check for understanding, and use clear analogies.""",
        "coder": """You are an expert software engineer. You write clean, efficient, 
                  well-commented code. You explain your choices and point out potential issues.""",
        "socrates": """You are Socrates. You answer questions with questions to help 
                     people discover answers themselves. You challenge assumptions gently.""",
        "eli5": "Explain everything like I'm 5 years old. Use simple words and fun analogies.",
    }
    
    COMMANDS = {
        "/help": "Show available commands",
        "/clear": "Clear conversation history",
        "/personality <name>": "Switch personality (assistant/teacher/coder/socrates/eli5)",
        "/model <name>": "Switch Ollama model",
        "/save": "Save conversation to file",
        "/load <file>": "Load conversation from file",
        "/history": "Show conversation history",
        "/quit": "Exit the chatbot",
    }
    
    def __init__(self, model="llama3", personality="assistant",
                 ollama_url="http://localhost:11434"):
        self.model = model
        self.personality = personality
        self.ollama_url = ollama_url
        self.history = []
        self.conversations_dir = Path("./conversations")
        self.conversations_dir.mkdir(exist_ok=True)
    
    def check_ollama(self):
        """Check if Ollama is running."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            return True, models
        except requests.ConnectionError:
            return False, []
    
    def send_message(self, user_message, stream=True):
        """Send message and get response from Ollama."""
        
        # Build messages with system prompt
        messages = [
            {
                "role": "system",
                "content": self.PERSONALITIES[self.personality]
            }
        ]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})
        
        # Add to history immediately
        self.history.append({"role": "user", "content": user_message})
        
        # Stream the response
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 4096,
                }
            },
            stream=stream,
        )
        
        full_response = ""
        
        if stream:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        token = chunk["message"].get("content", "")
                        print(token, end="", flush=True)
                        full_response += token
                    if chunk.get("done"):
                        print()
                        break
        else:
            full_response = response.json()["message"]["content"]
        
        # Add assistant response to history
        self.history.append({"role": "assistant", "content": full_response})
        return full_response
    
    def handle_command(self, command):
        """Handle slash commands."""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/help":
            print("\n📖 Available Commands:")
            for cmd_name, desc in self.COMMANDS.items():
                print(f"  {cmd_name:30} {desc}")
        
        elif cmd == "/clear":
            self.history = []
            print("✅ Conversation history cleared.")
        
        elif cmd == "/personality":
            if args in self.PERSONALITIES:
                self.personality = args
                print(f"✅ Personality switched to: {args}")
            else:
                print(f"❌ Unknown personality. Options: {', '.join(self.PERSONALITIES.keys())}")
        
        elif cmd == "/model":
            if args:
                self.model = args
                print(f"✅ Model switched to: {args}")
            else:
                _, models = self.check_ollama()
                print(f"Available models: {', '.join(models)}")
        
        elif cmd == "/save":
            filename = self.conversations_dir / f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump({
                    "model": self.model,
                    "personality": self.personality,
                    "history": self.history,
                    "saved_at": str(datetime.datetime.now())
                }, f, indent=2)
            print(f"✅ Saved to {filename}")
        
        elif cmd == "/history":
            if not self.history:
                print("No conversation history.")
            else:
                print(f"\n📜 Conversation History ({len(self.history)//2} turns):")
                for i, msg in enumerate(self.history):
                    role = "👤 You" if msg["role"] == "user" else "🤖 AI"
                    content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                    print(f"  {i+1}. {role}: {content}")
        
        elif cmd == "/quit":
            return False  # Signal to exit
        
        else:
            print(f"❌ Unknown command: {cmd}. Type /help for available commands.")
        
        return True  # Continue running
    
    def run(self):
        """Main chat loop."""
        # Check Ollama connection
        running, models = self.check_ollama()
        if not running:
            print("❌ Ollama is not running!")
            print("   Start it with: ollama serve")
            return
        
        print("\n" + "=" * 60)
        print(f"🤖 Local AI Chatbot")
        print(f"=" * 60)
        print(f"Model:       {self.model}")
        print(f"Personality: {self.personality}")
        print(f"Available models: {', '.join(models[:5])}")
        print(f"Type /help for commands, /quit to exit")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    should_continue = self.handle_command(user_input)
                    if not should_continue:
                        print("👋 Goodbye!")
                        break
                    continue
                
                print("AI: ", end="", flush=True)
                self.send_message(user_input, stream=True)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except requests.exceptions.ConnectionError:
                print("❌ Lost connection to Ollama. Is it still running?")

# Run the chatbot
if __name__ == "__main__":
    bot = LocalAIChatbot(
        model="llama3",
        personality="assistant"
    )
    bot.run()
```

---

## 25. Project: Document Q&A System

```python
"""
PROJECT: Document Q&A System
- Upload any text/PDF document
- Ask questions about it
- Get accurate answers with source citations
- Uses Ollama for local, private processing
"""

import requests
import json
from pathlib import Path

class DocumentQA:
    """
    A simple document Q&A system using Ollama.
    No external vector database required — uses context window directly.
    """
    
    def __init__(self, model="llama3", ollama_url="http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
        self.documents = {}  # {filename: content}
    
    def load_text_file(self, filepath):
        """Load a plain text file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        self.documents[path.name] = content
        print(f"✅ Loaded: {path.name} ({len(content)} characters, ~{len(content)//4} tokens)")
        return content
    
    def load_string(self, content, name="document"):
        """Load content from a string directly."""
        self.documents[name] = content
        print(f"✅ Loaded: {name} ({len(content)} characters)")
    
    def ask(self, question, doc_name=None, verbose=True):
        """Ask a question about the loaded documents."""
        
        # Build context from documents
        if doc_name and doc_name in self.documents:
            context = self.documents[doc_name]
        else:
            # Use all documents
            context_parts = []
            for name, content in self.documents.items():
                context_parts.append(f"--- Document: {name} ---\n{content}")
            context = "\n\n".join(context_parts)
        
        if not context:
            return "No documents loaded. Please load a document first."
        
        # Truncate if too long (leave room for question and answer)
        max_context_chars = 12000  # ~3000 tokens
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n\n[Document truncated...]"
        
        prompt = f"""You are a helpful assistant that answers questions about documents.
Use ONLY the information provided in the document below to answer questions.
If the answer is not in the document, say "I couldn't find that information in the document."
Always quote the relevant part of the document that supports your answer.

DOCUMENT:
{context}

QUESTION: {question}

ANSWER:"""
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": verbose,
                "options": {
                    "temperature": 0.1,   # Low temperature for factual answers
                    "num_ctx": 8192,
                    "num_predict": 512,
                }
            },
            stream=verbose
        )
        
        if verbose:
            print("Answer: ", end="", flush=True)
            full_answer = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    print(token, end="", flush=True)
                    full_answer += token
                    if chunk.get("done"):
                        print("\n")
                        break
            return full_answer
        else:
            return response.json()["response"]
    
    def interactive_session(self):
        """Start an interactive Q&A session."""
        print("\n📄 Document Q&A System")
        print("Commands: 'load <filepath>', 'docs', 'quit'")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                break
            
            elif user_input.lower().startswith("load "):
                filepath = user_input[5:].strip()
                try:
                    self.load_text_file(filepath)
                except FileNotFoundError as e:
                    print(f"❌ {e}")
            
            elif user_input.lower() == "docs":
                if self.documents:
                    print("Loaded documents:")
                    for name in self.documents:
                        print(f"  - {name}")
                else:
                    print("No documents loaded.")
            
            else:
                if not self.documents:
                    print("❌ No documents loaded. Use 'load <filepath>' first.")
                else:
                    self.ask(user_input)


# ── DEMO: Use the Q&A System ──────────────────────────────────

qa = DocumentQA(model="llama3")

# Load sample content
sample_content = """
PYTHON PROGRAMMING LANGUAGE - OVERVIEW

Python is a high-level, interpreted programming language created by Guido van Rossum.
It was first released in 1991. Python emphasizes code readability and uses significant 
whitespace for code blocks.

KEY FEATURES:
- Dynamic typing and garbage collection
- Support for multiple programming paradigms
- Large standard library often called "batteries included"
- Interpreted language (no compilation needed)

PYTHON VERSIONS:
Python 2 was released in 2000 and reached end-of-life on January 1, 2020.
Python 3 was released in December 2008 and is not backward-compatible with Python 2.
Python 3.11 brought significant performance improvements of 10-60% faster than Python 3.10.
Python 3.12 was released in October 2023.

USE CASES:
Python is widely used in:
1. Data Science and Machine Learning (libraries: NumPy, Pandas, TensorFlow, PyTorch)
2. Web Development (frameworks: Django, Flask, FastAPI)
3. Automation and Scripting
4. Scientific Computing
5. Artificial Intelligence Research

POPULARITY:
According to the TIOBE Index and Stack Overflow surveys, Python has been the most 
popular programming language since 2021. It is particularly dominant in data science 
and ML fields where over 90% of practitioners use Python.
"""

qa.load_string(sample_content, "python_overview.txt")

# Ask questions
questions = [
    "Who created Python and when?",
    "What are the main use cases for Python?",
    "What happened to Python 2?",
    "How much faster is Python 3.11 compared to 3.10?",
    "What is Python's most popular web framework?",  # Not in document
]

print("\n" + "=" * 60)
print("DOCUMENT Q&A DEMO")
print("=" * 60)

for question in questions:
    print(f"\nQ: {question}")
    qa.ask(question, verbose=False)  # Non-streaming for demo clarity
    answer = qa.ask(question, verbose=False)
    print(f"A: {answer[:200]}...")
    print()
```

---

## 26. Troubleshooting Master Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                    TROUBLESHOOTING GUIDE                        │
└─────────────────────────────────────────────────────────────────┘

OLLAMA ISSUES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ "connection refused" when calling localhost:11434
  → Ollama is not running
  → FIX: Run 'ollama serve' in a terminal (keep it open)
  → FIX (Linux): sudo systemctl start ollama

❌ "model not found"
  → Model hasn't been pulled
  → FIX: ollama pull <model-name>

❌ Very slow generation (< 1 token/sec)
  → Model is running on CPU, not GPU
  → CHECK: Run 'ollama ps' — look for GPU layers > 0
  → FIX: Make sure CUDA is installed and GPU is detected
  → FIX: ollama run llama3 --verbose  (shows GPU layers)

❌ Out of memory during model loading
  → Model too large for your VRAM
  → FIX: Use a smaller model (llama3:8b instead of 70b)
  → FIX: Use more quantized version (q4 instead of fp16)
  → FIX: Free up VRAM by closing other GPU applications

❌ Ollama using too much RAM
  → Multiple models loaded in memory
  → FIX: OLLAMA_MAX_LOADED_MODELS=1 ollama serve

TRANSFORMERS / PYTORCH ISSUES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ RuntimeError: CUDA out of memory
  → Not enough GPU memory
  → FIX 1: Reduce batch size
  → FIX 2: Use load_in_8bit=True or load_in_4bit=True
  → FIX 3: Enable gradient_checkpointing=True
  → FIX 4: Use torch.cuda.empty_cache() before loading
  → FIX 5: Set PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

❌ torch.cuda.is_available() returns False
  → CUDA not properly installed
  → CHECK: nvidia-smi (should show your GPU)
  → CHECK: nvcc --version (should show CUDA version)
  → FIX: Reinstall PyTorch with correct CUDA version
  → FIX: Make sure CUDA toolkit matches driver version

❌ "OSError: We couldn't connect to HuggingFace Hub"
  → No internet / Hub is down / firewall blocking
  → FIX: Use local_files_only=True if model already downloaded
  → FIX: Set TRANSFORMERS_OFFLINE=1

❌ "Repository Not Found" or 401 error for gated models
  → Need to authenticate for restricted models (LLaMA, Gemma)
  → FIX: Run 'huggingface-cli login'
  → FIX: Accept the model's license on HuggingFace website

❌ Poor model performance after fine-tuning
  → Common causes:
  → 1. Learning rate too high → reduce by 10x
  → 2. Too few training examples → need 100+ per class minimum
  → 3. Bad data quality → manually review training examples
  → 4. Mismatch between train and test distribution
  → 5. Not using correct tokenizer for the model

❌ "Expected input batch_size (X) to match target batch_size (Y)"
  → Labels shape doesn't match model output shape
  → FIX: Make sure labels are [batch_size], not [batch_size, 1]
  → FIX: labels = labels.squeeze(-1)  or  labels = labels.view(-1)

WINDOWS-SPECIFIC ISSUES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ bitsandbytes not working on Windows
  → bitsandbytes has limited Windows support
  → FIX: pip install bitsandbytes-windows
  → FIX: Use WSL2 (Windows Subsystem for Linux) for full support

❌ Long path errors when downloading models
  → Windows has 260-char path limit by default
  → FIX: Enable long paths in Windows registry:
         Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
         Set LongPathsEnabled = 1
  → FIX: Move HF cache to short path: 
         set HF_HOME=C:\hf

COLAB-SPECIFIC ISSUES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ "Runtime disconnected" during training
  → Colab session timed out
  → FIX: Save checkpoints to Google Drive frequently
  → FIX: Use resume_from_checkpoint=True
  → PREVENTION: Interact with the page every 60-90 minutes
  → PREVENTION: Use Colab Pro for longer sessions

❌ Not getting GPU in Colab
  → GPU not assigned (random allocation)
  → FIX: Runtime → Change runtime type → GPU
  → If T4 full: try again later or use Colab Pro
```

```python
# ── DIAGNOSTIC SCRIPT ────────────────────────────────────────
# Run this first when things aren't working

def full_diagnostic():
    print("=" * 60)
    print("AI ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    
    # Python version
    import sys
    print(f"\n✓ Python: {sys.version}")
    
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                used = torch.cuda.memory_allocated(i) / 1e9
                total = props.total_memory / 1e9
                print(f"✓ GPU {i}: {props.name} ({used:.1f}/{total:.1f} GB used)")
        else:
            print("⚠ CUDA: Not available (CPU only)")
    except ImportError:
        print("✗ PyTorch: NOT INSTALLED")
    
    # Transformers
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers: NOT INSTALLED — run: pip install transformers")
    
    # Datasets
    try:
        import datasets
        print(f"✓ Datasets: {datasets.__version__}")
    except ImportError:
        print("✗ Datasets: NOT INSTALLED — run: pip install datasets")
    
    # PEFT
    try:
        import peft
        print(f"✓ PEFT: {peft.__version__}")
    except ImportError:
        print("⚠ PEFT: NOT INSTALLED — run: pip install peft (needed for LoRA)")
    
    # bitsandbytes
    try:
        import bitsandbytes
        print(f"✓ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        print("⚠ bitsandbytes: NOT INSTALLED — run: pip install bitsandbytes (needed for quantization)")
    
    # Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"✓ Ollama: Running | Models: {', '.join(models[:3]) if models else 'none pulled'}")
    except:
        print("⚠ Ollama: Not running or not installed")
    
    # HuggingFace login
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✓ HF Hub: Logged in as '{user_info['name']}'")
    except Exception:
        print("⚠ HF Hub: Not logged in (needed for gated models like LLaMA)")
    
    print("\n" + "=" * 60)

full_diagnostic()
```

---

## 27. Learning Roadmap

### 27.1 Your 12-Week Path from Beginner to Practitioner

```
WEEK 1-2: FOUNDATIONS
├── Read sections 1-5 of this guide
├── Install Python, PyTorch, Transformers
├── Run your first sentiment analysis pipeline
├── Experiment with every pipeline type
└── Goal: Understand tokens, embeddings, model types

WEEK 3-4: LOCAL MODELS WITH OLLAMA
├── Install Ollama, pull LLaMA 3 and Gemma
├── Build the chatbot project (Section 24)
├── Experiment with Modelfiles
├── Use Ollama API from Python
└── Goal: Have a working local AI assistant

WEEK 5-6: HUGGINGFACE DEEP DIVE
├── Understand tokenizers deeply (Section 20)
├── Explore HuggingFace Hub — download 10+ different models
├── Run LLaMA/Gemma with transformers library
├── Understand fine-tuning theory (Section 8 from main guide)
└── Goal: Comfortable loading and running any HF model

WEEK 7-8: FINE-TUNING
├── Complete the spam classifier project (Section 23)
├── Fine-tune on YOUR own dataset
├── Learn LoRA/PEFT (Section 10 from main guide)
├── Track experiments with W&B or TensorBoard
└── Goal: Fine-tune a model for a real problem you have

WEEK 9-10: ADVANCED TOPICS
├── Implement RAG (Section 15 of this guide)
├── Learn about quantization (INT8, INT4, GGUF)
├── Try Google Colab for larger models
├── Experiment with different model architectures
└── Goal: Build a complete AI application

WEEK 11-12: CAPSTONE PROJECT
├── Identify a real problem to solve with AI
├── Collect/prepare your dataset
├── Choose and fine-tune the right model
├── Deploy locally or to cloud
└── Goal: A working AI project to show employers/clients

RESOURCES (Free):
  Courses:  fast.ai (Practical Deep Learning)
            DeepLearning.AI (Coursera)
            HuggingFace NLP Course (huggingface.co/course)
  
  Books:    "Natural Language Processing with Transformers" (O'Reilly)
            "Hands-On Machine Learning" (Aurélien Géron)
  
  Practice: Kaggle competitions (NLP category)
            HuggingFace Spaces (see others' implementations)
  
  Community: HuggingFace forums (discuss.huggingface.co)
             r/MachineLearning
             Papers With Code (paperswithcode.com)
```

### 27.2 Hardware Buying Guide for Beginners

```
BUDGET: $0 — Use Google Colab Free
  - Free T4 GPU (15GB VRAM)
  - Sessions reset (no persistence)
  - Good for: Learning, experimenting, small fine-tuning

BUDGET: $10/month — Google Colab Pro
  - Better GPU allocation, longer sessions
  - Good for: Regular experimentation, 7B model work

BUDGET: $300-500 — Used RTX 3060 12GB or RTX 2080 Ti
  - 12GB VRAM, run 7B models comfortably
  - Local, private, always available
  - Good for: Dedicated learner, small production use

BUDGET: $700-800 — RTX 3080/4070 Ti (16GB)
  - 16GB VRAM, comfortable 7B fine-tuning
  - Good for: Serious practitioner
  - Best value in this range

BUDGET: $1500-2000 — RTX 3090 Ti or 4090 (24GB)
  - 24GB VRAM, run 13B in FP16, 70B in Q4
  - Fine-tune 7B models with LoRA comfortably
  - Good for: Professional use, content creators

BUDGET: $3000+ — 2× RTX 3090 or A5000 48GB
  - Multiple GPUs or workstation GPU
  - Run and fine-tune large models
  - Good for: Startup, research, production

CLOUD ALTERNATIVES (Pay-per-use):
  RunPod:   From $0.20/hr for RTX 3090, great for burst work
  vast.ai:  Community GPUs, very cheap, less reliable
  Lambda:   Professional, reliable, good support
  GCP/AWS:  Enterprise grade, expensive, most reliable
```

---

*Document Version: 1.0*
*Compatible with: Ollama 0.2+, Transformers 4.40+, PyTorch 2.2+*
*Tested on: Ubuntu 22.04, Windows 11 WSL2, macOS Sonoma (Apple Silicon)*
