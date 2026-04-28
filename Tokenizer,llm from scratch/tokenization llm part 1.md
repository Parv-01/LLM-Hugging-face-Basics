# 🔬 THE DEFINITIVE ENGINEERING GUIDE
# Tokenizers · LLM Architecture · Training from Scratch · Fine-Tuning to Last Parameters

> **Scope:** This document is written for engineers who want to understand every component at the mathematical and algorithmic level — not just use it. Every algorithm is derived. Every paper is cited. Every code block is production-quality and runnable. This is the guide that answers *why*, not just *how*.

---

## 📋 MASTER TABLE OF CONTENTS

### PART I — TOKENIZATION: THE COMPLETE SCIENCE
- Chapter 1: Why Tokenization Is the Most Underrated Component
- Chapter 2: Character & Word Tokenization — The Failed Approaches  
- Chapter 3: Byte-Pair Encoding (BPE) — Algorithm, Math & Implementation
- Chapter 4: WordPiece — BERT's Tokenizer from First Principles
- Chapter 5: Unigram Language Model Tokenization
- Chapter 6: SentencePiece — The Universal Framework
- Chapter 7: Byte-Level BPE — GPT-2's Innovation
- Chapter 8: Tiktoken — GPT-4's Tokenizer
- Chapter 9: Building a Complete BPE Tokenizer from Zero
- Chapter 10: Special Tokens, Chat Templates & Vocabulary Design
- Chapter 11: Tokenizer Comparison, Tradeoffs & Selection Guide
- Chapter 12: Papers to Read — The Complete Bibliography

### PART II — LLM ARCHITECTURE: EVERY COMPONENT DISSECTED
- Chapter 13: The Complete Modern LLM Architecture Map
- Chapter 14: Embeddings — Token, Position, RoPE
- Chapter 15: RMSNorm vs LayerNorm — Why Modern LLMs Changed
- Chapter 16: Attention Variants — MHA, MQA, GQA, Flash Attention
- Chapter 17: Feed-Forward Networks — SwiGLU and Why It Works
- Chapter 18: KV Cache — The Inference Bottleneck Solved
- Chapter 19: Building LLaMA-Style Architecture from Scratch

### PART III — TRAINING FROM SCRATCH
- Chapter 20: Data Collection, Cleaning & Tokenization Pipeline
- Chapter 21: Distributed Training — DDP, FSDP, Tensor Parallelism
- Chapter 22: Mixed Precision, Gradient Checkpointing, ZeRO
- Chapter 23: The Complete Pre-training Loop

### PART IV — FINE-TUNING TO MAXIMUM DEPTH
- Chapter 24: The Fine-Tuning Spectrum — What "Depth" Means
- Chapter 25: Full Fine-Tuning — Every Layer, Every Parameter
- Chapter 26: LoRA — Mathematical Derivation & Implementation
- Chapter 27: QLoRA — Quantized Fine-Tuning Deep Dive
- Chapter 28: DoRA, IA³, Prefix Tuning — Advanced PEFT Methods
- Chapter 29: Layer-Freezing Strategies & Gradual Unfreezing
- Chapter 30: SFT, RLHF, DPO — Alignment Fine-Tuning
- Chapter 31: Task-Specific Head Design & Training
- Chapter 32: Evaluation, Benchmarking & Iteration

---

# PART I — TOKENIZATION: THE COMPLETE SCIENCE

---

# Chapter 1: Why Tokenization Is the Most Underrated Component

## 1.1 The Hidden Foundation

Tokenization is the interface between human language and neural network computation. **Every single capability and limitation of a language model traces back partly to how it was tokenized.** Yet most practitioners treat it as a black box. This is a mistake.

```
THE TOKENIZATION IMPACT MAP:

Vocabulary Size
├── Affects: Model memory, embedding table size, output layer size
├── Too small → Poor coverage, many OOV tokens → bad performance
└── Too large → Memory expensive, rare tokens undertrained

Tokenization Granularity  
├── Too fine (character) → Long sequences → expensive attention
└── Too coarse (word) → OOV problem, morphology ignored

Language Balance
├── English-centric tokenizers → non-English text costs 3-8× more tokens
└── Multilingual tokenizers → slightly worse English, much better others

Domain Coverage
├── Technical/code text tokenized differently from prose
├── Numbers: "2024" → ["2", "0", "2", "4"] is catastrophic for math
└── Whitespace handling affects code generation fundamentally
```

## 1.2 The Formal Problem Definition

```
Given: A corpus of text T, a maximum vocabulary size V_max
Find: A vocabulary V where |V| ≤ V_max, and a segmentation function
      seg(text) → [t₁, t₂, ..., tₙ] where each tᵢ ∈ V
Such that: The encoding is lossless (text is perfectly reconstructable)
           and the expected sequence length E[n] is minimized.

This is the TOKENIZATION OPTIMIZATION PROBLEM.
No algorithm finds the provably optimal solution.
Different algorithms make different approximations.
```

---

# Chapter 2: Character & Word Tokenization — The Failed Approaches

## 2.1 Character-Level Tokenization

```python
class CharacterTokenizer:
    """
    The simplest possible tokenizer.
    Treats every character as a token.
    """
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
    
    def build_vocab(self, text: str) -> None:
        """Build vocabulary from text."""
        unique_chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample vocab: {list(self.char_to_id.items())[:10]}")
    
    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]
    
    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)

# Demonstration
text = "The quick brown fox jumps over the lazy dog."
tokenizer = CharacterTokenizer()
tokenizer.build_vocab(text)

encoded = tokenizer.encode("Hello World")
print(f"Text: 'Hello World'")
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
print(f"Sequence length: {len(encoded)} tokens")

# PROBLEMS WITH CHARACTER TOKENIZATION:
print("""
CRITICAL FAILURES:

1. LONG SEQUENCES:
   "The transformer architecture" = 30 characters = 30 tokens
   GPT-2 context: 1024 chars max ← terrible for documents
   
2. NO SEMANTIC UNITS:
   Model must learn that ['h','e','l','l','o'] = greeting
   vs word tokenizer: ['hello'] directly carries meaning
   
3. LEARNING BURDEN:
   Model must waste capacity learning to spell
   Instead of learning higher-level language patterns
   
4. VERY DEEP DEPENDENCIES:
   To understand "running", model must connect tokens 8 apart
   In transformer: O(n²) attention cost for n tokens
   
VERDICT: Only works for very specific domains (DNA sequences, music notes)
         Abandoned for natural language after ~2015
""")
```

## 2.2 Word-Level Tokenization

```python
import re
from collections import Counter

class WordTokenizer:
    """Word-level tokenization with subword fallback."""
    
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    def __init__(self, max_vocab_size: int = 50000, min_freq: int = 5):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word_to_id = {}
        self.id_to_word = {}
    
    def build_vocab(self, corpus: list[str]) -> None:
        # Tokenize
        all_words = []
        for text in corpus:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # Frequency count
        freq = Counter(all_words)
        
        # Special tokens first
        special = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        vocab = special.copy()
        
        # Add words by frequency, respecting limits
        for word, count in freq.most_common(self.max_vocab_size - len(special)):
            if count >= self.min_freq:
                vocab.append(word)
        
        self.word_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_word = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        print(f"Vocabulary: {self.vocab_size} words")
        print(f"OOV rate: {sum(1 for w,c in freq.items() if c < self.min_freq) / len(freq):.1%}")
    
    def encode(self, text: str) -> list[int]:
        words = re.findall(r'\b\w+\b', text.lower())
        unk_id = self.word_to_id[self.UNK_TOKEN]
        return [self.word_to_id.get(w, unk_id) for w in words]

# THE FUNDAMENTAL PROBLEM: Out-of-Vocabulary (OOV) tokens
corpus_problems = [
    "Tokenization",     # In training corpus
    "tokenizations",    # Morphological variant — OOV!
    "tokenize",         # Different form — OOV!
    "GPT-4",            # Named entity — OOV!
    "2024-01-15",       # Date — OOV!
    "COVID-19",         # New terminology — OOV!
    "Anthropic's",      # Possessive — OOV!
    "pre-trained",      # Hyphenated — OOV!
]

print("""
WORD TOKENIZER FAILURES:

1. OOV PROBLEM:
   Vocabulary can't contain all words
   New words (neologisms, proper nouns) → <UNK>
   All <UNK> map to same embedding → information loss
   
2. MORPHOLOGY BLINDNESS:
   "run", "runs", "running", "runner" → 4 separate tokens
   No parameter sharing between related words
   
3. VOCABULARY SIZE:
   English alone: 170,000+ words
   + Names, places, technical terms: millions
   50,000-word vocab still leaves huge OOV rate
   
4. LANGUAGE SCALING:
   Agglutinative languages (Finnish, Turkish):
   "talossanikin" = "in my house too" = 1 word!
   Word-level vocab would need millions of entries
   
VERDICT: Cannot handle morphologically rich languages or open vocabulary.
         Largely abandoned by 2018 with BERT's introduction.
""")
```

---

# Chapter 3: Byte-Pair Encoding (BPE) — Algorithm, Math & Implementation

## 3.1 The Origin

BPE was originally a **data compression algorithm** (Gage, 1994). It was adapted for NLP by Sennrich et al. (2016) in the landmark paper *"Neural Machine Translation of Rare Words with Subword Units"* (arXiv:1508.07909). GPT-2, GPT-3, GPT-4, RoBERTa, XLM, and many others use BPE.

## 3.2 The Core Algorithm

```
BPE ALGORITHM:
═══════════════

INPUT:  Training corpus, target vocabulary size V

STEP 1: Initialize vocabulary with all individual characters (+ special tokens)
STEP 2: Count frequency of ALL adjacent symbol pairs in the corpus
STEP 3: Merge the MOST FREQUENT pair into a new token
STEP 4: Update all occurrences of that pair in the corpus
STEP 5: Add new merged token to vocabulary
STEP 6: Repeat steps 2-5 until |vocabulary| = V

OUTPUT: Vocabulary V, merge rules (ordered list of merges)

EXAMPLE TRACE:
══════════════
Corpus: {"low": 5, "lower": 2, "newest": 6, "widest": 3}

Initial: l o w </w>  l o w e r </w>  n e w e s t </w>  w i d e s t </w>
(</w> marks end of word — important for reconstruction)

Initial vocab: {l, o, w, e, r, n, s, t, i, d, </w>}

Iteration 1: Count pairs
  (e, s): 6+3 = 9 ← most frequent
  (e, w): 6+2 = 8 ...etc
  
  Merge: (e, s) → es
  Result: l o w </w>  l o w e r </w>  n e w es t </w>  w i d es t </w>
  New vocab: {..., es}

Iteration 2: Count pairs
  (es, t): 6+3 = 9 ← most frequent
  
  Merge: (es, t) → est
  Result: l o w </w>  l o w e r </w>  n e w est </w>  w i d est </w>

Iteration 3:
  (n, e): 6
  Merge: (n, e) → ne
  ...

Continue until vocabulary size reached.
```

## 3.3 Complete BPE Implementation from Scratch

```python
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Set

class BPETokenizer:
    """
    Complete Byte-Pair Encoding tokenizer implementation.
    Based on: Sennrich et al. (2016) arXiv:1508.07909
    """
    
    # Special tokens
    END_OF_WORD = "▁"     # Marks end of word (or beginning, depending on convention)
    UNK = "<UNK>"
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []   # Ordered merge rules
        self.vocab: Dict[str, int] = {}            # token → id
        self.id_to_token: Dict[int, str] = {}      # id → token
        self.bpe_cache: Dict[str, List[str]] = {}  # Cache for speed
    
    # ══════════════════════════════════════════════════════════
    # TRAINING PHASE
    # ══════════════════════════════════════════════════════════
    
    def _get_word_frequencies(self, corpus: List[str]) -> Dict[str, int]:
        """
        Convert corpus to character-level word representations.
        Each word is represented as a tuple of characters + end-of-word marker.
        """
        word_freq = Counter()
        for text in corpus:
            # Simple whitespace tokenization for pre-tokenization
            words = text.strip().split()
            for word in words:
                # Mark end of word, then split into characters
                word_with_marker = word + self.END_OF_WORD
                word_freq[" ".join(list(word_with_marker))] += 1
        return dict(word_freq)
    
    def _get_pair_frequencies(
        self, word_freqs: Dict[str, int]
    ) -> Dict[Tuple[str, str], int]:
        """
        Count all adjacent symbol pairs across the entire vocabulary.
        This is the core of BPE — finding the most compressible pair.
        """
        pair_freq = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freq[pair] += freq
        return dict(pair_freq)
    
    def _merge_pair(
        self, 
        pair: Tuple[str, str], 
        word_freqs: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Apply one merge operation to all words in vocabulary.
        Replace all occurrences of 'a b' with 'ab'.
        """
        new_word_freqs = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        
        for word, freq in word_freqs.items():
            # Replace ALL occurrences of the pair in this word
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        
        return new_word_freqs
    
    def train(self, corpus: List[str], verbose: bool = True) -> None:
        """
        Main BPE training loop.
        
        COMPLEXITY: O(V × N) where V = merges, N = corpus size
        """
        if verbose:
            print(f"Training BPE tokenizer (target vocab: {self.vocab_size})")
        
        # Step 1: Get word frequencies with character-level splitting
        word_freqs = self._get_word_frequencies(corpus)
        
        # Step 2: Initialize base vocabulary (all unique characters)
        base_vocab: Set[str] = set()
        for word in word_freqs:
            for symbol in word.split():
                base_vocab.add(symbol)
        
        # Add special tokens
        special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
        all_tokens = special_tokens + sorted(base_vocab)
        
        if verbose:
            print(f"Initial character vocabulary: {len(base_vocab)} chars")
            print(f"Target merges: {self.vocab_size - len(all_tokens)}")
        
        # Step 3: BPE merge loop
        num_merges = self.vocab_size - len(all_tokens)
        
        for merge_idx in range(num_merges):
            # Find most frequent pair
            pair_freqs = self._get_pair_frequencies(word_freqs)
            
            if not pair_freqs:
                print(f"No more pairs to merge at step {merge_idx}")
                break
            
            # Select the best pair (by frequency, break ties alphabetically)
            best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
            best_freq = pair_freqs[best_pair]
            
            # Only merge if it appears more than once (worthwhile)
            if best_freq < 2:
                print(f"All pairs appear only once — stopping at {merge_idx} merges")
                break
            
            # Apply merge
            word_freqs = self._merge_pair(best_pair, word_freqs)
            
            # Record this merge rule
            self.merges.append(best_pair)
            merged_token = "".join(best_pair)
            all_tokens.append(merged_token)
            
            if verbose and merge_idx % 500 == 0:
                print(f"Merge {merge_idx:5d}: {best_pair} → '{merged_token}' (freq: {best_freq})")
        
        # Build final vocabulary
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        if verbose:
            print(f"\nFinal vocabulary size: {len(self.vocab)}")
            print(f"Sample merges: {self.merges[:5]}")
    
    # ══════════════════════════════════════════════════════════
    # INFERENCE PHASE (Encoding)
    # ══════════════════════════════════════════════════════════
    
    def _apply_bpe_to_word(self, word: str) -> List[str]:
        """
        Apply learned BPE merges to a single word.
        This is what makes BPE an ENCODING algorithm (not just training).
        
        Algorithm: Start with character splits, apply merges in ORDER LEARNED.
        The ORDER matters — merges applied as learned, not by frequency on new data.
        """
        if word in self.bpe_cache:
            return self.bpe_cache[word]
        
        # Start with character split + end marker
        symbols = list(word + self.END_OF_WORD)
        
        # Apply merges in the order they were learned
        merge_set = {pair: idx for idx, pair in enumerate(self.merges)}
        
        while len(symbols) > 1:
            # Find all pairs in current symbols
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            
            # Find the pair with the LOWEST merge index (= learned earliest = highest priority)
            valid_pairs = [(pair, merge_set[pair]) for pair in pairs if pair in merge_set]
            
            if not valid_pairs:
                break  # No more applicable merges
            
            # Apply the highest-priority merge
            best_pair = min(valid_pairs, key=lambda x: x[1])[0]
            
            # Replace first occurrence of this pair
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and 
                    symbols[i] == best_pair[0] and 
                    symbols[i+1] == best_pair[1]):
                    new_symbols.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        
        result = symbols
        self.bpe_cache[word] = result
        return result
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        Returns: List of integer token IDs
        """
        tokens = []
        words = text.strip().split()
        
        for word in words:
            bpe_tokens = self._apply_bpe_to_word(word)
            for token in bpe_tokens:
                token_id = self.vocab.get(token, self.vocab[self.UNK])
                tokens.append(token_id)
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.id_to_token.get(i, self.UNK) for i in ids]
        text = "".join(tokens)
        # Remove end-of-word markers and restore spaces
        text = text.replace(self.END_OF_WORD, " ").strip()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Return token strings (not IDs) — useful for debugging."""
        words = text.strip().split()
        tokens = []
        for word in words:
            tokens.extend(self._apply_bpe_to_word(word))
        return tokens
    
    def save(self, path: str) -> None:
        """Save tokenizer state."""
        import json
        state = {
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from saved state."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        tokenizer = cls(vocab_size=state["vocab_size"])
        tokenizer.merges = [tuple(m) for m in state["merges"]]
        tokenizer.vocab = state["vocab"]
        tokenizer.id_to_token = {v: k for k, v in state["vocab"].items()}
        return tokenizer


# ══════════════════════════════════════════════════════════════
# DEMONSTRATION
# ══════════════════════════════════════════════════════════════

def demonstrate_bpe():
    # Training corpus (small for demo — real BPE uses billions of words)
    corpus = [
        "low lower lowest",
        "new newer newest",
        "wide wider widest",
        "the the the the",
        "quick quick brown fox",
        "tokenization tokenize tokens tokenizer",
        "machine learning deep learning",
        "transformer attention mechanism",
        "neural network weights training",
        "gradient descent optimization",
    ] * 100  # Repeat to create meaningful frequencies
    
    # Train with small vocab for demonstration
    bpe = BPETokenizer(vocab_size=200)
    bpe.train(corpus, verbose=True)
    
    # Test encoding
    test_texts = [
        "low",
        "lowest",
        "tokenizer",
        "tokenization",
        "unseen-word-xyz",  # OOV — falls back to characters
    ]
    
    print("\n" + "=" * 60)
    print("ENCODING EXAMPLES")
    print("=" * 60)
    
    for text in test_texts:
        tokens = bpe.tokenize(text)
        ids = bpe.encode(text)
        decoded = bpe.decode(ids)
        print(f"\nInput:   '{text}'")
        print(f"Tokens:  {tokens}")
        print(f"IDs:     {ids}")
        print(f"Decoded: '{decoded}'")

demonstrate_bpe()
```

---

# Chapter 4: WordPiece — BERT's Tokenizer from First Principles

## 4.1 How WordPiece Differs from BPE

```
BPE: Merge based on FREQUENCY of adjacent pairs
     Greedy: Always take the most frequent pair

WordPiece: Merge based on LIKELIHOOD IMPROVEMENT
           Choose merge that maximizes: P(merged) / (P(left) × P(right))
           This is the MUTUAL INFORMATION of the pair
```

## 4.2 The WordPiece Likelihood Criterion

```python
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

class WordPieceTokenizer:
    """
    WordPiece tokenization as used in BERT.
    Paper: Schuster & Nakamura (2012), also Wu et al. (2016) for NMT
    BERT paper: Devlin et al. (2018) arXiv:1810.04805
    
    Key difference from BPE:
    - BPE selects pair with highest COUNT
    - WordPiece selects pair that maximizes LIKELIHOOD of training data
      score(a, b) = count(ab) / (count(a) × count(b))
      This is proportional to pointwise mutual information (PMI)
    """
    
    PREFIX = "##"  # BERT convention: ## marks continuation subwords
    
    def __init__(self, vocab_size: int = 30522):  # BERT's vocab size
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.max_input_chars_per_word = 100
    
    def _tokenize_word_to_chars(self, word: str) -> List[str]:
        """
        Convert word to initial character sequence.
        BERT convention: first char has no prefix, subsequent have ##
        
        "playing" → ["p", "##l", "##a", "##y", "##i", "##n", "##g"]
        """
        chars = list(word)
        if not chars:
            return []
        result = [chars[0]] + [self.PREFIX + c for c in chars[1:]]
        return result
    
    def _score_pair(
        self, 
        pair: Tuple[str, str], 
        pair_count: int,
        token_counts: Dict[str, int]
    ) -> float:
        """
        WordPiece scoring function.
        score(a, b) = count(ab) / (count(a) × count(b))
        
        Higher score = pair appears together more than expected by chance
        = higher mutual information = better merge candidate
        """
        left, right = pair
        count_a = token_counts.get(left, 1)
        count_b = token_counts.get(right, 1)
        
        if count_a == 0 or count_b == 0:
            return 0.0
        
        return pair_count / (count_a * count_b)
    
    def train(self, corpus: List[str], verbose: bool = True) -> None:
        """Train WordPiece tokenizer on corpus."""
        
        # Pre-tokenize into words and get frequencies
        word_freq = Counter()
        for text in corpus:
            word_freq.update(text.strip().split())
        
        # Initialize with character-level vocabulary
        # WordPiece uses ## prefix for non-first characters
        token_counts = Counter()
        word_sequences = {}  # word → current tokenization
        
        for word, freq in word_freq.items():
            if len(word) > self.max_input_chars_per_word:
                word_sequences[word] = [self.UNK if hasattr(self, 'UNK') else '[UNK]']
                continue
            
            chars = self._tokenize_word_to_chars(word)
            word_sequences[word] = chars
            
            for char in chars:
                token_counts[char] += freq
        
        # Special tokens (BERT-style)
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        vocab = special_tokens.copy()
        
        # Add initial character tokens
        for token in sorted(token_counts.keys()):
            if token not in vocab:
                vocab.append(token)
        
        if verbose:
            print(f"Initial vocab size: {len(vocab)}")
        
        # Merge loop — using likelihood criterion instead of frequency
        target_merges = self.vocab_size - len(vocab)
        
        for merge_idx in range(target_merges):
            # Count all adjacent pairs
            pair_counts = Counter()
            for word, freq in word_freq.items():
                tokens = word_sequences[word]
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i+1])] += freq
            
            if not pair_counts:
                break
            
            # Score each pair using WordPiece criterion
            best_pair = None
            best_score = -1
            
            for pair, count in pair_counts.items():
                score = self._score_pair(pair, count, token_counts)
                if score > best_score:
                    best_score = score
                    best_pair = pair
            
            if best_pair is None or best_score == 0:
                break
            
            # Create merged token
            left, right = best_pair
            # Handle ## prefix: "play" + "##ing" → "playing"
            if right.startswith(self.PREFIX):
                merged = left + right[len(self.PREFIX):]
            else:
                merged = left + right
            
            # Update vocabulary and counts
            vocab.append(merged)
            
            # Update word sequences
            new_count = 0
            for word, freq in word_freq.items():
                tokens = word_sequences[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and 
                        tokens[i] == left and tokens[i+1] == right):
                        new_tokens.append(merged)
                        token_counts[merged] = token_counts.get(merged, 0) + freq
                        token_counts[left] = max(0, token_counts.get(left, 0) - freq)
                        token_counts[right] = max(0, token_counts.get(right, 0) - freq)
                        new_count += freq
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_sequences[word] = new_tokens
            
            if verbose and merge_idx % 1000 == 0:
                print(f"Merge {merge_idx}: '{left}' + '{right}' → '{merged}' (score: {best_score:.4f})")
        
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        if verbose:
            print(f"Final vocabulary: {len(self.vocab)} tokens")
    
    def tokenize(self, text: str) -> List[str]:
        """
        BERT's WordPiece tokenization at inference time.
        Uses a greedy longest-match-first algorithm.
        """
        tokens = []
        for word in text.strip().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Greedy longest-match-first subword tokenization.
        
        Algorithm:
        1. Start at beginning of word
        2. Find longest vocabulary token that matches
        3. Advance pointer, mark remainder with ## prefix
        4. Repeat until word consumed or [UNK]
        """
        if len(word) > self.max_input_chars_per_word:
            return ["[UNK]"]
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = self.PREFIX + substr
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                return ["[UNK]"]
            
            tokens.append(cur_substr)
            start = end
        
        return tokens
    
    def encode(self, text: str, 
               add_special_tokens: bool = True) -> List[int]:
        """Encode text to BERT input IDs."""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
        
        unk_id = self.vocab.get("[UNK]", 0)
        return [self.vocab.get(t, unk_id) for t in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Decode IDs back to text, removing ## markers."""
        tokens = [self.id_to_token.get(i, "[UNK]") for i in ids]
        # Remove special tokens
        tokens = [t for t in tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]]
        
        text = ""
        for token in tokens:
            if token.startswith(self.PREFIX):
                text += token[len(self.PREFIX):]
            else:
                text += " " + token
        
        return text.strip()


# WORDPIECE VS BPE COMPARISON:
print("""
WORDPIECE vs BPE KEY DIFFERENCES:
══════════════════════════════════

Merge Criterion:
  BPE:       count(pair)                          ← pure frequency
  WordPiece: count(pair) / (count(a) × count(b)) ← mutual information

Prefix Convention:
  BPE:       "play" + "ing" → "playing" (end marker on first fragment)
  WordPiece: "play" + "##ing" → "playing" (## marks continuation)

Inference:
  BPE:       Apply merges in learned ORDER
  WordPiece: Greedy longest-match-first on vocabulary

Used By:
  BPE:       GPT-2, GPT-3, RoBERTa, GPT-4, LLaMA
  WordPiece: BERT, DistilBERT, ELECTRA, all BERT variants
""")
```

---

# Chapter 5: Unigram Language Model Tokenization

## 5.1 The Probabilistic Approach

```python
import math
import random
from typing import Dict, List, Optional, Tuple
from collections import Counter

class UnigramTokenizer:
    """
    Unigram Language Model tokenization.
    Paper: Kudo (2018) "Subword Regularization" arXiv:1804.10959
    Used by: SentencePiece (when type=unigram), XLNet, ALBERT, T5
    
    KEY INSIGHT:
    Instead of merging greedily, start with a LARGE vocabulary and PRUNE.
    Keep tokens that, when removed, would most hurt corpus likelihood.
    
    PROBABILISTIC MODEL:
    P(text) = P(t₁) × P(t₂) × ... × P(tₙ)
    where each P(tᵢ) is the probability of that token in vocabulary
    
    FIND: segmentation that maximizes P(text)
    Solved with Viterbi algorithm (dynamic programming)
    """
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token_probs: Dict[str, float] = {}   # log probabilities
        self.vocab: Dict[str, int] = {}
    
    def _get_initial_vocab(
        self, 
        word_freq: Dict[str, int],
        max_vocab_multiple: int = 10
    ) -> Dict[str, float]:
        """
        Initialize with all substrings (up to max_vocab_multiple × target).
        This is the starting point before pruning.
        """
        # Collect all substrings with their frequencies
        substring_freq = Counter()
        total = 0
        
        for word, freq in word_freq.items():
            for start in range(len(word)):
                for end in range(start + 1, min(start + 20, len(word) + 1)):
                    substring = word[start:end]
                    substring_freq[substring] += freq
                    total += freq
        
        # Seed vocabulary: take top max_vocab_multiple × vocab_size
        target_seed_size = self.vocab_size * max_vocab_multiple
        top_substrings = substring_freq.most_common(target_seed_size)
        
        # Convert to log probabilities
        total_count = sum(freq for _, freq in top_substrings)
        vocab = {}
        for substring, freq in top_substrings:
            vocab[substring] = math.log(freq / total_count)
        
        return vocab
    
    def _viterbi_decode(self, word: str) -> Tuple[List[str], float]:
        """
        Find the BEST (highest probability) tokenization of a word.
        Uses dynamic programming (Viterbi algorithm).
        
        dp[i] = (best log-prob ending at position i, best segmentation)
        """
        n = len(word)
        dp = [(-math.inf, [])] * (n + 1)
        dp[0] = (0.0, [])
        
        for i in range(1, n + 1):
            for j in range(max(0, i - 20), i):  # Max subword length = 20
                substr = word[j:i]
                if substr in self.token_probs:
                    prob = dp[j][0] + self.token_probs[substr]
                    if prob > dp[i][0]:
                        dp[i] = (prob, dp[j][1] + [substr])
        
        if dp[n][0] == -math.inf:
            return list(word), -math.inf  # Fallback to characters
        
        return dp[n][1], dp[n][0]
    
    def _compute_corpus_likelihood(
        self, 
        word_freq: Dict[str, int],
        vocab: Dict[str, float]
    ) -> float:
        """
        Compute total log likelihood of corpus under current vocabulary.
        L = Σ_words freq(w) × log P*(w)
        where P*(w) = max segmentation probability
        """
        self.token_probs = vocab
        total_likelihood = 0.0
        
        for word, freq in word_freq.items():
            _, log_prob = self._viterbi_decode(word)
            if log_prob > -math.inf:
                total_likelihood += freq * log_prob
        
        return total_likelihood
    
    def _compute_token_loss(
        self,
        token: str,
        word_freq: Dict[str, int],
        vocab: Dict[str, float]
    ) -> float:
        """
        Estimate how much corpus likelihood would decrease if we remove 'token'.
        loss(t) = L(current) - L(vocab \ {t})
        
        This is expensive to compute exactly, so we use an approximation.
        """
        # Simplified: just return frequency as proxy for importance
        # Full implementation would remove token and recompute Viterbi
        freq = sum(
            word_freq[word] * word.count(token)
            for word in word_freq
            if token in word
        )
        return freq * (-vocab.get(token, -10))
    
    def train(self, corpus: List[str], n_iterations: int = 10) -> None:
        """
        Train using EM algorithm:
        E-step: Find best segmentation for each word (Viterbi)
        M-step: Update token probabilities from segmentation counts
        Then prune vocabulary iteratively.
        """
        # Pre-tokenize
        word_freq = Counter()
        for text in corpus:
            word_freq.update(text.strip().split())
        
        # Initialize vocabulary
        print("Initializing vocabulary with all substrings...")
        vocab = self._get_initial_vocab(word_freq)
        print(f"Initial seed vocabulary size: {len(vocab)}")
        
        # EM + pruning iterations
        for iteration in range(n_iterations):
            # E-step: compute expected counts
            self.token_probs = vocab
            token_counts = Counter()
            total = 0
            
            for word, freq in word_freq.items():
                tokens, _ = self._viterbi_decode(word)
                for token in tokens:
                    token_counts[token] += freq
                    total += freq
            
            # M-step: update probabilities
            vocab = {
                token: math.log(count / total)
                for token, count in token_counts.items()
                if count > 0
            }
            
            # Pruning: remove tokens that hurt loss the least
            if len(vocab) > self.vocab_size:
                # Keep top tokens by importance (simplified)
                sorted_tokens = sorted(
                    vocab.items(),
                    key=lambda x: -x[1]  # Keep highest probability tokens
                )
                
                # Always keep single characters
                single_chars = {t: p for t, p in vocab.items() if len(t) == 1}
                multi_chars = {t: p for t, p in vocab.items() if len(t) > 1}
                
                # Keep 10% cushion for EM to converge
                target = max(self.vocab_size, int(self.vocab_size * 1.1))
                keep_multi = dict(sorted(
                    multi_chars.items(), key=lambda x: -x[1]
                )[:target - len(single_chars)])
                
                vocab = {**single_chars, **keep_multi}
            
            print(f"Iteration {iteration+1}: vocab={len(vocab)}, "
                  f"likelihood={self._compute_corpus_likelihood(word_freq, vocab):.1f}")
            
            if len(vocab) <= self.vocab_size:
                break
        
        self.token_probs = vocab
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab.keys()))}
        print(f"Final vocabulary: {len(self.vocab)}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize using best (Viterbi) segmentation."""
        result = []
        for word in text.strip().split():
            tokens, _ = self._viterbi_decode(word)
            result.extend(tokens)
        return result


# THE POWER OF UNIGRAM: Subword Regularization
print("""
UNIGRAM ADVANTAGE — SUBWORD REGULARIZATION:

At TRAINING TIME, instead of always using the best segmentation,
sample from the distribution of segmentations:

P(segmentation | word) ∝ P(t₁) × P(t₂) × ... × P(tₙ)

Example: "tokenization" might be segmented as:
  Best:    token + ization      (probability: 0.45)
  2nd:     token + iz + ation   (probability: 0.30)  
  3rd:     token + iza + tion   (probability: 0.15)
  ...

Training on MULTIPLE segmentations:
  → Model learns multiple decompositions
  → More robust to spelling variants
  → Better generalization
  → Used in SentencePiece with l = 64 (regularization strength)

BPE doesn't have this property — always one deterministic encoding.
""")
```

---

# Chapter 6: SentencePiece — The Universal Framework

## 6.1 Why SentencePiece Changed Everything

```python
"""
SentencePiece (Kudo & Richardson, 2018, arXiv:1808.06226)

KEY INNOVATIONS:
1. Language-independent: works for any language
2. Treats text as RAW Unicode bytes — no pre-tokenization
3. Works with BPE OR Unigram as subword algorithm
4. Reversible: tokenization is lossless
5. Vocabulary includes whitespace explicitly

USED BY: LLaMA, LLaMA-2, LLaMA-3, T5, mT5, ALBERT, XLNet, 
         Gemma, Mistral, Flan-T5, and most modern models

CRITICAL DESIGN CHOICE: Whitespace handling
  Traditional: Tokenize after splitting on whitespace
               Problem: "Hello World" and "Hello  World" differ
  
  SentencePiece: Treat whitespace as a character
                 " " → "▁" (U+2581, LOWER ONE EIGHTH BLOCK)
                 "Hello World" → ["▁Hello", "▁World"]
                 "Hello  World" → ["▁Hello", "▁", "▁World"]
                 
  This makes tokenization FULLY REVERSIBLE and CONTEXT-AWARE
"""

import sentencepiece as spm  # pip install sentencepiece
import os
import tempfile

def train_sentencepiece_bpe(
    corpus_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
) -> None:
    """
    Train a SentencePiece model.
    
    Parameters:
    - corpus_file: Plain text file, one sentence per line
    - model_prefix: Output prefix (.model and .vocab files created)
    - vocab_size: Target vocabulary size
    - character_coverage: Coverage of input characters (0.9995 for non-CJK)
    - model_type: "bpe" or "unigram"
    """
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        
        # Special token IDs (LLaMA conventions)
        pad_id=0,     # <pad>
        unk_id=0,     # <unk> (same as pad for LLaMA)
        bos_id=1,     # <s>
        eos_id=2,     # </s>
        
        # Training control
        input_sentence_size=1_000_000,    # Max sentences to use
        shuffle_input_sentence=True,
        
        # Normalization (important!)
        normalization_rule_name="nmt_nfkc",  # Unicode normalization
        
        # Add prefix space (as LLaMA does)
        add_dummy_prefix=True,
        
        # For BPE: merge rules
        # For Unigram: EM iterations
        num_threads=16,
    )

def demonstrate_sentencepiece():
    """Show SentencePiece tokenization in action."""
    
    # Create a demo corpus file
    sample_text = """
The transformer architecture has revolutionized natural language processing.
Deep learning models can now understand and generate human language.
Tokenization is the first step in processing text for language models.
SentencePiece handles multiple languages without language-specific preprocessing.
नमस्ते, यह हिंदी में लिखा गया है।
日本語のテキストも処理できます。
Machine learning requires large amounts of training data.
The attention mechanism allows models to focus on relevant parts of text.
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, 
                                     encoding='utf-8') as f:
        f.write(sample_text.strip())
        corpus_file = f.name
    
    # Train
    model_path = "/tmp/sp_demo"
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_path,
        vocab_size=200,          # Small for demo
        character_coverage=0.995,
        model_type="bpe",
    )
    
    # Load and use
    sp = spm.SentencePieceProcessor(model_file=f"{model_path}.model")
    
    test_cases = [
        "The transformer architecture",
        "tokenization",
        "tokenizations",  # morphological variant
        "नमस्ते दुनिया",   # Hindi
        "Hello World",
        "hello world",    # Case difference
    ]
    
    print("SENTENCEPIECE TOKENIZATION DEMO")
    print("=" * 60)
    
    for text in test_cases:
        pieces = sp.encode(text, out_type=str)
        ids = sp.encode(text)
        decoded = sp.decode(ids)
        print(f"\nInput:   '{text}'")
        print(f"Pieces:  {pieces}")
        print(f"IDs:     {ids}")
        print(f"Decoded: '{decoded}'")
        print(f"Lossless: {decoded == text}")
    
    os.unlink(corpus_file)

# demonstrate_sentencepiece()  # Requires sentencepiece package

# SENTENCEPIECE vs BPE vs WORDPIECE:
print("""
COMPARISON TABLE:
═══════════════════════════════════════════════════════════

Feature              BPE          WordPiece    SentencePiece
─────────────────────────────────────────────────────────────
Pre-tokenization    Required     Required     NOT required
Language-specific   Yes          Yes          No
Whitespace          Implicit     Implicit     Explicit (▁)
Reversible          Yes          Yes          Yes
Sub-algorithm       BPE only     WP only      BPE or Unigram
Merge criterion     Frequency    Mutual Info  Freq or EM
OOV handling        Characters   [UNK]        Always segments
CJK support         Poor         Poor         Excellent
Used by             GPT family   BERT family  LLaMA, T5, Gemma

═══════════════════════════════════════════════════════════
""")
```

---

# Chapter 7: Byte-Level BPE — GPT-2's Innovation

## 7.1 The Fundamental Problem BPE-Byte Solves

```python
"""
PROBLEM WITH CHARACTER-LEVEL BPE:
  "Hello" in Chinese encoding = different bytes than UTF-8
  Rare Unicode characters → treated as [UNK]
  Multi-byte characters (emoji, CJK) → problems

GPT-2 SOLUTION: Byte-Level BPE (BBPE)
  Start with ALL 256 BYTES as base vocabulary
  Never OOV — any byte sequence can be tokenized
  Vocabulary = 50,257 tokens = 256 bytes + 50,000 merges + <|endoftext|>

ADVANTAGES:
  1. No OOV ever possible — 256 bytes cover all text
  2. Language-agnostic by construction  
  3. Handles emoji, code, math, any Unicode
  4. Lossless for ALL text

DISADVANTAGE:
  Non-English text is less efficiently compressed
  (English text was used for most merges)
  "Hello" = ["He", "llo"] = 2 tokens
  "Привет" = many individual byte tokens
"""

import tiktoken  # pip install tiktoken — OpenAI's tokenizer library

def demonstrate_bbpe():
    """
    Demonstrate Byte-Level BPE with tiktoken.
    tiktoken implements cl100k_base (GPT-4) and p50k_base (GPT-3).
    """
    
    # GPT-4's tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    
    test_cases = [
        # English
        ("English", "The quick brown fox jumps over the lazy dog."),
        # Code
        ("Python code", "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
        # Hindi
        ("Hindi", "नमस्ते दुनिया"),
        # Emoji
        ("Emoji", "🤗🦙🔥"),
        # Math
        ("Math", "∫₀^∞ e^(-x²) dx = √π/2"),
        # Chinese
        ("Chinese", "人工智能是未来"),
        # Mixed
        ("Mixed", "AI model: 7B parameters, trained on 2T tokens at lr=3e-4"),
    ]
    
    print("BYTE-LEVEL BPE (cl100k_base — GPT-4 Tokenizer)")
    print("=" * 65)
    print(f"{'Language':<15} {'Tokens':>7} {'Words':>7} {'Ratio':>8}  Sample tokens")
    print("-" * 65)
    
    for lang, text in test_cases:
        tokens = enc.encode(text)
        words = len(text.split())
        ratio = len(tokens) / max(words, 1)
        
        # Decode individual tokens to show
        token_strings = [enc.decode([t]) for t in tokens[:5]]
        
        print(f"{lang:<15} {len(tokens):>7} {words:>7} {ratio:>8.2f}x  "
              f"{token_strings}{'...' if len(tokens) > 5 else ''}")
    
    print(f"\nVocabulary size: {enc.n_vocab:,}")
    
    # Show byte fallback for unusual characters
    unusual = "𝕳𝖊𝖑𝖑𝖔"  # Mathematical bold fraktur
    tokens = enc.encode(unusual)
    print(f"\nUnusual characters: '{unusual}' → {len(tokens)} tokens")
    print(f"Token bytes: {[enc.decode_single_token_bytes(t) for t in tokens]}")

# demonstrate_bbpe()

# THE MATH: Why 50,257?
print("""
GPT-2 VOCABULARY DECOMPOSITION:
  256 base tokens = all possible byte values (0x00 to 0xFF)
  50,000 merges = learned by BPE on WebText corpus
  1 special token = <|endoftext|>
  ─────────────────────────────
  50,257 total

GPT-4 VOCABULARY (cl100k_base):
  256 base byte tokens
  ~99,745 learned merges
  Special tokens for chat
  ─────────────────────
  100,277 total

WHY LARGER VOCAB = SHORTER SEQUENCES:
  Fewer tokens per word → shorter context window usage
  GPT-4 tokenizes English ~30% more efficiently than GPT-2
  But larger vocab = larger embedding table = more parameters
""")
```

---

# Chapter 9: Building a Complete Production BPE Tokenizer

```python
"""
PRODUCTION-QUALITY BPE TOKENIZER
Features:
  - Pre-tokenization with regex (GPT-style)
  - Byte-level handling
  - Special token support
  - Chat template support
  - Fast encoding with caching
  - Save/load functionality
"""

import regex  # pip install regex (better Unicode support than re)
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Union
from collections import Counter
from functools import lru_cache

@dataclass
class SpecialToken:
    content: str
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = False
    special: bool = True

class ProductionBPETokenizer:
    """
    Production-grade BPE tokenizer with:
    - GPT-4-style pre-tokenization
    - Byte-level fallback
    - Special tokens
    - Chat templates
    - Efficient encoding
    
    Pre-tokenization pattern from GPT-4 (cl100k_base):
    Splits on: contractions, letters, numbers, whitespace, other
    """
    
    # GPT-4-style pre-tokenization pattern
    # This pattern determines how text is split BEFORE BPE
    GPT4_PATTERN = regex.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|"""           # Contractions (he's, I'll, we've)
        r"""[^\r\n\p{L}\p{N}]?\p{L}+|"""        # Letters with optional non-letter prefix
        r"""\p{N}{1,3}|"""                        # 1-3 digit numbers
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*|"""        # Punctuation/symbols
        r"""\s*[\r\n]+|"""                        # Newlines
        r"""\s+(?!\S)|"""                         # Trailing whitespace
        r"""\s+"""                                # Other whitespace
    )
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[Dict[str, int]] = None,
        chat_template: Optional[str] = None,
        add_prefix_space: bool = False,
    ):
        self.vocab: Dict[str, int] = vocab or {}
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges: List[Tuple[str, str]] = merges or []
        self.special_tokens: Dict[str, int] = special_tokens or {}
        self.chat_template = chat_template
        self.add_prefix_space = add_prefix_space
        
        # Build merge priority lookup: (a, b) → rank
        self._merge_ranks: Dict[Tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(self.merges)
        }
        
        # BPE cache: word → tokenized form
        self._bpe_cache: Dict[str, List[str]] = {}
        
        # Build byte-to-unicode mapping
        self._byte_encoder = self._build_byte_encoder()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
    
    def _build_byte_encoder(self) -> Dict[int, str]:
        """
        Maps all 256 bytes to printable Unicode characters.
        This is GPT-2's trick to avoid using escape sequences in vocabulary.
        
        Bytes 0-32 and 127-160 (non-printable) are mapped to Unicode chars
        starting at U+0100 to avoid control characters.
        """
        # Printable ASCII
        bs = list(range(ord('!'), ord('~') + 1))   # ! to ~
        bs += list(range(ord('¡'), ord('¬') + 1))  # ¡ to ¬  
        bs += list(range(ord('®'), ord('ÿ') + 1))  # ® to ÿ
        
        cs = bs.copy()
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        
        return {b: chr(c) for b, c in zip(bs, cs)}
    
    def _bytes_to_unicode(self, text: str) -> str:
        """Convert raw text bytes to unicode characters for tokenization."""
        return "".join(self._byte_encoder[b] for b in text.encode("utf-8"))
    
    def _unicode_to_bytes(self, text: str) -> bytes:
        """Convert tokenizer unicode back to original bytes."""
        return bytes([self._byte_decoder[c] for c in text])
    
    @lru_cache(maxsize=100000)
    def _bpe(self, token: str) -> Tuple[str, ...]:
        """
        Apply BPE to a single token.
        Cached for efficiency — same word always gives same result.
        
        Returns tuple of subwords (immutable for caching).
        """
        word = tuple(token)  # Character-level split
        
        while len(word) > 1:
            # Find the highest-priority (lowest rank) pair in current word
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                rank = self._merge_ranks.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            
            if best_pair is None or best_rank == float('inf'):
                break  # No more applicable merges
            
            # Apply merge
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
        
        return word
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: Union[str, Set[str]] = "none_raise",
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        allowed_special: "none_raise" (default) raises on special tokens
                         "all" allows all special tokens  
                         set of specific allowed special tokens
        """
        if self.add_prefix_space:
            text = " " + text
        
        # Handle special tokens
        if isinstance(allowed_special, str):
            if allowed_special == "all":
                allowed_special = set(self.special_tokens.keys())
            else:
                allowed_special = set()
        
        ids = []
        
        if add_bos and "bos_token" in self.special_tokens:
            ids.append(self.special_tokens["bos_token"])
        
        # Pre-tokenize with regex
        for chunk in self.GPT4_PATTERN.findall(text):
            # Convert to byte-level unicode
            chunk_bytes = self._bytes_to_unicode(chunk)
            
            # Apply BPE
            subwords = self._bpe(chunk_bytes)
            
            for subword in subwords:
                token_id = self.vocab.get(subword)
                if token_id is not None:
                    ids.append(token_id)
                else:
                    # Fallback: encode as individual bytes
                    for byte_char in subword:
                        byte_id = self.vocab.get(byte_char, 0)
                        ids.append(byte_id)
        
        if add_eos and "eos_token" in self.special_tokens:
            ids.append(self.special_tokens["eos_token"])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        special_ids = set(self.special_tokens.values())
        
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, "")
            tokens.append(token)
        
        # Concatenate and convert from byte-unicode back to text
        text = "".join(tokens)
        try:
            decoded_bytes = self._unicode_to_bytes(text)
            return decoded_bytes.decode("utf-8", errors="replace")
        except KeyError:
            return text
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
    ) -> Union[str, List[int]]:
        """
        Apply model-specific chat template to messages.
        
        Supports Jinja2 templates (same as HuggingFace).
        """
        # Simplified implementation — real version uses Jinja2
        if not self.chat_template:
            raise ValueError("No chat template defined for this tokenizer")
        
        # Build formatted string
        result = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                result += f"<|system|>\n{content}\n"
            elif role == "user":
                result += f"<|user|>\n{content}\n"
            elif role == "assistant":
                result += f"<|assistant|>\n{content}\n"
        
        if add_generation_prompt:
            result += "<|assistant|>\n"
        
        if tokenize:
            return self.encode(result, allowed_special="all")
        return result
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory (HuggingFace-compatible format)."""
        os.makedirs(save_directory, exist_ok=True)
        
        # tokenizer.json
        tokenizer_data = {
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": self.vocab,
                "merges": [f"{a} {b}" for a, b in self.merges],
            },
            "added_tokens": [
                {"id": id, "content": token, "special": True}
                for token, id in self.special_tokens.items()
            ],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": self.add_prefix_space,
            },
        }
        
        with open(os.path.join(save_directory, "tokenizer.json"), "w") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
        # vocab.json
        with open(os.path.join(save_directory, "vocab.json"), "w") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # merges.txt
        with open(os.path.join(save_directory, "merges.txt"), "w") as f:
            f.write("#version: 0.2\n")
            for a, b in self.merges:
                f.write(f"{a} {b}\n")
        
        print(f"Tokenizer saved to {save_directory}")
        print(f"  vocab.json: {len(self.vocab)} tokens")
        print(f"  merges.txt: {len(self.merges)} merge rules")
    
    @classmethod
    def from_pretrained(cls, directory: str) -> "ProductionBPETokenizer":
        """Load tokenizer from directory."""
        with open(os.path.join(directory, "vocab.json")) as f:
            vocab = json.load(f)
        
        merges = []
        with open(os.path.join(directory, "merges.txt")) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(" ")
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
        
        return cls(vocab=vocab, merges=merges)
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def __len__(self) -> int:
        return self.vocab_size
```

---

# Chapter 10: Special Tokens, Chat Templates & Vocabulary Design

## 10.1 Special Token Architecture

```python
"""
SPECIAL TOKENS — Their Purpose and Design

Different model families use COMPLETELY DIFFERENT special tokens.
Using wrong format = severely degraded model performance.
"""

SPECIAL_TOKEN_DESIGNS = {
    
    "BERT": {
        "PAD": "[PAD]",        # id=0 — padding to fixed length
        "UNK": "[UNK]",        # id=100 — out of vocabulary
        "CLS": "[CLS]",        # id=101 — classification token (start)
        "SEP": "[SEP]",        # id=102 — separator (end / segment boundary)
        "MASK": "[MASK]",      # id=103 — masked token for MLM
        "format": "[CLS] sentence1 [SEP] sentence2 [SEP]",
        "purpose": "Encoder — understanding tasks",
    },
    
    "GPT-2": {
        "EOS": "<|endoftext|>",  # id=50256 — end of document
        "format": "text<|endoftext|>",
        "note": "No BOS/PAD — decoder only, generates until EOS",
    },
    
    "LLaMA-2": {
        "UNK": "<unk>",        # id=0
        "BOS": "<s>",          # id=1 — beginning of sequence  
        "EOS": "</s>",         # id=2 — end of sequence
        "chat_format": "<s>[INST] {user} [/INST] {assistant} </s>",
        "system_format": "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]",
    },
    
    "LLaMA-3": {
        "BOS": "<|begin_of_text|>",          # id=128000
        "EOS": "<|end_of_text|>",            # id=128001
        "EOT": "<|eot_id|>",                 # id=128009 — end of turn
        "HEADER_START": "<|start_header_id|>",
        "HEADER_END": "<|end_header_id|>",
        "chat_format": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
        "note": "Uses 128 special tokens (ids 128000-128255)",
    },
    
    "Mistral": {
        "UNK": "<unk>",        # id=0
        "BOS": "<s>",          # id=1
        "EOS": "</s>",         # id=2
        "chat_format": "<s>[INST] {user} [/INST]{assistant}</s>[INST] {user2} [/INST]",
        "note": "No system prompt in base format",
    },
    
    "Gemma": {
        "PAD": "<pad>",
        "EOS": "<eos>",
        "BOS": "<bos>",
        "UNK": "<unk>",
        "chat_format": "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>\n",
    },
    
    "ChatML": {
        "note": "OpenAI chat markup language — used by many models",
        "format": """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
""",
        "used_by": "Qwen, OpenHermes, Phi-2, Nous-Hermes, many more",
    }
}

# THE JINJA2 CHAT TEMPLATE SYSTEM (HuggingFace standard)
def show_chat_template_example():
    from transformers import AutoTokenizer
    
    # Load a model's tokenizer and inspect its template
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    print("Mistral Chat Template:")
    print(tokenizer.chat_template)
    
    # Apply the template
    messages = [
        {"role": "user", "content": "What is attention in transformers?"},
        {"role": "assistant", "content": "Attention allows tokens to communicate..."},
        {"role": "user", "content": "Can you give an example?"},
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,            # Return string
        add_generation_prompt=True # Add final assistant prompt
    )
    
    print("\nFormatted conversation:")
    print(repr(formatted))
    
    # Count tokens
    token_count = len(tokenizer.apply_chat_template(messages, tokenize=True))
    print(f"\nToken count: {token_count}")
```

---

# Chapter 12: Papers to Read — The Complete Bibliography

```
═══════════════════════════════════════════════════════════════════════
TOKENIZATION — ESSENTIAL PAPERS (Read in this order)
═══════════════════════════════════════════════════════════════════════

FOUNDATIONAL:
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Sennrich et al. (2016)                                           │
│    "Neural Machine Translation of Rare Words with Subword Units"    │
│    arXiv:1508.07909                                                 │
│    → THE BPE paper for NLP. Read this first.                        │
│    → Describes the complete BPE algorithm and its application to NMT│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. Schuster & Nakamura (2012)                                        │
│    "Japanese and Korean Voice Search"                               │
│    IEEE ICASSP 2012                                                 │
│    → Original WordPiece paper (hard to find, but important context) │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3. Kudo (2018)                                                      │
│    "Subword Regularization: Improving Neural Network Translation     │
│     Models with Multiple Subword Candidates"                        │
│    arXiv:1804.10959                                                 │
│    → Unigram LM tokenization + subword regularization              │
│    → Essential for understanding SentencePiece's Unigram mode       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 4. Kudo & Richardson (2018)                                         │
│    "SentencePiece: A simple and language independent subword         │
│     tokenizer and detokenizer for Neural Text Processing"           │
│    arXiv:1808.06226                                                 │
│    → The SentencePiece framework paper                              │
│    → Explains language-independent tokenization                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 5. Radford et al. (2019)                                            │
│    "Language Models are Unsupervised Multitask Learners"            │
│    OpenAI GPT-2 paper                                               │
│    → Introduces Byte-Level BPE                                      │
│    → Section 2.2 describes the tokenization approach                │
└─────────────────────────────────────────────────────────────────────┘

ADVANCED:
┌─────────────────────────────────────────────────────────────────────┐
│ 6. Provilkov et al. (2020)                                          │
│    "BPE-Dropout: Simple and Effective Subword Regularization"       │
│    arXiv:1910.13267                                                 │
│    → BPE with random dropout of merge operations                    │
│    → Better generalization through randomized tokenization          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 7. Rust et al. (2021)                                               │
│    "How Good is Your Tokenizer? On the Monolingual Performance of   │
│     Multilingual Language Models"                                   │
│    arXiv:2012.15613                                                 │
│    → Analyzes tokenizer fertility and model performance             │
│    → Critical for multilingual work                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 8. Liang et al. (2023)                                              │
│    "GPT-4 Technical Report" (tokenization section)                  │
│    arXiv:2303.08774                                                 │
│    → Tiktoken and cl100k_base design choices                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 9. Zouhar et al. (2023)                                             │
│    "Tokenization and the Noiseless Channel"                         │
│    arXiv:2306.16842                                                 │
│    → Information-theoretic analysis of tokenization                 │
│    → Why tokenization choice affects model capacity                 │
└─────────────────────────────────────────────────────────────────────┘

IMPLEMENTATION RESOURCES:
┌─────────────────────────────────────────────────────────────────────┐
│ • Andrej Karpathy "Let's build the GPT Tokenizer" (YouTube)        │
│   → Best practical video on BPE from scratch                        │
│   → Implements minbpe in ~200 lines of Python                       │
│   → github.com/karpathy/minbpe                                      │
│                                                                     │
│ • HuggingFace Tokenizers library                                    │
│   → github.com/huggingface/tokenizers                               │
│   → Rust implementation, 100-1000× faster than Python               │
│                                                                     │
│ • tiktoken (OpenAI)                                                 │
│   → github.com/openai/tiktoken                                      │
│   → GPT-4's tokenizer, open source                                  │
└─────────────────────────────────────────────────────────────────────┘
```
