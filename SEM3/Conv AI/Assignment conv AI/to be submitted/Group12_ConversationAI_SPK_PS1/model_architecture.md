# Technical Architecture: Encoder-Only Transformer for Contextual Language Understanding

## 1. Overview

This document describes the architecture of an encoder-only transformer model trained from scratch using the Masked Language Modeling (MLM) objective on the SQuAD dataset. The model follows BERT-style pre‑training and is designed for language understanding tasks such as question answering and text classification.

The implementation is structured into six main tasks as per the assignment:

- **Task 1:** Data preprocessing and text cleaning  
- **Task 2:** Custom Byte Pair Encoding (BPE) tokenizer training  
- **Task 3:** Masked Language Modeling (MLM) setup  
- **Task 4:** Embedding layers (token, positional, segment)  
- **Task 5:** Transformer encoder construction (from scratch)  
- **Task 6:** Training loop and evaluation  

---

## 2. High‑Level Architecture (Mermaid Diagram)

The following diagram illustrates the complete forward pass of the model, from tokenized input to MLM logits.

```mermaid
flowchart TD
    subgraph Inputs
        A[Input IDs<br/>(batch, seq_len)]
        B[Segment IDs]
        M[Attention Mask]
    end

    subgraph Embeddings[Embedding Layer (Task 4)]
        TE[Token Embedding<br/>(vocab_size × embed_dim)]
        PE[Positional Embedding<br/>(max_seq_len × embed_dim)]
        SE[Segment Embedding<br/>(2 × embed_dim)]
        SUM[Element‑wise Sum]
        LN1[LayerNorm]
        D1[Dropout]
    end

    subgraph Encoder[Transformer Encoder (Task 5)]
        direction TB
        subgraph Layer[Encoder Layer (×4)]
            MHSA[Multi‑Head Self‑Attention<br/>8 heads, head_dim=32]
            ADD1[Residual Add]
            LN2[LayerNorm]
            FFN[Feed‑Forward Network<br/>256 → 1024 → 256]
            ADD2[Residual Add]
            LN3[LayerNorm]
        end
    end

    subgraph Head[MLM Prediction Head]
            LINEAR[Linear Layer<br/>embed_dim → vocab_size]
            LOGITS[Logits<br/>(batch, seq_len, vocab_size)]
    end

    A --> TE
    A --> PE
    B --> SE
    M --> MHSA

    TE --> SUM
    PE --> SUM
    SE --> SUM
    SUM --> LN1 --> D1

    D1 --> MHSA
    MHSA --> ADD1
    ADD1 --> LN2 --> FFN
    FFN --> ADD2 --> LN3
    LN3 -->|next layer| MHSA

    LN3 --> LINEAR --> LOGITS

## 2. Data Preparation (Task 1)

**Input:** SQuAD dataset (training + validation splits)

**Processing steps:**
1. Load the dataset using Hugging Face `datasets` library.
2. Clean each context and question:
   - Unescape HTML entities
   - Remove HTML tags, URLs, email addresses
   - Normalize Unicode (NFKD) and convert to ASCII
   - Remove non-ASCII characters and excessive punctuation
   - Collapse multiple whitespaces
3. Split cleaned passages into sentences using NLTK `sent_tokenize`.
4. Discard very short sentences (length ≤ 2 words).
5. Append both sentence fragments and cleaned questions to the corpus.

**Output:** `squad_clean_corpus.txt` containing ~593,966 sentences.

---

## 3. Custom BPE Tokenizer (Task 2)

**Why domain-specific tokenizer?**
- Trained on the SQuAD corpus to capture QA-specific vocabulary.
- Reduces sequence length compared to generic tokenizers (e.g., GPT-2) by ~12–28%.
- Improves handling of domain terminology and reduces OOV tokens.

**Tokenizer configuration:**
- Algorithm: Byte Pair Encoding (BPE)
- Vocabulary size: 10,000
- Special tokens: `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`
- Pre‑tokenization: Whitespace splitting
- Post‑processor: Adds `[CLS]` at the beginning and `[SEP]` at the end.

**Output:** `squad_bpe_tokenizer.json`

---

## 4. Masked Language Modeling Setup (Task 3)

The `MLMProcessor` class generates training samples following BERT’s masking strategy:

- **Masking probability:** 15% of tokens (excluding special tokens)
- **Token replacement:**
  - 80% → `[MASK]` token
  - 10% → random token from vocabulary
  - 10% → unchanged (original token)
- **Labels:** Only masked positions have the original token ID; others are set to `-100` (ignored by CrossEntropyLoss).

**Method:** `create_mlm_training_sample(text, max_length)` returns a dictionary with `input_ids`, `labels`, `attention_mask`.

---

## 5. Embedding Layers (Task 4)

Three independent embedding components are implemented as PyTorch modules:

| Component | Description | Shape |
|-----------|-------------|-------|
| TokenEmbedding | Learnable lookup table | `[vocab_size, embedding_dim]` |
| PositionalEmbedding | Learnable absolute position embeddings | `[max_seq_length, embedding_dim]` |
| SegmentEmbedding | Distinguishes sentence A vs B | `[num_segments, embedding_dim]` |

**BERTEmbedding** combines these by element-wise summation, followed by LayerNorm and Dropout.

**Hyperparameters:**
- `embedding_dim = 256`
- `max_seq_length = 512`
- `num_segments = 2`
- `dropout_rate = 0.1`

---

## 6. Transformer Encoder (Task 5)

The encoder is built from scratch using the following components:

### 6.1 Multi‑Head Self‑Attention (MHSA)
- `num_heads = 8`
- Head dimension = `embedding_dim / num_heads = 32`
- Linear projections for Q, K, V and output.
- Scaled dot‑product attention with mask support.

### 6.2 Feed‑Forward Network (FFN)
- Two linear layers with ReLU activation.
- Hidden dimension: `ffn_dim = 1024`
- Dropout between layers.

### 6.3 Transformer Encoder Layer
- Post‑LayerNorm architecture (original Transformer style):

- x = LayerNorm(x + Dropout(MHSA(x)))
- x = LayerNorm(x + Dropout(FFN(x)))

### 6.4 Encoder Stack
- Number of layers: `num_layers = 4`
- Each layer uses the same hyperparameters.

### 6.5 MLM Prediction Head
- Linear layer: `embedding_dim → vocab_size`

**Total trainable parameters:** 8,421,136

---

## 7. Training & Evaluation (Task 6)

### 7.1 Dataset Split
- Full cleaned corpus: 593,966 sentences.
- Train/validation split: 90/10 → **534,569** train, **59,397** validation.

### 7.2 DataLoader
- `batch_size = 32`
- `max_length = 128` (truncation/padding)
- Random shuffling for training.

### 7.3 Training Configuration
- **Device:** CUDA (GPU) or MPS (Apple Silicon) or CPU.
- **Optimizer:** AdamW (`lr=5e-4`, `weight_decay=0.01`)
- **Scheduler:** LinearLR with warmup over first 2 epochs.
- **Loss function:** CrossEntropyLoss (ignoring `-100` labels)
- **Gradient clipping:** max norm = 1.0
- **Epochs:** 50 (early stopping after no improvement for 5 epochs)

### 7.4 Evaluation Metrics
- **Validation loss** (cross‑entropy on masked tokens)
- **Masked token accuracy** (percentage of correctly predicted masked positions)

### 7.5 Early Stopping
Training stops if validation loss does not improve for 5 consecutive epochs after epoch 10.

---

## 8. Results

- **Best validation loss:** 3.0679 (epoch 42)
- **Final validation accuracy:** **45.70%**
- **Total training time:** ~30 minutes per epoch (on GPU); early stopping at epoch 48.

**Sample correct predictions (content words):**

| Masked Sentence (abbreviated) | Masked Token | Predicted Token |
|-------------------------------|--------------|------------------|
| In 1835 the [MASK] withdrew their right to vote. | legislature | legislature |
| … there is a large number of … cin as. | cin | cin |
| … were en by [MASK] … | ation | ation |
| Burmese migrant account for 80% of Thailand s migrant [MASK]. | workers | workers |
| vice president serves as [MASK] President of the Senate … | The | The |

---

## 9. Summary of Architectural Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Tokenizer | BPE (vocab=10k) | Domain adaptation, shorter sequences |
| Embedding dim | 256 | Balance between capacity and speed |
| Encoder layers | 4 | Sufficient for MLM; avoids overfitting |
| Attention heads | 8 | Standard for dim=256 |
| FFN hidden dim | 1024 | 4× embedding dimension (common practice) |
| Max sequence length | 128 | Covers most SQuAD sentences, reduces memory |
| Dropout | 0.1 | Regularisation |
| Optimizer | AdamW | Better weight decay handling |
| Learning rate | 5e-4 | Empirical optimum for this model size |
| Warmup steps | 2 epochs | Stabilises early training |

---

## 10. Conclusion

The implemented encoder‑only transformer successfully learns contextual representations via MLM, achieving **45.7% masked token prediction accuracy** on the SQuAD validation set – far above random chance (0.01%). The architecture follows BERT-style design with a custom BPE tokenizer, embedding layers, multi‑head self‑attention, and position‑wise FFNs. The model can be fine‑tuned for downstream tasks like extractive QA or classification.

--- 
**Technical Architecture Document Generated from:** `Group12_ConversationAI_SPK_PS1.ipynb`