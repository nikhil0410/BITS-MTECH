# Technical Report: Encoder-Only Transformer for SQuAD

## Design Decisions

1. **Encoder-Only Architecture**  
   - Selected because the task focuses on **language understanding** via Masked Language Modeling (MLM), not text generation.  
   - Bidirectional self-attention enables each token to incorporate both left and right context – crucial for predicting masked tokens.

2. **Custom BPE Tokenizer** (vocab_size=10,000)  
   - Trained from scratch on the cleaned SQuAD corpus. This captures domain‑specific vocabulary (e.g., “Napoleon”, “Paris”, “Revolution”) and reduces out‑of‑vocabulary tokens.  
   - Compared to a generic tokenizer (GPT‑2), our tokenizer produced **~25% shorter sequences** for typical QA sentences, improving efficiency.

3. **Model Architecture** (from scratch)  
   - **Embedding dimension**: 256  
   - **Encoder layers**: 4  
   - **Attention heads**: 8 (head dimension = 32)  
   - **Feed‑forward network**: 256 → 1024 → 256 with ReLU activation  
   - **Total trainable parameters**: 8,421,136  
   - **MLM prediction head**: Linear layer from 256 → vocab_size (10,000)  
   - This size balances representational power with feasible training on Apple Silicon (MPS).

4. **Training Setup**  
   - **Objective**: Masked Language Modeling – 15% of tokens masked per batch (80% → `[MASK]`, 10% → random token, 10% unchanged).  
   - **Optimizer**: AdamW (lr=5e-4, weight decay=0.01) with linear warmup over 10% of steps.  
   - **Batch size**: 32, sequence length: 128 tokens.  
   - **Gradient clipping**: 1.0 to stabilise training.

## Challenges Faced

- **Long training time**: The full SQuAD corpus contains ~590,000 sentences. With batch size 32, each epoch processed ~16,700 batches → ~30–40 minutes per epoch on MPS.  
  *Solution*: We still ran 50 epochs (total ~30 hours) because the model kept improving; early stopping could have been used, but we wanted to see peak performance.  
- **Memory limits on MPS (Apple Silicon)**: Sequence length beyond 128 caused out‑of‑memory errors. We fixed `max_length=128` and used gradient accumulation to simulate larger batches.  
- **Correct masking during evaluation**: Initially our manual `[MASK]` strings were not converted to the special token ID, leading to poor predictions. We corrected this by using the `MLMProcessor` or directly inserting the `[MASK]` token ID (4) after tokenization.

## Limitations

- **Model size is modest** – a larger transformer (e.g., 12 layers, 768 dim) would likely achieve higher accuracy but requires more data and compute.  
- **Only MLM pre‑training** – no fine‑tuning on downstream tasks (e.g., extractive QA). The model learns contextual representations but has not been adapted to a specific task.  
- **Sequence length truncation** – long SQuAD passages (>128 tokens) are truncated, potentially losing answer‑relevant context.  
- **Training time** – 50 epochs on the full dataset took ~30 hours; for faster iteration, a subset could be used, but we prioritised final accuracy.

## Results

- **Final validation accuracy**: **38.40%** (at epoch 50)  
- **Best validation loss**: **3.5829** (down from initial ~7.0)  
- **Training dataset**: 534,569 sentences (90% of cleaned corpus)  
- **Validation dataset**: 59,397 sentences (10%)  
- **Loss decreased steadily** without overfitting – validation loss remained lower than training loss.

**Sample predictions** (after fixing the `[MASK]` token handling):

| Sentence (with [MASK]) | Expected | Predicted | Correct |
|------------------------|----------|-----------|---------|
| The capital of [MASK] is Paris | France | France | ✓ |
| Machine [MASK] models learn from data | learning | learning | ✓ |
| Napoleon was a famous [MASK] leader | military | military | ✓ |

These examples demonstrate that the model has learned meaningful **contextual relationships** – e.g., “capital of … is Paris” strongly points to a country name, and “famous … leader” in a Napoleonic context points to “military”.

## Conclusion

The encoder‑only transformer, trained from scratch with a custom BPE tokenizer and the MLM objective, successfully learns rich contextual representations. It achieves **38.4% masked token prediction accuracy** on the SQuAD validation set – far above random chance (0.01%). The bidirectional self‑attention allows the model to leverage both left and right context, which is essential for language understanding tasks. While larger models and more pre‑training would improve performance, this work demonstrates that even a compact, from‑scratch encoder can learn useful semantics from a domain‑specific corpus.