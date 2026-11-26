# The problem NCM attacks

| paradigm | positions processed per step | sequential steps | KV cache | known weakness |
|---|---|---|---|---|
| Autoregressive next-token prediction | 1 (plus cached context) | L | yes | latency grows with L |
| Masked / discrete diffusion (MDLM, DUO, …) | all L, every step | T (often 128–1,024) | no (bidirectional) | quality degrades quickly when T is small; heavy training (MDLM/DUO: 1M steps, more than 500B tokens on OWT) |
| Block diffusion | block | L / block × T_block | yes, across blocks | parallelism limited to one block |
| **NCM** | **ℓ_k (short at first)** | **K ≪ L** | prefix only (nothing else is needed) | fixed L and K; needs a separate tokenizer |

The paper lists three practical problems with diffusion LMs (DLMs): 

**(i)** they need many sampling steps, 

**(ii)** each step is expensive because it processes the full sequence, and 

**(iii)** they need large training budgets. Distillation and flow maps reduce (i), but they add training stages, they lose sample diversity (lower generative perplexity at lower entropy), and they leave (ii) untouched. NCM changes both the representation and the generation order instead, borrowing next-scale prediction from images: VAR (Tian et al. 2024) and HMAR (Kumbong et al. 2025, same first author).

The hard part of moving this idea to text is that text has no natural "downsampled image". There is no blurry version of a sentence, so the coarse representation has to be learned. Reconstruction must also be nearly exact: a single wrong token can change the meaning, break code, or invalidate a Sudoku.

The word "concept" in the paper simply means the information carried by the coarse levels of the hierarchy.

# How NCM works

NCM generates text in two stages.

<img width="2385" height="2231" alt="image" src="https://github.com/user-attachments/assets/1fdf9e5e-4c75-4822-8b4a-b58c4507a214" />

- *Stage 1*, the tokenizer. It turns a length-L text into K levels of discrete codes. Level 1 has very few codes (often 1, a summary of the whole text); level K has L codes. Each level stores what the earlier levels still got wrong:
  
  **a) u_k = Down_k(z − ẑ^(k−1)) - shrink what is still unexplained to ℓ_k positions (an average);**

  **b) r_k = nearest codebook entry for each of those positions;**

  **c) ẑ^(k) = ẑ^(k−1) + R_k(e_k(r_k)) - stretch it back to full length and add it to the running reconstruction ẑ.**

- Nested dropout randomly cuts the hierarchy off during training. That forces the coarse levels to carry meaning instead of leaving everything to the last level.

- *Stage 2*, the generator. A transformer predicts the levels from coarse to fine. Within a level, all codes are predicted in parallel in one forward pass: p(r) = Π_k Π_i p(r_k^i | ẑ^(k−1)). Generation therefore takes K steps (12 for 128 tokens), and early steps are cheap because they are short.

- Attention is block-diagonal: each level sees only the prompt and itself, because its input already summarizes the earlier levels. Training all levels in one pass is therefore exactly equivalent to running them one by one.

- Corruption: during training, some input codes are swapped for random ones so the model tolerates its own mistakes at generation time.

- Maths: the generator loss is exactly a lower bound on log-likelihood (the reconstruction term plus the generation term). The guide derives it step by step.

- A useful identity: averaging is linear, so the generator's input is exactly the "already explained" part of what the tokenizer quantizes at that level. NCM is effectively a learned, discrete "increase-the-resolution" diffusion, which suggests alternative designs.

# Critical findings from my own analysis

<img width="1964" height="874" alt="image" src="https://github.com/user-attachments/assets/ff2714b6-3931-4ca6-8ed1-2fdff6940a94" />

1. The speedup is real against diffusion models. On OWT, NCM runs 11,440 token positions through the network per sample, versus 131,000 for a 128-step diffusion model.

2. Against autoregressive models it does 5–11× more work per sample, just in far fewer sequential steps. It is a latency win, not an efficiency win, and the paper doesn't benchmark against autoregressive wall-clock time.

3. The "4–5× fewer training tokens" claim counts text tokens. The generator actually processes 5× (LM1B) to 11× (OWT) more positions per example. Measured that way, LM1B training uses about 40B positions versus 66B, and OWT about 794B versus about 524B for the diffusion baselines. That is more, not fewer.

4. The codes hold 3–9× more bits than the text (9.3× on OWT).

5. On OpenWebText it is still well behind autoregressive sampling: generative perplexity 83.1 versus 35.5.

# Citation
```
@misc{kumbong2026nextconceptmodelingfastparallel,
      title={Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation}, 
      author={Hermann Kumbong and Ishan S. Khare and Iddah Mlauzi and Andrew Woen and Charles Ding and E. Kelly Buchanan and Simran Arora and Azalia Mirhoseini and Christopher Ré},
      year={2026},
      eprint={2610.14162},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      doi={10.48550/arXiv.2610.14162},
      url={https://arxiv.org/abs/2610.14162}
}
```
