**Environment setup**: pip install scikit-learn; pip install pandas; pip install deepgnn-torch==0.1.60 deepgnn-ge==0.1.60; pip install wandb==0.17.5; pip install peft; pip install accelerate; pip install openai; pip install Jinja2; pip install transformers==4.45.2

**Dataset**: https://drive.google.com/drive/folders/1RcJ2M1l5zWPHYuGd9l5Gibcs5w5aI3y6

ED2-**Single** Repo: https://huggingface.co/JayceAnova/Benchmark-Single/tree/main

ED2-**Dual** Repo: https://huggingface.co/JayceAnova/Benchmark-Dual/tree/main

**NOTE**: The Single Repo does not use user attributes, and no user-related tasks are added during SFT. The user dataset needs to be constructed separately. I will complete the user data soon.

**Reproduction**:
1. Pretrain RQ-VAE and save the checkpoint: Script locates at ED2/index/run.sh.
2. Train the ED2 framework: Script locates at ED2/instruments_pretrain.sh.
3. Finetune the ED2 framework without ID conflicts. If index collision is acceptable, this step can be skipped. Scripts locate at ED2/infer.sh and ED2/instruments_finetune.sh.
4. Evaluate ED2: Script locates at ED2/instruments_evaluate.sh.

# LLM Backbone

| LLM Backbone | Scale | Hidden Dim | Vocab Size |
|--------------|-------|------------|------------|
| GPT-2        | 124M  | 768        | 50257      |
| *Ministral*  | 3B    | 4096       | 32000      |
| LLaMA-3      | 8B    | 4096       | 128256     |

<img width="835" alt="image" src="https://github.com/user-attachments/assets/db633259-4187-4e0e-9447-f60f93596c2e" />
