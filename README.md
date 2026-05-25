Implementation of Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism. WebConf 2025 (Oral).
<img width="2272" height="780" alt="" src="https://github.com/user-attachments/assets/ed9aeda2-c7ed-47bf-95ed-fb1622353931" />

- --

**Environment setup**: pip install scikit-learn; pip install pandas; pip install deepgnn-torch==0.1.60 deepgnn-ge==0.1.60; pip install wandb==0.17.5; pip install peft; pip install accelerate; pip install openai; pip install Jinja2; pip install transformers==4.45.2

**Dataset**: https://drive.google.com/drive/folders/1RcJ2M1l5zWPHYuGd9l5Gibcs5w5aI3y6

ED2-**Single** Repo: https://huggingface.co/JayceAnova/Benchmark-Single/tree/main

ED2-**Dual** Repo: https://huggingface.co/JayceAnova/Benchmark-Dual/tree/main

**NOTE**: The Single Repo does not use user attributes, and no user-related tasks are added during SFT. The user dataset is in **Instruments.inter.user.json**.

**Reproduction**:
1. Pretrain RQ-VAE and save the checkpoint: Script locates at ED2/index/run.sh.
2. Train the ED2 framework: Script locates at ED2/instruments_pretrain.sh.
3. Finetune the ED2 framework without ID conflicts. If index collision is acceptable, this step can be skipped. Scripts locate at ED2/infer.sh and ED2/instruments_finetune.sh.
4. Evaluate ED2: Script locates at ED2/instruments_evaluate.sh.

### LLM Backbone

| LLM Backbone | Scale | Hidden Dim | Vocab Size |
|--------------|-------|------------|------------|
| GPT-2        | 124M  | 768        | 50257      |
| Ministral    | 3B    | 4096       | 32000      |
| LLaMA-3      | 8B    | 4096       | 128256     |

<img width="835" alt="image" src="https://github.com/user-attachments/assets/db633259-4187-4e0e-9447-f60f93596c2e" />

- --
If this work is helpful, please consider citing it through,
```bibtex
@inproceedings{ed22025,
  title={Rel-MOSS: Towards Imbalanced Relational Deep Learning on Relational Databases},
  author={Jun Yin and Zhengxin Zeng and Mingzheng Li and Hao Yan and Chaozhuo Li and Weihao Han and Jianjin Zhang and Ruochen Liu and Hao Sun and Weiwei Deng and Feng Sun and Qi Zhang and Shirui Pan and Senzhang Wang},
  journal={Proceedings of the ACM on Web Conference},
  year={2025}
}
```
