# AIRAWAT LLM/NLP Research Workflow Guide

## 1. Environment Status Verification

Your AIRAWAT setup is now correctly working.

Verified Components:

* CUDA Working
* NVIDIA A100 GPU Allocation Working
* PyTorch GPU Detection Working
* Conda Environment Working
* Hugging Face Authentication Working
* SLURM Submission Working
* AIRAWAT Metadata Policy Solved
* Transformers Stack Working
* A100 40GB GPU Accessible

Your successful output:

```bash
Torch Version: 2.6.0+cu124
CUDA Available: True
CUDA Version: 12.4
GPU Count: 1
GPU Name: NVIDIA A100-SXM4-40GB
```

This confirms your AIRAWAT GPU workflow is operational.

---

# 2. IMPORTANT AIRAWAT CONCEPTS

## Login Node vs Compute Node

| Login Node       | Compute Node          |
| ---------------- | --------------------- |
| File management  | GPU workloads         |
| Script editing   | Training              |
| Dataset handling | Inference             |
| Git operations   | Fine-tuning           |
| Small CPU tasks  | Transformer execution |

NEVER run:

* Large transformers
* Training
* Heavy inference
* GPU-intensive workloads

on login node.

Always use:

* sbatch
* compute nodes

---

# 3. Recommended Project Structure

```text
~/projects/
│
├── ayur_llm_project/
│   ├── data/
│   ├── scripts/
│   ├── outputs/
│   ├── logs/
│   ├── slurm/
│   ├── notebooks/
│   ├── evaluations/
│   └── checkpoints/
```

---

# 4. Daily Workflow

## Login to AIRAWAT

```bash
ssh username@login.airawat.in
```

---

## Activate Environment

```bash
llm_env
```

Expected:

```bash
(llm_env) agapar@login:~$
```

---

## Go to Project

```bash
cd ~/projects/ayur_llm_project
```

OR if alias exists:

```bash
ayur
```

---

# 5. Important Monitoring Commands

## Check Current Jobs

```bash
squeue --me
```

---

## Detailed Job Information

```bash
scontrol show job JOBID
```

Example:

```bash
scontrol show job 432425
```

---

## Live Queue Monitoring

```bash
watch -n 5 squeue --me
```

Exit:

```text
CTRL + C
```

---

## View GPU Queue

```bash
squeue -p dibdp
```

---

## Check Available GPU Nodes

```bash
sinfo -p dibdp
```

---

## Check Job Priority

```bash
sprio -j JOBID
```

---

# 6. GPU Verification Commands

## Check GPU

```bash
nvidia-smi
```

---

## Verify PyTorch GPU

```python
import torch

print(torch.cuda.is_available())

print(torch.cuda.get_device_name(0))
```

---

# 7. Conda Environment Commands

## Activate Environment

```bash
llm_env
```

---

## Deactivate Environment

```bash
conda deactivate
```

---

## List Environments

```bash
conda env list
```

---

## Install Package

```bash
pip install PACKAGE_NAME
```

Example:

```bash
pip install transformers
```

---

# 8. Hugging Face Commands

## Login

```bash
hf auth login
```

---

## Check Logged User

```bash
hf auth whoami
```

---

## Model Download Test

```python
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2"
)
```

---

# 9. Important Cache Locations

## Hugging Face Home

```text
~/hf_home
```

---

## Torch Cache

```text
~/torch_cache
```

---

## Triton Cache

```text
~/triton_cache
```

---

# 10. Uploading Files to AIRAWAT

## From Windows PowerShell

```powershell
scp "E:/path/file.csv" username@login.airawat.in:~/projects/ayur_llm_project/data/
```

---

## Upload Entire Folder

```powershell
scp -r "E:/folder" username@login.airawat.in:~/projects/
```

---

# 11. SLURM Job Workflow

## Create SLURM Script

```bash
nano slurm/run_mistral.slurm
```

---

## Submit Job

```bash
sbatch --export=ALL run_mistral.slurm
```

---

## Check Logs

```bash
ls logs
```

---

## Read Output Log

```bash
cat logs/output.out
```

---

## Read Error Log

```bash
cat logs/error.err
```

---

# 12. Mandatory AIRAWAT Metadata

Before EVERY sbatch submission:

```bash
export JOB_DESCRIPTION="Detailed 50 plus word description here"
```

```bash
export EXPECTED_OUTCOME="Detailed 50 plus word expected outcome here"
```

Then:

```bash
sbatch --export=ALL script.slurm
```

---

# 13. Recommended LLM Models for A100 40GB

## Zero-Shot Inference

| Model               | Suitable              |
| ------------------- | --------------------- |
| Mistral 7B Instruct | Excellent             |
| Llama 3 8B          | Excellent             |
| Gemma 7B            | Excellent             |
| Qwen 7B             | Excellent             |
| Param Bharat 7B     | Good                  |
| Param 17B Thinking  | Requires Quantization |

---

# 14. Recommended Quantization Strategy

## 4-bit QLoRA Loading

Recommended for:

* Mistral
* Llama
* Param 17B

Benefits:

* Lower VRAM
* Faster loading
* Multiple models possible

---

# 15. Standard 4-bit Loading Template

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

# 16. Recommended Research Workflow

## Phase 1

Dataset creation in Google Colab.

---

## Phase 2

Generate prompt categories.

Example:

* Cat1
* Cat2
* Cat3
* Cat4
* Cat5

---

## Phase 3

Upload CSV to AIRAWAT.

---

## Phase 4

Run zero-shot inference:

* Mistral
* Llama
* Param

---

## Phase 5

Save outputs into:

```text
MODELNAME_Cat1_op
MODELNAME_Cat2_op
```

etc.

---

## Phase 6

Evaluate:

* Accuracy
* Top-k Accuracy
* F1 Score
* BLEU
* ROUGE
* Semantic Similarity

---

## Phase 7

Fine-tuning using:

* PEFT
* QLoRA
* LoRA adapters

---

# 17. tmux Commands

## Start tmux

```bash
tmux
```

---

## Create New Session

```bash
tmux new -s mistral
```

---

## Detach Session

```text
CTRL + B
then D
```

---

## Reattach Session

```bash
tmux attach -t mistral
```

---

## List Sessions

```bash
tmux ls
```

---

# 18. Important Linux Commands

## List Files

```bash
ls
```

---

## Detailed List

```bash
ls -lh
```

---

## Change Directory

```bash
cd folder_name
```

---

## Current Directory

```bash
pwd
```

---

## Create Folder

```bash
mkdir folder_name
```

---

## Remove Folder

```bash
rm -r folder_name
```

---

## Check Disk Usage

```bash
du -sh folder_name
```

---

## Check Storage

```bash
df -h
```

---

# 19. Important AIRAWAT Rules

## Always Use Batch Jobs

Preferred:

```bash
sbatch
```

Avoid running heavy workloads directly.

---

## Always Monitor Logs

Check:

* .out
* .err

files after jobs.

---

## Always Save Outputs Properly

Store:

* predictions
* logs
* metrics
* checkpoints

systematically.

---

## Always Use Quantization for Large Models

Especially:

* 13B+
* 17B+
* 70B models

---

# 20. Your Immediate Next Steps

## 1. Create Ayur Project

```bash
mkdir -p ~/projects/ayur_llm_project
```

---

## 2. Upload CSV Dataset

Place inside:

```text
~/projects/ayur_llm_project/data/
```

---

## 3. Create Inference Scripts

Examples:

```text
scripts/run_mistral.py
scripts/run_llama.py
scripts/run_param.py
```

---

## 4. Create SLURM Files

Examples:

```text
slurm/mistral.slurm
slurm/llama.slurm
slurm/param.slurm
```

---

## 5. Run First Zero-Shot Benchmark

Pipeline:

CSV → Prompt → Model → Output Columns → Evaluation

---

# 21. Final Important Advice

Always think in this workflow:

```text
Local Machine
    ↓
Google Colab/Data Prep
    ↓
Upload to AIRAWAT
    ↓
SLURM Submission
    ↓
GPU Execution
    ↓
Logs + Outputs
    ↓
Evaluation
    ↓
Research Paper
```

This is the standard professional NLP and LLM research workflow on national HPC systems.
