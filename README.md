# Edge LLM Contest Submission - Training from Scratch

[![C4 Curated Dataset](https://img.shields.io/badge/🤗%20Dataset-C4%20Curated-blue.svg)](https://huggingface.co/datasets/neeleshg23/c4_curated) [![Jamba 1.9B](https://img.shields.io/badge/🤗%20Model-Jamba%201.9B-yellow.svg)](https://huggingface.co/neeleshg23/jamba-1.9b-alpaca-chinese)

### Overview
Jamba-based LM trained from scratch using the C4 and Alpaca datasets only using Llama 3.1-8B's tokenizer

## Data Preprocessing 
- Dependencies: Nvidia NeMo Data Curator, Nvidia CUDA GPU, and ~1TB disk space
- Given `allenai/c4`, run cleaning pipeline adapted from [RedPajama sample clean script](https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/pretraining-data-curation/red-pajama-v2-curation-tutorial.ipynb)

Data Preprocessing Steps:
  - language extraction
  - exact deduplication
  - fuzzy deduplication via jaccard similarity, jaccard compute, and connected components 
  - quality heuristic filtering

Code Reproduction:
- `cd 0_data_preprocessing`
- `conda env create -f nemo.yaml`

## Training
- Dependencies: Nvidia CUDA 12.4, PyTorch 2.4, mamba-ssm, causal-conv1d
- Given `datasets/neeleshg23/c4_curated`, train a 3 layer Jamba model for causal language modeling using source imported from [AI21labs](https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py)

Code Reproduction:
- `cd 1_training`
- `conda env create -f fa2.yaml`
- `torchrun --nproc-per-node $NUM_GPU 1_training.py`

## Evaluation
### Get Accuracies
- `conda activate opencompass`
- `pip install mamba-ssm causal-conv1d`
- `opencompass --datasets gsm8k_gen humaneval_gen commonsenseqa_7shot_cot_gen_734a22 truthfulqa_gen FewCLUE_chid_gen bbh_gen --hf-path neeleshg23/jamba-1.9b-8 --hf-type base --model-kwargs device_map='auto' trust_remote_code=True --max-out-len 1024 --max-num-workers $NUM_GPU`
### Get Memory and Throughput
- `python EvaluateThroughputAndMemory.py --model_name neeleshg23/jamba-1.9b-alpaca-chinese`
### Results
|       | GSM8K | HumanEval | CommonsenseQA | TruthfulQA | CHID-test | BBH | Throughput (Inf/s) | Memory (MB) |
|-------|-------|-----------|---------------|------------|-----------|-----|--------------------|-------------|
| Jamba |       |           |               |            |           |     | 5.26               | 8259.26     |


## Deployment
- `conda activate mlc-chat-venv`
- `python -c "import mlc_llm; print(mlc_llm.__path__)"` which returns `C:\\home\\miniconda3\\envs\\mlc-chat-venv\\Lib\\site-packages\\mlc_llm`
- For the sake of concision, let's call this directory $MYMLCPATH
- `cd 4_deployment`
- `cp -r jamba $MYMLCPATH/python/model`  
- Add the following code to `$MYMLCPATH/model/model.py`
```python
"jamba": Model(
      name="jamba",
      model=jamba_model.JambaForCausalLM,
      config=jamba_model.JambaConfig,
      source={
            "huggingface-torch": jamba_loader.huggingface,
            "huggingface-safetensor": jamba_loader.huggingface,
      },
      quantize={
            "no-quant": jamba_quantization.no_quant,
      },
    ),
```
## Current Deployment Status
- `mlc_llm convert_weight` works!
- `mlc_llm gen_config` works!
- `mlc_llm compile` fails :cry:
  - State space models' and their operations aren't natively supported within MLC LLM currently
  - We implement a custom forward pass recurrence for SSMs with using low-level `tvm.tir` operations
  - Huggingface models import properly, and compiling `embed` works
  - Compiling prefill and decode proved tough because the custom low-level operation don't fit well in the TensorIR compute graph with regular relax.frontend.nn operations
  - Currently the error is as follows:
    ```
    tvm._ffi.base.TVMError: Traceback (most recent call last):
    File "D:\a\package\package\tvm\src\relay\analysis\graph_partitioner.cc", line 464
    InternalError: Check failed: (group_node->pattern == kCommReduce) is false:
    ```
    
