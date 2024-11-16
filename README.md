# Edge LLM Contest Submission - Training from Scratch

[![C4 Curated Dataset](https://img.shields.io/badge/🤗%20Dataset-C4%20Curated-blue.svg)](https://huggingface.co/datasets/neeleshg23/c4_curated) [![Jamba 1.9B](https://img.shields.io/badge/🤗%20Model-Jamba%201.9B-yellow.svg)](https://huggingface.co/neeleshg23/jamba-1.9b-8)

## Data Preprocessing 
- Dependencies: Nvidia NeMo Data Curator, Nvidia CUDA GPU, and ~1TB disk space
- Given `allenai/c4`, run cleaning pipeline adapted from [RedPajama sample clean script](https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/pretraining-data-curation/red-pajama-v2-curation-tutorial.ipynb)

Data Preprocessing Steps:
- - language extraction
- - exact deduplication
- - fuzzy deduplication via jaccard similarity, jaccard compute, and connected components 
- - quality heuristic filtering

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
- Add the following code to `MYMLCPATH/model/model_preset.py`
```
    "jamba-1.9b": {
        "_name_or_path": "neeleshg23/jamba-1.9b-6",
        "architectures": [
            "JambaForCausalLM"
        ],
        "attention_dropout": 0.0,
        "attn_layer_offset": 4,
        "attn_layer_period": 8,
        "bos_token_id": 1,
        "d_model": 1024,
        "eos_token_id": 2,
        "expert_layer_offset": 1,
        "expert_layer_period": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "mamba_conv_bias": "true",
        "mamba_d_conv": 4,
        "mamba_d_state": 16,
        "mamba_dt_rank": 256,
        "mamba_expand": 2,
        "mamba_proj_bias": "false",
        "max_position_embeddings": 2048,
        "model_type": "jamba",
        "num_attention_heads": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 3,
        "num_key_value_heads": 8,
        "num_logits_to_keep": 1,
        "output_router_logits": "false",
        "pad_token_id": 0,
        "rms_norm_eps": 1e-06,
        "router_aux_loss_coef": 0.001,
        "sliding_window": "null",
        "tie_word_embeddings": "true",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.46.2",
        "use_cache": "true",
        "use_mamba_kernels": "true",
        "vocab_size": 128256
        },
```
