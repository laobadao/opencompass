"""Standalone WikiText-2 perplexity evaluation script.

Standard sliding-window PPL computation following the approach used in
GPT-2, LLaMA, and Qwen papers.  The full test set is concatenated into a
single token sequence and then evaluated with a strided sliding window.

Usage examples:

  # Qwen3-1.7B, default settings (seq_len=2048, stride=2048, test split)
  python tools/eval_wikitext2_ppl.py --model Qwen/Qwen3-1.7B

  # Custom sequence length and stride
  python tools/eval_wikitext2_ppl.py --model Qwen/Qwen3-1.7B \
      --seq-len 4096 --stride 2048

  # Use validation split
  python tools/eval_wikitext2_ppl.py --model Qwen/Qwen3-1.7B --split validation
"""

import argparse
import math

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description='Compute WikiText-2 perplexity for a HuggingFace causal LM')
    p.add_argument('--model', type=str, default='Qwen/Qwen3-1.7B',
                   help='HuggingFace model name or local path')
    p.add_argument('--tokenizer', type=str, default=None,
                   help='Tokenizer path (defaults to --model)')
    p.add_argument('--seq-len', type=int, default=2048,
                   help='Context window length for sliding window')
    p.add_argument('--stride', type=int, default=None,
                   help='Stride for sliding window (defaults to --seq-len)')
    p.add_argument('--split', type=str, default='test',
                   choices=['test', 'validation', 'train'],
                   help='WikiText-2 split to evaluate on')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Batch size (only 1 is supported for strided eval)')
    p.add_argument('--dtype', type=str, default='bfloat16',
                   choices=['float16', 'bfloat16', 'float32'],
                   help='Model dtype')
    p.add_argument('--device', type=str, default='auto',
                   help='Device map for model loading')
    p.add_argument('--dataset-path', type=str, default='wikitext',
                   help='HuggingFace dataset path')
    p.add_argument('--dataset-name', type=str, default='wikitext-2-raw-v1',
                   help='HuggingFace dataset config name')
    return p.parse_args()


def load_model_and_tokenizer(model_path, tokenizer_path, dtype_str, device):
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map[dtype_str]

    print(f'Loading tokenizer from {tokenizer_path or model_path} ...')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or model_path, trust_remote_code=True)

    print(f'Loading model from {model_path} (dtype={dtype_str}) ...')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def prepare_encodings(tokenizer, dataset_path, dataset_name, split):
    """Load WikiText-2 and tokenize the entire split into one long sequence."""
    print(f'Loading dataset {dataset_path}/{dataset_name} split={split} ...')
    dataset = load_dataset(dataset_path, dataset_name, split=split)

    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    return encodings


def compute_ppl(model, encodings, seq_len, stride, device):
    """Sliding-window perplexity following the Hugging Face standard recipe.

    For each window [begin, end), only the non-overlapping tail tokens
    contribute to the loss (via -100 label masking).  This avoids
    double-counting tokens when stride < seq_len.
    """
    input_ids = encodings['input_ids']
    total_len = input_ids.size(1)

    print(f'Total tokens: {total_len}, seq_len: {seq_len}, stride: {stride}')
    num_windows = max(1, math.ceil((total_len - seq_len) / stride) + 1)
    print(f'Number of windows: {num_windows}')

    nlls = []
    total_tokens = 0
    prev_end = 0

    for begin in tqdm(range(0, total_len, stride), desc='Evaluating'):
        end = min(begin + seq_len, total_len)
        trg_len = end - prev_end

        window_ids = input_ids[:, begin:end].to(device)
        target_ids = window_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(window_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_target_tokens = (target_ids[:, 1:] != -100).sum().item()
        if num_target_tokens > 0:
            nlls.append(neg_log_likelihood.item() * num_target_tokens)
            total_tokens += num_target_tokens

        prev_end = end
        if end == total_len:
            break

    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, total_tokens


def main():
    args = parse_args()
    stride = args.stride if args.stride is not None else args.seq_len

    model, tokenizer = load_model_and_tokenizer(
        args.model, args.tokenizer, args.dtype, args.device)

    device = next(model.parameters()).device

    encodings = prepare_encodings(
        tokenizer, args.dataset_path, args.dataset_name, args.split)

    ppl, avg_nll, total_tokens = compute_ppl(
        model, encodings, args.seq_len, stride, device)

    print('\n' + '=' * 60)
    print(f'Model:        {args.model}')
    print(f'Dataset:      {args.dataset_path}/{args.dataset_name}')
    print(f'Split:        {args.split}')
    print(f'Seq length:   {args.seq_len}')
    print(f'Stride:       {stride}')
    print(f'Total tokens: {total_tokens}')
    print(f'Avg NLL:      {avg_nll:.4f}')
    print(f'Perplexity:   {ppl:.2f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
