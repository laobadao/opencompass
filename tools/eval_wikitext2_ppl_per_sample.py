"""Per-sample WikiText-2 perplexity evaluation script.

Mirrors the OpenCompass PPLOnlyInferencer logic: treat each non-empty line
of WikiText-2 as an independent sample, compute its cross-entropy loss,
and report the average.

This is NOT the standard sliding-window approach used in papers (see
eval_wikitext2_ppl.py for that).  It is provided to demonstrate how
the OpenCompass per-sample PPL pipeline works on WikiText-2 and to
make the difference in results visible.

Usage:

  python tools/eval_wikitext2_ppl_per_sample.py --model Qwen/Qwen3-1.7B

  python tools/eval_wikitext2_ppl_per_sample.py --model Qwen/Qwen3-1.7B \
      --split validation --batch-size 8
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description='Per-sample WikiText-2 PPL (OpenCompass style)')
    p.add_argument('--model', type=str, default='Qwen/Qwen3-1.7B',
                   help='HuggingFace model name or local path')
    p.add_argument('--tokenizer', type=str, default=None,
                   help='Tokenizer path (defaults to --model)')
    p.add_argument('--max-seq-len', type=int, default=2048,
                   help='Max sequence length for tokenization')
    p.add_argument('--batch-size', type=int, default=4,
                   help='Batch size for inference')
    p.add_argument('--split', type=str, default='test',
                   choices=['test', 'validation', 'train'])
    p.add_argument('--dtype', type=str, default='bfloat16',
                   choices=['float16', 'bfloat16', 'float32'])
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--dataset-path', type=str, default='wikitext')
    p.add_argument('--dataset-name', type=str, default='wikitext-2-raw-v1')
    p.add_argument('--min-tokens', type=int, default=2,
                   help='Skip samples with fewer tokens than this')
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

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f'Set pad_token_id = eos_token_id = {tokenizer.pad_token_id}')

    print(f'Loading model from {model_path} (dtype={dtype_str}) ...')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_ppl_batch(model, tokenizer, texts, max_seq_len, device):
    """Compute per-sample average CE loss, replicating HuggingFaceBaseModel.get_ppl().

    Returns a list of per-sample CE losses (one float per text).
    """
    pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    tokens = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=max_seq_len,
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        logits = model(**tokens)[0]

    batch_size, seq_len, vocab_size = logits.shape
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = tokens['input_ids'][:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=pad_token_id,
        reduction='none',
    ).view(batch_size, seq_len - 1)

    lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()
    ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
    return ce_loss.tolist()


def main():
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.model, args.tokenizer, args.dtype, args.device)
    device = next(model.parameters()).device

    print(f'Loading dataset {args.dataset_path}/{args.dataset_name} '
          f'split={args.split} ...')
    dataset = load_dataset(
        args.dataset_path, args.dataset_name, split=args.split)

    texts = [row['text'] for row in dataset if row['text'].strip()]
    print(f'Total lines: {len(dataset)}, non-empty: {len(texts)}')

    filtered = []
    skipped = 0
    for t in texts:
        n_tokens = len(tokenizer.encode(t, add_special_tokens=True))
        if n_tokens >= args.min_tokens:
            filtered.append(t)
        else:
            skipped += 1
    texts = filtered
    print(f'After filtering (min_tokens={args.min_tokens}): '
          f'{len(texts)} samples, skipped {skipped}')

    all_losses = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc='Evaluating'):
        batch = texts[i:i + args.batch_size]
        losses = get_ppl_batch(
            model, tokenizer, batch, args.max_seq_len, device)
        all_losses.extend(losses)

    avg_ce = np.mean(all_losses)
    ppl = math.exp(avg_ce)

    token_counts = [
        len(tokenizer.encode(t, add_special_tokens=True)) - 1
        for t in texts
    ]
    weighted_nll = sum(l * n for l, n in zip(all_losses, token_counts))
    weighted_ppl = math.exp(weighted_nll / sum(token_counts))

    print('\n' + '=' * 65)
    print(f'Model:              {args.model}')
    print(f'Dataset:            {args.dataset_path}/{args.dataset_name}')
    print(f'Split:              {args.split}')
    print(f'Samples evaluated:  {len(texts)}')
    print(f'Max seq len:        {args.max_seq_len}')
    print('-' * 65)
    print(f'[Per-sample avg]  Avg CE loss: {avg_ce:.4f}  PPL: {ppl:.2f}')
    print(f'[Token-weighted]  Avg CE loss: '
          f'{weighted_nll / sum(token_counts):.4f}  '
          f'PPL: {weighted_ppl:.2f}')
    print('=' * 65)
    print()
    print('NOTE: These results differ from the standard sliding-window PPL')
    print('reported in papers. Use tools/eval_wikitext2_ppl.py for that.')


if __name__ == '__main__':
    main()
