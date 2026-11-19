"""Benchmark the native miner offline with synthetic inputs and no pool traffic.

Run from any directory: python /path/to/checkout/tools/benchmark_miner.py
Default timing reuses the header-dependent right operand, as the runtime does.
Use --uncached for preparation without reuse, or --stages to instrument native
operations individually (additional synchronizations affect overall timing).
Compilation and warmup are excluded from measured iterations. Hashrate counts
one operation for each m*n*k product, matching the runtime's rate accounting.
"""
import argparse
from functools import wraps
import json
from pathlib import Path
import statistics
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--k', type=int, default=16384)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--variant', type=int, choices=range(8), default=0,
                        help='transcript kernel variant; 0 selects automatically, 7 is the legacy large kernel')
    parser.add_argument('--uncached', action='store_true', help='prepare both operands for each batch')
    parser.add_argument('--stages', action='store_true', help='time native operations separately; adds synchronization overhead')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error('--warmup must be nonnegative and --iterations must be positive')
    return args


def main():
    args = arguments()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    import numpy as np
    import torch
    from LRAEP_trainer import native_cuda, protocol
    from LRAEP_trainer.protocol import TrainingConfig, load_official_helpers
    from LRAEP_trainer.wr_protocol import WRJob
    from LRAEP_trainer.wr_work import CapturedBatchPreparation, prepare_wr_batch

    cfg = TrainingConfig(m=args.m, n=args.n, k=args.k, tile_h=8, tile_w=8)
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required for this benchmark.')
    load_official_helpers(root / 'src' / 'LRAEP_helpers')
    compute = getattr(protocol, 'compute_batch', None) or protocol.search_batch
    # Synthetic valid header and deterministic nonce seeds; no user configuration
    # or identity is read and no connection or submission is made.
    header = bytes(72) + (0x207fffff).to_bytes(4, 'little')
    job = WRJob('offline-benchmark', header, 0, 3, 1,
                      update_payload=bytes(200), device_seed=bytes(range(8)))
    rng = np.random.default_rng(73)
    device = torch.device('cuda:0')
    native_cuda.load_native(device)
    preparation = None if args.uncached else CapturedBatchPreparation(cfg, device)
    saved_transcripts = native_cuda.transcripts_device
    saved_tail_root = native_cuda.ConstantTailTree.root
    if args.variant:
        def selected_transcripts(*values, **kwargs):
            kwargs['variant'] = args.variant
            return saved_transcripts(*values, **kwargs)
        native_cuda.transcripts_device = selected_transcripts
    stage_samples = {}
    originals = {}

    def instrument(name, function):
        @wraps(function)
        def measured(*values, **kwargs):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = function(*values, **kwargs)
            torch.cuda.synchronize(device)
            stage_samples.setdefault(name, []).append((time.perf_counter() - start) * 1000)
            return result
        return measured

    if args.stages:
        native_cuda.ConstantTailTree.root = instrument('constant_tail_root', saved_tail_root)
        for name in ('merkle_root', 'constant_merkle_root', 'constant_noised_operand',
                     'add_noise', 'transcripts_device', 'hash_candidates'):
            originals[name] = getattr(native_cuda, name)
            setattr(native_cuda, name, instrument(name, originals[name]))

    samples = []
    print(json.dumps({'gpu': torch.cuda.get_device_name(device), 'shape': [cfg.m, cfg.n, cfg.k],
                      'tile': [8, 8], 'variant': args.variant, 'cached': not args.uncached,
                      'warmup': args.warmup, 'iterations': args.iterations,
                      'stages': args.stages}), flush=True)
    try:
        for index in range(args.warmup + args.iterations):
            if index == args.warmup:
                stage_samples.clear()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            batch = prepare_wr_batch(job, cfg, rng, device=device, native=True,
                                           preparation=preparation)
            torch.cuda.synchronize(device)
            prepared = time.perf_counter()
            candidates = list(compute(batch, 1, device, lambda: False))
            torch.cuda.synchronize(device)
            end = time.perf_counter()
            if candidates:
                raise RuntimeError('Unexpected candidate at the synthetic acceptance threshold')
            values = {'prepare_ms': (prepared - start) * 1000,
                      'compute_ms': (end - prepared) * 1000,
                      'total_ms': (end - start) * 1000,
                      'TH/s': cfg.m * cfg.n * cfg.k / (end - start) / 1e12}
            print(json.dumps({'iteration': index, 'warmup': index < args.warmup, **values}), flush=True)
            if index >= args.warmup:
                samples.append(values)
            batch = candidates = None
        median = {key: statistics.median(sample[key] for sample in samples) for key in samples[0]}
        total_seconds = sum(sample['total_ms'] for sample in samples) / 1000
        print(json.dumps({'median': median,
                          'aggregate_TH/s': cfg.m * cfg.n * cfg.k * len(samples) / total_seconds / 1e12}), flush=True)
        if args.stages:
            print(json.dumps({'stages': {name: {'calls': len(values), 'total_ms': sum(values),
                                               'median_ms': statistics.median(values)}
                                        for name, values in stage_samples.items()}}), flush=True)
    finally:
        for name, function in originals.items():
            setattr(native_cuda, name, function)
        native_cuda.transcripts_device = saved_transcripts
        native_cuda.ConstantTailTree.root = saved_tail_root
        if preparation is not None:
            preparation.clear()


if __name__ == '__main__':
    main()
