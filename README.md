Tous les changements sont dans `examples/nvfp4`.

Run with
```bash
bazel run --config=release //examples/nvfp4 -- <file> [scalar-lookup|simd-shift] <nthreads>
```

For instance with [Qwen3-0.6B-FP4](https://huggingface.co/NVFP4/Qwen3-0.6B-FP4)
```bash
bazel run --config=release //examples/nvfp4 -- $PWD/Qwen3-0.6B-FP4.safetensors scalar-lookup 4

```
