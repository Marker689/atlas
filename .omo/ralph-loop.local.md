---
active: true
iteration: 3
max_iterations: 500
completion_promise: "VERIFIED"
initial_completion_promise: "DONE"
verification_attempt_id: "20d420ab-eca0-48f4-8f64-31663221d75f"
verification_session_id: "ses_14b9fc6b4ffeeTUcIRn35Z20ib"
started_at: "2026-06-11T01:22:50.153Z"
session_id: "ses_14bb9a6e1ffeAF63Aa0JTOTiYW"
ultrawork: true
verification_pending: true
strategy: "continue"
message_count_at_start: 0
---
So, we are continuing now working on adding support for prismaquant https://github.com/RobTand/prismaquant to atlas. We have 3 models, all qwens have visual encoder and mtp support (checked in vllm). The minimum goal for now is correcly loaded prismaquanted model with working mtp. For model info u can freely go to huggingface by its prefix/name. So the logs: marker@gx10-4837:~/atlas$ ./atlas.sh -m 1
┌─ Atlas ──────────────────────────────────────────────
│ Model : rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm
│ Port  : 8000
│ Parser: qwen3_coder
│ Name  : qwen-coder
└───────────────────────────────────────────────────────
2026-06-11T01:09:49.442562Z  INFO spark::main_modules::serve: Atlas Spark starting...
2026-06-11T01:09:49.442574Z  INFO spark::main_modules::serve: Licensed under AGPL-3.0-only — see /LICENSE in this container
2026-06-11T01:09:49.443856Z  INFO spark::model_resolver: Model: rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm (resolved to /root/.cache/huggingface/hub/models--rdtand--Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm/snapshots/9b5389d4a1e207daab2d47732efea57d7e946dcf)
2026-06-11T01:09:49.443861Z  INFO spark::main_modules::serve: Port: 8000
2026-06-11T01:09:49.443862Z  INFO spark::main_modules::serve: SSM decode dtype: f32 (full precision)
2026-06-11T01:09:49.444552Z  INFO spark::main_modules::serve: Quantization config: method="compressed-tensors", algo="NVFP4", format="mixed-precision", 175 module(s) in ignore list
2026-06-11T01:09:49.444555Z  INFO spark::main_modules::serve: Model config: 64 layers, 16 attention, 48 SSM, 0 experts, rope_theta=10000000, head_dim=256, rotary_dim=64
2026-06-11T01:09:49.444807Z  INFO spark::main_modules::serve: Selected kernel target: (sm_121, qwen3.6-27b, nvfp4) (134 modules) — quant compat: kernel=nvfp4 model=nvfp4 OK
2026-06-11T01:10:02.822137Z  INFO spark_runtime::cuda_backend: AtlasCudaBackend initialized on GPU 0 with 134 PTX modules
2026-06-11T01:10:02.822239Z  INFO spark::main_modules::serve_phases::preflight: GPU 0: 121.6 GB total, 117.0 GB free
2026-06-11T01:10:02.822289Z  INFO spark::main_modules::serve_phases::preflight: Preflight reserve: inference=17605 MB, buffer_arena=1497 MB (pre-load free: 117.0 GB)
2026-06-11T01:10:02.822332Z  INFO spark::main_modules::serve: OOM watchdog started (threshold: 2 GB, interval: 2s)
2026-06-11T01:10:02.822366Z  INFO spark::main_modules::serve: OOM guard reserve: 4096 MB
2026-06-11T01:10:02.822368Z  INFO spark::main_modules::serve_phases::weights: Using fast weight loader (O_DIRECT + pipelined read/copy)
2026-06-11T01:10:02.829853Z  INFO spark_runtime::fast_weights: Fast-load pre-flight: 18.79 GB on-disk, 1.3x overhead = 24.42 GB peak, 116.97 GB free, 4.0 GB reserve (FP8: false)
2026-06-11T01:10:02.829891Z  INFO spark_runtime::fast_weights: Fast-loading shard 1/4: model-00001-of-00004.safetensors (56 tensors)
2026-06-11T01:10:04.496707Z  INFO spark_runtime::fast_weights:   Shard 1/4 done — GPU memory: 5.21 GB used, 111.76 GB free
2026-06-11T01:10:04.496816Z  INFO spark_runtime::fast_weights: Fast-loading shard 2/4: model-00002-of-00004.safetensors (633 tensors)
2026-06-11T01:10:05.940570Z  INFO spark_runtime::fast_weights:   Shard 2/4 done — GPU memory: 10.29 GB used, 106.68 GB free
2026-06-11T01:10:05.940706Z  INFO spark_runtime::fast_weights: Fast-loading shard 3/4: model-00003-of-00004.safetensors (776 tensors)
2026-06-11T01:10:07.295653Z  INFO spark_runtime::fast_weights:   Shard 3/4 done — GPU memory: 15.38 GB used, 101.59 GB free
2026-06-11T01:10:07.295827Z  INFO spark_runtime::fast_weights: Fast-loading shard 4/4: model-00004-of-00004.safetensors (1222 tensors)
2026-06-11T01:10:08.367765Z  INFO spark_runtime::fast_weights:   Shard 4/4 done — GPU memory: 19.18 GB used, 97.80 GB free
2026-06-11T01:10:08.367847Z  INFO spark_runtime::fast_weights: Fast-loaded 2687 weight tensors
2026-06-11T01:10:08.367992Z  INFO spark::main_modules::serve_phases::weights: Loaded 2687 weight tensors
2026-06-11T01:10:08.367995Z  INFO spark::main_modules::serve_phases::weights: Weight prefix: model.language_model
2026-06-11T01:10:08.368485Z  INFO spark_model::preflight: Pre-flight checks passed
2026-06-11T01:10:08.368617Z  INFO spark_model::quant_format: QuantFormat: compressed-tensors (format="mixed-precision"), 175 ignored module(s), 1 config group(s)
2026-06-11T01:10:08.368709Z  INFO spark::main_modules::serve: Quantization format: compressed-tensors (base variant CompressedTensors), ignored globs = 175
2026-06-11T01:10:08.368757Z  INFO spark::main_modules::serve_phases::preflight: GDN chunked prefill reserve: 177 MB (chunk_size=4096, max_seq_len=32000)
2026-06-11T01:10:08.368760Z  INFO spark::main_modules::serve_phases::preflight: Weights: 18.79 GB, estimated free: 98.2 GB, actual free: 97.8 GB (reserve: 17605 MB)
2026-06-11T01:10:08.368794Z  INFO spark::main_modules::serve_phases::kv_cache:
