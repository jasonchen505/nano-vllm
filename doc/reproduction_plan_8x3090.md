# 8×3090 全流程复现 Plan
## 基于 nano-vllm 框架的完整实验计划

> **硬件资源**：8× RTX 3090，每卡 24GB VRAM，PCIe 4.0 互联（无 NVLink）
> **项目**：`/home/chenyizhou/nano-vllm`
> **目标**：完整跑通推理框架 → 理解每个模块 → 扩展改造 → 算法侧延伸

---

## 0. 硬件约束分析

### 0.1 RTX 3090 关键限制

| 参数 | 值 | 影响 |
|------|-----|------|
| **VRAM** | 24 GB | 限制单卡能跑的最大模型 |
| **互联** | PCIe 4.0（无 NVLink） | TP 通信带宽 ~32 GB/s，远低于 NVLink 的 600 GB/s |
| **Tensor Core** | FP16/BF16 35.6 TFLOPS | 计算能力足够 |
| **显存带宽** | 936 GB/s | decode 阶段 memory-bound 的瓶颈 |
| **CUDA Graph** | 支持 | 可以跑 nano-vllm 的 graph 路径 |

### 0.2 Qwen3 模型族内存预算（BF16）

| 模型 | 参数量 | 权重大小 | KV Cache/层/seq | 最大可并发 seq (4K) | 单卡可行？ | TP 需求 |
|------|--------|---------|----------------|-------------------|-----------|---------|
| Qwen3-0.6B | 0.6B | 1.2 GB | ~0.5 MB | ~500+ | ✅ 单卡绰绰有余 | TP=1 |
| Qwen3-1.7B | 1.7B | 3.4 GB | ~1.4 MB | ~300+ | ✅ 单卡够用 | TP=1 |
| Qwen3-4B | 4B | 8 GB | ~3.2 MB | ~150+ | ✅ 单卡刚好 | TP=1 |
| Qwen3-8B | 8B | 16 GB | ~6.4 MB | ~30+ | ⚠️ 单卡紧张，KV cache 很少 | TP=1/2 |
| Qwen3-14B | 14B | 28 GB | ~11 MB | ❌ 单卡放不下 | ❌ | TP=2 |
| Qwen3-32B | 32B | 64 GB | ~26 MB | ❌ | ❌ | TP=4 |
| Qwen3-72B | 72B | 144 GB | ~57 MB | ❌ | ❌ | TP=8 |

### 0.3 TP 可行性分析（PCIe 限制）

```
TP 通信量 = 2 × hidden_size × batch_size × dtype_bytes (per all_reduce)

Qwen3-8B: hidden=4096, TP=2
  all_reduce 通信量 = 2 × 4096 × bs × 2 = 16 KB × bs
  PCIe 4.0 x16: ~32 GB/s
  通信时间 = 16KB × 32 / 32GB/s ≈ 0.016 ms（可忽略）

但 decode 时每步都要通信，累积效应不可忽略。
结论：PCIe TP 对小模型（<14B）可接受，对大模型（32B+）会成为瓶颈。
```

**关键结论**：
- **TP=1**：Qwen3-0.6B/1.7B/4B 完全没问题，8B 勉强
- **TP=2**：Qwen3-14B 可行（PCIe 通信开销可接受）
- **TP=4**：Qwen3-32B 可行但通信开销明显
- **TP=8**：Qwen3-72B 可行但 PCIe 会成为严重瓶颈，吞吐不理想

---

## 1. 分阶段实验计划

### Phase 1：环境搭建 + 单卡基础跑通（Day 1-2）

#### 1.1 环境安装

```bash
# 创建 conda 环境
conda create -n nanovllm python=3.11 -y
conda activate nanovllm

# 安装 PyTorch（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install flash-attn --no-build-isolation  # 需要编译，约 10-20 分钟
pip install triton transformers xxhash

# 安装 nano-vllm
pip install -e /home/chenyizhou/nano-vllm
```

#### 1.2 下载模型

```bash
# 从 HuggingFace 下载（国内可用镜像）
# Phase 1 用最小模型
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False

# 后续 Phase 用更大模型
huggingface-cli download Qwen/Qwen3-1.7B \
  --local-dir ~/huggingface/Qwen3-1.7B/ \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen3-4B \
  --local-dir ~/huggingface/Qwen3-4B/ \
  --local-dir-use-symlinks False
```

#### 1.3 跑通 example.py

```bash
# 单卡 TP=1，enforce_eager=True（先不用 CUDA Graph）
python example.py
```

**验证点**：
- [ ] 模型加载成功
- [ ] Tokenizer 正确（检查 chat template）
- [ ] 生成结果合理（不是乱码）
- [ ] 显存使用 < 5 GB（0.6B 模型）

#### 1.4 跑通 bench.py

```bash
# 单卡 benchmark
python bench.py
```

**验证点**：
- [ ] 吞吐数字合理（0.6B 应该 > 1000 tok/s）
- [ ] 无 OOM
- [ ] 无 CUDA error

---

### Phase 2：单卡深度实验（Day 3-5）

#### 2.1 CUDA Graph 对比实验

**目的**：验证 CUDA Graph 对 decode 吞吐的提升

```python
# 修改 bench.py，分别跑 eager 和 graph
# 实验 A: enforce_eager=True
llm = LLM(path, enforce_eager=True, max_model_len=4096)

# 实验 B: enforce_eager=False
llm = LLM(path, enforce_eager=False, max_model_len=4096)
```

**记录指标**：
| 配置 | 输出 tokens | 时间 | 吞吐 (tok/s) | 预热时间 |
|------|-----------|------|-------------|---------|
| eager | 133,966 | ~?s | ~? | ~?s |
| graph | 133,966 | ~?s | ~? | ~?s |

**预期结论**：graph 模式下吞吐提升 10-20%，但首次 warmup + capture 会多花几秒。

#### 2.2 模型大小 vs 吞吐实验

**目的**：理解模型大小对推理性能的影响

```python
models = [
    ("Qwen3-0.6B", 1),   # TP=1
    ("Qwen3-1.7B", 1),   # TP=1
    ("Qwen3-4B", 1),     # TP=1
]
```

**记录指标**：
| 模型 | 权重大小 | 显存占用 | KV blocks | 吞吐 | Prefill tok/s |
|------|---------|---------|-----------|------|--------------|
| 0.6B | 1.2 GB | ~? GB | ~? | ~? | ~? |
| 1.7B | 3.4 GB | ~? GB | ~? | ~? | ~? |
| 4B | 8 GB | ~? GB | ~? | ~? | ~? |

**预期结论**：模型越大，单次 forward 越慢，但 KV cache 越少，并发能力越低。

#### 2.3 Batch Size vs 吞吐实验

**目的**：找到最优 batch size（吞吐 vs 延迟 trade-off）

```python
# 修改 bench.py 的 num_seqs
for num_seqs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    llm.generate(prompt_token_ids[:num_seqs], sampling_params[:num_seqs])
```

**预期结论**：batch size 越大吞吐越高，但单请求延迟也越高。存在一个"甜蜜点"。

#### 2.4 Prefix Cache 验证实验

**目的**：验证前缀缓存的效果

```python
# 构造共享 system prompt 的多个请求
system_prompt = "You are a helpful assistant." * 100  # ~500 tokens
user_queries = [f"Question {i}" for i in range(10)]

# 实验 A: 每次完整 prefill
# 实验 B: 开启 prefix cache（默认开启，但需要观察 hash_blocks 的命中率）
```

**验证方法**：在 `block_manager.py:110` 的 `hash_blocks()` 加日志：
```python
def hash_blocks(self, seq):
    ...
    for i in range(start, end):
        ...
        self.hash_to_block_id[h] = block.block_id
        print(f"[hash] seq={seq.seq_id} block={i} hash={h} "
              f"cached={h in old_hashes}")
```

#### 2.5 生成质量 vs 温度实验

**目的**：理解温度对生成质量的影响

```python
for temp in [0.1, 0.3, 0.6, 0.9, 1.2]:
    sp = SamplingParams(temperature=temp, max_tokens=256)
    outputs = llm.generate(["Explain PagedAttention"], sp)
    print(f"temp={temp}: {outputs[0]['text'][:200]}")
```

---

### Phase 3：张量并行实验（Day 6-8）

#### 3.1 TP=2 实验（Qwen3-8B）

**前提**：确认 `total_num_kv_heads % tp_size == 0`

```bash
# Qwen3-8B: num_kv_heads=8, tp_size=2 → 8%2=0 ✅
# Qwen3-8B: num_heads=32, tp_size=2 → 32%2=0 ✅

# 下载 Qwen3-8B
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir ~/huggingface/Qwen3-8B/ \
  --local-dir-use-symlinks False
```

```python
# 使用 2 张 GPU
llm = LLM("~/huggingface/Qwen3-8B/",
           tensor_parallel_size=2,
           enforce_eager=True)
```

**验证点**：
- [ ] NCCL 初始化成功（`tcp://localhost:2333`）
- [ ] 两个进程正确启动（`ps aux | grep python`）
- [ ] 生成结果正确（和 TP=1 结果一致）
- [ ] 显存使用均匀分布在 2 张卡上

**通信开销测量**：
```python
# 在 ModelRunner.run() 中加计时
t_comm = perf_counter()
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
t_comm_total = perf_counter() - t_comm
```

#### 3.2 TP=4 实验（Qwen3-14B）

```bash
# Qwen3-14B: num_kv_heads=8, tp_size=4 → 8%4=0 ✅
huggingface-cli download Qwen/Qwen3-14B \
  --local-dir ~/huggingface/Qwen3-14B/ \
  --local-dir-use-symlinks False
```

```python
llm = LLM("~/huggingface/Qwen3-14B/",
           tensor_parallel_size=4,
           enforce_eager=True)
```

**验证点**：
- [ ] 4 个进程正确启动
- [ ] 每卡显存 ~7 GB（28 GB / 4）
- [ ] 生成结果正确

#### 3.3 TP=8 实验（Qwen3-32B 或 72B）

```bash
# Qwen3-32B: num_kv_heads=8, tp_size=8 → 8%8=0 ✅
huggingface-cli download Qwen/Qwen3-32B \
  --local-dir ~/huggingface/Qwen3-32B/ \
  --local-dir-use-symlinks False
```

```python
llm = LLM("~/huggingface/Qwen3-32B/",
           tensor_parallel_size=8,
           enforce_eager=True)
```

**预期问题**：
- PCIe 通信开销明显，吞吐可能不理想
- 启动时间长（8 个进程同步初始化）
- 如果有其他进程占用 GPU，可能 OOM

#### 3.4 TP 通信开销对比

| 模型 | TP | 单步通信量 | 预估通信时间 | 占 decode 时间比例 |
|------|-----|-----------|-------------|------------------|
| 8B | 2 | 16 KB × bs | ~0.05 ms | ~5% |
| 14B | 4 | 22 KB × bs | ~0.1 ms | ~8% |
| 32B | 8 | 26 KB × bs | ~0.2 ms | ~15% |
| 72B | 8 | 40 KB × bs | ~0.3 ms | ~20% |

---

### Phase 4：代码改造实验（Day 9-14）

#### 4.1 添加 Top-P 采样

**目的**：扩展采样器，支持 nucleus sampling

**修改文件**：`nanovllm/layers/sampler.py`

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures, top_ps=None):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        
        if top_ps is not None:
            # 排序
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # 找到 cumsum > top_p 的位置
            mask = cumulative_probs - sorted_probs > top_ps.unsqueeze(1)
            sorted_probs[mask] = 0.0
            # 重新归一化
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            # 恢复原始顺序
            probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)
        
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

**同时修改**：
- `sampling_params.py`：添加 `top_p: float = 1.0`
- `engine/sequence.py`：存储 `top_p`
- `engine/model_runner.py`：`prepare_sample()` 传递 `top_ps`

**验证**：对比 top_p=0.9 vs top_p=1.0 的生成多样性

#### 4.2 添加 Greedy Decoding

**目的**：支持 deterministic 生成

```python
# 在 SamplingParams 中
temperature: float = 1.0

# 在 Sampler 中
if temperature < 1e-10:
    return logits.argmax(dim=-1)  # greedy
```

**验证**：temperature=0 时，相同 prompt 应始终生成相同结果

#### 4.3 添加 Sliding Window Attention

**目的**：支持 Mistral 风格的滑动窗口注意力

**修改**：`layers/attention.py`，在 flash_attn 调用中添加 `window_size` 参数

#### 4.4 添加日志系统

**目的**：在 Scheduler/BlockManager 中添加结构化日志

```python
import logging

logger = logging.getLogger("nanovllm")

# 在 Scheduler.schedule() 中
logger.info(f"[schedule] prefill={is_prefill} "
            f"num_seqs={len(scheduled_seqs)} "
            f"num_tokens={num_batched_tokens}")

# 在 BlockManager.can_allocate() 中
logger.info(f"[allocate] seq={seq.seq_id} "
            f"cached_blocks={num_cached_blocks} "
            f"new_blocks={num_new_blocks} "
            f"free_blocks={len(self.free_block_ids)}")
```

---

### Phase 5：后训练实验（Day 15-21）

#### 5.1 SFT 数据准备

**目的**：用 Qwen3-0.6B 做 SFT，验证后训练流程

```bash
# 准备 SFT 数据（100 条 QA 对）
# 格式：{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**使用工具**：LLaMA-Factory 或 transformers + trl

```bash
pip install llmtuner
llmtuner train --model_name_or_path ~/huggingface/Qwen3-0.6B/ \
  --dataset sft_data \
  --output_dir ./sft_output \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-5
```

#### 5.2 评估 SFT 效果

```python
# 对比 SFT 前后的生成质量
prompts = ["What is PagedAttention?", "Explain continuous batching"]
for prompt in prompts:
    # Base model
    base_output = base_llm.generate([prompt], sp)
    # SFT model
    sft_output = sft_llm.generate([prompt], sp)
    print(f"Base: {base_output[0]['text'][:200]}")
    print(f"SFT:  {sft_output[0]['text'][:200]}")
```

#### 5.3 DPO 数据准备与训练

**目的**：验证 DPO 流程

```bash
# 准备偏好数据
# 格式：{"prompt": "...", "chosen": "...", "rejected": "..."}

pip install trl
# 使用 trl 的 DPOTrainer
```

#### 5.4 用 nano-vllm 做 Rollout（RLHF 场景）

**目的**：用 nano-vllm 的 generate() 作为 RLHF 的 rollout engine

```python
# 伪代码：RLHF rollout
from nanovllm import LLM, SamplingParams

actor = LLM(actor_path, enforce_eager=False)
ref = LLM(ref_path, enforce_eager=True)  # reference model

for batch in prompt_batches:
    # Actor rollout
    actor_outputs = actor.generate(batch, SamplingParams(temperature=0.7))
    # Reference log probs
    ref_outputs = ref.generate(batch, SamplingParams(temperature=0.0))
    # 计算 KL 惩罚
    # ...
```

**验证点**：
- [ ] Actor 和 Reference 模型可以同时加载（不同 GPU）
- [ ] Rollout 吞吐足够高
- [ ] 可以计算 log prob（需要修改 generate() 返回 logits）

---

### Phase 6：Agent 场景实验（Day 22-28）

#### 6.1 多轮对话 + Prefix Cache 验证

**目的**：验证 Agent 多轮对话场景下 prefix cache 的效果

```python
system_prompt = "You are a helpful coding assistant..."  # 2K tokens
conversations = [
    [{"role": "system", "content": system_prompt},
     {"role": "user", "content": "Write a Python function to sort a list"},
     {"role": "assistant", "content": "..."},
     {"role": "user", "content": "Now add error handling"}],
    # ... 更多对话
]
```

**验证**：system prompt 的 KV cache 在第 2 轮被复用

#### 6.2 工具调用格式的生成

**目的**：验证模型能否生成结构化的 tool call JSON

```python
prompt = """You have access to:
- get_weather(city: str) -> dict
- search(query: str) -> list

User: What's the weather in Beijing?
Assistant: """

output = llm.generate([prompt], SamplingParams(temperature=0.1, max_tokens=256))
# 检查输出是否包含正确的 JSON 格式 tool call
```

---

## 2. 实验记录模板

每个实验记录以下内容：

```markdown
### 实验 X：[实验名称]

**日期**：YYYY-MM-DD
**假设**：[你预期会发生什么]
**配置**：
- 模型：
- GPU：
- TP：
- Batch Size：
- 其他参数：

**结果**：
| 指标 | 值 |
|------|-----|
| 吞吐 | |
| 延迟 | |
| 显存 | |

**分析**：
[为什么结果是这样的？和预期一致吗？]

**学到的点**：
[增量新知识]
```

---

## 3. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| flash-attn 编译失败 | 中 | 用 `pip install flash-attn --no-build-isolation`，或降级 PyTorch 版本 |
| NCCL 初始化失败 | 低 | 检查 `NCCL_DEBUG=INFO`，确认端口 2333 没被占用 |
| PCIe TP 吞吐太低 | 高 | 3090 无 NVLink，TP>2 时通信开销大，记录实际数据作为学习点 |
| 大模型 OOM | 中 | 调低 `gpu_memory_utilization` 或 `max_model_len` |
| CUDA Graph capture 失败 | 低 | 用 `enforce_eager=True` 绕过 |

---

## 4. 时间线总结

```
Day 1-2:   环境搭建 + 单卡跑通 Qwen3-0.6B
Day 3-5:   单卡深度实验（Graph/BS/Prefix/温度）
Day 6-8:   TP 实验（8B TP=2 → 14B TP=4 → 32B TP=8）
Day 9-14:  代码改造（Top-P/Greedy/Sliding Window/日志）
Day 15-21: 后训练实验（SFT → DPO → RLHF rollout）
Day 22-28: Agent 场景（多轮对话/工具调用/性能测试）
```

**总计 4 周，每天 2-3 小时，共约 70 小时。**
