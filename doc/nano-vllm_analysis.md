# nano-vllm 深度解析与面试复现指南

> **目标读者**：2028级研一学生，申请 LLM 算法/推理系统实习
> **项目地址**：`/home/chenyizhou/nano-vllm`
> **参考对象**：vLLM (NVIDIA/UC Berkeley 维护，工业界标杆推理引擎)

---

## 1. 项目定位与核心价值

### 1.1 nano-vllm 是什么？

| 维度 | 说明 |
|------|------|
| **代码量** | ~1,200 行 Python + 少量 Triton |
| **实现范围** | 离线推理核心流程：连续批处理、PagedAttention KV 缓存、前缀缓存、CUDA Graph 捕获、张量并行 |
| **支持模型** | Qwen3 系列（架构可扩展） |
| **性能** | RTX 4070 Laptop 上 Qwen3-0.6B 达 1,434 tok/s，**略超 vLLM 官方实现**（1,362 tok/s） |
| **设计哲学** | "极简教学级实现" —— 委托 FlashAttention 库、torch.compile、标准 NCCL，**不造轮子** |

### 1.2 为什么值得研一学生深度研究？

| 面试考点 | nano-vllm 对应模块 | 可展开深度 |
|----------|-------------------|-----------|
| **KV 缓存管理** | `BlockManager` + `Sequence.block_table` | PagedAttention 原理、前缀缓存哈希、写时复制 |
| **连续批处理调度** | `Scheduler.schedule()` | Prefill/Decode 交织、Chunked Prefill、抢占策略 |
| **张量并行** | `linear.py` (Col/Row/QKV/Merged) | Megatron-LM 切分策略、all-reduce 通信开销 |
| **注意力内核** | `attention.py` + FlashAttention | Varlen/KVCache 两种模式、Block Table 机制 |
| **CUDA Graph** | `model_runner.capture_cudagraph()` | 图捕获条件、内存池共享、回放时输入拷贝 |
| **推理加速技巧** | `embed_head.py` (仅最后 token) | Prefill 阶段只算最后位置 logits、减少通信量 |

---

## 2. 端到端推理流水线全景图

```
用户调用 LLM.generate(prompts, sampling_params)
        │
        ▼
┌─────────────────────────────────────┐
│ LLMEngine (llm_engine.py)           │
│  - tokenizer.encode()               │
│  - 创建 Sequence 对象               │
│  - Scheduler.add(seq) 入队          │
└─────────────────────────────────────┘
        │
        ▼
    while not scheduler.is_finished():
        │
        ├─▶ step() ──────────────────────────────┐
        │                                        │
        │  1. Scheduler.schedule()               │
        │     ├─ Prefill: 从 waiting 队列取       │
        │     │    BlockManager.can_allocate()   │
        │     │    BlockManager.allocate()       │
        │     │    chunked prefill (仅首 seq)    │
        │     └─ Decode: 从 running 队列取        │
        │          BlockManager.can_append()     │
        │          无块则 preempt 回 waiting     │
        │                                        │
        │  2. ModelRunner.run(seqs, is_prefill)  │
        │     ├─ prepare_prefill/decode()        │
        │     │   构建 input_ids, positions,     │
        │     │   cu_seqlens, slot_mapping,      │
        │     │   block_tables                   │
        │     │   set_context() 传给 attention   │
        │     │                                  │
        │     ├─ run_model()                     │
        │     │   ├─ Eager: 直接 forward         │
        │     │   └─ CUDA Graph: replay()        │
        │     │                                  │
        │     │   Qwen3ForCausalLM.forward()     │
        │     │   ├─ embed_tokens                │
        │     │   ├─ 28 层 DecoderLayer          │
        │     │   │   ├─ RMSNorm (fused residual)│
        │     │   │   ├─ Attention               │
        │     │   │   │   ├─ QKV 并行投影        │
        │     │   │   │   ├─ Q/K RMSNorm (Qwen3) │
        │     │   │   │   ├─ RoPE                │
        │     │   │   │   ├─ FlashAttention      │
        │     │   │   │   │   ├─ store_kvcache   │
        │     │   │   │   │   └─ varlen/kvcache  │
        │     │   │   │   └─ O proj (RowParallel)│
        │     │   │   ├─ RMSNorm (fused)         │
        │     │   │   └─ MLP (SiLU+Mul)          │
        │     │   └─ final RMSNorm               │
        │     │                                  │
        │     ├─ Sampler(logits, temps)          │
        │     │   温度缩放 → softmax → Gumbel-max│
        │     └─ reset_context()                 │
        │                                        │
        │  3. Scheduler.postprocess()            │
        │     ├─ BlockManager.hash_blocks()      │
        │     ├─ seq.append_token()              │
        │     ├─ EOS/max_tokens → FINISHED       │
        │     └─ deallocate finished seqs        │
        │                                        │
        │  4. 收集输出，更新 throughput 进度条     │
        │                                        │
        └────────────────────────────────────────┘
        │
        ▼
tokenizer.decode() → 返回文本
```

---

## 3. 核心模块深度解析

### 3.1 调度器 (`engine/scheduler.py`)

**核心职责**：把 `WAITING/RUNNING` 状态的 `Sequence` 组装成批次，决定每步是 prefill 还是 decode。

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # Phase 1: Prefill (优先)
    while self.waiting and len(scheduled) < self.max_num_seqs:
        seq = self.waiting[0]
        # 1. 检查 KV 缓存块是否足够（含前缀缓存命中）
        num_cached = self.block_manager.can_allocate(seq)  # -1 表示不够
        if num_cached == -1: break
        
        # 2. 计算本轮要处理的 token 数（支持 chunked prefill）
        num_tokens = seq.num_tokens - num_cached * block_size
        if remaining < num_tokens and scheduled: break  # 仅首 seq 允许分块
        
        # 3. 分配块，标记 scheduled
        self.block_manager.allocate(seq, num_cached)
        seq.num_scheduled_tokens = min(num_tokens, remaining)
        
        # 4. 整个 prompt 处理完 → 移入 running
        if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
            seq.status = RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        scheduled.append(seq)
    
    if scheduled: return scheduled, True  # is_prefill=True

    # Phase 2: Decode
    while self.running and len(scheduled) < self.max_num_seqs:
        seq = self.running.popleft()
        # 需要新块时检查空闲，不够则 preempt 最后一个 running
        while not self.block_manager.can_append(seq):
            if self.running: self.preempt(self.running.pop())
            else: self.preempt(seq); break
        else:
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)  # 跨块边界才分配
            scheduled.append(seq)
    self.running.extendleft(reversed(scheduled))
    return scheduled, False
```

**面试必问点**：
- **Chunked Prefill 为什么只允许首个序列？** 避免 pipeline bubble，保证 decode 不被长 prefill 无限阻塞
- **Preempt 策略为什么是 LIFO (pop 最后)？** 最后进入的通常 context 最短，丢弃代价最小
- **`max_num_batched_tokens` 如何同时约束 prefill 和 decode？** Prefill 累加 `num_scheduled_tokens`，Decode 每序列固定 1，统一在 budget 内

---

### 3.2 KV 缓存块管理 (`engine/block_manager.py`)

**设计亮点**：**单类完成** 物理块池、哈希索引、引用计数、前缀缓存。

```python
class Block:
    block_id: int
    ref_count: int      # 共享时 >1
    hash: int           # xxhash64 滚动哈希
    token_ids: list[int]

class BlockManager:
    blocks: list[Block]
    free_block_ids: deque[int]
    used_block_ids: set[int]
    hash_to_block_id: dict[intDiff[int, int]  # 内容哈希 → 物理块 ID
```

**关键算法：滚动哈希前缀匹配**

```python
def can_allocate(self, seq) -> int:
    h = -1
    num_cached = 0
    for i in range(seq.num_blocks - 1):  # 最后一个块可能不满，不缓存
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)  # 滚动 xxhash64
        block_id = self.hash_to_block_id.get(h, -1)
        # 关键：哈希命中后还要逐 token 比对，防止 xxhash 碰撞
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1  # 已被占用，仅增加 ref_count 即可
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached

def hash_blocks(self, seq):
    """每次 postprocess 时调用，把新填入的 block 哈希化进索引"""
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```

**xxhash 滚动链式哈希**：
```
h_0 = xxhash64(tokens_0)
h_1 = xxhash64(tokens_1, prefix=h_0)   # 把 h_0 的 8 字节拼接到输入
h_2 = xxhash64(tokens_2, prefix=h_1)
```
父块 hash 进子块的输入，使得"块 i 的 hash 唯一决定"——同前缀一定同 hash，不同前缀一定不同 hash。

**面试必问点**：
- **为什么仍然要做 `token_ids != expected` 校验？** xxhash 不是密码学哈希，碰撞概率虽低但非零；强校验保证语义正确
- **为什么最后一个 block 不参与哈希？** 末尾 block 可能不满 256 token，不应作为"完整前缀"被缓存
- **`can_append` 为何用 `len(seq) % self.block_size == 1` 判断？** 跨块边界的 decode（刚要写入新块的第 1 个 token）才需新块
- **引用计数 vs 写时复制 (COW)？** nano-vllm 选 ref_count：prefix cache 共享时多个 seq 指向同一物理块；如要做 beam search 需引入 COW

---

### 3.3 Sequence 状态机 (`engine/sequence.py`)

**三态机**：`WAITING → RUNNING → FINISHED`

```python
class SequenceStatus(Enum):
    WAITING = auto()    # 在 scheduler.waiting deque
    RUNNING = auto()    # 在 scheduler.running deque，每步生成 1 token
    FINISHED = auto()   # EOS 或 max_tokens 达到，blocks 已 deallocate
```

**关键字段**：
- `num_cached_tokens`: 已写入 KV 缓存的 token 数（支持 chunked prefill）
- `num_scheduled_tokens`: 本轮 schedule 选中的 token 数
- `is_prefill`: 当前是否处于 prefill 阶段（影响 Sampler 取哪些 logits）
- `block_table`: 逻辑块 → 物理块 ID 的映射
- `last_block_num_tokens`: 当前块已填充的 token 数（用于 decode 的 slot_mapping）

**`__getstate__/__setstate__` 序列化优化**：
- Prefill 时传完整 `token_ids`（worker 需要 rebuild KV）
- Decode 时只传 `last_token`（worker 只需要 1 个 token 增量）

---

### 3.4 张量并行 (`layers/linear.py`)

**Megatron-LM 风格的 4 种切分模式**：

```
                    ┌──────────┐
                    │  Input x  │  (batch, hidden_size)
                    └────┬─────┘
              ┌──────────┼──────────┐
              ▼                     ▼
     ColumnParallel            Replicated
   (沿输出维切, hidden→out)   (整块复制到所有 rank)
              │                     │
              ▼                     ▼
        (no comm)             (no comm)
              │                     │
              ▼                     ▼
   RowParallelLinear          QKVParallelLinear
   (沿输入维切, in→hidden)    (Q/K/V 分别切头数)
              │                     │
              ▼                     ▼
        all_reduce              (no comm, 局部即可)
              │
              ▼
         (batch, hidden_size)
```

| 类 | 切分维度 | 通信 |
|----|----------|------|
| `ReplicatedLinear` | 不切 | 无 |
| `ColumnParallelLinear` | dim=0 (输出) | 无 |
| `MergedColumnParallelLinear` | dim=0 (gate+up 合并) | 无 |
| `QKVParallelLinear` | dim=0 (Q/K/V 按头切) | 无 |
| `RowParallelLinear` | dim=1 (输入) | `all_reduce` |

**`QKVParallelLinear` 权重 layout**（沿 dim=0 拼接）：
```
[Q_head_0, Q_head_1, ..., K_head_0, K_head_1, V_head_0, V_head_1, ...]
   num_heads * head_dim      num_kv_heads * head_dim
```
加载时通过 `weight_loader(loaded_shard_id="q"|"k"|"v")` 区分。

**`RowParallelLinear` 的 bias 处理**：
```python
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
if self.tp_size > 1: dist.all_reduce(y)
```
仅 rank 0 加 bias，避免所有 rank 各加一次。

**面试必问点**：
- **为什么 `Q` 沿头切分不需要通信？** Attention 计算是按头独立的 `softmax(Q_i K_i^T) V_i`，分头切片互不依赖
- **为什么 `O_proj` 必须 all_reduce？** 它把多头结果拼起来，下一层 MLP 需要完整的 hidden_size
- **TP 通信开销来源？** RowParallel 的 `all_reduce`、LM Head 的 `dist.gather`；前者用 NCCL Ring-AllReduce，后者用 P2P gather

---

### 3.5 注意力层 (`layers/attention.py`)

**两条核心路径**，都委托给 FlashAttention 库：

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()  # 全局单例
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # Prefill 路径：varlen attention，多个 prompt 拼成一个大 batch
            if context.block_tables is not None:  # prefix cache 命中
                k, v = k_cache, v_cache  # 直接从缓存读，跳过本步未缓存部分
            o = flash_attn_varlen_func(q, k, v,
                cu_seqlens_q=..., cu_seqlens_k=...,
                max_seqlen_q=..., max_seqlen_k=...,
                softmax_scale=self.scale, causal=True,
                block_table=context.block_tables)
        else:
            # Decode 路径：增量 1 token，从 KV 缓存读历史
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,  # 每个 seq 当前总长度
                block_table=context.block_tables,
                softmax_scale=self.scale, causal=True)
        return o
```

**Triton `store_kvcache_kernel`**（KV 写入）：
```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                          k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return  # 跳过 padding
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

`slot_mapping` 由 `ModelRunner` 提前算好：逻辑 token 位置 → 物理缓存 offset。`block_table[i] * block_size + offset_in_block`。

**面试必问点**：
- **为什么 Prefill 用 varlen 而不是 pad？** 多个 prompt 长度不一（如 [100, 200, 1500]），padding 会浪费 GPU；varlen 用 cu_seqlens 直接变长处理
- **`flash_attn_with_kvcache` 比手写 PagedAttention 慢吗？** 不一定——它把 KV 缓存按 block 切分后传入 shared memory，命中率不输手写
- **GQA 怎么实现？** `num_kv_heads` 通常 < `num_heads`（如 8 vs 16），flash_attn 内部会对 Q 头做"组内广播"以匹配 K/V 头数

---

### 3.6 RoPE (`layers/rotary_embedding.py`)

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # (max_pos, rotary_dim/2)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)  # (max_pos, 1, rotary_dim)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
    
    @torch.compile
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

**算法本质**：把 x 的相邻两半视作复数的实部虚部，乘以 `cos + i*sin`：
```
(x1 + i·x2) · (cos + i·sin) = (x1·cos - x2·sin) + i·(x2·cos + x1·sin)
```

**面试必问点**：
- **预计算 cos/sin 的好处？** 在线算 `cos/sin` 是数百次 CUDA 三角函数调用，预算在 init 阶段省 100% 在线计算
- **为什么在 float32 下做旋转再 cast 回去？** RoPE 对低精度敏感，BFloat16 下累积误差会导致长序列 attention 偏移
- **Qwen3 与 Llama 的 RoPE 差异？** Qwen3 在 Q/K 上额外做 RMSNorm（`q_norm`, `k_norm`）后再 RoPE，训练更稳定；Llama 直接 RoPE

---

### 3.7 RMSNorm + 残差融合 (`layers/layernorm.py`)

```python
@torch.compile
def rms_forward(self, x):
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))  # 原地
    x = x.to(orig_dtype).mul_(self.weight)
    return x

@torch.compile
def add_rms_forward(self, x, residual):
    """融合 residual add + RMSNorm，省一次读 x 的 HBM"""
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)  # 留作下一层 residual
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x, residual
```

**公式**：
```
RMSNorm(x) = (x / sqrt(mean(x²) + ε)) · γ
```

**为什么比 LayerNorm 快？** 跳过均值中心化（不减均值），少一次 reduction。

**融合版为何要返回 `residual`？** 下一层（attention 或 MLP）的 `add_rms_forward` 需要未归一化的输入作为新 residual：
```
x_1 = add_rms_norm(x_0 + residual_0)  # 这一步融合了 x_0+residual_0 的 add 和 norm
x_2 = attention(x_1)
x_2, residual_1 = add_rms_norm(x_2, x_1)  # residual_1 = x_2 + x_1（归一化前的值）
x_3 = mlp(x_2)
x_3, residual_2 = add_rms_norm(x_3, residual_1)
```

---

### 3.8 采样器 (`layers/sampler.py`)

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max 采样：argmax(probs / Exp(1))
        # 等价于从 softmax(logits/temp) 分布中采样
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

**Gumbel-max 技巧推导**：
- 目标：采样 x ~ p(x) ∝ exp(f(x))
- 标准做法：u ~ Uniform(0,1)，x = argmax f(x) + log(-log(u))
- 改写：等价于 argmax f(x) + Gumbel = argmax log p(x) + Gumbel = argmax log p(x) - log(-log(u))
- 又：argmax exp(log p(x)) / (-log(u)) = argmax p(x) / Exp(1)
- 最后一步：Exp(1) 等价于 -log(u)，避开 log(-log(u)) 的数值不稳定

**面试必问点**：
- **为什么用 Gumbel-max 而不是 `torch.multinomial`？** 单次 `argmax` 调度，GPU 上更快
- **为什么没有 top-k / top-p？** nano-vllm 是教学实现，简化了采样
- **温度必须 > 1e-10？** 温度为 0（greedy）会导致除零；可用 `argmax(logits)` 代替

---

### 3.9 Embed + LM Head (`layers/embed_head.py`)

**`VocabParallelEmbedding`**：每个 rank 持 vocab_size / tp_size 份词嵌入
```python
def forward(self, x):
    mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
    x = mask * (x - self.vocab_start_idx)  # 偏移到局部索引
    y = F.embedding(x, self.weight)
    y = mask.unsqueeze(1) * y  # 越界 token 输出 0
    if self.tp_size > 1:
        dist.all_reduce(y)  # 跨 rank 汇总
    return y
```

**`ParallelLMHead` 关键优化：仅 prefilling 的最后 token**
```python
def forward(self, x):
    if context.is_prefill:
        # 只取每个序列的最后一个 token，对应 causal LM 预测的 next token
        last_indices = context.cu_seqlens_q[1:] - 1
        x = x[last_indices].contiguous()  # (num_seqs, hidden)
    logits = F.linear(x, self.weight)  # (num_seqs, vocab/tp)
    if self.tp_size > 1:
        # gather 到 rank 0，跨 rank 拼接 vocab 维
        all_logits = [...] if self.tp_rank == 0 else None
        dist.gather(logits, all_logits, 0)
        logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
    return logits
```

**为什么 prefill 只算最后 token？**
- Prefill 阶段每个 token 的 hidden state 都对应"预测下一个 token"的 logits
- 但只有最后一个位置的 logits 是要用的（之前的都对应 prompt 内部位置）
- 减少 (batch × seq_len × vocab) → (batch × vocab) 的计算和通信

---

### 3.10 Model Runner (`engine/model_runner.py`)

**核心职责**：把 sequence 列表 → GPU 张量 → forward → 采样

**KV 缓存分配**（启动时一次性）：
```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf.hidden_size // hf.num_attention_heads)
    # 单个 block 字节数 = 2(K+V) * num_layers * block_size * num_kv_heads * head_dim * dtype_bytes
    block_bytes = 2 * L * block_size * num_kv_heads * head_dim * dtype_bytes
    # 块数 = (总内存 × 利用率 - 已用 - 峰值) / 单块字节
    num_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
    self.kv_cache = torch.empty(2, L, num_blocks, block_size, num_kv_heads, head_dim)
    # 把切片指针挂到每个 Attention 层
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

**`prepare_prefill` 与 `prepare_decode` 的差异**：
- Prefill：varlen 模式，构造 `cu_seqlens_q/k`、`max_seqlen_q/k`、`slot_mapping`（所有 token 都要写 KV）
- Decode：1 token 模式，构造 `context_lens`（每 seq 当前长度）、`slot_mapping`（只写最新 1 个位置）

**CUDA Graph 捕获**：
```python
def capture_cudagraph(self):
    max_bs = min(self.config.max_num_seqs, 512)
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=..., context_lens=..., block_tables=...)
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
        with torch.cuda.graph(graph, self.graph_pool):  # 共享内存池
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
        if self.graph_pool is None: self.graph_pool = graph.pool()
        self.graphs[bs] = graph
```

**回放路径**：
```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        # 把当前 batch 拷进预分配 buffer
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()  # 一发指令，PyTorch 无 Python 开销
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**面试必问点**：
- **为什么只有 decode 用 CUDA Graph？** Prefill shape 多变（不同 prompt 长度），capture 一份图不够用；decode 每次恰好 1 token/seq，shape 稳定
- **为什么不 capture LM Head + Sampler？** 不同 seq 的 `temperatures` 不同，且采样结果非确定，不适合固定 graph
- **共享 `graph_pool` 的好处？** 多个 graph 复用同一段显存，避免各自独立分配导致 OOM

---

## 4. 关键算法与 vLLM 对比

| 算法/优化 | nano-vllm 实现 | vLLM 实现 | 教学价值 |
|----------|---------------|-----------|---------|
| **PagedAttention** | `BlockManager` + FlashAttention `block_table` 参数 | 自研 PagedAttention CUDA/Triton 内核 | ⭐⭐⭐⭐⭐ 必须懂 |
| **连续批处理** | `Scheduler.schedule()` prefill/decode 交织 | `Scheduler` + `SchedulerOutput` | ⭐⭐⭐⭐⭐ |
| **Chunked Prefill** | 仅首 seq 分块（避免 bubble） | 通用支持 | ⭐⭐⭐⭐ |
| **前缀缓存** | xxhash64 滚动哈希 | xxhash + LRU 淘汰 | ⭐⭐⭐⭐ |
| **CUDA Graph** | decode 路径 batch 1~512 | 全路径 + 多 shape | ⭐⭐⭐ |
| **张量并行** | Megatron 风格 4 类线性 | Megatron + DeepSpeed 风格 | ⭐⭐⭐⭐⭐ |
| **GQA** | flash_attn 内置 | flash_attn + 自研 PagedAttention | ⭐⭐⭐ |
| **滑动窗口 / 稀疏 Attn** | 不支持 | vLLM V1 支持 | ⭐⭐ |
| **KV 量化 (INT8/FP8)** | 不支持 | 支持 | ⭐⭐ |
| **推测解码** | 不支持 | 支持（EAGLE/MEDUSA） | ⭐ |

---

## 5. 复现与学习路线（4 周计划）

### Week 1：跑通端到端
- 装环境：`pip install git+https://github.com/GeeeekExplorer/nano-vllm.git`
- 跑 `example.py`，打印每步的 prefill/decode tok/s
- 改 `enforce_eager=True/False` 对比 CUDA Graph 加速效果
- 跑 `bench.py`，记录不同 batch size 的吞吐

### Week 2：吃透调度
- 在 `Scheduler.schedule()` 加 `print(scheduled_seqs, is_prefill, [s.num_scheduled_tokens for s in scheduled])`
- 故意构造长短混合 prompt（如 [50, 2000]），观察 chunked prefill 行为
- 故意构造超过 `max_model_len` 的 prompt，观察 preempt 行为
- 阅读 vLLM `v1/core/scheduler.py`，对比抽象层级

### Week 3：吃透 KV 缓存
- 在 `BlockManager.hash_blocks()` 打印每次哈希命中/未命中
- 跑同一 system prompt + 多个 user query，观察 prefix cache 命中
- 计算 hash collision：手动改一个 token 的 hash，验证会被 `token_ids != expected` 拦截
- 思考：如果 vLLM 升级到 v1 内存模型（`KvCacheManager`），与 nano-vllm 的 `BlockManager` 设计差异？

### Week 4：扩展点练习
- **加 top-p 采样**：改 `Sampler.forward()`，加一个 `top_ps` 张量参数
- **加 sliding window**：在 `Attention` 中只保留最近 N 个 token
- **加量化 KV**：用 `torch.ao` 把 `k_cache`/`v_cache` 量化到 INT8
- **加 OpenAI API**：用 FastAPI 包装 `LLM.generate`，支持 `/v1/chat/completions`

---

## 6. 面试高频问题速答

### Q1：PagedAttention 解决了什么问题？
**答**：传统 KV 缓存为每个请求预分配 `max_seq_len` 大小的连续显存，浪费严重（O(请求数 × max_len)）。PagedAttention 把 KV 切成固定大小的 block（nano-vllm 是 256 token），按需分配，碎片化在 block 级别，显存利用率从 ~30% 提升到 ~90%。

### Q2：连续批处理 vs 静态批处理？
**答**：静态批等所有请求都生成完才一起推进，GPU 经常空转；连续批每个 step 都重新组 batch，长短请求混跑，吞吐可提升 2-4 倍。

### Q3：Chunked Prefill 的意义？
**答**：长 prompt 的 prefill 阶段计算量大，如果一次跑完会阻塞后续 decode，破坏 TTFT 体验。分块后 prefill 拆分到多步，与 decode 交替执行，TTFT 和 TPS 都能兼顾。

### Q4：Prefix Cache 怎么避免 hash 冲突？
**答**：nano-vllm 用了**双重保险**：(1) 滚动链式哈希降低碰撞率；(2) 哈希命中后**逐 token 比对** `self.blocks[block_id].token_ids != token_ids`，确认不是碰撞才复用。

### Q5：CUDA Graph 的限制？
**答**：graph 内部不能有 CPU 逻辑、不能有动态 shape、不能有条件分支。nano-vllm 只对 shape 稳定的 decode 路径（bs ≤ 512）做 capture，且把 `temperature` 这种 per-seq 变量排除在 graph 外。

### Q6：为什么 TP>1 时只有 rank 0 做采样？
**答**：logits 维度 = vocab_size，TP 切分后跨 rank 拼起来才完整。`dist.gather` 把所有 rank 的 logits 集中到 rank 0，只有 rank 0 调用 Sampler，节省通信和重复计算。

### Q7：为什么 nano-vllm 比 vLLM 还快？
**答**：(1) 模型小（0.6B），TP=1 没有通信开销；(2) `enforce_eager=False` + CUDA Graph 减少了 Python 开销；(3) 测试场景（256 seq，bs 稳定）正好命中 Graph 优化的甜蜜点。**生产大模型场景** vLLM 的量化、调度、speculative decoding 优势会显著放大。

### Q8：Qwen3 相比 Llama 有什么特殊设计？
**答**：
1. **Q/K Norm**：在 RoPE 之前对 Q 和 K 各自做 RMSNorm（per-head），提高训练稳定性
2. **Qwen3 默认去掉 QKV bias**（`attention_bias=False`），触发 `q_norm`/`k_norm` 路径
3. **Tied embeddings**（小模型）：`tie_word_embeddings=True` 时 `lm_head.weight = embed_tokens.weight`

### Q9：nano-vllm 缺什么企业级功能？
**答**：
- 无 HTTP/OpenAI API
- 无 speculative decoding
- 无 KV 量化（INT8/FP8）
- 无 continuous batching 的细粒度策略（vLLM V1 支持 prefix caching-aware scheduling）
- 无异步 engine、无 request queue、无 priority scheduling
- 无 dynamic shape 优化

### Q10：实习面试怎么讲这个项目？
**建议答法**（STAR 法则）：
> **Situation**：自学推理引擎，目标是深入理解 vLLM 的设计
> **Task**：找一个 1,200 行量级的极简实现作为学习起点
> **Action**：逐模块读 + 跑实验 + 写笔记
>   1. 端到端跑通 Qwen3-0.6B，对比 enable/disable CUDA Graph 吞吐
>   2. 在 Scheduler 加日志，构造长短混合 prompt 观察 chunked prefill
>   3. 在 BlockManager 加日志，验证 prefix cache 命中与 hash 冲突处理
>   4. 扩展加 top-p 采样，对比生成质量
> **Result**：(a) 完整理解 vLLM 核心组件；(b) 输出可复现的工程笔记
> **讲亮点**：能深入讲 PagedAttention 原理、滚动哈希设计、CUDA Graph 边界、TP 通信模式

---

## 7. 配套代码导读地图

| 你想懂什么 | 读哪几个文件 | 行数 |
|-----------|-------------|------|
| **整体架构** | `nanovllm/__init__.py`、`llm.py`、`example.py` | 1+5+33 |
| **调度** | `engine/scheduler.py`、`engine/sequence.py` | 92+83 |
| **KV 缓存** | `engine/block_manager.py` | 120 |
| **GPU 执行** | `engine/model_runner.py` | 257 |
| **注意力** | `layers/attention.py`、`utils/context.py` | 75+27 |
| **RoPE** | `layers/rotary_embedding.py` | 59 |
| **采样** | `layers/sampler.py` | 12 |
| **TP 线性层** | `layers/linear.py` | 156 |
| **归一化 + 激活** | `layers/layernorm.py`、`layers/activation.py` | 50+11 |
| **Embed/Head** | `layers/embed_head.py` | 66 |
| **模型组装** | `models/qwen3.py` | 216 |
| **权重加载** | `utils/loader.py` | 28 |
| **配置** | `config.py`、`sampling_params.py` | 25+11 |

**总代码量约 1,200 行**，每天精读 200 行 + 1 个实验，**2 周可吃透**。

---

## 8. 进阶学习资源

| 资源 | 价值 |
|------|------|
| **vLLM 源码** | 工业级实现，重点看 `vllm/core/scheduler.py`、`vllm/attention/` |
| **FlashAttention 论文 + 代码** | Tri Dao 的 IO-aware attention，nano-vllm 委托的就是它 |
| **PagedAttention 论文** (SOSP'23) | Kwon et al., 原理解读 |
| **Megatron-LM 论文 + 代码** | 张量并行的祖师爷 |
| **Andrej Karpathy nanoGPT** | 训练侧极简实现，可对照学 |
| **llm.c** (Karpathy) | 纯 C/CUDA 训练，帮你理解 GPU 内核 |
| **《推理系统：性能优化》** (ZOMI) | 国产推理系统综述 |
| **vllm.readthedocs.io** | 官方文档，含架构图 |
| **SGLang 源码** | 另一个工业级引擎，RadixAttention 实现 prefix cache |
| **xformers / FlexAttention** | 理解 PyTorch 原生 attention 变体 |

---

## 9. 总结：nano-vllm 给你的"加速包"

1. **12 个核心文件，每个 100~250 行**——读起来不累
2. **完整跑通 PagedAttention + 连续批处理 + 调度 + TP + Graph**——面试 5 大考点全覆盖
3. **可扩展性强**——加新模型/新采样/新调度都是几十行代码的事
4. **性能对标 vLLM**——说明设计没有明显缺陷，可放心学
5. **附带 Triton 内核示例**——KV 存储那个 kernel 是入门 Triton 的好教材

**最后一句话**：用 nano-vllm 当跳板，**把 PagedAttention 和 Continuous Batching 讲透**，就能拿下 80% 的 LLM 推理系统面试。

