# 技术面试五维能力应对指南
## 以 nano-vllm 为锚点，从 infra 到算法的深度应对策略

> **核心原则**：面试官不考你"知不知道"，考你"有没有真正做过"
> **方法论**：每个回答都必须有**具体代码/数据/bug**作为锚点，而不是泛泛而谈

---

## 目录

- [维度一：底层原理深度](#dim1)
- [维度二：实验与方案验证](#dim2)
- [维度三：问题定位能力](#dim3)
- [维度四：工程落地能力](#dim4)
- [维度五：业务与场景理解](#dim5)

---

<a id="dim1"></a>
## 维度一：底层原理深度

> **面试官想听到的**：不是"这个东西是什么"，而是"为什么这么设计、解决什么问题、有什么局限、怎么改进"
> **致命回答**：背定义式的回答（"PagedAttention 是一种..."）
> **高分回答**：从问题出发，讲清楚 trade-off

---

### 1.1 PagedAttention — 为什么这么设计？

**问题**：面试官问"PagedAttention 解决了什么问题？为什么这么设计？"

**低分回答**：
> "PagedAttention 把 KV cache 分成固定大小的 block，按需分配，提高内存利用率。"

**高分回答**：

> "这个问题要从 KV cache 的内存模型说起。传统做法是为每个请求预分配 `max_seq_len` 大小的连续内存。但实际中用户回复长短不一——有人回 10 个 token，有人回 2000 个。预分配导致严重的**内部碎片**：短回复浪费大量显存，长回复又可能 OOM。
>
> nano-vllm 的 `BlockManager`（`block_manager.py:26`）把 KV cache 切成 256 token 一块的物理 block。`can_allocate()` 方法（`block_manager.py:58`）检查需要多少块，`allocate()` 按需分配。这样碎片化限制在 block 级别（最坏浪费 255 token），而不是 sequence 级别。
>
> 但这个设计有**三个局限**：
>
> **第一，block 内部碎片**。block_size=256 意味着一个 257 token 的序列需要 2 个 block，第二个 block 只用了 1 个位置，浪费 255 个 slot。vLLM 后来改成了更灵活的 block 管理。
>
> **第二，block table 的额外开销**。每个序列需要维护一个 `block_table` 列表（`sequence.py:28`），长序列的 block table 本身也占内存。128K context / 256 block_size = 512 个 block entries。
>
> **第三，不支持写时复制 (COW)**。nano-vllm 用引用计数（`block.py:12`）做 prefix cache 共享，但如果要做 beam search——多个 beam 共享 prefix 但各自有不同的 suffix——就需要 COW：共享的 block 只在被修改时才复制。nano-vllm 没有实现这个，所以它**不支持 beam search**。"

**追问：前缀缓存的 hash 碰撞怎么处理？**

> "nano-vllm 的 `compute_hash()`（`block_manager.py:36`）用 xxhash64 做滚动哈希，把父块的 hash 拼接到子块的输入里，形成链式结构。但 xxhash 不是密码学哈希，碰撞概率非零。
>
> 所以 `can_allocate()` 在哈希命中后还做了**逐 token 比对**（`block_manager.py:66`）：
> ```python
> if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
>     break
> ```
>
> 这是双重保险：哈希做快速过滤，token_ids 做精确校验。代价是哈希命中的 block 还要读 token_ids 做比较，但因为 block 只有 256 token，比较开销很小。"

---

### 1.2 连续批处理 — 解决什么问题，有什么代价？

**问题**：面试官问"Continuous Batching 和 Static Batching 有什么区别？为什么这么设计？"

**高分回答**：

> "Static Batching 的问题是**木桶效应**：一批请求中，短请求早就生成完了，但必须等最长的那个才能处理下一批。GPU 在等的这段时间完全空转。
>
> nano-vllm 的 `Scheduler.schedule()`（`scheduler.py:25`）实现了连续批处理：每个 step 重新组装 batch。具体看代码：
>
> ```python
> # scheduler.py:48-52
> if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
>     seq.status = SequenceStatus.RUNNING
>     self.waiting.popleft()
>     self.running.append(seq)
> ```
>
> Prefill 完成的序列从 waiting 移到 running，下一个 step 就可以参与 decode。同时新的 prefill 请求可以加入。
>
> **但连续批处理有代价**：
>
> **第一，调度开销**。每个 step 都要调用 `schedule()`，遍历 waiting 和 running 队列，检查 block 可用性。对于小模型（如 Qwen3-0.6B），调度的 Python 开销可能比 GPU 计算还大。这就是为什么 nano-vllm 用 CUDA Graph 来掩盖这个开销。
>
> **第二，Prefill-Decode 干扰**。一个 step 内既有 prefill 又有 decode 时，prefill 的大计算量会拖慢 decode 的延迟。nano-vllm 的策略是**prefill 优先**（`scheduler.py:29`），只有 waiting 队列空了才做 decode。这意味着如果有持续的新请求涌入，decode 可能被饿死。
>
> **第三，内存碎片化加剧**。连续批处理意味着 block 的分配和释放频繁发生，free list 会产生碎片。nano-vllm 的 `free_block_ids` 是 deque，分配是 FIFO 顺序，没有做碎片整理。"

**追问：Chunked Prefill 为什么只允许首个序列分块？**

> "看 `scheduler.py:42`：
> ```python
> if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
>     break
> ```
>
> 如果允许多个序列分块，会导致：(1) 每个序列都需要多次 prefill step，增加调度复杂度；(2) 多个序列的 partial prefill 混在一起，KV cache 的 slot_mapping 会很复杂；(3) 第一个序列分块后，后续序列可能在第一个序列完成前就开始 prefill，导致 pipeline 冲突。
>
> 只允许首个序列分块的**实际效果**：第一个长 prompt 会被切成多块，每块和其他序列的 decode 交替执行。后续序列要么完整 prefill，要么等到下一轮。这样既避免了长 prompt 阻塞 decode，又保持了调度的简单性。"

---

### 1.3 CUDA Graph — 原理与局限

**问题**：面试官问"CUDA Graph 是什么？为什么只在 decode 用？有什么限制？"

**高分回答**：

> "CUDA Graph 的核心思想是**把一系列 GPU 操作录制下来，之后重放**，避免每次 launch kernel 时的 CPU 开销。
>
> 看 `model_runner.py:222-248` 的 `capture_cudagraph()`：
> ```python
> for bs in reversed(self.graph_bs):
>     graph = torch.cuda.CUDAGraph()
>     set_context(False, ...)
>     outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
>     with torch.cuda.graph(graph, self.graph_pool):
>         outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
>     if self.graph_pool is None:
>         self.graph_pool = graph.pool()
>     self.graphs[bs] = graph
> ```
>
> 先 warmup（让 CUDA 分配好内存），再 capture（录制到 graph）。`graph_pool` 让多个 graph 共享同一段显存。
>
> **为什么只在 decode 用？** 看 `run_model()`（`model_runner.py:196-212`）：
> ```python
> if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
>     return self.model.compute_logits(self.model(input_ids, positions))
> ```
>
> Prefill 不用 CUDA Graph 的原因：
> 1. **Shape 不固定**：每个 prompt 长度不同，`cu_seqlens_q/k` 每次都不一样，无法用同一个 graph
> 2. **计算密集**：Prefill 本身就是大矩阵乘法，CPU launch 开销占比很小
> 3. **Block table 可能不同**：Prefill 时 prefix cache 命中情况不同，`block_tables` 每次变化
>
> Decode 用 CUDA Graph 的原因：
> 1. **Shape 固定**：每步都是 1 token/seq，只有 batch size 变化
> 2. **访存密集**：Decode 是 memory-bound，CPU launch 开销占比大
> 3. **高频调用**：每个生成的 token 都要调一次 decode
>
> **CUDA Graph 的三个硬限制**：
> 1. **不能有 CPU 逻辑**：graph 内部不能调用 Python 函数，不能有条件分支
> 2. **不能有动态 shape**：graph 录制时的 tensor shape 必须和回放时一致（所以 nano-vllm 预录了多种 bs）
> 3. **输入必须在预分配 buffer 上**：回放时要把数据拷到 graph 录制时用的那个 tensor 上"

---

### 1.4 RoPE — 为什么用旋转而不是可学习位置编码？

**问题**：面试官问"RoPE 比绝对位置编码好在哪？为什么 Qwen3 用它？"

**高分回答**：

> "绝对位置编码（如 GPT-2 的 learned position embedding）把位置信息加到输入上：`x = token_emb + position_emb`。问题是：(1) 位置编码是训练时学到的，没见过的长度无法泛化；(2) 位置信息只在第一层注入，深层可能丢失。
>
> RoPE 的核心 idea 是**把位置编码变成注意力计算的一部分**。看 `rotary_embedding.py:6-14`：
> ```python
> def apply_rotary_emb(x, cos, sin):
>     x1, x2 = torch.chunk(x.float(), 2, dim=-1)
>     y1 = x1 * cos - x2 * sin
>     y2 = x2 * cos + x1 * sin
>     return torch.cat((y1, y2), dim=-1).to(x.dtype)
> ```
>
> 本质是把 x 的相邻两半当作复数 `(x1 + i·x2)`，乘以旋转因子 `(cos + i·sin)`。关键性质：**两个位置的内积只依赖相对距离**，即 `<RoPE(q, m), RoPE(k, n)> = <q, k>_{m-n}`。这意味着注意力权重天然编码了相对位置，不需要额外的位置编码。
>
> **RoPE 的局限**：
> 1. **外推性有限**：超过训练长度后，高频分量的旋转角度超出训练分布，注意力质量下降。需要 NTK-aware scaling 或 YaRN 等技术
> 2. **Qwen3 的改进**：在 RoPE 之前对 Q 和 K 各自做 RMSNorm（`q_norm`, `k_norm`，见 `qwen3.py:69-70`），这稳定了训练过程中 Q/K 的范数，防止 attention logits 爆炸
> 3. **精度敏感**：`rotary_embedding.py:11` 在 float32 下做旋转再 cast 回去，因为 BF16 的精度不够，长序列下累积误差会导致 attention 偏移"

---

### 1.5 采样策略 — Gumbel-max vs Multinomial

**问题**：面试官问"nano-vllm 的采样是怎么实现的？为什么用这种方法？"

**高分回答**：

> "看 `sampler.py:8-11`：
> ```python
> logits = logits.float().div_(temperatures.unsqueeze(dim=1))
> probs = torch.softmax(logits, dim=-1)
> sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
> ```
>
> 这用的是 **Gumbel-max 技巧**：`argmax(p / Exp(1))` 等价于从 `softmax(logits/temp)` 分布中采样。
>
> 推导：标准做法是 `argmax(log(p) + Gumbel(0,1))`，其中 Gumbel = `-log(-log(U))`。数学上等价于 `argmax(p / (-log(U)))`，而 `-log(U)` 就是 `Exp(1)`。
>
> **为什么不用 `torch.multinomial`？** multinomial 需要先做 cumulative sum（前缀和），再 binary search，两次 kernel launch。Gumbel-max 只需要一次 elementwise 除法 + 一次 argmax，单 kernel 更快。
>
> **这个方法的局限**：
> 1. **不支持 top-k / top-p**：没有对 logits 做截断，所有 token 都参与采样
> 2. **不支持 beam search**：每个序列只有一个采样结果
> 3. **温度为 0 会除零**：`temperature > 1e-10` 是硬性约束，不支持 greedy（需要 argmax(logits) 的单独路径）
> 4. **数值稳定性**：`clamp_min_(1e-10)` 防止除零，但极小的 exponential 值可能让 argmax 结果不稳定"

---

<a id="dim2"></a>
## 维度二：实验与方案验证

> **面试官想听到的**：不是"我做了 XX 实验"，而是"我怎么设计实验来验证 XX 假设，遇到了什么意外，怎么解释"
> **致命回答**：只说结果不说过程（"我跑了 benchmark，吞吐提升了 20%"）
> **高分回答**：讲清楚实验设计、控制变量、结果分析、意外发现

---

### 2.1 实验设计模板：验证 CUDA Graph 的效果

**假设**：CUDA Graph 可以减少 decode 阶段的 Python/CUDA launch 开销，提高吞吐。

**实验设计**：
```python
# 实验 1: enforce_eager=True (无 CUDA Graph)
llm = LLM(path, enforce_eager=True, max_model_len=4096)
# 跑 256 条序列，记录总时间和吞吐

# 实验 2: enforce_eager=False (有 CUDA Graph)
llm = LLM(path, enforce_eager=False, max_model_len=4096)
# 跑同样 256 条序列，记录总时间和吞吐
```

**控制变量**：
- 相同的 prompt 长度分布（100-1024 token 随机）
- 相同的输出长度（100-1024 token 随机）
- 相同的 GPU（RTX 4070）
- 相同的 `max_model_len=4096`
- 先跑一轮 warmup（`bench.py:22` 中的 `llm.generate(["Benchmark: "], SamplingParams())`）

**结果分析**：
| 配置 | 输出 tokens | 时间 | 吞吐 |
|------|-----------|------|------|
| eager (无 Graph) | 133,966 | ~105s | ~1,276 tok/s |
| graph (有 Graph) | 133,966 | ~93s | ~1,434 tok/s |

**追问：为什么 CUDA Graph 对小模型效果更明显？**

> "小模型（0.6B）的单次 forward 很快，CPU launch 开销占比大。对于 70B 模型，单次 forward 可能要 10ms，launch 开销 0.1ms 只占 1%；但 0.6B 模型单次 forward 可能只要 0.5ms，launch 开销 0.1ms 就占了 20%。CUDA Graph 把这 0.1ms 降到了接近 0，所以小模型的相对收益更大。"

**追问：warmup 那一步为什么必要？**

> "CUDA Graph 录制时，所有 CUDA 内存必须已经分配好。warmup 让 PyTorch 的 memory allocator 跑一遍，分配好所有需要的 tensor。如果不 warmup，capture 阶段会触发新的内存分配，但 CUDA Graph 不允许在 capture 期间做动态分配——会报错。"

---

### 2.2 实验设计模板：验证 Prefix Cache 的效果

**假设**：Prefix Cache 可以让相同 system prompt 的多个请求复用 KV cache，减少 prefill 时间。

**实验设计**：
```python
# 构造 10 个不同 user query，共享同一个 2K token 的 system prompt
system_prompt = "..." * 2000 tokens
user_queries = ["query1", "query2", ..., "query10"]

# 实验 A: 无 prefix cache（每次重新 prefill 完整 prompt）
# 实验 B: 有 prefix cache（system prompt 只 prefill 一次）
```

**验证方法**：在 `BlockManager.hash_blocks()` 加日志：
```python
def hash_blocks(self, seq):
    ...
    for i in range(start, end):
        ...
        self.hash_to_block_id[h] = block.block_id
        print(f"[hash_blocks] seq={seq.seq_id} block={i} hash={h} cached={h in old_hashes}")
```

**预期结果**：第一个请求的 system prompt blocks 被哈希化，后续 9 个请求的 `can_allocate()` 返回 `num_cached_blocks > 0`，prefill 只需要处理 user query 部分。

**意外发现**：如果 system prompt 不是 block_size（256）的整数倍，最后一个 block 不会被缓存（`can_allocate()` 跳过最后一个 block）。这意味着 2001 token 的 system prompt，前 2000 token 可以缓存，最后 1 个 token 每次都要重新算。

---

### 2.3 面试中怎么讲实验？

**模板**：
> "我做了 XX 实验来验证 YY 假设。实验设计是：[控制变量]。结果发现 [预期/意外]。我分析原因是 [原因]。基于这个发现，我做了 [下一步]。"

**具体例子**：
> "我做了实验来验证 CUDA Graph 对 decode 吞吐的提升。控制变量是相同的 prompt 分布、相同的 GPU、相同的模型。结果发现 CUDA Graph 让吞吐提升了约 12%。但我注意到这个提升在 batch size 较小时更明显——bs=1 时提升 25%，bs=256 时只提升 5%。我分析原因是：bs 大时 GPU 计算占主导，launch 开销占比小；bs 小时 launch 开销占比大，Graph 的收益更明显。这让我理解了为什么 CUDA Graph 主要对 decode 路径有效——decode 时每步只处理 1 token/seq，计算量小，launch 开销相对占比大。"

---

### 2.4 追问应对：当实验结果和预期不一致

**场景**：你预期 Prefix Cache 会让吞吐翻倍，但实际只提升了 10%。

**低分回答**：
> "可能是代码有 bug，或者数据有问题。"

**高分回答**：
> "首先我排除了代码 bug——我打印了 `can_allocate()` 的返回值，确认 prefix cache 确实命中了。然后我发现问题出在**测试数据的 prompt 太短**——只有 100-1024 token，而 block_size 是 256。这意味着大多数 prompt 只有 1-4 个 block，prefix cache 最多省掉 3 个 block 的 prefill。但如果 prompt 有 4K token（16 个 block），prefix cache 可以省掉 15 个 block，效果就很明显了。
>
> 所以我调整了实验：用 4K token 的 system prompt + 100 token 的 user query。这次 prefix cache 的效果就很明显了——prefill 时间从 50ms 降到了 10ms。
>
> 这个经历让我理解了：**prefix cache 的收益和 prompt 长度/block_size 的比值成正比**。短 prompt 场景下，prefix cache 的收益可以忽略。"

---

<a id="dim3"></a>
## 维度三：问题定位能力

> **面试官想听到的**：不是"我遇到了 XX 问题"，而是"我是怎么定位到 XX 问题的根因的"
> **致命回答**：只说结果（"最后发现是内存不够"）
> **高分回答**：讲清楚排查思路、用了什么工具、排除了哪些可能、最终怎么定位

---

### 3.1 场景：模型上线后吞吐突然下降 50%

**排查思路**：

```
Step 1: 确认是 GPU 还是 CPU 问题
  → nvidia-smi 看 GPU 利用率
  → 如果 GPU 利用率低 → CPU 瓶颈（调度开销、数据加载）
  → 如果 GPU 利用率高 → GPU 瓶颈（计算、内存）

Step 2: 如果是 GPU 利用率低
  → 看 Scheduler 日志：每个 step 的 batch size 是多少？
  → 如果 batch size 小 → 新请求太少 or 调度策略问题
  → 如果 batch size 大但吞吐低 → CUDA Graph 可能失效了

Step 3: 如果是 GPU 利用率高但吞吐低
  → 看 CUDA Graph 是否命中：run_model() 走的是 eager 还是 graph 路径？
  → 看 KV cache 是否用完：BlockManager.can_allocate() 返回 -1 的频率
  → 看是否频繁 preempt：preempt() 的调用次数
```

**定位到具体原因**：

**Case A: CUDA Graph 未命中**
> "我发现吞吐下降是因为新增了一类 prompt，长度超过 512 token。看 `model_runner.py:197`：
> ```python
> if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
>     return self.model.compute_logits(self.model(input_ids, positions))
> ```
> 当 batch size > 512 时，CUDA Graph 不会命中。之前测试的 prompt 短，batch size 通常 < 512，所以一直走 graph 路径。新 prompt 长了，batch size 经常超过 512，导致 fallback 到 eager 模式。"

**Case B: KV Cache 耗尽导致频繁 preempt**
> "我发现吞吐下降是因为并发请求太多，KV cache 耗尽了。看 `scheduler.py:60-65`：
> ```python
> while not self.block_manager.can_append(seq):
>     if self.running:
>         self.preempt(self.running.pop())
>     else:
>         self.preempt(seq)
>         break
> ```
> 当 free block 不足时，会 preempt 正在运行的序列。被 preempt 的序列要重新 prefill，浪费了之前的计算。我通过日志发现 preempt 频率从 0 次/分钟 变成了 20 次/分钟。
>
> 根因是：新上线的 prompt 平均长度更长（2K → 4K），每个序列占更多 KV block，导致并发容量下降。解决方案：(1) 增加 GPU 内存利用率（`gpu_memory_utilization` 从 0.9 调到 0.95）；(2) 限制最大 prompt 长度；(3) 用更大的 block_size（256 → 512）减少 block table 开销。"

**Case C: Prefix Cache 未命中**
> "我发现吞吐下降是因为 system prompt 被修改了一个 token。前缀缓存是**精确匹配**——修改 system prompt 的任何一个 token，整个 prefix cache 就失效了。看 `block_manager.py:66`：
> ```python
> if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
>     break
> ```
> 只要第一个 block 的 token_ids 不匹配，后续所有 block 的缓存都不会被使用。这让我理解了为什么生产环境中 system prompt 应该固定不变——每次修改都会导致所有用户的 prefix cache 全部失效。"

---

### 3.2 场景：GPU OOM，但之前跑得好好的

**排查思路**：

```
Step 1: 确认 OOM 发生在哪个阶段
  → 启动时 OOM → 模型加载 or KV cache 分配
  → 运行时 OOM → KV cache 耗尽 or 中间激活过大

Step 2: 看内存分配
  → allocate_kv_cache() 中的计算：
    num_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
  → 如果 num_blocks 接近 0 → 模型权重 + 峰值激活 已经占满了 GPU
  → 如果 num_blocks > 0 但运行时 OOM → 峰值激活估计不准

Step 3: 看 warmup 是否覆盖了最坏情况
  → warmup_model() 用 min(max_num_batched_tokens, max_model_len) 作为 seq_len
  → 如果实际请求的 seq_len > warmup 的 seq_len → 峰值激活可能超过预估
```

**具体案例**：

> "我发现 OOM 是因为 `warmup_model()`（`model_runner.py:91-101`）用的 seq_len 是 `min(max_num_batched_tokens, max_model_len)` = `min(16384, 4096)` = 4096。但实际请求中，如果有 10 个序列每个 4096 token 同时 prefill，总 token 数是 40960，超过了 `max_num_batched_tokens=16384`，会被 Scheduler 拆成多个 chunk。
>
> 但问题是：warmup 时只跑了 `num_seqs = 16384 // 4096 = 4` 个序列，峰值激活是按 4 个序列估计的。实际运行时如果 batch size 更大，峰值激活可能超过预估。
>
> 解决方案：把 warmup 的 seq_len 调小，让 num_seqs 更大，覆盖更大的 batch size 场景。"

---

### 3.3 场景：生成结果质量突然变差

**排查思路**：

```
Step 1: 确认是模型问题还是数据问题
  → 同样的 prompt，对比 SFT 前后的输出
  → 如果 SFT 前好、SFT 后差 → 训练问题
  → 如果前后都差 → 数据问题 or prompt 问题

Step 2: 如果是训练问题
  → 检查训练数据：是否有低质量/错误标注的数据
  → 检查训练超参：学习率是否太高（过拟合）/太低（欠拟合）
  → 检查 loss 曲线：是否 train loss 低但 eval loss 高（过拟合）

Step 3: 如果是推理问题
  → 检查采样参数：temperature 是否正确、top-p 是否生效
  → 检查 KV cache：prefix cache 是否导致了错误的上下文复用
  → 检查 tokenizer：是否和训练时用的同一个 tokenizer
```

**具体案例**：

> "我发现生成质量下降是因为 prefix cache 导致了**错误的上下文复用**。两个用户共享了同一个 system prompt 的 prefix cache，但他们的对话历史不同。当用户 A 的对话结束后，他的 KV cache block 被释放（ref_count 减到 0），但 `hash_to_block_id` 中的映射还在。用户 B 的新请求命中了这个 hash，复用了已经被释放并重新分配的 block——里面的数据已经被用户 C 的 KV 覆盖了。
>
> 根因是 `_allocate_block()`（`block_manager.py:43-51`）在分配新 block 时会清理旧的 hash 映射：
> ```python
> if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
>     del self.hash_to_block_id[block.hash]
> ```
> 但如果 block 被释放后立即被重新分配（`free_block_ids.popleft()`），新的 block 会覆盖旧数据，而 `hash_to_block_id` 中指向这个 block_id 的旧 hash 仍然存在。下一个请求 `can_allocate()` 时，哈希命中但 token_ids 不匹配，会被逐 token 校验拦住。
>
> 这其实是一个**潜在的 bug**——虽然逐 token 校验拦住了错误的缓存命中，但如果两个不同的 token 序列恰好有相同的 xxhash64（碰撞），就会导致错误的缓存复用。不过 xxhash64 的碰撞概率极低（2^-64），实际中几乎不会发生。"

---

### 3.4 面试中怎么讲问题定位？

**模板**：
> "我遇到了 [问题现象]。首先我排除了 [可能原因 A]——通过 [具体方法]。然后我发现 [可能原因 B] 也不对——因为 [具体证据]。最终我定位到根因是 [根因]——通过 [具体日志/数据]。解决方案是 [方案]。"

**关键要素**：
1. **有条理**：从外到内，逐步缩小范围
2. **有证据**：每个排除都有具体数据/日志支持
3. **有方案**：定位到根因后有明确的解决方案
4. **有反思**：这个问题暴露了什么设计缺陷？怎么预防？

---

<a id="dim4"></a>
## 维度四：工程落地能力

> **面试官想听到的**：不是"理论上可以这么做"，而是"实际中我是怎么部署的，遇到了什么工程问题"
> **致命回答**：只讲算法不讲工程（"我用了 PPO 训练，效果很好"）
> **高分回答**：讲清楚部署、监控、回滚、稳定性

---

### 4.1 模型部署：从实验到生产

**问题**：面试官问"你的模型怎么部署上线的？"

**高分回答**：

> "部署分三层：
>
> **第一层：模型格式转换**。训练时用 PyTorch 的 checkpoint，部署时要转成推理引擎格式。vLLM 直接加载 HuggingFace 格式的 safetensors——看 `utils/loader.py`：
> ```python
> def load_model(model, path):
>     for file in glob(os.path.join(path, "*.safetensors")):
>         with safe_open(file, "pt", "cpu") as f:
>             for weight_name in f.keys():
>                 ...
> ```
> 生产环境中还需要考虑量化——把 BF16 权重量化到 INT8/INT4，减少显存和延迟。nano-vllm 没有量化，但 vLLM 支持 GPTQ、AWQ、FP8。
>
> **第二层：推理服务**。nano-vllm 是同步的 `generate()` 调用，生产环境需要异步服务：
> - vLLM 自带 OpenAI 兼容的 API server
> - 用 uvicorn/FastAPI 包装，支持并发请求
> - 需要 request queue + rate limiting
>
> **第三层：基础设施**。
> - **负载均衡**：多实例部署，用 Nginx/K8s 做负载均衡
> - **健康检查**：定期 ping 模型，不健康则重启
> - **自动扩缩容**：根据 QPS 动态调整实例数"

---

### 4.2 监控体系：上线后怎么知道系统正常？

**监控指标**：

| 指标 | 含义 | 报警阈值 |
|------|------|---------|
| **TTFT** (Time to First Token) | 首 token 延迟 | > 2s |
| **TPS** (Tokens per Second) | 生成速度 | < 10 tok/s |
| **Throughput** (req/s) | 每秒处理请求数 | < 预期的 50% |
| **GPU Utilization** | GPU 利用率 | < 30% 或 > 95% |
| **KV Cache Utilization** | KV cache 使用率 | > 90% (可能 OOM) |
| **Preemption Rate** | 每分钟抢占次数 | > 10 |
| **Error Rate** | 请求错误率 | > 1% |
| **Queue Length** | 等待队列长度 | > 100 |

**具体实现**：
```python
# 在 LLMEngine.generate() 中添加监控
import time
from prometheus_client import Histogram, Counter

TTFT_HISTOGRAM = Histogram('llm_ttft_seconds', 'Time to first token')
TPS_HISTOGRAM = Histogram('llm_tps', 'Tokens per second')
PREEMPTION_COUNTER = Counter('llm_preemptions_total', 'Total preemptions')

def generate(self, prompts, sampling_params, use_tqdm=True):
    ...
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()
        if num_tokens > 0:
            prefill_throughput = num_tokens / (perf_counter() - t)
            TTFT_HISTOGRAM.observe(perf_counter() - t)
        else:
            decode_throughput = -num_tokens / (perf_counter() - t)
            TPS_HISTOGRAM.observe(decode_throughput)
    ...
```

---

### 4.3 回滚策略：上线后效果变差怎么办？

**回滚流程**：

```
1. 灰度发布：新模型先接 5% 流量
2. A/B 对比：新旧模型同时运行，对比核心指标
3. 自动回滚：如果新模型指标低于阈值，自动切回旧模型
4. 数据回滚：如果问题是数据导致的，需要回滚训练数据
```

**具体案例**：

> "我上线了一个新的 SFT 模型，灰度 5% 流量后发现 TTFT 从 200ms 涨到了 800ms。排查发现新模型的 prompt 模板比旧模型多了 500 token 的 system prompt，导致 prefill 时间增加。解决方案：(1) 立即回滚到旧模型；(2) 优化 system prompt 长度；(3) 重新灰度。"

---

### 4.4 资源管理：GPU 显存怎么分配？

**从 nano-vllm 的 `allocate_kv_cache()` 看内存预算**：

```python
# model_runner.py:103-113
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    ...
    block_bytes = 2 * L * block_size * num_kv_heads * head_dim * dtype.itemsize
    num_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```

**内存预算公式**：
```
KV Cache 可用 = GPU 总内存 × 利用率 - 模型权重 - 峰值激活 + 当前已释放
```

**生产环境的挑战**：

| 挑战 | 原因 | 解决方案 |
|------|------|---------|
| **峰值激活估计不准** | 不同 prompt 长度的激活不同 | 用最坏情况估计，留 buffer |
| **多模型共享 GPU** | RM 和 Actor 在同一 GPU | 时间分片：先 rollout 再训练 |
| **CUDA 碎片化** | 频繁分配/释放导致 | 用 `torch.cuda.empty_cache()` |
| **OOM 不可预测** | 某些 prompt 特别长 | 设置 `max_model_len` 硬限制 |

---

### 4.5 工程落地的关键认知

**面试讲法**：

> "工程落地和理论最大的区别是**边界条件**。理论上 PagedAttention 可以提高内存利用率，但实际中：
>
> 1. **Block size 选多大？** 256 是经验值。太小（如 32）→ block table 开销大；太大（如 1024）→ 碎片浪费多。需要根据实际 prompt 长度分布来调。
>
> 2. **GPU memory utilization 设多少？** 0.9 留了 10% buffer。但如果峰值激活估计不准，这 10% 可能不够。生产环境通常要跑 stress test 来验证。
>
> 3. **怎么处理 OOM？** 不是不让它发生，而是让它**优雅地发生**——拒绝新请求、preempt 老请求、而不是直接 crash。
>
> 4. **怎么保证一致性？** 同一个 prompt 多次请求应该得到相似的结果（除非 temperature > 0）。需要固定 random seed、确保 prefix cache 行为一致。"

---

<a id="dim5"></a>
## 维度五：业务与场景理解

> **面试官想听到的**：不是"这个技术很好"，而是"这个技术在什么场景下有价值，成本多高，优先优化什么"
> **致命回答**：只讲技术不讲业务（"PagedAttention 很好，应该用"）
> **高分回答**：讲清楚场景适配、成本收益、优先级

---

### 5.1 场景适配：这个方案适合什么场景？

**问题**：面试官问"nano-vllm 的方案适合什么场景？不适合什么场景？"

**高分回答**：

> "**适合的场景**：
>
> 1. **离线批量推理**：nano-vllm 是同步 API，没有异步 server。适合 batch inference（如数据标注、离线评估），不适合在线服务。
>
> 2. **小模型高吞吐**：Qwen3-0.6B 这种小模型，单 GPU 就能跑，continuous batching 的吞吐优势明显。对于 70B 模型，单 GPU 放不下，需要 TP/PP，nano-vllm 的 TP 实现太简单（SharedMemory + pickle），不适合生产。
>
> 3. **教学和原型验证**：代码只有 1200 行，适合学习 vLLM 的核心思想，快速验证想法。
>
> **不适合的场景**：
>
> 1. **在线服务**：没有 HTTP server、没有 request queue、没有 load balancing
>
> 2. **长上下文**：block_size=256 固定，128K context 需要 512 个 block，block table 本身就有开销
>
> 3. **多模态**：只支持文本，不支持图像/音频
>
> 4. **Speculative Decoding**：没有实现 draft model，无法做投机解码
>
> 5. **量化推理**：不支持 GPTQ/AWQ/FP8，纯 BF16 推理"

---

### 5.2 成本收益分析：优化哪个最划算？

**问题**：面试官问"资源有限，你应该优先优化哪些部分？"

**高分回答**：

> "这取决于**瓶颈在哪**。我用一个决策树来分析：
>
> ```
> 如果 TTFT 太高（prefill 慢）：
>   → 首先：加 Prefix Cache（如果有多轮对话）→ 收益最大
>   → 其次：用 Chunked Prefill（如果 prompt 很长）→ 减少 decode 阻塞
>   → 最后：模型量化（INT8）→ 减少计算量
>
> 如果 TPS 太低（decode 慢）：
>   → 首先：增加 batch size（如果 GPU 内存够）→ 提高 GPU 利用率
>   → 其次：CUDA Graph（如果模型小）→ 减少 launch 开销
>   → 最后：Speculative Decoding → 2-3x 加速但工程复杂
>
> 如果并发不够（吞吐低）：
>   → 首先：Continuous Batching（已实现）→ 基础
>   → 其次：多实例部署 → 水平扩展
>   → 最后：Pipeline Parallelism → 跨 GPU 并行
>
> 如果内存不够（OOM）：
>   → 首先：PagedAttention（已实现）→ 基础
>   → 其次：KV Cache 量化（INT8）→ 减半内存
>   → 最后：Prefix Cache → 减少重复 KV
> ```
>
> **优先级原则**：先解决瓶颈，再优化非瓶颈。用 profiling 找到瓶颈，而不是凭感觉优化。"

---

### 5.3 用户关心什么？

**问题**：面试官问"用户更关心什么？TTFT 还是 TPS？"

**高分回答**：

> "取决于场景：
>
> | 场景 | 用户最关心 | 原因 |
> |------|-----------|------|
> | **聊天机器人** | TTFT | 用户等第一个字出现的耐心有限 |
> | **代码补全** | TTFT + TPS | 需要快速开始，也要流畅输出 |
> | **批量标注** | Throughput | 不在乎单个请求延迟，只在乎总吞吐 |
> | **Agent 工具调用** | TTFT | Agent 需要快速决定下一步动作 |
> | **长文本生成** | TPS | 生成 10K token，TPS 低会等很久 |
> | **实时翻译** | TTFT + TPS | 都很重要，用户期望即时响应 |
>
> 生产环境中，通常用 **P50/P95/P99 延迟**来衡量：
> - P50 TTFT < 500ms（一半请求在 500ms 内开始响应）
> - P95 TTFT < 2s（95% 的请求在 2s 内开始响应）
> - P99 TTFT < 5s（99% 的请求在 5s 内开始响应）
>
> 如果 P95 达标但 P99 不达标，说明有长尾请求——通常是特别长的 prompt 或 KV cache 耗尽导致的 preempt。"

---

### 5.4 上线成本有多高？

**问题**：面试官问"部署这个方案的成本有多高？"

**高分回答**：

> "成本分三块：
>
> **硬件成本**：
> - Qwen3-0.6B：单张 RTX 4070（~5000 元）就够，BF16 约 1.2GB 权重 + KV cache
> - Qwen3-72B：需要 4×A100-80GB（~40 万元），TP=4
> - 按云服务算：A100-80GB 约 15 元/小时，72B 模型 4 卡约 60 元/小时
>
> **推理成本**（以 Qwen3-72B 为例）：
> - 单次请求：平均 1K input + 500 output tokens
> - Prefill 时间：~200ms（1K tokens）
> - Decode 时间：~5s（500 tokens / 100 tok/s）
> - 成本：60 元/小时 × (5.2s / 3600s) ≈ 0.086 元/请求
>
> **优化收益**：
> - Prefix Cache：如果有 80% 命中率，prefill 成本降低 80%
> - Continuous Batching：batch size 从 1 提到 32，吞吐提升 20x，单请求成本降低 95%
> - 量化（INT8）：延迟降低 30%，成本降低 30%
>
> **优先级**：Continuous Batching > Prefix Cache > 量化 > Speculative Decoding。因为 batching 是成本收益比最高的——不需要改模型，只需要改调度策略。"

---

### 5.5 如果资源有限，优先做什么？

**问题**：面试官问"你只有 1 个工程师 + 1 张 A100，3 个月内上线一个 LLM 服务，你会怎么做？"

**高分回答**：

> "**Month 1：跑通基线**
> - 用 vLLM（不是 nano-vllm，因为需要 HTTP server）部署开源模型（如 Qwen3-7B）
> - 接入 FastAPI + uvicorn，提供 OpenAI 兼容 API
> - 部署到单张 A100，跑通端到端
> - 建立监控：TTFT、TPS、GPU 利用率
>
> **Month 2：优化体验**
> - 根据实际 prompt 长度分布调整 `max_model_len` 和 `max_num_seqs`
> - 开启 Prefix Cache（如果有 system prompt 场景）
> - 做 SFT：收集 1000 条业务数据，微调模型
> - 建立评估体系：自动化 eval + 人工抽检
>
> **Month 3：上线与迭代**
> - 灰度发布：5% → 20% → 100%
> - 收集用户反馈，迭代 SFT 数据
> - 如果 TPS 不够，考虑量化（AWQ INT4）
> - 如果需要更好的效果，考虑 DPO
>
> **不会做的事情**：
> - 不会自建推理引擎（用 vLLM/SGLang 就够了）
> - 不会做 RLHF（工程太复杂，DPO 足够）
> - 不会做 TP（单 GPU 不需要）
> - 不会做 Speculative Decoding（收益不确定，工程复杂）"

---

## 附录：面试回答框架速查

### 底层原理题回答框架
```
1. 这个技术解决什么问题？（从痛点出发）
2. 它是怎么解决的？（核心机制，用代码/公式说明）
3. 有什么 trade-off / 局限？（展示深度思考）
4. 有哪些改进方向？（展示视野）
```

### 实验验证题回答框架
```
1. 我的假设是什么？
2. 实验怎么设计的？（控制变量）
3. 结果是什么？（数据说话）
4. 和预期一致吗？不一致怎么解释？
5. 基于结果做了什么决策？
```

### 问题定位题回答框架
```
1. 现象是什么？
2. 我排除了哪些可能？（每步有证据）
3. 最终根因是什么？
4. 怎么解决的？
5. 怎么预防？（系统性改进）
```

### 工程落地题回答框架
```
1. 理论方案是什么？
2. 实际部署遇到了什么问题？
3. 怎么解决的？（工程 hack / trade-off）
4. 怎么监控？怎么保证稳定性？
5. 怎么回滚？
```

### 业务场景题回答框架
```
1. 这个方案适合什么场景？不适合什么？
2. 用户关心什么指标？
3. 成本多高？收益多大？
4. 资源有限时优先做什么？（给出决策框架）
```

---

**最后一句话**：面试的本质是**让面试官相信你真正做过**。要做到这一点，唯一的办法是**真的做过**——读代码、跑实验、踩坑、修 bug。nano-vllm 给了你一个 1200 行的沙盒，把上面每一道题都在这个沙盒里跑一遍，你就能在面试中讲出有血有肉的故事。
