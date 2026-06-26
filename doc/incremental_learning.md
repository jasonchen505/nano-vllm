# 增量学习笔记：8x3090 复现过程中的新发现
## 随实验推进持续记录，每条标注来源和学到什么

> **使用方法**：每完成一个实验，把新发现写进来
> **格式规范**：每条标注 [实验编号] [来源文件:行号] [分类标签]

---

## 使用说明

分类标签：
- [infra]     推理系统/工程相关
- [algo]      算法/模型设计相关
- [training]  后训练/SFT/RLHF 相关
- [agent]     Agent/应用相关
- [debug]     排查问题的经验
- [perf]      性能优化相关

---

## Phase 1：环境搭建 + 单卡跑通

### L-001: flash-attn 编译需要匹配 PyTorch + CUDA 版本

**来源**：安装过程
**分类**：[infra]

flash-attn 不是简单的 pip install，它需要从源码编译 CUDA kernel。关键依赖：
- PyTorch 版本（torch 2.4+）
- CUDA toolkit 版本（12.1+）
- GCC 版本（需要支持 C++17）

如果编译失败，可以用预编译 wheel：
```bash
pip install flash-attn --no-build-isolation
```

**为什么重要**：生产环境中 flash-attn 是性能关键组件，版本不匹配会导致性能回退或 crash。

---

### L-002: nano-vllm 的 NCCL 端口硬编码

**来源**：model_runner.py:26
**分类**：[infra]

端口 2333 是硬编码的。如果多实例同时运行（比如做 A/B 测试），会端口冲突。

**学到的点**：生产环境中的推理引擎需要支持动态端口分配或配置化。vLLM 通过环境变量 VLLM_PORT 来解决这个问题。

---

### L-003: Qwen3 的 tokenizer 和 chat template

**来源**：example.py:16-22
**分类**：[algo]

Qwen3 使用 ChatML 格式。apply_chat_template 会自动加上特殊标记。如果不用 chat template 直接传 prompt，模型可能不会正确回答。

**学到的点**：SFT 时必须和推理时用相同的 chat template，否则模型看到的 token 序列分布不同，效果会下降。

---

### L-004: torch.compile 的首次编译开销

**来源**：rotary_embedding.py:37, layernorm.py:17, activation.py:8, sampler.py:7
**分类**：[perf]

nano-vllm 中 4 个模块用了 @torch.compile：
- RotaryEmbedding.forward
- RMSNorm.rms_forward / add_rms_forward
- SiluAndMul.forward
- Sampler.forward

首次调用时 torch.compile 会触发图编译（TorchInductor），可能需要 10-30 秒。后续调用就很快了。

**学到的点**：生产环境中需要做 warmup 来预热编译。nano-vllm 的 warmup_model() 就是做这个的。

---

### L-005: set_default_device 的陷阱

**来源**：model_runner.py:29-38
**分类**：[debug]

model_runner 用 torch.set_default_device("cuda") 让所有 nn.Parameter 默认分配到 CUDA。创建完模型后必须恢复到 "cpu"，否则后续 CPU 上的 tensor 会意外跑到 GPU。

**学到的点**：更安全的做法是用 with torch.device("cuda"): 块，自动管理生命周期。

---

## Phase 2：单卡深度实验

### L-006: CUDA Graph capture 的 warmup 必须覆盖最坏情况

**来源**：model_runner.py:238-248
**分类**：[perf]

warmup 让 CUDA 分配好所有内存。如果 warmup 时的 tensor 和 capture 时不一致（比如某个分支没走到），capture 会失败。

**学到的点**：CUDA Graph 的限制是：录制时和回放时必须是完全相同的计算图。任何动态 shape、条件分支、CPU 逻辑都不行。

---

### L-007: graph_pool 共享内存池的意义

**来源**：model_runner.py:244-245
**分类**：[perf]

多个 CUDA graph 共享同一个内存池。好处：不同 batch size 的 graph 复用同一段显存，避免碎片化。代价：不能同时 replay 两个 graph（但推理是串行的，所以没问题）。

---

### L-008: Prefix Cache 的 hash 碰撞双重保险

**来源**：block_manager.py:36-41, 66
**分类**：[algo]

xxhash64 不是密码学哈希，碰撞概率非零。但 block_manager 做了双重保险：
1. 哈希做快速过滤（O(1) 查找）
2. 哈希命中后逐 token 比对（O(block_size) 校验）

**学到的点**：工程中对 hash 的使用应该是 hash + 精确校验，不能只依赖 hash。

---

### L-009: block_size=256 的 trade-off

**来源**：config.py:17
**分类**：[algo]

block_size 必须是 256 的倍数。这个选择的 trade-off：
- 太小（如 32）：block table 太长，管理开销大
- 太大（如 1024）：内部碎片严重（一个 block 最多浪费 1023 token）
- 256 是经验值：对 4K-8K context 长度，碎片率 < 10%

**追问**：如果 prompt 平均 300 token，block_size=256 意味着每个 prompt 要 2 个 block，第二个 block 只用 44 个位置，浪费 212 个位置。这种情况下 block_size=128 更合适。

---

## Phase 3：张量并行实验

### L-010: 3090 无 NVLink 的 TP 通信瓶颈

**来源**：实际测量
**分类**：[perf]

RTX 3090 没有 NVLink，TP 通信只能走 PCIe 4.0 x16（~32 GB/s 单向）。
- TP=2：每次 all_reduce 通信量 = 2 * hidden_size * bs * 2 bytes
- TP=8：通信量更大，但更重要的是 PCIe 是共享总线，多卡同时通信会争抢带宽

**学到的点**：消费级卡做 TP 主要用于学习目的，不是生产方案。生产用 A100/H100 + NVLink。

---

### L-011: TP 对 KV cache 的影响

**来源**：model_runner.py:110
**分类**：[infra]

TP=2 时，每卡只存一半的 KV heads。所以 TP 不仅分摊了权重，还分摊了 KV cache。

**学到的点**：TP 的好处不只是分摊权重，还能分摊 KV cache，提高并发能力。

---

### L-012: SharedMemory + pickle 的 TP IPC 设计

**来源**：model_runner.py:61-83
**分类**：[infra]

nano-vllm 的 TP 通信分两层：
1. 控制面：SharedMemory + pickle 序列化方法名和参数
2. 数据面：NCCL all_reduce / gather

局限：
- SharedMemory 大小固定（2^20 = 1MB），传大的 Sequence 列表可能溢出
- pickle 序列化有安全风险
- 没有错误处理

**学到的点**：生产级 TP 用 NCCL P2P 或 custom CUDA IPC，不用 SharedMemory + pickle。

---

## Phase 4：代码改造

### L-013: torch.compile 对动态 shape 的处理

**来源**：改造采样器时
**分类**：[perf]

添加 top_p 采样后，sorted_probs 的 shape 和 probs 一样，不会破坏 torch.compile。但如果引入了依赖输入数据的条件分支（如 if top_p < 1.0），torch.compile 会生成 guard，每次检查条件。

**学到的点**：torch.compile 对静态 shape 最友好。动态 shape 会导致 recompilation，首次调用会重新编译。

---

### L-014: Sequence 的 pickle 序列化优化

**来源**：sequence.py:72-83
**分类**：[infra]

Decode 时只序列化 last_token（1 个 int），而不是整个 token_ids 列表。这对 TP 场景很重要：每步都要把 Sequence 从 rank 0 传到 rank>0，如果每次都传完整的 token_ids（可能几千个 int），通信开销很大。

---

### L-015: @torch.compile 和 torch.inference_mode 的兼容性

**来源**：model_runner.py:195, 222
**分类**：[perf]

@torch.inference_mode 禁用 autograd，减少内存开销。但它和 @torch.compile 是兼容的——compile 在推理模式下也能工作。

**学到的点**：推理时应该总是用 inference_mode（比 no_grad 更激进，完全禁用版本计数）。

---

## Phase 5：后训练实验

### L-016: SFT 的 loss masking 必须和 chat template 对齐

**来源**：SFT 训练时
**分类**：[training]

SFT 时只在 assistant 回答的 token 上计算 loss。但 assistant 的 token 边界必须和 chat template 的 special token 对齐。如果 chat template 用 ChatML 格式，assistant 部分从 im_start 之后开始。如果 loss mask 边界不对，模型会学到格式 token 或漏掉部分回答。

---

### L-017: DPO 的 beta 和学习率要配合调

**来源**：DPO 训练时
**分类**：[training]

DPO 中 beta 控制偏离 reference model 的程度，学习率控制每步更新的幅度。两者是耦合的：
- beta 大 + 学习率大 → 模型剧烈震荡
- beta 小 + 学习率小 → 几乎不更新
- 经验：beta=0.1 + lr=5e-7 是安全起点

---

### L-018: Reference model 可以 offload 到 CPU 节省显存

**来源**：RLHF 训练时
**分类**：[perf]

RLHF 中 Actor + Critic + Reference + RM 四个模型同时在 GPU 上，显存压力很大。Reference model 只在计算 KL 惩罚时用，可以 offload 到 CPU，用的时候再搬到 GPU。代价是每次 CPU-GPU 搬运增加 ~100ms，但对于 RLHF 的 rollout 阶段来说可以接受。

---

## Phase 6：Agent 场景

### L-019: 多轮对话中 KV cache 复用的前提

**来源**：Agent 实验
**分类**：[agent]

多轮对话复用 prefix cache 的前提是：system prompt 的 token 序列完全一致。如果不同用户的 system prompt 有微小差异（如用户名不同），整个 prefix cache 就失效了。

**学到的点**：Agent 设计中，应该把动态部分（用户名、时间等）放在 user message 中，而不是放在 system prompt 中。

---

### L-020: Agent 的 context 膨胀是最大瓶颈

**来源**：Agent 实验
**分类**：[agent]

每轮对话都会增加 context 长度。如果 Agent 调用了 10 次工具，context 可能增长到 10K+ tokens。这导致：
1. KV cache 消耗增加
2. Prefill 时间增加（每轮都要重新 prefill 历史）
3. 注意力质量下降（长 context 下注意力分散）

**学到的点**：Agent 设计要控制 context 长度——定期压缩历史、只保留最近 N 轮、把工具返回结果做摘要。

---

## 模板：后续实验记录

### L-XXX: [标题]

**来源**：[文件:行号 或 实验过程]
**分类**：[infra/algo/training/agent/debug/perf]

[详细描述]

**学到的点**：
[核心 takeaway]
