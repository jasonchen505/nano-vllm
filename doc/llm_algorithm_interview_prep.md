# LLM 算法实习面试准备指南
## 从 nano-vllm 出发，覆盖后训练、Agent 应用、推理系统

> **作者**：2028级研一 MS 在读，目标方向 LLM 算法实习
> **定位**：面试不是考你背了多少论文，而是考你**能不能把一个点讲到让人信服你做过**
> **方法论**：以 nano-vllm 代码为锚点，向算法侧展开，每个知识点配"面试怎么讲"的模板

---

## 目录

- [Part A: 从 infra 到算法 — nano-vllm 教会你的底层直觉](#part-a)
- [Part B: 后训练 (Post-Training) 全链路](#part-b)
- [Part C: LLM Agent 应用](#part-c)
- [Part D: 评估、数据与工程实践](#part-d)
- [Part E: 模拟面试题库与讲法模板](#part-e)

---

<a id="part-a"></a>
## Part A: 从 infra 到算法 — nano-vllm 教会你的底层直觉

> 面试官问"你对推理系统了解多少"时，你需要的不是背 vLLM 架构，而是展示**算法和 infra 的交叉理解**。

### A.1 为什么算法岗也要懂推理？

| 你做的算法工作 | 底层推理知识怎么帮你 |
|---------------|---------------------|
| **SFT 数据筛选** | 理解 prefill/decode 成本 → 知道为什么要过滤长 prompt、截断冗余 |
| **RLHF online 采样** | 理解 continuous batching → 知道为什么 rollout 阶段吞吐是瓶颈 |
| **DPO 训练** | 理解 KV cache 内存模型 → 知道 why chosen/rejected response 长度影响 GPU 内存 |
| **Agent 工具调用** | 理解 prefix cache → 知道 why 多轮对话复用 system prompt KV 可以降低 TTFT |
| **长上下文** | 理解 PagedAttention → 知道 why 128K context 不是线性扩展 |
| **模型评估** | 理解 TP/PP → 知道 why 大模型评估需要分布式，如何配置 batch size |

**面试讲法**：
> "我学推理系统不只是为了做 infra，而是因为**算法和 infra 是同一个问题的两面**。比如我做 RLHF 的 rollout，瓶颈在在线采样的吞吐——如果不理解 continuous batching，就不知道为什么 rollout 要用 vLLM 而不是 naive generate。再比如做 Agent 多轮对话，如果不理解 prefix cache，就不知道为什么 system prompt 的 KV 可以跨轮复用。"

---

### A.2 nano-vllm 中 3 个算法岗必须内化的直觉

#### 直觉 1：Prefill 和 Decode 是两种完全不同的计算模式

```
Prefill:  处理 N 个 prompt tokens → compute-bound (矩阵乘)
Decode:   每步生成 1 个 token → memory-bound (权重读取 >> 计算)
```

**算法含义**：
- Prefill 是"一次算完所有 prompt"，瓶颈在 FLOPs → 所以 SFT 时长 prompt 训练慢
- Decode 是"每步读一次全部权重"，瓶颈在 HBM 带宽 → 所以推理时 batch 越大效率越高
- **Chunked Prefill** 的意义：把长 prompt 分块，避免 decode 被阻塞 → 对应 Agent 场景中"长 system prompt + 短回复"

**面试延伸**：
> "所以做 Agent 时，如果 system prompt 有 4K tokens，每次用户消息都要重新 prefill 4K，TTFT 很高。但如果用 prefix cache，system prompt 的 KV 只算一次，后续每轮只 prefill 用户的新消息。"

#### 直觉 2：KV Cache 是 LLM 推理的核心瓶颈

从 nano-vllm 的 `allocate_kv_cache()` 看内存模型：
```python
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_bytes
# 例：Qwen3-0.6B, 28 layers, 256 block_size, 8 kv_heads, 64 head_dim, bf16
# = 2 * 28 * 256 * 8 * 64 * 2 = 14.6 MB per block
# RTX 4070 8GB, 90% 利用率 → 约 500 blocks → 约 128K tokens 总 KV 容量
```

**算法含义**：
- KV Cache 大小 = `2 × L × seq_len × d_model × dtype_bytes`（每条序列）
- 这就是 why RLHF rollout 不能无限并发 — 每条 sequence 消耗 KV cache
- DPO 的 chosen + rejected 两条 response，如果长度差很大，KV cache 浪费严重
- 长上下文（128K）的 KV cache 是 4K 的 32 倍 → PagedAttention 按需分配是唯一出路

#### 直觉 3：采样温度影响的不只是质量，还有吞吐

nano-vllm 的 `Sampler`：
```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
```

**算法含义**：
- 温度低 → 分布尖锐 → 几乎 deterministic → 输出长度稳定
- 温度高 → 分布平坦 → 随机性大 → 容易产生长回复或 EOS 提前
- **top-p/top-k** 本质是在 decode 端做"概率截断"，影响生成长度分布
- 对 RLHF 来说：rollout 需要足够多样性（温度 0.7-1.0），但 reward model 训练需要确定性（温度 0）

---

### A.3 从 nano-vllm 代码看"算法选择的工程代价"

| 算法选择 | 推理代价 | nano-vllm 对应代码 |
|---------|---------|-------------------|
| **用 system prompt** | 首次 prefill 长，但后续可 prefix cache | `BlockManager.can_allocate()` 命中 |
| **Chain-of-Thought** | 输出 token 数暴增 → decode 时间线性增长 | `Scheduler.postprocess()` 的 max_tokens 检查 |
| **多轮 Agent** | 每轮 context 增长 → KV cache 线性增长 | `Sequence.num_tokens` 递增 |
| **DPO chosen/rejected** | 同时跑两条 → batch size 减半 | `max_num_seqs` 约束 |
| **ReAct 格式** | 输出含大量格式 token（`<tool_call>`）→ decode 开销 | `Sampler` 的 EOS 检查 |
| **Speculative Decoding** | draft model + target model 双倍 KV cache | nano-vllm 未实现，面试可谈 |

---

<a id="part-b"></a>
## Part B: 后训练 (Post-Training) 全链路

> 面试算法岗，后训练是**必考**。你需要讲清楚"为什么后训练"以及"每个阶段在做什么"。

### B.1 为什么需要后训练？

```
Pre-training (海量文本) → 基座模型 (会续写，不会对话)
    ↓
Post-training (高质量指令数据) → 对齐后的模型 (会对话、遵循指令、安全)
```

**核心问题**：预训练模型的训练目标是"预测下一个 token"，但用户要的是"回答问题"。这个 gap 就是后训练要解决的。

**面试讲法**：
> "预训练学到的是世界知识的压缩，但'怎么用这些知识'是后训练教的。SFT 教格式，RLHF 教偏好，DPO 把两者统一。类比：预训练是读书，SFT 是学写作文格式，RLHF 是老师批改作文告诉你哪种写法更好。"

---

### B.2 Supervised Fine-Tuning (SFT)

#### 核心思想
用 `(instruction, response)` 数据对模型做监督学习。

#### 数据格式
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is PagedAttention?"},
    {"role": "assistant", "content": "PagedAttention is a memory management technique..."}
  ]
}
```

#### 关键技术细节

**Loss Masking**：只在 assistant 回答的 token 上计算 loss，不在 system/user prompt 上计算。

```python
# 伪代码
for token_id, label in zip(input_ids, labels):
    if label == -100:  # prompt 部分
        continue       # 不计算 loss
    loss += cross_entropy(logits[token_id], label)
```

**面试必问**：为什么不 prompt 也计算 loss？
> "因为 prompt 是输入条件，不是模型要学习的输出。如果 prompt 也算 loss，模型会'记住'训练数据中的 prompt 格式，而不是学习'怎么根据 prompt 生成回答'。"

#### SFT 常见陷阱

| 问题 | 表现 | 原因 |
|------|------|------|
| **过拟合** | 训练 loss 低但评估差 | 数据量太少 / epoch 太多 |
| **格式崩坏** | 输出重复 / 格式混乱 | 数据质量差 / 学习率太高 |
| **灾难性遗忘** | 通用能力下降 | SFT 数据覆盖范围太窄 |
| **长度偏好** | 输出越来越长 | 长回复 loss 绝对值更大（token 更多） |

**面试讲法**：
> "SFT 最大的坑不是技术，是数据。1000 条高质量 SFT 数据 > 10 万条低质量数据。我理解的'高质量'是：(1) 指令多样化，覆盖不同任务；(2) 回答风格一致，不能有的长有的短；(3) 答案准确，不能有事实错误。"

---

### B.3 Reward Model (RM)

#### 核心思想
训练一个模型来评估"哪个回答更好"。

#### 训练数据格式
```json
{
  "prompt": "Explain PagedAttention",
  "chosen": "PagedAttention divides KV cache into fixed-size blocks...",
  "rejected": "PagedAttention is a type of attention mechanism that..."
}
```

#### 训练目标
```python
# Bradley-Terry 模型
loss = -log(sigmoid(rm_score(chosen) - rm_score(rejected)))
```

**关键点**：
- RM 不是生成模型，是**打分模型**
- 输入 `(prompt, response)` → 输出标量 score
- 训练数据来自人类标注的偏好对

#### RM 的常见问题

| 问题 | 解法 |
|------|------|
| **Reward Hacking** | 模型学会"讨好"RM 而非真正好 → DPO 可以缓解 |
| **标注一致性低** | 不同标注者偏好不同 → 需要清晰标注指南 + 多人投票 |
| **泛化性差** | RM 在分布外 prompt 上不可靠 → 需要多样化训练数据 |
| **长度偏好** | RM 倾向给长回答高分 → 需要长度正则化 |

**面试必问**：Reward Hacking 具体是什么？
> "模型发现 RM 有一些偏好漏洞，比如 RM 偏好长回答，模型就会生成又长又空洞的内容来获得高分，而不是真正有价值的回答。就像学生发现老师喜欢字写得多的，就开始凑字数。"

---

### B.4 RLHF (Reinforcement Learning from Human Feedback)

#### 核心思想
用 RM 的分数作为 reward，用 PPO 算法优化策略模型。

#### 算法流程
```
1. 从 prompt 池采样一批 prompt
2. 策略模型 (actor) 生成 response → rollout
3. RM 对 (prompt, response) 打分 → reward
4. 计算 KL 散度惩罚：KL(π_θ || π_ref) → 防止偏离太远
5. PPO 更新 actor
6. 重复
```

#### PPO 的 4 个模型

| 模型 | 作用 | 是否更新 |
|------|------|---------|
| **Actor** (策略模型) | 生成 response | 是 |
| **Critic** (价值模型) | 估计 state value | 是 |
| **Reference** (参考模型) | 计算 KL 惩罚 | 否 |
| **Reward Model** | 打分 | 否 |

**内存挑战**：同时 4 个大模型在 GPU 上 → 通常 Actor 和 Critic 用同一基座 + 不同 head，Reference 和 RM 可以 offload 到 CPU。

**面试必问**：为什么需要 Reference 模型？
> "没有 KL 惩罚，actor 会'hack' RM：生成 RM 给高分但人类觉得奇怪的回复。Reference 模型（初始 SFT 模型的冻结副本）提供一个锚点，通过 KL(π_θ || π_ref) 惩罚，让 actor 不要偏离原始能力太远。"

#### PPO 的实现要点

```python
# 简化版 PPO loss
advantages = rewards - values  # 优势函数
ratio = exp(log_prob_new - log_prob_old)  # 重要性采样比率
clipped_ratio = torch.clamp(ratio, 1-eps, 1+eps)
policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
value_loss = (values - returns) ** 2
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy  # entropy bonus
```

**核心概念**：
- **Advantage** = reward - baseline → 表示"比平均水平好多少"
- **Clipping** → 防止一步更新太大，稳定训练
- **Entropy Bonus** → 鼓励探索，防止过早收敛

---

### B.5 DPO (Direct Preference Optimization)

#### 核心思想
**不需要单独的 RM**，直接从偏好数据优化策略。

#### 核心公式
```python
# DPO loss
loss = -log(sigmoid(
    beta * (log_prob_chosen - log_prob_ref_chosen)
    - beta * (log_prob_rejected - log_prob_ref_rejected)
))
```

其中：
- `log_prob_chosen`: 当前模型对 chosen response 的 log prob
- `log_prob_ref_chosen`: reference 模型对 chosen response 的 log prob
- `beta`: 温度参数，控制偏离 reference 的程度

#### DPO vs RLHF 对比

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| **需要 RM** | 是 | 否 |
| **需要在线采样** | 是（rollout） | 否（离线数据） |
| **训练稳定性** | 低（PPO 很难调） | 高 |
| **数据需求** | 在线生成 + 偏好对 | 只需偏好对 |
| **理论等价** | RL + RM | 从 RLHF 公式推导得到 |
| **效果上限** | 更高（可以 online 探索） | 受限于离线数据质量 |
| **工程复杂度** | 高（4 个模型） | 低（2 个模型） |

**面试必问**：DPO 的数学推导？
> "RLHF 的目标是 `max E[r(x,y)] - β·KL(π||π_ref)`。这个优化问题的最优解有闭式解：`π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)`。反解出 `r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log(Z(x))`。代入 Bradley-Terry 模型 `P(y_w > y_l) = σ(r(x,y_w) - r(x,y_l))`，Z(x) 被消掉，得到 DPO loss。"

#### DPO 的变体

| 变体 | 改进点 | 核心区别 |
|------|-------|---------|
| **IPO** | 解决 DPO 的 overfitting | 用 squared loss 替代 log-sigmoid |
| **KTO** | 只需要 thumbs up/down，不需要 pair | 单条数据独立优化 |
| **ORPO** | 合并 SFT 和 DPO | 一个阶段完成 |
| **SimPO** | 不需要 reference model | 用 response 平均 log prob 做隐式长度正则 |
| **SPPO** | 自我对弈 | 模型自己生成 pair |
| **GRPO** | DeepSeek 的 group relative policy optimization | 去掉 Critic，用 group 内相对 reward |

**面试讲法**：
> "DPO 的核心 insight 是：RM 和 RL 都是手段，不是目的。我们真正想要的是'让 chosen 的概率比 rejected 高'。DPO 直接优化这个目标，绕过了 RM 和 PPO。但代价是它依赖离线数据——如果偏好对覆盖不够，DPO 的上限就不如 RLHF。"

---

### B.6 GRPO (Group Relative Policy Optimization)

> DeepSeek-R1 使用的算法，面试中**高频考点**。

#### 核心思想
去掉 Critic 模型，用同一 prompt 下多个采样的**组内相对排名**做 advantage 估计。

```
对每个 prompt x:
  1. 采样 G 个 response: {y_1, ..., y_G}
  2. 用 RM 打分: {r_1, ..., r_G}
  3. 组内标准化: advantage_i = (r_i - mean(r)) / std(r)
  4. PPO-style 更新: policy_loss = -advantage * log_prob
  5. KL 惩罚: + β * KL(π || π_ref)
```

**优势**：
- 不需要 Critic → 省一半显存
- 组内标准化 → 自然的 baseline，不需要 value function
- 适合数学/代码任务（可以有确定性 reward）

**面试讲法**：
> "GRPO 的 insight 是：与其学一个 Critic 来估计 baseline，不如直接用同一批采样的均值做 baseline。这样既省了 Critic 的显存，又避免了 value function 估计不准的问题。DeepSeek-R1 用 GRPO 训出了推理能力，证明了这个方法的有效性。"

---

### B.7 后训练链路总结

```
Base Model
    │
    ▼ SFT (指令数据，loss 只算 response)
    │
SFT Model
    │
    ├─────────────────────── DPO (偏好对，离线) ──→ 对齐模型
    │
    ├──────────── RM 训练 (偏好对)
    │                    │
    │                    ▼
    │               RLHF/PPO (在线采样 + RM 打分) ──→ 对齐模型
    │
    └──────────── GRPO (在线采样 + 组内相对排序) ──→ 推理增强模型
```

---

<a id="part-c"></a>
## Part C: LLM Agent 应用

> Agent 是 LLM 应用的最前沿方向，也是面试热点。

### C.1 什么是 LLM Agent？

```
LLM = 大脑 (理解 + 推理 + 生成)
Agent = 大脑 + 工具 + 记忆 + 规划
```

**核心公式**：
```
Agent = LLM + Tools + Memory + Planning
```

| 组件 | 作用 | 面试常问 |
|------|------|---------|
| **LLM** | 理解指令、推理、生成 | "你用的什么模型？为什么？" |
| **Tools** | 执行代码、调用 API、搜索 | "怎么让模型学会调工具？" |
| **Memory** | 短期 (对话历史) + 长期 (知识库) | "context window 不够怎么办？" |
| **Planning** | 分解任务、制定执行计划 | "怎么处理复杂多步任务？" |

---

### C.2 Agent 的核心设计模式

#### 模式 1：ReAct (Reasoning + Acting)

```
Thought: 我需要搜索 PagedAttention 的论文
Action: search("PagedAttention SOSP 2023")
Observation: Found paper "Efficient Memory Management for Large Language Model
             Serving with PagedAttention" by Kwon et al.
Thought: 我找到了论文，现在需要总结核心贡献
Action: finish("PagedAttention 的核心贡献是...")
```

**面试必问**：ReAct vs CoT vs ToT？

| 模式 | 特点 | 适用场景 |
|------|------|---------|
| **CoT** | 纯推理，逐步思考 | 数学、逻辑题 |
| **ReAct** | 推理 + 行动交替 | 需要外部信息的任务 |
| **ToT** | 树状搜索，多路径探索 | 需要回溯的复杂问题 |

#### 模式 2：Function Calling / Tool Use

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name"}
    },
    "required": ["city"]
  }
}
```

**实现方式**：
1. **Prompt-based**：在 system prompt 中描述工具，模型输出特殊格式
2. **Fine-tuned**：用工具调用数据做 SFT，模型原生输出 function call

**面试必问**：如何处理工具调用失败？
> "重试 + 降级 + 告知用户。(1) 设置重试次数和间隔；(2) 失败后尝试备用工具；(3) 如果所有工具都失败，告诉用户当前无法完成任务。关键是 Agent 不能'沉默'——必须有明确的失败反馈。"

#### 模式 3：Multi-Agent

```
用户 → Planner Agent → 分解任务
              │
              ├─→ Research Agent → 搜索信息
              ├─→ Code Agent → 写代码
              └─→ Review Agent → 审核结果
              │
              ← Planner Agent → 整合输出
```

**面试常问**：Multi-Agent vs Single-Agent？

| 维度 | Single Agent | Multi Agent |
|------|-------------|-------------|
| **复杂度** | 低 | 高（需要协调） |
| **可靠性** | 较高（无通信失败） | 较低（agent 间可能误解） |
| **适用场景** | 简单任务 | 复杂、需要多种能力的任务 |
| **成本** | 低 | 高（多次 LLM 调用） |

---

### C.3 Agent 面试深挖点

#### 深挖 1：Context Window 管理

**问题**：Agent 多轮对话后，context 超过 window size 怎么办？

**答案层次**：
1. **短期方案**：截断旧消息，保留最近 N 轮
2. **中期方案**：摘要压缩（让 LLM 总结之前对话）
3. **长期方案**：RAG（把历史存入向量数据库，按需检索）
4. **架构方案**：分层记忆（工作记忆 = context window，长期记忆 = 外部存储）

**联系 nano-vllm**：
> "从推理角度，context 越长，KV cache 越大，decode 越慢。PagedAttention 按需分配可以缓解内存问题，但 compute cost 是线性增长的。所以 Agent 设计要控制 context 膨胀——不是把所有历史塞进去，而是只塞相关信息。"

#### 深挖 2：工具调用的训练方法

**问题**：怎么让模型学会正确调用工具？

**答案层次**：
1. **Prompt Engineering**：在 system prompt 中写清楚工具 schema + 示例
2. **SFT**：用工具调用数据（用户指令 → function call JSON）做微调
3. **RLHF**：用工具调用成功/失败作为 reward 信号

**数据格式示例**：
```json
{
  "messages": [
    {"role": "system", "content": "You have access to: [get_weather, search]"},
    {"role": "user", "content": "What's the weather in Beijing?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}}
    ]},
    {"role": "tool", "content": "{\"temp\": 25, \"condition\": \"sunny\"}"},
    {"role": "assistant", "content": "The weather in Beijing is sunny, 25°C."}
  ]
}
```

#### 深挖 3：Planning 能力怎么评估？

**问题**：怎么测试 Agent 的规划能力？

**评估维度**：
| 维度 | 测试方法 |
|------|---------|
| **任务分解** | 给复杂任务，看能否拆成合理子步骤 |
| **工具选择** | 给多个工具，看能否选最合适的 |
| **错误恢复** | 故意让工具失败，看能否调整计划 |
| **效率** | 完成任务的步骤数和 token 消耗 |
| **终止条件** | 知道什么时候该停止并给出答案 |

#### 深挖 4：Agent 幻觉问题

**问题**：Agent 调用工具后，模型可能编造工具返回的结果怎么办？

**解法**：
1. **忠实引用**：强制模型引用工具返回的原始内容
2. **结果校验**：用另一个 LLM 或规则检查输出是否与工具结果一致
3. **Confidence Score**：让模型输出置信度，低置信度时再次调用工具

---

### C.4 RAG (Retrieval-Augmented Generation)

> RAG 是 Agent 的重要组成部分，单独拎出来讲。

#### 核心流程
```
用户问题
    │
    ▼
Query Encoding (问题向量化)
    │
    ▼
Vector Search (从知识库检索 Top-K 文档)
    │
    ▼
Context Assembly (把检索结果拼进 prompt)
    │
    ▼
LLM Generation (基于检索结果生成回答)
```

#### RAG 的关键设计决策

| 决策点 | 选项 | 权衡 |
|-------|------|------|
| **Chunk 策略** | 固定长度 / 按句 / 按段 | 小 chunk 精确但缺上下文，大 chunk 反之 |
| **Embedding 模型** | OpenAI / BGE / E5 | 效果 vs 成本 vs 延迟 |
| **检索方式** | 稠密 / 稀疏 / 混合 | 稠密语义好，稀疏关键词匹配好 |
| **Top-K** | 3 / 5 / 10 | K 小精度高，K 大召回高 |
| **重排序** | 有 / 无 | 重排序提高精度但增加延迟 |

**面试必问**：RAG 的常见失败模式？
> "(1) 检索失败——问题和文档表述不同，检索不到；(2) 检索到但不相关——embedding 质量差；(3) 检索到相关文档但模型忽略了——需要 prompt 引导；(4) 检索到矛盾信息——需要模型做冲突消解。"

---

<a id="part-d"></a>
## Part D: 评估、数据与工程实践

### D.1 模型评估

#### 评估维度

| 维度 | 基准 | 说明 |
|------|------|------|
| **通用能力** | MMLU, ARC, HellaSwag | 多选题，测知识广度 |
| **推理** | GSM8K, MATH, HumanEval | 数学/代码推理 |
| **中文** | C-Eval, CMMLU | 中文知识评测 |
| **对话** | MT-Bench, AlpacaEval | 开放式对话质量 |
| **安全** | TruthfulQA, BBQ | 幻觉/偏见测试 |
| **Agent** | ToolBench, API-Bank | 工具调用能力 |

#### 评估的关键指标

| 指标 | 含义 | 陷阱 |
|------|------|------|
| **Pass@k** | k 次采样至少一次正确的概率 | k>1 不代表实际可用性 |
| **Win Rate** | 和基座模型的 A/B 对比 | 受评估 prompt 分布影响 |
| **ELO Rating** | Chatbot Arena 排名 | 受用户偏好影响 |
| **Perplexity** | 困惑度 | 低 PPL ≠ 好回答 |

**面试讲法**：
> "评估是后训练最难的部分。自动评估（MMLU 等）只能测'对不对'，测不出'好不好'。人工评估（MT-Bench）能测质量但贵且慢。最好的方式是两者结合：自动评估做筛选，人工评估做最终判断。"

---

### D.2 数据工程

#### SFT 数据质量标准

| 维度 | 标准 | 检查方法 |
|------|------|---------|
| **准确性** | 答案事实正确 | 人工抽检 + LLM-as-judge |
| **多样性** | 覆盖不同任务类型 | 统计任务分布 |
| **一致性** | 风格/格式统一 | 格式校验脚本 |
| **难度梯度** | 从简单到复杂 | 统计 prompt 长度分布 |
| **安全性** | 无有害内容 | 安全分类器过滤 |

#### 偏好数据标注指南

```
标注者任务：给定 prompt 和两个回答 (A/B)，选择更好的一个

评判维度（按优先级）：
1. 事实正确性：哪个回答更准确？
2. 有用性：哪个回答更有帮助？
3. 无害性：哪个回答更安全？
4. 格式：哪个回答更清晰易读？
5. 语言质量：哪个回答更自然流畅？
```

---

### D.3 训练工程实践

#### 超参数经验

| 参数 | SFT 典型值 | DPO 典型值 | 说明 |
|------|-----------|-----------|------|
| **Learning Rate** | 1e-5 ~ 5e-5 | 5e-7 ~ 5e-6 | DPO 更小，怕偏离太远 |
| **Epochs** | 2~3 | 1~2 | 过拟合风险 |
| **Batch Size** | 128~512 | 64~256 | 大 batch 更稳定 |
| **Beta (DPO)** | - | 0.1~0.5 | 温度参数 |
| **Max Seq Len** | 2048~8192 | 2048~8192 | 受 GPU 内存限制 |
| **Warmup** | 3~10% steps | 3~10% steps | 标准 warmup |

#### 显存估算

```
训练显存 ≈ 模型参数 × dtype_bytes × 4 (参数 + 梯度 + 优化器状态 + 激活)
例：7B 模型, BF16, Adam
  = 7B × 2 × 4 = 56 GB (不含激活)
  + 激活 (取决于 seq_len 和 batch_size)
  ≈ 需要 2×A100-80GB

推理显存 ≈ 模型参数 × dtype_bytes + KV Cache
例：7B 模型, BF16, 4K context
  = 14 GB + KV Cache (~2GB)
  ≈ 需要 1×A100-40GB
```

**联系 nano-vllm**：
> "nano-vllm 的 `allocate_kv_cache()` 就是在做这个计算：把 GPU 总内存减去模型权重和峰值激活，剩下的都给 KV cache。这让我对推理内存有了直观感受。"

---

<a id="part-e"></a>
## Part E: 模拟面试题库与讲法模板

### E.1 基础概念题 (5 分钟快答)

**Q1: 解释 prefill 和 decode 的区别**
> "Prefill 是处理整个 prompt，一次算完所有 token 的 KV，计算密集型；decode 是逐个生成 token，每步只算 1 个 token 的 KV 但要读全部模型权重，访存密集型。这决定了推理优化的两个方向：prefill 靠 chunked prefill 和 prefix cache 优化；decode 靠 batching 提高 GPU 利用率。"

**Q2: 什么是 KV Cache？为什么需要？**
> "Transformer 的注意力需要历史所有 token 的 K 和 V。如果每次生成新 token 都重新算所有历史 token 的 KV，复杂度是 O(n²)。KV Cache 把已算过的 K/V 存起来，每步只算新 token 的 KV 并追加，复杂度降到 O(n)。代价是内存：每个 token 每层占 2×d_model×dtype_bytes。"

**Q3: 解释 PagedAttention**
> "传统 KV Cache 为每个请求预分配 max_seq_len 的连续内存，浪费严重。PagedAttention 借鉴操作系统的虚拟内存，把 KV 切成固定大小的 block（如 256 token），按需分配物理 block，通过 block_table 做逻辑-物理映射。好处：(1) 消除内部碎片；(2) 不同请求可以共享 prefix block；(3) 显存利用率从 ~30% 提到 ~90%。"

**Q4: 解释连续批处理**
> "静态批处理等所有请求都完成才处理下一批，长请求拖慢短请求。连续批处理每个 step 重新组 batch：完成的请求移出，新请求加入。这样 GPU 始终满载。实现上需要调度器管理每个请求的状态（prefill/decode/finished）。"

---

### E.2 后训练深度题 (10 分钟展开)

**Q5: 从 SFT 到 RLHF 到 DPO，讲讲你的理解**
> "SFT 用监督数据教模型'格式'——怎么按指令回答。但 SFT 只教了'什么是正确的回答'，没教'什么是更好的回答'。RLHF 通过 RM 量化'好'，用 PPO 优化。但 RLHF 工程复杂——4 个模型同时在 GPU，PPO 训练不稳定。DPO 的 insight 是：RM 和 RL 只是手段，我们真正要的是让 chosen 概率 > rejected 概率。DPO 直接从偏好数据优化这个目标，不需要 RM 和在线采样。代价是依赖离线数据质量。"

**Q6: DPO 的 beta 参数怎么选？**
> "Beta 控制偏离 reference model 的程度。beta 太大，模型几乎不动（过于保守）；beta 太小，模型可能过拟合到偏好数据。经验上 0.1~0.5，通常从 0.1 开始调。验证方法：看 DPO 训练过程中 chosen reward 和 rejected reward 的差距——应该稳定增长但不 diverge。"

**Q7: 怎么做 SFT 数据筛选？**
> "三个层面：(1) 质量过滤——用 GPT-4 或人工评分，去掉低质量回答；(2) 多样性采样——按任务类型/难度分层采样，避免某个类型过多；(3) 去重——semantically 相似的数据只保留一条。工具：embedding 去重、LLM-as-judge 打分、规则过滤（长度/格式/语言）。"

**Q8: Reward Hacking 怎么缓解？**
> "多管齐下：(1) RM 集成——多个 RM 投票，降低单个 RM 偏好的影响；(2) KL 惩罚——不让 actor 偏离 reference 太远；(3) 长度正则——RM 分数除以长度的某个幂；(4) 定期更新 RM——用最新的 actor 输出重新标注偏好数据；(5) 改用 DPO/GRPO——绕过 RM 的局限。"

---

### E.3 Agent 应用题 (10 分钟展开)

**Q9: 设计一个代码助手 Agent**

```
System Prompt:
你是一个代码助手，可以：
1. 搜索代码库 (search_code)
2. 读取文件 (read_file)
3. 运行代码 (run_code)
4. 写入文件 (write_file)

工作流：
- 理解用户需求
- 搜索相关代码
- 阅读并理解
- 修改并测试
- 返回结果

ReAct 格式：
Thought: <推理>
Action: <工具调用>
Observation: <工具返回>
... (循环)
Thought: <最终推理>
Answer: <给用户的回答>
```

**关键设计点**：
1. **工具描述要精确**——模糊的工具描述会导致模型误用
2. **限制最大步数**——防止 Agent 无限循环
3. **错误处理**——工具失败后要能降级或重试
4. **状态管理**——多轮对话中保持上下文

**Q10: Agent 的评估方法？**
> "三层评估：(1) 任务完成率——给定任务，Agent 是否成功完成？(2) 效率——用了多少步、多少 token、多少工具调用？(3) 用户体验——回答是否有用、格式是否清晰？具体 benchmark：SWE-Bench（代码）、WebArena（网页浏览）、ToolBench（工具调用）、GAIA（通用 Agent）。"

**Q11: 怎么让 Agent 更可靠？**
> "可靠性 = 正确性 + 一致性 + 可恢复性。具体措施：(1) 输出格式校验——用 JSON Schema 强制工具调用格式；(2) 自我反思——每步后问'这个结果合理吗'；(3) 备选方案——每个步骤有 Plan B；(4) 人类介入——关键操作前确认；(5) 日志追踪——每步记录，方便 debug。"

---

### E.4 系统设计题 (15 分钟白板)

**Q12: 设计一个 RLHF 训练系统**

```
┌─────────────────────────────────────────────────┐
│                  Prompt 池                       │
│  (从 SFT 训练数据 / 用户 query 中采样)           │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│              Rollout Worker (vLLM)                │
│  - Actor 模型生成 response                        │
│  - 连续批处理保证吞吐                             │
│  - Prefix cache 复用 system prompt                │
└──────────────────────┬───────────────────────────┘
                       │ (prompt, response)
                       ▼
┌──────────────────────────────────────────────────┐
│              Reward Worker                        │
│  - RM 对 (prompt, response) 打分                  │
│  - 批量推理，高吞吐                               │
└──────────────────────┬───────────────────────────┘
                       │ (prompt, response, reward)
                       ▼
┌──────────────────────────────────────────────────┐
│              PPO Trainer                          │
│  - 计算 advantage (reward - baseline)             │
│  - PPO 更新 actor                                │
│  - KL 惩罚 (vs reference model)                  │
│  - 保存 checkpoint                               │
└──────────────────────────────────────────────────┘
```

**关键设计决策**：
1. **Rollout 用 vLLM**：因为需要高吞吐在线采样，vLLM 的 continuous batching 最合适
2. **RM 和 Actor 分开部署**：避免显存争抢
3. **异步 pipeline**：Rollout 和 Training 可以流水线化——一组在训练，另一组在 rollout
4. **数据 buffer**：rollout 结果存 buffer，trainer 从 buffer 取，解耦速度差异

**Q13: 设计一个生产级 Agent 系统**

```
用户请求
    │
    ▼
┌─────────────────────┐
│    Gateway Layer     │ ← 限流、认证、日志
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Router Agent       │ ← 路由到合适的 Agent
└──────────┬──────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
  Agent A  Agent B  Agent C  ← 不同专长的 Agent
    │      │      │
    └──────┼──────┘
           │
           ▼
┌─────────────────────┐
│   Tool Layer         │ ← API / 数据库 / 代码执行
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Response Builder   │ ← 格式化、安全检查、引用
└─────────────────────┘
```

---

### E.5 开放讨论题 (5 分钟)

**Q14: LLM 推理的未来趋势？**
> "三个方向：(1) 更高效的注意力——线性注意力、稀疏注意力，从 O(n²) 降到 O(n)；(2) 更好的量化——INT4/INT2 甚至更低，让大模型跑在消费级 GPU；(3) 推测解码——用小模型 draft、大模型 verify，2-3x 加速。长期来看，硬件和算法会共同演进——HBM 带宽决定了 decode 的理论上限，算法要做的就是在有限带宽内挤出更多吞吐。"

**Q15: 后训练的未来趋势？**
> "三个方向：(1) RL 回归——DPO 虽然简单但上限有限，DeepSeek-R1 用 GRPO 证明了 RL 的价值；(2) 数据合成——用强模型生成训练数据，形成 self-play 循环；(3) 过程奖励——不只奖励最终答案，还奖励中间推理步骤（Process Reward Model）。最终目标是让模型'学会思考'，而不只是'学会回答'。"

**Q16: Agent 的未来趋势？**
> "三个方向：(1) 端到端训练——不是 prompt 拼接，而是把 tool use 作为模型能力训练进去；(2) 长期记忆——从 RAG 到真正的 episodic memory；(3) 多模态 Agent——不只处理文本，还能看图、操作 GUI、控制物理设备。终极目标是 AGI 级 Agent：给一个高层目标，自主完成所有子任务。"

---

## 附录：面试前 1 小时速览

```
✅ 后训练链路：SFT → RM → RLHF/DPO/GRPO，每步在做什么
✅ DPO 公式推导：从 RLHF 闭式解推导，Z(x) 消掉
✅ GRPO 核心：去 Critic，组内相对排序做 advantage
✅ Agent 四要素：LLM + Tools + Memory + Planning
✅ ReAct 模式：Thought → Action → Observation 循环
✅ RAG 流程：Query → Retrieve → Assemble → Generate
✅ PagedAttention：block 化 KV cache，按需分配
✅ Continuous Batching：prefill/decode 交织，每 step 重新组 batch
✅ Prefix Cache：滚动哈希 + token 逐位校验
✅ KV Cache 内存公式：2 × L × seq_len × d_model × dtype_bytes
✅ nano-vllm 讲法：STAR 法则，强调"从 infra 到算法"的交叉理解
```

---

**最后一句话**：面试不是考你背了多少，而是考你**能不能把一个点讲清楚，并且让面试官相信你真的理解了**。用 nano-vllm 当锚点，把推理系统和算法知识串起来，就是你最大的差异化优势。
