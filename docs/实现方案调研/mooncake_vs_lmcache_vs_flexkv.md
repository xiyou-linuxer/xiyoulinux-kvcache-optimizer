# Mooncake、LMCache、FlexKV 的关系与分工

## 目录

1. [[#1. 先说结论|先说结论]]
2. [[#2. 为什么这三个项目容易搞混|为什么这三个项目容易搞混]]
3. [[#3. 三者分别是什么|三者分别是什么]]
4. [[#4. 三者分别解决什么问题|三者分别解决什么问题]]
5. [[#5. 三者在整个系统中的位置|三者在整个系统中的位置]]
6. [[#6. 一个请求到来后，三者分别在做什么|一个请求到来后，三者分别在做什么]]
7. [[#7. 谁负责命中判断，谁负责搬运，谁负责存储|谁负责命中判断谁负责搬运谁负责存储]]
8. [[#8. 三者最大的实现差异|三者最大的实现差异]]
9. [[#9. 用一句话区分三者|用一句话区分三者]]
10. [[#10. 适合放进汇报里的总结|适合放进汇报里的总结]]

---

## 1. 先说结论

如果只用一句话概括三者关系：

> Mooncake 更像“高性能传输和对象池底座”，LMCache 更像“KV 对象缓存中间层”，FlexKV 更像“前缀匹配驱动的分层 KV 回填子系统”。

也可以更直白一点：

- `Mooncake`：负责 **怎么高速传**
- `LMCache`：负责 **怎么把 KV 当成对象缓存起来并查找/预取**
- `FlexKV`：负责 **怎么根据前缀命中结果把旧 KV 从多层缓存回填回来**

所以三者不是互相完全替代的关系，而更像不同层的能力：

- Mooncake 偏底座
- LMCache / FlexKV 偏上层 KV cache 系统

---

## 2. 为什么这三个项目容易搞混

因为它们都和同一件事有关：

- 已经算出来的 KV cache，不要浪费

而且它们都在提：

- CPU
- SSD
- Remote
- 异步加载
- 分布式共享
- 降低 TTFT
- 减少重复 Prefill

所以如果只看表面描述，很容易觉得：

- 它们都在做 KV cache 加速
- 那是不是只是三个名字不同的同类项目

但实际上它们的重心完全不同。

真正区分它们，关键不是看“都做不做 KV cache”，而是看：

> 它们分别把“KV cache 加速”问题切在了哪一层。

---

## 3. 三者分别是什么

### 3.1 Mooncake 是什么

Mooncake 更像一个 **高性能对象传输和对象池平台**。

它的核心是：

- `Transfer Engine`
- `Mooncake Store`
- `P2P Store`

它要解决的是：

- KV cache、hidden states、embedding 这类大对象
- 如何在 GPU、CPU、SSD、远端节点之间高效流动

所以它不是主要做“前缀判断”的，而是更偏：

- 高速传输
- 分离式推理中的对象共享
- 多介质、多协议支持

### 3.2 LMCache 是什么

LMCache 更像一个 **KV 对象缓存中间层**。

它把 KV cache 看成一批可以被：

- chunk 化
- segment 化
- key 化
- lookup
- prefetch
- retrieve
- store

的缓存对象。

它最突出的实现特点包括：

- `ChunkedTokenDatabase`
- `SegmentTokenDatabase`
- `lookup + prefetch`
- `layerwise pipeline`
- `CacheBlend`
- `L1/L2 controller`

所以 LMCache 更像是在做：

> 一个围绕 KV 对象生命周期组织起来的缓存服务层。

### 3.3 FlexKV 是什么

FlexKV 更像一个 **前缀匹配驱动的分层 KV 回填系统**。

它最核心的主线是：

- 先判断前缀命中了多少
- 再决定命中的 KV 在 CPU / SSD / Remote 哪一层
- 再构建明确的数据回填路径
- 让 GPU 驱逐后的旧 KV 还能继续复用

它最有辨识度的点包括：

- `block` 粒度前缀哈希
- `RadixTree`
- `GlobalCacheEngine`
- `TransferOpGraph`
- `TransferEngine`
- 分层回填和分布式索引快照

所以 FlexKV 的核心更像是：

> 一个围绕 prefix hit 结果来驱动多层回填和复用的 KV 子系统。

---

## 4. 三者分别解决什么问题

### Mooncake 主要解决什么

Mooncake 主要解决的是：

- 大对象怎么高效跨节点传
- 分离式 prefill/decode 场景下，KV 怎么在不同实例之间传
- 如何利用 RDMA、多 NIC、零拷贝等能力提高带宽和降低 CPU 开销

所以它主要解决的是：

> **传输问题** 和 **对象池问题**。

### LMCache 主要解决什么

LMCache 主要解决的是：

- 如何把 KV 拆成可缓存对象
- 如何查到已经存在的对象
- 如何预取这些对象
- 如何把它们重新注入 GPU
- 如何在更复杂场景下做非前缀复用

所以它主要解决的是：

> **KV 对象缓存问题**。

### FlexKV 主要解决什么

FlexKV 主要解决的是：

- 如何识别最长前缀命中
- GPU 放不下时，怎样把旧 KV 留在 CPU / SSD / Remote
- 下次命中时怎样按层取回
- 怎样把回填、继续计算、下层保存串起来

所以它主要解决的是：

> **前缀驱动的分层 KV 保存与回填问题**。

---

## 5. 三者在整个系统中的位置

可以用分层图来理解。

```text
应用 / Serving API
        ↓
推理引擎（vLLM / SGLang / TRT-LLM）
        ↓
KV Cache 管理层（LMCache / FlexKV）
        ↓
高性能传输与远端对象底座（Mooncake）
```

这里的意思不是说它们总是严格叠在一起，而是说：

- Mooncake 更靠底层
- LMCache / FlexKV 更靠近推理引擎

### Mooncake 的位置

Mooncake 更像：

- 底层的共享对象通道
- 高性能传输底座
- 分布式对象池基础设施

### LMCache 的位置

LMCache 更像：

- 推理引擎和底层存储/传输之间的 KV 中间层

### FlexKV 的位置

FlexKV 更像：

- 紧贴推理引擎的前缀匹配与多层回填控制面

所以如果一定要比谁更靠近引擎：

- FlexKV 往往更贴近“引擎内部 KV 路径”
- LMCache 稍微更像“服务中间层”
- Mooncake 更靠近“底层传输/存储基础设施”

---

## 6. 一个请求到来后，三者分别在做什么

我们用一个非常简单的请求流程来看。

### 第一步：请求到达推理引擎

这时候 vLLM / SGLang 会先准备 token、slot mapping、KV buffer 等。

### 第二步：判断旧 KV 能不能复用

这里通常是 `LMCache` 或 `FlexKV` 在起作用。

- LMCache：把输入切成 chunk/segment，做 lookup
- FlexKV：做 block 级前缀匹配

### 第三步：如果命中了旧 KV，决定从哪里拿

这时候：

- FlexKV 会判断 CPU / SSD / Remote 哪层命中
- LMCache 会通过 StorageManager/backend 决定从哪个 backend 取对象

### 第四步：真正把数据搬回来

这一步就可能会用到 Mooncake：

- 如果对象在远端
- 如果需要高性能跨节点传输
- 如果是分离式 prefill/decode

Mooncake 更像执行“运输”这一步。

### 第五步：继续生成剩余部分

推理引擎继续对没命中的那部分执行计算。

### 第六步：新生成的 KV 再保存

- LMCache / FlexKV 会决定新 KV 怎么保存
- Mooncake 如果在链路里，则负责更底层的数据传输或远端写入

所以一个请求里：

- Mooncake 往往负责“运”
- LMCache/FlexKV 往往负责“管”

---

## 7. 谁负责命中判断，谁负责搬运，谁负责存储

这是三者最容易混的地方，我直接拆开。

### 谁负责命中判断

主要是：

- `LMCache`
- `FlexKV`

因为它们都在做：

- lookup
- prefix/segment/chunk hit 判断
- 命中 token 数估计
- 对后续 retrieve/store 的调度

Mooncake 本身通常不是主要做这一步。

### 谁负责搬运

最底层最像“专业搬运工”的是：

- `Mooncake`

尤其是在：

- RDMA
- 多 NIC
- 远端对象池
- prefill/decode 解耦

这些场景下，它的角色最明确。

当然 LMCache/FlexKV 也有自己的 GPU connector / backend / transfer path，但 Mooncake 的项目主线就是“高性能搬运”。

### 谁负责存储

这取决于层次：

- `FlexKV` 有自己的 CPU/SSD/Remote 层次和索引管理
- `LMCache` 有 `LocalCPUBackend/LocalDiskBackend/RemoteBackend/P2PBackend/...`
- `Mooncake` 自己则有 `Mooncake Store` / `P2P Store`

所以三者都涉及存储，但方式不同：

- FlexKV：偏“多层回填式缓存”
- LMCache：偏“对象缓存后端矩阵”
- Mooncake：偏“远端对象池/共享对象系统”

---

## 8. 三者最大的实现差异

### 差异 1：FlexKV 最重“前缀树 + 回填路径”

FlexKV 非常强调：

- `RadixTree`
- longest prefix match
- local/remote index
- `TransferOpGraph`
- 多层命中后如何回填

所以它是典型的：

- 前缀控制面很重
- 回填路径很明确

### 差异 2：LMCache 最重“对象化 + 预取 + layerwise”

LMCache 最突出的点是：

- chunk / segment object
- lookup + prefetch
- layerwise retrieve/store
- CacheBlend
- L1/L2 控制器

所以它是典型的：

- KV 对象化程度很高
- 流水线和隐藏回填延迟做得很重

### 差异 3：Mooncake 最重“传输和对象池底座”

Mooncake 最突出的点是：

- Transfer Engine
- Store / P2P Store
- 多协议、多介质
- 高带宽低 CPU 开销

所以它是典型的：

- 传输底座很重
- 对象池和分离式服务能力很强

---

## 9. 用一句话区分三者

### Mooncake

> Mooncake 是一个面向 AI 大对象的高性能传输和对象池底座。

### LMCache

> LMCache 是一个围绕 chunk/segment 对象做 lookup、prefetch、retrieve、store 的 KV cache 中间层。

### FlexKV

> FlexKV 是一个围绕前缀命中做多层保存、迁移和回填的 KV cache 子系统。

---

## 10. 总结


> 这三个项目虽然都与 KV cache 加速有关，但它们的切入点并不相同。Mooncake 更偏底层基础设施，解决的是 KV cache 等大对象在分离式系统中的高性能传输和远端对象池问题；LMCache 更偏缓存中间层，解决的是 KV chunk 的查找、预取、回填和非前缀复用问题；FlexKV 更偏前缀驱动的分层 KV 回填子系统，解决的是 GPU 放不下时，旧 KV 如何在 CPU、SSD 和远端节点中继续保存并在命中时重新回填到 GPU 的问题。

如果再压成最短一句：

> Mooncake 负责“怎么快传”，LMCache 负责“怎么按对象缓存和预取”，FlexKV 负责“怎么按前缀命中结果做分层回填”。
