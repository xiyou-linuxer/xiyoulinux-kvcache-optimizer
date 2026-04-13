现代大语言模型（LLM）推理任务中，为了加速自注意力计算，通常会使用KV缓存（Key-Value Cache）来存储之前已计算的键值对。然而，随着agent、graph、工具利用带来的序列长度的增加，KV缓存会线性增长，成为显存占用的主要部分。这意味着，当处理长达8192甚至更长的上下文时，KV Cache本身就可能占用数GB乃至数十GB的显存，成为显存资源的主要消耗者。在资源受限的设备上，如何高效管理KV缓存，将频繁访问的缓存数据保留在显存中，将不常访问的数据迁移到内存或外存中，从而在有限的显存条件下优化更长的序列的吞吐性能，是提升大模型推理效率的关键。

## 赛题任务

*   设计并实现一个系统（可以是内核模块、用户态库或运行时库），能够监控LLM推理进程对KV缓存的访问模式（如访问频率、最近访问时间等），并以此定义每个KV缓存块的热度。
*   基于监控数据，实现一个动态的KV缓存迁移策略，将热数据（当前生成步骤频繁访问的KV缓存）保留在显存中，将温数据（偶尔访问的KV缓存）迁移到内存中，将冷数据（长时间未访问的KV缓存）迁移到外存交换区中。
*   实现一个针对KV缓存数据的存储分配器，优化显存和内存的碎片问题，特别是随着生成过程动态增长导致的碎片，提高缓存命中率，并支持与PagedAttention、vLLM等现有KV缓存优化技术的兼容与集成。
    *   **目标**：在不修改大语言模型推理代码（或仅需少量配置）的前提下，最大化推理吞吐速率，最小化GPU因等待KV缓存迁移而产生的等待时间，并支持更长的序列长度。

## 赛题特征

*   **KV缓存感知**：系统能理解KV缓存的特有访问模式（解码时按时间步顺序访问，但不同层、不同头的访问频率不同）
*   **自适应策略**：迁移策略能根据具体模型结构（如Transformer层数、注意力头数）和输入特征（如序列长度、任务类型）自适应调整
*   **透明集成**：无需修改主流推理框架（如vLLM、HuggingFace Transformers、TensorRT-LLM）的源代码
*   **性能评估**：以KV缓存命中率、平均访问延迟、有效上下文长度扩展倍数、吞吐率提升幅度为核心评估指标
*   **硬件支持**：支持Intel/AMD CPU + NVIDIA/AMD GPU硬件结构，或者国产异构计算硬件，支持高速互联如NVLINK、HCCS

## 参考资料

1.  Rhu M, Gimelshein N, Clemons J, et al. vDNN: Virtualized deep neural networks for scalable, memory-efficient neural network design[C]//2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 2016: 1-13.
2.  Foyer C, Goglin B, Proaño A R. A survey of software techniques to emulate heterogeneous memory systems in high-performance computing[J]. Parallel Computing, 2023, 116: 103023.
3.  Rajbhandari S, Rasley J, Ruwase O, et al. Zero: Memory optimizations toward training trillion parameter models[C]//SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2020: 1-16.
4.  Yanyu Liu, Jingying Fu, Sixiang Liu, Yitian Zou, Shouhua Zhang, and Jiehan Zhou. 2026. KV Cache Compression for Inference Efficiency in LLMs: A Review. In Proceedings of the 4th International Conference on Artificial Intelligence and Intelligent Information Processing (AIIIP '25). Association for Computing Machinery, New York, NY, USA, 207–212. https://doi.org/10.1145/3778534.3778567

## 评审要点

### 维度权重说明

*   **性能提升**：30% 在标准测例上的吞吐/延迟改善
*   **系统开销**：20% CPU 占用、迁移延迟、额外内存消耗
*   **通用性**：20% 是否支持多种模型/框架/国产硬件配置
*   **创新性**：15% 策略设计是否新颖（如结合 ML 预测）
*   **工程质量**：15% 代码结构、文档、可复现性