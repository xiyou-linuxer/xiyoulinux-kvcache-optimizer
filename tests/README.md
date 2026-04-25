# tests 目录说明

本目录用于存放 baseline 测试相关的可执行脚本，服务于原始 vLLM 基线验证与后续对比测试。

## 测试内容

当前测试工作主要围绕原始 vLLM 服务展开，覆盖以下内容：

1. 短上下文基础验证
2. 中长上下文真实数据基线
3. 超长上下文压力测试
4. 共享前缀命中验证
5. 混合压力并发测试
6. 固定长度 microbenchmark 参考线

## 脚本说明

- `vllm_baseline_runner.py`
  - 五组 baseline 场景的主执行脚本
  - 负责发送请求并采集 `/metrics`、GPU、CPU 等运行数据
- `summarize_baseline_runs.py`
  - 汇总各轮测试生成的 `summary.json`
  - 输出统一的 CSV 结果表，便于后续对比与汇报

## 文档索引

详细文档统一放在 [`docs/test`](../docs/test) 目录下。

- [基线测试报告.md](../docs/test/%E5%9F%BA%E7%BA%BF%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A.md)
  - 当前测试的主报告
  - 包含背景、环境、用例、结果、数据分析与测试结论
- [baseline测试记录.md](../docs/test/supporting/baseline%E6%B5%8B%E8%AF%95%E8%AE%B0%E5%BD%95.md)
  - baseline 过程记录文档
  - 保存各组测试结果、最终采用结果与关键观察
- [完整测试执行流程_方案B.md](../docs/test/supporting/%E5%AE%8C%E6%95%B4%E6%B5%8B%E8%AF%95%E6%89%A7%E8%A1%8C%E6%B5%81%E7%A8%8B_%E6%96%B9%E6%A1%88B.md)
  - 完整测试执行流程
  - 包含执行顺序、命令、路径和注意事项
- [比赛导向测试指标设计说明.md](../docs/test/supporting/%E6%AF%94%E8%B5%9B%E5%AF%BC%E5%90%91%E6%B5%8B%E8%AF%95%E6%8C%87%E6%A0%87%E8%AE%BE%E8%AE%A1%E8%AF%B4%E6%98%8E.md)
  - 指标设计说明
  - 解释当前阶段为什么选这些指标、这些指标分别说明什么
- [测试数据集设计方案.md](../docs/test/supporting/%E6%B5%8B%E8%AF%95%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AE%BE%E8%AE%A1%E6%96%B9%E6%A1%88.md)
  - 数据集设计与选型说明
  - 说明五组场景的数据来源和选取理由
