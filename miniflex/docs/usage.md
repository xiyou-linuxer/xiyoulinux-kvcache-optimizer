# MiniFlexKV 使用说明

MiniFlexKV 是一个从 0 编写的**单机单卡** KV Cache 插件，作为 vLLM **V1** KV connector
使用，外部缓存只含 **CPU + SSD** 两层。本文覆盖安装、启动和配置。
项目结构见 [project_structure.md](project_structure.md)，实测结论见 [validation.md](validation.md)。

## 1. 安装

### 系统依赖
C++ 扩展（`csrc/`）使用 io_uring 做 SSD I/O，需先装系统库：

```bash
sudo apt install liburing-dev
```

### 安装包
包含 torch C++ 扩展，**必须**用 `--no-build-isolation`，否则隔离构建会重新拉一个
可能与运行时 CUDA 不匹配的 torch：

```bash
# 在已装好 torch 的环境里
pip install --no-build-isolation .

# 需要 vLLM connector 时，带上可选依赖
pip install --no-build-isolation '.[vllm]'
```

- `requires-python >= 3.10`（代码使用了 match-case）。
- 运行依赖：`torch`、`numpy`、`pyzmq`；`vllm` 为可选依赖（不装也能 import 本包，
  只是用不了 connector）。

## 2. 在 vLLM 下启动（单机单卡）

通过 `--kv-transfer-config` 让 vLLM 加载 connector。**单卡必须**加
`--disable-hybrid-kv-cache-manager`（vLLM 的 HMA 不支持外部 connector）。

```bash
ENABLE_MINIFLEX=1 \
MINIFLEX_GPU_REGISTER_PORT=ipc:///tmp/miniflex.sock \
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"MiniFlexConnectorV1","kv_connector_module_path":"miniflex.integration.vllm.connector","kv_role":"kv_both"}' \
  --disable-hybrid-kv-cache-manager
```

> JSON 必须在同一行，不要被终端折行（折行会混入换行符导致 `Invalid JSON`）。
> 仓库根目录的 `run_vllm_miniflex.sh` 已封装好启动 + 残留清理，推荐直接用：
> `bash run_vllm_miniflex.sh`。

启动后是标准的 OpenAI 兼容服务：

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model>","prompt":"The capital of France is","max_tokens":16}'
```
- Base URL：`http://localhost:8000/v1`，API Key 填 `EMPTY`（默认不校验）。

### 常见启动坑
- **残留进程**：vLLM 的引擎进程名是 `VLLM::EngineCore`，`pkill -f "vllm serve"` 杀不到它，
  会残留并占显存。清理用：
  `pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "vllm serve"; rm -f /tmp/miniflex_*.sock`。
- **代理**：若设了 `socks://` 代理，vLLM 拉 HF 配置会报错。模型已在本地缓存时加
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 走本地。

## 3. 配置

配置分两类，优先级和归属不同。

### 模型结构字段（不可手动覆盖）
`num_layers / head_size / num_kv_heads / use_mla / dtype / tokens_per_block / tp_size / dp_size`
**只从 vLLM 的 config 推导**，不开放 env/文件覆盖——它们必须与真实模型一致，
写错会导致 KV 的 shape/layout 与实际不符、数据被静默写坏。

### 行为 / 缓存调优字段（可通过 env 或 JSON 文件配置）

顶层开关（`MiniFlexConfig`）：

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `ENABLE_MINIFLEX` | `1` | 是否启用 MiniFlex |
| `MINIFLEX_GPU_REGISTER_PORT` | `ipc:///tmp/miniflex_gpu_register.sock` | worker 向 TransferManager 注册 GPU 的 ZMQ 端点 |
| `MINIFLEX_ENABLE_BATCH` | `0` | 是否把多个任务合并成一个 batch task |
| `MINIFLEX_SYNC_GET` | `0` | 同步等待 GET 加载完成（见下方说明） |
| `MINIFLEX_CONFIG_PATH` | 空 | 指向一个 JSON 配置文件 |
| `MINIFLEX_DEBUG` | 空 | 设非空则打印 PUT/GET/launch/finished 等调试日志到 stderr |

缓存层（`CacheConfig`，env 名 → 字段）：

| 环境变量 | 字段 |
|---|---|
| `MINIFLEX_NUM_CPU_BLOCKS` | num_cpu_blocks |
| `MINIFLEX_ENABLE_SSD` / `MINIFLEX_NUM_SSD_BLOCKS` | enable_ssd / num_ssd_blocks |
| `MINIFLEX_SSD_CACHE_DIR` / `MINIFLEX_SSD_FILE_PREFIX` | ssd_cache_dir / ssd_file_prefix |
| `MINIFLEX_SSD_MAX_FILE_SIZE_GB` / `MINIFLEX_USE_DIRECT_IO` | ssd_max_file_size_gb / use_direct_io |
| `MINIFLEX_EVICTION_POLICY` / `MINIFLEX_EVICT_RATIO` | eviction_policy / evict_ratio |
| `MINIFLEX_EVICT_START_THRESHOLD` / `MINIFLEX_HIT_ADD_COUNTS` | evict_start_threshold / hit_add_counts |
| `MINIFLEX_PROTECTED_THRESHOLD` | protected_threshold |
| `MINIFLEX_CPU_LAYOUT_TYPE` / `MINIFLEX_SSD_LAYOUT_TYPE` | cpu_layout_type / ssd_layout_type |

默认 CPU-only（`enable_cpu=True`、`enable_ssd=False`、`num_cpu_blocks=1024`）。
启用 SSD 需同时给 `MINIFLEX_ENABLE_SSD=1` 和 `MINIFLEX_SSD_CACHE_DIR`。

JSON 文件示例（`MINIFLEX_CONFIG_PATH` 指向它）：

```json
{
  "enable_miniflex": true,
  "gpu_register_port": "ipc:///tmp/miniflex.sock",
  "enable_batch": false,
  "sync_get": false,
  "cache_config": {
    "num_cpu_blocks": 4096,
    "enable_ssd": true,
    "num_ssd_blocks": 65536,
    "ssd_cache_dir": "/data/miniflex_cache"
  }
}
```

### 关于 `MINIFLEX_SYNC_GET`
默认 `0`（异步）：GET 加载在后台进行，由调度循环轮询上报完成，性能最好。
设 `1` 时 connector 在 `build_connector_meta` 内**同步阻塞**等待 GET 加载完成，
保证前向读取前数据已就位——这会牺牲性能（阻塞热路径），仅用于调试，
或极低频且要求"发一次立刻可复用"的场景。一般保持默认 `0`。
