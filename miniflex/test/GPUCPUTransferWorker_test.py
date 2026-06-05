import time
import torch

from miniflex.common.transfer import TransferType
from miniflex.transfer.worker import GPUCPUTransferWorker


def make_test_worker(num_layers=2, num_blocks=4, block_shape=(4, 2, 8), dtype=torch.float16):
    """绕过 __init__，直接构造测试用的 worker
    
    _transfer_impl 只用 5 个属性，全部手动赋值
    每层 tensor 形状: [2, num_blocks, *block_shape]
    """
    worker = GPUCPUTransferWorker.__new__(GPUCPUTransferWorker)
    worker.num_layers = num_layers
    worker.gpu_num_blocks = num_blocks
    worker.cpu_num_blocks = num_blocks
    worker.gpu_tensors_list = [
        torch.zeros(2, num_blocks, *block_shape, device='cuda', dtype=dtype)
        for _ in range(num_layers)
    ]
    worker.cpu_tensors_list = [
        torch.zeros(2, num_blocks, *block_shape, pin_memory=True, dtype=dtype)
        for _ in range(num_layers)
    ]
    worker.gpu_stream = torch.cuda.Stream()
    return worker


def launch_transfer(worker, src_ids, dst_ids, transfer_type):
    with torch.cuda.stream(worker.gpu_stream):
        worker._transfer_impl(src_ids, dst_ids, transfer_type)
    worker.gpu_stream.synchronize()


def logical_transfer_bytes(worker, num_blocks):
    block_elements = worker.gpu_tensors_list[0][:, 0].numel()
    return worker.num_layers * num_blocks * block_elements * worker.gpu_tensors_list[0].element_size()


def benchmark_transfer(worker, transfer_type, num_blocks, warmup=5, repeats=20):
    src_ids = torch.randperm(worker.gpu_num_blocks, dtype=torch.int64)[:num_blocks]
    dst_ids = torch.randperm(worker.cpu_num_blocks, dtype=torch.int64)[:num_blocks]
    if transfer_type == TransferType.H2D:
        src_ids, dst_ids = dst_ids, src_ids

    for _ in range(warmup):
        launch_transfer(worker, src_ids, dst_ids, transfer_type)

    start = time.perf_counter()
    for _ in range(repeats):
        launch_transfer(worker, src_ids, dst_ids, transfer_type)
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / repeats * 1000
    payload_bytes = logical_transfer_bytes(worker, num_blocks)
    bandwidth_gbps = payload_bytes / (avg_ms / 1000) / 1e9
    print(
        f"{transfer_type.value:>3} blocks={num_blocks:<4} "
        f"payload={payload_bytes / 1024 / 1024:7.2f} MiB "
        f"avg={avg_ms:8.3f} ms "
        f"bandwidth={bandwidth_gbps:7.2f} GB/s"
    )


def test_d2h_roundtrip():
    """GPU 已知数据 → D2H → CPU 验证"""
    worker = make_test_worker(num_layers=2, num_blocks=4)
    
    # 每层 GPU 填不同已知值
    for layer_id in range(worker.num_layers):
        worker.gpu_tensors_list[layer_id].fill_(float(layer_id + 1))
    
    # 搬 src [0, 2] → dst [1, 3]
    src_ids = torch.tensor([0, 2], dtype=torch.int64)
    dst_ids = torch.tensor([1, 3], dtype=torch.int64)
    launch_transfer(worker, src_ids, dst_ids, TransferType.D2H)
    
    for layer_id in range(worker.num_layers):
        cpu_layer = worker.cpu_tensors_list[layer_id]
        v = float(layer_id + 1)
        assert cpu_layer[:, 1].eq(v).all(), f"Layer {layer_id} block 1 应该是 {v}"
        assert cpu_layer[:, 3].eq(v).all(), f"Layer {layer_id} block 3 应该是 {v}"
        assert cpu_layer[:, 0].eq(0).all(), f"Layer {layer_id} block 0 未涉及，应保持 0"
        assert cpu_layer[:, 2].eq(0).all(), f"Layer {layer_id} block 2 未涉及，应保持 0"
    print("✓ test_d2h_roundtrip")


def test_h2d_roundtrip():
    """CPU 已知数据 → H2D → GPU 验证"""
    worker = make_test_worker(num_layers=2, num_blocks=4)
    
    for layer_id in range(worker.num_layers):
        worker.cpu_tensors_list[layer_id].fill_(float(layer_id + 10))
    
    src_ids = torch.tensor([0, 2], dtype=torch.int64)
    dst_ids = torch.tensor([1, 3], dtype=torch.int64)
    launch_transfer(worker, src_ids, dst_ids, TransferType.H2D)
    
    for layer_id in range(worker.num_layers):
        gpu_layer = worker.gpu_tensors_list[layer_id]
        v = float(layer_id + 10)
        assert gpu_layer[:, 1].eq(v).all(), f"Layer {layer_id} block 1 应该是 {v}"
        assert gpu_layer[:, 3].eq(v).all(), f"Layer {layer_id} block 3 应该是 {v}"
        assert gpu_layer[:, 0].eq(0).all()
        assert gpu_layer[:, 2].eq(0).all()
    print("✓ test_h2d_roundtrip")


def test_distinct_values_preserved():
    """不同 block 填不同值，验证 src→dst 一一对应（不会窜数据）"""
    worker = make_test_worker(num_layers=1, num_blocks=4)
    
    # GPU 每个 block 填不同值: block 0=0, 1=100, 2=200, 3=300
    gpu_layer = worker.gpu_tensors_list[0]
    for blk in range(4):
        gpu_layer[:, blk].fill_(float(blk * 100))
    
    # 搬 src [0, 2, 3] → dst [3, 1, 0]
    # 期望: cpu[3]=0, cpu[1]=200, cpu[0]=300, cpu[2]=0(未动)
    src_ids = torch.tensor([0, 2, 3], dtype=torch.int64)
    dst_ids = torch.tensor([3, 1, 0], dtype=torch.int64)
    launch_transfer(worker, src_ids, dst_ids, TransferType.D2H)
    
    cpu_layer = worker.cpu_tensors_list[0]
    assert cpu_layer[:, 3].eq(0).all(),   "cpu[3] 应该是 0（来自 gpu[0]）"
    assert cpu_layer[:, 1].eq(200).all(), "cpu[1] 应该是 200（来自 gpu[2]）"
    assert cpu_layer[:, 0].eq(300).all(), "cpu[0] 应该是 300（来自 gpu[3]）"
    assert cpu_layer[:, 2].eq(0).all(),   "cpu[2] 未动，应保持 0"
    print("✓ test_distinct_values_preserved")


def test_empty_transfer_is_noop():
    worker = make_test_worker(num_layers=2, num_blocks=4)
    empty = torch.tensor([], dtype=torch.int64)
    # 不应抛错
    worker._transfer_impl(empty, empty, TransferType.D2H)
    worker._transfer_impl(empty, empty, TransferType.H2D)
    print("✓ test_empty_transfer_is_noop")


def test_out_of_range_raises():
    worker = make_test_worker(num_layers=2, num_blocks=4)
    
    cases = [
        # (src_ids, dst_ids, desc)
        ([0, 100], [0, 1],  "src 越界（大）"),
        ([0, 1],   [0, 100],"dst 越界（大）"),
        ([-1, 0],  [0, 1],  "src 负数"),
        ([0, 1],   [-1, 0], "dst 负数"),
    ]
    for src, dst, desc in cases:
        src_t = torch.tensor(src, dtype=torch.int64)
        dst_t = torch.tensor(dst, dtype=torch.int64)
        try:
            worker._transfer_impl(src_t, dst_t, TransferType.D2H)
            assert False, f"{desc}: 应该抛 ValueError"
        except ValueError:
            pass
    print("✓ test_out_of_range_raises")


def test_size_mismatch_raises():
    worker = make_test_worker(num_layers=1, num_blocks=4)
    src_t = torch.tensor([0, 1, 2], dtype=torch.int64)
    dst_t = torch.tensor([0, 1],    dtype=torch.int64)
    try:
        worker._transfer_impl(src_t, dst_t, TransferType.D2H)
        assert False, "size 不一致应该抛 ValueError"
    except ValueError:
        pass
    print("✓ test_size_mismatch_raises")


def test_invalid_transfer_type_raises():
    """非 D2H/H2D 应抛错（虽然 enum 限制了，但代码里有这层防御）"""
    worker = make_test_worker(num_layers=1, num_blocks=4)
    src = torch.tensor([0], dtype=torch.int64)
    dst = torch.tensor([1], dtype=torch.int64)
    try:
        # 传一个不存在的 transfer type
        worker._transfer_impl(src, dst, "not_a_type")
        assert False, "应该抛 ValueError"
    except ValueError:
        pass
    print("✓ test_invalid_transfer_type_raises")


def test_transfer_performance_report():
    """打印端到端传输耗时和逻辑 payload 带宽。"""
    worker = make_test_worker(
        num_layers=8,
        num_blocks=1024,
        block_shape=(16, 4, 64),
        dtype=torch.float16,
    )

    print("\nGPUCPUTransferWorker benchmark")
    print(
        f"config: layers={worker.num_layers}, blocks={worker.gpu_num_blocks}, "
        f"layer_shape={tuple(worker.gpu_tensors_list[0].shape)}, "
        f"dtype={worker.gpu_tensors_list[0].dtype}"
    )
    print("note: payload bandwidth counts logical KV bytes only, not temporary gather/scatter copies")

    for num_blocks in (1, 4, 16, 64, 256):
        benchmark_transfer(worker, TransferType.D2H, num_blocks)
    for num_blocks in (1, 4, 16, 64, 256):
        benchmark_transfer(worker, TransferType.H2D, num_blocks)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "测试需要 CUDA 环境"
    
    test_d2h_roundtrip()
    test_h2d_roundtrip()
    test_distinct_values_preserved()  # 这个最关键：验证 index 对应关系，不窜数据
    test_empty_transfer_is_noop()
    test_out_of_range_raises()
    test_size_mismatch_raises()
    test_invalid_transfer_type_raises()
    test_transfer_performance_report()
    
    print("\n🎉 全过")
