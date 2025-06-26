# TextNLP Infrastructure

This directory contains the core infrastructure components for the TextNLP synthetic data platform.

## Directory Structure

```
infrastructure/
├── gpu/                    # GPU resource management
│   ├── gpu_resource_manager.py    # Multi-cloud GPU instance management
│   ├── gpu_health_monitor.py      # GPU health monitoring and alerts
│   ├── gpu_autoscaler.py          # Intelligent GPU autoscaling
│   └── cuda_setup.py              # CUDA/cuDNN configuration
│
├── storage/               # Model storage infrastructure
│   ├── model_shard_manager.py     # Large model sharding (>10GB)
│   ├── chunked_transfer.py        # Resumable file transfers
│   ├── model_version_manager.py   # Model versioning and lifecycle
│   ├── delta_update_manager.py    # Efficient model updates
│   └── unified_storage.py         # Unified storage interface
│
└── adapter.py             # Service adapter for unified deployment
```

## Components Overview

### GPU Infrastructure (✅ Complete)

#### GPU Resource Manager
- Multi-cloud support (AWS, GCP, Azure)
- GPU type selection and cost optimization
- Instance recommendations based on workload

```python
from infrastructure.gpu import GPUResourceManager, GPUType

manager = GPUResourceManager()
manager.register_provider("aws", "us-east-1")
instance = manager.get_gpu_instance("aws", GPUType.NVIDIA_T4, count=2)
```

#### GPU Health Monitor
- Real-time health monitoring
- Configurable alerts and thresholds
- Multi-cloud metric collection

```python
from infrastructure.gpu import GPUHealthMonitorManager

monitor_manager = GPUHealthMonitorManager()
await monitor_manager.start_monitoring("aws", ["i-1234567890"])
```

#### GPU Autoscaler
- Intelligent scaling based on metrics
- Cost-aware decisions
- Spot instance support

```python
from infrastructure.gpu import GPUAutoscalerManager, ScalingPolicy

policy = ScalingPolicy(min_instances=1, max_instances=10)
autoscaler_manager = GPUAutoscalerManager()
await autoscaler_manager.start_autoscaling("aws")
```

### Storage Infrastructure (✅ Complete)

#### Model Sharding
- Handles models >10GB efficiently
- Multiple sharding strategies
- Parallel upload/download

```python
from infrastructure.storage import AdaptiveShardManager

shard_manager = AdaptiveShardManager(storage_backend, config)
manifest = await shard_manager.shard_model_adaptive(
    model_path="large_model.safetensors",
    output_dir="/shards",
    model_id="llama-13b",
    strategy="layer_based"
)
```

#### Chunked Transfer
- Resume capability for interrupted transfers
- Progress tracking
- Multi-cloud support

```python
from infrastructure.storage import ChunkedTransferManager

transfer_manager = ChunkedTransferManager()
await transfer_manager.upload_file_chunked(
    local_path="model.bin",
    remote_path="models/model.bin",
    storage_backend=backend,
    progress_callback=callback
)
```

#### Model Versioning
- Complete lifecycle management
- Semantic versioning
- Deployment tracking

```python
from infrastructure.storage import ModelRegistry

registry = ModelRegistry(version_manager)
model = await registry.publish_model(
    model_path="model.safetensors",
    name="gpt2-custom",
    version="1.0.0",
    architecture="transformer",
    created_by="user"
)
```

#### Delta Updates
- Efficient model updates
- Multiple strategies (sparse, low-rank)
- 50-90% size reduction

```python
from infrastructure.storage import DeltaUpdateManager

delta_manager = DeltaUpdateManager(storage_backend)
patch = await delta_manager.create_delta_patch(
    old_model_path="v1.0.safetensors",
    new_model_path="v1.1.safetensors",
    source_version="1.0.0",
    target_version="1.1.0"
)
```

#### Unified Storage
- Single interface for all operations
- Automatic feature selection
- Multi-cloud abstraction

```python
from infrastructure.storage import UnifiedModelStorage, StorageConfig

config = StorageConfig(
    provider="aws",
    bucket_name="textnlp-models",
    enable_sharding=True,
    enable_versioning=True
)

storage = UnifiedModelStorage(config)
await storage.upload_model(
    model_path="model.safetensors",
    model_name="custom-llm",
    version="1.0.0"
)
```

## Integration with TextNLP

The infrastructure components are designed to work seamlessly with the TextNLP platform:

1. **GPU Management**: Automatically provisions and scales GPU resources based on inference load
2. **Model Storage**: Handles large language models with intelligent sharding and versioning
3. **Unified Interface**: Simple APIs hide complexity of multi-cloud operations

## Configuration

### Environment Variables
```bash
# Cloud Provider Settings
AWS_REGION=us-east-1
GCP_PROJECT_ID=your-project
AZURE_RESOURCE_GROUP=textnlp-rg

# Storage Configuration
MODEL_STORAGE_BUCKET=textnlp-models
MAX_SHARD_SIZE_GB=2.0
ENABLE_COMPRESSION=true

# GPU Configuration
DEFAULT_GPU_TYPE=nvidia-t4
ENABLE_SPOT_INSTANCES=true
GPU_AUTOSCALING=true
```

### Configuration Files
- `config/development.yaml`: Development environment settings
- `config/.env.example`: Environment variable template

## Testing

Run infrastructure tests:
```bash
# Unit tests
pytest tests/infrastructure/

# Integration tests
pytest tests/infrastructure/integration/

# GPU tests (requires GPU)
pytest tests/infrastructure/gpu/ -m gpu
```

## Monitoring

Infrastructure components expose metrics for monitoring:

- **GPU Metrics**: Utilization, memory, temperature
- **Storage Metrics**: Transfer speeds, storage usage, version counts
- **Cost Metrics**: Per-hour costs, spot savings

## Security

- All transfers use encryption in transit
- Model checksums verify integrity
- IAM roles for cloud access
- Audit logging for all operations

## Next Steps

With the infrastructure layer complete, the next phase is to build:
1. **Inference Endpoints**: REST API for model serving
2. **Request Batching**: Optimize GPU utilization
3. **Model Loading**: Efficient model caching
4. **Load Balancing**: Distribute requests across GPUs

---

For detailed documentation on each component, see the individual module docstrings or the docs/ directory.