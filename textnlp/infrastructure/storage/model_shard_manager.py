"""
Model Sharding Manager for TextNLP
Handles sharding of large language models (>10GB) for efficient storage and loading
"""

import os
import hashlib
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, BinaryIO
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import torch
import safetensors
from safetensors.torch import save_file, load_file
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ShardInfo:
    """Information about a model shard"""
    shard_id: int
    shard_name: str
    size_bytes: int
    checksum: str
    tensor_keys: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelShardConfig:
    """Configuration for model sharding"""
    max_shard_size_gb: float = 2.0  # Maximum size per shard in GB
    compression: bool = True
    checksum_algorithm: str = "sha256"
    parallel_uploads: int = 4
    chunk_size_mb: int = 10  # For streaming uploads


class ModelShardManager:
    """Manages sharding of large language models"""
    
    def __init__(self, storage_backend: Any, config: Optional[ModelShardConfig] = None):
        self.storage = storage_backend
        self.config = config or ModelShardConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_uploads)
        
    def calculate_shard_size(self, model_size_bytes: int) -> Tuple[int, int]:
        """Calculate optimal number of shards and size per shard"""
        max_shard_bytes = int(self.config.max_shard_size_gb * 1024 * 1024 * 1024)
        
        # Calculate minimum number of shards needed
        num_shards = max(1, (model_size_bytes + max_shard_bytes - 1) // max_shard_bytes)
        
        # Distribute size evenly across shards
        shard_size = model_size_bytes // num_shards
        
        # Ensure last shard isn't too small (at least 10% of average)
        if model_size_bytes % num_shards < shard_size * 0.1:
            num_shards = max(1, num_shards - 1)
            shard_size = model_size_bytes // num_shards
            
        return num_shards, shard_size
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum of data"""
        if self.config.checksum_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.config.checksum_algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unknown checksum algorithm: {self.config.checksum_algorithm}")
    
    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Get size of tensor in bytes"""
        return tensor.element_size() * tensor.nelement()
    
    async def shard_model(self, model_path: str, output_dir: str, 
                         model_id: str) -> Dict[str, Any]:
        """Shard a model into multiple files"""
        logger.info(f"Starting to shard model: {model_path}")
        
        # Load model state dict
        if model_path.endswith('.safetensors'):
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
        # Calculate total model size
        total_size = sum(self._get_tensor_size(tensor) for tensor in state_dict.values())
        num_shards, shard_size = self.calculate_shard_size(total_size)
        
        logger.info(f"Model size: {total_size / 1e9:.2f}GB, creating {num_shards} shards")
        
        # Group tensors into shards
        shards = self._group_tensors_into_shards(state_dict, num_shards, shard_size)
        
        # Save shards
        shard_infos = []
        tasks = []
        
        for shard_id, shard_data in enumerate(shards):
            task = self._save_shard(
                shard_id=shard_id,
                shard_data=shard_data,
                output_dir=output_dir,
                model_id=model_id
            )
            tasks.append(task)
        
        # Execute saves in parallel
        shard_infos = await asyncio.gather(*tasks)
        
        # Create manifest
        manifest = {
            "model_id": model_id,
            "total_size_bytes": total_size,
            "num_shards": num_shards,
            "shard_config": {
                "max_shard_size_gb": self.config.max_shard_size_gb,
                "compression": self.config.compression,
                "checksum_algorithm": self.config.checksum_algorithm
            },
            "shards": [info.__dict__ for info in shard_infos],
            "metadata": {
                "model_path": model_path,
                "created_at": datetime.utcnow().isoformat(),
                "format": "safetensors" if model_path.endswith('.safetensors') else "pytorch"
            }
        }
        
        # Save manifest
        manifest_path = os.path.join(output_dir, f"{model_id}_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Model sharding complete. Manifest saved to {manifest_path}")
        
        return manifest
    
    def _group_tensors_into_shards(self, state_dict: Dict[str, torch.Tensor], 
                                  num_shards: int, target_shard_size: int) -> List[Dict[str, torch.Tensor]]:
        """Group tensors into shards of approximately equal size"""
        # Sort tensors by size (largest first for better packing)
        tensor_items = sorted(
            state_dict.items(),
            key=lambda x: self._get_tensor_size(x[1]),
            reverse=True
        )
        
        # Initialize shards
        shards = [{"tensors": {}, "size": 0} for _ in range(num_shards)]
        
        # Distribute tensors using a greedy algorithm
        for key, tensor in tensor_items:
            tensor_size = self._get_tensor_size(tensor)
            
            # Find the shard with minimum current size
            min_shard_idx = min(range(num_shards), key=lambda i: shards[i]["size"])
            
            # Add tensor to the smallest shard
            shards[min_shard_idx]["tensors"][key] = tensor
            shards[min_shard_idx]["size"] += tensor_size
        
        # Return just the tensor dictionaries
        return [shard["tensors"] for shard in shards]
    
    async def _save_shard(self, shard_id: int, shard_data: Dict[str, torch.Tensor],
                         output_dir: str, model_id: str) -> ShardInfo:
        """Save a single shard"""
        shard_name = f"{model_id}_shard_{shard_id:04d}.safetensors"
        shard_path = os.path.join(output_dir, shard_name)
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Save shard using safetensors for efficiency
        save_file(shard_data, shard_path)
        
        # Calculate size and checksum
        shard_size = os.path.getsize(shard_path)
        
        # Calculate checksum
        with open(shard_path, 'rb') as f:
            checksum = self._calculate_checksum(f.read())
        
        # Upload to storage backend
        await self._upload_shard_to_storage(shard_path, f"{model_id}/{shard_name}")
        
        # Create shard info
        shard_info = ShardInfo(
            shard_id=shard_id,
            shard_name=shard_name,
            size_bytes=shard_size,
            checksum=checksum,
            tensor_keys=list(shard_data.keys()),
            metadata={
                "num_tensors": len(shard_data),
                "compression": self.config.compression
            }
        )
        
        logger.info(f"Saved shard {shard_id}: {shard_size / 1e6:.2f}MB, {len(shard_data)} tensors")
        
        return shard_info
    
    async def _upload_shard_to_storage(self, local_path: str, remote_path: str):
        """Upload shard to storage backend"""
        # This is a placeholder - actual implementation depends on storage backend
        await self.storage.upload_file(local_path, remote_path)
    
    async def load_sharded_model(self, manifest_path: str, 
                                device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Load a sharded model from manifest"""
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        logger.info(f"Loading sharded model {manifest['model_id']} with {manifest['num_shards']} shards")
        
        # Download and load shards in parallel
        tasks = []
        for shard_info in manifest['shards']:
            task = self._load_shard(
                shard_info=ShardInfo(**shard_info),
                model_id=manifest['model_id'],
                device=device
            )
            tasks.append(task)
        
        # Load all shards
        shard_dicts = await asyncio.gather(*tasks)
        
        # Merge all shards
        state_dict = {}
        for shard_dict in shard_dicts:
            state_dict.update(shard_dict)
        
        logger.info(f"Successfully loaded {len(state_dict)} tensors")
        
        return state_dict
    
    async def _load_shard(self, shard_info: ShardInfo, model_id: str,
                         device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Load a single shard"""
        # Download from storage
        local_path = f"/tmp/{shard_info.shard_name}"
        remote_path = f"{model_id}/{shard_info.shard_name}"
        
        await self._download_shard_from_storage(remote_path, local_path)
        
        # Verify checksum
        with open(local_path, 'rb') as f:
            checksum = self._calculate_checksum(f.read())
        
        if checksum != shard_info.checksum:
            raise ValueError(f"Checksum mismatch for shard {shard_info.shard_id}")
        
        # Load shard
        shard_dict = load_file(local_path, device=device)
        
        # Clean up temporary file
        os.remove(local_path)
        
        logger.info(f"Loaded shard {shard_info.shard_id}: {len(shard_dict)} tensors")
        
        return shard_dict
    
    async def _download_shard_from_storage(self, remote_path: str, local_path: str):
        """Download shard from storage backend"""
        # This is a placeholder - actual implementation depends on storage backend
        await self.storage.download_file(remote_path, local_path)
    
    def verify_model_integrity(self, manifest_path: str) -> bool:
        """Verify integrity of all shards in a model"""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        logger.info(f"Verifying integrity of model {manifest['model_id']}")
        
        all_valid = True
        for shard_info in manifest['shards']:
            shard = ShardInfo(**shard_info)
            
            # Check if shard exists in storage
            remote_path = f"{manifest['model_id']}/{shard.shard_name}"
            if not self.storage.exists(remote_path):
                logger.error(f"Shard {shard.shard_id} not found in storage")
                all_valid = False
                continue
            
            # Verify size
            actual_size = self.storage.get_file_size(remote_path)
            if actual_size != shard.size_bytes:
                logger.error(f"Size mismatch for shard {shard.shard_id}: "
                           f"expected {shard.size_bytes}, got {actual_size}")
                all_valid = False
        
        return all_valid
    
    async def stream_load_shard(self, shard_info: ShardInfo, model_id: str,
                               callback=None) -> Dict[str, torch.Tensor]:
        """Stream load a shard with progress callback"""
        remote_path = f"{model_id}/{shard_info.shard_name}"
        
        # Create a temporary file for streaming
        temp_path = f"/tmp/{shard_info.shard_name}.partial"
        
        # Stream download with progress
        bytes_downloaded = 0
        chunk_size = self.config.chunk_size_mb * 1024 * 1024
        
        with open(temp_path, 'wb') as f:
            async for chunk in self.storage.stream_download(remote_path, chunk_size):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                
                if callback:
                    progress = bytes_downloaded / shard_info.size_bytes
                    callback(shard_info.shard_id, progress)
        
        # Verify checksum
        with open(temp_path, 'rb') as f:
            checksum = self._calculate_checksum(f.read())
        
        if checksum != shard_info.checksum:
            os.remove(temp_path)
            raise ValueError(f"Checksum mismatch for shard {shard_info.shard_id}")
        
        # Load the shard
        shard_dict = load_file(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return shard_dict
    
    def get_model_info(self, manifest_path: str) -> Dict[str, Any]:
        """Get information about a sharded model"""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        total_tensors = sum(len(shard['tensor_keys']) for shard in manifest['shards'])
        
        return {
            "model_id": manifest['model_id'],
            "total_size_gb": manifest['total_size_bytes'] / 1e9,
            "num_shards": manifest['num_shards'],
            "total_tensors": total_tensors,
            "average_shard_size_gb": manifest['total_size_bytes'] / manifest['num_shards'] / 1e9,
            "created_at": manifest['metadata']['created_at'],
            "format": manifest['metadata']['format']
        }


class AdaptiveShardManager(ModelShardManager):
    """Advanced shard manager with adaptive sharding strategies"""
    
    def __init__(self, storage_backend: Any, config: Optional[ModelShardConfig] = None):
        super().__init__(storage_backend, config)
        self.shard_strategies = {
            "layer_based": self._shard_by_layers,
            "size_balanced": self._shard_by_size,
            "attention_aware": self._shard_attention_aware
        }
    
    def _shard_by_layers(self, state_dict: Dict[str, torch.Tensor], 
                        num_shards: int) -> List[Dict[str, torch.Tensor]]:
        """Shard model by keeping layers together"""
        # Group tensors by layer
        layers = {}
        for key, tensor in state_dict.items():
            # Extract layer identifier (e.g., "layer.0", "layer.1")
            layer_match = re.match(r'.*\.(layer\.\d+)\..*', key)
            if layer_match:
                layer_name = layer_match.group(1)
            else:
                layer_name = "other"
            
            if layer_name not in layers:
                layers[layer_name] = {}
            layers[layer_name][key] = tensor
        
        # Distribute layers across shards
        shards = [{"tensors": {}, "size": 0} for _ in range(num_shards)]
        
        for layer_name, layer_tensors in sorted(layers.items()):
            layer_size = sum(self._get_tensor_size(t) for t in layer_tensors.values())
            
            # Find shard with minimum size
            min_shard_idx = min(range(num_shards), key=lambda i: shards[i]["size"])
            
            # Add entire layer to shard
            shards[min_shard_idx]["tensors"].update(layer_tensors)
            shards[min_shard_idx]["size"] += layer_size
        
        return [shard["tensors"] for shard in shards]
    
    def _shard_attention_aware(self, state_dict: Dict[str, torch.Tensor],
                              num_shards: int) -> List[Dict[str, torch.Tensor]]:
        """Shard model keeping attention mechanisms together"""
        # Group attention-related tensors
        attention_groups = {}
        other_tensors = {}
        
        for key, tensor in state_dict.items():
            if any(pattern in key for pattern in ['attention', 'attn', 'query', 'key', 'value']):
                # Extract attention block identifier
                block_match = re.match(r'.*\.(\d+)\..*', key)
                if block_match:
                    block_id = int(block_match.group(1))
                    if block_id not in attention_groups:
                        attention_groups[block_id] = {}
                    attention_groups[block_id][key] = tensor
                else:
                    other_tensors[key] = tensor
            else:
                other_tensors[key] = tensor
        
        # Create shards keeping attention blocks together
        shards = [{"tensors": {}, "size": 0} for _ in range(num_shards)]
        
        # First distribute attention blocks
        for block_id in sorted(attention_groups.keys()):
            block_tensors = attention_groups[block_id]
            block_size = sum(self._get_tensor_size(t) for t in block_tensors.values())
            
            min_shard_idx = min(range(num_shards), key=lambda i: shards[i]["size"])
            shards[min_shard_idx]["tensors"].update(block_tensors)
            shards[min_shard_idx]["size"] += block_size
        
        # Then distribute other tensors
        for key, tensor in other_tensors.items():
            tensor_size = self._get_tensor_size(tensor)
            min_shard_idx = min(range(num_shards), key=lambda i: shards[i]["size"])
            shards[min_shard_idx]["tensors"][key] = tensor
            shards[min_shard_idx]["size"] += tensor_size
        
        return [shard["tensors"] for shard in shards]
    
    async def shard_model_adaptive(self, model_path: str, output_dir: str,
                                  model_id: str, strategy: str = "layer_based") -> Dict[str, Any]:
        """Shard model using adaptive strategy"""
        if strategy not in self.shard_strategies:
            raise ValueError(f"Unknown sharding strategy: {strategy}")
        
        logger.info(f"Using {strategy} sharding strategy")
        
        # Load model
        if model_path.endswith('.safetensors'):
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
        # Calculate sharding parameters
        total_size = sum(self._get_tensor_size(tensor) for tensor in state_dict.values())
        num_shards, _ = self.calculate_shard_size(total_size)
        
        # Apply sharding strategy
        shards = self.shard_strategies[strategy](state_dict, num_shards)
        
        # Save shards
        shard_infos = []
        for shard_id, shard_data in enumerate(shards):
            shard_info = await self._save_shard(
                shard_id=shard_id,
                shard_data=shard_data,
                output_dir=output_dir,
                model_id=model_id
            )
            shard_infos.append(shard_info)
        
        # Create manifest with strategy info
        manifest = {
            "model_id": model_id,
            "total_size_bytes": total_size,
            "num_shards": num_shards,
            "sharding_strategy": strategy,
            "shards": [info.__dict__ for info in shard_infos],
            "metadata": {
                "model_path": model_path,
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
        manifest_path = os.path.join(output_dir, f"{model_id}_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest


# Placeholder for storage backend interface
class StorageBackend:
    """Abstract storage backend interface"""
    
    async def upload_file(self, local_path: str, remote_path: str):
        raise NotImplementedError
    
    async def download_file(self, remote_path: str, local_path: str):
        raise NotImplementedError
    
    async def stream_download(self, remote_path: str, chunk_size: int):
        raise NotImplementedError
    
    def exists(self, remote_path: str) -> bool:
        raise NotImplementedError
    
    def get_file_size(self, remote_path: str) -> int:
        raise NotImplementedError


# Example usage
if __name__ == "__main__":
    import re
    from datetime import datetime
    
    async def main():
        # Initialize storage backend (placeholder)
        storage = StorageBackend()
        
        # Create shard manager
        config = ModelShardConfig(
            max_shard_size_gb=2.0,
            compression=True,
            parallel_uploads=4
        )
        
        manager = AdaptiveShardManager(storage, config)
        
        # Shard a model
        manifest = await manager.shard_model_adaptive(
            model_path="/path/to/large_model.safetensors",
            output_dir="/path/to/shards",
            model_id="llama-13b-v1",
            strategy="layer_based"
        )
        
        print(f"Model sharded into {manifest['num_shards']} shards")
        
        # Load sharded model
        state_dict = await manager.load_sharded_model(
            manifest_path="/path/to/shards/llama-13b-v1_manifest.json",
            device="cuda:0"
        )
        
        print(f"Loaded model with {len(state_dict)} tensors")
    
    # Run example
    # asyncio.run(main())