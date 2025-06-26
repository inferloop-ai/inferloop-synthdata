"""
Delta Update Manager for TextNLP
Handles efficient delta updates for model weights to minimize transfer and storage
"""

import os
import json
import hashlib
import tempfile
import asyncio
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import aiofiles
import logging
from datetime import datetime
import zstandard as zstd
import msgpack
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


@dataclass
class DeltaPatch:
    """Represents a delta patch between model versions"""
    patch_id: str
    source_version: str
    target_version: str
    patch_type: str  # "weight_diff", "sparse_update", "low_rank"
    size_bytes: int
    compression_ratio: float
    affected_layers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass 
class WeightDelta:
    """Delta information for a single weight tensor"""
    key: str
    shape: Tuple[int, ...]
    dtype: str
    update_type: str  # "full", "sparse", "low_rank", "unchanged"
    data: Optional[Any] = None  # Actual delta data
    indices: Optional[np.ndarray] = None  # For sparse updates
    rank: Optional[int] = None  # For low-rank updates


class DeltaUpdateManager:
    """Manages delta updates between model versions"""
    
    def __init__(self, storage_backend: Any, compression_level: int = 3):
        self.storage = storage_backend
        self.compression_level = compression_level
        self.compressor = zstd.ZstdCompressor(level=compression_level)
        self.decompressor = zstd.ZstdDecompressor()
        
    def _calculate_tensor_hash(self, tensor: torch.Tensor) -> str:
        """Calculate hash of a tensor"""
        tensor_bytes = tensor.cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()[:16]
    
    def _compute_weight_diff(self, old_weight: torch.Tensor, 
                           new_weight: torch.Tensor) -> Tuple[str, Any]:
        """Compute difference between two weight tensors"""
        # Check if weights are identical
        if torch.equal(old_weight, new_weight):
            return "unchanged", None
        
        # Calculate absolute difference
        diff = new_weight - old_weight
        
        # Check sparsity of difference
        diff_abs = torch.abs(diff)
        threshold = torch.max(diff_abs) * 0.01  # 1% of max change
        sparse_mask = diff_abs > threshold
        sparsity = 1.0 - (torch.sum(sparse_mask).item() / diff.numel())
        
        # Decide update strategy based on sparsity
        if sparsity > 0.9:  # More than 90% sparse
            # Use sparse representation
            indices = torch.nonzero(sparse_mask).cpu().numpy()
            values = diff[sparse_mask].cpu().numpy()
            return "sparse", {"indices": indices, "values": values}
        
        elif old_weight.ndim >= 2 and min(old_weight.shape) > 10:
            # Try low-rank approximation for large matrices
            try:
                U, S, V = torch.svd(diff)
                
                # Determine rank for 99% energy retention
                cumsum_S = torch.cumsum(S**2, dim=0)
                total_energy = cumsum_S[-1]
                rank = torch.searchsorted(cumsum_S, 0.99 * total_energy).item() + 1
                
                # Check if low-rank is efficient
                original_size = diff.numel()
                low_rank_size = rank * (U.shape[0] + V.shape[0])
                
                if low_rank_size < original_size * 0.5:  # 50% size reduction
                    return "low_rank", {
                        "U": U[:, :rank].cpu().numpy(),
                        "S": S[:rank].cpu().numpy(),
                        "V": V[:, :rank].cpu().numpy(),
                        "rank": rank
                    }
            except:
                pass  # Fall back to full diff
        
        # Default to full difference
        return "full", diff.cpu().numpy()
    
    async def create_delta_patch(self, old_model_path: str, new_model_path: str,
                               source_version: str, target_version: str) -> DeltaPatch:
        """Create a delta patch between two model versions"""
        logger.info(f"Creating delta patch from {source_version} to {target_version}")
        
        # Load models
        old_state_dict = load_file(old_model_path) if old_model_path.endswith('.safetensors') \
                        else torch.load(old_model_path, map_location='cpu')
        new_state_dict = load_file(new_model_path) if new_model_path.endswith('.safetensors') \
                        else torch.load(new_model_path, map_location='cpu')
        
        # Compute deltas
        weight_deltas = {}
        affected_layers = []
        total_delta_size = 0
        
        # Process each weight
        all_keys = set(old_state_dict.keys()) | set(new_state_dict.keys())
        
        for key in all_keys:
            if key not in old_state_dict:
                # New weight added
                weight_deltas[key] = WeightDelta(
                    key=key,
                    shape=new_state_dict[key].shape,
                    dtype=str(new_state_dict[key].dtype),
                    update_type="full",
                    data=new_state_dict[key].cpu().numpy()
                )
                affected_layers.append(key)
                total_delta_size += new_state_dict[key].numel() * new_state_dict[key].element_size()
                
            elif key not in new_state_dict:
                # Weight removed
                weight_deltas[key] = WeightDelta(
                    key=key,
                    shape=old_state_dict[key].shape,
                    dtype=str(old_state_dict[key].dtype),
                    update_type="removed",
                    data=None
                )
                affected_layers.append(key)
                
            else:
                # Weight potentially changed
                old_weight = old_state_dict[key]
                new_weight = new_state_dict[key]
                
                if old_weight.shape != new_weight.shape:
                    # Shape changed, need full update
                    weight_deltas[key] = WeightDelta(
                        key=key,
                        shape=new_weight.shape,
                        dtype=str(new_weight.dtype),
                        update_type="full",
                        data=new_weight.cpu().numpy()
                    )
                    affected_layers.append(key)
                    total_delta_size += new_weight.numel() * new_weight.element_size()
                else:
                    # Compute difference
                    update_type, delta_data = self._compute_weight_diff(old_weight, new_weight)
                    
                    if update_type != "unchanged":
                        weight_deltas[key] = WeightDelta(
                            key=key,
                            shape=new_weight.shape,
                            dtype=str(new_weight.dtype),
                            update_type=update_type,
                            data=delta_data
                        )
                        affected_layers.append(key)
                        
                        # Estimate delta size
                        if update_type == "sparse":
                            total_delta_size += len(delta_data["values"]) * 4 + \
                                              len(delta_data["indices"]) * 4
                        elif update_type == "low_rank":
                            total_delta_size += sum(arr.nbytes for arr in delta_data.values() 
                                                  if isinstance(arr, np.ndarray))
                        else:
                            total_delta_size += delta_data.nbytes
        
        # Determine patch type
        if not affected_layers:
            patch_type = "no_change"
        elif len([d for d in weight_deltas.values() if d.update_type == "sparse"]) > len(weight_deltas) / 2:
            patch_type = "sparse_update"
        elif len([d for d in weight_deltas.values() if d.update_type == "low_rank"]) > len(weight_deltas) / 2:
            patch_type = "low_rank"
        else:
            patch_type = "weight_diff"
        
        # Create patch
        patch_id = f"delta_{source_version}_to_{target_version}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Serialize and compress patch data
        patch_data = {
            "patch_id": patch_id,
            "source_version": source_version,
            "target_version": target_version,
            "weight_deltas": {}
        }
        
        for key, delta in weight_deltas.items():
            patch_data["weight_deltas"][key] = {
                "shape": delta.shape,
                "dtype": delta.dtype,
                "update_type": delta.update_type,
                "data": delta.data
            }
        
        # Serialize with msgpack for efficiency
        serialized = msgpack.packb(patch_data, use_bin_type=True)
        compressed = self.compressor.compress(serialized)
        
        # Calculate compression ratio
        original_size = os.path.getsize(new_model_path)
        compression_ratio = len(compressed) / original_size
        
        # Save patch
        patch_path = f"patches/{patch_id}.delta"
        await self._save_patch(compressed, patch_path)
        
        # Create patch metadata
        patch = DeltaPatch(
            patch_id=patch_id,
            source_version=source_version,
            target_version=target_version,
            patch_type=patch_type,
            size_bytes=len(compressed),
            compression_ratio=compression_ratio,
            affected_layers=affected_layers,
            metadata={
                "num_affected_weights": len(affected_layers),
                "num_sparse_updates": len([d for d in weight_deltas.values() 
                                         if d.update_type == "sparse"]),
                "num_low_rank_updates": len([d for d in weight_deltas.values() 
                                           if d.update_type == "low_rank"]),
                "original_model_size": original_size
            },
            checksum=hashlib.sha256(compressed).hexdigest()
        )
        
        logger.info(f"Created delta patch {patch_id}: {len(compressed)/1024/1024:.2f}MB "
                   f"({compression_ratio*100:.1f}% of original)")
        
        return patch
    
    async def _save_patch(self, patch_data: bytes, patch_path: str):
        """Save patch to storage"""
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(patch_data)
            tmp_path = tmp_file.name
        
        # Upload to storage
        await self.storage.upload_file(tmp_path, patch_path)
        
        # Clean up
        os.unlink(tmp_path)
    
    async def apply_delta_patch(self, base_model_path: str, patch_path: str,
                              output_path: str) -> Dict[str, Any]:
        """Apply a delta patch to create updated model"""
        logger.info(f"Applying delta patch to {base_model_path}")
        
        # Load base model
        state_dict = load_file(base_model_path) if base_model_path.endswith('.safetensors') \
                    else torch.load(base_model_path, map_location='cpu')
        
        # Download and load patch
        local_patch_path = f"/tmp/{os.path.basename(patch_path)}"
        await self.storage.download_file(patch_path, local_patch_path)
        
        with open(local_patch_path, 'rb') as f:
            compressed_data = f.read()
        
        # Decompress and deserialize
        serialized = self.decompressor.decompress(compressed_data)
        patch_data = msgpack.unpackb(serialized, raw=False)
        
        # Apply deltas
        updated_weights = 0
        for key, delta_info in patch_data["weight_deltas"].items():
            update_type = delta_info["update_type"]
            
            if update_type == "removed":
                # Remove weight
                if key in state_dict:
                    del state_dict[key]
                    
            elif update_type == "unchanged":
                # No change needed
                continue
                
            elif update_type == "full":
                # Full replacement
                state_dict[key] = torch.from_numpy(delta_info["data"])
                updated_weights += 1
                
            elif update_type == "sparse":
                # Apply sparse update
                if key in state_dict:
                    indices = delta_info["data"]["indices"]
                    values = delta_info["data"]["values"]
                    
                    # Convert flat indices to multi-dimensional
                    weight = state_dict[key]
                    for idx, val in zip(indices, values):
                        weight.view(-1)[idx] += val
                    updated_weights += 1
                    
            elif update_type == "low_rank":
                # Apply low-rank update
                if key in state_dict:
                    U = torch.from_numpy(delta_info["data"]["U"])
                    S = torch.from_numpy(delta_info["data"]["S"])
                    V = torch.from_numpy(delta_info["data"]["V"])
                    
                    # Reconstruct delta
                    delta = torch.mm(torch.mm(U, torch.diag(S)), V.t())
                    
                    # Reshape if necessary
                    if len(delta_info["shape"]) > 2:
                        delta = delta.reshape(delta_info["shape"])
                    
                    state_dict[key] += delta
                    updated_weights += 1
        
        # Save updated model
        if output_path.endswith('.safetensors'):
            save_file(state_dict, output_path)
        else:
            torch.save(state_dict, output_path)
        
        # Clean up
        os.unlink(local_patch_path)
        
        logger.info(f"Applied patch successfully. Updated {updated_weights} weights")
        
        return {
            "updated_weights": updated_weights,
            "output_path": output_path,
            "patch_id": patch_data["patch_id"]
        }
    
    async def create_patch_chain(self, model_versions: List[Tuple[str, str]]) -> List[DeltaPatch]:
        """Create a chain of patches between consecutive model versions"""
        patches = []
        
        for i in range(len(model_versions) - 1):
            old_path, old_version = model_versions[i]
            new_path, new_version = model_versions[i + 1]
            
            patch = await self.create_delta_patch(
                old_path, new_path, old_version, new_version
            )
            patches.append(patch)
        
        return patches
    
    async def apply_patch_chain(self, base_model_path: str, 
                              patches: List[DeltaPatch],
                              target_version: str) -> str:
        """Apply a chain of patches to reach target version"""
        current_path = base_model_path
        temp_paths = []
        
        try:
            for i, patch in enumerate(patches):
                # Create temporary output path
                if i < len(patches) - 1:
                    output_path = f"/tmp/model_temp_{i}.safetensors"
                    temp_paths.append(output_path)
                else:
                    output_path = f"/tmp/model_{target_version}.safetensors"
                
                # Apply patch
                patch_path = f"patches/{patch.patch_id}.delta"
                await self.apply_delta_patch(current_path, patch_path, output_path)
                
                current_path = output_path
            
            return current_path
            
        finally:
            # Clean up temporary files
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def analyze_model_changes(self, old_state_dict: Dict[str, torch.Tensor],
                            new_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze changes between two model versions"""
        analysis = {
            "total_weights": len(new_state_dict),
            "unchanged_weights": 0,
            "modified_weights": 0,
            "added_weights": 0,
            "removed_weights": 0,
            "total_parameters": sum(w.numel() for w in new_state_dict.values()),
            "parameter_change": 0,
            "layer_changes": {},
            "change_magnitude": {}
        }
        
        old_keys = set(old_state_dict.keys())
        new_keys = set(new_state_dict.keys())
        
        # Added and removed
        analysis["added_weights"] = len(new_keys - old_keys)
        analysis["removed_weights"] = len(old_keys - new_keys)
        
        # Check modifications
        for key in old_keys & new_keys:
            old_weight = old_state_dict[key]
            new_weight = new_state_dict[key]
            
            if torch.equal(old_weight, new_weight):
                analysis["unchanged_weights"] += 1
            else:
                analysis["modified_weights"] += 1
                
                # Calculate change magnitude
                diff = torch.abs(new_weight - old_weight)
                relative_change = torch.mean(diff / (torch.abs(old_weight) + 1e-8))
                
                layer_name = key.split('.')[0]
                if layer_name not in analysis["layer_changes"]:
                    analysis["layer_changes"][layer_name] = []
                
                analysis["layer_changes"][layer_name].append({
                    "weight": key,
                    "relative_change": relative_change.item(),
                    "max_change": torch.max(diff).item()
                })
        
        # Parameter change
        old_params = sum(w.numel() for w in old_state_dict.values())
        new_params = sum(w.numel() for w in new_state_dict.values())
        analysis["parameter_change"] = new_params - old_params
        
        return analysis


class IncrementalTrainingManager:
    """Manages incremental training with delta updates"""
    
    def __init__(self, delta_manager: DeltaUpdateManager):
        self.delta_manager = delta_manager
    
    async def checkpoint_training(self, model: torch.nn.Module, 
                                checkpoint_path: str,
                                base_version: str,
                                checkpoint_id: str) -> DeltaPatch:
        """Create a delta checkpoint during training"""
        # Save current model state
        temp_path = f"/tmp/checkpoint_{checkpoint_id}.safetensors"
        save_file(model.state_dict(), temp_path)
        
        # Create delta from base version
        patch = await self.delta_manager.create_delta_patch(
            base_version,
            temp_path,
            base_version,
            f"checkpoint_{checkpoint_id}"
        )
        
        # Clean up
        os.unlink(temp_path)
        
        return patch
    
    async def merge_checkpoints(self, base_model_path: str,
                              checkpoint_patches: List[DeltaPatch],
                              output_path: str) -> Dict[str, Any]:
        """Merge multiple checkpoint patches into final model"""
        # Apply patches sequentially
        current_path = base_model_path
        
        for patch in checkpoint_patches:
            patch_path = f"patches/{patch.patch_id}.delta"
            await self.delta_manager.apply_delta_patch(
                current_path,
                patch_path,
                output_path
            )
            current_path = output_path
        
        return {
            "merged_checkpoints": len(checkpoint_patches),
            "output_path": output_path
        }


# Example usage
if __name__ == "__main__":
    async def example():
        # Dummy storage backend
        class DummyStorage:
            async def upload_file(self, local_path: str, remote_path: str):
                print(f"Uploading {local_path} to {remote_path}")
                
            async def download_file(self, remote_path: str, local_path: str):
                print(f"Downloading {remote_path} to {local_path}")
                # Create dummy file
                with open(local_path, 'wb') as f:
                    f.write(b"dummy patch data")
        
        storage = DummyStorage()
        
        # Create delta manager
        delta_manager = DeltaUpdateManager(storage)
        
        # Example: Create delta between two model versions
        # patch = await delta_manager.create_delta_patch(
        #     "/path/to/model_v1.safetensors",
        #     "/path/to/model_v2.safetensors",
        #     "1.0.0",
        #     "1.1.0"
        # )
        # print(f"Created patch: {patch.patch_id}, size: {patch.size_bytes/1024/1024:.2f}MB")
        
        # Example: Apply patch
        # result = await delta_manager.apply_delta_patch(
        #     "/path/to/model_v1.safetensors",
        #     "patches/delta_1.0.0_to_1.1.0.delta",
        #     "/path/to/model_v1.1.0_reconstructed.safetensors"
        # )
        # print(f"Applied patch: {result}")
        
        print("Delta update manager initialized")
    
    # asyncio.run(example())