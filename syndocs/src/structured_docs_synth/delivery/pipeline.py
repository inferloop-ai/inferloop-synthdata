"""
Delivery pipeline orchestrator for structured document synthesis.

Coordinates the entire delivery process from generation to final output.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..core.exceptions import DeliveryError, ConfigurationError
from .export.format_exporters import get_exporter
from .storage.database_storage import DatabaseStorage
from .storage.cloud_storage import CloudStorage
from .storage.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class DeliveryPipeline:
    """
    Main delivery pipeline that orchestrates document delivery process.
    
    Handles the complete workflow from document generation results to
    final delivery through various channels (API, export, storage).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize delivery pipeline.
        
        Args:
            config: Configuration dictionary for pipeline components
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage backends
        self.database = DatabaseStorage(self.config.get('database', {}))
        self.cloud_storage = CloudStorage(self.config.get('cloud_storage', {}))
        self.cache = CacheManager(self.config.get('cache', {}))
        
        # Pipeline state
        self.is_running = False
        self.processed_count = 0
        self.error_count = 0
        
    async def deliver_document(self, 
                             document_data: Dict[str, Any],
                             delivery_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deliver a single document through the pipeline.
        
        Args:
            document_data: Generated document data
            delivery_config: Delivery configuration (formats, targets, etc.)
            
        Returns:
            Delivery results dictionary
        """
        delivery_id = f"delivery_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            self.logger.info(f"Starting delivery: {delivery_id}")
            
            delivery_result = {
                'delivery_id': delivery_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'in_progress',
                'document_id': document_data.get('id'),
                'formats_generated': [],
                'storage_locations': [],
                'errors': []
            }
            
            # Step 1: Store document metadata in database
            await self._store_document_metadata(document_data, delivery_result)
            
            # Step 2: Generate requested formats
            formats = delivery_config.get('formats', ['pdf'])
            for format_type in formats:
                try:
                    format_result = await self._generate_format(
                        document_data, format_type, delivery_config
                    )
                    delivery_result['formats_generated'].append(format_result)
                    
                except Exception as e:
                    error_msg = f"Failed to generate {format_type}: {str(e)}"
                    self.logger.error(error_msg)
                    delivery_result['errors'].append(error_msg)
            
            # Step 3: Store outputs in configured storage locations
            storage_targets = delivery_config.get('storage_targets', ['local'])
            for target in storage_targets:
                try:
                    storage_result = await self._store_outputs(
                        delivery_result['formats_generated'], target, delivery_config
                    )
                    delivery_result['storage_locations'].append(storage_result)
                    
                except Exception as e:
                    error_msg = f"Failed to store to {target}: {str(e)}"
                    self.logger.error(error_msg)
                    delivery_result['errors'].append(error_msg)
            
            # Step 4: Cache results if enabled
            if delivery_config.get('enable_caching', True):
                await self._cache_delivery_result(delivery_result)
            
            # Update final status
            delivery_result['status'] = 'completed' if not delivery_result['errors'] else 'completed_with_errors'
            
            self.processed_count += 1
            if delivery_result['errors']:
                self.error_count += 1
                
            self.logger.info(f"Delivery completed: {delivery_id}")
            return delivery_result
            
        except Exception as e:
            self.error_count += 1
            error_msg = f"Delivery pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'delivery_id': delivery_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': error_msg,
                'document_id': document_data.get('id')
            }
    
    async def deliver_batch(self,
                          documents: List[Dict[str, Any]],
                          delivery_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deliver multiple documents in batch.
        
        Args:
            documents: List of document data dictionaries
            delivery_config: Batch delivery configuration
            
        Returns:
            Batch delivery results
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"Starting batch delivery: {batch_id} ({len(documents)} documents)")
            
            batch_result = {
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'total_documents': len(documents),
                'successful_deliveries': 0,
                'failed_deliveries': 0,
                'delivery_results': []
            }
            
            # Process documents concurrently or sequentially based on config
            max_concurrent = delivery_config.get('max_concurrent_deliveries', 5)
            
            if max_concurrent > 1:
                # Concurrent processing
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def process_document(doc):
                    async with semaphore:
                        return await self.deliver_document(doc, delivery_config)
                
                tasks = [process_document(doc) for doc in documents]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        batch_result['failed_deliveries'] += 1
                        batch_result['delivery_results'].append({
                            'status': 'failed',
                            'error': str(result)
                        })
                    else:
                        if result['status'] == 'failed':
                            batch_result['failed_deliveries'] += 1
                        else:
                            batch_result['successful_deliveries'] += 1
                        batch_result['delivery_results'].append(result)
            else:
                # Sequential processing
                for doc in documents:
                    result = await self.deliver_document(doc, delivery_config)
                    batch_result['delivery_results'].append(result)
                    
                    if result['status'] == 'failed':
                        batch_result['failed_deliveries'] += 1
                    else:
                        batch_result['successful_deliveries'] += 1
            
            self.logger.info(f"Batch delivery completed: {batch_id}")
            return batch_result
            
        except Exception as e:
            error_msg = f"Batch delivery failed: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': error_msg,
                'total_documents': len(documents),
                'successful_deliveries': 0,
                'failed_deliveries': len(documents)
            }
    
    async def _store_document_metadata(self, document_data: Dict[str, Any], 
                                     delivery_result: Dict[str, Any]):
        """Store document metadata in database"""
        try:
            metadata = {
                'document_id': document_data.get('id'),
                'delivery_id': delivery_result['delivery_id'],
                'generation_config': document_data.get('config', {}),
                'created_at': datetime.now().isoformat(),
                'status': 'delivered'
            }
            
            await self.database.store_document_metadata(metadata)
            
        except Exception as e:
            self.logger.warning(f"Failed to store document metadata: {e}")
    
    async def _generate_format(self, document_data: Dict[str, Any], 
                             format_type: str, delivery_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate document in specified format"""
        try:
            exporter = get_exporter(format_type)
            
            format_config = delivery_config.get('format_configs', {}).get(format_type, {})
            
            output_result = await exporter.export(document_data, format_config)
            
            return {
                'format': format_type,
                'status': 'success',
                'output_path': output_result.get('output_path'),
                'file_size': output_result.get('file_size'),
                'generation_time': output_result.get('generation_time')
            }
            
        except Exception as e:
            raise DeliveryError(f"Format generation failed for {format_type}: {str(e)}")
    
    async def _store_outputs(self, format_results: List[Dict[str, Any]], 
                           storage_target: str, delivery_config: Dict[str, Any]) -> Dict[str, Any]:
        """Store generated outputs to storage target"""
        try:
            storage_config = delivery_config.get('storage_configs', {}).get(storage_target, {})
            
            if storage_target == 'cloud':
                storage_result = await self.cloud_storage.store_files(
                    [result['output_path'] for result in format_results if result.get('output_path')],
                    storage_config
                )
            elif storage_target == 'database':
                storage_result = await self.database.store_files(
                    [result['output_path'] for result in format_results if result.get('output_path')],
                    storage_config
                )
            else:  # local storage
                storage_result = {
                    'storage_type': 'local',
                    'status': 'success',
                    'files_stored': len([r for r in format_results if r.get('output_path')])
                }
            
            return {
                'storage_target': storage_target,
                'status': 'success',
                **storage_result
            }
            
        except Exception as e:
            raise DeliveryError(f"Storage failed for {storage_target}: {str(e)}")
    
    async def _cache_delivery_result(self, delivery_result: Dict[str, Any]):
        """Cache delivery result for future retrieval"""
        try:
            cache_key = f"delivery:{delivery_result['delivery_id']}"
            await self.cache.set(cache_key, delivery_result, ttl=3600)  # 1 hour TTL
            
        except Exception as e:
            self.logger.warning(f"Failed to cache delivery result: {e}")
    
    async def get_delivery_status(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific delivery.
        
        Args:
            delivery_id: Delivery identifier
            
        Returns:
            Delivery status information or None if not found
        """
        try:
            # Try cache first
            cache_key = f"delivery:{delivery_id}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                return cached_result
            
            # Fallback to database
            return await self.database.get_delivery_status(delivery_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get delivery status: {e}")
            return None
    
    async def list_deliveries(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List recent deliveries.
        
        Args:
            limit: Maximum number of deliveries to return
            offset: Number of deliveries to skip
            
        Returns:
            List of delivery information
        """
        try:
            return await self.database.list_deliveries(limit, offset)
            
        except Exception as e:
            self.logger.error(f"Failed to list deliveries: {e}")
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Pipeline statistics dictionary
        """
        return {
            'is_running': self.is_running,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'success_rate': (self.processed_count - self.error_count) / max(self.processed_count, 1) * 100
        }