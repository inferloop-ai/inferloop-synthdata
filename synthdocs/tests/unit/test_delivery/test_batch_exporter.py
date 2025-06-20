"""
Unit tests for batch export functionality.

Tests the BatchExporter class for exporting large volumes of synthetic documents
in various formats with progress tracking and error handling.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from structured_docs_synth.delivery.export.batch_exporter import (
    BatchExporter,
    BatchExportConfig,
    ExportJob,
    ExportStatus
)
from structured_docs_synth.core.exceptions import DeliveryError


class TestBatchExportConfig:
    """Test BatchExportConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BatchExportConfig()
        
        assert config.batch_size == 100
        assert config.concurrent_exports == 5
        assert config.output_formats == ['pdf', 'json']
        assert config.compression_enabled is True
        assert config.compression_format == 'zip'
        assert config.include_metadata is True
        assert config.progress_callback is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        def progress_callback(current, total):
            pass
        
        config = BatchExportConfig(
            batch_size=50,
            output_formats=['docx', 'xml'],
            compression_enabled=False,
            progress_callback=progress_callback
        )
        
        assert config.batch_size == 50
        assert config.output_formats == ['docx', 'xml']
        assert config.compression_enabled is False
        assert config.progress_callback == progress_callback


class TestExportJob:
    """Test ExportJob class."""
    
    def test_job_creation(self):
        """Test export job creation."""
        job = ExportJob(
            job_id="test-123",
            total_documents=100,
            output_formats=['pdf', 'json']
        )
        
        assert job.job_id == "test-123"
        assert job.total_documents == 100
        assert job.output_formats == ['pdf', 'json']
        assert job.status == ExportStatus.PENDING
        assert job.processed_documents == 0
        assert job.failed_documents == 0
        assert job.start_time is None
        assert job.end_time is None
    
    def test_job_progress_update(self):
        """Test job progress updates."""
        job = ExportJob("test-123", 100, ['pdf'])
        
        job.start()
        assert job.status == ExportStatus.RUNNING
        assert job.start_time is not None
        
        job.update_progress(10, 2)
        assert job.processed_documents == 10
        assert job.failed_documents == 2
        
        job.complete()
        assert job.status == ExportStatus.COMPLETED
        assert job.end_time is not None
    
    def test_job_failure(self):
        """Test job failure handling."""
        job = ExportJob("test-123", 100, ['pdf'])
        
        job.start()
        job.fail("Export failed due to storage error")
        
        assert job.status == ExportStatus.FAILED
        assert job.error_message == "Export failed due to storage error"
        assert job.end_time is not None
    
    def test_job_statistics(self):
        """Test job statistics calculation."""
        job = ExportJob("test-123", 100, ['pdf'])
        
        job.start()
        job.processed_documents = 80
        job.failed_documents = 5
        job.complete()
        
        stats = job.get_statistics()
        
        assert stats['total_documents'] == 100
        assert stats['processed_documents'] == 80
        assert stats['failed_documents'] == 5
        assert stats['success_rate'] == 0.8
        assert stats['status'] == ExportStatus.COMPLETED
        assert 'duration' in stats


class TestBatchExporter:
    """Test BatchExporter class."""
    
    @pytest.fixture
    def exporter_config(self):
        """Provide exporter configuration."""
        return BatchExportConfig(
            batch_size=10,
            concurrent_exports=2,
            output_formats=['pdf', 'json']
        )
    
    @pytest.fixture
    def mock_format_exporter(self):
        """Provide mock format exporter."""
        mock = AsyncMock()
        mock.export = AsyncMock(return_value={
            'success': True,
            'output_path': '/tmp/exported_doc.pdf',
            'file_size': 1024
        })
        return mock
    
    @pytest.fixture
    def mock_storage(self):
        """Provide mock storage."""
        mock = AsyncMock()
        mock.store_files = AsyncMock(return_value={
            'successful_uploads': 10,
            'failed_uploads': 0
        })
        return mock
    
    @pytest.fixture
    def batch_exporter(self, exporter_config, mock_format_exporter, mock_storage):
        """Provide batch exporter instance."""
        exporter = BatchExporter(exporter_config)
        exporter.format_exporters = {'pdf': mock_format_exporter, 'json': mock_format_exporter}
        exporter.storage = mock_storage
        return exporter
    
    @pytest.mark.asyncio
    async def test_create_export_job(self, batch_exporter):
        """Test export job creation."""
        documents = [{'id': i, 'content': f'Document {i}'} for i in range(20)]
        
        job = await batch_exporter.create_export_job(
            documents=documents,
            output_dir=Path('/tmp/exports'),
            job_metadata={'project': 'test'}
        )
        
        assert job.total_documents == 20
        assert job.output_formats == ['pdf', 'json']
        assert job.status == ExportStatus.PENDING
        assert len(batch_exporter.active_jobs) == 1
        assert batch_exporter.active_jobs[job.job_id] == job
    
    @pytest.mark.asyncio
    async def test_export_batch_success(self, batch_exporter, temp_dir):
        """Test successful batch export."""
        documents = [{'id': i, 'content': f'Document {i}'} for i in range(5)]
        
        result = await batch_exporter._export_batch(
            documents=documents,
            output_dir=temp_dir,
            formats=['pdf', 'json']
        )
        
        assert result['success'] is True
        assert result['exported_count'] == 5
        assert result['failed_count'] == 0
        assert len(result['exported_files']) == 10  # 5 docs * 2 formats
    
    @pytest.mark.asyncio
    async def test_export_batch_with_failures(self, batch_exporter, temp_dir):
        """Test batch export with some failures."""
        documents = [{'id': i, 'content': f'Document {i}'} for i in range(5)]
        
        # Make some exports fail
        batch_exporter.format_exporters['pdf'].export = AsyncMock(side_effect=[
            {'success': True, 'output_path': f'/tmp/doc{i}.pdf', 'file_size': 1024}
            if i < 3 else {'success': False, 'error': 'Export failed'}
            for i in range(5)
        ])
        
        result = await batch_exporter._export_batch(
            documents=documents,
            output_dir=temp_dir,
            formats=['pdf']
        )
        
        assert result['exported_count'] == 3
        assert result['failed_count'] == 2
        assert len(result['failures']) == 2
    
    @pytest.mark.asyncio
    async def test_run_export_job(self, batch_exporter, temp_dir):
        """Test running a complete export job."""
        documents = [{'id': i, 'content': f'Document {i}'} for i in range(25)]
        
        job = await batch_exporter.create_export_job(documents, temp_dir)
        
        # Run the job
        await batch_exporter.run_export_job(job.job_id)
        
        assert job.status == ExportStatus.COMPLETED
        assert job.processed_documents == 25
        assert job.failed_documents == 0
        
        # Verify batches were processed
        # With batch_size=10, we should have 3 batches (10, 10, 5)
        assert batch_exporter.format_exporters['pdf'].export.call_count == 25
    
    @pytest.mark.asyncio
    async def test_run_export_job_with_compression(self, batch_exporter, temp_dir):
        """Test export job with compression enabled."""
        batch_exporter.config.compression_enabled = True
        documents = [{'id': i, 'content': f'Document {i}'} for i in range(10)]
        
        with patch('zipfile.ZipFile') as mock_zip:
            job = await batch_exporter.create_export_job(documents, temp_dir)
            await batch_exporter.run_export_job(job.job_id)
            
            # Verify compression was applied
            mock_zip.assert_called()
    
    @pytest.mark.asyncio
    async def test_export_job_cancellation(self, batch_exporter, temp_dir):
        """Test cancelling an export job."""
        documents = [{'id': i, 'content': f'Document {i}'} for i in range(100)]
        
        job = await batch_exporter.create_export_job(documents, temp_dir)
        
        # Start job in background
        export_task = asyncio.create_task(batch_exporter.run_export_job(job.job_id))
        
        # Cancel after short delay
        await asyncio.sleep(0.1)
        await batch_exporter.cancel_export_job(job.job_id)
        
        # Wait for task to complete
        try:
            await export_task
        except asyncio.CancelledError:
            pass
        
        assert job.status == ExportStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, batch_exporter, temp_dir):
        """Test progress callback functionality."""
        progress_updates = []
        
        def progress_callback(current, total):
            progress_updates.append((current, total))
        
        batch_exporter.config.progress_callback = progress_callback
        documents = [{'id': i, 'content': f'Document {i}'} for i in range(20)]
        
        job = await batch_exporter.create_export_job(documents, temp_dir)
        await batch_exporter.run_export_job(job.job_id)
        
        # Verify progress updates were called
        assert len(progress_updates) > 0
        assert progress_updates[-1] == (20, 20)  # Final update
    
    @pytest.mark.asyncio
    async def test_get_job_status(self, batch_exporter, temp_dir):
        """Test getting job status."""
        documents = [{'id': i} for i in range(10)]
        job = await batch_exporter.create_export_job(documents, temp_dir)
        
        status = await batch_exporter.get_job_status(job.job_id)
        
        assert status['job_id'] == job.job_id
        assert status['status'] == ExportStatus.PENDING
        assert status['total_documents'] == 10
        
        # Test non-existent job
        status = await batch_exporter.get_job_status('non-existent')
        assert status is None
    
    @pytest.mark.asyncio
    async def test_cleanup_completed_jobs(self, batch_exporter, temp_dir):
        """Test cleanup of completed jobs."""
        # Create multiple jobs
        for i in range(5):
            documents = [{'id': j} for j in range(5)]
            job = await batch_exporter.create_export_job(documents, temp_dir)
            await batch_exporter.run_export_job(job.job_id)
        
        assert len(batch_exporter.active_jobs) == 5
        
        # Cleanup completed jobs
        cleaned = await batch_exporter.cleanup_completed_jobs()
        
        assert cleaned == 5
        assert len(batch_exporter.active_jobs) == 0
    
    @pytest.mark.asyncio
    async def test_export_with_metadata(self, batch_exporter, temp_dir):
        """Test export with metadata included."""
        batch_exporter.config.include_metadata = True
        
        documents = [{
            'id': 1,
            'content': 'Test document',
            'metadata': {
                'author': 'Test Author',
                'created_date': '2024-01-01'
            }
        }]
        
        # Mock to capture metadata
        export_calls = []
        async def capture_export(*args, **kwargs):
            export_calls.append(kwargs)
            return {'success': True, 'output_path': '/tmp/doc.pdf', 'file_size': 1024}
        
        batch_exporter.format_exporters['pdf'].export = capture_export
        
        job = await batch_exporter.create_export_job(documents, temp_dir)
        await batch_exporter.run_export_job(job.job_id)
        
        # Verify metadata was passed
        assert len(export_calls) > 0
        assert 'metadata' in export_calls[0]
    
    def test_validate_export_config(self, exporter_config):
        """Test export configuration validation."""
        # Valid config should not raise
        BatchExporter._validate_config(exporter_config)
        
        # Invalid batch size
        exporter_config.batch_size = 0
        with pytest.raises(ValueError, match="Batch size must be positive"):
            BatchExporter._validate_config(exporter_config)
        
        # Invalid formats
        exporter_config.batch_size = 10
        exporter_config.output_formats = []
        with pytest.raises(ValueError, match="At least one output format"):
            BatchExporter._validate_config(exporter_config)