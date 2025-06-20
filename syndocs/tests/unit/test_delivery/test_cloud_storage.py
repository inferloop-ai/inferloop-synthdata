"""
Unit tests for cloud storage implementation.

Tests cover AWS S3, Azure Blob Storage, and Google Cloud Storage clients,
as well as the unified CloudStorage interface.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

from structured_docs_synth.delivery.storage.cloud_storage import (
    CloudStorage,
    CloudStorageConfig,
    CloudStorageClient,
    AWSS3Client,
    AzureBlobClient,
    GoogleCloudClient,
    create_cloud_storage
)
from structured_docs_synth.core.exceptions import DeliveryError, ConfigurationError


class TestCloudStorageConfig:
    """Test CloudStorageConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CloudStorageConfig()
        
        assert config.provider == 'aws'
        assert config.region == 'us-east-1'
        assert config.bucket_name == 'structured-docs-synth'
        assert config.prefix == 'documents/'
        assert config.encryption is True
        assert config.max_concurrent_uploads == 10
        assert config.chunk_size == 8 * 1024 * 1024
        assert config.retry_attempts == 3
        assert config.timeout == 300
        assert config.storage_class == 'STANDARD'
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_config = {
            'provider': 'azure',
            'region': 'westus2',
            'bucket_name': 'test-bucket',
            'encryption': False,
            'max_concurrent_uploads': 5
        }
        
        config = CloudStorageConfig(custom_config)
        
        assert config.provider == 'azure'
        assert config.region == 'westus2'
        assert config.bucket_name == 'test-bucket'
        assert config.encryption is False
        assert config.max_concurrent_uploads == 5


class TestAWSS3Client:
    """Test AWS S3 client implementation."""
    
    @pytest.fixture
    def s3_config(self):
        """Provide S3 configuration."""
        return CloudStorageConfig({
            'provider': 'aws',
            'bucket_name': 'test-bucket',
            'access_key_id': 'test-key',
            'secret_access_key': 'test-secret'
        })
    
    @pytest.fixture
    def s3_client(self, s3_config):
        """Provide S3 client instance."""
        return AWSS3Client(s3_config)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, s3_client):
        """Test successful S3 connection."""
        with patch('aioboto3.Session') as mock_session:
            mock_s3 = AsyncMock()
            mock_s3.head_bucket = AsyncMock(return_value={})
            mock_session.return_value.client.return_value = mock_s3
            
            await s3_client.connect()
            
            mock_s3.head_bucket.assert_called_once_with(Bucket='test-bucket')
            assert s3_client._client is not None
    
    @pytest.mark.asyncio
    async def test_connect_missing_sdk(self, s3_client):
        """Test connection failure when SDK is missing."""
        with patch('aioboto3.Session', side_effect=ImportError):
            with pytest.raises(ConfigurationError, match="AWS SDK.*not installed"):
                await s3_client.connect()
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self, s3_client, temp_dir):
        """Test successful file upload to S3."""
        # Create test file
        test_file = temp_dir / "test.pdf"
        test_file.write_text("test content")
        
        # Mock S3 client
        mock_s3 = AsyncMock()
        mock_s3.put_object = AsyncMock(return_value={
            'ETag': '"test-etag"',
            'VersionId': 'test-version'
        })
        s3_client._client = mock_s3
        
        # Upload file
        result = await s3_client.upload_file(test_file, "test/document.pdf")
        
        # Verify result
        assert result['success'] is True
        assert result['remote_path'] == "documents/test/document.pdf"
        assert result['file_size'] == len("test content")
        assert result['etag'] == 'test-etag'
        assert result['version_id'] == 'test-version'
        assert 's3://test-bucket/documents/test/document.pdf' in result['url']
        
        # Verify S3 call
        mock_s3.put_object.assert_called_once()
        call_args = mock_s3.put_object.call_args[1]
        assert call_args['Bucket'] == 'test-bucket'
        assert call_args['Key'] == 'documents/test/document.pdf'
        assert call_args['ServerSideEncryption'] == 'AES256'
    
    @pytest.mark.asyncio
    async def test_upload_file_failure(self, s3_client, temp_dir):
        """Test file upload failure handling."""
        test_file = temp_dir / "test.pdf"
        test_file.write_text("test content")
        
        mock_s3 = AsyncMock()
        mock_s3.put_object = AsyncMock(side_effect=Exception("Upload failed"))
        s3_client._client = mock_s3
        
        result = await s3_client.upload_file(test_file, "test/document.pdf")
        
        assert result['success'] is False
        assert 'Upload failed' in result['error']
        assert result['remote_path'] == 'test/document.pdf'
    
    @pytest.mark.asyncio
    async def test_download_file_success(self, s3_client, temp_dir):
        """Test successful file download from S3."""
        mock_s3 = AsyncMock()
        mock_body = AsyncMock()
        mock_body.read = AsyncMock(return_value=b"downloaded content")
        mock_s3.get_object = AsyncMock(return_value={
            'Body': mock_body,
            'Metadata': {'original_name': 'test.pdf'}
        })
        s3_client._client = mock_s3
        
        download_path = temp_dir / "downloaded.pdf"
        result = await s3_client.download_file("test/document.pdf", download_path)
        
        assert result['success'] is True
        assert result['local_path'] == str(download_path)
        assert result['file_size'] == len("downloaded content")
        assert download_path.read_text() == "downloaded content"
        assert result['metadata']['original_name'] == 'test.pdf'
    
    @pytest.mark.asyncio
    async def test_delete_file_success(self, s3_client):
        """Test successful file deletion from S3."""
        mock_s3 = AsyncMock()
        mock_s3.delete_object = AsyncMock()
        s3_client._client = mock_s3
        
        result = await s3_client.delete_file("test/document.pdf")
        
        assert result is True
        mock_s3.delete_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='documents/test/document.pdf'
        )
    
    @pytest.mark.asyncio
    async def test_list_files_success(self, s3_client):
        """Test listing files in S3."""
        mock_s3 = AsyncMock()
        mock_s3.list_objects_v2 = AsyncMock(return_value={
            'Contents': [
                {
                    'Key': 'documents/test1.pdf',
                    'Size': 1024,
                    'LastModified': datetime.now(),
                    'ETag': '"etag1"',
                    'StorageClass': 'STANDARD'
                },
                {
                    'Key': 'documents/test2.pdf',
                    'Size': 2048,
                    'LastModified': datetime.now(),
                    'ETag': '"etag2"'
                }
            ]
        })
        s3_client._client = mock_s3
        
        files = await s3_client.list_files("test/", limit=10)
        
        assert len(files) == 2
        assert files[0]['key'] == 'documents/test1.pdf'
        assert files[0]['size'] == 1024
        assert files[1]['key'] == 'documents/test2.pdf'
        assert files[1]['size'] == 2048


class TestAzureBlobClient:
    """Test Azure Blob Storage client implementation."""
    
    @pytest.fixture
    def azure_config(self):
        """Provide Azure configuration."""
        return CloudStorageConfig({
            'provider': 'azure',
            'bucket_name': 'test-container',
            'access_key_id': 'test-account',
            'secret_access_key': 'test-key'
        })
    
    @pytest.fixture
    def azure_client(self, azure_config):
        """Provide Azure client instance."""
        return AzureBlobClient(azure_config)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, azure_client):
        """Test successful Azure connection."""
        with patch('azure.storage.blob.aio.BlobServiceClient') as mock_blob_service:
            mock_client = AsyncMock()
            mock_client.get_account_information = AsyncMock(return_value={'sku_name': 'Standard_LRS'})
            mock_blob_service.from_connection_string.return_value = mock_client
            
            await azure_client.connect()
            
            assert azure_client._client is not None
            mock_client.get_account_information.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self, azure_client, temp_dir):
        """Test successful file upload to Azure."""
        test_file = temp_dir / "test.pdf"
        test_file.write_text("test content")
        
        mock_blob_client = AsyncMock()
        mock_blob_client.upload_blob = AsyncMock()
        mock_blob_client.url = "https://test.blob.core.windows.net/container/blob"
        
        mock_service_client = Mock()
        mock_service_client.get_blob_client.return_value = mock_blob_client
        azure_client._client = mock_service_client
        
        result = await azure_client.upload_file(test_file, "test/document.pdf")
        
        assert result['success'] is True
        assert result['remote_path'] == "documents/test/document.pdf"
        assert result['file_size'] == len("test content")
        assert result['url'] == "https://test.blob.core.windows.net/container/blob"
    
    @pytest.mark.asyncio
    async def test_download_file_success(self, azure_client, temp_dir):
        """Test successful file download from Azure."""
        mock_download_stream = AsyncMock()
        mock_download_stream.readall = AsyncMock(return_value=b"downloaded content")
        
        mock_blob_client = AsyncMock()
        mock_blob_client.download_blob = AsyncMock(return_value=mock_download_stream)
        
        mock_service_client = Mock()
        mock_service_client.get_blob_client.return_value = mock_blob_client
        azure_client._client = mock_service_client
        
        download_path = temp_dir / "downloaded.pdf"
        result = await azure_client.download_file("test/document.pdf", download_path)
        
        assert result['success'] is True
        assert result['local_path'] == str(download_path)
        assert result['file_size'] == len("downloaded content")
        assert download_path.read_text() == "downloaded content"


class TestGoogleCloudClient:
    """Test Google Cloud Storage client implementation."""
    
    @pytest.fixture
    def gcp_config(self):
        """Provide GCP configuration."""
        return CloudStorageConfig({
            'provider': 'gcp',
            'bucket_name': 'test-bucket'
        })
    
    @pytest.fixture
    def gcp_client(self, gcp_config):
        """Provide GCP client instance."""
        return GoogleCloudClient(gcp_config)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, gcp_client):
        """Test successful GCP connection."""
        with patch('google.cloud.storage.Client') as mock_storage:
            mock_client = Mock()
            mock_bucket = Mock()
            mock_bucket.reload = Mock()
            mock_client.bucket.return_value = mock_bucket
            mock_storage.return_value = mock_client
            
            await gcp_client.connect()
            
            assert gcp_client._client is not None
            mock_client.bucket.assert_called_once_with('test-bucket')
            mock_bucket.reload.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self, gcp_client, temp_dir):
        """Test successful file upload to GCS."""
        test_file = temp_dir / "test.pdf"
        test_file.write_text("test content")
        
        mock_blob = Mock()
        mock_blob.etag = "test-etag"
        mock_blob.generation = 123456
        mock_blob.upload_from_filename = Mock()
        
        mock_bucket = Mock()
        mock_bucket.blob.return_value = mock_blob
        
        mock_client = Mock()
        mock_client.bucket.return_value = mock_bucket
        gcp_client._client = mock_client
        
        result = await gcp_client.upload_file(test_file, "test/document.pdf")
        
        assert result['success'] is True
        assert result['remote_path'] == "documents/test/document.pdf"
        assert result['file_size'] == len("test content")
        assert result['etag'] == "test-etag"
        assert result['generation'] == 123456
        assert 'gs://test-bucket/documents/test/document.pdf' in result['url']


class TestCloudStorage:
    """Test unified CloudStorage interface."""
    
    @pytest.mark.asyncio
    async def test_create_aws_client(self):
        """Test creation of AWS client."""
        config = {'provider': 'aws'}
        storage = CloudStorage(config)
        
        assert isinstance(storage._client, AWSS3Client)
        assert storage.config.provider == 'aws'
    
    @pytest.mark.asyncio
    async def test_create_azure_client(self):
        """Test creation of Azure client."""
        config = {'provider': 'azure'}
        storage = CloudStorage(config)
        
        assert isinstance(storage._client, AzureBlobClient)
        assert storage.config.provider == 'azure'
    
    @pytest.mark.asyncio
    async def test_create_gcp_client(self):
        """Test creation of GCP client."""
        config = {'provider': 'gcp'}
        storage = CloudStorage(config)
        
        assert isinstance(storage._client, GoogleCloudClient)
        assert storage.config.provider == 'gcp'
    
    def test_unsupported_provider(self):
        """Test error on unsupported provider."""
        config = {'provider': 'unsupported'}
        
        with pytest.raises(ConfigurationError, match="Unsupported cloud provider"):
            CloudStorage(config)
    
    @pytest.mark.asyncio
    async def test_store_files_success(self, temp_dir):
        """Test storing multiple files."""
        # Create test files
        file1 = temp_dir / "file1.pdf"
        file2 = temp_dir / "file2.pdf"
        file1.write_text("content1")
        file2.write_text("content2")
        
        # Create storage with mocked client
        storage = CloudStorage({'provider': 'aws'})
        mock_client = AsyncMock()
        mock_client.upload_file = AsyncMock(side_effect=[
            {'success': True, 'file_size': 8, 'upload_time': 0.1},
            {'success': True, 'file_size': 8, 'upload_time': 0.1}
        ])
        storage._client = mock_client
        storage._connected = True
        
        # Store files
        result = await storage.store_files([file1, file2])
        
        assert result['total_files'] == 2
        assert result['successful_uploads'] == 2
        assert result['failed_uploads'] == 0
        assert result['total_size'] == 16
        assert len(result['upload_results']) == 2
    
    @pytest.mark.asyncio
    async def test_store_files_with_failures(self, temp_dir):
        """Test storing files with some failures."""
        file1 = temp_dir / "file1.pdf"
        file2 = temp_dir / "file2.pdf"
        file1.write_text("content1")
        
        storage = CloudStorage({'provider': 'aws'})
        mock_client = AsyncMock()
        mock_client.upload_file = AsyncMock(side_effect=[
            {'success': True, 'file_size': 8, 'upload_time': 0.1},
            {'success': False, 'error': 'File not found'}
        ])
        storage._client = mock_client
        storage._connected = True
        
        result = await storage.store_files([file1, file2])
        
        assert result['total_files'] == 2
        assert result['successful_uploads'] == 1
        assert result['failed_uploads'] == 1
    
    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test getting storage statistics."""
        storage = CloudStorage({'provider': 'aws'})
        mock_client = AsyncMock()
        mock_client.list_files = AsyncMock(return_value=[
            {'key': 'doc1.pdf', 'size': 1024},
            {'key': 'doc2.docx', 'size': 2048},
            {'key': 'doc3.pdf', 'size': 512}
        ])
        storage._client = mock_client
        storage._connected = True
        
        stats = await storage.get_statistics()
        
        assert stats['total_files'] == 3
        assert stats['total_size_bytes'] == 3584
        assert stats['total_size_mb'] == pytest.approx(0.00341796875)
        assert stats['file_types']['.pdf'] == 2
        assert stats['file_types']['.docx'] == 1
        assert stats['provider'] == 'aws'


def test_create_cloud_storage_factory():
    """Test cloud storage factory function."""
    storage = create_cloud_storage({'provider': 'azure'})
    
    assert isinstance(storage, CloudStorage)
    assert storage.config.provider == 'azure'