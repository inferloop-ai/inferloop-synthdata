package cloud

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// AWSProvider implements CloudStorageProvider for AWS S3
type AWSProvider struct {
	logger *logrus.Logger
	name   string
	config *CloudProviderConfig
}

// NewAWSProvider creates a new AWS S3 provider
func NewAWSProvider(name string, config *CloudProviderConfig, logger *logrus.Logger) (*AWSProvider, error) {
	return &AWSProvider{
		logger: logger,
		name:   name,
		config: config,
	}, nil
}

// Upload uploads data to S3
func (awsp *AWSProvider) Upload(ctx context.Context, key string, data io.Reader, metadata map[string]string) error {
	awsp.logger.WithFields(logrus.Fields{
		"bucket": awsp.config.Bucket,
		"key":    key,
		"region": awsp.config.Region,
	}).Info("Mock: Uploading to AWS S3")
	return nil
}

// Download downloads data from S3
func (awsp *AWSProvider) Download(ctx context.Context, key string) (io.ReadCloser, error) {
	awsp.logger.WithFields(logrus.Fields{
		"bucket": awsp.config.Bucket,
		"key":    key,
	}).Info("Mock: Downloading from AWS S3")
	return io.NopCloser(strings.NewReader("mock s3 data")), nil
}

// Delete deletes object from S3
func (awsp *AWSProvider) Delete(ctx context.Context, key string) error {
	awsp.logger.WithField("key", key).Info("Mock: Deleting from AWS S3")
	return nil
}

// List lists objects in S3
func (awsp *AWSProvider) List(ctx context.Context, prefix string) ([]StorageObject, error) {
	awsp.logger.WithField("prefix", prefix).Info("Mock: Listing AWS S3 objects")
	return []StorageObject{
		{
			Key:          prefix + "/file1.parquet",
			Size:         1024000,
			LastModified: time.Now(),
			StorageClass: "STANDARD",
		},
		{
			Key:          prefix + "/file2.parquet",
			Size:         2048000,
			LastModified: time.Now(),
			StorageClass: "STANDARD_IA",
		},
	}, nil
}

// Copy copies object in S3
func (awsp *AWSProvider) Copy(ctx context.Context, sourceKey, destKey string) error {
	awsp.logger.WithFields(logrus.Fields{
		"source": sourceKey,
		"dest":   destKey,
	}).Info("Mock: Copying in AWS S3")
	return nil
}

// GetMetadata gets object metadata
func (awsp *AWSProvider) GetMetadata(ctx context.Context, key string) (*ObjectMetadata, error) {
	return &ObjectMetadata{
		ContentType:   "application/octet-stream",
		ContentLength: 1024000,
		LastModified:  time.Now(),
		ETag:          "d41d8cd98f00b204e9800998ecf8427e",
		StorageClass:  "STANDARD",
		Encryption:    "AES256",
	}, nil
}

// CreateMultipartUpload creates multipart upload
func (awsp *AWSProvider) CreateMultipartUpload(ctx context.Context, key string) (string, error) {
	uploadID := fmt.Sprintf("upload_%d", time.Now().UnixNano())
	awsp.logger.WithFields(logrus.Fields{
		"key":      key,
		"uploadID": uploadID,
	}).Info("Mock: Created multipart upload")
	return uploadID, nil
}

// UploadPart uploads a part
func (awsp *AWSProvider) UploadPart(ctx context.Context, key, uploadID string, partNumber int, data io.Reader) (string, error) {
	etag := fmt.Sprintf("etag_part_%d", partNumber)
	awsp.logger.WithFields(logrus.Fields{
		"key":        key,
		"uploadID":   uploadID,
		"partNumber": partNumber,
		"etag":       etag,
	}).Info("Mock: Uploaded part")
	return etag, nil
}

// CompleteMultipartUpload completes multipart upload
func (awsp *AWSProvider) CompleteMultipartUpload(ctx context.Context, key, uploadID string, parts []CompletedPart) error {
	awsp.logger.WithFields(logrus.Fields{
		"key":      key,
		"uploadID": uploadID,
		"parts":    len(parts),
	}).Info("Mock: Completed multipart upload")
	return nil
}

// GetSignedURL gets a signed URL
func (awsp *AWSProvider) GetSignedURL(ctx context.Context, key string, expiry time.Duration) (string, error) {
	url := fmt.Sprintf("https://%s.s3.%s.amazonaws.com/%s?X-Amz-Expires=%d", 
		awsp.config.Bucket, awsp.config.Region, key, int(expiry.Seconds()))
	return url, nil
}

// GCPProvider implements CloudStorageProvider for Google Cloud Storage
type GCPProvider struct {
	logger *logrus.Logger
	name   string
	config *CloudProviderConfig
}

// NewGCPProvider creates a new GCP Storage provider
func NewGCPProvider(name string, config *CloudProviderConfig, logger *logrus.Logger) (*GCPProvider, error) {
	return &GCPProvider{
		logger: logger,
		name:   name,
		config: config,
	}, nil
}

// Upload uploads data to GCS
func (gcpp *GCPProvider) Upload(ctx context.Context, key string, data io.Reader, metadata map[string]string) error {
	gcpp.logger.WithFields(logrus.Fields{
		"bucket": gcpp.config.Bucket,
		"key":    key,
	}).Info("Mock: Uploading to Google Cloud Storage")
	return nil
}

// Download downloads data from GCS
func (gcpp *GCPProvider) Download(ctx context.Context, key string) (io.ReadCloser, error) {
	gcpp.logger.WithFields(logrus.Fields{
		"bucket": gcpp.config.Bucket,
		"key":    key,
	}).Info("Mock: Downloading from Google Cloud Storage")
	return io.NopCloser(strings.NewReader("mock gcs data")), nil
}

// Delete deletes object from GCS
func (gcpp *GCPProvider) Delete(ctx context.Context, key string) error {
	gcpp.logger.WithField("key", key).Info("Mock: Deleting from Google Cloud Storage")
	return nil
}

// List lists objects in GCS
func (gcpp *GCPProvider) List(ctx context.Context, prefix string) ([]StorageObject, error) {
	gcpp.logger.WithField("prefix", prefix).Info("Mock: Listing GCS objects")
	return []StorageObject{
		{
			Key:          prefix + "/file1.parquet",
			Size:         1024000,
			LastModified: time.Now(),
			StorageClass: "STANDARD",
		},
	}, nil
}

// Copy copies object in GCS
func (gcpp *GCPProvider) Copy(ctx context.Context, sourceKey, destKey string) error {
	gcpp.logger.WithFields(logrus.Fields{
		"source": sourceKey,
		"dest":   destKey,
	}).Info("Mock: Copying in Google Cloud Storage")
	return nil
}

// GetMetadata gets object metadata
func (gcpp *GCPProvider) GetMetadata(ctx context.Context, key string) (*ObjectMetadata, error) {
	return &ObjectMetadata{
		ContentType:   "application/octet-stream",
		ContentLength: 1024000,
		LastModified:  time.Now(),
		ETag:          "d41d8cd98f00b204e9800998ecf8427e",
		StorageClass:  "STANDARD",
	}, nil
}

// CreateMultipartUpload creates multipart upload
func (gcpp *GCPProvider) CreateMultipartUpload(ctx context.Context, key string) (string, error) {
	uploadID := fmt.Sprintf("gcs_upload_%d", time.Now().UnixNano())
	return uploadID, nil
}

// UploadPart uploads a part
func (gcpp *GCPProvider) UploadPart(ctx context.Context, key, uploadID string, partNumber int, data io.Reader) (string, error) {
	etag := fmt.Sprintf("gcs_etag_part_%d", partNumber)
	return etag, nil
}

// CompleteMultipartUpload completes multipart upload
func (gcpp *GCPProvider) CompleteMultipartUpload(ctx context.Context, key, uploadID string, parts []CompletedPart) error {
	gcpp.logger.Info("Mock: Completed GCS multipart upload")
	return nil
}

// GetSignedURL gets a signed URL
func (gcpp *GCPProvider) GetSignedURL(ctx context.Context, key string, expiry time.Duration) (string, error) {
	url := fmt.Sprintf("https://storage.googleapis.com/%s/%s?X-Goog-Expires=%d", 
		gcpp.config.Bucket, key, int(expiry.Seconds()))
	return url, nil
}

// AzureProvider implements CloudStorageProvider for Azure Blob Storage
type AzureProvider struct {
	logger *logrus.Logger
	name   string
	config *CloudProviderConfig
}

// NewAzureProvider creates a new Azure Blob Storage provider
func NewAzureProvider(name string, config *CloudProviderConfig, logger *logrus.Logger) (*AzureProvider, error) {
	return &AzureProvider{
		logger: logger,
		name:   name,
		config: config,
	}, nil
}

// Upload uploads data to Azure Blob Storage
func (azp *AzureProvider) Upload(ctx context.Context, key string, data io.Reader, metadata map[string]string) error {
	azp.logger.WithFields(logrus.Fields{
		"container": azp.config.Bucket,
		"blob":      key,
	}).Info("Mock: Uploading to Azure Blob Storage")
	return nil
}

// Download downloads data from Azure Blob Storage
func (azp *AzureProvider) Download(ctx context.Context, key string) (io.ReadCloser, error) {
	azp.logger.WithFields(logrus.Fields{
		"container": azp.config.Bucket,
		"blob":      key,
	}).Info("Mock: Downloading from Azure Blob Storage")
	return io.NopCloser(strings.NewReader("mock azure data")), nil
}

// Delete deletes blob from Azure
func (azp *AzureProvider) Delete(ctx context.Context, key string) error {
	azp.logger.WithField("blob", key).Info("Mock: Deleting from Azure Blob Storage")
	return nil
}

// List lists blobs in Azure
func (azp *AzureProvider) List(ctx context.Context, prefix string) ([]StorageObject, error) {
	azp.logger.WithField("prefix", prefix).Info("Mock: Listing Azure blobs")
	return []StorageObject{
		{
			Key:          prefix + "/file1.parquet",
			Size:         1024000,
			LastModified: time.Now(),
			StorageClass: "Hot",
		},
	}, nil
}

// Copy copies blob in Azure
func (azp *AzureProvider) Copy(ctx context.Context, sourceKey, destKey string) error {
	azp.logger.WithFields(logrus.Fields{
		"source": sourceKey,
		"dest":   destKey,
	}).Info("Mock: Copying in Azure Blob Storage")
	return nil
}

// GetMetadata gets blob metadata
func (azp *AzureProvider) GetMetadata(ctx context.Context, key string) (*ObjectMetadata, error) {
	return &ObjectMetadata{
		ContentType:   "application/octet-stream",
		ContentLength: 1024000,
		LastModified:  time.Now(),
		ETag:          "0x8D9F3A5B6C7D8E9",
		StorageClass:  "Hot",
	}, nil
}

// CreateMultipartUpload creates multipart upload
func (azp *AzureProvider) CreateMultipartUpload(ctx context.Context, key string) (string, error) {
	uploadID := fmt.Sprintf("azure_upload_%d", time.Now().UnixNano())
	return uploadID, nil
}

// UploadPart uploads a part
func (azp *AzureProvider) UploadPart(ctx context.Context, key, uploadID string, partNumber int, data io.Reader) (string, error) {
	blockID := fmt.Sprintf("block_%d", partNumber)
	return blockID, nil
}

// CompleteMultipartUpload completes multipart upload
func (azp *AzureProvider) CompleteMultipartUpload(ctx context.Context, key, uploadID string, parts []CompletedPart) error {
	azp.logger.Info("Mock: Completed Azure multipart upload")
	return nil
}

// GetSignedURL gets a signed URL
func (azp *AzureProvider) GetSignedURL(ctx context.Context, key string, expiry time.Duration) (string, error) {
	url := fmt.Sprintf("https://storage.blob.core.windows.net/%s/%s?sv=2020-08-04&se=%s", 
		azp.config.Bucket, key, time.Now().Add(expiry).Format(time.RFC3339))
	return url, nil
}

// LocalProvider implements CloudStorageProvider for local filesystem
type LocalProvider struct {
	logger   *logrus.Logger
	name     string
	config   *CloudProviderConfig
	basePath string
}

// NewLocalProvider creates a new local filesystem provider
func NewLocalProvider(name string, config *CloudProviderConfig, logger *logrus.Logger) (*LocalProvider, error) {
	basePath := config.Bucket
	if basePath == "" {
		basePath = "./storage"
	}

	if err := os.MkdirAll(basePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create local storage directory: %w", err)
	}

	return &LocalProvider{
		logger:   logger,
		name:     name,
		config:   config,
		basePath: basePath,
	}, nil
}

// Upload uploads data to local filesystem
func (lp *LocalProvider) Upload(ctx context.Context, key string, data io.Reader, metadata map[string]string) error {
	fullPath := filepath.Join(lp.basePath, key)
	dir := filepath.Dir(fullPath)

	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	file, err := os.Create(fullPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	_, err = io.Copy(file, data)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	// Store metadata as sidecar file
	if len(metadata) > 0 {
		metaPath := fullPath + ".meta"
		metaFile, err := os.Create(metaPath)
		if err == nil {
			json.NewEncoder(metaFile).Encode(metadata)
			metaFile.Close()
		}
	}

	lp.logger.WithField("path", fullPath).Info("Uploaded to local filesystem")
	return nil
}

// Download downloads data from local filesystem
func (lp *LocalProvider) Download(ctx context.Context, key string) (io.ReadCloser, error) {
	fullPath := filepath.Join(lp.basePath, key)
	
	file, err := os.Open(fullPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	return file, nil
}

// Delete deletes file from local filesystem
func (lp *LocalProvider) Delete(ctx context.Context, key string) error {
	fullPath := filepath.Join(lp.basePath, key)
	
	if err := os.Remove(fullPath); err != nil {
		return fmt.Errorf("failed to delete file: %w", err)
	}

	// Try to remove metadata file
	os.Remove(fullPath + ".meta")

	return nil
}

// List lists files in local filesystem
func (lp *LocalProvider) List(ctx context.Context, prefix string) ([]StorageObject, error) {
	var objects []StorageObject
	
	searchPath := filepath.Join(lp.basePath, prefix)
	
	err := filepath.Walk(searchPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		if !info.IsDir() && !strings.HasSuffix(path, ".meta") {
			relPath, _ := filepath.Rel(lp.basePath, path)
			objects = append(objects, StorageObject{
				Key:          relPath,
				Size:         info.Size(),
				LastModified: info.ModTime(),
				StorageClass: "LOCAL",
			})
		}
		
		return nil
	})

	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("failed to list files: %w", err)
	}

	return objects, nil
}

// Copy copies file in local filesystem
func (lp *LocalProvider) Copy(ctx context.Context, sourceKey, destKey string) error {
	sourcePath := filepath.Join(lp.basePath, sourceKey)
	destPath := filepath.Join(lp.basePath, destKey)
	
	// Create destination directory
	destDir := filepath.Dir(destPath)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	// Copy file
	source, err := os.Open(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to open source file: %w", err)
	}
	defer source.Close()

	dest, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}
	defer dest.Close()

	_, err = io.Copy(dest, source)
	if err != nil {
		return fmt.Errorf("failed to copy file: %w", err)
	}

	// Copy metadata if exists
	os.Rename(sourcePath+".meta", destPath+".meta")

	return nil
}

// GetMetadata gets file metadata
func (lp *LocalProvider) GetMetadata(ctx context.Context, key string) (*ObjectMetadata, error) {
	fullPath := filepath.Join(lp.basePath, key)
	
	info, err := os.Stat(fullPath)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	metadata := &ObjectMetadata{
		ContentType:   "application/octet-stream",
		ContentLength: info.Size(),
		LastModified:  info.ModTime(),
		StorageClass:  "LOCAL",
	}

	// Load custom metadata from sidecar file
	metaPath := fullPath + ".meta"
	if metaFile, err := os.Open(metaPath); err == nil {
		defer metaFile.Close()
		customMeta := make(map[string]string)
		if err := json.NewDecoder(metaFile).Decode(&customMeta); err == nil {
			metadata.CustomMetadata = customMeta
		}
	}

	return metadata, nil
}

// CreateMultipartUpload creates multipart upload (not supported for local)
func (lp *LocalProvider) CreateMultipartUpload(ctx context.Context, key string) (string, error) {
	uploadID := fmt.Sprintf("local_upload_%d", time.Now().UnixNano())
	return uploadID, nil
}

// UploadPart uploads a part (not supported for local)
func (lp *LocalProvider) UploadPart(ctx context.Context, key, uploadID string, partNumber int, data io.Reader) (string, error) {
	// For local storage, we'll just return a mock etag
	etag := fmt.Sprintf("local_part_%d", partNumber)
	return etag, nil
}

// CompleteMultipartUpload completes multipart upload (not supported for local)
func (lp *LocalProvider) CompleteMultipartUpload(ctx context.Context, key, uploadID string, parts []CompletedPart) error {
	// For local storage, this is a no-op
	return nil
}

// GetSignedURL gets a signed URL (returns file:// URL for local)
func (lp *LocalProvider) GetSignedURL(ctx context.Context, key string, expiry time.Duration) (string, error) {
	fullPath := filepath.Join(lp.basePath, key)
	absPath, err := filepath.Abs(fullPath)
	if err != nil {
		return "", fmt.Errorf("failed to get absolute path: %w", err)
	}
	return "file://" + absPath, nil
}