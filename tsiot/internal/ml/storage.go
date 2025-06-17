package ml

import (
	"context"
	"crypto/md5"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// LocalModelStorage implements ModelStorage for local filesystem
type LocalModelStorage struct {
	logger   *logrus.Logger
	basePath string
}

// NewLocalModelStorage creates a new local model storage
func NewLocalModelStorage(basePath string, logger *logrus.Logger) (*LocalModelStorage, error) {
	if err := os.MkdirAll(basePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage directory: %w", err)
	}

	return &LocalModelStorage{
		logger:   logger,
		basePath: basePath,
	}, nil
}

// Store stores a model artifact
func (lms *LocalModelStorage) Store(ctx context.Context, modelID, versionID string, artifact io.Reader) (string, error) {
	// Create directory structure: basePath/modelID/versionID/
	modelDir := filepath.Join(lms.basePath, modelID)
	versionDir := filepath.Join(modelDir, versionID)
	
	if err := os.MkdirAll(versionDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create version directory: %w", err)
	}

	// Store artifact as model.bin
	artifactPath := filepath.Join(versionDir, "model.bin")
	file, err := os.Create(artifactPath)
	if err != nil {
		return "", fmt.Errorf("failed to create artifact file: %w", err)
	}
	defer file.Close()

	// Copy artifact data
	_, err = io.Copy(file, artifact)
	if err != nil {
		return "", fmt.Errorf("failed to store artifact: %w", err)
	}

	lms.logger.WithFields(logrus.Fields{
		"model_id":   modelID,
		"version_id": versionID,
		"path":       artifactPath,
	}).Info("Stored model artifact")

	return artifactPath, nil
}

// Retrieve retrieves a model artifact
func (lms *LocalModelStorage) Retrieve(ctx context.Context, path string) (io.ReadCloser, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open artifact: %w", err)
	}

	return file, nil
}

// Delete deletes a model artifact
func (lms *LocalModelStorage) Delete(ctx context.Context, path string) error {
	if err := os.Remove(path); err != nil {
		return fmt.Errorf("failed to delete artifact: %w", err)
	}

	// Try to remove empty directories
	dir := filepath.Dir(path)
	for dir != lms.basePath {
		if err := os.Remove(dir); err != nil {
			break // Directory not empty or other error
		}
		dir = filepath.Dir(dir)
	}

	lms.logger.WithField("path", path).Info("Deleted model artifact")
	return nil
}

// Exists checks if an artifact exists
func (lms *LocalModelStorage) Exists(ctx context.Context, path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

// ListVersions lists all versions for a model
func (lms *LocalModelStorage) ListVersions(ctx context.Context, modelID string) ([]string, error) {
	modelDir := filepath.Join(lms.basePath, modelID)
	
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, fmt.Errorf("failed to list versions: %w", err)
	}

	var versions []string
	for _, entry := range entries {
		if entry.IsDir() {
			versions = append(versions, entry.Name())
		}
	}

	return versions, nil
}

// GetMetadata returns metadata about a stored artifact
func (lms *LocalModelStorage) GetMetadata(ctx context.Context, path string) (*StorageMetadata, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}

	// Calculate checksum
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file for checksum: %w", err)
	}
	defer file.Close()

	hash := md5.New()
	_, err = io.Copy(hash, file)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate checksum: %w", err)
	}

	checksum := fmt.Sprintf("%x", hash.Sum(nil))

	metadata := &StorageMetadata{
		Size:         info.Size(),
		Checksum:     checksum,
		ContentType:  "application/octet-stream",
		LastModified: info.ModTime(),
		Metadata:     make(map[string]string),
	}

	return metadata, nil
}

// S3ModelStorage implements ModelStorage for Amazon S3 (mock implementation)
type S3ModelStorage struct {
	logger   *logrus.Logger
	bucket   string
	region   string
	endpoint string
}

// NewS3ModelStorage creates a new S3 model storage
func NewS3ModelStorage(bucket, region, endpoint string, logger *logrus.Logger) *S3ModelStorage {
	return &S3ModelStorage{
		logger:   logger,
		bucket:   bucket,
		region:   region,
		endpoint: endpoint,
	}
}

// Store stores a model artifact in S3
func (s3s *S3ModelStorage) Store(ctx context.Context, modelID, versionID string, artifact io.Reader) (string, error) {
	key := fmt.Sprintf("models/%s/%s/model.bin", modelID, versionID)
	
	// Mock S3 upload
	s3s.logger.WithFields(logrus.Fields{
		"bucket":     s3s.bucket,
		"key":        key,
		"model_id":   modelID,
		"version_id": versionID,
	}).Info("Mock: Stored model artifact in S3")

	return fmt.Sprintf("s3://%s/%s", s3s.bucket, key), nil
}

// Retrieve retrieves a model artifact from S3
func (s3s *S3ModelStorage) Retrieve(ctx context.Context, path string) (io.ReadCloser, error) {
	// Mock S3 download
	s3s.logger.WithField("path", path).Info("Mock: Retrieved model artifact from S3")
	
	// Return empty reader for mock
	return io.NopCloser(strings.NewReader("")), nil
}

// Delete deletes a model artifact from S3
func (s3s *S3ModelStorage) Delete(ctx context.Context, path string) error {
	// Mock S3 delete
	s3s.logger.WithField("path", path).Info("Mock: Deleted model artifact from S3")
	return nil
}

// Exists checks if an artifact exists in S3
func (s3s *S3ModelStorage) Exists(ctx context.Context, path string) (bool, error) {
	// Mock S3 exists check
	return true, nil
}

// ListVersions lists all versions for a model in S3
func (s3s *S3ModelStorage) ListVersions(ctx context.Context, modelID string) ([]string, error) {
	// Mock S3 list
	return []string{"v1.0.0", "v1.1.0", "v2.0.0"}, nil
}

// GetMetadata returns metadata about a stored artifact in S3
func (s3s *S3ModelStorage) GetMetadata(ctx context.Context, path string) (*StorageMetadata, error) {
	// Mock S3 metadata
	return &StorageMetadata{
		Size:         1024000,
		Checksum:     "d41d8cd98f00b204e9800998ecf8427e",
		ContentType:  "application/octet-stream",
		LastModified: time.Now(),
		Metadata:     map[string]string{"source": "s3"},
	}, nil
}

// GCSModelStorage implements ModelStorage for Google Cloud Storage (mock implementation)
type GCSModelStorage struct {
	logger     *logrus.Logger
	bucket     string
	projectID  string
}

// NewGCSModelStorage creates a new GCS model storage
func NewGCSModelStorage(bucket, projectID string, logger *logrus.Logger) *GCSModelStorage {
	return &GCSModelStorage{
		logger:    logger,
		bucket:    bucket,
		projectID: projectID,
	}
}

// Store stores a model artifact in GCS
func (gcs *GCSModelStorage) Store(ctx context.Context, modelID, versionID string, artifact io.Reader) (string, error) {
	objectName := fmt.Sprintf("models/%s/%s/model.bin", modelID, versionID)
	
	// Mock GCS upload
	gcs.logger.WithFields(logrus.Fields{
		"bucket":      gcs.bucket,
		"object":      objectName,
		"model_id":    modelID,
		"version_id":  versionID,
	}).Info("Mock: Stored model artifact in GCS")

	return fmt.Sprintf("gs://%s/%s", gcs.bucket, objectName), nil
}

// Retrieve retrieves a model artifact from GCS
func (gcs *GCSModelStorage) Retrieve(ctx context.Context, path string) (io.ReadCloser, error) {
	// Mock GCS download
	gcs.logger.WithField("path", path).Info("Mock: Retrieved model artifact from GCS")
	
	// Return empty reader for mock
	return io.NopCloser(strings.NewReader("")), nil
}

// Delete deletes a model artifact from GCS
func (gcs *GCSModelStorage) Delete(ctx context.Context, path string) error {
	// Mock GCS delete
	gcs.logger.WithField("path", path).Info("Mock: Deleted model artifact from GCS")
	return nil
}

// Exists checks if an artifact exists in GCS
func (gcs *GCSModelStorage) Exists(ctx context.Context, path string) (bool, error) {
	// Mock GCS exists check
	return true, nil
}

// ListVersions lists all versions for a model in GCS
func (gcs *GCSModelStorage) ListVersions(ctx context.Context, modelID string) ([]string, error) {
	// Mock GCS list
	return []string{"v1.0.0", "v1.1.0", "v2.0.0"}, nil
}

// GetMetadata returns metadata about a stored artifact in GCS
func (gcs *GCSModelStorage) GetMetadata(ctx context.Context, path string) (*StorageMetadata, error) {
	// Mock GCS metadata
	return &StorageMetadata{
		Size:         1024000,
		Checksum:     "d41d8cd98f00b204e9800998ecf8427e",
		ContentType:  "application/octet-stream",
		LastModified: time.Now(),
		Metadata:     map[string]string{"source": "gcs"},
	}, nil
}

// AzureModelStorage implements ModelStorage for Azure Blob Storage (mock implementation)
type AzureModelStorage struct {
	logger        *logrus.Logger
	accountName   string
	containerName string
}

// NewAzureModelStorage creates a new Azure model storage
func NewAzureModelStorage(accountName, containerName string, logger *logrus.Logger) *AzureModelStorage {
	return &AzureModelStorage{
		logger:        logger,
		accountName:   accountName,
		containerName: containerName,
	}
}

// Store stores a model artifact in Azure Blob Storage
func (azs *AzureModelStorage) Store(ctx context.Context, modelID, versionID string, artifact io.Reader) (string, error) {
	blobName := fmt.Sprintf("models/%s/%s/model.bin", modelID, versionID)
	
	// Mock Azure upload
	azs.logger.WithFields(logrus.Fields{
		"account":    azs.accountName,
		"container":  azs.containerName,
		"blob":       blobName,
		"model_id":   modelID,
		"version_id": versionID,
	}).Info("Mock: Stored model artifact in Azure Blob Storage")

	return fmt.Sprintf("https://%s.blob.core.windows.net/%s/%s", azs.accountName, azs.containerName, blobName), nil
}

// Retrieve retrieves a model artifact from Azure Blob Storage
func (azs *AzureModelStorage) Retrieve(ctx context.Context, path string) (io.ReadCloser, error) {
	// Mock Azure download
	azs.logger.WithField("path", path).Info("Mock: Retrieved model artifact from Azure Blob Storage")
	
	// Return empty reader for mock
	return io.NopCloser(strings.NewReader("")), nil
}

// Delete deletes a model artifact from Azure Blob Storage
func (azs *AzureModelStorage) Delete(ctx context.Context, path string) error {
	// Mock Azure delete
	azs.logger.WithField("path", path).Info("Mock: Deleted model artifact from Azure Blob Storage")
	return nil
}

// Exists checks if an artifact exists in Azure Blob Storage
func (azs *AzureModelStorage) Exists(ctx context.Context, path string) (bool, error) {
	// Mock Azure exists check
	return true, nil
}

// ListVersions lists all versions for a model in Azure Blob Storage
func (azs *AzureModelStorage) ListVersions(ctx context.Context, modelID string) ([]string, error) {
	// Mock Azure list
	return []string{"v1.0.0", "v1.1.0", "v2.0.0"}, nil
}

// GetMetadata returns metadata about a stored artifact in Azure Blob Storage
func (azs *AzureModelStorage) GetMetadata(ctx context.Context, path string) (*StorageMetadata, error) {
	// Mock Azure metadata
	return &StorageMetadata{
		Size:         1024000,
		Checksum:     "d41d8cd98f00b204e9800998ecf8427e",
		ContentType:  "application/octet-stream",
		LastModified: time.Now(),
		Metadata:     map[string]string{"source": "azure"},
	}, nil
}

// StorageFactory creates storage backends based on configuration
type StorageFactory struct{}

// NewStorageFactory creates a new storage factory
func NewStorageFactory() *StorageFactory {
	return &StorageFactory{}
}

// CreateStorage creates a storage backend based on the configuration
func (sf *StorageFactory) CreateStorage(config *RegistryConfig, logger *logrus.Logger) (ModelStorage, error) {
	switch config.StorageBackend {
	case "local":
		return NewLocalModelStorage(config.StoragePath, logger)
	case "s3":
		// Parse S3 configuration from storage path or environment
		bucket := "ml-models-bucket"
		region := "us-west-2"
		endpoint := ""
		return NewS3ModelStorage(bucket, region, endpoint, logger), nil
	case "gcs":
		// Parse GCS configuration
		bucket := "ml-models-bucket"
		projectID := "my-project"
		return NewGCSModelStorage(bucket, projectID, logger), nil
	case "azure":
		// Parse Azure configuration
		accountName := "mlmodels"
		containerName := "models"
		return NewAzureModelStorage(accountName, containerName, logger), nil
	default:
		return nil, fmt.Errorf("unsupported storage backend: %s", config.StorageBackend)
	}
}