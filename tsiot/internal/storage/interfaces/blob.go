package interfaces

import (
	"context"
	"io"
	"time"
)

// BlobStorage defines the interface for blob/object storage operations
type BlobStorage interface {
	// Put stores a blob with the given key
	Put(ctx context.Context, key string, data io.Reader, size int64, metadata map[string]string) error

	// Get retrieves a blob by key
	Get(ctx context.Context, key string) (io.ReadCloser, error)

	// GetWithMetadata retrieves a blob by key along with its metadata
	GetWithMetadata(ctx context.Context, key string) (io.ReadCloser, map[string]string, error)

	// Head gets metadata about a blob without downloading it
	Head(ctx context.Context, key string) (*BlobInfo, error)

	// Delete removes a blob by key
	Delete(ctx context.Context, key string) error

	// List lists blobs with optional prefix filter
	List(ctx context.Context, prefix string, limit int) ([]*BlobInfo, error)

	// Exists checks if a blob exists
	Exists(ctx context.Context, key string) (bool, error)

	// Copy copies a blob from source to destination key
	Copy(ctx context.Context, sourceKey, destKey string, metadata map[string]string) error

	// Move moves a blob from source to destination key
	Move(ctx context.Context, sourceKey, destKey string, metadata map[string]string) error

	// GetURL generates a pre-signed URL for accessing the blob
	GetURL(ctx context.Context, key string, expiry time.Duration, options *URLOptions) (string, error)

	// PutURL generates a pre-signed URL for uploading to the blob
	PutURL(ctx context.Context, key string, expiry time.Duration, options *URLOptions) (string, error)

	// Batch operations
	BatchPut(ctx context.Context, operations []*BlobPutOperation) error
	BatchDelete(ctx context.Context, keys []string) error

	// Storage management
	GetStorageStats(ctx context.Context) (*BlobStorageStats, error)
	SetMetadata(ctx context.Context, key string, metadata map[string]string) error
	GetMetadata(ctx context.Context, key string) (map[string]string, error)
}

// BlobInfo contains information about a blob
type BlobInfo struct {
	Key          string            `json:"key"`
	Size         int64             `json:"size"`
	ContentType  string            `json:"content_type"`
	ETag         string            `json:"etag"`
	LastModified time.Time         `json:"last_modified"`
	Metadata     map[string]string `json:"metadata"`
	StorageClass string            `json:"storage_class,omitempty"`
	IsDirectory  bool              `json:"is_directory"`
}

// BlobPutOperation represents a single put operation in a batch
type BlobPutOperation struct {
	Key      string            `json:"key"`
	Data     io.Reader         `json:"-"`
	Size     int64             `json:"size"`
	Metadata map[string]string `json:"metadata"`
}

// URLOptions contains options for generating URLs
type URLOptions struct {
	ContentType        string            `json:"content_type,omitempty"`
	ContentDisposition string            `json:"content_disposition,omitempty"`
	Headers            map[string]string `json:"headers,omitempty"`
	Method             string            `json:"method,omitempty"` // GET, PUT, POST, etc.
}

// BlobStorageStats contains storage statistics
type BlobStorageStats struct {
	TotalObjects    int64             `json:"total_objects"`
	TotalSize       int64             `json:"total_size"`
	StorageClass    map[string]int64  `json:"storage_class"`
	LastModified    time.Time         `json:"last_modified"`
	BandwidthUsed   int64             `json:"bandwidth_used"`
	RequestCount    int64             `json:"request_count"`
	ErrorCount      int64             `json:"error_count"`
	AvailableSpace  int64             `json:"available_space,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// MultipartUpload interface for handling large file uploads
type MultipartUpload interface {
	// InitiateMultipartUpload starts a multipart upload
	InitiateMultipartUpload(ctx context.Context, key string, metadata map[string]string) (*MultipartUploadInfo, error)

	// UploadPart uploads a part of the multipart upload
	UploadPart(ctx context.Context, uploadID string, partNumber int, data io.Reader, size int64) (*PartInfo, error)

	// CompleteMultipartUpload completes the multipart upload
	CompleteMultipartUpload(ctx context.Context, uploadID string, parts []*PartInfo) error

	// AbortMultipartUpload aborts the multipart upload
	AbortMultipartUpload(ctx context.Context, uploadID string) error

	// ListParts lists uploaded parts
	ListParts(ctx context.Context, uploadID string) ([]*PartInfo, error)

	// ListMultipartUploads lists ongoing multipart uploads
	ListMultipartUploads(ctx context.Context, prefix string) ([]*MultipartUploadInfo, error)
}

// MultipartUploadInfo contains information about a multipart upload
type MultipartUploadInfo struct {
	UploadID    string            `json:"upload_id"`
	Key         string            `json:"key"`
	Initiated   time.Time         `json:"initiated"`
	Metadata    map[string]string `json:"metadata"`
	StorageClass string           `json:"storage_class,omitempty"`
}

// PartInfo contains information about an uploaded part
type PartInfo struct {
	PartNumber   int       `json:"part_number"`
	ETag         string    `json:"etag"`
	Size         int64     `json:"size"`
	LastModified time.Time `json:"last_modified"`
}

// BlobStreamReader provides streaming read access to blobs
type BlobStreamReader interface {
	io.ReadCloser
	// Seek sets the offset for the next Read
	Seek(offset int64, whence int) (int64, error)
	// Size returns the total size of the blob
	Size() int64
	// ContentType returns the content type of the blob
	ContentType() string
	// Metadata returns the blob metadata
	Metadata() map[string]string
}

// BlobStreamWriter provides streaming write access to blobs
type BlobStreamWriter interface {
	io.WriteCloser
	// SetMetadata sets metadata for the blob
	SetMetadata(metadata map[string]string)
	// SetContentType sets the content type
	SetContentType(contentType string)
	// Abort cancels the upload
	Abort() error
}

// VersionedBlobStorage extends BlobStorage with versioning support
type VersionedBlobStorage interface {
	BlobStorage

	// PutVersion stores a new version of a blob
	PutVersion(ctx context.Context, key string, data io.Reader, size int64, metadata map[string]string) (string, error)

	// GetVersion retrieves a specific version of a blob
	GetVersion(ctx context.Context, key, version string) (io.ReadCloser, error)

	// ListVersions lists all versions of a blob
	ListVersions(ctx context.Context, key string) ([]*BlobVersionInfo, error)

	// DeleteVersion deletes a specific version
	DeleteVersion(ctx context.Context, key, version string) error

	// SetVersionPolicy sets the versioning policy
	SetVersionPolicy(ctx context.Context, policy *VersionPolicy) error

	// GetVersionPolicy gets the current versioning policy
	GetVersionPolicy(ctx context.Context) (*VersionPolicy, error)
}

// BlobVersionInfo contains information about a blob version
type BlobVersionInfo struct {
	Version      string            `json:"version"`
	Size         int64             `json:"size"`
	ETag         string            `json:"etag"`
	LastModified time.Time         `json:"last_modified"`
	IsLatest     bool              `json:"is_latest"`
	Metadata     map[string]string `json:"metadata"`
}

// VersionPolicy defines the versioning policy
type VersionPolicy struct {
	Enabled         bool          `json:"enabled"`
	MaxVersions     int           `json:"max_versions,omitempty"`
	RetentionPeriod time.Duration `json:"retention_period,omitempty"`
	DeleteMarkers   bool          `json:"delete_markers"`
}

// EncryptedBlobStorage extends BlobStorage with encryption support
type EncryptedBlobStorage interface {
	BlobStorage

	// PutEncrypted stores an encrypted blob
	PutEncrypted(ctx context.Context, key string, data io.Reader, size int64, 
		encryptionKey []byte, metadata map[string]string) error

	// GetDecrypted retrieves and decrypts a blob
	GetDecrypted(ctx context.Context, key string, decryptionKey []byte) (io.ReadCloser, error)

	// SetEncryptionPolicy sets the encryption policy
	SetEncryptionPolicy(ctx context.Context, policy *EncryptionPolicy) error

	// GetEncryptionPolicy gets the current encryption policy
	GetEncryptionPolicy(ctx context.Context) (*EncryptionPolicy, error)
}

// EncryptionPolicy defines the encryption policy
type EncryptionPolicy struct {
	Enabled           bool   `json:"enabled"`
	Algorithm         string `json:"algorithm"`
	KeyManagement     string `json:"key_management"`
	DefaultKeyID      string `json:"default_key_id,omitempty"`
	RotationPeriod    time.Duration `json:"rotation_period,omitempty"`
	RequireEncryption bool   `json:"require_encryption"`
}

// BlobNotification represents a blob storage event notification
type BlobNotification struct {
	EventType    string            `json:"event_type"`
	Bucket       string            `json:"bucket"`
	Key          string            `json:"key"`
	Size         int64             `json:"size"`
	ETag         string            `json:"etag"`
	Timestamp    time.Time         `json:"timestamp"`
	UserMetadata map[string]string `json:"user_metadata"`
	Source       string            `json:"source"`
}

// BlobNotificationHandler handles blob storage events
type BlobNotificationHandler interface {
	// HandleNotification processes a blob notification
	HandleNotification(ctx context.Context, notification *BlobNotification) error
}

// BlobLifecycleManager manages blob lifecycle policies
type BlobLifecycleManager interface {
	// SetLifecyclePolicy sets a lifecycle policy
	SetLifecyclePolicy(ctx context.Context, policy *LifecyclePolicy) error

	// GetLifecyclePolicy gets the current lifecycle policy
	GetLifecyclePolicy(ctx context.Context) (*LifecyclePolicy, error)

	// DeleteLifecyclePolicy removes the lifecycle policy
	DeleteLifecyclePolicy(ctx context.Context) error

	// ApplyLifecycleRules manually applies lifecycle rules
	ApplyLifecycleRules(ctx context.Context) (*LifecycleResult, error)
}

// LifecyclePolicy defines blob lifecycle rules
type LifecyclePolicy struct {
	Rules []LifecycleRule `json:"rules"`
}

// LifecycleRule defines a single lifecycle rule
type LifecycleRule struct {
	ID          string                 `json:"id"`
	Status      string                 `json:"status"` // Enabled, Disabled
	Filter      *LifecycleFilter       `json:"filter,omitempty"`
	Transitions []LifecycleTransition  `json:"transitions,omitempty"`
	Expiration  *LifecycleExpiration   `json:"expiration,omitempty"`
}

// LifecycleFilter defines which objects the rule applies to
type LifecycleFilter struct {
	Prefix string            `json:"prefix,omitempty"`
	Tags   map[string]string `json:"tags,omitempty"`
	Size   *SizeFilter       `json:"size,omitempty"`
}

// SizeFilter defines size-based filtering
type SizeFilter struct {
	GreaterThan int64 `json:"greater_than,omitempty"`
	LessThan    int64 `json:"less_than,omitempty"`
}

// LifecycleTransition defines storage class transitions
type LifecycleTransition struct {
	Days         int    `json:"days"`
	StorageClass string `json:"storage_class"`
}

// LifecycleExpiration defines when objects expire
type LifecycleExpiration struct {
	Days                      int  `json:"days,omitempty"`
	ExpiredObjectDeleteMarker bool `json:"expired_object_delete_marker,omitempty"`
}

// LifecycleResult contains the result of applying lifecycle rules
type LifecycleResult struct {
	ProcessedObjects int64                    `json:"processed_objects"`
	TransitionedObjects int64                 `json:"transitioned_objects"`
	DeletedObjects   int64                    `json:"deleted_objects"`
	Errors          []string                  `json:"errors,omitempty"`
	Details         map[string]interface{}    `json:"details,omitempty"`
}