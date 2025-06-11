package interfaces

import (
	"context"
	"time"
)

// Cache defines the interface for caching operations
type Cache interface {
	// Get retrieves a value by key
	Get(ctx context.Context, key string) ([]byte, error)

	// Set stores a value with optional TTL
	Set(ctx context.Context, key string, value []byte, ttl time.Duration) error

	// Delete removes a key
	Delete(ctx context.Context, key string) error

	// Exists checks if a key exists
	Exists(ctx context.Context, key string) (bool, error)

	// Expire sets TTL for an existing key
	Expire(ctx context.Context, key string, ttl time.Duration) error

	// TTL returns the remaining TTL for a key
	TTL(ctx context.Context, key string) (time.Duration, error)

	// Clear removes all keys
	Clear(ctx context.Context) error

	// Keys returns all keys matching a pattern
	Keys(ctx context.Context, pattern string) ([]string, error)

	// Size returns the number of keys
	Size(ctx context.Context) (int64, error)

	// Stats returns cache statistics
	Stats(ctx context.Context) (*CacheStats, error)

	// Health checks cache health
	Health(ctx context.Context) error

	// Close closes the cache connection
	Close() error
}

// CacheStats contains cache performance statistics
type CacheStats struct {
	Hits           int64         `json:"hits"`
	Misses         int64         `json:"misses"`
	Sets           int64         `json:"sets"`
	Deletes        int64         `json:"deletes"`
	Evictions      int64         `json:"evictions"`
	Errors         int64         `json:"errors"`
	KeyCount       int64         `json:"key_count"`
	MemoryUsed     int64         `json:"memory_used"`
	MemoryMax      int64         `json:"memory_max"`
	Uptime         time.Duration `json:"uptime"`
	HitRate        float64       `json:"hit_rate"`
	LastAccess     time.Time     `json:"last_access"`
	ConnectedSince time.Time     `json:"connected_since"`
}

// MultiCache extends Cache with batch operations
type MultiCache interface {
	Cache

	// MGet retrieves multiple values by keys
	MGet(ctx context.Context, keys []string) (map[string][]byte, error)

	// MSet stores multiple key-value pairs
	MSet(ctx context.Context, items map[string][]byte, ttl time.Duration) error

	// MDelete removes multiple keys
	MDelete(ctx context.Context, keys []string) error

	// MExists checks existence of multiple keys
	MExists(ctx context.Context, keys []string) (map[string]bool, error)
}

// DistributedCache extends MultiCache with distributed caching features
type DistributedCache interface {
	MultiCache

	// Lock acquires a distributed lock
	Lock(ctx context.Context, key string, ttl time.Duration) (*Lock, error)

	// TryLock attempts to acquire a lock without blocking
	TryLock(ctx context.Context, key string, ttl time.Duration) (*Lock, error)

	// Unlock releases a distributed lock
	Unlock(ctx context.Context, lock *Lock) error

	// Increment atomically increments a counter
	Increment(ctx context.Context, key string, delta int64) (int64, error)

	// Decrement atomically decrements a counter
	Decrement(ctx context.Context, key string, delta int64) (int64, error)

	// Add sets a key only if it doesn't exist
	Add(ctx context.Context, key string, value []byte, ttl time.Duration) error

	// Replace sets a key only if it exists
	Replace(ctx context.Context, key string, value []byte, ttl time.Duration) error

	// CompareAndSwap atomically updates a value if it matches expected
	CompareAndSwap(ctx context.Context, key string, expected, new []byte, ttl time.Duration) (bool, error)

	// Subscribe subscribes to key changes
	Subscribe(ctx context.Context, pattern string) (<-chan *CacheEvent, error)

	// Unsubscribe unsubscribes from key changes
	Unsubscribe(ctx context.Context, pattern string) error

	// Publish publishes a message to subscribers
	Publish(ctx context.Context, channel string, message []byte) error
}

// Lock represents a distributed lock
type Lock struct {
	Key       string    `json:"key"`
	Value     string    `json:"value"`
	TTL       time.Duration `json:"ttl"`
	AcquiredAt time.Time `json:"acquired_at"`
	Owner     string    `json:"owner"`
}

// CacheEvent represents a cache event notification
type CacheEvent struct {
	Type      string    `json:"type"`      // set, delete, expire, evict
	Key       string    `json:"key"`
	Value     []byte    `json:"value,omitempty"`
	Timestamp time.Time `json:"timestamp"`
	Source    string    `json:"source"`
}

// TaggedCache extends Cache with tag-based operations
type TaggedCache interface {
	Cache

	// SetWithTags stores a value with associated tags
	SetWithTags(ctx context.Context, key string, value []byte, ttl time.Duration, tags []string) error

	// GetTags returns tags associated with a key
	GetTags(ctx context.Context, key string) ([]string, error)

	// DeleteByTag removes all keys associated with a tag
	DeleteByTag(ctx context.Context, tag string) error

	// KeysByTag returns all keys associated with a tag
	KeysByTag(ctx context.Context, tag string) ([]string, error)

	// SetTags sets tags for an existing key
	SetTags(ctx context.Context, key string, tags []string) error

	// AddTag adds a tag to an existing key
	AddTag(ctx context.Context, key string, tag string) error

	// RemoveTag removes a tag from a key
	RemoveTag(ctx context.Context, key string, tag string) error
}

// LayeredCache implements cache layering (L1, L2, etc.)
type LayeredCache interface {
	Cache

	// SetLayer stores a value in a specific layer
	SetLayer(ctx context.Context, layer int, key string, value []byte, ttl time.Duration) error

	// GetLayer retrieves a value from a specific layer
	GetLayer(ctx context.Context, layer int, key string) ([]byte, error)

	// Promote moves a value to a higher cache layer
	Promote(ctx context.Context, key string, fromLayer, toLayer int) error

	// Demote moves a value to a lower cache layer
	Demote(ctx context.Context, key string, fromLayer, toLayer int) error

	// LayerStats returns statistics for a specific layer
	LayerStats(ctx context.Context, layer int) (*CacheStats, error)

	// Invalidate removes a key from all layers
	Invalidate(ctx context.Context, key string) error
}

// CompressionCache extends Cache with compression support
type CompressionCache interface {
	Cache

	// SetCompressed stores a compressed value
	SetCompressed(ctx context.Context, key string, value []byte, algorithm string, ttl time.Duration) error

	// GetDecompressed retrieves and decompresses a value
	GetDecompressed(ctx context.Context, key string) ([]byte, error)

	// SetCompressionPolicy sets the compression policy
	SetCompressionPolicy(ctx context.Context, policy *CompressionPolicy) error

	// GetCompressionPolicy gets the current compression policy
	GetCompressionPolicy(ctx context.Context) (*CompressionPolicy, error)
}

// CompressionPolicy defines compression settings
type CompressionPolicy struct {
	Enabled           bool   `json:"enabled"`
	Algorithm         string `json:"algorithm"`         // gzip, lz4, snappy, zstd
	MinSize           int    `json:"min_size"`          // minimum size to compress
	CompressionLevel  int    `json:"compression_level"` // compression level (algorithm specific)
	AutoDetect        bool   `json:"auto_detect"`       // auto-detect best algorithm
}

// CacheMetrics provides detailed cache metrics
type CacheMetrics struct {
	// Performance metrics
	ResponseTime     time.Duration `json:"response_time"`
	Throughput       float64       `json:"throughput"`       // operations per second
	ErrorRate        float64       `json:"error_rate"`       // errors per second
	SuccessRate      float64       `json:"success_rate"`     // success rate percentage

	// Memory metrics
	MemoryUtilization float64 `json:"memory_utilization"` // percentage
	MemoryFragmentation float64 `json:"memory_fragmentation"`
	KeyspaceUtilization float64 `json:"keyspace_utilization"`

	// Network metrics
	NetworkBytesIn  int64 `json:"network_bytes_in"`
	NetworkBytesOut int64 `json:"network_bytes_out"`
	ConnectionCount int   `json:"connection_count"`

	// Operation metrics
	GetOperations    int64 `json:"get_operations"`
	SetOperations    int64 `json:"set_operations"`
	DeleteOperations int64 `json:"delete_operations"`
	ScanOperations   int64 `json:"scan_operations"`

	// Distribution metrics (for distributed caches)
	NodeCount      int                    `json:"node_count"`
	PartitionCount int                    `json:"partition_count"`
	ReplicationFactor int                 `json:"replication_factor"`
	NodeStats      map[string]*CacheStats `json:"node_stats,omitempty"`
}