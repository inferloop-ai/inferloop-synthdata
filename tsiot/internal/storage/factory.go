package storage

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	. "github.com/inferloop/tsiot/internal/storage/implementations/timescaledb"
	. "github.com/inferloop/tsiot/internal/storage/implementations/s3"
	. "github.com/inferloop/tsiot/internal/storage/implementations/redis"
)

// Factory implements the StorageFactory interface
type Factory struct {
	creators map[string]interfaces.StorageCreateFunc
	mu       sync.RWMutex
	logger   *logrus.Logger
}

// NewFactory creates a new storage factory
func NewFactory(logger *logrus.Logger) *Factory {
	if logger == nil {
		logger = logrus.New()
	}

	factory := &Factory{
		creators: make(map[string]interfaces.StorageCreateFunc),
		logger:   logger,
	}

	// Register default storage types
	factory.registerDefaults()

	return factory
}

// CreateStorage creates a new storage instance
func (f *Factory) CreateStorage(storageType string, config interfaces.StorageConfig) (interfaces.Storage, error) {
	f.mu.RLock()
	createFunc, exists := f.creators[storageType]
	f.mu.RUnlock()

	if !exists {
		return nil, errors.NewStorageError("UNSUPPORTED_TYPE", fmt.Sprintf("Storage type '%s' is not supported", storageType))
	}

	storage, err := createFunc(config)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "CREATION_FAILED", fmt.Sprintf("Failed to create %s storage", storageType))
	}

	f.logger.WithFields(logrus.Fields{
		"storage_type": storageType,
	}).Info("Created storage instance")

	return storage, nil
}

// GetSupportedTypes returns all supported storage types
func (f *Factory) GetSupportedTypes() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()

	types := make([]string, 0, len(f.creators))
	for storageType := range f.creators {
		types = append(types, storageType)
	}

	return types
}

// RegisterStorage registers a new storage type
func (f *Factory) RegisterStorage(storageType string, createFunc interfaces.StorageCreateFunc) error {
	if storageType == "" {
		return errors.NewValidationError("INVALID_TYPE", "Storage type cannot be empty")
	}

	if createFunc == nil {
		return errors.NewValidationError("INVALID_CREATOR", "Storage create function cannot be nil")
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	f.creators[storageType] = createFunc

	f.logger.WithFields(logrus.Fields{
		"storage_type": storageType,
	}).Info("Registered storage type")

	return nil
}

// IsSupported checks if a storage type is supported
func (f *Factory) IsSupported(storageType string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()

	_, exists := f.creators[storageType]
	return exists
}

// registerDefaults registers the default storage implementations
func (f *Factory) registerDefaults() {
	// Register InfluxDB storage
	f.RegisterStorage(constants.StorageTypeInfluxDB, func(config interfaces.StorageConfig) (interfaces.Storage, error) {
		influxConfig := &InfluxDBConfig{
			URL:           config.ConnectionString,
			Token:         config.Password,
			Organization:  config.Database,
			Bucket:        config.Database,
			Timeout:       config.Timeout,
			BatchSize:     config.BatchSize,
			UseGZip:       config.Compression,
		}

		return NewInfluxDBStorage(influxConfig, f.logger)
	})

	// Register TimescaleDB storage
	f.RegisterStorage(constants.StorageTypeTimescaleDB, func(config interfaces.StorageConfig) (interfaces.Storage, error) {
		timescaleConfig := &TimescaleDBConfig{
			Host:              "localhost",
			Port:              5432,
			Database:          config.Database,
			Username:          config.Username,
			Password:          config.Password,
			SSLMode:           "prefer",
			ConnectTimeout:    config.Timeout,
			QueryTimeout:      config.Timeout,
			MaxConnections:    config.MaxConnections,
			MaxIdleConns:      config.MaxConnections / 2,
			ConnMaxLifetime:   time.Hour,
			ChunkTimeInterval: "1 day",
			CompressionPolicy: config.Compression,
			RetentionPolicy:   config.RetentionPolicy,
		}

		// Parse connection string if provided
		if config.ConnectionString != "" {
			// Parse connection string and override defaults
			// For now, use defaults with connection string override
		}

		return NewTimescaleDBStorage(timescaleConfig, f.logger)
	})

	// Register ClickHouse storage (placeholder)
	f.RegisterStorage(constants.StorageTypeClickhouse, func(config interfaces.StorageConfig) (interfaces.Storage, error) {
		return nil, errors.NewStorageError("NOT_IMPLEMENTED", "ClickHouse storage not yet implemented")
	})

	// Register S3 storage
	f.RegisterStorage(constants.StorageTypeS3, func(config interfaces.StorageConfig) (interfaces.Storage, error) {
		s3Config := &S3Config{
			Region:          "us-east-1",
			Bucket:          config.Database, // Use database field as bucket name
			AccessKeyID:     config.Username,
			SecretAccessKey: config.Password,
			Prefix:          "timeseries",
			Timeout:         config.Timeout,
			MaxRetries:      3,
			PartSize:        64 * 1024 * 1024, // 64MB
			UseCompression:  config.Compression,
			StorageClass:    "STANDARD",
		}

		// Parse additional config from metadata
		if config.Metadata != nil {
			if region, ok := config.Metadata["region"].(string); ok {
				s3Config.Region = region
			}
			if endpoint, ok := config.Metadata["endpoint"].(string); ok {
				s3Config.Endpoint = endpoint
			}
			if forcePathStyle, ok := config.Metadata["force_path_style"].(bool); ok {
				s3Config.ForcePathStyle = forcePathStyle
			}
			if disableSSL, ok := config.Metadata["disable_ssl"].(bool); ok {
				s3Config.DisableSSL = disableSSL
			}
			if prefix, ok := config.Metadata["prefix"].(string); ok {
				s3Config.Prefix = prefix
			}
			if storageClass, ok := config.Metadata["storage_class"].(string); ok {
				s3Config.StorageClass = storageClass
			}
		}

		return NewS3Storage(s3Config, f.logger)
	})

	// Register File storage (placeholder)
	f.RegisterStorage(constants.StorageTypeFile, func(config interfaces.StorageConfig) (interfaces.Storage, error) {
		return nil, errors.NewStorageError("NOT_IMPLEMENTED", "File storage not yet implemented")
	})

	// Register Redis storage
	f.RegisterStorage(constants.StorageTypeRedis, func(config interfaces.StorageConfig) (interfaces.Storage, error) {
		redisConfig := &RedisConfig{
			Addr:            config.ConnectionString,
			Password:        config.Password,
			DB:              0,
			DialTimeout:     config.Timeout,
			ReadTimeout:     config.Timeout,
			WriteTimeout:    config.Timeout,
			PoolSize:        config.MaxConnections,
			MinIdleConns:    config.MaxConnections / 4,
			MaxRetries:      3,
			RetryBackoff:    time.Millisecond * 100,
			IdleTimeout:     time.Minute * 5,
			TTL:             time.Hour * 24, // Default 24 hour TTL
			KeyPrefix:       "tsiot",
			UseStreams:      false, // Default to sorted sets
			StreamMaxLen:    10000,
			UseClustering:   false,
		}

		// Parse additional config from metadata
		if config.Metadata != nil {
			if db, ok := config.Metadata["db"].(float64); ok {
				redisConfig.DB = int(db)
			}
			if ttlStr, ok := config.Metadata["ttl"].(string); ok {
				if ttl, err := time.ParseDuration(ttlStr); err == nil {
					redisConfig.TTL = ttl
				}
			}
			if keyPrefix, ok := config.Metadata["key_prefix"].(string); ok {
				redisConfig.KeyPrefix = keyPrefix
			}
			if useStreams, ok := config.Metadata["use_streams"].(bool); ok {
				redisConfig.UseStreams = useStreams
			}
			if streamMaxLen, ok := config.Metadata["stream_max_len"].(float64); ok {
				redisConfig.StreamMaxLen = int64(streamMaxLen)
			}
			if useClustering, ok := config.Metadata["use_clustering"].(bool); ok {
				redisConfig.UseClustering = useClustering
			}
			if clusterAddrs, ok := config.Metadata["cluster_addrs"].([]interface{}); ok {
				var addrs []string
				for _, addr := range clusterAddrs {
					if addrStr, ok := addr.(string); ok {
						addrs = append(addrs, addrStr)
					}
				}
				redisConfig.ClusterAddrs = addrs
			}
		}

		// Default addr if not provided
		if redisConfig.Addr == "" && !redisConfig.UseClustering {
			redisConfig.Addr = "localhost:6379"
		}

		return NewRedisStorage(redisConfig, f.logger)
	})
}

// Pool implements the StoragePool interface
type Pool struct {
	factory     *Factory
	pools       map[string]*storagePool
	mu          sync.RWMutex
	logger      *logrus.Logger
	maxSize     int
	maxIdleTime time.Duration
}

type storagePool struct {
	storageType string
	config      interfaces.StorageConfig
	instances   []pooledStorage
	mu          sync.Mutex
	factory     *Factory
}

type pooledStorage struct {
	storage  interfaces.Storage
	lastUsed time.Time
	inUse    bool
}

// NewPool creates a new storage pool
func NewPool(factory *Factory, maxSize int, maxIdleTime time.Duration, logger *logrus.Logger) *Pool {
	if logger == nil {
		logger = logrus.New()
	}

	if maxSize <= 0 {
		maxSize = 10
	}

	if maxIdleTime <= 0 {
		maxIdleTime = 30 * time.Minute
	}

	pool := &Pool{
		factory:     factory,
		pools:       make(map[string]*storagePool),
		logger:      logger,
		maxSize:     maxSize,
		maxIdleTime: maxIdleTime,
	}

	// Start cleanup goroutine
	go pool.cleanup()

	return pool
}

// Get gets a storage connection from the pool
func (p *Pool) Get(ctx context.Context, storageType string, config interfaces.StorageConfig) (interfaces.Storage, error) {
	poolKey := fmt.Sprintf("%s:%s", storageType, config.ConnectionString)

	p.mu.Lock()
	pool, exists := p.pools[poolKey]
	if !exists {
		pool = &storagePool{
			storageType: storageType,
			config:      config,
			instances:   make([]pooledStorage, 0),
			factory:     p.factory,
		}
		p.pools[poolKey] = pool
	}
	p.mu.Unlock()

	return pool.get(ctx)
}

// Put returns a storage connection to the pool
func (p *Pool) Put(storage interfaces.Storage) error {
	// In a real implementation, you would identify which pool this storage belongs to
	// and return it to the appropriate pool
	// For now, we'll just close it
	return storage.Close()
}

// Close closes the pool and all connections
func (p *Pool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, pool := range p.pools {
		pool.close()
	}

	p.pools = make(map[string]*storagePool)

	p.logger.Info("Storage pool closed")
	return nil
}

// Stats returns pool statistics
func (p *Pool) Stats() *interfaces.StoragePoolStats {
	p.mu.RLock()
	defer p.mu.RUnlock()

	stats := &interfaces.StoragePoolStats{
		ActiveConnections: 0,
		IdleConnections:   0,
		TotalCreated:      0,
		TotalReused:       0,
		TotalErrors:       0,
	}

	for _, pool := range p.pools {
		pool.mu.Lock()
		for _, instance := range pool.instances {
			if instance.inUse {
				stats.ActiveConnections++
			} else {
				stats.IdleConnections++
			}
		}
		pool.mu.Unlock()
	}

	return stats
}

// get gets a storage instance from the specific pool
func (sp *storagePool) get(ctx context.Context) (interfaces.Storage, error) {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	// Look for an available instance
	for i := range sp.instances {
		if !sp.instances[i].inUse {
			sp.instances[i].inUse = true
			sp.instances[i].lastUsed = time.Now()
			return sp.instances[i].storage, nil
		}
	}

	// If no available instance and under max size, create new one
	if len(sp.instances) < 10 { // maxSize per pool
		storage, err := sp.factory.CreateStorage(sp.storageType, sp.config)
		if err != nil {
			return nil, err
		}

		// Connect the storage
		if err := storage.Connect(ctx); err != nil {
			storage.Close()
			return nil, err
		}

		// Add to pool
		sp.instances = append(sp.instances, pooledStorage{
			storage:  storage,
			lastUsed: time.Now(),
			inUse:    true,
		})

		return storage, nil
	}

	// If pool is full, create a new instance without pooling
	storage, err := sp.factory.CreateStorage(sp.storageType, sp.config)
	if err != nil {
		return nil, err
	}

	if err := storage.Connect(ctx); err != nil {
		storage.Close()
		return nil, err
	}

	return storage, nil
}

// close closes all instances in the pool
func (sp *storagePool) close() {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	for _, instance := range sp.instances {
		instance.storage.Close()
	}

	sp.instances = sp.instances[:0]
}

// cleanup periodically cleans up idle connections
func (p *Pool) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		p.cleanupIdleConnections()
	}
}

// cleanupIdleConnections removes idle connections that have exceeded maxIdleTime
func (p *Pool) cleanupIdleConnections() {
	p.mu.RLock()
	pools := make([]*storagePool, 0, len(p.pools))
	for _, pool := range p.pools {
		pools = append(pools, pool)
	}
	p.mu.RUnlock()

	for _, pool := range pools {
		pool.mu.Lock()
		
		activeInstances := make([]pooledStorage, 0, len(pool.instances))
		for _, instance := range pool.instances {
			if instance.inUse || time.Since(instance.lastUsed) < p.maxIdleTime {
				activeInstances = append(activeInstances, instance)
			} else {
				// Close idle instance
				instance.storage.Close()
				p.logger.Debug("Closed idle storage connection")
			}
		}
		
		pool.instances = activeInstances
		pool.mu.Unlock()
	}
}