package interfaces

import (
	"context"
	"time"
)

// VectorStorage defines the interface for vector database operations
type VectorStorage interface {
	// Collection operations
	CreateCollection(ctx context.Context, collection *VectorCollection) error
	GetCollection(ctx context.Context, name string) (*VectorCollection, error)
	UpdateCollection(ctx context.Context, collection *VectorCollection) error
	DeleteCollection(ctx context.Context, name string) error
	ListCollections(ctx context.Context) ([]*VectorCollection, error)

	// Vector operations
	Insert(ctx context.Context, collection string, vectors []*Vector) error
	Update(ctx context.Context, collection string, vectors []*Vector) error
	Delete(ctx context.Context, collection string, ids []string) error
	Get(ctx context.Context, collection string, ids []string) ([]*Vector, error)

	// Search operations
	Search(ctx context.Context, collection string, query *VectorQuery) (*VectorSearchResult, error)
	SearchByVector(ctx context.Context, collection string, vector []float32, limit int, filters map[string]interface{}) (*VectorSearchResult, error)
	SearchByText(ctx context.Context, collection string, text string, limit int, filters map[string]interface{}) (*VectorSearchResult, error)
	SearchSimilar(ctx context.Context, collection string, id string, limit int, filters map[string]interface{}) (*VectorSearchResult, error)

	// Batch operations
	BatchInsert(ctx context.Context, collection string, vectors []*Vector, batchSize int) error
	BatchUpdate(ctx context.Context, collection string, vectors []*Vector, batchSize int) error
	BatchDelete(ctx context.Context, collection string, ids []string, batchSize int) error

	// Index operations
	CreateIndex(ctx context.Context, collection string, index *VectorIndex) error
	DeleteIndex(ctx context.Context, collection string, indexName string) error
	ListIndexes(ctx context.Context, collection string) ([]*VectorIndex, error)
	RebuildIndex(ctx context.Context, collection string, indexName string) error

	// Health and monitoring
	Health(ctx context.Context) (*VectorStorageHealth, error)
	GetMetrics(ctx context.Context) (*VectorStorageMetrics, error)
	GetCollectionStats(ctx context.Context, collection string) (*VectorCollectionStats, error)
}

// EmbeddingVectorStorage extends VectorStorage with embedding generation
type EmbeddingVectorStorage interface {
	VectorStorage

	// Generate embeddings from text
	GenerateEmbedding(ctx context.Context, text string, model string) ([]float32, error)
	GenerateEmbeddings(ctx context.Context, texts []string, model string) ([][]float32, error)

	// Insert with automatic embedding generation
	InsertWithText(ctx context.Context, collection string, items []*TextVectorItem) error
	UpdateWithText(ctx context.Context, collection string, items []*TextVectorItem) error

	// Search with automatic embedding generation
	SearchByTextWithEmbedding(ctx context.Context, collection string, text string, model string, limit int, filters map[string]interface{}) (*VectorSearchResult, error)

	// Embedding model management
	ListEmbeddingModels(ctx context.Context) ([]*EmbeddingModel, error)
	GetEmbeddingModel(ctx context.Context, name string) (*EmbeddingModel, error)
	SetDefaultEmbeddingModel(ctx context.Context, name string) error
}

// MultimodalVectorStorage extends VectorStorage with multimodal support
type MultimodalVectorStorage interface {
	VectorStorage

	// Generate embeddings from different modalities
	GenerateImageEmbedding(ctx context.Context, imageData []byte, model string) ([]float32, error)
	GenerateAudioEmbedding(ctx context.Context, audioData []byte, model string) ([]float32, error)
	GenerateVideoEmbedding(ctx context.Context, videoData []byte, model string) ([]float32, error)

	// Cross-modal search
	SearchImageByText(ctx context.Context, collection string, text string, limit int, filters map[string]interface{}) (*VectorSearchResult, error)
	SearchTextByImage(ctx context.Context, collection string, imageData []byte, limit int, filters map[string]interface{}) (*VectorSearchResult, error)
	SearchMultimodal(ctx context.Context, collection string, query *MultimodalQuery) (*VectorSearchResult, error)
}

// Vector represents a vector with metadata
type Vector struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Text     string                 `json:"text,omitempty"`
	Image    []byte                 `json:"image,omitempty"`
	Audio    []byte                 `json:"audio,omitempty"`
	Video    []byte                 `json:"video,omitempty"`
}

// TextVectorItem represents text with metadata for embedding generation
type TextVectorItem struct {
	ID       string                 `json:"id"`
	Text     string                 `json:"text"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Model    string                 `json:"model,omitempty"`
}

// VectorCollection represents a collection of vectors
type VectorCollection struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Schema      *VectorSchema          `json:"schema"`
	IndexConfig *VectorIndexConfig     `json:"index_config"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// VectorSchema defines the schema for vectors in a collection
type VectorSchema struct {
	Dimension   int                    `json:"dimension"`
	DataType    string                 `json:"data_type"`    // float32, float64, int8, binary
	MetricType  string                 `json:"metric_type"`  // L2, IP, COSINE, HAMMING, JACCARD
	Properties  []*VectorProperty      `json:"properties"`
	Constraints map[string]interface{} `json:"constraints,omitempty"`
}

// VectorProperty defines a metadata property
type VectorProperty struct {
	Name        string      `json:"name"`
	Type        string      `json:"type"`        // string, int64, float64, bool, array
	Required    bool        `json:"required"`
	Indexed     bool        `json:"indexed"`
	Description string      `json:"description"`
	Default     interface{} `json:"default,omitempty"`
}

// VectorIndexConfig defines indexing configuration
type VectorIndexConfig struct {
	IndexType   string                 `json:"index_type"`   // IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, ANNOY
	MetricType  string                 `json:"metric_type"`  // L2, IP, COSINE
	Parameters  map[string]interface{} `json:"parameters"`
	AutoIndex   bool                   `json:"auto_index"`
}

// VectorIndex represents a vector index
type VectorIndex struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Parameters  map[string]interface{} `json:"parameters"`
	Status      string                 `json:"status"`      // building, built, failed
	Progress    float64                `json:"progress"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// VectorQuery represents a vector search query
type VectorQuery struct {
	Vector      []float32              `json:"vector,omitempty"`
	Text        string                 `json:"text,omitempty"`
	ID          string                 `json:"id,omitempty"`
	Limit       int                    `json:"limit"`
	Offset      int                    `json:"offset"`
	Filters     map[string]interface{} `json:"filters,omitempty"`
	MetricType  string                 `json:"metric_type,omitempty"`
	SearchParams map[string]interface{} `json:"search_params,omitempty"`
	OutputFields []string              `json:"output_fields,omitempty"`
}

// MultimodalQuery represents a multimodal search query
type MultimodalQuery struct {
	Text        string                 `json:"text,omitempty"`
	Image       []byte                 `json:"image,omitempty"`
	Audio       []byte                 `json:"audio,omitempty"`
	Video       []byte                 `json:"video,omitempty"`
	Weights     map[string]float32     `json:"weights,omitempty"` // modality weights
	Limit       int                    `json:"limit"`
	Offset      int                    `json:"offset"`
	Filters     map[string]interface{} `json:"filters,omitempty"`
	MetricType  string                 `json:"metric_type,omitempty"`
	OutputFields []string              `json:"output_fields,omitempty"`
}

// VectorSearchResult represents search results
type VectorSearchResult struct {
	Vectors     []*VectorMatch         `json:"vectors"`
	Total       int64                  `json:"total"`
	SearchTime  time.Duration          `json:"search_time"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// VectorMatch represents a search result match
type VectorMatch struct {
	ID       string                 `json:"id"`
	Score    float32                `json:"score"`
	Vector   []float32              `json:"vector,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Text     string                 `json:"text,omitempty"`
	Distance float32                `json:"distance"`
}

// EmbeddingModel represents an embedding model
type EmbeddingModel struct {
	Name        string                 `json:"name"`
	Provider    string                 `json:"provider"`
	Dimension   int                    `json:"dimension"`
	MaxTokens   int                    `json:"max_tokens"`
	Languages   []string               `json:"languages"`
	Modalities  []string               `json:"modalities"` // text, image, audio, video
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
}

// VectorStorageHealth represents the health status of vector storage
type VectorStorageHealth struct {
	Status          string                 `json:"status"` // healthy, degraded, unhealthy
	LastCheck       time.Time              `json:"last_check"`
	ResponseTime    time.Duration          `json:"response_time"`
	Collections     int                    `json:"collections"`
	TotalVectors    int64                  `json:"total_vectors"`
	IndexStatus     map[string]string      `json:"index_status"`
	MemoryUsage     int64                  `json:"memory_usage"`
	DiskUsage       int64                  `json:"disk_usage"`
	SearchQPS       float64                `json:"search_qps"`
	InsertQPS       float64                `json:"insert_qps"`
	Errors          []string               `json:"errors,omitempty"`
	Warnings        []string               `json:"warnings,omitempty"`
	Details         map[string]interface{} `json:"details,omitempty"`
}

// VectorStorageMetrics contains detailed performance metrics
type VectorStorageMetrics struct {
	// Operation counts
	SearchOperations int64 `json:"search_operations"`
	InsertOperations int64 `json:"insert_operations"`
	UpdateOperations int64 `json:"update_operations"`
	DeleteOperations int64 `json:"delete_operations"`

	// Performance metrics
	AverageSearchTime   time.Duration `json:"average_search_time"`
	AverageInsertTime   time.Duration `json:"average_insert_time"`
	P95SearchTime       time.Duration `json:"p95_search_time"`
	P95InsertTime       time.Duration `json:"p95_insert_time"`
	SearchThroughput    float64       `json:"search_throughput"`    // searches per second
	InsertThroughput    float64       `json:"insert_throughput"`    // inserts per second

	// Accuracy metrics
	SearchAccuracy      float64 `json:"search_accuracy"`      // recall@k
	IndexAccuracy       float64 `json:"index_accuracy"`       // index quality
	EmbeddingQuality    float64 `json:"embedding_quality"`    // embedding consistency

	// Resource metrics
	MemoryUsage         int64 `json:"memory_usage"`
	DiskUsage           int64 `json:"disk_usage"`
	NetworkBytesIn      int64 `json:"network_bytes_in"`
	NetworkBytesOut     int64 `json:"network_bytes_out"`
	CPUUsage            float64 `json:"cpu_usage"`
	GPUUsage            float64 `json:"gpu_usage,omitempty"`

	// Data metrics
	CollectionCount     int   `json:"collection_count"`
	VectorCount         int64 `json:"vector_count"`
	IndexCount          int   `json:"index_count"`
	AverageDimension    int   `json:"average_dimension"`

	// Cache metrics
	CacheHitRate        float64 `json:"cache_hit_rate"`
	CacheMissRate       float64 `json:"cache_miss_rate"`
	CacheEvictions      int64   `json:"cache_evictions"`

	// Error metrics
	ErrorCount          int64   `json:"error_count"`
	ErrorRate           float64 `json:"error_rate"`
	TimeoutCount        int64   `json:"timeout_count"`

	// Index metrics
	IndexBuildTime      time.Duration `json:"index_build_time"`
	IndexSize           int64         `json:"index_size"`
	IndexFragmentation  float64       `json:"index_fragmentation"`

	// Timestamp and uptime
	CollectedAt         time.Time     `json:"collected_at"`
	Uptime              time.Duration `json:"uptime"`
}

// VectorCollectionStats contains statistics for a specific collection
type VectorCollectionStats struct {
	Name            string    `json:"name"`
	VectorCount     int64     `json:"vector_count"`
	Dimension       int       `json:"dimension"`
	IndexType       string    `json:"index_type"`
	IndexStatus     string    `json:"index_status"`
	IndexProgress   float64   `json:"index_progress"`
	StorageSize     int64     `json:"storage_size"`
	MemoryUsage     int64     `json:"memory_usage"`
	SearchCount     int64     `json:"search_count"`
	InsertCount     int64     `json:"insert_count"`
	UpdateCount     int64     `json:"update_count"`
	DeleteCount     int64     `json:"delete_count"`
	AverageSearchTime time.Duration `json:"average_search_time"`
	LastAccess      time.Time `json:"last_access"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// VectorSimilarity represents vector similarity calculations
type VectorSimilarity struct {
	ID1        string  `json:"id1"`
	ID2        string  `json:"id2"`
	Similarity float32 `json:"similarity"`
	Distance   float32 `json:"distance"`
	MetricType string  `json:"metric_type"`
}

// VectorCluster represents a cluster of similar vectors
type VectorCluster struct {
	ID          string    `json:"id"`
	Centroid    []float32 `json:"centroid"`
	VectorIDs   []string  `json:"vector_ids"`
	Size        int       `json:"size"`
	Cohesion    float32   `json:"cohesion"`    // intra-cluster similarity
	Separation  float32   `json:"separation"`  // inter-cluster distance
	CreatedAt   time.Time `json:"created_at"`
}

// VectorAnalytics provides analytics capabilities for vector data
type VectorAnalytics interface {
	// Similarity analysis
	CalculateSimilarity(ctx context.Context, collection string, id1, id2 string, metricType string) (*VectorSimilarity, error)
	FindSimilarVectors(ctx context.Context, collection string, id string, threshold float32, limit int) ([]*VectorSimilarity, error)
	CalculatePairwiseSimilarity(ctx context.Context, collection string, ids []string, metricType string) ([]*VectorSimilarity, error)

	// Clustering analysis
	ClusterVectors(ctx context.Context, collection string, algorithm string, parameters map[string]interface{}) ([]*VectorCluster, error)
	GetCluster(ctx context.Context, collection string, clusterID string) (*VectorCluster, error)
	FindVectorCluster(ctx context.Context, collection string, id string) (*VectorCluster, error)

	// Dimensionality reduction
	ReduceDimensionality(ctx context.Context, collection string, targetDimension int, algorithm string) error
	VisualizationEmbedding(ctx context.Context, collection string, algorithm string) ([][]float32, error)

	// Quality analysis
	AnalyzeDistribution(ctx context.Context, collection string) (*VectorDistribution, error)
	DetectOutliers(ctx context.Context, collection string, threshold float32) ([]string, error)
	CalculateDiversity(ctx context.Context, collection string) (*VectorDiversity, error)
}

// VectorDistribution represents the distribution of vectors in a collection
type VectorDistribution struct {
	Mean               []float32 `json:"mean"`
	StandardDeviation  []float32 `json:"standard_deviation"`
	Variance           []float32 `json:"variance"`
	Skewness          []float32 `json:"skewness"`
	Kurtosis          []float32 `json:"kurtosis"`
	PercentileP25     []float32 `json:"percentile_p25"`
	PercentileP50     []float32 `json:"percentile_p50"`
	PercentileP75     []float32 `json:"percentile_p75"`
	PercentileP95     []float32 `json:"percentile_p95"`
	Range             []float32 `json:"range"`
}

// VectorDiversity represents diversity metrics for vector collections
type VectorDiversity struct {
	InterVectorDistance     float32   `json:"inter_vector_distance"`
	AverageDistance         float32   `json:"average_distance"`
	DistanceVariance        float32   `json:"distance_variance"`
	ClusterCount           int       `json:"cluster_count"`
	SilhouetteScore        float32   `json:"silhouette_score"`
	CalinhariIndex         float32   `json:"calinski_harabasz_index"`
	DaviesBouldinIndex     float32   `json:"davies_bouldin_index"`
	DimensionalEntropy     []float32 `json:"dimensional_entropy"`
	EffectiveDimension     int       `json:"effective_dimension"`
}

// VectorBackup represents a vector database backup
type VectorBackup struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Collections  []string               `json:"collections"`
	Type         string                 `json:"type"`         // full, incremental
	Status       string                 `json:"status"`       // pending, running, completed, failed
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time"`
	Duration     time.Duration          `json:"duration"`
	Size         int64                  `json:"size"`
	VectorCount  int64                  `json:"vector_count"`
	Location     string                 `json:"location"`
	Compression  string                 `json:"compression"`
	Encryption   bool                   `json:"encryption"`
	Checksum     string                 `json:"checksum"`
	Metadata     map[string]interface{} `json:"metadata"`
	ErrorMessage string                 `json:"error_message,omitempty"`
}

// VectorRestore represents a vector database restore operation
type VectorRestore struct {
	ID           string                 `json:"id"`
	BackupID     string                 `json:"backup_id"`
	Collections  []string               `json:"collections"`
	Status       string                 `json:"status"`       // pending, running, completed, failed
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time"`
	Duration     time.Duration          `json:"duration"`
	VectorCount  int64                  `json:"vector_count"`
	Destination  string                 `json:"destination"`
	Overwrite    bool                   `json:"overwrite"`
	Metadata     map[string]interface{} `json:"metadata"`
	ErrorMessage string                 `json:"error_message,omitempty"`
}