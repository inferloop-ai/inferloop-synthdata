package weaviate

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/auth"
	"github.com/weaviate/weaviate/entities/models"

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/internal/storage/interfaces"
)

// WeaviateConfig holds configuration for Weaviate
type WeaviateConfig struct {
	Host      string        `json:"host"`
	Scheme    string        `json:"scheme"`
	APIKey    string        `json:"api_key,omitempty"`
	Username  string        `json:"username,omitempty"`
	Password  string        `json:"password,omitempty"`
	Timeout   time.Duration `json:"timeout"`
	Headers   map[string]string `json:"headers,omitempty"`
}

// WeaviateStorage implements the VectorStorage interface for Weaviate
type WeaviateStorage struct {
	config  *WeaviateConfig
	client  *weaviate.Client
	logger  *logrus.Logger
	mu      sync.RWMutex
	metrics *storageMetrics
	closed  bool
}

type storageMetrics struct {
	searchOps    int64
	insertOps    int64
	updateOps    int64
	deleteOps    int64
	errorCount   int64
	startTime    time.Time
	mu           sync.RWMutex
}

// NewWeaviateStorage creates a new Weaviate storage instance
func NewWeaviateStorage(config *WeaviateConfig, logger *logrus.Logger) (*WeaviateStorage, error) {
	if config == nil {
		return nil, errors.NewStorageError("INVALID_CONFIG", "Weaviate config cannot be nil")
	}

	if config.Host == "" {
		return nil, errors.NewStorageError("INVALID_CONFIG", "Weaviate host is required")
	}

	if logger == nil {
		logger = logrus.New()
	}

	storage := &WeaviateStorage{
		config: config,
		logger: logger,
		metrics: &storageMetrics{
			startTime: time.Now(),
		},
	}

	return storage, nil
}

// Connect establishes connection to Weaviate
func (w *WeaviateStorage) Connect(ctx context.Context) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.client != nil {
		return nil // Already connected
	}

	cfg := weaviate.Config{
		Host:   w.config.Host,
		Scheme: w.config.Scheme,
	}

	if w.config.Timeout > 0 {
		cfg.ConnectionClient = &weaviate.ConnectionParams{
			Timeout: w.config.Timeout,
		}
	}

	if len(w.config.Headers) > 0 {
		cfg.Headers = w.config.Headers
	}

	// Set authentication
	if w.config.APIKey != "" {
		cfg.AuthConfig = auth.ApiKey{Value: w.config.APIKey}
	} else if w.config.Username != "" && w.config.Password != "" {
		cfg.AuthConfig = auth.UserPasswordCredentials{
			Username: w.config.Username,
			Password: w.config.Password,
		}
	}

	client, err := weaviate.NewClient(cfg)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "CONNECTION_FAILED", "Failed to create Weaviate client")
	}

	// Test connection
	isReady, err := client.Misc().ReadyChecker().Do(ctx)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "CONNECTION_FAILED", "Failed to connect to Weaviate")
	}

	if !isReady {
		return errors.NewStorageError("CONNECTION_FAILED", "Weaviate is not ready")
	}

	w.client = client

	w.logger.WithFields(logrus.Fields{
		"host":   w.config.Host,
		"scheme": w.config.Scheme,
	}).Info("Connected to Weaviate")

	return nil
}

// Close closes the Weaviate connection
func (w *WeaviateStorage) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return nil
	}

	w.client = nil
	w.closed = true

	w.logger.Info("Weaviate connection closed")
	return nil
}

// CreateCollection creates a new collection (class) in Weaviate
func (w *WeaviateStorage) CreateCollection(ctx context.Context, collection *interfaces.VectorCollection) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	// Convert to Weaviate class
	class := w.convertToWeaviateClass(collection)

	err := w.client.Schema().ClassCreator().WithClass(class).Do(ctx)
	if err != nil {
		w.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "CREATE_COLLECTION_FAILED", "Failed to create collection")
	}

	w.logger.WithField("collection", collection.Name).Info("Created Weaviate collection")
	return nil
}

// GetCollection retrieves a collection (class) from Weaviate
func (w *WeaviateStorage) GetCollection(ctx context.Context, name string) (*interfaces.VectorCollection, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	class, err := w.client.Schema().ClassGetter().WithClassName(name).Do(ctx)
	if err != nil {
		w.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "GET_COLLECTION_FAILED", "Failed to get collection")
	}

	return w.convertFromWeaviateClass(class), nil
}

// UpdateCollection updates a collection (class) in Weaviate
func (w *WeaviateStorage) UpdateCollection(ctx context.Context, collection *interfaces.VectorCollection) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	// Weaviate doesn't support direct class updates, so we need to handle property additions
	existingClass, err := w.client.Schema().ClassGetter().WithClassName(collection.Name).Do(ctx)
	if err != nil {
		w.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "UPDATE_COLLECTION_FAILED", "Failed to get existing collection")
	}

	// Add new properties if any
	newClass := w.convertToWeaviateClass(collection)
	for _, newProp := range newClass.Properties {
		exists := false
		for _, existingProp := range existingClass.Properties {
			if existingProp.Name == newProp.Name {
				exists = true
				break
			}
		}
		
		if !exists {
			err := w.client.Schema().PropertyCreator().
				WithClassName(collection.Name).
				WithProperty(newProp).
				Do(ctx)
			if err != nil {
				w.incrementErrorCount()
				return errors.WrapError(err, errors.ErrorTypeStorage, "UPDATE_COLLECTION_FAILED", 
					fmt.Sprintf("Failed to add property %s", newProp.Name))
			}
		}
	}

	w.logger.WithField("collection", collection.Name).Info("Updated Weaviate collection")
	return nil
}

// DeleteCollection deletes a collection (class) from Weaviate
func (w *WeaviateStorage) DeleteCollection(ctx context.Context, name string) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	err := w.client.Schema().ClassDeleter().WithClassName(name).Do(ctx)
	if err != nil {
		w.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_COLLECTION_FAILED", "Failed to delete collection")
	}

	w.logger.WithField("collection", name).Info("Deleted Weaviate collection")
	return nil
}

// ListCollections lists all collections (classes) in Weaviate
func (w *WeaviateStorage) ListCollections(ctx context.Context) ([]*interfaces.VectorCollection, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	schema, err := w.client.Schema().Getter().Do(ctx)
	if err != nil {
		w.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "LIST_COLLECTIONS_FAILED", "Failed to list collections")
	}

	var collections []*interfaces.VectorCollection
	if schema.Classes != nil {
		for _, class := range *schema.Classes {
			collections = append(collections, w.convertFromWeaviateClass(&class))
		}
	}

	return collections, nil
}

// Insert inserts vectors into a collection
func (w *WeaviateStorage) Insert(ctx context.Context, collection string, vectors []*interfaces.Vector) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	start := time.Now()
	defer func() {
		w.incrementInsertOps()
		w.logger.WithField("duration", time.Since(start)).Debug("Insert operation completed")
	}()

	batcher := w.client.Batch().ObjectsBatcher()

	for _, vector := range vectors {
		obj := w.convertToWeaviateObject(collection, vector)
		batcher = batcher.WithObjects(obj)
	}

	_, err := batcher.Do(ctx)
	if err != nil {
		w.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "INSERT_FAILED", "Failed to insert vectors")
	}

	return nil
}

// Update updates vectors in a collection
func (w *WeaviateStorage) Update(ctx context.Context, collection string, vectors []*interfaces.Vector) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	start := time.Now()
	defer func() {
		w.incrementUpdateOps()
		w.logger.WithField("duration", time.Since(start)).Debug("Update operation completed")
	}()

	for _, vector := range vectors {
		properties := make(map[string]interface{})
		for k, v := range vector.Metadata {
			properties[k] = v
		}
		if vector.Text != "" {
			properties["text"] = vector.Text
		}

		err := w.client.Data().Updater().
			WithClassName(collection).
			WithID(vector.ID).
			WithProperties(properties).
			WithVector(vector.Vector).
			Do(ctx)

		if err != nil {
			w.incrementErrorCount()
			return errors.WrapError(err, errors.ErrorTypeStorage, "UPDATE_FAILED", 
				fmt.Sprintf("Failed to update vector %s", vector.ID))
		}
	}

	return nil
}

// Delete deletes vectors from a collection
func (w *WeaviateStorage) Delete(ctx context.Context, collection string, ids []string) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	start := time.Now()
	defer func() {
		w.incrementDeleteOps()
		w.logger.WithField("duration", time.Since(start)).Debug("Delete operation completed")
	}()

	for _, id := range ids {
		err := w.client.Data().Deleter().
			WithClassName(collection).
			WithID(id).
			Do(ctx)

		if err != nil {
			w.incrementErrorCount()
			return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_FAILED", 
				fmt.Sprintf("Failed to delete vector %s", id))
		}
	}

	return nil
}

// Get retrieves vectors by IDs
func (w *WeaviateStorage) Get(ctx context.Context, collection string, ids []string) ([]*interfaces.Vector, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	var vectors []*interfaces.Vector

	for _, id := range ids {
		obj, err := w.client.Data().ObjectsGetter().
			WithClassName(collection).
			WithID(id).
			WithVector().
			Do(ctx)

		if err != nil {
			w.logger.WithError(err).Warnf("Failed to get vector %s", id)
			continue
		}

		if len(obj) > 0 {
			vector := w.convertFromWeaviateObject(&obj[0])
			vectors = append(vectors, vector)
		}
	}

	return vectors, nil
}

// Search performs vector search
func (w *WeaviateStorage) Search(ctx context.Context, collection string, query *interfaces.VectorQuery) (*interfaces.VectorSearchResult, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if w.closed || w.client == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Weaviate not connected")
	}

	start := time.Now()
	defer func() {
		w.incrementSearchOps()
		w.logger.WithField("duration", time.Since(start)).Debug("Search operation completed")
	}()

	builder := w.client.GraphQL().Get().
		WithClassName(collection).
		WithLimit(query.Limit)

	if query.Offset > 0 {
		builder = builder.WithOffset(query.Offset)
	}

	// Add vector search
	if len(query.Vector) > 0 {
		builder = builder.WithNearVector(w.client.GraphQL().NearVectorArgBuilder().
			WithVector(query.Vector))
	}

	// Add text search
	if query.Text != "" {
		builder = builder.WithNearText(w.client.GraphQL().NearTextArgBuilder().
			WithConcepts([]string{query.Text}))
	}

	// Add filters
	if len(query.Filters) > 0 {
		whereFilter := w.buildWhereFilter(query.Filters)
		if whereFilter != nil {
			builder = builder.WithWhere(whereFilter)
		}
	}

	// Add output fields
	fields := []string{"_additional { id, distance, vector }"}
	if len(query.OutputFields) > 0 {
		fields = append(fields, query.OutputFields...)
	} else {
		fields = append(fields, "text") // Default field
	}
	builder = builder.WithFields(fields...)

	result, err := builder.Do(ctx)
	if err != nil {
		w.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "SEARCH_FAILED", "Failed to search vectors")
	}

	return w.convertSearchResult(result, time.Since(start)), nil
}

// SearchByVector performs vector similarity search
func (w *WeaviateStorage) SearchByVector(ctx context.Context, collection string, vector []float32, limit int, filters map[string]interface{}) (*interfaces.VectorSearchResult, error) {
	query := &interfaces.VectorQuery{
		Vector:  vector,
		Limit:   limit,
		Filters: filters,
	}
	return w.Search(ctx, collection, query)
}

// SearchByText performs text-based search
func (w *WeaviateStorage) SearchByText(ctx context.Context, collection string, text string, limit int, filters map[string]interface{}) (*interfaces.VectorSearchResult, error) {
	query := &interfaces.VectorQuery{
		Text:    text,
		Limit:   limit,
		Filters: filters,
	}
	return w.Search(ctx, collection, query)
}

// SearchSimilar finds similar vectors to a given vector ID
func (w *WeaviateStorage) SearchSimilar(ctx context.Context, collection string, id string, limit int, filters map[string]interface{}) (*interfaces.VectorSearchResult, error) {
	query := &interfaces.VectorQuery{
		ID:      id,
		Limit:   limit,
		Filters: filters,
	}
	return w.Search(ctx, collection, query)
}

// BatchInsert inserts vectors in batches
func (w *WeaviateStorage) BatchInsert(ctx context.Context, collection string, vectors []*interfaces.Vector, batchSize int) error {
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}
		
		batch := vectors[i:end]
		if err := w.Insert(ctx, collection, batch); err != nil {
			return err
		}
	}
	return nil
}

// BatchUpdate updates vectors in batches
func (w *WeaviateStorage) BatchUpdate(ctx context.Context, collection string, vectors []*interfaces.Vector, batchSize int) error {
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}
		
		batch := vectors[i:end]
		if err := w.Update(ctx, collection, batch); err != nil {
			return err
		}
	}
	return nil
}

// BatchDelete deletes vectors in batches
func (w *WeaviateStorage) BatchDelete(ctx context.Context, collection string, ids []string, batchSize int) error {
	for i := 0; i < len(ids); i += batchSize {
		end := i + batchSize
		if end > len(ids) {
			end = len(ids)
		}
		
		batch := ids[i:end]
		if err := w.Delete(ctx, collection, batch); err != nil {
			return err
		}
	}
	return nil
}

// CreateIndex creates an index (not directly applicable to Weaviate)
func (w *WeaviateStorage) CreateIndex(ctx context.Context, collection string, index *interfaces.VectorIndex) error {
	// Weaviate handles indexing automatically
	w.logger.Info("Weaviate handles indexing automatically")
	return nil
}

// DeleteIndex deletes an index (not directly applicable to Weaviate)
func (w *WeaviateStorage) DeleteIndex(ctx context.Context, collection string, indexName string) error {
	// Weaviate handles indexing automatically
	w.logger.Info("Weaviate handles indexing automatically")
	return nil
}

// ListIndexes lists indexes (not directly applicable to Weaviate)
func (w *WeaviateStorage) ListIndexes(ctx context.Context, collection string) ([]*interfaces.VectorIndex, error) {
	// Weaviate handles indexing automatically
	return []*interfaces.VectorIndex{}, nil
}

// RebuildIndex rebuilds an index (not directly applicable to Weaviate)
func (w *WeaviateStorage) RebuildIndex(ctx context.Context, collection string, indexName string) error {
	// Weaviate handles indexing automatically
	w.logger.Info("Weaviate handles indexing automatically")
	return nil
}

// Health returns the health status of Weaviate
func (w *WeaviateStorage) Health(ctx context.Context) (*interfaces.VectorStorageHealth, error) {
	start := time.Now()
	status := "healthy"
	var healthErrors []string

	if w.closed || w.client == nil {
		status = "unhealthy"
		healthErrors = append(healthErrors, "Not connected to Weaviate")
	} else {
		isReady, err := w.client.Misc().ReadyChecker().Do(ctx)
		if err != nil || !isReady {
			status = "unhealthy"
			if err != nil {
				healthErrors = append(healthErrors, err.Error())
			} else {
				healthErrors = append(healthErrors, "Weaviate is not ready")
			}
		}
	}

	return &interfaces.VectorStorageHealth{
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Errors:       healthErrors,
	}, nil
}

// GetMetrics returns storage metrics
func (w *WeaviateStorage) GetMetrics(ctx context.Context) (*interfaces.VectorStorageMetrics, error) {
	w.metrics.mu.RLock()
	defer w.metrics.mu.RUnlock()

	return &interfaces.VectorStorageMetrics{
		SearchOperations: w.metrics.searchOps,
		InsertOperations: w.metrics.insertOps,
		UpdateOperations: w.metrics.updateOps,
		DeleteOperations: w.metrics.deleteOps,
		ErrorCount:       w.metrics.errorCount,
		Uptime:           time.Since(w.metrics.startTime),
		CollectedAt:      time.Now(),
	}, nil
}

// GetCollectionStats returns statistics for a specific collection
func (w *WeaviateStorage) GetCollectionStats(ctx context.Context, collection string) (*interfaces.VectorCollectionStats, error) {
	// This would require custom implementation to gather stats from Weaviate
	return &interfaces.VectorCollectionStats{
		Name:      collection,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}, nil
}

// Helper methods

func (w *WeaviateStorage) convertToWeaviateClass(collection *interfaces.VectorCollection) *models.Class {
	class := &models.Class{
		Class:       collection.Name,
		Description: collection.Description,
		Properties:  []*models.Property{},
	}

	if collection.Schema != nil {
		// Add default text property
		textProp := &models.Property{
			Name:     "text",
			DataType: []string{"text"},
		}
		class.Properties = append(class.Properties, textProp)

		// Add metadata properties
		for _, prop := range collection.Schema.Properties {
			weaviateProp := &models.Property{
				Name:        prop.Name,
				Description: prop.Description,
			}

			switch prop.Type {
			case "string":
				weaviateProp.DataType = []string{"string"}
			case "int64":
				weaviateProp.DataType = []string{"int"}
			case "float64":
				weaviateProp.DataType = []string{"number"}
			case "bool":
				weaviateProp.DataType = []string{"boolean"}
			default:
				weaviateProp.DataType = []string{"string"}
			}

			class.Properties = append(class.Properties, weaviateProp)
		}
	}

	return class
}

func (w *WeaviateStorage) convertFromWeaviateClass(class *models.Class) *interfaces.VectorCollection {
	collection := &interfaces.VectorCollection{
		Name:        class.Class,
		Description: class.Description,
		Schema: &interfaces.VectorSchema{
			Properties: []*interfaces.VectorProperty{},
		},
	}

	if class.Properties != nil {
		for _, prop := range class.Properties {
			if len(prop.DataType) > 0 {
				property := &interfaces.VectorProperty{
					Name:        prop.Name,
					Description: prop.Description,
				}

				switch prop.DataType[0] {
				case "string", "text":
					property.Type = "string"
				case "int":
					property.Type = "int64"
				case "number":
					property.Type = "float64"
				case "boolean":
					property.Type = "bool"
				default:
					property.Type = "string"
				}

				collection.Schema.Properties = append(collection.Schema.Properties, property)
			}
		}
	}

	return collection
}

func (w *WeaviateStorage) convertToWeaviateObject(collection string, vector *interfaces.Vector) *models.Object {
	properties := make(map[string]interface{})
	
	// Add metadata as properties
	for k, v := range vector.Metadata {
		properties[k] = v
	}
	
	// Add text if present
	if vector.Text != "" {
		properties["text"] = vector.Text
	}

	obj := &models.Object{
		Class:      collection,
		Properties: properties,
		Vector:     vector.Vector,
	}

	// Set ID if provided, otherwise generate one
	if vector.ID != "" {
		obj.ID = vector.ID
	} else {
		obj.ID = uuid.New().String()
	}

	return obj
}

func (w *WeaviateStorage) convertFromWeaviateObject(obj *models.Object) *interfaces.Vector {
	vector := &interfaces.Vector{
		ID:       obj.ID.String(),
		Vector:   obj.Vector,
		Metadata: make(map[string]interface{}),
	}

	if obj.Properties != nil {
		for k, v := range obj.Properties.(map[string]interface{}) {
			if k == "text" {
				if text, ok := v.(string); ok {
					vector.Text = text
				}
			} else {
				vector.Metadata[k] = v
			}
		}
	}

	return vector
}

func (w *WeaviateStorage) buildWhereFilter(filters map[string]interface{}) *models.WhereFilter {
	if len(filters) == 0 {
		return nil
	}

	var operands []*models.WhereFilter
	for key, value := range filters {
		operand := &models.WhereFilter{
			Path:     []string{key},
			Operator: "Equal",
			ValueText: fmt.Sprintf("%v", value),
		}
		operands = append(operands, operand)
	}

	if len(operands) == 1 {
		return operands[0]
	}

	// Combine multiple filters with AND
	filter := &models.WhereFilter{
		Operator: "And",
		Operands: operands,
	}

	return filter
}

func (w *WeaviateStorage) convertSearchResult(result *models.GraphQLResponse, searchTime time.Duration) *interfaces.VectorSearchResult {
	searchResult := &interfaces.VectorSearchResult{
		Vectors:    []*interfaces.VectorMatch{},
		SearchTime: searchTime,
	}

	if result.Data == nil {
		return searchResult
	}

	// Parse GraphQL response - this is simplified
	// In practice, you'd need to parse the nested JSON structure
	data := result.Data.(map[string]interface{})
	for className, classData := range data {
		if strings.HasPrefix(className, "Get") {
			objects := classData.([]interface{})
			for _, obj := range objects {
				objMap := obj.(map[string]interface{})
				
				match := &interfaces.VectorMatch{
					Metadata: make(map[string]interface{}),
				}

				// Extract additional information
				if additional, ok := objMap["_additional"].(map[string]interface{}); ok {
					if id, ok := additional["id"].(string); ok {
						match.ID = id
					}
					if distance, ok := additional["distance"].(float64); ok {
						match.Distance = float32(distance)
					}
					if vector, ok := additional["vector"].([]interface{}); ok {
						match.Vector = make([]float32, len(vector))
						for i, v := range vector {
							if f, ok := v.(float64); ok {
								match.Vector[i] = float32(f)
							}
						}
					}
				}

				// Extract other properties
				for k, v := range objMap {
					if k != "_additional" {
						if k == "text" {
							if text, ok := v.(string); ok {
								match.Text = text
							}
						} else {
							match.Metadata[k] = v
						}
					}
				}

				searchResult.Vectors = append(searchResult.Vectors, match)
			}
		}
	}

	searchResult.Total = int64(len(searchResult.Vectors))
	return searchResult
}

func (w *WeaviateStorage) incrementSearchOps() {
	w.metrics.mu.Lock()
	w.metrics.searchOps++
	w.metrics.mu.Unlock()
}

func (w *WeaviateStorage) incrementInsertOps() {
	w.metrics.mu.Lock()
	w.metrics.insertOps++
	w.metrics.mu.Unlock()
}

func (w *WeaviateStorage) incrementUpdateOps() {
	w.metrics.mu.Lock()
	w.metrics.updateOps++
	w.metrics.mu.Unlock()
}

func (w *WeaviateStorage) incrementDeleteOps() {
	w.metrics.mu.Lock()
	w.metrics.deleteOps++
	w.metrics.mu.Unlock()
}

func (w *WeaviateStorage) incrementErrorCount() {
	w.metrics.mu.Lock()
	w.metrics.errorCount++
	w.metrics.mu.Unlock()
}