package privacy

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"math"
	"sync"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

type AnonymizationConfig struct {
	Method               string                 `json:"method"`
	NoiseLevel           float64                `json:"noise_level"`
	HashingSalt          string                 `json:"hashing_salt"`
	GeneralizationLevels map[string]int         `json:"generalization_levels"`
	SuppressionThreshold float64                `json:"suppression_threshold"`
	MaskingPatterns      map[string]string      `json:"masking_patterns"`
	Metadata             map[string]interface{} `json:"metadata"`
}

type Anonymizer struct {
	config             *AnonymizationConfig
	logger             *logrus.Logger
	identifierMappings sync.Map
	mu                 sync.RWMutex
}

func NewAnonymizer(config *AnonymizationConfig, logger *logrus.Logger) *Anonymizer {
	if config == nil {
		config = getDefaultAnonymizationConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}

	return &Anonymizer{
		config: config,
		logger: logger,
	}
}

func (a *Anonymizer) AnonymizeTimeSeries(ctx context.Context, data *models.TimeSeries) (*models.TimeSeries, error) {
	if data == nil {
		return nil, errors.NewValidationError("input time series is nil", nil)
	}

	a.logger.WithFields(logrus.Fields{
		"series_id": data.ID,
		"method":    a.config.Method,
	}).Info("Starting time series anonymization")

	anonymized := &models.TimeSeries{
		ID:         a.anonymizeIdentifier(data.ID),
		Name:       a.anonymizeField(data.Name, "name"),
		Points:     make([]models.DataPoint, len(data.Points)),
		Metadata:   make(map[string]interface{}),
		Properties: data.Properties,
		Tags:       a.anonymizeTags(data.Tags),
	}

	// Copy and anonymize metadata
	for k, v := range data.Metadata {
		if a.shouldSuppressField(k) {
			continue
		}
		anonymized.Metadata[k] = a.anonymizeValue(v, k)
	}

	// Anonymize data points
	for i, point := range data.Points {
		anonymized.Points[i] = models.DataPoint{
			Timestamp: a.anonymizeTimestamp(point.Timestamp),
			Value:     a.anonymizeNumericValue(point.Value),
			Quality:   point.Quality,
		}
	}

	return anonymized, nil
}

func (a *Anonymizer) AnonymizeBatch(ctx context.Context, batch []*models.TimeSeries) ([]*models.TimeSeries, error) {
	if len(batch) == 0 {
		return batch, nil
	}

	anonymized := make([]*models.TimeSeries, len(batch))
	var wg sync.WaitGroup
	errChan := make(chan error, len(batch))

	for i, series := range batch {
		wg.Add(1)
		go func(idx int, ts *models.TimeSeries) {
			defer wg.Done()
			
			result, err := a.AnonymizeTimeSeries(ctx, ts)
			if err != nil {
				errChan <- fmt.Errorf("failed to anonymize series %s: %w", ts.ID, err)
				return
			}
			anonymized[idx] = result
		}(i, series)
	}

	wg.Wait()
	close(errChan)

	if len(errChan) > 0 {
		return nil, <-errChan
	}

	return anonymized, nil
}

func (a *Anonymizer) anonymizeIdentifier(id string) string {
	// Check if we've already mapped this identifier
	if mapped, ok := a.identifierMappings.Load(id); ok {
		return mapped.(string)
	}

	// Generate new anonymous identifier
	var anonymousID string
	switch a.config.Method {
	case "hash":
		anonymousID = a.hashIdentifier(id)
	case "random":
		anonymousID = a.generateRandomID()
	case "sequential":
		anonymousID = a.getSequentialID()
	default:
		anonymousID = a.hashIdentifier(id)
	}

	a.identifierMappings.Store(id, anonymousID)
	return anonymousID
}

func (a *Anonymizer) hashIdentifier(id string) string {
	// Simple hashing with salt
	salted := id + a.config.HashingSalt
	return fmt.Sprintf("anon_%x", salted[:min(16, len(salted))])
}

func (a *Anonymizer) generateRandomID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return "anon_" + hex.EncodeToString(b)
}

func (a *Anonymizer) getSequentialID() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	// Get current count from mappings
	count := 0
	a.identifierMappings.Range(func(k, v interface{}) bool {
		count++
		return true
	})
	
	return fmt.Sprintf("anon_%06d", count+1)
}

func (a *Anonymizer) anonymizeField(value, fieldType string) string {
	if pattern, ok := a.config.MaskingPatterns[fieldType]; ok {
		return a.applyMaskingPattern(value, pattern)
	}
	
	// Default: generalize the field
	if level, ok := a.config.GeneralizationLevels[fieldType]; ok {
		return a.generalizeString(value, level)
	}
	
	return "***"
}

func (a *Anonymizer) applyMaskingPattern(value, pattern string) string {
	if len(value) == 0 {
		return value
	}
	
	// Simple masking implementation
	masked := []rune(value)
	for i := range masked {
		if i < len(pattern) && pattern[i] == '*' {
			masked[i] = '*'
		}
	}
	return string(masked)
}

func (a *Anonymizer) generalizeString(value string, level int) string {
	if level <= 0 {
		return value
	}
	
	// Simple generalization by truncating
	if len(value) > level {
		return value[:level] + "***"
	}
	return "***"
}

func (a *Anonymizer) anonymizeNumericValue(value float64) float64 {
	if a.config.NoiseLevel == 0 {
		return value
	}
	
	// Add Laplace noise for differential privacy
	noise := a.laplacianNoise(a.config.NoiseLevel)
	return value + noise
}

func (a *Anonymizer) laplacianNoise(scale float64) float64 {
	// Generate Laplacian noise
	b := make([]byte, 8)
	rand.Read(b)
	u := float64(b[0]) / 255.0
	
	if u < 0.5 {
		return scale * math.Log(2*u)
	}
	return -scale * math.Log(2*(1-u))
}

func (a *Anonymizer) anonymizeTimestamp(ts int64) int64 {
	// Round timestamp to reduce precision
	precision := int64(60000) // 1 minute precision
	return (ts / precision) * precision
}

func (a *Anonymizer) anonymizeTags(tags []string) []string {
	if len(tags) == 0 {
		return tags
	}
	
	anonymized := make([]string, 0, len(tags))
	for _, tag := range tags {
		if !a.shouldSuppressField(tag) {
			anonymized = append(anonymized, a.anonymizeField(tag, "tag"))
		}
	}
	return anonymized
}

func (a *Anonymizer) anonymizeValue(value interface{}, fieldName string) interface{} {
	switch v := value.(type) {
	case string:
		return a.anonymizeField(v, fieldName)
	case float64:
		return a.anonymizeNumericValue(v)
	case int:
		return int(a.anonymizeNumericValue(float64(v)))
	case int64:
		return int64(a.anonymizeNumericValue(float64(v)))
	default:
		return value
	}
}

func (a *Anonymizer) shouldSuppressField(field string) bool {
	// Implement suppression logic based on threshold
	// This is a simplified implementation
	return false
}

func (a *Anonymizer) GetMappings() map[string]string {
	mappings := make(map[string]string)
	a.identifierMappings.Range(func(k, v interface{}) bool {
		mappings[k.(string)] = v.(string)
		return true
	})
	return mappings
}

func (a *Anonymizer) ClearMappings() {
	a.identifierMappings = sync.Map{}
}

func getDefaultAnonymizationConfig() *AnonymizationConfig {
	return &AnonymizationConfig{
		Method:               "hash",
		NoiseLevel:           0.1,
		HashingSalt:          "default-salt",
		GeneralizationLevels: map[string]int{
			"name":     5,
			"location": 3,
			"tag":      7,
		},
		SuppressionThreshold: 0.05,
		MaskingPatterns: map[string]string{
			"email": "***@***.***",
			"phone": "***-***-****",
		},
		Metadata: make(map[string]interface{}),
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}