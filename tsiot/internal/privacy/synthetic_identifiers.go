package privacy

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

type SyntheticIdentifierConfig struct {
	Format             string                 `json:"format"` // uuid, sequential, custom
	Prefix             string                 `json:"prefix"`
	Suffix             string                 `json:"suffix"`
	Length             int                    `json:"length"`
	IncludeTimestamp   bool                   `json:"include_timestamp"`
	PreserveMappings   bool                   `json:"preserve_mappings"`
	CustomPattern      string                 `json:"custom_pattern"`
	ConsistencyGroups  map[string][]string    `json:"consistency_groups"`
	Metadata           map[string]interface{} `json:"metadata"`
}

type SyntheticIdentifierGenerator struct {
	config   *SyntheticIdentifierConfig
	logger   *logrus.Logger
	mappings sync.Map // Original ID -> Synthetic ID
	counter  int64
	mu       sync.Mutex
}

type IdentifierMapping struct {
	Original  string
	Synthetic string
	Timestamp time.Time
	Group     string
}

func NewSyntheticIdentifierGenerator(config *SyntheticIdentifierConfig, logger *logrus.Logger) *SyntheticIdentifierGenerator {
	if config == nil {
		config = getDefaultSyntheticIdentifierConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}

	return &SyntheticIdentifierGenerator{
		config:  config,
		logger:  logger,
		counter: 0,
	}
}

func (s *SyntheticIdentifierGenerator) GenerateIdentifiers(ctx context.Context, dataset []*models.TimeSeries) ([]*models.TimeSeries, error) {
	if len(dataset) == 0 {
		return dataset, nil
	}

	s.logger.WithFields(logrus.Fields{
		"dataset_size": len(dataset),
		"format":       s.config.Format,
	}).Info("Generating synthetic identifiers")

	// Process consistency groups first
	groupMappings := s.processConsistencyGroups(dataset)

	// Generate synthetic identifiers
	result := make([]*models.TimeSeries, len(dataset))
	for i, series := range dataset {
		syntheticSeries, err := s.generateForSeries(series, groupMappings)
		if err != nil {
			return nil, fmt.Errorf("failed to generate identifier for series %s: %w", series.ID, err)
		}
		result[i] = syntheticSeries
	}

	return result, nil
}

func (s *SyntheticIdentifierGenerator) processConsistencyGroups(dataset []*models.TimeSeries) map[string]string {
	groupMappings := make(map[string]string)

	// Process each consistency group
	for groupName, fields := range s.config.ConsistencyGroups {
		// Find all unique values for this group
		uniqueValues := make(map[string]bool)
		for _, series := range dataset {
			for _, field := range fields {
				value := s.getFieldValue(series, field)
				if value != "" {
					uniqueValues[value] = true
				}
			}
		}

		// Generate consistent synthetic values for each unique value
		for value := range uniqueValues {
			key := fmt.Sprintf("%s:%s", groupName, value)
			groupMappings[key] = s.generateConsistentIdentifier(groupName, value)
		}
	}

	return groupMappings
}

func (s *SyntheticIdentifierGenerator) generateForSeries(series *models.TimeSeries, groupMappings map[string]string) (*models.TimeSeries, error) {
	// Check if we already have a mapping
	if s.config.PreserveMappings {
		if synthetic, ok := s.mappings.Load(series.ID); ok {
			return s.applySyntheticID(series, synthetic.(string)), nil
		}
	}

	// Generate new synthetic identifier
	syntheticID := s.generateIdentifier(series.ID)

	// Store mapping
	if s.config.PreserveMappings {
		s.mappings.Store(series.ID, syntheticID)
	}

	// Apply synthetic identifier
	syntheticSeries := s.applySyntheticID(series, syntheticID)

	// Apply consistency group mappings
	s.applyGroupMappings(syntheticSeries, groupMappings)

	return syntheticSeries, nil
}

func (s *SyntheticIdentifierGenerator) generateIdentifier(originalID string) string {
	switch s.config.Format {
	case "uuid":
		return s.generateUUID()
	case "sequential":
		return s.generateSequential()
	case "custom":
		return s.generateCustom(originalID)
	default:
		return s.generateUUID()
	}
}

func (s *SyntheticIdentifierGenerator) generateUUID() string {
	// Generate UUID-like identifier
	b := make([]byte, 16)
	rand.Read(b)
	
	uuid := fmt.Sprintf("%x-%x-%x-%x-%x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])

	return s.formatIdentifier(uuid)
}

func (s *SyntheticIdentifierGenerator) generateSequential() string {
	s.mu.Lock()
	s.counter++
	count := s.counter
	s.mu.Unlock()

	sequential := fmt.Sprintf("%0*d", s.config.Length, count)
	return s.formatIdentifier(sequential)
}

func (s *SyntheticIdentifierGenerator) generateCustom(originalID string) string {
	if s.config.CustomPattern == "" {
		return s.generateUUID()
	}

	// Parse custom pattern
	// Supported tokens: {random}, {sequential}, {timestamp}, {original_hash}
	result := s.config.CustomPattern

	if strings.Contains(result, "{random}") {
		b := make([]byte, 8)
		rand.Read(b)
		random := base64.RawURLEncoding.EncodeToString(b)[:8]
		result = strings.ReplaceAll(result, "{random}", random)
	}

	if strings.Contains(result, "{sequential}") {
		s.mu.Lock()
		s.counter++
		count := s.counter
		s.mu.Unlock()
		result = strings.ReplaceAll(result, "{sequential}", fmt.Sprintf("%d", count))
	}

	if strings.Contains(result, "{timestamp}") {
		ts := time.Now().UnixNano()
		result = strings.ReplaceAll(result, "{timestamp}", fmt.Sprintf("%d", ts))
	}

	if strings.Contains(result, "{original_hash}") {
		hash := s.hashString(originalID)
		result = strings.ReplaceAll(result, "{original_hash}", hash[:8])
	}

	return s.formatIdentifier(result)
}

func (s *SyntheticIdentifierGenerator) formatIdentifier(base string) string {
	var parts []string

	if s.config.Prefix != "" {
		parts = append(parts, s.config.Prefix)
	}

	if s.config.IncludeTimestamp {
		ts := time.Now().Format("20060102")
		parts = append(parts, ts)
	}

	parts = append(parts, base)

	if s.config.Suffix != "" {
		parts = append(parts, s.config.Suffix)
	}

	return strings.Join(parts, "_")
}

func (s *SyntheticIdentifierGenerator) generateConsistentIdentifier(groupName, value string) string {
	// Generate a consistent identifier for a group value
	combined := fmt.Sprintf("%s:%s", groupName, value)
	hash := s.hashString(combined)
	
	// Use first 8 characters of hash
	return fmt.Sprintf("grp_%s", hash[:8])
}

func (s *SyntheticIdentifierGenerator) hashString(input string) string {
	// Simple hash function for demonstration
	// In production, use a proper hash function
	h := 0
	for _, c := range input {
		h = 31*h + int(c)
	}
	return fmt.Sprintf("%x", h)
}

func (s *SyntheticIdentifierGenerator) applySyntheticID(series *models.TimeSeries, syntheticID string) *models.TimeSeries {
	// Create a copy with synthetic identifier
	synthetic := &models.TimeSeries{
		ID:         syntheticID,
		Name:       s.anonymizeName(series.Name),
		Points:     series.Points,
		Metadata:   make(map[string]interface{}),
		Properties: make(map[string]interface{}),
		Tags:       s.anonymizeTags(series.Tags),
	}

	// Copy and anonymize metadata
	for k, v := range series.Metadata {
		if s.shouldAnonymizeField(k) {
			synthetic.Metadata[k] = s.anonymizeValue(k, v)
		} else {
			synthetic.Metadata[k] = v
		}
	}

	// Copy and anonymize properties
	for k, v := range series.Properties {
		if s.shouldAnonymizeField(k) {
			synthetic.Properties[k] = s.anonymizeValue(k, v)
		} else {
			synthetic.Properties[k] = v
		}
	}

	return synthetic
}

func (s *SyntheticIdentifierGenerator) applyGroupMappings(series *models.TimeSeries, groupMappings map[string]string) {
	// Apply consistency group mappings
	for groupName, fields := range s.config.ConsistencyGroups {
		for _, field := range fields {
			value := s.getFieldValue(series, field)
			if value != "" {
				key := fmt.Sprintf("%s:%s", groupName, value)
				if synthetic, ok := groupMappings[key]; ok {
					s.setFieldValue(series, field, synthetic)
				}
			}
		}
	}
}

func (s *SyntheticIdentifierGenerator) getFieldValue(series *models.TimeSeries, field string) string {
	// Check metadata
	if val, ok := series.Metadata[field]; ok {
		return fmt.Sprintf("%v", val)
	}

	// Check properties
	if val, ok := series.Properties[field]; ok {
		return fmt.Sprintf("%v", val)
	}

	// Check name
	if field == "name" {
		return series.Name
	}

	// Check tags
	for _, tag := range series.Tags {
		if tag == field {
			return tag
		}
	}

	return ""
}

func (s *SyntheticIdentifierGenerator) setFieldValue(series *models.TimeSeries, field, value string) {
	// Set in metadata by default
	if _, ok := series.Metadata[field]; ok {
		series.Metadata[field] = value
	} else if _, ok := series.Properties[field]; ok {
		series.Properties[field] = value
	} else if field == "name" {
		series.Name = value
	} else {
		// Add to metadata if not found
		series.Metadata[field] = value
	}
}

func (s *SyntheticIdentifierGenerator) shouldAnonymizeField(field string) bool {
	// Define fields that should be anonymized
	anonymizeFields := []string{
		"device_id", "user_id", "sensor_id", "location_id",
		"ip_address", "mac_address", "serial_number",
	}

	for _, af := range anonymizeFields {
		if strings.Contains(strings.ToLower(field), af) {
			return true
		}
	}

	return false
}

func (s *SyntheticIdentifierGenerator) anonymizeValue(field string, value interface{}) interface{} {
	// Generate synthetic value based on field type
	switch v := value.(type) {
	case string:
		if v == "" {
			return v
		}
		return s.generateSyntheticValue(field, v)
	default:
		return value
	}
}

func (s *SyntheticIdentifierGenerator) generateSyntheticValue(field, original string) string {
	// Check if we have a mapping
	if s.config.PreserveMappings {
		key := fmt.Sprintf("%s:%s", field, original)
		if synthetic, ok := s.mappings.Load(key); ok {
			return synthetic.(string)
		}
	}

	// Generate new synthetic value
	synthetic := fmt.Sprintf("syn_%s_%s", field, s.hashString(original)[:8])

	// Store mapping
	if s.config.PreserveMappings {
		key := fmt.Sprintf("%s:%s", field, original)
		s.mappings.Store(key, synthetic)
	}

	return synthetic
}

func (s *SyntheticIdentifierGenerator) anonymizeName(name string) string {
	if name == "" {
		return name
	}

	// Generate synthetic name
	return fmt.Sprintf("Series_%s", s.hashString(name)[:8])
}

func (s *SyntheticIdentifierGenerator) anonymizeTags(tags []string) []string {
	if len(tags) == 0 {
		return tags
	}

	anonymized := make([]string, len(tags))
	for i, tag := range tags {
		anonymized[i] = fmt.Sprintf("tag_%s", s.hashString(tag)[:6])
	}

	return anonymized
}

func (s *SyntheticIdentifierGenerator) GetMappings() []IdentifierMapping {
	var mappings []IdentifierMapping

	s.mappings.Range(func(k, v interface{}) bool {
		mapping := IdentifierMapping{
			Original:  k.(string),
			Synthetic: v.(string),
			Timestamp: time.Now(),
		}
		mappings = append(mappings, mapping)
		return true
	})

	return mappings
}

func (s *SyntheticIdentifierGenerator) ExportMappings() map[string]string {
	mappings := make(map[string]string)

	s.mappings.Range(func(k, v interface{}) bool {
		mappings[k.(string)] = v.(string)
		return true
	})

	return mappings
}

func (s *SyntheticIdentifierGenerator) ImportMappings(mappings map[string]string) {
	for k, v := range mappings {
		s.mappings.Store(k, v)
	}
}

func (s *SyntheticIdentifierGenerator) ClearMappings() {
	s.mappings = sync.Map{}
	s.mu.Lock()
	s.counter = 0
	s.mu.Unlock()
}

func getDefaultSyntheticIdentifierConfig() *SyntheticIdentifierConfig {
	return &SyntheticIdentifierConfig{
		Format:           "uuid",
		Prefix:           "syn",
		Suffix:           "",
		Length:           8,
		IncludeTimestamp: false,
		PreserveMappings: true,
		CustomPattern:    "",
		ConsistencyGroups: map[string][]string{
			"location": {"location_id", "site_id", "zone_id"},
			"device":   {"device_id", "device_type", "manufacturer"},
		},
		Metadata: make(map[string]interface{}),
	}
}