package privacy

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

type TClosenessConfig struct {
	T                   float64                `json:"t"`
	DistanceMetric      string                 `json:"distance_metric"` // EMD, KL
	QuasiIdentifiers    []string               `json:"quasi_identifiers"`
	SensitiveAttributes []string               `json:"sensitive_attributes"`
	NumericBins         int                    `json:"numeric_bins"`
	Metadata            map[string]interface{} `json:"metadata"`
}

type TClosenessProcessor struct {
	config                  *TClosenessConfig
	logger                  *logrus.Logger
	globalDistributions     map[string]Distribution
	mu                      sync.RWMutex
}

type Distribution struct {
	Values      []interface{}
	Frequencies map[interface{}]float64
	Total       int
	IsNumeric   bool
}

type ClosenessClass struct {
	Records          []*models.TimeSeries
	QuasiIdentifiers map[string]string
	Distributions    map[string]Distribution
}

func NewTClosenessProcessor(config *TClosenessConfig, logger *logrus.Logger) *TClosenessProcessor {
	if config == nil {
		config = getDefaultTClosenessConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}

	return &TClosenessProcessor{
		config:              config,
		logger:              logger,
		globalDistributions: make(map[string]Distribution),
	}
}

func (t *TClosenessProcessor) ApplyTCloseness(ctx context.Context, dataset []*models.TimeSeries) ([]*models.TimeSeries, error) {
	if len(dataset) == 0 {
		return dataset, nil
	}

	t.logger.WithFields(logrus.Fields{
		"dataset_size": len(dataset),
		"t_value":      t.config.T,
		"metric":       t.config.DistanceMetric,
	}).Info("Applying t-closeness")

	// Step 1: Calculate global distributions for sensitive attributes
	t.calculateGlobalDistributions(dataset)

	// Step 2: Group by quasi-identifiers
	groups := t.groupByQuasiIdentifiers(dataset)

	// Step 3: Check and enforce t-closeness
	closeGroups := t.enforceTCloseness(groups)

	// Step 4: Extract records from t-close groups
	result := t.extractRecords(closeGroups)

	return result, nil
}

func (t *TClosenessProcessor) calculateGlobalDistributions(dataset []*models.TimeSeries) {
	t.mu.Lock()
	defer t.mu.Unlock()

	for _, attr := range t.config.SensitiveAttributes {
		dist := Distribution{
			Frequencies: make(map[interface{}]float64),
			IsNumeric:   false,
		}

		// Collect all values for this attribute
		var values []interface{}
		for _, series := range dataset {
			value := t.getAttributeValue(series, attr)
			if value != nil {
				values = append(values, value)
				dist.Total++
			}
		}

		// Check if numeric
		if len(values) > 0 {
			_, isFloat := values[0].(float64)
			_, isInt := values[0].(int)
			dist.IsNumeric = isFloat || isInt
		}

		// Calculate frequencies
		if dist.IsNumeric && t.config.NumericBins > 0 {
			// Bin numeric values
			dist = t.binNumericValues(values, t.config.NumericBins)
		} else {
			// Calculate frequencies for categorical values
			for _, v := range values {
				dist.Frequencies[v]++
			}
			// Normalize
			for k := range dist.Frequencies {
				dist.Frequencies[k] /= float64(dist.Total)
			}
		}

		dist.Values = t.getSortedKeys(dist.Frequencies)
		t.globalDistributions[attr] = dist
	}
}

func (t *TClosenessProcessor) binNumericValues(values []interface{}, numBins int) Distribution {
	// Convert to float64
	floatValues := make([]float64, 0, len(values))
	for _, v := range values {
		switch val := v.(type) {
		case float64:
			floatValues = append(floatValues, val)
		case int:
			floatValues = append(floatValues, float64(val))
		}
	}

	if len(floatValues) == 0 {
		return Distribution{
			Frequencies: make(map[interface{}]float64),
			Total:       0,
			IsNumeric:   true,
		}
	}

	// Find min and max
	min, max := floatValues[0], floatValues[0]
	for _, v := range floatValues {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	// Create bins
	binWidth := (max - min) / float64(numBins)
	dist := Distribution{
		Frequencies: make(map[interface{}]float64),
		Total:       len(floatValues),
		IsNumeric:   true,
	}

	// Assign values to bins
	for _, v := range floatValues {
		binIndex := int((v - min) / binWidth)
		if binIndex >= numBins {
			binIndex = numBins - 1
		}
		binKey := fmt.Sprintf("bin_%d", binIndex)
		dist.Frequencies[binKey]++
	}

	// Normalize
	for k := range dist.Frequencies {
		dist.Frequencies[k] /= float64(dist.Total)
	}

	return dist
}

func (t *TClosenessProcessor) groupByQuasiIdentifiers(dataset []*models.TimeSeries) map[string]*ClosenessClass {
	groups := make(map[string]*ClosenessClass)

	for _, series := range dataset {
		groupKey := t.createGroupKey(series)

		if group, exists := groups[groupKey]; exists {
			group.Records = append(group.Records, series)
			t.updateGroupDistribution(group, series)
		} else {
			newGroup := &ClosenessClass{
				Records:          []*models.TimeSeries{series},
				QuasiIdentifiers: t.extractQuasiIdentifiers(series),
				Distributions:    make(map[string]Distribution),
			}
			t.updateGroupDistribution(newGroup, series)
			groups[groupKey] = newGroup
		}
	}

	// Finalize distributions
	for _, group := range groups {
		for attr, dist := range group.Distributions {
			// Normalize frequencies
			for k := range dist.Frequencies {
				dist.Frequencies[k] /= float64(dist.Total)
			}
			dist.Values = t.getSortedKeys(dist.Frequencies)
			group.Distributions[attr] = dist
		}
	}

	return groups
}

func (t *TClosenessProcessor) createGroupKey(series *models.TimeSeries) string {
	values := make([]string, 0, len(t.config.QuasiIdentifiers))

	for _, qi := range t.config.QuasiIdentifiers {
		value := t.getAttributeValue(series, qi)
		values = append(values, fmt.Sprintf("%v", value))
	}

	return fmt.Sprintf("%v", values)
}

func (t *TClosenessProcessor) extractQuasiIdentifiers(series *models.TimeSeries) map[string]string {
	identifiers := make(map[string]string)

	for _, qi := range t.config.QuasiIdentifiers {
		value := t.getAttributeValue(series, qi)
		identifiers[qi] = fmt.Sprintf("%v", value)
	}

	return identifiers
}

func (t *TClosenessProcessor) updateGroupDistribution(group *ClosenessClass, series *models.TimeSeries) {
	for _, attr := range t.config.SensitiveAttributes {
		value := t.getAttributeValue(series, attr)
		if value == nil {
			continue
		}

		if _, exists := group.Distributions[attr]; !exists {
			group.Distributions[attr] = Distribution{
				Frequencies: make(map[interface{}]float64),
				IsNumeric:   t.globalDistributions[attr].IsNumeric,
			}
		}

		dist := group.Distributions[attr]
		dist.Total++
		dist.Frequencies[value]++
		group.Distributions[attr] = dist
	}
}

func (t *TClosenessProcessor) getAttributeValue(series *models.TimeSeries, attribute string) interface{} {
	if val, ok := series.Metadata[attribute]; ok {
		return val
	}

	if val, ok := series.Properties[attribute]; ok {
		return val
	}

	for _, tag := range series.Tags {
		if tag == attribute {
			return attribute
		}
	}

	return nil
}

func (t *TClosenessProcessor) enforceTCloseness(groups map[string]*ClosenessClass) []*ClosenessClass {
	var closeGroups []*ClosenessClass

	for _, group := range groups {
		if t.checkTCloseness(group) {
			closeGroups = append(closeGroups, group)
		} else {
			// Try to merge with other groups
			merged := t.mergeForCloseness(group, groups)
			if merged != nil && t.checkTCloseness(merged) {
				closeGroups = append(closeGroups, merged)
			}
		}
	}

	return closeGroups
}

func (t *TClosenessProcessor) checkTCloseness(group *ClosenessClass) bool {
	for _, attr := range t.config.SensitiveAttributes {
		if !t.checkAttributeCloseness(group, attr) {
			return false
		}
	}
	return true
}

func (t *TClosenessProcessor) checkAttributeCloseness(group *ClosenessClass, attribute string) bool {
	groupDist, ok := group.Distributions[attribute]
	if !ok || groupDist.Total == 0 {
		return true // No data for this attribute
	}

	globalDist := t.globalDistributions[attribute]
	
	var distance float64
	switch t.config.DistanceMetric {
	case "EMD":
		distance = t.earthMoversDistance(groupDist, globalDist)
	case "KL":
		distance = t.kullbackLeiblerDivergence(groupDist, globalDist)
	default:
		distance = t.earthMoversDistance(groupDist, globalDist)
	}

	return distance <= t.config.T
}

func (t *TClosenessProcessor) earthMoversDistance(dist1, dist2 Distribution) float64 {
	// Simplified EMD for categorical data
	// For numeric data, this assumes ordered bins
	
	// Get all unique values
	allValues := make(map[interface{}]bool)
	for v := range dist1.Frequencies {
		allValues[v] = true
	}
	for v := range dist2.Frequencies {
		allValues[v] = true
	}

	// Calculate cumulative distributions
	cumDist1 := 0.0
	cumDist2 := 0.0
	distance := 0.0

	sortedValues := t.getSortedKeys(allValues)
	for _, v := range sortedValues {
		cumDist1 += dist1.Frequencies[v]
		cumDist2 += dist2.Frequencies[v]
		distance += math.Abs(cumDist1 - cumDist2)
	}

	return distance / float64(len(sortedValues))
}

func (t *TClosenessProcessor) kullbackLeiblerDivergence(dist1, dist2 Distribution) float64 {
	// KL divergence from dist1 to dist2
	divergence := 0.0
	
	for v, p := range dist1.Frequencies {
		q, exists := dist2.Frequencies[v]
		if !exists || q == 0 {
			// Handle zero probability in reference distribution
			continue
		}
		if p > 0 {
			divergence += p * math.Log(p/q)
		}
	}

	return divergence
}

func (t *TClosenessProcessor) mergeForCloseness(group *ClosenessClass, allGroups map[string]*ClosenessClass) *ClosenessClass {
	// Find the best group to merge with
	var bestMerge *ClosenessClass
	bestDistance := math.MaxFloat64

	for _, other := range allGroups {
		if other == group {
			continue
		}

		// Calculate distance after merge
		merged := t.mergeGroups(group, other)
		avgDistance := 0.0
		count := 0

		for _, attr := range t.config.SensitiveAttributes {
			if groupDist, ok := merged.Distributions[attr]; ok && groupDist.Total > 0 {
				globalDist := t.globalDistributions[attr]
				distance := t.earthMoversDistance(groupDist, globalDist)
				avgDistance += distance
				count++
			}
		}

		if count > 0 {
			avgDistance /= float64(count)
			if avgDistance < bestDistance && avgDistance <= t.config.T {
				bestDistance = avgDistance
				bestMerge = merged
			}
		}
	}

	return bestMerge
}

func (t *TClosenessProcessor) mergeGroups(group1, group2 *ClosenessClass) *ClosenessClass {
	merged := &ClosenessClass{
		Records:          append(group1.Records, group2.Records...),
		QuasiIdentifiers: make(map[string]string),
		Distributions:    make(map[string]Distribution),
	}

	// Merge quasi-identifiers
	for qi := range group1.QuasiIdentifiers {
		if group1.QuasiIdentifiers[qi] == group2.QuasiIdentifiers[qi] {
			merged.QuasiIdentifiers[qi] = group1.QuasiIdentifiers[qi]
		} else {
			merged.QuasiIdentifiers[qi] = "*"
		}
	}

	// Recalculate distributions
	for _, record := range merged.Records {
		for _, attr := range t.config.SensitiveAttributes {
			value := t.getAttributeValue(record, attr)
			if value == nil {
				continue
			}

			if _, exists := merged.Distributions[attr]; !exists {
				merged.Distributions[attr] = Distribution{
					Frequencies: make(map[interface{}]float64),
					IsNumeric:   t.globalDistributions[attr].IsNumeric,
				}
			}

			dist := merged.Distributions[attr]
			dist.Total++
			dist.Frequencies[value]++
			merged.Distributions[attr] = dist
		}
	}

	// Normalize
	for attr, dist := range merged.Distributions {
		for k := range dist.Frequencies {
			dist.Frequencies[k] /= float64(dist.Total)
		}
		dist.Values = t.getSortedKeys(dist.Frequencies)
		merged.Distributions[attr] = dist
	}

	return merged
}

func (t *TClosenessProcessor) extractRecords(groups []*ClosenessClass) []*models.TimeSeries {
	var result []*models.TimeSeries

	for _, group := range groups {
		for _, record := range group.Records {
			anonymized := t.applyQuasiIdentifiers(record, group.QuasiIdentifiers)
			result = append(result, anonymized)
		}
	}

	return result
}

func (t *TClosenessProcessor) applyQuasiIdentifiers(series *models.TimeSeries, identifiers map[string]string) *models.TimeSeries {
	anonymized := &models.TimeSeries{
		ID:         series.ID,
		Name:       series.Name,
		Points:     series.Points,
		Metadata:   make(map[string]interface{}),
		Properties: make(map[string]interface{}),
		Tags:       series.Tags,
	}

	// Copy all metadata
	for k, v := range series.Metadata {
		anonymized.Metadata[k] = v
	}

	// Apply generalized quasi-identifiers
	for qi, value := range identifiers {
		anonymized.Metadata[qi] = value
	}

	return anonymized
}

func (t *TClosenessProcessor) getSortedKeys(m interface{}) []interface{} {
	var keys []interface{}
	
	switch v := m.(type) {
	case map[interface{}]float64:
		for k := range v {
			keys = append(keys, k)
		}
	case map[interface{}]bool:
		for k := range v {
			keys = append(keys, k)
		}
	}

	// Sort keys for consistent ordering
	sort.Slice(keys, func(i, j int) bool {
		return fmt.Sprintf("%v", keys[i]) < fmt.Sprintf("%v", keys[j])
	})

	return keys
}

func (t *TClosenessProcessor) ValidateTCloseness(dataset []*models.TimeSeries) (bool, error) {
	t.calculateGlobalDistributions(dataset)
	groups := t.groupByQuasiIdentifiers(dataset)

	for groupKey, group := range groups {
		if !t.checkTCloseness(group) {
			return false, fmt.Errorf("group %s does not satisfy %.2f-closeness", groupKey, t.config.T)
		}
	}

	return true, nil
}

func getDefaultTClosenessConfig() *TClosenessConfig {
	return &TClosenessConfig{
		T:                   0.2,
		DistanceMetric:      "EMD",
		QuasiIdentifiers:    []string{"location", "age_group", "device_type"},
		SensitiveAttributes: []string{"health_status", "income_level"},
		NumericBins:         10,
		Metadata:            make(map[string]interface{}),
	}
}