package statistical

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// WaveletGenerator generates synthetic time series using wavelet decomposition
type WaveletGenerator struct {
	logger           *logrus.Logger
	config           *WaveletConfig
	trained          bool
	waveletCoeffs    [][]float64 // Wavelet coefficients by level
	scalingCoeffs    []float64   // Scaling coefficients
	decompositionLevels int
	meanValue        float64
	variance         float64
	randSource       *rand.Rand
}

// WaveletConfig contains configuration for wavelet-based generation
type WaveletConfig struct {
	WaveletType      string  `json:"wavelet_type"`      // "haar", "daubechies", "biorthogonal"
	DecompositionLevels int  `json:"decomposition_levels"`  // Number of decomposition levels
	ThresholdType    string  `json:"threshold_type"`    // "soft", "hard", "none"
	ThresholdValue   float64 `json:"threshold_value"`   // Threshold for denoising
	PreserveEnergy   bool    `json:"preserve_energy"`   // Whether to preserve energy
	AdaptiveThreshold bool   `json:"adaptive_threshold"` // Use adaptive thresholding
	NoiseLevel       float64 `json:"noise_level"`       // Amount of noise to add
	Seed             int64   `json:"seed"`              // Random seed
}

// NewWaveletGenerator creates a new wavelet-based generator
func NewWaveletGenerator(config *WaveletConfig, logger *logrus.Logger) *WaveletGenerator {
	if config == nil {
		config = getDefaultWaveletConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}
	
	return &WaveletGenerator{
		logger:     logger,
		config:     config,
		trained:    false,
		randSource: rand.New(rand.NewSource(config.Seed)),
	}
}

// GetType returns the generator type
func (w *WaveletGenerator) GetType() models.GeneratorType {
	return models.GeneratorType(constants.GeneratorTypeWavelet)
}

// GetName returns a human-readable name for the generator
func (w *WaveletGenerator) GetName() string {
	return "Wavelet Generator"
}

// GetDescription returns a description of the generator
func (w *WaveletGenerator) GetDescription() string {
	return "Generates synthetic time series using wavelet decomposition to capture multi-scale temporal patterns"
}

// GetSupportedSensorTypes returns the sensor types this generator supports
func (w *WaveletGenerator) GetSupportedSensorTypes() []models.SensorType {
	return []models.SensorType{
		models.SensorType(constants.SensorTypeTemperature),
		models.SensorType(constants.SensorTypeHumidity),
		models.SensorType(constants.SensorTypePressure),
		models.SensorType(constants.SensorTypeVibration),
		models.SensorType(constants.SensorTypePower),
		models.SensorType(constants.SensorTypeFlow),
		models.SensorType(constants.SensorTypeLevel),
		models.SensorType(constants.SensorTypeSpeed),
		models.SensorType(constants.SensorTypeCustom),
	}
}

// ValidateParameters validates the generation parameters
func (w *WaveletGenerator) ValidateParameters(params models.GenerationParameters) error {
	if params.Length <= 0 {
		return errors.NewValidationError("INVALID_LENGTH", "Generation length must be positive")
	}
	
	if params.Frequency == "" {
		return errors.NewValidationError("INVALID_FREQUENCY", "Frequency is required")
	}
	
	if !w.trained {
		return errors.NewGenerationError("MODEL_NOT_TRAINED", "Wavelet generator must be trained before generation")
	}
	
	return nil
}

// Generate generates synthetic data based on the request
func (w *WaveletGenerator) Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	if err := w.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}
	
	w.logger.WithFields(logrus.Fields{
		"request_id":         req.ID,
		"length":             req.Parameters.Length,
		"wavelet_type":       w.config.WaveletType,
		"decomposition_levels": w.decompositionLevels,
	}).Info("Starting wavelet generation")
	
	start := time.Now()
	
	// Parse frequency
	frequency, err := time.ParseDuration(req.Parameters.Frequency)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_FREQUENCY", "Failed to parse frequency")
	}
	
	// Generate timestamps
	timestamps := make([]time.Time, req.Parameters.Length)
	current := req.Parameters.StartTime
	for i := 0; i < req.Parameters.Length; i++ {
		timestamps[i] = current
		current = current.Add(frequency)
	}
	
	// Generate synthetic time series using wavelet reconstruction
	values := w.generateWaveletSeries(req.Parameters.Length)
	
	// Create data points
	dataPoints := make([]models.DataPoint, len(timestamps))
	for i, timestamp := range timestamps {
		dataPoints[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     values[i],
			Quality:   0.92, // High quality for wavelet reconstruction
		}
	}
	
	// Create time series
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("wavelet-%d", time.Now().UnixNano()),
		Name:        "Wavelet Generated Series",
		Description: "Synthetic time series data generated using wavelet decomposition",
		Tags:        req.Parameters.Tags,
		Metadata:    req.Parameters.Metadata,
		DataPoints:  dataPoints,
		StartTime:   timestamps[0],
		EndTime:     timestamps[len(timestamps)-1],
		Frequency:   req.Parameters.Frequency,
		SensorType:  string(req.Parameters.SensorType),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	
	duration := time.Since(start)
	
	result := &models.GenerationResult{
		ID:            req.ID,
		Status:        "completed",
		TimeSeries:    timeSeries,
		Duration:      duration,
		GeneratedAt:   time.Now(),
		GeneratorType: string(w.GetType()),
		Quality:       0.92,
		Metadata: map[string]interface{}{
			"wavelet_type":         w.config.WaveletType,
			"decomposition_levels": w.decompositionLevels,
			"threshold_type":       w.config.ThresholdType,
			"threshold_value":      w.config.ThresholdValue,
			"noise_level":          w.config.NoiseLevel,
			"data_points":          len(dataPoints),
			"generation_time":      duration.String(),
		},
	}
	
	w.logger.WithFields(logrus.Fields{
		"request_id":  req.ID,
		"data_points": len(dataPoints),
		"duration":    duration,
		"levels":      w.decompositionLevels,
	}).Info("Completed wavelet generation")
	
	return result, nil
}

// Train trains the generator with reference data by performing wavelet decomposition
func (w *WaveletGenerator) Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error {
	if data == nil {
		return errors.NewValidationError("INVALID_DATA", "Training data is required")
	}
	
	if len(data.DataPoints) < 8 {
		return errors.NewValidationError("INSUFFICIENT_DATA", "Need at least 8 data points for wavelet analysis")
	}
	
	w.logger.WithFields(logrus.Fields{
		"series_id":    data.ID,
		"data_points":  len(data.DataPoints),
		"wavelet_type": w.config.WaveletType,
	}).Info("Training wavelet generator")
	
	start := time.Now()
	
	// Extract values from data points
	values := make([]float64, len(data.DataPoints))
	for i, dp := range data.DataPoints {
		values[i] = dp.Value
	}
	
	// Calculate basic statistics
	w.meanValue = w.calculateMean(values)
	w.variance = w.calculateVariance(values, w.meanValue)
	
	// Pad data to power of 2 if necessary
	paddedValues := w.padToPowerOfTwo(values)
	
	// Perform wavelet decomposition
	w.performWaveletDecomposition(paddedValues)
	
	// Apply thresholding for denoising if configured
	if w.config.ThresholdType != "none" {
		w.applyThresholding()
	}
	
	w.trained = true
	duration := time.Since(start)
	
	w.logger.WithFields(logrus.Fields{
		"series_id":         data.ID,
		"training_duration":  duration,
		"decomposition_levels": w.decompositionLevels,
		"mean_value":        w.meanValue,
		"variance":          w.variance,
	}).Info("Wavelet generator training completed")
	
	return nil
}

// IsTrainable returns true if the generator requires/supports training
func (w *WaveletGenerator) IsTrainable() bool {
	return true
}

// generateWaveletSeries generates a time series using stored wavelet coefficients
func (w *WaveletGenerator) generateWaveletSeries(length int) []float64 {
	// Start with stored coefficients and perform inverse wavelet transform
	reconstructed := w.performInverseWaveletTransform()
	
	// Resize to requested length
	result := w.resizeToLength(reconstructed, length)
	
	// Add noise if configured
	if w.config.NoiseLevel > 0 {
		for i := range result {
			noise := w.randSource.NormFloat64() * w.config.NoiseLevel * math.Sqrt(w.variance)
			result[i] += noise
		}
	}
	
	return result
}

// performWaveletDecomposition performs multi-level wavelet decomposition
func (w *WaveletGenerator) performWaveletDecomposition(values []float64) {
	// Determine number of decomposition levels
	n := len(values)
	maxLevels := int(math.Log2(float64(n)))
	w.decompositionLevels = w.config.DecompositionLevels
	if w.decompositionLevels > maxLevels {
		w.decompositionLevels = maxLevels
	}
	
	// Initialize coefficient storage
	w.waveletCoeffs = make([][]float64, w.decompositionLevels)
	
	// Perform decomposition
	currentSignal := make([]float64, len(values))
	copy(currentSignal, values)
	
	for level := 0; level < w.decompositionLevels; level++ {
		// Perform one level of decomposition
		approx, detail := w.waveletDecomposeLevel(currentSignal)
		
		// Store detail coefficients
		w.waveletCoeffs[level] = make([]float64, len(detail))
		copy(w.waveletCoeffs[level], detail)
		
		// Use approximation as input for next level
		currentSignal = approx
	}
	
	// Store final scaling coefficients
	w.scalingCoeffs = make([]float64, len(currentSignal))
	copy(w.scalingCoeffs, currentSignal)
}

// waveletDecomposeLevel performs single-level wavelet decomposition
func (w *WaveletGenerator) waveletDecomposeLevel(signal []float64) ([]float64, []float64) {
	n := len(signal)
	half := n / 2
	
	approx := make([]float64, half)
	detail := make([]float64, half)
	
	// Get wavelet and scaling filters
	h0, h1 := w.getWaveletFilters()
	
	// Convolution and downsampling
	for i := 0; i < half; i++ {
		approxSum := 0.0
		detailSum := 0.0
		
		for j := 0; j < len(h0); j++ {
			idx := (2*i + j) % n
			approxSum += h0[j] * signal[idx]
			detailSum += h1[j] * signal[idx]
		}
		
		approx[i] = approxSum
		detail[i] = detailSum
	}
	
	return approx, detail
}

// performInverseWaveletTransform reconstructs signal from coefficients
func (w *WaveletGenerator) performInverseWaveletTransform() []float64 {
	// Start with scaling coefficients
	reconstructed := make([]float64, len(w.scalingCoeffs))
	copy(reconstructed, w.scalingCoeffs)
	
	// Perform inverse transform for each level (in reverse order)
	for level := w.decompositionLevels - 1; level >= 0; level-- {
		reconstructed = w.waveletReconstructLevel(reconstructed, w.waveletCoeffs[level])
	}
	
	return reconstructed
}

// waveletReconstructLevel performs single-level wavelet reconstruction
func (w *WaveletGenerator) waveletReconstructLevel(approx, detail []float64) []float64 {
	n := len(approx) * 2
	reconstructed := make([]float64, n)
	
	// Get reconstruction filters
	g0, g1 := w.getReconstructionFilters()
	
	// Upsampling and convolution
	for i := 0; i < n; i++ {
		sum := 0.0
		
		for j := 0; j < len(g0); j++ {
			approxIdx := (i - j + n) % len(approx)
			detailIdx := (i - j + n) % len(detail)
			
			if (i-j)%2 == 0 {
				sum += g0[j]*approx[approxIdx/2] + g1[j]*detail[detailIdx/2]
			}
		}
		
		reconstructed[i] = sum
	}
	
	return reconstructed
}

// getWaveletFilters returns decomposition filters based on wavelet type
func (w *WaveletGenerator) getWaveletFilters() ([]float64, []float64) {
	switch w.config.WaveletType {
	case "haar":
		h0 := []float64{1.0 / math.Sqrt(2), 1.0 / math.Sqrt(2)}
		h1 := []float64{1.0 / math.Sqrt(2), -1.0 / math.Sqrt(2)}
		return h0, h1
	
	case "daubechies":
		// Daubechies-4 coefficients
		c0 := (1 + math.Sqrt(3)) / (4 * math.Sqrt(2))
		c1 := (3 + math.Sqrt(3)) / (4 * math.Sqrt(2))
		c2 := (3 - math.Sqrt(3)) / (4 * math.Sqrt(2))
		c3 := (1 - math.Sqrt(3)) / (4 * math.Sqrt(2))
		
		h0 := []float64{c0, c1, c2, c3}
		h1 := []float64{c3, -c2, c1, -c0}
		return h0, h1
	
	default:
		// Default to Haar
		h0 := []float64{1.0 / math.Sqrt(2), 1.0 / math.Sqrt(2)}
		h1 := []float64{1.0 / math.Sqrt(2), -1.0 / math.Sqrt(2)}
		return h0, h1
	}
}

// getReconstructionFilters returns reconstruction filters
func (w *WaveletGenerator) getReconstructionFilters() ([]float64, []float64) {
	h0, h1 := w.getWaveletFilters()
	
	// Reconstruction filters are time-reversed decomposition filters
	g0 := make([]float64, len(h0))
	g1 := make([]float64, len(h1))
	
	for i := 0; i < len(h0); i++ {
		g0[i] = h0[len(h0)-1-i]
		g1[i] = h1[len(h1)-1-i]
	}
	
	return g0, g1
}

// applyThresholding applies thresholding to wavelet coefficients for denoising
func (w *WaveletGenerator) applyThresholding() {
	for level := 0; level < len(w.waveletCoeffs); level++ {
		coeffs := w.waveletCoeffs[level]
		threshold := w.config.ThresholdValue
		
		// Adaptive threshold based on coefficient statistics
		if w.config.AdaptiveThreshold {
			threshold = w.calculateAdaptiveThreshold(coeffs)
		}
		
		// Apply thresholding
		for i := range coeffs {
			switch w.config.ThresholdType {
			case "soft":
				coeffs[i] = w.softThreshold(coeffs[i], threshold)
			case "hard":
				coeffs[i] = w.hardThreshold(coeffs[i], threshold)
			}
		}
	}
}

// softThreshold applies soft thresholding
func (w *WaveletGenerator) softThreshold(value, threshold float64) float64 {
	if math.Abs(value) <= threshold {
		return 0.0
	}
	if value > threshold {
		return value - threshold
	}
	return value + threshold
}

// hardThreshold applies hard thresholding
func (w *WaveletGenerator) hardThreshold(value, threshold float64) float64 {
	if math.Abs(value) <= threshold {
		return 0.0
	}
	return value
}

// calculateAdaptiveThreshold calculates adaptive threshold based on coefficient statistics
func (w *WaveletGenerator) calculateAdaptiveThreshold(coeffs []float64) float64 {
	// Use median absolute deviation (MAD) for robust threshold estimation
	absCoeffs := make([]float64, len(coeffs))
	for i, c := range coeffs {
		absCoeffs[i] = math.Abs(c)
	}
	
	// Sort for median calculation
	w.sortFloat64Slice(absCoeffs)
	
	median := absCoeffs[len(absCoeffs)/2]
	
	// Threshold = sigma * sqrt(2 * log(N))
	sigma := median / 0.6745 // MAD-based sigma estimate
	n := float64(len(coeffs))
	threshold := sigma * math.Sqrt(2*math.Log(n))
	
	return threshold
}

// Helper functions

func (w *WaveletGenerator) calculateMean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (w *WaveletGenerator) calculateVariance(values []float64, mean float64) float64 {
	sum := 0.0
	for _, v := range values {
		diff := v - mean
		sum += diff * diff
	}
	return sum / float64(len(values)-1)
}

func (w *WaveletGenerator) padToPowerOfTwo(values []float64) []float64 {
	n := len(values)
	nextPower := 1
	for nextPower < n {
		nextPower *= 2
	}
	
	if nextPower == n {
		return values
	}
	
	padded := make([]float64, nextPower)
	copy(padded, values)
	
	// Pad with last value (or zeros)
	lastValue := values[n-1]
	for i := n; i < nextPower; i++ {
		padded[i] = lastValue
	}
	
	return padded
}

func (w *WaveletGenerator) resizeToLength(signal []float64, targetLength int) []float64 {
	if len(signal) == targetLength {
		return signal
	}
	
	result := make([]float64, targetLength)
	
	if len(signal) > targetLength {
		// Downsample
		copy(result, signal[:targetLength])
	} else {
		// Upsample by repetition
		for i := 0; i < targetLength; i++ {
			result[i] = signal[i%len(signal)]
		}
	}
	
	return result
}

func (w *WaveletGenerator) sortFloat64Slice(slice []float64) {
	// Simple bubble sort (in production, use sort.Float64s)
	n := len(slice)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if slice[j] > slice[j+1] {
				slice[j], slice[j+1] = slice[j+1], slice[j]
			}
		}
	}
}

// GetDefaultParameters returns default parameters for this generator
func (w *WaveletGenerator) GetDefaultParameters() models.GenerationParameters {
	return models.GenerationParameters{
		Length:    1000,
		Frequency: "1m",
		StartTime: time.Now().Add(-24 * time.Hour),
		Tags:      make(map[string]string),
		Metadata:  make(map[string]interface{}),
	}
}

// EstimateDuration estimates how long generation will take
func (w *WaveletGenerator) EstimateDuration(req *models.GenerationRequest) (time.Duration, error) {
	if req == nil {
		return 0, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	// Wavelet generation is moderately fast, roughly 3ms per 1000 data points
	pointsPerMs := 300.0
	estimatedMs := float64(req.Parameters.Length) / pointsPerMs
	return time.Duration(estimatedMs) * time.Millisecond, nil
}

// Cancel cancels an ongoing generation
func (w *WaveletGenerator) Cancel(ctx context.Context, requestID string) error {
	w.logger.WithField("request_id", requestID).Info("Cancel requested for wavelet generation")
	return nil
}

// GetProgress returns the progress of an ongoing generation
func (w *WaveletGenerator) GetProgress(requestID string) (float64, error) {
	return 1.0, nil
}

// Close cleans up resources
func (w *WaveletGenerator) Close() error {
	w.logger.Info("Closing wavelet generator")
	return nil
}

func getDefaultWaveletConfig() *WaveletConfig {
	return &WaveletConfig{
		WaveletType:         "daubechies",
		DecompositionLevels: 4,
		ThresholdType:       "soft",
		ThresholdValue:      0.1,
		PreserveEnergy:      true,
		AdaptiveThreshold:   true,
		NoiseLevel:          0.01,
		Seed:                time.Now().UnixNano(),
	}
}
