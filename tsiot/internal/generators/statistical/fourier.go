package statistical

import (
	"context"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// FourierGenerator generates synthetic time series using Fourier analysis
type FourierGenerator struct {
	logger       *logrus.Logger
	config       *FourierConfig
	trained      bool
	frequencies  []float64
	amplitudes   []float64
	phases       []float64
	meanValue    float64
	noiseLevel   float64
	randSource   *rand.Rand
}

// FourierConfig contains configuration for Fourier-based generation
type FourierConfig struct {
	MaxHarmonics     int     `json:"max_harmonics"`     // Maximum number of harmonics to use
	FrequencyCutoff  float64 `json:"frequency_cutoff"`  // Frequency cutoff for filtering
	NoiseLevel       float64 `json:"noise_level"`       // Amount of noise to add
	WindowType       string  `json:"window_type"`       // Windowing function: "none", "hann", "hamming", "blackman"
	PreserveDC       bool    `json:"preserve_dc"`       // Whether to preserve DC component
	NormalizePhases  bool    `json:"normalize_phases"`  // Whether to normalize phase information
	MinAmplitude     float64 `json:"min_amplitude"`     // Minimum amplitude threshold
	Seed             int64   `json:"seed"`              // Random seed
}

// NewFourierGenerator creates a new Fourier-based generator
func NewFourierGenerator(config *FourierConfig, logger *logrus.Logger) *FourierGenerator {
	if config == nil {
		config = getDefaultFourierConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}
	
	return &FourierGenerator{
		logger:     logger,
		config:     config,
		trained:    false,
		randSource: rand.New(rand.NewSource(config.Seed)),
	}
}

// GetType returns the generator type
func (f *FourierGenerator) GetType() models.GeneratorType {
	return models.GeneratorType(constants.GeneratorTypeFourier)
}

// GetName returns a human-readable name for the generator
func (f *FourierGenerator) GetName() string {
	return "Fourier Generator"
}

// GetDescription returns a description of the generator
func (f *FourierGenerator) GetDescription() string {
	return "Generates synthetic time series using Fourier transform analysis to decompose and reconstruct frequency components"
}

// GetSupportedSensorTypes returns the sensor types this generator supports
func (f *FourierGenerator) GetSupportedSensorTypes() []models.SensorType {
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
func (f *FourierGenerator) ValidateParameters(params models.GenerationParameters) error {
	if params.Length <= 0 {
		return errors.NewValidationError("INVALID_LENGTH", "Generation length must be positive")
	}
	
	if params.Frequency == "" {
		return errors.NewValidationError("INVALID_FREQUENCY", "Frequency is required")
	}
	
	if !f.trained {
		return errors.NewGenerationError("MODEL_NOT_TRAINED", "Fourier generator must be trained before generation")
	}
	
	return nil
}

// Generate generates synthetic data based on the request
func (f *FourierGenerator) Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	if err := f.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}
	
	f.logger.WithFields(logrus.Fields{
		"request_id":    req.ID,
		"length":        req.Parameters.Length,
		"harmonics":     len(f.frequencies),
		"max_harmonics": f.config.MaxHarmonics,
	}).Info("Starting Fourier generation")
	
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
	
	// Generate synthetic time series using Fourier components
	values := f.generateFourierSeries(req.Parameters.Length)
	
	// Create data points
	dataPoints := make([]models.DataPoint, len(timestamps))
	for i, timestamp := range timestamps {
		dataPoints[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     values[i],
			Quality:   0.9, // High quality for Fourier reconstruction
		}
	}
	
	// Create time series
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("fourier-%d", time.Now().UnixNano()),
		Name:        "Fourier Generated Series",
		Description: "Synthetic time series data generated using Fourier transform analysis",
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
		GeneratorType: string(f.GetType()),
		Quality:       0.9,
		Metadata: map[string]interface{}{
			"harmonics_used":     len(f.frequencies),
			"max_harmonics":      f.config.MaxHarmonics,
			"frequency_cutoff":   f.config.FrequencyCutoff,
			"noise_level":        f.config.NoiseLevel,
			"window_type":        f.config.WindowType,
			"data_points":        len(dataPoints),
			"generation_time":    duration.String(),
		},
	}
	
	f.logger.WithFields(logrus.Fields{
		"request_id":   req.ID,
		"data_points":  len(dataPoints),
		"duration":     duration,
		"harmonics":    len(f.frequencies),
	}).Info("Completed Fourier generation")
	
	return result, nil
}

// Train trains the generator with reference data by analyzing frequency components
func (f *FourierGenerator) Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error {
	if data == nil {
		return errors.NewValidationError("INVALID_DATA", "Training data is required")
	}
	
	if len(data.DataPoints) < 4 {
		return errors.NewValidationError("INSUFFICIENT_DATA", "Need at least 4 data points for Fourier analysis")
	}
	
	f.logger.WithFields(logrus.Fields{
		"series_id":    data.ID,
		"data_points":  len(data.DataPoints),
		"max_harmonics": f.config.MaxHarmonics,
	}).Info("Training Fourier generator")
	
	start := time.Now()
	
	// Extract values from data points
	values := make([]float64, len(data.DataPoints))
	for i, dp := range data.DataPoints {
		values[i] = dp.Value
	}
	
	// Calculate mean (DC component)
	f.meanValue = f.calculateMean(values)
	
	// Remove DC component if not preserving it
	if !f.config.PreserveDC {
		for i := range values {
			values[i] -= f.meanValue
		}
	}
	
	// Apply windowing function
	windowedValues := f.applyWindow(values)
	
	// Perform FFT analysis
	frequencies, amplitudes, phases := f.performFFT(windowedValues)
	
	// Filter and select significant frequency components
	f.filterFrequencyComponents(frequencies, amplitudes, phases)
	
	// Estimate noise level from high-frequency components
	f.estimateNoiseLevel(amplitudes)
	
	f.trained = true
	duration := time.Since(start)
	
	f.logger.WithFields(logrus.Fields{
		"series_id":        data.ID,
		"training_duration": duration,
		"harmonics_found":   len(f.frequencies),
		"mean_value":        f.meanValue,
		"noise_level":       f.noiseLevel,
	}).Info("Fourier generator training completed")
	
	return nil
}

// IsTrainable returns true if the generator requires/supports training
func (f *FourierGenerator) IsTrainable() bool {
	return true
}

// generateFourierSeries generates a time series using stored Fourier components
func (f *FourierGenerator) generateFourierSeries(length int) []float64 {
	values := make([]float64, length)
	
	// Generate time points
	for i := 0; i < length; i++ {
		t := float64(i)
		value := f.meanValue // Start with DC component
		
		// Add each harmonic component
		for j := 0; j < len(f.frequencies); j++ {
			freq := f.frequencies[j]
			amp := f.amplitudes[j]
			phase := f.phases[j]
			
			// Add harmonic: amplitude * sin(2À * frequency * t + phase)
			value += amp * math.Sin(2*math.Pi*freq*t/float64(length) + phase)
		}
		
		// Add noise
		noise := f.randSource.NormFloat64() * f.noiseLevel
		value += noise
		
		values[i] = value
	}
	
	return values
}

// performFFT performs Fast Fourier Transform analysis
func (f *FourierGenerator) performFFT(values []float64) ([]float64, []float64, []float64) {
	n := len(values)
	
	// Convert to complex numbers
	complexValues := make([]complex128, n)
	for i, v := range values {
		complexValues[i] = complex(v, 0)
	}
	
	// Perform DFT (simplified - in production would use optimized FFT)
	fftResult := f.dft(complexValues)
	
	// Extract frequency, amplitude, and phase information
	frequencies := make([]float64, n/2) // Only positive frequencies
	amplitudes := make([]float64, n/2)
	phases := make([]float64, n/2)
	
	for i := 0; i < n/2; i++ {
		frequencies[i] = float64(i)
		amplitudes[i] = cmplx.Abs(fftResult[i]) * 2.0 / float64(n) // Normalize
		phases[i] = cmplx.Phase(fftResult[i])
	}
	
	return frequencies, amplitudes, phases
}

// dft performs Discrete Fourier Transform (simplified implementation)
func (f *FourierGenerator) dft(input []complex128) []complex128 {
	n := len(input)
	output := make([]complex128, n)
	
	for k := 0; k < n; k++ {
		sum := complex(0, 0)
		for j := 0; j < n; j++ {
			angle := -2 * math.Pi * float64(k) * float64(j) / float64(n)
			w := cmplx.Exp(complex(0, angle))
			sum += input[j] * w
		}
		output[k] = sum
	}
	
	return output
}

// filterFrequencyComponents filters and selects significant frequency components
func (f *FourierGenerator) filterFrequencyComponents(frequencies, amplitudes, phases []float64) {
	// Create list of component indices sorted by amplitude
	type component struct {
		index     int
		frequency float64
		amplitude float64
		phase     float64
	}
	
	components := make([]component, len(frequencies))
	for i := range frequencies {
		components[i] = component{
			index:     i,
			frequency: frequencies[i],
			amplitude: amplitudes[i],
			phase:     phases[i],
		}
	}
	
	// Sort by amplitude (descending)
	f.sortComponentsByAmplitude(components)
	
	// Select top components within constraints
	maxComponents := f.config.MaxHarmonics
	if maxComponents > len(components) {
		maxComponents = len(components)
	}
	
	// Filter by amplitude threshold and frequency cutoff
	selectedComponents := make([]component, 0, maxComponents)
	for i := 0; i < len(components) && len(selectedComponents) < maxComponents; i++ {
		comp := components[i]
		
		// Skip DC component if not preserving it
		if !f.config.PreserveDC && comp.frequency == 0 {
			continue
		}
		
		// Apply amplitude threshold
		if comp.amplitude < f.config.MinAmplitude {
			continue
		}
		
		// Apply frequency cutoff
		if f.config.FrequencyCutoff > 0 && comp.frequency > f.config.FrequencyCutoff {
			continue
		}
		
		selectedComponents = append(selectedComponents, comp)
	}
	
	// Store selected components
	f.frequencies = make([]float64, len(selectedComponents))
	f.amplitudes = make([]float64, len(selectedComponents))
	f.phases = make([]float64, len(selectedComponents))
	
	for i, comp := range selectedComponents {
		f.frequencies[i] = comp.frequency
		f.amplitudes[i] = comp.amplitude
		f.phases[i] = comp.phase
		
		// Normalize phases if requested
		if f.config.NormalizePhases {
			f.phases[i] = math.Mod(f.phases[i], 2*math.Pi)
		}
	}
}

// estimateNoiseLevel estimates noise level from high-frequency components
func (f *FourierGenerator) estimateNoiseLevel(amplitudes []float64) {
	if len(amplitudes) < 10 {
		f.noiseLevel = f.config.NoiseLevel
		return
	}
	
	// Use high-frequency components to estimate noise
	highFreqStart := len(amplitudes) * 3 / 4
	noiseSum := 0.0
	noiseCount := 0
	
	for i := highFreqStart; i < len(amplitudes); i++ {
		noiseSum += amplitudes[i]
		noiseCount++
	}
	
	if noiseCount > 0 {
		f.noiseLevel = noiseSum / float64(noiseCount)
	} else {
		f.noiseLevel = f.config.NoiseLevel
	}
	
	// Ensure minimum noise level
	if f.noiseLevel < f.config.NoiseLevel {
		f.noiseLevel = f.config.NoiseLevel
	}
}

// applyWindow applies windowing function to reduce spectral leakage
func (f *FourierGenerator) applyWindow(values []float64) []float64 {
	if f.config.WindowType == "none" {
		return values
	}
	
	n := len(values)
	windowed := make([]float64, n)
	
	for i := 0; i < n; i++ {
		var windowValue float64
		
		switch f.config.WindowType {
		case "hann":
			windowValue = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
		case "hamming":
			windowValue = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n-1))
		case "blackman":
			a0, a1, a2 := 0.42, 0.5, 0.08
			windowValue = a0 - a1*math.Cos(2*math.Pi*float64(i)/float64(n-1)) + a2*math.Cos(4*math.Pi*float64(i)/float64(n-1))
		default:
			windowValue = 1.0
		}
		
		windowed[i] = values[i] * windowValue
	}
	
	return windowed
}

// calculateMean calculates the mean of a series
func (f *FourierGenerator) calculateMean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// sortComponentsByAmplitude sorts components by amplitude in descending order
func (f *FourierGenerator) sortComponentsByAmplitude(components []struct {
	index     int
	frequency float64
	amplitude float64
	phase     float64
}) {
	// Simple bubble sort (in production, use sort.Slice)
	n := len(components)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if components[j].amplitude < components[j+1].amplitude {
				components[j], components[j+1] = components[j+1], components[j]
			}
		}
	}
}

// GetDefaultParameters returns default parameters for this generator
func (f *FourierGenerator) GetDefaultParameters() models.GenerationParameters {
	return models.GenerationParameters{
		Length:    1000,
		Frequency: "1m",
		StartTime: time.Now().Add(-24 * time.Hour),
		Tags:      make(map[string]string),
		Metadata:  make(map[string]interface{}),
	}
}

// EstimateDuration estimates how long generation will take
func (f *FourierGenerator) EstimateDuration(req *models.GenerationRequest) (time.Duration, error) {
	if req == nil {
		return 0, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	// Fourier generation is moderately fast, roughly 5ms per 1000 data points
	pointsPerMs := 200.0
	estimatedMs := float64(req.Parameters.Length) / pointsPerMs
	return time.Duration(estimatedMs) * time.Millisecond, nil
}

// Cancel cancels an ongoing generation
func (f *FourierGenerator) Cancel(ctx context.Context, requestID string) error {
	f.logger.WithField("request_id", requestID).Info("Cancel requested for Fourier generation")
	return nil
}

// GetProgress returns the progress of an ongoing generation
func (f *FourierGenerator) GetProgress(requestID string) (float64, error) {
	return 1.0, nil
}

// Close cleans up resources
func (f *FourierGenerator) Close() error {
	f.logger.Info("Closing Fourier generator")
	return nil
}

func getDefaultFourierConfig() *FourierConfig {
	return &FourierConfig{
		MaxHarmonics:    20,
		FrequencyCutoff: 0.5,
		NoiseLevel:      0.01,
		WindowType:      "hann",
		PreserveDC:      true,
		NormalizePhases: true,
		MinAmplitude:    0.001,
		Seed:            time.Now().UnixNano(),
	}
}
