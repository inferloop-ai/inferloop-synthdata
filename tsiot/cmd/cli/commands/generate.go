package commands

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	
	"github.com/inferloop/tsiot/internal/generators"
	"github.com/inferloop/tsiot/pkg/models"
)

type GenerateOptions struct {
	Generator  string
	StartTime  string
	EndTime    string
	Duration   string
	Length     int
	Frequency  string
	OutputFile string
	Format     string
	SensorType string
	NoiseLevel float64
	Anomalies  int
	Privacy    bool
	BatchSize  int
	TrainData  string
	Config     string
}

func NewGenerateCmd() *cobra.Command {
	opts := &GenerateOptions{}

	cmd := &cobra.Command{
		Use:   "generate",
		Short: "Generate synthetic time series data",
		Long: `Generate synthetic time series data using various algorithms including
TimeGAN, ARIMA, RNN, and statistical methods.`,
		Example: `  # Generate temperature sensor data using TimeGAN
  tsiot-cli generate --generator timegan --sensor-type temperature --start-time "2024-01-01" --duration 7d

  # Generate with anomalies
  tsiot-cli generate --generator arima --anomalies 10 --output data.csv

  # Generate privacy-preserving synthetic data
  tsiot-cli generate --generator ydata --privacy --output private_data.json`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenerate(opts)
		},
	}

	// Add flags
	cmd.Flags().StringVarP(&opts.Generator, "generator", "g", "statistical", "Generator algorithm (statistical, timegan, arima, rnn, lstm, gru, ydata)")
	cmd.Flags().StringVar(&opts.StartTime, "start-time", "", "Start time (RFC3339 format or relative like 'now-24h')")
	cmd.Flags().StringVar(&opts.EndTime, "end-time", "", "End time (RFC3339 format or relative)")
	cmd.Flags().StringVar(&opts.Duration, "duration", "24h", "Duration of data to generate (e.g., 24h, 7d, 30d)")
	cmd.Flags().IntVarP(&opts.Length, "length", "l", 0, "Number of data points to generate (overrides duration)")
	cmd.Flags().StringVarP(&opts.Frequency, "frequency", "f", "1m", "Data frequency (1s, 1m, 5m, 1h, etc.)")
	cmd.Flags().StringVarP(&opts.OutputFile, "output", "o", "-", "Output file (- for stdout)")
	cmd.Flags().StringVar(&opts.Format, "format", "csv", "Output format (csv, json)")
	cmd.Flags().StringVarP(&opts.SensorType, "sensor-type", "s", "temperature", "Sensor type to simulate")
	cmd.Flags().Float64Var(&opts.NoiseLevel, "noise", 0.1, "Noise level (0.0 to 1.0)")
	cmd.Flags().IntVar(&opts.Anomalies, "anomalies", 0, "Number of anomalies to inject")
	cmd.Flags().BoolVar(&opts.Privacy, "privacy", false, "Enable privacy-preserving generation")
	cmd.Flags().IntVar(&opts.BatchSize, "batch-size", 1000, "Batch size for generation")
	cmd.Flags().StringVar(&opts.TrainData, "train-data", "", "Training data file for ML generators (required for some generators)")
	cmd.Flags().StringVar(&opts.Config, "config", "", "Generator configuration file (JSON)")

	return cmd
}

func runGenerate(opts *GenerateOptions) error {
	// Validate options
	if err := validateOptions(opts); err != nil {
		return err
	}

	// Parse time parameters
	startTime, length, err := parseTimeParameters(opts)
	if err != nil {
		return err
	}

	fmt.Printf("Generating synthetic data...\n")
	fmt.Printf("Generator: %s\n", opts.Generator)
	fmt.Printf("Sensor Type: %s\n", opts.SensorType)
	fmt.Printf("Start Time: %s\n", startTime.Format(time.RFC3339))
	fmt.Printf("Length: %d data points\n", length)
	fmt.Printf("Frequency: %s\n", opts.Frequency)
	fmt.Printf("Output Format: %s\n", opts.Format)

	// Create generator factory
	logger := logrus.New()
	factory := generators.NewFactory(logger)
	
	// Create generator based on the type
	var genType models.GeneratorType
	switch opts.Generator {
	case "timegan":
		genType = models.GeneratorTypeTimeGAN
	case "arima":
		genType = models.GeneratorTypeARIMA
	case "rnn":
		genType = models.GeneratorTypeRNN
	case "lstm":
		genType = models.GeneratorTypeLSTM
	case "gru":
		genType = models.GeneratorTypeGRU
	case "ydata":
		genType = models.GeneratorTypeYData
	default:
		genType = models.GeneratorTypeStatistical
	}
	
	generator, err := factory.CreateGenerator(genType)
	if err != nil {
		return fmt.Errorf("failed to create generator: %w", err)
	}
	defer generator.Close()

	// Build generation parameters
	params := buildGenerationParameters(opts, startTime, length)
	
	// Train generator if required and training data provided
	if generator.IsTrainable() && opts.TrainData != "" {
		fmt.Printf("Training generator with data from %s...\n", opts.TrainData)
		trainingData, err := loadTrainingData(opts.TrainData)
		if err != nil {
			return fmt.Errorf("failed to load training data: %w", err)
		}
		
		if err := generator.Train(context.Background(), trainingData, params); err != nil {
			return fmt.Errorf("failed to train generator: %w", err)
		}
		fmt.Printf("Training completed successfully.\n")
	}
	
	// Parse frequency
	frequency, err := time.ParseDuration(opts.Frequency)
	if err != nil {
		return fmt.Errorf("invalid frequency: %w", err)
	}

	// Create generation request
	request := &models.GenerationRequest{
		ID:            fmt.Sprintf("cli-%d", time.Now().Unix()),
		GeneratorType: genType,
		SensorType:    opts.SensorType,
		Parameters: models.GenerationParameters{
			Count:     length,
			StartTime: startTime,
			EndTime:   startTime.Add(time.Duration(length) * frequency),
			Frequency: frequency,
			Schema: map[string]interface{}{
				"fields": []map[string]interface{}{
					{
						"name": "timestamp",
						"type": "timestamp",
					},
					{
						"name": opts.SensorType,
						"type": "float64",
					},
				},
			},
			NoiseLevel: opts.NoiseLevel,
		},
		OutputFormat: opts.Format,
		Metadata: map[string]interface{}{
			"generator":   opts.Generator,
			"sensor_type": opts.SensorType,
			"cli":         true,
		},
		CreatedAt: time.Now(),
	}
	
	// Generate data
	ctx := context.Background()
	result, err := generator.Generate(ctx, request)
	if err != nil {
		return fmt.Errorf("generation failed: %w", err)
	}
	
	// Apply post-processing
	if opts.Anomalies > 0 {
		result.TimeSeries = injectAnomalies(result.TimeSeries, opts.Anomalies)
	}
	
	// Output data
	if err := outputGeneratedData(result, opts); err != nil {
		return fmt.Errorf("failed to output data: %w", err)
	}
	
	// Print summary
	fmt.Printf("\nGeneration completed successfully!\n")
	fmt.Printf("Generated %d data points\n", result.RecordsGenerated)
	fmt.Printf("Time Range: %s to %s\n", result.StartTime.Format(time.RFC3339), result.EndTime.Format(time.RFC3339))
	fmt.Printf("Generation Time: %s\n", result.ProcessingTime.String())
	if opts.OutputFile != "-" {
		fmt.Printf("Output saved to: %s\n", opts.OutputFile)
	}
	
	return nil
}

func validateOptions(opts *GenerateOptions) error {
	// Validate generator type
	validGenerators := []string{"statistical", "timegan", "arima", "rnn", "lstm", "gru", "ydata"}
	isValid := false
	for _, gen := range validGenerators {
		if opts.Generator == gen {
			isValid = true
			break
		}
	}
	if !isValid {
		return fmt.Errorf("invalid generator: %s. Valid options: %s", opts.Generator, strings.Join(validGenerators, ", "))
	}
	
	// Validate output format
	if opts.Format != "csv" && opts.Format != "json" {
		return fmt.Errorf("invalid output format: %s. Valid options: csv, json", opts.Format)
	}
	
	// Check if training data is required
	trainableGenerators := []string{"timegan", "rnn", "lstm", "gru", "ydata"}
	needsTraining := false
	for _, gen := range trainableGenerators {
		if opts.Generator == gen {
			needsTraining = true
			break
		}
	}
	
	if needsTraining && opts.TrainData == "" {
		fmt.Printf("Warning: %s generator works better with training data. Consider using --train-data flag.\n", opts.Generator)
	}
	
	return nil
}

func parseTimeParameters(opts *GenerateOptions) (time.Time, int, error) {
	var startTime time.Time
	var err error

	// Parse start time
	if opts.StartTime == "" {
		startTime = time.Now().Add(-24 * time.Hour)
	} else {
		startTime, err = parseTime(opts.StartTime)
		if err != nil {
			return time.Time{}, 0, fmt.Errorf("invalid start time: %w", err)
		}
	}

	// Calculate length
	var length int
	if opts.Length > 0 {
		length = opts.Length
	} else {
		// Calculate from duration and frequency
		duration, err := time.ParseDuration(opts.Duration)
		if err != nil {
			return time.Time{}, 0, fmt.Errorf("invalid duration: %w", err)
		}
		
		frequency, err := time.ParseDuration(opts.Frequency)
		if err != nil {
			return time.Time{}, 0, fmt.Errorf("invalid frequency: %w", err)
		}
		
		length = int(duration / frequency)
		if length <= 0 {
			return time.Time{}, 0, fmt.Errorf("calculated length is zero or negative: check duration and frequency")
		}
	}

	return startTime, length, nil
}

func loadTrainingData(filename string) (*models.TimeSeries, error) {
	// Reuse the loadTimeSeriesFromFile function from analyze.go
	return loadTimeSeriesFromFile(filename)
}

func buildGenerationParameters(opts *GenerateOptions, startTime time.Time, length int) models.GenerationParameters {
	return models.GenerationParameters{
		Length:     length,
		Frequency:  opts.Frequency,
		StartTime:  startTime,
		SensorType: models.SensorType(opts.SensorType),
		Tags: map[string]string{
			"generator": opts.Generator,
			"cli":       "true",
		},
		Metadata: map[string]interface{}{
			"noise_level": opts.NoiseLevel,
			"anomalies":   opts.Anomalies,
			"privacy":     opts.Privacy,
			"batch_size":  opts.BatchSize,
		},
	}
}

func injectAnomalies(timeSeries *models.TimeSeries, anomalyCount int) *models.TimeSeries {
	if len(timeSeries.DataPoints) == 0 || anomalyCount <= 0 {
		return timeSeries
	}

	// Calculate basic statistics for anomaly injection
	values := make([]float64, len(timeSeries.DataPoints))
	for i, dp := range timeSeries.DataPoints {
		values[i] = dp.Value
	}
	
	mean := calculateMean(values)
	variance := calculateVariance(values, mean)
	stdDev := calculateStdDev(variance)
	
	// Randomly select points to make anomalous
	anomalyIndices := make(map[int]bool)
	for len(anomalyIndices) < anomalyCount && len(anomalyIndices) < len(timeSeries.DataPoints) {
		idx := int(time.Now().UnixNano()) % len(timeSeries.DataPoints)
		anomalyIndices[idx] = true
	}
	
	// Create copy of time series
	result := *timeSeries
	result.DataPoints = make([]models.DataPoint, len(timeSeries.DataPoints))
	copy(result.DataPoints, timeSeries.DataPoints)
	
	// Inject anomalies
	for idx := range anomalyIndices {
		// Create spike or drop anomaly
		factor := 3.0 // 3 standard deviations
		if time.Now().UnixNano()%2 == 0 {
			factor = -factor // Make it a drop instead of spike
		}
		
		result.DataPoints[idx].Value = mean + factor*stdDev
		result.DataPoints[idx].Quality = 0.5 // Mark as lower quality
	}
	
	return &result
}

func getSensorUnit(sensorType string) string {
	units := map[string]string{
		"temperature": "°C",
		"humidity":    "%RH",
		"pressure":    "Pa",
		"vibration":   "m/s²",
		"power":       "W",
		"flow":        "L/min",
		"level":       "m",
		"speed":       "m/s",
	}
	
	if unit, exists := units[sensorType]; exists {
		return unit
	}
	return ""
}

func outputGeneratedData(result *models.GenerationResult, opts *GenerateOptions) error {
	var output *os.File
	var err error
	
	if opts.OutputFile == "-" {
		output = os.Stdout
	} else {
		output, err = os.Create(opts.OutputFile)
		if err != nil {
			return fmt.Errorf("failed to create output file: %w", err)
		}
		defer output.Close()
	}
	
	// Extract TimeSeries from result.Data
	var timeSeries *models.TimeSeries
	if ts, ok := result.Data.(*models.TimeSeries); ok {
		timeSeries = ts
	} else {
		return fmt.Errorf("generated data is not a TimeSeries")
	}
	
	switch opts.Format {
	case "csv":
		return outputTimeSeriesCSV(output, timeSeries)
	case "json":
		return outputTimeSeriesJSON(output, timeSeries)
	default:
		return fmt.Errorf("unsupported output format: %s", opts.Format)
	}
}

func outputTimeSeriesCSV(output *os.File, timeSeries *models.TimeSeries) error {
	writer := csv.NewWriter(output)
	defer writer.Flush()
	
	// Write header
	if err := writer.Write([]string{"timestamp", "value", "quality"}); err != nil {
		return err
	}
	
	// Write data points
	for _, point := range timeSeries.DataPoints {
		record := []string{
			point.Timestamp.Format(time.RFC3339),
			strconv.FormatFloat(point.Value, 'f', 6, 64),
			strconv.FormatFloat(point.Quality, 'f', 3, 64),
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	
	return nil
}

func outputTimeSeriesJSON(output *os.File, timeSeries *models.TimeSeries) error {
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(timeSeries)
}

func parseTime(timeStr string) (time.Time, error) {
	// Try parsing as RFC3339 first
	if t, err := time.Parse(time.RFC3339, timeStr); err == nil {
		return t, nil
	}

	// Handle relative times like "now-24h"
	if timeStr == "now" {
		return time.Now(), nil
	}
	
	// Handle relative time expressions
	if strings.HasPrefix(timeStr, "now-") {
		durationStr := timeStr[4:] // Remove "now-"
		duration, err := time.ParseDuration(durationStr)
		if err != nil {
			return time.Time{}, fmt.Errorf("invalid relative time duration: %s", durationStr)
		}
		return time.Now().Add(-duration), nil
	}
	
	if strings.HasPrefix(timeStr, "now+") {
		durationStr := timeStr[4:] // Remove "now+"
		duration, err := time.ParseDuration(durationStr)
		if err != nil {
			return time.Time{}, fmt.Errorf("invalid relative time duration: %s", durationStr)
		}
		return time.Now().Add(duration), nil
	}

	// Try other common formats
	formats := []string{
		"2006-01-02 15:04:05",
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05.000",
		"2006-01-02T15:04:05.000",
		"01/02/2006 15:04:05",
		"2006-01-02",
		"01/02/2006",
	}

	for _, format := range formats {
		if t, err := time.Parse(format, timeStr); err == nil {
			return t, nil
		}
	}

	// Try Unix timestamp
	if unixTime, err := strconv.ParseInt(timeStr, 10, 64); err == nil {
		// Check if it's seconds or milliseconds
		if unixTime > 1e10 {
			return time.Unix(unixTime/1000, (unixTime%1000)*1e6), nil
		}
		return time.Unix(unixTime, 0), nil
	}

	return time.Time{}, fmt.Errorf("unable to parse time: %s", timeStr)
}