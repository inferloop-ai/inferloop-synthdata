package commands

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
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
	Frequency  string
	OutputFile string
	Format     string
	SensorType string
	NoiseLevel float64
	Anomalies  int
	Privacy    bool
	BatchSize  int
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
	cmd.Flags().StringVarP(&opts.Generator, "generator", "g", "timegan", "Generator algorithm (timegan, arima, rnn, statistical, ydata)")
	cmd.Flags().StringVar(&opts.StartTime, "start-time", "", "Start time (RFC3339 format or relative like 'now-24h')")
	cmd.Flags().StringVar(&opts.EndTime, "end-time", "", "End time (RFC3339 format or relative)")
	cmd.Flags().StringVarP(&opts.Frequency, "frequency", "f", "1m", "Data frequency (1s, 1m, 5m, 1h, etc.)")
	cmd.Flags().StringVarP(&opts.OutputFile, "output", "o", "-", "Output file (- for stdout)")
	cmd.Flags().StringVar(&opts.Format, "format", "csv", "Output format (csv, json, parquet, influx)")
	cmd.Flags().StringVarP(&opts.SensorType, "sensor-type", "s", "temperature", "Sensor type to simulate")
	cmd.Flags().Float64Var(&opts.NoiseLevel, "noise", 0.1, "Noise level (0.0 to 1.0)")
	cmd.Flags().IntVar(&opts.Anomalies, "anomalies", 0, "Number of anomalies to inject")
	cmd.Flags().BoolVar(&opts.Privacy, "privacy", false, "Enable privacy-preserving generation")
	cmd.Flags().IntVar(&opts.BatchSize, "batch-size", 1000, "Batch size for generation")

	return cmd
}

func runGenerate(opts *GenerateOptions) error {
	// Parse times
	var startTime, endTime time.Time
	var err error

	if opts.StartTime == "" {
		startTime = time.Now().Add(-24 * time.Hour)
	} else {
		startTime, err = parseTime(opts.StartTime)
		if err != nil {
			return fmt.Errorf("invalid start time: %w", err)
		}
	}

	if opts.EndTime == "" {
		endTime = time.Now()
	} else {
		endTime, err = parseTime(opts.EndTime)
		if err != nil {
			return fmt.Errorf("invalid end time: %w", err)
		}
	}

	fmt.Printf("Generating synthetic data...\n")
	fmt.Printf("Generator: %s\n", opts.Generator)
	fmt.Printf("Sensor Type: %s\n", opts.SensorType)
	fmt.Printf("Time Range: %s to %s\n", startTime.Format(time.RFC3339), endTime.Format(time.RFC3339))
	fmt.Printf("Frequency: %s\n", opts.Frequency)
	fmt.Printf("Output Format: %s\n", opts.Format)

	// Create generator factory
	logger := logrus.New()
	factory := generators.NewFactory(logger)
	
	// Create generation request
	request := &models.GenerationRequest{
		ID:         fmt.Sprintf("cli-%d", time.Now().Unix()),
		Generator:  models.GeneratorType(opts.Generator),
		Parameters: buildGenerationParameters(opts),
		DataSpec:   buildDataSpecification(opts, startTime, endTime),
		OutputConfig: models.OutputConfiguration{
			Format:      opts.Format,
			Destination: opts.OutputFile,
			Headers:     true,
		},
		Priority:  models.PriorityNormal,
		Status:    models.StatusPending,
		CreatedAt: time.Now(),
	}
	
	// Validate the request
	if err := request.Validate(); err != nil {
		return fmt.Errorf("invalid generation request: %w", err)
	}
	
	// Create generator
	generator, err := factory.CreateGenerator(request.Generator)
	if err != nil {
		return fmt.Errorf("failed to create generator: %w", err)
	}
	defer generator.Close()
	
	// Generate data
	ctx := context.Background()
	result, err := generator.Generate(ctx, request)
	if err != nil {
		return fmt.Errorf("generation failed: %w", err)
	}
	
	// Output data
	if err := outputData(result, opts); err != nil {
		return fmt.Errorf("failed to output data: %w", err)
	}
	
	fmt.Printf("\nGeneration completed successfully!\n")
	fmt.Printf("Generated %d data points\n", len(result.DataPoints))
	return nil
}

func buildGenerationParameters(opts *GenerateOptions) models.GenerationParameters {
	params := models.GenerationParameters{
		BatchSize: opts.BatchSize,
	}
	
	// Set generator-specific parameters
	switch opts.Generator {
	case "statistical":
		params.Statistical = &models.StatisticalParams{
			Method:     "gaussian",
			NoiseLevel: opts.NoiseLevel,
		}
	case "timegan":
		params.TimeGAN = &models.TimeGANParams{
			HiddenDim:      24,
			NumLayers:      3,
			SequenceLength: 24,
			Temperature:    1.0,
		}
	case "arima":
		params.ARIMA = &models.ARIMAParams{
			Order:     []int{1, 1, 1},
			AutoARIMA: true,
		}
	}
	
	// Add post-processing if anomalies requested
	if opts.Anomalies > 0 {
		params.PostProcessing = &models.PostProcessingParams{
			Anomalies: &models.AnomaliesParams{
				Enabled:   true,
				Count:     opts.Anomalies,
				Type:      "spike",
				Magnitude: 2.0,
			},
		}
	}
	
	return params
}

func buildDataSpecification(opts *GenerateOptions, startTime, endTime time.Time) models.DataSpecification {
	return models.DataSpecification{
		SensorType: models.SensorType(opts.SensorType),
		StartTime:  startTime,
		EndTime:    endTime,
		Frequency:  opts.Frequency,
		Features: []models.Feature{
			{
				Name:         "value",
				Type:         "numeric",
				Range:        &models.Range{Min: -10, Max: 50},
				Unit:         getSensorUnit(opts.SensorType),
				Distribution: "normal",
			},
		},
	}
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

func outputData(result *models.GenerationResult, opts *GenerateOptions) error {
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
	
	switch opts.Format {
	case "csv":
		return outputCSV(output, result)
	case "json":
		return outputJSON(output, result)
	default:
		return fmt.Errorf("unsupported output format: %s", opts.Format)
	}
}

func outputCSV(output *os.File, result *models.GenerationResult) error {
	writer := csv.NewWriter(output)
	defer writer.Flush()
	
	// Write header
	if err := writer.Write([]string{"timestamp", "value", "quality"}); err != nil {
		return err
	}
	
	// Write data points
	for _, point := range result.Statistics.TimeRange {
		record := []string{
			point.Start.Format(time.RFC3339),
			"0.0", // Placeholder value
			"1.0", // Placeholder quality
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	
	return nil
}

func outputJSON(output *os.File, result *models.GenerationResult) error {
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result)
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

	// TODO: Implement more sophisticated time parsing
	return time.Parse("2006-01-02", timeStr)
}