package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

type Config struct {
	NumSeries       int           `json:"num_series"`
	PointsPerSeries int           `json:"points_per_series"`
	TimeInterval    time.Duration `json:"time_interval"`
	OutputFormat    string        `json:"output_format"` // json, csv
	OutputFile      string        `json:"output_file"`
	Patterns        []Pattern     `json:"patterns"`
	Noise           NoiseConfig   `json:"noise"`
}

type Pattern struct {
	Type        string  `json:"type"`        // sine, linear, seasonal, random
	Amplitude   float64 `json:"amplitude"`
	Frequency   float64 `json:"frequency"`
	Phase       float64 `json:"phase"`
	Trend       float64 `json:"trend"`
	Seasonality int     `json:"seasonality"`
}

type NoiseConfig struct {
	Enabled bool    `json:"enabled"`
	Type    string  `json:"type"`    // gaussian, uniform
	Level   float64 `json:"level"`
	Mean    float64 `json:"mean"`
	StdDev  float64 `json:"std_dev"`
}

type Generator struct {
	config *Config
	logger *logrus.Logger
	rand   *rand.Rand
}

func main() {
	var (
		configFile = flag.String("config", "", "Configuration file path")
		numSeries  = flag.Int("series", 10, "Number of time series to generate")
		points     = flag.Int("points", 1000, "Number of points per series")
		output     = flag.String("output", "test_data.json", "Output file")
		format     = flag.String("format", "json", "Output format (json/csv)")
		verbose    = flag.Bool("verbose", false, "Enable verbose logging")
	)
	flag.Parse()

	// Setup logging
	logger := logrus.New()
	if *verbose {
		logger.SetLevel(logrus.DebugLevel)
	}

	// Load or create config
	var config *Config
	if *configFile != "" {
		var err error
		config, err = loadConfig(*configFile)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}
	} else {
		config = getDefaultConfig()
		config.NumSeries = *numSeries
		config.PointsPerSeries = *points
		config.OutputFile = *output
		config.OutputFormat = *format
	}

	generator := NewGenerator(config, logger)

	logger.WithFields(logrus.Fields{
		"num_series":        config.NumSeries,
		"points_per_series": config.PointsPerSeries,
		"output_file":       config.OutputFile,
		"output_format":     config.OutputFormat,
	}).Info("Starting test data generation")

	// Generate test data
	data, err := generator.Generate(context.Background())
	if err != nil {
		log.Fatalf("Failed to generate data: %v", err)
	}

	// Save to file
	err = generator.SaveToFile(data, config.OutputFile, config.OutputFormat)
	if err != nil {
		log.Fatalf("Failed to save data: %v", err)
	}

	logger.WithFields(logrus.Fields{
		"series_generated": len(data),
		"output_file":      config.OutputFile,
	}).Info("Test data generation completed")
}

func NewGenerator(config *Config, logger *logrus.Logger) *Generator {
	return &Generator{
		config: config,
		logger: logger,
		rand:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (g *Generator) Generate(ctx context.Context) ([]*models.TimeSeries, error) {
	series := make([]*models.TimeSeries, g.config.NumSeries)

	for i := 0; i < g.config.NumSeries; i++ {
		ts, err := g.generateSeries(ctx, i)
		if err != nil {
			return nil, fmt.Errorf("failed to generate series %d: %w", i, err)
		}
		series[i] = ts
	}

	return series, nil
}

func (g *Generator) generateSeries(ctx context.Context, seriesIndex int) (*models.TimeSeries, error) {
	now := time.Now()

	ts := &models.TimeSeries{
		ID:     fmt.Sprintf("test_series_%d", seriesIndex),
		Name:   fmt.Sprintf("Test Series %d", seriesIndex),
		Points: make([]models.DataPoint, g.config.PointsPerSeries),
		Metadata: map[string]interface{}{
			"generator":    "test-data-generator",
			"series_type":  "synthetic",
			"created_at":   now.Format(time.RFC3339),
			"series_index": seriesIndex,
		},
		Properties: map[string]interface{}{
			"unit":        "units",
			"description": fmt.Sprintf("Generated test time series %d", seriesIndex),
		},
		Tags: []string{
			"test",
			"synthetic",
			fmt.Sprintf("series_%d", seriesIndex),
		},
	}

	// Generate data points
	for i := 0; i < g.config.PointsPerSeries; i++ {
		timestamp := now.Add(-time.Duration(g.config.PointsPerSeries-i-1) * g.config.TimeInterval).Unix()

		// Generate base value using patterns
		value := g.generateValue(i, seriesIndex)

		// Add noise if enabled
		if g.config.Noise.Enabled {
			value += g.generateNoise()
		}

		ts.Points[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     value,
			Quality:   g.generateQuality(),
		}
	}

	return ts, nil
}

func (g *Generator) generateValue(pointIndex, seriesIndex int) float64 {
	if len(g.config.Patterns) == 0 {
		// Default random walk
		return g.rand.Float64() * 100
	}

	value := 0.0
	t := float64(pointIndex)

	for _, pattern := range g.config.Patterns {
		switch pattern.Type {
		case "sine":
			value += pattern.Amplitude * math.Sin(2*math.Pi*pattern.Frequency*t/100+pattern.Phase)
		case "cosine":
			value += pattern.Amplitude * math.Cos(2*math.Pi*pattern.Frequency*t/100+pattern.Phase)
		case "linear":
			value += pattern.Trend * t
		case "seasonal":
			if pattern.Seasonality > 0 {
				seasonalValue := pattern.Amplitude * math.Sin(2*math.Pi*t/float64(pattern.Seasonality))
				value += seasonalValue
			}
		case "exponential":
			value += pattern.Amplitude * math.Exp(pattern.Trend*t/100)
		case "logarithmic":
			if t > 0 {
				value += pattern.Amplitude * math.Log(1+t)
			}
		case "random":
			value += pattern.Amplitude * (g.rand.Float64() - 0.5)
		case "step":
			if int(t)%50 == 0 {
				value += pattern.Amplitude
			}
		default:
			value += g.rand.Float64() * pattern.Amplitude
		}
	}

	// Add series-specific offset
	value += float64(seriesIndex) * 10

	return value
}

func (g *Generator) generateNoise() float64 {
	switch g.config.Noise.Type {
	case "gaussian":
		return g.rand.NormFloat64()*g.config.Noise.StdDev + g.config.Noise.Mean
	case "uniform":
		return (g.rand.Float64()-0.5) * g.config.Noise.Level
	default:
		return g.rand.NormFloat64() * g.config.Noise.Level
	}
}

func (g *Generator) generateQuality() float64 {
	// Generate quality score between 0.8 and 1.0
	return 0.8 + g.rand.Float64()*0.2
}

func (g *Generator) SaveToFile(data []*models.TimeSeries, filename, format string) error {
	switch format {
	case "json":
		return g.saveAsJSON(data, filename)
	case "csv":
		return g.saveAsCSV(data, filename)
	default:
		return fmt.Errorf("unsupported format: %s", format)
	}
}

func (g *Generator) saveAsJSON(data []*models.TimeSeries, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	return encoder.Encode(map[string]interface{}{
		"metadata": map[string]interface{}{
			"generated_at": time.Now().Format(time.RFC3339),
			"generator":    "test-data-generator",
			"version":      "1.0.0",
			"series_count": len(data),
		},
		"data": data,
	})
}

func (g *Generator) saveAsCSV(data []*models.TimeSeries, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	// Write CSV header
	_, err = file.WriteString("series_id,series_name,timestamp,value,quality\n")
	if err != nil {
		return err
	}

	// Write data points
	for _, series := range data {
		for _, point := range series.Points {
			line := fmt.Sprintf("%s,%s,%d,%.6f,%.3f\n",
				series.ID, series.Name, point.Timestamp, point.Value, point.Quality)
			_, err = file.WriteString(line)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func loadConfig(filename string) (*Config, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var config Config
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&config)
	if err != nil {
		return nil, err
	}

	return &config, nil
}

func getDefaultConfig() *Config {
	return &Config{
		NumSeries:       10,
		PointsPerSeries: 1000,
		TimeInterval:    time.Minute,
		OutputFormat:    "json",
		OutputFile:      "test_data.json",
		Patterns: []Pattern{
			{
				Type:      "sine",
				Amplitude: 50.0,
				Frequency: 0.1,
				Phase:     0.0,
			},
			{
				Type:        "seasonal",
				Amplitude:   20.0,
				Seasonality: 24, // Daily pattern
			},
			{
				Type:  "linear",
				Trend: 0.1,
			},
			{
				Type:      "random",
				Amplitude: 10.0,
			},
		},
		Noise: NoiseConfig{
			Enabled: true,
			Type:    "gaussian",
			Level:   5.0,
			Mean:    0.0,
			StdDev:  2.0,
		},
	}
}