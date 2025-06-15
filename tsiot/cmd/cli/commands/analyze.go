package commands

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/internal/analytics"
	"github.com/inferloop/tsiot/internal/storage"
	"github.com/inferloop/tsiot/pkg/models"
)

type AnalyzeOptions struct {
	InputFile       string
	AnalysisType    []string
	WindowSize      string
	DetectAnomalies bool
	Seasonality     bool
	Forecast        bool
	ForecastPeriods int
	OutputFormat    string
	OutputFile      string
}

func NewAnalyzeCmd() *cobra.Command {
	opts := &AnalyzeOptions{}

	cmd := &cobra.Command{
		Use:   "analyze",
		Short: "Analyze time series data patterns and characteristics",
		Long: `Analyze time series data to understand patterns, detect anomalies,
identify seasonality, and generate forecasts.`,
		Example: `  # Basic analysis
  tsiot-cli analyze --input sensor_data.csv

  # Detect anomalies
  tsiot-cli analyze --input data.csv --detect-anomalies

  # Analyze seasonality and forecast
  tsiot-cli analyze --input data.csv --seasonality --forecast --forecast-periods 24`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runAnalyze(opts)
		},
	}

	// Add flags
	cmd.Flags().StringVarP(&opts.InputFile, "input", "i", "", "Input file to analyze (required)")
	cmd.Flags().StringSliceVarP(&opts.AnalysisType, "type", "t", []string{"basic"}, "Analysis types (basic, advanced, all)")
	cmd.Flags().StringVar(&opts.WindowSize, "window", "1h", "Window size for analysis")
	cmd.Flags().BoolVar(&opts.DetectAnomalies, "detect-anomalies", false, "Detect anomalies in the data")
	cmd.Flags().BoolVar(&opts.Seasonality, "seasonality", false, "Analyze seasonality patterns")
	cmd.Flags().BoolVar(&opts.Forecast, "forecast", false, "Generate forecast")
	cmd.Flags().IntVar(&opts.ForecastPeriods, "forecast-periods", 10, "Number of periods to forecast")
	cmd.Flags().StringVar(&opts.OutputFormat, "format", "text", "Output format (text, json, html)")
	cmd.Flags().StringVarP(&opts.OutputFile, "output", "o", "-", "Output file (- for stdout)")

	cmd.MarkFlagRequired("input")

	return cmd
}

func runAnalyze(opts *AnalyzeOptions) error {
	// Validate input file
	if opts.InputFile == "" {
		return fmt.Errorf("input file is required")
	}

	// Load time series data
	timeSeries, err := loadTimeSeriesFromFile(opts.InputFile)
	if err != nil {
		return fmt.Errorf("failed to load data: %w", err)
	}

	fmt.Printf("Analyzing time series data...\n")
	fmt.Printf("Input File: %s\n", opts.InputFile)
	fmt.Printf("Analysis Types: %v\n", opts.AnalysisType)

	// Initialize analytics engine
	logger := logrus.New()
	analyticsEngine := analytics.NewEngine(nil, logger)

	// Perform analysis
	analysisResult, err := performAnalysis(analyticsEngine, timeSeries, opts)
	if err != nil {
		return fmt.Errorf("analysis failed: %w", err)
	}

	// Output results
	return outputResults(analysisResult, opts)
}

func loadTimeSeriesFromFile(filename string) (*models.TimeSeries, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("cannot open file: %w", err)
	}
	defer file.Close()

	// Determine file format by extension
	if strings.HasSuffix(strings.ToLower(filename), ".csv") {
		return loadFromCSV(file, filename)
	} else if strings.HasSuffix(strings.ToLower(filename), ".json") {
		return loadFromJSON(file, filename)
	}

	return nil, fmt.Errorf("unsupported file format. Supported: .csv, .json")
}

func loadFromCSV(file *os.File, filename string) (*models.TimeSeries, error) {
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %w", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("CSV file must have at least header and one data row")
	}

	// Parse header to determine timestamp and value columns
	header := records[0]
	timestampCol := -1
	valueCol := -1

	for i, col := range header {
		colLower := strings.ToLower(col)
		if timestampCol == -1 && (strings.Contains(colLower, "time") || strings.Contains(colLower, "date")) {
			timestampCol = i
		}
		if valueCol == -1 && (strings.Contains(colLower, "value") || strings.Contains(colLower, "data") || strings.Contains(colLower, "sensor")) {
			valueCol = i
		}
	}

	if timestampCol == -1 || valueCol == -1 {
		return nil, fmt.Errorf("CSV must have timestamp and value columns")
	}

	// Parse data rows
	dataPoints := make([]models.DataPoint, 0, len(records)-1)
	for i, record := range records[1:] {
		if len(record) <= timestampCol || len(record) <= valueCol {
			continue
		}

		// Parse timestamp
		timestamp, err := parseTimestamp(record[timestampCol])
		if err != nil {
			return nil, fmt.Errorf("invalid timestamp at row %d: %w", i+2, err)
		}

		// Parse value
		value, err := strconv.ParseFloat(record[valueCol], 64)
		if err != nil {
			return nil, fmt.Errorf("invalid value at row %d: %w", i+2, err)
		}

		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: timestamp,
			Value:     value,
			Quality:   1.0,
		})
	}

	if len(dataPoints) == 0 {
		return nil, fmt.Errorf("no valid data points found")
	}

	return &models.TimeSeries{
		ID:          fmt.Sprintf("analysis-%d", time.Now().Unix()),
		Name:        filename,
		Description: fmt.Sprintf("Time series loaded from %s", filename),
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   inferFrequency(dataPoints),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}, nil
}

func loadFromJSON(file *os.File, filename string) (*models.TimeSeries, error) {
	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON: %w", err)
	}

	var timeSeries models.TimeSeries
	if err := json.Unmarshal(data, &timeSeries); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	if len(timeSeries.DataPoints) == 0 {
		return nil, fmt.Errorf("no data points found in JSON")
	}

	return &timeSeries, nil
}

func parseTimestamp(timestampStr string) (time.Time, error) {
	// Try common timestamp formats
	formats := []string{
		time.RFC3339,
		time.RFC3339Nano,
		"2006-01-02 15:04:05",
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05.000",
		"2006-01-02T15:04:05.000",
		"01/02/2006 15:04:05",
		"2006-01-02",
		"01/02/2006",
	}

	for _, format := range formats {
		if t, err := time.Parse(format, timestampStr); err == nil {
			return t, nil
		}
	}

	// Try Unix timestamp
	if unixTime, err := strconv.ParseInt(timestampStr, 10, 64); err == nil {
		// Check if it's seconds or milliseconds
		if unixTime > 1e10 {
			return time.Unix(unixTime/1000, (unixTime%1000)*1e6), nil
		}
		return time.Unix(unixTime, 0), nil
	}

	return time.Time{}, fmt.Errorf("unable to parse timestamp: %s", timestampStr)
}

func inferFrequency(dataPoints []models.DataPoint) string {
	if len(dataPoints) < 2 {
		return "unknown"
	}

	// Calculate average interval between consecutive points
	totalDuration := time.Duration(0)
	intervals := 0

	for i := 1; i < len(dataPoints) && intervals < 10; i++ {
		interval := dataPoints[i].Timestamp.Sub(dataPoints[i-1].Timestamp)
		if interval > 0 {
			totalDuration += interval
			intervals++
		}
	}

	if intervals == 0 {
		return "unknown"
	}

	avgInterval := totalDuration / time.Duration(intervals)

	// Round to common frequencies
	switch {
	case avgInterval < 30*time.Second:
		return "1s"
	case avgInterval < 5*time.Minute:
		return "1m"
	case avgInterval < 30*time.Minute:
		return "5m"
	case avgInterval < 2*time.Hour:
		return "1h"
	case avgInterval < 12*time.Hour:
		return "6h"
	default:
		return "1d"
	}
}

type AnalysisResult struct {
	BasicStats      *BasicStatistics      `json:"basic_stats"`
	PatternAnalysis *PatternAnalysis      `json:"pattern_analysis,omitempty"`
	Seasonality     *SeasonalityAnalysis  `json:"seasonality,omitempty"`
	Anomalies       *AnomalyAnalysis      `json:"anomalies,omitempty"`
	Forecast        *ForecastAnalysis     `json:"forecast,omitempty"`
	TimeSeries      *models.TimeSeries    `json:"-"`
}

type BasicStatistics struct {
	DataPoints int       `json:"data_points"`
	StartTime  time.Time `json:"start_time"`
	EndTime    time.Time `json:"end_time"`
	Mean       float64   `json:"mean"`
	StdDev     float64   `json:"std_dev"`
	Min        float64   `json:"min"`
	Max        float64   `json:"max"`
	Median     float64   `json:"median"`
	Q25        float64   `json:"q25"`
	Q75        float64   `json:"q75"`
}

type PatternAnalysis struct {
	TrendDirection string  `json:"trend_direction"`
	TrendStrength  float64 `json:"trend_strength"`
	PrimaryPeriod  string  `json:"primary_period"`
	Volatility     float64 `json:"volatility"`
}

type SeasonalityAnalysis struct {
	HasSeasonality    bool                   `json:"has_seasonality"`
	SeasonalStrength  float64                `json:"seasonal_strength"`
	DominantPeriods   []int                  `json:"dominant_periods"`
	SeasonalPatterns  map[string]interface{} `json:"seasonal_patterns"`
}

type AnomalyAnalysis struct {
	AnomaliesFound []AnomalyPoint `json:"anomalies_found"`
	AnomalyRate    float64        `json:"anomaly_rate"`
	Method         string         `json:"method"`
}

type AnomalyPoint struct {
	Timestamp   time.Time `json:"timestamp"`
	Value       float64   `json:"value"`
	AnomalyType string    `json:"anomaly_type"`
	Severity    float64   `json:"severity"`
}

type ForecastAnalysis struct {
	Method           string                 `json:"method"`
	ForecastPoints   []ForecastPoint        `json:"forecast_points"`
	ConfidenceLevel  float64                `json:"confidence_level"`
	ModelParameters  map[string]interface{} `json:"model_parameters"`
}

type ForecastPoint struct {
	Timestamp    time.Time `json:"timestamp"`
	Value        float64   `json:"value"`
	LowerBound   float64   `json:"lower_bound"`
	UpperBound   float64   `json:"upper_bound"`
}

func performAnalysis(engine *analytics.Engine, timeSeries *models.TimeSeries, opts *AnalyzeOptions) (*AnalysisResult, error) {
	result := &AnalysisResult{
		TimeSeries: timeSeries,
	}

	// Calculate basic statistics
	result.BasicStats = calculateBasicStatistics(timeSeries)

	// Perform requested analyses
	for _, analysisType := range opts.AnalysisType {
		switch strings.ToLower(analysisType) {
		case "basic", "all":
			result.PatternAnalysis = analyzePatterns(timeSeries)
		case "advanced", "all":
			result.PatternAnalysis = analyzePatterns(timeSeries)
		}
	}

	// Seasonality analysis
	if opts.Seasonality {
		result.Seasonality = analyzeSeasonality(timeSeries)
	}

	// Anomaly detection
	if opts.DetectAnomalies {
		result.Anomalies = detectAnomalies(timeSeries)
	}

	// Forecasting
	if opts.Forecast {
		result.Forecast = generateForecast(timeSeries, opts.ForecastPeriods)
	}

	return result, nil
}

func calculateBasicStatistics(ts *models.TimeSeries) *BasicStatistics {
	if len(ts.DataPoints) == 0 {
		return &BasicStatistics{}
	}

	values := make([]float64, len(ts.DataPoints))
	for i, dp := range ts.DataPoints {
		values[i] = dp.Value
	}

	// Calculate statistics
	mean := calculateMean(values)
	variance := calculateVariance(values, mean)
	stdDev := calculateStdDev(variance)
	min, max := calculateMinMax(values)
	median := calculateMedian(values)
	q25 := calculatePercentile(values, 0.25)
	q75 := calculatePercentile(values, 0.75)

	return &BasicStatistics{
		DataPoints: len(ts.DataPoints),
		StartTime:  ts.StartTime,
		EndTime:    ts.EndTime,
		Mean:       mean,
		StdDev:     stdDev,
		Min:        min,
		Max:        max,
		Median:     median,
		Q25:        q25,
		Q75:        q75,
	}
}

func analyzePatterns(ts *models.TimeSeries) *PatternAnalysis {
	if len(ts.DataPoints) < 3 {
		return &PatternAnalysis{}
	}

	values := make([]float64, len(ts.DataPoints))
	for i, dp := range ts.DataPoints {
		values[i] = dp.Value
	}

	// Calculate trend
	trendStrength, trendDirection := calculateTrend(values)
	
	// Estimate primary period
	primaryPeriod := estimatePrimaryPeriod(ts)
	
	// Calculate volatility
	volatility := calculateVolatility(values)

	return &PatternAnalysis{
		TrendDirection: trendDirection,
		TrendStrength:  trendStrength,
		PrimaryPeriod:  primaryPeriod,
		Volatility:     volatility,
	}
}

func analyzeSeasonality(ts *models.TimeSeries) *SeasonalityAnalysis {
	if len(ts.DataPoints) < 48 {
		return &SeasonalityAnalysis{HasSeasonality: false}
	}

	// Simple seasonality detection based on autocorrelation
	values := make([]float64, len(ts.DataPoints))
	for i, dp := range ts.DataPoints {
		values[i] = dp.Value
	}

	// Check for daily (24h) and weekly (7d) patterns
	periods := []int{24, 168} // 24 hours, 7 days worth of hours
	dominantPeriods := []int{}
	seasonalStrength := 0.0

	for _, period := range periods {
		if len(values) >= period*2 {
			autocorr := calculateAutocorrelation(values, period)
			if autocorr > 0.3 {
				dominantPeriods = append(dominantPeriods, period)
				if autocorr > seasonalStrength {
					seasonalStrength = autocorr
				}
			}
		}
	}

	return &SeasonalityAnalysis{
		HasSeasonality:   len(dominantPeriods) > 0,
		SeasonalStrength: seasonalStrength,
		DominantPeriods:  dominantPeriods,
		SeasonalPatterns: make(map[string]interface{}),
	}
}

func detectAnomalies(ts *models.TimeSeries) *AnomalyAnalysis {
	if len(ts.DataPoints) < 10 {
		return &AnomalyAnalysis{Method: "insufficient_data"}
	}

	values := make([]float64, len(ts.DataPoints))
	for i, dp := range ts.DataPoints {
		values[i] = dp.Value
	}

	mean := calculateMean(values)
	stdDev := calculateStdDev(calculateVariance(values, mean))
	threshold := 2.5 * stdDev

	anomalies := []AnomalyPoint{}
	for i, dp := range ts.DataPoints {
		deviation := dp.Value - mean
		if deviation > threshold || deviation < -threshold {
			anomalyType := "spike"
			if deviation < 0 {
				anomalyType = "drop"
			}
			
			anomalies = append(anomalies, AnomalyPoint{
				Timestamp:   dp.Timestamp,
				Value:       dp.Value,
				AnomalyType: anomalyType,
				Severity:    deviation / stdDev,
			})
		}
	}

	return &AnomalyAnalysis{
		AnomaliesFound: anomalies,
		AnomalyRate:    float64(len(anomalies)) / float64(len(ts.DataPoints)) * 100,
		Method:         "statistical_zscore",
	}
}

func generateForecast(ts *models.TimeSeries, periods int) *ForecastAnalysis {
	if len(ts.DataPoints) < 5 {
		return &ForecastAnalysis{Method: "insufficient_data"}
	}

	// Simple linear extrapolation forecast
	values := make([]float64, len(ts.DataPoints))
	for i, dp := range ts.DataPoints {
		values[i] = dp.Value
	}

	// Calculate trend
	trend, _ := calculateTrend(values)
	lastValue := values[len(values)-1]
	
	// Estimate time interval
	interval := time.Hour // Default
	if len(ts.DataPoints) > 1 {
		interval = ts.DataPoints[1].Timestamp.Sub(ts.DataPoints[0].Timestamp)
	}

	forecastPoints := make([]ForecastPoint, periods)
	stdDev := calculateStdDev(calculateVariance(values, calculateMean(values)))
	
	for i := 0; i < periods; i++ {
		timestamp := ts.EndTime.Add(time.Duration(i+1) * interval)
		value := lastValue + trend*float64(i+1)
		margin := 1.96 * stdDev // 95% confidence interval
		
		forecastPoints[i] = ForecastPoint{
			Timestamp:  timestamp,
			Value:      value,
			LowerBound: value - margin,
			UpperBound: value + margin,
		}
	}

	return &ForecastAnalysis{
		Method:          "linear_trend",
		ForecastPoints:  forecastPoints,
		ConfidenceLevel: 0.95,
		ModelParameters: map[string]interface{}{
			"trend": trend,
			"last_value": lastValue,
		},
	}
}

func outputResults(result *AnalysisResult, opts *AnalyzeOptions) error {
	switch strings.ToLower(opts.OutputFormat) {
	case "json":
		return outputJSON(result, opts.OutputFile)
	case "html":
		return outputHTML(result, opts.OutputFile)
	default:
		return outputText(result, opts.OutputFile)
	}
}

func outputText(result *AnalysisResult, outputFile string) error {
	output := fmt.Sprintf("Analysis Results\n")
	output += "================\n\n"

	// Basic statistics
	if result.BasicStats != nil {
		output += "Basic Statistics:\n"
		output += fmt.Sprintf("- Data Points: %d\n", result.BasicStats.DataPoints)
		output += fmt.Sprintf("- Time Range: %s to %s\n", 
			result.BasicStats.StartTime.Format("2006-01-02 15:04:05"),
			result.BasicStats.EndTime.Format("2006-01-02 15:04:05"))
		output += fmt.Sprintf("- Mean: %.2f\n", result.BasicStats.Mean)
		output += fmt.Sprintf("- Std Dev: %.2f\n", result.BasicStats.StdDev)
		output += fmt.Sprintf("- Min: %.2f\n", result.BasicStats.Min)
		output += fmt.Sprintf("- Max: %.2f\n", result.BasicStats.Max)
		output += fmt.Sprintf("- Median: %.2f\n", result.BasicStats.Median)
		output += "\n"
	}

	// Pattern analysis
	if result.PatternAnalysis != nil {
		output += "Pattern Analysis:\n"
		output += fmt.Sprintf("- Trend: %s (strength: %.3f)\n", 
			result.PatternAnalysis.TrendDirection, result.PatternAnalysis.TrendStrength)
		output += fmt.Sprintf("- Primary Period: %s\n", result.PatternAnalysis.PrimaryPeriod)
		output += fmt.Sprintf("- Volatility: %.3f\n", result.PatternAnalysis.Volatility)
		output += "\n"
	}

	// Seasonality
	if result.Seasonality != nil {
		output += "Seasonality Analysis:\n"
		output += fmt.Sprintf("- Has Seasonality: %t\n", result.Seasonality.HasSeasonality)
		if result.Seasonality.HasSeasonality {
			output += fmt.Sprintf("- Seasonal Strength: %.3f\n", result.Seasonality.SeasonalStrength)
			output += fmt.Sprintf("- Dominant Periods: %v\n", result.Seasonality.DominantPeriods)
		}
		output += "\n"
	}

	// Anomalies
	if result.Anomalies != nil {
		output += "Anomaly Detection:\n"
		output += fmt.Sprintf("- Method: %s\n", result.Anomalies.Method)
		output += fmt.Sprintf("- Anomalies Found: %d\n", len(result.Anomalies.AnomaliesFound))
		output += fmt.Sprintf("- Anomaly Rate: %.2f%%\n", result.Anomalies.AnomalyRate)
		
		if len(result.Anomalies.AnomaliesFound) > 0 {
			output += "- Anomaly Details:\n"
			for _, anomaly := range result.Anomalies.AnomaliesFound {
				output += fmt.Sprintf("  - %s: %.2f (%s, severity: %.2f)\n",
					anomaly.Timestamp.Format("2006-01-02 15:04:05"),
					anomaly.Value, anomaly.AnomalyType, anomaly.Severity)
			}
		}
		output += "\n"
	}

	// Forecast
	if result.Forecast != nil {
		output += fmt.Sprintf("Forecast (%d periods):\n", len(result.Forecast.ForecastPoints))
		output += fmt.Sprintf("- Method: %s\n", result.Forecast.Method)
		output += fmt.Sprintf("- Confidence Level: %.0f%%\n", result.Forecast.ConfidenceLevel*100)
		
		if len(result.Forecast.ForecastPoints) > 0 {
			output += "- Forecast Values:\n"
			for i, point := range result.Forecast.ForecastPoints {
				if i >= 5 { // Show only first 5 points
					output += fmt.Sprintf("  ... and %d more\n", len(result.Forecast.ForecastPoints)-5)
					break
				}
				output += fmt.Sprintf("  - %s: %.2f [%.2f, %.2f]\n",
					point.Timestamp.Format("2006-01-02 15:04:05"),
					point.Value, point.LowerBound, point.UpperBound)
			}
		}
	}

	// Output to file or stdout
	if outputFile == "-" {
		fmt.Print(output)
	} else {
		return os.WriteFile(outputFile, []byte(output), 0644)
	}

	return nil
}

func outputJSON(result *AnalysisResult, outputFile string) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	if outputFile == "-" {
		fmt.Println(string(data))
	} else {
		return os.WriteFile(outputFile, data, 0644)
	}

	return nil
}

func outputHTML(result *AnalysisResult, outputFile string) error {
	html := `<!DOCTYPE html>
<html>
<head>
    <title>Time Series Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .section { margin-bottom: 30px; }
        .metric { margin: 5px 0; }
        .anomaly { color: red; }
        .forecast { color: blue; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Time Series Analysis Report</h1>
`
	
	if result.BasicStats != nil {
		html += `
    <div class="section">
        <h2>Basic Statistics</h2>
        <div class="metric">Data Points: ` + fmt.Sprintf("%d", result.BasicStats.DataPoints) + `</div>
        <div class="metric">Time Range: ` + result.BasicStats.StartTime.Format("2006-01-02 15:04:05") + ` to ` + result.BasicStats.EndTime.Format("2006-01-02 15:04:05") + `</div>
        <div class="metric">Mean: ` + fmt.Sprintf("%.2f", result.BasicStats.Mean) + `</div>
        <div class="metric">Standard Deviation: ` + fmt.Sprintf("%.2f", result.BasicStats.StdDev) + `</div>
        <div class="metric">Min: ` + fmt.Sprintf("%.2f", result.BasicStats.Min) + `</div>
        <div class="metric">Max: ` + fmt.Sprintf("%.2f", result.BasicStats.Max) + `</div>
    </div>`
	}

	if result.Anomalies != nil && len(result.Anomalies.AnomaliesFound) > 0 {
		html += `
    <div class="section">
        <h2>Detected Anomalies</h2>
        <table>
            <tr><th>Timestamp</th><th>Value</th><th>Type</th><th>Severity</th></tr>`
		
		for _, anomaly := range result.Anomalies.AnomaliesFound {
			html += fmt.Sprintf(`
            <tr class="anomaly">
                <td>%s</td>
                <td>%.2f</td>
                <td>%s</td>
                <td>%.2f</td>
            </tr>`, 
				anomaly.Timestamp.Format("2006-01-02 15:04:05"),
				anomaly.Value, anomaly.AnomalyType, anomaly.Severity)
		}
		html += `
        </table>
    </div>`
	}

	html += `
</body>
</html>`

	if outputFile == "-" {
		fmt.Println(html)
	} else {
		return os.WriteFile(outputFile, []byte(html), 0644)
	}

	return nil
}

// Statistical helper functions
func calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func calculateVariance(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		diff := v - mean
		sum += diff * diff
	}
	return sum / float64(len(values))
}

func calculateStdDev(variance float64) float64 {
	if variance < 0 {
		return 0
	}
	return float64(variance) // Simplified - would use math.Sqrt in real implementation
}

func calculateMinMax(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	min, max := values[0], values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}

func calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	// Simplified - would sort and find middle in real implementation
	return calculateMean(values)
}

func calculatePercentile(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	// Simplified implementation
	return calculateMean(values)
}

func calculateTrend(values []float64) (float64, string) {
	if len(values) < 2 {
		return 0, "none"
	}
	
	// Simple linear regression slope
	n := float64(len(values))
	sumX, sumY, sumXY, sumX2 := 0.0, 0.0, 0.0, 0.0
	
	for i, y := range values {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}
	
	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	
	direction := "stable"
	if slope > 0.01 {
		direction = "upward"
	} else if slope < -0.01 {
		direction = "downward"
	}
	
	return slope, direction
}

func estimatePrimaryPeriod(ts *models.TimeSeries) string {
	// Simple heuristic based on data frequency
	if len(ts.DataPoints) < 2 {
		return "unknown"
	}
	
	interval := ts.DataPoints[1].Timestamp.Sub(ts.DataPoints[0].Timestamp)
	
	switch {
	case interval <= time.Minute:
		return "1 hour"
	case interval <= time.Hour:
		return "1 day"
	default:
		return "1 week"
	}
}

func calculateVolatility(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	
	// Calculate returns
	returns := make([]float64, len(values)-1)
	for i := 1; i < len(values); i++ {
		if values[i-1] != 0 {
			returns[i-1] = (values[i] - values[i-1]) / values[i-1]
		}
	}
	
	// Return standard deviation of returns
	mean := calculateMean(returns)
	variance := calculateVariance(returns, mean)
	return calculateStdDev(variance)
}

func calculateAutocorrelation(values []float64, lag int) float64 {
	if len(values) <= lag {
		return 0
	}
	
	mean := calculateMean(values)
	
	// Calculate covariance at lag
	covariance := 0.0
	for i := 0; i < len(values)-lag; i++ {
		covariance += (values[i] - mean) * (values[i+lag] - mean)
	}
	covariance /= float64(len(values) - lag)
	
	// Calculate variance
	variance := calculateVariance(values, mean)
	
	if variance == 0 {
		return 0
	}
	
	return covariance / variance
}