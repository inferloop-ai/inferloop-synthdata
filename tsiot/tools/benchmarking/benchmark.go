package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v2"

	"github.com/inferloop/tsiot/internal/generators"
	"github.com/inferloop/tsiot/internal/protocols/grpc/client"
	"github.com/inferloop/tsiot/internal/validation"
	"github.com/inferloop/tsiot/pkg/models"
)

type BenchmarkConfig struct {
	Name            string                 `json:"name" yaml:"name"`
	Description     string                 `json:"description" yaml:"description"`
	Duration        time.Duration          `json:"duration" yaml:"duration"`
	Warmup          time.Duration          `json:"warmup" yaml:"warmup"`
	Concurrency     int                    `json:"concurrency" yaml:"concurrency"`
	Operations      []OperationConfig      `json:"operations" yaml:"operations"`
	DataConfig      DataGenerationConfig   `json:"data_config" yaml:"data_config"`
	TargetConfig    TargetConfig           `json:"target_config" yaml:"target_config"`
	MetricsConfig   MetricsConfig          `json:"metrics_config" yaml:"metrics_config"`
	ReportConfig    ReportConfig           `json:"report_config" yaml:"report_config"`
}

type OperationConfig struct {
	Name        string                 `json:"name" yaml:"name"`
	Type        string                 `json:"type" yaml:"type"`
	Weight      int                    `json:"weight" yaml:"weight"`
	Parameters  map[string]interface{} `json:"parameters" yaml:"parameters"`
}

type DataGenerationConfig struct {
	NumSeries       int                    `json:"num_series" yaml:"num_series"`
	PointsPerSeries int                    `json:"points_per_series" yaml:"points_per_series"`
	BatchSize       int                    `json:"batch_size" yaml:"batch_size"`
	GeneratorType   string                 `json:"generator_type" yaml:"generator_type"`
	GeneratorConfig map[string]interface{} `json:"generator_config" yaml:"generator_config"`
}

type TargetConfig struct {
	Type        string                 `json:"type" yaml:"type"` // grpc, http, kafka, mqtt
	Endpoint    string                 `json:"endpoint" yaml:"endpoint"`
	Timeout     time.Duration          `json:"timeout" yaml:"timeout"`
	MaxRetries  int                    `json:"max_retries" yaml:"max_retries"`
	TLS         bool                   `json:"tls" yaml:"tls"`
	AuthConfig  map[string]string      `json:"auth_config" yaml:"auth_config"`
}

type MetricsConfig struct {
	CollectInterval time.Duration          `json:"collect_interval" yaml:"collect_interval"`
	Percentiles     []float64              `json:"percentiles" yaml:"percentiles"`
	Histogram       bool                   `json:"histogram" yaml:"histogram"`
	DetailedErrors  bool                   `json:"detailed_errors" yaml:"detailed_errors"`
}

type ReportConfig struct {
	Format      string   `json:"format" yaml:"format"` // json, yaml, html, markdown
	OutputFile  string   `json:"output_file" yaml:"output_file"`
	IncludeRaw  bool     `json:"include_raw" yaml:"include_raw"`
	Comparisons []string `json:"comparisons" yaml:"comparisons"`
}

type Benchmark struct {
	config      *BenchmarkConfig
	logger      *logrus.Logger
	client      interface{} // Protocol-specific client
	generator   generators.Generator
	metrics     *MetricsCollector
	operations  []Operation
	startTime   time.Time
	endTime     time.Time
}

type Operation interface {
	Execute(ctx context.Context, data interface{}) error
	Name() string
}

type MetricsCollector struct {
	mu              sync.RWMutex
	operations      map[string]*OperationMetrics
	totalOps        int64
	totalErrors     int64
	totalBytes      int64
	systemMetrics   []SystemMetrics
	collectionStart time.Time
}

type OperationMetrics struct {
	Count       int64
	Errors      int64
	Latencies   []time.Duration
	Bytes       int64
	ErrorTypes  map[string]int64
}

type SystemMetrics struct {
	Timestamp   time.Time
	CPUUsage    float64
	MemoryUsage uint64
	Goroutines  int
	GCPauses    uint64
	GCDuration  time.Duration
}

type BenchmarkResult struct {
	Name           string                          `json:"name"`
	Description    string                          `json:"description"`
	Duration       time.Duration                   `json:"duration"`
	StartTime      time.Time                       `json:"start_time"`
	EndTime        time.Time                       `json:"end_time"`
	TotalOps       int64                           `json:"total_operations"`
	TotalErrors    int64                           `json:"total_errors"`
	TotalBytes     int64                           `json:"total_bytes"`
	Throughput     float64                         `json:"throughput_ops_sec"`
	ErrorRate      float64                         `json:"error_rate"`
	Operations     map[string]*OperationResult     `json:"operations"`
	SystemMetrics  *SystemMetricsSummary           `json:"system_metrics"`
}

type OperationResult struct {
	Name           string                          `json:"name"`
	Count          int64                           `json:"count"`
	Errors         int64                           `json:"errors"`
	ErrorRate      float64                         `json:"error_rate"`
	Throughput     float64                         `json:"throughput_ops_sec"`
	Latency        *LatencyMetrics                 `json:"latency"`
	BytesProcessed int64                           `json:"bytes_processed"`
	ErrorBreakdown map[string]int64                `json:"error_breakdown,omitempty"`
}

type LatencyMetrics struct {
	Min         time.Duration                   `json:"min"`
	Max         time.Duration                   `json:"max"`
	Mean        time.Duration                   `json:"mean"`
	StdDev      time.Duration                   `json:"std_dev"`
	Median      time.Duration                   `json:"median"`
	Percentiles map[string]time.Duration        `json:"percentiles"`
}

type SystemMetricsSummary struct {
	AvgCPUUsage    float64                        `json:"avg_cpu_usage"`
	MaxCPUUsage    float64                        `json:"max_cpu_usage"`
	AvgMemoryMB    float64                        `json:"avg_memory_mb"`
	MaxMemoryMB    float64                        `json:"max_memory_mb"`
	AvgGoroutines  float64                        `json:"avg_goroutines"`
	MaxGoroutines  int                            `json:"max_goroutines"`
	TotalGCPauses  uint64                         `json:"total_gc_pauses"`
	TotalGCTime    time.Duration                  `json:"total_gc_time"`
}

func main() {
	var (
		configFile   = flag.String("config", "", "Benchmark configuration file")
		name         = flag.String("name", "default", "Benchmark name")
		duration     = flag.Duration("duration", 60*time.Second, "Benchmark duration")
		warmup       = flag.Duration("warmup", 10*time.Second, "Warmup duration")
		concurrency  = flag.Int("concurrency", 10, "Number of concurrent workers")
		operation    = flag.String("operation", "mixed", "Operation type: insert, query, mixed")
		target       = flag.String("target", "localhost:50051", "Target endpoint")
		outputFormat = flag.String("format", "json", "Output format: json, yaml, html, markdown")
		outputFile   = flag.String("output", "", "Output file (default: stdout)")
		verbose      = flag.Bool("verbose", false, "Enable verbose logging")
	)
	flag.Parse()

	// Setup logging
	logger := logrus.New()
	if *verbose {
		logger.SetLevel(logrus.DebugLevel)
	}

	// Load configuration
	var config *BenchmarkConfig
	if *configFile != "" {
		var err error
		config, err = loadConfig(*configFile)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}
	} else {
		// Create default configuration from flags
		config = &BenchmarkConfig{
			Name:        *name,
			Duration:    *duration,
			Warmup:      *warmup,
			Concurrency: *concurrency,
			Operations: []OperationConfig{
				{
					Name:   *operation,
					Type:   *operation,
					Weight: 100,
				},
			},
			DataConfig: DataGenerationConfig{
				NumSeries:       100,
				PointsPerSeries: 1000,
				BatchSize:       100,
				GeneratorType:   "synthetic",
			},
			TargetConfig: TargetConfig{
				Type:     "grpc",
				Endpoint: *target,
				Timeout:  30 * time.Second,
			},
			MetricsConfig: MetricsConfig{
				CollectInterval: 1 * time.Second,
				Percentiles:     []float64{0.5, 0.9, 0.95, 0.99},
				Histogram:       true,
				DetailedErrors:  true,
			},
			ReportConfig: ReportConfig{
				Format:     *outputFormat,
				OutputFile: *outputFile,
			},
		}
	}

	// Create and run benchmark
	benchmark := NewBenchmark(config, logger)
	
	ctx := context.Background()
	result, err := benchmark.Run(ctx)
	if err != nil {
		log.Fatalf("Benchmark failed: %v", err)
	}

	// Generate report
	if err := generateReport(result, config.ReportConfig); err != nil {
		log.Fatalf("Failed to generate report: %v", err)
	}
}

func NewBenchmark(config *BenchmarkConfig, logger *logrus.Logger) *Benchmark {
	b := &Benchmark{
		config:  config,
		logger:  logger,
		metrics: NewMetricsCollector(),
	}

	// Initialize client based on target type
	switch config.TargetConfig.Type {
	case "grpc":
		// Initialize gRPC client
		// b.client = grpc.NewClient(config.TargetConfig)
	case "http":
		// Initialize HTTP client
	case "kafka":
		// Initialize Kafka client
	case "mqtt":
		// Initialize MQTT client
	default:
		log.Fatalf("Unknown target type: %s", config.TargetConfig.Type)
	}

	// Initialize data generator
	// b.generator = generators.NewGenerator(config.DataConfig)

	// Create operations
	b.createOperations()

	return b
}

func (b *Benchmark) Run(ctx context.Context) (*BenchmarkResult, error) {
	b.logger.WithFields(logrus.Fields{
		"name":        b.config.Name,
		"duration":    b.config.Duration,
		"concurrency": b.config.Concurrency,
	}).Info("Starting benchmark")

	// Start metrics collection
	metricsCtx, cancelMetrics := context.WithCancel(ctx)
	defer cancelMetrics()
	go b.collectSystemMetrics(metricsCtx)

	// Warmup phase
	if b.config.Warmup > 0 {
		b.logger.WithField("duration", b.config.Warmup).Info("Starting warmup")
		warmupCtx, cancel := context.WithTimeout(ctx, b.config.Warmup)
		b.runWorkers(warmupCtx, b.config.Concurrency/2)
		cancel()
		
		// Reset metrics after warmup
		b.metrics.Reset()
	}

	// Main benchmark phase
	b.startTime = time.Now()
	benchCtx, cancel := context.WithTimeout(ctx, b.config.Duration)
	defer cancel()

	b.runWorkers(benchCtx, b.config.Concurrency)

	b.endTime = time.Now()
	
	// Generate result
	result := b.generateResult()
	
	b.logger.WithFields(logrus.Fields{
		"total_operations": result.TotalOps,
		"total_errors":     result.TotalErrors,
		"throughput":       result.Throughput,
		"error_rate":       result.ErrorRate,
	}).Info("Benchmark completed")

	return result, nil
}

func (b *Benchmark) runWorkers(ctx context.Context, numWorkers int) {
	var wg sync.WaitGroup
	
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			b.worker(ctx, workerID)
		}(i)
	}
	
	wg.Wait()
}

func (b *Benchmark) worker(ctx context.Context, workerID int) {
	// Create worker-specific random generator
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerID)))
	
	for {
		select {
		case <-ctx.Done():
			return
		default:
			// Select operation based on weights
			op := b.selectOperation(rng)
			
			// Generate data
			data := b.generateData()
			
			// Execute operation
			start := time.Now()
			err := op.Execute(ctx, data)
			duration := time.Since(start)
			
			// Record metrics
			b.metrics.RecordOperation(op.Name(), duration, err, b.getDataSize(data))
		}
	}
}

func (b *Benchmark) selectOperation(rng *rand.Rand) Operation {
	// Calculate total weight
	totalWeight := 0
	for _, opConfig := range b.config.Operations {
		totalWeight += opConfig.Weight
	}
	
	// Select based on weight
	r := rng.Intn(totalWeight)
	cumWeight := 0
	
	for i, opConfig := range b.config.Operations {
		cumWeight += opConfig.Weight
		if r < cumWeight {
			return b.operations[i]
		}
	}
	
	return b.operations[0] // Fallback
}

func (b *Benchmark) generateData() interface{} {
	// Generate synthetic time series data
	// This is a placeholder implementation
	return &models.TimeSeries{
		ID:         fmt.Sprintf("series_%d", rand.Int63()),
		Name:       "Benchmark Series",
		SensorType: "temperature",
		DataPoints: generateDataPoints(b.config.DataConfig.PointsPerSeries),
	}
}

func (b *Benchmark) getDataSize(data interface{}) int64 {
	// Calculate approximate data size
	// This is a simplified implementation
	if ts, ok := data.(*models.TimeSeries); ok {
		return int64(len(ts.DataPoints) * 16) // Approximate bytes per point
	}
	return 0
}

func (b *Benchmark) createOperations() {
	b.operations = make([]Operation, len(b.config.Operations))
	
	for i, opConfig := range b.config.Operations {
		switch opConfig.Type {
		case "insert":
			b.operations[i] = &InsertOperation{
				name:   opConfig.Name,
				client: b.client,
				params: opConfig.Parameters,
			}
		case "query":
			b.operations[i] = &QueryOperation{
				name:   opConfig.Name,
				client: b.client,
				params: opConfig.Parameters,
			}
		case "aggregate":
			b.operations[i] = &AggregateOperation{
				name:   opConfig.Name,
				client: b.client,
				params: opConfig.Parameters,
			}
		default:
			// Create a mixed operation
			b.operations[i] = &MixedOperation{
				name:   opConfig.Name,
				client: b.client,
				params: opConfig.Parameters,
			}
		}
	}
}

func (b *Benchmark) collectSystemMetrics(ctx context.Context) {
	ticker := time.NewTicker(b.config.MetricsConfig.CollectInterval)
	defer ticker.Stop()
	
	var lastGCStats runtime.MemStats
	runtime.ReadMemStats(&lastGCStats)
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			var memStats runtime.MemStats
			runtime.ReadMemStats(&memStats)
			
			metrics := SystemMetrics{
				Timestamp:   time.Now(),
				CPUUsage:    getCPUUsage(),
				MemoryUsage: memStats.Alloc,
				Goroutines:  runtime.NumGoroutine(),
				GCPauses:    memStats.NumGC - lastGCStats.NumGC,
				GCDuration:  time.Duration(memStats.PauseTotalNs - lastGCStats.PauseTotalNs),
			}
			
			b.metrics.RecordSystemMetrics(metrics)
			lastGCStats = memStats
		}
	}
}

func (b *Benchmark) generateResult() *BenchmarkResult {
	duration := b.endTime.Sub(b.startTime)
	
	result := &BenchmarkResult{
		Name:        b.config.Name,
		Description: b.config.Description,
		Duration:    duration,
		StartTime:   b.startTime,
		EndTime:     b.endTime,
		TotalOps:    atomic.LoadInt64(&b.metrics.totalOps),
		TotalErrors: atomic.LoadInt64(&b.metrics.totalErrors),
		TotalBytes:  atomic.LoadInt64(&b.metrics.totalBytes),
		Operations:  make(map[string]*OperationResult),
	}
	
	result.Throughput = float64(result.TotalOps) / duration.Seconds()
	result.ErrorRate = float64(result.TotalErrors) / float64(result.TotalOps)
	
	// Generate operation results
	b.metrics.mu.RLock()
	for name, metrics := range b.metrics.operations {
		opResult := &OperationResult{
			Name:           name,
			Count:          metrics.Count,
			Errors:         metrics.Errors,
			ErrorRate:      float64(metrics.Errors) / float64(metrics.Count),
			Throughput:     float64(metrics.Count) / duration.Seconds(),
			BytesProcessed: metrics.Bytes,
			ErrorBreakdown: metrics.ErrorTypes,
		}
		
		// Calculate latency metrics
		if len(metrics.Latencies) > 0 {
			opResult.Latency = calculateLatencyMetrics(metrics.Latencies, b.config.MetricsConfig.Percentiles)
		}
		
		result.Operations[name] = opResult
	}
	
	// Calculate system metrics summary
	result.SystemMetrics = b.calculateSystemMetricsSummary()
	b.metrics.mu.RUnlock()
	
	return result
}

func (b *Benchmark) calculateSystemMetricsSummary() *SystemMetricsSummary {
	if len(b.metrics.systemMetrics) == 0 {
		return &SystemMetricsSummary{}
	}
	
	summary := &SystemMetricsSummary{}
	
	var totalCPU float64
	var totalMemory uint64
	var totalGoroutines int
	
	for _, m := range b.metrics.systemMetrics {
		totalCPU += m.CPUUsage
		totalMemory += m.MemoryUsage
		totalGoroutines += m.Goroutines
		
		if m.CPUUsage > summary.MaxCPUUsage {
			summary.MaxCPUUsage = m.CPUUsage
		}
		if float64(m.MemoryUsage) > summary.MaxMemoryMB*1024*1024 {
			summary.MaxMemoryMB = float64(m.MemoryUsage) / (1024 * 1024)
		}
		if m.Goroutines > summary.MaxGoroutines {
			summary.MaxGoroutines = m.Goroutines
		}
		
		summary.TotalGCPauses += m.GCPauses
		summary.TotalGCTime += m.GCDuration
	}
	
	n := float64(len(b.metrics.systemMetrics))
	summary.AvgCPUUsage = totalCPU / n
	summary.AvgMemoryMB = float64(totalMemory) / n / (1024 * 1024)
	summary.AvgGoroutines = float64(totalGoroutines) / n
	
	return summary
}

// MetricsCollector implementation

func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		operations:      make(map[string]*OperationMetrics),
		collectionStart: time.Now(),
	}
}

func (m *MetricsCollector) RecordOperation(name string, latency time.Duration, err error, bytes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	atomic.AddInt64(&m.totalOps, 1)
	atomic.AddInt64(&m.totalBytes, bytes)
	
	if _, exists := m.operations[name]; !exists {
		m.operations[name] = &OperationMetrics{
			ErrorTypes: make(map[string]int64),
		}
	}
	
	op := m.operations[name]
	op.Count++
	op.Bytes += bytes
	op.Latencies = append(op.Latencies, latency)
	
	if err != nil {
		atomic.AddInt64(&m.totalErrors, 1)
		op.Errors++
		
		errType := fmt.Sprintf("%T", err)
		op.ErrorTypes[errType]++
	}
}

func (m *MetricsCollector) RecordSystemMetrics(metrics SystemMetrics) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.systemMetrics = append(m.systemMetrics, metrics)
}

func (m *MetricsCollector) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.operations = make(map[string]*OperationMetrics)
	m.totalOps = 0
	m.totalErrors = 0
	m.totalBytes = 0
	m.systemMetrics = nil
	m.collectionStart = time.Now()
}

// Operation implementations

type InsertOperation struct {
	name   string
	client interface{}
	params map[string]interface{}
}

func (o *InsertOperation) Name() string {
	return o.name
}

func (o *InsertOperation) Execute(ctx context.Context, data interface{}) error {
	// Implement insert logic based on client type
	// This is a placeholder
	return nil
}

type QueryOperation struct {
	name   string
	client interface{}
	params map[string]interface{}
}

func (o *QueryOperation) Name() string {
	return o.name
}

func (o *QueryOperation) Execute(ctx context.Context, data interface{}) error {
	// Implement query logic based on client type
	// This is a placeholder
	return nil
}

type AggregateOperation struct {
	name   string
	client interface{}
	params map[string]interface{}
}

func (o *AggregateOperation) Name() string {
	return o.name
}

func (o *AggregateOperation) Execute(ctx context.Context, data interface{}) error {
	// Implement aggregation logic based on client type
	// This is a placeholder
	return nil
}

type MixedOperation struct {
	name   string
	client interface{}
	params map[string]interface{}
}

func (o *MixedOperation) Name() string {
	return o.name
}

func (o *MixedOperation) Execute(ctx context.Context, data interface{}) error {
	// Randomly select between insert and query
	if rand.Intn(2) == 0 {
		return (&InsertOperation{name: o.name, client: o.client, params: o.params}).Execute(ctx, data)
	}
	return (&QueryOperation{name: o.name, client: o.client, params: o.params}).Execute(ctx, data)
}

// Helper functions

func loadConfig(path string) (*BenchmarkConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	
	var config BenchmarkConfig
	if strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml") {
		err = yaml.Unmarshal(data, &config)
	} else {
		err = json.Unmarshal(data, &config)
	}
	
	return &config, err
}

func generateReport(result *BenchmarkResult, config ReportConfig) error {
	var output []byte
	var err error
	
	switch config.Format {
	case "json":
		output, err = json.MarshalIndent(result, "", "  ")
	case "yaml":
		output, err = yaml.Marshal(result)
	case "markdown":
		output = []byte(generateMarkdownReport(result))
	case "html":
		output = []byte(generateHTMLReport(result))
	default:
		return fmt.Errorf("unsupported format: %s", config.Format)
	}
	
	if err != nil {
		return err
	}
	
	if config.OutputFile != "" {
		return os.WriteFile(config.OutputFile, output, 0644)
	}
	
	fmt.Println(string(output))
	return nil
}

func generateDataPoints(count int) []models.DataPoint {
	points := make([]models.DataPoint, count)
	now := time.Now()
	
	for i := 0; i < count; i++ {
		points[i] = models.DataPoint{
			Timestamp: now.Add(time.Duration(i) * time.Second),
			Value:     rand.Float64() * 100,
			Quality:   0.95,
		}
	}
	
	return points
}

func calculateLatencyMetrics(latencies []time.Duration, percentiles []float64) *LatencyMetrics {
	if len(latencies) == 0 {
		return &LatencyMetrics{}
	}
	
	// Sort latencies
	sortedLatencies := make([]time.Duration, len(latencies))
	copy(sortedLatencies, latencies)
	
	// Simple bubble sort for demonstration
	for i := 0; i < len(sortedLatencies); i++ {
		for j := i + 1; j < len(sortedLatencies); j++ {
			if sortedLatencies[i] > sortedLatencies[j] {
				sortedLatencies[i], sortedLatencies[j] = sortedLatencies[j], sortedLatencies[i]
			}
		}
	}
	
	// Calculate basic metrics
	min := sortedLatencies[0]
	max := sortedLatencies[len(sortedLatencies)-1]
	
	var sum time.Duration
	for _, l := range latencies {
		sum += l
	}
	mean := sum / time.Duration(len(latencies))
	
	// Calculate percentiles
	percentilesMap := make(map[string]time.Duration)
	for _, p := range percentiles {
		index := int(float64(len(sortedLatencies)-1) * p)
		percentilesMap[fmt.Sprintf("p%.0f", p*100)] = sortedLatencies[index]
	}
	
	// Median is 50th percentile
	median := sortedLatencies[len(sortedLatencies)/2]
	
	// Calculate standard deviation
	var variance float64
	for _, l := range latencies {
		diff := float64(l - mean)
		variance += diff * diff
	}
	variance /= float64(len(latencies))
	stdDev := time.Duration(math.Sqrt(variance))
	
	return &LatencyMetrics{
		Min:         min,
		Max:         max,
		Mean:        mean,
		StdDev:      stdDev,
		Median:      median,
		Percentiles: percentilesMap,
	}
}

func getCPUUsage() float64 {
	// Placeholder for CPU usage calculation
	// In a real implementation, you would use system-specific calls
	return rand.Float64() * 100
}

func generateMarkdownReport(result *BenchmarkResult) string {
	var report strings.Builder
	
	report.WriteString(fmt.Sprintf("# Benchmark Report: %s\n\n", result.Name))
	if result.Description != "" {
		report.WriteString(fmt.Sprintf("%s\n\n", result.Description))
	}
	
	report.WriteString("## Summary\n\n")
	report.WriteString(fmt.Sprintf("- **Duration**: %s\n", result.Duration))
	report.WriteString(fmt.Sprintf("- **Total Operations**: %d\n", result.TotalOps))
	report.WriteString(fmt.Sprintf("- **Total Errors**: %d (%.2f%%)\n", result.TotalErrors, result.ErrorRate*100))
	report.WriteString(fmt.Sprintf("- **Throughput**: %.2f ops/sec\n", result.Throughput))
	report.WriteString(fmt.Sprintf("- **Total Data**: %d bytes\n\n", result.TotalBytes))
	
	report.WriteString("## Operations\n\n")
	for name, op := range result.Operations {
		report.WriteString(fmt.Sprintf("### %s\n\n", name))
		report.WriteString(fmt.Sprintf("- **Count**: %d\n", op.Count))
		report.WriteString(fmt.Sprintf("- **Errors**: %d (%.2f%%)\n", op.Errors, op.ErrorRate*100))
		report.WriteString(fmt.Sprintf("- **Throughput**: %.2f ops/sec\n", op.Throughput))
		
		if op.Latency != nil {
			report.WriteString("\n**Latency**:\n")
			report.WriteString(fmt.Sprintf("- Min: %s\n", op.Latency.Min))
			report.WriteString(fmt.Sprintf("- Max: %s\n", op.Latency.Max))
			report.WriteString(fmt.Sprintf("- Mean: %s\n", op.Latency.Mean))
			report.WriteString(fmt.Sprintf("- Median: %s\n", op.Latency.Median))
			
			if len(op.Latency.Percentiles) > 0 {
				report.WriteString("\n**Percentiles**:\n")
				for k, v := range op.Latency.Percentiles {
					report.WriteString(fmt.Sprintf("- %s: %s\n", k, v))
				}
			}
		}
		report.WriteString("\n")
	}
	
	if result.SystemMetrics != nil {
		report.WriteString("## System Metrics\n\n")
		report.WriteString(fmt.Sprintf("- **Avg CPU**: %.2f%%\n", result.SystemMetrics.AvgCPUUsage))
		report.WriteString(fmt.Sprintf("- **Max CPU**: %.2f%%\n", result.SystemMetrics.MaxCPUUsage))
		report.WriteString(fmt.Sprintf("- **Avg Memory**: %.2f MB\n", result.SystemMetrics.AvgMemoryMB))
		report.WriteString(fmt.Sprintf("- **Max Memory**: %.2f MB\n", result.SystemMetrics.MaxMemoryMB))
		report.WriteString(fmt.Sprintf("- **Avg Goroutines**: %.0f\n", result.SystemMetrics.AvgGoroutines))
		report.WriteString(fmt.Sprintf("- **Max Goroutines**: %d\n", result.SystemMetrics.MaxGoroutines))
	}
	
	return report.String()
}

func generateHTMLReport(result *BenchmarkResult) string {
	// Simplified HTML report
	return fmt.Sprintf(`
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report: %s</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Benchmark Report: %s</h1>
    <p>Duration: %s | Total Operations: %d | Throughput: %.2f ops/sec</p>
    <!-- Add more detailed HTML content here -->
</body>
</html>
`, result.Name, result.Name, result.Duration, result.TotalOps, result.Throughput)
}

func math.Sqrt(x float64) float64 {
	// Simple square root implementation
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}
	
	// Newton's method
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}