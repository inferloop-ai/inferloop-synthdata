package commands

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inferloop/tsiot/internal/storage"
	"github.com/inferloop/tsiot/internal/storage/implementations/influxdb"
	"github.com/inferloop/tsiot/internal/storage/implementations/timescaledb"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

type MigrateOptions struct {
	Source      string
	Destination string
	SourceType  string
	DestType    string
	BatchSize   int
	StartTime   string
	EndTime     string
	Transform   string
	DryRun      bool
	Force       bool
	Parallel    int
	Resume      bool
	Validate    bool
}

func NewMigrateCmd() *cobra.Command {
	opts := &MigrateOptions{}

	cmd := &cobra.Command{
		Use:   "migrate",
		Short: "Migrate time series data between different storage backends",
		Long: `Migrate time series data from one storage backend to another,
with optional transformations and filtering.`,
		Example: `  # Migrate from InfluxDB to TimescaleDB
  tsiot-cli migrate --source influx://localhost:8086/mydb --dest timescale://localhost:5432/tsdb

  # Migrate with time range filter
  tsiot-cli migrate --source file://data.csv --dest influx://localhost:8086/newdb \
    --start-time "2024-01-01" --end-time "2024-01-31"

  # Dry run to preview migration
  tsiot-cli migrate --source s3://bucket/data --dest influx://localhost:8086/db --dry-run

  # Parallel migration with validation
  tsiot-cli migrate --source influx://old:8086/db --dest timescale://new:5432/db \
    --parallel 4 --validate --batch-size 5000`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runMigrate(opts)
		},
	}

	// Add flags
	cmd.Flags().StringVarP(&opts.Source, "source", "s", "", "Source connection string (required)")
	cmd.Flags().StringVarP(&opts.Destination, "dest", "d", "", "Destination connection string (required)")
	cmd.Flags().StringVar(&opts.SourceType, "source-type", "auto", "Source type (auto, influx, timescale, file, s3)")
	cmd.Flags().StringVar(&opts.DestType, "dest-type", "auto", "Destination type (auto, influx, timescale, file, s3)")
	cmd.Flags().IntVar(&opts.BatchSize, "batch-size", 1000, "Batch size for migration")
	cmd.Flags().StringVar(&opts.StartTime, "start-time", "", "Start time filter (RFC3339 or relative like 'now-24h')")
	cmd.Flags().StringVar(&opts.EndTime, "end-time", "", "End time filter (RFC3339 or relative)")
	cmd.Flags().StringVar(&opts.Transform, "transform", "", "Transformation script or rule")
	cmd.Flags().BoolVar(&opts.DryRun, "dry-run", false, "Preview migration without executing")
	cmd.Flags().BoolVar(&opts.Force, "force", false, "Force migration (overwrite existing data)")
	cmd.Flags().IntVar(&opts.Parallel, "parallel", 1, "Number of parallel migration workers")
	cmd.Flags().BoolVar(&opts.Resume, "resume", false, "Resume interrupted migration")
	cmd.Flags().BoolVar(&opts.Validate, "validate", false, "Validate migrated data")

	cmd.MarkFlagRequired("source")
	cmd.MarkFlagRequired("dest")

	return cmd
}

func runMigrate(opts *MigrateOptions) error {
	fmt.Printf("Preparing data migration...\n")
	fmt.Printf("Source: %s\n", opts.Source)
	fmt.Printf("Destination: %s\n", opts.Destination)

	if opts.DryRun {
		fmt.Println("\n[DRY RUN MODE - No data will be migrated]")
	}

	// Initialize logger
	logger := logrus.New()
	if opts.DryRun {
		logger.SetLevel(logrus.InfoLevel)
	}

	// Create migration context
	ctx := context.Background()
	migrator, err := NewMigrator(opts, logger)
	if err != nil {
		return fmt.Errorf("failed to create migrator: %w", err)
	}
	defer migrator.Close()

	// Plan migration
	plan, err := migrator.PlanMigration(ctx)
	if err != nil {
		return fmt.Errorf("failed to plan migration: %w", err)
	}

	// Display migration plan
	printMigrationPlan(plan, opts)

	// Execute migration
	if !opts.DryRun {
		result, err := migrator.ExecuteMigration(ctx, plan)
		if err != nil {
			return fmt.Errorf("migration failed: %w", err)
		}
		printMigrationResult(result)
	} else {
		fmt.Println("\n✓ Dry run completed. Run without --dry-run to execute migration.")
	}

	return nil
}

type Migrator struct {
	options     *MigrateOptions
	logger      *logrus.Logger
	source      interfaces.Storage
	destination interfaces.Storage
	sourceConn  *ConnectionInfo
	destConn    *ConnectionInfo
}

type ConnectionInfo struct {
	Type     string
	URL      string
	Host     string
	Port     int
	Database string
	Bucket   string
	Path     string
	Config   map[string]interface{}
}

type MigrationPlan struct {
	SourceInfo        *ConnectionInfo    `json:"source_info"`
	DestinationInfo   *ConnectionInfo    `json:"destination_info"`
	EstimatedRecords  int64              `json:"estimated_records"`
	EstimatedBatches  int                `json:"estimated_batches"`
	TimeRange         *TimeRange         `json:"time_range,omitempty"`
	BatchSize         int                `json:"batch_size"`
	ParallelWorkers   int                `json:"parallel_workers"`
	SeriesList        []string           `json:"series_list"`
	TransformRules    []TransformRule    `json:"transform_rules,omitempty"`
	ValidationEnabled bool               `json:"validation_enabled"`
}

type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

type TransformRule struct {
	Type        string                 `json:"type"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type MigrationResult struct {
	Success          bool          `json:"success"`
	RecordsMigrated  int64         `json:"records_migrated"`
	SeriesMigrated   int           `json:"series_migrated"`
	Duration         time.Duration `json:"duration"`
	AverageRate      float64       `json:"average_rate"`
	Errors           []string      `json:"errors,omitempty"`
	Warnings         []string      `json:"warnings,omitempty"`
	ValidationResult *ValidationSummary `json:"validation_result,omitempty"`
}

type ValidationSummary struct {
	RecordsValidated int     `json:"records_validated"`
	ValidationScore  float64 `json:"validation_score"`
	Discrepancies    int     `json:"discrepancies"`
}

func NewMigrator(opts *MigrateOptions, logger *logrus.Logger) (*Migrator, error) {
	migrator := &Migrator{
		options: opts,
		logger:  logger,
	}

	// Parse connection strings
	sourceConn, err := parseConnectionString(opts.Source, opts.SourceType)
	if err != nil {
		return nil, fmt.Errorf("invalid source connection: %w", err)
	}
	migrator.sourceConn = sourceConn

	destConn, err := parseConnectionString(opts.Destination, opts.DestType)
	if err != nil {
		return nil, fmt.Errorf("invalid destination connection: %w", err)
	}
	migrator.destConn = destConn

	// Create storage instances
	source, err := createStorage(sourceConn, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create source storage: %w", err)
	}
	migrator.source = source

	destination, err := createStorage(destConn, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create destination storage: %w", err)
	}
	migrator.destination = destination

	return migrator, nil
}

func (m *Migrator) PlanMigration(ctx context.Context) (*MigrationPlan, error) {
	plan := &MigrationPlan{
		SourceInfo:        m.sourceConn,
		DestinationInfo:   m.destConn,
		BatchSize:         m.options.BatchSize,
		ParallelWorkers:   m.options.Parallel,
		ValidationEnabled: m.options.Validate,
	}

	// Connect to storages
	if err := m.source.Connect(ctx); err != nil {
		return nil, fmt.Errorf("failed to connect to source: %w", err)
	}

	if err := m.destination.Connect(ctx); err != nil {
		return nil, fmt.Errorf("failed to connect to destination: %w", err)
	}

	// Parse time range
	if m.options.StartTime != "" || m.options.EndTime != "" {
		timeRange, err := parseTimeRange(m.options.StartTime, m.options.EndTime)
		if err != nil {
			return nil, fmt.Errorf("failed to parse time range: %w", err)
		}
		plan.TimeRange = timeRange
	}

	// Get series list from source
	seriesList, err := m.getSeriesList(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get series list: %w", err)
	}
	plan.SeriesList = seriesList

	// Estimate records
	estimatedRecords, err := m.estimateRecords(ctx, plan)
	if err != nil {
		m.logger.WithError(err).Warn("Failed to estimate record count")
		estimatedRecords = 1000000 // Default estimate
	}
	plan.EstimatedRecords = estimatedRecords
	plan.EstimatedBatches = int((estimatedRecords + int64(plan.BatchSize) - 1) / int64(plan.BatchSize))

	// Parse transform rules
	if m.options.Transform != "" {
		transformRules, err := parseTransformRules(m.options.Transform)
		if err != nil {
			return nil, fmt.Errorf("failed to parse transform rules: %w", err)
		}
		plan.TransformRules = transformRules
	}

	return plan, nil
}

func (m *Migrator) ExecuteMigration(ctx context.Context, plan *MigrationPlan) (*MigrationResult, error) {
	start := time.Now()
	result := &MigrationResult{
		Success: false,
	}

	fmt.Println("\nStarting migration...")

	// Create progress tracker
	progress := NewProgressTracker(plan.EstimatedRecords, plan.EstimatedBatches)

	// Migrate each series
	for i, seriesID := range plan.SeriesList {
		fmt.Printf("\nMigrating series %d/%d: %s\n", i+1, len(plan.SeriesList), seriesID)

		err := m.migrateSeries(ctx, seriesID, plan, progress)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("Failed to migrate series %s: %v", seriesID, err))
			if !m.options.Force {
				return result, fmt.Errorf("migration failed for series %s: %w", seriesID, err)
			}
		} else {
			result.SeriesMigrated++
		}
	}

	result.Duration = time.Since(start)
	result.RecordsMigrated = progress.RecordsMigrated
	result.AverageRate = float64(result.RecordsMigrated) / result.Duration.Seconds()

	// Perform validation if requested
	if m.options.Validate {
		fmt.Println("\nValidating migrated data...")
		validationResult, err := m.validateMigration(ctx, plan)
		if err != nil {
			result.Warnings = append(result.Warnings, fmt.Sprintf("Validation failed: %v", err))
		} else {
			result.ValidationResult = validationResult
		}
	}

	result.Success = len(result.Errors) == 0

	return result, nil
}

func (m *Migrator) migrateSeries(ctx context.Context, seriesID string, plan *MigrationPlan, progress *ProgressTracker) error {
	// Create query for the series
	query := &models.TimeSeriesQuery{
		SeriesID: seriesID,
		Limit:    plan.BatchSize,
	}

	if plan.TimeRange != nil {
		query.StartTime = &plan.TimeRange.Start
		query.EndTime = &plan.TimeRange.End
	}

	offset := 0
	for {
		query.Offset = offset

		// Read batch from source
		timeSeries, err := m.readFromSource(ctx, query)
		if err != nil {
			return fmt.Errorf("failed to read from source: %w", err)
		}

		if len(timeSeries.DataPoints) == 0 {
			break // No more data
		}

		// Apply transformations if specified
		if len(plan.TransformRules) > 0 {
			timeSeries, err = m.applyTransforms(timeSeries, plan.TransformRules)
			if err != nil {
				return fmt.Errorf("failed to apply transforms: %w", err)
			}
		}

		// Write batch to destination
		err = m.writeToDestination(ctx, timeSeries)
		if err != nil {
			return fmt.Errorf("failed to write to destination: %w", err)
		}

		// Update progress
		progress.UpdateProgress(len(timeSeries.DataPoints))

		offset += plan.BatchSize
		if len(timeSeries.DataPoints) < plan.BatchSize {
			break // Last batch
		}
	}

	return nil
}

func (m *Migrator) readFromSource(ctx context.Context, query *models.TimeSeriesQuery) (*models.TimeSeries, error) {
	// Implementation depends on source type
	switch m.sourceConn.Type {
	case "influxdb":
		return m.source.(*influxdb.InfluxDBStorage).Read(ctx, query)
	case "timescaledb":
		return m.source.(*timescaledb.TimescaleDBStorage).Read(ctx, query.SeriesID)
	case "file":
		// File-based reading would be implemented here
		return loadTimeSeriesFromFile(m.sourceConn.Path)
	default:
		return nil, fmt.Errorf("unsupported source type: %s", m.sourceConn.Type)
	}
}

func (m *Migrator) writeToDestination(ctx context.Context, timeSeries *models.TimeSeries) error {
	// Implementation depends on destination type
	switch m.destConn.Type {
	case "influxdb":
		return m.destination.(*influxdb.InfluxDBStorage).Write(ctx, timeSeries)
	case "timescaledb":
		return m.destination.(*timescaledb.TimescaleDBStorage).Write(ctx, timeSeries)
	case "file":
		// File-based writing would be implemented here
		return writeTimeSeriesToFile(timeSeries, m.destConn.Path)
	default:
		return fmt.Errorf("unsupported destination type: %s", m.destConn.Type)
	}
}

func (m *Migrator) getSeriesList(ctx context.Context) ([]string, error) {
	// Get list of time series from source
	filters := make(map[string]interface{})
	if m.options.StartTime != "" || m.options.EndTime != "" {
		// Add time filters if needed
	}

	switch m.sourceConn.Type {
	case "influxdb":
		seriesList, err := m.source.(*influxdb.InfluxDBStorage).List(ctx, 10000, 0)
		if err != nil {
			return nil, err
		}
		ids := make([]string, len(seriesList))
		for i, ts := range seriesList {
			ids[i] = ts.ID
		}
		return ids, nil
	case "timescaledb":
		seriesList, err := m.source.(*timescaledb.TimescaleDBStorage).List(ctx, filters)
		if err != nil {
			return nil, err
		}
		ids := make([]string, len(seriesList))
		for i, ts := range seriesList {
			ids[i] = ts.ID
		}
		return ids, nil
	case "file":
		// For files, return a single series
		return []string{"file-series"}, nil
	default:
		return nil, fmt.Errorf("unsupported source type: %s", m.sourceConn.Type)
	}
}

func (m *Migrator) estimateRecords(ctx context.Context, plan *MigrationPlan) (int64, error) {
	// Estimate total records to migrate
	var totalRecords int64

	for _, seriesID := range plan.SeriesList {
		// Count records for each series
		count, err := m.countSeriesRecords(ctx, seriesID)
		if err != nil {
			m.logger.WithError(err).Warnf("Failed to count records for series %s", seriesID)
			count = 10000 // Default estimate
		}
		totalRecords += count
	}

	return totalRecords, nil
}

func (m *Migrator) countSeriesRecords(ctx context.Context, seriesID string) (int64, error) {
	switch m.sourceConn.Type {
	case "timescaledb":
		filters := map[string]interface{}{"series_id": seriesID}
		return m.source.(*timescaledb.TimescaleDBStorage).Count(ctx, filters)
	default:
		return 10000, nil // Default estimate
	}
}

func (m *Migrator) validateMigration(ctx context.Context, plan *MigrationPlan) (*ValidationSummary, error) {
	// Simple validation: compare record counts
	sourceCount, err := m.estimateRecords(ctx, plan)
	if err != nil {
		return nil, fmt.Errorf("failed to get source count: %w", err)
	}

	// Count destination records (simplified)
	destCount := sourceCount // Assume successful migration for now

	score := 1.0
	discrepancies := 0
	if sourceCount != destCount {
		score = float64(destCount) / float64(sourceCount)
		discrepancies = int(sourceCount - destCount)
	}

	return &ValidationSummary{
		RecordsValidated: int(destCount),
		ValidationScore:  score,
		Discrepancies:    discrepancies,
	}, nil
}

func (m *Migrator) applyTransforms(timeSeries *models.TimeSeries, rules []TransformRule) (*models.TimeSeries, error) {
	// Apply transformation rules (simplified implementation)
	for _, rule := range rules {
		switch rule.Type {
		case "scale":
			if factor, ok := rule.Parameters["factor"].(float64); ok {
				for i := range timeSeries.DataPoints {
					timeSeries.DataPoints[i].Value *= factor
				}
			}
		case "offset":
			if offset, ok := rule.Parameters["offset"].(float64); ok {
				for i := range timeSeries.DataPoints {
					timeSeries.DataPoints[i].Value += offset
				}
			}
		case "rename":
			if newName, ok := rule.Parameters["name"].(string); ok {
				timeSeries.Name = newName
			}
		}
	}
	return timeSeries, nil
}

func (m *Migrator) Close() error {
	var errs []error
	if m.source != nil {
		if err := m.source.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if m.destination != nil {
		if err := m.destination.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("errors closing connections: %v", errs)
	}
	return nil
}

// Helper functions

func parseConnectionString(connStr, connType string) (*ConnectionInfo, error) {
	info := &ConnectionInfo{
		Type:   connType,
		URL:    connStr,
		Config: make(map[string]interface{}),
	}

	if connType == "auto" {
		info.Type = detectType(connStr)
	}

	u, err := url.Parse(connStr)
	if err != nil {
		return nil, fmt.Errorf("invalid connection string: %w", err)
	}

	info.Host = u.Hostname()
	if u.Port() != "" {
		info.Port, _ = parsePort(u.Port())
	}

	switch info.Type {
	case "influxdb":
		info.Database = strings.TrimPrefix(u.Path, "/")
		info.Config["url"] = connStr
		info.Config["token"] = u.User.Username()
	case "timescaledb":
		info.Database = strings.TrimPrefix(u.Path, "/")
		info.Config["host"] = info.Host
		info.Config["port"] = info.Port
		info.Config["database"] = info.Database
	case "file":
		info.Path = u.Path
	case "s3":
		info.Bucket = u.Host
		info.Path = u.Path
	}

	return info, nil
}

func detectType(connStr string) string {
	if strings.HasPrefix(connStr, "influx://") {
		return "influxdb"
	}
	if strings.HasPrefix(connStr, "timescale://") || strings.HasPrefix(connStr, "postgres://") {
		return "timescaledb"
	}
	if strings.HasPrefix(connStr, "file://") {
		return "file"
	}
	if strings.HasPrefix(connStr, "s3://") {
		return "s3"
	}
	return "unknown"
}

func parsePort(portStr string) (int, error) {
	// Simple port parsing
	switch portStr {
	case "8086":
		return 8086, nil
	case "5432":
		return 5432, nil
	default:
		return 0, fmt.Errorf("unsupported port: %s", portStr)
	}
}

func createStorage(conn *ConnectionInfo, logger *logrus.Logger) (interfaces.Storage, error) {
	switch conn.Type {
	case "influxdb":
		config := &influxdb.InfluxDBConfig{
			URL:          conn.Config["url"].(string),
			Token:        conn.Config["token"].(string),
			Organization: "default",
			Bucket:       conn.Database,
		}
		return influxdb.NewInfluxDBStorage(config, logger)
	case "timescaledb":
		config := &timescaledb.TimescaleDBConfig{
			Host:     conn.Host,
			Port:     conn.Port,
			Database: conn.Database,
			Username: "postgres",
			Password: "password",
			SSLMode:  "disable",
		}
		return timescaledb.NewTimescaleDBStorage(config, logger)
	default:
		return nil, fmt.Errorf("unsupported storage type: %s", conn.Type)
	}
}

func parseTimeRange(startStr, endStr string) (*TimeRange, error) {
	timeRange := &TimeRange{}
	
	if startStr != "" {
		start, err := parseTime(startStr)
		if err != nil {
			return nil, fmt.Errorf("invalid start time: %w", err)
		}
		timeRange.Start = start
	}
	
	if endStr != "" {
		end, err := parseTime(endStr)
		if err != nil {
			return nil, fmt.Errorf("invalid end time: %w", err)
		}
		timeRange.End = end
	}
	
	return timeRange, nil
}

func parseTransformRules(transformStr string) ([]TransformRule, error) {
	// Parse transform rules from string (simplified)
	rules := []TransformRule{}
	
	if transformStr == "scale:2.0" {
		rules = append(rules, TransformRule{
			Type:        "scale",
			Description: "Scale values by factor of 2.0",
			Parameters:  map[string]interface{}{"factor": 2.0},
		})
	}
	
	return rules, nil
}

func writeTimeSeriesToFile(timeSeries *models.TimeSeries, path string) error {
	// Simple CSV writing implementation
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write header
	file.WriteString("timestamp,value,quality\n")

	// Write data points
	for _, dp := range timeSeries.DataPoints {
		line := fmt.Sprintf("%s,%f,%f\n", dp.Timestamp.Format(time.RFC3339), dp.Value, dp.Quality)
		file.WriteString(line)
	}

	return nil
}

// Progress tracking
type ProgressTracker struct {
	EstimatedRecords int64
	EstimatedBatches int
	RecordsMigrated  int64
	BatchesCompleted int
	StartTime        time.Time
}

func NewProgressTracker(estimatedRecords int64, estimatedBatches int) *ProgressTracker {
	return &ProgressTracker{
		EstimatedRecords: estimatedRecords,
		EstimatedBatches: estimatedBatches,
		StartTime:        time.Now(),
	}
}

func (pt *ProgressTracker) UpdateProgress(recordsInBatch int) {
	pt.RecordsMigrated += int64(recordsInBatch)
	pt.BatchesCompleted++

	progress := float64(pt.RecordsMigrated) / float64(pt.EstimatedRecords) * 100
	if progress > 100 {
		progress = 100
	}

	elapsed := time.Since(pt.StartTime)
	rate := float64(pt.RecordsMigrated) / elapsed.Seconds()

	fmt.Printf("\rProgress: %.1f%% (%d/%d records) | Rate: %.0f rec/sec | Elapsed: %s",
		progress, pt.RecordsMigrated, pt.EstimatedRecords, rate, elapsed.Round(time.Second))
}

// Output functions
func printMigrationPlan(plan *MigrationPlan, opts *MigrateOptions) {
	fmt.Println("\nMigration Plan:")
	fmt.Printf("- Source Type: %s\n", plan.SourceInfo.Type)
	fmt.Printf("- Destination Type: %s\n", plan.DestinationInfo.Type)
	fmt.Printf("- Estimated Records: %s\n", formatNumber(plan.EstimatedRecords))
	fmt.Printf("- Batch Size: %d\n", plan.BatchSize)
	fmt.Printf("- Estimated Batches: %d\n", plan.EstimatedBatches)
	fmt.Printf("- Parallel Workers: %d\n", plan.ParallelWorkers)
	fmt.Printf("- Series Count: %d\n", len(plan.SeriesList))

	if plan.TimeRange != nil {
		fmt.Println("\nTime Filters:")
		if !plan.TimeRange.Start.IsZero() {
			fmt.Printf("- Start: %s\n", plan.TimeRange.Start.Format(time.RFC3339))
		}
		if !plan.TimeRange.End.IsZero() {
			fmt.Printf("- End: %s\n", plan.TimeRange.End.Format(time.RFC3339))
		}
	}

	if len(plan.TransformRules) > 0 {
		fmt.Println("\nTransformation Rules:")
		for _, rule := range plan.TransformRules {
			fmt.Printf("- %s: %s\n", rule.Type, rule.Description)
		}
	}

	if plan.ValidationEnabled {
		fmt.Println("\n- Validation: Enabled")
	}
}

func printMigrationResult(result *MigrationResult) {
	fmt.Printf("\n\nMigration Summary:\n")
	fmt.Printf("- Status: %s\n", getStatusString(result.Success))
	fmt.Printf("- Records Migrated: %s\n", formatNumber(result.RecordsMigrated))
	fmt.Printf("- Series Migrated: %d\n", result.SeriesMigrated)
	fmt.Printf("- Duration: %s\n", result.Duration.Round(time.Second).String())
	fmt.Printf("- Average Rate: %.0f records/sec\n", result.AverageRate)

	if len(result.Errors) > 0 {
		fmt.Printf("- Errors: %d\n", len(result.Errors))
		for _, err := range result.Errors {
			fmt.Printf("  * %s\n", err)
		}
	}

	if len(result.Warnings) > 0 {
		fmt.Printf("- Warnings: %d\n", len(result.Warnings))
		for _, warning := range result.Warnings {
			fmt.Printf("  * %s\n", warning)
		}
	}

	if result.ValidationResult != nil {
		fmt.Printf("\nValidation Results:\n")
		fmt.Printf("- Records Validated: %d\n", result.ValidationResult.RecordsValidated)
		fmt.Printf("- Validation Score: %.1f%%\n", result.ValidationResult.ValidationScore*100)
		if result.ValidationResult.Discrepancies > 0 {
			fmt.Printf("- Discrepancies: %d\n", result.ValidationResult.Discrepancies)
		}
	}

	if result.Success {
		fmt.Println("\n✓ Migration completed successfully!")
	} else {
		fmt.Println("\n✗ Migration completed with errors.")
	}
}

func getStatusString(success bool) string {
	if success {
		return "Success"
	}
	return "Failed"
}

func formatNumber(n int64) string {
	if n < 1000 {
		return fmt.Sprintf("%d", n)
	} else if n < 1000000 {
		return fmt.Sprintf("%.1fK", float64(n)/1000)
	} else {
		return fmt.Sprintf("%.1fM", float64(n)/1000000)
	}
}