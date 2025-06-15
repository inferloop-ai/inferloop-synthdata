package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v2"

	"github.com/inferloop/tsiot/pkg/models"
	_ "github.com/lib/pq"
	_ "github.com/go-sql-driver/mysql"
	_ "github.com/mattn/go-sqlite3"
)

type MigrationConfig struct {
	Source            string                 `json:"source" yaml:"source"`
	Target            string                 `json:"target" yaml:"target"`
	Direction         string                 `json:"direction" yaml:"direction"`
	Version           string                 `json:"version" yaml:"version"`
	Steps             int                    `json:"steps" yaml:"steps"`
	Mode              string                 `json:"mode" yaml:"mode"`
	BatchSize         int                    `json:"batch_size" yaml:"batch_size"`
	Workers           int                    `json:"workers" yaml:"workers"`
	DryRun            bool                   `json:"dry_run" yaml:"dry_run"`
	Validate          bool                   `json:"validate" yaml:"validate"`
	RollbackOnError   bool                   `json:"rollback_on_error" yaml:"rollback_on_error"`
	TransformConfig   string                 `json:"transform_config" yaml:"transform_config"`
	MigrationsDir     string                 `json:"migrations_dir" yaml:"migrations_dir"`
}

type Migration struct {
	Version     string    `db:"version"`
	Description string    `db:"description"`
	AppliedAt   time.Time `db:"applied_at"`
	ExecutionTime time.Duration `db:"execution_time"`
	Checksum    string    `db:"checksum"`
}

type MigrationFile struct {
	Version     string
	Description string
	Direction   string
	Path        string
	SQL         string
	Checksum    string
}

type TransformConfig struct {
	Version          string                 `yaml:"version"`
	Transformations  []Transformation       `yaml:"transformations"`
}

type Transformation struct {
	Name            string                 `yaml:"name"`
	SourceTable     string                 `yaml:"source_table"`
	TargetTable     string                 `yaml:"target_table"`
	Mappings        []FieldMapping         `yaml:"mappings"`
	Filters         []Filter               `yaml:"filters"`
	BatchSize       int                    `yaml:"batch_size"`
	ValidateFunc    string                 `yaml:"validate_func"`
}

type FieldMapping struct {
	Source    string                 `yaml:"source"`
	Target    string                 `yaml:"target"`
	Transform string                 `yaml:"transform"`
	Default   interface{}            `yaml:"default"`
}

type Filter struct {
	Field    string                 `yaml:"field"`
	Operator string                 `yaml:"operator"`
	Value    interface{}            `yaml:"value"`
}

type Migrator struct {
	config       *MigrationConfig
	logger       *logrus.Logger
	sourceDB     *sql.DB
	targetDB     *sql.DB
	transforms   map[string]TransformFunc
	migrations   []MigrationFile
}

type TransformFunc func(value interface{}) (interface{}, error)

type MigrationStats struct {
	StartTime       time.Time
	EndTime         time.Time
	RecordsProcessed int64
	RecordsFailed   int64
	BytesProcessed  int64
	Errors          []error
}

func main() {
	var (
		source          = flag.String("source", "", "Source connection string")
		target          = flag.String("target", "", "Target connection string")
		direction       = flag.String("direction", "up", "Migration direction: up, down")
		version         = flag.String("version", "", "Target version to migrate to")
		steps           = flag.Int("steps", 0, "Number of migration steps")
		mode            = flag.String("mode", "schema", "Migration mode: schema, data, transform")
		batchSize       = flag.Int("batch-size", 1000, "Batch size for data migrations")
		workers         = flag.Int("workers", 4, "Number of parallel workers")
		dryRun          = flag.Bool("dry-run", false, "Perform dry run")
		validate        = flag.Bool("validate", false, "Validate data after migration")
		rollbackOnError = flag.Bool("rollback-on-error", true, "Rollback on error")
		configFile      = flag.String("config", "", "Configuration file")
		transformConfig = flag.String("transform-config", "", "Transformation config file")
		migrationsDir   = flag.String("migrations-dir", "./migrations", "Migrations directory")
		verbose         = flag.Bool("verbose", false, "Enable verbose logging")
		createMigration = flag.String("create", "", "Create new migration with given name")
	)
	flag.Parse()

	// Setup logging
	logger := logrus.New()
	if *verbose {
		logger.SetLevel(logrus.DebugLevel)
	}

	// Handle migration creation
	if *createMigration != "" {
		if err := createMigrationFiles(*createMigration, *migrationsDir); err != nil {
			log.Fatalf("Failed to create migration: %v", err)
		}
		fmt.Printf("Created migration files for: %s\n", *createMigration)
		return
	}

	// Load configuration
	config := &MigrationConfig{
		Source:          *source,
		Target:          *target,
		Direction:       *direction,
		Version:         *version,
		Steps:           *steps,
		Mode:            *mode,
		BatchSize:       *batchSize,
		Workers:         *workers,
		DryRun:          *dryRun,
		Validate:        *validate,
		RollbackOnError: *rollbackOnError,
		TransformConfig: *transformConfig,
		MigrationsDir:   *migrationsDir,
	}

	if *configFile != "" {
		if err := loadConfig(*configFile, config); err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}
	}

	// Validate configuration
	if config.Source == "" {
		log.Fatal("Source connection string is required")
	}

	migrator := NewMigrator(config, logger)

	ctx := context.Background()
	
	logger.WithFields(logrus.Fields{
		"mode":      config.Mode,
		"source":    maskConnectionString(config.Source),
		"target":    maskConnectionString(config.Target),
		"direction": config.Direction,
		"dry_run":   config.DryRun,
	}).Info("Starting migration")

	var err error
	switch config.Mode {
	case "schema":
		err = migrator.RunSchemaMigration(ctx)
	case "data":
		err = migrator.RunDataMigration(ctx)
	case "transform":
		err = migrator.RunTransformMigration(ctx)
	default:
		err = fmt.Errorf("unknown migration mode: %s", config.Mode)
	}

	if err != nil {
		log.Fatalf("Migration failed: %v", err)
	}

	logger.Info("Migration completed successfully")
}

func NewMigrator(config *MigrationConfig, logger *logrus.Logger) *Migrator {
	m := &Migrator{
		config:     config,
		logger:     logger,
		transforms: make(map[string]TransformFunc),
	}

	// Register built-in transforms
	m.registerBuiltinTransforms()

	return m
}

func (m *Migrator) RunSchemaMigration(ctx context.Context) error {
	// Connect to source database
	db, err := m.connectDB(m.config.Source)
	if err != nil {
		return fmt.Errorf("failed to connect to source: %w", err)
	}
	m.sourceDB = db
	defer m.sourceDB.Close()

	// Ensure migration table exists
	if err := m.ensureMigrationTable(); err != nil {
		return fmt.Errorf("failed to create migration table: %w", err)
	}

	// Load migration files
	if err := m.loadMigrationFiles(); err != nil {
		return fmt.Errorf("failed to load migrations: %w", err)
	}

	// Get current version
	currentVersion, err := m.getCurrentVersion()
	if err != nil {
		return fmt.Errorf("failed to get current version: %w", err)
	}

	m.logger.WithField("current_version", currentVersion).Info("Current migration version")

	// Determine migrations to run
	migrations := m.getMigrationsToRun(currentVersion)
	
	if len(migrations) == 0 {
		m.logger.Info("No migrations to run")
		return nil
	}

	m.logger.WithField("count", len(migrations)).Info("Migrations to run")

	// Execute migrations
	for _, migration := range migrations {
		if err := m.executeMigration(ctx, migration); err != nil {
			if m.config.RollbackOnError {
				m.logger.WithError(err).Error("Migration failed, rolling back")
				if rollbackErr := m.rollbackMigration(ctx, migration); rollbackErr != nil {
					return fmt.Errorf("migration failed and rollback failed: %v, %v", err, rollbackErr)
				}
			}
			return fmt.Errorf("migration %s failed: %w", migration.Version, err)
		}
	}

	return nil
}

func (m *Migrator) RunDataMigration(ctx context.Context) error {
	if m.config.Target == "" {
		return fmt.Errorf("target connection string required for data migration")
	}

	// Connect to source and target
	sourceDB, err := m.connectDB(m.config.Source)
	if err != nil {
		return fmt.Errorf("failed to connect to source: %w", err)
	}
	m.sourceDB = sourceDB
	defer m.sourceDB.Close()

	targetDB, err := m.connectDB(m.config.Target)
	if err != nil {
		return fmt.Errorf("failed to connect to target: %w", err)
	}
	m.targetDB = targetDB
	defer m.targetDB.Close()

	stats := &MigrationStats{
		StartTime: time.Now(),
	}

	// Create worker pool
	jobs := make(chan *models.TimeSeries, m.config.BatchSize)
	results := make(chan error, m.config.Workers)
	
	var wg sync.WaitGroup
	for i := 0; i < m.config.Workers; i++ {
		wg.Add(1)
		go m.dataWorker(ctx, jobs, results, stats, &wg)
	}

	// Start result collector
	done := make(chan bool)
	go m.collectResults(results, stats, done)

	// Read and process data
	if err := m.streamData(ctx, jobs); err != nil {
		return fmt.Errorf("failed to stream data: %w", err)
	}

	close(jobs)
	wg.Wait()
	close(results)
	<-done

	stats.EndTime = time.Now()

	// Log statistics
	m.logger.WithFields(logrus.Fields{
		"duration":         stats.EndTime.Sub(stats.StartTime),
		"records_processed": stats.RecordsProcessed,
		"records_failed":   stats.RecordsFailed,
		"bytes_processed":  stats.BytesProcessed,
		"errors":           len(stats.Errors),
	}).Info("Data migration completed")

	if m.config.Validate {
		return m.validateMigration(ctx, stats)
	}

	return nil
}

func (m *Migrator) RunTransformMigration(ctx context.Context) error {
	if m.config.TransformConfig == "" {
		return fmt.Errorf("transform config file required")
	}

	// Load transform configuration
	transformConfig, err := m.loadTransformConfig(m.config.TransformConfig)
	if err != nil {
		return fmt.Errorf("failed to load transform config: %w", err)
	}

	// Connect to databases
	if err := m.connectDatabases(); err != nil {
		return fmt.Errorf("failed to connect to databases: %w", err)
	}
	defer m.closeDatabases()

	// Execute transformations
	for _, transform := range transformConfig.Transformations {
		m.logger.WithField("transformation", transform.Name).Info("Running transformation")
		
		if err := m.executeTransformation(ctx, transform); err != nil {
			return fmt.Errorf("transformation %s failed: %w", transform.Name, err)
		}
	}

	return nil
}

func (m *Migrator) connectDB(connStr string) (*sql.DB, error) {
	// Parse connection string to determine driver
	var driver string
	switch {
	case strings.HasPrefix(connStr, "postgres://"):
		driver = "postgres"
	case strings.HasPrefix(connStr, "mysql://"):
		driver = "mysql"
	case strings.HasPrefix(connStr, "sqlite://"):
		driver = "sqlite3"
		connStr = strings.TrimPrefix(connStr, "sqlite://")
	default:
		return nil, fmt.Errorf("unsupported database type in connection string")
	}

	db, err := sql.Open(driver, connStr)
	if err != nil {
		return nil, err
	}

	// Test connection
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, err
	}

	// Configure connection pool
	db.SetMaxOpenConns(m.config.Workers * 2)
	db.SetMaxIdleConns(m.config.Workers)
	db.SetConnMaxLifetime(5 * time.Minute)

	return db, nil
}

func (m *Migrator) ensureMigrationTable() error {
	query := `
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version VARCHAR(255) PRIMARY KEY,
			description TEXT,
			applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			execution_time BIGINT,
			checksum VARCHAR(64)
		)
	`

	_, err := m.sourceDB.Exec(query)
	return err
}

func (m *Migrator) loadMigrationFiles() error {
	files, err := ioutil.ReadDir(m.config.MigrationsDir)
	if err != nil {
		return err
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		// Parse filename: YYYYMMDD_HHMMSS_description.up.sql
		parts := strings.Split(file.Name(), "_")
		if len(parts) < 3 {
			continue
		}

		// Extract version and direction
		version := parts[0] + "_" + parts[1]
		
		var direction string
		if strings.HasSuffix(file.Name(), ".up.sql") {
			direction = "up"
		} else if strings.HasSuffix(file.Name(), ".down.sql") {
			direction = "down"
		} else {
			continue
		}

		// Read SQL content
		path := filepath.Join(m.config.MigrationsDir, file.Name())
		content, err := ioutil.ReadFile(path)
		if err != nil {
			return fmt.Errorf("failed to read migration file %s: %w", path, err)
		}

		// Calculate checksum
		checksum := calculateChecksum(string(content))

		migration := MigrationFile{
			Version:     version,
			Description: extractDescription(file.Name()),
			Direction:   direction,
			Path:        path,
			SQL:         string(content),
			Checksum:    checksum,
		}

		m.migrations = append(m.migrations, migration)
	}

	// Sort migrations by version
	sort.Slice(m.migrations, func(i, j int) bool {
		return m.migrations[i].Version < m.migrations[j].Version
	})

	return nil
}

func (m *Migrator) getCurrentVersion() (string, error) {
	var version string
	query := `SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1`
	
	err := m.sourceDB.QueryRow(query).Scan(&version)
	if err == sql.ErrNoRows {
		return "", nil
	}
	
	return version, err
}

func (m *Migrator) getMigrationsToRun(currentVersion string) []MigrationFile {
	var migrations []MigrationFile

	for _, migration := range m.migrations {
		// Skip if wrong direction
		if migration.Direction != m.config.Direction {
			continue
		}

		// For up migrations, run if version > current
		if m.config.Direction == "up" && migration.Version > currentVersion {
			migrations = append(migrations, migration)
			if m.config.Steps > 0 && len(migrations) >= m.config.Steps {
				break
			}
		}

		// For down migrations, run if version <= current
		if m.config.Direction == "down" && migration.Version <= currentVersion {
			migrations = append(migrations, migration)
			if m.config.Steps > 0 && len(migrations) >= m.config.Steps {
				break
			}
		}
	}

	// Reverse order for down migrations
	if m.config.Direction == "down" {
		for i := 0; i < len(migrations)/2; i++ {
			j := len(migrations) - 1 - i
			migrations[i], migrations[j] = migrations[j], migrations[i]
		}
	}

	return migrations
}

func (m *Migrator) executeMigration(ctx context.Context, migration MigrationFile) error {
	m.logger.WithFields(logrus.Fields{
		"version":   migration.Version,
		"direction": migration.Direction,
		"file":      migration.Path,
	}).Info("Executing migration")

	if m.config.DryRun {
		m.logger.Info("DRY RUN - Would execute:")
		fmt.Println(migration.SQL)
		return nil
	}

	startTime := time.Now()

	// Begin transaction
	tx, err := m.sourceDB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Execute migration SQL
	if _, err := tx.Exec(migration.SQL); err != nil {
		return fmt.Errorf("failed to execute SQL: %w", err)
	}

	// Record migration
	if m.config.Direction == "up" {
		query := `
			INSERT INTO schema_migrations (version, description, execution_time, checksum)
			VALUES ($1, $2, $3, $4)
		`
		_, err = tx.Exec(query, 
			migration.Version, 
			migration.Description,
			time.Since(startTime).Nanoseconds(),
			migration.Checksum,
		)
		if err != nil {
			return fmt.Errorf("failed to record migration: %w", err)
		}
	} else {
		// Remove migration record for down migrations
		query := `DELETE FROM schema_migrations WHERE version = $1`
		_, err = tx.Exec(query, migration.Version)
		if err != nil {
			return fmt.Errorf("failed to remove migration record: %w", err)
		}
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit migration: %w", err)
	}

	m.logger.WithFields(logrus.Fields{
		"version":  migration.Version,
		"duration": time.Since(startTime),
	}).Info("Migration completed")

	return nil
}

func (m *Migrator) rollbackMigration(ctx context.Context, migration MigrationFile) error {
	// Find corresponding down migration
	var downMigration *MigrationFile
	for _, mig := range m.migrations {
		if mig.Version == migration.Version && mig.Direction == "down" {
			downMigration = &mig
			break
		}
	}

	if downMigration == nil {
		return fmt.Errorf("no down migration found for version %s", migration.Version)
	}

	return m.executeMigration(ctx, *downMigration)
}

func (m *Migrator) dataWorker(ctx context.Context, jobs <-chan *models.TimeSeries, results chan<- error, stats *MigrationStats, wg *sync.WaitGroup) {
	defer wg.Done()

	for ts := range jobs {
		select {
		case <-ctx.Done():
			results <- ctx.Err()
			return
		default:
			if err := m.migrateTimeSeries(ts); err != nil {
				results <- fmt.Errorf("failed to migrate series %s: %w", ts.ID, err)
			} else {
				results <- nil
			}
		}
	}
}

func (m *Migrator) migrateTimeSeries(ts *models.TimeSeries) error {
	// Implement actual migration logic based on target type
	// This is a placeholder implementation
	
	if m.config.DryRun {
		m.logger.WithField("series_id", ts.ID).Debug("DRY RUN - Would migrate time series")
		return nil
	}

	// Example: Insert into target database
	query := `
		INSERT INTO timeseries (id, name, description, sensor_type, start_time, end_time)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (id) DO UPDATE SET
			name = EXCLUDED.name,
			description = EXCLUDED.description,
			updated_at = NOW()
	`

	_, err := m.targetDB.Exec(query, 
		ts.ID, ts.Name, ts.Description, ts.SensorType, ts.StartTime, ts.EndTime)
	
	return err
}

func (m *Migrator) streamData(ctx context.Context, jobs chan<- *models.TimeSeries) error {
	query := `SELECT id, name, description, sensor_type, start_time, end_time FROM timeseries`
	
	rows, err := m.sourceDB.QueryContext(ctx, query)
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var ts models.TimeSeries
		err := rows.Scan(&ts.ID, &ts.Name, &ts.Description, &ts.SensorType, &ts.StartTime, &ts.EndTime)
		if err != nil {
			return err
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case jobs <- &ts:
		}
	}

	return rows.Err()
}

func (m *Migrator) collectResults(results <-chan error, stats *MigrationStats, done chan<- bool) {
	for err := range results {
		if err != nil {
			stats.RecordsFailed++
			stats.Errors = append(stats.Errors, err)
			m.logger.WithError(err).Error("Migration error")
		} else {
			stats.RecordsProcessed++
		}
	}
	done <- true
}

func (m *Migrator) validateMigration(ctx context.Context, stats *MigrationStats) error {
	m.logger.Info("Validating migration...")

	// Count records in source
	var sourceCount int64
	err := m.sourceDB.QueryRow("SELECT COUNT(*) FROM timeseries").Scan(&sourceCount)
	if err != nil {
		return fmt.Errorf("failed to count source records: %w", err)
	}

	// Count records in target
	var targetCount int64
	err = m.targetDB.QueryRow("SELECT COUNT(*) FROM timeseries").Scan(&targetCount)
	if err != nil {
		return fmt.Errorf("failed to count target records: %w", err)
	}

	m.logger.WithFields(logrus.Fields{
		"source_count": sourceCount,
		"target_count": targetCount,
		"processed":    stats.RecordsProcessed,
	}).Info("Validation results")

	if sourceCount != targetCount {
		return fmt.Errorf("record count mismatch: source=%d, target=%d", sourceCount, targetCount)
	}

	return nil
}

func (m *Migrator) registerBuiltinTransforms() {
	// Unix timestamp to time.Time
	m.transforms["unix_to_timestamp"] = func(value interface{}) (interface{}, error) {
		switch v := value.(type) {
		case int64:
			return time.Unix(v, 0), nil
		case float64:
			return time.Unix(int64(v), 0), nil
		default:
			return nil, fmt.Errorf("invalid unix timestamp type: %T", value)
		}
	}

	// Float precision
	m.transforms["float_precision"] = func(value interface{}) (interface{}, error) {
		// This would need the precision parameter from config
		switch v := value.(type) {
		case float64:
			return fmt.Sprintf("%.2f", v), nil
		default:
			return value, nil
		}
	}

	// String to lowercase
	m.transforms["lowercase"] = func(value interface{}) (interface{}, error) {
		if str, ok := value.(string); ok {
			return strings.ToLower(str), nil
		}
		return value, nil
	}
}

func (m *Migrator) loadTransformConfig(path string) (*TransformConfig, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config TransformConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

func (m *Migrator) connectDatabases() error {
	sourceDB, err := m.connectDB(m.config.Source)
	if err != nil {
		return fmt.Errorf("failed to connect to source: %w", err)
	}
	m.sourceDB = sourceDB

	if m.config.Target != "" {
		targetDB, err := m.connectDB(m.config.Target)
		if err != nil {
			m.sourceDB.Close()
			return fmt.Errorf("failed to connect to target: %w", err)
		}
		m.targetDB = targetDB
	}

	return nil
}

func (m *Migrator) closeDatabases() {
	if m.sourceDB != nil {
		m.sourceDB.Close()
	}
	if m.targetDB != nil {
		m.targetDB.Close()
	}
}

func (m *Migrator) executeTransformation(ctx context.Context, transform Transformation) error {
	// This is a simplified implementation
	// In a real implementation, you would:
	// 1. Read from source table with proper pagination
	// 2. Apply field mappings and transformations
	// 3. Apply filters
	// 4. Write to target table in batches
	// 5. Handle errors and rollbacks

	m.logger.WithFields(logrus.Fields{
		"source": transform.SourceTable,
		"target": transform.TargetTable,
	}).Info("Executing transformation")

	return nil
}

// Helper functions

func loadConfig(path string, config *MigrationConfig) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	if strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml") {
		return yaml.Unmarshal(data, config)
	}

	return json.Unmarshal(data, config)
}

func maskConnectionString(connStr string) string {
	if connStr == "" {
		return ""
	}

	// Simple masking - in production use proper URL parsing
	parts := strings.Split(connStr, "@")
	if len(parts) > 1 {
		return parts[0][:len(parts[0])/2] + "****@" + parts[1]
	}
	return connStr[:len(connStr)/3] + "****"
}

func calculateChecksum(content string) string {
	// Simplified - in production use crypto/sha256
	return fmt.Sprintf("%x", len(content))
}

func extractDescription(filename string) string {
	// Remove timestamp prefix and file extension
	parts := strings.Split(filename, "_")
	if len(parts) > 2 {
		desc := strings.Join(parts[2:], "_")
		desc = strings.TrimSuffix(desc, ".up.sql")
		desc = strings.TrimSuffix(desc, ".down.sql")
		return strings.ReplaceAll(desc, "_", " ")
	}
	return filename
}

func createMigrationFiles(name, dir string) error {
	timestamp := time.Now().Format("20060102_150405")
	baseFilename := fmt.Sprintf("%s_%s", timestamp, strings.ReplaceAll(name, " ", "_"))

	// Create migrations directory if it doesn't exist
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Create up migration
	upPath := filepath.Join(dir, baseFilename+".up.sql")
	upContent := fmt.Sprintf("-- Migration: %s\n-- Description: %s\n\n-- Add your UP migration SQL here\n", baseFilename, name)
	if err := ioutil.WriteFile(upPath, []byte(upContent), 0644); err != nil {
		return err
	}

	// Create down migration
	downPath := filepath.Join(dir, baseFilename+".down.sql")
	downContent := fmt.Sprintf("-- Migration: %s\n-- Description: %s\n\n-- Add your DOWN migration SQL here\n", baseFilename, name)
	if err := ioutil.WriteFile(downPath, []byte(downContent), 0644); err != nil {
		return err
	}

	return nil
}