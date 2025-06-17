package migrations

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/errors"
)

// MigrationManager manages database migrations
type MigrationManager struct {
	storage     interfaces.Storage
	logger      *logrus.Logger
	migrations  map[string]*Migration
	versions    []string
	config      *MigrationConfig
}

// Migration represents a database migration
type Migration struct {
	Version     string             `json:"version"`
	Name        string             `json:"name"`
	Description string             `json:"description"`
	UpFunc      func(context.Context, interfaces.Storage) error
	DownFunc    func(context.Context, interfaces.Storage) error
	Dependencies []string          `json:"dependencies"`
	Tags        []string           `json:"tags"`
	CreatedAt   time.Time          `json:"created_at"`
}

// MigrationConfig contains migration configuration
type MigrationConfig struct {
	TableName       string `json:"table_name"`
	LockTableName   string `json:"lock_table_name"`
	LockTimeout     time.Duration `json:"lock_timeout"`
	TransactionMode bool   `json:"transaction_mode"`
	DryRun          bool   `json:"dry_run"`
	SkipValidation  bool   `json:"skip_validation"`
}

// MigrationRecord represents a migration record in the database
type MigrationRecord struct {
	Version     string    `json:"version" db:"version"`
	Name        string    `json:"name" db:"name"`
	AppliedAt   time.Time `json:"applied_at" db:"applied_at"`
	Checksum    string    `json:"checksum" db:"checksum"`
	ExecutionTime time.Duration `json:"execution_time" db:"execution_time"`
	Success     bool      `json:"success" db:"success"`
	ErrorMessage string   `json:"error_message" db:"error_message"`
}

// MigrationLock represents a migration lock
type MigrationLock struct {
	ID        string    `json:"id" db:"id"`
	LockedAt  time.Time `json:"locked_at" db:"locked_at"`
	LockedBy  string    `json:"locked_by" db:"locked_by"`
	ExpiresAt time.Time `json:"expires_at" db:"expires_at"`
}

// MigrationStatus represents the status of migrations
type MigrationStatus struct {
	CurrentVersion  string              `json:"current_version"`
	TargetVersion   string              `json:"target_version"`
	PendingCount    int                 `json:"pending_count"`
	AppliedCount    int                 `json:"applied_count"`
	FailedCount     int                 `json:"failed_count"`
	LastMigration   *MigrationRecord    `json:"last_migration"`
	PendingMigrations []*Migration      `json:"pending_migrations"`
	AppliedMigrations []*MigrationRecord `json:"applied_migrations"`
}

// MigrationResult represents the result of a migration operation
type MigrationResult struct {
	Version       string        `json:"version"`
	Name          string        `json:"name"`
	Success       bool          `json:"success"`
	ExecutionTime time.Duration `json:"execution_time"`
	ErrorMessage  string        `json:"error_message,omitempty"`
}

// NewMigrationManager creates a new migration manager
func NewMigrationManager(storage interfaces.Storage, config *MigrationConfig, logger *logrus.Logger) *MigrationManager {
	if config == nil {
		config = &MigrationConfig{
			TableName:       "schema_migrations",
			LockTableName:   "migration_locks",
			LockTimeout:     time.Minute * 5,
			TransactionMode: true,
			DryRun:          false,
			SkipValidation:  false,
		}
	}

	if logger == nil {
		logger = logrus.New()
	}

	return &MigrationManager{
		storage:    storage,
		logger:     logger,
		migrations: make(map[string]*Migration),
		versions:   make([]string, 0),
		config:     config,
	}
}

// RegisterMigration registers a new migration
func (m *MigrationManager) RegisterMigration(migration *Migration) error {
	if migration.Version == "" {
		return errors.NewValidationError("INVALID_VERSION", "Migration version cannot be empty")
	}

	if migration.Name == "" {
		return errors.NewValidationError("INVALID_NAME", "Migration name cannot be empty")
	}

	if migration.UpFunc == nil {
		return errors.NewValidationError("INVALID_UP_FUNC", "Migration up function cannot be nil")
	}

	if _, exists := m.migrations[migration.Version]; exists {
		return errors.NewValidationError("DUPLICATE_VERSION", fmt.Sprintf("Migration version %s already exists", migration.Version))
	}

	migration.CreatedAt = time.Now()
	m.migrations[migration.Version] = migration
	m.versions = append(m.versions, migration.Version)
	
	// Sort versions
	sort.Strings(m.versions)

	m.logger.WithFields(logrus.Fields{
		"version": migration.Version,
		"name":    migration.Name,
	}).Info("Registered migration")

	return nil
}

// Initialize initializes the migration system
func (m *MigrationManager) Initialize(ctx context.Context) error {
	m.logger.Info("Initializing migration system")

	if err := m.createMigrationTables(ctx); err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "INIT_FAILED", "Failed to create migration tables")
	}

	if err := m.validateMigrations(); err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "VALIDATION_FAILED", "Migration validation failed")
	}

	m.logger.Info("Migration system initialized successfully")
	return nil
}

// Migrate runs pending migrations to bring the database up to the latest version
func (m *MigrationManager) Migrate(ctx context.Context) (*MigrationStatus, error) {
	return m.MigrateTo(ctx, "")
}

// MigrateTo runs migrations to bring the database to a specific version
func (m *MigrationManager) MigrateTo(ctx context.Context, targetVersion string) (*MigrationStatus, error) {
	m.logger.WithField("target_version", targetVersion).Info("Starting migration")

	// Acquire migration lock
	lock, err := m.acquireLock(ctx)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "LOCK_FAILED", "Failed to acquire migration lock")
	}
	defer m.releaseLock(ctx, lock)

	// Get current status
	status, err := m.GetStatus(ctx)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "STATUS_FAILED", "Failed to get migration status")
	}

	// Determine target version
	if targetVersion == "" {
		if len(m.versions) > 0 {
			targetVersion = m.versions[len(m.versions)-1]
		}
	}

	// Find migrations to run
	migrationsToRun := m.getMigrationsToRun(status.CurrentVersion, targetVersion)
	if len(migrationsToRun) == 0 {
		m.logger.Info("No pending migrations")
		return status, nil
	}

	m.logger.WithField("count", len(migrationsToRun)).Info("Found pending migrations")

	// Run migrations
	for _, migration := range migrationsToRun {
		result, err := m.runMigration(ctx, migration, true)
		if err != nil || !result.Success {
			m.logger.WithError(err).WithField("version", migration.Version).Error("Migration failed")
			return nil, errors.NewStorageError("MIGRATION_FAILED", 
				fmt.Sprintf("Migration %s failed: %v", migration.Version, err))
		}

		m.logger.WithFields(logrus.Fields{
			"version":        migration.Version,
			"execution_time": result.ExecutionTime,
		}).Info("Migration completed successfully")
	}

	// Get updated status
	return m.GetStatus(ctx)
}

// Rollback rolls back migrations to a specific version
func (m *MigrationManager) Rollback(ctx context.Context, targetVersion string) (*MigrationStatus, error) {
	m.logger.WithField("target_version", targetVersion).Info("Starting rollback")

	// Acquire migration lock
	lock, err := m.acquireLock(ctx)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "LOCK_FAILED", "Failed to acquire migration lock")
	}
	defer m.releaseLock(ctx, lock)

	// Get current status
	status, err := m.GetStatus(ctx)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "STATUS_FAILED", "Failed to get migration status")
	}

	// Find migrations to rollback
	migrationsToRollback := m.getMigrationsToRollback(status.CurrentVersion, targetVersion)
	if len(migrationsToRollback) == 0 {
		m.logger.Info("No migrations to rollback")
		return status, nil
	}

	m.logger.WithField("count", len(migrationsToRollback)).Info("Found migrations to rollback")

	// Rollback migrations (in reverse order)
	for i := len(migrationsToRollback) - 1; i >= 0; i-- {
		migration := migrationsToRollback[i]
		result, err := m.runMigration(ctx, migration, false)
		if err != nil || !result.Success {
			m.logger.WithError(err).WithField("version", migration.Version).Error("Rollback failed")
			return nil, errors.NewStorageError("ROLLBACK_FAILED", 
				fmt.Sprintf("Rollback %s failed: %v", migration.Version, err))
		}

		m.logger.WithFields(logrus.Fields{
			"version":        migration.Version,
			"execution_time": result.ExecutionTime,
		}).Info("Rollback completed successfully")
	}

	// Get updated status
	return m.GetStatus(ctx)
}

// GetStatus returns the current migration status
func (m *MigrationManager) GetStatus(ctx context.Context) (*MigrationStatus, error) {
	appliedRecords, err := m.getAppliedMigrations(ctx)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "STATUS_FAILED", "Failed to get applied migrations")
	}

	status := &MigrationStatus{
		AppliedMigrations: appliedRecords,
		AppliedCount:      len(appliedRecords),
		PendingMigrations: []*Migration{},
	}

	// Determine current version
	if len(appliedRecords) > 0 {
		status.CurrentVersion = appliedRecords[len(appliedRecords)-1].Version
		status.LastMigration = appliedRecords[len(appliedRecords)-1]
	}

	// Determine target version
	if len(m.versions) > 0 {
		status.TargetVersion = m.versions[len(m.versions)-1]
	}

	// Find pending migrations
	currentIdx := -1
	if status.CurrentVersion != "" {
		for i, version := range m.versions {
			if version == status.CurrentVersion {
				currentIdx = i
				break
			}
		}
	}

	for i := currentIdx + 1; i < len(m.versions); i++ {
		version := m.versions[i]
		if migration, exists := m.migrations[version]; exists {
			status.PendingMigrations = append(status.PendingMigrations, migration)
		}
	}

	status.PendingCount = len(status.PendingMigrations)

	// Count failed migrations
	for _, record := range appliedRecords {
		if !record.Success {
			status.FailedCount++
		}
	}

	return status, nil
}

// ValidateMigrations validates the migration chain
func (m *MigrationManager) ValidateMigrations() error {
	return m.validateMigrations()
}

// ListMigrations returns all registered migrations
func (m *MigrationManager) ListMigrations() []*Migration {
	migrations := make([]*Migration, 0, len(m.migrations))
	for _, version := range m.versions {
		if migration, exists := m.migrations[version]; exists {
			migrations = append(migrations, migration)
		}
	}
	return migrations
}

// GetMigration returns a specific migration by version
func (m *MigrationManager) GetMigration(version string) (*Migration, error) {
	migration, exists := m.migrations[version]
	if !exists {
		return nil, errors.NewValidationError("MIGRATION_NOT_FOUND", fmt.Sprintf("Migration %s not found", version))
	}
	return migration, nil
}

// Private methods

func (m *MigrationManager) createMigrationTables(ctx context.Context) error {
	// This would be implementation-specific based on the storage type
	// For now, we'll assume it's handled by the storage implementation
	m.logger.Info("Migration tables created (implementation-specific)")
	return nil
}

func (m *MigrationManager) validateMigrations() error {
	if m.config.SkipValidation {
		return nil
	}

	// Check for duplicate versions
	seen := make(map[string]bool)
	for _, version := range m.versions {
		if seen[version] {
			return errors.NewValidationError("DUPLICATE_VERSION", fmt.Sprintf("Duplicate migration version: %s", version))
		}
		seen[version] = true
	}

	// Validate dependencies
	for _, migration := range m.migrations {
		for _, dep := range migration.Dependencies {
			if _, exists := m.migrations[dep]; !exists {
				return errors.NewValidationError("MISSING_DEPENDENCY", 
					fmt.Sprintf("Migration %s depends on %s which does not exist", migration.Version, dep))
			}
		}
	}

	return nil
}

func (m *MigrationManager) acquireLock(ctx context.Context) (*MigrationLock, error) {
	lockID := fmt.Sprintf("migration_%d", time.Now().UnixNano())
	lock := &MigrationLock{
		ID:        lockID,
		LockedAt:  time.Now(),
		LockedBy:  "migration_manager", // Could be hostname or process ID
		ExpiresAt: time.Now().Add(m.config.LockTimeout),
	}

	// Implementation would depend on storage type
	m.logger.WithField("lock_id", lockID).Info("Acquired migration lock")
	return lock, nil
}

func (m *MigrationManager) releaseLock(ctx context.Context, lock *MigrationLock) error {
	// Implementation would depend on storage type
	m.logger.WithField("lock_id", lock.ID).Info("Released migration lock")
	return nil
}

func (m *MigrationManager) getAppliedMigrations(ctx context.Context) ([]*MigrationRecord, error) {
	// Implementation would depend on storage type
	// For now, return empty slice
	return []*MigrationRecord{}, nil
}

func (m *MigrationManager) getMigrationsToRun(currentVersion, targetVersion string) []*Migration {
	var migrations []*Migration

	currentIdx := -1
	targetIdx := len(m.versions) - 1

	if currentVersion != "" {
		for i, version := range m.versions {
			if version == currentVersion {
				currentIdx = i
				break
			}
		}
	}

	if targetVersion != "" {
		for i, version := range m.versions {
			if version == targetVersion {
				targetIdx = i
				break
			}
		}
	}

	for i := currentIdx + 1; i <= targetIdx; i++ {
		version := m.versions[i]
		if migration, exists := m.migrations[version]; exists {
			migrations = append(migrations, migration)
		}
	}

	return migrations
}

func (m *MigrationManager) getMigrationsToRollback(currentVersion, targetVersion string) []*Migration {
	var migrations []*Migration

	currentIdx := len(m.versions) - 1
	targetIdx := -1

	if currentVersion != "" {
		for i, version := range m.versions {
			if version == currentVersion {
				currentIdx = i
				break
			}
		}
	}

	if targetVersion != "" {
		for i, version := range m.versions {
			if version == targetVersion {
				targetIdx = i
				break
			}
		}
	}

	for i := currentIdx; i > targetIdx; i-- {
		version := m.versions[i]
		if migration, exists := m.migrations[version]; exists {
			migrations = append(migrations, migration)
		}
	}

	return migrations
}

func (m *MigrationManager) runMigration(ctx context.Context, migration *Migration, up bool) (*MigrationResult, error) {
	start := time.Now()
	result := &MigrationResult{
		Version: migration.Version,
		Name:    migration.Name,
	}

	if m.config.DryRun {
		m.logger.WithField("version", migration.Version).Info("DRY RUN: Would run migration")
		result.Success = true
		result.ExecutionTime = time.Since(start)
		return result, nil
	}

	var err error
	if up {
		if migration.UpFunc != nil {
			err = migration.UpFunc(ctx, m.storage)
		}
	} else {
		if migration.DownFunc != nil {
			err = migration.DownFunc(ctx, m.storage)
		} else {
			err = errors.NewValidationError("NO_DOWN_FUNC", 
				fmt.Sprintf("Migration %s has no down function", migration.Version))
		}
	}

	result.ExecutionTime = time.Since(start)
	result.Success = err == nil

	if err != nil {
		result.ErrorMessage = err.Error()
	}

	// Record migration
	record := &MigrationRecord{
		Version:       migration.Version,
		Name:          migration.Name,
		AppliedAt:     time.Now(),
		ExecutionTime: result.ExecutionTime,
		Success:       result.Success,
		ErrorMessage:  result.ErrorMessage,
	}

	if err := m.recordMigration(ctx, record, up); err != nil {
		m.logger.WithError(err).Error("Failed to record migration")
	}

	return result, err
}

func (m *MigrationManager) recordMigration(ctx context.Context, record *MigrationRecord, applied bool) error {
	// Implementation would depend on storage type
	action := "applied"
	if !applied {
		action = "rolled back"
	}

	m.logger.WithFields(logrus.Fields{
		"version":   record.Version,
		"action":    action,
		"success":   record.Success,
		"duration":  record.ExecutionTime,
	}).Info("Recorded migration")

	return nil
}

// GenerateChecksum generates a checksum for a migration
func (m *MigrationManager) GenerateChecksum(migration *Migration) string {
	// Simple checksum based on version and name
	return fmt.Sprintf("%x", []byte(migration.Version+migration.Name))
}

// Helper function to create a new migration
func NewMigration(version, name, description string) *Migration {
	return &Migration{
		Version:      version,
		Name:         name,
		Description:  description,
		Dependencies: []string{},
		Tags:         []string{},
		CreatedAt:    time.Now(),
	}
}

// Helper function to set migration up function
func (m *Migration) Up(fn func(context.Context, interfaces.Storage) error) *Migration {
	m.UpFunc = fn
	return m
}

// Helper function to set migration down function
func (m *Migration) Down(fn func(context.Context, interfaces.Storage) error) *Migration {
	m.DownFunc = fn
	return m
}

// Helper function to add dependencies
func (m *Migration) DependsOn(versions ...string) *Migration {
	m.Dependencies = append(m.Dependencies, versions...)
	return m
}

// Helper function to add tags
func (m *Migration) WithTags(tags ...string) *Migration {
	m.Tags = append(m.Tags, tags...)
	return m
}