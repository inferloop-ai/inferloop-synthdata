package helpers

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/influxdata/influxdb-client-go/v2"
	"github.com/stretchr/testify/require"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// TestCleanup provides utilities for cleaning up test resources
type TestCleanup struct {
	t         *testing.T
	resources []CleanupResource
	mu        sync.Mutex
}

// CleanupResource represents a resource that needs cleanup
type CleanupResource interface {
	Cleanup() error
	String() string
}

// FileCleanup represents file system cleanup
type FileCleanup struct {
	paths []string
}

// DatabaseCleanup represents database cleanup
type DatabaseCleanup struct {
	db     *sql.DB
	tables []string
}

// RedisCleanup represents Redis cleanup
type RedisCleanup struct {
	client *redis.Client
	keys   []string
}

// InfluxDBCleanup represents InfluxDB cleanup
type InfluxDBCleanup struct {
	client influxdb2.Client
	bucket string
	org    string
}

// MongoDBCleanup represents MongoDB cleanup
type MongoDBCleanup struct {
	client         *mongo.Client
	database       string
	collections    []string
	dropDatabase   bool
	dropCollection bool
}

// NewTestCleanup creates a new test cleanup helper
func NewTestCleanup(t *testing.T) *TestCleanup {
	tc := &TestCleanup{
		t:         t,
		resources: make([]CleanupResource, 0),
	}

	// Register cleanup to run at test end
	t.Cleanup(func() {
		tc.CleanupAll()
	})

	return tc
}

// AddResource adds a resource to be cleaned up
func (tc *TestCleanup) AddResource(resource CleanupResource) {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	tc.resources = append(tc.resources, resource)
}

// CleanupAll cleans up all registered resources
func (tc *TestCleanup) CleanupAll() {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	var errors []string
	for i := len(tc.resources) - 1; i >= 0; i-- {
		resource := tc.resources[i]
		if err := resource.Cleanup(); err != nil {
			errors = append(errors, fmt.Sprintf("failed to cleanup %s: %v", resource.String(), err))
		}
	}

	if len(errors) > 0 {
		tc.t.Errorf("cleanup errors: %s", strings.Join(errors, "; "))
	}

	tc.resources = tc.resources[:0]
}

// CreateTempDir creates a temporary directory for testing
func (tc *TestCleanup) CreateTempDir(pattern string) string {
	tempDir, err := os.MkdirTemp("", pattern)
	require.NoError(tc.t, err)

	tc.AddResource(&FileCleanup{paths: []string{tempDir}})
	return tempDir
}

// CreateTempFile creates a temporary file for testing
func (tc *TestCleanup) CreateTempFile(pattern string) (*os.File, string) {
	tempFile, err := os.CreateTemp("", pattern)
	require.NoError(tc.t, err)

	tc.AddResource(&FileCleanup{paths: []string{tempFile.Name()}})
	return tempFile, tempFile.Name()
}

// RegisterFileCleanup registers files/directories for cleanup
func (tc *TestCleanup) RegisterFileCleanup(paths ...string) {
	tc.AddResource(&FileCleanup{paths: paths})
}

// RegisterDatabaseCleanup registers database tables for cleanup
func (tc *TestCleanup) RegisterDatabaseCleanup(db *sql.DB, tables ...string) {
	tc.AddResource(&DatabaseCleanup{db: db, tables: tables})
}

// RegisterRedisCleanup registers Redis keys for cleanup
func (tc *TestCleanup) RegisterRedisCleanup(client *redis.Client, keys ...string) {
	tc.AddResource(&RedisCleanup{client: client, keys: keys})
}

// RegisterInfluxDBCleanup registers InfluxDB bucket for cleanup
func (tc *TestCleanup) RegisterInfluxDBCleanup(client influxdb2.Client, org, bucket string) {
	tc.AddResource(&InfluxDBCleanup{client: client, bucket: bucket, org: org})
}

// RegisterMongoDBCleanup registers MongoDB collections for cleanup
func (tc *TestCleanup) RegisterMongoDBCleanup(client *mongo.Client, database string, collections ...string) {
	tc.AddResource(&MongoDBCleanup{
		client:      client,
		database:    database,
		collections: collections,
	})
}

// RegisterMongoDBDatabaseCleanup registers entire MongoDB database for cleanup
func (tc *TestCleanup) RegisterMongoDBDatabaseCleanup(client *mongo.Client, database string) {
	tc.AddResource(&MongoDBCleanup{
		client:       client,
		database:     database,
		dropDatabase: true,
	})
}

// FileCleanup implementation
func (fc *FileCleanup) Cleanup() error {
	var errors []string
	for _, path := range fc.paths {
		if err := os.RemoveAll(path); err != nil && !os.IsNotExist(err) {
			errors = append(errors, fmt.Sprintf("failed to remove %s: %v", path, err))
		}
	}
	if len(errors) > 0 {
		return fmt.Errorf("file cleanup errors: %s", strings.Join(errors, "; "))
	}
	return nil
}

func (fc *FileCleanup) String() string {
	return fmt.Sprintf("FileCleanup(%s)", strings.Join(fc.paths, ", "))
}

// DatabaseCleanup implementation
func (dc *DatabaseCleanup) Cleanup() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var errors []string
	for _, table := range dc.tables {
		// Try TRUNCATE first (faster), fall back to DELETE
		if _, err := dc.db.ExecContext(ctx, fmt.Sprintf("TRUNCATE TABLE %s", table)); err != nil {
			if _, err := dc.db.ExecContext(ctx, fmt.Sprintf("DELETE FROM %s", table)); err != nil {
				errors = append(errors, fmt.Sprintf("failed to clean table %s: %v", table, err))
			}
		}
	}
	if len(errors) > 0 {
		return fmt.Errorf("database cleanup errors: %s", strings.Join(errors, "; "))
	}
	return nil
}

func (dc *DatabaseCleanup) String() string {
	return fmt.Sprintf("DatabaseCleanup(%s)", strings.Join(dc.tables, ", "))
}

// RedisCleanup implementation
func (rc *RedisCleanup) Cleanup() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if len(rc.keys) == 0 {
		// If no specific keys, flush all
		return rc.client.FlushAll(ctx).Err()
	}

	var errors []string
	for _, key := range rc.keys {
		if err := rc.client.Del(ctx, key).Err(); err != nil && err != redis.Nil {
			errors = append(errors, fmt.Sprintf("failed to delete key %s: %v", key, err))
		}
	}
	if len(errors) > 0 {
		return fmt.Errorf("redis cleanup errors: %s", strings.Join(errors, "; "))
	}
	return nil
}

func (rc *RedisCleanup) String() string {
	if len(rc.keys) == 0 {
		return "RedisCleanup(all keys)"
	}
	return fmt.Sprintf("RedisCleanup(%s)", strings.Join(rc.keys, ", "))
}

// InfluxDBCleanup implementation
func (ic *InfluxDBCleanup) Cleanup() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	deleteAPI := ic.client.DeleteAPI()
	now := time.Now()
	start := now.Add(-24 * time.Hour)

	// Delete all data in the bucket
	err := deleteAPI.DeleteWithName(ctx, ic.org, ic.bucket, start, now, "")
	if err != nil {
		return fmt.Errorf("failed to delete InfluxDB data: %v", err)
	}

	return nil
}

func (ic *InfluxDBCleanup) String() string {
	return fmt.Sprintf("InfluxDBCleanup(org=%s, bucket=%s)", ic.org, ic.bucket)
}

// MongoDBCleanup implementation
func (mc *MongoDBCleanup) Cleanup() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	db := mc.client.Database(mc.database)

	if mc.dropDatabase {
		return db.Drop(ctx)
	}

	var errors []string
	for _, collection := range mc.collections {
		coll := db.Collection(collection)
		if mc.dropCollection {
			if err := coll.Drop(ctx); err != nil {
				errors = append(errors, fmt.Sprintf("failed to drop collection %s: %v", collection, err))
			}
		} else {
			if _, err := coll.DeleteMany(ctx, map[string]interface{}{}); err != nil {
				errors = append(errors, fmt.Sprintf("failed to clear collection %s: %v", collection, err))
			}
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("mongodb cleanup errors: %s", strings.Join(errors, "; "))
	}
	return nil
}

func (mc *MongoDBCleanup) String() string {
	if mc.dropDatabase {
		return fmt.Sprintf("MongoDBCleanup(database=%s)", mc.database)
	}
	return fmt.Sprintf("MongoDBCleanup(collections=%s)", strings.Join(mc.collections, ", "))
}

// CleanupTestFiles removes test files matching patterns
func CleanupTestFiles(t *testing.T, patterns ...string) {
	t.Helper()
	
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			t.Logf("Failed to glob pattern %s: %v", pattern, err)
			continue
		}
		
		for _, match := range matches {
			if err := os.RemoveAll(match); err != nil {
				t.Logf("Failed to remove %s: %v", match, err)
			}
		}
	}
}

// CleanupTestDatabases truncates test database tables
func CleanupTestDatabases(t *testing.T, db *sql.DB, tables ...string) {
	t.Helper()
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	for _, table := range tables {
		// Try TRUNCATE first, then DELETE
		if _, err := db.ExecContext(ctx, fmt.Sprintf("TRUNCATE TABLE %s", table)); err != nil {
			if _, err := db.ExecContext(ctx, fmt.Sprintf("DELETE FROM %s", table)); err != nil {
				t.Logf("Failed to clean table %s: %v", table, err)
			}
		}
	}
}

// CleanupTestRedis clears Redis test data
func CleanupTestRedis(t *testing.T, client *redis.Client, keys ...string) {
	t.Helper()
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	if len(keys) == 0 {
		// Flush all keys
		if err := client.FlushAll(ctx).Err(); err != nil {
			t.Logf("Failed to flush Redis: %v", err)
		}
		return
	}
	
	for _, key := range keys {
		if err := client.Del(ctx, key).Err(); err != nil && err != redis.Nil {
			t.Logf("Failed to delete Redis key %s: %v", key, err)
		}
	}
}

// WaitForCleanup waits for cleanup operations to complete
func WaitForCleanup(timeout time.Duration, checkFn func() bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if checkFn() {
			return true
		}
		time.Sleep(100 * time.Millisecond)
	}
	return false
}

// CleanupWithRetry retries cleanup operations
func CleanupWithRetry(maxRetries int, delay time.Duration, cleanupFn func() error) error {
	var lastErr error
	for i := 0; i < maxRetries; i++ {
		if err := cleanupFn(); err == nil {
			return nil
		} else {
			lastErr = err
			if i < maxRetries-1 {
				time.Sleep(delay)
			}
		}
	}
	return fmt.Errorf("cleanup failed after %d retries: %v", maxRetries, lastErr)
}

// EnsureCleanState ensures the system is in a clean state before tests
func EnsureCleanState(t *testing.T, checkers ...func() error) {
	t.Helper()
	
	for _, checker := range checkers {
		if err := checker(); err != nil {
			t.Fatalf("System not in clean state: %v", err)
		}
	}
}

// CreateTestEnvironment creates a complete test environment with cleanup
func CreateTestEnvironment(t *testing.T) *TestCleanup {
	tc := NewTestCleanup(t)
	
	// Set test environment variables
	os.Setenv("ENVIRONMENT", "test")
	os.Setenv("LOG_LEVEL", "debug")
	
	// Create test data directory
	testDataDir := tc.CreateTempDir("tsiot-test-data-*")
	os.Setenv("TEST_DATA_DIR", testDataDir)
	
	return tc
}