# Migration Scripts Tool

This tool provides utilities for migrating time series data between different storage backends, versions, and schemas in the TSIoT platform.

## Overview

The migration tool supports:
- Database schema migrations (PostgreSQL, MySQL, SQLite)
- Time series data format conversions
- Storage backend migrations (file-based to database, database to object storage)
- Version upgrades with data transformation
- Batch processing with progress tracking
- Rollback capabilities

## Usage

### Basic Migration

```bash
# Run all pending migrations
./migrate -source postgres://localhost/tsiot -direction up

# Rollback last migration
./migrate -source postgres://localhost/tsiot -direction down -steps 1

# Migrate to specific version
./migrate -source postgres://localhost/tsiot -version 20240115_add_indexes

# Dry run mode
./migrate -source postgres://localhost/tsiot -direction up -dry-run
```

### Data Migration

```bash
# Migrate data between storage backends
./migrate -mode data \
  -source file://./data/timeseries \
  -target s3://mybucket/tsiot-data \
  -batch-size 1000

# Transform data format during migration
./migrate -mode transform \
  -source postgres://localhost/tsiot \
  -transform-config ./transforms/v2_upgrade.yaml
```

### Configuration Options

```bash
Flags:
  -source string
        Source connection string (required)
  -target string
        Target connection string (for data migrations)
  -direction string
        Migration direction: up, down (default "up")
  -version string
        Target version to migrate to
  -steps int
        Number of migration steps to execute
  -mode string
        Migration mode: schema, data, transform (default "schema")
  -batch-size int
        Batch size for data migrations (default 1000)
  -workers int
        Number of parallel workers (default 4)
  -dry-run
        Perform dry run without applying changes
  -verbose
        Enable verbose logging
  -config string
        Configuration file path
  -transform-config string
        Transformation configuration file
  -validate
        Validate data after migration
  -rollback-on-error
        Automatically rollback on error
```

## Migration Files

Schema migrations should be placed in the `migrations/` directory with the following naming convention:

```
YYYYMMDD_HHMMSS_description.up.sql
YYYYMMDD_HHMMSS_description.down.sql
```

Example:
```
20240115_143022_create_timeseries_table.up.sql
20240115_143022_create_timeseries_table.down.sql
```

## Transformation Configuration

For data transformations, create a YAML configuration file:

```yaml
version: "1.0"
transformations:
  - name: "upgrade_sensor_data"
    source_table: "sensor_data_v1"
    target_table: "sensor_data_v2"
    mappings:
      - source: "timestamp"
        target: "recorded_at"
        transform: "unix_to_timestamp"
      - source: "value"
        target: "measurement"
        transform: "float_precision:2"
    filters:
      - field: "quality"
        operator: ">="
        value: 0.8
    batch_size: 5000
```

## Supported Storage Backends

- **Databases**: PostgreSQL, MySQL, SQLite, MongoDB
- **File Systems**: Local filesystem, NFS
- **Object Storage**: AWS S3, Google Cloud Storage, Azure Blob Storage
- **Time Series Databases**: InfluxDB, TimescaleDB, Cassandra
- **Message Queues**: Kafka, MQTT (for streaming migrations)

## Best Practices

1. **Always backup data** before running migrations
2. **Test migrations** in a staging environment first
3. **Use dry-run mode** to preview changes
4. **Monitor progress** for large data migrations
5. **Implement rollback scripts** for all migrations
6. **Validate data integrity** after migration

## Examples

### Example 1: PostgreSQL Schema Migration

```bash
# Create migration files
./migrate create add_privacy_fields

# Run migration
./migrate -source postgres://user:pass@localhost/tsiot -direction up
```

### Example 2: Migrate from Files to PostgreSQL

```bash
./migrate -mode data \
  -source file://./data/2024 \
  -target postgres://user:pass@localhost/tsiot \
  -batch-size 5000 \
  -workers 8 \
  -validate
```

### Example 3: Transform and Migrate to New Schema

```bash
./migrate -mode transform \
  -source postgres://localhost/old_tsiot \
  -target postgres://localhost/new_tsiot \
  -transform-config ./transforms/schema_v2.yaml \
  -rollback-on-error
```

## Troubleshooting

### Common Issues

1. **Connection errors**: Verify connection strings and network access
2. **Permission errors**: Ensure proper database/file permissions
3. **Memory issues**: Reduce batch size for large migrations
4. **Transformation errors**: Validate transformation configuration

### Debug Mode

Enable debug logging:
```bash
./migrate -source ... -verbose -log-level debug
```

### Recovery

If a migration fails:
1. Check the migration status table
2. Review error logs
3. Fix the issue
4. Resume or rollback as needed

## Development

### Adding New Storage Backends

Implement the `StorageBackend` interface:

```go
type StorageBackend interface {
    Connect(ctx context.Context) error
    Read(query Query) ([]TimeSeries, error)
    Write(data []TimeSeries) error
    Close() error
}
```

### Custom Transformations

Register custom transformation functions:

```go
RegisterTransform("custom_transform", func(value interface{}) (interface{}, error) {
    // Implementation
})
```

## Safety Features

- **Checkpoints**: Progress saved at regular intervals
- **Atomic Operations**: Migrations use transactions where possible
- **Validation**: Built-in data validation after migration
- **Rollback**: Automatic rollback on critical errors
- **Audit Log**: All operations logged for compliance

## Performance Considerations

- Use appropriate batch sizes based on data volume
- Enable parallel processing for large migrations
- Consider network bandwidth for remote migrations
- Monitor database load during migration
- Use compression for network transfers

## Related Tools

- `backup-tool`: Create backups before migration
- `validate-tool`: Validate data integrity
- `benchmark-tool`: Test migration performance
- `monitor-tool`: Monitor migration progress