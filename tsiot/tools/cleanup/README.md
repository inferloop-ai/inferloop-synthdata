# Cleanup Tools

This directory contains utilities for cleaning up and maintaining the TSIoT platform's data and logs.

## Tools Overview

### 1. data-cleanup.go
Cleans up old time series data files based on age, size, and other criteria.

**Features:**
- Age-based cleanup (delete files older than X days)
- Pattern matching for selective cleanup
- Dry run mode for safety
- Backup before deletion
- Compression support (gzip, zstd)
- Free space monitoring
- Recursive directory traversal

**Usage:**
```bash
# Clean files older than 30 days
./data-cleanup -dir ./data -max-age 720h

# Dry run to see what would be deleted
./data-cleanup -dir ./data -max-age 30d -dry-run

# Clean with backup
./data-cleanup -dir ./data -max-age 30d -backup ./backups

# Clean specific file patterns
./data-cleanup -dir ./data -pattern "*.tmp" -max-age 1h

# Maintain minimum free space
./data-cleanup -dir ./data -min-free-gb 50
```

### 2. log-cleanup.go
Manages log file rotation, compression, and archival.

**Features:**
- Log rotation based on size and age
- Compression of old logs
- Archive to remote storage (S3, GCS)
- Pattern-based filtering
- Multi-format support (json, text, structured)
- Aggregation of similar log entries
- Retention policies

**Usage:**
```bash
# Basic log cleanup
./log-cleanup -dir ./logs -max-age 7d

# Rotate and compress logs
./log-cleanup -dir ./logs -rotate-size 100MB -compress gzip

# Archive to S3
./log-cleanup -dir ./logs -archive s3://mybucket/logs/

# Clean specific log levels
./log-cleanup -dir ./logs -level-filter "DEBUG,TRACE" -max-age 1d

# Aggregate similar entries
./log-cleanup -dir ./logs -aggregate -threshold 0.9
```

## Configuration

Both tools support configuration files in YAML or JSON format:

### data-cleanup.yaml
```yaml
data_dir: ./data
max_age: 720h
file_pattern: "*.json"
recursive: true
min_free_space_gb: 50
backup_location: ./backups
compression_type: gzip
exclude_patterns:
  - "*.lock"
  - ".DS_Store"
cleanup_rules:
  - pattern: "*.tmp"
    max_age: 1h
  - pattern: "*.cache"
    max_age: 24h
  - pattern: "archived/*"
    max_age: 90d
```

### log-cleanup.yaml
```yaml
log_dir: ./logs
max_age: 7d
max_size_mb: 100
compression: gzip
archive:
  enabled: true
  destination: s3://mybucket/logs/
  retention_days: 90
rotation:
  size_mb: 100
  count: 10
  compress_old: true
aggregation:
  enabled: true
  threshold: 0.85
  time_window: 1h
level_policies:
  DEBUG:
    max_age: 1d
  INFO:
    max_age: 7d
  ERROR:
    max_age: 30d
```

## Common Use Cases

### 1. Daily Cleanup Cron Job
```bash
# Add to crontab
0 2 * * * /path/to/data-cleanup -config /etc/tsiot/cleanup.yaml
0 3 * * * /path/to/log-cleanup -config /etc/tsiot/log-cleanup.yaml
```

### 2. Pre-deployment Cleanup
```bash
#!/bin/bash
# Clean temporary files before deployment
./data-cleanup -dir ./data -pattern "*.tmp" -max-age 0
./log-cleanup -dir ./logs -level-filter "DEBUG" -max-age 0
```

### 3. Storage Management
```bash
#!/bin/bash
# Monitor and maintain free space
while true; do
  FREE_SPACE=$(df -BG /data | tail -1 | awk '{print $4}' | sed 's/G//')
  if [ "$FREE_SPACE" -lt 50 ]; then
    ./data-cleanup -dir /data -max-age 7d -min-free-gb 100
  fi
  sleep 3600
done
```

### 4. Log Archival Pipeline
```bash
# Compress, aggregate, and archive logs
./log-cleanup \
  -dir ./logs \
  -compress gzip \
  -aggregate \
  -archive s3://logs-archive/tsiot/$(date +%Y/%m/%d)/ \
  -max-age 7d
```

## Best Practices

1. **Always use dry-run first** to verify what will be deleted
2. **Enable backups** for critical data before cleanup
3. **Set up monitoring** for cleanup job failures
4. **Use configuration files** for consistent cleanup policies
5. **Test patterns** carefully to avoid deleting important files
6. **Monitor free space** to prevent disk full issues
7. **Archive before deletion** for compliance requirements

## Performance Considerations

- Use appropriate batch sizes for large directories
- Enable parallel processing for faster cleanup
- Consider I/O impact on production systems
- Schedule cleanup during off-peak hours
- Monitor system resources during cleanup

## Safety Features

1. **Dry Run Mode**: Preview changes without deletion
2. **Backup Support**: Create backups before deletion
3. **Pattern Validation**: Verify file patterns before execution
4. **Atomic Operations**: Ensure cleanup can be safely interrupted
5. **Audit Logging**: Track all cleanup operations
6. **Rollback Support**: Restore from backups if needed

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Ensure proper file permissions
   - Run with appropriate user privileges

2. **Disk Space Issues**
   - Check available space before backup
   - Use compression to save space

3. **Pattern Not Matching**
   - Verify glob pattern syntax
   - Use `-verbose` flag for debugging

4. **Slow Performance**
   - Reduce batch size
   - Enable parallel processing
   - Check disk I/O metrics

## Integration

### With Monitoring Systems
```bash
# Send metrics to Prometheus
./data-cleanup -dir ./data -metrics-endpoint http://prometheus:9090/metrics
```

### With Alerting
```bash
# Send alerts on failure
./log-cleanup -dir ./logs -alert-webhook https://slack.com/webhook
```

### With CI/CD
```yaml
# GitLab CI example
cleanup-job:
  stage: maintenance
  script:
    - ./data-cleanup -config cleanup.yaml -dry-run
    - ./data-cleanup -config cleanup.yaml
  only:
    - schedules
```