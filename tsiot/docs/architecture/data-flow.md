# TSIOT Data Flow Architecture

## Overview

This document describes the data flow architecture for the Time Series IoT Synthetic Data (TSIOT) platform. It covers how data moves through the system from generation requests to final storage and analysis.

## Data Flow Overview

```
                                                                
   Client       �   API          � Generation     �  Storage    
 Application       Gateway          Service          Backend    
                                                                
                                                                
                          �                   �                   �
                                                                  
                   Validation        Analytics         Streaming  
                   Service           Service           Output     
                                                                  
```

## Core Data Flow Patterns

### 1. Synchronous Generation Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant G as API Gateway
    participant A as Auth Service
    participant GS as Generation Service
    participant S as Storage Service
    participant V as Validation Service

    C->>G: POST /api/v1/generate
    G->>A: Validate API Key/JWT
    A->>G: Authentication Result
    G->>GS: Generation Request
    GS->>GS: Generate Time Series
    GS->>V: Validate Generated Data
    V->>GS: Validation Result
    GS->>S: Store Time Series
    S->>GS: Storage Confirmation
    GS->>G: Generated Time Series
    G->>C: Response with Time Series
```

### 2. Asynchronous Batch Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant G as API Gateway
    participant Q as Job Queue
    participant W as Worker Pool
    participant S as Storage Service
    participant N as Notification Service

    C->>G: POST /api/v1/batch/generate
    G->>Q: Enqueue Batch Job
    Q->>C: Job ID (202 Accepted)
    Q->>W: Dispatch Job to Worker
    W->>W: Process Batch
    W->>S: Store Results
    W->>N: Send Completion Notification
    N->>C: Webhook/Email Notification
```

### 3. Streaming Data Flow

```mermaid
graph LR
    A[Data Source] --> B[Kafka Producer]
    B --> C[Kafka Topic]
    C --> D[Stream Processor]
    D --> E[Real-time Validator]
    D --> F[Analytics Engine]
    E --> G[Alert System]
    F --> H[Dashboard]
    D --> I[Storage Sink]
```

## Detailed Component Flows

### 1. Request Processing Pipeline

#### Input Validation Layer
```go
type RequestProcessor struct {
    validator  *RequestValidator
    limiter    *RateLimiter
    auth       *AuthService
    metrics    *MetricsCollector
}

func (rp *RequestProcessor) ProcessRequest(req *http.Request) (*ProcessedRequest, error) {
    // 1. Rate limiting check
    if err := rp.limiter.Allow(req); err != nil {
        rp.metrics.IncrementRateLimit()
        return nil, err
    }
    
    // 2. Authentication
    user, err := rp.auth.Authenticate(req)
    if err != nil {
        rp.metrics.IncrementAuthFailure()
        return nil, err
    }
    
    // 3. Request validation
    validatedReq, err := rp.validator.Validate(req, user)
    if err != nil {
        rp.metrics.IncrementValidationError()
        return nil, err
    }
    
    return validatedReq, nil
}
```

#### Generation Pipeline
```go
type GenerationPipeline struct {
    generators map[string]Generator
    validators []Validator
    storage    StorageManager
    cache      CacheManager
}

func (gp *GenerationPipeline) Generate(req *GenerationRequest) (*TimeSeries, error) {
    // 1. Check cache first
    if cached := gp.cache.Get(req.CacheKey()); cached != nil {
        return cached, nil
    }
    
    // 2. Select appropriate generator
    generator := gp.generators[req.Type]
    if generator == nil {
        return nil, errors.New("unknown generator type")
    }
    
    // 3. Generate time series
    ts, err := generator.Generate(req.Parameters)
    if err != nil {
        return nil, err
    }
    
    // 4. Run validation pipeline
    for _, validator := range gp.validators {
        if err := validator.Validate(ts); err != nil {
            return nil, fmt.Errorf("validation failed: %w", err)
        }
    }
    
    // 5. Store in cache and persistent storage
    gp.cache.Set(req.CacheKey(), ts)
    gp.storage.Store(ts)
    
    return ts, nil
}
```

### 2. Storage Data Flow

#### Multi-Backend Storage Strategy
```go
type StorageRouter struct {
    backends map[string]StorageBackend
    rules    []RoutingRule
}

type RoutingRule struct {
    Condition func(*TimeSeries) bool
    Backend   string
    Priority  int
}

func (sr *StorageRouter) Store(ts *TimeSeries) error {
    // Route to appropriate storage backends
    for _, rule := range sr.rules {
        if rule.Condition(ts) {
            backend := sr.backends[rule.Backend]
            if err := backend.Store(ts); err != nil {
                // Log error but continue to other backends
                log.Errorf("Failed to store in %s: %v", rule.Backend, err)
            }
        }
    }
    return nil
}

// Example routing rules
func CreateStorageRules() []RoutingRule {
    return []RoutingRule{
        {
            Condition: func(ts *TimeSeries) bool { 
                return ts.Metadata.Priority == "realtime" 
            },
            Backend:  "timescaledb",
            Priority: 1,
        },
        {
            Condition: func(ts *TimeSeries) bool { 
                return ts.Length > 100000 
            },
            Backend:  "clickhouse",
            Priority: 2,
        },
        {
            Condition: func(ts *TimeSeries) bool { 
                return time.Since(ts.CreatedAt) > 24*time.Hour 
            },
            Backend:  "s3",
            Priority: 3,
        },
    }
}
```

#### Data Transformation Pipeline
```go
type TransformationPipeline struct {
    stages []TransformationStage
}

type TransformationStage interface {
    Transform(*TimeSeries) (*TimeSeries, error)
    CanProcess(*TimeSeries) bool
}

// Example stages
type CompressionStage struct {
    Algorithm string
    Threshold int
}

func (cs *CompressionStage) Transform(ts *TimeSeries) (*TimeSeries, error) {
    if len(ts.DataPoints) < cs.Threshold {
        return ts, nil // Skip compression for small series
    }
    
    compressed, err := compress(ts.DataPoints, cs.Algorithm)
    if err != nil {
        return nil, err
    }
    
    ts.Metadata.Compressed = true
    ts.Metadata.CompressionAlgorithm = cs.Algorithm
    ts.CompressedData = compressed
    
    return ts, nil
}
```

### 3. Real-time Analytics Flow

#### Stream Processing Architecture
```go
type StreamProcessor struct {
    input      chan *TimeSeries
    output     chan *AnalysisResult
    windows    map[string]*SlidingWindow
    analyzers  []StreamAnalyzer
    alerter    *AlertManager
}

func (sp *StreamProcessor) ProcessStream() {
    for ts := range sp.input {
        // Add to appropriate windows
        for _, window := range sp.windows {
            if window.ShouldInclude(ts) {
                window.Add(ts)
            }
        }
        
        // Run analysis on each window
        for _, analyzer := range sp.analyzers {
            for _, window := range sp.windows {
                if window.IsReady() {
                    result := analyzer.Analyze(window.GetData())
                    sp.output <- result
                    
                    // Check for alerts
                    if result.ShouldAlert() {
                        sp.alerter.TriggerAlert(result)
                    }
                }
            }
        }
    }
}
```

#### Sliding Window Implementation
```go
type SlidingWindow struct {
    size     time.Duration
    data     []*TimeSeries
    mutex    sync.RWMutex
    lastClean time.Time
}

func (sw *SlidingWindow) Add(ts *TimeSeries) {
    sw.mutex.Lock()
    defer sw.mutex.Unlock()
    
    sw.data = append(sw.data, ts)
    
    // Clean old data periodically
    if time.Since(sw.lastClean) > time.Minute {
        sw.cleanOldData()
        sw.lastClean = time.Now()
    }
}

func (sw *SlidingWindow) cleanOldData() {
    cutoff := time.Now().Add(-sw.size)
    i := 0
    for i < len(sw.data) && sw.data[i].CreatedAt.Before(cutoff) {
        i++
    }
    sw.data = sw.data[i:]
}
```

### 4. Event-Driven Architecture

#### Event System
```go
type EventBus struct {
    subscribers map[string][]EventHandler
    mutex       sync.RWMutex
}

type Event struct {
    Type      string
    Timestamp time.Time
    Source    string
    Data      interface{}
    Metadata  map[string]interface{}
}

type EventHandler interface {
    Handle(Event) error
    GetEventTypes() []string
}

func (eb *EventBus) Publish(event Event) {
    eb.mutex.RLock()
    handlers := eb.subscribers[event.Type]
    eb.mutex.RUnlock()
    
    for _, handler := range handlers {
        go func(h EventHandler) {
            if err := h.Handle(event); err != nil {
                log.Errorf("Event handler error: %v", err)
            }
        }(handler)
    }
}

// Example event handlers
type ValidationEventHandler struct {
    validator *ValidationService
}

func (veh *ValidationEventHandler) Handle(event Event) error {
    if ts, ok := event.Data.(*TimeSeries); ok {
        return veh.validator.ValidateAsync(ts)
    }
    return nil
}

func (veh *ValidationEventHandler) GetEventTypes() []string {
    return []string{"time_series.generated", "time_series.updated"}
}
```

### 5. Data Quality and Monitoring Flow

#### Quality Monitoring Pipeline
```go
type QualityMonitor struct {
    metrics    []QualityMetric
    collectors []DataCollector
    alerter    *AlertManager
    dashboard  *Dashboard
}

type QualityMetric interface {
    Calculate(*TimeSeries) float64
    GetThreshold() float64
    GetName() string
}

func (qm *QualityMonitor) MonitorQuality(ts *TimeSeries) {
    qualityReport := &QualityReport{
        SeriesID:  ts.Metadata.SeriesID,
        Timestamp: time.Now(),
        Metrics:   make(map[string]float64),
    }
    
    for _, metric := range qm.metrics {
        value := metric.Calculate(ts)
        qualityReport.Metrics[metric.GetName()] = value
        
        if value < metric.GetThreshold() {
            alert := &QualityAlert{
                SeriesID:    ts.Metadata.SeriesID,
                MetricName:  metric.GetName(),
                Value:       value,
                Threshold:   metric.GetThreshold(),
                Severity:    "warning",
                Timestamp:   time.Now(),
            }
            qm.alerter.SendAlert(alert)
        }
    }
    
    // Send to collectors and dashboard
    for _, collector := range qm.collectors {
        collector.Collect(qualityReport)
    }
    qm.dashboard.UpdateQualityMetrics(qualityReport)
}
```

### 6. Caching Strategy and Data Flow

#### Multi-Level Cache Architecture
```go
type CacheManager struct {
    l1      *sync.Map          // In-memory cache
    l2      *redis.Client      // Redis cache
    l3      *s3.Client        // S3 cache
    ttl     map[string]time.Duration
    metrics *CacheMetrics
}

func (cm *CacheManager) Get(key string) (*TimeSeries, error) {
    // L1 Cache (fastest)
    if value, ok := cm.l1.Load(key); ok {
        cm.metrics.RecordHit("l1")
        return value.(*TimeSeries), nil
    }
    
    // L2 Cache (medium speed)
    if data, err := cm.l2.Get(key).Bytes(); err == nil {
        ts := &TimeSeries{}
        if err := json.Unmarshal(data, ts); err == nil {
            cm.l1.Store(key, ts) // Promote to L1
            cm.metrics.RecordHit("l2")
            return ts, nil
        }
    }
    
    // L3 Cache (slowest)
    if ts, err := cm.getFromS3(key); err == nil {
        // Promote to L1 and L2
        cm.l1.Store(key, ts)
        data, _ := json.Marshal(ts)
        cm.l2.Set(key, data, cm.ttl["l2"])
        cm.metrics.RecordHit("l3")
        return ts, nil
    }
    
    cm.metrics.RecordMiss()
    return nil, errors.New("not found in cache")
}

func (cm *CacheManager) Set(key string, ts *TimeSeries) {
    // Store in all levels
    cm.l1.Store(key, ts)
    
    data, _ := json.Marshal(ts)
    cm.l2.Set(key, data, cm.ttl["l2"])
    
    go func() {
        cm.storeInS3(key, ts) // Async S3 storage
    }()
}
```

### 7. Error Handling and Recovery Flow

#### Circuit Breaker Pattern
```go
type CircuitBreaker struct {
    state        State
    failures     int
    threshold    int
    timeout      time.Duration
    lastFailTime time.Time
    mutex        sync.Mutex
}

type State int

const (
    Closed State = iota
    Open
    HalfOpen
)

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    switch cb.state {
    case Open:
        if time.Since(cb.lastFailTime) > cb.timeout {
            cb.state = HalfOpen
            cb.failures = 0
        } else {
            return errors.New("circuit breaker is open")
        }
    case HalfOpen:
        // Allow one request through
    }
    
    err := fn()
    if err != nil {
        cb.failures++
        cb.lastFailTime = time.Now()
        
        if cb.failures >= cb.threshold {
            cb.state = Open
        }
        return err
    }
    
    // Success
    cb.failures = 0
    cb.state = Closed
    return nil
}
```

## Data Consistency and Integrity

### 1. ACID Properties in Distributed System
```go
type TransactionManager struct {
    participants []TransactionParticipant
    coordinator  *TransactionCoordinator
}

type TransactionParticipant interface {
    Prepare(txID string) error
    Commit(txID string) error
    Abort(txID string) error
}

// Two-Phase Commit implementation
func (tm *TransactionManager) ExecuteTransaction(operations []Operation) error {
    txID := generateTransactionID()
    
    // Phase 1: Prepare
    for _, participant := range tm.participants {
        if err := participant.Prepare(txID); err != nil {
            // Abort all participants
            for _, p := range tm.participants {
                p.Abort(txID)
            }
            return err
        }
    }
    
    // Phase 2: Commit
    for _, participant := range tm.participants {
        if err := participant.Commit(txID); err != nil {
            // This is a critical error - log and alert
            log.Errorf("Commit failed for participant: %v", err)
            // Attempt compensation
        }
    }
    
    return nil
}
```

### 2. Event Sourcing for Audit Trail
```go
type EventStore struct {
    events []Event
    snapshots map[string]Snapshot
    mutex  sync.RWMutex
}

func (es *EventStore) AppendEvent(event Event) error {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    // Validate event
    if err := es.validateEvent(event); err != nil {
        return err
    }
    
    // Store event
    es.events = append(es.events, event)
    
    // Create snapshot if needed
    if len(es.events)%1000 == 0 {
        snapshot := es.createSnapshot()
        es.snapshots[event.AggregateID] = snapshot
    }
    
    return nil
}

func (es *EventStore) Replay(aggregateID string) (*TimeSeries, error) {
    // Start from latest snapshot
    snapshot := es.snapshots[aggregateID]
    ts := snapshot.State
    
    // Apply events since snapshot
    for _, event := range es.events {
        if event.AggregateID == aggregateID && 
           event.Timestamp.After(snapshot.Timestamp) {
            ts = ApplyEvent(ts, event)
        }
    }
    
    return ts, nil
}
```

## Performance Optimization Patterns

### 1. Batch Processing Optimization
```go
type BatchProcessor struct {
    batchSize    int
    flushTimeout time.Duration
    buffer       []*TimeSeries
    mutex        sync.Mutex
    ticker       *time.Ticker
}

func (bp *BatchProcessor) Add(ts *TimeSeries) {
    bp.mutex.Lock()
    defer bp.mutex.Unlock()
    
    bp.buffer = append(bp.buffer, ts)
    
    if len(bp.buffer) >= bp.batchSize {
        go bp.flush()
    }
}

func (bp *BatchProcessor) flush() {
    bp.mutex.Lock()
    batch := make([]*TimeSeries, len(bp.buffer))
    copy(batch, bp.buffer)
    bp.buffer = bp.buffer[:0] // Clear buffer
    bp.mutex.Unlock()
    
    // Process batch
    if err := bp.processBatch(batch); err != nil {
        log.Errorf("Batch processing failed: %v", err)
        // Add to retry queue
    }
}
```

### 2. Connection Pooling
```go
type ConnectionPool struct {
    connections chan *Connection
    factory     ConnectionFactory
    maxSize     int
    currentSize int32
    mutex       sync.Mutex
}

func (cp *ConnectionPool) Get() (*Connection, error) {
    select {
    case conn := <-cp.connections:
        if conn.IsValid() {
            return conn, nil
        }
        // Connection is invalid, create new one
    default:
        // No available connections
    }
    
    // Create new connection if under limit
    if atomic.LoadInt32(&cp.currentSize) < int32(cp.maxSize) {
        conn, err := cp.factory.Create()
        if err != nil {
            return nil, err
        }
        atomic.AddInt32(&cp.currentSize, 1)
        return conn, nil
    }
    
    // Wait for available connection
    return <-cp.connections, nil
}

func (cp *ConnectionPool) Put(conn *Connection) {
    if conn.IsValid() {
        select {
        case cp.connections <- conn:
            // Successfully returned to pool
        default:
            // Pool is full, close connection
            conn.Close()
            atomic.AddInt32(&cp.currentSize, -1)
        }
    } else {
        conn.Close()
        atomic.AddInt32(&cp.currentSize, -1)
    }
}
```

## Conclusion

The TSIOT data flow architecture is designed for high throughput, low latency, and robust error handling. Key principles include:

1. **Asynchronous Processing**: Non-blocking operations where possible
2. **Event-Driven Architecture**: Loose coupling between components
3. **Multi-Level Caching**: Optimized data access patterns
4. **Circuit Breakers**: Fault tolerance and graceful degradation
5. **Batch Processing**: Efficient resource utilization
6. **Quality Monitoring**: Continuous data quality assessment
7. **Event Sourcing**: Complete audit trail and state reconstruction

This architecture supports both real-time and batch processing workloads while maintaining data consistency and providing comprehensive monitoring and alerting capabilities.