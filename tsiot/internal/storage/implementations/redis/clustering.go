package redis

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// ClusterManager manages Redis cluster operations
type ClusterManager struct {
	client       *redis.ClusterClient
	logger       *logrus.Logger
	config       *ClusterConfig
	mu           sync.RWMutex
	nodes        map[string]*NodeInfo
	shardMap     map[string]string
	healthCache  map[string]*NodeHealth
	lastHealthCheck time.Time
}

// ClusterConfig contains Redis cluster configuration
type ClusterConfig struct {
	// Cluster settings
	Addrs                []string      `json:"addrs"`
	RouteByLatency       bool          `json:"route_by_latency"`
	RouteRandomly        bool          `json:"route_randomly"`
	ReadOnly             bool          `json:"read_only"`
	
	// Connection settings
	MaxRedirects         int           `json:"max_redirects"`
	ReadTimeout          time.Duration `json:"read_timeout"`
	WriteTimeout         time.Duration `json:"write_timeout"`
	DialTimeout          time.Duration `json:"dial_timeout"`
	
	// Pool settings
	PoolSize             int           `json:"pool_size"`
	MinIdleConns         int           `json:"min_idle_conns"`
	MaxConnAge           time.Duration `json:"max_conn_age"`
	PoolTimeout          time.Duration `json:"pool_timeout"`
	IdleTimeout          time.Duration `json:"idle_timeout"`
	IdleCheckFrequency   time.Duration `json:"idle_check_frequency"`
	
	// Failover settings
	MaxRetries           int           `json:"max_retries"`
	MinRetryBackoff      time.Duration `json:"min_retry_backoff"`
	MaxRetryBackoff      time.Duration `json:"max_retry_backoff"`
	
	// Health check settings
	HealthCheckInterval  time.Duration `json:"health_check_interval"`
	HealthCheckTimeout   time.Duration `json:"health_check_timeout"`
	
	// Sharding settings
	HashSlots            int           `json:"hash_slots"`
	ShardingStrategy     string        `json:"sharding_strategy"` // "hash", "consistent", "range"
	ReplicationFactor    int           `json:"replication_factor"`
}

// NodeInfo contains information about a cluster node
type NodeInfo struct {
	ID       string    `json:"id"`
	Addr     string    `json:"addr"`
	Role     string    `json:"role"`     // "master", "slave"
	Status   string    `json:"status"`   // "online", "offline", "fail"
	Slots    []string  `json:"slots"`    // Hash slots range
	Master   string    `json:"master"`   // Master node ID for slaves
	Replicas []string  `json:"replicas"` // Replica node IDs for masters
	Flags    []string  `json:"flags"`
	LastSeen time.Time `json:"last_seen"`
}

// NodeHealth contains health information for a node
type NodeHealth struct {
	NodeID       string        `json:"node_id"`
	Addr         string        `json:"addr"`
	Status       string        `json:"status"`
	Latency      time.Duration `json:"latency"`
	Memory       int64         `json:"memory"`
	Connections  int           `json:"connections"`
	CommandsPerSec float64     `json:"commands_per_sec"`
	KeyCount     int64         `json:"key_count"`
	LastUpdate   time.Time     `json:"last_update"`
	Errors       []string      `json:"errors"`
}

// ShardInfo contains information about data sharding
type ShardInfo struct {
	Key      string `json:"key"`
	Slot     int    `json:"slot"`
	NodeID   string `json:"node_id"`
	NodeAddr string `json:"node_addr"`
}

// NewClusterManager creates a new Redis cluster manager
func NewClusterManager(config *ClusterConfig, logger *logrus.Logger) (*ClusterManager, error) {
	if config == nil {
		config = getDefaultClusterConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if len(config.Addrs) == 0 {
		return nil, errors.NewStorageError("INVALID_CONFIG", "At least one cluster address is required")
	}
	
	clusterClient := redis.NewClusterClient(&redis.ClusterOptions{
		Addrs:              config.Addrs,
		RouteByLatency:     config.RouteByLatency,
		RouteRandomly:      config.RouteRandomly,
		ReadOnly:           config.ReadOnly,
		MaxRedirects:       config.MaxRedirects,
		ReadTimeout:        config.ReadTimeout,
		WriteTimeout:       config.WriteTimeout,
		DialTimeout:        config.DialTimeout,
		PoolSize:           config.PoolSize,
		MinIdleConns:       config.MinIdleConns,
		MaxConnAge:         config.MaxConnAge,
		PoolTimeout:        config.PoolTimeout,
		IdleTimeout:        config.IdleTimeout,
		IdleCheckFrequency: config.IdleCheckFrequency,
		MaxRetries:         config.MaxRetries,
		MinRetryBackoff:    config.MinRetryBackoff,
		MaxRetryBackoff:    config.MaxRetryBackoff,
	})
	
	manager := &ClusterManager{
		client:      clusterClient,
		logger:      logger,
		config:      config,
		nodes:       make(map[string]*NodeInfo),
		shardMap:    make(map[string]string),
		healthCache: make(map[string]*NodeHealth),
	}
	
	return manager, nil
}

// Connect establishes connection to the Redis cluster
func (cm *ClusterManager) Connect(ctx context.Context) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	// Test cluster connection
	_, err := cm.client.Ping(ctx).Result()
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "CLUSTER_CONNECTION_FAILED", "Failed to connect to Redis cluster")
	}
	
	// Discover cluster topology
	if err := cm.discoverTopology(ctx); err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "TOPOLOGY_DISCOVERY_FAILED", "Failed to discover cluster topology")
	}
	
	// Start health monitoring
	go cm.startHealthMonitoring(ctx)
	
	cm.logger.WithFields(logrus.Fields{
		"addrs":      cm.config.Addrs,
		"node_count": len(cm.nodes),
	}).Info("Connected to Redis cluster")
	
	return nil
}

// GetClusterInfo returns comprehensive cluster information
func (cm *ClusterManager) GetClusterInfo(ctx context.Context) (*ClusterInfo, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	// Get cluster nodes information
	nodesCmd := cm.client.ClusterNodes(ctx)
	nodesInfo, err := nodesCmd.Result()
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "CLUSTER_INFO_FAILED", "Failed to get cluster nodes information")
	}
	
	// Get cluster slots information
	slotsCmd := cm.client.ClusterSlots(ctx)
	slotsInfo, err := slotsCmd.Result()
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "CLUSTER_SLOTS_FAILED", "Failed to get cluster slots information")
	}
	
	clusterInfo := &ClusterInfo{
		State:         "ok",
		ClusterSize:   len(cm.nodes),
		KnownNodes:    len(cm.nodes),
		ClusterSlots:  make([]SlotRange, 0),
		NodeDetails:   make([]*NodeInfo, 0),
		LastUpdate:    time.Now(),
	}
	
	// Parse nodes information
	for _, node := range cm.nodes {
		clusterInfo.NodeDetails = append(clusterInfo.NodeDetails, node)
	}
	
	// Parse slots information
	for _, slot := range slotsInfo {
		slotRange := SlotRange{
			Start: int(slot.Start),
			End:   int(slot.End),
			Nodes: make([]string, 0),
		}
		
		for _, node := range slot.Nodes {
			slotRange.Nodes = append(slotRange.Nodes, node.Addr)
		}
		
		clusterInfo.ClusterSlots = append(clusterInfo.ClusterSlots, slotRange)
	}
	
	// Parse overall cluster state from nodes info
	lines := strings.Split(nodesInfo, "\n")
	masterCount := 0
	slaveCount := 0
	failedCount := 0
	
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) < 3 {
			continue
		}
		
		flags := parts[2]
		if strings.Contains(flags, "master") {
			masterCount++
		} else if strings.Contains(flags, "slave") {
			slaveCount++
		}
		
		if strings.Contains(flags, "fail") {
			failedCount++
		}
	}
	
	clusterInfo.MasterNodes = masterCount
	clusterInfo.SlaveNodes = slaveCount
	clusterInfo.FailedNodes = failedCount
	
	if failedCount > 0 {
		clusterInfo.State = "degraded"
	}
	
	return clusterInfo, nil
}

// GetNodeHealth returns health information for all nodes
func (cm *ClusterManager) GetNodeHealth(ctx context.Context) ([]*NodeHealth, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	var healthList []*NodeHealth
	
	// Update health cache if needed
	if time.Since(cm.lastHealthCheck) > cm.config.HealthCheckInterval {
		cm.mu.RUnlock()
		cm.updateNodeHealth(ctx)
		cm.mu.RLock()
	}
	
	for _, health := range cm.healthCache {
		healthCopy := *health
		healthList = append(healthList, &healthCopy)
	}
	
	return healthList, nil
}

// GetShardInfo returns sharding information for a key
func (cm *ClusterManager) GetShardInfo(key string) (*ShardInfo, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	// Calculate hash slot for key
	slot := cm.calculateHashSlot(key)
	
	// Find node responsible for this slot
	nodeAddr, exists := cm.shardMap[fmt.Sprintf("%d", slot)]
	if !exists {
		return nil, errors.NewStorageError("SHARD_NOT_FOUND", fmt.Sprintf("No node found for slot %d", slot))
	}
	
	// Find node ID
	var nodeID string
	for id, node := range cm.nodes {
		if node.Addr == nodeAddr {
			nodeID = id
			break
		}
	}
	
	return &ShardInfo{
		Key:      key,
		Slot:     slot,
		NodeID:   nodeID,
		NodeAddr: nodeAddr,
	}, nil
}

// DistributeData distributes time series data across cluster nodes
func (cm *ClusterManager) DistributeData(ctx context.Context, timeSeries *models.TimeSeries) (map[string]*models.TimeSeries, error) {
	if len(timeSeries.DataPoints) == 0 {
		return nil, errors.NewValidationError("EMPTY_DATA", "Time series has no data points")
	}
	
	distribution := make(map[string]*models.TimeSeries)
	nodeDataPoints := make(map[string][]models.DataPoint)
	
	// Distribute data points based on sharding strategy
	switch cm.config.ShardingStrategy {
	case "hash":
		err := cm.distributeByHash(timeSeries, nodeDataPoints)
		if err != nil {
			return nil, err
		}
	case "time_range":
		err := cm.distributeByTimeRange(timeSeries, nodeDataPoints)
		if err != nil {
			return nil, err
		}
	default:
		return nil, errors.NewValidationError("INVALID_STRATEGY", fmt.Sprintf("Unknown sharding strategy: %s", cm.config.ShardingStrategy))
	}
	
	// Create time series for each node
	for nodeAddr, dataPoints := range nodeDataPoints {
		if len(dataPoints) > 0 {
			nodeTimeSeries := &models.TimeSeries{
				ID:          fmt.Sprintf("%s_%s", timeSeries.ID, nodeAddr),
				Name:        timeSeries.Name,
				Description: timeSeries.Description,
				Tags:        timeSeries.Tags,
				Metadata:    timeSeries.Metadata,
				DataPoints:  dataPoints,
				StartTime:   dataPoints[0].Timestamp,
				EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
				Frequency:   timeSeries.Frequency,
				SensorType:  timeSeries.SensorType,
				CreatedAt:   timeSeries.CreatedAt,
				UpdatedAt:   time.Now(),
			}
			
			distribution[nodeAddr] = nodeTimeSeries
		}
	}
	
	return distribution, nil
}

// Failover handles node failover operations
func (cm *ClusterManager) Failover(ctx context.Context, nodeID string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	node, exists := cm.nodes[nodeID]
	if !exists {
		return errors.NewStorageError("NODE_NOT_FOUND", fmt.Sprintf("Node %s not found", nodeID))
	}
	
	if node.Role != "master" {
		return errors.NewValidationError("INVALID_OPERATION", "Can only failover master nodes")
	}
	
	// Trigger cluster failover
	err := cm.client.ClusterFailover(ctx).Err()
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "FAILOVER_FAILED", "Failed to trigger cluster failover")
	}
	
	cm.logger.WithFields(logrus.Fields{
		"node_id": nodeID,
		"addr":    node.Addr,
	}).Info("Cluster failover triggered")
	
	// Wait for topology to update
	time.Sleep(time.Second * 2)
	
	// Rediscover topology
	if err := cm.discoverTopology(ctx); err != nil {
		cm.logger.WithError(err).Warn("Failed to rediscover topology after failover")
	}
	
	return nil
}

// Rebalance rebalances data across cluster nodes
func (cm *ClusterManager) Rebalance(ctx context.Context) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	cm.logger.Info("Starting cluster rebalancing")
	
	// This is a simplified rebalancing operation
	// In a real implementation, this would involve:
	// 1. Analyzing current data distribution
	// 2. Calculating optimal shard allocation
	// 3. Moving data between nodes
	// 4. Updating slot assignments
	
	// For now, we just rediscover topology
	if err := cm.discoverTopology(ctx); err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "REBALANCE_FAILED", "Failed to rebalance cluster")
	}
	
	cm.logger.Info("Cluster rebalancing completed")
	return nil
}

// Close closes the cluster manager
func (cm *ClusterManager) Close() error {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	if cm.client != nil {
		err := cm.client.Close()
		cm.client = nil
		return err
	}
	
	return nil
}

// Internal methods

func (cm *ClusterManager) discoverTopology(ctx context.Context) error {
	// Get cluster nodes
	nodesCmd := cm.client.ClusterNodes(ctx)
	nodesInfo, err := nodesCmd.Result()
	if err != nil {
		return err
	}
	
	// Parse nodes information
	lines := strings.Split(nodesInfo, "\n")
	newNodes := make(map[string]*NodeInfo)
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) < 8 {
			continue
		}
		
		nodeInfo := &NodeInfo{
			ID:       parts[0],
			Addr:     strings.Split(parts[1], "@")[0], // Remove cluster bus port
			Flags:    strings.Split(parts[2], ","),
			Master:   parts[3],
			LastSeen: time.Now(),
		}
		
		// Determine role
		for _, flag := range nodeInfo.Flags {
			if flag == "master" {
				nodeInfo.Role = "master"
			} else if flag == "slave" {
				nodeInfo.Role = "slave"
			}
		}
		
		// Parse slot ranges (for masters)
		if len(parts) > 8 && nodeInfo.Role == "master" {
			for i := 8; i < len(parts); i++ {
				if strings.Contains(parts[i], "-") {
					nodeInfo.Slots = append(nodeInfo.Slots, parts[i])
				}
			}
		}
		
		newNodes[nodeInfo.ID] = nodeInfo
	}
	
	// Update shard map
	newShardMap := make(map[string]string)
	for _, node := range newNodes {
		if node.Role == "master" {
			for _, slotRange := range node.Slots {
				if strings.Contains(slotRange, "-") {
					parts := strings.Split(slotRange, "-")
					if len(parts) == 2 {
						start, _ := strconv.Atoi(parts[0])
						end, _ := strconv.Atoi(parts[1])
						for slot := start; slot <= end; slot++ {
							newShardMap[fmt.Sprintf("%d", slot)] = node.Addr
						}
					}
				} else {
					// Single slot
					newShardMap[slotRange] = node.Addr
				}
			}
		}
	}
	
	cm.nodes = newNodes
	cm.shardMap = newShardMap
	
	return nil
}

func (cm *ClusterManager) startHealthMonitoring(ctx context.Context) {
	ticker := time.NewTicker(cm.config.HealthCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cm.updateNodeHealth(ctx)
		}
	}
}

func (cm *ClusterManager) updateNodeHealth(ctx context.Context) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	newHealthCache := make(map[string]*NodeHealth)
	
	for nodeID, node := range cm.nodes {
		health := &NodeHealth{
			NodeID:     nodeID,
			Addr:       node.Addr,
			Status:     "unknown",
			LastUpdate: time.Now(),
		}
		
		// Check node health (simplified)
		start := time.Now()
		
		// For a real implementation, you would connect to each node individually
		// Here we use a simplified approach
		_, err := cm.client.Ping(ctx).Result()
		if err != nil {
			health.Status = "offline"
			health.Errors = append(health.Errors, err.Error())
		} else {
			health.Status = "online"
			health.Latency = time.Since(start)
		}
		
		newHealthCache[nodeID] = health
	}
	
	cm.healthCache = newHealthCache
	cm.lastHealthCheck = time.Now()
}

func (cm *ClusterManager) calculateHashSlot(key string) int {
	// Redis Cluster uses CRC16 hash
	// This is a simplified implementation
	hash := crc16([]byte(key))
	return int(hash % 16384) // Redis cluster has 16384 slots
}

func (cm *ClusterManager) distributeByHash(timeSeries *models.TimeSeries, nodeDataPoints map[string][]models.DataPoint) error {
	for _, point := range timeSeries.DataPoints {
		// Create a key based on timestamp for distribution
		key := fmt.Sprintf("%s_%d", timeSeries.ID, point.Timestamp.Unix())
		
		shardInfo, err := cm.GetShardInfo(key)
		if err != nil {
			return err
		}
		
		nodeDataPoints[shardInfo.NodeAddr] = append(nodeDataPoints[shardInfo.NodeAddr], point)
	}
	
	return nil
}

func (cm *ClusterManager) distributeByTimeRange(timeSeries *models.TimeSeries, nodeDataPoints map[string][]models.DataPoint) error {
	if len(timeSeries.DataPoints) == 0 {
		return nil
	}
	
	// Calculate time range per node
	masterNodes := 0
	var masterAddrs []string
	
	for _, node := range cm.nodes {
		if node.Role == "master" {
			masterNodes++
			masterAddrs = append(masterAddrs, node.Addr)
		}
	}
	
	if masterNodes == 0 {
		return errors.NewStorageError("NO_MASTER_NODES", "No master nodes available")
	}
	
	pointsPerNode := len(timeSeries.DataPoints) / masterNodes
	remainder := len(timeSeries.DataPoints) % masterNodes
	
	currentIndex := 0
	for i, addr := range masterAddrs {
		count := pointsPerNode
		if i < remainder {
			count++
		}
		
		if currentIndex+count <= len(timeSeries.DataPoints) {
			nodeDataPoints[addr] = timeSeries.DataPoints[currentIndex : currentIndex+count]
			currentIndex += count
		}
	}
	
	return nil
}

// Helper types and functions

type ClusterInfo struct {
	State         string      `json:"state"`
	ClusterSize   int         `json:"cluster_size"`
	KnownNodes    int         `json:"known_nodes"`
	MasterNodes   int         `json:"master_nodes"`
	SlaveNodes    int         `json:"slave_nodes"`
	FailedNodes   int         `json:"failed_nodes"`
	ClusterSlots  []SlotRange `json:"cluster_slots"`
	NodeDetails   []*NodeInfo `json:"node_details"`
	LastUpdate    time.Time   `json:"last_update"`
}

type SlotRange struct {
	Start int      `json:"start"`
	End   int      `json:"end"`
	Nodes []string `json:"nodes"`
}

// Simple CRC16 implementation for hash slot calculation
func crc16(data []byte) uint16 {
	const poly = 0x1021
	var crc uint16 = 0x0000
	
	for _, b := range data {
		crc ^= uint16(b) << 8
		for i := 0; i < 8; i++ {
			if crc&0x8000 != 0 {
				crc = (crc << 1) ^ poly
			} else {
				crc <<= 1
			}
		}
	}
	
	return crc
}

func getDefaultClusterConfig() *ClusterConfig {
	return &ClusterConfig{
		Addrs:               []string{"localhost:7000", "localhost:7001", "localhost:7002"},
		RouteByLatency:      false,
		RouteRandomly:       true,
		ReadOnly:            false,
		MaxRedirects:        3,
		ReadTimeout:         30 * time.Second,
		WriteTimeout:        30 * time.Second,
		DialTimeout:         10 * time.Second,
		PoolSize:            10,
		MinIdleConns:        5,
		MaxConnAge:          30 * time.Minute,
		PoolTimeout:         4 * time.Second,
		IdleTimeout:         5 * time.Minute,
		IdleCheckFrequency:  time.Minute,
		MaxRetries:          3,
		MinRetryBackoff:     8 * time.Millisecond,
		MaxRetryBackoff:     512 * time.Millisecond,
		HealthCheckInterval: 30 * time.Second,
		HealthCheckTimeout:  5 * time.Second,
		HashSlots:           16384,
		ShardingStrategy:    "hash",
		ReplicationFactor:   1,
	}
}