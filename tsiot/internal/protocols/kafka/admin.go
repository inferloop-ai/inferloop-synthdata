package kafka

import (
	"context"
	"fmt"
	"time"

	"github.com/Shopify/sarama"
	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/pkg/errors"
)

// Admin provides Kafka administrative operations
type Admin struct {
	config *AdminConfig
	admin  sarama.ClusterAdmin
	logger *logrus.Logger
}

// AdminConfig contains configuration for Kafka admin operations
type AdminConfig struct {
	Brokers   []string       `json:"brokers"`
	ClientID  string         `json:"client_id"`
	Timeout   time.Duration  `json:"timeout"`
	Security  SecurityConfig `json:"security"`
	Metadata  MetadataConfig `json:"metadata"`
}

// TopicConfig contains configuration for creating topics
type TopicConfig struct {
	Name              string            `json:"name"`
	NumPartitions     int32             `json:"num_partitions"`
	ReplicationFactor int16             `json:"replication_factor"`
	ConfigEntries     map[string]string `json:"config_entries"`
}

// TopicDetail contains detailed information about a topic
type TopicDetail struct {
	Name              string                        `json:"name"`
	NumPartitions     int32                         `json:"num_partitions"`
	ReplicationFactor int16                         `json:"replication_factor"`
	ConfigEntries     map[string]string             `json:"config_entries"`
	Partitions        []PartitionDetail             `json:"partitions"`
}

// PartitionDetail contains information about a partition
type PartitionDetail struct {
	ID       int32   `json:"id"`
	Leader   int32   `json:"leader"`
	Replicas []int32 `json:"replicas"`
	ISR      []int32 `json:"isr"`
}

// ConsumerGroupDetail contains information about a consumer group
type ConsumerGroupDetail struct {
	GroupID     string                           `json:"group_id"`
	State       string                           `json:"state"`
	Protocol    string                           `json:"protocol"`
	ProtocolType string                          `json:"protocol_type"`
	Members     []ConsumerGroupMemberDetail      `json:"members"`
	Coordinator int32                            `json:"coordinator"`
}

// ConsumerGroupMemberDetail contains information about a consumer group member
type ConsumerGroupMemberDetail struct {
	MemberID     string            `json:"member_id"`
	ClientID     string            `json:"client_id"`
	ClientHost   string            `json:"client_host"`
	Assignment   map[string][]int32 `json:"assignment"`
}

// NewAdmin creates a new Kafka admin client
func NewAdmin(config *AdminConfig, logger *logrus.Logger) (*Admin, error) {
	if config == nil {
		return nil, errors.NewValidationError("INVALID_CONFIG", "Admin config cannot be nil")
	}
	
	if len(config.Brokers) == 0 {
		return nil, errors.NewValidationError("INVALID_BROKERS", "At least one broker must be specified")
	}
	
	// Create Sarama config
	saramaConfig := sarama.NewConfig()
	
	// Set client ID
	if config.ClientID != "" {
		saramaConfig.ClientID = config.ClientID
	}
	
	// Set timeout
	if config.Timeout > 0 {
		saramaConfig.Admin.Timeout = config.Timeout
	}
	
	// Set metadata configuration
	saramaConfig.Metadata.RefreshFrequency = config.Metadata.RefreshFrequency
	saramaConfig.Metadata.Full = config.Metadata.FullRefresh
	saramaConfig.Metadata.Retry.Max = config.Metadata.RetryMax
	saramaConfig.Metadata.Retry.Backoff = config.Metadata.RetryBackoff
	saramaConfig.Metadata.Timeout = config.Metadata.Timeout
	
	// Configure security
	if config.Security.Enabled {
		configureSecurity(saramaConfig, &config.Security)
	}
	
	// Create cluster admin
	admin, err := sarama.NewClusterAdmin(config.Brokers, saramaConfig)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeNetwork, "KAFKA_ADMIN_CREATE_FAILED", "Failed to create Kafka admin client")
	}
	
	return &Admin{
		config: config,
		admin:  admin,
		logger: logger,
	}, nil
}

// Close closes the admin client
func (a *Admin) Close() error {
	if err := a.admin.Close(); err != nil {
		a.logger.WithError(err).Error("Error closing Kafka admin client")
		return err
	}
	return nil
}

// CreateTopic creates a new topic
func (a *Admin) CreateTopic(ctx context.Context, topicConfig *TopicConfig) error {
	if topicConfig == nil {
		return errors.NewValidationError("INVALID_TOPIC_CONFIG", "Topic config cannot be nil")
	}
	
	if topicConfig.Name == "" {
		return errors.NewValidationError("INVALID_TOPIC_NAME", "Topic name cannot be empty")
	}
	
	// Check if topic already exists
	exists, err := a.TopicExists(ctx, topicConfig.Name)
	if err != nil {
		return err
	}
	
	if exists {
		return errors.NewValidationError("TOPIC_ALREADY_EXISTS", fmt.Sprintf("Topic '%s' already exists", topicConfig.Name))
	}
	
	// Create topic detail
	topicDetail := &sarama.TopicDetail{
		NumPartitions:     topicConfig.NumPartitions,
		ReplicationFactor: topicConfig.ReplicationFactor,
		ConfigEntries:     topicConfig.ConfigEntries,
	}
	
	// Create the topic
	err = a.admin.CreateTopic(topicConfig.Name, topicDetail, false)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "TOPIC_CREATE_FAILED", fmt.Sprintf("Failed to create topic '%s'", topicConfig.Name))
	}
	
	a.logger.WithFields(logrus.Fields{
		"topic":       topicConfig.Name,
		"partitions":  topicConfig.NumPartitions,
		"replication": topicConfig.ReplicationFactor,
	}).Info("Topic created successfully")
	
	return nil
}

// DeleteTopic deletes a topic
func (a *Admin) DeleteTopic(ctx context.Context, topicName string) error {
	if topicName == "" {
		return errors.NewValidationError("INVALID_TOPIC_NAME", "Topic name cannot be empty")
	}
	
	// Check if topic exists
	exists, err := a.TopicExists(ctx, topicName)
	if err != nil {
		return err
	}
	
	if !exists {
		return errors.NewValidationError("TOPIC_NOT_FOUND", fmt.Sprintf("Topic '%s' does not exist", topicName))
	}
	
	// Delete the topic
	err = a.admin.DeleteTopic(topicName)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "TOPIC_DELETE_FAILED", fmt.Sprintf("Failed to delete topic '%s'", topicName))
	}
	
	a.logger.WithField("topic", topicName).Info("Topic deleted successfully")
	
	return nil
}

// TopicExists checks if a topic exists
func (a *Admin) TopicExists(ctx context.Context, topicName string) (bool, error) {
	metadata, err := a.admin.DescribeTopics([]string{topicName})
	if err != nil {
		return false, errors.WrapError(err, errors.ErrorTypeStorage, "TOPIC_DESCRIBE_FAILED", fmt.Sprintf("Failed to describe topic '%s'", topicName))
	}
	
	_, exists := metadata[topicName]
	return exists, nil
}

// ListTopics lists all topics
func (a *Admin) ListTopics(ctx context.Context) ([]string, error) {
	metadata, err := a.admin.ListTopics()
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "TOPIC_LIST_FAILED", "Failed to list topics")
	}
	
	topics := make([]string, 0, len(metadata))
	for topic := range metadata {
		topics = append(topics, topic)
	}
	
	return topics, nil
}

// DescribeTopic gets detailed information about a topic
func (a *Admin) DescribeTopic(ctx context.Context, topicName string) (*TopicDetail, error) {
	if topicName == "" {
		return nil, errors.NewValidationError("INVALID_TOPIC_NAME", "Topic name cannot be empty")
	}
	
	metadata, err := a.admin.DescribeTopics([]string{topicName})
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "TOPIC_DESCRIBE_FAILED", fmt.Sprintf("Failed to describe topic '%s'", topicName))
	}
	
	topicMetadata, exists := metadata[topicName]
	if !exists {
		return nil, errors.NewValidationError("TOPIC_NOT_FOUND", fmt.Sprintf("Topic '%s' does not exist", topicName))
	}
	
	// Get topic configuration
	configResource := sarama.ConfigResource{
		Type: sarama.TopicResource,
		Name: topicName,
	}
	
	configs, err := a.admin.DescribeConfig(configResource)
	if err != nil {
		a.logger.WithError(err).Warn("Failed to get topic configuration")
	}
	
	// Convert to our format
	topicDetail := &TopicDetail{
		Name:              topicName,
		NumPartitions:     int32(len(topicMetadata.Partitions)),
		ConfigEntries:     make(map[string]string),
		Partitions:        make([]PartitionDetail, len(topicMetadata.Partitions)),
	}
	
	// Add configuration entries
	if configs != nil {
		for _, config := range configs {
			topicDetail.ConfigEntries[config.Name] = config.Value
		}
	}
	
	// Add partition details
	for i, partition := range topicMetadata.Partitions {
		topicDetail.Partitions[i] = PartitionDetail{
			ID:       partition.ID,
			Leader:   partition.Leader,
			Replicas: partition.Replicas,
			ISR:      partition.Isr,
		}
		
		// Set replication factor from first partition
		if i == 0 {
			topicDetail.ReplicationFactor = int16(len(partition.Replicas))
		}
	}
	
	return topicDetail, nil
}

// CreateConsumerGroup creates a new consumer group (this is more of a logical operation)
func (a *Admin) CreateConsumerGroup(ctx context.Context, groupID string) error {
	if groupID == "" {
		return errors.NewValidationError("INVALID_GROUP_ID", "Consumer group ID cannot be empty")
	}
	
	// Consumer groups are created automatically when consumers join
	// This is mainly for validation and logging
	a.logger.WithField("group_id", groupID).Info("Consumer group will be created when consumers join")
	
	return nil
}

// DeleteConsumerGroup deletes a consumer group
func (a *Admin) DeleteConsumerGroup(ctx context.Context, groupID string) error {
	if groupID == "" {
		return errors.NewValidationError("INVALID_GROUP_ID", "Consumer group ID cannot be empty")
	}
	
	err := a.admin.DeleteConsumerGroup(groupID)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "CONSUMER_GROUP_DELETE_FAILED", fmt.Sprintf("Failed to delete consumer group '%s'", groupID))
	}
	
	a.logger.WithField("group_id", groupID).Info("Consumer group deleted successfully")
	
	return nil
}

// ListConsumerGroups lists all consumer groups
func (a *Admin) ListConsumerGroups(ctx context.Context) ([]string, error) {
	groups, err := a.admin.ListConsumerGroups()
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "CONSUMER_GROUP_LIST_FAILED", "Failed to list consumer groups")
	}
	
	groupIDs := make([]string, 0, len(groups))
	for groupID := range groups {
		groupIDs = append(groupIDs, groupID)
	}
	
	return groupIDs, nil
}

// DescribeConsumerGroup gets detailed information about a consumer group
func (a *Admin) DescribeConsumerGroup(ctx context.Context, groupID string) (*ConsumerGroupDetail, error) {
	if groupID == "" {
		return nil, errors.NewValidationError("INVALID_GROUP_ID", "Consumer group ID cannot be empty")
	}
	
	descriptions, err := a.admin.DescribeConsumerGroups([]string{groupID})
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "CONSUMER_GROUP_DESCRIBE_FAILED", fmt.Sprintf("Failed to describe consumer group '%s'", groupID))
	}
	
	description, exists := descriptions[groupID]
	if !exists {
		return nil, errors.NewValidationError("CONSUMER_GROUP_NOT_FOUND", fmt.Sprintf("Consumer group '%s' does not exist", groupID))
	}
	
	// Convert to our format
	groupDetail := &ConsumerGroupDetail{
		GroupID:      groupID,
		State:        description.State,
		Protocol:     description.Protocol,
		ProtocolType: description.ProtocolType,
		Members:      make([]ConsumerGroupMemberDetail, len(description.Members)),
	}
	
	// Add member details
	for i, member := range description.Members {
		groupDetail.Members[i] = ConsumerGroupMemberDetail{
			MemberID:   member.MemberID,
			ClientID:   member.ClientID,
			ClientHost: member.ClientHost,
			Assignment: make(map[string][]int32),
		}
		
		// Parse member assignment (simplified)
		// In a real implementation, you would properly decode the assignment
		for topic, partitions := range member.MemberAssignment.Topics {
			groupDetail.Members[i].Assignment[topic] = partitions
		}
	}
	
	return groupDetail, nil
}

// GetBrokerInfo gets information about Kafka brokers
func (a *Admin) GetBrokerInfo(ctx context.Context) ([]BrokerInfo, error) {
	// Get cluster metadata
	req := &sarama.MetadataRequest{
		Version: 1,
	}
	
	// This is a simplified approach - in a real implementation you would
	// need to establish a connection to get broker metadata
	brokers := make([]BrokerInfo, len(a.config.Brokers))
	for i, broker := range a.config.Brokers {
		brokers[i] = BrokerInfo{
			ID:      int32(i),
			Address: broker,
			Rack:    "",
		}
	}
	
	return brokers, nil
}

// BrokerInfo contains information about a Kafka broker
type BrokerInfo struct {
	ID      int32  `json:"id"`
	Address string `json:"address"`
	Rack    string `json:"rack,omitempty"`
}

// CreateTimeSeriesTopic creates a topic specifically configured for time series data
func (a *Admin) CreateTimeSeriesTopic(ctx context.Context, topicName string, partitions int32, replicationFactor int16) error {
	topicConfig := &TopicConfig{
		Name:              topicName,
		NumPartitions:     partitions,
		ReplicationFactor: replicationFactor,
		ConfigEntries: map[string]string{
			"cleanup.policy":   "delete",
			"retention.ms":     "604800000", // 7 days
			"segment.ms":       "86400000",  // 1 day
			"compression.type": "snappy",
			"message.timestamp.type": "CreateTime",
		},
	}
	
	return a.CreateTopic(ctx, topicConfig)
}

// CreateCompactedTopic creates a topic with log compaction enabled
func (a *Admin) CreateCompactedTopic(ctx context.Context, topicName string, partitions int32, replicationFactor int16) error {
	topicConfig := &TopicConfig{
		Name:              topicName,
		NumPartitions:     partitions,
		ReplicationFactor: replicationFactor,
		ConfigEntries: map[string]string{
			"cleanup.policy":     "compact",
			"compression.type":   "snappy",
			"min.cleanable.dirty.ratio": "0.1",
			"segment.ms":         "86400000", // 1 day
		},
	}
	
	return a.CreateTopic(ctx, topicConfig)
}

// Health checks the health of the Kafka cluster
func (a *Admin) Health(ctx context.Context) error {
	// Try to list topics as a health check
	_, err := a.ListTopics(ctx)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeNetwork, "KAFKA_HEALTH_CHECK_FAILED", "Kafka cluster health check failed")
	}
	
	return nil
}

// DefaultAdminConfig returns a default admin configuration
func DefaultAdminConfig() *AdminConfig {
	return &AdminConfig{
		Brokers:  []string{"localhost:9092"},
		ClientID: "tsiot-admin",
		Timeout:  30 * time.Second,
		Security: SecurityConfig{
			Enabled: false,
		},
		Metadata: MetadataConfig{
			RefreshFrequency: 10 * time.Minute,
			FullRefresh:      true,
			RetryMax:         3,
			RetryBackoff:     250 * time.Millisecond,
			Timeout:          60 * time.Second,
		},
	}
}