package encoding

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"encoding/xml"
	"fmt"

	"github.com/vmihailenco/msgpack/v5"
	"google.golang.org/protobuf/proto"
	"gopkg.in/yaml.v3"
)

// SerializationFormat represents different serialization formats
type SerializationFormat int

const (
	JSON SerializationFormat = iota
	XML
	YAML
	MessagePack
	Gob
	Protobuf
)

// Serializer interface for different serialization implementations
type Serializer interface {
	Serialize(data interface{}) ([]byte, error)
	Deserialize(data []byte, target interface{}) error
	Format() SerializationFormat
	ContentType() string
}

// JSONSerializer implements JSON serialization
type JSONSerializer struct {
	indent bool
}

// NewJSONSerializer creates a new JSON serializer
func NewJSONSerializer(indent bool) *JSONSerializer {
	return &JSONSerializer{indent: indent}
}

// Serialize serializes data to JSON
func (j *JSONSerializer) Serialize(data interface{}) ([]byte, error) {
	if j.indent {
		return json.MarshalIndent(data, "", "  ")
	}
	return json.Marshal(data)
}

// Deserialize deserializes JSON data
func (j *JSONSerializer) Deserialize(data []byte, target interface{}) error {
	return json.Unmarshal(data, target)
}

// Format returns the serialization format
func (j *JSONSerializer) Format() SerializationFormat {
	return JSON
}

// ContentType returns the MIME content type
func (j *JSONSerializer) ContentType() string {
	return "application/json"
}

// XMLSerializer implements XML serialization
type XMLSerializer struct {
	indent bool
}

// NewXMLSerializer creates a new XML serializer
func NewXMLSerializer(indent bool) *XMLSerializer {
	return &XMLSerializer{indent: indent}
}

// Serialize serializes data to XML
func (x *XMLSerializer) Serialize(data interface{}) ([]byte, error) {
	if x.indent {
		return xml.MarshalIndent(data, "", "  ")
	}
	return xml.Marshal(data)
}

// Deserialize deserializes XML data
func (x *XMLSerializer) Deserialize(data []byte, target interface{}) error {
	return xml.Unmarshal(data, target)
}

// Format returns the serialization format
func (x *XMLSerializer) Format() SerializationFormat {
	return XML
}

// ContentType returns the MIME content type
func (x *XMLSerializer) ContentType() string {
	return "application/xml"
}

// YAMLSerializer implements YAML serialization
type YAMLSerializer struct{}

// NewYAMLSerializer creates a new YAML serializer
func NewYAMLSerializer() *YAMLSerializer {
	return &YAMLSerializer{}
}

// Serialize serializes data to YAML
func (y *YAMLSerializer) Serialize(data interface{}) ([]byte, error) {
	return yaml.Marshal(data)
}

// Deserialize deserializes YAML data
func (y *YAMLSerializer) Deserialize(data []byte, target interface{}) error {
	return yaml.Unmarshal(data, target)
}

// Format returns the serialization format
func (y *YAMLSerializer) Format() SerializationFormat {
	return YAML
}

// ContentType returns the MIME content type
func (y *YAMLSerializer) ContentType() string {
	return "application/x-yaml"
}

// MessagePackSerializer implements MessagePack serialization
type MessagePackSerializer struct{}

// NewMessagePackSerializer creates a new MessagePack serializer
func NewMessagePackSerializer() *MessagePackSerializer {
	return &MessagePackSerializer{}
}

// Serialize serializes data to MessagePack
func (m *MessagePackSerializer) Serialize(data interface{}) ([]byte, error) {
	return msgpack.Marshal(data)
}

// Deserialize deserializes MessagePack data
func (m *MessagePackSerializer) Deserialize(data []byte, target interface{}) error {
	return msgpack.Unmarshal(data, target)
}

// Format returns the serialization format
func (m *MessagePackSerializer) Format() SerializationFormat {
	return MessagePack
}

// ContentType returns the MIME content type
func (m *MessagePackSerializer) ContentType() string {
	return "application/msgpack"
}

// GobSerializer implements Go's gob serialization
type GobSerializer struct{}

// NewGobSerializer creates a new Gob serializer
func NewGobSerializer() *GobSerializer {
	return &GobSerializer{}
}

// Serialize serializes data using gob encoding
func (g *GobSerializer) Serialize(data interface{}) ([]byte, error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	if err := encoder.Encode(data); err != nil {
		return nil, fmt.Errorf("gob encode failed: %w", err)
	}
	return buf.Bytes(), nil
}

// Deserialize deserializes gob data
func (g *GobSerializer) Deserialize(data []byte, target interface{}) error {
	buf := bytes.NewBuffer(data)
	decoder := gob.NewDecoder(buf)
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("gob decode failed: %w", err)
	}
	return nil
}

// Format returns the serialization format
func (g *GobSerializer) Format() SerializationFormat {
	return Gob
}

// ContentType returns the MIME content type
func (g *GobSerializer) ContentType() string {
	return "application/gob"
}

// ProtobufSerializer implements Protocol Buffers serialization
type ProtobufSerializer struct{}

// NewProtobufSerializer creates a new Protobuf serializer
func NewProtobufSerializer() *ProtobufSerializer {
	return &ProtobufSerializer{}
}

// Serialize serializes data using Protocol Buffers
func (p *ProtobufSerializer) Serialize(data interface{}) ([]byte, error) {
	if msg, ok := data.(proto.Message); ok {
		return proto.Marshal(msg)
	}
	return nil, fmt.Errorf("data must implement proto.Message interface")
}

// Deserialize deserializes Protocol Buffers data
func (p *ProtobufSerializer) Deserialize(data []byte, target interface{}) error {
	if msg, ok := target.(proto.Message); ok {
		return proto.Unmarshal(data, msg)
	}
	return fmt.Errorf("target must implement proto.Message interface")
}

// Format returns the serialization format
func (p *ProtobufSerializer) Format() SerializationFormat {
	return Protobuf
}

// ContentType returns the MIME content type
func (p *ProtobufSerializer) ContentType() string {
	return "application/protobuf"
}

// SerializationFactory creates serializers based on format
type SerializationFactory struct{}

// NewSerializationFactory creates a new serialization factory
func NewSerializationFactory() *SerializationFactory {
	return &SerializationFactory{}
}

// CreateSerializer creates a serializer for the specified format
func (sf *SerializationFactory) CreateSerializer(format SerializationFormat, options ...interface{}) (Serializer, error) {
	switch format {
	case JSON:
		indent := false
		if len(options) > 0 {
			if i, ok := options[0].(bool); ok {
				indent = i
			}
		}
		return NewJSONSerializer(indent), nil
	case XML:
		indent := false
		if len(options) > 0 {
			if i, ok := options[0].(bool); ok {
				indent = i
			}
		}
		return NewXMLSerializer(indent), nil
	case YAML:
		return NewYAMLSerializer(), nil
	case MessagePack:
		return NewMessagePackSerializer(), nil
	case Gob:
		return NewGobSerializer(), nil
	case Protobuf:
		return NewProtobufSerializer(), nil
	default:
		return nil, fmt.Errorf("unsupported serialization format: %d", format)
	}
}

// SerializationConfig holds serialization configuration
type SerializationConfig struct {
	Format           SerializationFormat
	Indent           bool
	Compression      bool
	Encryption       bool
	CompressionType  CompressionType
	EncAlgorithm     EncryptionAlgorithm
}

// SerializationManager manages serialization operations with optional compression and encryption
type SerializationManager struct {
	serializer  Serializer
	config      *SerializationConfig
	compressor  Compressor
	encryptor   Encryptor
}

// NewSerializationManager creates a new serialization manager
func NewSerializationManager(config *SerializationConfig) (*SerializationManager, error) {
	factory := NewSerializationFactory()
	serializer, err := factory.CreateSerializer(config.Format, config.Indent)
	if err != nil {
		return nil, err
	}

	manager := &SerializationManager{
		serializer: serializer,
		config:     config,
	}

	// Initialize compressor if compression is enabled
	if config.Compression {
		compFactory := NewCompressionFactory()
		compConfig := CompressorConfig{
			Type:  config.CompressionType,
			Level: CompressionLevelDefault,
		}
		manager.compressor, err = compFactory.CreateCompressor(compConfig)
		if err != nil {
			return nil, err
		}
	}

	// Initialize encryptor if encryption is enabled
	if config.Encryption {
		encFactory := NewEncryptionFactory()
		manager.encryptor, err = encFactory.CreateEncryptor(config.EncAlgorithm)
		if err != nil {
			return nil, err
		}
	}

	return manager, nil
}

// Serialize serializes data with optional compression and encryption
func (sm *SerializationManager) Serialize(data interface{}, key []byte) ([]byte, error) {
	// First serialize the data
	serialized, err := sm.serializer.Serialize(data)
	if err != nil {
		return nil, fmt.Errorf("serialization failed: %w", err)
	}

	result := serialized

	// Apply compression if enabled
	if sm.config.Compression && sm.compressor != nil {
		compressed, err := sm.compressor.Compress(result)
		if err != nil {
			return nil, fmt.Errorf("compression failed: %w", err)
		}
		result = compressed
	}

	// Apply encryption if enabled
	if sm.config.Encryption && sm.encryptor != nil {
		if len(key) == 0 {
			return nil, fmt.Errorf("encryption key required")
		}
		encrypted, err := sm.encryptor.Encrypt(result, key)
		if err != nil {
			return nil, fmt.Errorf("encryption failed: %w", err)
		}
		result = encrypted
	}

	return result, nil
}

// Deserialize deserializes data with optional decryption and decompression
func (sm *SerializationManager) Deserialize(data []byte, target interface{}, key []byte) error {
	result := data

	// Apply decryption if enabled
	if sm.config.Encryption && sm.encryptor != nil {
		if len(key) == 0 {
			return fmt.Errorf("decryption key required")
		}
		decrypted, err := sm.encryptor.Decrypt(result, key)
		if err != nil {
			return fmt.Errorf("decryption failed: %w", err)
		}
		result = decrypted
	}

	// Apply decompression if enabled
	if sm.config.Compression && sm.compressor != nil {
		decompressed, err := sm.compressor.Decompress(result)
		if err != nil {
			return fmt.Errorf("decompression failed: %w", err)
		}
		result = decompressed
	}

	// Finally deserialize the data
	if err := sm.serializer.Deserialize(result, target); err != nil {
		return fmt.Errorf("deserialization failed: %w", err)
	}

	return nil
}

// SerializedData represents serialized data with metadata
type SerializedData struct {
	Format      SerializationFormat `json:"format"`
	Data        []byte              `json:"data"`
	Compressed  bool                `json:"compressed"`
	Encrypted   bool                `json:"encrypted"`
	ContentType string              `json:"content_type"`
	Size        int                 `json:"size"`
	Metadata    map[string]string   `json:"metadata,omitempty"`
}

// NewSerializedData creates a new serialized data structure
func NewSerializedData(format SerializationFormat, data []byte, contentType string) *SerializedData {
	return &SerializedData{
		Format:      format,
		Data:        data,
		ContentType: contentType,
		Size:        len(data),
		Metadata:    make(map[string]string),
	}
}

// SerializationBenchmark benchmarks different serialization formats
func SerializationBenchmark(data interface{}) map[SerializationFormat]*SerializationStats {
	results := make(map[SerializationFormat]*SerializationStats)
	factory := NewSerializationFactory()

	formats := []SerializationFormat{JSON, XML, YAML, MessagePack, Gob}

	for _, format := range formats {
		serializer, err := factory.CreateSerializer(format)
		if err != nil {
			continue
		}

		// Serialize
		serialized, err := serializer.Serialize(data)
		if err != nil {
			continue
		}

		// Test deserialization
		var target interface{}
		err = serializer.Deserialize(serialized, &target)
		if err != nil {
			continue
		}

		stats := &SerializationStats{
			Format:       format,
			Size:         len(serialized),
			ContentType:  serializer.ContentType(),
			SerializeOK:  true,
			DeserializeOK: true,
		}

		results[format] = stats
	}

	return results
}

// SerializationStats holds serialization benchmark statistics
type SerializationStats struct {
	Format        SerializationFormat
	Size          int
	ContentType   string
	SerializeOK   bool
	DeserializeOK bool
}

// StreamSerializer provides streaming serialization for large datasets
type StreamSerializer struct {
	serializer Serializer
	buffer     bytes.Buffer
}

// NewStreamSerializer creates a new stream serializer
func NewStreamSerializer(format SerializationFormat) (*StreamSerializer, error) {
	factory := NewSerializationFactory()
	serializer, err := factory.CreateSerializer(format)
	if err != nil {
		return nil, err
	}

	return &StreamSerializer{
		serializer: serializer,
	}, nil
}

// SerializeItem serializes a single item and appends to buffer
func (ss *StreamSerializer) SerializeItem(item interface{}) error {
	data, err := ss.serializer.Serialize(item)
	if err != nil {
		return err
	}

	// Add length prefix for streaming
	length := len(data)
	lengthBytes := make([]byte, 4)
	lengthBytes[0] = byte(length >> 24)
	lengthBytes[1] = byte(length >> 16)
	lengthBytes[2] = byte(length >> 8)
	lengthBytes[3] = byte(length)

	ss.buffer.Write(lengthBytes)
	ss.buffer.Write(data)

	return nil
}

// GetBuffer returns the serialized buffer
func (ss *StreamSerializer) GetBuffer() []byte {
	return ss.buffer.Bytes()
}

// Reset resets the stream serializer
func (ss *StreamSerializer) Reset() {
	ss.buffer.Reset()
}

// StreamDeserializer provides streaming deserialization
type StreamDeserializer struct {
	serializer Serializer
	buffer     *bytes.Buffer
}

// NewStreamDeserializer creates a new stream deserializer
func NewStreamDeserializer(format SerializationFormat, data []byte) (*StreamDeserializer, error) {
	factory := NewSerializationFactory()
	serializer, err := factory.CreateSerializer(format)
	if err != nil {
		return nil, err
	}

	return &StreamDeserializer{
		serializer: serializer,
		buffer:     bytes.NewBuffer(data),
	}, nil
}

// DeserializeNext deserializes the next item from the stream
func (sd *StreamDeserializer) DeserializeNext(target interface{}) error {
	// Read length prefix
	lengthBytes := make([]byte, 4)
	n, err := sd.buffer.Read(lengthBytes)
	if err != nil || n != 4 {
		return fmt.Errorf("failed to read length prefix: %w", err)
	}

	length := int(lengthBytes[0])<<24 | int(lengthBytes[1])<<16 | int(lengthBytes[2])<<8 | int(lengthBytes[3])

	// Read data
	data := make([]byte, length)
	n, err = sd.buffer.Read(data)
	if err != nil || n != length {
		return fmt.Errorf("failed to read data: %w", err)
	}

	return sd.serializer.Deserialize(data, target)
}

// HasNext checks if there are more items to deserialize
func (sd *StreamDeserializer) HasNext() bool {
	return sd.buffer.Len() >= 4
}

// ConvertFormat converts data from one serialization format to another
func ConvertFormat(data []byte, fromFormat, toFormat SerializationFormat) ([]byte, error) {
	factory := NewSerializationFactory()

	// Create deserializer for source format
	fromSerializer, err := factory.CreateSerializer(fromFormat)
	if err != nil {
		return nil, err
	}

	// Create serializer for target format
	toSerializer, err := factory.CreateSerializer(toFormat)
	if err != nil {
		return nil, err
	}

	// Deserialize from source format
	var intermediate interface{}
	if err := fromSerializer.Deserialize(data, &intermediate); err != nil {
		return nil, fmt.Errorf("failed to deserialize from %d: %w", fromFormat, err)
	}

	// Serialize to target format
	result, err := toSerializer.Serialize(intermediate)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize to %d: %w", toFormat, err)
	}

	return result, nil
}

// ValidateSerializedData validates if data can be properly deserialized
func ValidateSerializedData(data []byte, format SerializationFormat) error {
	factory := NewSerializationFactory()
	serializer, err := factory.CreateSerializer(format)
	if err != nil {
		return err
	}

	var target interface{}
	return serializer.Deserialize(data, &target)
}

// GetOptimalFormat analyzes data and recommends the best serialization format
func GetOptimalFormat(data interface{}, prioritizeSize bool) (SerializationFormat, error) {
	benchmarks := SerializationBenchmark(data)

	var bestFormat SerializationFormat
	var bestScore float64

	for format, stats := range benchmarks {
		if !stats.SerializeOK || !stats.DeserializeOK {
			continue
		}

		var score float64
		if prioritizeSize {
			// Lower size is better
			score = 1.0 / float64(stats.Size)
		} else {
			// Balance between size and compatibility
			score = 0.7/float64(stats.Size) + 0.3 // Add base compatibility score
		}

		if score > bestScore {
			bestScore = score
			bestFormat = format
		}
	}

	if bestScore == 0 {
		return JSON, fmt.Errorf("no suitable format found, defaulting to JSON")
	}

	return bestFormat, nil
}