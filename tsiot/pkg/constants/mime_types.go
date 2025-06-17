package constants

// MIME types for various content formats used in the TSIOT system
const (
	// Standard Web MIME types
	MimeTypeJSON            = "application/json"
	MimeTypeXML             = "application/xml"
	MimeTypeHTML            = "text/html"
	MimeTypePlainText       = "text/plain"
	MimeTypeFormURLEncoded  = "application/x-www-form-urlencoded"
	MimeTypeMultipartForm   = "multipart/form-data"
	MimeTypeOctetStream     = "application/octet-stream"
	
	// Data format MIME types
	MimeTypeCSV             = "text/csv"
	MimeTypeParquet         = "application/parquet"
	MimeTypeAvro            = "application/avro"
	MimeTypeHDF5            = "application/x-hdf5"
	MimeTypeArrow           = "application/vnd.apache.arrow.file"
	MimeTypePickle          = "application/x-pickle"
	MimeTypeNumPy           = "application/x-numpy"
	
	// Archive and compression MIME types
	MimeTypeGzip            = "application/gzip"
	MimeTypeBzip2           = "application/x-bzip2"
	MimeTypeXZ              = "application/x-xz"
	MimeTypeZip             = "application/zip"
	MimeTypeTar             = "application/x-tar"
	MimeTypeSnappy          = "application/x-snappy"
	MimeTypeLZ4             = "application/x-lz4"
	MimeTypeZstd            = "application/zstd"
	
	// Configuration and markup MIME types
	MimeTypeYAML            = "application/x-yaml"
	MimeTypeTOML            = "application/toml"
	MimeTypeINI             = "text/x-ini"
	MimeTypeProperties      = "text/x-java-properties"
	
	// Protocol buffer and schema MIME types
	MimeTypeProtobuf        = "application/x-protobuf"
	MimeTypeThrift          = "application/x-thrift"
	MimeTypeCapnProto       = "application/x-capnproto"
	MimeTypeFlatBuffers     = "application/x-flatbuffers"
	MimeTypeMsgPack         = "application/msgpack"
	
	// Time series specific MIME types
	MimeTypeInfluxLineProtocol = "application/x-influxdb-line-protocol"
	MimeTypePrometheusMetrics  = "text/plain; version=0.0.4"
	MimeTypeOpenMetrics        = "application/openmetrics-text"
	MimeTypeGraphite           = "text/x-graphite"
	MimeTypeCarbon             = "application/x-carbon"
	
	// Machine learning and model MIME types
	MimeTypeONNX            = "application/onnx"
	MimeTypeTensorFlow      = "application/x-tensorflow"
	MimeTypePyTorch         = "application/x-pytorch"
	MimeTypeH5              = "application/x-hdf5"
	MimeTypePKL             = "application/x-pickle"
	MimeTypeJoblib          = "application/x-joblib"
	MimeTypeMLflow          = "application/x-mlflow"
	
	// Database export MIME types
	MimeTypeSQL             = "application/sql"
	MimeTypeSQLite          = "application/x-sqlite3"
	MimeTypeMySQL           = "application/x-mysql"
	MimeTypePostgreSQL      = "application/x-postgresql"
	MimeTypeInfluxDB        = "application/x-influxdb"
	MimeTypeTimescaleDB     = "application/x-timescaledb"
	
	// Cloud storage MIME types
	MimeTypeS3Object        = "application/x-amazon-s3-object"
	MimeTypeGCSObject       = "application/x-google-cloud-storage-object"
	MimeTypeAzureBlob       = "application/x-azure-blob"
	
	// Streaming and messaging MIME types
	MimeTypeKafka           = "application/x-kafka"
	MimeTypeMQTT            = "application/x-mqtt"
	MimeTypeAMQP            = "application/x-amqp"
	MimeTypeRedisStream     = "application/x-redis-stream"
	MimeTypePulsar          = "application/x-pulsar"
	
	// Security and encryption MIME types
	MimeTypePEM             = "application/x-pem-file"
	MimeTypeX509            = "application/x-x509-ca-cert"
	MimeTypeJWS             = "application/jose+json"
	MimeTypeJWT             = "application/jwt"
	MimeTypeOpenPGP         = "application/pgp-encrypted"
	
	// Image MIME types (for charts/visualizations)
	MimeTypePNG             = "image/png"
	MimeTypeJPEG            = "image/jpeg"
	MimeTypeSVG             = "image/svg+xml"
	MimeTypePDF             = "application/pdf"
	MimeTypePS              = "application/postscript"
	MimeTypeEPS             = "application/eps"
	
	// Document MIME types
	MimeTypeMarkdown        = "text/markdown"
	MimeTypeRST             = "text/x-rst"
	MimeTypeAsciidoc        = "text/x-asciidoc"
	MimeTypeLaTeX           = "text/x-tex"
	
	// Log and monitoring MIME types
	MimeTypeLog             = "text/x-log"
	MimeTypeSyslog          = "text/x-syslog"
	MimeTypeJSONLines       = "application/x-ndjson"
	MimeTypeLogfmt          = "text/x-logfmt"
	
	// Statistical and scientific MIME types
	MimeTypeR               = "application/x-r-data"
	MimeTypeStata           = "application/x-stata"
	MimeTypeSPSS            = "application/x-spss"
	MimeTypeSAS             = "application/x-sas"
	MimeTypeMatlab          = "application/x-matlab-data"
	
	// Notebook MIME types
	MimeTypeJupyter         = "application/x-ipynb+json"
	MimeTypeZeppelin        = "application/x-zeppelin-notebook"
	MimeTypeObservable      = "application/x-observable-notebook"
	
	// API and schema MIME types
	MimeTypeOpenAPI         = "application/vnd.oai.openapi"
	MimeTypeAsyncAPI        = "application/vnd.aai.asyncapi"
	MimeTypeGraphQL         = "application/graphql"
	MimeTypeJSONSchema      = "application/schema+json"
	MimeTypeXMLSchema       = "application/xml-schema"
	
	// Synthetic data specific MIME types
	MimeTypeSyntheticData   = "application/x-synthetic-data"
	MimeTypeTimeGAN         = "application/x-timegan-model"
	MimeTypeARIMA           = "application/x-arima-model"
	MimeTypeLSTM            = "application/x-lstm-model"
	MimeTypeStatistical     = "application/x-statistical-model"
	MimeTypeValidationReport = "application/x-validation-report"
	MimeTypeQualityMetrics  = "application/x-quality-metrics"
	MimeTypePrivacyReport   = "application/x-privacy-report"
)

// MimeTypeMap provides a mapping from file extensions to MIME types
var MimeTypeMap = map[string]string{
	// Data formats
	".json":     MimeTypeJSON,
	".xml":      MimeTypeXML,
	".csv":      MimeTypeCSV,
	".parquet":  MimeTypeParquet,
	".avro":     MimeTypeAvro,
	".hdf5":     MimeTypeHDF5,
	".h5":       MimeTypeHDF5,
	".arrow":    MimeTypeArrow,
	".pkl":      MimeTypePickle,
	".pickle":   MimeTypePickle,
	".npy":      MimeTypeNumPy,
	".npz":      MimeTypeNumPy,
	
	// Compression
	".gz":       MimeTypeGzip,
	".bz2":      MimeTypeBzip2,
	".xz":       MimeTypeXZ,
	".zip":      MimeTypeZip,
	".tar":      MimeTypeTar,
	".snappy":   MimeTypeSnappy,
	".lz4":      MimeTypeLZ4,
	".zst":      MimeTypeZstd,
	
	// Configuration
	".yaml":     MimeTypeYAML,
	".yml":      MimeTypeYAML,
	".toml":     MimeTypeTOML,
	".ini":      MimeTypeINI,
	".properties": MimeTypeProperties,
	
	// Protocol formats
	".proto":    MimeTypeProtobuf,
	".pb":       MimeTypeProtobuf,
	".thrift":   MimeTypeThrift,
	".capnp":    MimeTypeCapnProto,
	".fbs":      MimeTypeFlatBuffers,
	".msgpack":  MimeTypeMsgPack,
	
	// Machine learning
	".onnx":     MimeTypeONNX,
	".pb":       MimeTypeTensorFlow,
	".pth":      MimeTypePyTorch,
	".pt":       MimeTypePyTorch,
	".joblib":   MimeTypeJoblib,
	
	// Database
	".sql":      MimeTypeSQL,
	".sqlite":   MimeTypeSQLite,
	".sqlite3":  MimeTypeSQLite,
	".db":       MimeTypeSQLite,
	
	// Security
	".pem":      MimeTypePEM,
	".crt":      MimeTypeX509,
	".cert":     MimeTypeX509,
	".cer":      MimeTypeX509,
	".p12":      "application/x-pkcs12",
	".pfx":      "application/x-pkcs12",
	
	// Images
	".png":      MimeTypePNG,
	".jpg":      MimeTypeJPEG,
	".jpeg":     MimeTypeJPEG,
	".svg":      MimeTypeSVG,
	".pdf":      MimeTypePDF,
	".ps":       MimeTypePS,
	".eps":      MimeTypeEPS,
	
	// Documents
	".md":       MimeTypeMarkdown,
	".markdown": MimeTypeMarkdown,
	".rst":      MimeTypeRST,
	".adoc":     MimeTypeAsciidoc,
	".tex":      MimeTypeLaTeX,
	
	// Logs
	".log":      MimeTypeLog,
	".jsonl":    MimeTypeJSONLines,
	".ndjson":   MimeTypeJSONLines,
	
	// Scientific
	".rdata":    MimeTypeR,
	".rds":      MimeTypeR,
	".dta":      MimeTypeStata,
	".sav":      MimeTypeSPSS,
	".sas7bdat": MimeTypeSAS,
	".mat":      MimeTypeMatlab,
	
	// Notebooks
	".ipynb":    MimeTypeJupyter,
	
	// Text
	".txt":      MimeTypePlainText,
	".text":     MimeTypePlainText,
	".html":     MimeTypeHTML,
	".htm":      MimeTypeHTML,
}

// GetMimeTypeByExtension returns the MIME type for a given file extension
func GetMimeTypeByExtension(ext string) string {
	if mimeType, exists := MimeTypeMap[ext]; exists {
		return mimeType
	}
	return MimeTypeOctetStream // Default fallback
}

// IsDataFormat checks if the MIME type represents a data format
func IsDataFormat(mimeType string) bool {
	dataFormats := []string{
		MimeTypeJSON, MimeTypeXML, MimeTypeCSV, MimeTypeParquet,
		MimeTypeAvro, MimeTypeHDF5, MimeTypeArrow, MimeTypePickle,
		MimeTypeNumPy, MimeTypeInfluxLineProtocol,
	}
	
	for _, format := range dataFormats {
		if mimeType == format {
			return true
		}
	}
	return false
}

// IsCompressed checks if the MIME type represents a compressed format
func IsCompressed(mimeType string) bool {
	compressedFormats := []string{
		MimeTypeGzip, MimeTypeBzip2, MimeTypeXZ, MimeTypeZip,
		MimeTypeTar, MimeTypeSnappy, MimeTypeLZ4, MimeTypeZstd,
	}
	
	for _, format := range compressedFormats {
		if mimeType == format {
			return true
		}
	}
	return false
}

// IsMachineLearningFormat checks if the MIME type represents an ML model format
func IsMachineLearningFormat(mimeType string) bool {
	mlFormats := []string{
		MimeTypeONNX, MimeTypeTensorFlow, MimeTypePyTorch,
		MimeTypeH5, MimeTypePKL, MimeTypeJoblib, MimeTypeMLflow,
		MimeTypeTimeGAN, MimeTypeARIMA, MimeTypeLSTM, MimeTypeStatistical,
	}
	
	for _, format := range mlFormats {
		if mimeType == format {
			return true
		}
	}
	return false
}

// IsStreamingFormat checks if the MIME type represents a streaming format
func IsStreamingFormat(mimeType string) bool {
	streamingFormats := []string{
		MimeTypeKafka, MimeTypeMQTT, MimeTypeAMQP,
		MimeTypeRedisStream, MimeTypePulsar,
	}
	
	for _, format := range streamingFormats {
		if mimeType == format {
			return true
		}
	}
	return false
}

// IsTimeSeriesFormat checks if the MIME type represents a time series format
func IsTimeSeriesFormat(mimeType string) bool {
	timeSeriesFormats := []string{
		MimeTypeInfluxLineProtocol, MimeTypePrometheusMetrics,
		MimeTypeOpenMetrics, MimeTypeGraphite, MimeTypeCarbon,
		MimeTypeInfluxDB, MimeTypeTimescaleDB,
	}
	
	for _, format := range timeSeriesFormats {
		if mimeType == format {
			return true
		}
	}
	return false
}