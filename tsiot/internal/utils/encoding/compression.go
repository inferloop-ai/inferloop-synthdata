package encoding

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"compress/lzw"
	"compress/zlib"
	"fmt"
	"io"
)

// CompressionType represents different compression algorithms
type CompressionType int

const (
	CompressionGZIP CompressionType = iota
	CompressionZLIB
	CompressionDEFLATE
	CompressionLZW
)

// CompressionLevel defines compression levels
type CompressionLevel int

const (
	CompressionLevelDefault CompressionLevel = iota
	CompressionLevelFast
	CompressionLevelBest
	CompressionLevelNone
)

// Compressor interface for different compression implementations
type Compressor interface {
	Compress(data []byte) ([]byte, error)
	Decompress(data []byte) ([]byte, error)
	Type() CompressionType
	Level() CompressionLevel
}

// CompressorConfig holds configuration for compression
type CompressorConfig struct {
	Type  CompressionType
	Level CompressionLevel
}

// GZIPCompressor implements GZIP compression
type GZIPCompressor struct {
	level int
}

// NewGZIPCompressor creates a new GZIP compressor
func NewGZIPCompressor(level CompressionLevel) *GZIPCompressor {
	var gzipLevel int
	switch level {
	case CompressionLevelFast:
		gzipLevel = gzip.BestSpeed
	case CompressionLevelBest:
		gzipLevel = gzip.BestCompression
	case CompressionLevelNone:
		gzipLevel = gzip.NoCompression
	default:
		gzipLevel = gzip.DefaultCompression
	}
	return &GZIPCompressor{level: gzipLevel}
}

// Compress compresses data using GZIP
func (g *GZIPCompressor) Compress(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer, err := gzip.NewWriterLevel(&buf, g.level)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip writer: %w", err)
	}
	defer writer.Close()

	if _, err := writer.Write(data); err != nil {
		return nil, fmt.Errorf("failed to write data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close gzip writer: %w", err)
	}

	return buf.Bytes(), nil
}

// Decompress decompresses GZIP data
func (g *GZIPCompressor) Decompress(data []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer reader.Close()

	result, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read decompressed data: %w", err)
	}

	return result, nil
}

// Type returns the compression type
func (g *GZIPCompressor) Type() CompressionType {
	return CompressionGZIP
}

// Level returns the compression level
func (g *GZIPCompressor) Level() CompressionLevel {
	switch g.level {
	case gzip.BestSpeed:
		return CompressionLevelFast
	case gzip.BestCompression:
		return CompressionLevelBest
	case gzip.NoCompression:
		return CompressionLevelNone
	default:
		return CompressionLevelDefault
	}
}

// ZLIBCompressor implements ZLIB compression
type ZLIBCompressor struct {
	level int
}

// NewZLIBCompressor creates a new ZLIB compressor
func NewZLIBCompressor(level CompressionLevel) *ZLIBCompressor {
	var zlibLevel int
	switch level {
	case CompressionLevelFast:
		zlibLevel = zlib.BestSpeed
	case CompressionLevelBest:
		zlibLevel = zlib.BestCompression
	case CompressionLevelNone:
		zlibLevel = zlib.NoCompression
	default:
		zlibLevel = zlib.DefaultCompression
	}
	return &ZLIBCompressor{level: zlibLevel}
}

// Compress compresses data using ZLIB
func (z *ZLIBCompressor) Compress(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer, err := zlib.NewWriterLevel(&buf, z.level)
	if err != nil {
		return nil, fmt.Errorf("failed to create zlib writer: %w", err)
	}
	defer writer.Close()

	if _, err := writer.Write(data); err != nil {
		return nil, fmt.Errorf("failed to write data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close zlib writer: %w", err)
	}

	return buf.Bytes(), nil
}

// Decompress decompresses ZLIB data
func (z *ZLIBCompressor) Decompress(data []byte) ([]byte, error) {
	reader, err := zlib.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create zlib reader: %w", err)
	}
	defer reader.Close()

	result, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read decompressed data: %w", err)
	}

	return result, nil
}

// Type returns the compression type
func (z *ZLIBCompressor) Type() CompressionType {
	return CompressionZLIB
}

// Level returns the compression level
func (z *ZLIBCompressor) Level() CompressionLevel {
	switch z.level {
	case zlib.BestSpeed:
		return CompressionLevelFast
	case zlib.BestCompression:
		return CompressionLevelBest
	case zlib.NoCompression:
		return CompressionLevelNone
	default:
		return CompressionLevelDefault
	}
}

// DEFLATECompressor implements DEFLATE compression
type DEFLATECompressor struct {
	level int
}

// NewDEFLATECompressor creates a new DEFLATE compressor
func NewDEFLATECompressor(level CompressionLevel) *DEFLATECompressor {
	var deflateLevel int
	switch level {
	case CompressionLevelFast:
		deflateLevel = flate.BestSpeed
	case CompressionLevelBest:
		deflateLevel = flate.BestCompression
	case CompressionLevelNone:
		deflateLevel = flate.NoCompression
	default:
		deflateLevel = flate.DefaultCompression
	}
	return &DEFLATECompressor{level: deflateLevel}
}

// Compress compresses data using DEFLATE
func (d *DEFLATECompressor) Compress(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer, err := flate.NewWriter(&buf, d.level)
	if err != nil {
		return nil, fmt.Errorf("failed to create deflate writer: %w", err)
	}
	defer writer.Close()

	if _, err := writer.Write(data); err != nil {
		return nil, fmt.Errorf("failed to write data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close deflate writer: %w", err)
	}

	return buf.Bytes(), nil
}

// Decompress decompresses DEFLATE data
func (d *DEFLATECompressor) Decompress(data []byte) ([]byte, error) {
	reader := flate.NewReader(bytes.NewReader(data))
	defer reader.Close()

	result, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read decompressed data: %w", err)
	}

	return result, nil
}

// Type returns the compression type
func (d *DEFLATECompressor) Type() CompressionType {
	return CompressionDEFLATE
}

// Level returns the compression level
func (d *DEFLATECompressor) Level() CompressionLevel {
	switch d.level {
	case flate.BestSpeed:
		return CompressionLevelFast
	case flate.BestCompression:
		return CompressionLevelBest
	case flate.NoCompression:
		return CompressionLevelNone
	default:
		return CompressionLevelDefault
	}
}

// LZWCompressor implements LZW compression
type LZWCompressor struct {
	order  lzw.Order
	litWidth int
}

// NewLZWCompressor creates a new LZW compressor
func NewLZWCompressor() *LZWCompressor {
	return &LZWCompressor{
		order:    lzw.MSB,
		litWidth: 8,
	}
}

// Compress compresses data using LZW
func (l *LZWCompressor) Compress(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer := lzw.NewWriter(&buf, l.order, l.litWidth)
	defer writer.Close()

	if _, err := writer.Write(data); err != nil {
		return nil, fmt.Errorf("failed to write data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close lzw writer: %w", err)
	}

	return buf.Bytes(), nil
}

// Decompress decompresses LZW data
func (l *LZWCompressor) Decompress(data []byte) ([]byte, error) {
	reader := lzw.NewReader(bytes.NewReader(data), l.order, l.litWidth)
	defer reader.Close()

	result, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read decompressed data: %w", err)
	}

	return result, nil
}

// Type returns the compression type
func (l *LZWCompressor) Type() CompressionType {
	return CompressionLZW
}

// Level returns the compression level (LZW doesn't have levels)
func (l *LZWCompressor) Level() CompressionLevel {
	return CompressionLevelDefault
}

// CompressionFactory creates compressors based on configuration
type CompressionFactory struct{}

// NewCompressionFactory creates a new compression factory
func NewCompressionFactory() *CompressionFactory {
	return &CompressionFactory{}
}

// CreateCompressor creates a compressor based on configuration
func (cf *CompressionFactory) CreateCompressor(config CompressorConfig) (Compressor, error) {
	switch config.Type {
	case CompressionGZIP:
		return NewGZIPCompressor(config.Level), nil
	case CompressionZLIB:
		return NewZLIBCompressor(config.Level), nil
	case CompressionDEFLATE:
		return NewDEFLATECompressor(config.Level), nil
	case CompressionLZW:
		return NewLZWCompressor(), nil
	default:
		return nil, fmt.Errorf("unsupported compression type: %d", config.Type)
	}
}

// CompressData compresses data using the specified algorithm and level
func CompressData(data []byte, compressionType CompressionType, level CompressionLevel) ([]byte, error) {
	factory := NewCompressionFactory()
	config := CompressorConfig{
		Type:  compressionType,
		Level: level,
	}

	compressor, err := factory.CreateCompressor(config)
	if err != nil {
		return nil, err
	}

	return compressor.Compress(data)
}

// DecompressData decompresses data using the specified algorithm
func DecompressData(data []byte, compressionType CompressionType) ([]byte, error) {
	factory := NewCompressionFactory()
	config := CompressorConfig{
		Type:  compressionType,
		Level: CompressionLevelDefault,
	}

	compressor, err := factory.CreateCompressor(config)
	if err != nil {
		return nil, err
	}

	return compressor.Decompress(data)
}

// CompressionStats holds compression statistics
type CompressionStats struct {
	OriginalSize   int64
	CompressedSize int64
	CompressionRatio float64
	Algorithm      CompressionType
	Level          CompressionLevel
}

// CalculateCompressionRatio calculates compression ratio
func CalculateCompressionRatio(originalSize, compressedSize int64) float64 {
	if originalSize == 0 {
		return 0
	}
	return float64(compressedSize) / float64(originalSize)
}

// GetCompressionStats returns compression statistics
func GetCompressionStats(original, compressed []byte, compressionType CompressionType, level CompressionLevel) *CompressionStats {
	return &CompressionStats{
		OriginalSize:     int64(len(original)),
		CompressedSize:   int64(len(compressed)),
		CompressionRatio: CalculateCompressionRatio(int64(len(original)), int64(len(compressed))),
		Algorithm:        compressionType,
		Level:            level,
	}
}

// BenchmarkCompression tests different compression algorithms on data
func BenchmarkCompression(data []byte) map[CompressionType]*CompressionStats {
	results := make(map[CompressionType]*CompressionStats)
	factory := NewCompressionFactory()

	algorithms := []CompressionType{
		CompressionGZIP,
		CompressionZLIB,
		CompressionDEFLATE,
		CompressionLZW,
	}

	levels := []CompressionLevel{
		CompressionLevelFast,
		CompressionLevelDefault,
		CompressionLevelBest,
	}

	for _, alg := range algorithms {
		for _, level := range levels {
			// Skip levels for LZW as it doesn't support them
			if alg == CompressionLZW && level != CompressionLevelDefault {
				continue
			}

			config := CompressorConfig{Type: alg, Level: level}
			compressor, err := factory.CreateCompressor(config)
			if err != nil {
				continue
			}

			compressed, err := compressor.Compress(data)
			if err != nil {
				continue
			}

			stats := GetCompressionStats(data, compressed, alg, level)
			key := CompressionType(int(alg)*10 + int(level))
			results[key] = stats
		}
	}

	return results
}

// OptimalCompressionConfig finds the best compression configuration for given data
func OptimalCompressionConfig(data []byte, prioritizeSpeed bool) (CompressorConfig, error) {
	benchmarks := BenchmarkCompression(data)
	
	var bestConfig CompressorConfig
	var bestRatio float64 = 1.0
	var bestAlg CompressionType
	var bestLevel CompressionLevel

	for key, stats := range benchmarks {
		// Extract algorithm and level from key
		alg := CompressionType(int(key) / 10)
		level := CompressionLevel(int(key) % 10)

		if prioritizeSpeed {
			// Prefer faster algorithms and levels
			if level == CompressionLevelFast && stats.CompressionRatio < bestRatio {
				bestRatio = stats.CompressionRatio
				bestAlg = alg
				bestLevel = level
			}
		} else {
			// Prefer better compression
			if stats.CompressionRatio < bestRatio {
				bestRatio = stats.CompressionRatio
				bestAlg = alg
				bestLevel = level
			}
		}
	}

	if bestRatio == 1.0 {
		return CompressorConfig{}, fmt.Errorf("no suitable compression algorithm found")
	}

	bestConfig.Type = bestAlg
	bestConfig.Level = bestLevel

	return bestConfig, nil
}

// StreamCompressor provides streaming compression
type StreamCompressor struct {
	compressor Compressor
	buffer     bytes.Buffer
}

// NewStreamCompressor creates a new stream compressor
func NewStreamCompressor(compressionType CompressionType, level CompressionLevel) (*StreamCompressor, error) {
	factory := NewCompressionFactory()
	config := CompressorConfig{Type: compressionType, Level: level}
	
	compressor, err := factory.CreateCompressor(config)
	if err != nil {
		return nil, err
	}

	return &StreamCompressor{
		compressor: compressor,
	}, nil
}

// Write writes data to the compression stream
func (sc *StreamCompressor) Write(data []byte) error {
	_, err := sc.buffer.Write(data)
	return err
}

// Flush compresses and returns all buffered data
func (sc *StreamCompressor) Flush() ([]byte, error) {
	data := sc.buffer.Bytes()
	sc.buffer.Reset()
	return sc.compressor.Compress(data)
}

// Reset resets the compression stream
func (sc *StreamCompressor) Reset() {
	sc.buffer.Reset()
}

// CompressionPool manages a pool of compressors for reuse
type CompressionPool struct {
	gzipPool    chan *GZIPCompressor
	zlibPool    chan *ZLIBCompressor
	deflatePool chan *DEFLATECompressor
	lzwPool     chan *LZWCompressor
}

// NewCompressionPool creates a new compression pool
func NewCompressionPool(poolSize int) *CompressionPool {
	return &CompressionPool{
		gzipPool:    make(chan *GZIPCompressor, poolSize),
		zlibPool:    make(chan *ZLIBCompressor, poolSize),
		deflatePool: make(chan *DEFLATECompressor, poolSize),
		lzwPool:     make(chan *LZWCompressor, poolSize),
	}
}

// GetCompressor gets a compressor from the pool
func (cp *CompressionPool) GetCompressor(compressionType CompressionType, level CompressionLevel) Compressor {
	switch compressionType {
	case CompressionGZIP:
		select {
		case compressor := <-cp.gzipPool:
			return compressor
		default:
			return NewGZIPCompressor(level)
		}
	case CompressionZLIB:
		select {
		case compressor := <-cp.zlibPool:
			return compressor
		default:
			return NewZLIBCompressor(level)
		}
	case CompressionDEFLATE:
		select {
		case compressor := <-cp.deflatePool:
			return compressor
		default:
			return NewDEFLATECompressor(level)
		}
	case CompressionLZW:
		select {
		case compressor := <-cp.lzwPool:
			return compressor
		default:
			return NewLZWCompressor()
		}
	default:
		return NewGZIPCompressor(level)
	}
}

// PutCompressor returns a compressor to the pool
func (cp *CompressionPool) PutCompressor(compressor Compressor) {
	switch c := compressor.(type) {
	case *GZIPCompressor:
		select {
		case cp.gzipPool <- c:
		default:
			// Pool is full, discard
		}
	case *ZLIBCompressor:
		select {
		case cp.zlibPool <- c:
		default:
			// Pool is full, discard
		}
	case *DEFLATECompressor:
		select {
		case cp.deflatePool <- c:
		default:
			// Pool is full, discard
		}
	case *LZWCompressor:
		select {
		case cp.lzwPool <- c:
		default:
			// Pool is full, discard
		}
	}
}