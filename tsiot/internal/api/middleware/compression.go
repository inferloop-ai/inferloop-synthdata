package middleware

import (
	"bufio"
	"compress/flate"
	"compress/gzip"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/sirupsen/logrus"
)

// CompressionConfig contains compression middleware configuration
type CompressionConfig struct {
	Enabled           bool     `json:"enabled" yaml:"enabled"`
	Level             int      `json:"level" yaml:"level"` // 1-9, where 9 is best compression
	MinSize           int      `json:"min_size" yaml:"min_size"` // Minimum response size to compress
	Types             []string `json:"types" yaml:"types"` // MIME types to compress
	ExcludePaths      []string `json:"exclude_paths" yaml:"exclude_paths"`
	ExcludeExtensions []string `json:"exclude_extensions" yaml:"exclude_extensions"`
	Vary              bool     `json:"vary" yaml:"vary"` // Add Vary: Accept-Encoding header
	BrotliEnabled     bool     `json:"brotli_enabled" yaml:"brotli_enabled"`
	DeflateEnabled    bool     `json:"deflate_enabled" yaml:"deflate_enabled"`
}

// CompressionMiddleware provides HTTP response compression
type CompressionMiddleware struct {
	config      *CompressionConfig
	logger      *logrus.Logger
	gzipPool    sync.Pool
	deflatePool sync.Pool
}

// compressedWriter wraps http.ResponseWriter to provide compression
type compressedWriter struct {
	http.ResponseWriter
	writer     io.Writer
	encoding   string
	minSize    int
	buffer     []byte
	size       int
	headerSent bool
	closed     bool
}

// NewCompressionMiddleware creates a new compression middleware
func NewCompressionMiddleware(config *CompressionConfig, logger *logrus.Logger) *CompressionMiddleware {
	if logger == nil {
		logger = logrus.New()
	}

	if config == nil {
		config = &CompressionConfig{
			Enabled:        true,
			Level:          6, // Default compression level
			MinSize:        1024, // Only compress responses >= 1KB
			Vary:           true,
			BrotliEnabled:  false, // Brotli requires additional library
			DeflateEnabled: true,
			Types: []string{
				"text/html",
				"text/css",
				"text/javascript",
				"text/plain",
				"text/xml",
				"application/json",
				"application/javascript",
				"application/xml",
				"application/rss+xml",
				"application/atom+xml",
				"image/svg+xml",
			},
			ExcludePaths: []string{
				"/health",
				"/metrics",
			},
			ExcludeExtensions: []string{
				".jpg", ".jpeg", ".png", ".gif", ".ico",
				".zip", ".gz", ".bz2", ".7z", ".rar",
				".mp4", ".avi", ".mov", ".mp3", ".wav",
				".pdf", ".doc", ".docx", ".xls", ".xlsx",
			},
		}
	}

	cm := &CompressionMiddleware{
		config: config,
		logger: logger,
	}

	// Initialize pools for better performance
	cm.gzipPool = sync.Pool{
		New: func() interface{} {
			w, _ := gzip.NewWriterLevel(io.Discard, config.Level)
			return w
		},
	}

	cm.deflatePool = sync.Pool{
		New: func() interface{} {
			w, _ := flate.NewWriter(io.Discard, config.Level)
			return w
		},
	}

	return cm
}

// Middleware returns the HTTP middleware function
func (cm *CompressionMiddleware) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip compression if disabled
			if !cm.config.Enabled {
				next.ServeHTTP(w, r)
				return
			}

			// Check if request should be excluded
			if cm.shouldExclude(r) {
				next.ServeHTTP(w, r)
				return
			}

			// Check if client accepts compression
			encoding := cm.selectEncoding(r.Header.Get("Accept-Encoding"))
			if encoding == "" {
				next.ServeHTTP(w, r)
				return
			}

			// Add Vary header if configured
			if cm.config.Vary {
				w.Header().Add("Vary", "Accept-Encoding")
			}

			// Create compressed writer
			cw := &compressedWriter{
				ResponseWriter: w,
				encoding:       encoding,
				minSize:        cm.config.MinSize,
				buffer:         make([]byte, 0, cm.config.MinSize),
			}

			// Setup compressor based on encoding
			if err := cm.setupCompressor(cw); err != nil {
				cm.logger.WithError(err).Error("Failed to setup compressor")
				next.ServeHTTP(w, r)
				return
			}

			// Ensure we close the compressor
			defer cm.closeCompressor(cw)

			// Process request
			next.ServeHTTP(cw, r)

			// Flush any remaining data
			cw.Close()
		})
	}
}

// shouldExclude checks if the request should be excluded from compression
func (cm *CompressionMiddleware) shouldExclude(r *http.Request) bool {
	// Check excluded paths
	for _, excludePath := range cm.config.ExcludePaths {
		if strings.HasPrefix(r.URL.Path, excludePath) {
			return true
		}
	}

	// Check excluded extensions
	path := strings.ToLower(r.URL.Path)
	for _, ext := range cm.config.ExcludeExtensions {
		if strings.HasSuffix(path, strings.ToLower(ext)) {
			return true
		}
	}

	return false
}

// selectEncoding selects the best compression encoding supported by the client
func (cm *CompressionMiddleware) selectEncoding(acceptEncoding string) string {
	if acceptEncoding == "" {
		return ""
	}

	acceptEncoding = strings.ToLower(acceptEncoding)

	// Parse Accept-Encoding header and select best encoding
	encodings := strings.Split(acceptEncoding, ",")
	
	// Priority order: brotli (if enabled), gzip, deflate
	for _, encoding := range encodings {
		encoding = strings.TrimSpace(encoding)
		
		// Remove quality values (e.g., "gzip;q=0.8" -> "gzip")
		if idx := strings.Index(encoding, ";"); idx != -1 {
			encoding = encoding[:idx]
		}

		switch encoding {
		case "br":
			if cm.config.BrotliEnabled {
				return "br"
			}
		case "gzip":
			return "gzip"
		case "deflate":
			if cm.config.DeflateEnabled {
				return "deflate"
			}
		}
	}

	return ""
}

// setupCompressor sets up the appropriate compressor for the writer
func (cm *CompressionMiddleware) setupCompressor(cw *compressedWriter) error {
	switch cw.encoding {
	case "gzip":
		if gzWriter := cm.gzipPool.Get().(*gzip.Writer); gzWriter != nil {
			gzWriter.Reset(cw.ResponseWriter)
			cw.writer = gzWriter
			return nil
		}
		// Fallback if pool is empty
		gzWriter, err := gzip.NewWriterLevel(cw.ResponseWriter, cm.config.Level)
		if err != nil {
			return err
		}
		cw.writer = gzWriter
		return nil

	case "deflate":
		if deflateWriter := cm.deflatePool.Get().(*flate.Writer); deflateWriter != nil {
			deflateWriter.Reset(cw.ResponseWriter)
			cw.writer = deflateWriter
			return nil
		}
		// Fallback if pool is empty
		deflateWriter, err := flate.NewWriter(cw.ResponseWriter, cm.config.Level)
		if err != nil {
			return err
		}
		cw.writer = deflateWriter
		return nil

	default:
		cw.writer = cw.ResponseWriter
		return nil
	}
}

// closeCompressor properly closes and returns compressor to pool
func (cm *CompressionMiddleware) closeCompressor(cw *compressedWriter) {
	if cw.writer == nil || cw.closed {
		return
	}

	switch cw.encoding {
	case "gzip":
		if gzWriter, ok := cw.writer.(*gzip.Writer); ok {
			gzWriter.Close()
			cm.gzipPool.Put(gzWriter)
		}
	case "deflate":
		if deflateWriter, ok := cw.writer.(*flate.Writer); ok {
			deflateWriter.Close()
			cm.deflatePool.Put(deflateWriter)
		}
	}

	cw.closed = true
}

// shouldCompress checks if the response should be compressed based on content type
func (cm *CompressionMiddleware) shouldCompress(contentType string) bool {
	if contentType == "" {
		return false
	}

	// Extract main content type (ignore charset, etc.)
	if idx := strings.Index(contentType, ";"); idx != -1 {
		contentType = contentType[:idx]
	}
	contentType = strings.TrimSpace(strings.ToLower(contentType))

	// Check if content type is in the compression list
	for _, compressibleType := range cm.config.Types {
		if contentType == strings.ToLower(compressibleType) {
			return true
		}
		// Check for wildcard matches
		if strings.HasSuffix(compressibleType, "/*") {
			prefix := compressibleType[:len(compressibleType)-2]
			if strings.HasPrefix(contentType, prefix) {
				return true
			}
		}
	}

	return false
}

// Header returns the response header
func (cw *compressedWriter) Header() http.Header {
	return cw.ResponseWriter.Header()
}

// WriteHeader sets the response status code and headers
func (cw *compressedWriter) WriteHeader(code int) {
	if cw.headerSent {
		return
	}

	cw.headerSent = true

	// Check if we should compress based on content type
	contentType := cw.Header().Get("Content-Type")
	if contentType == "" {
		// Try to detect content type from buffered content
		if len(cw.buffer) > 0 {
			contentType = http.DetectContentType(cw.buffer)
			cw.Header().Set("Content-Type", contentType)
		}
	}

	// Only compress if content type is compressible
	if cw.encoding != "" && contentType != "" {
		middleware := cw.ResponseWriter.(*http.ResponseWriter)
		if cm, ok := (*middleware).(*CompressionMiddleware); ok {
			if !cm.shouldCompress(contentType) {
				cw.encoding = ""
				cw.writer = cw.ResponseWriter
			}
		}
	}

	// Set compression headers if we're compressing
	if cw.encoding != "" && cw.writer != cw.ResponseWriter {
		cw.Header().Set("Content-Encoding", cw.encoding)
		cw.Header().Del("Content-Length") // Remove content-length as it will change
	}

	cw.ResponseWriter.WriteHeader(code)

	// If we have buffered data and we're not compressing, flush it directly
	if cw.encoding == "" && len(cw.buffer) > 0 {
		cw.ResponseWriter.Write(cw.buffer)
		cw.buffer = nil
	}
}

// Write writes data to the response
func (cw *compressedWriter) Write(data []byte) (int, error) {
	if cw.closed {
		return 0, io.ErrClosedPipe
	}

	// If headers haven't been sent, buffer the data to check size
	if !cw.headerSent {
		cw.buffer = append(cw.buffer, data...)
		
		// If buffer exceeds minimum size or we have a complete response, send headers
		if len(cw.buffer) >= cw.minSize {
			cw.WriteHeader(http.StatusOK)
		} else {
			return len(data), nil // Buffer the data
		}
	}

	// Write buffered data first if any
	if len(cw.buffer) > 0 {
		written, err := cw.writer.Write(cw.buffer)
		cw.size += written
		if err != nil {
			return 0, err
		}
		cw.buffer = nil
	}

	// Write current data
	written, err := cw.writer.Write(data)
	cw.size += written
	return len(data), err // Return original length to maintain interface contract
}

// Close closes the compressed writer
func (cw *compressedWriter) Close() error {
	if cw.closed {
		return nil
	}

	// Send headers if not sent yet
	if !cw.headerSent {
		// If we have buffered data but it's less than minimum size, don't compress
		if len(cw.buffer) > 0 && len(cw.buffer) < cw.minSize {
			cw.encoding = ""
			cw.writer = cw.ResponseWriter
		}
		cw.WriteHeader(http.StatusOK)
	}

	// Write any remaining buffered data
	if len(cw.buffer) > 0 {
		cw.writer.Write(cw.buffer)
		cw.buffer = nil
	}

	// Close compressor if it's not the response writer
	if cw.writer != cw.ResponseWriter {
		switch w := cw.writer.(type) {
		case *gzip.Writer:
			return w.Close()
		case *flate.Writer:
			return w.Close()
		}
	}

	cw.closed = true
	return nil
}

// Hijack implements http.Hijacker interface
func (cw *compressedWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	hijacker, ok := cw.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, http.ErrNotSupported
	}
	return hijacker.Hijack()
}

// Flush implements http.Flusher interface
func (cw *compressedWriter) Flush() {
	// Ensure headers are sent
	if !cw.headerSent {
		cw.WriteHeader(http.StatusOK)
	}

	// Flush compressor if possible
	switch w := cw.writer.(type) {
	case *gzip.Writer:
		w.Flush()
	case *flate.Writer:
		w.Flush()
	}

	// Flush underlying response writer
	if flusher, ok := cw.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// GetCompressionStats returns compression middleware statistics
func (cm *CompressionMiddleware) GetCompressionStats() map[string]interface{} {
	return map[string]interface{}{
		"config": cm.config,
		"pools": map[string]interface{}{
			"gzip_pool_size":    "dynamic", // Can't easily get pool size
			"deflate_pool_size": "dynamic",
		},
	}
}