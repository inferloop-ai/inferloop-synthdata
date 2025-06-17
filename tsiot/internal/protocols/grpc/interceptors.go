package grpc

import (
	"context"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/pkg/errors"
)

// InterceptorSet contains unary and stream interceptors
type InterceptorSet struct {
	unary  []grpc.UnaryServerInterceptor
	stream []grpc.StreamServerInterceptor
}

// ClientInterceptorSet contains client-side interceptors
type ClientInterceptorSet struct {
	unary  []grpc.UnaryClientInterceptor
	stream []grpc.StreamClientInterceptor
}

// createInterceptors creates server-side interceptors
func (s *Server) createInterceptors() InterceptorSet {
	var unary []grpc.UnaryServerInterceptor
	var stream []grpc.StreamServerInterceptor
	
	// Add logging interceptor
	if s.config.EnableLogging {
		unary = append(unary, s.loggingUnaryInterceptor())
		stream = append(stream, s.loggingStreamInterceptor())
	}
	
	// Add metrics interceptor
	if s.config.EnableMetrics {
		unary = append(unary, s.metricsUnaryInterceptor())
		stream = append(stream, s.metricsStreamInterceptor())
	}
	
	// Add recovery interceptor
	unary = append(unary, s.recoveryUnaryInterceptor())
	stream = append(stream, s.recoveryStreamInterceptor())
	
	// Add validation interceptor
	unary = append(unary, s.validationUnaryInterceptor())
	stream = append(stream, s.validationStreamInterceptor())
	
	// Add timeout interceptor
	unary = append(unary, s.timeoutUnaryInterceptor())
	
	return InterceptorSet{
		unary:  unary,
		stream: stream,
	}
}

// createInterceptors creates client-side interceptors
func (c *Client) createInterceptors() ClientInterceptorSet {
	var unary []grpc.UnaryClientInterceptor
	var stream []grpc.StreamClientInterceptor
	
	// Add logging interceptor
	if c.config.EnableLogging {
		unary = append(unary, c.loggingUnaryClientInterceptor())
		stream = append(stream, c.loggingStreamClientInterceptor())
	}
	
	// Add metrics interceptor
	if c.config.EnableMetrics {
		unary = append(unary, c.metricsUnaryClientInterceptor())
		stream = append(stream, c.metricsStreamClientInterceptor())
	}
	
	// Add retry interceptor
	if c.config.RetryPolicy.Enabled {
		unary = append(unary, c.retryUnaryClientInterceptor())
	}
	
	// Add timeout interceptor
	if c.config.RequestTimeout > 0 {
		unary = append(unary, c.timeoutUnaryClientInterceptor())
	}
	
	return ClientInterceptorSet{
		unary:  unary,
		stream: stream,
	}
}

// Server-side interceptors

// loggingUnaryInterceptor logs unary requests
func (s *Server) loggingUnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		start := time.Now()
		
		// Extract metadata
		md, _ := metadata.FromIncomingContext(ctx)
		
		resp, err := handler(ctx, req)
		
		duration := time.Since(start)
		
		fields := logrus.Fields{
			"method":   info.FullMethod,
			"duration": duration,
		}
		
		if err != nil {
			fields["error"] = err
			s.logger.WithFields(fields).Error("gRPC unary request failed")
		} else {
			s.logger.WithFields(fields).Info("gRPC unary request completed")
		}
		
		// Log metadata if present
		if len(md) > 0 {
			s.logger.WithField("metadata", md).Debug("Request metadata")
		}
		
		return resp, err
	}
}

// loggingStreamInterceptor logs stream requests
func (s *Server) loggingStreamInterceptor() grpc.StreamServerInterceptor {
	return func(srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		start := time.Now()
		
		err := handler(srv, stream)
		
		duration := time.Since(start)
		
		fields := logrus.Fields{
			"method":       info.FullMethod,
			"duration":     duration,
			"client_stream": info.IsClientStream,
			"server_stream": info.IsServerStream,
		}
		
		if err != nil {
			fields["error"] = err
			s.logger.WithFields(fields).Error("gRPC stream request failed")
		} else {
			s.logger.WithFields(fields).Info("gRPC stream request completed")
		}
		
		return err
	}
}

// metricsUnaryInterceptor collects metrics for unary requests
func (s *Server) metricsUnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		start := time.Now()
		
		s.mu.Lock()
		s.metrics.RequestsTotal++
		s.metrics.RequestsInFlight++
		s.mu.Unlock()
		
		resp, err := handler(ctx, req)
		
		duration := time.Since(start)
		
		s.mu.Lock()
		s.metrics.RequestsInFlight--
		s.metrics.RequestDuration = duration
		s.metrics.LastRequestTime = start
		
		if err != nil {
			s.metrics.ErrorsTotal++
		}
		s.mu.Unlock()
		
		return resp, err
	}
}

// metricsStreamInterceptor collects metrics for stream requests
func (s *Server) metricsStreamInterceptor() grpc.StreamServerInterceptor {
	return func(srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		start := time.Now()
		
		s.mu.Lock()
		s.metrics.RequestsTotal++
		s.metrics.RequestsInFlight++
		s.mu.Unlock()
		
		err := handler(srv, stream)
		
		duration := time.Since(start)
		
		s.mu.Lock()
		s.metrics.RequestsInFlight--
		s.metrics.RequestDuration = duration
		s.metrics.LastRequestTime = start
		
		if err != nil {
			s.metrics.ErrorsTotal++
		}
		s.mu.Unlock()
		
		return err
	}
}

// recoveryUnaryInterceptor recovers from panics in unary handlers
func (s *Server) recoveryUnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
		defer func() {
			if r := recover(); r != nil {
				s.logger.WithFields(logrus.Fields{
					"method": info.FullMethod,
					"panic":  r,
				}).Error("Recovered from panic in gRPC handler")
				
				err = status.Errorf(codes.Internal, "Internal server error")
			}
		}()
		
		return handler(ctx, req)
	}
}

// recoveryStreamInterceptor recovers from panics in stream handlers
func (s *Server) recoveryStreamInterceptor() grpc.StreamServerInterceptor {
	return func(srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) (err error) {
		defer func() {
			if r := recover(); r != nil {
				s.logger.WithFields(logrus.Fields{
					"method": info.FullMethod,
					"panic":  r,
				}).Error("Recovered from panic in gRPC stream handler")
				
				err = status.Errorf(codes.Internal, "Internal server error")
			}
		}()
		
		return handler(srv, stream)
	}
}

// validationUnaryInterceptor validates requests
func (s *Server) validationUnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// Basic request validation
		if req == nil {
			return nil, status.Errorf(codes.InvalidArgument, "Request cannot be nil")
		}
		
		// Add more validation logic as needed
		// For example, validate required fields, formats, etc.
		
		return handler(ctx, req)
	}
}

// validationStreamInterceptor validates stream requests
func (s *Server) validationStreamInterceptor() grpc.StreamServerInterceptor {
	return func(srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		// Add stream validation logic as needed
		return handler(srv, stream)
	}
}

// timeoutUnaryInterceptor adds timeout to unary requests
func (s *Server) timeoutUnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// Set default timeout if not already set
		if _, hasDeadline := ctx.Deadline(); !hasDeadline {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, 30*time.Second)
			defer cancel()
		}
		
		return handler(ctx, req)
	}
}

// Client-side interceptors

// loggingUnaryClientInterceptor logs client requests
func (c *Client) loggingUnaryClientInterceptor() grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		start := time.Now()
		
		err := invoker(ctx, method, req, reply, cc, opts...)
		
		duration := time.Since(start)
		
		fields := logrus.Fields{
			"method":   method,
			"duration": duration,
		}
		
		if err != nil {
			fields["error"] = err
			c.logger.WithFields(fields).Error("gRPC client request failed")
		} else {
			c.logger.WithFields(fields).Debug("gRPC client request completed")
		}
		
		return err
	}
}

// loggingStreamClientInterceptor logs client stream requests
func (c *Client) loggingStreamClientInterceptor() grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		start := time.Now()
		
		stream, err := streamer(ctx, desc, cc, method, opts...)
		
		duration := time.Since(start)
		
		fields := logrus.Fields{
			"method":        method,
			"duration":      duration,
			"client_stream": desc.ClientStreams,
			"server_stream": desc.ServerStreams,
		}
		
		if err != nil {
			fields["error"] = err
			c.logger.WithFields(fields).Error("gRPC client stream failed")
		} else {
			c.logger.WithFields(fields).Debug("gRPC client stream created")
		}
		
		return stream, err
	}
}

// metricsUnaryClientInterceptor collects client metrics
func (c *Client) metricsUnaryClientInterceptor() grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		start := time.Now()
		
		c.mu.Lock()
		c.metrics.RequestsTotal++
		c.metrics.RequestsInFlight++
		c.mu.Unlock()
		
		err := invoker(ctx, method, req, reply, cc, opts...)
		
		duration := time.Since(start)
		
		c.mu.Lock()
		c.metrics.RequestsInFlight--
		c.metrics.RequestDuration = duration
		c.metrics.LastRequestTime = start
		
		if err != nil {
			c.metrics.RequestsFailed++
			c.metrics.LastError = err
			c.metrics.LastErrorTime = time.Now()
		} else {
			c.metrics.RequestsSuccessful++
		}
		c.mu.Unlock()
		
		return err
	}
}

// metricsStreamClientInterceptor collects client stream metrics
func (c *Client) metricsStreamClientInterceptor() grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		start := time.Now()
		
		c.mu.Lock()
		c.metrics.RequestsTotal++
		c.metrics.RequestsInFlight++
		c.mu.Unlock()
		
		stream, err := streamer(ctx, desc, cc, method, opts...)
		
		duration := time.Since(start)
		
		c.mu.Lock()
		c.metrics.RequestsInFlight--
		c.metrics.RequestDuration = duration
		c.metrics.LastRequestTime = start
		
		if err != nil {
			c.metrics.RequestsFailed++
			c.metrics.LastError = err
			c.metrics.LastErrorTime = time.Now()
		} else {
			c.metrics.RequestsSuccessful++
		}
		c.mu.Unlock()
		
		return stream, err
	}
}

// retryUnaryClientInterceptor implements retry logic
func (c *Client) retryUnaryClientInterceptor() grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		var lastErr error
		
		for attempt := 0; attempt < c.config.RetryPolicy.MaxAttempts; attempt++ {
			if attempt > 0 {
				// Calculate backoff
				backoff := time.Duration(float64(c.config.RetryPolicy.InitialBackoff) * 
					pow(c.config.RetryPolicy.BackoffMultiplier, float64(attempt-1)))
				
				if backoff > c.config.RetryPolicy.MaxBackoff {
					backoff = c.config.RetryPolicy.MaxBackoff
				}
				
				// Wait before retry
				select {
				case <-time.After(backoff):
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			
			err := invoker(ctx, method, req, reply, cc, opts...)
			if err == nil {
				return nil
			}
			
			lastErr = err
			
			// Check if error is retryable
			if !c.isRetryableError(err) {
				break
			}
		}
		
		return lastErr
	}
}

// timeoutUnaryClientInterceptor adds timeout to client requests
func (c *Client) timeoutUnaryClientInterceptor() grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		// Add timeout if not already set
		if _, hasDeadline := ctx.Deadline(); !hasDeadline {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, c.config.RequestTimeout)
			defer cancel()
		}
		
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}

// isRetryableError checks if an error is retryable
func (c *Client) isRetryableError(err error) bool {
	st, ok := status.FromError(err)
	if !ok {
		return false
	}
	
	code := st.Code()
	for _, retryableCode := range c.config.RetryPolicy.RetryableStatusCodes {
		if code.String() == retryableCode {
			return true
		}
	}
	
	return false
}

// pow calculates power (simple implementation)
func pow(base, exp float64) float64 {
	if exp == 0 {
		return 1
	}
	result := base
	for i := 1; i < int(exp); i++ {
		result *= base
	}
	return result
}