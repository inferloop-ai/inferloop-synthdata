package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
)

func main() {
	config := ParseFlags()
	
	logger := setupLogger(config.LogLevel, config.LogFormat)
	
	logger.WithFields(logrus.Fields{
		"version":   Version,
		"commit":    GitCommit,
		"buildDate": BuildDate,
	}).Info("Starting Time Series Synthetic Data MCP Server")

	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Initialize router
	router := mux.NewRouter()
	
	// Health check endpoint
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"ok","version":"%s"}`, Version)
	}).Methods("GET")

	// Version endpoint
	router.HandleFunc("/version", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		info := GetBuildInfo()
		fmt.Fprintf(w, `{"version":"%s","commit":"%s","buildDate":"%s","goVersion":"%s","platform":"%s"}`,
			info.Version, info.GitCommit, info.BuildDate, info.GoVersion, info.Platform)
	}).Methods("GET")

	// API routes would be registered here
	// router.PathPrefix("/api/v1/").Handler(api.NewRouter())

	// Start metrics server
	go func() {
		metricsAddr := fmt.Sprintf(":%d", config.MetricsPort)
		logger.WithField("address", metricsAddr).Info("Starting metrics server")
		
		metricsMux := http.NewServeMux()
		metricsMux.Handle("/metrics", promhttp.Handler())
		
		if err := http.ListenAndServe(metricsAddr, metricsMux); err != nil {
			logger.WithError(err).Error("Metrics server failed")
		}
	}()

	// Configure main server
	serverAddr := fmt.Sprintf("%s:%d", config.Host, config.Port)
	srv := &http.Server{
		Addr:         serverAddr,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server
	go func() {
		logger.WithField("address", serverAddr).Info("Starting HTTP server")
		
		var err error
		if config.EnableTLS && config.TLSCert != "" && config.TLSKey != "" {
			err = srv.ListenAndServeTLS(config.TLSCert, config.TLSKey)
		} else {
			err = srv.ListenAndServe()
		}
		
		if err != nil && err != http.ErrServerClosed {
			logger.WithError(err).Fatal("Server failed to start")
		}
	}()

	// Start MCP server if enabled
	if config.EnableMCP {
		logger.WithField("transport", config.MCPTransport).Info("Starting MCP server")
		// TODO: Initialize and start MCP server
		// mcpServer := mcp.NewServer(config.MCPTransport)
		// go mcpServer.Start(ctx)
	}

	// Wait for shutdown signal
	<-sigChan
	logger.Info("Shutdown signal received")

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.WithError(err).Error("Server shutdown failed")
	}

	logger.Info("Server stopped")
}

func setupLogger(level, format string) *logrus.Logger {
	logger := logrus.New()

	// Set log level
	logLevel, err := logrus.ParseLevel(level)
	if err != nil {
		logLevel = logrus.InfoLevel
	}
	logger.SetLevel(logLevel)

	// Set log format
	if format == "json" {
		logger.SetFormatter(&logrus.JSONFormatter{})
	} else {
		logger.SetFormatter(&logrus.TextFormatter{
			FullTimestamp: true,
		})
	}

	return logger
}