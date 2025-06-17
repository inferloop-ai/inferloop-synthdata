package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sirupsen/logrus"
)

type WorkerConfig struct {
	WorkerID       string
	ServerURL      string
	Concurrency    int
	PollInterval   time.Duration
	MaxRetries     int
	StorageBackend string
	LogLevel       string
	LogFormat      string
}

var logger *logrus.Logger

func main() {
	config := parseFlags()
	
	logger = setupLogger(config.LogLevel, config.LogFormat)
	
	logger.WithFields(logrus.Fields{
		"workerID":    config.WorkerID,
		"concurrency": config.Concurrency,
		"serverURL":   config.ServerURL,
	}).Info("Starting Time Series Synthetic Data Worker")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Initialize scheduler
	scheduler := NewScheduler(config, logger)
	
	// Initialize job processor
	processor := NewJobProcessor(config, logger)

	// Start worker components
	go scheduler.Start(ctx)
	go processor.Start(ctx)

	// Monitor worker health
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				logger.WithFields(logrus.Fields{
					"activeJobs":    processor.ActiveJobs(),
					"completedJobs": processor.CompletedJobs(),
					"failedJobs":    processor.FailedJobs(),
				}).Debug("Worker health check")
			}
		}
	}()

	// Wait for shutdown signal
	<-sigChan
	logger.Info("Shutdown signal received")

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := gracefulShutdown(shutdownCtx, scheduler, processor); err != nil {
		logger.WithError(err).Error("Worker shutdown failed")
		os.Exit(1)
	}

	logger.Info("Worker stopped successfully")
}

func parseFlags() *WorkerConfig {
	config := &WorkerConfig{}

	flag.StringVar(&config.WorkerID, "worker-id", generateWorkerID(), "Unique worker ID")
	flag.StringVar(&config.ServerURL, "server-url", "http://localhost:8080", "Server URL")
	flag.IntVar(&config.Concurrency, "concurrency", 4, "Number of concurrent jobs")
	flag.DurationVar(&config.PollInterval, "poll-interval", 5*time.Second, "Job polling interval")
	flag.IntVar(&config.MaxRetries, "max-retries", 3, "Maximum job retries")
	flag.StringVar(&config.StorageBackend, "storage", "influxdb", "Storage backend")
	flag.StringVar(&config.LogLevel, "log-level", "info", "Log level")
	flag.StringVar(&config.LogFormat, "log-format", "json", "Log format")

	flag.Parse()

	return config
}

func setupLogger(level, format string) *logrus.Logger {
	logger := logrus.New()

	logLevel, err := logrus.ParseLevel(level)
	if err != nil {
		logLevel = logrus.InfoLevel
	}
	logger.SetLevel(logLevel)

	if format == "json" {
		logger.SetFormatter(&logrus.JSONFormatter{})
	} else {
		logger.SetFormatter(&logrus.TextFormatter{
			FullTimestamp: true,
		})
	}

	return logger
}

func generateWorkerID() string {
	hostname, _ := os.Hostname()
	return fmt.Sprintf("%s-%d", hostname, os.Getpid())
}

func gracefulShutdown(ctx context.Context, scheduler *Scheduler, processor *JobProcessor) error {
	logger.Info("Starting graceful shutdown")

	// Stop accepting new jobs
	scheduler.Stop()

	// Wait for active jobs to complete
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("shutdown timeout exceeded")
		case <-ticker.C:
			if processor.ActiveJobs() == 0 {
				logger.Info("All jobs completed")
				return nil
			}
			logger.WithField("activeJobs", processor.ActiveJobs()).Info("Waiting for jobs to complete")
		}
	}
}