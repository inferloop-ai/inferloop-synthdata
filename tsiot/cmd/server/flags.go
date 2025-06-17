package main

import (
	"flag"
	"fmt"
	"os"
)

type Config struct {
	Port           int
	Host           string
	ConfigFile     string
	LogLevel       string
	LogFormat      string
	MetricsPort    int
	EnableMCP      bool
	MCPTransport   string
	StorageBackend string
	TLSCert        string
	TLSKey         string
	EnableTLS      bool
	Version        bool
}

func ParseFlags() *Config {
	config := &Config{}

	flag.IntVar(&config.Port, "port", 8080, "Server port")
	flag.StringVar(&config.Host, "host", "0.0.0.0", "Server host")
	flag.StringVar(&config.ConfigFile, "config", "", "Path to configuration file")
	flag.StringVar(&config.LogLevel, "log-level", "info", "Log level (debug, info, warn, error)")
	flag.StringVar(&config.LogFormat, "log-format", "json", "Log format (json, text)")
	flag.IntVar(&config.MetricsPort, "metrics-port", 9090, "Prometheus metrics port")
	flag.BoolVar(&config.EnableMCP, "enable-mcp", true, "Enable Model Context Protocol")
	flag.StringVar(&config.MCPTransport, "mcp-transport", "stdio", "MCP transport (stdio, websocket)")
	flag.StringVar(&config.StorageBackend, "storage", "influxdb", "Storage backend (influxdb, timescaledb)")
	flag.StringVar(&config.TLSCert, "tls-cert", "", "Path to TLS certificate")
	flag.StringVar(&config.TLSKey, "tls-key", "", "Path to TLS key")
	flag.BoolVar(&config.EnableTLS, "enable-tls", false, "Enable TLS")
	flag.BoolVar(&config.Version, "version", false, "Show version information")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nTime Series Synthetic Data MCP Server\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
	}

	flag.Parse()

	if config.Version {
		info := GetBuildInfo()
		fmt.Printf("Version: %s\n", info.Version)
		fmt.Printf("Git Commit: %s\n", info.GitCommit)
		fmt.Printf("Build Date: %s\n", info.BuildDate)
		fmt.Printf("Go Version: %s\n", info.GoVersion)
		fmt.Printf("Platform: %s\n", info.Platform)
		os.Exit(0)
	}

	return config
}