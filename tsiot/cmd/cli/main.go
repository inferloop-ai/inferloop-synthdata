package main

import (
	"fmt"
	"os"

	"github.com/inferloop/tsiot/cmd/cli/commands"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	cfgFile string
	verbose bool
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "tsiot-cli",
		Short: "Time Series IoT Synthetic Data CLI",
		Long: `A command-line interface for generating, validating, and managing
synthetic time series data for IoT applications.`,
		Version: "0.1.0",
	}

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.tsiot.yaml)")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")

	// Initialize Viper
	cobra.OnInitialize(initConfig)

	// Add commands
	rootCmd.AddCommand(commands.NewGenerateCmd())
	rootCmd.AddCommand(commands.NewValidateCmd())
	rootCmd.AddCommand(commands.NewAnalyzeCmd())
	rootCmd.AddCommand(commands.NewMigrateCmd())

	// Execute
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		home, err := os.UserHomeDir()
		cobra.CheckErr(err)

		viper.AddConfigPath(home)
		viper.SetConfigType("yaml")
		viper.SetConfigName(".tsiot")
	}

	viper.AutomaticEnv()
	viper.SetEnvPrefix("TSIOT")

	if err := viper.ReadInConfig(); err == nil && verbose {
		fmt.Fprintln(os.Stderr, "Using config file:", viper.ConfigFileUsed())
	}
}