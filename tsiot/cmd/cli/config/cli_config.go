package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/viper"
)

type CLIConfig struct {
	ServerURL      string                 `mapstructure:"server_url"`
	DefaultOutput  string                 `mapstructure:"default_output"`
	DefaultFormat  string                 `mapstructure:"default_format"`
	Generators     map[string]GenConfig   `mapstructure:"generators"`
	Storage        map[string]StorageConfig `mapstructure:"storage"`
	Preferences    Preferences            `mapstructure:"preferences"`
}

type GenConfig struct {
	Enabled    bool                   `mapstructure:"enabled"`
	Parameters map[string]interface{} `mapstructure:"parameters"`
}

type StorageConfig struct {
	Type        string `mapstructure:"type"`
	URL         string `mapstructure:"url"`
	Username    string `mapstructure:"username"`
	Password    string `mapstructure:"password"`
	Database    string `mapstructure:"database"`
	Bucket      string `mapstructure:"bucket"`
	Region      string `mapstructure:"region"`
}

type Preferences struct {
	ColorOutput  bool   `mapstructure:"color_output"`
	TableFormat  string `mapstructure:"table_format"`
	TimeZone     string `mapstructure:"timezone"`
	ProgressBars bool   `mapstructure:"progress_bars"`
}

func LoadConfig(cfgFile string) (*CLIConfig, error) {
	config := &CLIConfig{
		ServerURL:     "http://localhost:8080",
		DefaultOutput: "-",
		DefaultFormat: "csv",
		Preferences: Preferences{
			ColorOutput:  true,
			TableFormat:  "simple",
			TimeZone:     "UTC",
			ProgressBars: true,
		},
	}

	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, err
		}

		configPath := filepath.Join(home, ".tsiot")
		viper.AddConfigPath(configPath)
		viper.SetConfigName("config")
		viper.SetConfigType("yaml")
	}

	viper.SetEnvPrefix("TSIOT")
	viper.AutomaticEnv()

	// Set defaults
	viper.SetDefault("server_url", config.ServerURL)
	viper.SetDefault("default_output", config.DefaultOutput)
	viper.SetDefault("default_format", config.DefaultFormat)
	viper.SetDefault("preferences.color_output", config.Preferences.ColorOutput)
	viper.SetDefault("preferences.table_format", config.Preferences.TableFormat)
	viper.SetDefault("preferences.timezone", config.Preferences.TimeZone)
	viper.SetDefault("preferences.progress_bars", config.Preferences.ProgressBars)

	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("error reading config file: %w", err)
		}
	}

	if err := viper.Unmarshal(config); err != nil {
		return nil, fmt.Errorf("error unmarshaling config: %w", err)
	}

	return config, nil
}

func SaveConfig(config *CLIConfig, cfgFile string) error {
	if cfgFile == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		
		configDir := filepath.Join(home, ".tsiot")
		if err := os.MkdirAll(configDir, 0755); err != nil {
			return fmt.Errorf("error creating config directory: %w", err)
		}
		
		cfgFile = filepath.Join(configDir, "config.yaml")
	}

	viper.Set("server_url", config.ServerURL)
	viper.Set("default_output", config.DefaultOutput)
	viper.Set("default_format", config.DefaultFormat)
	viper.Set("generators", config.Generators)
	viper.Set("storage", config.Storage)
	viper.Set("preferences", config.Preferences)

	return viper.WriteConfigAs(cfgFile)
}

func GetDefaultConfigPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".tsiot", "config.yaml")
}