/**
 * TSIOT JavaScript/TypeScript SDK
 * 
 * A comprehensive client library for interacting with the Time Series IoT 
 * Synthetic Data (TSIOT) service. This SDK provides functionality for generating,
 * validating, and analyzing time series data.
 * 
 * @version 1.0.0
 * @author TSIOT Development Team
 * @license MIT
 */

// Core exports
const { TSIOTClient } = require('./client');
const { TimeSeries, DataPoint, TimeSeriesMetadata } = require('./timeseries');
const { TimeSeriesGenerator } = require('./generators');

// Constants and enums
const DataFormat = {
  JSON: 'json',
  CSV: 'csv',
  PARQUET: 'parquet',
  AVRO: 'avro',
  ARROW: 'arrow'
};

const Frequency = {
  NANOSECOND: 'ns',
  MICROSECOND: 'us',
  MILLISECOND: 'ms',
  SECOND: 's',
  MINUTE: 'min',
  HOUR: 'h',
  DAY: 'd',
  WEEK: 'w',
  MONTH: 'M',
  QUARTER: 'Q',
  YEAR: 'Y'
};

const ValidationLevel = {
  BASIC: 'basic',
  STANDARD: 'standard',
  COMPREHENSIVE: 'comprehensive'
};

// Error classes
class TSIOTError extends Error {
  constructor(message, errorCode = 'UNKNOWN_ERROR', details = {}) {
    super(message);
    this.name = 'TSIOTError';
    this.errorCode = errorCode;
    this.details = details;
    this.timestamp = new Date().toISOString();
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      errorCode: this.errorCode,
      details: this.details,
      timestamp: this.timestamp,
      stack: this.stack
    };
  }
}

class ValidationError extends TSIOTError {
  constructor(message, field = null, value = null, details = {}) {
    if (field) details.field = field;
    if (value !== null) details.value = value;
    super(message, 'VALIDATION_ERROR', details);
    this.name = 'ValidationError';
  }
}

class GenerationError extends TSIOTError {
  constructor(message, generatorType = null, details = {}) {
    if (generatorType) details.generatorType = generatorType;
    super(message, 'GENERATION_ERROR', details);
    this.name = 'GenerationError';
  }
}

class NetworkError extends TSIOTError {
  constructor(message, statusCode = null, responseBody = null, details = {}) {
    if (statusCode) details.statusCode = statusCode;
    if (responseBody) details.responseBody = responseBody;
    super(message, 'NETWORK_ERROR', details);
    this.name = 'NetworkError';
  }
}

class AuthenticationError extends TSIOTError {
  constructor(message = 'Authentication failed', details = {}) {
    super(message, 'AUTHENTICATION_ERROR', details);
    this.name = 'AuthenticationError';
  }
}

class RateLimitError extends TSIOTError {
  constructor(message, retryAfter = null, details = {}) {
    if (retryAfter) details.retryAfter = retryAfter;
    super(message, 'RATE_LIMIT_ERROR', details);
    this.name = 'RateLimitError';
  }
}

class TimeoutError extends TSIOTError {
  constructor(message, timeoutSeconds = null, details = {}) {
    if (timeoutSeconds) details.timeoutSeconds = timeoutSeconds;
    super(message, 'TIMEOUT_ERROR', details);
    this.name = 'TimeoutError';
  }
}

// Utility functions
const utils = {
  /**
   * Validate that a value is a positive number
   * @param {number} value - Value to validate
   * @param {string} name - Parameter name for error messages
   * @returns {number} Validated value
   */
  validatePositiveNumber(value, name) {
    if (typeof value !== 'number' || isNaN(value)) {
      throw new ValidationError(`${name} must be a number`, name, value);
    }
    if (value <= 0) {
      throw new ValidationError(`${name} must be positive`, name, value);
    }
    return value;
  },

  /**
   * Validate that a value is a non-negative number
   * @param {number} value - Value to validate
   * @param {string} name - Parameter name for error messages
   * @returns {number} Validated value
   */
  validateNonNegativeNumber(value, name) {
    if (typeof value !== 'number' || isNaN(value)) {
      throw new ValidationError(`${name} must be a number`, name, value);
    }
    if (value < 0) {
      throw new ValidationError(`${name} must be non-negative`, name, value);
    }
    return value;
  },

  /**
   * Validate that a string is not empty
   * @param {string} value - Value to validate
   * @param {string} name - Parameter name for error messages
   * @returns {string} Validated and trimmed value
   */
  validateStringNotEmpty(value, name) {
    if (typeof value !== 'string') {
      throw new ValidationError(`${name} must be a string`, name, typeof value);
    }
    const trimmed = value.trim();
    if (trimmed.length === 0) {
      throw new ValidationError(`${name} cannot be empty`, name);
    }
    return trimmed;
  },

  /**
   * Validate that an array is not empty
   * @param {Array} value - Value to validate
   * @param {string} name - Parameter name for error messages
   * @returns {Array} Validated value
   */
  validateArrayNotEmpty(value, name) {
    if (!Array.isArray(value)) {
      throw new ValidationError(`${name} must be an array`, name, typeof value);
    }
    if (value.length === 0) {
      throw new ValidationError(`${name} cannot be empty`, name);
    }
    return value;
  },

  /**
   * Format a timestamp as ISO 8601 string
   * @param {Date} date - Date to format
   * @param {boolean} includeTimezone - Whether to include timezone
   * @returns {string} Formatted timestamp
   */
  formatTimestamp(date, includeTimezone = true) {
    if (!(date instanceof Date)) {
      throw new ValidationError('date must be a Date object');
    }
    return includeTimezone ? date.toISOString() : date.toISOString().slice(0, -1);
  },

  /**
   * Parse an ISO 8601 timestamp string
   * @param {string} timestampStr - Timestamp string to parse
   * @returns {Date} Parsed date
   */
  parseTimestamp(timestampStr) {
    if (typeof timestampStr !== 'string') {
      throw new ValidationError('timestamp must be a string');
    }
    
    const date = new Date(timestampStr);
    if (isNaN(date.getTime())) {
      throw new ValidationError(`Invalid timestamp format: ${timestampStr}`);
    }
    
    return date;
  },

  /**
   * Get current UTC timestamp
   * @returns {Date} Current UTC date
   */
  currentTimestamp() {
    return new Date();
  },

  /**
   * Build query parameter string from object
   * @param {Object} params - Parameters object
   * @returns {string} Query string
   */
  buildQueryParams(params) {
    const filtered = {};
    
    Object.keys(params).forEach(key => {
      const value = params[key];
      if (value !== null && value !== undefined) {
        if (Array.isArray(value)) {
          value.forEach((item, index) => {
            filtered[`${key}[${index}]`] = String(item);
          });
        } else if (typeof value === 'boolean') {
          filtered[key] = String(value).toLowerCase();
        } else {
          filtered[key] = String(value);
        }
      }
    });
    
    return new URLSearchParams(filtered).toString();
  },

  /**
   * Sleep for specified milliseconds
   * @param {number} ms - Milliseconds to sleep
   * @returns {Promise} Promise that resolves after delay
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  },

  /**
   * Retry a function with exponential backoff
   * @param {Function} fn - Function to retry
   * @param {Object} options - Retry options
   * @returns {Promise} Promise with function result
   */
  async exponentialBackoffRetry(fn, options = {}) {
    const {
      maxRetries = 3,
      baseDelay = 1000,
      maxDelay = 60000,
      exponentialBase = 2,
      jitter = true
    } = options;

    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;

        if (attempt === maxRetries) {
          break;
        }

        // Calculate delay
        let delay = Math.min(baseDelay * Math.pow(exponentialBase, attempt), maxDelay);

        if (jitter) {
          delay *= (0.5 + Math.random() * 0.5); // Add ±25% jitter
        }

        await this.sleep(delay);
      }
    }

    throw lastError;
  }
};

// Convenience functions
/**
 * Create a TSIOT client instance
 * @param {string} baseUrl - Base URL of the TSIOT service
 * @param {Object} options - Client options
 * @returns {TSIOTClient} Client instance
 */
function createClient(baseUrl, options = {}) {
  return new TSIOTClient(baseUrl, options);
}

/**
 * Create a time series generator
 * @param {TSIOTClient} client - TSIOT client instance
 * @returns {TimeSeriesGenerator} Generator instance
 */
function createGenerator(client) {
  return new TimeSeriesGenerator(client);
}

/**
 * Create a time series from arrays
 * @param {Array<number>} values - Array of values
 * @param {Array<Date>} timestamps - Array of timestamps (optional)
 * @param {Object} metadata - Time series metadata (optional)
 * @returns {TimeSeries} Time series instance
 */
function createTimeSeries(values, timestamps = null, metadata = {}) {
  if (!Array.isArray(values)) {
    throw new ValidationError('values must be an array');
  }

  // Generate timestamps if not provided
  if (!timestamps) {
    const startTime = new Date();
    timestamps = values.map((_, index) => new Date(startTime.getTime() + index * 1000));
  }

  if (values.length !== timestamps.length) {
    throw new ValidationError('values and timestamps arrays must have the same length');
  }

  // Create data points
  const dataPoints = values.map((value, index) => new DataPoint(timestamps[index], value));

  // Create metadata
  const tsMetadata = new TimeSeriesMetadata(
    metadata.seriesId || 'default',
    metadata
  );

  return new TimeSeries(dataPoints, tsMetadata);
}

// Main exports
module.exports = {
  // Core classes
  TSIOTClient,
  TimeSeries,
  DataPoint,
  TimeSeriesMetadata,
  TimeSeriesGenerator,

  // Constants
  DataFormat,
  Frequency,
  ValidationLevel,

  // Error classes
  TSIOTError,
  ValidationError,
  GenerationError,
  NetworkError,
  AuthenticationError,
  RateLimitError,
  TimeoutError,

  // Utility functions
  utils,

  // Convenience functions
  createClient,
  createGenerator,
  createTimeSeries,

  // Aliases for convenience
  Client: TSIOTClient,
  TS: TimeSeries,
  Generator: TimeSeriesGenerator
};

// Package information
module.exports.version = '1.0.0';
module.exports.name = '@tsiot/sdk';
module.exports.description = 'JavaScript/TypeScript SDK for Time Series IoT Synthetic Data service';

// Default configuration
const defaultConfig = {
  timeout: 30000,
  retries: 3,
  batchSize: 1000,
  defaultFormat: DataFormat.JSON,
  verifySSL: true
};

let globalConfig = { ...defaultConfig };

/**
 * Configure global SDK settings
 * @param {Object} config - Configuration options
 */
function configure(config = {}) {
  globalConfig = { ...globalConfig, ...config };
}

/**
 * Get current global configuration
 * @returns {Object} Current configuration
 */
function getConfig() {
  return { ...globalConfig };
}

module.exports.configure = configure;
module.exports.getConfig = getConfig;
module.exports.defaultConfig = defaultConfig;