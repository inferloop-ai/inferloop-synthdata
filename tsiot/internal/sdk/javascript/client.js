/**
 * TSIOT JavaScript SDK Client
 * 
 * This module provides the main client class for interacting with the TSIOT service.
 * It includes comprehensive error handling, authentication, and retry logic.
 */

const axios = require('axios');
const { 
  TSIOTError, 
  NetworkError, 
  AuthenticationError, 
  RateLimitError, 
  TimeoutError, 
  ValidationError,
  utils 
} = require('./index');
const { TimeSeries } = require('./timeseries');

/**
 * TSIOT service client
 * 
 * Provides methods for generating, validating, and analyzing time series data.
 * Handles authentication, retries, and error handling automatically.
 * 
 * @example
 * const client = new TSIOTClient('http://localhost:8080', {
 *   apiKey: 'your-api-key',
 *   timeout: 30000,
 *   retries: 3
 * });
 * 
 * // Generate time series data
 * const request = {
 *   type: 'arima',
 *   length: 1000,
 *   parameters: {
 *     ar_params: [0.5, -0.3],
 *     ma_params: [0.2]
 *   }
 * };
 * 
 * const timeSeries = await client.generate(request);
 * console.log(`Generated ${timeSeries.length} data points`);
 */
class TSIOTClient {
  /**
   * Create a new TSIOT client
   * @param {string} baseUrl - Base URL of the TSIOT service
   * @param {Object} options - Client configuration options
   * @param {string} options.apiKey - API key for authentication
   * @param {string} options.jwtToken - JWT token for authentication
   * @param {number} options.timeout - Request timeout in milliseconds
   * @param {number} options.retries - Number of retry attempts
   * @param {boolean} options.verifySSL - Whether to verify SSL certificates
   * @param {string} options.userAgent - Custom user agent string
   */
  constructor(baseUrl, options = {}) {
    this.baseUrl = utils.validateStringNotEmpty(baseUrl.replace(/\/$/, ''), 'baseUrl');
    this.apiKey = options.apiKey || null;
    this.jwtToken = options.jwtToken || null;
    this.timeout = options.timeout || 30000;
    this.retries = Math.max(0, options.retries || 3);
    this.verifySSL = options.verifySSL !== false;
    this.userAgent = options.userAgent || 'tsiot-javascript-sdk/1.0.0';

    // Validate authentication
    if (!this.apiKey && !this.jwtToken) {
      console.warn('No authentication provided. Some endpoints may not be accessible.');
    }

    // Create axios instance
    this._createHttpClient();

    console.log(`Initialized TSIOT client for ${this.baseUrl}`);
  }

  /**
   * Create and configure the HTTP client
   * @private
   */
  _createHttpClient() {
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': this.userAgent
    };

    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    } else if (this.jwtToken) {
      headers['Authorization'] = `Bearer ${this.jwtToken}`;
    }

    this.httpClient = axios.create({
      baseURL: this.baseUrl,
      timeout: this.timeout,
      headers,
      validateStatus: () => true, // Handle all status codes manually
      httpsAgent: this.verifySSL ? undefined : new (require('https')).Agent({
        rejectUnauthorized: false
      })
    });

    // Add request interceptor for logging
    this.httpClient.interceptors.request.use(
      (config) => {
        console.debug(`Making ${config.method.toUpperCase()} request to ${config.url}`);
        return config;
      },
      (error) => Promise.reject(error)
    );
  }

  /**
   * Make an HTTP request with error handling and retries
   * @param {string} method - HTTP method
   * @param {string} endpoint - API endpoint
   * @param {Object} data - Request body data
   * @param {Object} params - Query parameters
   * @param {number} timeout - Request timeout override
   * @returns {Promise<Object>} Response data
   * @private
   */
  async _makeRequest(method, endpoint, data = null, params = null, timeout = null) {
    const requestConfig = {
      method: method.toLowerCase(),
      url: endpoint,
      data,
      params,
      timeout: timeout || this.timeout
    };

    const makeRequestOnce = async () => {
      try {
        const startTime = Date.now();
        const response = await this.httpClient.request(requestConfig);
        const duration = Date.now() - startTime;

        console.debug(`${method} ${endpoint} completed in ${duration}ms with status ${response.status}`);

        return this._handleResponse(response);
      } catch (error) {
        throw this._handleRequestError(error);
      }
    };

    // Retry logic
    if (this.retries > 0) {
      return utils.exponentialBackoffRetry(makeRequestOnce, {
        maxRetries: this.retries,
        baseDelay: 1000,
        maxDelay: 30000
      });
    } else {
      return makeRequestOnce();
    }
  }

  /**
   * Handle HTTP response and extract data
   * @param {Object} response - Axios response object
   * @returns {Object} Response data
   * @private
   */
  _handleResponse(response) {
    const { status, data, headers } = response;

    switch (status) {
      case 200:
      case 201:
        return data;
      
      case 204:
        return {};
      
      case 400:
        const errorData = data || {};
        throw new ValidationError(
          errorData.message || 'Bad request',
          errorData.field,
          errorData.value,
          { details: errorData }
        );
      
      case 401:
        throw new AuthenticationError('Authentication failed');
      
      case 403:
        throw new AuthenticationError('Access forbidden');
      
      case 404:
        throw new NetworkError(`Endpoint not found: ${response.config.url}`, status);
      
      case 429:
        const retryAfter = parseInt(headers['retry-after']) || 60;
        throw new RateLimitError('Rate limit exceeded', retryAfter);
      
      default:
        if (status >= 500) {
          throw new NetworkError(
            `Server error: ${status}`,
            status,
            typeof data === 'string' ? data : JSON.stringify(data)
          );
        } else {
          throw new NetworkError(
            `Unexpected status code: ${status}`,
            status,
            typeof data === 'string' ? data : JSON.stringify(data)
          );
        }
    }
  }

  /**
   * Handle request errors (network, timeout, etc.)
   * @param {Error} error - Request error
   * @returns {Error} Transformed error
   * @private
   */
  _handleRequestError(error) {
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      return new TimeoutError(`Request timed out after ${this.timeout}ms`);
    }
    
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return new NetworkError(`Network error: ${error.message}`);
    }
    
    if (error instanceof TSIOTError) {
      return error;
    }
    
    return new NetworkError(`HTTP error: ${error.message}`);
  }

  /**
   * Check the health status of the TSIOT service
   * @returns {Promise<Object>} Health status information
   */
  async healthCheck() {
    return this._makeRequest('GET', '/health');
  }

  /**
   * Get information about the TSIOT service
   * @returns {Promise<Object>} Service information
   */
  async getInfo() {
    return this._makeRequest('GET', '/api/v1/info');
  }

  /**
   * Generate synthetic time series data
   * @param {Object} request - Generation request parameters
   * @param {number} timeout - Request timeout override
   * @returns {Promise<TimeSeries>} Generated time series data
   * 
   * @example
   * const request = {
   *   type: 'arima',
   *   length: 1000,
   *   parameters: {
   *     ar_params: [0.5, -0.3],
   *     ma_params: [0.2],
   *     trend: 'linear'
   *   },
   *   metadata: {
   *     series_id: 'test-series',
   *     name: 'Test ARIMA Series'
   *   }
   * };
   * 
   * const timeSeries = await client.generate(request);
   */
  async generate(request, timeout = null) {
    if (!request || typeof request !== 'object') {
      throw new ValidationError('request must be an object');
    }

    const response = await this._makeRequest('POST', '/api/v1/generate', request, null, timeout);
    return TimeSeries.fromObject(response);
  }

  /**
   * Validate time series data
   * @param {TimeSeries|Object} timeSeries - Time series data to validate
   * @param {Array<string>} validationTypes - Types of validation to perform
   * @param {number} timeout - Request timeout override
   * @returns {Promise<Object>} Validation results
   * 
   * @example
   * const validationResult = await client.validate(
   *   timeSeries,
   *   ['quality', 'statistical', 'privacy']
   * );
   * 
   * console.log(`Quality score: ${validationResult.quality_score}`);
   */
  async validate(timeSeries, validationTypes = null, timeout = null) {
    let data;
    if (timeSeries instanceof TimeSeries) {
      data = timeSeries.toObject();
    } else if (typeof timeSeries === 'object') {
      data = timeSeries;
    } else {
      throw new ValidationError('timeSeries must be a TimeSeries instance or object');
    }

    const requestBody = {
      time_series: data,
      validation_types: validationTypes || ['quality', 'statistical']
    };

    return this._makeRequest('POST', '/api/v1/validate', requestBody, null, timeout);
  }

  /**
   * Perform analytics on time series data
   * @param {TimeSeries|Object} timeSeries - Time series data to analyze
   * @param {Array<string>} analysisTypes - Types of analysis to perform
   * @param {Object} parameters - Analysis parameters
   * @param {number} timeout - Request timeout override
   * @returns {Promise<Object>} Analysis results
   * 
   * @example
   * const analysis = await client.analyze(
   *   timeSeries,
   *   ['basic', 'trend', 'seasonality', 'anomaly'],
   *   { forecast_horizon: 24 }
   * );
   * 
   * console.log(`Trend: ${analysis.trend.direction}`);
   * console.log(`Seasonality: ${analysis.seasonality.has_seasonality}`);
   */
  async analyze(timeSeries, analysisTypes = null, parameters = null, timeout = null) {
    let data;
    if (timeSeries instanceof TimeSeries) {
      data = timeSeries.toObject();
    } else if (typeof timeSeries === 'object') {
      data = timeSeries;
    } else {
      throw new ValidationError('timeSeries must be a TimeSeries instance or object');
    }

    const requestBody = {
      time_series: data,
      analysis_types: analysisTypes || ['basic', 'trend'],
      parameters: parameters || {}
    };

    return this._makeRequest('POST', '/api/v1/analyze', requestBody, null, timeout);
  }

  /**
   * List available data generators
   * @returns {Promise<Array>} List of available generators
   */
  async listGenerators() {
    return this._makeRequest('GET', '/api/v1/generators');
  }

  /**
   * List available validators
   * @returns {Promise<Array>} List of available validators
   */
  async listValidators() {
    return this._makeRequest('GET', '/api/v1/validators');
  }

  /**
   * Export time series data in specified format
   * @param {TimeSeries|Object} timeSeries - Time series data to export
   * @param {string} format - Export format ('json', 'csv', 'parquet')
   * @param {boolean} compression - Whether to compress the output
   * @param {number} timeout - Request timeout override
   * @returns {Promise<string|Buffer>} Exported data
   */
  async exportData(timeSeries, format = 'json', compression = false, timeout = null) {
    let data;
    if (timeSeries instanceof TimeSeries) {
      data = timeSeries.toObject();
    } else if (typeof timeSeries === 'object') {
      data = timeSeries;
    } else {
      throw new ValidationError('timeSeries must be a TimeSeries instance or object');
    }

    const requestBody = {
      time_series: data,
      format,
      compression
    };

    const response = await this._makeRequest('POST', '/api/v1/export', requestBody, null, timeout);

    if (compression) {
      // Decode base64 data
      return Buffer.from(response.data, 'base64');
    } else {
      return response.data;
    }
  }

  /**
   * Get service metrics and statistics
   * @returns {Promise<Object>} Service metrics
   */
  async getMetrics() {
    return this._makeRequest('GET', '/metrics');
  }

  /**
   * Generate multiple time series in a single batch request
   * @param {Array<Object>} requests - Array of generation requests
   * @param {number} timeout - Request timeout override
   * @returns {Promise<Array<TimeSeries>>} Array of generated time series
   */
  async batchGenerate(requests, timeout = null) {
    if (!Array.isArray(requests)) {
      throw new ValidationError('requests must be an array');
    }

    const batchRequest = { requests };
    
    const response = await this._makeRequest(
      'POST', 
      '/api/v1/batch/generate', 
      batchRequest, 
      null, 
      timeout
    );

    return response.results.map(tsData => TimeSeries.fromObject(tsData));
  }

  /**
   * Generate multiple time series concurrently
   * @param {Array<Object>} requests - Array of generation requests
   * @param {number} maxConcurrent - Maximum number of concurrent requests
   * @returns {Promise<Array<TimeSeries>>} Array of generated time series
   */
  async concurrentGenerate(requests, maxConcurrent = 10) {
    if (!Array.isArray(requests)) {
      throw new ValidationError('requests must be an array');
    }

    const semaphore = new Semaphore(maxConcurrent);
    
    const generateWithSemaphore = async (request) => {
      await semaphore.acquire();
      try {
        return await this.generate(request);
      } finally {
        semaphore.release();
      }
    };

    const promises = requests.map(request => generateWithSemaphore(request));
    return Promise.all(promises);
  }

  /**
   * Stream generation (placeholder for future WebSocket/SSE implementation)
   * @param {Object} request - Generation request parameters
   * @param {number} chunkSize - Size of each chunk
   * @returns {AsyncGenerator<Array>} Async generator yielding data chunks
   */
  async* streamGenerate(request, chunkSize = 1000) {
    // For now, simulate streaming by generating full data and chunking it
    // In a real implementation, this would use WebSocket or Server-Sent Events
    const fullTimeSeries = await this.generate(request);
    
    for (let i = 0; i < fullTimeSeries.length; i += chunkSize) {
      const chunk = fullTimeSeries.dataPoints.slice(i, i + chunkSize);
      yield chunk;
    }
  }

  /**
   * Close the client and clean up resources
   */
  close() {
    // No specific cleanup needed for axios
    console.log('TSIOT client closed');
  }
}

/**
 * Simple semaphore implementation for controlling concurrency
 */
class Semaphore {
  constructor(permits) {
    this.permits = permits;
    this.queue = [];
  }

  async acquire() {
    return new Promise((resolve) => {
      if (this.permits > 0) {
        this.permits--;
        resolve();
      } else {
        this.queue.push(resolve);
      }
    });
  }

  release() {
    this.permits++;
    if (this.queue.length > 0) {
      const next = this.queue.shift();
      this.permits--;
      next();
    }
  }
}

/**
 * Convenience function to create a TSIOT client
 * @param {string} baseUrl - Base URL of the TSIOT service
 * @param {Object} options - Client options
 * @returns {TSIOTClient} Client instance
 */
function createClient(baseUrl, options = {}) {
  return new TSIOTClient(baseUrl, options);
}

module.exports = {
  TSIOTClient,
  createClient,
  Semaphore
};