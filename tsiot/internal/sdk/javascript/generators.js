/**
 * Time series data generators for the TSIOT JavaScript SDK.
 * 
 * This module provides various generator classes for creating synthetic time series data
 * using different algorithms including ARIMA, LSTM, GRU, TimeGAN, and statistical methods.
 */

const { ValidationError, utils } = require('./index');
const { TimeSeries, DataPoint, TimeSeriesMetadata } = require('./timeseries');

/**
 * Base class for all time series generators
 */
class BaseGenerator {
  /**
   * Create a new generator
   * @param {Object} options - Generator configuration options
   */
  constructor(options = {}) {
    this.options = options;
    this.seed = options.seed || null;
    this.validateOptions();
  }

  /**
   * Validate generator options
   * @protected
   */
  validateOptions() {
    // Base validation - override in subclasses
  }

  /**
   * Generate time series data
   * @param {number} length - Number of data points to generate
   * @param {Object} metadata - Optional metadata for the time series
   * @returns {Promise<TimeSeries>} Generated time series
   * @abstract
   */
  async generate(length, metadata = {}) {
    throw new Error('generate() method must be implemented by subclasses');
  }

  /**
   * Set random seed for reproducible generation
   * @param {number} seed - Random seed
   */
  setSeed(seed) {
    this.seed = seed;
    if (typeof seed === 'number') {
      // Simple seeded random number generator
      this._random = this._createSeededRandom(seed);
    }
  }

  /**
   * Create a seeded random number generator
   * @param {number} seed - Random seed
   * @returns {Function} Random number generator
   * @private
   */
  _createSeededRandom(seed) {
    let state = seed;
    return () => {
      state = (state * 1664525 + 1013904223) % Math.pow(2, 32);
      return state / Math.pow(2, 32);
    };
  }

  /**
   * Get random number (seeded or standard)
   * @returns {number} Random number between 0 and 1
   * @protected
   */
  random() {
    return this._random ? this._random() : Math.random();
  }
}

/**
 * ARIMA (AutoRegressive Integrated Moving Average) generator
 */
class ARIMAGenerator extends BaseGenerator {
  /**
   * Create an ARIMA generator
   * @param {Object} options - ARIMA configuration
   * @param {Array<number>} options.arParams - Autoregressive parameters
   * @param {Array<number>} options.maParams - Moving average parameters
   * @param {number} options.differencing - Degree of differencing (default: 0)
   * @param {number} options.noise - Noise level (default: 1.0)
   * @param {string} options.trend - Trend type ('none', 'linear', 'quadratic')
   */
  constructor(options = {}) {
    super(options);
    this.arParams = options.arParams || [0.5];
    this.maParams = options.maParams || [0.3];
    this.differencing = options.differencing || 0;
    this.noise = options.noise || 1.0;
    this.trend = options.trend || 'none';
  }

  validateOptions() {
    if (!Array.isArray(this.arParams)) {
      throw new ValidationError('arParams must be an array');
    }
    if (!Array.isArray(this.maParams)) {
      throw new ValidationError('maParams must be an array');
    }
    if (this.differencing < 0) {
      throw new ValidationError('differencing must be non-negative');
    }
  }

  /**
   * Generate ARIMA time series
   * @param {number} length - Number of data points
   * @param {Object} metadata - Time series metadata
   * @returns {Promise<TimeSeries>} Generated time series
   */
  async generate(length, metadata = {}) {
    utils.validatePositiveNumber(length, 'length');

    const values = [];
    const errors = [];
    const startTime = new Date();

    // Initialize with small random values
    for (let i = 0; i < Math.max(this.arParams.length, this.maParams.length); i++) {
      values.push(this.random() - 0.5);
      errors.push(this.random() - 0.5);
    }

    // Generate ARIMA process
    for (let t = Math.max(this.arParams.length, this.maParams.length); t < length; t++) {
      let value = 0;
      const error = (this.random() - 0.5) * this.noise;

      // Autoregressive component
      for (let i = 0; i < this.arParams.length; i++) {
        if (t - i - 1 >= 0) {
          value += this.arParams[i] * values[t - i - 1];
        }
      }

      // Moving average component
      for (let i = 0; i < this.maParams.length; i++) {
        if (t - i - 1 >= 0) {
          value += this.maParams[i] * errors[t - i - 1];
        }
      }

      // Add current error
      value += error;

      // Add trend
      value += this._getTrendValue(t, length);

      values.push(value);
      errors.push(error);
    }

    // Apply differencing (integrate)
    if (this.differencing > 0) {
      for (let d = 0; d < this.differencing; d++) {
        for (let i = 1; i < values.length; i++) {
          values[i] += values[i - 1];
        }
      }
    }

    // Create data points
    const dataPoints = values.map((value, index) => {
      const timestamp = new Date(startTime.getTime() + index * 1000);
      return new DataPoint(timestamp, value);
    });

    const tsMetadata = new TimeSeriesMetadata(
      metadata.seriesId || 'arima-generated',
      {
        name: metadata.name || 'ARIMA Generated Series',
        description: `ARIMA(${this.arParams.length},${this.differencing},${this.maParams.length}) generated series`,
        ...metadata
      }
    );

    return new TimeSeries(dataPoints, tsMetadata);
  }

  /**
   * Get trend value for given time point
   * @param {number} t - Time index
   * @param {number} length - Total length
   * @returns {number} Trend value
   * @private
   */
  _getTrendValue(t, length) {
    const normalizedT = t / length;
    switch (this.trend) {
      case 'linear':
        return normalizedT;
      case 'quadratic':
        return normalizedT * normalizedT;
      case 'exponential':
        return Math.exp(normalizedT) - 1;
      default:
        return 0;
    }
  }
}

/**
 * LSTM (Long Short-Term Memory) generator
 * Note: This is a simplified implementation for demonstration
 */
class LSTMGenerator extends BaseGenerator {
  /**
   * Create an LSTM generator
   * @param {Object} options - LSTM configuration
   * @param {number} options.hiddenSize - Hidden layer size (default: 50)
   * @param {number} options.sequenceLength - Input sequence length (default: 10)
   * @param {number} options.numLayers - Number of LSTM layers (default: 2)
   * @param {Array<number>} options.pattern - Pattern to learn and repeat
   */
  constructor(options = {}) {
    super(options);
    this.hiddenSize = options.hiddenSize || 50;
    this.sequenceLength = options.sequenceLength || 10;
    this.numLayers = options.numLayers || 2;
    this.pattern = options.pattern || this._generateSinePattern();
  }

  validateOptions() {
    utils.validatePositiveNumber(this.hiddenSize, 'hiddenSize');
    utils.validatePositiveNumber(this.sequenceLength, 'sequenceLength');
    utils.validatePositiveNumber(this.numLayers, 'numLayers');
  }

  /**
   * Generate default sine wave pattern
   * @returns {Array<number>} Sine wave pattern
   * @private
   */
  _generateSinePattern() {
    const pattern = [];
    for (let i = 0; i < 100; i++) {
      pattern.push(Math.sin(2 * Math.PI * i / 20) + 0.1 * (this.random() - 0.5));
    }
    return pattern;
  }

  /**
   * Generate LSTM-style time series
   * @param {number} length - Number of data points
   * @param {Object} metadata - Time series metadata
   * @returns {Promise<TimeSeries>} Generated time series
   */
  async generate(length, metadata = {}) {
    utils.validatePositiveNumber(length, 'length');

    const values = [];
    const startTime = new Date();

    // Simple pattern-based generation (simulating LSTM output)
    for (let i = 0; i < length; i++) {
      let value = 0;

      // Use pattern with some variation
      const patternIndex = i % this.pattern.length;
      value = this.pattern[patternIndex];

      // Add some noise and dependency on previous values
      if (i > 0) {
        value += 0.1 * values[i - 1];
      }
      if (i > 1) {
        value += 0.05 * values[i - 2];
      }

      // Add noise
      value += 0.1 * (this.random() - 0.5);

      values.push(value);
    }

    // Create data points
    const dataPoints = values.map((value, index) => {
      const timestamp = new Date(startTime.getTime() + index * 1000);
      return new DataPoint(timestamp, value);
    });

    const tsMetadata = new TimeSeriesMetadata(
      metadata.seriesId || 'lstm-generated',
      {
        name: metadata.name || 'LSTM Generated Series',
        description: `LSTM generated series with ${this.hiddenSize} hidden units`,
        ...metadata
      }
    );

    return new TimeSeries(dataPoints, tsMetadata);
  }
}

/**
 * GRU (Gated Recurrent Unit) generator
 */
class GRUGenerator extends BaseGenerator {
  /**
   * Create a GRU generator
   * @param {Object} options - GRU configuration
   * @param {number} options.hiddenSize - Hidden layer size (default: 32)
   * @param {number} options.sequenceLength - Input sequence length (default: 8)
   * @param {Array<number>} options.seasonality - Seasonal pattern
   */
  constructor(options = {}) {
    super(options);
    this.hiddenSize = options.hiddenSize || 32;
    this.sequenceLength = options.sequenceLength || 8;
    this.seasonality = options.seasonality || [1, 0.8, 0.6, 0.4, 0.6, 0.8];
  }

  validateOptions() {
    utils.validatePositiveNumber(this.hiddenSize, 'hiddenSize');
    utils.validatePositiveNumber(this.sequenceLength, 'sequenceLength');
  }

  /**
   * Generate GRU-style time series
   * @param {number} length - Number of data points
   * @param {Object} metadata - Time series metadata
   * @returns {Promise<TimeSeries>} Generated time series
   */
  async generate(length, metadata = {}) {
    utils.validatePositiveNumber(length, 'length');

    const values = [];
    const startTime = new Date();
    let hiddenState = 0;

    for (let i = 0; i < length; i++) {
      // Simplified GRU-like computation
      const seasonalValue = this.seasonality[i % this.seasonality.length];
      const input = seasonalValue + 0.1 * (this.random() - 0.5);

      // Update gate (simplified)
      const updateGate = 1 / (1 + Math.exp(-(input + hiddenState)));
      
      // Reset gate (simplified)
      const resetGate = 1 / (1 + Math.exp(-(input * 0.5 + hiddenState * 0.5)));
      
      // Candidate state
      const candidateState = Math.tanh(input + resetGate * hiddenState);
      
      // Update hidden state
      hiddenState = (1 - updateGate) * hiddenState + updateGate * candidateState;
      
      values.push(hiddenState);
    }

    // Create data points
    const dataPoints = values.map((value, index) => {
      const timestamp = new Date(startTime.getTime() + index * 1000);
      return new DataPoint(timestamp, value);
    });

    const tsMetadata = new TimeSeriesMetadata(
      metadata.seriesId || 'gru-generated',
      {
        name: metadata.name || 'GRU Generated Series',
        description: `GRU generated series with ${this.hiddenSize} hidden units`,
        ...metadata
      }
    );

    return new TimeSeries(dataPoints, tsMetadata);
  }
}

/**
 * TimeGAN (Time-series Generative Adversarial Network) generator
 */
class TimeGANGenerator extends BaseGenerator {
  /**
   * Create a TimeGAN generator
   * @param {Object} options - TimeGAN configuration
   * @param {number} options.latentDim - Latent dimension (default: 24)
   * @param {number} options.sequenceLength - Sequence length (default: 24)
   * @param {Array<number>} options.referencePattern - Reference pattern to learn
   */
  constructor(options = {}) {
    super(options);
    this.latentDim = options.latentDim || 24;
    this.sequenceLength = options.sequenceLength || 24;
    this.referencePattern = options.referencePattern || this._generateComplexPattern();
  }

  validateOptions() {
    utils.validatePositiveNumber(this.latentDim, 'latentDim');
    utils.validatePositiveNumber(this.sequenceLength, 'sequenceLength');
  }

  /**
   * Generate complex reference pattern
   * @returns {Array<number>} Complex pattern
   * @private
   */
  _generateComplexPattern() {
    const pattern = [];
    for (let i = 0; i < 100; i++) {
      const value = Math.sin(2 * Math.PI * i / 12) * 0.8 +
                   Math.sin(2 * Math.PI * i / 24) * 0.5 +
                   Math.sin(2 * Math.PI * i / 7) * 0.3 +
                   0.05 * (this.random() - 0.5);
      pattern.push(value);
    }
    return pattern;
  }

  /**
   * Generate TimeGAN-style time series
   * @param {number} length - Number of data points
   * @param {Object} metadata - Time series metadata
   * @returns {Promise<TimeSeries>} Generated time series
   */
  async generate(length, metadata = {}) {
    utils.validatePositiveNumber(length, 'length');

    const values = [];
    const startTime = new Date();

    // Simulate TimeGAN generation process
    for (let i = 0; i < length; i++) {
      // Generate latent vector (random noise)
      const latentVector = Array.from({ length: this.latentDim }, () => this.random() - 0.5);
      
      // Simplified "generator" network simulation
      let generatedValue = 0;
      for (let j = 0; j < this.latentDim; j++) {
        generatedValue += latentVector[j] * Math.sin(2 * Math.PI * j / this.latentDim);
      }
      
      // Add reference pattern influence
      const patternIndex = i % this.referencePattern.length;
      generatedValue = 0.7 * generatedValue + 0.3 * this.referencePattern[patternIndex];
      
      // Temporal consistency (look-back)
      if (i > 0) {
        generatedValue = 0.8 * generatedValue + 0.2 * values[i - 1];
      }
      
      values.push(generatedValue);
    }

    // Create data points
    const dataPoints = values.map((value, index) => {
      const timestamp = new Date(startTime.getTime() + index * 1000);
      return new DataPoint(timestamp, value);
    });

    const tsMetadata = new TimeSeriesMetadata(
      metadata.seriesId || 'timegan-generated',
      {
        name: metadata.name || 'TimeGAN Generated Series',
        description: `TimeGAN generated series with latent dimension ${this.latentDim}`,
        ...metadata
      }
    );

    return new TimeSeries(dataPoints, tsMetadata);
  }
}

/**
 * Statistical pattern generator
 */
class StatisticalGenerator extends BaseGenerator {
  /**
   * Create a statistical generator
   * @param {Object} options - Statistical configuration
   * @param {string} options.distribution - Distribution type ('normal', 'uniform', 'exponential')
   * @param {Object} options.parameters - Distribution parameters
   * @param {string} options.trend - Trend type ('none', 'linear', 'exponential')
   * @param {Array<number>} options.seasonal - Seasonal pattern
   */
  constructor(options = {}) {
    super(options);
    this.distribution = options.distribution || 'normal';
    this.parameters = options.parameters || { mean: 0, std: 1 };
    this.trend = options.trend || 'none';
    this.seasonal = options.seasonal || [];
  }

  validateOptions() {
    const validDistributions = ['normal', 'uniform', 'exponential', 'poisson'];
    if (!validDistributions.includes(this.distribution)) {
      throw new ValidationError(`distribution must be one of: ${validDistributions.join(', ')}`);
    }
  }

  /**
   * Generate statistical time series
   * @param {number} length - Number of data points
   * @param {Object} metadata - Time series metadata
   * @returns {Promise<TimeSeries>} Generated time series
   */
  async generate(length, metadata = {}) {
    utils.validatePositiveNumber(length, 'length');

    const values = [];
    const startTime = new Date();

    for (let i = 0; i < length; i++) {
      let value = this._generateFromDistribution();
      
      // Add trend
      value += this._getTrendValue(i, length);
      
      // Add seasonal component
      if (this.seasonal.length > 0) {
        value += this.seasonal[i % this.seasonal.length];
      }
      
      values.push(value);
    }

    // Create data points
    const dataPoints = values.map((value, index) => {
      const timestamp = new Date(startTime.getTime() + index * 1000);
      return new DataPoint(timestamp, value);
    });

    const tsMetadata = new TimeSeriesMetadata(
      metadata.seriesId || 'statistical-generated',
      {
        name: metadata.name || 'Statistical Generated Series',
        description: `Statistical series with ${this.distribution} distribution`,
        ...metadata
      }
    );

    return new TimeSeries(dataPoints, tsMetadata);
  }

  /**
   * Generate value from specified distribution
   * @returns {number} Generated value
   * @private
   */
  _generateFromDistribution() {
    switch (this.distribution) {
      case 'normal':
        return this._generateNormal(this.parameters.mean || 0, this.parameters.std || 1);
      case 'uniform':
        return this._generateUniform(this.parameters.min || 0, this.parameters.max || 1);
      case 'exponential':
        return this._generateExponential(this.parameters.lambda || 1);
      case 'poisson':
        return this._generatePoisson(this.parameters.lambda || 1);
      default:
        return this.random();
    }
  }

  /**
   * Generate normal random variable (Box-Muller transform)
   * @param {number} mean - Mean
   * @param {number} std - Standard deviation
   * @returns {number} Normal random variable
   * @private
   */
  _generateNormal(mean, std) {
    if (!this._spare) {
      const u = this.random();
      const v = this.random();
      const z0 = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      const z1 = Math.sqrt(-2 * Math.log(u)) * Math.sin(2 * Math.PI * v);
      this._spare = z1;
      return z0 * std + mean;
    } else {
      const temp = this._spare;
      this._spare = null;
      return temp * std + mean;
    }
  }

  /**
   * Generate uniform random variable
   * @param {number} min - Minimum value
   * @param {number} max - Maximum value
   * @returns {number} Uniform random variable
   * @private
   */
  _generateUniform(min, max) {
    return min + (max - min) * this.random();
  }

  /**
   * Generate exponential random variable
   * @param {number} lambda - Rate parameter
   * @returns {number} Exponential random variable
   * @private
   */
  _generateExponential(lambda) {
    return -Math.log(this.random()) / lambda;
  }

  /**
   * Generate Poisson random variable (Knuth's algorithm)
   * @param {number} lambda - Rate parameter
   * @returns {number} Poisson random variable
   * @private
   */
  _generatePoisson(lambda) {
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    
    do {
      k++;
      p *= this.random();
    } while (p > L);
    
    return k - 1;
  }

  /**
   * Get trend value for given time point
   * @param {number} t - Time index
   * @param {number} length - Total length
   * @returns {number} Trend value
   * @private
   */
  _getTrendValue(t, length) {
    const normalizedT = t / length;
    switch (this.trend) {
      case 'linear':
        return normalizedT;
      case 'exponential':
        return Math.exp(normalizedT) - 1;
      case 'logarithmic':
        return Math.log(normalizedT + 1);
      default:
        return 0;
    }
  }
}

/**
 * Generator factory for creating generators by type
 */
class GeneratorFactory {
  /**
   * Create a generator by type
   * @param {string} type - Generator type
   * @param {Object} options - Generator options
   * @returns {BaseGenerator} Generator instance
   */
  static createGenerator(type, options = {}) {
    switch (type.toLowerCase()) {
      case 'arima':
        return new ARIMAGenerator(options);
      case 'lstm':
        return new LSTMGenerator(options);
      case 'gru':
        return new GRUGenerator(options);
      case 'timegan':
        return new TimeGANGenerator(options);
      case 'statistical':
        return new StatisticalGenerator(options);
      default:
        throw new ValidationError(`Unknown generator type: ${type}`);
    }
  }

  /**
   * Get list of available generator types
   * @returns {Array<string>} Available generator types
   */
  static getAvailableTypes() {
    return ['arima', 'lstm', 'gru', 'timegan', 'statistical'];
  }
}

module.exports = {
  BaseGenerator,
  ARIMAGenerator,
  LSTMGenerator,
  GRUGenerator,
  TimeGANGenerator,
  StatisticalGenerator,
  GeneratorFactory
};