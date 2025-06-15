# JavaScript SDK Guide

The TSIoT JavaScript SDK provides a modern, TypeScript-ready interface for generating and validating synthetic time series data in both Node.js and browser environments.

## Installation

### Using npm

```bash
npm install @inferloop/tsiot-sdk
```

### Using yarn

```bash
yarn add @inferloop/tsiot-sdk
```

### Using CDN (Browser)

```html
<script src="https://unpkg.com/@inferloop/tsiot-sdk@latest/dist/tsiot.min.js"></script>
```

### Requirements

- Node.js 16+ (for server-side)
- Modern browsers with ES2020 support
- TypeScript 4.5+ (optional)

## Quick Start

### Node.js

```javascript
const { TSIoTClient } = require('@inferloop/tsiot-sdk');
// or with ES modules
import { TSIoTClient } from '@inferloop/tsiot-sdk';

// Create client
const client = new TSIoTClient('http://localhost:8080');

// Generate synthetic data
async function generateData() {
  try {
    const data = await client.generate({
      generator: 'statistical',
      sensorType: 'temperature',
      points: 1000,
      frequency: '1m'
    });
    
    console.log(`Generated ${data.points.length} data points`);
    
    // Save to file (Node.js only)
    await data.saveCSV('temperature_data.csv');
  } catch (error) {
    console.error('Generation failed:', error);
  }
}

generateData();
```

### Browser

```html
<!DOCTYPE html>
<html>
<head>
  <script type="module">
    import { TSIoTClient } from 'https://unpkg.com/@inferloop/tsiot-sdk@latest/dist/esm/index.js';
    
    const client = new TSIoTClient('http://localhost:8080');
    
    async function generateAndVisualize() {
      const data = await client.generate({
        generator: 'arima',
        sensorType: 'temperature',
        points: 100,
        frequency: '1m'
      });
      
      // Visualize with Chart.js or similar
      const chartData = {
        labels: data.timestamps(),
        datasets: [{
          label: 'Temperature',
          data: data.values(),
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1
        }]
      };
      
      // Render chart...
    }
    
    generateAndVisualize();
  </script>
</head>
<body>
  <canvas id="chart"></canvas>
</body>
</html>
```

### TypeScript

```typescript
import { TSIoTClient, GenerateRequest, TimeSeriesData } from '@inferloop/tsiot-sdk';

const client = new TSIoTClient('http://localhost:8080');

interface SensorConfig {
  id: string;
  type: string;
  location: string;
}

async function generateSensorData(config: SensorConfig): Promise<TimeSeriesData> {
  const request: GenerateRequest = {
    generator: 'timegan',
    sensorType: config.type,
    sensorId: config.id,
    points: 1440, // 24 hours of minute data
    frequency: '1m',
    metadata: {
      location: config.location
    }
  };
  
  return await client.generate(request);
}
```

## API Reference

### TSIoTClient

```typescript
class TSIoTClient {
  constructor(baseURL: string, options?: ClientOptions)
  
  // Generation
  generate(request: GenerateRequest): Promise<TimeSeriesData>
  
  // Validation
  validate(request: ValidateRequest): Promise<ValidationResult>
  
  // Analysis
  analyze(request: AnalyzeRequest): Promise<AnalysisResult>
  
  // Streaming
  stream(request: StreamRequest): AsyncIterableIterator<DataPoint>
  
  // Utilities
  health(): Promise<HealthStatus>
  convert(data: TimeSeriesData, format: string): Promise<string>
}
```

#### Client Options

```typescript
interface ClientOptions {
  apiKey?: string;
  timeout?: number;           // milliseconds
  retries?: number;
  retryDelay?: number;        // milliseconds
  userAgent?: string;
  headers?: Record<string, string>;
}
```

#### Examples

```javascript
// Basic client
const client = new TSIoTClient('http://localhost:8080');

// With authentication
const client = new TSIoTClient('https://tsiot.example.com', {
  apiKey: 'your-api-key',
  timeout: 30000
});

// With custom headers
const client = new TSIoTClient('http://localhost:8080', {
  headers: {
    'X-Custom-Header': 'value'
  }
});
```

### Generate Data

```typescript
interface GenerateRequest {
  generator: 'statistical' | 'arima' | 'timegan' | 'rnn' | 'ydata';
  sensorType: string;
  sensorId?: string;
  points?: number;
  startTime?: string | Date;
  endTime?: string | Date;
  duration?: string;
  frequency?: string;
  anomalies?: number;
  anomalyMagnitude?: number;
  noiseLevel?: number;
  trend?: number;
  seasonality?: boolean;
  seasonalPeriod?: string;
  seed?: number;
  privacy?: boolean;
  privacyEpsilon?: number;
  metadata?: Record<string, any>;
  customParams?: Record<string, any>;
}
```

#### Examples

```javascript
// Basic generation
const data = await client.generate({
  generator: 'statistical',
  sensorType: 'temperature',
  points: 1000
});

// Time range generation
const data = await client.generate({
  generator: 'arima',
  sensorType: 'humidity',
  startTime: new Date('2024-01-01'),
  endTime: new Date('2024-01-31'),
  frequency: '5m'
});

// With anomalies and seasonality
const data = await client.generate({
  generator: 'timegan',
  sensorType: 'energy',
  points: 10000,
  anomalies: 20,
  anomalyMagnitude: 2.5,
  seasonality: true,
  seasonalPeriod: '24h',
  seed: 42
});

// Privacy-preserving
const data = await client.generate({
  generator: 'ydata',
  sensorType: 'medical',
  points: 5000,
  privacy: true,
  privacyEpsilon: 0.5
});
```

### Validate Data

```typescript
interface ValidateRequest {
  data: TimeSeriesData | string; // Data or file path
  reference?: TimeSeriesData | string;
  metrics?: string[];
  statisticalTests?: string[];
  qualityThreshold?: number;
}

interface ValidationResult {
  qualityScore: number;
  metrics: Record<string, number>;
  statisticalTests: Record<string, TestResult>;
  anomalies: Anomaly[];
  recommendations: string[];
  passed: boolean;
}
```

#### Examples

```javascript
// Basic validation
const result = await client.validate({ data });
console.log(`Quality score: ${result.qualityScore}`);

// Compare with reference
const referenceData = await TimeSeriesData.fromCSV('real_data.csv');
const result = await client.validate({
  data: syntheticData,
  reference: referenceData,
  metrics: ['distribution', 'temporal', 'similarity']
});

// Validate from URL
const result = await client.validate({
  data: 'https://example.com/synthetic_data.csv',
  reference: 'https://example.com/real_data.csv',
  statisticalTests: ['ks', 'anderson', 'ljung_box']
});
```

### Analyze Data

```typescript
interface AnalyzeRequest {
  data: TimeSeriesData | string;
  detectAnomalies?: boolean;
  anomalyMethod?: string;
  decompose?: boolean;
  forecast?: boolean;
  forecastPeriods?: number;
  patterns?: boolean;
}

interface AnalysisResult {
  summaryStats: Record<string, number>;
  anomalies: Anomaly[];
  decomposition?: Decomposition;
  forecast?: Forecast;
  patterns: Pattern[];
}
```

#### Examples

```javascript
// Basic analysis
const result = await client.analyze({ data });
console.log('Summary stats:', result.summaryStats);

// Anomaly detection
const result = await client.analyze({
  data: sensorData,
  detectAnomalies: true,
  anomalyMethod: 'isolation_forest'
});
console.log(`Found ${result.anomalies.length} anomalies`);

// Time series decomposition and forecasting
const result = await client.analyze({
  data: historicalData,
  decompose: true,
  forecast: true,
  forecastPeriods: 24
});
```

### Stream Data

```typescript
interface StreamRequest {
  generator: string;
  sensorType: string;
  sensorId?: string;
  frequency: string;
  customParams?: Record<string, any>;
}
```

#### Examples

```javascript
// Basic streaming (Node.js)
for await (const point of client.stream({
  generator: 'statistical',
  sensorType: 'temperature',
  frequency: '5s'
})) {
  console.log(`${point.timestamp}: ${point.value}`);
}

// With WebSocket (Browser)
const stream = client.streamWebSocket({
  generator: 'arima',
  sensorType: 'pressure',
  frequency: '1s'
});

stream.on('data', (point) => {
  updateChart(point);
});

stream.on('error', (error) => {
  console.error('Stream error:', error);
});

stream.on('close', () => {
  console.log('Stream closed');
});
```

## Data Classes

### TimeSeriesData

```typescript
class TimeSeriesData {
  points: DataPoint[];
  metadata?: Record<string, any>;
  
  // Properties
  get length(): number;
  
  // Methods
  timestamps(): Date[];
  values(): number[];
  sensorIds(): string[];
  
  // Filtering and transformation
  filter(startTime?: Date, endTime?: Date): TimeSeriesData;
  resample(frequency: string, method?: 'mean' | 'sum' | 'max' | 'min'): TimeSeriesData;
  addNoise(level: number): TimeSeriesData;
  normalize(method?: 'minmax' | 'zscore'): TimeSeriesData;
  
  // Export (Node.js only)
  saveCSV(filename: string): Promise<void>;
  saveJSON(filename: string): Promise<void>;
  saveParquet(filename: string): Promise<void>;
  
  // Export (Browser compatible)
  toCSV(): string;
  toJSON(): string;
  toBlob(format: 'csv' | 'json'): Blob;
  
  // Static methods
  static fromCSV(data: string | File): Promise<TimeSeriesData>;
  static fromJSON(data: string | object): TimeSeriesData;
  static concat(datasets: TimeSeriesData[]): TimeSeriesData;
}
```

### DataPoint

```typescript
interface DataPoint {
  timestamp: Date;
  sensorId: string;
  sensorType: string;
  value: number;
  unit?: string;
  metadata?: Record<string, any>;
}
```

## Advanced Usage

### Parallel Generation

```javascript
async function generateMultipleSensors() {
  const sensorIds = ['sensor_001', 'sensor_002', 'sensor_003'];
  
  // Generate in parallel
  const promises = sensorIds.map(sensorId => 
    client.generate({
      generator: 'arima',
      sensorType: 'temperature',
      sensorId,
      points: 1440, // 24 hours
      frequency: '1m'
    })
  );
  
  const results = await Promise.all(promises);
  
  // Combine all data
  const allData = TimeSeriesData.concat(results);
  console.log(`Generated ${allData.length} total points`);
  
  return allData;
}
```

### Real-time Visualization

```javascript
// Real-time chart updates (Browser)
import Chart from 'chart.js/auto';

class RealTimeChart {
  constructor(canvasId) {
    this.chart = new Chart(document.getElementById(canvasId), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Temperature',
          data: [],
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1
        }]
      },
      options: {
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'second'
            }
          }
        },
        animation: false
      }
    });
    
    this.maxPoints = 100;
  }
  
  addPoint(point) {
    const data = this.chart.data;
    data.labels.push(point.timestamp);
    data.datasets[0].data.push(point.value);
    
    // Keep only last N points
    if (data.labels.length > this.maxPoints) {
      data.labels.shift();
      data.datasets[0].data.shift();
    }
    
    this.chart.update('none');
  }
}

// Usage
const chart = new RealTimeChart('myChart');
const stream = client.streamWebSocket({
  generator: 'arima',
  sensorType: 'temperature',
  frequency: '1s'
});

stream.on('data', (point) => {
  chart.addPoint(point);
});
```

### Custom Generators with Parameters

```javascript
const customData = await client.generate({
  generator: 'statistical',
  sensorType: 'custom',
  points: 5000,
  customParams: {
    distribution: 'exponential',
    scale: 2.0,
    offset: 10.0,
    seasonalityComponents: [
      { period: '24h', amplitude: 5.0 },
      { period: '7d', amplitude: 2.0 }
    ]
  }
});
```

### Integration with Data Processing Libraries

```javascript
// With D3.js
import * as d3 from 'd3';

const data = await client.generate({
  generator: 'timegan',
  sensorType: 'temperature',
  points: 1000
});

// Convert to D3-friendly format
const d3Data = data.points.map(p => ({
  date: p.timestamp,
  value: p.value
}));

// Create scales
const xScale = d3.scaleTime()
  .domain(d3.extent(d3Data, d => d.date))
  .range([0, width]);

const yScale = d3.scaleLinear()
  .domain(d3.extent(d3Data, d => d.value))
  .range([height, 0]);

// Draw line
const line = d3.line()
  .x(d => xScale(d.date))
  .y(d => yScale(d.value));

svg.append('path')
  .datum(d3Data)
  .attr('d', line);
```

### Data Processing Pipeline

```javascript
class DataPipeline {
  constructor(client) {
    this.client = client;
    this.processors = [];
  }
  
  addProcessor(fn) {
    this.processors.push(fn);
    return this;
  }
  
  async process(request) {
    let data = await this.client.generate(request);
    
    for (const processor of this.processors) {
      data = await processor(data);
    }
    
    return data;
  }
}

// Usage
const pipeline = new DataPipeline(client)
  .addProcessor(data => data.addNoise(0.05))
  .addProcessor(data => data.normalize('zscore'))
  .addProcessor(async (data) => {
    // Validate processed data
    const result = await client.validate({ data });
    if (result.qualityScore < 0.8) {
      throw new Error('Data quality too low');
    }
    return data;
  });

const processedData = await pipeline.process({
  generator: 'arima',
  sensorType: 'temperature',
  points: 1000
});
```

### Error Handling

```javascript
import { TSIoTError, NetworkError, ValidationError } from '@inferloop/tsiot-sdk';

try {
  const data = await client.generate({
    generator: 'invalid_generator',
    points: 1000
  });
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Validation failed:', error.details);
  } else if (error instanceof NetworkError) {
    console.error('Network error:', error.message);
    // Implement retry logic
  } else if (error instanceof TSIoTError) {
    console.error('TSIoT API error:', error.message, error.code);
  } else {
    console.error('Unknown error:', error);
  }
}
```

### Configuration Management

```javascript
// config.js
export const config = {
  serverUrl: process.env.TSIOT_SERVER_URL || 'http://localhost:8080',
  apiKey: process.env.TSIOT_API_KEY,
  timeout: parseInt(process.env.TSIOT_TIMEOUT) || 30000,
  defaults: {
    generator: 'statistical',
    sensorType: 'temperature',
    points: 1000,
    frequency: '1m'
  }
};

// client.js
import { TSIoTClient } from '@inferloop/tsiot-sdk';
import { config } from './config.js';

export const client = new TSIoTClient(config.serverUrl, {
  apiKey: config.apiKey,
  timeout: config.timeout
});

export function generateWithDefaults(overrides = {}) {
  return client.generate({
    ...config.defaults,
    ...overrides
  });
}
```

## Testing

### Mock Client

```javascript
import { MockTSIoTClient } from '@inferloop/tsiot-sdk/testing';

// For unit testing
const mockClient = new MockTSIoTClient();

mockClient.setGenerateResponse({
  points: [
    {
      timestamp: new Date(),
      sensorId: 'test',
      sensorType: 'temperature',
      value: 25.0
    }
  ]
});

// Test your code with mock client
const data = await mockClient.generate({ points: 1 });
expect(data.points).toHaveLength(1);
```

### Jest Integration

```javascript
// tsiot.test.js
import { TSIoTClient } from '@inferloop/tsiot-sdk';

describe('TSIoT Integration', () => {
  let client;
  
  beforeAll(() => {
    client = new TSIoTClient('http://localhost:8080');
  });
  
  test('should generate temperature data', async () => {
    const data = await client.generate({
      generator: 'statistical',
      sensorType: 'temperature',
      points: 100
    });
    
    expect(data.points).toHaveLength(100);
    expect(data.values().every(v => typeof v === 'number')).toBe(true);
  });
  
  test('should validate data quality', async () => {
    const data = await client.generate({
      generator: 'statistical',
      sensorType: 'temperature',
      points: 100
    });
    
    const result = await client.validate({ data });
    expect(result.qualityScore).toBeGreaterThan(0.7);
  });
});
```

## Framework Integration

### React Hook

```javascript
import { useState, useEffect } from 'react';
import { TSIoTClient } from '@inferloop/tsiot-sdk';

function useTSIoTData(request, dependencies = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    if (!request) return;
    
    setLoading(true);
    setError(null);
    
    const client = new TSIoTClient('http://localhost:8080');
    
    client.generate(request)
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, dependencies);
  
  return { data, loading, error };
}

// Usage in component
function TemperatureChart({ sensorId }) {
  const { data, loading, error } = useTSIoTData({
    generator: 'arima',
    sensorType: 'temperature',
    sensorId,
    points: 100
  }, [sensorId]);
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!data) return null;
  
  return <LineChart data={data.values()} labels={data.timestamps()} />;
}
```

### Vue Composition API

```javascript
import { ref, watch, computed } from 'vue';
import { TSIoTClient } from '@inferloop/tsiot-sdk';

export function useTSIoT() {
  const client = new TSIoTClient('http://localhost:8080');
  const data = ref(null);
  const loading = ref(false);
  const error = ref(null);
  
  async function generate(request) {
    loading.value = true;
    error.value = null;
    
    try {
      data.value = await client.generate(request);
    } catch (err) {
      error.value = err;
    } finally {
      loading.value = false;
    }
  }
  
  const chartData = computed(() => {
    if (!data.value) return null;
    
    return {
      labels: data.value.timestamps(),
      datasets: [{
        label: 'Sensor Data',
        data: data.value.values()
      }]
    };
  });
  
  return {
    data,
    loading,
    error,
    generate,
    chartData
  };
}
```

## Examples

See the [examples directory](../../../examples/javascript-sdk/) for complete examples:

- [Basic Usage](../../../examples/javascript-sdk/basic/)
- [React Integration](../../../examples/javascript-sdk/react/)
- [Vue Integration](../../../examples/javascript-sdk/vue/)
- [Real-time Dashboard](../../../examples/javascript-sdk/dashboard/)
- [Data Visualization](../../../examples/javascript-sdk/visualization/)

## Support

- **Documentation**: https://docs.inferloop.com/tsiot/javascript-sdk
- **GitHub Issues**: https://github.com/inferloop/tsiot-javascript-sdk/issues
- **npm Package**: https://www.npmjs.com/package/@inferloop/tsiot-sdk
- **Examples**: https://github.com/inferloop/tsiot/tree/main/examples/javascript-sdk