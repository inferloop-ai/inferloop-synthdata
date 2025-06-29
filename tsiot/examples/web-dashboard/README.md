# TSIoT Web Dashboard

A comprehensive web-based dashboard for visualizing and managing TSIoT synthetic time series data generation, analysis, and quality monitoring.

## Overview

The TSIoT Web Dashboard provides an intuitive interface for:

- **Interactive Data Generation**: Configure and generate synthetic IoT sensor data
- **Real-time Visualization**: View time series data with interactive charts and graphs
- **Quality Analytics**: Monitor data quality metrics and validation results
- **Batch Processing**: Manage large-scale data generation workflows
- **Export & Integration**: Download data in various formats and integrate with external systems

## Features

### <� Core Features

- **Multi-Sensor Support**: Generate data for various IoT sensor types
- **Advanced Generators**: Support for Statistical, TimeGAN, and WGAN-GP models
- **Interactive Charts**: Real-time plotting with zoom, pan, and filtering
- **Quality Metrics**: Comprehensive data quality assessment and scoring
- **Batch Operations**: Queue and manage multiple generation jobs
- **Export Options**: JSON, CSV, Parquet, and other format support

### =� Visualization Components

- **Time Series Plot**: Interactive line charts with multiple series
- **Statistical Dashboard**: Histograms, box plots, and distribution analysis
- **Quality Scorecard**: Real-time quality metrics and validation status
- **Correlation Matrix**: Multi-variate sensor correlation analysis
- **Anomaly Detection**: Visual highlighting of detected anomalies
- **Pattern Analysis**: Seasonal decomposition and trend visualization

### � Configuration Management

- **Generator Settings**: Configure model parameters and training options
- **Sensor Profiles**: Define custom sensor characteristics and behaviors
- **Quality Thresholds**: Set validation criteria and alert conditions
- **Export Preferences**: Customize output formats and destinations

## Technology Stack

### Frontend
- **React 18**: Modern component-based UI framework
- **TypeScript**: Type-safe JavaScript development
- **Material-UI (MUI)**: Professional UI component library
- **Chart.js / Recharts**: Interactive charting and visualization
- **React Query**: Efficient data fetching and caching
- **Zustand**: Lightweight state management

### Backend Integration
- **RESTful API**: Integration with TSIoT backend services
- **WebSocket**: Real-time updates for long-running operations
- **File Upload**: Drag-and-drop file handling for reference data
- **Streaming**: Progressive data loading for large datasets

### Development Tools
- **Vite**: Fast build tool and development server
- **ESLint**: Code quality and consistency
- **Prettier**: Code formatting
- **Jest**: Unit testing framework
- **Cypress**: End-to-end testing

## Quick Start

### Prerequisites

- Node.js 18+ and npm/yarn
- TSIoT backend service running
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# TSIoT Backend Configuration
VITE_API_BASE_URL=http://localhost:8080/api
VITE_WS_BASE_URL=ws://localhost:8080/ws

# Feature Flags
VITE_ENABLE_ADVANCED_GENERATORS=true
VITE_ENABLE_BATCH_PROCESSING=true
VITE_ENABLE_EXPORT_FORMATS=json,csv,parquet

# UI Configuration
VITE_DEFAULT_CHART_THEME=light
VITE_MAX_DATA_POINTS=10000
VITE_REFRESH_INTERVAL=5000
```

## Usage Guide

### 1. Data Generation

1. **Select Sensor Type**: Choose from temperature, humidity, pressure, etc.
2. **Configure Generator**: Select statistical, TimeGAN, or WGAN-GP
3. **Set Parameters**: Define count, frequency, and time range
4. **Generate Data**: Click "Generate" and monitor progress
5. **View Results**: Interactive charts appear automatically

### 2. Quality Analysis

1. **Upload Reference Data** (optional): Provide real sensor data for comparison
2. **Configure Validators**: Select statistical, distributional, or temporal validation
3. **Set Thresholds**: Define quality score requirements
4. **Run Validation**: Automatic quality assessment with detailed reports
5. **Review Results**: Quality scorecard with pass/fail indicators

### 3. Batch Processing

1. **Create Job Queue**: Add multiple generation tasks
2. **Configure Dependencies**: Set up sequential or parallel execution
3. **Monitor Progress**: Real-time job status and completion tracking
4. **Review Outputs**: Consolidated results and quality reports
5. **Export Batch**: Download all generated data as archive

### 4. Advanced Analytics

1. **Multi-Sensor Correlation**: Analyze relationships between sensor types
2. **Seasonal Decomposition**: Break down trends, seasonality, and residuals
3. **Anomaly Detection**: Identify outliers and unusual patterns
4. **Pattern Mining**: Discover recurring behaviors and cycles
5. **Comparative Analysis**: Compare different generators and parameters

## API Integration

### TSIoT Backend Endpoints

```typescript
// Data Generation
POST /api/v1/generate
{
  "generator": "statistical",
  "sensorType": "temperature",
  "count": 1440,
  "frequency": "1m",
  "startTime": "2023-01-01T00:00:00Z"
}

// Data Analysis
POST /api/v1/analyze
{
  "data": [...],
  "analysisTypes": ["basic_stats", "trend", "seasonality"]
}

// Quality Validation
POST /api/v1/validate
{
  "synthetic": [...],
  "reference": [...],
  "validators": ["statistical", "distributional"],
  "threshold": 0.8
}

// Batch Operations
POST /api/v1/batch/create
{
  "jobs": [...],
  "executionMode": "parallel"
}

GET /api/v1/batch/{id}/status
```

### WebSocket Events

```typescript
// Real-time Updates
interface WebSocketMessage {
  type: 'generation_progress' | 'job_completed' | 'quality_alert';
  payload: {
    jobId: string;
    progress?: number;
    status?: 'running' | 'completed' | 'failed';
    data?: any;
  };
}
```

## Component Architecture

### Core Components

```
src/
   components/
      DataGeneration/
         GeneratorConfig.tsx
         SensorSelector.tsx
         ParameterForm.tsx
      Visualization/
         TimeSeriesChart.tsx
         StatisticsPanel.tsx
         QualityScorecard.tsx
      BatchProcessing/
         JobQueue.tsx
         JobMonitor.tsx
         BatchResults.tsx
      Export/
          FormatSelector.tsx
          ExportPreview.tsx
          DownloadManager.tsx
   hooks/
      useDataGeneration.ts
      useQualityValidation.ts
      useBatchProcessing.ts
   services/
      api.ts
      websocket.ts
      export.ts
   types/
      sensor.ts
      generator.ts
      quality.ts
   utils/
       chartHelpers.ts
       dataTransform.ts
       validation.ts
```

### State Management

```typescript
// Global State Store
interface AppState {
  // Data Generation
  generators: Generator[];
  sensorTypes: SensorType[];
  generationJobs: GenerationJob[];
  
  // Visualization
  chartData: ChartData[];
  selectedMetrics: string[];
  chartConfig: ChartConfig;
  
  // Quality
  validationResults: ValidationResult[];
  qualityThresholds: QualityThresholds;
  
  // Batch Processing
  jobQueue: BatchJob[];
  batchResults: BatchResult[];
  
  // UI State
  activeTab: string;
  isLoading: boolean;
  error: string | null;
}
```

## Customization

### Themes and Styling

```typescript
// Custom Theme Configuration
export const customTheme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
  },
});
```

### Chart Customization

```typescript
// Chart Configuration
export const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top' as const,
    },
    title: {
      display: true,
      text: 'Synthetic IoT Data Visualization',
    },
  },
  scales: {
    x: {
      type: 'time',
      time: {
        displayFormats: {
          hour: 'HH:mm',
          day: 'MMM DD',
        },
      },
    },
    y: {
      beginAtZero: false,
    },
  },
};
```

## Deployment

### Development Environment

```bash
# Start development server
npm run dev

# Access dashboard
open http://localhost:3000
```

### Production Build

```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview

# Deploy to static hosting
npm run deploy
```

### Docker Deployment

```dockerfile
# Multi-stage build
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment-Specific Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  tsiot-dashboard:
    build: .
    ports:
      - "3000:80"
    environment:
      - VITE_API_BASE_URL=http://tsiot-backend:8080/api
      - VITE_WS_BASE_URL=ws://tsiot-backend:8080/ws
    depends_on:
      - tsiot-backend
    networks:
      - tsiot-network

networks:
  tsiot-network:
    driver: bridge
```

## Testing

### Unit Testing

```bash
# Run unit tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode for development
npm run test:watch
```

### End-to-End Testing

```bash
# Run Cypress tests
npm run test:e2e

# Open Cypress interface
npm run cypress:open
```

### Example Test

```typescript
// DataGeneration.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { DataGeneration } from './DataGeneration';

describe('DataGeneration Component', () => {
  test('generates data with correct parameters', async () => {
    render(<DataGeneration />);
    
    // Select sensor type
    fireEvent.click(screen.getByLabelText('Sensor Type'));
    fireEvent.click(screen.getByText('Temperature'));
    
    // Configure generator
    fireEvent.click(screen.getByLabelText('Generator'));
    fireEvent.click(screen.getByText('Statistical'));
    
    // Set parameters
    fireEvent.change(screen.getByLabelText('Data Points'), {
      target: { value: '1440' }
    });
    
    // Generate data
    fireEvent.click(screen.getByText('Generate'));
    
    // Verify API call
    expect(mockApiCall).toHaveBeenCalledWith({
      generator: 'statistical',
      sensorType: 'temperature',
      count: 1440
    });
  });
});
```

## Performance Optimization

### Code Splitting

```typescript
// Lazy loading for better performance
const DataGeneration = lazy(() => import('./components/DataGeneration'));
const Visualization = lazy(() => import('./components/Visualization'));
const BatchProcessing = lazy(() => import('./components/BatchProcessing'));

// Route-based splitting
const AppRoutes = () => (
  <Routes>
    <Route path="/generate" element={
      <Suspense fallback={<Loading />}>
        <DataGeneration />
      </Suspense>
    } />
    <Route path="/visualize" element={
      <Suspense fallback={<Loading />}>
        <Visualization />
      </Suspense>
    } />
  </Routes>
);
```

### Data Virtualization

```typescript
// Large dataset handling
import { FixedSizeList as List } from 'react-window';

const VirtualizedDataTable = ({ data }) => (
  <List
    height={600}
    itemCount={data.length}
    itemSize={50}
    itemData={data}
  >
    {({ index, style, data }) => (
      <div style={style}>
        {data[index].timestamp}: {data[index].value}
      </div>
    )}
  </List>
);
```

## Troubleshooting

### Common Issues

**Connection Errors**
```bash
# Verify backend is running
curl http://localhost:8080/api/health

# Check WebSocket connection
wscat -c ws://localhost:8080/ws
```

**Memory Issues with Large Datasets**
```typescript
// Enable data pagination
const useDataPagination = (data, pageSize = 1000) => {
  const [currentPage, setCurrentPage] = useState(0);
  const paginatedData = useMemo(() => 
    data.slice(currentPage * pageSize, (currentPage + 1) * pageSize),
    [data, currentPage, pageSize]
  );
  return { paginatedData, currentPage, setCurrentPage };
};
```

**Chart Performance**
```typescript
// Optimize chart rendering
const chartOptions = {
  animation: {
    duration: 0, // Disable animations for large datasets
  },
  elements: {
    point: {
      radius: 0, // Hide points for better performance
    },
  },
  plugins: {
    decimation: {
      enabled: true,
      algorithm: 'lttb', // Largest-Triangle-Three-Buckets
    },
  },
};
```

## Support and Documentation

### Additional Resources

- [TSIoT Core Documentation](../docs/)
- [API Reference](../docs/api.md)
- [CLI Examples](../cli-examples/)
- [Jupyter Notebooks](../jupyter-notebooks/)

### Getting Help

- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Community forum for questions and ideas
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Sample implementations and use cases

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### License

This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

---

**TSIoT Web Dashboard** - Making synthetic IoT data generation accessible through an intuitive web interface.