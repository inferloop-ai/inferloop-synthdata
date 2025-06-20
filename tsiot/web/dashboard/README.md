# TSIoT Web Dashboard

A modern, responsive web dashboard for the Time Series Synthetic IoT data generation platform. Built with React and TypeScript, providing an intuitive interface for managing synthetic data generation, validation, and monitoring.

## Features

### Core Functionality
- **Data Generation Management**: Configure and monitor synthetic data generation jobs
- **Real-time Monitoring**: Live dashboards with metrics and system health
- **Validation Results**: Interactive visualization of data quality metrics
- **Agent Management**: Control and monitor generation agents
- **Storage Management**: View and manage data across different storage backends

### Key Components
- **Dashboard Overview**: System health, active jobs, and key metrics
- **Generation Studio**: Visual interface for configuring data generators
- **Validation Center**: Interactive data quality assessment tools
- **Agent Console**: Real-time agent monitoring and control
- **Settings Panel**: System configuration and user preferences

## Technology Stack

- **Frontend Framework**: React 18 with TypeScript
- **State Management**: Redux Toolkit with RTK Query
- **UI Components**: Material-UI (MUI) v5
- **Charts & Visualization**: Chart.js with react-chartjs-2
- **Real-time Communication**: Socket.IO client
- **Data Tables**: React Table v8
- **Forms**: React Hook Form with Yup validation
- **Routing**: React Router v6
- **Styling**: Emotion/styled with MUI theme system
- **Testing**: Jest, React Testing Library, Cypress
- **Build Tool**: Vite
- **Code Quality**: ESLint, Prettier, Husky

## Quick Start

### Prerequisites
- Node.js 18+ and npm/yarn
- TSIoT backend server running

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

# Run E2E tests
npm run test:e2e
```

### Environment Configuration
Create a `.env.local` file:
```env
REACT_APP_API_BASE_URL=http://localhost:8080/api/v1
REACT_APP_WS_URL=ws://localhost:8080/ws
REACT_APP_MQTT_BROKER_URL=ws://localhost:9001
REACT_APP_ENABLE_DEBUG=true
```

## Project Structure

```
web/dashboard/
   public/                 # Static assets
   src/
      components/        # Reusable UI components
         common/        # Generic components
         charts/        # Chart components
         forms/         # Form components
         layout/        # Layout components
      pages/             # Page components
         Dashboard/     # Main dashboard
         Generation/    # Data generation interface
         Validation/    # Validation results
         Agents/        # Agent management
         Settings/      # Configuration
      store/             # Redux store and slices
      services/          # API services and utilities
      hooks/             # Custom React hooks
      utils/             # Utility functions
      types/             # TypeScript type definitions
      constants/         # Application constants
      assets/            # Images, icons, etc.
   cypress/               # E2E tests
   tests/                 # Unit and integration tests
```

## Key Features

### Dashboard Overview
- System health indicators and alerts
- Active generation jobs with progress tracking
- Resource utilization charts (CPU, memory, storage)
- Recent activity timeline
- Quick action buttons for common tasks

### Generation Studio
- Visual drag-and-drop interface for configuring generators
- Template library with pre-built configurations
- Real-time preview of generated data patterns
- Parameter tuning with live feedback
- Batch job scheduling and management

### Validation Center
- Interactive data quality scorecards
- Statistical test results with visualizations
- Privacy compliance dashboards
- Comparative analysis tools
- Custom validation rule builder

### Agent Management
- Real-time agent status monitoring
- Performance metrics and health checks
- Agent deployment and scaling controls
- Log streaming and debugging tools
- Configuration management interface

## API Integration

The dashboard integrates with the TSIoT backend through:

- **REST API**: Standard CRUD operations and configuration
- **WebSocket**: Real-time updates and notifications
- **MQTT**: Live data streaming for monitoring
- **gRPC-Web**: High-performance data operations

### Key API Endpoints
- `/api/v1/generate` - Data generation management
- `/api/v1/validate` - Validation operations
- `/api/v1/agents` - Agent management
- `/api/v1/timeseries` - Time series data operations
- `/api/v1/metrics` - System metrics and monitoring

## Development

### Code Style
- TypeScript strict mode enabled
- ESLint with Airbnb configuration
- Prettier for code formatting
- Husky for pre-commit hooks

### Testing Strategy
- Unit tests for utilities and hooks
- Component tests with React Testing Library
- Integration tests for API interactions
- E2E tests with Cypress for critical flows

### Performance Optimization
- Code splitting with React.lazy
- Virtual scrolling for large datasets
- Debounced API calls
- Memoization for expensive computations
- Progressive loading strategies

## Deployment

### Docker Deployment
```bash
# Build production image
docker build -t tsiot-dashboard .

# Run container
docker run -p 3000:80 tsiot-dashboard
```

### Kubernetes Deployment
See `deployments/kubernetes/` for complete manifests.

### Environment Variables
- `REACT_APP_API_BASE_URL`: Backend API URL
- `REACT_APP_WS_URL`: WebSocket server URL
- `REACT_APP_ENABLE_DEBUG`: Enable debug features
- `REACT_APP_THEME_MODE`: Default theme (light/dark)

## Contributing

1. Follow the established code style and patterns
2. Write tests for new features
3. Update documentation for API changes
4. Use conventional commit messages
5. Ensure all tests pass before submitting PRs

## Troubleshooting

### Common Issues

**Development Server Won't Start**
- Check Node.js version (18+ required)
- Clear node_modules and reinstall dependencies
- Verify environment variables

**API Connection Issues**
- Confirm backend server is running
- Check CORS configuration
- Verify API endpoints and authentication

**Build Failures**
- Check TypeScript errors
- Verify all dependencies are installed
- Clear build cache with `npm run clean`

### Debug Mode
Enable debug mode with `REACT_APP_ENABLE_DEBUG=true` for:
- Detailed API request/response logging
- Redux state inspection tools
- Performance profiling information
- Additional development utilities

## License

Licensed under the same terms as the main TSIoT project.