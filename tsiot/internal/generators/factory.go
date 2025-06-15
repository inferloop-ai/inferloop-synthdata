package generators

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/internal/generators/arima"
	"github.com/inferloop/tsiot/internal/generators/rnn"
	"github.com/inferloop/tsiot/internal/generators/ydata"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// Factory implements the GeneratorFactory interface
type Factory struct {
	creators map[models.GeneratorType]interfaces.GeneratorCreateFunc
	mu       sync.RWMutex
	logger   *logrus.Logger
}

// NewGeneratorFactory creates a new generator factory with default logger
func NewGeneratorFactory() *Factory {
	return NewFactory(nil)
}

// NewFactory creates a new generator factory
func NewFactory(logger *logrus.Logger) *Factory {
	if logger == nil {
		logger = logrus.New()
	}

	factory := &Factory{
		creators: make(map[models.GeneratorType]interfaces.GeneratorCreateFunc),
		logger:   logger,
	}

	// Register default generator types
	factory.registerDefaults()

	return factory
}

// CreateGenerator creates a new generator instance
func (f *Factory) CreateGenerator(generatorType models.GeneratorType) (interfaces.Generator, error) {
	f.mu.RLock()
	createFunc, exists := f.creators[generatorType]
	f.mu.RUnlock()

	if !exists {
		return nil, errors.NewGenerationError("UNSUPPORTED_TYPE", fmt.Sprintf("Generator type '%s' is not supported", generatorType))
	}

	generator := createFunc()
	if generator == nil {
		return nil, errors.NewGenerationError("CREATION_FAILED", fmt.Sprintf("Failed to create %s generator", generatorType))
	}

	f.logger.WithFields(logrus.Fields{
		"generator_type": generatorType,
	}).Info("Created generator instance")

	return generator, nil
}

// GetAvailableGenerators returns all available generator types
func (f *Factory) GetAvailableGenerators() []models.GeneratorType {
	f.mu.RLock()
	defer f.mu.RUnlock()

	types := make([]models.GeneratorType, 0, len(f.creators))
	for generatorType := range f.creators {
		types = append(types, generatorType)
	}

	return types
}

// RegisterGenerator registers a new generator type
func (f *Factory) RegisterGenerator(generatorType models.GeneratorType, createFunc interfaces.GeneratorCreateFunc) error {
	if generatorType == "" {
		return errors.NewValidationError("INVALID_TYPE", "Generator type cannot be empty")
	}

	if createFunc == nil {
		return errors.NewValidationError("INVALID_CREATOR", "Generator create function cannot be nil")
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	f.creators[generatorType] = createFunc

	f.logger.WithFields(logrus.Fields{
		"generator_type": generatorType,
	}).Info("Registered generator type")

	return nil
}

// IsSupported checks if a generator type is supported
func (f *Factory) IsSupported(generatorType models.GeneratorType) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()

	_, exists := f.creators[generatorType]
	return exists
}

// registerDefaults registers the default generator implementations
func (f *Factory) registerDefaults() {
	// Register Statistical generator
	f.RegisterGenerator(models.GeneratorType(constants.GeneratorTypeStatistical), func() interfaces.Generator {
		return NewStatisticalGenerator(nil, f.logger)
	})

	// Register TimeGAN generator
	f.RegisterGenerator(models.GeneratorType(constants.GeneratorTypeTimeGAN), func() interfaces.Generator {
		return NewTimeGANGenerator(nil, f.logger)
	})

	// Register ARIMA generator
	f.RegisterGenerator(models.GeneratorType(constants.GeneratorTypeARIMA), func() interfaces.Generator {
		return arima.NewARIMAGenerator(nil, f.logger)
	})

	// Register RNN generator
	f.RegisterGenerator(models.GeneratorType(constants.GeneratorTypeRNN), func() interfaces.Generator {
		return rnn.NewRNNGenerator(nil, f.logger)
	})

	// Register LSTM generator
	f.RegisterGenerator(models.GeneratorType(constants.GeneratorTypeLSTM), func() interfaces.Generator {
		return rnn.NewLSTMGenerator(nil, f.logger)
	})

	// Register GRU generator
	f.RegisterGenerator(models.GeneratorType(constants.GeneratorTypeGRU), func() interfaces.Generator {
		return rnn.NewGRUGenerator(nil, f.logger)
	})

	// Register YData generator
	f.RegisterGenerator(models.GeneratorType(constants.GeneratorTypeYData), func() interfaces.Generator {
		return ydata.NewYDataGenerator(nil, f.logger)
	})
}

// Registry implements the GeneratorRegistry interface
type Registry struct {
	generators map[models.GeneratorType]interfaces.Generator
	mu         sync.RWMutex
	logger     *logrus.Logger
}

// NewRegistry creates a new generator registry
func NewRegistry(logger *logrus.Logger) *Registry {
	if logger == nil {
		logger = logrus.New()
	}

	return &Registry{
		generators: make(map[models.GeneratorType]interfaces.Generator),
		logger:     logger,
	}
}

// Register registers a generator
func (r *Registry) Register(generator interfaces.Generator) error {
	if generator == nil {
		return errors.NewValidationError("INVALID_GENERATOR", "Generator cannot be nil")
	}

	generatorType := generator.GetType()

	r.mu.Lock()
	defer r.mu.Unlock()

	r.generators[generatorType] = generator

	r.logger.WithFields(logrus.Fields{
		"generator_type": generatorType,
		"generator_name": generator.GetName(),
	}).Info("Registered generator")

	return nil
}

// Get retrieves a generator by type
func (r *Registry) Get(generatorType models.GeneratorType) (interfaces.Generator, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	generator, exists := r.generators[generatorType]
	if !exists {
		return nil, errors.NewGenerationError("GENERATOR_NOT_FOUND", fmt.Sprintf("Generator type '%s' not found in registry", generatorType))
	}

	return generator, nil
}

// List returns all registered generators
func (r *Registry) List() []interfaces.Generator {
	r.mu.RLock()
	defer r.mu.RUnlock()

	generators := make([]interfaces.Generator, 0, len(r.generators))
	for _, generator := range r.generators {
		generators = append(generators, generator)
	}

	return generators
}

// Remove removes a generator from the registry
func (r *Registry) Remove(generatorType models.GeneratorType) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	generator, exists := r.generators[generatorType]
	if !exists {
		return errors.NewGenerationError("GENERATOR_NOT_FOUND", fmt.Sprintf("Generator type '%s' not found in registry", generatorType))
	}

	// Close the generator if it supports it
	if err := generator.Close(); err != nil {
		r.logger.WithFields(logrus.Fields{
			"generator_type": generatorType,
			"error":          err.Error(),
		}).Warn("Error closing generator during removal")
	}

	delete(r.generators, generatorType)

	r.logger.WithFields(logrus.Fields{
		"generator_type": generatorType,
	}).Info("Removed generator from registry")

	return nil
}

// Count returns the number of registered generators
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return len(r.generators)
}

// Pool implements the GeneratorPool interface
type Pool struct {
	factory     *Factory
	pools       map[models.GeneratorType]*generatorPool
	mu          sync.RWMutex
	logger      *logrus.Logger
	maxSize     int
	maxIdleTime time.Duration
}

type generatorPool struct {
	generatorType models.GeneratorType
	instances     []pooledGenerator
	mu            sync.Mutex
	factory       *Factory
}

type pooledGenerator struct {
	generator interfaces.Generator
	lastUsed  time.Time
	inUse     bool
}

// NewPool creates a new generator pool
func NewPool(factory *Factory, maxSize int, maxIdleTime time.Duration, logger *logrus.Logger) *Pool {
	if logger == nil {
		logger = logrus.New()
	}

	if maxSize <= 0 {
		maxSize = 5
	}

	if maxIdleTime <= 0 {
		maxIdleTime = 30 * time.Minute
	}

	pool := &Pool{
		factory:     factory,
		pools:       make(map[models.GeneratorType]*generatorPool),
		logger:      logger,
		maxSize:     maxSize,
		maxIdleTime: maxIdleTime,
	}

	// Start cleanup goroutine
	go pool.cleanup()

	return pool
}

// Get gets a generator from the pool
func (p *Pool) Get(ctx context.Context, generatorType models.GeneratorType) (interfaces.Generator, error) {
	p.mu.Lock()
	pool, exists := p.pools[generatorType]
	if !exists {
		pool = &generatorPool{
			generatorType: generatorType,
			instances:     make([]pooledGenerator, 0),
			factory:       p.factory,
		}
		p.pools[generatorType] = pool
	}
	p.mu.Unlock()

	return pool.get(ctx)
}

// Put returns a generator to the pool
func (p *Pool) Put(generator interfaces.Generator) error {
	if generator == nil {
		return errors.NewValidationError("INVALID_GENERATOR", "Generator cannot be nil")
	}

	generatorType := generator.GetType()

	p.mu.RLock()
	pool, exists := p.pools[generatorType]
	p.mu.RUnlock()

	if !exists {
		// If no pool exists, just close the generator
		return generator.Close()
	}

	return pool.put(generator)
}

// Close closes the pool and all generators
func (p *Pool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, pool := range p.pools {
		pool.close()
	}

	p.pools = make(map[models.GeneratorType]*generatorPool)

	p.logger.Info("Generator pool closed")
	return nil
}

// Stats returns pool statistics
func (p *Pool) Stats() interfaces.PoolStats {
	p.mu.RLock()
	defer p.mu.RUnlock()

	stats := interfaces.PoolStats{
		ActiveConnections: 0,
		IdleConnections:   0,
		TotalCreated:      0,
		TotalReused:       0,
	}

	for _, pool := range p.pools {
		pool.mu.Lock()
		for _, instance := range pool.instances {
			if instance.inUse {
				stats.ActiveConnections++
			} else {
				stats.IdleConnections++
			}
		}
		pool.mu.Unlock()
	}

	return stats
}

// get gets a generator instance from the specific pool
func (gp *generatorPool) get(ctx context.Context) (interfaces.Generator, error) {
	gp.mu.Lock()
	defer gp.mu.Unlock()

	// Look for an available instance
	for i := range gp.instances {
		if !gp.instances[i].inUse {
			gp.instances[i].inUse = true
			gp.instances[i].lastUsed = time.Now()
			return gp.instances[i].generator, nil
		}
	}

	// If no available instance and under max size, create new one
	if len(gp.instances) < 5 { // maxSize per pool
		generator, err := gp.factory.CreateGenerator(gp.generatorType)
		if err != nil {
			return nil, err
		}

		// Add to pool
		gp.instances = append(gp.instances, pooledGenerator{
			generator: generator,
			lastUsed:  time.Now(),
			inUse:     true,
		})

		return generator, nil
	}

	// If pool is full, create a new instance without pooling
	return gp.factory.CreateGenerator(gp.generatorType)
}

// put returns a generator to the pool
func (gp *generatorPool) put(generator interfaces.Generator) error {
	gp.mu.Lock()
	defer gp.mu.Unlock()

	// Find the generator in the pool and mark it as not in use
	for i := range gp.instances {
		if gp.instances[i].generator == generator {
			gp.instances[i].inUse = false
			gp.instances[i].lastUsed = time.Now()
			return nil
		}
	}

	// If not found in pool, just close it
	return generator.Close()
}

// close closes all instances in the pool
func (gp *generatorPool) close() {
	gp.mu.Lock()
	defer gp.mu.Unlock()

	for _, instance := range gp.instances {
		instance.generator.Close()
	}

	gp.instances = gp.instances[:0]
}

// cleanup periodically cleans up idle generators
func (p *Pool) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		p.cleanupIdleGenerators()
	}
}

// cleanupIdleGenerators removes idle generators that have exceeded maxIdleTime
func (p *Pool) cleanupIdleGenerators() {
	p.mu.RLock()
	pools := make([]*generatorPool, 0, len(p.pools))
	for _, pool := range p.pools {
		pools = append(pools, pool)
	}
	p.mu.RUnlock()

	for _, pool := range pools {
		pool.mu.Lock()
		
		activeInstances := make([]pooledGenerator, 0, len(pool.instances))
		for _, instance := range pool.instances {
			if instance.inUse || time.Since(instance.lastUsed) < p.maxIdleTime {
				activeInstances = append(activeInstances, instance)
			} else {
				// Close idle instance
				instance.generator.Close()
				p.logger.Debug("Closed idle generator")
			}
		}
		
		pool.instances = activeInstances
		pool.mu.Unlock()
	}
}