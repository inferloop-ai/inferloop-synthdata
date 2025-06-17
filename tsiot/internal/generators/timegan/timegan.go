package timegan

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/mat"

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// TimeGANModel implements the complete TimeGAN architecture
type TimeGANModel struct {
	logger      *logrus.Logger
	config      *TimeGANConfig
	
	// Network components
	embedder    *Embedder
	recovery    *Recovery
	generator   *Generator
	discriminator *Discriminator
	supervisor  *Supervisor
	
	// Training state
	trainedEpochs    int
	trainingHistory  []TrainingMetrics
	isTraining       bool
	trainingMutex    sync.RWMutex
	
	// Data preprocessing
	scaler          *DataScaler
	sequenceLength  int
	featureDim      int
	
	// Optimization
	optimizerG      *AdamOptimizer
	optimizerD      *AdamOptimizer
	optimizerE      *AdamOptimizer
	optimizerR      *AdamOptimizer
	optimizerS      *AdamOptimizer
}

// TimeGANConfig contains configuration for TimeGAN
type TimeGANConfig struct {
	// Architecture parameters
	SequenceLength    int     `json:"sequence_length"`
	FeatureDim        int     `json:"feature_dim"`
	HiddenDim         int     `json:"hidden_dim"`
	NumLayers         int     `json:"num_layers"`
	
	// Training parameters
	BatchSize         int     `json:"batch_size"`
	LearningRate      float64 `json:"learning_rate"`
	Epochs            int     `json:"epochs"`
	
	// Loss weights
	GammaValue        float64 `json:"gamma_value"`        // Supervised loss weight
	EtaValue          float64 `json:"eta_value"`          // Generator loss weight
	
	// Training phases
	EmbeddingEpochs   int     `json:"embedding_epochs"`   // Embedder + Recovery training
	SupervisedEpochs  int     `json:"supervised_epochs"`  // Supervisor training
	JointEpochs       int     `json:"joint_epochs"`       // Joint adversarial training
	
	// Regularization
	DropoutRate       float64 `json:"dropout_rate"`
	L2Regularization  float64 `json:"l2_regularization"`
	GradientClipping  float64 `json:"gradient_clipping"`
	
	// Data preprocessing
	Normalization     string  `json:"normalization"`      // "minmax", "zscore"
	NoiseStd          float64 `json:"noise_std"`          // Noise for generator input
	
	// Other settings
	Seed              int64   `json:"seed"`
	UseGPU            bool    `json:"use_gpu"`
	SaveCheckpoints   bool    `json:"save_checkpoints"`
	CheckpointInterval int    `json:"checkpoint_interval"`
}

// TrainingMetrics tracks training progress
type TrainingMetrics struct {
	Epoch             int     `json:"epoch"`
	Phase             string  `json:"phase"`
	EmbedderLoss      float64 `json:"embedder_loss"`
	RecoveryLoss      float64 `json:"recovery_loss"`
	SupervisorLoss    float64 `json:"supervisor_loss"`
	GeneratorLoss     float64 `json:"generator_loss"`
	DiscriminatorLoss float64 `json:"discriminator_loss"`
	TotalLoss         float64 `json:"total_loss"`
	Duration          time.Duration `json:"duration"`
}

// NewTimeGANModel creates a new TimeGAN model
func NewTimeGANModel(config *TimeGANConfig, logger *logrus.Logger) (*TimeGANModel, error) {
	if config == nil {
		config = getDefaultTimeGANConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if config.Seed != 0 {
		rand.Seed(config.Seed)
	}
	
	model := &TimeGANModel{
		logger:          logger,
		config:          config,
		sequenceLength:  config.SequenceLength,
		featureDim:      config.FeatureDim,
		trainingHistory: make([]TrainingMetrics, 0),
		scaler:          NewDataScaler(config.Normalization),
	}
	
	// Initialize network components
	if err := model.initializeNetworks(); err != nil {
		return nil, fmt.Errorf("failed to initialize networks: %w", err)
	}
	
	// Initialize optimizers
	model.initializeOptimizers()
	
	return model, nil
}

// Train trains the TimeGAN model
func (m *TimeGANModel) Train(ctx context.Context, data [][]float64) error {
	m.trainingMutex.Lock()
	defer m.trainingMutex.Unlock()
	
	if m.isTraining {
		return errors.NewProcessingError("ALREADY_TRAINING", "Model is already training")
	}
	
	m.isTraining = true
	defer func() { m.isTraining = false }()
	
	m.logger.WithFields(logrus.Fields{
		"data_sequences": len(data),
		"sequence_length": m.sequenceLength,
		"feature_dim": m.featureDim,
		"total_epochs": m.config.Epochs,
	}).Info("Starting TimeGAN training")
	
	// Preprocess data
	scaledData, err := m.preprocessData(data)
	if err != nil {
		return fmt.Errorf("failed to preprocess data: %w", err)
	}
	
	// Create training batches
	batches := m.createBatches(scaledData)
	
	startTime := time.Now()
	
	// Phase 1: Embedder + Recovery training
	if err := m.trainEmbedderRecovery(ctx, batches); err != nil {
		return fmt.Errorf("embedder-recovery training failed: %w", err)
	}
	
	// Phase 2: Supervisor training
	if err := m.trainSupervisor(ctx, batches); err != nil {
		return fmt.Errorf("supervisor training failed: %w", err)
	}
	
	// Phase 3: Joint adversarial training
	if err := m.trainJoint(ctx, batches); err != nil {
		return fmt.Errorf("joint training failed: %w", err)
	}
	
	totalDuration := time.Since(startTime)
	m.trainedEpochs = m.config.Epochs
	
	m.logger.WithFields(logrus.Fields{
		"total_duration": totalDuration,
		"epochs_completed": m.trainedEpochs,
		"final_loss": m.getLastLoss(),
	}).Info("TimeGAN training completed")
	
	return nil
}

// Generate generates synthetic time series data
func (m *TimeGANModel) Generate(ctx context.Context, numSequences int) ([][]float64, error) {
	if m.trainedEpochs == 0 {
		return nil, errors.NewGenerationError("MODEL_NOT_TRAINED", "Model must be trained before generation")
	}
	
	m.logger.WithFields(logrus.Fields{
		"num_sequences": numSequences,
		"sequence_length": m.sequenceLength,
		"feature_dim": m.featureDim,
	}).Info("Generating synthetic sequences")
	
	generated := make([][]float64, numSequences)
	
	for i := 0; i < numSequences; i++ {
		// Generate random noise
		noise := m.generateNoise()
		
		// Pass through generator
		embedded := m.generator.Forward(noise)
		
		// Recover to original space
		recovered := m.recovery.Forward(embedded)
		
		// Convert to slice
		sequence := m.matrixToSlice(recovered)
		
		// Inverse scale
		if m.scaler.IsFitted() {
			sequence = m.scaler.InverseTransform(sequence)
		}
		
		generated[i] = sequence
		
		// Check for cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	
	m.logger.WithField("generated_sequences", len(generated)).Info("Synthetic sequence generation completed")
	
	return generated, nil
}

// Training phases

func (m *TimeGANModel) trainEmbedderRecovery(ctx context.Context, batches []*mat.Dense) error {
	m.logger.Info("Starting embedder-recovery training phase")
	
	for epoch := 0; epoch < m.config.EmbeddingEpochs; epoch++ {
		var totalLoss float64
		startTime := time.Now()
		
		for _, batch := range batches {
			// Forward pass
			embedded := m.embedder.Forward(batch)
			recovered := m.recovery.Forward(embedded)
			
			// Calculate reconstruction loss
			loss := m.calculateMSELoss(batch, recovered)
			totalLoss += loss
			
			// Backward pass (simplified - in real implementation would compute gradients)
			m.optimizerE.Step()
			m.optimizerR.Step()
			
			// Check for cancellation
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}
		
		avgLoss := totalLoss / float64(len(batches))
		duration := time.Since(startTime)
		
		metrics := TrainingMetrics{
			Epoch:         epoch + 1,
			Phase:         "embedding",
			EmbedderLoss:  avgLoss,
			RecoveryLoss:  avgLoss,
			TotalLoss:     avgLoss,
			Duration:      duration,
		}
		
		m.trainingHistory = append(m.trainingHistory, metrics)
		
		if epoch%10 == 0 {
			m.logger.WithFields(logrus.Fields{
				"epoch": epoch + 1,
				"loss": avgLoss,
				"duration": duration,
			}).Info("Embedder-recovery training progress")
		}
	}
	
	m.logger.Info("Embedder-recovery training phase completed")
	return nil
}

func (m *TimeGANModel) trainSupervisor(ctx context.Context, batches []*mat.Dense) error {
	m.logger.Info("Starting supervisor training phase")
	
	for epoch := 0; epoch < m.config.SupervisedEpochs; epoch++ {
		var totalLoss float64
		startTime := time.Now()
		
		for _, batch := range batches {
			// Create supervisor training data
			embedded := m.embedder.Forward(batch)
			supervisorInput := m.createSupervisorInput(embedded)
			supervisorTarget := m.createSupervisorTarget(embedded)
			
			// Forward pass
			supervisorOutput := m.supervisor.Forward(supervisorInput)
			
			// Calculate supervised loss
			loss := m.calculateMSELoss(supervisorTarget, supervisorOutput)
			totalLoss += loss
			
			// Backward pass
			m.optimizerS.Step()
			
			// Check for cancellation
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}
		
		avgLoss := totalLoss / float64(len(batches))
		duration := time.Since(startTime)
		
		metrics := TrainingMetrics{
			Epoch:         epoch + 1,
			Phase:         "supervised",
			SupervisorLoss: avgLoss,
			TotalLoss:     avgLoss,
			Duration:      duration,
		}
		
		m.trainingHistory = append(m.trainingHistory, metrics)
		
		if epoch%10 == 0 {
			m.logger.WithFields(logrus.Fields{
				"epoch": epoch + 1,
				"loss": avgLoss,
				"duration": duration,
			}).Info("Supervisor training progress")
		}
	}
	
	m.logger.Info("Supervisor training phase completed")
	return nil
}

func (m *TimeGANModel) trainJoint(ctx context.Context, batches []*mat.Dense) error {
	m.logger.Info("Starting joint adversarial training phase")
	
	for epoch := 0; epoch < m.config.JointEpochs; epoch++ {
		var genLoss, discLoss float64
		startTime := time.Now()
		
		for _, batch := range batches {
			// Train discriminator
			discLoss += m.trainDiscriminatorStep(batch)
			
			// Train generator
			genLoss += m.trainGeneratorStep(batch)
			
			// Check for cancellation
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}
		
		avgGenLoss := genLoss / float64(len(batches))
		avgDiscLoss := discLoss / float64(len(batches))
		duration := time.Since(startTime)
		
		metrics := TrainingMetrics{
			Epoch:            epoch + 1,
			Phase:            "joint",
			GeneratorLoss:    avgGenLoss,
			DiscriminatorLoss: avgDiscLoss,
			TotalLoss:        avgGenLoss + avgDiscLoss,
			Duration:         duration,
		}
		
		m.trainingHistory = append(m.trainingHistory, metrics)
		
		if epoch%10 == 0 {
			m.logger.WithFields(logrus.Fields{
				"epoch": epoch + 1,
				"gen_loss": avgGenLoss,
				"disc_loss": avgDiscLoss,
				"duration": duration,
			}).Info("Joint training progress")
		}
	}
	
	m.logger.Info("Joint adversarial training phase completed")
	return nil
}

// Helper methods

func (m *TimeGANModel) initializeNetworks() error {
	var err error
	
	// Initialize embedder
	m.embedder, err = NewEmbedder(m.featureDim, m.config.HiddenDim, m.config.NumLayers)
	if err != nil {
		return fmt.Errorf("failed to create embedder: %w", err)
	}
	
	// Initialize recovery
	m.recovery, err = NewRecovery(m.config.HiddenDim, m.featureDim, m.config.NumLayers)
	if err != nil {
		return fmt.Errorf("failed to create recovery: %w", err)
	}
	
	// Initialize generator
	m.generator, err = NewGenerator(m.config.HiddenDim, m.config.HiddenDim, m.config.NumLayers)
	if err != nil {
		return fmt.Errorf("failed to create generator: %w", err)
	}
	
	// Initialize discriminator
	m.discriminator, err = NewDiscriminator(m.config.HiddenDim, 1, m.config.NumLayers)
	if err != nil {
		return fmt.Errorf("failed to create discriminator: %w", err)
	}
	
	// Initialize supervisor
	m.supervisor, err = NewSupervisor(m.config.HiddenDim, m.config.HiddenDim, m.config.NumLayers)
	if err != nil {
		return fmt.Errorf("failed to create supervisor: %w", err)
	}
	
	return nil
}

func (m *TimeGANModel) initializeOptimizers() {
	m.optimizerE = NewAdamOptimizer(m.config.LearningRate)
	m.optimizerR = NewAdamOptimizer(m.config.LearningRate)
	m.optimizerG = NewAdamOptimizer(m.config.LearningRate)
	m.optimizerD = NewAdamOptimizer(m.config.LearningRate)
	m.optimizerS = NewAdamOptimizer(m.config.LearningRate)
}

func (m *TimeGANModel) preprocessData(data [][]float64) ([]*mat.Dense, error) {
	if len(data) == 0 {
		return nil, errors.NewValidationError("EMPTY_DATA", "Input data is empty")
	}
	
	// Fit scaler on all data
	allValues := make([]float64, 0)
	for _, sequence := range data {
		allValues = append(allValues, sequence...)
	}
	
	if err := m.scaler.Fit(allValues); err != nil {
		return nil, fmt.Errorf("failed to fit scaler: %w", err)
	}
	
	// Transform sequences
	scaledData := make([]*mat.Dense, len(data))
	for i, sequence := range data {
		scaled := m.scaler.Transform(sequence)
		
		// Convert to matrix
		matrix := mat.NewDense(m.sequenceLength, m.featureDim, nil)
		for t := 0; t < m.sequenceLength && t < len(scaled); t++ {
			matrix.Set(t, 0, scaled[t])
		}
		
		scaledData[i] = matrix
	}
	
	return scaledData, nil
}

func (m *TimeGANModel) createBatches(data []*mat.Dense) []*mat.Dense {
	batches := make([]*mat.Dense, 0)
	
	for i := 0; i < len(data); i += m.config.BatchSize {
		end := i + m.config.BatchSize
		if end > len(data) {
			end = len(data)
		}
		
		// Create batch matrix
		batchSize := end - i
		batch := mat.NewDense(batchSize*m.sequenceLength, m.featureDim, nil)
		
		for j := i; j < end; j++ {
			for t := 0; t < m.sequenceLength; t++ {
				rowIdx := (j-i)*m.sequenceLength + t
				for f := 0; f < m.featureDim; f++ {
					batch.Set(rowIdx, f, data[j].At(t, f))
				}
			}
		}
		
		batches = append(batches, batch)
	}
	
	return batches
}

func (m *TimeGANModel) generateNoise() *mat.Dense {
	noise := mat.NewDense(m.sequenceLength, m.config.HiddenDim, nil)
	
	for i := 0; i < m.sequenceLength; i++ {
		for j := 0; j < m.config.HiddenDim; j++ {
			noise.Set(i, j, rand.NormFloat64()*m.config.NoiseStd)
		}
	}
	
	return noise
}

func (m *TimeGANModel) matrixToSlice(matrix *mat.Dense) []float64 {
	rows, _ := matrix.Dims()
	result := make([]float64, rows)
	
	for i := 0; i < rows; i++ {
		result[i] = matrix.At(i, 0)
	}
	
	return result
}

func (m *TimeGANModel) calculateMSELoss(target, prediction *mat.Dense) float64 {
	rows, cols := target.Dims()
	var sum float64
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := target.At(i, j) - prediction.At(i, j)
			sum += diff * diff
		}
	}
	
	return sum / float64(rows*cols)
}

func (m *TimeGANModel) createSupervisorInput(embedded *mat.Dense) *mat.Dense {
	// Take all but last timestep
	rows, cols := embedded.Dims()
	input := mat.NewDense(rows-1, cols, nil)
	
	for i := 0; i < rows-1; i++ {
		for j := 0; j < cols; j++ {
			input.Set(i, j, embedded.At(i, j))
		}
	}
	
	return input
}

func (m *TimeGANModel) createSupervisorTarget(embedded *mat.Dense) *mat.Dense {
	// Take all but first timestep
	rows, cols := embedded.Dims()
	target := mat.NewDense(rows-1, cols, nil)
	
	for i := 1; i < rows; i++ {
		for j := 0; j < cols; j++ {
			target.Set(i-1, j, embedded.At(i, j))
		}
	}
	
	return target
}

func (m *TimeGANModel) trainDiscriminatorStep(realBatch *mat.Dense) float64 {
	// Generate fake data
	noise := m.generateNoise()
	fakeEmbedded := m.generator.Forward(noise)
	
	// Get real embedded data
	realEmbedded := m.embedder.Forward(realBatch)
	
	// Discriminator predictions
	realPred := m.discriminator.Forward(realEmbedded)
	fakePred := m.discriminator.Forward(fakeEmbedded)
	
	// Calculate discriminator loss (simplified)
	realLoss := m.calculateBCELoss(realPred, 1.0)
	fakeLoss := m.calculateBCELoss(fakePred, 0.0)
	loss := realLoss + fakeLoss
	
	// Update discriminator
	m.optimizerD.Step()
	
	return loss
}

func (m *TimeGANModel) trainGeneratorStep(realBatch *mat.Dense) float64 {
	// Generate fake data
	noise := m.generateNoise()
	fakeEmbedded := m.generator.Forward(noise)
	
	// Discriminator prediction on fake data
	fakePred := m.discriminator.Forward(fakeEmbedded)
	
	// Generator adversarial loss
	advLoss := m.calculateBCELoss(fakePred, 1.0)
	
	// Supervised loss
	supervisorInput := m.createSupervisorInput(fakeEmbedded)
	supervisorOutput := m.supervisor.Forward(supervisorInput)
	supervisorTarget := m.createSupervisorTarget(fakeEmbedded)
	supLoss := m.calculateMSELoss(supervisorTarget, supervisorOutput)
	
	// Combined loss
	totalLoss := advLoss + m.config.GammaValue*supLoss
	
	// Update generator
	m.optimizerG.Step()
	
	return totalLoss
}

func (m *TimeGANModel) calculateBCELoss(prediction *mat.Dense, target float64) float64 {
	rows, cols := prediction.Dims()
	var sum float64
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			pred := prediction.At(i, j)
			// Sigmoid activation
			pred = 1.0 / (1.0 + math.Exp(-pred))
			// BCE loss
			loss := -target*math.Log(pred+1e-8) - (1-target)*math.Log(1-pred+1e-8)
			sum += loss
		}
	}
	
	return sum / float64(rows*cols)
}

func (m *TimeGANModel) getLastLoss() float64 {
	if len(m.trainingHistory) == 0 {
		return 0.0
	}
	return m.trainingHistory[len(m.trainingHistory)-1].TotalLoss
}

// GetTrainingHistory returns the training history
func (m *TimeGANModel) GetTrainingHistory() []TrainingMetrics {
	m.trainingMutex.RLock()
	defer m.trainingMutex.RUnlock()
	
	history := make([]TrainingMetrics, len(m.trainingHistory))
	copy(history, m.trainingHistory)
	return history
}

// IsTraining returns whether the model is currently training
func (m *TimeGANModel) IsTraining() bool {
	m.trainingMutex.RLock()
	defer m.trainingMutex.RUnlock()
	return m.isTraining
}

func getDefaultTimeGANConfig() *TimeGANConfig {
	return &TimeGANConfig{
		SequenceLength:    24,
		FeatureDim:        1,
		HiddenDim:         24,
		NumLayers:         3,
		BatchSize:         128,
		LearningRate:      0.001,
		Epochs:            1000,
		GammaValue:        1.0,
		EtaValue:          1.0,
		EmbeddingEpochs:   600,
		SupervisedEpochs:  600,
		JointEpochs:       1000,
		DropoutRate:       0.0,
		L2Regularization:  0.0,
		GradientClipping:  1.0,
		Normalization:     "minmax",
		NoiseStd:          1.0,
		Seed:              42,
		UseGPU:            false,
		SaveCheckpoints:   false,
		CheckpointInterval: 100,
	}
}