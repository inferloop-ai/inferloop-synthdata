package encoding

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"fmt"
	"io"

	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/chacha20poly1305"
	"golang.org/x/crypto/hkdf"
)

// EncryptionAlgorithm represents different encryption algorithms
type EncryptionAlgorithm int

const (
	AES256GCM EncryptionAlgorithm = iota
	ChaCha20Poly1305
)

// KeyDerivationFunction represents different key derivation functions
type KeyDerivationFunction int

const (
	HKDF_SHA256 KeyDerivationFunction = iota
	Argon2ID
)

// EncryptionConfig holds encryption configuration
type EncryptionConfig struct {
	Algorithm EncryptionAlgorithm
	KeySize   int
	NonceSize int
}

// Encryptor interface for different encryption implementations
type Encryptor interface {
	Encrypt(plaintext []byte, key []byte) ([]byte, error)
	Decrypt(ciphertext []byte, key []byte) ([]byte, error)
	GenerateKey() ([]byte, error)
	KeySize() int
	NonceSize() int
}

// AESGCMEncryptor implements AES-256-GCM encryption
type AESGCMEncryptor struct {
	keySize   int
	nonceSize int
}

// NewAESGCMEncryptor creates a new AES-GCM encryptor
func NewAESGCMEncryptor() *AESGCMEncryptor {
	return &AESGCMEncryptor{
		keySize:   32, // 256 bits
		nonceSize: 12, // 96 bits for GCM
	}
}

// Encrypt encrypts plaintext using AES-256-GCM
func (a *AESGCMEncryptor) Encrypt(plaintext []byte, key []byte) ([]byte, error) {
	if len(key) != a.keySize {
		return nil, fmt.Errorf("invalid key size: expected %d, got %d", a.keySize, len(key))
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	nonce := make([]byte, a.nonceSize)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	ciphertext := aead.Seal(nonce, nonce, plaintext, nil)
	return ciphertext, nil
}

// Decrypt decrypts ciphertext using AES-256-GCM
func (a *AESGCMEncryptor) Decrypt(ciphertext []byte, key []byte) ([]byte, error) {
	if len(key) != a.keySize {
		return nil, fmt.Errorf("invalid key size: expected %d, got %d", a.keySize, len(key))
	}

	if len(ciphertext) < a.nonceSize {
		return nil, errors.New("ciphertext too short")
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	nonce := ciphertext[:a.nonceSize]
	ciphertext = ciphertext[a.nonceSize:]

	plaintext, err := aead.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt: %w", err)
	}

	return plaintext, nil
}

// GenerateKey generates a random key for AES-256
func (a *AESGCMEncryptor) GenerateKey() ([]byte, error) {
	key := make([]byte, a.keySize)
	if _, err := io.ReadFull(rand.Reader, key); err != nil {
		return nil, fmt.Errorf("failed to generate key: %w", err)
	}
	return key, nil
}

// KeySize returns the key size in bytes
func (a *AESGCMEncryptor) KeySize() int {
	return a.keySize
}

// NonceSize returns the nonce size in bytes
func (a *AESGCMEncryptor) NonceSize() int {
	return a.nonceSize
}

// ChaCha20Poly1305Encryptor implements ChaCha20-Poly1305 encryption
type ChaCha20Poly1305Encryptor struct {
	keySize   int
	nonceSize int
}

// NewChaCha20Poly1305Encryptor creates a new ChaCha20-Poly1305 encryptor
func NewChaCha20Poly1305Encryptor() *ChaCha20Poly1305Encryptor {
	return &ChaCha20Poly1305Encryptor{
		keySize:   32, // 256 bits
		nonceSize: 12, // 96 bits
	}
}

// Encrypt encrypts plaintext using ChaCha20-Poly1305
func (c *ChaCha20Poly1305Encryptor) Encrypt(plaintext []byte, key []byte) ([]byte, error) {
	if len(key) != c.keySize {
		return nil, fmt.Errorf("invalid key size: expected %d, got %d", c.keySize, len(key))
	}

	aead, err := chacha20poly1305.New(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create ChaCha20-Poly1305 cipher: %w", err)
	}

	nonce := make([]byte, c.nonceSize)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	ciphertext := aead.Seal(nonce, nonce, plaintext, nil)
	return ciphertext, nil
}

// Decrypt decrypts ciphertext using ChaCha20-Poly1305
func (c *ChaCha20Poly1305Encryptor) Decrypt(ciphertext []byte, key []byte) ([]byte, error) {
	if len(key) != c.keySize {
		return nil, fmt.Errorf("invalid key size: expected %d, got %d", c.keySize, len(key))
	}

	if len(ciphertext) < c.nonceSize {
		return nil, errors.New("ciphertext too short")
	}

	aead, err := chacha20poly1305.New(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create ChaCha20-Poly1305 cipher: %w", err)
	}

	nonce := ciphertext[:c.nonceSize]
	ciphertext = ciphertext[c.nonceSize:]

	plaintext, err := aead.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt: %w", err)
	}

	return plaintext, nil
}

// GenerateKey generates a random key for ChaCha20-Poly1305
func (c *ChaCha20Poly1305Encryptor) GenerateKey() ([]byte, error) {
	key := make([]byte, c.keySize)
	if _, err := io.ReadFull(rand.Reader, key); err != nil {
		return nil, fmt.Errorf("failed to generate key: %w", err)
	}
	return key, nil
}

// KeySize returns the key size in bytes
func (c *ChaCha20Poly1305Encryptor) KeySize() int {
	return c.keySize
}

// NonceSize returns the nonce size in bytes
func (c *ChaCha20Poly1305Encryptor) NonceSize() int {
	return c.nonceSize
}

// EncryptionFactory creates encryptors based on algorithm
type EncryptionFactory struct{}

// NewEncryptionFactory creates a new encryption factory
func NewEncryptionFactory() *EncryptionFactory {
	return &EncryptionFactory{}
}

// CreateEncryptor creates an encryptor for the specified algorithm
func (ef *EncryptionFactory) CreateEncryptor(algorithm EncryptionAlgorithm) (Encryptor, error) {
	switch algorithm {
	case AES256GCM:
		return NewAESGCMEncryptor(), nil
	case ChaCha20Poly1305:
		return NewChaCha20Poly1305Encryptor(), nil
	default:
		return nil, fmt.Errorf("unsupported encryption algorithm: %d", algorithm)
	}
}

// KeyDerivation provides key derivation functionality
type KeyDerivation struct {
	function KeyDerivationFunction
	salt     []byte
}

// NewKeyDerivation creates a new key derivation instance
func NewKeyDerivation(function KeyDerivationFunction, salt []byte) *KeyDerivation {
	return &KeyDerivation{
		function: function,
		salt:     salt,
	}
}

// DeriveKey derives a key from a password
func (kd *KeyDerivation) DeriveKey(password []byte, keyLength int) ([]byte, error) {
	switch kd.function {
	case HKDF_SHA256:
		return kd.deriveHKDF(password, keyLength)
	case Argon2ID:
		return kd.deriveArgon2ID(password, keyLength)
	default:
		return nil, fmt.Errorf("unsupported key derivation function: %d", kd.function)
	}
}

// deriveHKDF derives a key using HKDF-SHA256
func (kd *KeyDerivation) deriveHKDF(password []byte, keyLength int) ([]byte, error) {
	hash := sha256.New
	hkdf := hkdf.New(hash, password, kd.salt, nil)
	
	key := make([]byte, keyLength)
	if _, err := io.ReadFull(hkdf, key); err != nil {
		return nil, fmt.Errorf("failed to derive key with HKDF: %w", err)
	}
	
	return key, nil
}

// deriveArgon2ID derives a key using Argon2id
func (kd *KeyDerivation) deriveArgon2ID(password []byte, keyLength int) ([]byte, error) {
	// Argon2id parameters (these should be tuned based on security requirements)
	time := uint32(3)
	memory := uint32(64 * 1024) // 64 MB
	threads := uint8(4)
	
	key := argon2.IDKey(password, kd.salt, time, memory, threads, uint32(keyLength))
	return key, nil
}

// GenerateSalt generates a random salt
func GenerateSalt(size int) ([]byte, error) {
	salt := make([]byte, size)
	if _, err := io.ReadFull(rand.Reader, salt); err != nil {
		return nil, fmt.Errorf("failed to generate salt: %w", err)
	}
	return salt, nil
}

// SecureRandom provides cryptographically secure random number generation
type SecureRandom struct{}

// NewSecureRandom creates a new secure random generator
func NewSecureRandom() *SecureRandom {
	return &SecureRandom{}
}

// GenerateBytes generates random bytes
func (sr *SecureRandom) GenerateBytes(size int) ([]byte, error) {
	bytes := make([]byte, size)
	if _, err := io.ReadFull(rand.Reader, bytes); err != nil {
		return nil, fmt.Errorf("failed to generate random bytes: %w", err)
	}
	return bytes, nil
}

// GenerateKey generates a random key for the specified algorithm
func (sr *SecureRandom) GenerateKey(algorithm EncryptionAlgorithm) ([]byte, error) {
	factory := NewEncryptionFactory()
	encryptor, err := factory.CreateEncryptor(algorithm)
	if err != nil {
		return nil, err
	}
	return encryptor.GenerateKey()
}

// EncryptedData represents encrypted data with metadata
type EncryptedData struct {
	Algorithm  EncryptionAlgorithm `json:"algorithm"`
	Ciphertext []byte              `json:"ciphertext"`
	KeyHash    []byte              `json:"key_hash,omitempty"`
	Metadata   map[string]string   `json:"metadata,omitempty"`
}

// NewEncryptedData creates a new encrypted data structure
func NewEncryptedData(algorithm EncryptionAlgorithm, ciphertext []byte) *EncryptedData {
	return &EncryptedData{
		Algorithm:  algorithm,
		Ciphertext: ciphertext,
		Metadata:   make(map[string]string),
	}
}

// SetKeyHash sets the key hash for verification
func (ed *EncryptedData) SetKeyHash(key []byte) {
	hash := sha256.Sum256(key)
	ed.KeyHash = hash[:]
}

// VerifyKey verifies if the provided key matches the stored hash
func (ed *EncryptedData) VerifyKey(key []byte) bool {
	if len(ed.KeyHash) == 0 {
		return true // No hash stored, assume valid
	}
	
	hash := sha256.Sum256(key)
	return subtle.ConstantTimeCompare(ed.KeyHash, hash[:]) == 1
}

// Decrypt decrypts the data using the provided key
func (ed *EncryptedData) Decrypt(key []byte) ([]byte, error) {
	if !ed.VerifyKey(key) {
		return nil, errors.New("invalid key")
	}
	
	factory := NewEncryptionFactory()
	encryptor, err := factory.CreateEncryptor(ed.Algorithm)
	if err != nil {
		return nil, err
	}
	
	return encryptor.Decrypt(ed.Ciphertext, key)
}

// EncryptionManager manages encryption operations
type EncryptionManager struct {
	defaultAlgorithm EncryptionAlgorithm
	encryptor        Encryptor
}

// NewEncryptionManager creates a new encryption manager
func NewEncryptionManager(algorithm EncryptionAlgorithm) (*EncryptionManager, error) {
	factory := NewEncryptionFactory()
	encryptor, err := factory.CreateEncryptor(algorithm)
	if err != nil {
		return nil, err
	}
	
	return &EncryptionManager{
		defaultAlgorithm: algorithm,
		encryptor:        encryptor,
	}, nil
}

// Encrypt encrypts data and returns an EncryptedData structure
func (em *EncryptionManager) Encrypt(plaintext []byte, key []byte) (*EncryptedData, error) {
	ciphertext, err := em.encryptor.Encrypt(plaintext, key)
	if err != nil {
		return nil, err
	}
	
	encData := NewEncryptedData(em.defaultAlgorithm, ciphertext)
	encData.SetKeyHash(key)
	
	return encData, nil
}

// EncryptWithPassword encrypts data using a password-derived key
func (em *EncryptionManager) EncryptWithPassword(plaintext []byte, password []byte) (*EncryptedData, error) {
	salt, err := GenerateSalt(32)
	if err != nil {
		return nil, err
	}
	
	kd := NewKeyDerivation(Argon2ID, salt)
	key, err := kd.DeriveKey(password, em.encryptor.KeySize())
	if err != nil {
		return nil, err
	}
	
	encData, err := em.Encrypt(plaintext, key)
	if err != nil {
		return nil, err
	}
	
	// Store salt in metadata for later decryption
	encData.Metadata["salt"] = string(salt)
	encData.Metadata["kdf"] = "argon2id"
	
	return encData, nil
}

// DecryptWithPassword decrypts data using a password
func (em *EncryptionManager) DecryptWithPassword(encData *EncryptedData, password []byte) ([]byte, error) {
	saltStr, exists := encData.Metadata["salt"]
	if !exists {
		return nil, errors.New("salt not found in metadata")
	}
	
	salt := []byte(saltStr)
	kd := NewKeyDerivation(Argon2ID, salt)
	key, err := kd.DeriveKey(password, em.encryptor.KeySize())
	if err != nil {
		return nil, err
	}
	
	return encData.Decrypt(key)
}

// EncryptionPool manages a pool of encryptors for performance
type EncryptionPool struct {
	aesPool     chan *AESGCMEncryptor
	chachaPool  chan *ChaCha20Poly1305Encryptor
}

// NewEncryptionPool creates a new encryption pool
func NewEncryptionPool(poolSize int) *EncryptionPool {
	return &EncryptionPool{
		aesPool:    make(chan *AESGCMEncryptor, poolSize),
		chachaPool: make(chan *ChaCha20Poly1305Encryptor, poolSize),
	}
}

// GetEncryptor gets an encryptor from the pool
func (ep *EncryptionPool) GetEncryptor(algorithm EncryptionAlgorithm) Encryptor {
	switch algorithm {
	case AES256GCM:
		select {
		case encryptor := <-ep.aesPool:
			return encryptor
		default:
			return NewAESGCMEncryptor()
		}
	case ChaCha20Poly1305:
		select {
		case encryptor := <-ep.chachaPool:
			return encryptor
		default:
			return NewChaCha20Poly1305Encryptor()
		}
	default:
		return NewAESGCMEncryptor()
	}
}

// PutEncryptor returns an encryptor to the pool
func (ep *EncryptionPool) PutEncryptor(encryptor Encryptor) {
	switch e := encryptor.(type) {
	case *AESGCMEncryptor:
		select {
		case ep.aesPool <- e:
		default:
			// Pool is full, discard
		}
	case *ChaCha20Poly1305Encryptor:
		select {
		case ep.chachaPool <- e:
		default:
			// Pool is full, discard
		}
	}
}

// SecureZero securely zeros out sensitive data
func SecureZero(data []byte) {
	for i := range data {
		data[i] = 0
	}
}

// SecureCompare performs constant-time comparison of two byte slices
func SecureCompare(a, b []byte) bool {
	return subtle.ConstantTimeCompare(a, b) == 1
}