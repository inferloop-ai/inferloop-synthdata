# TSIOT Security Architecture

## Overview

This document outlines the comprehensive security architecture for the Time Series IoT Synthetic Data (TSIOT) platform. Security is implemented at multiple layers with defense-in-depth principles, covering authentication, authorization, data protection, network security, and compliance.

## Security Principles

### 1. Zero Trust Architecture
- Never trust, always verify
- Least privilege access
- Assume breach mentality
- Continuous verification

### 2. Defense in Depth
- Multiple security layers
- Fail-safe defaults
- Comprehensive monitoring
- Rapid incident response

### 3. Data Privacy by Design
- Privacy-preserving synthetic data generation
- Differential privacy techniques
- K-anonymity and L-diversity
- Secure multi-party computation

## Security Architecture Overview

```
                                                             
                    External Layer                           
                                                      
      WAF             CDN          DDoS               
   Protection     Filtering      Protection           
                                                      
                                                             
                                                             
                   Network Layer                             
                                                      
      VPC           Network         TLS               
   Isolation       Policies      Encryption           
                                                      
                                                             
                                                             
                 Application Layer                           
                                                      
      JWT            RBAC          Input              
   Validation     Controls       Validation           
                                                      
                                                             
                                                             
                    Data Layer                               
                                                      
   Encryption       Access          Audit             
   at Rest         Logging        Logging             
                                                      
                                                             
```

## Authentication and Authorization

### 1. Multi-Factor Authentication (MFA)

#### JWT Token Implementation
```go
type JWTService struct {
    signingKey     []byte
    refreshKey     []byte
    issuer         string
    accessExpiry   time.Duration
    refreshExpiry  time.Duration
    blacklist      TokenBlacklist
}

type Claims struct {
    UserID    string   `json:"user_id"`
    Email     string   `json:"email"`
    Roles     []string `json:"roles"`
    Scopes    []string `json:"scopes"`
    SessionID string   `json:"session_id"`
    jwt.StandardClaims
}

func (js *JWTService) GenerateTokenPair(user *User) (*TokenPair, error) {
    sessionID := generateSessionID()
    
    // Access Token (short-lived)
    accessClaims := &Claims{
        UserID:    user.ID,
        Email:     user.Email,
        Roles:     user.Roles,
        Scopes:    user.Scopes,
        SessionID: sessionID,
        StandardClaims: jwt.StandardClaims{
            ExpiresAt: time.Now().Add(js.accessExpiry).Unix(),
            Issuer:    js.issuer,
            IssuedAt:  time.Now().Unix(),
            Subject:   user.ID,
        },
    }
    
    accessToken := jwt.NewWithClaims(jwt.SigningMethodHS256, accessClaims)
    accessString, err := accessToken.SignedString(js.signingKey)
    if err != nil {
        return nil, err
    }
    
    // Refresh Token (long-lived)
    refreshClaims := &Claims{
        UserID:    user.ID,
        SessionID: sessionID,
        StandardClaims: jwt.StandardClaims{
            ExpiresAt: time.Now().Add(js.refreshExpiry).Unix(),
            Issuer:    js.issuer,
            IssuedAt:  time.Now().Unix(),
            Subject:   user.ID,
        },
    }
    
    refreshToken := jwt.NewWithClaims(jwt.SigningMethodHS256, refreshClaims)
    refreshString, err := refreshToken.SignedString(js.refreshKey)
    if err != nil {
        return nil, err
    }
    
    return &TokenPair{
        AccessToken:  accessString,
        RefreshToken: refreshString,
        ExpiresIn:    int(js.accessExpiry.Seconds()),
        TokenType:    "Bearer",
    }, nil
}

func (js *JWTService) ValidateToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return js.signingKey, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    claims, ok := token.Claims.(*Claims)
    if !ok || !token.Valid {
        return nil, errors.New("invalid token")
    }
    
    // Check blacklist
    if js.blacklist.IsBlacklisted(tokenString) {
        return nil, errors.New("token is blacklisted")
    }
    
    return claims, nil
}
```

#### API Key Management
```go
type APIKeyService struct {
    storage   APIKeyStorage
    hasher    PasswordHasher
    generator SecureGenerator
}

type APIKey struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    KeyHash     string    `json:"key_hash"`
    UserID      string    `json:"user_id"`
    Scopes      []string  `json:"scopes"`
    LastUsed    time.Time `json:"last_used"`
    ExpiresAt   time.Time `json:"expires_at"`
    CreatedAt   time.Time `json:"created_at"`
    RateLimit   int       `json:"rate_limit"`
    IsActive    bool      `json:"is_active"`
}

func (aks *APIKeyService) CreateAPIKey(userID, name string, scopes []string, expiry time.Duration) (*APIKey, string, error) {
    // Generate secure API key
    keyValue, err := aks.generator.GenerateAPIKey(32)
    if err != nil {
        return nil, "", err
    }
    
    // Hash the key for storage
    keyHash, err := aks.hasher.HashPassword(keyValue)
    if err != nil {
        return nil, "", err
    }
    
    apiKey := &APIKey{
        ID:        generateID(),
        Name:      name,
        KeyHash:   keyHash,
        UserID:    userID,
        Scopes:    scopes,
        ExpiresAt: time.Now().Add(expiry),
        CreatedAt: time.Now(),
        RateLimit: 1000, // Default rate limit
        IsActive:  true,
    }
    
    if err := aks.storage.Create(apiKey); err != nil {
        return nil, "", err
    }
    
    // Return the plain key only once
    return apiKey, keyValue, nil
}

func (aks *APIKeyService) ValidateAPIKey(keyValue string) (*APIKey, error) {
    // Hash the provided key
    keyHash, err := aks.hasher.HashPassword(keyValue)
    if err != nil {
        return nil, err
    }
    
    // Find API key by hash
    apiKey, err := aks.storage.FindByHash(keyHash)
    if err != nil {
        return nil, err
    }
    
    // Validate expiry and status
    if !apiKey.IsActive {
        return nil, errors.New("API key is inactive")
    }
    
    if time.Now().After(apiKey.ExpiresAt) {
        return nil, errors.New("API key has expired")
    }
    
    // Update last used
    apiKey.LastUsed = time.Now()
    aks.storage.Update(apiKey)
    
    return apiKey, nil
}
```

### 2. Role-Based Access Control (RBAC)

#### Permission System
```go
type Permission string

const (
    PermissionGenerateData    Permission = "generate:data"
    PermissionValidateData    Permission = "validate:data"
    PermissionAnalyzeData     Permission = "analyze:data"
    PermissionViewData        Permission = "view:data"
    PermissionDeleteData      Permission = "delete:data"
    PermissionManageUsers     Permission = "manage:users"
    PermissionManageAPIKeys   Permission = "manage:api_keys"
    PermissionViewMetrics     Permission = "view:metrics"
    PermissionManageSystem    Permission = "manage:system"
)

type Role struct {
    Name        string       `json:"name"`
    Description string       `json:"description"`
    Permissions []Permission `json:"permissions"`
}

type RBACService struct {
    roles       map[string]*Role
    userRoles   map[string][]string
    roleCache   *cache.Cache
}

func (rbac *RBACService) InitializeDefaultRoles() {
    rbac.roles = map[string]*Role{
        "admin": {
            Name:        "Administrator",
            Description: "Full system access",
            Permissions: []Permission{
                PermissionGenerateData,
                PermissionValidateData,
                PermissionAnalyzeData,
                PermissionViewData,
                PermissionDeleteData,
                PermissionManageUsers,
                PermissionManageAPIKeys,
                PermissionViewMetrics,
                PermissionManageSystem,
            },
        },
        "data_scientist": {
            Name:        "Data Scientist",
            Description: "Data generation and analysis",
            Permissions: []Permission{
                PermissionGenerateData,
                PermissionValidateData,
                PermissionAnalyzeData,
                PermissionViewData,
                PermissionViewMetrics,
            },
        },
        "viewer": {
            Name:        "Viewer",
            Description: "Read-only access",
            Permissions: []Permission{
                PermissionViewData,
                PermissionViewMetrics,
            },
        },
    }
}

func (rbac *RBACService) HasPermission(userID string, permission Permission) bool {
    userRoles := rbac.userRoles[userID]
    
    for _, roleName := range userRoles {
        if role, exists := rbac.roles[roleName]; exists {
            for _, perm := range role.Permissions {
                if perm == permission {
                    return true
                }
            }
        }
    }
    
    return false
}

func (rbac *RBACService) RequirePermission(permission Permission) gin.HandlerFunc {
    return func(c *gin.Context) {
        userID := c.GetString("user_id")
        
        if !rbac.HasPermission(userID, permission) {
            c.JSON(http.StatusForbidden, gin.H{
                "error": "insufficient permissions",
                "required_permission": permission,
            })
            c.Abort()
            return
        }
        
        c.Next()
    }
}
```

### 3. Secure Session Management

#### Session Store Implementation
```go
type SessionStore struct {
    redis    *redis.Client
    sessions sync.Map
    ttl      time.Duration
}

type Session struct {
    ID          string            `json:"id"`
    UserID      string            `json:"user_id"`
    CreatedAt   time.Time         `json:"created_at"`
    LastAccess  time.Time         `json:"last_access"`
    IPAddress   string            `json:"ip_address"`
    UserAgent   string            `json:"user_agent"`
    Attributes  map[string]string `json:"attributes"`
    IsActive    bool              `json:"is_active"`
}

func (ss *SessionStore) CreateSession(userID, ipAddress, userAgent string) (*Session, error) {
    session := &Session{
        ID:         generateSecureID(),
        UserID:     userID,
        CreatedAt:  time.Now(),
        LastAccess: time.Now(),
        IPAddress:  ipAddress,
        UserAgent:  userAgent,
        Attributes: make(map[string]string),
        IsActive:   true,
    }
    
    // Store in Redis
    sessionData, _ := json.Marshal(session)
    err := ss.redis.Set(session.ID, sessionData, ss.ttl).Err()
    if err != nil {
        return nil, err
    }
    
    // Store in local cache
    ss.sessions.Store(session.ID, session)
    
    return session, nil
}

func (ss *SessionStore) GetSession(sessionID string) (*Session, error) {
    // Try local cache first
    if cached, ok := ss.sessions.Load(sessionID); ok {
        session := cached.(*Session)
        if session.IsActive && time.Since(session.LastAccess) < ss.ttl {
            return session, nil
        }
    }
    
    // Try Redis
    sessionData, err := ss.redis.Get(sessionID).Result()
    if err != nil {
        return nil, err
    }
    
    var session Session
    if err := json.Unmarshal([]byte(sessionData), &session); err != nil {
        return nil, err
    }
    
    // Update local cache
    ss.sessions.Store(sessionID, &session)
    
    return &session, nil
}

func (ss *SessionStore) InvalidateSession(sessionID string) error {
    // Remove from Redis
    ss.redis.Del(sessionID)
    
    // Remove from local cache
    ss.sessions.Delete(sessionID)
    
    return nil
}
```

## Data Protection

### 1. Encryption at Rest

#### Database Encryption
```go
type EncryptionService struct {
    key        []byte
    aead       cipher.AEAD
    keyManager *KeyManager
}

func NewEncryptionService(keyManager *KeyManager) (*EncryptionService, error) {
    key, err := keyManager.GetCurrentKey()
    if err != nil {
        return nil, err
    }
    
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, err
    }
    
    aead, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    return &EncryptionService{
        key:        key,
        aead:       aead,
        keyManager: keyManager,
    }, nil
}

func (es *EncryptionService) Encrypt(plaintext []byte) ([]byte, error) {
    nonce := make([]byte, es.aead.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, err
    }
    
    ciphertext := es.aead.Seal(nonce, nonce, plaintext, nil)
    return ciphertext, nil
}

func (es *EncryptionService) Decrypt(ciphertext []byte) ([]byte, error) {
    nonceSize := es.aead.NonceSize()
    if len(ciphertext) < nonceSize {
        return nil, errors.New("ciphertext too short")
    }
    
    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
    plaintext, err := es.aead.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return nil, err
    }
    
    return plaintext, nil
}

// Encrypt time series data before storage
func (es *EncryptionService) EncryptTimeSeries(ts *TimeSeries) (*EncryptedTimeSeries, error) {
    // Serialize time series
    data, err := json.Marshal(ts)
    if err != nil {
        return nil, err
    }
    
    // Encrypt data
    encryptedData, err := es.Encrypt(data)
    if err != nil {
        return nil, err
    }
    
    return &EncryptedTimeSeries{
        ID:            ts.ID,
        SeriesID:      ts.SeriesID,
        EncryptedData: encryptedData,
        KeyVersion:    es.keyManager.GetCurrentVersion(),
        CreatedAt:     ts.CreatedAt,
        UpdatedAt:     time.Now(),
    }, nil
}
```

#### Key Management System
```go
type KeyManager struct {
    keys        map[int][]byte
    currentVersion int
    rotationInterval time.Duration
    kms         ExternalKMS
}

type ExternalKMS interface {
    Encrypt(keyID string, plaintext []byte) ([]byte, error)
    Decrypt(keyID string, ciphertext []byte) ([]byte, error)
    GenerateDataKey(keyID string) ([]byte, []byte, error)
}

func (km *KeyManager) RotateKeys() error {
    // Generate new key
    newKey := make([]byte, 32)
    if _, err := rand.Read(newKey); err != nil {
        return err
    }
    
    // Encrypt with KMS
    encryptedKey, err := km.kms.Encrypt("data-encryption-key", newKey)
    if err != nil {
        return err
    }
    
    // Store new version
    km.currentVersion++
    km.keys[km.currentVersion] = encryptedKey
    
    // Schedule cleanup of old keys
    go km.scheduleKeyCleanup()
    
    return nil
}

func (km *KeyManager) GetKey(version int) ([]byte, error) {
    encryptedKey, exists := km.keys[version]
    if !exists {
        return nil, errors.New("key version not found")
    }
    
    // Decrypt with KMS
    key, err := km.kms.Decrypt("data-encryption-key", encryptedKey)
    if err != nil {
        return nil, err
    }
    
    return key, nil
}
```

### 2. Privacy-Preserving Data Generation

#### Differential Privacy Implementation
```go
type DifferentialPrivacy struct {
    epsilon    float64
    delta      float64
    mechanism  PrivacyMechanism
}

type PrivacyMechanism interface {
    AddNoise(value float64, sensitivity float64) float64
    AddLaplaceNoise(value float64, sensitivity float64) float64
    AddGaussianNoise(value float64, sensitivity float64) float64
}

type LaplaceMechanism struct {
    epsilon float64
    rng     *rand.Rand
}

func (lm *LaplaceMechanism) AddNoise(value, sensitivity float64) float64 {
    scale := sensitivity / lm.epsilon
    
    // Generate Laplace noise
    u := lm.rng.Float64() - 0.5
    noise := scale * math.Log(1-2*math.Abs(u)) * math.Copysign(1, u)
    
    return value + noise
}

func (dp *DifferentialPrivacy) PrivatizeTimeSeries(ts *TimeSeries) *TimeSeries {
    privatized := &TimeSeries{
        SeriesID:  ts.SeriesID + "_private",
        Metadata:  ts.Metadata,
        CreatedAt: time.Now(),
    }
    
    for _, point := range ts.DataPoints {
        noisyValue := dp.mechanism.AddNoise(point.Value, 1.0) // Sensitivity = 1
        
        privatized.DataPoints = append(privatized.DataPoints, &DataPoint{
            Timestamp: point.Timestamp,
            Value:     noisyValue,
            Quality:   point.Quality * 0.9, // Reduce quality due to noise
            Metadata:  point.Metadata,
        })
    }
    
    privatized.Metadata.PrivacyLevel = "differential"
    privatized.Metadata.Epsilon = dp.epsilon
    privatized.Metadata.Delta = dp.delta
    
    return privatized
}
```

#### K-Anonymity Implementation
```go
type KAnonymityProcessor struct {
    k             int
    quasiIdentifiers []string
    generalizer   *Generalizer
}

type Generalizer struct {
    hierarchies map[string][]GeneralizationLevel
}

type GeneralizationLevel struct {
    Level int
    Transform func(string) string
}

func (kap *KAnonymityProcessor) EnsureKAnonymity(dataset []*TimeSeries) []*TimeSeries {
    // Group by quasi-identifiers
    groups := kap.groupByQuasiIdentifiers(dataset)
    
    var result []*TimeSeries
    
    for _, group := range groups {
        if len(group) < kap.k {
            // Need to generalize or suppress
            generalized := kap.generalizeGroup(group)
            result = append(result, generalized...)
        } else {
            result = append(result, group...)
        }
    }
    
    return result
}

func (kap *KAnonymityProcessor) groupByQuasiIdentifiers(dataset []*TimeSeries) map[string][]*TimeSeries {
    groups := make(map[string][]*TimeSeries)
    
    for _, ts := range dataset {
        key := kap.buildGroupKey(ts)
        groups[key] = append(groups[key], ts)
    }
    
    return groups
}

func (kap *KAnonymityProcessor) buildGroupKey(ts *TimeSeries) string {
    var parts []string
    
    for _, qi := range kap.quasiIdentifiers {
        if value, exists := ts.Metadata.Tags[qi]; exists {
            parts = append(parts, value)
        } else {
            parts = append(parts, "*")
        }
    }
    
    return strings.Join(parts, "|")
}
```

## Network Security

### 1. TLS Configuration

#### TLS Server Setup
```go
type TLSConfig struct {
    CertFile       string
    KeyFile        string
    CAFile         string
    MinVersion     uint16
    CipherSuites   []uint16
    ClientAuth     tls.ClientAuthType
    InsecureSkipVerify bool
}

func (tc *TLSConfig) BuildTLSConfig() (*tls.Config, error) {
    cert, err := tls.LoadX509KeyPair(tc.CertFile, tc.KeyFile)
    if err != nil {
        return nil, err
    }
    
    var clientCAs *x509.CertPool
    if tc.CAFile != "" {
        clientCAs = x509.NewCertPool()
        caCert, err := ioutil.ReadFile(tc.CAFile)
        if err != nil {
            return nil, err
        }
        clientCAs.AppendCertsFromPEM(caCert)
    }
    
    return &tls.Config{
        Certificates: []tls.Certificate{cert},
        ClientCAs:    clientCAs,
        ClientAuth:   tc.ClientAuth,
        MinVersion:   tc.MinVersion,
        CipherSuites: tc.CipherSuites,
        InsecureSkipVerify: tc.InsecureSkipVerify,
    }, nil
}

func CreateSecureTLSConfig() *TLSConfig {
    return &TLSConfig{
        MinVersion: tls.VersionTLS13,
        CipherSuites: []uint16{
            tls.TLS_AES_256_GCM_SHA384,
            tls.TLS_AES_128_GCM_SHA256,
            tls.TLS_CHACHA20_POLY1305_SHA256,
        },
        ClientAuth: tls.RequireAndVerifyClientCert,
    }
}
```

### 2. Network Policies (Kubernetes)

#### Pod Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tsiot-network-policy
  namespace: tsiot
spec:
  podSelector:
    matchLabels:
      app: tsiot-api
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: tsiot-frontend
    ports:
    - protocol: TCP
      port: 8080
  
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: tsiot-database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS outbound
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### 3. Service Mesh Security (Istio)

#### Security Policies
```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: tsiot-peer-auth
  namespace: tsiot
spec:
  mtls:
    mode: STRICT
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: tsiot-authz
  namespace: tsiot
spec:
  selector:
    matchLabels:
      app: tsiot-api
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/tsiot/sa/tsiot-frontend"]
  - to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/v1/*"]
  - when:
    - key: request.headers[authorization]
      values: ["Bearer *"]
```

## Security Monitoring and Incident Response

### 1. Security Event Logging

#### Audit Logger
```go
type AuditLogger struct {
    logger     *logrus.Logger
    storage    AuditStorage
    enricher   *EventEnricher
    alerter    *SecurityAlerter
}

type AuditEvent struct {
    ID          string                 `json:"id"`
    Timestamp   time.Time             `json:"timestamp"`
    EventType   string                `json:"event_type"`
    UserID      string                `json:"user_id"`
    IPAddress   string                `json:"ip_address"`
    UserAgent   string                `json:"user_agent"`
    Resource    string                `json:"resource"`
    Action      string                `json:"action"`
    Result      string                `json:"result"`
    Details     map[string]interface{} `json:"details"`
    RiskScore   int                   `json:"risk_score"`
    SessionID   string                `json:"session_id"`
}

func (al *AuditLogger) LogSecurityEvent(eventType, userID, resource, action, result string, details map[string]interface{}) {
    event := &AuditEvent{
        ID:        generateEventID(),
        Timestamp: time.Now(),
        EventType: eventType,
        UserID:    userID,
        Resource:  resource,
        Action:    action,
        Result:    result,
        Details:   details,
    }
    
    // Enrich with additional context
    al.enricher.EnrichEvent(event)
    
    // Calculate risk score
    event.RiskScore = al.calculateRiskScore(event)
    
    // Log to storage
    al.storage.Store(event)
    
    // Check for suspicious activity
    if event.RiskScore > 80 {
        al.alerter.TriggerSecurityAlert(event)
    }
    
    // Log to file
    al.logger.WithFields(logrus.Fields{
        "event_id":   event.ID,
        "event_type": event.EventType,
        "user_id":    event.UserID,
        "resource":   event.Resource,
        "action":     event.Action,
        "result":     event.Result,
        "risk_score": event.RiskScore,
    }).Info("Security event logged")
}

func (al *AuditLogger) calculateRiskScore(event *AuditEvent) int {
    score := 0
    
    // Failed authentication attempts
    if event.EventType == "authentication" && event.Result == "failure" {
        score += 30
    }
    
    // Admin actions
    if strings.Contains(event.Action, "admin") || strings.Contains(event.Action, "manage") {
        score += 20
    }
    
    // Unusual time access
    hour := event.Timestamp.Hour()
    if hour < 6 || hour > 22 {
        score += 15
    }
    
    // Multiple failures from same IP
    if al.getRecentFailures(event.IPAddress) > 5 {
        score += 40
    }
    
    return score
}
```

### 2. Intrusion Detection System

#### Anomaly Detection
```go
type IntrusionDetector struct {
    baselines    map[string]*Baseline
    rules        []DetectionRule
    alerter      *SecurityAlerter
    ml_model     AnomalyModel
}

type Baseline struct {
    UserID           string
    NormalBehavior   BehaviorProfile
    LastUpdated      time.Time
}

type BehaviorProfile struct {
    TypicalHours     []int
    TypicalEndpoints []string
    TypicalIPRanges  []string
    RequestRate      float64
    DataVolumeRange  [2]float64
}

type DetectionRule interface {
    Evaluate(event *AuditEvent, baseline *Baseline) (bool, string)
    GetSeverity() string
}

type BruteForceDetectionRule struct {
    threshold    int
    timeWindow   time.Duration
    severity     string
}

func (bfdr *BruteForceDetectionRule) Evaluate(event *AuditEvent, baseline *Baseline) (bool, string) {
    if event.EventType != "authentication" || event.Result != "failure" {
        return false, ""
    }
    
    // Count recent failures from same IP
    recentFailures := countRecentAuthFailures(event.IPAddress, bfdr.timeWindow)
    
    if recentFailures >= bfdr.threshold {
        return true, fmt.Sprintf("Brute force attack detected: %d failures from %s", 
                                recentFailures, event.IPAddress)
    }
    
    return false, ""
}

func (bfdr *BruteForceDetectionRule) GetSeverity() string {
    return bfdr.severity
}

func (id *IntrusionDetector) AnalyzeEvent(event *AuditEvent) {
    baseline := id.baselines[event.UserID]
    if baseline == nil {
        // Create baseline for new user
        baseline = id.createBaselineForUser(event.UserID)
        id.baselines[event.UserID] = baseline
    }
    
    // Run rule-based detection
    for _, rule := range id.rules {
        if triggered, message := rule.Evaluate(event, baseline); triggered {
            alert := &SecurityAlert{
                ID:          generateAlertID(),
                Timestamp:   time.Now(),
                EventID:     event.ID,
                RuleType:    reflect.TypeOf(rule).Name(),
                Severity:    rule.GetSeverity(),
                Message:     message,
                UserID:      event.UserID,
                IPAddress:   event.IPAddress,
            }
            
            id.alerter.TriggerAlert(alert)
        }
    }
    
    // Run ML-based anomaly detection
    if id.ml_model != nil {
        anomalyScore := id.ml_model.Score(event, baseline)
        if anomalyScore > 0.8 {
            alert := &SecurityAlert{
                ID:          generateAlertID(),
                Timestamp:   time.Now(),
                EventID:     event.ID,
                RuleType:    "ML_Anomaly",
                Severity:    "medium",
                Message:     fmt.Sprintf("Anomalous behavior detected (score: %.2f)", anomalyScore),
                UserID:      event.UserID,
                IPAddress:   event.IPAddress,
            }
            
            id.alerter.TriggerAlert(alert)
        }
    }
    
    // Update baseline
    id.updateBaseline(baseline, event)
}
```

### 3. Automated Incident Response

#### Response Automation
```go
type IncidentResponder struct {
    rules       []ResponseRule
    actions     map[string]ResponseAction
    escalation  *EscalationManager
}

type ResponseRule struct {
    Condition   func(*SecurityAlert) bool
    Actions     []string
    Automatic   bool
    Priority    int
}

type ResponseAction interface {
    Execute(alert *SecurityAlert) error
    GetDescription() string
    RequiresApproval() bool
}

type BlockIPAction struct {
    firewall FirewallManager
    duration time.Duration
}

func (bia *BlockIPAction) Execute(alert *SecurityAlert) error {
    return bia.firewall.BlockIP(alert.IPAddress, bia.duration)
}

func (bia *BlockIPAction) GetDescription() string {
    return fmt.Sprintf("Block IP address for %s", bia.duration)
}

func (bia *BlockIPAction) RequiresApproval() bool {
    return false
}

type DisableUserAction struct {
    userService UserService
}

func (dua *DisableUserAction) Execute(alert *SecurityAlert) error {
    return dua.userService.DisableUser(alert.UserID, "Security incident")
}

func (dua *DisableUserAction) GetDescription() string {
    return "Disable user account"
}

func (dua *DisableUserAction) RequiresApproval() bool {
    return true
}

func (ir *IncidentResponder) HandleAlert(alert *SecurityAlert) {
    for _, rule := range ir.rules {
        if rule.Condition(alert) {
            for _, actionName := range rule.Actions {
                action := ir.actions[actionName]
                if action == nil {
                    continue
                }
                
                if action.RequiresApproval() && !rule.Automatic {
                    ir.escalation.RequestApproval(alert, action)
                } else {
                    if err := action.Execute(alert); err != nil {
                        log.Errorf("Failed to execute action %s: %v", actionName, err)
                    }
                }
            }
        }
    }
}
```

## Compliance and Governance

### 1. GDPR Compliance

#### Data Subject Rights
```go
type GDPRService struct {
    storage     DataStorage
    encryption  *EncryptionService
    audit       *AuditLogger
    anonymizer  *DataAnonymizer
}

func (gs *GDPRService) HandleDataSubjectRequest(request *DataSubjectRequest) error {
    switch request.Type {
    case "access":
        return gs.handleAccessRequest(request)
    case "rectification":
        return gs.handleRectificationRequest(request)
    case "erasure":
        return gs.handleErasureRequest(request)
    case "portability":
        return gs.handlePortabilityRequest(request)
    default:
        return errors.New("unknown request type")
    }
}

func (gs *GDPRService) handleErasureRequest(request *DataSubjectRequest) error {
    // Find all data for the subject
    data, err := gs.storage.FindBySubjectID(request.SubjectID)
    if err != nil {
        return err
    }
    
    // Log the request
    gs.audit.LogSecurityEvent("gdpr_erasure", "system", "personal_data", "delete", "started", map[string]interface{}{
        "subject_id": request.SubjectID,
        "request_id": request.ID,
    })
    
    // Anonymize or delete data
    for _, item := range data {
        if item.CanBeDeleted() {
            if err := gs.storage.Delete(item.ID); err != nil {
                return err
            }
        } else {
            // Anonymize if deletion would break business logic
            anonymized := gs.anonymizer.Anonymize(item)
            if err := gs.storage.Update(anonymized); err != nil {
                return err
            }
        }
    }
    
    // Log completion
    gs.audit.LogSecurityEvent("gdpr_erasure", "system", "personal_data", "delete", "completed", map[string]interface{}{
        "subject_id": request.SubjectID,
        "request_id": request.ID,
        "items_processed": len(data),
    })
    
    return nil
}
```

### 2. SOC 2 Compliance

#### Access Controls and Monitoring
```go
type SOC2Compliance struct {
    accessLog     *AccessLogger
    changeTracker *ChangeTracker
    monitor       *ComplianceMonitor
}

type AccessLogger struct {
    storage AuditStorage
}

func (al *AccessLogger) LogDataAccess(userID, resource string, operation string) {
    event := &AccessEvent{
        ID:        generateID(),
        Timestamp: time.Now(),
        UserID:    userID,
        Resource:  resource,
        Operation: operation,
        IPAddress: getCurrentIPAddress(),
    }
    
    al.storage.StoreAccessEvent(event)
}

type ChangeTracker struct {
    storage ChangeStorage
}

func (ct *ChangeTracker) TrackChange(entity string, oldValue, newValue interface{}, userID string) {
    change := &ChangeEvent{
        ID:        generateID(),
        Timestamp: time.Now(),
        Entity:    entity,
        OldValue:  oldValue,
        NewValue:  newValue,
        UserID:    userID,
        Approved:  false,
    }
    
    ct.storage.StoreChange(change)
}

func (ct *ChangeTracker) RequireApproval(changeType string) bool {
    sensitiveChanges := []string{
        "user_permissions",
        "security_settings",
        "encryption_keys",
        "system_configuration",
    }
    
    for _, sensitive := range sensitiveChanges {
        if strings.Contains(changeType, sensitive) {
            return true
        }
    }
    
    return false
}
```

## Security Testing and Validation

### 1. Automated Security Testing

#### Security Test Suite
```go
type SecurityTestSuite struct {
    client     *http.Client
    baseURL    string
    testDB     *sql.DB
    config     *SecurityTestConfig
}

func (sts *SecurityTestSuite) RunAllTests() error {
    tests := []SecurityTest{
        sts.TestSQLInjection,
        sts.TestXSS,
        sts.TestCSRF,
        sts.TestAuthenticationBypass,
        sts.TestAuthorizationBypass,
        sts.TestRateLimiting,
        sts.TestInputValidation,
        sts.TestSessionManagement,
    }
    
    var results []TestResult
    
    for _, test := range tests {
        result := test()
        results = append(results, result)
        
        if result.Failed && result.Severity == "critical" {
            return fmt.Errorf("critical security test failed: %s", result.Name)
        }
    }
    
    sts.generateSecurityReport(results)
    return nil
}

func (sts *SecurityTestSuite) TestSQLInjection() TestResult {
    payloads := []string{
        "' OR '1'='1",
        "'; DROP TABLE users; --",
        "1' UNION SELECT * FROM users --",
    }
    
    for _, payload := range payloads {
        resp, err := sts.client.Get(fmt.Sprintf("%s/api/users?id=%s", sts.baseURL, url.QueryEscape(payload)))
        if err != nil {
            continue
        }
        
        body, _ := ioutil.ReadAll(resp.Body)
        resp.Body.Close()
        
        // Check for SQL error messages
        if strings.Contains(string(body), "SQL syntax") ||
           strings.Contains(string(body), "ORA-") ||
           strings.Contains(string(body), "PostgreSQL") {
            return TestResult{
                Name:     "SQL Injection",
                Failed:   true,
                Severity: "critical",
                Message:  "SQL injection vulnerability detected",
                Payload:  payload,
            }
        }
    }
    
    return TestResult{
        Name:     "SQL Injection",
        Failed:   false,
        Severity: "info",
        Message:  "No SQL injection vulnerabilities detected",
    }
}
```

### 2. Penetration Testing Framework

#### Automated Penetration Testing
```go
type PenTestFramework struct {
    target      string
    tests       []PenTest
    reporter    *PenTestReporter
    scanner     *VulnScanner
}

type PenTest interface {
    Name() string
    Execute() PenTestResult
    Prerequisites() []string
    Severity() string
}

type AuthenticationPenTest struct {
    target   string
    client   *http.Client
    wordlist []string
}

func (apt *AuthenticationPenTest) Execute() PenTestResult {
    result := PenTestResult{
        TestName:  "Authentication Security",
        StartTime: time.Now(),
        Findings:  []Finding{},
    }
    
    // Test for default credentials
    defaultCreds := []Credential{
        {"admin", "admin"},
        {"admin", "password"},
        {"root", "root"},
    }
    
    for _, cred := range defaultCreds {
        if apt.testCredential(cred) {
            result.Findings = append(result.Findings, Finding{
                Type:        "Default Credentials",
                Severity:    "high",
                Description: fmt.Sprintf("Default credentials found: %s:%s", cred.Username, cred.Password),
                Risk:        "Unauthorized access to the system",
                Recommendation: "Change default credentials immediately",
            })
        }
    }
    
    // Test for brute force protection
    if !apt.testBruteForceProtection() {
        result.Findings = append(result.Findings, Finding{
            Type:        "Missing Brute Force Protection",
            Severity:    "medium",
            Description: "No rate limiting detected on authentication endpoint",
            Risk:        "Account takeover via brute force attacks",
            Recommendation: "Implement account lockout and rate limiting",
        })
    }
    
    result.EndTime = time.Now()
    return result
}
```

## Conclusion

The TSIOT security architecture implements comprehensive security measures across all layers:

1. **Authentication & Authorization**: Multi-factor authentication, RBAC, secure session management
2. **Data Protection**: End-to-end encryption, privacy-preserving techniques, key management
3. **Network Security**: TLS 1.3, network policies, service mesh security
4. **Monitoring & Response**: Audit logging, intrusion detection, automated incident response
5. **Compliance**: GDPR, SOC 2, and other regulatory compliance
6. **Testing & Validation**: Automated security testing, penetration testing, vulnerability scanning

This multi-layered approach ensures robust protection against a wide range of security threats while maintaining compliance with industry standards and regulations. Regular security assessments and updates ensure the platform remains secure against evolving threats.