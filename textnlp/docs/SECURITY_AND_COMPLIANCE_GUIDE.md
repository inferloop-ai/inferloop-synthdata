# TextNLP Security and Compliance Guide

## Table of Contents
1. [Security Overview](#security-overview)
2. [Model Security](#model-security)
3. [API Security](#api-security)
4. [Data Privacy](#data-privacy)
5. [Compliance Requirements](#compliance-requirements)
6. [Access Management](#access-management)
7. [Threat Mitigation](#threat-mitigation)
8. [Security Monitoring](#security-monitoring)
9. [Incident Response](#incident-response)
10. [Security Best Practices](#security-best-practices)

## Security Overview

TextNLP implements comprehensive security measures to protect language models, generated content, and user data while ensuring compliance with industry standards and regulations.

### Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Perimeter                        │
├─────────────────────────────────────────────────────────────┤
│  WAF │ Rate Limiting │ DDoS Protection │ Bot Detection      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                    API Security Layer                       │
├────────────┬────────────┬────────────┬────────────────────┤
│   OAuth2   │    JWT     │   RBAC     │ Input Validation   │
└────────────┴────────────┴────────────┴────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                  Model Security Layer                       │
├────────────┬────────────┬────────────┬────────────────────┤
│Model Access│ Prompt     │ Output     │ Content            │
│ Control    │ Filtering  │ Filtering  │ Moderation         │
└────────────┴────────────┴────────────┴────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                    Data Security Layer                      │
├────────────┬────────────┬────────────┬────────────────────┤
│ Encryption │ Anonymize  │ Audit      │ Compliance         │
└────────────┴────────────┴────────────┴────────────────────┘
```

### Security Principles

1. **Zero Trust Architecture**: Verify every request
2. **Defense in Depth**: Multiple security layers
3. **Least Privilege**: Minimal access rights
4. **Privacy by Design**: Built-in privacy protection
5. **Continuous Monitoring**: Real-time threat detection

## Model Security

### 1. Model Access Control

```python
# model_security.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
import jwt

class ModelAccessLevel(Enum):
    PUBLIC = "public"
    RESTRICTED = "restricted"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"

@dataclass
class ModelSecurityPolicy:
    model_id: str
    access_level: ModelAccessLevel
    allowed_users: List[str]
    allowed_roles: List[str]
    rate_limits: Dict[str, int]
    prompt_filters: List[str]
    output_filters: List[str]

class ModelSecurityManager:
    def __init__(self):
        self.policies = {}
        self.model_checksums = {}
        
    def register_model(self, model_id: str, model_path: str, policy: ModelSecurityPolicy):
        """Register model with security policy"""
        # Calculate model checksum for integrity
        checksum = self.calculate_model_checksum(model_path)
        self.model_checksums[model_id] = checksum
        self.policies[model_id] = policy
        
    def calculate_model_checksum(self, model_path: str) -> str:
        """Calculate SHA-256 checksum of model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def verify_model_integrity(self, model_id: str, model_path: str) -> bool:
        """Verify model hasn't been tampered with"""
        current_checksum = self.calculate_model_checksum(model_path)
        return current_checksum == self.model_checksums.get(model_id)
        
    def check_access(self, model_id: str, user_id: str, user_roles: List[str]) -> bool:
        """Check if user has access to model"""
        policy = self.policies.get(model_id)
        if not policy:
            return False
            
        # Check access level
        if policy.access_level == ModelAccessLevel.PUBLIC:
            return True
            
        # Check user-specific access
        if user_id in policy.allowed_users:
            return True
            
        # Check role-based access
        if any(role in policy.allowed_roles for role in user_roles):
            return True
            
        return False
        
    def apply_prompt_filters(self, model_id: str, prompt: str) -> str:
        """Apply security filters to prompts"""
        policy = self.policies.get(model_id)
        if not policy:
            return prompt
            
        filtered_prompt = prompt
        for filter_pattern in policy.prompt_filters:
            # Apply filter (e.g., remove sensitive patterns)
            filtered_prompt = self.apply_filter(filtered_prompt, filter_pattern)
            
        return filtered_prompt
```

### 2. Prompt Injection Prevention

```python
# prompt_security.py
import re
from typing import List, Tuple, Optional
import spacy

class PromptSecurityScanner:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.injection_patterns = self.load_injection_patterns()
        self.sensitive_patterns = self.load_sensitive_patterns()
        
    def load_injection_patterns(self) -> List[Tuple[str, str]]:
        """Load known prompt injection patterns"""
        return [
            (r"ignore\s+previous\s+instructions", "instruction_override"),
            (r"disregard\s+all\s+prior", "instruction_override"),
            (r"system\s*:\s*", "system_prompt_injection"),
            (r"</?(script|iframe|object|embed)", "code_injection"),
            (r"(\$|#|!)\{.*\}", "template_injection"),
            (r"(union\s+select|drop\s+table|delete\s+from)", "sql_injection"),
            (r"<\|endoftext\|>|<\|startoftext\|>", "token_manipulation"),
        ]
        
    def load_sensitive_patterns(self) -> List[Tuple[str, str]]:
        """Load patterns for sensitive information"""
        return [
            (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
            (r"\b\d{16}\b", "credit_card"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
            (r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+", "password"),
            (r"(?i)(api[_-]?key|apikey)\s*[:=]\s*\S+", "api_key"),
        ]
        
    def scan_prompt(self, prompt: str) -> Dict[str, Any]:
        """Comprehensive security scan of prompt"""
        results = {
            "safe": True,
            "risk_score": 0,
            "detected_threats": [],
            "sensitive_data": [],
            "recommendations": []
        }
        
        # Check for injection attempts
        for pattern, threat_type in self.injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                results["safe"] = False
                results["risk_score"] += 50
                results["detected_threats"].append({
                    "type": threat_type,
                    "pattern": pattern,
                    "severity": "high"
                })
                
        # Check for sensitive data
        for pattern, data_type in self.sensitive_patterns:
            matches = re.findall(pattern, prompt)
            if matches:
                results["risk_score"] += 20
                results["sensitive_data"].append({
                    "type": data_type,
                    "count": len(matches),
                    "severity": "medium"
                })
                
        # Analyze prompt structure
        doc = self.nlp(prompt)
        
        # Check for suspicious entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG"] and self.is_suspicious_entity(ent.text):
                results["risk_score"] += 10
                results["detected_threats"].append({
                    "type": "suspicious_entity",
                    "entity": ent.text,
                    "severity": "low"
                })
                
        # Generate recommendations
        if results["risk_score"] > 0:
            results["recommendations"] = self.generate_recommendations(results)
            
        return results
        
    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt by removing dangerous patterns"""
        sanitized = prompt
        
        # Remove injection attempts
        for pattern, _ in self.injection_patterns:
            sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)
            
        # Mask sensitive data
        for pattern, data_type in self.sensitive_patterns:
            if data_type == "email":
                sanitized = re.sub(pattern, "[EMAIL_REDACTED]", sanitized)
            elif data_type == "ssn":
                sanitized = re.sub(pattern, "[SSN_REDACTED]", sanitized)
            elif data_type in ["password", "api_key"]:
                sanitized = re.sub(pattern, "[CREDENTIAL_REDACTED]", sanitized)
                
        return sanitized
```

### 3. Output Security Filtering

```python
# output_security.py
from typing import List, Dict, Any
import re

class OutputSecurityFilter:
    def __init__(self):
        self.content_filters = self.initialize_filters()
        self.toxicity_detector = self.load_toxicity_model()
        
    def initialize_filters(self) -> Dict[str, Any]:
        """Initialize content filtering rules"""
        return {
            "pii_removal": {
                "patterns": [
                    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REMOVED]"),
                    (r"\b\d{16}\b", "[CC_REMOVED]"),
                    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REMOVED]"),
                    (r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+", "[PASSWORD_REMOVED]")
                ],
                "enabled": True
            },
            "toxicity_filter": {
                "threshold": 0.7,
                "action": "block",
                "enabled": True
            },
            "content_moderation": {
                "categories": ["violence", "hate_speech", "explicit", "self_harm"],
                "action": "flag",
                "enabled": True
            }
        }
        
    def filter_output(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply security filters to model output"""
        result = {
            "filtered_text": text,
            "removed_content": [],
            "warnings": [],
            "blocked": False
        }
        
        # Apply PII removal
        if self.content_filters["pii_removal"]["enabled"]:
            for pattern, replacement in self.content_filters["pii_removal"]["patterns"]:
                matches = re.findall(pattern, result["filtered_text"])
                if matches:
                    result["filtered_text"] = re.sub(pattern, replacement, result["filtered_text"])
                    result["removed_content"].append({
                        "type": "pii",
                        "count": len(matches),
                        "pattern": pattern
                    })
                    
        # Check toxicity
        if self.content_filters["toxicity_filter"]["enabled"]:
            toxicity_score = self.check_toxicity(result["filtered_text"])
            if toxicity_score > self.content_filters["toxicity_filter"]["threshold"]:
                if self.content_filters["toxicity_filter"]["action"] == "block":
                    result["blocked"] = True
                    result["filtered_text"] = "[Content blocked due to policy violation]"
                result["warnings"].append({
                    "type": "toxicity",
                    "score": toxicity_score,
                    "threshold": self.content_filters["toxicity_filter"]["threshold"]
                })
                
        # Content moderation
        if self.content_filters["content_moderation"]["enabled"]:
            moderation_results = self.moderate_content(result["filtered_text"])
            for category, score in moderation_results.items():
                if category in self.content_filters["content_moderation"]["categories"] and score > 0.5:
                    result["warnings"].append({
                        "type": "content_moderation",
                        "category": category,
                        "score": score
                    })
                    
        return result
        
    def check_toxicity(self, text: str) -> float:
        """Check text toxicity level"""
        # Placeholder for actual toxicity detection
        # In production, use a proper toxicity detection model
        toxic_keywords = ["hate", "violence", "abuse", "threat"]
        score = sum(1 for keyword in toxic_keywords if keyword in text.lower()) / len(toxic_keywords)
        return min(score * 2, 1.0)  # Scale to 0-1
```

## API Security

### 1. Authentication and Authorization

```python
# api_security.py
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import redis

class APISecurityManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.redis_client = redis.Redis(decode_responses=True)
        
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
            
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        # Add token ID for revocation support
        token_id = secrets.token_urlsafe(16)
        to_encode["jti"] = token_id
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Store token in Redis for revocation checking
        self.redis_client.setex(
            f"token:{token_id}",
            int(expires_delta.total_seconds() if expires_delta else 900),
            "valid"
        )
        
        return encoded_jwt
        
    async def get_current_user(self, token: str = Depends(oauth2_scheme)):
        """Validate token and get current user"""
        credentials_exception = HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            token_id: str = payload.get("jti")
            
            if username is None or token_id is None:
                raise credentials_exception
                
            # Check if token is revoked
            if not self.redis_client.get(f"token:{token_id}"):
                raise HTTPException(status_code=401, detail="Token has been revoked")
                
        except JWTError:
            raise credentials_exception
            
        return username
        
    def revoke_token(self, token_id: str):
        """Revoke a token"""
        self.redis_client.delete(f"token:{token_id}")

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if request is within rate limit"""
        current = self.redis.incr(key)
        
        if current == 1:
            self.redis.expire(key, window)
            
        if current > limit:
            return False
            
        return True
        
    async def get_rate_limit_middleware(self, request, call_next):
        """Rate limiting middleware"""
        # Get client identifier
        client_id = request.client.host
        
        # Check rate limit (100 requests per minute)
        key = f"rate_limit:{client_id}:{datetime.utcnow().minute}"
        
        if not await self.check_rate_limit(key, 100, 60):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"}
            )
            
        response = await call_next(request)
        return response
```

### 2. Input Validation and Sanitization

```python
# input_validation.py
from pydantic import BaseModel, validator, constr, conint
from typing import Optional, List, Dict, Any
import re
import bleach

class TextGenerationRequest(BaseModel):
    prompt: constr(min_length=1, max_length=2000)
    max_tokens: conint(ge=1, le=2048) = 100
    temperature: float = 1.0
    top_p: float = 1.0
    model_id: str
    
    @validator('prompt')
    def sanitize_prompt(cls, v):
        """Sanitize prompt input"""
        # Remove any HTML/script tags
        v = bleach.clean(v, tags=[], strip=True)
        
        # Remove control characters
        v = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', v)
        
        # Limit consecutive whitespace
        v = re.sub(r'\s+', ' ', v)
        
        return v.strip()
        
    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature range"""
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
        
    @validator('top_p')
    def validate_top_p(cls, v):
        """Validate top_p range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('top_p must be between 0.0 and 1.0')
        return v
        
    @validator('model_id')
    def validate_model_id(cls, v):
        """Validate model ID format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid model ID format')
        return v

class InputSanitizer:
    def __init__(self):
        self.sql_injection_patterns = [
            r"(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
            r"(\s+or\s+1\s*=\s*1|--\s*$|;\s*$)",
            r"(exec\s*\(|execute\s+immediate)"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript\s*:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>"
        ]
        
    def sanitize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive input sanitization"""
        sanitized = {}
        
        for key, value in input_data.items():
            if isinstance(value, str):
                # Check for SQL injection
                for pattern in self.sql_injection_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        raise ValueError(f"Potential SQL injection detected in {key}")
                        
                # Check for XSS
                for pattern in self.xss_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        raise ValueError(f"Potential XSS detected in {key}")
                        
                # Sanitize HTML
                sanitized[key] = bleach.clean(value, tags=[], strip=True)
                
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_input(value)
                
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_input(item) if isinstance(item, dict)
                    else bleach.clean(item, tags=[], strip=True) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
                
        return sanitized
```

### 3. API Security Middleware

```python
# security_middleware.py
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import time
import uuid
import logging
from typing import Callable

class SecurityMiddleware:
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger(__name__)
        
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # Add request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Security headers
        start_time = time.time()
        
        # Log request
        self.logger.info(f"Request {request_id}: {request.method} {request.url.path}")
        
        # Check for common attack patterns in headers
        if self.detect_header_injection(request.headers):
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid request headers"}
            )
            
        try:
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
            
            # Log response
            process_time = time.time() - start_time
            self.logger.info(
                f"Request {request_id} completed: "
                f"status={response.status_code} "
                f"duration={process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request {request_id} failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id}
            )
            
    def detect_header_injection(self, headers: dict) -> bool:
        """Detect header injection attempts"""
        suspicious_patterns = [
            r"[\r\n]",  # CRLF injection
            r"<script",  # XSS in headers
            r"javascript:",
            r"\${",  # Template injection
        ]
        
        for header_name, header_value in headers.items():
            for pattern in suspicious_patterns:
                if re.search(pattern, str(header_value), re.IGNORECASE):
                    self.logger.warning(
                        f"Suspicious header detected: {header_name}={header_value}"
                    )
                    return True
                    
        return False
```

## Data Privacy

### 1. Privacy-Preserving Generation

```python
# privacy_generation.py
from typing import Dict, List, Any, Optional
import hashlib
from dataclasses import dataclass

@dataclass
class PrivacySettings:
    anonymize_entities: bool = True
    remove_pii: bool = True
    differential_privacy: bool = False
    epsilon: float = 1.0
    k_anonymity: int = 5
    encryption_enabled: bool = True

class PrivacyPreservingGenerator:
    def __init__(self, privacy_settings: PrivacySettings):
        self.settings = privacy_settings
        self.entity_map = {}  # Maps real entities to anonymized versions
        
    def generate_with_privacy(self, 
                            prompt: str, 
                            model,
                            context: Optional[Dict[str, Any]] = None) -> str:
        """Generate text with privacy preservation"""
        # Pre-process prompt
        if self.settings.anonymize_entities:
            prompt = self.anonymize_prompt_entities(prompt)
            
        # Generate text
        generated_text = model.generate(prompt)
        
        # Post-process output
        if self.settings.remove_pii:
            generated_text = self.remove_pii_from_output(generated_text)
            
        if self.settings.anonymize_entities:
            generated_text = self.anonymize_output_entities(generated_text)
            
        if self.settings.differential_privacy:
            generated_text = self.apply_differential_privacy(generated_text)
            
        return generated_text
        
    def anonymize_prompt_entities(self, prompt: str) -> str:
        """Anonymize entities in prompt"""
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(prompt)
        
        anonymized = prompt
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                anonymous_id = self.get_anonymous_id(ent.text, ent.label_)
                anonymized = anonymized.replace(ent.text, anonymous_id)
                
        return anonymized
        
    def get_anonymous_id(self, entity: str, entity_type: str) -> str:
        """Get consistent anonymous ID for entity"""
        if entity not in self.entity_map:
            # Generate deterministic but anonymous ID
            hash_input = f"{entity}:{entity_type}:salt"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
            
            if entity_type == "PERSON":
                self.entity_map[entity] = f"Person_{hash_value}"
            elif entity_type == "ORG":
                self.entity_map[entity] = f"Org_{hash_value}"
            elif entity_type == "GPE":
                self.entity_map[entity] = f"Location_{hash_value}"
                
        return self.entity_map[entity]
        
    def remove_pii_from_output(self, text: str) -> str:
        """Remove PII from generated text"""
        patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CREDIT_CARD]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]'),
            (r'\b\d{5}(?:-\d{4})?\b', '[ZIPCODE]')
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
            
        return text
```

### 2. Data Retention and Deletion

```python
# data_retention.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
import shutil
import os

class DataRetentionManager:
    def __init__(self, retention_policies: Dict[str, int]):
        """
        retention_policies: Dict mapping data types to retention days
        Example: {"prompts": 30, "generated_text": 90, "user_data": 365}
        """
        self.retention_policies = retention_policies
        self.deletion_log = []
        
    async def enforce_retention_policies(self):
        """Enforce data retention policies"""
        while True:
            try:
                for data_type, retention_days in self.retention_policies.items():
                    await self.cleanup_old_data(data_type, retention_days)
                    
                # Run daily
                await asyncio.sleep(86400)
                
            except Exception as e:
                logging.error(f"Error in retention enforcement: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
                
    async def cleanup_old_data(self, data_type: str, retention_days: int):
        """Clean up data older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        if data_type == "prompts":
            deleted_count = await self.delete_old_prompts(cutoff_date)
        elif data_type == "generated_text":
            deleted_count = await self.delete_old_generated_text(cutoff_date)
        elif data_type == "user_data":
            deleted_count = await self.delete_old_user_data(cutoff_date)
        elif data_type == "model_cache":
            deleted_count = await self.cleanup_model_cache(cutoff_date)
            
        if deleted_count > 0:
            self.log_deletion(data_type, deleted_count, cutoff_date)
            
    async def delete_old_prompts(self, cutoff_date: datetime) -> int:
        """Delete prompts older than cutoff date"""
        # Implementation depends on storage backend
        # Example for database:
        query = """
            DELETE FROM prompts 
            WHERE created_at < %s 
            AND NOT is_flagged_for_retention
            RETURNING id
        """
        deleted_ids = await db.fetch_all(query, cutoff_date)
        
        # Secure deletion from any file storage
        for prompt_id in deleted_ids:
            file_path = f"/data/prompts/{prompt_id}.txt"
            if os.path.exists(file_path):
                self.secure_delete_file(file_path)
                
        return len(deleted_ids)
        
    def secure_delete_file(self, file_path: str):
        """Securely delete file by overwriting before deletion"""
        if not os.path.exists(file_path):
            return
            
        file_size = os.path.getsize(file_path)
        
        # Overwrite with random data 3 times
        for _ in range(3):
            with open(file_path, "rb+") as f:
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
                
        # Finally delete
        os.remove(file_path)
        
    def handle_deletion_request(self, user_id: str) -> Dict[str, Any]:
        """Handle user data deletion request (GDPR right to be forgotten)"""
        deletion_tasks = []
        
        # Delete from all data stores
        deletion_tasks.append(self.delete_user_prompts(user_id))
        deletion_tasks.append(self.delete_user_generated_text(user_id))
        deletion_tasks.append(self.delete_user_preferences(user_id))
        deletion_tasks.append(self.delete_user_audit_logs(user_id))
        
        # Execute all deletions
        results = asyncio.gather(*deletion_tasks)
        
        # Log deletion
        self.log_user_deletion(user_id, results)
        
        return {
            "user_id": user_id,
            "deletion_timestamp": datetime.utcnow().isoformat(),
            "deleted_items": results,
            "status": "completed"
        }
```

## Compliance Requirements

### 1. GDPR Compliance for Text Generation

```python
# gdpr_text_compliance.py
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

class GDPRTextCompliance:
    def __init__(self):
        self.consent_records = {}
        self.processing_activities = []
        
    def record_consent(self, 
                      user_id: str, 
                      purpose: str, 
                      scope: List[str],
                      duration_days: int = 365):
        """Record user consent for text generation"""
        consent = {
            "user_id": user_id,
            "purpose": purpose,
            "scope": scope,
            "granted_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=duration_days)).isoformat(),
            "version": "2.0",
            "withdrawal_method": "DELETE /api/v1/consent/{user_id}"
        }
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
            
        self.consent_records[user_id].append(consent)
        
        # Log processing activity
        self.log_processing_activity({
            "activity": "consent_recorded",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "legal_basis": "consent",
            "purpose": purpose
        })
        
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has valid consent for purpose"""
        if user_id not in self.consent_records:
            return False
            
        for consent in self.consent_records[user_id]:
            if (consent["purpose"] == purpose and 
                datetime.fromisoformat(consent["expires_at"]) > datetime.utcnow()):
                return True
                
        return False
        
    def generate_privacy_notice(self) -> Dict[str, Any]:
        """Generate GDPR-compliant privacy notice"""
        return {
            "controller": {
                "name": "TextNLP Service",
                "contact": "privacy@textnlp.example.com",
                "dpo_contact": "dpo@textnlp.example.com"
            },
            "purposes": [
                {
                    "purpose": "text_generation",
                    "description": "Generate synthetic text based on user prompts",
                    "legal_basis": "consent",
                    "data_categories": ["prompts", "preferences", "usage_patterns"],
                    "retention_period": "90 days",
                    "recipients": ["internal_ml_team"]
                },
                {
                    "purpose": "service_improvement",
                    "description": "Improve model quality and service performance",
                    "legal_basis": "legitimate_interest",
                    "data_categories": ["anonymized_prompts", "performance_metrics"],
                    "retention_period": "2 years",
                    "recipients": ["internal_research_team"]
                }
            ],
            "rights": [
                "right_to_access",
                "right_to_rectification",
                "right_to_erasure",
                "right_to_data_portability",
                "right_to_object",
                "right_to_withdraw_consent"
            ],
            "international_transfers": {
                "occurs": False,
                "safeguards": "N/A"
            },
            "automated_decision_making": {
                "occurs": True,
                "description": "Content filtering and moderation",
                "right_to_human_review": True
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for GDPR data portability"""
        user_data = {
            "export_date": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "data_categories": {}
        }
        
        # Collect all user data
        user_data["data_categories"]["prompts"] = self.get_user_prompts(user_id)
        user_data["data_categories"]["generated_texts"] = self.get_user_generated_texts(user_id)
        user_data["data_categories"]["preferences"] = self.get_user_preferences(user_id)
        user_data["data_categories"]["consent_history"] = self.consent_records.get(user_id, [])
        user_data["data_categories"]["processing_activities"] = [
            activity for activity in self.processing_activities
            if activity.get("user_id") == user_id
        ]
        
        return user_data
```

### 2. AI Ethics and Compliance

```python
# ai_ethics_compliance.py
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class EthicalPrinciple(Enum):
    TRANSPARENCY = "transparency"
    FAIRNESS = "fairness"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    SAFETY = "safety"
    HUMAN_OVERSIGHT = "human_oversight"

@dataclass
class EthicalAssessment:
    principle: EthicalPrinciple
    score: float  # 0-1
    issues: List[str]
    recommendations: List[str]

class AIEthicsCompliance:
    def __init__(self):
        self.ethical_guidelines = self.load_ethical_guidelines()
        self.bias_detector = self.initialize_bias_detector()
        
    def assess_generation_request(self, 
                                prompt: str, 
                                context: Dict[str, Any]) -> List[EthicalAssessment]:
        """Assess ethical implications of generation request"""
        assessments = []
        
        # Transparency assessment
        transparency_score = self.assess_transparency(prompt, context)
        assessments.append(transparency_score)
        
        # Fairness assessment
        fairness_score = self.assess_fairness(prompt, context)
        assessments.append(fairness_score)
        
        # Safety assessment
        safety_score = self.assess_safety(prompt, context)
        assessments.append(safety_score)
        
        return assessments
        
    def assess_transparency(self, prompt: str, context: Dict[str, Any]) -> EthicalAssessment:
        """Assess transparency of AI generation"""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check if AI-generated content is clearly marked
        if not context.get("ai_disclosure", False):
            issues.append("AI-generated content not clearly disclosed")
            recommendations.append("Add clear disclosure that content is AI-generated")
            score -= 0.3
            
        # Check if model limitations are communicated
        if not context.get("limitations_disclosed", False):
            issues.append("Model limitations not communicated")
            recommendations.append("Include information about model capabilities and limitations")
            score -= 0.2
            
        return EthicalAssessment(
            principle=EthicalPrinciple.TRANSPARENCY,
            score=max(0, score),
            issues=issues,
            recommendations=recommendations
        )
        
    def assess_fairness(self, prompt: str, context: Dict[str, Any]) -> EthicalAssessment:
        """Assess fairness and bias in generation"""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check for bias in prompt
        bias_analysis = self.bias_detector.analyze(prompt)
        
        if bias_analysis["gender_bias"] > 0.3:
            issues.append("Potential gender bias detected in prompt")
            recommendations.append("Consider rephrasing to be more gender-neutral")
            score -= 0.2
            
        if bias_analysis["racial_bias"] > 0.3:
            issues.append("Potential racial bias detected")
            recommendations.append("Review prompt for stereotypes or discriminatory language")
            score -= 0.3
            
        return EthicalAssessment(
            principle=EthicalPrinciple.FAIRNESS,
            score=max(0, score),
            issues=issues,
            recommendations=recommendations
        )
        
    def generate_ethics_report(self, 
                             generation_id: str,
                             assessments: List[EthicalAssessment]) -> Dict[str, Any]:
        """Generate ethics compliance report"""
        return {
            "generation_id": generation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_score": sum(a.score for a in assessments) / len(assessments),
            "assessments": [
                {
                    "principle": a.principle.value,
                    "score": a.score,
                    "issues": a.issues,
                    "recommendations": a.recommendations
                }
                for a in assessments
            ],
            "compliance_status": "compliant" if all(a.score > 0.7 for a in assessments) else "review_required",
            "human_review_required": any(a.score < 0.5 for a in assessments)
        }
```

## Access Management

### 1. Fine-Grained Access Control

```python
# access_management.py
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import json

class ResourceType(Enum):
    MODEL = "model"
    PROMPT_TEMPLATE = "prompt_template"
    GENERATION_API = "generation_api"
    ANALYTICS = "analytics"
    ADMIN = "admin"

class Action(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    SHARE = "share"

@dataclass
class Permission:
    resource_type: ResourceType
    resource_id: str
    actions: Set[Action]
    conditions: Optional[Dict[str, Any]] = None

class AccessManager:
    def __init__(self):
        self.user_permissions = {}
        self.role_permissions = {}
        self.resource_policies = {}
        
    def grant_permission(self, 
                        user_id: str, 
                        permission: Permission):
        """Grant permission to user"""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = []
            
        self.user_permissions[user_id].append(permission)
        
        # Log permission grant
        self.audit_log({
            "action": "permission_granted",
            "user_id": user_id,
            "permission": permission,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    def check_permission(self,
                        user_id: str,
                        resource_type: ResourceType,
                        resource_id: str,
                        action: Action,
                        context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has permission"""
        # Check direct user permissions
        if user_id in self.user_permissions:
            for perm in self.user_permissions[user_id]:
                if (perm.resource_type == resource_type and
                    (perm.resource_id == resource_id or perm.resource_id == "*") and
                    action in perm.actions):
                    
                    # Check conditions if any
                    if perm.conditions:
                        if not self.evaluate_conditions(perm.conditions, context):
                            continue
                            
                    return True
                    
        # Check role-based permissions
        user_roles = self.get_user_roles(user_id)
        for role in user_roles:
            if role in self.role_permissions:
                for perm in self.role_permissions[role]:
                    if (perm.resource_type == resource_type and
                        (perm.resource_id == resource_id or perm.resource_id == "*") and
                        action in perm.actions):
                        
                        if perm.conditions:
                            if not self.evaluate_conditions(perm.conditions, context):
                                continue
                                
                        return True
                        
        return False
        
    def evaluate_conditions(self, 
                          conditions: Dict[str, Any], 
                          context: Optional[Dict[str, Any]]) -> bool:
        """Evaluate permission conditions"""
        if not context:
            return False
            
        for key, expected_value in conditions.items():
            if key not in context:
                return False
                
            if key == "time_range":
                # Check if current time is within allowed range
                current_hour = datetime.utcnow().hour
                if not (expected_value["start"] <= current_hour < expected_value["end"]):
                    return False
                    
            elif key == "ip_range":
                # Check if request IP is in allowed range
                if not self.ip_in_range(context.get("ip_address"), expected_value):
                    return False
                    
            elif context[key] != expected_value:
                return False
                
        return True
```

### 2. Session Management

```python
# session_management.py
from typing import Dict, Optional, List
import secrets
import redis
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session_timeout = 1800  # 30 minutes
        
    def create_session(self, 
                      user_id: str, 
                      user_data: Dict[str, Any]) -> str:
        """Create new user session"""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "ip_address": user_data.get("ip_address"),
            "user_agent": user_data.get("user_agent"),
            "permissions": user_data.get("permissions", []),
            "metadata": user_data.get("metadata", {})
        }
        
        # Store in Redis with expiration
        self.redis.setex(
            f"session:{session_id}",
            self.session_timeout,
            json.dumps(session_data)
        )
        
        # Track active sessions for user
        self.redis.sadd(f"user_sessions:{user_id}", session_id)
        
        return session_id
        
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and refresh session"""
        session_key = f"session:{session_id}"
        session_data = self.redis.get(session_key)
        
        if not session_data:
            return None
            
        session = json.loads(session_data)
        
        # Update last activity
        session["last_activity"] = datetime.utcnow().isoformat()
        
        # Extend session
        self.redis.setex(
            session_key,
            self.session_timeout,
            json.dumps(session)
        )
        
        return session
        
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        session_data = self.redis.get(f"session:{session_id}")
        
        if session_data:
            session = json.loads(session_data)
            user_id = session["user_id"]
            
            # Remove from active sessions
            self.redis.srem(f"user_sessions:{user_id}", session_id)
            
        # Delete session
        self.redis.delete(f"session:{session_id}")
        
    def invalidate_all_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user"""
        session_ids = self.redis.smembers(f"user_sessions:{user_id}")
        
        for session_id in session_ids:
            self.redis.delete(f"session:{session_id}")
            
        self.redis.delete(f"user_sessions:{user_id}")
        
    def get_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for user"""
        session_ids = self.redis.smembers(f"user_sessions:{user_id}")
        sessions = []
        
        for session_id in session_ids:
            session_data = self.redis.get(f"session:{session_id}")
            if session_data:
                session = json.loads(session_data)
                session["session_id"] = session_id
                sessions.append(session)
                
        return sessions
```

## Threat Mitigation

### 1. Common Attack Prevention

```python
# threat_mitigation.py
from typing import Dict, List, Any, Optional
import re
import ipaddress
from collections import defaultdict
from datetime import datetime, timedelta

class ThreatMitigator:
    def __init__(self):
        self.threat_patterns = self.load_threat_patterns()
        self.blocked_ips = set()
        self.suspicious_activity = defaultdict(list)
        
    def load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load known threat patterns"""
        return {
            "prompt_injection": [
                r"ignore\s+previous\s+instructions",
                r"system\s+prompt",
                r"admin\s+mode",
                r"bypass\s+security",
                r"reveal\s+prompt"
            ],
            "data_exfiltration": [
                r"list\s+all\s+users",
                r"dump\s+database",
                r"export\s+all\s+data",
                r"show\s+me\s+everything"
            ],
            "model_manipulation": [
                r"jailbreak",
                r"unlock\s+restrictions",
                r"disable\s+safety",
                r"turn\s+off\s+filters"
            ]
        }
        
    def analyze_request(self, 
                       request_data: Dict[str, Any],
                       user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request for threats"""
        threat_analysis = {
            "risk_level": "low",
            "threats_detected": [],
            "recommended_actions": [],
            "allow_request": True
        }
        
        # Check prompt for injection attempts
        prompt = request_data.get("prompt", "")
        prompt_threats = self.detect_prompt_threats(prompt)
        
        if prompt_threats:
            threat_analysis["threats_detected"].extend(prompt_threats)
            threat_analysis["risk_level"] = "high"
            
        # Check for abnormal behavior
        abnormal_behavior = self.detect_abnormal_behavior(user_context)
        
        if abnormal_behavior:
            threat_analysis["threats_detected"].extend(abnormal_behavior)
            threat_analysis["risk_level"] = "medium" if threat_analysis["risk_level"] == "low" else "high"
            
        # Check request frequency
        if self.is_rate_limit_exceeded(user_context):
            threat_analysis["threats_detected"].append({
                "type": "rate_limit_exceeded",
                "severity": "medium"
            })
            threat_analysis["recommended_actions"].append("throttle_user")
            
        # Determine if request should be blocked
        if threat_analysis["risk_level"] == "high":
            threat_analysis["allow_request"] = False
            threat_analysis["recommended_actions"].append("block_request")
            
        return threat_analysis
        
    def detect_prompt_threats(self, prompt: str) -> List[Dict[str, Any]]:
        """Detect threats in prompt"""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    threats.append({
                        "type": threat_type,
                        "pattern": pattern,
                        "severity": "high" if threat_type == "prompt_injection" else "medium"
                    })
                    
        return threats
        
    def detect_abnormal_behavior(self, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect abnormal user behavior"""
        abnormalities = []
        user_id = user_context.get("user_id")
        
        # Track activity
        activity_key = f"activity:{user_id}"
        self.suspicious_activity[activity_key].append(datetime.utcnow())
        
        # Clean old activity
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        self.suspicious_activity[activity_key] = [
            ts for ts in self.suspicious_activity[activity_key] if ts > cutoff
        ]
        
        # Check for suspicious patterns
        recent_activity = self.suspicious_activity[activity_key]
        
        # Rapid requests (more than 50 in 5 minutes)
        if len(recent_activity) > 50:
            abnormalities.append({
                "type": "rapid_requests",
                "count": len(recent_activity),
                "severity": "medium"
            })
            
        # Check for pattern changes
        if self.detect_usage_pattern_change(user_context):
            abnormalities.append({
                "type": "pattern_change",
                "severity": "low"
            })
            
        return abnormalities
```

### 2. DDoS Protection

```python
# ddos_protection.py
from typing import Dict, Set, Optional
import time
from collections import defaultdict
import asyncio

class DDoSProtection:
    def __init__(self):
        self.request_counts = defaultdict(lambda: defaultdict(int))
        self.blocked_ips = set()
        self.connection_pools = defaultdict(set)
        
    async def check_request(self, 
                          ip_address: str, 
                          user_agent: str) -> Dict[str, Any]:
        """Check if request should be allowed"""
        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            return {
                "allowed": False,
                "reason": "ip_blocked",
                "retry_after": 3600
            }
            
        # Check rate limits
        current_minute = int(time.time() / 60)
        
        # Clean old data
        self.cleanup_old_data(current_minute)
        
        # Increment counters
        self.request_counts[current_minute][ip_address] += 1
        
        # Check limits
        if self.request_counts[current_minute][ip_address] > 100:
            # Block IP for 1 hour
            self.blocked_ips.add(ip_address)
            asyncio.create_task(self.unblock_ip_after(ip_address, 3600))
            
            return {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "retry_after": 3600
            }
            
        # Check connection pool size
        if len(self.connection_pools[ip_address]) > 10:
            return {
                "allowed": False,
                "reason": "too_many_connections",
                "retry_after": 60
            }
            
        return {"allowed": True}
        
    def cleanup_old_data(self, current_minute: int):
        """Clean up old request data"""
        old_minutes = [m for m in self.request_counts if m < current_minute - 5]
        for minute in old_minutes:
            del self.request_counts[minute]
            
    async def unblock_ip_after(self, ip_address: str, seconds: int):
        """Unblock IP after specified time"""
        await asyncio.sleep(seconds)
        self.blocked_ips.discard(ip_address)
        
    def add_syn_cookie_protection(self):
        """Enable SYN cookie protection"""
        # This would be implemented at the OS/network level
        # Example sysctl settings:
        return {
            "net.ipv4.tcp_syncookies": 1,
            "net.ipv4.tcp_max_syn_backlog": 2048,
            "net.ipv4.tcp_synack_retries": 2
        }
```

## Security Monitoring

### 1. Real-time Security Monitoring

```python
# security_monitoring.py
from typing import Dict, List, Any
import asyncio
from datetime import datetime
from collections import deque

class SecurityMonitor:
    def __init__(self):
        self.event_queue = asyncio.Queue()
        self.alert_handlers = []
        self.metrics = defaultdict(int)
        self.recent_events = deque(maxlen=1000)
        
    async def monitor_events(self):
        """Main monitoring loop"""
        while True:
            try:
                event = await self.event_queue.get()
                await self.process_event(event)
            except Exception as e:
                logging.error(f"Error processing event: {e}")
                
    async def process_event(self, event: Dict[str, Any]):
        """Process security event"""
        event["processed_at"] = datetime.utcnow().isoformat()
        
        # Store recent events
        self.recent_events.append(event)
        
        # Update metrics
        self.metrics[event["type"]] += 1
        
        # Check for security patterns
        if event["type"] == "authentication_failure":
            await self.check_brute_force(event)
        elif event["type"] == "prompt_injection_attempt":
            await self.check_injection_patterns(event)
        elif event["type"] == "data_access":
            await self.check_data_exfiltration(event)
            
        # Log event
        await self.log_security_event(event)
        
    async def check_brute_force(self, event: Dict[str, Any]):
        """Check for brute force attacks"""
        user_id = event.get("user_id")
        ip_address = event.get("ip_address")
        
        # Count recent failures
        recent_failures = sum(
            1 for e in self.recent_events
            if e["type"] == "authentication_failure" and
            e.get("ip_address") == ip_address and
            datetime.fromisoformat(e["timestamp"]) > 
            datetime.utcnow() - timedelta(minutes=5)
        )
        
        if recent_failures >= 5:
            await self.trigger_alert({
                "alert_type": "brute_force_detected",
                "severity": "high",
                "ip_address": ip_address,
                "failure_count": recent_failures,
                "recommended_action": "block_ip"
            })
            
    async def trigger_alert(self, alert: Dict[str, Any]):
        """Trigger security alert"""
        alert["timestamp"] = datetime.utcnow().isoformat()
        
        # Execute all alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logging.error(f"Alert handler error: {e}")
                
    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security dashboard"""
        return {
            "total_events": len(self.recent_events),
            "event_types": dict(self.metrics),
            "recent_alerts": self.get_recent_alerts(),
            "threat_level": self.calculate_threat_level(),
            "active_threats": self.get_active_threats(),
            "recommendations": self.generate_recommendations()
        }
        
    def calculate_threat_level(self) -> str:
        """Calculate current threat level"""
        high_severity_events = sum(
            1 for e in self.recent_events
            if e.get("severity") == "high"
        )
        
        if high_severity_events > 10:
            return "critical"
        elif high_severity_events > 5:
            return "high"
        elif high_severity_events > 0:
            return "medium"
        else:
            return "low"
```

### 2. Security Metrics and Reporting

```python
# security_metrics.py
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta

class SecurityMetricsCollector:
    def __init__(self):
        self.metrics_store = []
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive security metrics"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "authentication": self.collect_auth_metrics(),
            "api_security": self.collect_api_metrics(),
            "model_security": self.collect_model_metrics(),
            "compliance": self.collect_compliance_metrics(),
            "incidents": self.collect_incident_metrics()
        }
        
        self.metrics_store.append(metrics)
        return metrics
        
    def collect_auth_metrics(self) -> Dict[str, Any]:
        """Collect authentication metrics"""
        return {
            "total_logins": self.count_events("login_success"),
            "failed_logins": self.count_events("login_failure"),
            "mfa_usage_rate": self.calculate_mfa_usage(),
            "password_reset_requests": self.count_events("password_reset"),
            "account_lockouts": self.count_events("account_locked"),
            "session_hijack_attempts": self.count_events("session_hijack_attempt")
        }
        
    def generate_security_report(self, period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "executive_summary": self.generate_executive_summary(start_date, end_date),
            "detailed_metrics": self.generate_detailed_metrics(start_date, end_date),
            "incidents": self.summarize_incidents(start_date, end_date),
            "compliance_status": self.assess_compliance_status(),
            "recommendations": self.generate_security_recommendations()
        }
        
        return report
        
    def generate_executive_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate executive summary"""
        return {
            "overall_security_score": self.calculate_security_score(),
            "critical_incidents": self.count_critical_incidents(start_date, end_date),
            "compliance_violations": self.count_compliance_violations(start_date, end_date),
            "key_improvements": self.identify_improvements(),
            "key_risks": self.identify_risks()
        }
```

## Incident Response

### 1. Automated Incident Response

```python
# incident_response.py
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio

class IncidentSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class IncidentType(Enum):
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    PROMPT_INJECTION = "prompt_injection"
    MODEL_MANIPULATION = "model_manipulation"
    API_ABUSE = "api_abuse"
    COMPLIANCE_VIOLATION = "compliance_violation"

class IncidentResponseSystem:
    def __init__(self):
        self.response_playbooks = self.load_playbooks()
        self.incident_queue = asyncio.Queue()
        self.active_incidents = {}
        
    async def handle_incident(self, incident: Dict[str, Any]):
        """Handle security incident"""
        incident_id = self.generate_incident_id()
        
        incident_record = {
            "id": incident_id,
            "type": incident["type"],
            "severity": incident["severity"],
            "timestamp": datetime.utcnow().isoformat(),
            "status": "detected",
            "details": incident
        }
        
        self.active_incidents[incident_id] = incident_record
        
        # Execute response playbook
        playbook = self.response_playbooks.get(incident["type"])
        
        if playbook:
            await self.execute_playbook(incident_id, playbook)
        else:
            await self.execute_default_response(incident_id)
            
        return incident_id
        
    async def execute_playbook(self, incident_id: str, playbook: Dict[str, Any]):
        """Execute incident response playbook"""
        incident = self.active_incidents[incident_id]
        
        try:
            # Containment phase
            incident["status"] = "containing"
            await self.execute_containment(incident, playbook["containment"])
            
            # Investigation phase
            incident["status"] = "investigating"
            investigation_results = await self.execute_investigation(
                incident, playbook["investigation"]
            )
            incident["investigation_results"] = investigation_results
            
            # Remediation phase
            incident["status"] = "remediating"
            await self.execute_remediation(incident, playbook["remediation"])
            
            # Recovery phase
            incident["status"] = "recovering"
            await self.execute_recovery(incident, playbook["recovery"])
            
            # Post-incident phase
            incident["status"] = "closed"
            await self.post_incident_review(incident)
            
        except Exception as e:
            incident["status"] = "error"
            incident["error"] = str(e)
            await self.escalate_incident(incident_id)
            
    async def execute_containment(self, incident: Dict[str, Any], containment_steps: List[Dict]):
        """Execute containment actions"""
        for step in containment_steps:
            if step["action"] == "block_user":
                await self.block_user(incident["details"]["user_id"])
            elif step["action"] == "block_ip":
                await self.block_ip(incident["details"]["ip_address"])
            elif step["action"] == "disable_model":
                await self.disable_model(incident["details"]["model_id"])
            elif step["action"] == "isolate_system":
                await self.isolate_system(incident["details"]["system_id"])
                
    async def execute_investigation(self, 
                                  incident: Dict[str, Any], 
                                  investigation_steps: List[Dict]) -> Dict[str, Any]:
        """Execute investigation actions"""
        results = {}
        
        for step in investigation_steps:
            if step["action"] == "analyze_logs":
                results["log_analysis"] = await self.analyze_logs(
                    incident["timestamp"],
                    step["parameters"]
                )
            elif step["action"] == "trace_requests":
                results["request_trace"] = await self.trace_requests(
                    incident["details"]["user_id"],
                    step["parameters"]
                )
            elif step["action"] == "check_data_access":
                results["data_access"] = await self.check_data_access(
                    incident["details"]["user_id"],
                    step["parameters"]
                )
                
        return results
```

### 2. Incident Documentation and Learning

```python
# incident_documentation.py
class IncidentDocumentation:
    def __init__(self):
        self.incident_database = []
        self.lessons_learned = []
        
    def document_incident(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive incident documentation"""
        documentation = {
            "incident_id": incident["id"],
            "executive_summary": self.generate_executive_summary(incident),
            "timeline": self.create_incident_timeline(incident),
            "impact_assessment": self.assess_impact(incident),
            "root_cause_analysis": self.analyze_root_cause(incident),
            "response_evaluation": self.evaluate_response(incident),
            "lessons_learned": self.extract_lessons(incident),
            "recommendations": self.generate_recommendations(incident),
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.incident_database.append(documentation)
        return documentation
        
    def generate_post_mortem(self, incident_id: str) -> Dict[str, Any]:
        """Generate incident post-mortem report"""
        incident = self.get_incident(incident_id)
        
        return {
            "incident_overview": {
                "id": incident_id,
                "type": incident["type"],
                "severity": incident["severity"],
                "duration": self.calculate_incident_duration(incident),
                "systems_affected": incident.get("systems_affected", [])
            },
            "what_happened": self.describe_incident(incident),
            "why_it_happened": self.analyze_causes(incident),
            "how_we_responded": self.describe_response(incident),
            "what_we_learned": self.summarize_lessons(incident),
            "action_items": self.create_action_items(incident),
            "prevention_measures": self.define_prevention_measures(incident)
        }
```

## Security Best Practices

### 1. Development Security

```python
# dev_security_practices.py
class DevelopmentSecurityPractices:
    def __init__(self):
        self.security_checks = []
        
    def pre_commit_security_check(self, code_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run security checks before code commit"""
        results = {
            "passed": True,
            "checks": [],
            "vulnerabilities": []
        }
        
        for change in code_changes:
            # Check for hardcoded secrets
            secrets_check = self.check_hardcoded_secrets(change["content"])
            if secrets_check["found"]:
                results["passed"] = False
                results["vulnerabilities"].append({
                    "type": "hardcoded_secret",
                    "file": change["file"],
                    "line": secrets_check["line"],
                    "severity": "critical"
                })
                
            # Check for vulnerable dependencies
            if change["file"].endswith("requirements.txt") or change["file"] == "pyproject.toml":
                dep_check = self.check_dependencies(change["content"])
                if dep_check["vulnerable"]:
                    results["passed"] = False
                    results["vulnerabilities"].extend(dep_check["vulnerabilities"])
                    
            # Check for security anti-patterns
            antipatterns = self.check_security_antipatterns(change["content"])
            if antipatterns:
                results["passed"] = False
                results["vulnerabilities"].extend(antipatterns)
                
        return results
        
    def check_hardcoded_secrets(self, content: str) -> Dict[str, Any]:
        """Check for hardcoded secrets in code"""
        secret_patterns = [
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "api_key"),
            (r'password\s*=\s*["\'][^"\']+["\']', "password"),
            (r'secret[_-]?key\s*=\s*["\'][^"\']+["\']', "secret_key"),
            (r'aws[_-]?access[_-]?key[_-]?id\s*=\s*["\'][^"\']+["\']', "aws_key"),
            (r'private[_-]?key\s*=\s*["\'][^"\']+["\']', "private_key")
        ]
        
        for pattern, secret_type in secret_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return {
                    "found": True,
                    "type": secret_type,
                    "line": content[:match.start()].count('\n') + 1
                }
                
        return {"found": False}
```

### 2. Security Training and Awareness

```python
# security_training.py
class SecurityTrainingProgram:
    def __init__(self):
        self.training_modules = self.define_training_modules()
        self.completion_records = {}
        
    def define_training_modules(self) -> Dict[str, Dict[str, Any]]:
        """Define security training modules"""
        return {
            "text_generation_security": {
                "title": "Secure Text Generation Practices",
                "duration": "2 hours",
                "topics": [
                    "Prompt injection prevention",
                    "Output filtering techniques",
                    "Model security best practices",
                    "Privacy-preserving generation"
                ],
                "required_for": ["developers", "ml_engineers"],
                "frequency": "quarterly"
            },
            "api_security": {
                "title": "API Security for AI Systems",
                "duration": "3 hours",
                "topics": [
                    "Authentication and authorization",
                    "Rate limiting and DDoS protection",
                    "Input validation and sanitization",
                    "Security headers and CORS"
                ],
                "required_for": ["developers", "devops"],
                "frequency": "bi-annual"
            },
            "compliance_awareness": {
                "title": "AI Compliance and Ethics",
                "duration": "1.5 hours",
                "topics": [
                    "GDPR and data privacy",
                    "AI ethics principles",
                    "Bias detection and mitigation",
                    "Transparency requirements"
                ],
                "required_for": ["all"],
                "frequency": "annual"
            },
            "incident_response": {
                "title": "Security Incident Response",
                "duration": "2 hours",
                "topics": [
                    "Incident identification",
                    "Response procedures",
                    "Communication protocols",
                    "Post-incident review"
                ],
                "required_for": ["security_team", "team_leads"],
                "frequency": "bi-annual"
            }
        }
        
    def track_completion(self, user_id: str, module_id: str, score: float):
        """Track training completion"""
        if user_id not in self.completion_records:
            self.completion_records[user_id] = []
            
        self.completion_records[user_id].append({
            "module_id": module_id,
            "completion_date": datetime.utcnow().isoformat(),
            "score": score,
            "passed": score >= 0.8
        })
        
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate training compliance report"""
        return {
            "overall_completion_rate": self.calculate_completion_rate(),
            "module_statistics": self.get_module_statistics(),
            "overdue_training": self.identify_overdue_training(),
            "upcoming_training": self.get_upcoming_training(),
            "recommendations": self.generate_training_recommendations()
        }
```

## Conclusion

This comprehensive security and compliance guide for TextNLP provides:

1. **Multi-layered security architecture** protecting models, APIs, and data
2. **Advanced threat detection** including prompt injection and model manipulation prevention
3. **Privacy-preserving techniques** for text generation
4. **Comprehensive compliance frameworks** (GDPR, AI Ethics)
5. **Robust access management** with fine-grained permissions
6. **Real-time security monitoring** and alerting
7. **Automated incident response** with detailed playbooks
8. **Security best practices** for development and operations
9. **Training programs** for security awareness
10. **Continuous improvement** through metrics and reporting

Regular review and updates ensure the TextNLP system maintains the highest security standards while enabling innovative text generation capabilities.