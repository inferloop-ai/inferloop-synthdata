# Tabular Data Security and Compliance Guide

## Table of Contents
1. [Security Overview](#security-overview)
2. [Data Security](#data-security)
3. [Access Control](#access-control)
4. [Network Security](#network-security)
5. [Compliance Frameworks](#compliance-frameworks)
6. [Audit and Logging](#audit-and-logging)
7. [Incident Response](#incident-response)
8. [Security Hardening](#security-hardening)
9. [Compliance Checklists](#compliance-checklists)
10. [Best Practices](#best-practices)

## Security Overview

The Tabular synthetic data system implements defense-in-depth security architecture to protect sensitive data and ensure compliance with regulatory requirements.

### Security Principles
- **Data Minimization**: Only process necessary data
- **Privacy by Design**: Built-in privacy protection
- **Zero Trust**: Verify everything, trust nothing
- **Least Privilege**: Minimal access rights
- **Defense in Depth**: Multiple security layers

### Security Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Security Perimeter                        │
├─────────────────────────────────────────────────────────────┤
│  WAF │ DDoS Protection │ IDS/IPS │ Rate Limiting           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                    Application Security                     │
├────────────┬────────────┬────────────┬────────────────────┤
│    Auth    │   RBAC     │ Encryption │ Input Validation   │
└────────────┴────────────┴────────────┴────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                      Data Security                          │
├────────────┬────────────┬────────────┬────────────────────┤
│ Encryption │ Masking    │ Anonymize  │ Access Control     │
└────────────┴────────────┴────────────┴────────────────────┘
```

## Data Security

### 1. Encryption at Rest

#### Database Encryption
```yaml
# PostgreSQL transparent data encryption
postgresql:
  encryption:
    enabled: true
    algorithm: AES-256
    key_management: AWS_KMS  # or Azure Key Vault, HashiCorp Vault
    
# Configuration
CREATE EXTENSION pgcrypto;
ALTER DATABASE synthdata SET encryption = 'on';
```

#### File System Encryption
```bash
# Linux disk encryption setup
# Install cryptsetup
sudo apt-get install cryptsetup

# Create encrypted partition
sudo cryptsetup luksFormat /dev/sdb1
sudo cryptsetup open /dev/sdb1 encrypted_data
sudo mkfs.ext4 /dev/mapper/encrypted_data
sudo mount /dev/mapper/encrypted_data /mnt/secure_data

# Auto-mount configuration
echo "encrypted_data /dev/sdb1 none luks" >> /etc/crypttab
```

### 2. Encryption in Transit

#### TLS Configuration
```nginx
# Nginx SSL configuration
server {
    listen 443 ssl http2;
    server_name synthdata.example.com;
    
    # Modern TLS configuration
    ssl_certificate /etc/ssl/certs/synthdata.crt;
    ssl_certificate_key /etc/ssl/private/synthdata.key;
    
    # TLS 1.2 and 1.3 only
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Strong ciphers only
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    
    # Enable HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Additional security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

### 3. Data Anonymization

```python
# anonymizer.py
import hashlib
import secrets
from typing import Dict, Any
from cryptography.fernet import Fernet

class DataAnonymizer:
    def __init__(self, key: bytes = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
        
    def anonymize_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize personally identifiable information"""
        anonymized = data.copy()
        
        # Hash email addresses
        if 'email' in data:
            anonymized['email'] = self.hash_email(data['email'])
            
        # Mask SSN
        if 'ssn' in data:
            anonymized['ssn'] = self.mask_ssn(data['ssn'])
            
        # Generalize dates
        if 'dob' in data:
            anonymized['dob'] = self.generalize_date(data['dob'])
            
        return anonymized
    
    def hash_email(self, email: str) -> str:
        """One-way hash of email addresses"""
        salt = "synthdata_salt_2024"  # Use secure salt management
        return hashlib.sha256(f"{email}{salt}".encode()).hexdigest()[:16] + "@anonymous.local"
    
    def mask_ssn(self, ssn: str) -> str:
        """Mask SSN keeping format"""
        if len(ssn) >= 9:
            return f"XXX-XX-{ssn[-4:]}"
        return "XXX-XX-XXXX"
    
    def generalize_date(self, date: str) -> str:
        """Generalize dates to year only"""
        try:
            year = date.split('-')[0]
            return f"{year}-01-01"
        except:
            return "1900-01-01"
```

### 4. Privacy-Preserving Techniques

```python
# differential_privacy.py
import numpy as np
from typing import List, Tuple

class DifferentialPrivacy:
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        
    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def private_count(self, true_count: int, max_contribution: int = 1) -> int:
        """Differentially private count"""
        noisy_count = self.add_laplace_noise(true_count, max_contribution)
        return max(0, int(round(noisy_count)))
    
    def private_mean(self, values: List[float], bounds: Tuple[float, float]) -> float:
        """Differentially private mean with bounded values"""
        true_mean = np.mean(values)
        sensitivity = (bounds[1] - bounds[0]) / len(values)
        return self.add_laplace_noise(true_mean, sensitivity)
```

## Access Control

### 1. Role-Based Access Control (RBAC)

```python
# rbac.py
from enum import Enum
from typing import List, Set
from dataclasses import dataclass

class Permission(Enum):
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    GENERATE_SYNTHETIC = "generate_synthetic"
    MANAGE_MODELS = "manage_models"
    VIEW_ANALYTICS = "view_analytics"
    ADMIN = "admin"

@dataclass
class Role:
    name: str
    permissions: Set[Permission]

# Define roles
ROLES = {
    "viewer": Role("viewer", {Permission.READ_DATA, Permission.VIEW_ANALYTICS}),
    "analyst": Role("analyst", {
        Permission.READ_DATA, 
        Permission.GENERATE_SYNTHETIC,
        Permission.VIEW_ANALYTICS
    }),
    "data_engineer": Role("data_engineer", {
        Permission.READ_DATA,
        Permission.WRITE_DATA,
        Permission.GENERATE_SYNTHETIC,
        Permission.MANAGE_MODELS,
        Permission.VIEW_ANALYTICS
    }),
    "admin": Role("admin", {Permission.ADMIN})
}

class AccessControl:
    def __init__(self):
        self.user_roles = {}
        
    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user"""
        if role_name in ROLES:
            self.user_roles[user_id] = ROLES[role_name]
            
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has permission"""
        if user_id not in self.user_roles:
            return False
            
        role = self.user_roles[user_id]
        return permission in role.permissions or Permission.ADMIN in role.permissions
```

### 2. API Authentication

```python
# auth.py
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class JWTAuth:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer()
        
    def create_token(self, user_id: str, role: str) -> str:
        """Create JWT token"""
        payload = {
            "user_id": user_id,
            "role": role,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials) -> dict:
        """Verify JWT token"""
        token = credentials.credentials
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
```

## Network Security

### 1. Firewall Configuration

```bash
#!/bin/bash
# firewall_setup.sh

# Reset firewall
sudo iptables -F
sudo iptables -X

# Default policies
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT

# Allow loopback
sudo iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (restrict source IPs)
sudo iptables -A INPUT -p tcp --dport 22 -s 10.0.0.0/8 -j ACCEPT

# Allow HTTPS
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow application ports (internal only)
sudo iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 5432 -s 10.0.0.0/8 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 6379 -s 10.0.0.0/8 -j ACCEPT

# Rate limiting
sudo iptables -A INPUT -p tcp --dport 443 -m limit --limit 100/minute --limit-burst 200 -j ACCEPT

# DDoS protection
sudo iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT

# Log dropped packets
sudo iptables -A INPUT -j LOG --log-prefix "IPTables-Dropped: " --log-level 4

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
```

### 2. Network Segmentation

```yaml
# network_config.yaml
networks:
  dmz:
    subnet: 10.0.1.0/24
    vlan: 100
    services:
      - load_balancer
      - web_gateway
      
  application:
    subnet: 10.0.2.0/24
    vlan: 200
    services:
      - api_servers
      - processing_nodes
      
  data:
    subnet: 10.0.3.0/24
    vlan: 300
    services:
      - database_cluster
      - cache_cluster
      - storage_systems
      
  management:
    subnet: 10.0.4.0/24
    vlan: 400
    services:
      - monitoring
      - logging
      - backup_systems

firewall_rules:
  - from: dmz
    to: application
    allow: ["tcp/8000", "tcp/8001"]
    
  - from: application
    to: data
    allow: ["tcp/5432", "tcp/6379", "tcp/9000"]
    
  - from: management
    to: all
    allow: ["tcp/22", "tcp/9090", "tcp/3000"]
```

## Compliance Frameworks

### 1. GDPR Compliance

```python
# gdpr_compliance.py
from datetime import datetime
from typing import Dict, List
import json

class GDPRCompliance:
    def __init__(self):
        self.consent_records = {}
        self.data_inventory = {}
        
    def record_consent(self, user_id: str, purpose: str, granted: bool):
        """Record user consent for data processing"""
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
            
        self.consent_records[user_id].append({
            "purpose": purpose,
            "granted": granted,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0"
        })
        
    def handle_data_request(self, user_id: str, request_type: str) -> Dict:
        """Handle GDPR data subject requests"""
        if request_type == "access":
            return self.export_user_data(user_id)
        elif request_type == "deletion":
            return self.delete_user_data(user_id)
        elif request_type == "portability":
            return self.export_portable_data(user_id)
        elif request_type == "rectification":
            return {"status": "manual_review_required"}
            
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data for access requests"""
        return {
            "user_id": user_id,
            "data": self.data_inventory.get(user_id, {}),
            "consent_history": self.consent_records.get(user_id, []),
            "export_date": datetime.utcnow().isoformat()
        }
        
    def delete_user_data(self, user_id: str) -> Dict:
        """Delete user data (right to be forgotten)"""
        # Implement secure deletion
        if user_id in self.data_inventory:
            del self.data_inventory[user_id]
        if user_id in self.consent_records:
            del self.consent_records[user_id]
            
        return {
            "status": "deleted",
            "user_id": user_id,
            "deletion_date": datetime.utcnow().isoformat()
        }
```

### 2. HIPAA Compliance

```python
# hipaa_compliance.py
import logging
from typing import Dict, Any
from cryptography.fernet import Fernet

class HIPAACompliance:
    def __init__(self):
        self.audit_logger = self.setup_audit_logging()
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
    def setup_audit_logging(self):
        """Setup HIPAA-compliant audit logging"""
        logger = logging.getLogger('hipaa_audit')
        handler = logging.FileHandler('/secure/logs/hipaa_audit.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
        
    def log_access(self, user_id: str, patient_id: str, action: str, data_accessed: str):
        """Log all PHI access"""
        self.audit_logger.info(json.dumps({
            "event": "phi_access",
            "user_id": user_id,
            "patient_id": patient_id,
            "action": action,
            "data_accessed": data_accessed,
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": self.get_client_ip()
        }))
        
    def encrypt_phi(self, data: str) -> str:
        """Encrypt Protected Health Information"""
        return self.cipher.encrypt(data.encode()).decode()
        
    def decrypt_phi(self, encrypted_data: str) -> str:
        """Decrypt Protected Health Information"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
        
    def validate_minimum_necessary(self, user_role: str, requested_data: List[str]) -> List[str]:
        """Ensure minimum necessary standard"""
        role_permissions = {
            "physician": ["demographics", "medical_history", "medications", "lab_results"],
            "nurse": ["demographics", "vitals", "medications"],
            "billing": ["demographics", "insurance", "procedures"],
            "researcher": ["anonymized_data"]
        }
        
        allowed = role_permissions.get(user_role, [])
        return [field for field in requested_data if field in allowed]
```

### 3. SOC 2 Compliance

```yaml
# soc2_controls.yaml
security_controls:
  logical_access:
    - unique_user_ids: true
    - password_policy:
        min_length: 12
        complexity: true
        rotation_days: 90
    - mfa_required: true
    - session_timeout: 30
    
  data_encryption:
    - at_rest: AES-256
    - in_transit: TLS 1.2+
    - key_management: HSM
    
  change_management:
    - code_review_required: true
    - testing_required: true
    - approval_required: true
    - rollback_plan: true
    
  incident_response:
    - detection_time: "< 1 hour"
    - response_time: "< 4 hours"
    - notification_time: "< 24 hours"
    
  backup_recovery:
    - backup_frequency: daily
    - retention_period: 30_days
    - recovery_time_objective: 4_hours
    - recovery_point_objective: 24_hours
```

## Audit and Logging

### 1. Comprehensive Audit System

```python
# audit_system.py
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class AuditEvent:
    timestamp: str
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    result: str
    metadata: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    def calculate_hash(self) -> str:
        """Calculate hash for integrity verification"""
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

class AuditLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.setup_logging()
        
    def setup_logging(self):
        """Setup secure audit logging"""
        self.logger = logging.getLogger('audit')
        
        # File handler with rotation
        handler = RotatingFileHandler(
            self.log_path,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )
        
        # Secure formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S.%f'
        )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def log_event(self, event: AuditEvent):
        """Log audit event with integrity check"""
        event_dict = asdict(event)
        event_dict['hash'] = event.calculate_hash()
        
        # Log to file
        self.logger.info(json.dumps(event_dict))
        
        # Also send to SIEM if configured
        self.send_to_siem(event_dict)
        
    def log_data_access(self, user_id: str, dataset: str, query: str, rows_returned: int):
        """Log data access events"""
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="data_access",
            user_id=user_id,
            resource=dataset,
            action="query",
            result="success",
            metadata={
                "query": query,
                "rows_returned": rows_returned
            }
        )
        self.log_event(event)
```

### 2. Security Event Monitoring

```python
# security_monitoring.py
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class SecurityMonitor:
    def __init__(self):
        self.failed_login_attempts = defaultdict(list)
        self.anomaly_detectors = []
        self.alert_handlers = []
        
    async def monitor_failed_logins(self, user_id: str, ip_address: str):
        """Monitor failed login attempts"""
        key = f"{user_id}:{ip_address}"
        now = datetime.utcnow()
        
        # Add failed attempt
        self.failed_login_attempts[key].append(now)
        
        # Clean old attempts
        cutoff = now - timedelta(minutes=15)
        self.failed_login_attempts[key] = [
            attempt for attempt in self.failed_login_attempts[key]
            if attempt > cutoff
        ]
        
        # Check threshold
        if len(self.failed_login_attempts[key]) >= 5:
            await self.trigger_alert({
                "type": "excessive_failed_logins",
                "user_id": user_id,
                "ip_address": ip_address,
                "attempts": len(self.failed_login_attempts[key]),
                "severity": "high"
            })
            
    async def detect_data_exfiltration(self, user_id: str, bytes_downloaded: int):
        """Detect potential data exfiltration"""
        # Track download volumes
        if bytes_downloaded > 1_000_000_000:  # 1GB
            await self.trigger_alert({
                "type": "potential_data_exfiltration",
                "user_id": user_id,
                "bytes_downloaded": bytes_downloaded,
                "severity": "critical"
            })
            
    async def trigger_alert(self, alert: Dict[str, Any]):
        """Trigger security alert"""
        alert["timestamp"] = datetime.utcnow().isoformat()
        
        # Log alert
        logging.critical(f"SECURITY ALERT: {json.dumps(alert)}")
        
        # Execute alert handlers
        for handler in self.alert_handlers:
            await handler(alert)
```

## Incident Response

### 1. Incident Response Plan

```python
# incident_response.py
from enum import Enum
from typing import List, Dict, Any
import subprocess

class IncidentSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class IncidentType(Enum):
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALWARE = "malware"
    DDOS = "ddos"
    SYSTEM_COMPROMISE = "system_compromise"

class IncidentResponse:
    def __init__(self):
        self.playbooks = self.load_playbooks()
        
    def handle_incident(self, incident_type: IncidentType, severity: IncidentSeverity):
        """Execute incident response playbook"""
        playbook = self.playbooks.get(incident_type)
        
        if not playbook:
            self.log_error(f"No playbook found for {incident_type}")
            return
            
        # Execute playbook steps
        for step in playbook['steps']:
            self.execute_step(step, severity)
            
    def execute_step(self, step: Dict[str, Any], severity: IncidentSeverity):
        """Execute individual playbook step"""
        if step['type'] == 'isolate':
            self.isolate_system(step['target'])
        elif step['type'] == 'snapshot':
            self.create_forensic_snapshot(step['target'])
        elif step['type'] == 'notify':
            self.notify_stakeholders(step['recipients'], severity)
        elif step['type'] == 'block':
            self.block_network_access(step['target'])
            
    def isolate_system(self, system_id: str):
        """Isolate compromised system"""
        commands = [
            f"iptables -I INPUT -s {system_id} -j DROP",
            f"iptables -I OUTPUT -d {system_id} -j DROP"
        ]
        for cmd in commands:
            subprocess.run(cmd.split(), check=True)
            
    def create_forensic_snapshot(self, system_id: str):
        """Create forensic snapshot for investigation"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"forensic_{system_id}_{timestamp}"
        
        # Create memory dump
        subprocess.run([
            "sudo", "dd", 
            f"if=/proc/{system_id}/mem",
            f"of=/forensics/{snapshot_name}_memory.dump"
        ])
        
        # Create disk image
        subprocess.run([
            "sudo", "dd",
            f"if=/dev/disk/{system_id}",
            f"of=/forensics/{snapshot_name}_disk.img"
        ])
```

### 2. Incident Response Playbooks

```yaml
# incident_playbooks.yaml
playbooks:
  data_breach:
    steps:
      - type: isolate
        target: affected_systems
        description: "Isolate affected systems from network"
        
      - type: snapshot
        target: affected_systems
        description: "Create forensic snapshots"
        
      - type: notify
        recipients: ["security_team", "legal", "executives"]
        description: "Notify key stakeholders"
        
      - type: investigate
        actions:
          - analyze_logs
          - identify_breach_vector
          - determine_data_affected
          
      - type: contain
        actions:
          - patch_vulnerability
          - reset_credentials
          - update_firewall_rules
          
      - type: recover
        actions:
          - restore_from_backup
          - verify_integrity
          - monitor_for_reoccurrence
          
      - type: document
        requirements:
          - timeline_of_events
          - systems_affected
          - data_compromised
          - remediation_steps
          
  unauthorized_access:
    steps:
      - type: block
        target: source_ip
        description: "Block source IP address"
        
      - type: disable
        target: compromised_account
        description: "Disable compromised user account"
        
      - type: audit
        scope: access_logs
        timeframe: "last_7_days"
        description: "Audit access logs for suspicious activity"
```

## Security Hardening

### 1. System Hardening Script

```bash
#!/bin/bash
# system_hardening.sh

echo "Starting system hardening..."

# Update system
apt-get update && apt-get upgrade -y

# Install security tools
apt-get install -y \
    fail2ban \
    aide \
    rkhunter \
    lynis \
    auditd

# Kernel hardening
cat >> /etc/sysctl.conf << EOF
# IP Spoofing protection
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0

# Log Martians
net.ipv4.conf.all.log_martians = 1

# Ignore ICMP ping requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Syn flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5
EOF

sysctl -p

# Secure shared memory
echo "tmpfs /run/shm tmpfs defaults,noexec,nosuid 0 0" >> /etc/fstab

# Configure fail2ban
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true

[apache]
enabled = true

[nginx-http-auth]
enabled = true
EOF

systemctl enable fail2ban
systemctl start fail2ban

# Configure auditd
cat >> /etc/audit/audit.rules << EOF
# Monitor user/group modifications
-w /etc/passwd -p wa -k passwd_changes
-w /etc/group -p wa -k group_changes
-w /etc/shadow -p wa -k shadow_changes

# Monitor sudo usage
-w /etc/sudoers -p wa -k sudoers_changes
-w /usr/bin/sudo -p x -k sudo_usage

# Monitor system calls
-a exit,always -F arch=b64 -S execve -k exec_tracking
EOF

systemctl enable auditd
systemctl start auditd

# Disable unnecessary services
systemctl disable bluetooth
systemctl disable cups
systemctl disable avahi-daemon

# Set secure permissions
chmod 644 /etc/passwd
chmod 644 /etc/group
chmod 600 /etc/shadow
chmod 600 /etc/gshadow

echo "System hardening complete!"
```

### 2. Application Hardening

```python
# app_hardening.py
from functools import wraps
import secrets
from typing import Callable
import bleach

class SecurityMiddleware:
    def __init__(self, app):
        self.app = app
        
    def __call__(self, environ, start_response):
        # Add security headers
        def custom_start_response(status, headers):
            security_headers = [
                ('X-Content-Type-Options', 'nosniff'),
                ('X-Frame-Options', 'DENY'),
                ('X-XSS-Protection', '1; mode=block'),
                ('Strict-Transport-Security', 'max-age=31536000; includeSubDomains'),
                ('Content-Security-Policy', "default-src 'self'"),
                ('Referrer-Policy', 'strict-origin-when-cross-origin')
            ]
            headers.extend(security_headers)
            return start_response(status, headers)
            
        return self.app(environ, custom_start_response)

def sanitize_input(func: Callable) -> Callable:
    """Decorator to sanitize user input"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Sanitize string arguments
        sanitized_args = []
        for arg in args:
            if isinstance(arg, str):
                # Remove potential XSS
                arg = bleach.clean(arg)
                # Remove SQL injection attempts
                arg = arg.replace("'", "''")
                arg = arg.replace(";", "")
                arg = arg.replace("--", "")
            sanitized_args.append(arg)
            
        # Sanitize keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, str):
                kwargs[key] = bleach.clean(value)
                kwargs[key] = kwargs[key].replace("'", "''")
                
        return func(*sanitized_args, **kwargs)
    return wrapper

class SessionSecurity:
    def __init__(self):
        self.sessions = {}
        
    def create_secure_session(self, user_id: str) -> str:
        """Create cryptographically secure session"""
        session_id = secrets.token_urlsafe(32)
        
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'ip_address': self.get_client_ip(),
            'user_agent': self.get_user_agent()
        }
        
        return session_id
        
    def validate_session(self, session_id: str) -> bool:
        """Validate session with multiple checks"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        
        # Check session timeout (30 minutes)
        if datetime.utcnow() - session['last_activity'] > timedelta(minutes=30):
            del self.sessions[session_id]
            return False
            
        # Check IP address consistency
        if session['ip_address'] != self.get_client_ip():
            self.log_security_event("ip_mismatch", session_id)
            del self.sessions[session_id]
            return False
            
        # Update last activity
        session['last_activity'] = datetime.utcnow()
        return True
```

## Compliance Checklists

### 1. Daily Security Checklist

```yaml
# daily_security_checklist.yaml
daily_tasks:
  - task: "Review security alerts"
    responsible: "Security Team"
    sla: "09:00"
    
  - task: "Check failed login attempts"
    responsible: "Security Team"
    sla: "09:30"
    
  - task: "Verify backup completion"
    responsible: "Operations"
    sla: "10:00"
    
  - task: "Review system logs for anomalies"
    responsible: "Security Team"
    sla: "11:00"
    
  - task: "Check certificate expiration"
    responsible: "Operations"
    sla: "14:00"
    
  - task: "Verify patch status"
    responsible: "Operations"
    sla: "15:00"
    
  - task: "Review access control changes"
    responsible: "Security Team"
    sla: "16:00"
```

### 2. Monthly Compliance Review

```python
# compliance_review.py
class ComplianceReview:
    def __init__(self):
        self.checks = []
        
    def run_monthly_review(self) -> Dict[str, Any]:
        """Run comprehensive monthly compliance review"""
        results = {
            "review_date": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Access control review
        results["checks"]["access_control"] = self.review_access_control()
        
        # Data retention compliance
        results["checks"]["data_retention"] = self.check_data_retention()
        
        # Encryption verification
        results["checks"]["encryption"] = self.verify_encryption()
        
        # Audit log integrity
        results["checks"]["audit_logs"] = self.verify_audit_logs()
        
        # Vulnerability scan results
        results["checks"]["vulnerabilities"] = self.run_vulnerability_scan()
        
        # Generate report
        self.generate_compliance_report(results)
        
        return results
        
    def review_access_control(self) -> Dict[str, Any]:
        """Review user access and permissions"""
        return {
            "total_users": self.count_users(),
            "inactive_users": self.find_inactive_users(days=90),
            "excessive_permissions": self.find_excessive_permissions(),
            "service_accounts": self.audit_service_accounts()
        }
        
    def check_data_retention(self) -> Dict[str, Any]:
        """Verify data retention policies"""
        return {
            "expired_data": self.find_expired_data(),
            "retention_violations": self.check_retention_violations(),
            "deletion_confirmations": self.verify_deletions()
        }
```

## Best Practices

### 1. Security Development Lifecycle

```yaml
# sdl_practices.yaml
development:
  code_review:
    - security_focused: true
    - automated_scanning: true
    - peer_review_required: true
    
  testing:
    - unit_tests: required
    - integration_tests: required
    - security_tests: required
    - penetration_tests: quarterly
    
  dependencies:
    - vulnerability_scanning: true
    - license_compliance: true
    - update_frequency: monthly

deployment:
  pre_deployment:
    - security_scan: required
    - compliance_check: required
    - risk_assessment: required
    
  deployment:
    - blue_green: true
    - rollback_plan: required
    - monitoring_enabled: true
    
  post_deployment:
    - security_validation: true
    - performance_monitoring: true
    - incident_response_ready: true
```

### 2. Security Training Program

```python
# security_training.py
class SecurityTraining:
    def __init__(self):
        self.training_modules = {
            "basic_security": {
                "duration": "2 hours",
                "topics": [
                    "Password security",
                    "Phishing awareness",
                    "Data handling",
                    "Incident reporting"
                ],
                "frequency": "annual"
            },
            "developer_security": {
                "duration": "8 hours",
                "topics": [
                    "Secure coding practices",
                    "OWASP Top 10",
                    "Security testing",
                    "Threat modeling"
                ],
                "frequency": "bi-annual"
            },
            "data_privacy": {
                "duration": "4 hours",
                "topics": [
                    "GDPR requirements",
                    "Data classification",
                    "Privacy by design",
                    "Data breach response"
                ],
                "frequency": "annual"
            }
        }
        
    def assign_training(self, user_role: str) -> List[str]:
        """Assign training based on role"""
        role_training = {
            "developer": ["basic_security", "developer_security", "data_privacy"],
            "analyst": ["basic_security", "data_privacy"],
            "admin": ["basic_security", "developer_security", "data_privacy"]
        }
        return role_training.get(user_role, ["basic_security"])
```

### 3. Security Metrics Dashboard

```python
# security_metrics.py
class SecurityMetrics:
    def __init__(self):
        self.metrics = {}
        
    def calculate_security_score(self) -> float:
        """Calculate overall security posture score"""
        scores = {
            "patch_compliance": self.get_patch_compliance_score(),
            "vulnerability_management": self.get_vulnerability_score(),
            "access_control": self.get_access_control_score(),
            "incident_response": self.get_incident_response_score(),
            "training_completion": self.get_training_score()
        }
        
        weights = {
            "patch_compliance": 0.25,
            "vulnerability_management": 0.25,
            "access_control": 0.20,
            "incident_response": 0.20,
            "training_completion": 0.10
        }
        
        total_score = sum(
            scores[metric] * weights[metric] 
            for metric in scores
        )
        
        return round(total_score, 2)
        
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for security dashboard"""
        return {
            "overall_score": self.calculate_security_score(),
            "critical_vulnerabilities": self.count_critical_vulnerabilities(),
            "failed_login_attempts": self.get_failed_login_stats(),
            "patch_compliance": f"{self.get_patch_compliance_score()}%",
            "incidents_mttr": self.calculate_mttr(),
            "training_completion": f"{self.get_training_score()}%",
            "audit_findings": self.get_audit_findings_count(),
            "last_update": datetime.utcnow().isoformat()
        }
```

## Conclusion

This comprehensive security and compliance guide provides:

1. **Defense-in-depth security architecture**
2. **Data protection through encryption and anonymization**
3. **Robust access control and authentication**
4. **Network security and segmentation**
5. **Compliance with major frameworks (GDPR, HIPAA, SOC 2)**
6. **Comprehensive audit and logging**
7. **Incident response procedures**
8. **Security hardening guidelines**
9. **Regular compliance reviews**
10. **Security best practices and training**

Regular review and updates of these security measures ensure continued protection of synthetic data systems and compliance with evolving regulations.