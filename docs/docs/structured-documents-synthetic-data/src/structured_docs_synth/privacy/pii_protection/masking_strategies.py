"""
Advanced data masking strategies for privacy protection
"""

import re
import random
import string
import hashlib
import hmac
import base64
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import secrets

from ...core import get_logger, PrivacyError
from .pii_detector import PIIType, PIIDetectionResult


class MaskingMethod(Enum):
    """Types of masking methods available"""
    FULL_MASK = "full_mask"
    PARTIAL_MASK = "partial_mask"
    FORMAT_PRESERVING = "format_preserving"
    SUBSTITUTION = "substitution"
    GENERALIZATION = "generalization"
    SUPPRESSION = "suppression"
    TOKENIZATION = "tokenization"
    HASHING = "hashing"
    ENCRYPTION = "encryption"
    PERTURBATION = "perturbation"
    SYNTHETIC_REPLACEMENT = "synthetic_replacement"


class PreservationLevel(Enum):
    """Level of data structure/format preservation"""
    NONE = "none"
    FORMAT_ONLY = "format_only"
    LENGTH_ONLY = "length_only"
    PATTERN_ONLY = "pattern_only"
    FULL_STRUCTURE = "full_structure"


@dataclass
class MaskingRule:
    """Configuration for a specific masking operation"""
    pii_type: PIIType
    method: MaskingMethod
    preservation_level: PreservationLevel
    mask_character: str = "*"
    reveal_positions: List[int] = None  # Positions to leave unmasked
    custom_function: Optional[Callable] = None
    parameters: Dict[str, Any] = None


@dataclass
class MaskingResult:
    """Result of a masking operation"""
    original_value: str
    masked_value: str
    method_used: MaskingMethod
    preservation_level: PreservationLevel
    is_reversible: bool
    metadata: Dict[str, Any] = None


class MaskingStrategies:
    """Advanced data masking strategies implementation"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.logger = get_logger(__name__)
        
        # Encryption key for reversible masking
        self.encryption_key = encryption_key or secrets.token_bytes(32)
        
        # Token mapping for consistent tokenization
        self.token_mapping = {}
        
        # Substitution dictionaries
        self.substitution_maps = self._initialize_substitution_maps()
        
        # Format patterns for different PII types
        self.format_patterns = self._initialize_format_patterns()
        
        self.logger.info("Masking Strategies initialized")
    
    def _initialize_substitution_maps(self) -> Dict[PIIType, List[str]]:
        """Initialize substitution mappings for different PII types"""
        
        return {
            PIIType.NAME: [
                "John Smith", "Jane Doe", "Michael Johnson", "Sarah Wilson",
                "David Brown", "Emily Davis", "Christopher Miller", "Jessica Garcia"
            ],
            PIIType.EMAIL: [
                "user1@example.com", "user2@example.com", "contact@company.com",
                "info@business.org", "admin@service.net"
            ],
            PIIType.PHONE: [
                "(555) 123-4567", "(555) 987-6543", "(555) 246-8135",
                "(555) 369-2580", "(555) 147-2589"
            ],
            PIIType.ADDRESS: [
                "123 Main St, Anytown, ST 12345",
                "456 Oak Ave, Somewhere, ST 67890",
                "789 Pine Rd, Elsewhere, ST 54321"
            ]
        }
    
    def _initialize_format_patterns(self) -> Dict[PIIType, Dict[str, str]]:
        """Initialize format patterns for different PII types"""
        
        return {
            PIIType.SSN: {
                "pattern": r"(\d{3})-(\d{2})-(\d{4})",
                "format": "{}-{}-{}"
            },
            PIIType.CREDIT_CARD: {
                "pattern": r"(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})",
                "format": "{}-{}-{}-{}"
            },
            PIIType.PHONE: {
                "pattern": r"\((\d{3})\)\s*(\d{3})-(\d{4})",
                "format": "({}) {}-{}"
            },
            PIIType.DATE_OF_BIRTH: {
                "pattern": r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
                "format": "{}/{}/{}"
            }
        }
    
    def apply_masking_rule(self, value: str, rule: MaskingRule) -> MaskingResult:
        """Apply a specific masking rule to a value"""
        
        try:
            if rule.custom_function:
                masked_value = rule.custom_function(value, rule)
                method_used = MaskingMethod.SUBSTITUTION
            else:
                masked_value, method_used = self._apply_method(value, rule)
            
            is_reversible = method_used in [
                MaskingMethod.TOKENIZATION, 
                MaskingMethod.ENCRYPTION
            ]
            
            result = MaskingResult(
                original_value=value,
                masked_value=masked_value,
                method_used=method_used,
                preservation_level=rule.preservation_level,
                is_reversible=is_reversible,
                metadata={
                    "pii_type": rule.pii_type.value,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.logger.debug(f"Applied {method_used.value} masking to {rule.pii_type.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying masking rule: {str(e)}")
            raise PrivacyError(
                f"Masking failed for {rule.pii_type.value}: {str(e)}",
                details={"value_length": len(value), "method": rule.method.value}
            )
    
    def _apply_method(self, value: str, rule: MaskingRule) -> tuple[str, MaskingMethod]:
        """Apply the specified masking method"""
        
        method_map = {
            MaskingMethod.FULL_MASK: self._full_mask,
            MaskingMethod.PARTIAL_MASK: self._partial_mask,
            MaskingMethod.FORMAT_PRESERVING: self._format_preserving_mask,
            MaskingMethod.SUBSTITUTION: self._substitution_mask,
            MaskingMethod.GENERALIZATION: self._generalization_mask,
            MaskingMethod.SUPPRESSION: self._suppression_mask,
            MaskingMethod.TOKENIZATION: self._tokenization_mask,
            MaskingMethod.HASHING: self._hashing_mask,
            MaskingMethod.ENCRYPTION: self._encryption_mask,
            MaskingMethod.PERTURBATION: self._perturbation_mask,
            MaskingMethod.SYNTHETIC_REPLACEMENT: self._synthetic_replacement_mask
        }
        
        if rule.method not in method_map:
            raise PrivacyError(f"Unsupported masking method: {rule.method.value}")
        
        masked_value = method_map[rule.method](value, rule)
        return masked_value, rule.method
    
    def _full_mask(self, value: str, rule: MaskingRule) -> str:
        """Replace entire value with mask characters"""
        
        if rule.preservation_level == PreservationLevel.LENGTH_ONLY:
            return rule.mask_character * len(value)
        elif rule.preservation_level == PreservationLevel.FORMAT_ONLY:
            return self._preserve_format(value, rule.mask_character)
        else:
            return rule.mask_character * 8  # Default length
    
    def _partial_mask(self, value: str, rule: MaskingRule) -> str:
        """Mask part of the value, preserving some characters"""
        
        if not value:
            return value
        
        if rule.reveal_positions:
            # Mask all except specified positions
            result = list(rule.mask_character * len(value))
            for pos in rule.reveal_positions:
                if 0 <= pos < len(value):
                    result[pos] = value[pos]
            return ''.join(result)
        
        # Default partial masking based on PII type
        if rule.pii_type == PIIType.SSN:
            # Show last 4 digits: ***-**-1234
            if len(value) >= 4:
                return rule.mask_character * (len(value) - 4) + value[-4:]
        elif rule.pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits: ****-****-****-1234
            if len(value) >= 4:
                masked_part = re.sub(r'\d', rule.mask_character, value[:-4])
                return masked_part + value[-4:]
        elif rule.pii_type == PIIType.EMAIL:
            # Mask username: ****@domain.com
            if '@' in value:
                username, domain = value.rsplit('@', 1)
                return rule.mask_character * len(username) + '@' + domain
        elif rule.pii_type == PIIType.PHONE:
            # Show last 4 digits: (***) ***-1234
            digits_only = re.sub(r'\D', '', value)
            if len(digits_only) >= 4:
                pattern = re.sub(r'\d', rule.mask_character, value)
                # Replace last 4 mask characters with actual digits
                last_4 = digits_only[-4:]
                for digit in reversed(last_4):
                    pattern = pattern[::-1].replace(rule.mask_character, digit, 1)[::-1]
                return pattern
        
        # Default: mask first half
        mid_point = len(value) // 2
        return rule.mask_character * mid_point + value[mid_point:]
    
    def _format_preserving_mask(self, value: str, rule: MaskingRule) -> str:
        """Preserve the format/pattern while masking data"""
        
        if rule.pii_type in self.format_patterns:
            pattern_info = self.format_patterns[rule.pii_type]
            pattern = pattern_info["pattern"]
            format_str = pattern_info["format"]
            
            match = re.match(pattern, value)
            if match:
                # Generate replacement digits/characters
                groups = []
                for group in match.groups():
                    if group.isdigit():
                        # Replace with random digits
                        replacement = ''.join(random.choices(string.digits, k=len(group)))
                    else:
                        # Replace with random characters
                        replacement = ''.join(random.choices(string.ascii_letters, k=len(group)))
                    groups.append(replacement)
                
                return format_str.format(*groups)
        
        # Fallback: preserve character types
        return self._preserve_character_types(value)
    
    def _preserve_format(self, value: str, mask_char: str) -> str:
        """Preserve the format structure of the original value"""
        
        result = []
        for char in value:
            if char.isalnum():
                result.append(mask_char)
            else:
                result.append(char)  # Preserve special characters
        
        return ''.join(result)
    
    def _preserve_character_types(self, value: str) -> str:
        """Preserve character types (digit/letter/special) while randomizing"""
        
        result = []
        for char in value:
            if char.isdigit():
                result.append(random.choice(string.digits))
            elif char.isalpha():
                if char.isupper():
                    result.append(random.choice(string.ascii_uppercase))
                else:
                    result.append(random.choice(string.ascii_lowercase))
            else:
                result.append(char)  # Preserve special characters
        
        return ''.join(result)
    
    def _substitution_mask(self, value: str, rule: MaskingRule) -> str:
        """Replace with a value from substitution dictionary"""
        
        if rule.pii_type in self.substitution_maps:
            substitutes = self.substitution_maps[rule.pii_type]
            return random.choice(substitutes)
        
        # Fallback to synthetic replacement
        return self._synthetic_replacement_mask(value, rule)
    
    def _generalization_mask(self, value: str, rule: MaskingRule) -> str:
        """Generalize the value to a broader category"""
        
        if rule.pii_type == PIIType.DATE_OF_BIRTH:
            # Generalize to birth year only
            date_match = re.search(r'\b(19|20)\d{2}\b', value)
            if date_match:
                year = int(date_match.group())
                # Generalize to decade
                decade = (year // 10) * 10
                return f"{decade}s"
        
        elif rule.pii_type == PIIType.ADDRESS:
            # Generalize to city/state only
            # Simple heuristic: keep last two parts (city, state)
            parts = value.split(',')
            if len(parts) >= 2:
                return ', '.join(parts[-2:]).strip()
        
        elif rule.pii_type == PIIType.PHONE:
            # Generalize to area code only
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 3:
                return f"({digits[:3]}) ***-****"
        
        elif rule.pii_type == PIIType.EMAIL:
            # Generalize to domain only
            if '@' in value:
                return f"*@{value.split('@')[1]}"
        
        # Default: replace with category name
        return f"<{rule.pii_type.value.upper()}>"
    
    def _suppression_mask(self, value: str, rule: MaskingRule) -> str:
        """Completely suppress/remove the value"""
        
        if rule.preservation_level == PreservationLevel.LENGTH_ONLY:
            return " " * len(value)  # Preserve length with spaces
        elif rule.preservation_level == PreservationLevel.FORMAT_ONLY:
            return "[SUPPRESSED]"
        else:
            return ""  # Complete removal
    
    def _tokenization_mask(self, value: str, rule: MaskingRule) -> str:
        """Replace with a consistent token (reversible)"""
        
        # Generate a hash-based token
        token_key = f"{rule.pii_type.value}:{value}"
        
        if token_key in self.token_mapping:
            return self.token_mapping[token_key]
        
        # Generate new token
        hash_obj = hashlib.sha256()
        hash_obj.update(token_key.encode('utf-8'))
        hash_obj.update(self.encryption_key)
        token_hash = hash_obj.hexdigest()[:16]
        
        if rule.preservation_level == PreservationLevel.FORMAT_ONLY:
            # Preserve format while tokenizing
            token = self._format_token(token_hash, value, rule)
        else:
            token = f"TKN_{rule.pii_type.value.upper()}_{token_hash}"
        
        self.token_mapping[token_key] = token
        return token
    
    def _format_token(self, token_hash: str, original: str, rule: MaskingRule) -> str:
        """Format a token to match original structure"""
        
        if rule.pii_type == PIIType.SSN:
            # Format as XXX-XX-XXXX
            digits = ''.join(c for c in token_hash if c.isdigit())[:9]
            if len(digits) < 9:
                digits += '0' * (9 - len(digits))
            return f"{digits[:3]}-{digits[3:5]}-{digits[5:9]}"
        
        elif rule.pii_type == PIIType.PHONE:
            # Format as (XXX) XXX-XXXX
            digits = ''.join(c for c in token_hash if c.isdigit())[:10]
            if len(digits) < 10:
                digits += '0' * (10 - len(digits))
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:10]}"
        
        elif rule.pii_type == PIIType.EMAIL:
            # Format as token@domain.com
            if '@' in original:
                domain = original.split('@')[1]
                return f"user{token_hash[:8]}@{domain}"
        
        return token_hash
    
    def _hashing_mask(self, value: str, rule: MaskingRule) -> str:
        """Replace with a cryptographic hash (irreversible)"""
        
        # Use HMAC for additional security
        hash_obj = hmac.new(
            self.encryption_key,
            f"{rule.pii_type.value}:{value}".encode('utf-8'),
            hashlib.sha256
        )
        
        hash_value = hash_obj.hexdigest()
        
        if rule.preservation_level == PreservationLevel.LENGTH_ONLY:
            # Truncate or pad to match original length
            if len(hash_value) > len(value):
                return hash_value[:len(value)]
            else:
                return hash_value.ljust(len(value), '0')
        
        return f"HASH_{hash_value[:16]}"
    
    def _encryption_mask(self, value: str, rule: MaskingRule) -> str:
        """Encrypt the value (reversible with key)"""
        
        try:
            from cryptography.fernet import Fernet
            
            # Generate key from our encryption key
            key = base64.urlsafe_b64encode(self.encryption_key[:32])
            fernet = Fernet(key)
            
            # Encrypt the value
            encrypted = fernet.encrypt(value.encode('utf-8'))
            encrypted_str = base64.urlsafe_b64encode(encrypted).decode('utf-8')
            
            if rule.preservation_level == PreservationLevel.FORMAT_ONLY:
                return f"ENC_{encrypted_str[:16]}"
            
            return encrypted_str
            
        except ImportError:
            self.logger.warning("Cryptography library not available, falling back to hashing")
            return self._hashing_mask(value, rule)
    
    def _perturbation_mask(self, value: str, rule: MaskingRule) -> str:
        """Add noise/perturbation to numerical values"""
        
        # Extract numbers from the value
        numbers = re.findall(r'\d+', value)
        
        if not numbers:
            return self._substitution_mask(value, rule)
        
        perturbed_value = value
        
        for num_str in numbers:
            original_num = int(num_str)
            
            # Add random noise (±10% of value)
            noise_range = max(1, original_num // 10)
            noise = random.randint(-noise_range, noise_range)
            perturbed_num = max(0, original_num + noise)
            
            # Replace in string
            perturbed_value = perturbed_value.replace(num_str, str(perturbed_num), 1)
        
        return perturbed_value
    
    def _synthetic_replacement_mask(self, value: str, rule: MaskingRule) -> str:
        """Replace with synthetically generated data"""
        
        try:
            from faker import Faker
            fake = Faker()
            
            replacements = {
                PIIType.NAME: fake.name(),
                PIIType.EMAIL: fake.email(),
                PIIType.PHONE: fake.phone_number(),
                PIIType.ADDRESS: fake.address().replace('\n', ', '),
                PIIType.SSN: fake.ssn(),
                PIIType.DATE_OF_BIRTH: fake.date_of_birth().strftime('%m/%d/%Y'),
                PIIType.CREDIT_CARD: fake.credit_card_number(),
                PIIType.BANK_ACCOUNT: str(fake.random_number(digits=12))
            }
            
            if rule.pii_type in replacements:
                return replacements[rule.pii_type]
            
        except ImportError:
            self.logger.warning("Faker library not available, using substitution")
        
        # Fallback to substitution
        return self._substitution_mask(value, rule)
    
    def create_masking_rules(
        self,
        pii_detection_result: PIIDetectionResult,
        default_method: MaskingMethod = MaskingMethod.PARTIAL_MASK,
        preservation_level: PreservationLevel = PreservationLevel.FORMAT_ONLY
    ) -> List[MaskingRule]:
        """Create appropriate masking rules based on detected PII"""
        
        rules = []
        
        # Default rules for different PII types
        default_rules = {
            PIIType.SSN: MaskingRule(
                PIIType.SSN, MaskingMethod.PARTIAL_MASK, 
                PreservationLevel.FORMAT_ONLY, reveal_positions=[-4, -3, -2, -1]
            ),
            PIIType.CREDIT_CARD: MaskingRule(
                PIIType.CREDIT_CARD, MaskingMethod.PARTIAL_MASK,
                PreservationLevel.FORMAT_ONLY, reveal_positions=[-4, -3, -2, -1]
            ),
            PIIType.EMAIL: MaskingRule(
                PIIType.EMAIL, MaskingMethod.PARTIAL_MASK,
                PreservationLevel.FORMAT_ONLY
            ),
            PIIType.PHONE: MaskingRule(
                PIIType.PHONE, MaskingMethod.PARTIAL_MASK,
                PreservationLevel.FORMAT_ONLY
            ),
            PIIType.NAME: MaskingRule(
                PIIType.NAME, MaskingMethod.SYNTHETIC_REPLACEMENT,
                PreservationLevel.NONE
            ),
            PIIType.ADDRESS: MaskingRule(
                PIIType.ADDRESS, MaskingMethod.GENERALIZATION,
                PreservationLevel.PATTERN_ONLY
            ),
            PIIType.DATE_OF_BIRTH: MaskingRule(
                PIIType.DATE_OF_BIRTH, MaskingMethod.GENERALIZATION,
                PreservationLevel.PATTERN_ONLY
            ),
            PIIType.MEDICAL_RECORD: MaskingRule(
                PIIType.MEDICAL_RECORD, MaskingMethod.TOKENIZATION,
                PreservationLevel.FORMAT_ONLY
            ),
            PIIType.BANK_ACCOUNT: MaskingRule(
                PIIType.BANK_ACCOUNT, MaskingMethod.HASHING,
                PreservationLevel.NONE
            )
        }
        
        # Create rules for detected PII types
        for pii_type in pii_detection_result.pii_types_found:
            if pii_type in default_rules:
                rules.append(default_rules[pii_type])
            else:
                # Create default rule
                rules.append(MaskingRule(
                    pii_type, default_method, preservation_level
                ))
        
        return rules
    
    def unmask_value(self, masked_value: str, masking_result: MaskingResult) -> Optional[str]:
        """Attempt to unmask a value (only works for reversible methods)"""
        
        if not masking_result.is_reversible:
            self.logger.warning("Cannot unmask irreversible masking result")
            return None
        
        try:
            if masking_result.method_used == MaskingMethod.TOKENIZATION:
                # Look up in token mapping
                for key, token in self.token_mapping.items():
                    if token == masked_value:
                        original_value = key.split(':', 1)[1]
                        return original_value
            
            elif masking_result.method_used == MaskingMethod.ENCRYPTION:
                try:
                    from cryptography.fernet import Fernet
                    
                    key = base64.urlsafe_b64encode(self.encryption_key[:32])
                    fernet = Fernet(key)
                    
                    # Handle different encryption formats
                    if masked_value.startswith("ENC_"):
                        self.logger.warning("Partial encryption format cannot be fully reversed")
                        return None
                    
                    # Decrypt
                    encrypted_bytes = base64.urlsafe_b64decode(masked_value.encode('utf-8'))
                    decrypted = fernet.decrypt(encrypted_bytes)
                    return decrypted.decode('utf-8')
                    
                except ImportError:
                    self.logger.error("Cryptography library not available for decryption")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error unmasking value: {str(e)}")
            return None
    
    def get_masking_statistics(self) -> Dict[str, Any]:
        """Get statistics about masking operations"""
        
        return {
            "total_tokens": len(self.token_mapping),
            "available_methods": [method.value for method in MaskingMethod],
            "preservation_levels": [level.value for level in PreservationLevel],
            "supported_pii_types": [pii_type.value for pii_type in PIIType],
            "substitution_map_sizes": {
                pii_type.value: len(substitutes) 
                for pii_type, substitutes in self.substitution_maps.items()
            }
        }