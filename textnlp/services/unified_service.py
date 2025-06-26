"""
Unified TextNLP Service

Integrates all safety, metrics, and generation components into a single service
"""

import asyncio
import logging
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# Safety imports
from ..safety import (
    PIIDetector, ToxicityClassifier, BiasDetector, 
    ComplianceChecker, AuditLogger, AuditLoggerFactory, 
    AuditEvent, AuditEventType
)

# Metrics imports
from ..metrics import (
    MetricsCollector, QualityMetricsCalculator, 
    ResourceTracker, BusinessMetricsCollector
)

# SDK imports
from ..sdk import BaseGenerator

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for unified service"""
    safety_config: Dict[str, Any]
    metrics_config: Dict[str, Any] 
    integration_config: Dict[str, Any]
    
    @classmethod
    def from_files(cls, config_dir: str = "config"):
        """Load configuration from YAML files"""
        config_path = Path(config_dir)
        
        # Load safety config
        safety_config_path = config_path / "safety_config.yaml"
        with open(safety_config_path) as f:
            safety_config = yaml.safe_load(f)
        
        # Load metrics config
        metrics_config_path = config_path / "metrics_config.yaml"
        with open(metrics_config_path) as f:
            metrics_config = yaml.safe_load(f)
            
        # Load integration config
        integration_config_path = config_path / "integration_config.yaml"
        with open(integration_config_path) as f:
            integration_config = yaml.safe_load(f)
        
        return cls(
            safety_config=safety_config,
            metrics_config=metrics_config,
            integration_config=integration_config
        )


class UnifiedTextNLPService:
    """
    Unified service that orchestrates all TextNLP components:
    - Text generation
    - Safety filtering (PII, toxicity, bias, compliance)
    - Quality metrics (BLEU, ROUGE, etc.)
    - Resource monitoring
    - Business analytics
    - Audit logging
    """
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        
        # Initialize components
        self._initialize_safety_components()
        self._initialize_metrics_components()
        self._initialize_generators()
        
        logger.info("Unified TextNLP Service initialized")
    
    def _initialize_safety_components(self):
        """Initialize all safety components"""
        safety_config = self.config.safety_config
        
        # PII Detection
        if safety_config.get("pii_detection", {}).get("enabled", True):
            self.pii_detector = PIIDetector(
                languages=safety_config["pii_detection"].get("languages", ["en"]),
                confidence_threshold=safety_config["pii_detection"].get("confidence_threshold", 0.5),
                mask_mode=safety_config["pii_detection"].get("mask_mode", "replace")
            )
            logger.info("PII detector initialized")
        else:
            self.pii_detector = None
        
        # Toxicity Classification
        if safety_config.get("toxicity_classification", {}).get("enabled", True):
            toxicity_config = safety_config["toxicity_classification"]
            self.toxicity_classifier = ToxicityClassifier(toxicity_config)
            logger.info("Toxicity classifier initialized")
        else:
            self.toxicity_classifier = None
        
        # Bias Detection
        if safety_config.get("bias_detection", {}).get("enabled", True):
            bias_config = safety_config["bias_detection"]
            self.bias_detector = BiasDetector(bias_config)
            logger.info("Bias detector initialized")
        else:
            self.bias_detector = None
        
        # Compliance Checking
        if safety_config.get("compliance_checking", {}).get("enabled", True):
            compliance_config = safety_config["compliance_checking"]
            self.compliance_checker = ComplianceChecker(compliance_config)
            logger.info("Compliance checker initialized")
        else:
            self.compliance_checker = None
        
        # Audit Logging
        if safety_config.get("audit_logging", {}).get("enabled", True):
            audit_config = safety_config["audit_logging"]
            self.audit_logger = AuditLoggerFactory.create_logger(audit_config)
            logger.info("Audit logger initialized")
        else:
            self.audit_logger = None
    
    def _initialize_metrics_components(self):
        """Initialize all metrics components"""
        metrics_config = self.config.metrics_config
        
        # Generation Metrics
        if metrics_config.get("generation_metrics", {}).get("enabled", True):
            gen_config = metrics_config["generation_metrics"]
            self.metrics_collector = MetricsCollector(gen_config)
            logger.info("Generation metrics collector initialized")
        else:
            self.metrics_collector = None
        
        # Quality Metrics
        if metrics_config.get("quality_metrics", {}).get("enabled", True):
            quality_config = metrics_config["quality_metrics"]
            self.quality_calculator = QualityMetricsCalculator(quality_config)
            logger.info("Quality metrics calculator initialized")
        else:
            self.quality_calculator = None
        
        # Resource Tracking
        if metrics_config.get("resource_tracking", {}).get("enabled", True):
            resource_config = metrics_config["resource_tracking"]
            self.resource_tracker = ResourceTracker(resource_config)
            self.resource_tracker.start_monitoring()
            logger.info("Resource tracker initialized and started")
        else:
            self.resource_tracker = None
        
        # Business Metrics
        if metrics_config.get("business_dashboard", {}).get("enabled", True):
            business_config = metrics_config["business_dashboard"]
            self.business_collector = BusinessMetricsCollector(business_config)
            logger.info("Business metrics collector initialized")
        else:
            self.business_collector = None
    
    def _initialize_generators(self):
        """Initialize text generators"""
        # This would integrate with existing SDK generators
        self.generators = {}
        logger.info("Text generators initialized")
    
    async def generate_text(self, prompt: str, model_name: str = "gpt2", 
                           user_id: Optional[str] = None,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate text with comprehensive safety, quality, and metrics tracking
        """
        request_id = f"req_{int(datetime.now().timestamp() * 1000)}"
        session_id = context.get("session_id") if context else None
        
        # Log generation request
        if self.audit_logger:
            event = AuditEvent(
                event_type=AuditEventType.GENERATION_REQUEST,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                component="unified_service",
                action="generate_text",
                resource=model_name,
                message=f"Text generation request for model {model_name}",
                details={
                    "model": model_name,
                    "prompt_length": len(prompt),
                    "user_id": user_id
                }
            )
            self.audit_logger.log_event(event)
        
        # Start metrics tracking
        metrics_context = None
        token_tracker = None
        if self.metrics_collector:
            metrics_context = self.metrics_collector.track_generation(
                request_id, user_id, model_name, prompt
            )
            metrics_context_manager = metrics_context.__enter__()
            metrics, token_tracker = metrics_context_manager
        
        try:
            # Step 1: Input Safety Check
            input_safety_results = await self._check_input_safety(prompt, context)
            
            if not input_safety_results["safe"]:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": "Input failed safety checks",
                    "safety_results": input_safety_results,
                    "generated_text": None
                }
            
            # Step 2: Generate Text (placeholder - integrate with actual generators)
            generated_text = await self._generate_text_internal(prompt, model_name, token_tracker)
            
            if self.metrics_collector and metrics:
                metrics.generated_length = len(generated_text.split())
            
            # Step 3: Output Safety Check
            output_safety_results = await self._check_output_safety(generated_text, context)
            
            # Step 4: Quality Assessment
            quality_results = None
            if self.quality_calculator:
                quality_results = await self.quality_calculator.evaluate_quality(
                    reference=prompt,  # Using prompt as reference for now
                    candidate=generated_text,
                    context=prompt
                )
            
            # Step 5: Compliance Check
            compliance_results = None
            if self.compliance_checker:
                compliance_results = await self.compliance_checker.check_compliance(
                    generated_text,
                    context=context
                )
            
            # Step 6: Record Business Metrics
            if self.business_collector:
                # Usage metrics
                self.business_collector.record_usage_metrics({
                    "tokens_generated": len(generated_text.split()),
                    "model": model_name,
                    "user_id": user_id
                })
                
                # Quality metrics
                if quality_results:
                    self.business_collector.record_quality_metrics({
                        "overall_quality": quality_results.overall_quality
                    })
                
                # Safety metrics
                safety_scores = {}
                if output_safety_results.get("toxicity"):
                    safety_scores["toxicity"] = output_safety_results["toxicity"]["score"]
                if output_safety_results.get("bias"):
                    safety_scores["bias"] = output_safety_results["bias"]["overall_score"]
                
                if safety_scores:
                    self.business_collector.record_safety_metrics({
                        "safety_score": min(safety_scores.values()),
                        "violations": output_safety_results.get("violations", {})
                    })
            
            # Step 7: Update Generation Metrics
            if self.metrics_collector and metrics:
                # Add safety metrics to generation metrics
                self.metrics_collector.add_safety_metrics(metrics, {
                    "toxicity": output_safety_results.get("toxicity", {}).get("score", 0.0),
                    "bias": output_safety_results.get("bias", {}).get("overall_score", 0.0)
                })
                
                # Add text metrics
                if generated_text:
                    text_metrics = self.metrics_collector.calculate_text_metrics(generated_text, prompt)
                    metrics.vocabulary_size = text_metrics.get("vocabulary_size", 0)
                    metrics.repetition_ratio = text_metrics.get("repetition_ratio", 0.0)
            
            # Determine overall success
            is_safe = output_safety_results["safe"]
            is_compliant = compliance_results.is_compliant if compliance_results else True
            success = is_safe and is_compliant
            
            # Log generation completion
            if self.audit_logger:
                event = AuditEvent(
                    event_type=AuditEventType.GENERATION_COMPLETED if success else AuditEventType.GENERATION_FAILED,
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    component="unified_service",
                    action="generate_text",
                    resource=model_name,
                    message=f"Text generation {'completed' if success else 'failed'}",
                    details={
                        "success": success,
                        "generated_length": len(generated_text.split()) if generated_text else 0,
                        "safety_passed": is_safe,
                        "compliance_passed": is_compliant
                    }
                )
                self.audit_logger.log_event(event)
            
            # Return comprehensive results
            return {
                "request_id": request_id,
                "success": success,
                "generated_text": generated_text if success else None,
                "input_safety": input_safety_results,
                "output_safety": output_safety_results,
                "quality_metrics": quality_results.to_dict() if quality_results else None,
                "compliance_results": compliance_results.to_dict() if compliance_results else None,
                "metadata": {
                    "model": model_name,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            
            # Log error
            if self.audit_logger:
                event = AuditEvent(
                    event_type=AuditEventType.GENERATION_FAILED,
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    component="unified_service",
                    action="generate_text",
                    message=f"Text generation failed: {str(e)}",
                    details={"error": str(e)}
                )
                self.audit_logger.log_event(event)
            
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "generated_text": None
            }
        
        finally:
            # Clean up metrics tracking
            if metrics_context:
                metrics_context.__exit__(None, None, None)
    
    async def _check_input_safety(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check input text for safety issues"""
        results = {"safe": True, "violations": []}
        
        # PII Detection
        if self.pii_detector:
            pii_result = await self.pii_detector.detect_pii(text)
            results["pii"] = {
                "detected": len(pii_result.pii_matches) > 0,
                "risk_level": pii_result.risk_level,
                "count": len(pii_result.pii_matches)
            }
            if pii_result.risk_level in ["medium", "high"]:
                results["safe"] = False
                results["violations"].append("pii_detected")
        
        # Toxicity Classification
        if self.toxicity_classifier:
            toxicity_result = await self.toxicity_classifier.classify_toxicity(text)
            results["toxicity"] = {
                "is_toxic": toxicity_result.is_toxic,
                "score": toxicity_result.overall_toxicity,
                "types": [score.toxicity_type.value for score in toxicity_result.toxicity_scores if score.is_toxic]
            }
            if toxicity_result.is_toxic:
                results["safe"] = False
                results["violations"].append("toxicity_detected")
        
        return results
    
    async def _check_output_safety(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check output text for safety issues"""
        results = {"safe": True, "violations": {}}
        
        # Run all safety checks in parallel
        safety_tasks = []
        
        if self.pii_detector:
            safety_tasks.append(self.pii_detector.detect_pii(text))
        
        if self.toxicity_classifier:
            safety_tasks.append(self.toxicity_classifier.classify_toxicity(text))
        
        if self.bias_detector:
            safety_tasks.append(self.bias_detector.detect_bias(text))
        
        # Execute safety checks
        safety_results = await asyncio.gather(*safety_tasks, return_exceptions=True)
        
        # Process PII results
        pii_idx = 0
        if self.pii_detector:
            pii_result = safety_results[pii_idx]
            if not isinstance(pii_result, Exception):
                results["pii"] = {
                    "detected": len(pii_result.pii_matches) > 0,
                    "risk_level": pii_result.risk_level,
                    "count": len(pii_result.pii_matches)
                }
                if pii_result.risk_level in ["medium", "high"]:
                    results["safe"] = False
                    results["violations"]["pii"] = len(pii_result.pii_matches)
            pii_idx += 1
        
        # Process toxicity results  
        if self.toxicity_classifier:
            toxicity_result = safety_results[pii_idx]
            if not isinstance(toxicity_result, Exception):
                results["toxicity"] = {
                    "is_toxic": toxicity_result.is_toxic,
                    "score": toxicity_result.overall_toxicity
                }
                if toxicity_result.is_toxic:
                    results["safe"] = False
                    results["violations"]["toxicity"] = toxicity_result.overall_toxicity
            pii_idx += 1
        
        # Process bias results
        if self.bias_detector:
            bias_result = safety_results[pii_idx]
            if not isinstance(bias_result, Exception):
                results["bias"] = {
                    "detected": len(bias_result.bias_indicators) > 0,
                    "overall_score": bias_result.overall_bias_score,
                    "types": [indicator.bias_type.value for indicator in bias_result.bias_indicators]
                }
                if bias_result.overall_bias_score > 0.5:
                    results["safe"] = False
                    results["violations"]["bias"] = bias_result.overall_bias_score
        
        return results
    
    async def _generate_text_internal(self, prompt: str, model_name: str, 
                                    token_tracker=None) -> str:
        """Internal text generation (placeholder)"""
        # This is a placeholder - in practice, this would integrate with actual generators
        # from the SDK (GPT-2, GPT-J, etc.)
        
        if token_tracker:
            token_tracker.record_first_token()
        
        # Simulate text generation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Return a simple generated response
        return f"Generated response to: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        status = {
            "service": "TextNLP Unified Service",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Safety components
        status["components"]["safety"] = {
            "pii_detector": self.pii_detector is not None,
            "toxicity_classifier": self.toxicity_classifier is not None,
            "bias_detector": self.bias_detector is not None,
            "compliance_checker": self.compliance_checker is not None,
            "audit_logger": self.audit_logger is not None
        }
        
        # Metrics components  
        status["components"]["metrics"] = {
            "metrics_collector": self.metrics_collector is not None,
            "quality_calculator": self.quality_calculator is not None,
            "resource_tracker": self.resource_tracker is not None,
            "business_collector": self.business_collector is not None
        }
        
        # Resource status
        if self.resource_tracker:
            current_utilization = self.resource_tracker.get_current_utilization()
            if current_utilization:
                status["resources"] = current_utilization.to_dict()
        
        # Active alerts
        if self.resource_tracker:
            active_alerts = self.resource_tracker.get_alerts(resolved=False)
            status["alerts"] = {
                "count": len(active_alerts),
                "critical": len([a for a in active_alerts if a.severity.value == "critical"])
            }
        
        return status
    
    def shutdown(self):
        """Shutdown the service and cleanup resources"""
        logger.info("Shutting down Unified TextNLP Service")
        
        # Stop resource monitoring
        if self.resource_tracker:
            self.resource_tracker.stop_monitoring()
        
        # Stop metrics collection
        if self.metrics_collector:
            self.metrics_collector.stop()
        
        # Stop audit logging
        if self.audit_logger:
            self.audit_logger.stop()
        
        # Shutdown quality calculator
        if self.quality_calculator:
            self.quality_calculator.shutdown()
        
        logger.info("Service shutdown complete")