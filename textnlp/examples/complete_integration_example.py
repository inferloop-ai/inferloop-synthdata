"""
Complete Integration Example for TextNLP

Demonstrates how to use all Phase 3 components together:
- Safety filtering (PII, toxicity, bias, compliance)
- Quality metrics (BLEU, ROUGE, etc.)
- Resource monitoring
- Business analytics
- Audit logging
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services import ServiceOrchestrator
from safety import PIIDetector, ToxicityClassifier, BiasDetector
from metrics import QualityMetricsCalculator, ResourceTracker, BusinessMetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_safety_features():
    """Demonstrate all safety features independently"""
    print("\n" + "="*60)
    print("DEMONSTRATING SAFETY FEATURES")
    print("="*60)
    
    # Test texts with various safety issues
    test_texts = [
        "Hello, my name is John Doe and my email is john.doe@example.com",  # PII
        "You are a stupid idiot and I hate you!",  # Toxicity
        "Women are naturally worse at math than men",  # Bias
        "We process personal data without consent for marketing purposes",  # GDPR violation
        "This is a normal, safe text with no issues"  # Safe text
    ]
    
    # Initialize safety components
    pii_detector = PIIDetector()
    toxicity_classifier = ToxicityClassifier({})
    bias_detector = BiasDetector({})
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i}: {text[:50]}{'...' if len(text) > 50 else ''} ---")
        
        # PII Detection
        pii_result = await pii_detector.detect_pii(text)
        print(f"PII: {len(pii_result.pii_matches)} instances, Risk: {pii_result.risk_level}")
        
        # Toxicity Classification
        toxicity_result = await toxicity_classifier.classify_toxicity(text)
        print(f"Toxicity: {toxicity_result.is_toxic}, Score: {toxicity_result.overall_toxicity:.3f}")
        
        # Bias Detection
        bias_result = await bias_detector.detect_bias(text)
        print(f"Bias: {len(bias_result.bias_indicators)} indicators, Score: {bias_result.overall_bias_score:.3f}")


async def demonstrate_quality_metrics():
    """Demonstrate quality metrics calculation"""
    print("\n" + "="*60)
    print("DEMONSTRATING QUALITY METRICS")
    print("="*60)
    
    # Test pairs (reference, candidate)
    test_pairs = [
        (
            "The quick brown fox jumps over the lazy dog.",
            "A fast brown fox leaps over a sleepy dog."
        ),
        (
            "Artificial intelligence will transform many industries.",
            "AI technology is going to change various business sectors."
        ),
        (
            "Climate change is a serious global challenge.",
            "Global warming poses significant worldwide problems."
        )
    ]
    
    # Initialize quality calculator
    quality_calculator = QualityMetricsCalculator({
        "enabled_metrics": ["bleu", "rouge_1", "rouge_2", "semantic_similarity", "fluency"]
    })
    
    for i, (reference, candidate) in enumerate(test_pairs, 1):
        print(f"\n--- Quality Test {i} ---")
        print(f"Reference: {reference}")
        print(f"Candidate: {candidate}")
        
        evaluation = await quality_calculator.evaluate_quality(reference, candidate)
        
        print(f"Overall Quality: {evaluation.overall_quality:.3f}")
        for score in evaluation.scores:
            print(f"  {score.metric_type.value}: {score.score:.3f}")
    
    quality_calculator.shutdown()


async def demonstrate_resource_monitoring():
    """Demonstrate resource monitoring"""
    print("\n" + "="*60)
    print("DEMONSTRATING RESOURCE MONITORING")
    print("="*60)
    
    # Initialize resource tracker
    tracker = ResourceTracker({
        "sample_interval": 2,
        "alert_thresholds": {
            "cpu_warning": 50.0,  # Lower threshold for demo
            "memory_warning": 50.0
        }
    })
    
    # Start monitoring
    tracker.start_monitoring()
    
    # Let it collect some data
    print("Collecting resource data for 10 seconds...")
    await asyncio.sleep(10)
    
    # Get current utilization
    current = tracker.get_current_utilization()
    if current:
        print(f"\nCurrent Resource Utilization:")
        print(f"  CPU: {current.cpu_usage:.1f}%")
        print(f"  Memory: {current.memory_usage:.1f}%")
        print(f"  Available Memory: {current.memory_available:.2f} GB")
        if current.gpu_usage is not None:
            print(f"  GPU: {current.gpu_usage:.1f}%")
    
    # Get statistics
    cpu_stats = tracker.get_resource_statistics("cpu", "usage", 5)
    if cpu_stats:
        print(f"\nCPU Statistics (5 min):")
        print(f"  Mean: {cpu_stats['mean']:.1f}%")
        print(f"  Max: {cpu_stats['max']:.1f}%")
        print(f"  Samples: {cpu_stats['samples']}")
    
    # Check for alerts
    alerts = tracker.get_alerts(resolved=False)
    print(f"\nActive Alerts: {len(alerts)}")
    for alert in alerts[:3]:  # Show first 3
        print(f"  {alert.severity.value.upper()}: {alert.message}")
    
    tracker.stop_monitoring()


async def demonstrate_business_analytics():
    """Demonstrate business analytics"""
    print("\n" + "="*60)
    print("DEMONSTRATING BUSINESS ANALYTICS")
    print("="*60)
    
    # Initialize business metrics collector
    collector = BusinessMetricsCollector({"db_path": ":memory:"})  # In-memory DB for demo
    
    # Simulate business metrics
    print("Generating sample business metrics...")
    
    for i in range(20):
        # Usage metrics
        collector.record_usage_metrics({
            "tokens_generated": 100 + i * 10,
            "model": "gpt2" if i % 2 == 0 else "gpt-j",
            "user_id": f"user_{i % 5}"
        })
        
        # Performance metrics
        collector.record_performance_metrics({
            "response_time": 1.0 + (i % 3) * 0.5,
            "throughput": 20 + i,
            "errors": 0 if i % 10 != 0 else 1
        })
        
        # Quality metrics
        collector.record_quality_metrics({
            "bleu_score": 0.7 + (i % 3) * 0.1,
            "rouge_score": 0.6 + (i % 4) * 0.1
        })
    
    # Get aggregated metrics
    api_requests = collector.get_metric_aggregation(
        "api_requests", 
        start_time=collector.metrics_history[0].timestamp,
        end_time=collector.metrics_history[-1].timestamp,
        granularity="day",
        aggregation_function="sum"
    )
    
    print(f"\nBusiness Metrics Summary:")
    print(f"  Total API Requests: {sum(d['value'] for d in api_requests)}")
    print(f"  Metrics Records: {len(collector.metrics_history)}")
    
    # Get metrics by category
    from collections import defaultdict
    metrics_by_category = defaultdict(int)
    for metric in collector.metrics_history:
        metrics_by_category[metric.category.value] += 1
    
    print(f"\nMetrics by Category:")
    for category, count in metrics_by_category.items():
        print(f"  {category}: {count}")


async def demonstrate_unified_service():
    """Demonstrate the unified service integration"""
    print("\n" + "="*60)
    print("DEMONSTRATING UNIFIED SERVICE INTEGRATION")
    print("="*60)
    
    try:
        # Initialize service orchestrator
        orchestrator = ServiceOrchestrator()
        
        async with orchestrator.service_context():
            print("Unified service started successfully!")
            
            # Test prompts with various characteristics
            test_prompts = [
                "Write a helpful explanation about renewable energy",  # Safe, good
                "Write something toxic and harmful",  # Should be blocked
                "My SSN is 123-45-6789, please help me",  # PII issue
                "Generate a story about space exploration"  # Safe, creative
            ]
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n--- Unified Test {i} ---")
                print(f"Prompt: {prompt}")
                
                result = await orchestrator.generate_text(
                    prompt=prompt,
                    model_name="gpt2",
                    user_id=f"demo_user_{i}",
                    context={"session_id": "demo_session"}
                )
                
                print(f"Success: {result['success']}")
                if result['success']:
                    print(f"Generated: {result['generated_text'][:100]}...")
                    print(f"Input Safety: {result['input_safety']['safe']}")
                    print(f"Output Safety: {result['output_safety']['safe']}")
                    if result.get('quality_metrics'):
                        print(f"Quality Score: {result['quality_metrics']['overall_quality']:.3f}")
                else:
                    print(f"Blocked: {result.get('error', 'Unknown error')}")
                    if 'safety_results' in result:
                        violations = result['safety_results'].get('violations', [])
                        print(f"Violations: {', '.join(violations)}")
            
            # Get service status
            status = await orchestrator.get_service_status()
            print(f"\n--- Service Status ---")
            print(f"Status: {status['status']}")
            print(f"Safety Components: {sum(status['components']['safety'].values())}")
            print(f"Metrics Components: {sum(status['components']['metrics'].values())}")
            
            if 'alerts' in status:
                print(f"Active Alerts: {status['alerts']['count']}")
    
    except Exception as e:
        print(f"Error in unified service demo: {e}")
        logger.exception("Unified service demo failed")


async def main():
    """Run all demonstrations"""
    print("TextNLP Phase 3 Complete Integration Demonstration")
    print("="*60)
    
    try:
        # Individual component demonstrations
        await demonstrate_safety_features()
        await demonstrate_quality_metrics()
        await demonstrate_resource_monitoring()
        await demonstrate_business_analytics()
        
        # Unified service demonstration
        await demonstrate_unified_service()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        logger.exception("Demo failed")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the complete demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code)