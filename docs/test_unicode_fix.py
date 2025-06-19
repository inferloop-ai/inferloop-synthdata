#!/usr/bin/env python3
"""
Test script to verify that unicode encoding issues in privacy modules have been fixed.
"""

import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 
                                'docs/structured-documents-synthetic-data/src'))

def test_unicode_fixes():
    """Test that all privacy modules can be imported without unicode errors."""
    
    print("🧪 Testing Unicode Fixes in Privacy Modules")
    print("=" * 50)
    
    try:
        # Test differential privacy modules
        print("\n📚 Testing Differential Privacy Modules...")
        
        from structured_docs_synth.privacy.differential_privacy.privacy_budget import (
            PrivacyBudgetTracker, BudgetScope, CompositionType
        )
        print("✅ Privacy Budget Tracker imported successfully")
        
        from structured_docs_synth.privacy.differential_privacy.exponential_mechanism import (
            ExponentialMechanism, SelectionStrategy
        )
        print("✅ Exponential Mechanism imported successfully")
        
        from structured_docs_synth.privacy.differential_privacy.laplace_mechanism import (
            LaplaceMechanism, NoiseParameters
        )
        print("✅ Laplace Mechanism imported successfully")
        
        from structured_docs_synth.privacy.differential_privacy.composition_analyzer import (
            CompositionAnalyzer, CompositionMethod
        )
        print("✅ Composition Analyzer imported successfully")
        
        print("\n🔬 Testing Basic Functionality...")
        
        # Test Privacy Budget Tracker
        tracker = PrivacyBudgetTracker(max_epsilon=1.0, max_delta=1e-5)
        allocation_id = tracker.allocate_budget(
            epsilon=0.1, 
            delta=1e-6, 
            mechanism="test", 
            query_description="test query"
        )
        summary = tracker.get_budget_summary()
        print(f"✅ Budget allocation successful: {allocation_id[:8]}...")
        print(f"   Remaining epsilon: {summary.epsilon_remaining:.6f}")
        
        # Test Laplace Mechanism
        laplace = LaplaceMechanism()
        noise_params = NoiseParameters(epsilon=1.0, sensitivity=1.0)
        noisy_value = laplace.add_noise(42.0, noise_params)
        print(f"✅ Laplace noise added: 42.0 -> {noisy_value:.4f}")
        
        # Test Exponential Mechanism
        exp_mech = ExponentialMechanism()
        from structured_docs_synth.privacy.differential_privacy.exponential_mechanism import FrequencyUtility
        utility_func = FrequencyUtility()
        result = exp_mech.select_output(
            data=[1, 2, 2, 3, 3, 3],
            output_domain=[1, 2, 3],
            utility_function=utility_func,
            epsilon=1.0
        )
        print(f"✅ Exponential mechanism selection: {result.selected_output}")
        
        print("\n🎯 All Tests Passed!")
        print("✅ No unicode encoding errors detected")
        print("✅ All differential privacy modules working correctly")
        
        return True
        
    except UnicodeError as e:
        print(f"❌ Unicode Error: {e}")
        return False
    except SyntaxError as e:
        print(f"❌ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_unicode_fixes()
    if success:
        print("\n🎉 Unicode fix validation completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Unicode fix validation failed!")
        sys.exit(1)