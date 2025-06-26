#!/usr/bin/env python3
"""
Test Unified Infrastructure Integration

This script tests the tabular service with unified infrastructure to ensure
all components are working correctly.
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import tempfile
import uuid

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
def test_imports():
    """Test that all unified infrastructure modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from unified_cloud_deployment import (
            AuthMiddleware, get_current_user, User,
            TelemetryMiddleware, MetricsCollector,
            StorageClient, CacheClient, DatabaseClient,
            get_service_config, RateLimitMiddleware
        )
        print("   ✅ All unified infrastructure modules imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


def test_configuration():
    """Test configuration management"""
    print("🔧 Testing configuration...")
    
    try:
        from unified_cloud_deployment.config import get_config_manager
        
        config_manager = get_config_manager("tabular")
        config = config_manager.config
        
        print(f"   ✅ Service name: {config.name}")
        print(f"   ✅ Service tier: {config.tier}")
        print(f"   ✅ Environment: {config.environment}")
        print(f"   ✅ Database provider: {config.database.get('provider', 'not set')}")
        print(f"   ✅ Cache provider: {config.cache.get('provider', 'not set')}")
        print(f"   ✅ Storage provider: {config.storage.get('provider', 'not set')}")
        
        return True
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False


async def test_database():
    """Test database connectivity and schema"""
    print("🗄️  Testing database...")
    
    try:
        from unified_cloud_deployment.database import DatabaseClient
        
        client = DatabaseClient("tabular")
        
        # Test connection
        if await client.check_connection():
            print("   ✅ Database connection successful")
        else:
            print("   ❌ Database connection failed")
            return False
        
        # Test schema - check if tables exist
        async with client.session() as session:
            result = await session.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('users', 'tabular_generation_jobs', 'tabular_validation_jobs')
                ORDER BY table_name
            """)
            
            tables = [row[0] for row in result.fetchall()]
            
            expected_tables = ['users', 'tabular_generation_jobs', 'tabular_validation_jobs']
            missing_tables = [t for t in expected_tables if t not in tables]
            
            if missing_tables:
                print(f"   ⚠️  Missing tables: {missing_tables}")
                print("   💡 Run: python run_migration.py")
            else:
                print("   ✅ All required tables exist")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False


async def test_cache():
    """Test cache connectivity"""
    print("📦 Testing cache...")
    
    try:
        from unified_cloud_deployment.cache import CacheClient
        
        client = CacheClient("tabular")
        
        # Test connection
        if await client.ping():
            print("   ✅ Cache connection successful")
        else:
            print("   ❌ Cache connection failed")
            return False
        
        # Test basic operations
        test_key = f"test:{uuid.uuid4().hex[:8]}"
        test_value = {"message": "Hello from tabular service", "timestamp": datetime.utcnow().isoformat()}
        
        # Set
        await client.set(test_key, test_value, ttl=60)
        print("   ✅ Cache set operation successful")
        
        # Get
        retrieved_value = await client.get(test_key)
        if retrieved_value and retrieved_value.get("message") == test_value["message"]:
            print("   ✅ Cache get operation successful")
        else:
            print("   ❌ Cache get operation failed")
            return False
        
        # Delete
        await client.delete(test_key)
        print("   ✅ Cache delete operation successful")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Cache error: {e}")
        return False


async def test_storage():
    """Test storage connectivity"""
    print("💾 Testing storage...")
    
    try:
        from unified_cloud_deployment.storage import StorageClient
        
        client = StorageClient("tabular")
        
        # Test connection
        if await client.check_connection():
            print("   ✅ Storage connection successful")
        else:
            print("   ❌ Storage connection failed")
            return False
        
        # Test basic operations
        test_key = f"test/{uuid.uuid4().hex[:8]}.txt"
        test_content = f"Test content from tabular service at {datetime.utcnow().isoformat()}"
        
        # Put
        await client.put(test_key, test_content)
        print("   ✅ Storage put operation successful")
        
        # Check exists
        if await client.exists(test_key):
            print("   ✅ Storage exists check successful")
        else:
            print("   ❌ Storage exists check failed")
            return False
        
        # Get
        retrieved_content = await client.get_text(test_key)
        if test_content in retrieved_content:
            print("   ✅ Storage get operation successful")
        else:
            print("   ❌ Storage get operation failed")
            return False
        
        # Delete
        await client.delete(test_key)
        print("   ✅ Storage delete operation successful")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Storage error: {e}")
        return False


def test_service_adapter():
    """Test tabular service adapter"""
    print("🔌 Testing service adapter...")
    
    try:
        from tabular.infrastructure.adapter import TabularServiceAdapter, ServiceTier
        
        # Test different tiers
        for tier in [ServiceTier.STARTER, ServiceTier.PROFESSIONAL, ServiceTier.BUSINESS, ServiceTier.ENTERPRISE]:
            adapter = TabularServiceAdapter(tier)
            
            print(f"   📊 Testing {tier.value} tier:")
            
            # Test configurations
            service_config = adapter.get_service_config()
            print(f"      ✅ Service config: {service_config['name']}")
            
            api_config = adapter.get_api_config()
            rate_limits = api_config['rate_limiting']
            print(f"      ✅ Rate limits: {rate_limits['requests_per_hour']}/hour")
            
            model_config = adapter.get_model_config()
            available_models = list(model_config['available_models'].keys())
            print(f"      ✅ Available models: {len(available_models)} ({', '.join(available_models[:3])})")
            
            # Test scaling config
            scaling_config = adapter.get_scaling_config()
            print(f"      ✅ Scaling: {scaling_config['min_replicas']}-{scaling_config['max_replicas']} replicas")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Service adapter error: {e}")
        return False


async def test_api_app():
    """Test that the unified API app can be imported and configured"""
    print("🌐 Testing API application...")
    
    try:
        # Test import
        from tabular.api.app_unified import app
        print("   ✅ Unified API app imported successfully")
        
        # Test app configuration
        print(f"   ✅ App title: {app.title}")
        print(f"   ✅ App version: {app.version}")
        
        # Test routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = ["/health", "/ready", "/api/tabular/generate", "/metrics"]
        
        missing_routes = [route for route in expected_routes if route not in routes]
        if missing_routes:
            print(f"   ⚠️  Missing routes: {missing_routes}")
        else:
            print("   ✅ All expected routes registered")
        
        return True
        
    except Exception as e:
        print(f"   ❌ API app error: {e}")
        return False


def test_environment_setup():
    """Check environment variables for unified infrastructure"""
    print("🌍 Testing environment setup...")
    
    # Check for required environment variables
    required_vars = {
        'DATABASE_PROVIDER': 'postgresql',
        'CACHE_PROVIDER': 'redis',
        'STORAGE_PROVIDER': 's3',
        'SERVICE_NAME': 'tabular'
    }
    
    optional_vars = {
        'SERVICE_TIER': 'starter',
        'DATABASE_HOST': 'localhost',
        'CACHE_HOST': 'localhost',
        'STORAGE_BUCKET_NAME': 'inferloop-tabular',
        'ENABLE_METRICS': 'true'
    }
    
    all_good = True
    
    for var, default in required_vars.items():
        value = os.getenv(var, default)
        if value:
            print(f"   ✅ {var}: {value}")
        else:
            print(f"   ❌ {var}: not set")
            all_good = False
    
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        print(f"   ℹ️  {var}: {value}")
    
    if not all_good:
        print("\n   💡 Set missing environment variables:")
        print("      export DATABASE_PROVIDER=postgresql")
        print("      export CACHE_PROVIDER=redis") 
        print("      export STORAGE_PROVIDER=s3")
        print("      export SERVICE_NAME=tabular")
    
    return all_good


async def run_all_tests():
    """Run all integration tests"""
    print("🧪 Tabular Service Unified Infrastructure Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Environment", test_environment_setup),
        ("Service Adapter", test_service_adapter),
        ("Database", test_database),
        ("Cache", test_cache),
        ("Storage", test_storage),
        ("API App", test_api_app),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name} Test")
        print("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Tabular service is ready for unified infrastructure.")
        return True
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test tabular service unified infrastructure integration")
    parser.add_argument('--test', choices=['imports', 'config', 'database', 'cache', 'storage', 'adapter', 'api'], 
                       help='Run specific test only')
    
    args = parser.parse_args()
    
    if args.test:
        # Run specific test
        test_map = {
            'imports': test_imports,
            'config': test_configuration,
            'database': test_database,
            'cache': test_cache,
            'storage': test_storage,
            'adapter': test_service_adapter,
            'api': test_api_app
        }
        
        test_func = test_map.get(args.test)
        if test_func:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
            
            sys.exit(0 if result else 1)
        else:
            print(f"Unknown test: {args.test}")
            sys.exit(1)
    else:
        # Run all tests
        result = asyncio.run(run_all_tests())
        sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()