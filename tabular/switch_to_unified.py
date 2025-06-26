#!/usr/bin/env python3
"""
Switch Tabular Service to Unified Infrastructure

This script switches the tabular service to use the unified infrastructure
by updating the main app.py to use app_unified.py instead of the legacy version.
"""

import os
import shutil
from pathlib import Path
import argparse
import sys


def switch_to_unified():
    """Switch to unified infrastructure version"""
    api_dir = Path(__file__).parent / "api"
    
    # Check if unified version exists
    app_unified_path = api_dir / "app_unified.py"
    if not app_unified_path.exists():
        print("❌ app_unified.py not found!")
        print("   The unified infrastructure version is not available.")
        return False
    
    # Backup current app.py
    app_path = api_dir / "app.py"
    backup_path = api_dir / "app_legacy.py"
    
    print("📁 Backing up current app.py to app_legacy.py...")
    shutil.copy2(app_path, backup_path)
    
    # Create new app.py that imports from app_unified
    new_app_content = '''"""
Tabular Synthetic Data API - Unified Infrastructure Version

This module imports the unified infrastructure version of the API.
"""

from .app_unified import app

# Export the app for uvicorn
__all__ = ["app"]
'''
    
    print("🔄 Updating app.py to use unified infrastructure...")
    with open(app_path, 'w') as f:
        f.write(new_app_content)
    
    print("✅ Successfully switched to unified infrastructure!")
    print("\n📋 Next steps:")
    print("   1. Run database migrations: python run_migration.py")
    print("   2. Set environment variables for unified infrastructure")
    print("   3. Start the service: uvicorn api.app:app --reload")
    print("\n🔧 To switch back: python switch_to_unified.py --revert")
    
    return True


def revert_to_legacy():
    """Revert to legacy version"""
    api_dir = Path(__file__).parent / "api"
    
    # Check if backup exists
    backup_path = api_dir / "app_legacy.py"
    if not backup_path.exists():
        print("❌ No backup found (app_legacy.py)")
        print("   Cannot revert to legacy version.")
        return False
    
    app_path = api_dir / "app.py"
    
    print("🔄 Reverting to legacy version...")
    shutil.copy2(backup_path, app_path)
    
    print("✅ Successfully reverted to legacy version!")
    print("\n📋 The service is now using the original implementation.")
    
    return True


def show_status():
    """Show current configuration status"""
    api_dir = Path(__file__).parent / "api"
    app_path = api_dir / "app.py"
    app_unified_path = api_dir / "app_unified.py"
    backup_path = api_dir / "app_legacy.py"
    
    print("📊 Tabular Service Configuration Status")
    print("=" * 40)
    
    # Check if files exist
    print(f"app.py exists: {'✅' if app_path.exists() else '❌'}")
    print(f"app_unified.py exists: {'✅' if app_unified_path.exists() else '❌'}")
    print(f"app_legacy.py exists: {'✅' if backup_path.exists() else '❌'}")
    
    # Check current configuration
    if app_path.exists():
        with open(app_path, 'r') as f:
            content = f.read()
        
        if 'app_unified' in content:
            print("\n🔧 Current mode: Unified Infrastructure")
            print("   Using unified cloud deployment infrastructure")
        elif 'Legacy' in content[:200]:
            print("\n🏠 Current mode: Legacy")
            print("   Using original tabular service implementation")
        else:
            print("\n❓ Current mode: Unknown")
            print("   Cannot determine current configuration")
    
    # Check environment variables
    print("\n🌍 Environment Configuration:")
    env_vars = [
        'DATABASE_PROVIDER',
        'CACHE_PROVIDER', 
        'STORAGE_PROVIDER',
        'SERVICE_TIER',
        'ENABLE_METRICS'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        print(f"   {var}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Switch tabular service between legacy and unified infrastructure")
    parser.add_argument('--revert', action='store_true', help='Revert to legacy version')
    parser.add_argument('--status', action='store_true', help='Show current status')
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    elif args.revert:
        if revert_to_legacy():
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        if switch_to_unified():
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()