#!/usr/bin/env python3
"""
Database migration runner for tabular service

This script runs database migrations for the tabular service to set up
the unified infrastructure schema.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to Python path to import unified_cloud_deployment
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_cloud_deployment.database import DatabaseClient, DatabaseManager, create_database_if_not_exists
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_sql_file(client: DatabaseClient, sql_file_path: Path):
    """Run SQL commands from a file"""
    logger.info(f"Running migration: {sql_file_path.name}")
    
    with open(sql_file_path, 'r') as f:
        sql_content = f.read()
    
    # Split by statements (simple approach - assumes ; ends statements)
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    async with client.session() as session:
        for i, statement in enumerate(statements):
            if statement:
                try:
                    logger.debug(f"Executing statement {i+1}/{len(statements)}")
                    await session.execute(statement)
                except Exception as e:
                    logger.error(f"Error executing statement {i+1}: {e}")
                    logger.error(f"Statement: {statement[:100]}...")
                    raise
    
    logger.info(f"Successfully completed migration: {sql_file_path.name}")


async def run_migrations():
    """Run all database migrations for tabular service"""
    logger.info("Starting tabular service database migration")
    
    # Ensure database exists
    try:
        await create_database_if_not_exists("inferloop")
        logger.info("Database 'inferloop' is ready")
    except Exception as e:
        logger.warning(f"Could not create database (may already exist): {e}")
    
    # Get database client
    db_manager = DatabaseManager()
    client = db_manager.get_client("tabular")
    
    # Check connection
    if not await client.check_connection():
        logger.error("Cannot connect to database")
        return False
    
    logger.info("Database connection successful")
    
    # Find migration files
    migrations_dir = Path(__file__).parent / "migrations"
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    if not migration_files:
        logger.warning("No migration files found")
        return True
    
    logger.info(f"Found {len(migration_files)} migration files")
    
    # Run migrations in order
    for migration_file in migration_files:
        try:
            await run_sql_file(client, migration_file)
        except Exception as e:
            logger.error(f"Migration failed: {migration_file.name} - {e}")
            return False
    
    # Create service-specific tables using the database module
    try:
        logger.info("Creating additional service tables...")
        await db_manager.create_service_tables("tabular")
        logger.info("Service tables created successfully")
    except Exception as e:
        logger.warning(f"Error creating service tables (may already exist): {e}")
    
    # Verify tables exist
    async with client.session() as session:
        result = await session.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'tabular_%'
            ORDER BY table_name
        """)
        
        tables = [row[0] for row in result.fetchall()]
        logger.info(f"Tabular service tables: {', '.join(tables)}")
    
    # Close connections
    await db_manager.close_all()
    
    logger.info("All migrations completed successfully!")
    return True


async def verify_schema():
    """Verify the database schema is correctly set up"""
    logger.info("Verifying database schema...")
    
    db_manager = DatabaseManager()
    client = db_manager.get_client("tabular")
    
    # Check if all required tables exist
    required_tables = [
        'users',
        'organizations', 
        'api_keys',
        'service_usage',
        'service_jobs',
        'billing_records',
        'tabular_generation_jobs',
        'tabular_validation_jobs',
        'tabular_dataset_profiles'
    ]
    
    async with client.session() as session:
        for table in required_tables:
            result = await session.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table}'
                )
            """)
            
            exists = result.scalar()
            if exists:
                logger.info(f"✓ Table '{table}' exists")
            else:
                logger.error(f"✗ Table '{table}' missing")
                return False
    
    # Check indexes
    async with client.session() as session:
        result = await session.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename LIKE 'tabular_%'
            ORDER BY indexname
        """)
        
        indexes = [row[0] for row in result.fetchall()]
        logger.info(f"Found {len(indexes)} indexes on tabular tables")
    
    await db_manager.close_all()
    logger.info("Schema verification completed successfully!")
    return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tabular service database migrations")
    parser.add_argument('--verify-only', action='store_true', help='Only verify schema, do not run migrations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.verify_only:
        success = asyncio.run(verify_schema())
    else:
        success = asyncio.run(run_migrations())
        if success and not args.verify_only:
            # Also run verification after migration
            asyncio.run(verify_schema())
    
    if success:
        logger.info("Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("Migration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()