"""
Unified Database Module

Provides unified database abstraction for all Inferloop services.
Supports PostgreSQL, MySQL, SQLite with async operations using SQLAlchemy 2.0.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
import logging

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, Text, Boolean, JSON
from fastapi import Depends
import asyncpg
import aiomysql


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class DatabaseConfig:
    """Database configuration"""
    
    def __init__(self):
        self.provider = os.getenv('DATABASE_PROVIDER', 'postgresql').lower()
        self.host = os.getenv('DATABASE_HOST', 'localhost')
        self.port = int(os.getenv('DATABASE_PORT', '5432'))
        self.username = os.getenv('DATABASE_USERNAME', 'postgres')
        self.password = os.getenv('DATABASE_PASSWORD', 'password')
        self.database = os.getenv('DATABASE_NAME', 'inferloop')
        self.pool_size = int(os.getenv('DATABASE_POOL_SIZE', '10'))
        self.max_overflow = int(os.getenv('DATABASE_MAX_OVERFLOW', '20'))
        self.pool_timeout = int(os.getenv('DATABASE_POOL_TIMEOUT', '30'))
        self.echo = os.getenv('DATABASE_ECHO', 'false').lower() == 'true'
        self.ssl_mode = os.getenv('DATABASE_SSL_MODE', 'prefer')
    
    @property
    def url(self) -> str:
        """Get database URL"""
        if self.provider == 'postgresql':
            return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.provider == 'mysql':
            return f"mysql+aiomysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.provider == 'sqlite':
            db_path = os.getenv('DATABASE_PATH', './inferloop.db')
            return f"sqlite+aiosqlite:///{db_path}"
        else:
            raise ValueError(f"Unsupported database provider: {self.provider}")


class DatabaseClient:
    """Unified database client"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.config = DatabaseConfig()
        self.engine = None
        self.session_factory = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Setup database engine"""
        engine_kwargs = {
            'echo': self.config.echo,
            'future': True,
        }
        
        # Configure connection pool
        if self.config.provider != 'sqlite':
            engine_kwargs.update({
                'poolclass': QueuePool,
                'pool_size': self.config.pool_size,
                'max_overflow': self.config.max_overflow,
                'pool_timeout': self.config.pool_timeout,
                'pool_pre_ping': True,
                'pool_recycle': 3600,
            })
        else:
            # SQLite doesn't support connection pooling
            engine_kwargs['poolclass'] = NullPool
        
        self.engine = create_async_engine(self.config.url, **engine_kwargs)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        return self.session_factory()
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for database session"""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def execute(self, query: Union[str, sa.sql.Executable], parameters: Optional[Dict] = None) -> Any:
        """Execute SQL query"""
        async with self.session() as session:
            if isinstance(query, str):
                result = await session.execute(sa.text(query), parameters or {})
            else:
                result = await session.execute(query, parameters or {})
            return result
    
    async def fetch_one(self, query: Union[str, sa.sql.Executable], parameters: Optional[Dict] = None) -> Optional[Any]:
        """Fetch one row"""
        result = await self.execute(query, parameters)
        return result.first()
    
    async def fetch_all(self, query: Union[str, sa.sql.Executable], parameters: Optional[Dict] = None) -> List[Any]:
        """Fetch all rows"""
        result = await self.execute(query, parameters)
        return result.fetchall()
    
    async def create_tables(self, metadata: MetaData):
        """Create tables from metadata"""
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
    
    async def drop_tables(self, metadata: MetaData):
        """Drop tables from metadata"""
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.drop_all)
    
    async def check_connection(self) -> bool:
        """Check database connection"""
        try:
            async with self.session() as session:
                await session.execute(sa.text("SELECT 1"))
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()


# Common database models for all services
class User(Base):
    """User model"""
    __tablename__ = 'users'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    tier: Mapped[str] = mapped_column(String(50), nullable=False, default='starter')
    organization_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    permissions: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Organization(Base):
    """Organization model"""
    __tablename__ = 'organizations'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    tier: Mapped[str] = mapped_column(String(50), nullable=False, default='starter')
    settings: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class APIKey(Base):
    """API key model"""
    __tablename__ = 'api_keys'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    permissions: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class ServiceUsage(Base):
    """Service usage tracking model"""
    __tablename__ = 'service_usage'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    service_name: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)  # tokens, generations, validations
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class ServiceJob(Base):
    """Generic job tracking model"""
    __tablename__ = 'service_jobs'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    service_name: Mapped[str] = mapped_column(String(100), nullable=False)
    job_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default='pending')
    input_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    output_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class BillingRecord(Base):
    """Billing record model"""
    __tablename__ = 'billing_records'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    service_name: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[float] = mapped_column(sa.Float, nullable=False)
    total_amount: Mapped[float] = mapped_column(sa.Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default='USD')
    billing_period: Mapped[str] = mapped_column(String(20), nullable=False)  # monthly, daily, etc.
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


# Service-specific table creation helpers
def create_tabular_tables() -> MetaData:
    """Create tables specific to tabular service"""
    metadata = MetaData()
    
    # Generation jobs table
    Table(
        'tabular_generation_jobs',
        metadata,
        Column('id', String(36), primary_key=True),
        Column('user_id', String(36), nullable=False),
        Column('algorithm', String(100), nullable=False),
        Column('dataset_rows', Integer, nullable=False),
        Column('dataset_columns', Integer, nullable=False),
        Column('config', JSON, nullable=False),
        Column('status', String(50), nullable=False, default='pending'),
        Column('progress', Integer, nullable=False, default=0),
        Column('output_path', String(500), nullable=True),
        Column('error_message', Text, nullable=True),
        Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
        Column('started_at', DateTime, nullable=True),
        Column('completed_at', DateTime, nullable=True),
    )
    
    # Validation jobs table
    Table(
        'tabular_validation_jobs',
        metadata,
        Column('id', String(36), primary_key=True),
        Column('user_id', String(36), nullable=False),
        Column('generation_job_id', String(36), nullable=True),
        Column('validator_type', String(100), nullable=False),
        Column('metrics', JSON, nullable=False, default=dict),
        Column('results', JSON, nullable=False, default=dict),
        Column('status', String(50), nullable=False, default='pending'),
        Column('error_message', Text, nullable=True),
        Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
        Column('completed_at', DateTime, nullable=True),
    )
    
    # Dataset profiles table
    Table(
        'tabular_dataset_profiles',
        metadata,
        Column('id', String(36), primary_key=True),
        Column('user_id', String(36), nullable=False),
        Column('dataset_name', String(255), nullable=False),
        Column('dataset_hash', String(64), nullable=False),
        Column('column_count', Integer, nullable=False),
        Column('row_count', Integer, nullable=False),
        Column('profile_data', JSON, nullable=False),
        Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
    )
    
    return metadata


def create_textnlp_tables() -> MetaData:
    """Create tables specific to textnlp service"""
    metadata = MetaData()
    
    # Generations table
    Table(
        'textnlp_generations',
        metadata,
        Column('id', String(36), primary_key=True),
        Column('user_id', String(36), nullable=False),
        Column('model', String(100), nullable=False),
        Column('prompt', Text, nullable=False),
        Column('generated_text', Text, nullable=False),
        Column('tokens_used', Integer, nullable=False),
        Column('duration', sa.Float, nullable=True),
        Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
    )
    
    # Validations table
    Table(
        'textnlp_validations',
        metadata,
        Column('id', String(36), primary_key=True),
        Column('user_id', String(36), nullable=False),
        Column('metrics', JSON, nullable=False),
        Column('results', JSON, nullable=False),
        Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
    )
    
    # Templates table
    Table(
        'textnlp_templates',
        metadata,
        Column('id', String(36), primary_key=True),
        Column('user_id', String(36), nullable=False),
        Column('name', String(255), nullable=False),
        Column('template_content', Text, nullable=False),
        Column('variables', JSON, nullable=False, default=list),
        Column('category', String(100), nullable=False, default='general'),
        Column('is_public', Boolean, nullable=False, default=False),
        Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
        Column('updated_at', DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow),
    )
    
    return metadata


class DatabaseManager:
    """Manager for database operations across services"""
    
    def __init__(self):
        self.clients: Dict[str, DatabaseClient] = {}
        self.config = DatabaseConfig()
    
    def get_client(self, service_name: str) -> DatabaseClient:
        """Get database client for service"""
        if service_name not in self.clients:
            self.clients[service_name] = DatabaseClient(service_name)
        return self.clients[service_name]
    
    async def create_service_tables(self, service_name: str):
        """Create tables for a specific service"""
        client = self.get_client(service_name)
        
        # Create common tables
        await client.create_tables(Base.metadata)
        
        # Create service-specific tables
        if service_name == 'tabular':
            metadata = create_tabular_tables()
            await client.create_tables(metadata)
        elif service_name == 'textnlp':
            metadata = create_textnlp_tables()
            await client.create_tables(metadata)
    
    async def run_migrations(self, service_name: str):
        """Run database migrations for service"""
        # This is a simplified migration system
        # In production, you'd use Alembic or similar
        await self.create_service_tables(service_name)
    
    async def check_all_connections(self) -> Dict[str, bool]:
        """Check connections for all clients"""
        results = {}
        for service_name, client in self.clients.items():
            results[service_name] = await client.check_connection()
        return results
    
    async def close_all(self):
        """Close all database connections"""
        for client in self.clients.values():
            await client.close()


# Global database manager
db_manager = DatabaseManager()


# Dependency for FastAPI
async def get_db_session(service_name: str = "default") -> AsyncSession:
    """Dependency to get database session"""
    client = db_manager.get_client(service_name)
    return await client.get_session()


# Utility functions
async def create_database_if_not_exists(database_name: str):
    """Create database if it doesn't exist (PostgreSQL/MySQL only)"""
    config = DatabaseConfig()
    
    if config.provider == 'postgresql':
        # Connect to postgres database to create new database
        admin_url = f"postgresql+asyncpg://{config.username}:{config.password}@{config.host}:{config.port}/postgres"
        admin_engine = create_async_engine(admin_url)
        
        async with admin_engine.begin() as conn:
            # Check if database exists
            result = await conn.execute(
                sa.text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": database_name}
            )
            
            if not result.fetchone():
                # Create database
                await conn.execute(sa.text(f"CREATE DATABASE {database_name}"))
        
        await admin_engine.dispose()
    
    elif config.provider == 'mysql':
        # Connect to MySQL to create new database
        admin_url = f"mysql+aiomysql://{config.username}:{config.password}@{config.host}:{config.port}"
        admin_engine = create_async_engine(admin_url)
        
        async with admin_engine.begin() as conn:
            await conn.execute(sa.text(f"CREATE DATABASE IF NOT EXISTS {database_name}"))
        
        await admin_engine.dispose()


class TransactionManager:
    """Helper for managing database transactions"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._savepoints = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        else:
            await self.commit()
    
    async def savepoint(self, name: str = None):
        """Create a savepoint"""
        if name is None:
            name = f"sp_{len(self._savepoints)}"
        
        await self.session.execute(sa.text(f"SAVEPOINT {name}"))
        self._savepoints.append(name)
        return name
    
    async def rollback_to_savepoint(self, name: str):
        """Rollback to specific savepoint"""
        await self.session.execute(sa.text(f"ROLLBACK TO SAVEPOINT {name}"))
        
        # Remove savepoints created after this one
        try:
            index = self._savepoints.index(name)
            self._savepoints = self._savepoints[:index + 1]
        except ValueError:
            pass
    
    async def commit(self):
        """Commit transaction"""
        await self.session.commit()
        self._savepoints.clear()
    
    async def rollback(self):
        """Rollback transaction"""
        await self.session.rollback()
        self._savepoints.clear()


# Query builder helpers
class QueryBuilder:
    """Helper for building common queries"""
    
    @staticmethod
    def paginate(query, page: int = 1, size: int = 20):
        """Add pagination to query"""
        offset = (page - 1) * size
        return query.offset(offset).limit(size)
    
    @staticmethod
    def filter_by_date_range(query, date_column, start_date: datetime = None, end_date: datetime = None):
        """Filter query by date range"""
        if start_date:
            query = query.where(date_column >= start_date)
        if end_date:
            query = query.where(date_column <= end_date)
        return query
    
    @staticmethod
    def order_by_created_at(query, desc: bool = True):
        """Order query by created_at"""
        return query.order_by(sa.desc('created_at') if desc else sa.asc('created_at'))


# Health check function
async def check_database_health() -> Dict[str, Any]:
    """Check database health across all services"""
    health_status = {
        "status": "healthy",
        "services": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    connections = await db_manager.check_all_connections()
    
    for service, is_healthy in connections.items():
        health_status["services"][service] = {
            "status": "healthy" if is_healthy else "unhealthy",
            "provider": db_manager.config.provider
        }
    
    # Overall status
    if not all(connections.values()):
        health_status["status"] = "degraded"
    
    return health_status