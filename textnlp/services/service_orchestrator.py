"""
Service Orchestrator for TextNLP

Manages service lifecycle, health monitoring, and coordination between components
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from datetime import datetime

from .unified_service import UnifiedTextNLPService, ServiceConfig

logger = logging.getLogger(__name__)


class ServiceOrchestrator:
    """
    Orchestrates the entire TextNLP service ecosystem:
    - Service lifecycle management
    - Health monitoring
    - Graceful shutdown
    - Service coordination
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.services: Dict[str, Any] = {}
        self.running = False
        self.health_check_interval = 30  # seconds
        
        # Load configuration
        self.config = ServiceConfig.from_files(config_dir)
        
        logger.info("Service orchestrator initialized")
    
    async def start_services(self):
        """Start all services"""
        logger.info("Starting TextNLP services...")
        
        try:
            # Initialize unified service
            self.unified_service = UnifiedTextNLPService(self.config)
            self.services["unified_service"] = self.unified_service
            
            # Start health monitoring
            self.running = True
            asyncio.create_task(self._health_monitor_loop())
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            logger.info("All services started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            await self.shutdown_services()
            raise
    
    async def shutdown_services(self):
        """Shutdown all services gracefully"""
        logger.info("Shutting down TextNLP services...")
        
        self.running = False
        
        # Shutdown unified service
        if hasattr(self, 'unified_service'):
            self.unified_service.shutdown()
        
        # Clear services
        self.services.clear()
        
        logger.info("All services shut down")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown_services())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self.running:
            try:
                await self._check_service_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_service_health(self):
        """Check health of all services"""
        if hasattr(self, 'unified_service'):
            try:
                status = await self.unified_service.get_service_status()
                
                # Log any critical alerts
                if status.get("alerts", {}).get("critical", 0) > 0:
                    logger.warning(f"Critical alerts detected: {status['alerts']['critical']}")
                
                # Check component health
                for component_type, components in status.get("components", {}).items():
                    for component, healthy in components.items():
                        if not healthy:
                            logger.warning(f"Component {component_type}.{component} is not healthy")
                
            except Exception as e:
                logger.error(f"Failed to check unified service health: {e}")
    
    async def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using the unified service"""
        if not hasattr(self, 'unified_service'):
            raise RuntimeError("Unified service not initialized")
        
        return await self.unified_service.generate_text(prompt, **kwargs)
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        if not hasattr(self, 'unified_service'):
            return {
                "status": "not_initialized",
                "message": "Services not started"
            }
        
        return await self.unified_service.get_service_status()
    
    async def get_metrics_dashboard_data(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get dashboard data for monitoring"""
        if not hasattr(self, 'unified_service') or not self.unified_service.business_collector:
            return {"error": "Business metrics not available"}
        
        # This would integrate with the business dashboard
        # For now, return basic metrics
        return {
            "time_range": time_range,
            "services_running": len(self.services),
            "health_status": "healthy" if self.running else "stopped",
            "timestamp": datetime.now().isoformat()
        }
    
    @asynccontextmanager
    async def service_context(self):
        """Context manager for service lifecycle"""
        try:
            await self.start_services()
            yield self
        finally:
            await self.shutdown_services()


# Example usage and testing
async def main():
    """Example usage of the service orchestrator"""
    orchestrator = ServiceOrchestrator()
    
    async with orchestrator.service_context():
        # Test text generation
        result = await orchestrator.generate_text(
            prompt="Write a short story about AI safety",
            model_name="gpt2",
            user_id="test_user",
            context={"session_id": "test_session"}
        )
        
        print("Generation Result:")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Generated Text: {result['generated_text']}")
            print(f"Safety Status: {result['output_safety']['safe']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Get service status
        status = await orchestrator.get_service_status()
        print(f"\nService Status: {status['status']}")
        print(f"Components: {len(status['components']['safety'])} safety, {len(status['components']['metrics'])} metrics")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())