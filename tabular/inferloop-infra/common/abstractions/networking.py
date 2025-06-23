"""
Networking resource abstractions for VPCs, load balancers, and firewalls
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Network, IPv6Network, ip_network

from .base import BaseResource, ResourceConfig, ResourceType


class NetworkProtocol(Enum):
    """Network protocols"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ALL = "all"


class LoadBalancerType(Enum):
    """Types of load balancers"""
    APPLICATION = "application"  # Layer 7
    NETWORK = "network"  # Layer 4
    CLASSIC = "classic"
    GATEWAY = "gateway"


class HealthCheckProtocol(Enum):
    """Health check protocols"""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    GRPC = "grpc"


@dataclass
class NetworkConfig(ResourceConfig):
    """Configuration for network resources"""
    cidr_block: str = "10.0.0.0/16"
    enable_dns: bool = True
    enable_ipv6: bool = False
    availability_zones: List[str] = field(default_factory=list)
    nat_gateway: bool = True
    vpn_gateway: bool = False
    
    def validate(self) -> List[str]:
        errors = super().validate()
        try:
            ip_network(self.cidr_block)
        except ValueError:
            errors.append(f"Invalid CIDR block: {self.cidr_block}")
        return errors


@dataclass
class SubnetConfig:
    """Configuration for subnet"""
    name: str
    cidr_block: str
    availability_zone: str
    public: bool = False
    map_public_ip: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SecurityRule:
    """Security rule for firewall/security group"""
    protocol: NetworkProtocol
    from_port: int
    to_port: int
    source: str  # CIDR or security group
    description: str = ""
    
    def validate(self) -> List[str]:
        errors = []
        if self.from_port < 0 or self.from_port > 65535:
            errors.append(f"Invalid from_port: {self.from_port}")
        if self.to_port < 0 or self.to_port > 65535:
            errors.append(f"Invalid to_port: {self.to_port}")
        if self.from_port > self.to_port:
            errors.append("from_port cannot be greater than to_port")
        return errors


@dataclass
class FirewallConfig(ResourceConfig):
    """Configuration for firewall/security group"""
    vpc_id: str = ""
    ingress_rules: List[SecurityRule] = field(default_factory=list)
    egress_rules: List[SecurityRule] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.vpc_id:
            errors.append("VPC ID is required")
        for rule in self.ingress_rules + self.egress_rules:
            errors.extend(rule.validate())
        return errors


@dataclass
class LoadBalancerConfig(ResourceConfig):
    """Configuration for load balancer"""
    type: LoadBalancerType = LoadBalancerType.APPLICATION
    subnets: List[str] = field(default_factory=list)
    security_groups: List[str] = field(default_factory=list)
    internal: bool = False
    cross_zone: bool = True
    deletion_protection: bool = False
    access_logs: Optional[Dict[str, Any]] = None
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if len(self.subnets) < 2:
            errors.append("At least 2 subnets required for load balancer")
        return errors


@dataclass
class TargetGroupConfig:
    """Configuration for load balancer target group"""
    name: str
    protocol: str = "HTTP"
    port: int = 80
    vpc_id: str = ""
    health_check: Dict[str, Any] = field(default_factory=dict)
    targets: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ListenerConfig:
    """Configuration for load balancer listener"""
    protocol: str = "HTTP"
    port: int = 80
    default_action: Dict[str, Any] = field(default_factory=dict)
    rules: List[Dict[str, Any]] = field(default_factory=list)
    certificates: List[str] = field(default_factory=list)


@dataclass
class NetworkResource:
    """Representation of a network resource"""
    id: str
    name: str
    type: str
    state: str
    config: NetworkConfig
    subnets: List[Dict[str, Any]] = field(default_factory=list)
    route_tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseNetwork(BaseResource[NetworkConfig, NetworkResource]):
    """Base class for network resources (VPC)"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.NETWORK
    
    @abstractmethod
    async def create_subnet(self, vpc_id: str, config: SubnetConfig) -> str:
        """Create subnet in VPC"""
        pass
    
    @abstractmethod
    async def delete_subnet(self, subnet_id: str) -> bool:
        """Delete subnet"""
        pass
    
    @abstractmethod
    async def create_route_table(self, vpc_id: str, name: str) -> str:
        """Create route table"""
        pass
    
    @abstractmethod
    async def add_route(self, route_table_id: str, destination: str, target: str) -> bool:
        """Add route to route table"""
        pass
    
    @abstractmethod
    async def associate_route_table(self, subnet_id: str, route_table_id: str) -> bool:
        """Associate route table with subnet"""
        pass
    
    @abstractmethod
    async def create_internet_gateway(self, vpc_id: str) -> str:
        """Create and attach internet gateway"""
        pass
    
    @abstractmethod
    async def create_nat_gateway(self, subnet_id: str, allocation_id: str) -> str:
        """Create NAT gateway"""
        pass
    
    @abstractmethod
    async def create_peering_connection(self, vpc_id: str, peer_vpc_id: str, peer_region: Optional[str] = None) -> str:
        """Create VPC peering connection"""
        pass
    
    @abstractmethod
    async def enable_flow_logs(self, vpc_id: str, destination: str) -> bool:
        """Enable VPC flow logs"""
        pass


class BaseLoadBalancer(BaseResource[LoadBalancerConfig, Dict[str, Any]]):
    """Base class for load balancers"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.LOADBALANCER
    
    @abstractmethod
    async def create_target_group(self, config: TargetGroupConfig) -> str:
        """Create target group"""
        pass
    
    @abstractmethod
    async def register_targets(self, target_group_id: str, targets: List[Dict[str, Any]]) -> bool:
        """Register targets with target group"""
        pass
    
    @abstractmethod
    async def deregister_targets(self, target_group_id: str, targets: List[Dict[str, Any]]) -> bool:
        """Deregister targets from target group"""
        pass
    
    @abstractmethod
    async def create_listener(self, load_balancer_id: str, config: ListenerConfig) -> str:
        """Create listener for load balancer"""
        pass
    
    @abstractmethod
    async def add_listener_rule(self, listener_id: str, rule: Dict[str, Any]) -> str:
        """Add rule to listener"""
        pass
    
    @abstractmethod
    async def modify_health_check(self, target_group_id: str, health_check: Dict[str, Any]) -> bool:
        """Modify target group health check"""
        pass
    
    @abstractmethod
    async def get_metrics(self, load_balancer_id: str, metric_name: str, period: int = 300) -> List[Dict[str, Any]]:
        """Get load balancer metrics"""
        pass
    
    @abstractmethod
    async def enable_access_logs(self, load_balancer_id: str, s3_bucket: str, prefix: str = "") -> bool:
        """Enable access logging to S3"""
        pass


class BaseFirewall(BaseResource[FirewallConfig, Dict[str, Any]]):
    """Base class for firewalls/security groups"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.SECURITY
    
    @abstractmethod
    async def add_ingress_rule(self, firewall_id: str, rule: SecurityRule) -> bool:
        """Add ingress rule"""
        pass
    
    @abstractmethod
    async def add_egress_rule(self, firewall_id: str, rule: SecurityRule) -> bool:
        """Add egress rule"""
        pass
    
    @abstractmethod
    async def remove_ingress_rule(self, firewall_id: str, rule: SecurityRule) -> bool:
        """Remove ingress rule"""
        pass
    
    @abstractmethod
    async def remove_egress_rule(self, firewall_id: str, rule: SecurityRule) -> bool:
        """Remove egress rule"""
        pass
    
    @abstractmethod
    async def attach_to_instance(self, firewall_id: str, instance_id: str) -> bool:
        """Attach firewall to compute instance"""
        pass
    
    @abstractmethod
    async def detach_from_instance(self, firewall_id: str, instance_id: str) -> bool:
        """Detach firewall from compute instance"""
        pass