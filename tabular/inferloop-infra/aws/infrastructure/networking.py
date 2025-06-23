"""AWS networking resources implementation."""

import boto3
from typing import Dict, Any, List, Optional
from datetime import datetime
import ipaddress

from common.core.base_provider import (
    ResourceInfo,
    ResourceStatus,
    NetworkConfig,
)
from common.core.config import InfrastructureConfig
from common.core.exceptions import (
    ResourceCreationError,
    ResourceNotFoundError,
    NetworkError,
)


class AWSNetworking:
    """AWS networking resources management."""
    
    def __init__(self, session: boto3.Session, config: InfrastructureConfig):
        """Initialize AWS networking manager."""
        self.session = session
        self.config = config
        self.ec2_client = session.client("ec2")
        self.ec2_resource = session.resource("ec2")
        self.elbv2_client = session.client("elbv2")
        self.route53_client = session.client("route53")
        self.cloudfront_client = session.client("cloudfront")
    
    def create_vpc(self, config: NetworkConfig) -> ResourceInfo:
        """Create a VPC with subnets."""
        try:
            # Create VPC
            vpc_response = self.ec2_client.create_vpc(
                CidrBlock=config.cidr_block,
                TagSpecifications=[
                    {
                        "ResourceType": "vpc",
                        "Tags": self._format_tags({**config.tags, "Name": config.name}),
                    }
                ],
            )
            
            vpc_id = vpc_response["Vpc"]["VpcId"]
            vpc = self.ec2_resource.Vpc(vpc_id)
            
            # Wait for VPC to be available
            vpc.wait_until_available()
            
            # Enable DNS
            if config.enable_dns:
                self.ec2_client.modify_vpc_attribute(
                    VpcId=vpc_id,
                    EnableDnsHostnames={"Value": True},
                )
                self.ec2_client.modify_vpc_attribute(
                    VpcId=vpc_id,
                    EnableDnsSupport={"Value": True},
                )
            
            # Create Internet Gateway
            igw = self.ec2_resource.create_internet_gateway(
                TagSpecifications=[
                    {
                        "ResourceType": "internet-gateway",
                        "Tags": self._format_tags({"Name": f"{config.name}-igw"}),
                    }
                ]
            )
            vpc.attach_internet_gateway(InternetGatewayId=igw.id)
            
            # Calculate subnet CIDR blocks
            network = ipaddress.IPv4Network(config.cidr_block)
            subnets = list(network.subnets(new_prefix=24))[:6]  # Get 6 /24 subnets
            
            # Get availability zones
            azs_response = self.ec2_client.describe_availability_zones(
                Filters=[{"Name": "state", "Values": ["available"]}]
            )
            azs = [az["ZoneName"] for az in azs_response["AvailabilityZones"][:3]]
            
            # Create subnets
            public_subnet_ids = []
            private_subnet_ids = []
            
            for i, az in enumerate(azs):
                # Public subnet
                public_subnet = self._create_subnet(
                    vpc_id,
                    str(subnets[i]),
                    az,
                    f"{config.name}-public-{az}",
                    True,
                )
                public_subnet_ids.append(public_subnet.id)
                
                # Private subnet
                if i + 3 < len(subnets):
                    private_subnet = self._create_subnet(
                        vpc_id,
                        str(subnets[i + 3]),
                        az,
                        f"{config.name}-private-{az}",
                        False,
                    )
                    private_subnet_ids.append(private_subnet.id)
            
            # Create route table for public subnets
            public_rt = self._create_route_table(
                vpc_id,
                f"{config.name}-public-rt",
                igw.id,
                public_subnet_ids,
            )
            
            # Create NAT Gateway for private subnets (if enabled)
            if config.enable_dns and private_subnet_ids:
                nat_gateway_id = self._create_nat_gateway(
                    public_subnet_ids[0],
                    f"{config.name}-nat",
                )
                
                # Create route table for private subnets
                private_rt = self._create_route_table(
                    vpc_id,
                    f"{config.name}-private-rt",
                    nat_gateway_id,
                    private_subnet_ids,
                    is_nat=True,
                )
            
            # Create default security group
            if config.enable_firewall:
                sg_id = self._create_default_security_group(
                    vpc_id,
                    config.name,
                    config.allowed_ports,
                    config.allowed_ip_ranges,
                )
            
            # Create VPC endpoints if enabled
            if self.config.enable_vpc_endpoints:
                self._create_vpc_endpoints(vpc_id, route_table_ids=[public_rt])
            
            return ResourceInfo(
                resource_id=vpc_id,
                resource_type="vpc",
                name=config.name,
                status=ResourceStatus.RUNNING,
                region=config.region,
                created_at=datetime.utcnow(),
                metadata={
                    "cidr_block": config.cidr_block,
                    "public_subnets": public_subnet_ids,
                    "private_subnets": private_subnet_ids,
                    "internet_gateway": igw.id,
                    "nat_gateway": nat_gateway_id if private_subnet_ids else None,
                },
            )
            
        except Exception as e:
            raise ResourceCreationError("VPC", str(e))
    
    def create_load_balancer(self, lb_config: Dict[str, Any]) -> ResourceInfo:
        """Create an Application Load Balancer."""
        try:
            # Create ALB
            response = self.elbv2_client.create_load_balancer(
                Name=lb_config["name"],
                Subnets=lb_config["subnets"],
                SecurityGroups=lb_config.get("security_groups", []),
                Scheme=lb_config.get("scheme", "internet-facing"),
                Type="application",
                IpAddressType="ipv4",
                Tags=self._format_tags(lb_config.get("tags", {})),
            )
            
            alb = response["LoadBalancers"][0]
            alb_arn = alb["LoadBalancerArn"]
            alb_dns = alb["DNSName"]
            
            # Create target group
            tg_response = self.elbv2_client.create_target_group(
                Name=f"{lb_config['name']}-tg"[:32],
                Protocol="HTTP",
                Port=lb_config.get("target_port", 80),
                VpcId=lb_config["vpc_id"],
                TargetType=lb_config.get("target_type", "instance"),
                HealthCheckProtocol="HTTP",
                HealthCheckPath=lb_config.get("health_check_path", "/"),
                HealthCheckIntervalSeconds=30,
                HealthCheckTimeoutSeconds=5,
                HealthyThresholdCount=2,
                UnhealthyThresholdCount=3,
                Tags=self._format_tags({"Name": f"{lb_config['name']}-tg"}),
            )
            
            target_group_arn = tg_response["TargetGroups"][0]["TargetGroupArn"]
            
            # Create listener
            listener_response = self.elbv2_client.create_listener(
                LoadBalancerArn=alb_arn,
                Protocol="HTTP",
                Port=80,
                DefaultActions=[
                    {
                        "Type": "forward",
                        "TargetGroupArn": target_group_arn,
                    }
                ],
            )
            
            # Add HTTPS listener if certificate provided
            if self.config.ssl_certificate_arn:
                self.elbv2_client.create_listener(
                    LoadBalancerArn=alb_arn,
                    Protocol="HTTPS",
                    Port=443,
                    Certificates=[
                        {
                            "CertificateArn": self.config.ssl_certificate_arn,
                        }
                    ],
                    DefaultActions=[
                        {
                            "Type": "forward",
                            "TargetGroupArn": target_group_arn,
                        }
                    ],
                )
            
            return ResourceInfo(
                resource_id=alb_arn,
                resource_type="alb",
                name=lb_config["name"],
                status=ResourceStatus.RUNNING,
                region=self.config.region,
                created_at=alb["CreatedTime"],
                endpoint=alb_dns,
                metadata={
                    "dns_name": alb_dns,
                    "target_group_arn": target_group_arn,
                    "scheme": alb["Scheme"],
                    "vpc_id": alb["VpcId"],
                },
            )
            
        except Exception as e:
            raise ResourceCreationError("Load Balancer", str(e))
    
    def create_cloudfront_distribution(self, cf_config: Dict[str, Any]) -> ResourceInfo:
        """Create a CloudFront distribution."""
        try:
            # Create distribution configuration
            distribution_config = {
                "CallerReference": f"{cf_config['name']}-{datetime.utcnow().timestamp()}",
                "Comment": cf_config.get("comment", f"Distribution for {cf_config['name']}"),
                "Enabled": True,
                "Origins": {
                    "Quantity": 1,
                    "Items": [
                        {
                            "Id": "primary-origin",
                            "DomainName": cf_config["origin_domain"],
                            "CustomOriginConfig": {
                                "HTTPPort": 80,
                                "HTTPSPort": 443,
                                "OriginProtocolPolicy": "https-only",
                                "OriginSslProtocols": {
                                    "Quantity": 1,
                                    "Items": ["TLSv1.2"],
                                },
                            },
                        }
                    ],
                },
                "DefaultRootObject": cf_config.get("default_root_object", "index.html"),
                "DefaultCacheBehavior": {
                    "TargetOriginId": "primary-origin",
                    "ViewerProtocolPolicy": "redirect-to-https",
                    "AllowedMethods": {
                        "Quantity": 7,
                        "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
                        "CachedMethods": {
                            "Quantity": 2,
                            "Items": ["GET", "HEAD"],
                        },
                    },
                    "Compress": True,
                    "ForwardedValues": {
                        "QueryString": True,
                        "Cookies": {"Forward": "all"},
                        "Headers": {
                            "Quantity": 1,
                            "Items": ["*"],
                        },
                    },
                    "TrustedSigners": {
                        "Enabled": False,
                        "Quantity": 0,
                    },
                    "MinTTL": 0,
                    "DefaultTTL": 86400,
                    "MaxTTL": 31536000,
                },
                "PriceClass": cf_config.get("price_class", "PriceClass_100"),
            }
            
            # Add custom domain if provided
            if cf_config.get("domain_names"):
                distribution_config["Aliases"] = {
                    "Quantity": len(cf_config["domain_names"]),
                    "Items": cf_config["domain_names"],
                }
                
                if self.config.ssl_certificate_arn:
                    distribution_config["ViewerCertificate"] = {
                        "ACMCertificateArn": self.config.ssl_certificate_arn,
                        "SSLSupportMethod": "sni-only",
                        "MinimumProtocolVersion": "TLSv1.2_2021",
                    }
            
            # Create distribution
            response = self.cloudfront_client.create_distribution(
                DistributionConfig=distribution_config
            )
            
            distribution = response["Distribution"]
            
            return ResourceInfo(
                resource_id=distribution["Id"],
                resource_type="cloudfront_distribution",
                name=cf_config["name"],
                status=ResourceStatus.CREATING,
                region="global",  # CloudFront is global
                created_at=datetime.utcnow(),
                endpoint=distribution["DomainName"],
                metadata={
                    "domain_name": distribution["DomainName"],
                    "status": distribution["Status"],
                    "enabled": distribution["DistributionConfig"]["Enabled"],
                },
            )
            
        except Exception as e:
            raise ResourceCreationError("CloudFront Distribution", str(e))
    
    def create_route53_zone(self, zone_config: Dict[str, Any]) -> ResourceInfo:
        """Create a Route53 hosted zone."""
        try:
            # Create hosted zone
            response = self.route53_client.create_hosted_zone(
                Name=zone_config["domain_name"],
                CallerReference=f"{zone_config['domain_name']}-{datetime.utcnow().timestamp()}",
                HostedZoneConfig={
                    "Comment": zone_config.get("comment", f"Zone for {zone_config['domain_name']}"),
                    "PrivateZone": zone_config.get("private", False),
                },
            )
            
            zone = response["HostedZone"]
            zone_id = zone["Id"].split("/")[-1]
            
            # Get name servers
            ns_response = self.route53_client.list_resource_record_sets(
                HostedZoneId=zone_id,
                StartRecordType="NS",
                StartRecordName=zone_config["domain_name"],
            )
            
            name_servers = []
            for record_set in ns_response["ResourceRecordSets"]:
                if record_set["Type"] == "NS":
                    name_servers = [r["Value"] for r in record_set["ResourceRecords"]]
                    break
            
            return ResourceInfo(
                resource_id=zone_id,
                resource_type="route53_zone",
                name=zone_config["domain_name"],
                status=ResourceStatus.RUNNING,
                region="global",  # Route53 is global
                created_at=datetime.utcnow(),
                metadata={
                    "name_servers": name_servers,
                    "private": zone_config.get("private", False),
                },
            )
            
        except Exception as e:
            raise ResourceCreationError("Route53 Zone", str(e))
    
    def get_vpc(self, vpc_id: str) -> ResourceInfo:
        """Get VPC information."""
        try:
            response = self.ec2_client.describe_vpcs(VpcIds=[vpc_id])
            
            if not response["Vpcs"]:
                raise ResourceNotFoundError(vpc_id, "VPC")
            
            vpc = response["Vpcs"][0]
            
            # Get associated resources
            subnets_response = self.ec2_client.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )
            
            public_subnets = []
            private_subnets = []
            
            for subnet in subnets_response["Subnets"]:
                if subnet.get("MapPublicIpOnLaunch", False):
                    public_subnets.append(subnet["SubnetId"])
                else:
                    private_subnets.append(subnet["SubnetId"])
            
            return ResourceInfo(
                resource_id=vpc_id,
                resource_type="vpc",
                name=self._get_tag_value(vpc.get("Tags", []), "Name"),
                status=ResourceStatus.RUNNING,
                region=self.config.region,
                created_at=datetime.utcnow(),  # VPC doesn't have creation time in API
                metadata={
                    "cidr_block": vpc["CidrBlock"],
                    "public_subnets": public_subnets,
                    "private_subnets": private_subnets,
                    "is_default": vpc.get("IsDefault", False),
                },
            )
            
        except Exception:
            raise ResourceNotFoundError(vpc_id, "VPC")
    
    def list_vpcs(self) -> List[ResourceInfo]:
        """List all VPCs."""
        vpcs = []
        
        try:
            response = self.ec2_client.describe_vpcs(
                Filters=[
                    {
                        "Name": "tag:Project",
                        "Values": [self.config.project_name],
                    }
                ]
            )
            
            for vpc in response["Vpcs"]:
                try:
                    vpcs.append(self.get_vpc(vpc["VpcId"]))
                except Exception:
                    pass
                    
        except Exception:
            pass
        
        return vpcs
    
    def delete_vpc(self, vpc_id: str) -> bool:
        """Delete a VPC and all associated resources."""
        try:
            # Delete subnets
            subnets = self.ec2_client.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )
            for subnet in subnets["Subnets"]:
                self.ec2_client.delete_subnet(SubnetId=subnet["SubnetId"])
            
            # Delete route tables
            route_tables = self.ec2_client.describe_route_tables(
                Filters=[
                    {"Name": "vpc-id", "Values": [vpc_id]},
                    {"Name": "association.main", "Values": ["false"]},
                ]
            )
            for rt in route_tables["RouteTables"]:
                self.ec2_client.delete_route_table(RouteTableId=rt["RouteTableId"])
            
            # Detach and delete internet gateways
            igws = self.ec2_client.describe_internet_gateways(
                Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}]
            )
            for igw in igws["InternetGateways"]:
                self.ec2_client.detach_internet_gateway(
                    InternetGatewayId=igw["InternetGatewayId"],
                    VpcId=vpc_id,
                )
                self.ec2_client.delete_internet_gateway(
                    InternetGatewayId=igw["InternetGatewayId"]
                )
            
            # Delete NAT gateways
            nat_gateways = self.ec2_client.describe_nat_gateways(
                Filters=[
                    {"Name": "vpc-id", "Values": [vpc_id]},
                    {"Name": "state", "Values": ["available"]},
                ]
            )
            for nat in nat_gateways["NatGateways"]:
                self.ec2_client.delete_nat_gateway(NatGatewayId=nat["NatGatewayId"])
            
            # Delete VPC
            self.ec2_client.delete_vpc(VpcId=vpc_id)
            
            return True
            
        except Exception:
            return False
    
    def _create_subnet(
        self,
        vpc_id: str,
        cidr_block: str,
        az: str,
        name: str,
        public: bool,
    ) -> Any:
        """Create a subnet."""
        subnet = self.ec2_resource.create_subnet(
            VpcId=vpc_id,
            CidrBlock=cidr_block,
            AvailabilityZone=az,
            TagSpecifications=[
                {
                    "ResourceType": "subnet",
                    "Tags": self._format_tags({"Name": name}),
                }
            ],
        )
        
        if public:
            self.ec2_client.modify_subnet_attribute(
                SubnetId=subnet.id,
                MapPublicIpOnLaunch={"Value": True},
            )
        
        return subnet
    
    def _create_route_table(
        self,
        vpc_id: str,
        name: str,
        gateway_id: str,
        subnet_ids: List[str],
        is_nat: bool = False,
    ) -> str:
        """Create and configure a route table."""
        # Create route table
        rt_response = self.ec2_client.create_route_table(
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    "ResourceType": "route-table",
                    "Tags": self._format_tags({"Name": name}),
                }
            ],
        )
        
        rt_id = rt_response["RouteTable"]["RouteTableId"]
        
        # Add route
        if is_nat:
            self.ec2_client.create_route(
                RouteTableId=rt_id,
                DestinationCidrBlock="0.0.0.0/0",
                NatGatewayId=gateway_id,
            )
        else:
            self.ec2_client.create_route(
                RouteTableId=rt_id,
                DestinationCidrBlock="0.0.0.0/0",
                GatewayId=gateway_id,
            )
        
        # Associate with subnets
        for subnet_id in subnet_ids:
            self.ec2_client.associate_route_table(
                RouteTableId=rt_id,
                SubnetId=subnet_id,
            )
        
        return rt_id
    
    def _create_nat_gateway(self, subnet_id: str, name: str) -> str:
        """Create a NAT gateway."""
        # Allocate Elastic IP
        eip_response = self.ec2_client.allocate_address(
            Domain="vpc",
            TagSpecifications=[
                {
                    "ResourceType": "elastic-ip",
                    "Tags": self._format_tags({"Name": f"{name}-eip"}),
                }
            ],
        )
        
        # Create NAT gateway
        nat_response = self.ec2_client.create_nat_gateway(
            SubnetId=subnet_id,
            AllocationId=eip_response["AllocationId"],
            TagSpecifications=[
                {
                    "ResourceType": "natgateway",
                    "Tags": self._format_tags({"Name": name}),
                }
            ],
        )
        
        nat_gateway_id = nat_response["NatGateway"]["NatGatewayId"]
        
        # Wait for NAT gateway to be available
        waiter = self.ec2_client.get_waiter("nat_gateway_available")
        waiter.wait(NatGatewayIds=[nat_gateway_id])
        
        return nat_gateway_id
    
    def _create_default_security_group(
        self,
        vpc_id: str,
        name: str,
        allowed_ports: List[int],
        allowed_ips: List[str],
    ) -> str:
        """Create a default security group."""
        sg_response = self.ec2_client.create_security_group(
            GroupName=f"{name}-default-sg",
            Description=f"Default security group for {name}",
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    "ResourceType": "security-group",
                    "Tags": self._format_tags({"Name": f"{name}-default-sg"}),
                }
            ],
        )
        
        sg_id = sg_response["GroupId"]
        
        # Add ingress rules
        rules = []
        
        # Allow internal VPC traffic
        rules.append({
            "IpProtocol": "-1",
            "UserIdGroupPairs": [{"GroupId": sg_id}],
        })
        
        # Allow specified ports from allowed IPs
        for port in allowed_ports:
            rules.append({
                "IpProtocol": "tcp",
                "FromPort": port,
                "ToPort": port,
                "IpRanges": [{"CidrIp": ip} for ip in allowed_ips],
            })
        
        if rules:
            self.ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=rules,
            )
        
        return sg_id
    
    def _create_vpc_endpoints(self, vpc_id: str, route_table_ids: List[str]) -> None:
        """Create VPC endpoints for AWS services."""
        # S3 endpoint
        try:
            self.ec2_client.create_vpc_endpoint(
                VpcId=vpc_id,
                ServiceName=f"com.amazonaws.{self.config.region}.s3",
                VpcEndpointType="Gateway",
                RouteTableIds=route_table_ids,
                TagSpecifications=[
                    {
                        "ResourceType": "vpc-endpoint",
                        "Tags": self._format_tags({"Name": f"{vpc_id}-s3-endpoint"}),
                    }
                ],
            )
        except:
            pass
        
        # DynamoDB endpoint
        try:
            self.ec2_client.create_vpc_endpoint(
                VpcId=vpc_id,
                ServiceName=f"com.amazonaws.{self.config.region}.dynamodb",
                VpcEndpointType="Gateway",
                RouteTableIds=route_table_ids,
                TagSpecifications=[
                    {
                        "ResourceType": "vpc-endpoint",
                        "Tags": self._format_tags({"Name": f"{vpc_id}-dynamodb-endpoint"}),
                    }
                ],
            )
        except:
            pass
    
    def _format_tags(self, tags: Dict[str, str]) -> List[Dict[str, str]]:
        """Format tags for AWS API."""
        return [{"Key": k, "Value": v} for k, v in tags.items()]
    
    def _get_tag_value(self, tags: List[Dict[str, str]], key: str) -> Optional[str]:
        """Get tag value from AWS tags list."""
        for tag in tags:
            if tag.get("Key") == key:
                return tag.get("Value")
        return None