"""AWS storage resources implementation."""

import boto3
from typing import Dict, Any, List, Optional, BinaryIO, Union
from datetime import datetime
from pathlib import Path
import json

from common.core.base_provider import (
    ResourceInfo,
    ResourceStatus,
    StorageConfig,
)
from common.core.config import InfrastructureConfig
from common.core.exceptions import (
    ResourceCreationError,
    ResourceNotFoundError,
    StorageError,
)
from common.storage.base_storage import (
    BaseStorage,
    StorageObject,
    StorageMetadata,
)


class AWSStorage(BaseStorage):
    """AWS S3 storage implementation."""
    
    def __init__(self, session: boto3.Session, config: InfrastructureConfig):
        """Initialize AWS storage manager."""
        super().__init__(config.to_provider_config())
        self.session = session
        self.config = config
        self.s3_client = session.client("s3")
        self.s3_resource = session.resource("s3")
        self.sts_client = session.client("sts")
        
        # Get account ID
        self.account_id = self.sts_client.get_caller_identity()["Account"]
    
    def create_bucket(
        self,
        bucket_name: str,
        region: Optional[str] = None,
        versioning: bool = False,
        encryption: bool = True,
        lifecycle_rules: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Create an S3 bucket."""
        try:
            # Ensure bucket name is valid
            bucket_name = self._validate_bucket_name(bucket_name)
            
            # Create bucket
            create_params = {"Bucket": bucket_name}
            
            # Add location constraint for non us-east-1 regions
            region = region or self.config.region
            if region != "us-east-1":
                create_params["CreateBucketConfiguration"] = {
                    "LocationConstraint": region
                }
            
            self.s3_client.create_bucket(**create_params)
            
            # Enable versioning if requested
            if versioning:
                self.enable_versioning(bucket_name)
            
            # Enable encryption
            if encryption:
                self._enable_bucket_encryption(bucket_name)
            
            # Set lifecycle rules
            if lifecycle_rules:
                self.set_lifecycle_policy(bucket_name, lifecycle_rules)
            
            # Add tags
            if tags:
                self._tag_bucket(bucket_name, tags)
            
            # Block public access
            self._block_public_access(bucket_name)
            
            return True
            
        except Exception as e:
            raise StorageError("create_bucket", bucket_name, str(e))
    
    def create_bucket_from_config(self, config: StorageConfig) -> ResourceInfo:
        """Create S3 bucket from StorageConfig."""
        try:
            bucket_name = self._generate_bucket_name(config.name)
            
            # Create bucket
            self.create_bucket(
                bucket_name=bucket_name,
                region=config.region,
                versioning=config.versioning_enabled,
                encryption=config.encryption_enabled,
                lifecycle_rules=config.lifecycle_rules,
                tags=config.tags,
            )
            
            # Create bucket policy if needed
            if config.access_control == "private":
                self._set_bucket_policy_private(bucket_name)
            
            return ResourceInfo(
                resource_id=bucket_name,
                resource_type="s3_bucket",
                name=config.name,
                status=ResourceStatus.RUNNING,
                region=config.region,
                created_at=datetime.utcnow(),
                endpoint=f"https://{bucket_name}.s3.{config.region}.amazonaws.com",
                metadata={
                    "encryption": config.encryption_enabled,
                    "versioning": config.versioning_enabled,
                    "storage_class": config.storage_class,
                },
            )
            
        except Exception as e:
            raise ResourceCreationError("S3 bucket", str(e))
    
    def delete_bucket(self, bucket_name: str, force: bool = False) -> bool:
        """Delete an S3 bucket."""
        try:
            if force:
                # Empty bucket first
                self.empty_bucket(bucket_name)
            
            # Delete bucket
            self.s3_client.delete_bucket(Bucket=bucket_name)
            return True
            
        except Exception:
            return False
    
    def list_buckets(self) -> List[Dict[str, Any]]:
        """List all S3 buckets."""
        try:
            response = self.s3_client.list_buckets()
            
            buckets = []
            for bucket in response["Buckets"]:
                # Get bucket tags to filter
                try:
                    tags_response = self.s3_client.get_bucket_tagging(
                        Bucket=bucket["Name"]
                    )
                    tags = {
                        tag["Key"]: tag["Value"]
                        for tag in tags_response.get("TagSet", [])
                    }
                    
                    # Only include buckets managed by this project
                    if tags.get("Project") == self.config.project_name:
                        buckets.append({
                            "name": bucket["Name"],
                            "created": bucket["CreationDate"],
                            "tags": tags,
                        })
                except:
                    pass
            
            return buckets
            
        except Exception as e:
            raise StorageError("list_buckets", "", str(e))
    
    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists."""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except:
            return False
    
    def upload_file(
        self,
        file_path: Union[str, Path],
        bucket: str,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encryption: Optional[str] = None,
    ) -> StorageObject:
        """Upload a file to S3."""
        try:
            file_path = Path(file_path)
            
            # Prepare upload parameters
            extra_args = {}
            
            if metadata:
                extra_args["Metadata"] = metadata
            
            if content_type:
                extra_args["ContentType"] = content_type
            
            if encryption or self.config.storage_encryption:
                extra_args["ServerSideEncryption"] = "AES256"
            
            # Upload file
            self.s3_client.upload_file(
                str(file_path),
                bucket,
                key,
                ExtraArgs=extra_args,
            )
            
            # Get object metadata
            obj_metadata = self.get_object_metadata(bucket, key)
            
            return StorageObject(
                bucket=bucket,
                key=key,
                metadata=obj_metadata,
            )
            
        except Exception as e:
            raise StorageError("upload_file", bucket, str(e))
    
    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        bucket: str,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encryption: Optional[str] = None,
    ) -> StorageObject:
        """Upload a file object to S3."""
        try:
            # Prepare upload parameters
            extra_args = {}
            
            if metadata:
                extra_args["Metadata"] = metadata
            
            if content_type:
                extra_args["ContentType"] = content_type
            
            if encryption or self.config.storage_encryption:
                extra_args["ServerSideEncryption"] = "AES256"
            
            # Upload file object
            self.s3_client.upload_fileobj(
                file_obj,
                bucket,
                key,
                ExtraArgs=extra_args,
            )
            
            # Get object metadata
            obj_metadata = self.get_object_metadata(bucket, key)
            
            return StorageObject(
                bucket=bucket,
                key=key,
                metadata=obj_metadata,
            )
            
        except Exception as e:
            raise StorageError("upload_fileobj", bucket, str(e))
    
    def download_file(
        self,
        bucket: str,
        key: str,
        file_path: Union[str, Path],
        version_id: Optional[str] = None,
    ) -> bool:
        """Download a file from S3."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            extra_args = {}
            if version_id:
                extra_args["VersionId"] = version_id
            
            self.s3_client.download_file(
                bucket,
                key,
                str(file_path),
                ExtraArgs=extra_args if extra_args else None,
            )
            
            return True
            
        except Exception as e:
            raise StorageError("download_file", bucket, str(e))
    
    def download_fileobj(
        self,
        bucket: str,
        key: str,
        file_obj: BinaryIO,
        version_id: Optional[str] = None,
    ) -> bool:
        """Download to a file object from S3."""
        try:
            extra_args = {}
            if version_id:
                extra_args["VersionId"] = version_id
            
            self.s3_client.download_fileobj(
                bucket,
                key,
                file_obj,
                ExtraArgs=extra_args if extra_args else None,
            )
            
            return True
            
        except Exception as e:
            raise StorageError("download_fileobj", bucket, str(e))
    
    def get_object(
        self, bucket: str, key: str, version_id: Optional[str] = None
    ) -> bytes:
        """Get object content as bytes."""
        try:
            params = {"Bucket": bucket, "Key": key}
            if version_id:
                params["VersionId"] = version_id
            
            response = self.s3_client.get_object(**params)
            return response["Body"].read()
            
        except Exception as e:
            raise StorageError("get_object", bucket, str(e))
    
    def put_object(
        self,
        bucket: str,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encryption: Optional[str] = None,
    ) -> StorageObject:
        """Put object data directly to S3."""
        try:
            params = {
                "Bucket": bucket,
                "Key": key,
                "Body": data,
            }
            
            if metadata:
                params["Metadata"] = metadata
            
            if content_type:
                params["ContentType"] = content_type
            
            if encryption or self.config.storage_encryption:
                params["ServerSideEncryption"] = "AES256"
            
            response = self.s3_client.put_object(**params)
            
            # Get object metadata
            obj_metadata = self.get_object_metadata(bucket, key)
            
            return StorageObject(
                bucket=bucket,
                key=key,
                metadata=obj_metadata,
                version_id=response.get("VersionId"),
            )
            
        except Exception as e:
            raise StorageError("put_object", bucket, str(e))
    
    def delete_object(
        self, bucket: str, key: str, version_id: Optional[str] = None
    ) -> bool:
        """Delete an object from S3."""
        try:
            params = {"Bucket": bucket, "Key": key}
            if version_id:
                params["VersionId"] = version_id
            
            self.s3_client.delete_object(**params)
            return True
            
        except Exception:
            return False
    
    def delete_objects(
        self, bucket: str, keys: List[str]
    ) -> Dict[str, Union[bool, str]]:
        """Delete multiple objects from S3."""
        try:
            # S3 delete_objects accepts max 1000 keys at a time
            results = {}
            
            for i in range(0, len(keys), 1000):
                batch = keys[i:i + 1000]
                
                response = self.s3_client.delete_objects(
                    Bucket=bucket,
                    Delete={
                        "Objects": [{"Key": key} for key in batch],
                        "Quiet": False,
                    },
                )
                
                # Process successful deletes
                for deleted in response.get("Deleted", []):
                    results[deleted["Key"]] = True
                
                # Process errors
                for error in response.get("Errors", []):
                    results[error["Key"]] = error["Message"]
            
            return results
            
        except Exception as e:
            raise StorageError("delete_objects", bucket, str(e))
    
    def list_objects(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List objects in an S3 bucket."""
        try:
            params = {
                "Bucket": bucket,
                "MaxKeys": max_keys,
            }
            
            if prefix:
                params["Prefix"] = prefix
            
            if delimiter:
                params["Delimiter"] = delimiter
            
            if continuation_token:
                params["ContinuationToken"] = continuation_token
            
            response = self.s3_client.list_objects_v2(**params)
            
            objects = []
            for obj in response.get("Contents", []):
                objects.append(
                    StorageObject(
                        bucket=bucket,
                        key=obj["Key"],
                        metadata=StorageMetadata(
                            size=obj["Size"],
                            content_type="application/octet-stream",
                            etag=obj["ETag"],
                            last_modified=obj["LastModified"],
                            storage_class=obj.get("StorageClass", "STANDARD"),
                        ),
                    )
                )
            
            return {
                "objects": objects,
                "prefixes": response.get("CommonPrefixes", []),
                "is_truncated": response.get("IsTruncated", False),
                "next_continuation_token": response.get("NextContinuationToken"),
                "key_count": response.get("KeyCount", 0),
            }
            
        except Exception as e:
            raise StorageError("list_objects", bucket, str(e))
    
    def get_object_metadata(
        self, bucket: str, key: str, version_id: Optional[str] = None
    ) -> StorageMetadata:
        """Get object metadata from S3."""
        try:
            params = {"Bucket": bucket, "Key": key}
            if version_id:
                params["VersionId"] = version_id
            
            response = self.s3_client.head_object(**params)
            
            return StorageMetadata(
                size=response["ContentLength"],
                content_type=response.get("ContentType", "application/octet-stream"),
                etag=response.get("ETag", "").strip('"'),
                last_modified=response["LastModified"],
                storage_class=response.get("StorageClass", "STANDARD"),
                encryption=response.get("ServerSideEncryption"),
                custom_metadata=response.get("Metadata", {}),
            )
            
        except Exception as e:
            raise StorageError("get_object_metadata", bucket, str(e))
    
    def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Copy an object in S3."""
        try:
            copy_source = {"Bucket": source_bucket, "Key": source_key}
            
            params = {
                "Bucket": dest_bucket,
                "Key": dest_key,
                "CopySource": copy_source,
            }
            
            if metadata:
                params["Metadata"] = metadata
                params["MetadataDirective"] = "REPLACE"
            else:
                params["MetadataDirective"] = "COPY"
            
            if self.config.storage_encryption:
                params["ServerSideEncryption"] = "AES256"
            
            response = self.s3_client.copy_object(**params)
            
            # Get object metadata
            obj_metadata = self.get_object_metadata(dest_bucket, dest_key)
            
            return StorageObject(
                bucket=dest_bucket,
                key=dest_key,
                metadata=obj_metadata,
                version_id=response.get("VersionId"),
            )
            
        except Exception as e:
            raise StorageError("copy_object", dest_bucket, str(e))
    
    def generate_presigned_url(
        self,
        bucket: str,
        key: str,
        operation: str = "get_object",
        expires_in: int = 3600,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a presigned URL for S3."""
        try:
            url_params = {"Bucket": bucket, "Key": key}
            if params:
                url_params.update(params)
            
            return self.s3_client.generate_presigned_url(
                ClientMethod=operation,
                Params=url_params,
                ExpiresIn=expires_in,
            )
            
        except Exception as e:
            raise StorageError("generate_presigned_url", bucket, str(e))
    
    def enable_versioning(self, bucket: str) -> bool:
        """Enable versioning on an S3 bucket."""
        try:
            self.s3_client.put_bucket_versioning(
                Bucket=bucket,
                VersioningConfiguration={"Status": "Enabled"},
            )
            return True
            
        except Exception as e:
            raise StorageError("enable_versioning", bucket, str(e))
    
    def set_lifecycle_policy(
        self, bucket: str, rules: List[Dict[str, Any]]
    ) -> bool:
        """Set lifecycle policy on an S3 bucket."""
        try:
            # Format rules for S3 API
            lifecycle_rules = []
            
            for rule in rules:
                s3_rule = {
                    "ID": rule.get("id", f"rule-{len(lifecycle_rules)}"),
                    "Status": "Enabled" if rule.get("enabled", True) else "Disabled",
                    "Filter": {},
                }
                
                # Add prefix filter if specified
                if "prefix" in rule:
                    s3_rule["Filter"]["Prefix"] = rule["prefix"]
                
                # Add transitions
                if "transitions" in rule:
                    s3_rule["Transitions"] = []
                    for transition in rule["transitions"]:
                        s3_rule["Transitions"].append({
                            "Days": transition["days"],
                            "StorageClass": transition.get("storage_class", "GLACIER"),
                        })
                
                # Add expiration
                if "expiration_days" in rule:
                    s3_rule["Expiration"] = {"Days": rule["expiration_days"]}
                
                # Add noncurrent version transitions
                if "noncurrent_transitions" in rule:
                    s3_rule["NoncurrentVersionTransitions"] = []
                    for transition in rule["noncurrent_transitions"]:
                        s3_rule["NoncurrentVersionTransitions"].append({
                            "NoncurrentDays": transition["days"],
                            "StorageClass": transition.get("storage_class", "GLACIER"),
                        })
                
                # Add noncurrent version expiration
                if "noncurrent_expiration_days" in rule:
                    s3_rule["NoncurrentVersionExpiration"] = {
                        "NoncurrentDays": rule["noncurrent_expiration_days"]
                    }
                
                lifecycle_rules.append(s3_rule)
            
            # Apply lifecycle configuration
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket,
                LifecycleConfiguration={"Rules": lifecycle_rules},
            )
            
            return True
            
        except Exception as e:
            raise StorageError("set_lifecycle_policy", bucket, str(e))
    
    def get_bucket(self, bucket_name: str) -> ResourceInfo:
        """Get S3 bucket information."""
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=bucket_name)
            
            # Get bucket location
            location_response = self.s3_client.get_bucket_location(Bucket=bucket_name)
            region = location_response.get("LocationConstraint") or "us-east-1"
            
            # Get bucket tags
            tags = {}
            try:
                tags_response = self.s3_client.get_bucket_tagging(Bucket=bucket_name)
                tags = {
                    tag["Key"]: tag["Value"]
                    for tag in tags_response.get("TagSet", [])
                }
            except:
                pass
            
            # Get versioning status
            versioning = False
            try:
                ver_response = self.s3_client.get_bucket_versioning(Bucket=bucket_name)
                versioning = ver_response.get("Status") == "Enabled"
            except:
                pass
            
            # Get encryption status
            encryption = False
            try:
                enc_response = self.s3_client.get_bucket_encryption(Bucket=bucket_name)
                encryption = bool(enc_response.get("ServerSideEncryptionConfiguration"))
            except:
                pass
            
            return ResourceInfo(
                resource_id=bucket_name,
                resource_type="s3_bucket",
                name=bucket_name,
                status=ResourceStatus.RUNNING,
                region=region,
                created_at=datetime.utcnow(),  # S3 doesn't provide creation time via API
                endpoint=f"https://{bucket_name}.s3.{region}.amazonaws.com",
                metadata={
                    "versioning": versioning,
                    "encryption": encryption,
                    "tags": tags,
                },
            )
            
        except Exception:
            raise ResourceNotFoundError(bucket_name, "S3 bucket")
    
    def create_efs_filesystem(self, filesystem_config: Dict[str, Any]) -> ResourceInfo:
        """Create an EFS filesystem for shared storage."""
        try:
            efs_client = self.session.client("efs")
            
            # Create filesystem
            response = efs_client.create_file_system(
                CreationToken=filesystem_config["name"],
                PerformanceMode=filesystem_config.get("performance_mode", "generalPurpose"),
                ThroughputMode=filesystem_config.get("throughput_mode", "bursting"),
                Encrypted=self.config.storage_encryption,
                Tags=self._format_tags({
                    **filesystem_config.get("tags", {}),
                    "Name": filesystem_config["name"],
                }),
            )
            
            filesystem_id = response["FileSystemId"]
            
            # Create mount targets in each subnet
            subnets = filesystem_config.get("subnets", self._get_private_subnets())
            security_groups = filesystem_config.get(
                "security_groups",
                [self._get_or_create_efs_security_group()],
            )
            
            for subnet in subnets:
                efs_client.create_mount_target(
                    FileSystemId=filesystem_id,
                    SubnetId=subnet,
                    SecurityGroups=security_groups,
                )
            
            return ResourceInfo(
                resource_id=filesystem_id,
                resource_type="efs_filesystem",
                name=filesystem_config["name"],
                status=ResourceStatus.CREATING,
                region=self.config.region,
                created_at=datetime.utcnow(),
                endpoint=f"{filesystem_id}.efs.{self.config.region}.amazonaws.com",
                metadata={
                    "performance_mode": filesystem_config.get("performance_mode", "generalPurpose"),
                    "throughput_mode": filesystem_config.get("throughput_mode", "bursting"),
                    "encrypted": self.config.storage_encryption,
                },
            )
            
        except Exception as e:
            raise ResourceCreationError("EFS filesystem", str(e))
    
    def _validate_bucket_name(self, name: str) -> str:
        """Validate and format S3 bucket name."""
        # S3 bucket names must be globally unique and follow specific rules
        name = name.lower().replace("_", "-")
        
        # Ensure it starts and ends with alphanumeric
        if not name[0].isalnum():
            name = "a" + name[1:]
        if not name[-1].isalnum():
            name = name[:-1] + "a"
        
        # Limit length
        if len(name) > 63:
            name = name[:63]
        
        return name
    
    def _generate_bucket_name(self, base_name: str) -> str:
        """Generate a unique S3 bucket name."""
        # Add account ID for uniqueness
        name = f"{base_name}-{self.account_id}"
        return self._validate_bucket_name(name)
    
    def _enable_bucket_encryption(self, bucket_name: str) -> None:
        """Enable default encryption on S3 bucket."""
        self.s3_client.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        }
                    }
                ]
            },
        )
    
    def _block_public_access(self, bucket_name: str) -> None:
        """Block all public access to S3 bucket."""
        self.s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
    
    def _tag_bucket(self, bucket_name: str, tags: Dict[str, str]) -> None:
        """Add tags to S3 bucket."""
        # Add default tags
        all_tags = {
            **self.config.default_tags,
            **tags,
        }
        
        self.s3_client.put_bucket_tagging(
            Bucket=bucket_name,
            Tagging={
                "TagSet": [
                    {"Key": k, "Value": v}
                    for k, v in all_tags.items()
                ]
            },
        )
    
    def _set_bucket_policy_private(self, bucket_name: str) -> None:
        """Set bucket policy to enforce private access."""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "DenyInsecureConnections",
                    "Effect": "Deny",
                    "Principal": "*",
                    "Action": "s3:*",
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}/*",
                        f"arn:aws:s3:::{bucket_name}",
                    ],
                    "Condition": {
                        "Bool": {
                            "aws:SecureTransport": "false"
                        }
                    },
                }
            ],
        }
        
        self.s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(policy),
        )
    
    def _get_private_subnets(self) -> List[str]:
        """Get private subnet IDs."""
        ec2_client = self.session.client("ec2")
        
        # For now, return default subnets
        # In production, this should return actual private subnets
        response = ec2_client.describe_subnets(
            Filters=[
                {"Name": "default-for-az", "Values": ["true"]},
            ]
        )
        
        return [subnet["SubnetId"] for subnet in response["Subnets"][:2]]
    
    def _get_or_create_efs_security_group(self) -> str:
        """Get or create EFS security group."""
        ec2_client = self.session.client("ec2")
        group_name = f"{self.config.resource_name}-efs-sg"
        
        try:
            response = ec2_client.describe_security_groups(
                GroupNames=[group_name]
            )
            return response["SecurityGroups"][0]["GroupId"]
        except:
            # Get default VPC
            vpc_response = ec2_client.describe_vpcs(
                Filters=[{"Name": "is-default", "Values": ["true"]}]
            )
            vpc_id = vpc_response["Vpcs"][0]["VpcId"]
            
            # Create security group
            response = ec2_client.create_security_group(
                GroupName=group_name,
                Description=f"Security group for {self.config.project_name} EFS",
                VpcId=vpc_id,
                TagSpecifications=[
                    {
                        "ResourceType": "security-group",
                        "Tags": self._format_tags({"Name": group_name}),
                    }
                ],
            )
            
            group_id = response["GroupId"]
            
            # Add ingress rule for NFS
            ec2_client.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 2049,
                        "ToPort": 2049,
                        "UserIdGroupPairs": [{"GroupId": group_id}],
                    }
                ],
            )
            
            return group_id
    
    def _format_tags(self, tags: Dict[str, str]) -> List[Dict[str, str]]:
        """Format tags for AWS API."""
        return [{"Key": k, "Value": v} for k, v in tags.items()]