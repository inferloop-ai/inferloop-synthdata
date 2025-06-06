# src/delivery/exporters.py
import boto3
import os
from typing import Dict, Any, List

class S3Exporter:
    """Export generated data to S3"""
    
    def __init__(self, bucket_name: str, aws_access_key: str = None, aws_secret_key: str = None):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key or os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
    
    def export_jsonl(self, data: str, key: str) -> str:
        """Export JSONL data to S3"""
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType='application/jsonl'
        )
        return f"s3://{self.bucket_name}/{key}"
    
    def export_csv(self, data: str, key: str) -> str:
        """Export CSV data to S3"""
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType='text/csv'
        )
        return f"s3://{self.bucket_name}/{key}"
