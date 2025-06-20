#!/usr/bin/env python3
"""
Export command for converting documents to various formats.

Provides CLI interface for exporting synthetic documents to different
formats including COCO, YOLO, Pascal VOC, JSON, and CSV.
"""

import asyncio
import click
import json
from pathlib import Path
from typing import List, Optional

from ...core import get_logger
from ...delivery.export import create_batch_exporter, get_supported_export_formats
from ..utils import OutputFormatter, ProgressTracker


logger = get_logger(__name__)


@click.command(name='export')
@click.argument(
    'input_path',
    type=click.Path(exists=True, path_type=Path)
)
@click.argument(
    'output_path',
    type=click.Path(path_type=Path)
)
@click.option(
    '--format', '-f',
    type=click.Choice(['coco', 'yolo', 'pascal_voc', 'json', 'jsonl', 'csv']),
    required=True,
    help='Export format'
)
@click.option(
    '--batch-size',
    type=int,
    default=100,
    help='Batch size for processing'
)
@click.option(
    '--recursive', '-R',
    is_flag=True,
    help='Process directories recursively'
)
@click.option(
    '--filter',
    type=str,
    help='Filter pattern for files (e.g., "*.pdf")'
)
@click.option(
    '--privacy/--no-privacy',
    default=True,
    help='Apply privacy protection during export'
)
@click.option(
    '--copy-images/--no-copy-images',
    default=True,
    help='Copy images to output directory (for image formats)'
)
@click.option(
    '--split-ratio',
    type=str,
    help='Train/val/test split ratio (e.g., "70:20:10")'
)
@click.option(
    '--max-workers',
    type=int,
    default=4,
    help='Maximum parallel workers'
)
@click.option(
    '--resume',
    is_flag=True,
    help='Resume interrupted export job'
)
@click.option(
    '--job-id',
    type=str,
    help='Specific job ID to resume'
)
@click.option(
    '--metadata-file',
    type=click.Path(exists=True, path_type=Path),
    help='Additional metadata JSON file'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be exported without doing it'
)
def export_command(
    input_path: Path,
    output_path: Path,
    format: str,
    batch_size: int,
    recursive: bool,
    filter: Optional[str],
    privacy: bool,
    copy_images: bool,
    split_ratio: Optional[str],
    max_workers: int,
    resume: bool,
    job_id: Optional[str],
    metadata_file: Optional[Path],
    verbose: bool,
    dry_run: bool
):
    """
    Export documents to various formats.
    
    Examples:
        # Export to COCO format
        synthdata export ./documents ./coco_dataset -f coco
        
        # Export to YOLO with train/val split
        synthdata export ./data ./yolo_data -f yolo --split-ratio 80:20
        
        # Export JSON files to CSV
        synthdata export ./json_docs ./csv_output -f csv --filter "*.json"
        
        # Resume interrupted export
        synthdata export ./input ./output -f coco --resume --job-id abc123
    """
    asyncio.run(_export_async(
        input_path=input_path,
        output_path=output_path,
        format=format,
        batch_size=batch_size,
        recursive=recursive,
        filter=filter,
        privacy=privacy,
        copy_images=copy_images,
        split_ratio=split_ratio,
        max_workers=max_workers,
        resume=resume,
        job_id=job_id,
        metadata_file=metadata_file,
        verbose=verbose,
        dry_run=dry_run
    ))


async def _export_async(
    input_path: Path,
    output_path: Path,
    format: str,
    batch_size: int,
    recursive: bool,
    filter: Optional[str],
    privacy: bool,
    copy_images: bool,
    split_ratio: Optional[str],
    max_workers: int,
    resume: bool,
    job_id: Optional[str],
    metadata_file: Optional[Path],
    verbose: bool,
    dry_run: bool
):
    """Async implementation of export command"""
    formatter = OutputFormatter(verbose=verbose)
    
    try:
        # Find documents to export
        documents = await _find_documents(
            input_path, recursive, filter, formatter
        )
        
        if not documents:
            formatter.warning("No documents found to export")
            return
        
        formatter.info(f"Found {len(documents)} documents to export")
        
        # Load additional metadata if provided
        extra_metadata = {}
        if metadata_file:
            with open(metadata_file) as f:
                extra_metadata = json.load(f)
            formatter.info(f"Loaded metadata from {metadata_file}")
        
        # Parse split ratio if provided
        splits = None
        if split_ratio:
            try:
                ratios = [int(x) for x in split_ratio.split(':')]
                if len(ratios) not in [2, 3] or sum(ratios) != 100:
                    raise ValueError("Invalid split ratio")
                splits = {
                    'train': ratios[0] / 100,
                    'val': ratios[1] / 100
                }
                if len(ratios) == 3:
                    splits['test'] = ratios[2] / 100
            except Exception:
                formatter.error("Invalid split ratio format. Use format like '70:20:10'")
                return
        
        if dry_run:
            formatter.warning("DRY RUN MODE - No actual export will be performed")
            formatter.info(f"Would export {len(documents)} documents to {format} format")
            formatter.info(f"Output path: {output_path}")
            if splits:
                formatter.info(f"Split ratios: {split_ratio}")
            return
        
        # Create batch exporter
        exporter = create_batch_exporter(
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        # Prepare export options
        export_options = {
            'privacy_protection': privacy,
            'copy_images': copy_images,
            'metadata': extra_metadata,
            'format': format
        }
        
        if splits:
            export_options['splits'] = splits
        
        # Handle resume functionality
        if resume and job_id:
            formatter.info(f"Resuming export job {job_id}")
            job_status = await exporter.get_job_status(job_id)
            if not job_status:
                formatter.error(f"Job {job_id} not found")
                return
            if job_status['status'] == 'completed':
                formatter.info("Job already completed")
                return
        
        # Create or resume export job
        if not (resume and job_id):
            # Get document IDs (in real implementation, this would extract from files)
            document_ids = [str(doc) for doc in documents]
            
            job_id = await exporter.create_export_job(
                document_ids=document_ids,
                format_type=format,
                output_path=str(output_path),
                options=export_options
            )
            formatter.info(f"Created export job: {job_id}")
        
        # Progress callback
        progress = ProgressTracker(total=len(documents), desc="Exporting documents")
        
        async def progress_callback(job_info: dict):
            if job_info['status'] == 'running':
                progress.update(
                    job_info['processed_documents'] - progress.n
                )
            elif job_info['status'] == 'completed':
                progress.close()
                formatter.success(f"\nExport completed successfully")
                formatter.info(f"Output: {job_info['output_path']}")
                formatter.info(f"Processed: {job_info['processed_documents']} documents")
                if job_info['failed_documents'] > 0:
                    formatter.warning(
                        f"Failed: {job_info['failed_documents']} documents"
                    )
            elif job_info['status'] == 'failed':
                progress.close()
                formatter.error(f"Export failed: {job_info['error_message']}")
        
        # Start export job
        await exporter.start_export_job(job_id, progress_callback)
        
        # Wait for completion
        while True:
            job_status = await exporter.get_job_status(job_id)
            if job_status['status'] in ['completed', 'failed', 'cancelled']:
                break
            await asyncio.sleep(1)
        
        if job_status['status'] == 'failed':
            raise click.ClickException(f"Export failed: {job_status['error_message']}")
        
    except KeyboardInterrupt:
        formatter.warning("\nExport interrupted by user")
        if 'exporter' in locals() and 'job_id' in locals():
            await exporter.cancel_job(job_id)
    except Exception as e:
        formatter.error(f"Export failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise


async def _find_documents(
    input_path: Path, 
    recursive: bool, 
    filter_pattern: Optional[str],
    formatter: OutputFormatter
) -> List[Path]:
    """Find documents to export"""
    documents = []
    
    if input_path.is_file():
        documents.append(input_path)
    elif input_path.is_dir():
        if recursive:
            if filter_pattern:
                documents.extend(input_path.rglob(filter_pattern))
            else:
                documents.extend([
                    f for f in input_path.rglob('*') if f.is_file()
                ])
        else:
            if filter_pattern:
                documents.extend(input_path.glob(filter_pattern))
            else:
                documents.extend([
                    f for f in input_path.glob('*') if f.is_file()
                ])
    
    # Filter to relevant file types
    supported_extensions = {'.json', '.pdf', '.docx', '.xml', '.txt', '.png', '.jpg'}
    documents = [
        d for d in documents 
        if d.suffix.lower() in supported_extensions
    ]
    
    return sorted(documents)


def create_export_command():
    """Factory function to create export command"""
    return export_command


__all__ = ['export_command', 'create_export_command']