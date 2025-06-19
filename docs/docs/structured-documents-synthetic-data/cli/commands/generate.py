#!/usr/bin/env python3
"""
Generate command for creating synthetic documents.

Provides CLI interface for generating synthetic documents with various
configurations, domains, and privacy settings.
"""

import asyncio
import click
import json
from pathlib import Path
from typing import Optional

from ...core import get_logger, get_config
from ...generation import create_layout_engine
from ...privacy import create_privacy_engine
from ..utils import OutputFormatter, ProgressTracker


logger = get_logger(__name__)


@click.command(name='generate')
@click.option(
    '--count', '-c',
    type=int,
    default=10,
    help='Number of documents to generate'
)
@click.option(
    '--domain', '-d',
    type=click.Choice(['general', 'finance', 'healthcare', 'legal', 'government']),
    default='general',
    help='Document domain/type'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['pdf', 'docx', 'html', 'latex']),
    default='pdf',
    help='Output document format'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    default='./output',
    help='Output directory path'
)
@click.option(
    '--privacy-level',
    type=click.Choice(['none', 'low', 'medium', 'high']),
    default='medium',
    help='Privacy protection level'
)
@click.option(
    '--config-file',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file path'
)
@click.option(
    '--template',
    type=str,
    help='Template name or path'
)
@click.option(
    '--batch-size',
    type=int,
    default=10,
    help='Batch size for generation'
)
@click.option(
    '--parallel/--sequential',
    default=True,
    help='Enable parallel generation'
)
@click.option(
    '--seed',
    type=int,
    help='Random seed for reproducibility'
)
@click.option(
    '--metadata',
    type=str,
    help='Additional metadata as JSON string'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Perform dry run without actual generation'
)
def generate_command(
    count: int,
    domain: str,
    format: str,
    output: Path,
    privacy_level: str,
    config_file: Optional[Path],
    template: Optional[str],
    batch_size: int,
    parallel: bool,
    seed: Optional[int],
    metadata: Optional[str],
    verbose: bool,
    dry_run: bool
):
    """
    Generate synthetic structured documents.
    
    Examples:
        # Generate 100 financial PDFs
        synthdata generate -c 100 -d finance -f pdf
        
        # Generate healthcare documents with high privacy
        synthdata generate -d healthcare --privacy-level high
        
        # Use custom template and config
        synthdata generate --template invoice --config-file config.yaml
    """
    # Run async function
    asyncio.run(_generate_async(
        count=count,
        domain=domain,
        format=format,
        output=output,
        privacy_level=privacy_level,
        config_file=config_file,
        template=template,
        batch_size=batch_size,
        parallel=parallel,
        seed=seed,
        metadata=metadata,
        verbose=verbose,
        dry_run=dry_run
    ))


async def _generate_async(
    count: int,
    domain: str,
    format: str,
    output: Path,
    privacy_level: str,
    config_file: Optional[Path],
    template: Optional[str],
    batch_size: int,
    parallel: bool,
    seed: Optional[int],
    metadata: Optional[str],
    verbose: bool,
    dry_run: bool
):
    """Async implementation of generate command"""
    formatter = OutputFormatter(verbose=verbose)
    progress = ProgressTracker(total=count, desc="Generating documents")
    
    try:
        # Load configuration
        config = get_config()
        if config_file:
            formatter.info(f"Loading configuration from {config_file}")
            with open(config_file) as f:
                custom_config = json.load(f) if config_file.suffix == '.json' else {}
                config.update(custom_config)
        
        # Parse metadata
        extra_metadata = {}
        if metadata:
            try:
                extra_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                formatter.error("Invalid metadata JSON format")
                return
        
        # Validate output directory
        output.mkdir(parents=True, exist_ok=True)
        formatter.info(f"Output directory: {output}")
        
        if dry_run:
            formatter.warning("DRY RUN MODE - No documents will be generated")
            formatter.info(f"Would generate {count} {domain} documents in {format} format")
            formatter.info(f"Privacy level: {privacy_level}")
            formatter.info(f"Output path: {output}")
            return
        
        # Initialize engines
        layout_engine = create_layout_engine()
        privacy_engine = create_privacy_engine() if privacy_level != 'none' else None
        
        # Set seed if provided
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            formatter.info(f"Random seed set to {seed}")
        
        # Generate documents
        generated_docs = []
        batches = [list(range(i, min(i + batch_size, count))) 
                  for i in range(0, count, batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            formatter.info(f"Processing batch {batch_idx + 1}/{len(batches)}")
            
            if parallel:
                # Parallel generation
                tasks = [
                    _generate_single_document(
                        idx=idx,
                        domain=domain,
                        format=format,
                        template=template,
                        layout_engine=layout_engine,
                        privacy_engine=privacy_engine,
                        privacy_level=privacy_level,
                        metadata=extra_metadata
                    )
                    for idx in batch
                ]
                batch_docs = await asyncio.gather(*tasks)
            else:
                # Sequential generation
                batch_docs = []
                for idx in batch:
                    doc = await _generate_single_document(
                        idx=idx,
                        domain=domain,
                        format=format,
                        template=template,
                        layout_engine=layout_engine,
                        privacy_engine=privacy_engine,
                        privacy_level=privacy_level,
                        metadata=extra_metadata
                    )
                    batch_docs.append(doc)
                    progress.update(1)
            
            # Save batch documents
            for doc in batch_docs:
                if doc:
                    doc_path = output / f"{domain}_doc_{doc['id']}.{format}"
                    await _save_document(doc, doc_path, format)
                    generated_docs.append(doc_path)
            
            if parallel:
                progress.update(len(batch_docs))
        
        progress.close()
        
        # Summary
        formatter.success(f"\nGenerated {len(generated_docs)} documents successfully")
        formatter.info(f"Output directory: {output}")
        
        # Generate manifest
        manifest = {
            'generation_info': {
                'total_documents': len(generated_docs),
                'domain': domain,
                'format': format,
                'privacy_level': privacy_level,
                'template': template,
                'seed': seed,
                'metadata': extra_metadata
            },
            'documents': [str(p) for p in generated_docs]
        }
        
        manifest_path = output / 'generation_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        formatter.info(f"Generation manifest saved to {manifest_path}")
        
    except KeyboardInterrupt:
        formatter.warning("\nGeneration interrupted by user")
        progress.close()
    except Exception as e:
        formatter.error(f"Generation failed: {str(e)}")
        progress.close()
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.ClickException(str(e))


async def _generate_single_document(
    idx: int,
    domain: str,
    format: str,
    template: Optional[str],
    layout_engine,
    privacy_engine,
    privacy_level: str,
    metadata: dict
) -> Optional[dict]:
    """Generate a single document"""
    try:
        # Generate document with layout engine
        doc = await layout_engine.generate_document(
            domain=domain,
            format=format,
            template=template,
            metadata=metadata
        )
        
        # Apply privacy protection
        if privacy_engine and privacy_level != 'none':
            doc = await privacy_engine.apply_protection(
                doc, 
                level=privacy_level
            )
        
        # Add generation metadata
        doc['id'] = f"{idx:06d}"
        doc['generation_metadata'] = {
            'domain': domain,
            'format': format,
            'privacy_level': privacy_level,
            'template': template
        }
        
        return doc
        
    except Exception as e:
        logger.error(f"Failed to generate document {idx}: {e}")
        return None


async def _save_document(doc: dict, path: Path, format: str):
    """Save document to file"""
    try:
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(doc, f, indent=2)
        else:
            # For other formats, save the content
            # This would integrate with actual rendering engines
            content = doc.get('content', '')
            with open(path, 'wb' if format in ['pdf', 'docx'] else 'w') as f:
                if isinstance(content, bytes):
                    f.write(content)
                else:
                    f.write(str(content))
                    
    except Exception as e:
        logger.error(f"Failed to save document to {path}: {e}")
        raise


def create_generate_command():
    """Factory function to create generate command"""
    return generate_command


__all__ = ['generate_command', 'create_generate_command']