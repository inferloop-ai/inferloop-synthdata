"""
CLI commands for cache management
"""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
# Use a simple size formatter if humanize is not available
try:
    from humanize import naturalsize
except ImportError:
    def naturalsize(value):
        """Simple bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if value < 1024.0:
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} TB"

from sdk.cache import get_cache, set_cache, SyntheticDataCache, FileSystemCache, MemoryCache

app = typer.Typer(help="Cache management commands")
console = Console()


@app.command()
def status():
    """Show cache status and statistics"""
    cache = get_cache()
    stats = cache.get_stats()
    
    # Create status panel
    status_text = f"""[bold]Cache Status[/bold]
Backend: {type(cache.backend).__name__}
Default TTL: {cache.default_ttl} seconds
Cache Data: {'✓' if cache.cache_generated_data else '✗'}
Cache Models: {'✓' if cache.cache_models else '✗'}
    """
    
    console.print(Panel(status_text, title="Configuration", box=box.ROUNDED))
    
    # Show statistics if available
    if stats:
        stats_table = Table(title="Cache Statistics", box=box.SIMPLE)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("Total Entries", f"{stats.get('entries', 0):,}")
        stats_table.add_row("Total Size", naturalsize(stats.get('total_size_mb', 0) * 1024 * 1024))
        stats_table.add_row("Max Size", naturalsize(stats.get('max_size_mb', 0) * 1024 * 1024))
        stats_table.add_row("Utilization", f"{stats.get('utilization', 0):.1f}%")
        stats_table.add_row("Total Hits", f"{stats.get('total_hits', 0):,}")
        stats_table.add_row("Avg Hits/Entry", f"{stats.get('avg_hits_per_entry', 0):.2f}")
        
        console.print(stats_table)
    
    # Show cache directory for filesystem cache
    if hasattr(cache.backend, 'cache_dir'):
        console.print(f"\n[dim]Cache Directory: {cache.backend.cache_dir}[/dim]")


@app.command()
def clear(
    pattern: Optional[str] = typer.Option(None, help="Clear only entries matching pattern"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """Clear cache entries"""
    cache = get_cache()
    
    if not force:
        if pattern:
            confirm = typer.confirm(f"Clear cache entries matching '{pattern}'?")
        else:
            confirm = typer.confirm("Clear ALL cache entries?")
        
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)
    
    with console.status("Clearing cache..."):
        if pattern:
            # Clear matching entries
            cleared = 0
            if hasattr(cache.backend, 'index'):
                keys_to_delete = []
                for key_hash, entry in cache.backend.index.items():
                    if pattern in entry.key:
                        keys_to_delete.append(entry.key)
                
                for key in keys_to_delete:
                    if cache.backend.delete(key):
                        cleared += 1
            
            console.print(f"[green]Cleared {cleared} entries matching '{pattern}'[/green]")
        else:
            # Clear all
            cleared = cache.clear_all()
            console.print(f"[green]Cleared {cleared} cache entries[/green]")


@app.command()
def list(
    limit: int = typer.Option(20, help="Number of entries to show"),
    sort_by: str = typer.Option("created", help="Sort by: created, size, hits")
):
    """List cache entries"""
    cache = get_cache()
    
    if not hasattr(cache.backend, 'index'):
        console.print("[yellow]Cache backend doesn't support listing entries[/yellow]")
        return
    
    entries = list(cache.backend.index.values())
    
    if not entries:
        console.print("[yellow]No cache entries found[/yellow]")
        return
    
    # Sort entries
    if sort_by == "created":
        entries.sort(key=lambda x: x.created_at, reverse=True)
    elif sort_by == "size":
        entries.sort(key=lambda x: x.size_bytes, reverse=True)
    elif sort_by == "hits":
        entries.sort(key=lambda x: x.hits, reverse=True)
    
    # Create table
    table = Table(title="Cache Entries", box=box.ROUNDED)
    table.add_column("Key", style="cyan", max_width=40)
    table.add_column("Created", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Hits", style="magenta")
    table.add_column("Expires", style="red")
    
    for entry in entries[:limit]:
        # Format key
        key = entry.key
        if len(key) > 40:
            key = key[:37] + "..."
        
        # Format dates
        created = entry.created_at.strftime("%Y-%m-%d %H:%M")
        expires = entry.expires_at.strftime("%Y-%m-%d %H:%M") if entry.expires_at else "Never"
        
        table.add_row(
            key,
            created,
            naturalsize(entry.size_bytes),
            str(entry.hits),
            expires
        )
    
    console.print(table)
    
    if len(entries) > limit:
        console.print(f"\n[dim]Showing {limit} of {len(entries)} entries[/dim]")


@app.command()
def configure(
    backend: str = typer.Option("filesystem", help="Cache backend: filesystem or memory"),
    cache_dir: Optional[Path] = typer.Option(None, help="Cache directory (filesystem only)"),
    max_size_mb: int = typer.Option(1024, help="Maximum cache size in MB"),
    default_ttl: int = typer.Option(3600, help="Default TTL in seconds"),
    enable_data_cache: bool = typer.Option(True, help="Cache generated data"),
    enable_model_cache: bool = typer.Option(True, help="Cache trained models")
):
    """Configure cache settings"""
    console.print(f"[green]Configuring cache...[/green]")
    
    # Create backend
    if backend == "filesystem":
        cache_backend = FileSystemCache(
            cache_dir=str(cache_dir) if cache_dir else None,
            max_size_mb=max_size_mb
        )
        console.print(f"✓ Filesystem backend configured")
        if cache_dir:
            console.print(f"  Directory: {cache_dir}")
    
    elif backend == "memory":
        # Estimate max entries
        max_entries = (max_size_mb * 1024) // 100  # Assume 100KB average
        cache_backend = MemoryCache(max_entries=max_entries)
        console.print(f"✓ Memory backend configured")
        console.print(f"  Max entries: {max_entries:,}")
    
    else:
        console.print(f"[red]Unknown backend: {backend}[/red]")
        raise typer.Exit(1)
    
    # Create cache instance
    new_cache = SyntheticDataCache(
        backend=cache_backend,
        default_ttl=default_ttl,
        cache_generated_data=enable_data_cache,
        cache_models=enable_model_cache
    )
    
    # Set as global cache
    set_cache(new_cache)
    
    console.print(f"✓ Default TTL: {default_ttl} seconds")
    console.print(f"✓ Cache data: {'Enabled' if enable_data_cache else 'Disabled'}")
    console.print(f"✓ Cache models: {'Enabled' if enable_model_cache else 'Disabled'}")
    console.print("\n[green]Cache configured successfully![/green]")


@app.command()
def inspect(
    key: str = typer.Argument(..., help="Cache key to inspect")
):
    """Inspect a specific cache entry"""
    cache = get_cache()
    
    if not hasattr(cache.backend, 'index'):
        console.print("[yellow]Cache backend doesn't support inspection[/yellow]")
        return
    
    # Find entry
    entry = None
    for key_hash, e in cache.backend.index.items():
        if key in e.key or e.key == key:
            entry = e
            break
    
    if not entry:
        console.print(f"[red]Cache entry not found: {key}[/red]")
        raise typer.Exit(1)
    
    # Display entry details
    details = f"""[bold]Cache Entry Details[/bold]

Key: {entry.key}
Created: {entry.created_at}
Expires: {entry.expires_at if entry.expires_at else 'Never'}
Size: {naturalsize(entry.size_bytes)}
Hits: {entry.hits}
File: {entry.file_path}

[dim]Expired: {'Yes' if entry.is_expired() else 'No'}[/dim]
    """
    
    console.print(Panel(details, title="Entry Information", box=box.ROUNDED))
    
    # Check if file exists
    if Path(entry.file_path).exists():
        console.print("[green]✓ Cache file exists[/green]")
    else:
        console.print("[red]✗ Cache file missing![/red]")


@app.command()
def delete(
    key: str = typer.Argument(..., help="Cache key to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """Delete a specific cache entry"""
    cache = get_cache()
    
    if not force:
        confirm = typer.confirm(f"Delete cache entry '{key}'?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)
    
    # Try to delete
    deleted = cache.backend.delete(key)
    
    if not deleted:
        # Try with prefixes
        for prefix in ['data:', 'model:']:
            if cache.backend.delete(f"{prefix}{key}"):
                deleted = True
                break
    
    if deleted:
        console.print(f"[green]✓ Deleted cache entry: {key}[/green]")
    else:
        console.print(f"[red]Cache entry not found: {key}[/red]")
        raise typer.Exit(1)


@app.command()
def benchmark():
    """Run cache performance benchmark"""
    import time
    import pandas as pd
    import numpy as np
    
    console.print("[green]Running cache benchmark...[/green]")
    
    cache = get_cache()
    
    # Create test data
    test_sizes = [100, 1000, 10000]
    results = []
    
    for size in test_sizes:
        # Generate test dataframe
        df = pd.DataFrame({
            'col1': np.random.randn(size),
            'col2': np.random.choice(['A', 'B', 'C'], size),
            'col3': np.random.randint(0, 100, size)
        })
        
        # Test write
        key = f"benchmark_test_{size}"
        start = time.time()
        cache.backend.set(key, df, ttl=60)
        write_time = time.time() - start
        
        # Test read
        start = time.time()
        cached_df = cache.backend.get(key)
        read_time = time.time() - start
        
        # Clean up
        cache.backend.delete(key)
        
        results.append({
            'size': size,
            'write_ms': write_time * 1000,
            'read_ms': read_time * 1000
        })
    
    # Display results
    table = Table(title="Cache Performance Benchmark", box=box.ROUNDED)
    table.add_column("Data Size", style="cyan")
    table.add_column("Write Time", style="yellow")
    table.add_column("Read Time", style="green")
    
    for result in results:
        table.add_row(
            f"{result['size']:,} rows",
            f"{result['write_ms']:.2f} ms",
            f"{result['read_ms']:.2f} ms"
        )
    
    console.print(table)


if __name__ == "__main__":
    app()