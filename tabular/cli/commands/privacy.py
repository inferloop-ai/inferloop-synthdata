"""
CLI commands for privacy evaluation
"""

from pathlib import Path
from typing import Optional, List

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from sdk.privacy import (
    PrivacyEvaluator,
    DifferentialPrivacyValidator,
    KAnonymityValidator,
    LDiversityValidator,
    TClosenessValidator
)

app = typer.Typer(help="Privacy evaluation commands")
console = Console()


@app.command()
def evaluate(
    real_data: Path = typer.Argument(..., help="Path to real/original data"),
    synthetic_data: Path = typer.Argument(..., help="Path to synthetic data"),
    epsilon: float = typer.Option(1.0, help="Differential privacy epsilon"),
    delta: float = typer.Option(1e-5, help="Differential privacy delta"),
    k_threshold: int = typer.Option(5, help="k-anonymity threshold"),
    l_threshold: int = typer.Option(2, help="l-diversity threshold"),
    t_threshold: float = typer.Option(0.2, help="t-closeness threshold"),
    output_report: Optional[Path] = typer.Option(None, help="Save report to file")
):
    """Evaluate privacy of synthetic data"""
    
    # Load data
    console.print(f"[green]Loading data files...[/green]")
    
    try:
        real_df = pd.read_csv(real_data)
        synthetic_df = pd.read_csv(synthetic_data)
    except Exception as e:
        console.print(f"[red]Error loading data: {str(e)}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Real data: {len(real_df)} rows, {len(real_df.columns)} columns")
    console.print(f"Synthetic data: {len(synthetic_df)} rows, {len(synthetic_df.columns)} columns")
    
    # Create evaluator
    evaluator = PrivacyEvaluator(
        epsilon=epsilon,
        delta=delta,
        k_threshold=k_threshold,
        l_threshold=l_threshold,
        t_threshold=t_threshold
    )
    
    # Evaluate privacy
    console.print("\n[cyan]Evaluating privacy metrics...[/cyan]")
    
    with console.status("Running privacy evaluation..."):
        metrics = evaluator.evaluate_privacy(real_df, synthetic_df)
        report = evaluator.generate_privacy_report(real_df, synthetic_df)
    
    # Display results
    console.print("\n[bold]Privacy Evaluation Results[/bold]\n")
    
    # Create summary table
    summary_table = Table(title="Privacy Metrics Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")
    summary_table.add_column("Threshold", style="blue")
    summary_table.add_column("Status", style="green")
    
    # Differential Privacy
    dp_status = "✓ Pass" if metrics.epsilon <= epsilon else "✗ Fail"
    summary_table.add_row(
        "Differential Privacy (ε)",
        f"{metrics.epsilon:.3f}",
        f"{epsilon:.3f}",
        dp_status
    )
    
    # k-Anonymity
    k_status = "✓ Pass" if metrics.k_anonymity >= k_threshold else "✗ Fail"
    summary_table.add_row(
        "k-Anonymity",
        str(metrics.k_anonymity),
        str(k_threshold),
        k_status
    )
    
    # l-Diversity
    l_status = "✓ Pass" if metrics.l_diversity >= l_threshold else "✗ Fail"
    summary_table.add_row(
        "l-Diversity",
        f"{metrics.l_diversity:.2f}",
        str(l_threshold),
        l_status
    )
    
    # t-Closeness
    t_status = "✓ Pass" if metrics.t_closeness <= t_threshold else "✗ Fail"
    summary_table.add_row(
        "t-Closeness",
        f"{metrics.t_closeness:.3f}",
        f"{t_threshold:.3f}",
        t_status
    )
    
    console.print(summary_table)
    
    # Risk assessment
    risk_table = Table(title="Privacy Risk Assessment", box=box.SIMPLE)
    risk_table.add_column("Risk Type", style="cyan")
    risk_table.add_column("Level", style="yellow")
    
    risk_table.add_row(
        "Membership Disclosure",
        f"{metrics.membership_disclosure_risk:.1%}"
    )
    risk_table.add_row(
        "Attribute Disclosure",
        f"{metrics.attribute_disclosure_risk:.1%}"
    )
    
    console.print("\n")
    console.print(risk_table)
    
    # Overall score
    score_color = "green" if metrics.privacy_score >= 0.8 else "yellow" if metrics.privacy_score >= 0.6 else "red"
    console.print(f"\n[bold]Overall Privacy Score: [{score_color}]{metrics.privacy_score:.2f}/1.00[/{score_color}][/bold]")
    
    # Detailed report
    if output_report:
        with open(output_report, 'w') as f:
            f.write(report)
        console.print(f"\n[green]Detailed report saved to: {output_report}[/green]")
    else:
        console.print("\n[dim]Use --output-report to save detailed report[/dim]")


@app.command()
def check_k_anonymity(
    data_file: Path = typer.Argument(..., help="Data file to check"),
    k_threshold: int = typer.Option(5, help="k-anonymity threshold"),
    quasi_identifiers: Optional[str] = typer.Option(None, help="Comma-separated quasi-identifiers"),
    show_risky: bool = typer.Option(False, help="Show risky groups")
):
    """Check k-anonymity of a dataset"""
    
    # Load data
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        console.print(f"[red]Error loading data: {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Parse quasi-identifiers
    qi_list = None
    if quasi_identifiers:
        qi_list = [col.strip() for col in quasi_identifiers.split(',')]
    
    # Create validator
    validator = KAnonymityValidator(quasi_identifiers=qi_list)
    
    # Check k-anonymity
    results = validator.check_k_anonymity(df, k_threshold)
    
    # Display results
    console.print(f"\n[bold]k-Anonymity Analysis[/bold]")
    console.print(f"Dataset: {data_file.name}")
    console.print(f"Rows: {len(df)}")
    
    if qi_list:
        console.print(f"Quasi-identifiers: {', '.join(qi_list)}")
    else:
        console.print("Quasi-identifiers: All columns")
    
    console.print(f"\nk-value: [yellow]{results['k_value']}[/yellow]")
    console.print(f"Threshold: {k_threshold}")
    
    if results['satisfies_k_anonymity']:
        console.print(f"Status: [green]✓ Satisfies {k_threshold}-anonymity[/green]")
    else:
        console.print(f"Status: [red]✗ Violates {k_threshold}-anonymity[/red]")
    
    # Show risky groups if requested
    if show_risky and not results['satisfies_k_anonymity']:
        risky_groups = validator.get_risky_groups(df, k_threshold)
        
        if not risky_groups.empty:
            console.print(f"\n[yellow]Found {len(risky_groups)} risky groups:[/yellow]")
            
            risky_table = Table(box=box.SIMPLE)
            for col in risky_groups.columns:
                risky_table.add_column(col)
            
            for _, row in risky_groups.head(10).iterrows():
                risky_table.add_row(*[str(val) for val in row])
            
            if len(risky_groups) > 10:
                risky_table.add_row(*["..." for _ in risky_groups.columns])
            
            console.print(risky_table)


@app.command()
def check_l_diversity(
    data_file: Path = typer.Argument(..., help="Data file to check"),
    l_threshold: int = typer.Option(2, help="l-diversity threshold"),
    quasi_identifiers: Optional[str] = typer.Option(None, help="Comma-separated quasi-identifiers"),
    sensitive_attributes: Optional[str] = typer.Option(None, help="Comma-separated sensitive attributes"),
    entropy: bool = typer.Option(False, help="Calculate entropy l-diversity")
):
    """Check l-diversity of a dataset"""
    
    # Load data
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        console.print(f"[red]Error loading data: {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Parse columns
    qi_list = None
    if quasi_identifiers:
        qi_list = [col.strip() for col in quasi_identifiers.split(',')]
    
    sa_list = None
    if sensitive_attributes:
        sa_list = [col.strip() for col in sensitive_attributes.split(',')]
    
    # Create validator
    validator = LDiversityValidator(
        quasi_identifiers=qi_list,
        sensitive_attributes=sa_list
    )
    
    # Check l-diversity
    results = validator.check_l_diversity(df, l_threshold)
    
    # Display results
    console.print(f"\n[bold]l-Diversity Analysis[/bold]")
    console.print(f"Dataset: {data_file.name}")
    
    if qi_list:
        console.print(f"Quasi-identifiers: {', '.join(qi_list)}")
    if sa_list:
        console.print(f"Sensitive attributes: {', '.join(sa_list)}")
    
    console.print(f"\nl-value: [yellow]{results['l_value']:.2f}[/yellow]")
    console.print(f"Threshold: {l_threshold}")
    
    if results['satisfies_l_diversity']:
        console.print(f"Status: [green]✓ Satisfies {l_threshold}-diversity[/green]")
    else:
        console.print(f"Status: [red]✗ Violates {l_threshold}-diversity[/red]")
    
    # Calculate entropy l-diversity if requested
    if entropy:
        entropy_l = validator.compute_entropy_l_diversity(df)
        console.print(f"\nEntropy l-diversity: [yellow]{entropy_l:.2f}[/yellow]")


@app.command()
def differential_privacy(
    real_data: Path = typer.Argument(..., help="Path to real data"),
    synthetic_data: Path = typer.Argument(..., help="Path to synthetic data"),
    epsilon: float = typer.Option(1.0, help="Target epsilon value"),
    delta: float = typer.Option(1e-5, help="Target delta value"),
    num_trials: int = typer.Option(100, help="Number of trials for estimation")
):
    """Check differential privacy guarantees"""
    
    # Load data
    try:
        real_df = pd.read_csv(real_data)
        synthetic_df = pd.read_csv(synthetic_data)
    except Exception as e:
        console.print(f"[red]Error loading data: {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Create validator
    dp_validator = DifferentialPrivacyValidator(epsilon=epsilon, delta=delta)
    
    console.print(f"\n[cyan]Estimating differential privacy parameters...[/cyan]")
    
    with console.status("Running DP analysis..."):
        # Check differential privacy
        dp_results = dp_validator.check_differential_privacy(real_df, synthetic_df)
        
        # Check Laplace mechanism
        laplace_results = dp_validator.laplace_mechanism_check(real_df, synthetic_df)
    
    # Display results
    console.print(f"\n[bold]Differential Privacy Analysis[/bold]\n")
    
    dp_table = Table(box=box.ROUNDED)
    dp_table.add_column("Parameter", style="cyan")
    dp_table.add_column("Value", style="yellow")
    
    dp_table.add_row("Target ε", f"{epsilon:.3f}")
    dp_table.add_row("Estimated ε", f"{dp_results['estimated_epsilon']:.3f}")
    dp_table.add_row("δ (delta)", f"{delta}")
    dp_table.add_row("Privacy Loss", f"{dp_results['privacy_loss']:.3f}")
    
    console.print(dp_table)
    
    if dp_results['satisfies_dp']:
        console.print(f"\n[green]✓ Satisfies ({epsilon}, {delta})-differential privacy[/green]")
    else:
        console.print(f"\n[red]✗ Violates ({epsilon}, {delta})-differential privacy[/red]")
    
    # Laplace mechanism analysis
    if laplace_results:
        console.print("\n[bold]Laplace Mechanism Analysis[/bold]")
        
        laplace_table = Table(box=box.SIMPLE)
        laplace_table.add_column("Column", style="cyan")
        laplace_table.add_column("Noise Scale", style="yellow")
        laplace_table.add_column("Follows Laplace", style="green")
        
        for col, info in list(laplace_results.items())[:10]:
            follows = "✓" if info['follows_laplace'] else "✗"
            laplace_table.add_row(
                col,
                f"{info['expected_scale']:.6f}",
                follows
            )
        
        if len(laplace_results) > 10:
            laplace_table.add_row("...", "...", "...")
        
        console.print(laplace_table)


@app.command()
def attack_risk(
    real_data: Path = typer.Argument(..., help="Path to real data"),
    synthetic_data: Path = typer.Argument(..., help="Path to synthetic data"),
    n_neighbors: int = typer.Option(5, help="Number of neighbors for membership inference"),
    sensitive_columns: Optional[str] = typer.Option(None, help="Comma-separated sensitive columns")
):
    """Assess privacy attack risks"""
    
    # Load data
    try:
        real_df = pd.read_csv(real_data)
        synthetic_df = pd.read_csv(synthetic_data)
    except Exception as e:
        console.print(f"[red]Error loading data: {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Parse sensitive columns
    sensitive_cols = None
    if sensitive_columns:
        sensitive_cols = [col.strip() for col in sensitive_columns.split(',')]
    
    console.print(f"\n[cyan]Assessing privacy attack risks...[/cyan]")
    
    # Membership inference attack
    from sdk.privacy import MembershipInferenceAttack
    membership_attack = MembershipInferenceAttack(n_neighbors=n_neighbors)
    
    with console.status("Running membership inference attack..."):
        membership_risk = membership_attack.compute_membership_risk(real_df, synthetic_df)
    
    # Attribute disclosure risk
    from sdk.privacy import AttributeDisclosureRisk
    attribute_risk_assessor = AttributeDisclosureRisk()
    
    with console.status("Assessing attribute disclosure risk..."):
        attribute_risk = attribute_risk_assessor.compute_attribute_risk(
            real_df, synthetic_df, sensitive_cols
        )
    
    # Display results
    console.print(f"\n[bold]Privacy Attack Risk Assessment[/bold]\n")
    
    # Risk panel
    risk_level = "Low" if membership_risk < 0.2 else "Medium" if membership_risk < 0.5 else "High"
    risk_color = "green" if risk_level == "Low" else "yellow" if risk_level == "Medium" else "red"
    
    panel_content = f"""
[bold]Membership Inference Risk:[/bold] [{risk_color}]{membership_risk:.1%}[/{risk_color}]
Risk Level: [{risk_color}]{risk_level}[/{risk_color}]

[bold]Attribute Disclosure Risk:[/bold] {attribute_risk:.1%}
Sensitive Columns: {len(sensitive_cols) if sensitive_cols else 'All non-numeric'}
"""
    
    console.print(Panel(panel_content, title="Attack Risk Summary", box=box.ROUNDED))
    
    # Recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    
    if membership_risk > 0.5:
        console.print("• [red]High membership inference risk detected[/red]")
        console.print("  - Consider adding more noise to the generation process")
        console.print("  - Increase the differential privacy budget (lower epsilon)")
    
    if attribute_risk > 0.3:
        console.print("• [yellow]Moderate attribute disclosure risk[/yellow]")
        console.print("  - Review handling of rare values in sensitive attributes")
        console.print("  - Consider suppression or generalization techniques")
    
    if membership_risk < 0.2 and attribute_risk < 0.2:
        console.print("• [green]Privacy risks are within acceptable bounds[/green]")


if __name__ == "__main__":
    app()