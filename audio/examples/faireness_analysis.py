
# examples/fairness_analysis.py
"""
Fairness analysis example for synthetic audio
"""

import torch
import torchaudio
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from audio_synth.sdk.client import AudioSynthSDK

def fairness_analysis_example():
    """Demonstrate fairness analysis across demographics"""
    
    print("=== Fairness Analysis Example ===\n")
    
    sdk = AudioSynthSDK()
    
    # Define demographic groups to test
    demographics = [
        {"gender": "male", "age_group": "adult", "accent": "american"},
        {"gender": "female", "age_group": "adult", "accent": "american"},
        {"gender": "male", "age_group": "elderly", "accent": "british"},
        {"gender": "female", "age_group": "elderly", "accent": "british"},
        {"gender": "other", "age_group": "adult", "accent": "australian"},
        {"gender": "male", "age_group": "child", "accent": "canadian"},
        {"gender": "female", "age_group": "child", "accent": "indian"}
    ]
    
    output_dir = Path("./output/fairness_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_audios = []
    all_metadata = []
    fairness_results = {}
    
    print("Generating samples for different demographic groups...")
    
    for i, demo in enumerate(demographics):
        demo_key = f"{demo['gender']}_{demo['age_group']}_{demo['accent']}"
        print(f"  {i+1}. {demo_key}")
        
        # Generate samples for this demographic
        result = sdk.generate_and_validate(
            method="diffusion",
            prompt="Hello, how are you today? This is a test of speech synthesis.",
            num_samples=3,
            validators=["quality", "fairness"],
            conditions={
                "demographics": demo
            }
        )
        
        # Save samples
        for j, audio in enumerate(result["audios"]):
            filename = f"{demo_key}_sample_{j+1}.wav"
            torchaudio.save(
                str(output_dir / filename),
                audio.unsqueeze(0),
                22050
            )
            
            all_audios.append(audio)
            
            metadata = {
                "demographic_group": demo_key,
                "demographics": demo,
                "sample_index": j,
                "filename": filename
            }
            all_metadata.append(metadata)
        
        # Store fairness results
        fairness_metrics = result["validation"]["fairness"]
        quality_metrics = result["validation"]["quality"]
        
        fairness_results[demo_key] = {
            "demographics": demo,
            "fairness_metrics": fairness_metrics,
            "quality_metrics": quality_metrics,
            "avg_fairness": sum(m.get("bias_score", 0) for m in fairness_metrics) / len(fairness_metrics),
            "avg_quality": sum(m.get("realism_score", 0) for m in quality_metrics) / len(quality_metrics)
        }
    
    print("\nRunning comprehensive fairness validation...")
    
    # Run batch fairness analysis
    batch_validation = sdk.validate(
        audios=all_audios,
        metadata=all_metadata,
        validators=["fairness"]
    )
    
    # Analyze group fairness
    group_analysis = analyze_group_fairness(fairness_results)
    
    # Save results
    analysis_results = {
        "individual_results": fairness_results,
        "group_analysis": group_analysis,
        "batch_validation": batch_validation,
        "demographics_tested": demographics
    }
    
    with open(output_dir / "fairness_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Generate fairness report
    generate_fairness_report(analysis_results, output_dir)
    
    print(f"\nFairness analysis completed. Results saved to: {output_dir}")
    
    # Print summary
    print_fairness_summary(group_analysis)

def analyze_group_fairness(fairness_results):
    """Analyze fairness across demographic groups"""
    
    # Group by attributes
    gender_groups = {}
    age_groups = {}
    accent_groups = {}
    
    for demo_key, results in fairness_results.items():
        demo = results["demographics"]
        
        # Group by gender
        gender = demo["gender"]
        if gender not in gender_groups:
            gender_groups[gender] = []
        gender_groups[gender].append(results["avg_quality"])
        
        # Group by age
        age = demo["age_group"]
        if age not in age_groups:
            age_groups[age] = []
        age_groups[age].append(results["avg_quality"])
        
        # Group by accent
        accent = demo["accent"]
        if accent not in accent_groups:
            accent_groups[accent] = []
        accent_groups[accent].append(results["avg_quality"])
    
    # Calculate fairness metrics
    def calculate_parity(groups):
        group_means = {k: sum(v)/len(v) for k, v in groups.items()}
        overall_mean = sum(group_means.values()) / len(group_means)
        variance = sum((mean - overall_mean)**2 for mean in group_means.values()) / len(group_means)
        parity_score = 1.0 / (1.0 + variance)  # Higher is better
        return {
            "group_means": group_means,
            "overall_mean": overall_mean,
            "variance": variance,
            "parity_score": parity_score
        }
    
    return {
        "gender_parity": calculate_parity(gender_groups),
        "age_parity": calculate_parity(age_groups),
        "accent_parity": calculate_parity(accent_groups)
    }

def generate_fairness_report(analysis_results, output_dir):
    """Generate HTML fairness report"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Audio Synthesis Fairness Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .good {{ color: green; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        .poor {{ color: red; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Audio Synthesis Fairness Report</h1>
        <p>Analysis of fairness across demographic groups</p>
    </div>
    
    <div class="section">
        <h2>Demographic Parity Analysis</h2>
        {_generate_parity_table(analysis_results["group_analysis"])}
    </div>
    
    <div class="section">
        <h2>Individual Group Results</h2>
        {_generate_individual_results_table(analysis_results["individual_results"])}
    </div>
</body>
</html>
"""
    
    with open(output_dir / "fairness_report.html", 'w') as f:
        f.write(html_content)

def _generate_parity_table(group_analysis):
    """Generate parity analysis table"""
    
    html = "<table><tr><th>Attribute</th><th>Parity Score</th><th>Variance</th><th>Status</th></tr>"
    
    for attr, analysis in group_analysis.items():
        parity_score = analysis["parity_score"]
        variance = analysis["variance"]
        
        if parity_score > 0.8:
            status = '<span class="good">Good</span>'
        elif parity_score > 0.6:
            status = '<span class="warning">Fair</span>'
        else:
            status = '<span class="poor">Poor</span>'
        
        html += f"<tr><td>{attr}</td><td>{parity_score:.3f}</td><td>{variance:.3f}</td><td>{status}</td></tr>"
    
    html += "</table>"
    return html

def _generate_individual_results_table(individual_results):
    """Generate individual results table"""
    
    html = "<table><tr><th>Group</th><th>Gender</th><th>Age</th><th>Accent</th><th>Avg Quality</th><th>Avg Fairness</th></tr>"
    
    for group, results in individual_results.items():
        demo = results["demographics"]
        avg_quality = results["avg_quality"]
        avg_fairness = results["avg_fairness"]
        
        html += f"""<tr>
            <td>{group}</td>
            <td>{demo['gender']}</td>
            <td>{demo['age_group']}</td>
            <td>{demo['accent']}</td>
            <td>{avg_quality:.3f}</td>
            <td>{avg_fairness:.3f}</td>
        </tr>"""
    
    html += "</table>"
    return html

def print_fairness_summary(group_analysis):
    """Print fairness summary to console"""
    
    print("\n" + "="*60)
    print("FAIRNESS ANALYSIS SUMMARY")
    print("="*60)
    
    for attr, analysis in group_analysis.items():
        parity_score = analysis["parity_score"]
        print(f"\n{attr.upper()}:")
        print(f"  Parity Score: {parity_score:.3f}")
        print(f"  Group Means:")
        
        for group, mean in analysis["group_means"].items():
            print(f"    {group}: {mean:.3f}")
        
        # Determine status
        if parity_score > 0.8:
            status = "GOOD - High fairness"
        elif parity_score > 0.6:
            status = "FAIR - Moderate fairness"
        else:
            status = "POOR - Low fairness"
        
        print(f"  Status: {status}")

if __name__ == "__main__":
    fairness_analysis_example()


