"""
Privacy evaluation API endpoints
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd

from sdk.privacy import (
    PrivacyEvaluator,
    DifferentialPrivacyValidator,
    KAnonymityValidator,
    LDiversityValidator,
    TClosenessValidator,
    MembershipInferenceAttack,
    AttributeDisclosureRisk
)
from api.deps import get_current_user

router = APIRouter(prefix="/privacy", tags=["privacy"])


class PrivacyEvaluationRequest(BaseModel):
    """Request model for privacy evaluation"""
    epsilon: float = Field(default=1.0, description="Differential privacy epsilon")
    delta: float = Field(default=1e-5, description="Differential privacy delta")
    k_threshold: int = Field(default=5, description="k-anonymity threshold")
    l_threshold: int = Field(default=2, description="l-diversity threshold")
    t_threshold: float = Field(default=0.2, description="t-closeness threshold")


class PrivacyMetricsResponse(BaseModel):
    """Response model for privacy metrics"""
    privacy_score: float
    epsilon: float
    delta: float
    k_anonymity: int
    l_diversity: float
    t_closeness: float
    membership_disclosure_risk: float
    attribute_disclosure_risk: float
    satisfies_differential_privacy: bool
    satisfies_k_anonymity: bool
    satisfies_l_diversity: bool
    satisfies_t_closeness: bool


class KAnonymityRequest(BaseModel):
    """Request model for k-anonymity check"""
    k_threshold: int = 5
    quasi_identifiers: Optional[List[str]] = None


class LDiversityRequest(BaseModel):
    """Request model for l-diversity check"""
    l_threshold: int = 2
    quasi_identifiers: Optional[List[str]] = None
    sensitive_attributes: Optional[List[str]] = None


class AttackRiskRequest(BaseModel):
    """Request model for attack risk assessment"""
    n_neighbors: int = 5
    sensitive_columns: Optional[List[str]] = None
    test_size: float = 0.2


@router.post("/evaluate", response_model=PrivacyMetricsResponse)
async def evaluate_privacy(
    real_file: UploadFile = File(..., description="Real/original data file"),
    synthetic_file: UploadFile = File(..., description="Synthetic data file"),
    request: PrivacyEvaluationRequest = PrivacyEvaluationRequest(),
    current_user = Depends(get_current_user)
):
    """Comprehensive privacy evaluation of synthetic data"""
    
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as real_tmp:
        content = await real_file.read()
        real_tmp.write(content)
        real_path = real_tmp.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as synth_tmp:
        content = await synthetic_file.read()
        synth_tmp.write(content)
        synth_path = synth_tmp.name
    
    try:
        # Load data
        real_df = pd.read_csv(real_path)
        synthetic_df = pd.read_csv(synth_path)
        
        # Create evaluator
        evaluator = PrivacyEvaluator(
            epsilon=request.epsilon,
            delta=request.delta,
            k_threshold=request.k_threshold,
            l_threshold=request.l_threshold,
            t_threshold=request.t_threshold
        )
        
        # Evaluate privacy
        metrics = evaluator.evaluate_privacy(real_df, synthetic_df)
        
        return PrivacyMetricsResponse(
            privacy_score=metrics.privacy_score,
            epsilon=metrics.epsilon,
            delta=metrics.delta,
            k_anonymity=metrics.k_anonymity,
            l_diversity=metrics.l_diversity,
            t_closeness=metrics.t_closeness,
            membership_disclosure_risk=metrics.membership_disclosure_risk,
            attribute_disclosure_risk=metrics.attribute_disclosure_risk,
            satisfies_differential_privacy=metrics.epsilon <= request.epsilon,
            satisfies_k_anonymity=metrics.k_anonymity >= request.k_threshold,
            satisfies_l_diversity=metrics.l_diversity >= request.l_threshold,
            satisfies_t_closeness=metrics.t_closeness <= request.t_threshold
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        # Clean up temp files
        import os
        for path in [real_path, synth_path]:
            if os.path.exists(path):
                os.unlink(path)


@router.post("/evaluate/differential-privacy")
async def check_differential_privacy(
    real_file: UploadFile = File(...),
    synthetic_file: UploadFile = File(...),
    epsilon: float = 1.0,
    delta: float = 1e-5,
    num_trials: int = 100,
    current_user = Depends(get_current_user)
):
    """Check differential privacy guarantees"""
    
    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as real_tmp:
        content = await real_file.read()
        real_tmp.write(content)
        real_path = real_tmp.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as synth_tmp:
        content = await synthetic_file.read()
        synth_tmp.write(content)
        synth_path = synth_tmp.name
    
    try:
        # Load data
        real_df = pd.read_csv(real_path)
        synthetic_df = pd.read_csv(synth_path)
        
        # Create validator
        dp_validator = DifferentialPrivacyValidator(epsilon=epsilon, delta=delta)
        
        # Check DP
        dp_results = dp_validator.check_differential_privacy(real_df, synthetic_df)
        
        # Estimate epsilon
        estimated_epsilon = dp_validator.estimate_epsilon(
            real_df, synthetic_df, num_trials
        )
        
        return {
            'satisfies_dp': dp_results['satisfies_dp'],
            'target_epsilon': epsilon,
            'estimated_epsilon': estimated_epsilon,
            'delta': delta,
            'privacy_loss': dp_results['privacy_loss']
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        import os
        for path in [real_path, synth_path]:
            if os.path.exists(path):
                os.unlink(path)


@router.post("/evaluate/k-anonymity")
async def check_k_anonymity(
    file: UploadFile = File(...),
    request: KAnonymityRequest = KAnonymityRequest(),
    current_user = Depends(get_current_user)
):
    """Check k-anonymity of a dataset"""
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        file_path = tmp.name
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Create validator
        validator = KAnonymityValidator(quasi_identifiers=request.quasi_identifiers)
        
        # Check k-anonymity
        results = validator.check_k_anonymity(df, request.k_threshold)
        
        # Get risky groups if any
        risky_groups = []
        if not results['satisfies_k_anonymity']:
            risky_df = validator.get_risky_groups(df, request.k_threshold)
            risky_groups = risky_df.head(10).to_dict('records')
        
        return {
            'k_value': results['k_value'],
            'k_threshold': request.k_threshold,
            'satisfies_k_anonymity': results['satisfies_k_anonymity'],
            'quasi_identifiers': results['quasi_identifiers'],
            'risky_groups': risky_groups,
            'risky_groups_count': len(risky_groups)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        import os
        if os.path.exists(file_path):
            os.unlink(file_path)


@router.post("/evaluate/l-diversity")
async def check_l_diversity(
    file: UploadFile = File(...),
    request: LDiversityRequest = LDiversityRequest(),
    current_user = Depends(get_current_user)
):
    """Check l-diversity of a dataset"""
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        file_path = tmp.name
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Create validator
        validator = LDiversityValidator(
            quasi_identifiers=request.quasi_identifiers,
            sensitive_attributes=request.sensitive_attributes
        )
        
        # Check l-diversity
        results = validator.check_l_diversity(df, request.l_threshold)
        
        # Compute entropy l-diversity
        entropy_l = validator.compute_entropy_l_diversity(df)
        
        return {
            'l_value': results['l_value'],
            'l_threshold': request.l_threshold,
            'satisfies_l_diversity': results['satisfies_l_diversity'],
            'entropy_l_diversity': entropy_l,
            'quasi_identifiers': results['quasi_identifiers'],
            'sensitive_attributes': results['sensitive_attributes']
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        import os
        if os.path.exists(file_path):
            os.unlink(file_path)


@router.post("/evaluate/t-closeness")
async def check_t_closeness(
    file: UploadFile = File(...),
    t_threshold: float = 0.2,
    quasi_identifiers: Optional[List[str]] = None,
    sensitive_attributes: Optional[List[str]] = None,
    current_user = Depends(get_current_user)
):
    """Check t-closeness of a dataset"""
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        file_path = tmp.name
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Create validator
        validator = TClosenessValidator(
            quasi_identifiers=quasi_identifiers,
            sensitive_attributes=sensitive_attributes
        )
        
        # Check t-closeness
        results = validator.check_t_closeness(df, t_threshold)
        
        return {
            't_value': results['t_value'],
            't_threshold': t_threshold,
            'satisfies_t_closeness': results['satisfies_t_closeness'],
            'quasi_identifiers': results['quasi_identifiers'],
            'sensitive_attributes': results['sensitive_attributes']
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        import os
        if os.path.exists(file_path):
            os.unlink(file_path)


@router.post("/evaluate/attack-risk")
async def assess_attack_risk(
    real_file: UploadFile = File(...),
    synthetic_file: UploadFile = File(...),
    request: AttackRiskRequest = AttackRiskRequest(),
    current_user = Depends(get_current_user)
):
    """Assess privacy attack risks"""
    
    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as real_tmp:
        content = await real_file.read()
        real_tmp.write(content)
        real_path = real_tmp.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as synth_tmp:
        content = await synthetic_file.read()
        synth_tmp.write(content)
        synth_path = synth_tmp.name
    
    try:
        # Load data
        real_df = pd.read_csv(real_path)
        synthetic_df = pd.read_csv(synth_path)
        
        # Membership inference attack
        membership_attack = MembershipInferenceAttack(n_neighbors=request.n_neighbors)
        membership_risk = membership_attack.compute_membership_risk(
            real_df, synthetic_df, request.test_size
        )
        
        # Attribute disclosure risk
        attribute_assessor = AttributeDisclosureRisk()
        attribute_risk = attribute_assessor.compute_attribute_risk(
            real_df, synthetic_df, request.sensitive_columns
        )
        
        # Determine risk levels
        membership_level = (
            "Low" if membership_risk < 0.2 else
            "Medium" if membership_risk < 0.5 else
            "High"
        )
        
        attribute_level = (
            "Low" if attribute_risk < 0.2 else
            "Medium" if attribute_risk < 0.5 else
            "High"
        )
        
        return {
            'membership_inference_risk': membership_risk,
            'membership_risk_level': membership_level,
            'attribute_disclosure_risk': attribute_risk,
            'attribute_risk_level': attribute_level,
            'overall_risk': max(membership_risk, attribute_risk),
            'recommendations': generate_risk_recommendations(
                membership_risk, attribute_risk
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        import os
        for path in [real_path, synth_path]:
            if os.path.exists(path):
                os.unlink(path)


@router.get("/report/{file_id}")
async def generate_privacy_report(
    file_id: str,
    synthetic_file_id: str,
    current_user = Depends(get_current_user)
):
    """Generate comprehensive privacy report"""
    
    # This would load files from storage based on file_ids
    # For now, return a template response
    
    return {
        'report_id': f"privacy_report_{file_id}_{synthetic_file_id}",
        'status': 'generated',
        'sections': [
            'differential_privacy',
            'k_anonymity',
            'l_diversity',
            't_closeness',
            'attack_risks',
            'recommendations'
        ],
        'download_url': f'/privacy/report/download/{file_id}_{synthetic_file_id}'
    }


def generate_risk_recommendations(membership_risk: float, attribute_risk: float) -> List[str]:
    """Generate recommendations based on risk levels"""
    
    recommendations = []
    
    if membership_risk > 0.5:
        recommendations.extend([
            "High membership inference risk detected",
            "Consider increasing noise levels in the generation process",
            "Use stronger differential privacy guarantees (lower epsilon)",
            "Implement output perturbation techniques"
        ])
    elif membership_risk > 0.2:
        recommendations.extend([
            "Moderate membership inference risk",
            "Review the generation parameters",
            "Consider adding post-processing privacy filters"
        ])
    
    if attribute_risk > 0.5:
        recommendations.extend([
            "High attribute disclosure risk detected",
            "Suppress or generalize rare values in sensitive attributes",
            "Implement k-anonymity with appropriate quasi-identifiers",
            "Consider using l-diversity for sensitive attributes"
        ])
    elif attribute_risk > 0.2:
        recommendations.extend([
            "Moderate attribute disclosure risk",
            "Review handling of categorical variables",
            "Consider increasing the minimum group size (k-anonymity)"
        ])
    
    if membership_risk < 0.2 and attribute_risk < 0.2:
        recommendations.append("Privacy risks are within acceptable bounds")
    
    return recommendations