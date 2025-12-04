"""
Prompt templates for Qwen 2.5 1.5B Instruct feature interpretation.

These prompts are used to generate explanations based on statistical feature
interactions and spatial regions, without making assumptions about image content.
"""

import json


def calculate_interaction_strength(coherency):
    """
    Calculate interaction strength based on coherency value.
    
    Args:
        coherency: Coherency score (float)
    
    Returns:
        Tuple of (strength_label, description)
    """
    if coherency >= 0.40:
        return "Strong", "Pair appears frequently and has domain_similarity 1.0 or 0.8"
    elif coherency >= 0.15:
        return "Moderate", "Pair appears occasionally and/or domains partially match"
    elif coherency >= 0.05:
        return "Weak", "Pair appears rarely and/or domains unrelated"
    else:
        return "Very Weak", "Pair appears very rarely and/or domains unrelated"


def calculate_sign_status(value1, value2):
    """
    Determine if signs are mixed or aligned.
    
    Args:
        value1: First feature value (z-score)
        value2: Second feature value (z-score)
    
    Returns:
        Tuple of (status, description)
    """
    sign1 = "above" if value1 >= 0 else "below"
    sign2 = "above" if value2 >= 0 else "below"
    
    if (value1 >= 0 and value2 >= 0) or (value1 < 0 and value2 < 0):
        return "aligned", f"both {sign1} mean"
    else:
        return "mixed", f"one above mean, one below mean"


def calculate_magnitude_comparison(value1, value2):
    """
    Determine if magnitudes are similar or different.
    
    Args:
        value1: First feature value (z-score)
        value2: Second feature value (z-score)
    
    Returns:
        Tuple of (comparison, description)
    """
    abs1 = abs(value1)
    abs2 = abs(value2)
    
    # Consider similar if within 30% of each other
    ratio = min(abs1, abs2) / max(abs1, abs2) if max(abs1, abs2) > 0 else 1.0
    
    if ratio >= 0.7:
        return "similar", f"magnitudes are similar ({abs1:.2f} vs {abs2:.2f})"
    else:
        return "different", f"magnitudes differ ({abs1:.2f} vs {abs2:.2f})"


def calculate_classification_fields(interactions_json):
    """
    Calculate all classification fields programmatically from interactions JSON.
    
    Args:
        interactions_json: JSON string containing top_pairs with features, coherency, and values
    
    Returns:
        Dictionary with calculated fields for each pair and summary statistics
    """
    try:
        data = json.loads(interactions_json)
        top_pairs = data.get("top_pairs", [])
    except (json.JSONDecodeError, TypeError):
        # If it's already a dict, use it directly
        if isinstance(interactions_json, dict):
            top_pairs = interactions_json.get("top_pairs", [])
        else:
            return {}
    
    calculated_fields = []
    
    for pair in top_pairs:
        features = pair.get("features", [])
        coherency = pair.get("coherency", 0.0)
        values = pair.get("values", [])
        
        if len(features) != 2 or len(values) != 2:
            continue
        
        value1, value2 = values[0], values[1]
        feat1, feat2 = features[0], features[1]
        
        # Calculate all fields
        strength_label, strength_desc = calculate_interaction_strength(coherency)
        sign_status, sign_desc = calculate_sign_status(value1, value2)
        mag_comparison, mag_desc = calculate_magnitude_comparison(value1, value2)
        
        sign1 = "above" if value1 >= 0 else "below"
        sign2 = "above" if value2 >= 0 else "below"
        
        calculated_fields.append({
            "pair_name": f"{feat1} & {feat2}",
            "coherency": coherency,
            "strength": strength_label,
            "strength_description": strength_desc,
            "value1": value1,
            "value2": value2,
            "sign1": sign1,
            "sign2": sign2,
            "sign_status": sign_status,
            "sign_description": sign_desc,
            "magnitude_comparison": mag_comparison,
            "magnitude_description": mag_desc,
        })
    
    # Calculate dominant behavior across all pairs
    if calculated_fields:
        aligned_count = sum(1 for f in calculated_fields if f["sign_status"] == "aligned")
        mixed_count = len(calculated_fields) - aligned_count
        
        if aligned_count > mixed_count:
            dominant_behavior = "aligned"
            dominant_desc = "features tend to be aligned (both above or both below mean)"
        elif mixed_count > aligned_count:
            dominant_behavior = "contrasting"
            dominant_desc = "features tend to be contrasting (mixed signs)"
        else:
            dominant_behavior = "mixed"
            dominant_desc = "features show mixed alignment patterns"
    else:
        dominant_behavior = "unknown"
        dominant_desc = "insufficient data"
    
    return {
        "pairs": calculated_fields,
        "dominant_behavior": dominant_behavior,
        "dominant_description": dominant_desc
    }


def format_calculated_fields(fields_dict):
    """
    Format calculated fields into a readable string for the prompt.
    
    Args:
        fields_dict: Dictionary returned from calculate_classification_fields
    
    Returns:
        Formatted string with all calculated fields
    """
    if not fields_dict or not fields_dict.get("pairs"):
        return "No valid feature pairs found."
    
    lines = []
    lines.append("CALCULATED CLASSIFICATION FIELDS:")
    lines.append("=" * 50)
    
    for idx, pair_data in enumerate(fields_dict["pairs"], 1):
        lines.append(f"\nPair {idx}: {pair_data['pair_name']}")
        lines.append(f"  Coherency: {pair_data['coherency']:.4f}")
        lines.append(f"  Strength: {pair_data['strength']} ({pair_data['strength_description']})")
        lines.append(f"  Feature 1 ({pair_data['pair_name'].split(' & ')[0]}): {pair_data['value1']:.4f} ({pair_data['sign1']} mean)")
        lines.append(f"  Feature 2 ({pair_data['pair_name'].split(' & ')[1]}): {pair_data['value2']:.4f} ({pair_data['sign2']} mean)")
        lines.append(f"  Sign Status: {pair_data['sign_status']} ({pair_data['sign_description']})")
        lines.append(f"  Magnitude: {pair_data['magnitude_comparison']} ({pair_data['magnitude_description']})")
    
    lines.append(f"\nSummary:")
    lines.append(f"  Dominant Behavior: {fields_dict['dominant_behavior']} ({fields_dict['dominant_description']})")
    
    return "\n".join(lines)


def get_captioning_prompt(prob_fake, interactions_json):
    """
    Generate prompt for DEEPFAKE classification.
    
    Args:
        prob_fake: Probability of being fake (0-1)
        interactions_json: Raw JSON object (as string) of feature interactions with features, coherency, and values
        top_features_str: Formatted string of top feature values
        location_str: Formatted string of spatial regions
    
    Returns:
        Complete prompt string
    """
    # Calculate all classification fields programmatically
    calculated_fields = calculate_classification_fields(interactions_json)
    calculated_fields_str = format_calculated_fields(calculated_fields)
    
    return f"""



Task: Write 2-3 sentences in sentence format interpreting these image-style feature interactions.
Rules:
- Begin your response with a <START> tag.
- Focus on the NUMERICAL VALUES and their relationships
- Feature values are z-scores
- A negative value means the feature is below the dataset mean.
- A positive value means the feature is above the dataset mean.
- The sign DOES NOT indicate correlation direction between features.
- The magnitude DOES NOT indicate absolute "noise" or "color" <E2><80><94> only relative deviation.

{calculated_fields_str}

Output Format:
- Sentence 1 (Strongest Interaction): Use the Top Pair information above. Reference the exact coherency score and calculated strength.
- Sentence 2 (Magnitude & Signs): Reference the sign status (above/below mean) for both features and the magnitude comparison (similar/different) from the calculated fields.
- Sentence 3 (Secondary Interaction): If a second pair exists, reference its name, coherency score, strength, and sign status (mixed/aligned) from the calculated fields.
- Sentence 4 (Summary): Reference the dominant behavior (aligned/contrasting) from the calculated summary.

INTERACTING FEATURES (Raw Data):
{interactions_json}

Explanation:"""




def get_inference_prompt(prob_fake, interactions_json):
    """
    Generate prompt for DEEPFAKE classification.
    
    Args:
        prob_fake: Probability of being fake (0-1)
        interactions_json: Raw JSON object (as string) of feature interactions with features, coherency, and values
        top_features_str: Formatted string of top feature values
        location_str: Formatted string of spatial regions
    
    Returns:
        Complete prompt string
    """
    
    return f"""


Task: Write 2-3 sentences in sentence format interpreting these image-style feature interactions. 
Rules: 
- Focus on the NUMERICAL VALUES and their relationships 
- Feature values are z-scores
- A negative value means the feature is below the dataset mean.
- A positive value means the feature is above the dataset mean.
- The sign DOES NOT indicate correlation direction between features.
- The magnitude DOES NOT indicate absolute "noise" or "color" — only relative deviation.
 
INTERACTING FEATURES (Raw Data):
{interactions_json}

- Sentence 1 (Strongest Interaction): Use the Top Pair information above. Reference the exact coherency score and calculated strength.
- Sentence 2 (Secondary Interaction): Identify the second pair, reference Reference the exact coherency score, magnitude, and strength.

RELATIVE DIFFERENCE RULE:
- Calculate the percentage difference between the two absolute feature values.
- If the difference is greater than 5%, state 'magnitudes differ significantly'; otherwise, state 'magnitudes are similar'."

Explanation:"""




def get_training_prompt(prob_fake, interactions_json):
    """
    Generate prompt for DEEPFAKE classification.
    
    Args:
        prob_fake: Probability of being fake (0-1)
        interactions_json: Raw JSON object (as string) of feature interactions with features, coherency, and values
        top_features_str: Formatted string of top feature values
        location_str: Formatted string of spatial regions
    
    Returns:
        Complete prompt string
    """
    
    return f"""


Task: Write 2-3 sentences in sentence format interpreting these image-style feature interactions. 
Rules: 
- Begin your response with a <START> tag. 
- Focus on the NUMERICAL VALUES and their relationships 
- Feature values are z-scores
- A negative value means the feature is below the dataset mean.
- A positive value means the feature is above the dataset mean.
- The sign DOES NOT indicate correlation direction between features.
- The magnitude DOES NOT indicate absolute "noise" or "color" — only relative deviation.
 
INTERACTING FEATURES (Raw Data):
{interactions_json}

Explanation:"""