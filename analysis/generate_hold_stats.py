#!/usr/bin/env python3
"""
Generate Hold Statistics for Moonboard Analytics

This standalone script processes the Moonboard problems dataset and generates
statistics for each hold position on the board. The output is a JSON file
containing difficulty ratings, usage frequencies, and pre-computed heatmaps.

Usage:
    py analysis/generate_hold_stats.py

Output:
    analysis/hold_stats.json
"""

import json
from collections import defaultdict, Counter
from statistics import mean, median
from pathlib import Path

# Grade definitions (Font grades from easiest to hardest)
GRADES = [
    "5+",
    "6A", "6A+", "6B", "6B+", "6C", "6C+",
    "7A", "7A+", "7B", "7B+", "7C", "7C+",
    "8A", "8A+", "8B", "8B+", "8C", "8C+"
]
GRADE_TO_INDEX = {grade: idx for idx, grade in enumerate(GRADES)}

# Board dimensions
COLUMNS = "ABCDEFGHIJK"
ROWS = 18
MIN_REPEATS = 5  # Minimum repeats to include a problem

# Global grade weights (calculated from dataset distribution)
GRADE_WEIGHTS = {}  # Populated by calculate_grade_weights()


def load_problems(json_path: Path) -> list:
    """Load problems from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['data']


def filter_problems(problems: list, min_repeats: int) -> list:
    """Filter problems to only include those with sufficient repeats."""
    filtered = [p for p in problems if p.get('repeats', 0) >= min_repeats]
    print(f"Filtered {len(problems)} problems to {len(filtered)} (min {min_repeats} repeats)")
    return filtered


def calculate_grade_weights(problems: list) -> dict:
    """
    Calculate inverse frequency weights for each grade.
    
    This helps correct for dataset imbalance where easy grades are over-represented.
    Rarer grades get higher weights, more common grades get lower weights.
    
    Returns:
        Dictionary mapping grade string to its weight
    """
    global GRADE_WEIGHTS
    
    # Count problems by grade
    grade_counts = Counter()
    for problem in problems:
        grade = problem.get('grade')
        if grade in GRADE_TO_INDEX:
            grade_counts[grade] += 1
    
    total_problems = sum(grade_counts.values())
    
    if total_problems == 0:
        return {g: 1.0 for g in GRADES}
    
    # Calculate inverse frequency weights
    # Weight = (total / count) normalized so average weight = 1.0
    raw_weights = {}
    for grade in GRADES:
        count = grade_counts.get(grade, 0)
        if count > 0:
            raw_weights[grade] = total_problems / count
        else:
            raw_weights[grade] = 0.0  # No weight for grades not in dataset
    
    # Normalize so average weight across present grades is 1.0
    present_weights = [w for w in raw_weights.values() if w > 0]
    if present_weights:
        avg_weight = sum(present_weights) / len(present_weights)
        GRADE_WEIGHTS = {g: (w / avg_weight if w > 0 else 0.0) 
                         for g, w in raw_weights.items()}
    else:
        GRADE_WEIGHTS = {g: 1.0 for g in GRADES}
    
    print(f"\nGrade distribution and weights:")
    for grade in GRADES:
        count = grade_counts.get(grade, 0)
        weight = GRADE_WEIGHTS.get(grade, 0)
        if count > 0:
            print(f"  {grade}: {count:5d} problems, weight: {weight:.3f}")
    
    return GRADE_WEIGHTS


def get_dataset_grade_distribution(problems: list) -> dict:
    """
    Get the overall grade distribution of the dataset.
    
    Returns:
        Dictionary mapping grade string to its proportion in the dataset
    """
    grade_counts = Counter()
    for problem in problems:
        grade = problem.get('grade')
        if grade in GRADE_TO_INDEX:
            grade_counts[grade] += 1
    
    total = sum(grade_counts.values())
    if total == 0:
        return {}
    
    return {grade: count / total for grade, count in grade_counts.items()}


def index_to_grade(index: float) -> tuple[str, int]:
    """
    Convert a grade index (possibly float) to the nearest discrete grade.
    
    Returns:
        Tuple of (grade_string, grade_index)
    """
    rounded_index = round(index)
    # Clamp to valid range
    rounded_index = max(0, min(len(GRADES) - 1, rounded_index))
    return GRADES[rounded_index], rounded_index


def calculate_weighted_mean(grade_indices: list) -> float:
    """
    Calculate weighted mean grade index using inverse frequency weights.
    
    Args:
        grade_indices: List of grade indices for problems using a hold
        
    Returns:
        Weighted mean grade index
    """
    if not grade_indices or not GRADE_WEIGHTS:
        return mean(grade_indices) if grade_indices else 0.0
    
    weighted_sum = 0.0
    weight_total = 0.0
    
    for idx in grade_indices:
        grade = GRADES[idx]
        weight = GRADE_WEIGHTS.get(grade, 1.0)
        weighted_sum += idx * weight
        weight_total += weight
    
    return weighted_sum / weight_total if weight_total > 0 else mean(grade_indices)


def calculate_weighted_median(grade_indices: list) -> int:
    """
    Calculate weighted median grade index using inverse frequency weights.
    
    Args:
        grade_indices: List of grade indices for problems using a hold
        
    Returns:
        Weighted median grade index (integer)
    """
    if not grade_indices or not GRADE_WEIGHTS:
        return round(median(grade_indices)) if grade_indices else 0
    
    # Create weighted list of (grade_index, cumulative_weight)
    grade_counts = Counter(grade_indices)
    sorted_grades = sorted(grade_counts.keys())
    
    # Calculate cumulative weights
    cumulative = []
    running_total = 0.0
    for idx in sorted_grades:
        grade = GRADES[idx]
        weight = GRADE_WEIGHTS.get(grade, 1.0)
        count = grade_counts[idx]
        running_total += weight * count
        cumulative.append((idx, running_total))
    
    # Find the median (50% point)
    half_total = running_total / 2.0
    for idx, cum_weight in cumulative:
        if cum_weight >= half_total:
            return idx
    
    return sorted_grades[-1] if sorted_grades else 0


def analyze_holds(problems: list, dataset_distribution: dict) -> dict:
    """
    Analyze all holds and calculate statistics for each position.
    
    Args:
        problems: List of problems to analyze
        dataset_distribution: Overall grade distribution of the dataset
    
    Returns:
        Dictionary mapping hold position (e.g., "F7") to stats
    """
    # Collect data for each hold
    hold_data = defaultdict(lambda: {
        'grades': [],      # List of grade indices
        'as_start': 0,
        'as_middle': 0,
        'as_end': 0,
    })
    
    for problem in problems:
        grade = problem.get('grade')
        if grade not in GRADE_TO_INDEX:
            continue
        
        grade_idx = GRADE_TO_INDEX[grade]
        
        for move in problem.get('moves', []):
            pos = move.get('description', '').upper()
            if not pos:
                continue
            
            hold_data[pos]['grades'].append(grade_idx)
            
            if move.get('isStart', False):
                hold_data[pos]['as_start'] += 1
            elif move.get('isEnd', False):
                hold_data[pos]['as_end'] += 1
            else:
                hold_data[pos]['as_middle'] += 1
    
    # Calculate statistics for each hold
    hold_stats = {}
    
    for pos, data in hold_data.items():
        grades = data['grades']
        
        if not grades:
            continue
        
        # Calculate grade statistics
        min_grade_idx = min(grades)
        
        # Weighted mean (corrects for dataset imbalance)
        weighted_mean_idx = calculate_weighted_mean(grades)
        mean_grade, _ = index_to_grade(weighted_mean_idx)
        
        # Weighted median (corrects for dataset imbalance)
        weighted_median_idx = calculate_weighted_median(grades)
        median_grade = GRADES[weighted_median_idx]
        
        # Calculate grade distribution (raw counts)
        grade_distribution = defaultdict(int)
        for g in grades:
            grade_distribution[GRADES[g]] += 1
        
        # Calculate normalized grade distribution (ratio vs dataset)
        # >1 means over-represented, <1 means under-represented
        total_uses = len(grades)
        grade_distribution_normalized = {}
        for grade_str, count in grade_distribution.items():
            hold_proportion = count / total_uses
            dataset_proportion = dataset_distribution.get(grade_str, 0)
            if dataset_proportion > 0:
                # Ratio: how much more/less frequent this grade is for this hold
                grade_distribution_normalized[grade_str] = round(
                    hold_proportion / dataset_proportion, 2
                )
            else:
                grade_distribution_normalized[grade_str] = 0.0
        
        hold_stats[pos] = {
            'minGrade': GRADES[min_grade_idx],
            'minGradeIndex': min_grade_idx,
            'meanGrade': mean_grade,
            'medianGrade': median_grade,
            'frequency': len(grades),
            'asStart': data['as_start'],
            'asMiddle': data['as_middle'],
            'asEnd': data['as_end'],
            'gradeDistribution': dict(grade_distribution),
            'gradeDistributionNormalized': grade_distribution_normalized,
        }
    
    return hold_stats


def generate_heatmaps(hold_stats: dict) -> dict:
    """
    Generate pre-computed 18x11 heatmaps for different metrics.
    
    Each heatmap is normalized to 0-1 range.
    Grid layout: [row][col] where row 0 = row 1, col 0 = column A
    
    The meanGrade heatmap uses weighted mean values (already corrected for dataset imbalance when stored in stats; no additional weighting is applied here).
    
    Returns:
        Dictionary with keys: 'meanGrade', 'minGrade', 'frequency'
    """
    # Initialize grids with None (for missing holds)
    def empty_grid():
        return [[None for _ in range(len(COLUMNS))] for _ in range(ROWS)]
    
    mean_grid = empty_grid()
    min_grid = empty_grid()
    freq_grid = empty_grid()
    
    # Populate grids
    for pos, stats in hold_stats.items():
        if len(pos) < 2:
            continue
        
        col_char = pos[0].upper()
        try:
            row_num = int(pos[1:])
        except ValueError:
            continue
        
        if col_char not in COLUMNS or row_num < 1 or row_num > ROWS:
            continue
        
        col_idx = COLUMNS.index(col_char)
        row_idx = row_num - 1
        
        mean_grid[row_idx][col_idx] = GRADE_TO_INDEX[stats['meanGrade']]
        min_grid[row_idx][col_idx] = stats['minGradeIndex']
        freq_grid[row_idx][col_idx] = stats['frequency']
    
    def normalize_grid(grid: list) -> list:
        """Normalize grid values to 0-1 range."""
        # Collect all non-None values
        values = [v for row in grid for v in row if v is not None]
        
        if not values:
            return [[0.0 for _ in range(len(COLUMNS))] for _ in range(ROWS)]
        
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1
        
        normalized = []
        for row in grid:
            norm_row = []
            for val in row:
                if val is None:
                    norm_row.append(0.0)  # Default for missing holds
                else:
                    norm_row.append((val - min_val) / range_val)
            normalized.append(norm_row)
        
        return normalized
    
    return {
        'meanGrade': normalize_grid(mean_grid),
        'minGrade': normalize_grid(min_grid),
        'frequency': normalize_grid(freq_grid),
    }


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    problems_path = project_root / 'data' / 'problems.json'
    output_path = script_dir / 'hold_stats.json'
    
    print(f"Loading problems from: {problems_path}")
    
    # Load and filter problems
    problems = load_problems(problems_path)
    print(f"Loaded {len(problems)} total problems")
    
    filtered_problems = filter_problems(problems, MIN_REPEATS)
    
    # Calculate grade weights to correct for dataset imbalance
    print("\nCalculating grade weights...")
    calculate_grade_weights(filtered_problems)
    
    # Get dataset-wide grade distribution for normalization
    dataset_distribution = get_dataset_grade_distribution(filtered_problems)
    
    # Analyze holds
    print("\nAnalyzing hold statistics...")
    hold_stats = analyze_holds(filtered_problems, dataset_distribution)
    print(f"Analyzed {len(hold_stats)} unique hold positions")
    
    # Generate heatmaps
    print("Generating heatmaps...")
    heatmaps = generate_heatmaps(hold_stats)
    
    # Compile output
    output = {
        'holds': hold_stats,
        'heatmaps': heatmaps,
        'meta': {
            'totalProblems': len(filtered_problems),
            'totalProblemsUnfiltered': len(problems),
            'minRepeatsFilter': MIN_REPEATS,
        }
    }
    
    # Save output
    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Problems analyzed: {len(filtered_problems)}")
    print(f"Unique holds: {len(hold_stats)}")
    
    # Find easiest and hardest holds by weighted mean grade
    sorted_by_mean = sorted(hold_stats.items(), key=lambda x: GRADE_TO_INDEX[x[1]['meanGrade']])
    print(f"\nEasiest holds (by weighted mean grade):")
    for pos, stats in sorted_by_mean[:5]:
        print(f"  {pos}: {stats['meanGrade']} (freq: {stats['frequency']})")
    
    print(f"\nHardest holds (by weighted mean grade):")
    for pos, stats in sorted_by_mean[-5:]:
        print(f"  {pos}: {stats['meanGrade']} (freq: {stats['frequency']})")
    
    # Most used holds
    sorted_by_freq = sorted(hold_stats.items(), key=lambda x: x[1]['frequency'], reverse=True)
    print(f"\nMost used holds:")
    for pos, stats in sorted_by_freq[:5]:
        print(f"  {pos}: {stats['frequency']} problems ({stats['meanGrade']} weighted mean)")
    
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()

