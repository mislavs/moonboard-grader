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
from collections import defaultdict
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


def analyze_holds(problems: list) -> dict:
    """
    Analyze all holds and calculate statistics for each position.
    
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
        mean_grade_idx = mean(grades)
        median_grade_idx = median(grades)
        
        # Convert to discrete grades
        min_grade, min_grade_idx_discrete = GRADES[min_grade_idx], min_grade_idx
        mean_grade, mean_grade_idx_discrete = index_to_grade(mean_grade_idx)
        median_grade, median_grade_idx_discrete = index_to_grade(median_grade_idx)
        
        # Calculate grade distribution
        grade_distribution = defaultdict(int)
        for g in grades:
            grade_distribution[GRADES[g]] += 1
        
        hold_stats[pos] = {
            'minGrade': min_grade,
            'minGradeIndex': min_grade_idx_discrete,
            'meanGrade': mean_grade,
            'medianGrade': median_grade,
            'frequency': len(grades),
            'asStart': data['as_start'],
            'asMiddle': data['as_middle'],
            'asEnd': data['as_end'],
            'gradeDistribution': dict(grade_distribution),
        }
    
    return hold_stats


def generate_heatmaps(hold_stats: dict) -> dict:
    """
    Generate pre-computed 18x11 heatmaps for different metrics.
    
    Each heatmap is normalized to 0-1 range.
    Grid layout: [row][col] where row 0 = row 1, col 0 = column A
    
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
    
    # Analyze holds
    print("Analyzing hold statistics...")
    hold_stats = analyze_holds(filtered_problems)
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
    
    # Find easiest and hardest holds by mean grade
    sorted_by_mean = sorted(hold_stats.items(), key=lambda x: GRADE_TO_INDEX[x[1]['meanGrade']])
    print(f"\nEasiest holds (by mean grade):")
    for pos, stats in sorted_by_mean[:5]:
        print(f"  {pos}: {stats['meanGrade']} (freq: {stats['frequency']})")
    
    print(f"\nHardest holds (by mean grade):")
    for pos, stats in sorted_by_mean[-5:]:
        print(f"  {pos}: {stats['meanGrade']} (freq: {stats['frequency']})")
    
    # Most used holds
    sorted_by_freq = sorted(hold_stats.items(), key=lambda x: x[1]['frequency'], reverse=True)
    print(f"\nMost used holds:")
    for pos, stats in sorted_by_freq[:5]:
        print(f"  {pos}: {stats['frequency']} problems ({stats['meanGrade']} mean)")
    
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()

