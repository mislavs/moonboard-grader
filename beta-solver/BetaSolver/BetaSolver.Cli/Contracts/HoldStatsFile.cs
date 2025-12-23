using System.Text.Json.Serialization;

namespace BetaSolver.Contracts;

/// <summary>
/// Represents the top-level structure of the hold_stats.json file.
/// </summary>
public sealed record HoldStatsFile(
    [property: JsonPropertyName("holds")] Dictionary<string, HoldStats> Holds);

/// <summary>
/// Statistics for a single hold position.
/// </summary>
public sealed record HoldStats(
    [property: JsonPropertyName("minGrade")] string MinGrade,
    [property: JsonPropertyName("minGradeIndex")] int MinGradeIndex,
    [property: JsonPropertyName("meanGrade")] string MeanGrade,
    [property: JsonPropertyName("medianGrade")] string MedianGrade,
    [property: JsonPropertyName("frequency")] int Frequency,
    [property: JsonPropertyName("asStart")] int AsStart,
    [property: JsonPropertyName("asMiddle")] int AsMiddle,
    [property: JsonPropertyName("asEnd")] int AsEnd)
{
    /// <summary>
    /// Ordered list of moonboard grades from easiest to hardest.
    /// </summary>
    private static readonly string[] GradeOrder =
    [
        "5+", "6A", "6A+", "6B", "6B+", "6C", "6C+",
        "7A", "7A+", "7B", "7B+", "7C", "7C+",
        "8A", "8A+", "8B", "8B+", "8C", "8C+"
    ];

    /// <summary>
    /// Maps grade string to index (0-17).
    /// </summary>
    private static readonly Dictionary<string, int> GradeToIndex = GradeOrder
        .Select((grade, index) => (grade, index))
        .ToDictionary(x => x.grade, x => x.index);

    /// <summary>
    /// Gets the mean grade as an index (0-18).
    /// </summary>
    public int MeanGradeIndex => GradeToIndex.TryGetValue(MeanGrade, out var idx) ? idx : 0;

    /// <summary>
    /// Calculates difficulty on a 1-10 scale from the mean grade.
    /// Formula: difficulty = 1 + (meanGradeIndex * 9.0 / 18.0)
    /// </summary>
    public double Difficulty => 1.0 + (MeanGradeIndex * 9.0 / 18.0);
}

