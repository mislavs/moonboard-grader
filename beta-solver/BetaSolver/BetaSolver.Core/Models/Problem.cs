namespace BetaSolver.Core.Models;

/// <summary>
/// Represents a MoonBoard climbing problem
/// </summary>
public sealed class Problem
{
    /// <summary>
    /// The name of the problem.
    /// </summary>
    public string Name { get; }
    
    /// <summary>
    /// The grade of the problem (e.g., "6B+").
    /// </summary>
    public string Grade { get; }
    
    /// <summary>
    /// The list of holds in this problem, sorted by Y-coordinate (bottom to top).
    /// </summary>
    public IReadOnlyList<Hold> Holds { get; }
    
    /// <summary>
    /// The API ID of the problem (optional).
    /// </summary>
    public int? ApiId { get; }
    
    /// <summary>
    /// Gets the start holds.
    /// </summary>
    public IEnumerable<Hold> StartHolds => Holds.Where(h => h.IsStart);
    
    /// <summary>
    /// Gets the end holds.
    /// </summary>
    public IEnumerable<Hold> EndHolds => Holds.Where(h => h.IsEnd);

    public Problem(string name, string grade, IEnumerable<Hold> holds, int? apiId = null)
    {
        Name = name;
        Grade = grade;
        ApiId = apiId;
        
        // Sort holds by Y-coordinate (bottom to top), then by X for consistency
        Holds = holds
            .OrderBy(h => h.Y)
            .ThenBy(h => h.X)
            .ToList()
            .AsReadOnly();
        
        ValidateProblem();
    }
    
    private void ValidateProblem()
    {
        if (StartHolds.Count() is < 1 or > 2)
        {
            throw new ArgumentException($"Problem must have 1-2 start holds, but has {StartHolds.Count()}.");
        }
        
        if (EndHolds.Count() is < 1 or > 2)
        {
            throw new ArgumentException($"Problem must have 1-2 end holds, but has {EndHolds.Count()}.");
        }
    }
    
    public override string ToString() => $"{Name} ({Grade}) - {Holds.Count} holds";
}
