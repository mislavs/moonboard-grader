namespace BetaSolver.Core.Models;

/// <summary>
/// Represents a single move in a beta sequence.
/// </summary>
/// <param name="Hold">The hold being grabbed</param>
/// <param name="Hand">Which hand grabs the hold</param>
public readonly record struct Move(Hold Hold, Hand Hand)
{
    private string HandAbbreviation => Hand == Hand.Left ? "LH" : "RH";
    
    /// <summary>
    /// Returns a string representation of the move (e.g., "E10, LH")
    /// </summary>
    public override string ToString() => $"{Hold.Description}, {HandAbbreviation}";
}
