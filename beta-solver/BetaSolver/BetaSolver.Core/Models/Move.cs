namespace BetaSolver.Core.Models;

/// <summary>
/// Represents a single move in a beta sequence.
/// </summary>
/// <param name="HoldIndex">Index of the hold being grabbed (index into the problem's hold list)</param>
/// <param name="Hand">Which hand grabs the hold</param>
public readonly record struct Move(int HoldIndex, Hand Hand)
{
    private string HandAbbreviation => Hand == Hand.Left ? "LH" : "RH";
    
    /// <summary>
    /// Returns a string representation of the move (e.g., "Hold 3, LH")
    /// </summary>
    public override string ToString() => $"Hold {HoldIndex}, {HandAbbreviation}";
}
