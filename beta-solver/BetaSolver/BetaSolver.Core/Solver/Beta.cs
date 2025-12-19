using BetaSolver.Core.Models;

namespace BetaSolver.Core.Solver;

/// <summary>
/// Represents the result of solving for the optimal beta sequence.
/// </summary>
/// <param name="Moves">The ordered list of scored moves in the sequence with full spatial context</param>
/// <param name="StartHands">Maps each hand to the starting hold it is on</param>
public sealed record Beta(
    IReadOnlyList<Move> Moves,
    IReadOnlyDictionary<Hand, Hold> StartHands)
{
    /// <summary>
    /// Creates an empty result with no moves.
    /// </summary>
    public static Beta Empty => new([], new Dictionary<Hand, Hold>());
}
