using BetaSolver.Core.Models;

namespace BetaSolver.Core.Solver;

/// <summary>
/// Represents the result of solving for the optimal beta sequence.
/// </summary>
/// <param name="Moves">The ordered list of moves in the sequence</param>
public sealed record BetaSequenceResult(IReadOnlyList<Move> Moves)
{
    /// <summary>
    /// Creates an empty result with no moves.
    /// </summary>
    public static BetaSequenceResult Empty => new(Array.Empty<Move>());
}
