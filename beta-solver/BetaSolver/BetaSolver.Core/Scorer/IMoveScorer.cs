using BetaSolver.Core.Models;

namespace BetaSolver.Core.Scorer;

public interface IMoveScorer
{
    /// <summary>
    /// Calculates the ease factor for completing a move (0-1 range).
    /// Higher values are better (1.0 = no penalty, approaching 0 = very difficult).
    /// </summary>
    /// <param name="targetIndex">Index of the hold being grabbed</param>
    /// <param name="originIndex">Index of the hold the moving hand is currently on (origin)</param>
    /// <param name="otherHandIndex">Index of the hold the other (stationary) hand is on</param>
    /// <param name="hand">Which hand is moving</param>
    /// <param name="holds">The full list of holds for coordinate lookup</param>
    /// <returns>Ease factor between 0 and 1</returns>
    double ScoreMove(int targetIndex, int originIndex, int otherHandIndex, Hand hand, IReadOnlyList<Hold> holds);
}
