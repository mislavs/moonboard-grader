using BetaSolver.Core.Models;

namespace BetaSolver.Core.Scorer;

/// <summary>
/// Scores climbing moves based on biomechanical ease.
/// Each component returns a factor (0-1), and the total is their product.
/// </summary>
public sealed class DistanceBasedMoveScorer : IMoveScorer
{
    /// <summary>
    /// Decay rate for distance (quadratic).
    /// Higher values = steeper penalty for longer moves.
    /// At rate 0.02: 2 units → ~92%, 4 units → ~73%, 6 units → ~49%, 8 units → ~28%
    /// </summary>
    private const double DistanceDecayRate = 0.02;

    /// <summary>
    /// How many grid units of crossing is tolerable before penalty.
    /// </summary>
    private const double CrossBodyThreshold = 1.0;

    /// <summary>
    /// Decay rate for cross-body moves beyond threshold.
    /// At rate 0.8: 2 units crossing → ~45%, 3 units → ~20%, 4 units → ~9%
    /// </summary>
    private const double CrossBodyDecayRate = 0.8;

    /// <summary>
    /// Calculates the ease factor for completing a move (0-1 range).
    /// Higher values are better (1.0 = no penalty, approaching 0 = very difficult).
    /// </summary>
    public double ScoreMove(int targetIndex, int otherHandIndex, Hand hand, IReadOnlyList<Hold> holds)
    {
        var targetHold = holds[targetIndex];
        var otherHandHold = holds[otherHandIndex];

        var distanceFactor = CalculateDistanceFactor(targetHold, otherHandHold);
        var crossBodyFactor = CalculateCrossBodyFactor(targetHold, otherHandHold, hand);

        // Multiply factors for combined ease score
        return distanceFactor * crossBodyFactor;
    }

    /// <summary>
    /// Calculates the distance-based ease factor.
    /// Uses quadratic exponential decay: gentle penalty for close moves,
    /// steep penalty for far moves.
    /// </summary>
    private static double CalculateDistanceFactor(Hold target, Hold otherHandHold)
    {
        var distance = otherHandHold.DistanceTo(target);

        // Quadratic decay: gentle near 0, steep as distance grows
        return Math.Exp(-DistanceDecayRate * distance * distance);
    }

    /// <summary>
    /// Calculates the cross-body ease factor.
    /// No penalty when hands don't cross (1.0); exponential decay when they do.
    /// </summary>
    private static double CalculateCrossBodyFactor(Hold target, Hold otherHandHold, Hand movingHand)
    {
        var horizontalDiff = target.X - otherHandHold.X;

        // For right hand: penalize if target is significantly left of other hand (negative diff)
        // For left hand: penalize if target is significantly right of other hand (positive diff)
        var crossAmount = movingHand == Hand.Right
            ? Math.Max(0, -horizontalDiff - CrossBodyThreshold)
            : Math.Max(0, horizontalDiff - CrossBodyThreshold);

        if (crossAmount <= 0)
        {
            return 1.0; // No penalty
        }

        return Math.Exp(-CrossBodyDecayRate * crossAmount);
    }
}
