using BetaSolver.Core.Models;
using BetaSolver.Core.Scorer;

namespace BetaSolver.Core.Solver;

/// <summary>
/// Minimal decision info stored in memoization.
/// </summary>
/// <param name="TargetIndex">Index of the hold we're moving to</param>
/// <param name="MovedHand">Which hand made the move</param>
/// <param name="MoveScore">Score for this individual move</param>
internal readonly record struct MoveDecision(int TargetIndex, Hand MovedHand, double MoveScore);

/// <summary>
/// Solves for the optimal beta sequence using dynamic programming with memoization.
/// </summary>
public sealed class DpBetaSolver(IMoveScorer _scorer) : IBetaSolver
{
    /// <summary>
    /// Finds the optimal beta sequence for the given problem.
    /// </summary>
    /// <param name="problem">The climbing problem to solve</param>
    /// <returns>The optimal sequence of moves with scores</returns>
    public Beta Solve(Problem problem)
    {
        var holds = problem.Holds;
        var startIndices = problem.StartHoldIndices;

        if (startIndices.Count == 0)
        {
            return Beta.Empty;
        }

        // Try all valid start configurations and return the best result
        return startIndices.Count switch
        {
            1 => SolveFromSingleStart(holds, startIndices[0]),
            2 => SolveFromDoubleStart(holds, startIndices[0], startIndices[1]),
            _ => throw new InvalidOperationException($"Unexpected number of start holds: {startIndices.Count}")
        };
    }

    private Beta SolveFromSingleStart(IReadOnlyList<Hold> holds, int startIndex)
    {
        // Both hands start on the same hold
        var initialVisited = 1L << startIndex;
        var memo = new Dictionary<(int, int, long), (double Score, MoveDecision? Decision)>();

        SolveRecursive(startIndex, startIndex, initialVisited, holds, memo);
        var moves = ReconstructPath(startIndex, startIndex, initialVisited, holds, memo);
        var startHold = holds[startIndex];

        return new Beta(moves, new Dictionary<Hand, Hold>
        {
            { Hand.Left, startHold },
            { Hand.Right, startHold }
        });
    }

    private Beta SolveFromDoubleStart(IReadOnlyList<Hold> holds, int startIndex1, int startIndex2)
    {
        var initialVisited = (1L << startIndex1) | (1L << startIndex2);
        var memo1 = new Dictionary<(int, int, long), (double Score, MoveDecision? Decision)>();
        var memo2 = new Dictionary<(int, int, long), (double Score, MoveDecision? Decision)>();

        // Try assignment 1: LH on start1, RH on start2
        var (score1, _) = SolveRecursive(startIndex1, startIndex2, initialVisited, holds, memo1);

        // Try assignment 2: LH on start2, RH on start1
        var (score2, _) = SolveRecursive(startIndex2, startIndex1, initialVisited, holds, memo2);

        if (score1 >= score2)
        {
            var moves = ReconstructPath(startIndex1, startIndex2, initialVisited, holds, memo1);
            return new Beta(moves, new Dictionary<Hand, Hold>
            {
                { Hand.Left, holds[startIndex1] },
                { Hand.Right, holds[startIndex2] }
            });
        }
        else
        {
            var moves = ReconstructPath(startIndex2, startIndex1, initialVisited, holds, memo2);
            return new Beta(moves, new Dictionary<Hand, Hold>
            {
                { Hand.Left, holds[startIndex2] },
                { Hand.Right, holds[startIndex1] }
            });
        }
    }

    private (double Score, MoveDecision? Decision) SolveRecursive(
        int lhPosition,
        int rhPosition,
        long visitedMask,
        IReadOnlyList<Hold> holds,
        Dictionary<(int, int, long), (double Score, MoveDecision? Decision)> memo)
    {
        var stateKey = (lhPosition, rhPosition, visitedMask);

        if (memo.TryGetValue(stateKey, out var cached))
        {
            return cached;
        }

        // Check if terminal state (all holds visited)
        var allVisited = (1L << holds.Count) - 1;
        if (visitedMask == allVisited)
        {
            var result = (1.0, (MoveDecision?)null);
            memo[stateKey] = result;
            return result;
        }

        var bestScore = 0.0;
        MoveDecision? bestDecision = null;

        // Try all possible moves
        for (var targetIndex = 0; targetIndex < holds.Count; targetIndex++)
        {
            // Skip already visited holds
            if ((visitedMask & (1L << targetIndex)) != 0)
            {
                continue;
            }

            var newVisitedMask = visitedMask | (1L << targetIndex);

            // Try moving left hand
            var lhScore = _scorer.ScoreMove(targetIndex, lhPosition, rhPosition, Hand.Left, holds);
            if (lhScore > 0)
            {
                var (futureScore, _) = SolveRecursive(
                    targetIndex, rhPosition, newVisitedMask, holds, memo);

                var totalScore = lhScore * futureScore;
                if (totalScore > bestScore)
                {
                    bestScore = totalScore;
                    bestDecision = new MoveDecision(targetIndex, Hand.Left, lhScore);
                }
            }

            // Try moving right hand
            var rhScore = _scorer.ScoreMove(targetIndex, rhPosition, lhPosition, Hand.Right, holds);
            if (rhScore > 0)
            {
                var (futureScore, _) = SolveRecursive(
                    lhPosition, targetIndex, newVisitedMask, holds, memo);

                var totalScore = rhScore * futureScore;
                if (totalScore > bestScore)
                {
                    bestScore = totalScore;
                    bestDecision = new MoveDecision(targetIndex, Hand.Right, rhScore);
                }
            }
        }

        var finalResult = (bestScore, bestDecision);
        memo[stateKey] = finalResult;
        return finalResult;
    }

    /// <summary>
    /// Reconstructs the optimal path by walking through the memoization table.
    /// This is O(n) where n is the path length - done once at the end instead of at every recursion step.
    /// </summary>
    private static List<Move> ReconstructPath(
        int lhPosition,
        int rhPosition,
        long visitedMask,
        IReadOnlyList<Hold> holds,
        Dictionary<(int, int, long), (double Score, MoveDecision? Decision)> memo)
    {
        var moves = new List<Move>();

        while (memo.TryGetValue((lhPosition, rhPosition, visitedMask), out var entry) && entry.Decision.HasValue)
        {
            var decision = entry.Decision.Value;

            var move = new Move(
                TargetHold: holds[decision.TargetIndex],
                Hand: decision.MovedHand,
                Score: decision.MoveScore,
                StationaryHold: decision.MovedHand == Hand.Left ? holds[rhPosition] : holds[lhPosition],
                OriginHold: decision.MovedHand == Hand.Left ? holds[lhPosition] : holds[rhPosition]);

            moves.Add(move);

            // Update state to follow the path
            visitedMask |= 1L << decision.TargetIndex;
            if (decision.MovedHand == Hand.Left)
            {
                lhPosition = decision.TargetIndex;
            }
            else
            {
                rhPosition = decision.TargetIndex;
            }
        }

        return moves;
    }
}
