using BetaSolver.Core.Models;
using BetaSolver.Core.Scorer;

namespace BetaSolver.Core.Solver;

/// <summary>
/// Solves for the optimal beta sequence using dynamic programming with memoization.
/// </summary>
public sealed class DpBetaSolver(IMoveScorer _scorer)
{
    /// <summary>
    /// Finds the optimal beta sequence for the given problem.
    /// </summary>
    /// <param name="problem">The climbing problem to solve</param>
    /// <returns>The optimal sequence of moves with scores</returns>
    public BetaSequenceResult Solve(Problem problem)
    {
        var holds = problem.Holds;
        var startIndices = problem.StartHoldIndices;

        if (startIndices.Count == 0)
        {
            return BetaSequenceResult.Empty;
        }

        // Try all valid start configurations and return the best result
        return startIndices.Count switch
        {
            1 => SolveFromSingleStart(holds, startIndices[0]),
            2 => SolveFromDoubleStart(holds, startIndices[0], startIndices[1]),
            _ => throw new InvalidOperationException($"Unexpected number of start holds: {startIndices.Count}")
        };
    }

    private BetaSequenceResult SolveFromSingleStart(IReadOnlyList<Hold> holds, int startIndex)
    {
        // Both hands start on the same hold
        var initialVisited = 1L << startIndex;
        var memo = new Dictionary<(int, int, long), (double Score, List<Move> Moves)>();

        var (_, moves) = SolveRecursive(startIndex, startIndex, initialVisited, holds, memo);
        return new BetaSequenceResult(moves);
    }

    private BetaSequenceResult SolveFromDoubleStart(IReadOnlyList<Hold> holds, int startIndex1, int startIndex2)
    {
        var initialVisited = (1L << startIndex1) | (1L << startIndex2);
        var memo = new Dictionary<(int, int, long), (double Score, List<Move> Moves)>();

        // Try assignment 1: LH on start1, RH on start2
        var (score1, moves1) = SolveRecursive(startIndex1, startIndex2, initialVisited, holds, memo);

        // Try assignment 2: LH on start2, RH on start1
        memo.Clear();
        var (score2, moves2) = SolveRecursive(startIndex2, startIndex1, initialVisited, holds, memo);

        // Return the better result
        var bestMoves = score1 >= score2 ? moves1 : moves2;
        return new BetaSequenceResult(bestMoves);
    }

    private (double Score, List<Move> Moves) SolveRecursive(
        int lhPosition,
        int rhPosition,
        long visitedMask,
        IReadOnlyList<Hold> holds,
        Dictionary<(int, int, long), (double Score, List<Move> Moves)> memo)
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
            var result = (1.0, new List<Move>());
            memo[stateKey] = result;
            return result;
        }

        var bestScore = 0.0;
        List<Move>? bestMoves = null;

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
            var lhScore = _scorer.ScoreMove(targetIndex, rhPosition, Hand.Left, holds);
            if (lhScore > 0)
            {
                var (futureScore, futureMoves) = SolveRecursive(
                    targetIndex, rhPosition, newVisitedMask, holds, memo);

                var totalScore = lhScore * futureScore;
                if (totalScore > bestScore)
                {
                    bestScore = totalScore;
                    bestMoves = [new Move(holds[targetIndex], Hand.Left), .. futureMoves];
                }
            }

            // Try moving right hand
            var rhScore = _scorer.ScoreMove(targetIndex, lhPosition, Hand.Right, holds);
            if (rhScore > 0)
            {
                var (futureScore, futureMoves) = SolveRecursive(
                    lhPosition, targetIndex, newVisitedMask, holds, memo);

                var totalScore = rhScore * futureScore;
                if (totalScore > bestScore)
                {
                    bestScore = totalScore;
                    bestMoves = [new Move(holds[targetIndex], Hand.Right), .. futureMoves];
                }
            }
        }

        var finalResult = (bestScore, bestMoves ?? []);
        memo[stateKey] = finalResult;
        return finalResult;
    }
}
