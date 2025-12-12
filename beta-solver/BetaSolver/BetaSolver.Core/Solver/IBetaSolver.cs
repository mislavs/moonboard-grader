using BetaSolver.Core.Models;

namespace BetaSolver.Core.Solver;

public interface IBetaSolver
{
    /// <summary>
    /// Finds the optimal beta sequence for the given problem.
    /// </summary>
    /// <param name="problem">The climbing problem to solve</param>
    /// <returns>The optimal sequence of moves with scores</returns>
    BetaSequenceResult Solve(Problem problem);
}