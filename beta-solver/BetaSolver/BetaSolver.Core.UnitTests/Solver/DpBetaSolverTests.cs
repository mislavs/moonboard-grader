using AwesomeAssertions;
using BetaSolver.Core.Models;
using BetaSolver.Core.Scorer;
using BetaSolver.Core.Solver;

namespace BetaSolver.Core.UnitTests.Solver;

public sealed class DpBetaSolverTests
{
    private readonly DpBetaSolver _solver = new(new DistanceBasedMoveScorer());

    // Hold format: "E5" = intermediate, "E5*" = start, "E5!" = end, "E5*!" = start+end
    public static TheoryData<string[], string[]> SolveTestCases => new()
    {
        // Trivial three hold problem
        { ["E5*", "E10", "F18!"], ["E10 LH", "F18 RH"] },
        // Two start holds
        { ["D5*", "F5*", "F10", "F18!"], ["F10 LH", "F18 RH"] },
        // Prefers non-crossing moves
        { ["C5*", "B10", "H10", "E18!"], ["H10 RH", "B10 LH", "E18 LH"] },
        // Single hold problem
        { ["E10*!"], Array.Empty<string>() },
        // White Jughaul
        { ["F5*", "I8", "H10", "K13", "K15", "K17", "H18!"], ["I8 RH", "H10 LH", "K13 RH", "K15 LH", "K17 RH", "H18 LH"] },
        // To Jug or not to Jug
        { ["F5*", "D9", "H10", "F12", "G13", "D15", "H18!"], ["D9 LH", "H10 RH", "F12 LH", "G13 RH", "D15 LH", "H18 RH"] }
    };

    [Theory]
    [MemberData(nameof(SolveTestCases))]
    public void Solve_ReturnsExpectedSequence(string[] holdDescriptions, string[] expected)
    {
        // Arrange
        var holds = holdDescriptions.Select(ParseHold).ToArray();

        // Act
        var result = Solve(holds);

        // Assert
        result.Should().BeEquivalentTo(expected);
    }

    private string[] Solve(Hold[] holds)
    {
        var problem = new Problem("Test", "6A", holds);
        var result = _solver.Solve(problem);
        return result.Moves
            .Select(m => $"{m.Hold.Description} {(m.Hand == Hand.Left ? "LH" : "RH")}")
            .ToArray();
    }

    private static Hold ParseHold(string description)
    {
        var isStart = description.Contains('*');
        var isEnd = description.Contains('!');
        var holdName = description.Replace("*", "").Replace("!", "");
        return new Hold(holdName, isStart, isEnd);
    }
}
