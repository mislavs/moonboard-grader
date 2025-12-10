using System.Text.Json;
using AwesomeAssertions;
using BetaSolver.Core.Models;

namespace BetaSolver.Core.UnitTests.Models;

public class ProblemTests
{
    [Fact]
    public void Problem_SortsHoldsByYCoordinate()
    {
        // Arrange
        var holds = new[]
        {
            new Hold("F18", isEnd: true),     // Y = 17
            new Hold("J4", isStart: true),    // Y = 3
            new Hold("G6"),                   // Y = 5
            new Hold("E10"),                  // Y = 9
        };

        // Act
        var problem = new Problem("Test", "6B+", holds);

        // Assert
        problem.Holds[0].Y.Should().Be(3);   // J4 first (bottom)
        problem.Holds[1].Y.Should().Be(5);   // G6
        problem.Holds[2].Y.Should().Be(9);   // E10
        problem.Holds[3].Y.Should().Be(17);  // F18 last (top)
    }

    [Fact]
    public void Problem_StartHolds_ReturnsOnlyStartHolds()
    {
        // Arrange
        var startHold1 = new Hold("J4", isStart: true); 
        var startHold2 = new Hold("E5", isStart: true);
        var endHold = new Hold("F18", isEnd: true); 
        var midHold = new Hold("G6");
        
        var problem = new Problem("Test", "6B+", [startHold1, startHold2, endHold, midHold]);

        // Act
        var startHolds = problem.StartHolds.ToList();

        // Assert
        startHolds.Should().HaveCount(2);
        startHolds.Should().BeEquivalentTo([startHold1, startHold2]);
    }

    [Fact]
    public void Problem_EndHolds_ReturnsOnlyEndHolds()
    {
        // Arrange
        var startHold = new Hold("J4", isStart: true);
        var endHold = new Hold("F18", isEnd: true);
        var midHold = new Hold("G6");

        var problem = new Problem("Test", "6B+", [startHold, endHold, midHold]);

        // Act
        var endHolds = problem.EndHolds.ToList();

        // Assert
        endHolds.Should().HaveCount(1);
        endHolds.Should().BeEquivalentTo([endHold]);
    }

    [Fact]
    public void Problem_ThrowsIfNoStartHolds()
    {
        // Arrange
        var holds = new[]
        {
            new Hold("G6"),
            new Hold("F18", isEnd: true)
        };

        // Act
        var act = () => new Problem("Test", "6B+", holds);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Problem_ThrowsIfNoEndHolds()
    {
        // Arrange
        var holds = new[]
        {
            new Hold("J4", isStart: true),
            new Hold("G6")
        };

        // Act
        var act = () => new Problem("Test", "6B+", holds);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Problem_ThrowsIfTooManyStartHolds()
    {
        // Arrange
        var holds = new[]
        {
            new Hold("J4", isStart: true),
            new Hold("E5", isStart: true),
            new Hold("A3", isStart: true),
            new Hold("F18", isEnd: true)
        };

        // Act
        var act = () => new Problem("Test", "6B+", holds);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ToString_ReturnsCorrectFormat()
    {
        // Arrange
        var holds = new[]
        {
            new Hold("J4", isStart: true),
            new Hold("G6"),
            new Hold("F18", isEnd: true)
        };
        var problem = new Problem("Test Problem", "6B+", holds);

        // Act
        var result = problem.ToString();

        // Assert
        result.Should().Be("Test Problem (6B+) - 3 holds");
    }
}
