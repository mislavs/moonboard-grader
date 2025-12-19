using AwesomeAssertions;
using BetaSolver.Core.Models;

namespace BetaSolver.Core.UnitTests.Models;

public class MoveTests
{
    private static readonly Hold DefaultOrigin = new("A1");
    private static readonly Hold DefaultStationary = new("A2");

    [Theory]
    [InlineData("E10", Hand.Left, 0.85, "E10, LH (0.85)")]
    [InlineData("J4", Hand.Right, 0.50, "J4, RH (0.50)")]
    public void ToString_ReturnsCorrectFormat(string holdDescription, Hand hand, double score, string expected)
    {
        // Arrange
        var move = new Move(new Hold(holdDescription), hand, score, DefaultStationary, DefaultOrigin);

        // Act
        var result = move.ToString();

        // Assert
        result.Should().Be(expected);
    }

    [Fact]
    public void Equality_SameValues_ShouldBeEqual()
    {
        // Arrange
        var move1 = new Move(new Hold("E10"), Hand.Left, 0.9, DefaultStationary, DefaultOrigin);
        var move2 = new Move(new Hold("E10"), Hand.Left, 0.9, DefaultStationary, DefaultOrigin);

        // Assert
        move1.Should().Be(move2);
    }

    [Fact]
    public void Equality_DifferentHand_ShouldNotBeEqual()
    {
        // Arrange
        var move1 = new Move(new Hold("E10"), Hand.Left, 0.9, DefaultStationary, DefaultOrigin);
        var move2 = new Move(new Hold("E10"), Hand.Right, 0.9, DefaultStationary, DefaultOrigin);

        // Assert
        move1.Should().NotBe(move2);
    }

    [Fact]
    public void Equality_DifferentHold_ShouldNotBeEqual()
    {
        // Arrange
        var move1 = new Move(new Hold("E10"), Hand.Left, 0.9, DefaultStationary, DefaultOrigin);
        var move2 = new Move(new Hold("F10"), Hand.Left, 0.9, DefaultStationary, DefaultOrigin);

        // Assert
        move1.Should().NotBe(move2);
    }

    [Fact]
    public void Equality_DifferentScore_ShouldNotBeEqual()
    {
        // Arrange
        var move1 = new Move(new Hold("E10"), Hand.Left, 0.9, DefaultStationary, DefaultOrigin);
        var move2 = new Move(new Hold("E10"), Hand.Left, 0.5, DefaultStationary, DefaultOrigin);

        // Assert
        move1.Should().NotBe(move2);
    }
}
