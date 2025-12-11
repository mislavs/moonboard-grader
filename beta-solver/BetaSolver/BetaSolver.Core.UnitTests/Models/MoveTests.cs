using AwesomeAssertions;
using BetaSolver.Core.Models;

namespace BetaSolver.Core.UnitTests.Models;

public class MoveTests
{
    [Theory]
    [InlineData("E10", Hand.Left, "E10, LH")]
    [InlineData("J4", Hand.Right, "J4, RH")]
    public void ToString_ReturnsCorrectFormat(string holdDescription, Hand hand, string expected)
    {
        // Arrange
        var move = new Move(new Hold(holdDescription), hand);

        // Act
        var result = move.ToString();

        // Assert
        result.Should().Be(expected);
    }

    [Fact]
    public void Equality_SameValues_ShouldBeEqual()
    {
        // Arrange
        var move1 = new Move(new Hold("E10"), Hand.Left);
        var move2 = new Move(new Hold("E10"), Hand.Left);

        // Assert
        move1.Should().Be(move2);
    }

    [Fact]
    public void Equality_DifferentHand_ShouldNotBeEqual()
    {
        // Arrange
        var move1 = new Move(new Hold("E10"), Hand.Left);
        var move2 = new Move(new Hold("E10"), Hand.Right);

        // Assert
        move1.Should().NotBe(move2);
    }

    [Fact]
    public void Equality_DifferentHold_ShouldNotBeEqual()
    {
        // Arrange
        var move1 = new Move(new Hold("E10"), Hand.Left);
        var move2 = new Move(new Hold("F10"), Hand.Left);

        // Assert
        move1.Should().NotBe(move2);
    }
}
