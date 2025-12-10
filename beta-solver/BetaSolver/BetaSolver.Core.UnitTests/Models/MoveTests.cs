using AwesomeAssertions;
using BetaSolver.Core.Models;

namespace BetaSolver.Core.UnitTests.Models;

public class MoveTests
{
    [Theory]
    [InlineData(3, Hand.Left)]
    [InlineData(0, Hand.Right)]
    public void Constructor_ShouldStoreValuesCorrectly(int holdIndex, Hand hand)
    {
        // Act
        var move = new Move(holdIndex, hand);

        // Assert
        move.HoldIndex.Should().Be(holdIndex);
        move.Hand.Should().Be(hand);
    }

    [Theory]
    [InlineData(0, Hand.Left, "Hold 0, LH")]
    [InlineData(5, Hand.Right, "Hold 5, RH")]
    public void ToString_ReturnsCorrectFormat(int holdIndex, Hand hand, string expected)
    {
        // Arrange
        var move = new Move(holdIndex, hand);

        // Act
        var result = move.ToString();

        // Assert
        result.Should().Be(expected);
    }

    [Fact]
    public void Equality_SameValues_ShouldBeEqual()
    {
        // Arrange
        var move1 = new Move(3, Hand.Left);
        var move2 = new Move(3, Hand.Left);

        // Assert
        move1.Should().Be(move2);
    }

    [Fact]
    public void Equality_DifferentHand_ShouldNotBeEqual()
    {
        // Arrange
        var move1 = new Move(3, Hand.Left);
        var move2 = new Move(3, Hand.Right);

        // Assert
        move1.Should().NotBe(move2);
    }

    [Fact]
    public void Equality_DifferentHoldIndex_ShouldNotBeEqual()
    {
        // Arrange
        var move1 = new Move(3, Hand.Left);
        var move2 = new Move(4, Hand.Left);

        // Assert
        move1.Should().NotBe(move2);
    }
}
