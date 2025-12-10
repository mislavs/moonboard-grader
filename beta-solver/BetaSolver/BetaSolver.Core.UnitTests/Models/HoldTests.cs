using AwesomeAssertions;
using BetaSolver.Core.Models;

namespace BetaSolver.Core.UnitTests.Models;

public class HoldTests
{
    [Theory]
    [InlineData("A1", 0, 0)]
    [InlineData("K18", 10, 17)]
    [InlineData("E9", 4, 8)]
    [InlineData("J4", 9, 3)]
    [InlineData("F7", 5, 6)]
    public void Constructor_WithDescription_ReturnsCorrectCoordinates(string description, int expectedX, int expectedY)
    {
        // Act
        var hold = new Hold(description);

        // Assert
        hold.X.Should().Be(expectedX);
        hold.Y.Should().Be(expectedY);
    }

    [Theory]
    [InlineData("a1", 0, 0)]
    [InlineData("k18", 10, 17)]
    [InlineData("e9", 4, 8)]
    public void Constructor_WithLowercaseDescription_ReturnsCorrectCoordinates(string description, int expectedX, int expectedY)
    {
        // Act
        var hold = new Hold(description);

        // Assert
        hold.X.Should().Be(expectedX);
        hold.Y.Should().Be(expectedY);
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("A")]
    [InlineData("10")]
    [InlineData("Z10")]
    [InlineData("A99")]
    [InlineData("A-3")]
    [InlineData("A 3")]
    [InlineData("AAA")]
    public void Constructor_WithInvalidDescription_ThrowsArgumentException(string description)
    {
        // Act
        var act = () => new Hold(description);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Theory]
    [InlineData(0, 0, "A1")]
    [InlineData(10, 17, "K18")]
    [InlineData(4, 8, "E9")]
    [InlineData(9, 3, "J4")]
    public void Description_ReturnsCorrectFormat(int x, int y, string expected)
    {
        // Arrange
        var hold = new Hold(x, y, false, false);

        // Act
        var description = hold.Description;

        // Assert
        description.Should().Be(expected);
    }

    [Theory]
    [InlineData("A1")]
    [InlineData("K18")]
    [InlineData("E9")]
    [InlineData("J4")]
    [InlineData("F7")]
    [InlineData("B8")]
    public void Constructor_AndDescription_AreInverse(string description)
    {
        // Act
        var hold = new Hold(description);

        // Assert
        hold.Description.Should().Be(description);
    }

    [Fact]
    public void DistanceTo_SameHold_ReturnsZero()
    {
        // Arrange
        var hold = new Hold("E5");

        // Act
        var distance = hold.DistanceTo(hold);

        // Assert
        distance.Should().Be(0);
    }

    [Fact]
    public void DistanceTo_HorizontalMove_ReturnsCorrectDistance()
    {
        // Arrange
        var from = new Hold("A5");
        var to = new Hold("D5");  // 3 units to the right

        // Act
        var distance = from.DistanceTo(to);

        // Assert
        distance.Should().Be(3);
    }

    [Fact]
    public void DistanceTo_VerticalMove_ReturnsCorrectDistance()
    {
        // Arrange
        var from = new Hold("E1");
        var to = new Hold("E5");  // 4 units up

        // Act
        var distance = from.DistanceTo(to);

        // Assert
        distance.Should().Be(4);
    }

    [Fact]
    public void DistanceTo_DiagonalMove_ReturnsCorrectDistance()
    {
        // Arrange
        var from = new Hold("A1");
        var to = new Hold("D4");  // 3 right, 3 up

        // Act
        var distance = from.DistanceTo(to);

        // Assert
        distance.Should().BeApproximately(Math.Sqrt(18), 0.0001);
    }
}
