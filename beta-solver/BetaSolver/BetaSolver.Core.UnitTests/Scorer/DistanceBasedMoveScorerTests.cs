using AwesomeAssertions;
using BetaSolver.Core.Models;
using BetaSolver.Core.Scorer;

namespace BetaSolver.Core.UnitTests.Scorer;

public class DistanceBasedMoveScorerTests
{
    private readonly DistanceBasedMoveScorer _scorer = new();

    #region Ease Factor Tests

    [Fact]
    public void ScoreMove_CloseTarget_ReturnsHighFactor()
    {
        // Arrange
        var holds = new List<Hold>
        {
            new("E5"),  // Index 0 - moving hand origin
            new("D5"),  // Index 1 - other hand position
            new("F6"),  // Index 2 - target (close to origin)
        };

        // Act
        var factor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Right, holds);

        // Assert - close moves should have high ease factor
        factor.Should().BeGreaterThan(0.8);
        factor.Should().BeLessThanOrEqualTo(1.0);
    }

    [Fact]
    public void ScoreMove_FartherTarget_ReturnsLowerFactor()
    {
        // Arrange
        var holds = new List<Hold>
        {
            new("E5"),   // Index 0 - moving hand origin
            new("D5"),   // Index 1 - other hand position
            new("F6"),   // Index 2 - close target
            new("J12"),  // Index 3 - far target
        };

        // Act
        var closeFactor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Right, holds);
        var farFactor = _scorer.ScoreMove(targetIndex: 3, originIndex: 0, otherHandIndex: 1, Hand.Right, holds);

        // Assert
        farFactor.Should().BeLessThan(closeFactor);
    }

    [Fact]
    public void ScoreMove_VeryFarTarget_ReturnsVeryLowFactor()
    {
        // Arrange - distance across the board
        var holds = new List<Hold>
        {
            new("A1"),   // Index 0 - moving hand at bottom left
            new("B1"),   // Index 1 - other hand nearby
            new("K18"),  // Index 2 - target at top right (diagonal across board)
        };

        // Act
        var factor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Right, holds);

        // Assert - very low factor for extreme distance
        factor.Should().BeLessThan(0.01);
        factor.Should().BeGreaterThan(0);
    }

    [Fact]
    public void ScoreMove_AllScores_AreBetweenZeroAndOne()
    {
        // Arrange - various moves (origin, other, target)
        var testCases = new[]
        {
            (new Hold("E5"), new Hold("D5"), new Hold("F6")),   // Close
            (new Hold("A1"), new Hold("B1"), new Hold("K18")),  // Far
            (new Hold("G5"), new Hold("H5"), new Hold("C5")),   // Crossing
        };

        foreach (var (origin, other, target) in testCases)
        {
            var holds = new List<Hold> { origin, other, target };

            // Act
            var factor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Right, holds);

            // Assert
            factor.Should().BeGreaterThan(0);
            factor.Should().BeLessThanOrEqualTo(1.0);
        }
    }

    #endregion

    #region Cross-Body Penalty Tests

    [Fact]
    public void ScoreMove_RightHandCrossingLeft_ReturnsLowerFactor()
    {
        // Arrange - right hand reaching far to the left of left hand
        var crossingHolds = new List<Hold>
        {
            new("H5"),  // Index 0 - right hand (moving) origin
            new("G5"),  // Index 1 - left hand (other) position
            new("C5"),  // Index 2 - target to the left (crossing)
        };

        var nonCrossingHolds = new List<Hold>
        {
            new("H5"),  // Index 0 - right hand origin
            new("G5"),  // Index 1 - left hand position
            new("I5"),  // Index 2 - target to the right (not crossing)
        };

        // Act
        var crossingFactor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Right, crossingHolds);
        var nonCrossingFactor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Right, nonCrossingHolds);

        // Assert
        crossingFactor.Should().BeLessThan(nonCrossingFactor);
    }

    [Fact]
    public void ScoreMove_LeftHandCrossingRight_ReturnsLowerFactor()
    {
        // Arrange - left hand reaching far to the right of right hand
        var crossingHolds = new List<Hold>
        {
            new("B5"),  // Index 0 - left hand (moving) origin
            new("C5"),  // Index 1 - right hand (other) position
            new("G5"),  // Index 2 - target to the right (crossing)
        };

        var nonCrossingHolds = new List<Hold>
        {
            new("B5"),  // Index 0 - left hand origin
            new("C5"),  // Index 1 - right hand position
            new("A5"),  // Index 2 - target to the left (not crossing)
        };

        // Act
        var crossingFactor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Left, crossingHolds);
        var nonCrossingFactor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Left, nonCrossingHolds);

        // Assert
        crossingFactor.Should().BeLessThan(nonCrossingFactor);
    }

    [Fact]
    public void ScoreMove_SlightCrossing_WithinThreshold_NoCrossBodyPenalty()
    {
        // Arrange - hands crossing by only 1 grid unit (within threshold)
        // Both moves have same reach and stability distances
        var crossingHolds = new List<Hold>
        {
            new("F5"),  // Index 0 - moving hand origin
            new("E5"),  // Index 1 - other hand
            new("D5"),  // Index 2 - target 1 unit to the left of other hand (within 1-unit threshold)
        };

        var nonCrossingHolds = new List<Hold>
        {
            new("D5"),  // Index 0 - moving hand origin
            new("E5"),  // Index 1 - other hand
            new("F5"),  // Index 2 - target 1 unit to the right of other hand
        };

        // Act
        var crossingFactor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Right, crossingHolds);
        var nonCrossingFactor = _scorer.ScoreMove(targetIndex: 2, originIndex: 0, otherHandIndex: 1, Hand.Right, nonCrossingHolds);

        // Assert - same reach distance (2 units), same stability distance (1 unit), within cross-body threshold
        crossingFactor.Should().BeApproximately(nonCrossingFactor, 0.001);
    }

    #endregion
}
