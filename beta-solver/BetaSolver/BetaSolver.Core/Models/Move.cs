using System.Globalization;

namespace BetaSolver.Core.Models;

/// <summary>
/// Represents a single move in a beta sequence.
/// </summary>
/// <param name="TargetHold">The hold being grabbed</param>
/// <param name="Hand">Which hand grabs the hold (Left or Right)</param>
/// <param name="Score">Calculated ease factor for this move (0-1)</param>
/// <param name="StationaryHold">The hold where the other hand remains during this move</param>
/// <param name="OriginHold">The hold the moving hand is leaving</param>
public readonly record struct Move(
    Hold TargetHold,
    Hand Hand,
    double Score,
    Hold StationaryHold,
    Hold OriginHold)
{
    private string HandAbbreviation => Hand == Hand.Left ? "LH" : "RH";

    /// <summary>
    /// Returns a string representation of the move (e.g., "E10, LH (0.85)")
    /// </summary>
    public override string ToString() => $"{TargetHold.Description}, {HandAbbreviation} ({Score.ToString("F2", CultureInfo.InvariantCulture)})";
}
