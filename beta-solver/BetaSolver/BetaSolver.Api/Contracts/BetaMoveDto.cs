namespace BetaSolver.Api.Contracts;

/// <summary>
/// Represents a single move in the beta sequence.
/// </summary>
public sealed record BetaMoveDto(
    string Hold,
    string Hand);
