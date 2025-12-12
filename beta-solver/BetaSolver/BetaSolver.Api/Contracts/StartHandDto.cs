namespace BetaSolver.Api.Contracts;

/// <summary>
/// Represents which hand starts on which hold.
/// </summary>
public sealed record StartHandDto(
    string Hand,
    string Description);
