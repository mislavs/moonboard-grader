using BetaSolver.Core.Models;

namespace BetaSolver.Api.Contracts;

/// <summary>
/// Represents which hand starts on which hold.
/// </summary>
public sealed record StartHandDto(
    Hand Hand,
    string Description);
