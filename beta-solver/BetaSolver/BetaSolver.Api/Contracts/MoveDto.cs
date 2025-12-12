using BetaSolver.Core.Models;

namespace BetaSolver.Api.Contracts;

/// <summary>
/// Represents a hold position in a boulder problem.
/// </summary>
public sealed record MoveDto(
    string Description,
    bool IsStart,
    bool IsEnd)
{
    public Hold ToHold() => new(Description, IsStart, IsEnd);
}
