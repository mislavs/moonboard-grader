namespace BetaSolver.Api.Contracts;

/// <summary>
/// Request to solve a boulder problem.
/// </summary>
public sealed record SolveProblemRequest(List<MoveDto> Moves);
