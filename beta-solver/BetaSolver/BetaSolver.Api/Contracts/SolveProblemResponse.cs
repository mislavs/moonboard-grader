using BetaSolver.Core.Models;
using BetaSolver.Core.Solver;

namespace BetaSolver.Api.Contracts;

/// <summary>
/// Response containing the beta sequence for a problem.
/// </summary>
public sealed record SolveProblemResponse(
    List<StartHandDto> StartHands,
    List<BetaMoveDto> Moves)
{
    public SolveProblemResponse(Beta beta)
        : this(
            beta.StartHands
                .Select(kvp => new StartHandDto(kvp.Key == Hand.Left ? "LH" : "RH", kvp.Value.Description))
                .ToList(),
            beta.Moves
                .Select(m => new BetaMoveDto(m.TargetHold.Description, m.Hand == Hand.Left ? "LH" : "RH"))
                .ToList())
    {
    }
}
