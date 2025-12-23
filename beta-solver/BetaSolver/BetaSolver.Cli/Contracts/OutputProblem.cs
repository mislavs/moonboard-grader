using System.Text.Json.Serialization;

namespace BetaSolver.Contracts;

/// <summary>
/// Output DTO for a solved problem with beta sequence.
/// </summary>
public sealed record OutputProblem(
    [property: JsonPropertyName("apiId")] int? ApiId,
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("grade")] string Grade,
    [property: JsonPropertyName("moves")] List<OutputMove> Moves);

