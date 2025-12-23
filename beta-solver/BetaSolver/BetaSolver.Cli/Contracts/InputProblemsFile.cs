using System.Text.Json.Serialization;

namespace BetaSolver.Contracts;

/// <summary>
/// Represents the top-level structure of the moonboard problems JSON file.
/// </summary>
public sealed record InputProblemsFile(
    [property: JsonPropertyName("total")] int Total,
    [property: JsonPropertyName("data")] List<InputProblem> Data);

/// <summary>
/// Represents a single problem from the moonboard problems JSON.
/// </summary>
public sealed record InputProblem(
    [property: JsonPropertyName("apiId")] int? ApiId,
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("grade")] string Grade,
    [property: JsonPropertyName("moves")] List<InputMove> Moves);

/// <summary>
/// Represents a single move/hold in an input problem.
/// </summary>
public sealed record InputMove(
    [property: JsonPropertyName("problemId")] int ProblemId,
    [property: JsonPropertyName("description")] string Description,
    [property: JsonPropertyName("isStart")] bool IsStart,
    [property: JsonPropertyName("isEnd")] bool IsEnd);
