using System.Text.Json.Serialization;

namespace BetaSolver.Contracts;

/// <summary>
/// Output DTO for a single move in the beta sequence with full spatial context.
/// </summary>
public sealed record OutputMove(
    [property: JsonPropertyName("targetX")] int TargetX,
    [property: JsonPropertyName("targetY")] int TargetY,
    [property: JsonPropertyName("hand")] int Hand,
    [property: JsonPropertyName("targetDifficulty")] double TargetDifficulty,
    [property: JsonPropertyName("stationaryX")] int StationaryX,
    [property: JsonPropertyName("stationaryY")] int StationaryY,
    [property: JsonPropertyName("stationaryDifficulty")] double StationaryDifficulty,
    [property: JsonPropertyName("bodyStretchDx")] int BodyStretchDx,
    [property: JsonPropertyName("bodyStretchDy")] int BodyStretchDy,
    [property: JsonPropertyName("originX")] int OriginX,
    [property: JsonPropertyName("originY")] int OriginY,
    [property: JsonPropertyName("originDifficulty")] double OriginDifficulty,
    [property: JsonPropertyName("travelDx")] int TravelDx,
    [property: JsonPropertyName("travelDy")] int TravelDy,
    [property: JsonPropertyName("successScore")] double SuccessScore);

