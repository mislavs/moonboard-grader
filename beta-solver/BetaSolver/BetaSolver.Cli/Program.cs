using System.Text.Encodings.Web;
using System.Text.Json;
using BetaSolver.Contracts;
using BetaSolver.Core.Models;
using BetaSolver.Core.Scorer;
using BetaSolver.Core.Solver;

namespace BetaSolver;

public static class Program
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    public static async Task<int> Main(string[] args)
    {
        var (inputPath, outputPath, holdStatsPath) = ParseArguments(args);

        if (inputPath is null || outputPath is null)
        {
            PrintUsage();
            return 1;
        }

        try
        {
            // Load hold stats for difficulty lookup
            var holdStats = await LoadHoldStatsAsync(holdStatsPath);
            Console.WriteLine($"Loaded hold stats for {holdStats.Holds.Count} holds");

            // Load input problems
            var inputFile = await LoadInputProblemsAsync(inputPath);
            Console.WriteLine($"Loaded {inputFile.Data.Count} problems from {inputPath}");

            // Solve each problem
            var solver = new DpBetaSolver(new DistanceBasedMoveScorer());
            var outputProblems = new List<OutputProblem>();

            var solvedCount = 0;
            var errorCount = 0;

            foreach (var inputProblem in inputFile.Data)
            {
                try
                {
                    var outputProblem = SolveProblem(inputProblem, solver, holdStats);
                    outputProblems.Add(outputProblem);
                    solvedCount++;

                    if (solvedCount % 1000 == 0)
                    {
                        Console.WriteLine($"  Solved {solvedCount}/{inputFile.Data.Count} problems...");
                    }
                }
                catch (Exception ex)
                {
                    errorCount++;
                    Console.Error.WriteLine($"Error solving '{inputProblem.Name}': {ex.Message}");
                }
            }

            // Write output
            await WriteOutputAsync(outputPath, outputProblems);
            Console.WriteLine($"Wrote {outputProblems.Count} solved problems to {outputPath}");

            if (errorCount > 0)
            {
                Console.WriteLine($"Encountered {errorCount} errors during processing");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Fatal error: {ex.Message}");
            return 1;
        }
    }

    private static (string? InputPath, string? OutputPath, string HoldStatsPath) ParseArguments(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;
        string holdStatsPath = GetDefaultHoldStatsPath();

        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input" or "-i" when i + 1 < args.Length:
                    inputPath = args[++i];
                    break;
                case "--output" or "-o" when i + 1 < args.Length:
                    outputPath = args[++i];
                    break;
                case "--hold-stats" when i + 1 < args.Length:
                    holdStatsPath = args[++i];
                    break;
            }
        }

        return (inputPath, outputPath, holdStatsPath);
    }

    private static string GetDefaultHoldStatsPath()
    {
        // Try to find hold_stats.json relative to the executable or project
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "analysis", "hold_stats.json"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "backend", "data", "hold_stats.json"),
            Path.Combine(Environment.CurrentDirectory, "analysis", "hold_stats.json"),
            Path.Combine(Environment.CurrentDirectory, "backend", "data", "hold_stats.json"),
            Path.Combine(Environment.CurrentDirectory, "..", "analysis", "hold_stats.json"),
            Path.Combine(Environment.CurrentDirectory, "..", "..", "analysis", "hold_stats.json"),
        };

        foreach (var candidate in candidates)
        {
            var fullPath = Path.GetFullPath(candidate);
            if (File.Exists(fullPath))
            {
                return fullPath;
            }
        }

        // Fallback - let it fail later with a clear error
        return Path.Combine(Environment.CurrentDirectory, "analysis", "hold_stats.json");
    }

    private static void PrintUsage()
    {
        Console.WriteLine("BetaSolver CLI - Solves MoonBoard problems and outputs beta sequences with ML features");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run -- --input <problems.json> --output <solved.json>");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  -i, --input <file>      Input JSON file with moonboard problems (required)");
        Console.WriteLine("  -o, --output <file>     Output JSON file for solved problems (required)");
        Console.WriteLine("  --hold-stats <file>     Path to hold_stats.json (auto-detected by default)");
    }

    private static async Task<HoldStatsFile> LoadHoldStatsAsync(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Hold stats file not found: {path}");
        }

        await using var stream = File.OpenRead(path);
        var holdStats = await JsonSerializer.DeserializeAsync<HoldStatsFile>(stream, JsonOptions)
            ?? throw new InvalidOperationException("Failed to deserialize hold stats");

        return holdStats;
    }

    private static async Task<InputProblemsFile> LoadInputProblemsAsync(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Input file not found: {path}");
        }

        await using var stream = File.OpenRead(path);
        var inputFile = await JsonSerializer.DeserializeAsync<InputProblemsFile>(stream, JsonOptions)
            ?? throw new InvalidOperationException("Failed to deserialize input problems");

        return inputFile;
    }

    private static OutputProblem SolveProblem(InputProblem input, DpBetaSolver solver, HoldStatsFile holdStats)
    {
        // Convert input moves to holds
        var holds = input.Moves
            .Select(m => new Hold(m.Description, m.IsStart, m.IsEnd))
            .ToList();

        // Create the problem
        var problem = new Problem(input.Name, input.Grade, holds, input.ApiId);

        // Solve it
        var beta = solver.Solve(problem);

        // Convert to output format
        var outputMoves = beta.Moves
            .Select(m => ConvertToOutputMove(m, holdStats))
            .ToList();

        return new OutputProblem(
            ApiId: input.ApiId,
            Name: input.Name,
            Grade: input.Grade,
            Moves: outputMoves);
    }

    private static OutputMove ConvertToOutputMove(Move move, HoldStatsFile holdStats)
    {
        var targetDifficulty = GetHoldDifficulty(move.TargetHold.Description, holdStats);
        var stationaryDifficulty = GetHoldDifficulty(move.StationaryHold.Description, holdStats);
        var originDifficulty = GetHoldDifficulty(move.OriginHold.Description, holdStats);

        return new OutputMove(
            TargetX: move.TargetHold.X,
            TargetY: move.TargetHold.Y,
            Hand: move.Hand == Hand.Left ? 0 : 1,
            TargetDifficulty: targetDifficulty,
            StationaryX: move.StationaryHold.X,
            StationaryY: move.StationaryHold.Y,
            StationaryDifficulty: stationaryDifficulty,
            BodyStretchDx: move.TargetHold.X - move.StationaryHold.X,
            BodyStretchDy: move.TargetHold.Y - move.StationaryHold.Y,
            OriginX: move.OriginHold.X,
            OriginY: move.OriginHold.Y,
            OriginDifficulty: originDifficulty,
            TravelDx: move.TargetHold.X - move.OriginHold.X,
            TravelDy: move.TargetHold.Y - move.OriginHold.Y,
            SuccessScore: move.Score);
    }

    private static double GetHoldDifficulty(string holdDescription, HoldStatsFile holdStats)
    {
        if (holdStats.Holds.TryGetValue(holdDescription, out var stats))
        {
            return stats.Difficulty;
        }

        // Default to middle difficulty if hold not found in stats
        return 5.0;
    }

    private static async Task WriteOutputAsync(string path, List<OutputProblem> problems)
    {
        await using var stream = File.Create(path);
        await JsonSerializer.SerializeAsync(stream, problems, JsonOptions);
    }
}
