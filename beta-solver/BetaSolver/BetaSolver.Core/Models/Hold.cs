using System.Globalization;

namespace BetaSolver.Core.Models;

/// <summary>
/// Represents a hold on the MoonBoard grid.
/// </summary>
/// <param name="X">Column position (0-10, where 0=A, 10=K)</param>
/// <param name="Y">Row position (0-17, where 0=row 1, 17=row 18)</param>
/// <param name="IsStart">True if this is a starting hold</param>
/// <param name="IsEnd">True if this is a finishing hold</param>
public readonly record struct Hold(int X, int Y, bool IsStart, bool IsEnd)
{
    /// <summary>
    /// Creates a Hold from a MoonBoard description string (e.g., "J4")
    /// </summary>
    /// <param name="description">The hold description (column letter A-K followed by row number 1-18)</param>
    /// <param name="isStart">Whether this is a start hold</param>
    /// <param name="isEnd">Whether this is an end hold</param>
    /// <exception cref="ArgumentException">Thrown when the description format is invalid</exception>
    public Hold(string description, bool isStart = false, bool isEnd = false)
        : this(ParseCoordinates(description).X, ParseCoordinates(description).Y, isStart, isEnd)
    {
    }

    /// <summary>
    /// Returns the hold description in MoonBoard notation (e.g., "J4")
    /// </summary>
    public string Description => $"{(char)('A' + X)}{Y + 1}";

    /// <summary>
    /// Calculates the Euclidean distance to another hold.
    /// </summary>
    public double DistanceTo(Hold other)
    {
        var dx = other.X - X;
        var dy = other.Y - Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    private static (int X, int Y) ParseCoordinates(string description)
    {
        if (string.IsNullOrWhiteSpace(description) || description.Length < 2)
        {
            throw new ArgumentException($"Invalid hold description: '{description}'", nameof(description));
        }

        var columnChar = char.ToUpper(description[0]);
        if (columnChar is < 'A' or > 'K')
        {
            throw new ArgumentException($"Invalid column '{columnChar}'. Must be A-K.", nameof(description));
        }

        var rowPart = description[1..];
        if (!int.TryParse(rowPart, NumberStyles.None, CultureInfo.InvariantCulture, out var row) 
            || row is < 1 or > 18)
        {
            throw new ArgumentException($"Invalid row '{rowPart}'. Must be 1-18.", nameof(description));
        }

        return (columnChar - 'A', row - 1);
    }
}
