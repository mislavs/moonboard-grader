using BetaSolver.Api.Contracts;
using BetaSolver.Core.Models;
using BetaSolver.Core.Scorer;
using BetaSolver.Core.Solver;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenApi();
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});
builder.Services.AddScoped<IMoveScorer, DistanceBasedMoveScorer>();
builder.Services.AddScoped<IBetaSolver, DpBetaSolver>();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors();
app.UseHttpsRedirection();

app.MapPost("/solve", (SolveProblemRequest request, IBetaSolver solver) =>
    {
        var beta = solver.Solve(new Problem(request.Moves.Select(m => m.ToHold())));
        return new SolveProblemResponse(beta);
    })
    .WithName("SolveProblem");

app.Run();
