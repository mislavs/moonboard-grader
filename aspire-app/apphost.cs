#:sdk Aspire.AppHost.Sdk@13.1.1
#:package Aspire.Hosting.JavaScript@13.1.1
#:package Aspire.Hosting.Python@13.1.1

var builder = DistributedApplication.CreateBuilder(args);

var backend = builder.AddUvicornApp("backend", "../backend", "app.main:app")
    .WithUv()
    .WithExternalHttpEndpoints()
    .WithHttpHealthCheck("/health");

var solverBackend = builder.AddProject("solver-backend", "../beta-solver/BetaSolver/BetaSolver.Api/BetaSolver.Api.csproj");

var frontend = builder.AddViteApp("frontend", "../frontend")
    .WithReference(backend)
    .WaitFor(backend);

builder.Build().Run();
