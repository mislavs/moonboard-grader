# Building

- use `dotnet build` to build the solution

# Running tests

- use `dotnet test` to run the tests in the solution
- NEVER run only a subset of tests, ALWAYS run all tests

# File structure

- each file should have one empty line at the end

# Class conventions

- each class should be in its own file
- classes should be `sealed` unless they are designed for inheritance

# Test conventions

- test should follow the arrange-act-assert pattern