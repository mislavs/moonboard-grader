# OpenTelemetry Logging Implementation Plan

## Overview

Enable OpenTelemetry-based logging and tracing across all three systems, exporting to .NET Aspire's OTLP endpoint for unified observability in the Aspire dashboard.

## Architecture

```
Frontend (browser)                    Backend (FastAPI)                Beta Solver (.NET)
       │                                     │                                │
       │ OTLP HTTP                           │ OTLP gRPC                      │ OTLP gRPC
       │ (traces/logs)                       │ (traces/logs)                  │ (traces/logs)
       └─────────────┬───────────────────────┴────────────────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Aspire OTLP    │
            │  Endpoint       │
            │  (via OTEL_EXPORTER_OTLP_ENDPOINT env var)
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Aspire Dashboard│
            │ (logs, traces)  │
            └─────────────────┘
```

**Trace Context Flow:**
```
Frontend (generates trace) ──traceparent header──▶ Backend ──traceparent header──▶ Beta Solver
```

---

## Step 1: Backend (Python/FastAPI)

### 1.1 Add dependencies ✅

**File:** `backend/pyproject.toml`

Add to dependencies:
```toml
"opentelemetry-api>=1.20.0",
"opentelemetry-sdk>=1.20.0",
"opentelemetry-exporter-otlp>=1.20.0",
"opentelemetry-instrumentation-fastapi>=0.41b0",
"opentelemetry-instrumentation-logging>=0.41b0",
```

**Implementation Summary:** Added the five OpenTelemetry dependencies to `backend/pyproject.toml`. Ran `uv sync` to install packages - resolved 113 packages total, with 22 new packages installed including opentelemetry-api 1.39.1, opentelemetry-sdk 1.39.1, opentelemetry-exporter-otlp 1.39.1, and the instrumentation packages. All 133 existing passing tests continue to pass.

### 1.2 Create telemetry configuration ✅

**New file:** `backend/app/core/telemetry.py`

- Configure `TracerProvider` with OTLP exporter
- Configure `LoggerProvider` with OTLP exporter
- Read OTLP endpoint from environment variable `OTEL_EXPORTER_OTLP_ENDPOINT` (set by Aspire)
- Set service name: `moonboard-backend`
- Instrument FastAPI app
- Bridge Python logging to OpenTelemetry

**Implementation Summary:** Created `backend/app/core/telemetry.py` with `setup_telemetry()` function that configures TracerProvider and LoggerProvider with OTLP gRPC exporters. Reads endpoint from `OTEL_EXPORTER_OTLP_ENDPOINT` env var; gracefully disables export if not set. Also added `get_otel_logging_handler()` to provide a LoggingHandler for Python's logging integration.

### 1.3 Update logging configuration ✅

**File:** `backend/app/core/logging.py`

- Add OpenTelemetry logging handler
- Use `opentelemetry.sdk._logs` to export logs

**Implementation Summary:** Updated `setup_logging()` to call `setup_telemetry()` first, then add the OpenTelemetry logging handler alongside the console handler. Refactored to use a single `log_level` variable to avoid repeated `getattr()` calls.

### 1.4 Initialize telemetry on startup ✅

**File:** `backend/app/main.py`

- Import and call telemetry setup before app creation
- Instrument FastAPI with `FastAPIInstrumentor`

**Implementation Summary:** Added `FastAPIInstrumentor` import and call to `FastAPIInstrumentor.instrument_app(app)` in `create_application()`. Telemetry is initialized via `setup_logging()` which is called at module load.

### 1.5 Propagate trace context to beta solver ⏭️ (N/A)

**File:** backend client/service that calls beta solver

- Use `opentelemetry.propagate.inject()` to add trace headers to outbound requests
- Headers automatically include `traceparent` and `tracestate`

**Implementation Summary:** Not applicable - the backend does not directly call the beta solver. The frontend calls the beta solver directly, so trace propagation will be handled in Step 2.

---

## Step 2: Frontend (React/Vite)

### 2.1 Add npm packages

**File:** `frontend/package.json`

```json
"dependencies": {
  "@opentelemetry/api": "^1.9.0",
  "@opentelemetry/sdk-trace-web": "^1.25.0",
  "@opentelemetry/exporter-trace-otlp-http": "^0.52.0",
  "@opentelemetry/instrumentation-fetch": "^0.52.0",
  "@opentelemetry/context-zone": "^1.25.0",
  "@opentelemetry/resources": "^1.25.0",
  "@opentelemetry/semantic-conventions": "^1.25.0"
}
```

### 2.2 Create telemetry setup

**New file:** `frontend/src/telemetry/setup.ts`

- Initialize `WebTracerProvider`
- Configure `OTLPTraceExporter` with HTTP endpoint
- Register `FetchInstrumentation` to auto-instrument fetch calls
- Set service name: `moonboard-frontend`
- Export function to get current trace context for manual propagation

### 2.3 Configure OTLP endpoint proxy

**File:** `frontend/vite.config.ts`

Add proxy for OTLP endpoint to avoid CORS issues:
```typescript
'/otlp': {
  target: 'http://localhost:19222',  // Aspire OTLP endpoint
  changeOrigin: true,
  rewrite: (path) => path.replace(/^\/otlp/, '')
}
```

### 2.4 Initialize telemetry

**File:** `frontend/src/main.tsx`

- Import and call telemetry setup before React renders
- Ensure telemetry is initialized early

### 2.5 Update error handling in hooks

Files to update (replace `console.error` with span exception recording):
- `frontend/src/hooks/usePrediction.ts`
- `frontend/src/hooks/useProblems.ts`
- `frontend/src/hooks/useBeta.ts`
- `frontend/src/hooks/useGeneration.ts`
- `frontend/src/hooks/useDuplicateCheck.ts`
- `frontend/src/hooks/useBackendHealth.ts`
- `frontend/src/contexts/BoardSetupContext.tsx`
- `frontend/src/components/ViewMode.tsx`
- `frontend/src/components/AnalyticsMode.tsx`

---

## Step 3: Beta Solver (.NET)

### 3.1 Add NuGet packages

**File:** `beta-solver/BetaSolver/BetaSolver.Api/BetaSolver.Api.csproj`

```xml
<PackageReference Include="Serilog.AspNetCore" Version="9.0.0" />
<PackageReference Include="Serilog.Sinks.OpenTelemetry" Version="4.1.1" />
```

### 3.2 Configure Serilog with OpenTelemetry sink

**File:** `beta-solver/BetaSolver/BetaSolver.Api/Program.cs`

```csharp
using Serilog;

Log.Logger = new LoggerConfiguration()
    .Enrich.WithProperty("service.name", "moonboard-betasolver")
    .WriteTo.OpenTelemetry(options =>
    {
        options.Endpoint = Environment.GetEnvironmentVariable("OTEL_EXPORTER_OTLP_ENDPOINT")
            ?? "http://localhost:4317";
        options.ResourceAttributes = new Dictionary<string, object>
        {
            ["service.name"] = "moonboard-betasolver"
        };
    })
    .WriteTo.Console()
    .CreateLogger();

builder.Host.UseSerilog();
```

The OTLP endpoint is read from `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable (set by Aspire).

---

## Files Summary

### Backend (Python)
| File | Action |
|------|--------|
| `backend/pyproject.toml` | Add OpenTelemetry dependencies |
| `backend/app/core/telemetry.py` | Create (new file) |
| `backend/app/core/logging.py` | Add OTel logging handler |
| `backend/app/main.py` | Initialize telemetry |

### Frontend (React)
| File | Action |
|------|--------|
| `frontend/package.json` | Add OpenTelemetry packages |
| `frontend/src/telemetry/setup.ts` | Create (new file) |
| `frontend/vite.config.ts` | Add OTLP proxy |
| `frontend/src/main.tsx` | Initialize telemetry |
| 9 hooks/components | Replace console.error with spans |

### Beta Solver (.NET)
| File | Action |
|------|--------|
| `BetaSolver.Api.csproj` | Add Serilog packages |
| `Program.cs` | Configure Serilog with OpenTelemetry sink |

---

## Verification

1. **Backend**: Run via Aspire, make API call, verify traces appear in Aspire dashboard
2. **Frontend**: Open app, make prediction, verify frontend spans in dashboard
3. **Beta Solver**: Make `/solve` request, verify traces in dashboard with same trace ID as caller
4. **End-to-end**: Single user action should show connected spans across all three services with same trace ID
5. **Tests**: Run existing tests to ensure no regressions
   - `cd backend && pytest`
   - `cd frontend && npm run lint`
   - `cd beta-solver/BetaSolver && dotnet test`
