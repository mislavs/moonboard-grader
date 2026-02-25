# Frontend OpenTelemetry with .NET Aspire

How to add browser-side distributed tracing to a React/Vite frontend, with traces flowing into the Aspire Dashboard alongside your backend services.

## Architecture

```
Browser (spans) ──OTLP/proto──▸ Aspire Dashboard OTLP endpoint
Backend (spans) ──OTLP/gRPC──▸ Aspire Dashboard OTLP endpoint
                                        │
                              Aspire Dashboard UI
                          (correlated distributed traces)
```

## How It Works

There are three layers to this setup:

### 1. Aspire Orchestration (the magic glue)

`Aspire.Hosting.JavaScript` and `AddViteApp()` in the app host automatically inject these environment variables into the `vite` process at runtime:

| Variable | Purpose |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Aspire Dashboard's OTLP collector URL |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers for the collector |
| `OTEL_RESOURCE_ATTRIBUTES` | Resource attributes (e.g., `service.instance.id`) |
| `OTEL_SERVICE_NAME` | The resource name (e.g., `"frontend"`) |

You never set these manually; Aspire provides them at runtime.

### 2. Vite Bridges Env Vars into the Browser

Since `process.env` doesn't exist in browsers, `vite.config.ts` uses `define` to bake these values into the JS bundle at build time.

### 3. Frontend Telemetry Initialization

`telemetry.ts` creates a `WebTracerProvider` that sends traces via OTLP protobuf directly to the Aspire Dashboard. `main.tsx` calls it before React renders. API calls are wrapped in spans with context propagation so backend services can correlate traces.

## Recreation Steps

### Prerequisites

- A React + Vite frontend
- A .NET Aspire app host (SDK `Aspire.AppHost.Sdk` 13.1+)

### Step 1: Install OTEL packages

```bash
npm install @opentelemetry/api \
  @opentelemetry/sdk-trace-web \
  @opentelemetry/sdk-trace-base \
  @opentelemetry/exporter-trace-otlp-proto \
  @opentelemetry/instrumentation \
  @opentelemetry/instrumentation-document-load \
  @opentelemetry/resources \
  @opentelemetry/semantic-conventions \
  @opentelemetry/context-zone
```

### Step 2: Create `src/telemetry.ts`

```typescript
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web'
import { SimpleSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/sdk-trace-base'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto'
import { registerInstrumentations } from '@opentelemetry/instrumentation'
import { DocumentLoadInstrumentation } from '@opentelemetry/instrumentation-document-load'
import { resourceFromAttributes } from '@opentelemetry/resources'
import { ATTR_SERVICE_NAME } from '@opentelemetry/semantic-conventions'
import { ZoneContextManager } from '@opentelemetry/context-zone'

export function initializeTelemetry(
  otlpEndpoint: string,
  headers: string = '',
  resourceAttributes: string = '',
  serviceName: string = ''
) {
  const exporter = new OTLPTraceExporter({
    url: `${otlpEndpoint}/v1/traces`,
    headers: parseDelimitedValues(headers),
  })

  const attributes = parseDelimitedValues(resourceAttributes)
  attributes[ATTR_SERVICE_NAME] = serviceName?.trim() || 'browser'

  const provider = new WebTracerProvider({
    resource: resourceFromAttributes(attributes),
    spanProcessors: [
      new SimpleSpanProcessor(new ConsoleSpanExporter()),
      new SimpleSpanProcessor(exporter),
    ],
  })

  provider.register({
    contextManager: new ZoneContextManager(),
  })

  registerInstrumentations({
    instrumentations: [new DocumentLoadInstrumentation()],
  })
}

function parseDelimitedValues(value: string) {
  if (!value || !value.trim()) return {}
  return Object.fromEntries(
    value.split(',').map((pair) => {
      const [k, v] = pair.split('=')
      return [k.trim(), (v ?? '').trim()]
    })
  )
}
```

### Step 3: Initialize from `src/main.tsx`

Call `initializeTelemetry` before React renders:

```typescript
import { initializeTelemetry } from './telemetry'

initializeTelemetry(
  import.meta.env.OTEL_EXPORTER_OTLP_ENDPOINT,
  import.meta.env.OTEL_EXPORTER_OTLP_HEADERS,
  import.meta.env.OTEL_RESOURCE_ATTRIBUTES,
  import.meta.env.OTEL_SERVICE_NAME
)

// ... rest of React bootstrap
```

### Step 4: Add TypeScript types in `src/vite-env.d.ts`

```typescript
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly OTEL_EXPORTER_OTLP_ENDPOINT: string
  readonly OTEL_EXPORTER_OTLP_HEADERS: string
  readonly OTEL_RESOURCE_ATTRIBUTES: string
  readonly OTEL_SERVICE_NAME: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
```

### Step 5: Bridge env vars in `vite.config.ts`

Add a `define` block so Vite replaces `import.meta.env.OTEL_*` references with the actual values from `process.env` at build time:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(() => ({
  plugins: [react()],
  define: {
    'import.meta.env.OTEL_EXPORTER_OTLP_ENDPOINT': JSON.stringify(
      process.env.OTEL_EXPORTER_OTLP_ENDPOINT ?? ''
    ),
    'import.meta.env.OTEL_EXPORTER_OTLP_HEADERS': JSON.stringify(
      process.env.OTEL_EXPORTER_OTLP_HEADERS ?? ''
    ),
    'import.meta.env.OTEL_RESOURCE_ATTRIBUTES': JSON.stringify(
      process.env.OTEL_RESOURCE_ATTRIBUTES ?? ''
    ),
    'import.meta.env.OTEL_SERVICE_NAME': JSON.stringify(
      process.env.OTEL_SERVICE_NAME ?? ''
    ),
  },
}))
```

### Step 6: Instrument your API calls

Wrap fetch calls in spans with context propagation. The `propagation.inject` call adds `traceparent`/`tracestate` headers so backend services join the same distributed trace.

```typescript
import { context, propagation, trace, SpanKind, SpanStatusCode } from '@opentelemetry/api'

async function tracedFetch<T>(
  url: string,
  spanName: string,
  options?: RequestInit
): Promise<T> {
  const tracer = trace.getTracer('frontend')
  const span = tracer.startSpan(spanName, {
    kind: SpanKind.CLIENT,
    attributes: {
      'http.method': options?.method ?? 'GET',
      'http.url': url,
    },
  })

  const spanContext = trace.setSpan(context.active(), span)
  const headers = new Headers(options?.headers)
  propagation.inject(spanContext, headers, {
    set(carrier, key, value) {
      carrier.set(key, value)
    },
  })

  try {
    const response = await context.with(spanContext, () =>
      fetch(url, { ...options, headers })
    )
    const result = await response.json()
    span.setStatus({ code: SpanStatusCode.OK })
    return result
  } catch (error) {
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error instanceof Error ? error.message : String(error),
    })
    throw error
  } finally {
    span.end()
  }
}
```

Then use `tracedFetch` instead of raw `fetch` for all API calls:

```typescript
export async function getItems(): Promise<Item[]> {
  return tracedFetch<Item[]>('/api/items', 'GET /items')
}

export async function createItem(data: NewItem): Promise<Item> {
  return tracedFetch<Item>('/api/items', 'POST /items', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
}
```

### Step 7: Wire up the Aspire app host

Add the JavaScript hosting package and register the frontend:

```csharp
#:package Aspire.Hosting.JavaScript@13.1.1

var backend = builder.AddProject("backend", "../backend/Backend.csproj");

var frontend = builder.AddViteApp("frontend", "../frontend")
    .WithReference(backend)
    .WaitFor(backend);
```

### Step 8: Configure `apphost.run.json`

Add `DOTNET_DASHBOARD_OTLP_HTTP_ENDPOINT_URL` to each profile, matching the value of `ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL`:

```json
{
  "profiles": {
    "https": {
      "environmentVariables": {
        "DOTNET_DASHBOARD_OTLP_HTTP_ENDPOINT_URL": "https://localhost:21040",
        "ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL": "https://localhost:21040"
      }
    },
    "http": {
      "environmentVariables": {
        "DOTNET_DASHBOARD_OTLP_HTTP_ENDPOINT_URL": "http://localhost:19222",
        "ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL": "http://localhost:19222"
      }
    }
  }
}
```

## What Happens at Runtime

1. `aspire run` starts the Aspire host
2. Aspire launches `vite` with `OTEL_EXPORTER_OTLP_ENDPOINT` pointing at its dashboard collector
3. Vite bakes that URL into the JS bundle via `define`
4. The browser sends traces directly to the Aspire Dashboard
5. Backend services also send traces to Aspire
6. The Aspire Dashboard shows correlated end-to-end traces (browser → backend)

## Key Gotcha

`DOTNET_DASHBOARD_OTLP_HTTP_ENDPOINT_URL` in `apphost.run.json` is **required** for the browser OTLP exporter to work. Without it, the Aspire Dashboard only exposes a gRPC OTLP endpoint, which browsers can't use.
