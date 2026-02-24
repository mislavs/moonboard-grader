import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { initializeTelemetry } from './telemetry'

initializeTelemetry(
  import.meta.env.OTEL_EXPORTER_OTLP_ENDPOINT,
  import.meta.env.OTEL_EXPORTER_OTLP_HEADERS,
  import.meta.env.OTEL_RESOURCE_ATTRIBUTES,
  import.meta.env.OTEL_SERVICE_NAME
)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
