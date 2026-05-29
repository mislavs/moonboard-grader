import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

const telemetryEndpoint = import.meta.env.OTEL_EXPORTER_OTLP_ENDPOINT?.trim()

if (telemetryEndpoint) {
  window.__MOONBOARD_TELEMETRY_ENABLED = true
  void import('./telemetry')
    .then(({ initializeTelemetry }) => {
      initializeTelemetry(
        telemetryEndpoint,
        import.meta.env.OTEL_EXPORTER_OTLP_HEADERS,
        import.meta.env.OTEL_RESOURCE_ATTRIBUTES,
        import.meta.env.OTEL_SERVICE_NAME
      )
    })
    .catch((error) => {
      window.__MOONBOARD_TELEMETRY_ENABLED = false
      console.warn('Telemetry initialization failed:', error)
    })
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
