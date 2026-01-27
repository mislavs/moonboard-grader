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
