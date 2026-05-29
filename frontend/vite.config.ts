import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(() => ({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return undefined
          }

          if (id.includes('@opentelemetry')) {
            return 'vendor-otel'
          }

          if (id.includes('react')) {
            return 'vendor-react'
          }

          if (id.includes('rc-slider')) {
            return 'vendor-slider'
          }

          return 'vendor'
        },
      },
    },
  },
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
  server: {
    proxy: {
      '/api': {
        target:
          process.env.BACKEND_HTTPS ||
          process.env.BACKEND_HTTP ||
          'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/beta-api': {
        target:
          process.env.BETA_SOLVER_HTTPS ||
          process.env.BETA_SOLVER_HTTP ||
          'https://localhost:7068',
        changeOrigin: true,
        secure: false, // Allow self-signed certificates in development
        rewrite: (path) => path.replace(/^\/beta-api/, ''),
      },
    },
  },
}))
