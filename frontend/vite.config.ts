import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: process.env.BACKEND_HTTPS || process.env.BACKEND_HTTP ||'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/beta-api': {
        target: process.env.BETA_SOLVER_HTTPS || process.env.BETA_SOLVER_HTTP || 'https://localhost:7068',
        changeOrigin: true,
        secure: false, // Allow self-signed certificates in development
        rewrite: (path) => path.replace(/^\/beta-api/, ''),
      },
    },
  },
})
