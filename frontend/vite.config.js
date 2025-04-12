import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  server: {
    proxy: {
      '/compare': 'http://127.0.0.1:8000'
    }
  }
})
