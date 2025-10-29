import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: { outDir: 'dist' },
  server: { https: false, host: '0.0.0.0' },
});