import { defineConfig } from 'vite';

export default defineConfig({
  // Base path for deployment (useful for Vercel/GitHub Pages)
  base: './', 
  build: {
    // Ensures assets and modules are built correctly
    outDir: 'dist', 
  },
  server: {
    // Required for the camera to work on local development (HTTPS required)
    https: false, 
    // You can set this to true if you are using HTTPS setup (recommended for AR)
    host: '0.0.0.0' 
  },
});
