import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.svg"],
      manifest: {
        name: "Distance Checker",
        short_name: "Distance",
        description:
          "Real-time webcam distance and posture tracking with on-device deep learning (MediaPipe Face Landmarker), runs entirely in your browser.",
        theme_color: "#14120f",
        background_color: "#14120f",
        display: "standalone",
        start_url: "/",
        icons: [
          {
            src: "pwa-192x192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "pwa-512x512.png",
            sizes: "512x512",
            type: "image/png",
          },
          {
            src: "pwa-512x512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
      workbox: {
        // Cache the app shell plus MediaPipe's WASM/model assets (served from
        // jsdelivr/googleapis CDNs) so a repeat visit doesn't have to
        // re-download the ~10MB of WASM + model weights.
        globPatterns: ["**/*.{js,css,html,ico,svg,png}"],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/npm\/@mediapipe\/.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "mediapipe-wasm-cache",
              expiration: { maxEntries: 20, maxAgeSeconds: 60 * 60 * 24 * 30 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
          {
            urlPattern: /^https:\/\/storage\.googleapis\.com\/mediapipe-models\/.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "mediapipe-model-cache",
              expiration: { maxEntries: 5, maxAgeSeconds: 60 * 60 * 24 * 30 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
    }),
  ],
});
