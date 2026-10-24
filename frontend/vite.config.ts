import { defineConfig } from "vite";

// base "./" makes every asset path relative, so the built page works both on
// GitHub Pages (https://<user>.github.io/<repo>/) and when previewed locally.
export default defineConfig({
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
