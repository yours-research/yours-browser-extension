import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function syncModelBundleToDist(
  bundleDir: string,
  destDir: string,
  skipNames = new Set<string>(),
): void {
  fs.mkdirSync(destDir, { recursive: true });
  const entries = fs
    .readdirSync(bundleDir, { withFileTypes: true })
    .filter((entry) => !entry.name.endsWith('.tmp') && !skipNames.has(entry.name));
  const keepNames = new Set(entries.map((entry) => entry.name));

  for (const existing of fs.readdirSync(destDir)) {
    if (!keepNames.has(existing)) {
      fs.rmSync(path.join(destDir, existing), { recursive: true, force: true });
    }
  }

  for (const entry of entries) {
    const srcPath = path.join(bundleDir, entry.name);
    const destPath = path.join(destDir, entry.name);
    if (entry.isDirectory()) {
      syncModelBundleToDist(srcPath, destPath);
      continue;
    }
    if (!entry.isFile()) {
      continue;
    }
    const srcSize = fs.statSync(srcPath).size;
    let needsSync = !fs.existsSync(destPath);
    if (!needsSync) {
      const destStat = fs.statSync(destPath);
      if (!destStat.isFile() || destStat.size !== srcSize) {
        needsSync = true;
      } else if (srcSize < 65536) {
        needsSync = !fs.readFileSync(srcPath).equals(fs.readFileSync(destPath));
      }
    }
    if (!needsSync) {
      continue;
    }
    if (fs.existsSync(destPath)) {
      fs.rmSync(destPath, { recursive: true, force: true });
    }
    try {
      fs.linkSync(srcPath, destPath);
    } catch {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function prepareDistAndAssetsPlugin(): Plugin {
  let outDir = path.resolve(__dirname, 'dist');
  let isBuild = true;

  return {
    name: 'prepare-dist-and-assets',
    configResolved(config) {
      outDir = path.resolve(__dirname, config.build.outDir);
      isBuild = config.command === 'build';
    },
    buildStart() {
      if (!isBuild || !fs.existsSync(outDir)) {
        return;
      }
      for (const entry of fs.readdirSync(outDir)) {
        if (entry === 'models') {
          continue;
        }
        fs.rmSync(path.join(outDir, entry), { recursive: true, force: true });
      }
    },
    writeBundle() {
      const wasmBuilt = path.resolve(
        __dirname,
        'crates/tinygpt-wgsl-engine/target/wasm32-unknown-unknown/release/tinygpt_wgsl_engine.wasm',
      );
      if (!fs.existsSync(wasmBuilt)) {
        throw new Error(`Missing Rust WASM engine artifact at ${wasmBuilt}. Run npm run build:wasm first.`);
      }
      const wasmOutDir = path.join(outDir, 'wasm');
      fs.mkdirSync(wasmOutDir, { recursive: true });
      fs.copyFileSync(wasmBuilt, path.join(wasmOutDir, 'tinygpt_wgsl_engine.wasm'));

      const modelsOutDir = path.join(outDir, 'models');
      if (fs.existsSync(modelsOutDir)) {
        for (const entry of fs.readdirSync(modelsOutDir)) {
          if (entry !== 'pragma-1-4b-rc1') {
            fs.rmSync(path.join(modelsOutDir, entry), { recursive: true, force: true });
          }
        }
      }

      const rc1BundleDir = path.resolve(__dirname, 'model-bundle/pragma-1-4b-rc1');
      if (fs.existsSync(path.join(rc1BundleDir, 'manifest.json'))) {
        syncModelBundleToDist(rc1BundleDir, path.join(modelsOutDir, 'pragma-1-4b-rc1'));
      }
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [react(), prepareDistAndAssetsPlugin()],
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
    outDir: 'dist',
    emptyOutDir: false,
    sourcemap: false,
    rollupOptions: {
      input: {
        sidepanel: path.resolve(__dirname, 'sidepanel.html'),
        offscreen: path.resolve(__dirname, 'offscreen.html'),
        permissions: path.resolve(__dirname, 'permissions.html'),
        background: path.resolve(__dirname, 'src/background/service-worker.ts'),
      },
      output: {
        entryFileNames: (chunkInfo) => {
          if (chunkInfo.name === 'background') {
            return 'background.js';
          }
          return 'assets/[name]-[hash].js';
        },
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
});
