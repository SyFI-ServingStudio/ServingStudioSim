// Run the timing predictor of the browser module (`just build-wasm`) under Node.
//
//   node tests/wasm/run_predict.mjs <pkg-dir> <input.json>
//
// input.json: {"config": <predict config: arch, gpu, backends?>,
//              "kernel_data_file": <kernel API config documents path>,
//              "files": {<path the arch block names>: <file to read it from>},
//              "cases": <cases array>}
// Prints {"version", "info", "manifest", "cases"} as JSON on stdout. On a
// failure prints the error (and the panic, if one trapped) and exits 1.
// tests/test_wasm.py compares the output with native timing-predict.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [pkgDir, inputPath] = process.argv.slice(2);
const wasm = await import(pathToFileURL(join(pkgDir, "simulator_wasm.js")).href);
await wasm.default({ module_or_path: readFileSync(join(pkgDir, "simulator_wasm_bg.wasm")) });

const input = JSON.parse(readFileSync(inputPath, "utf8"));
const files = Object.fromEntries(
  Object.entries(input.files).map(([name, path]) => [name, readFileSync(path, "utf8")]),
);
try {
  const predictor = new wasm.Predictor(
    JSON.stringify(input.config),
    readFileSync(input.kernel_data_file, "utf8"),
    JSON.stringify(files),
  );
  const out = {
    version: JSON.parse(wasm.version()),
    info: JSON.parse(predictor.info()),
    manifest: JSON.parse(predictor.manifest()),
    cases: JSON.parse(predictor.predict(JSON.stringify(input.cases))).cases,
  };
  process.stdout.write(JSON.stringify(out));
} catch (err) {
  console.error(err?.message ?? String(err));
  const panic = wasm.last_panic();
  if (panic) console.error(`panic: ${panic}`);
  process.exit(1);
}
