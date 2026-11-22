// Formats values with Streamlit's own frontend code, for tests/test_display.py.
// Usage: node streamlit_render.mjs <streamlit static/js dir>  (cases as JSON on stdin)
// Each case is {"printf": fmt, "value": number} or {"moment": fmt, "utc": [y, m, d, h, mi, s]}.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const dir = process.argv[2];
const find = (prefix) => {
  const name = readdirSync(dir).find((f) => f.startsWith(prefix) && f.endsWith(".js"));
  if (!name) throw new Error(`no ${prefix}*.js in ${dir}`);
  return pathToFileURL(join(dir, name)).href;
};

// The bundles read a few browser globals when they load.
globalThis.window ??= globalThis;

const sprintfModule = await import(find("sprintfjs."));
const sprintf = Object.values(sprintfModule).find((f) => typeof f === "function");

const momentModule = await import(find("moment."));
let moment = Object.values(momentModule).find((f) => typeof f?.utc === "function");
if (!moment) {
  // The bundle may export a lazy initializer that has to run first.
  for (const f of Object.values(momentModule)) {
    if (typeof f === "function" && f.length === 0) {
      try {
        f();
      } catch {}
    }
  }
  moment = Object.values(momentModule).find((f) => typeof f?.utc === "function");
}
if (!moment) throw new Error(`moment not found: ${Object.keys(momentModule)}`);

const cases = JSON.parse(readFileSync(0, "utf8"));
const out = cases.map((c) => {
  if ("printf" in c) return sprintf(c.printf, c.value);
  const [y, mo, d, h, mi, s] = c.utc;
  return moment.utc(Date.UTC(y, mo - 1, d, h, mi, s)).format(c.moment);
});
process.stdout.write(JSON.stringify(out));
