// Drive render() and buildKvTable() through the page's own script with a DOM stub, over every
// model x speculation x KV x context combination.
//
// config-calculator.test.mjs exercises the arithmetic helpers -- pageRoundUp, perToken,
// fixedBytes, graphBytes, decodeAtDepth, buildKvTable -- but never render(), which is where most
// of the page's own code lives and where a wrong property name or an out-of-scope variable would
// only show up as a blank page in a browser. Adding per-mode speculative memory touched render()
// in four places, so this exists to catch that class of mistake without a browser.
//
// It asserts two things: that nothing throws across all 420 combinations, and that the reported
// largest-fitting context actually responds to the speculation mode. The second one matters --
// before per-mode terms the page reported 262,144 tokens for every mode on the 35B, including
// DFlash-3, which really tops out at 240,192.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const html = readFileSync("docs/config-calculator.html", "utf8");
const m = html.match(/<script>\n([\s\S]*?)<\/script>/);
const source = m[1];

const els = new Map();
function el(id) {
  if (!els.has(id)) {
    els.set(id, { id, value: "", textContent: "", innerHTML: "", className: "", style: {},
                  firstChild: { nodeValue: "" }, lastChild: { nodeValue: "" },
                  dataset: {}, hidden: false, children: [],
                  addEventListener() {}, appendChild(c) { this.children.push(c); },
                  querySelector() { return el(id + "-q"); },
                  querySelectorAll() { return []; },
                  setAttribute() {}, getAttribute() { return null; },
                  classList: { add() {}, remove() {}, toggle() {}, contains: () => false } });
  }
  return els.get(id);
}
const document = {
  getElementById: el,
  querySelector: (s) => el("sel:" + s),
  querySelectorAll: () => [],
  createElement: (t) => el("new:" + t + ":" + Math.random()),
  addEventListener() {},
};
const context = { document, window: { addEventListener() {}, matchMedia: () => ({ matches: false, addEventListener() {} }) },
                  console, Math, Number, String, Object, Array, JSON, isNaN, parseFloat, parseInt,
                  Infinity, NaN, location: { hash: "" }, history: { replaceState() {} } };
context.globalThis = context;
// The page runs buildControls() at load, which reads these, so seed them first.
el("model").value = "27b";
el("spec").value = "none";
el("kv").value = "int8";
el("ctx").value = "8192";
el("reserve").value = "307";
vm.createContext(context);
vm.runInContext(source, context, { filename: "config-calculator inline script" });

const { DATA, render, buildKvTable } = vm.runInContext("({DATA, render, buildKvTable})", context);
let runs = 0, failures = 0;
for (const modelKey of Object.keys(DATA.models)) {
  const model = DATA.models[modelKey];
  for (const specKey of Object.keys(model.spec)) {
    for (const kv of Object.keys(model.kv)) {
      for (const ctx of [256, 4096, 8192, 8193, 40960, 262144]) {
        el("model").value = modelKey;
        el("spec").value = specKey;
        el("kv").value = kv;
        el("ctx").value = String(ctx);
        el("reserve").value = "307";
        try { render(); buildKvTable(); runs++; }
        catch (e) {
          failures++;
          if (failures <= 3) console.log(`FAIL ${modelKey}/${specKey}/${kv}/${ctx}: ${e.message}`);
        }
      }
    }
  }
}
console.log(`render()+buildKvTable() over ${runs + failures} combinations: ${failures} threw`);

// The largest-fitting context must respond to the speculation mode, or the page is silently
// promising context a speculative configuration cannot deliver.
el("model").value = "35b"; el("kv").value = "int8"; el("ctx").value = "32768"; el("reserve").value = "307";
for (const specKey of ["none", "mtp3+head", "dflash-3"]) {
  el("spec").value = specKey; render();
  const mx = el("out-maxctx").firstChild.nodeValue;
  console.log(`  35b/${specKey.padEnd(10)} largest ctx=${mx}  |  ${el("out-maxctx-sub").textContent}`);
}
el("spec").value = "none"; render();
const noneCtx = el("out-maxctx").firstChild.nodeValue;
el("spec").value = "dflash-3"; render();
const dflashCtx = el("out-maxctx").firstChild.nodeValue;
if (noneCtx === dflashCtx) {
  console.log(`FAIL largest context did not move with the speculation mode (${noneCtx} both ways)`);
  failures++;
}
console.log(failures ? `FAIL (${failures})` : "PASS");
process.exit(failures ? 1 : 0);
