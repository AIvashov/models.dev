import { expect, test } from "bun:test";

import { patchModelV2 } from "../src/model-file-v2.js";

test("patchModelV2 keeps v2 reasoning data when sync inherits reasoning from the lab", () => {
  const doc = {
    schema: 2,
    base_model: "xai/grok-4.7",
    capabilities: {
      reasoning: { supported: true, default: { type: "effort", value: "high" }, effort: ["low", "high"] },
      cache: { supported: true, implicit: true, explicit: false },
    },
    cost: {
      text: { input: { price: 9 }, output: { price: 6 } },
      multipliers: [{ when: { region: { equals: "us" } }, factor: 1.1 }],
    },
  };
  const reasoning_options = [{ type: "effort", values: ["low", "high"] }];
  const before = { base_model: "xai/grok-4.7", reasoning: true, reasoning_options, cost: { input: 9, output: 6 } };
  const after = { base_model: "xai/grok-4.7", reasoning_options, cost: { input: 2, output: 6 } };

  const next = patchModelV2(doc, before, after);

  expect(next.capabilities).toEqual(doc.capabilities);
  expect(next.cost).toEqual({
    text: { input: { price: 2 }, output: { price: 6 } },
    multipliers: [{ when: { region: { equals: "us" } }, factor: 1.1 }],
  });
});
