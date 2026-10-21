import { describe, expect, test } from "bun:test";
import path from "node:path";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";

import { generate, generateCatalog, generateV2 } from "../src/index.js";

async function withFixture<T>(callback: (root: string) => Promise<T>) {
  const root = await mkdtemp(path.join(tmpdir(), "models-dev-test-"));
  try {
    return await callback(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

async function write(root: string, file: string, content: string) {
  const filePath = path.join(root, file);
  await mkdir(path.dirname(filePath), { recursive: true });
  await Bun.write(filePath, content);
}

function stable(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stable).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}:${stable(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

describe("catalog generation", () => {
  test("rejects providers with no models", async () => {
    await withFixture(async (root) => {
      await write(root, "providers/empty/provider.toml", providerToml("Empty"));

      expect(generate(path.join(root, "providers"))).rejects.toThrow(
        'Provider "empty" has no models',
      );
    });
  });

  test("base_model publishes canonical identity with the provider model", async () => {
    await withFixture(async (root) => {
      await write(root, "providers/direct/provider.toml", providerToml("Direct"));
      await write(root, "providers/factored/provider.toml", providerToml("Factored"));
      await write(root, "models/lab/model.toml", modelMetadataToml());
      await write(
        root,
        "providers/direct/models/model.toml",
        `${providerFieldsToml()}

[cost]
input = 1.25
output = 2.50
cache_read = 0.125
`,
      );
      await write(
        root,
        "providers/factored/models/model.toml",
        `base_model = "lab/model"
reasoning_options = []

[cost]
input = 1.25
output = 2.50
cache_read = 0.125
`,
      );

      const catalog = await generateCatalog(root);

      expect(catalog.models["lab/model"]?.benchmarks).toEqual([
        {
          name: "SWE-Bench Verified",
          score: 71.2,
          metric: "resolved",
          harness: "Example Harness",
          variant: "high",
          dataset: "verified",
          version: "1",
          source: "https://example.com/benchmarks",
        },
      ]);
      expect(catalog.models["lab/model"]?.weights).toEqual([
        {
          label: "Weights",
          url: "https://huggingface.co/lab/model",
          format: "safetensors",
        },
      ]);

      expect(catalog.providers.factored?.models.model).toEqual({
        ...catalog.providers.direct?.models.model,
        canonical_model_id: "lab/model",
      });
      expect(catalog.providers.factored?.models.model).not.toHaveProperty(
        "base_model",
      );
      expect(catalog.providers.factored?.models.model).not.toHaveProperty(
        "benchmarks",
      );
    });
  });

  test("base_model_omit removes inherited metadata fields", async () => {
    await withFixture(async (root) => {
      await write(root, "providers/provider/provider.toml", providerToml("Provider"));
      await write(root, "models/lab/model.toml", modelMetadataToml());
      await write(
        root,
        "providers/provider/models/model.toml",
        `base_model = "lab/model"
base_model_omit = ["limit.input", "structured_output"]
reasoning_options = []

[cost]
input = 1.25
output = 2.50

[limit]
context = 200_000
output = 32_000
`,
      );

      const providers = await generate(path.join(root, "providers"));
      const model = providers.provider?.models.model;

      expect(model?.structured_output).toBeUndefined();
      expect(model?.limit).toEqual({
        context: 200_000,
        output: 32_000,
      });
    });
  });

  test("base_model can inherit sibling fields from partial object overrides", async () => {
    await withFixture(async (root) => {
      await write(root, "providers/provider/provider.toml", providerToml("Provider"));
      await write(root, "models/lab/model.toml", modelMetadataToml());
      await write(
        root,
        "providers/provider/models/model.toml",
        `base_model = "lab/model"
open_weights = true
reasoning_options = []

[cost]
input = 1.25
output = 2.50

[limit]
context = 200_000

[modalities]
input = ["text"]
`,
      );

      const providers = await generate(path.join(root, "providers"));
      const model = providers.provider?.models.model;

      expect(model?.open_weights).toBe(true);
      expect(model?.limit).toEqual({
        context: 200_000,
        input: 272_000,
        output: 128_000,
      });
      expect(model?.modalities).toEqual({
        input: ["text"],
        output: ["text"],
      });
    });
  });

  test("v1_alias providers serve the target's models with renamed env vars", async () => {
    await withFixture(async (root) => {
      await write(root, "providers/target/provider.toml", providerToml("Target"));
      await write(
        root,
        "providers/target/models/model.toml",
        `${providerFieldsToml()}
[cost]
input = 1
output = 2

[provider]
api = "https://\${TARGET_RESOURCE}.example.com/v1"
`,
      );
      await write(
        root,
        "providers/legacy/provider.toml",
        `${providerToml("Legacy")}
[v1_alias]
provider = "target"
env = { TARGET_RESOURCE = "LEGACY_RESOURCE" }
`,
      );

      const providers = await generate(path.join(root, "providers"));
      expect(Object.keys(providers.legacy!)).not.toContain("v1_alias");
      expect(providers.legacy?.name).toBe("Legacy");
      expect(providers.legacy?.models.model?.provider?.api).toBe(
        "https://${LEGACY_RESOURCE}.example.com/v1",
      );
      expect(providers.target?.models.model?.provider?.api).toBe(
        "https://${TARGET_RESOURCE}.example.com/v1",
      );
    });
  });

  test("v1_alias renames the target's provider-level api and rejects model files", async () => {
    await withFixture(async (root) => {
      await write(
        root,
        "providers/target/provider.toml",
        `name = "Target"
npm = "@ai-sdk/openai"
env = ["TARGET_RESOURCE", "TARGET_KEY"]
api = "https://\${TARGET_RESOURCE}.example.com/v1"
doc = "https://example.com/models"
`,
      );
      await write(root, "providers/target/models/plain.toml", `${providerFieldsToml()}
[cost]
input = 1
output = 2
`);
      const alias = `name = "Legacy"
npm = "@ai-sdk/openai"
env = ["LEGACY_RESOURCE", "LEGACY_KEY"]
doc = "https://example.com/models"

[v1_alias]
provider = "target"
env = { TARGET_RESOURCE = "LEGACY_RESOURCE", TARGET_KEY = "LEGACY_KEY" }
`;
      await write(root, "providers/legacy/provider.toml", alias);

      const providers = await generate(path.join(root, "providers"));
      expect(providers.legacy?.api).toBe("https://${LEGACY_RESOURCE}.example.com/v1");
      expect(providers.legacy?.models.plain?.provider).toBeUndefined();
      expect(JSON.stringify(providers.legacy)).not.toContain("TARGET_");

      await write(root, "providers/legacy/models/plain.toml", `${providerFieldsToml()}`);
      expect(generate(path.join(root, "providers"))).rejects.toThrow("can't have a models directory");
    });
  });

  test("repository provider TOMLs do not use legacy extends tables", async () => {
    const root = path.join(import.meta.dirname, "..", "..", "..");
    const matches: string[] = [];

    for await (const file of new Bun.Glob("providers/**/*.toml").scan({
      cwd: root,
    })) {
      const text = await Bun.file(path.join(root, file)).text();
      if (/^\[extends\]/m.test(text)) matches.push(file);
    }

    expect(matches).toEqual([]);
  });

  test("repository provider JSON strips authored metadata pointers", async () => {
    const root = path.join(import.meta.dirname, "..", "..", "..");
    const providers = await generate(path.join(root, "providers"));
    const leaked: string[] = [];

    for (const [providerID, provider] of Object.entries(providers)) {
      for (const [modelID, model] of Object.entries(provider.models)) {
        const encoded = stable(model);
        if (encoded.includes("base_model") || encoded.includes("base_model_omit")) {
          leaked.push(`${providerID}/${modelID}`);
        }
      }
    }

    expect(leaked).toEqual([]);
  });

  test("repository provider JSON excludes model-only metadata", async () => {
    const root = path.join(import.meta.dirname, "..", "..", "..");
    const providers = await generate(path.join(root, "providers"));
    const modelOnlyFields = ["benchmarks", "license", "links", "weights"];
    const leaked: string[] = [];

    for (const [providerID, provider] of Object.entries(providers)) {
      for (const [modelID, model] of Object.entries(provider.models)) {
        const leakedFields = modelOnlyFields.filter((field) => field in model);
        if (leakedFields.length > 0) {
          leaked.push(`${providerID}/${modelID}: ${leakedFields.join(", ")}`);
        }
      }
    }

    expect(leaked).toEqual([]);
  });

  test("repository model metadata avoids provider-only namespaces", async () => {
    const root = path.join(import.meta.dirname, "..", "..", "..");
    const providerNamespaces = [
      "amazon-bedrock",
      "llama",
      "opencode",
      "tencent-tokenhub",
      "zai",
    ];
    const namespaceDirs = providerNamespaces.filter((namespace) =>
      existsSync(path.join(root, "models", namespace))
    );
    const baseModelRefs: string[] = [];

    for await (const file of new Bun.Glob("providers/**/*.toml").scan({
      cwd: root,
    })) {
      const text = await Bun.file(path.join(root, file)).text();
      const match = /^base_model = "([^/"]+)\//m.exec(text);
      if (match?.[1] !== undefined && providerNamespaces.includes(match[1])) {
        baseModelRefs.push(file);
      }
    }

    expect(namespaceDirs).toEqual([]);
    expect(baseModelRefs).toEqual([]);
  });

  test("repository open-weight model metadata includes weights links", async () => {
    const root = path.join(import.meta.dirname, "..", "..", "..");
    const catalog = await generateCatalog(root);
    const missingWeights: string[] = [];
    const closedWithWeights: string[] = [];

    for (const [modelID, model] of Object.entries(catalog.models)) {
      const hasWeights = (model.weights?.length ?? 0) > 0;
      if (model.open_weights === true && !hasWeights) {
        missingWeights.push(modelID);
      }
      if (model.open_weights !== true && hasWeights) {
        closedWithWeights.push(modelID);
      }
    }

    expect(missingWeights).toEqual([]);
    expect(closedWithWeights).toEqual([]);
  });

  test("repository benchmark metadata is sourced", async () => {
    const root = path.join(import.meta.dirname, "..", "..", "..");
    const catalog = await generateCatalog(root);
    const unsourced: string[] = [];

    for (const [modelID, model] of Object.entries(catalog.models)) {
      for (const benchmark of model.benchmarks ?? []) {
        if (benchmark.source === undefined) {
          unsourced.push(`${modelID}: ${benchmark.name}`);
        }
      }
    }

    expect(unsourced).toEqual([]);
  });

  test("repository benchmark names are normalized", async () => {
    const root = path.join(import.meta.dirname, "..", "..", "..");
    const catalog = await generateCatalog(root);
    const qualifiedNames: string[] = [];

    for (const [modelID, model] of Object.entries(catalog.models)) {
      for (const benchmark of model.benchmarks ?? []) {
        if (/\s\([^)]+\)$/.test(benchmark.name)) {
          qualifiedNames.push(`${modelID}: ${benchmark.name}`);
        }
      }
    }

    expect(qualifiedNames).toEqual([]);
  });

  test("legacy_shape is published in v1 and never in v2", async () => {
    await withFixture(async (root) => {
      await write(root, "providers/provider/provider.toml", providerToml("Provider"));
      await write(
        root,
        "providers/provider/models/model.toml",
        `schema = 2
name = "Lab Model"
description = "Example model for catalog generation and inheritance tests"
open_weights = true
release_date = "2026-01-02"

[modalities]
input = ["text"]
output = ["text"]

[capabilities]
tools = { supported = true }
reasoning = { supported = false }

[limit]
context = 400_000
output = 128_000

[api.responses]
base_url = "https://example.com/v1"
sdk = { aisdk = "@ai-sdk/openai" }
legacy_shape = "responses"

[cost.text.input]
price = 1

[cost.text.output]
price = 2
`,
      );

      const providers = await generate(path.join(root, "providers"));
      expect(providers.provider?.models.model?.provider).toEqual({
        npm: "@ai-sdk/openai",
        api: "https://example.com/v1",
        shape: "responses",
      });

      const providersV2 = await generateV2(path.join(root, "providers"));
      expect(providersV2.provider?.models.model?.api).toEqual({
        responses: { base_url: "https://example.com/v1", sdk: { aisdk: "@ai-sdk/openai" , opencodeai: "@opencode/ai/providers/openai/responses" } },
      });
      expect(JSON.stringify(providersV2)).not.toContain("legacy_shape");
    });
  });
});

describe("provider [api.*] tables", () => {
  const apisProvider = `name = "Gateway"
env = ["API_KEY"]
doc = "https://example.com/models"

[api.chat-completions]
base_url = "https://example.com/v1"
sdk = { aisdk = "@ai-sdk/openai-compatible", opencodeai = "@opencode/ai/providers/openai-compatible" }

[api.responses]
base_url = "https://example.com/v1"
sdk = { aisdk = "@ai-sdk/openai", opencodeai = "@opencode/ai/providers/openai/responses" }

[api.messages]
base_url = "https://example.com/anthropic/v1"
sdk = { aisdk = "@ai-sdk/anthropic", opencodeai = "@opencode/ai/providers/anthropic-compatible" }

[api.images]
base_url = "https://example.com/v1"
sdk = { aisdk = "@example/ai-sdk-provider" }
opt_in = true
`;

  const v2Model = (api = "") => `schema = 2
name = "Lab Model"
description = "Example model for catalog generation and inheritance tests"
open_weights = true
release_date = "2026-01-02"

[modalities]
input = ["text"]
output = ["text"]

[capabilities]
tools = { supported = true }
reasoning = { supported = false }

[limit]
context = 400_000
output = 128_000
${api}
[cost.text.input]
price = 1

[cost.text.output]
price = 2
`;

  test("models get every API except opt-in ones, in order, and v1 gets the first", async () => {
    await withFixture(async (root) => {
      await write(root, "providers/gateway/provider.toml", apisProvider);
      await write(root, "providers/gateway/models/model.toml", v2Model());

      const providers = await generate(path.join(root, "providers"));
      expect(providers.gateway?.npm).toBe("@ai-sdk/openai-compatible");
      expect(providers.gateway?.api).toBe("https://example.com/v1");
      expect(providers.gateway?.models.model?.provider).toBeUndefined();

      const api = (await generateV2(path.join(root, "providers"))).gateway?.models.model?.api;
      expect(Object.keys(api ?? {})).toEqual(["chat-completions", "responses", "messages"]);
      expect(api?.messages).toEqual({
        base_url: "https://example.com/anthropic/v1",
        sdk: { aisdk: "@ai-sdk/anthropic", opencodeai: "@opencode/ai/providers/anthropic-compatible" },
      });
    });
  });

  test("v1_api = false keeps v1 without a provider api", async () => {
    await withFixture(async (root) => {
      await write(
        root,
        "providers/gateway/provider.toml",
        apisProvider.replace(
          'sdk = { aisdk = "@ai-sdk/openai-compatible", opencodeai = "@opencode/ai/providers/openai-compatible" }\n',
          'sdk = { aisdk = "@example/gateway" }\nv1_api = false\n',
        ),
      );
      await write(root, "providers/gateway/models/model.toml", v2Model());

      const providers = await generate(path.join(root, "providers"));
      expect(providers.gateway?.npm).toBe("@example/gateway");
      expect(providers.gateway).not.toHaveProperty("api");
      const api = (await generateV2(path.join(root, "providers"))).gateway?.models.model?.api;
      expect(api?.["chat-completions"]?.base_url).toBe("https://example.com/v1");
      expect(JSON.stringify(api)).not.toContain("v1_api");
    });
  });

  test("a model's own tables replace the list and take missing fields from the provider", async () => {
    await withFixture(async (root) => {
      await write(root, "providers/gateway/provider.toml", apisProvider);
      // Restricts the list to one provider API, with a reasoning field on it.
      await write(
        root,
        "providers/gateway/models/chat-only.toml",
        v2Model(`
[api.chat-completions]
compatibility = { reasoning_field = "reasoning_content" }
`),
      );
      // Uses an opt-in API, and one the provider doesn't list.
      await write(
        root,
        "providers/gateway/models/image.toml",
        v2Model(`
[api.images]

[api.videos]
base_url = "https://example.com/video/v1"
sdk = { aisdk = "@example/video" }
`),
      );
      // Swaps the AI SDK package and keeps the provider's OpenCode package.
      await write(
        root,
        "providers/gateway/models/custom.toml",
        v2Model(`
[api.responses]
sdk = { aisdk = "@example/responses" }
`),
      );

      const models = (await generateV2(path.join(root, "providers"))).gateway?.models;
      expect(models?.["chat-only"]?.api).toEqual({
        "chat-completions": {
          base_url: "https://example.com/v1",
          sdk: { aisdk: "@ai-sdk/openai-compatible", opencodeai: "@opencode/ai/providers/openai-compatible" },
          compatibility: { reasoning_field: "reasoning_content" },
        },
      });
      expect(models?.image?.api).toEqual({
        images: { base_url: "https://example.com/v1", sdk: { aisdk: "@example/ai-sdk-provider" } },
        videos: { base_url: "https://example.com/video/v1", sdk: { aisdk: "@example/video" } },
      });
      expect(models?.custom?.api).toEqual({
        responses: {
          base_url: "https://example.com/v1",
          sdk: { aisdk: "@example/responses", opencodeai: "@opencode/ai/providers/openai/responses" },
        },
      });

      // v1 keeps reading only the first table's own fields.
      const providers = await generate(path.join(root, "providers"));
      expect(providers.gateway?.models["chat-only"]?.interleaved).toEqual({ field: "reasoning_content" });
      expect(providers.gateway?.models.image?.provider).toBeUndefined();
    });
  });

  test("a model table without an SDK takes the provider's for its v1 shape", async () => {
    await withFixture(async (root) => {
      await write(
        root,
        "providers/openai/provider.toml",
        `name = "OpenAI"
env = ["API_KEY"]
doc = "https://example.com/models"

[api.responses]
base_url = "https://example.com/v1"
sdk = { aisdk = "@ai-sdk/openai" }
v1_api = false

[api.chat-completions]
base_url = "https://example.com/v1"
sdk = { aisdk = "@ai-sdk/openai" }
`,
      );
      await write(root, "providers/openai/models/responses-only.toml", v2Model("\n[api.responses]\n"));
      await write(root, "providers/openai/models/chat-only.toml", v2Model("\n[api.chat-completions]\n"));

      const models = (await generate(path.join(root, "providers"))).openai?.models;
      // Responses is @ai-sdk/openai's default, so it is no v1 shape; Chat is.
      expect(models?.["responses-only"]?.provider).toBeUndefined();
      expect(models?.["chat-only"]?.provider).toEqual({ shape: "completions" });
    });
  });

  test("rejects tables that contradict v1 or miss a base URL", async () => {
    const cases: Array<[string, string, string | RegExp]> = [
      [apisProvider.replace('env = ["API_KEY"]', 'env = ["API_KEY"]\nnpm = "@ai-sdk/openai"'), v2Model(), "can't set npm"],
      [apisProvider.replace("[api.chat-completions]", "[api.chat-completions]\nopt_in = true"), v2Model(), /can't be opt_in/],
      [apisProvider, v2Model("\n[api.rerank]\n"), "[api.rerank] needs a base_url"],
    ];
    for (const [provider, model, message] of cases) {
      await withFixture(async (root) => {
        await write(root, "providers/gateway/provider.toml", provider);
        await write(root, "providers/gateway/models/model.toml", model);
        await expect(generateV2(path.join(root, "providers"))).rejects.toThrow(message);
      });
    }
  });

  test("without provider tables, v2 still derives the one API and rejects model opencodeai", async () => {
    await withFixture(async (root) => {
      await write(root, "providers/provider/provider.toml", providerToml("Provider"));
      await write(
        root,
        "providers/provider/models/model.toml",
        v2Model(`
[api.chat-completions]
compatibility = { reasoning_field = "reasoning_content" }
`),
      );
      expect((await generateV2(path.join(root, "providers"))).provider?.models.model?.api).toEqual({
        responses: {
          base_url: "https://api.openai.com/v1",
          sdk: { aisdk: "@ai-sdk/openai", opencodeai: "@opencode/ai/providers/openai/responses" },
          compatibility: { reasoning_field: "reasoning_content" },
        },
      });

    });
    await withFixture(async (root) => {
      await write(root, "providers/provider/provider.toml", providerToml("Provider"));
      await write(
        root,
        "providers/provider/models/model.toml",
        v2Model(`
[api.responses]
sdk = { opencodeai = "@opencode/ai/providers/openai/responses" }
`),
      );
      await expect(generateV2(path.join(root, "providers"))).rejects.toThrow(
        "sdk.opencodeai in a model file needs the provider's [api.*] tables",
      );
    });
  });
});

function providerToml(name: string) {
  return `name = "${name}"
npm = "@ai-sdk/openai"
env = ["API_KEY"]
doc = "https://example.com/models"
`;
}

function modelMetadataToml() {
  return `name = "Lab Model"
description = "Example model for catalog generation and inheritance tests"
family = "gpt"
release_date = "2026-01-02"
last_updated = "2026-01-03"
attachment = true
reasoning = true
temperature = false
tool_call = true
structured_output = true
knowledge = "2025-12"
open_weights = true
license = "Example License"

[limit]
context = 400_000
input = 272_000
output = 128_000

[modalities]
input = ["text", "image"]
output = ["text"]

[[links]]
label = "Model card"
url = "https://example.com/model"
type = "model_card"

[[weights]]
label = "Weights"
url = "https://huggingface.co/lab/model"
format = "safetensors"

[[benchmarks]]
name = "SWE-Bench Verified"
score = 71.2
metric = "resolved"
harness = "Example Harness"
variant = "high"
dataset = "verified"
version = "1"
source = "https://example.com/benchmarks"
`;
}

function providerFieldsToml() {
  return `name = "Lab Model"
description = "Example model for catalog generation and inheritance tests"
family = "gpt"
release_date = "2026-01-02"
last_updated = "2026-01-03"
attachment = true
reasoning = true
reasoning_options = []
temperature = false
tool_call = true
structured_output = true
knowledge = "2025-12"
open_weights = true

[limit]
context = 400_000
input = 272_000
output = 128_000

[modalities]
input = ["text", "image"]
output = ["text"]
`;
}
