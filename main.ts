// free-model-aggregator-02: one server, two wires, live free OpenRouter models.
// Reads the brief at $MAKEFUN_PROMPT_FILE, serves OpenAI Chat Completions and
// Anthropic Messages, dials every model on the gateway's Chat Completions route
// ($OPENAI_BASE_URL/chat/completions), routes each prompt to the highest-rated
// free model that meets its requirements, and fails over in rating order.
// Deterministic: fixed timestamps/ids, no wall clock in any response, no
// randomness. Only timers are used for timeouts and the /work poll loop.
import http from "node:http";
import fs from "node:fs";

const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = "2026-01-01T00:00:00Z";
const FIXED_CHAT_ID = "chatcmpl-aggregator02";
const FIXED_MSG_ID = "msg_aggregator02_000000000000000000000000";
const GATEWAY_TIMEOUT_FIRST_MS = 20000;
const PROMPT_DEADLINE_MS = 28500;
const MAX_ATTEMPTS = 3;

type ModelRow = {
  id: string;
  display_name: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
};

type PromptSpec = {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: any;
};

type Failure = { status: number; code: string; model: string | null };

let MODELS: ModelRow[] = [];
let MANIFEST: PromptSpec[] = [];
let manifestUsed: boolean[] = [];

function parseBrief(): void {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) return;
  let text: string;
  try {
    text = fs.readFileSync(file, "utf8");
  } catch {
    return;
  }
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t.startsWith("{")) continue;
    let doc: any = null;
    try {
      doc = JSON.parse(t);
    } catch {
      continue;
    }
    if (!doc || typeof doc.schema !== "string") continue;
    if (doc.schema === "free-model-aggregator-02/models/v1" && Array.isArray(doc.models)) {
      MODELS = doc.models
        .filter((m: any) => m && m.free === true && typeof m.id === "string")
        .map((m: any) => ({
          id: m.id,
          display_name: typeof m.display_name === "string" ? m.display_name : m.id,
          rating_milli: typeof m.rating_milli === "number" ? m.rating_milli : 0,
          context_length: typeof m.context_length === "number" ? m.context_length : 0,
          tools: m.tools === true,
          vision: m.vision === true,
          json: m.json === true,
          free: true,
        }));
    }
    if (doc.schema === "free-model-aggregator-02/prompts/v1" && Array.isArray(doc.prompts)) {
      MANIFEST = doc.prompts
        .filter((p: any) => p && typeof p.id === "string")
        .map((p: any) => ({
          id: p.id,
          api: typeof p.api === "string" ? p.api : "openai",
          stream: p.stream === true,
          weight: typeof p.weight === "number" ? p.weight : 1,
          requires: p.requires && typeof p.requires === "object" ? p.requires : {},
        }));
    }
  }
  manifestUsed = MANIFEST.map(() => false);
}

function catalogObject(): any {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: MODELS.map((m) => ({ id: m.id, free: true, context_length: m.context_length })),
      },
    ],
  };
}

function emptyCatalog(): any {
  return { providers: [] };
}

function meetsRequirements(m: ModelRow, req: any): boolean {
  if (!req) return true;
  if (req.json === true && !m.json) return false;
  if (req.tools === true && !m.tools) return false;
  if (req.vision === true && !m.vision) return false;
  if (typeof req.min_context === "number" && m.context_length < req.min_context) return false;
  return true;
}

// Ranked candidates for a prompt: rating desc, id asc as the deterministic tie-break.
function rankedModels(req: any): ModelRow[] {
  return MODELS.filter((m) => meetsRequirements(m, req)).sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
}

function inferRequirements(api: string, body: any): any {
  const req: any = {};
  if (api === "openai") {
    if (body && body.response_format && typeof body.response_format === "object") req.json = true;
    if (body && Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
    if (body && Array.isArray(body.messages)) {
      for (const msg of body.messages) {
        if (msg && Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part && part.type === "image_url") req.vision = true;
          }
        }
      }
    }
  } else {
    if (body && Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
    if (body && Array.isArray(body.messages)) {
      for (const msg of body.messages) {
        if (msg && Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part && part.type === "image") req.vision = true;
          }
        }
      }
    }
  }
  return req;
}

// Take the next unconsumed manifest entry matching this request's wire/stream;
// fall back to the next unconsumed entry, then to inference from the body.
function requirementsFor(api: string, stream: boolean, body: any): any {
  let idx = -1;
  for (let i = 0; i < MANIFEST.length; i++) {
    if (!manifestUsed[i] && MANIFEST[i].api === api && MANIFEST[i].stream === stream) {
      idx = i;
      break;
    }
  }
  if (idx === -1) {
    for (let i = 0; i < MANIFEST.length; i++) {
      if (!manifestUsed[i]) {
        idx = i;
        break;
      }
    }
  }
  if (idx !== -1) {
    manifestUsed[idx] = true;
    return MANIFEST[idx].requires;
  }
  return inferRequirements(api, body);
}

function clampMaxTokens(v: any): number {
  if (typeof v !== "number" || !Number.isFinite(v) || v <= 0) return 1024;
  if (v > 8192) return 8192;
  return Math.floor(v);
}

function anthropicSystemText(system: any): string {
  if (typeof system === "string") return system;
  if (Array.isArray(system)) {
    const parts: string[] = [];
    for (const b of system) if (b && b.type === "text" && typeof b.text === "string") parts.push(b.text);
    return parts.join("\n");
  }
  return "";
}

function anthropicContentToText(content: any): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const b of content) {
      if (b && b.type === "text" && typeof b.text === "string") parts.push(b.text);
    }
    return parts.join("");
  }
  return "";
}

function anthropicMessagesToOpenAI(system: any, messages: any): any[] {
  const out: any[] = [];
  const sys = anthropicSystemText(system);
  if (sys) out.push({ role: "system", content: sys });
  if (Array.isArray(messages)) {
    for (const m of messages) {
      if (!m || typeof m.role !== "string") continue;
      const role = m.role === "assistant" ? "assistant" : "user";
      const content = m.content;
      if (typeof content === "string") {
        out.push({ role, content });
      } else if (Array.isArray(content)) {
        const parts: any[] = [];
        for (const b of content) {
          if (!b || typeof b.type !== "string") continue;
          if (b.type === "text" && typeof b.text === "string") parts.push({ type: "text", text: b.text });
          else if (b.type === "image" && b.source && b.source.type === "base64") {
            parts.push({
              type: "image_url",
              image_url: { url: `data:${b.source.media_type || "image/png"};base64,${b.source.data || ""}` },
            });
          }
        }
        out.push({ role, content: parts });
      }
    }
  }
  return out;
}

function buildGatewayBody(api: string, body: any, model: string, stream: boolean): any {
  const maxTokens = clampMaxTokens(api === "openai" ? body.max_tokens ?? body.max_completion_tokens : body.max_tokens);
  const out: any = { model, max_tokens: maxTokens, stream };
  if (api === "openai") {
    out.messages = Array.isArray(body.messages) ? body.messages : [];
    if (body.temperature !== undefined) out.temperature = body.temperature;
    if (body.top_p !== undefined) out.top_p = body.top_p;
    if (body.stop !== undefined) out.stop = body.stop;
    if (body.response_format !== undefined) out.response_format = body.response_format;
    if (Array.isArray(body.tools) && body.tools.length > 0) {
      out.tools = body.tools;
      if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
    }
    if (stream) out.stream_options = { include_usage: true };
  } else {
    out.messages = anthropicMessagesToOpenAI(body.system, body.messages);
    if (body.temperature !== undefined) out.temperature = body.temperature;
    if (body.top_p !== undefined) out.top_p = body.top_p;
    if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
    if (Array.isArray(body.tools) && body.tools.length > 0) {
      out.tools = body.tools.map((t: any) => ({
        type: "function",
        function: {
          name: t.name,
          description: typeof t.description === "string" ? t.description : "",
          parameters: t.input_schema && typeof t.input_schema === "object" ? t.input_schema : { type: "object", properties: {} },
        },
      }));
      if (body.tool_choice !== undefined) {
        const tc = body.tool_choice;
        if (tc && tc.type === "tool" && typeof tc.name === "string") {
          out.tool_choice = { type: "function", function: { name: tc.name } };
        } else if (tc && tc.type === "auto") out.tool_choice = "auto";
        else if (tc && tc.type === "none") out.tool_choice = "none";
      }
    }
    if (stream) out.stream_options = { include_usage: true };
  }
  return out;
}

function errorCodeOf(bodyText: string, status: number): string {
  try {
    const v = JSON.parse(bodyText);
    if (v && typeof v === "object") {
      const e = (v as any).error;
      if (e && typeof e === "object") {
        if (typeof e.code === "string" && e.code) return e.code.slice(0, 64);
        if (typeof e.type === "string" && e.type) return e.type.slice(0, 64);
      }
      if (typeof (v as any).code === "string" && (v as any).code) return (v as any).code.slice(0, 64);
      if (typeof (v as any).type === "string" && (v as any).type) return (v as any).type.slice(0, 64);
    }
  } catch {
    // not JSON
  }
  return `http_${status}`;
}

function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

type GatewayResult =
  | { ok: true; status: number; json: any; text: string }
  | { ok: false; status: number; code: string; bodyText: string; network: boolean };

async function callGateway(model: string, gwBody: any, timeoutMs: number): Promise<GatewayResult> {
  const base = process.env.OPENAI_BASE_URL || "";
  const key = process.env.OPENAI_API_KEY || "";
  const url = base.replace(/\/$/, "") + "/chat/completions";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
      body: JSON.stringify(gwBody),
      signal: controller.signal,
    });
    const text = await res.text();
    if (res.status >= 200 && res.status < 300) {
      let json: any = null;
      if (!gwBody.stream) {
        try {
          json = JSON.parse(text);
        } catch {
          json = null;
        }
      }
      return { ok: true, status: res.status, json, text };
    }
    return { ok: false, status: res.status, code: errorCodeOf(text, res.status), bodyText: text, network: false };
  } catch (err: any) {
    const timedOut = err && err.name === "AbortError";
    return { ok: false, status: timedOut ? 504 : 502, code: timedOut ? "timeout" : "upstream_error", bodyText: "", network: true };
  } finally {
    clearTimeout(timer);
  }
}

function parseSse(text: string): any[] {
  const frames: any[] = [];
  const normalized = text.replace(/\r\n/g, "\n");
  for (const frame of normalized.split("\n\n")) {
    if (frame.trim() === "") continue;
    const data: string[] = [];
    let event: string | null = null;
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function mapFinishToAnthropic(fr: string | null): string {
  if (fr === "length") return "max_tokens";
  if (fr === "tool_calls") return "tool_use";
  if (fr === "content_filter") return "refusal";
  return "end_turn";
}

function openAIUsageOf(providerJson: any, text: string, promptText: string): any {
  const u = providerJson && providerJson.usage && typeof providerJson.usage === "object" ? providerJson.usage : null;
  const promptTokens = u && typeof u.prompt_tokens === "number" ? u.prompt_tokens : Math.max(0, Math.ceil(promptText.length / 4));
  const completionTokens =
    u && typeof u.completion_tokens === "number" && u.completion_tokens > 0
      ? u.completion_tokens
      : estimateTokens(text);
  return { prompt_tokens: promptTokens, completion_tokens: completionTokens };
}

function providerTextOf(providerJson: any): string {
  try {
    const choice = providerJson && Array.isArray(providerJson.choices) ? providerJson.choices[0] : null;
    const msg = choice && choice.message ? choice.message : null;
    if (!msg) return "";
    if (typeof msg.content === "string") return msg.content;
    if (Array.isArray(msg.content)) {
      const parts: string[] = [];
      for (const p of msg.content) if (p && p.type === "text" && typeof p.text === "string") parts.push(p.text);
      return parts.join("");
    }
    return "";
  } catch {
    return "";
  }
}

function providerToolCallsOf(providerJson: any): any[] {
  try {
    const choice = providerJson && Array.isArray(providerJson.choices) ? providerJson.choices[0] : null;
    const msg = choice && choice.message ? choice.message : null;
    if (msg && Array.isArray(msg.tool_calls)) return msg.tool_calls;
    return [];
  } catch {
    return [];
  }
}

function providerFinishOf(providerJson: any): string {
  try {
    const choice = providerJson && Array.isArray(providerJson.choices) ? providerJson.choices[0] : null;
    if (choice && typeof choice.finish_reason === "string") return choice.finish_reason;
    return "stop";
  } catch {
    return "stop";
  }
}

function promptTextForUsage(api: string, body: any): string {
  const parts: string[] = [];
  if (api === "openai") {
    if (Array.isArray(body.messages)) {
      for (const m of body.messages) {
        if (typeof m.content === "string") parts.push(m.content);
        else if (Array.isArray(m.content)) {
          for (const p of m.content) if (p && typeof p.text === "string") parts.push(p.text);
        }
      }
    }
  } else {
    parts.push(anthropicSystemText(body.system));
    if (Array.isArray(body.messages)) {
      for (const m of body.messages) parts.push(anthropicContentToText(m.content));
    }
  }
  return parts.join("\n");
}

// Dial the ranked models in order until one answers. Returns the success or
// the last non-retryable refusal. Retries only 429 / 5xx / network errors.
async function dialRanked(
  ranked: ModelRow[],
  api: string,
  body: any,
  stream: boolean
): Promise<{ model: string; result: GatewayResult; failures: Failure[] } | { model: null; result: GatewayResult; failures: Failure[] }> {
  const failures: Failure[] = [];
  const started = Date.now();
  // Synthetic fallback so the caller never sees a null result (e.g. if the
  // per-prompt deadline is already spent before the first attempt).
  let last: GatewayResult = { ok: false, status: 504, code: "timeout", bodyText: "", network: true };
  const attempts = Math.min(MAX_ATTEMPTS, ranked.length);
  for (let i = 0; i < attempts; i++) {
    const model = ranked[i];
    const remaining = PROMPT_DEADLINE_MS - (Date.now() - started);
    if (remaining <= 1000) break;
    const timeoutMs = Math.min(GATEWAY_TIMEOUT_FIRST_MS, remaining);
    const gwBody = buildGatewayBody(api, body, model.id, stream);
    const result = await callGateway(model.id, gwBody, timeoutMs);
    if (result.ok) return { model: model.id, result, failures };
    last = result;
    const retryable = result.status === 429 || result.status >= 500;
    if (!retryable) return { model: null, result, failures };
    failures.push({ status: result.status, code: result.code, model: model.id });
  }
  return { model: null, result: last, failures };
}

function failuresHeader(failures: Failure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { "x-makefun-upstream-failures": JSON.stringify(failures.slice(0, 8)) };
}

function sendJson(res: any, status: number, obj: any, extraHeaders: Record<string, string>): void {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "content-type": "application/json", "content-length": String(Buffer.byteLength(body)), ...extraHeaders });
  res.end(body);
}

function openAIErrorBody(status: number, code: string, message: string): any {
  return { error: { message, type: code, code } };
}

function anthropicErrorBody(status: number, code: string, message: string): any {
  return { type: "error", error: { type: code, message } };
}

function sendUpstreamError(
  res: any,
  api: string,
  result: GatewayResult,
  failures: Failure[]
): void {
  const status = result.ok ? 502 : result.status;
  let message = `upstream error ${status}`;
  let code = result.ok ? "upstream_error" : result.code;
  if (!result.ok && result.bodyText) {
    try {
      const v = JSON.parse(result.bodyText);
      if (v && v.error && typeof v.error.message === "string") message = v.error.message;
      else if (typeof v.message === "string") message = v.message;
    } catch {
      message = result.bodyText.slice(0, 500);
    }
  }
  const headers = failuresHeader(failures);
  if (api === "openai") sendJson(res, status, openAIErrorBody(status, code, message), headers);
  else sendJson(res, status, anthropicErrorBody(status, code, message), headers);
}

function handleOpenAIJson(res: any, model: string, providerJson: any, body: any, failures: Failure[]): void {
  const text = providerTextOf(providerJson);
  const toolCalls = providerToolCallsOf(providerJson);
  const finish = providerFinishOf(providerJson);
  const usage = openAIUsageOf(providerJson, text, promptTextForUsage("openai", body));
  const message: any = { role: "assistant", content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const out: any = {
    id: FIXED_CHAT_ID,
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finish }],
    usage,
  };
  sendJson(res, 200, out, failuresHeader(failures));
}

function handleAnthropicJson(res: any, model: string, providerJson: any, body: any, failures: Failure[]): void {
  const text = providerTextOf(providerJson);
  const toolCalls = providerToolCallsOf(providerJson);
  const finish = providerFinishOf(providerJson);
  const usage = openAIUsageOf(providerJson, text, promptTextForUsage("anthropic", body));
  const content: any[] = [];
  if (text) content.push({ type: "text", text });
  for (const tc of toolCalls) {
    let input: any = {};
    try {
      input = JSON.parse(tc.function && typeof tc.function.arguments === "string" ? tc.function.arguments : "{}");
    } catch {
      input = {};
    }
    content.push({
      type: "tool_use",
      id: typeof tc.id === "string" ? tc.id : "toolu_aggregator02",
      name: tc.function && typeof tc.function.name === "string" ? tc.function.name : "",
      input,
    });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  const out: any = {
    id: FIXED_MSG_ID,
    type: "message",
    role: "assistant",
    content,
    model,
    stop_reason: mapFinishToAnthropic(finish),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
  sendJson(res, 200, out, failuresHeader(failures));
}

function handleOpenAIStream(res: any, model: string, sseText: string, body: any, failures: Failure[]): void {
  const frames = parseSse(sseText);
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    ...failuresHeader(failures),
  };
  res.writeHead(200, headers);
  let sawFinish = false;
  let sawUsage = false;
  let text = "";
  let finish = "stop";
  let usage: any = null;
  for (const frame of frames) {
    if (frame.data === "[DONE]") continue;
    let chunk: any = null;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!chunk || typeof chunk !== "object") continue;
    chunk.id = FIXED_CHAT_ID;
    chunk.created = FIXED_CREATED;
    chunk.model = model;
    if (chunk.usage && typeof chunk.usage === "object") {
      sawUsage = true;
      usage = chunk.usage;
    }
    if (Array.isArray(chunk.choices)) {
      for (const ch of chunk.choices) {
        if (ch && ch.delta && typeof ch.delta.content === "string") text += ch.delta.content;
        if (ch && typeof ch.finish_reason === "string" && ch.finish_reason) {
          sawFinish = true;
          finish = ch.finish_reason;
        }
      }
    }
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  }
  if (!sawFinish) {
    const fin: any = {
      id: FIXED_CHAT_ID,
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    };
    res.write(`data: ${JSON.stringify(fin)}\n\n`);
    finish = "stop";
  }
  if (!sawUsage) {
    const u = openAIUsageOf(
      { usage: usage || undefined },
      text,
      promptTextForUsage("openai", body)
    );
    const uchunk: any = {
      id: FIXED_CHAT_ID,
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: u.prompt_tokens, completion_tokens: u.completion_tokens },
    };
    res.write(`data: ${JSON.stringify(uchunk)}\n\n`);
  }
  res.write("data: [DONE]\n\n");
  res.end();
}

function handleAnthropicStream(res: any, model: string, sseText: string, body: any, failures: Failure[]): void {
  const frames = parseSse(sseText);
  let text = "";
  let finish = "stop";
  let usage: any = null;
  const toolCalls: any[] = [];
  for (const frame of frames) {
    if (frame.data === "[DONE]") continue;
    let chunk: any = null;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!chunk || typeof chunk !== "object") continue;
    if (chunk.usage && typeof chunk.usage === "object") usage = chunk.usage;
    if (Array.isArray(chunk.choices)) {
      for (const ch of chunk.choices) {
        if (ch && ch.delta && typeof ch.delta.content === "string") text += ch.delta.content;
        if (ch && ch.delta && Array.isArray(ch.delta.tool_calls)) {
          for (const tc of ch.delta.tool_calls) {
            const idx = typeof tc.index === "number" ? tc.index : toolCalls.length;
            while (toolCalls.length <= idx) toolCalls.push({ id: "", name: "", args: "" });
            if (tc.id) toolCalls[idx].id = tc.id;
            if (tc.function && typeof tc.function.name === "string" && tc.function.name) toolCalls[idx].name = tc.function.name;
            if (tc.function && typeof tc.function.arguments === "string") toolCalls[idx].args += tc.function.arguments;
          }
        }
        if (ch && typeof ch.finish_reason === "string" && ch.finish_reason) finish = ch.finish_reason;
      }
    }
  }
  const promptText = promptTextForUsage("anthropic", body);
  const u = openAIUsageOf({ usage: usage || undefined }, text, promptText);
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    ...failuresHeader(failures),
  };
  res.writeHead(200, headers);
  const send = (event: string, data: any): void => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };
  send("message_start", {
    type: "message_start",
    message: {
      id: FIXED_MSG_ID,
      type: "message",
      role: "assistant",
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: u.prompt_tokens, output_tokens: 0 },
    },
  });
  let blockIndex = 0;
  if (text) {
    send("content_block_start", { type: "content_block_start", index: blockIndex, content_block: { type: "text", text: "" } });
    send("content_block_delta", { type: "content_block_delta", index: blockIndex, delta: { type: "text_delta", text } });
    send("content_block_stop", { type: "content_block_stop", index: blockIndex });
    blockIndex++;
  }
  for (const tc of toolCalls) {
    if (!tc.name) continue;
    const id = tc.id || "toolu_aggregator02";
    send("content_block_start", {
      type: "content_block_start",
      index: blockIndex,
      content_block: { type: "tool_use", id, name: tc.name, input: {} },
    });
    send("content_block_delta", {
      type: "content_block_delta",
      index: blockIndex,
      delta: { type: "input_json_delta", partial_json: tc.args || "{}" },
    });
    send("content_block_stop", { type: "content_block_stop", index: blockIndex });
    blockIndex++;
  }
  if (blockIndex === 0) {
    send("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    send("content_block_stop", { type: "content_block_stop", index: 0 });
  }
  send("message_delta", {
    type: "message_delta",
    delta: { stop_reason: mapFinishToAnthropic(finish), stop_sequence: null },
    usage: { output_tokens: u.completion_tokens },
  });
  send("message_stop", { type: "message_stop" });
  res.end();
}

function readBody(req: any): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > 2 * 1024 * 1024) {
        reject(new Error("body too large"));
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function handleCompletion(req: any, res: any, api: string): Promise<void> {
  let raw = "";
  try {
    raw = await readBody(req);
  } catch {
    sendJson(res, 413, api === "openai" ? openAIErrorBody(413, "request_too_large", "body too large") : anthropicErrorBody(413, "request_too_large", "body too large"), {});
    return;
  }
  let body: any = null;
  try {
    body = JSON.parse(raw);
  } catch {
    sendJson(res, 400, api === "openai" ? openAIErrorBody(400, "invalid_json", "body is not JSON") : anthropicErrorBody(400, "invalid_json", "body is not JSON"), {});
    return;
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    sendJson(res, 400, api === "openai" ? openAIErrorBody(400, "invalid_json", "body is not a JSON object") : anthropicErrorBody(400, "invalid_json", "body is not a JSON object"), {});
    return;
  }
  const stream = body.stream === true;
  const reqs = requirementsFor(api, stream, body);
  const ranked = rankedModels(reqs);
  if (ranked.length === 0) {
    sendJson(
      res,
      503,
      api === "openai"
        ? openAIErrorBody(503, "no_model", "no free model meets this prompt's requirements")
        : anthropicErrorBody(503, "no_model", "no free model meets this prompt's requirements"),
      {}
    );
    return;
  }
  const dialed = await dialRanked(ranked, api, body, stream);
  if (!dialed.model || !dialed.result.ok) {
    sendUpstreamError(res, api, dialed.result, dialed.failures);
    return;
  }
  if (api === "openai") {
    if (stream) handleOpenAIStream(res, dialed.model, dialed.result.text, body, dialed.failures);
    else {
      if (!dialed.result.json) {
        sendUpstreamError(res, api, { ok: false, status: 502, code: "upstream_error", bodyText: "", network: true }, dialed.failures);
        return;
      }
      handleOpenAIJson(res, dialed.model, dialed.result.json, body, dialed.failures);
    }
  } else {
    if (stream) handleAnthropicStream(res, dialed.model, dialed.result.text, body, dialed.failures);
    else {
      if (!dialed.result.json) {
        sendUpstreamError(res, api, { ok: false, status: 502, code: "upstream_error", bodyText: "", network: true }, dialed.failures);
        return;
      }
      handleAnthropicJson(res, dialed.model, dialed.result.json, body, dialed.failures);
    }
  }
}

function handleModels(req: any, res: any): void {
  const anthropic = req.headers["x-api-key"] !== undefined;
  if (anthropic) {
    const data = MODELS.map((m) => ({
      type: "model",
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_CREATED_AT,
    }));
    sendJson(res, 200, { data, has_more: false }, {});
  } else {
    const data = MODELS.map((m) => ({ id: m.id, object: "model", created: FIXED_CREATED, owned_by: "openrouter" }));
    sendJson(res, 200, { object: "list", data }, {});
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function register(port: number, catalog: any): Promise<boolean> {
  const envUrl = (process.env.MAKEFUN_ENV_URL || "").replace(/\/$/, "");
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 5000);
      try {
        const res = await fetch(envUrl + "/register", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ protocol_version: 1, port, catalog }),
          signal: controller.signal,
        });
        if (res.status >= 200 && res.status < 300) return true;
      } finally {
        clearTimeout(timer);
      }
    } catch {
      // retry below
    }
    await sleep(500);
  }
  return false;
}

async function pollWork(server: any): Promise<void> {
  const envUrl = (process.env.MAKEFUN_ENV_URL || "").replace(/\/$/, "");
  const timer = setInterval(async () => {
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 5000);
      try {
        const res = await fetch(envUrl + "/work", { signal: controller.signal });
        if (res.status >= 200 && res.status < 300) {
          const v: any = await res.json();
          if (v && v.done === true) {
            clearInterval(timer);
            server.close(() => process.exit(0));
            setTimeout(() => process.exit(0), 1000);
          }
        }
      } finally {
        clearTimeout(t);
      }
    } catch {
      // keep polling
    }
  }, 100);
}

async function main(): Promise<void> {
  parseBrief();
  const args = process.argv.slice(2);
  if (args.includes("--list")) {
    const hasBrief = !!process.env.MAKEFUN_PROMPT_FILE;
    const catalog = hasBrief && MODELS.length > 0 ? catalogObject() : emptyCatalog();
    process.stdout.write(JSON.stringify(catalog) + "\n");
    process.exit(0);
    return;
  }
  const catalog = MODELS.length > 0 ? catalogObject() : emptyCatalog();
  const server = http.createServer((req: any, res: any) => {
    const url = req.url || "/";
    const path = url.split("?")[0];
    if (req.method === "GET" && path === "/v1/models") {
      handleModels(req, res);
      return;
    }
    if (req.method === "POST" && path === "/v1/chat/completions") {
      handleCompletion(req, res, "openai").catch(() => {
        try {
          sendJson(res, 500, openAIErrorBody(500, "internal_error", "internal error"), {});
        } catch {
          // socket gone
        }
      });
      return;
    }
    if (req.method === "POST" && path === "/v1/messages") {
      handleCompletion(req, res, "anthropic").catch(() => {
        try {
          sendJson(res, 500, anthropicErrorBody(500, "internal_error", "internal error"), {});
        } catch {
          // socket gone
        }
      });
      return;
    }
    sendJson(res, 404, { error: { message: "not found", type: "not_found", code: "not_found" } }, {});
  });
  await new Promise<void>((resolve) => {
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const addr: any = server.address();
  const port = typeof addr === "object" && addr !== null ? addr.port : 0;
  const envUrl = process.env.MAKEFUN_ENV_URL || "";
  if (envUrl) {
    const ok = await register(port, catalog);
    if (!ok) {
      process.stderr.write("free-model-aggregator-02: register failed\n");
      process.exit(1);
      return;
    }
    await pollWork(server);
  }
}

main().catch((err) => {
  process.stderr.write(`free-model-aggregator-02: ${err && err.stack ? err.stack : String(err)}\n`);
  process.exit(1);
});
