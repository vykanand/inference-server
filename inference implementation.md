1. The fundamental architecture

Build this:

                         ┌─────────────────────────┐
                         │ Cline / OpenCode / IDE  │
                         │ Roo / Continue / etc.   │
                         └────────────┬────────────┘
                                      │
                       OpenAI-compatible API
                                      │
                    ┌─────────────────▼─────────────────┐
                    │       YOUR INFERENCE GATEWAY      │
                    │                                   │
                    │  Auth / Rate Limits / Billing    │
                    │  Request Validation              │
                    │  Session / Request IDs           │
                    │  Model Resolution                │
                    │  Capability Resolution           │
                    │  Prompt Normalization             │
                    │  Tool Normalization               │
                    │  Provider Routing                 │
                    │  Retry / Fallback                 │
                    │  Stream Normalization              │
                    │  Error Normalization              │
                    └───────────────┬───────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
       ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
       │ OpenAI    │          │ Anthropic │          │ Gemini    │
       │ Adapter   │          │ Adapter   │          │ Adapter   │
       └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
             │                      │                      │
       ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
       │ Together  │          │ Fireworks │          │ Groq      │
       └───────────┘          └───────────┘          └───────────┘

                         + local vLLM
                         + SGLang
                         + llama.cpp
                         + Ollama
                         + custom providers

The most important principle:

Your public API should be stable even when the underlying provider is not.

OpenCode itself separates provider configuration/runtime from the client-facing experience, including base URLs, headers, model limits and provider packages.

2. Support these endpoints first

For maximum compatibility, implement:

GET  /v1/models

POST /v1/chat/completions

POST /v1/responses

GET  /v1/models/:model

POST /v1/embeddings       optional

And:

GET  /health
GET  /ready
GET  /version

Your absolute minimum for coding agents is:

GET  /v1/models
POST /v1/chat/completions

But if you're trying to be broadly compatible in 2026, implement /v1/responses as well.

OpenCode's current documentation specifically says OpenAI-compatible providers using /v1/chat/completions should use the OpenAI-compatible runtime, while providers using /v1/responses should use the OpenAI runtime.

3. Your internal API should NOT be OpenAI-specific

This is one of the biggest architectural decisions.

Don't make your internal representation:

OpenAIChatCompletionRequest

Instead create your own canonical IR:

type InferenceRequest = {
  requestId: string
  model: ModelRef

  system?: Content[]
  messages: Message[]

  tools?: ToolDefinition[]
  toolChoice?: ToolChoice

  responseFormat?: ResponseFormat

  temperature?: number
  topP?: number
  topK?: number

  maxTokens?: number

  stop?: string[]

  stream: boolean

  reasoning?: ReasoningConfig

  metadata?: Record<string, unknown>

  providerPreferences?: ProviderPreferences

  conversationId?: string
}

Then:

OpenAI request
       ↓
Canonical IR
       ↓
Provider adapter
       ↓
Anthropic/Gemini/OpenAI/etc.

And responses:

Provider response
       ↓
Provider adapter
       ↓
Canonical Response IR
       ↓
OpenAI-compatible serializer

This prevents your entire system from becoming a pile of provider-specific conditionals.

4. Build a canonical message format

You need to support much more than:

{
  "role": "user",
  "content": "hello"
}

Your internal message model should support:

type Message = {
  id?: string
  role:
    | "system"
    | "developer"
    | "user"
    | "assistant"
    | "tool"

  content: ContentPart[]

  toolCalls?: ToolCall[]

  toolCallId?: string

  name?: string

  metadata?: Record<string, unknown>
}

Content:

type ContentPart =
  | {
      type: "text"
      text: string
    }
  | {
      type: "image"
      url: string
    }
  | {
      type: "image"
      base64: string
      mediaType: string
    }
  | {
      type: "file"
      url: string
    }
  | {
      type: "reasoning"
      text: string
    }

This matters because coding agents increasingly send multimodal context and reasoning/tool information.

5. Tool calling is the #1 compatibility requirement

If you want Cline/OpenCode-style agents to work, tool calling must be first-class.

Do not treat tools as prompt text.

Support:

{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string"
            }
          },
          "required": ["path"]
        }
      }
    }
  ]
}

And:

{
  "tool_choice": "auto"
}

plus:

{
  "tool_choice": {
    "type": "function",
    "function": {
      "name": "read_file"
    }
  }
}

Your canonical representation:

type ToolDefinition = {
  id?: string
  name: string
  description?: string
  inputSchema: JSONSchema
}
6. Tool calls must have stable IDs

This is critical.

Generate:

call_01J...

rather than relying on a provider's internal ID being compatible.

Example:

{
  "id": "call_abc123",
  "name": "read_file",
  "arguments": {
    "path": "src/index.ts"
  }
}

Then the agent responds:

{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "..."
}

Your gateway must preserve this relationship across:

assistant
   ↓
tool_call
   ↓
client executes tool
   ↓
tool result
   ↓
gateway
   ↓
model

Never randomly regenerate the tool call ID during retries or stream reconstruction.

7. Streaming must be designed separately

Do not implement streaming as:

generate full response
↓
split string
↓
send chunks

That's technically streaming but operationally terrible.

You need a canonical event stream:

type InferenceEvent =
  | { type: "response.started" }
  | { type: "text.delta"; text: string }
  | { type: "reasoning.delta"; text: string }

  | {
      type: "tool_call.started"
      id: string
      name: string
    }

  | {
      type: "tool_call.delta"
      id: string
      argumentsDelta: string
    }

  | {
      type: "tool_call.completed"
      id: string
      arguments: unknown
    }

  | {
      type: "usage"
      inputTokens: number
      outputTokens: number
      totalTokens: number
    }

  | {
      type: "response.completed"
    }

  | {
      type: "error"
      error: InferenceError
    }

Then serialize that to:

OpenAI SSE
Anthropic SSE
Responses API SSE

as required.

OpenRouter's current request tooling explicitly supports streaming and tool calls, which illustrates the feature set an agent-oriented compatibility layer needs.

8. Never assume one tool call per response

Your system needs to support:

assistant
 ├── tool_call A
 ├── tool_call B
 └── tool_call C

Potentially:

[
  {
    "id": "call_1",
    "name": "read_file",
    "arguments": {...}
  },
  {
    "id": "call_2",
    "name": "read_file",
    "arguments": {...}
  },
  {
    "id": "call_3",
    "name": "grep",
    "arguments": {...}
  }
]

Do not serialize these into one call.

9. Streaming tool calls are harder

Providers can stream:

{name:"read_file", arguments:"{"}
arguments:"\"pa"
arguments:"th\""
arguments:":"
arguments:"\"src/a.ts\""
arguments:"}"

You need a streaming JSON accumulator.

Conceptually:

class ToolCallAccumulator {
  id = ""
  name = ""
  argumentsBuffer = ""

  append(delta) {
    this.argumentsBuffer += delta
  }

  finalize() {
    return JSON.parse(this.argumentsBuffer)
  }
}

But production code should also handle:

partial JSON
escaped strings
unicode splits
multiple tool calls
malformed arguments
provider-specific deltas
10. Never trust streamed JSON blindly

Use:

incremental parser
       ↓
buffer
       ↓
UTF-8 safe decoding
       ↓
JSON parse
       ↓
schema validation
       ↓
canonical tool call

If the provider generates:

{"path":

and then disconnects, don't crash the whole inference service.

Return a normalized error.

11. Implement model capability metadata

This is essential for reliability.

Your model registry should contain:

type ModelCapabilities = {
  toolCalling: boolean
  parallelToolCalling: boolean

  structuredOutputs: boolean

  streaming: boolean

  vision: boolean
  audioInput: boolean

  reasoning: boolean

  jsonMode: boolean

  promptCaching: boolean

  maxContextTokens: number
  maxOutputTokens: number

  supportedParameters: Set<string>
}

Example:

{
  "id": "provider/model",
  "capabilities": {
    "toolCalling": true,
    "parallelToolCalling": true,
    "structuredOutputs": true,
    "streaming": true,
    "vision": true,
    "reasoning": true
  },
  "limits": {
    "context": 200000,
    "output": 65536
  }
}

OpenCode exposes context/output limits in its provider/model configuration so that the client knows how much context remains.

12. Don't advertise capabilities you haven't tested

This is a major mistake.

For example:

{
  "toolCalling": true
}

doesn't mean:

"The model technically accepts a tools parameter."

It means:

"This exact model/provider combination successfully completes real coding-agent tool-call tests."

Use capability levels:

UNKNOWN
SUPPORTED
DEGRADED
UNSUPPORTED

Even better:

tool_calling:
  native
  emulated
  unreliable
  unsupported
13. Build a capability test suite

Every model should be tested against:

01 basic generation
02 streaming
03 system message
04 developer message
05 tool declaration
06 simple tool call
07 multiple tools
08 parallel tool calls
09 tool result
10 tool call + streaming
11 malformed tool arguments
12 structured output
13 JSON mode
14 long context
15 context overflow
16 reasoning
17 vision
18 stop sequences
19 max_tokens
20 temperature
21 cancellation
22 timeout
23 retry
24 provider error

And coding-agent scenarios:

read_file
write_file
edit_file
grep
glob
bash
terminal
git
search
browser
MCP
14. Build a real model registry

Don't hardcode:

if (model === "foo") ...

Use a database/config:

providers
models
model_capabilities
provider_models
routing_policies

Example:

models
------
id
provider_id
model_id
display_name
context_length
max_output_tokens
status
created_at
updated_at

Capabilities:

model_capabilities
------------------
model_id
tool_calling
parallel_tools
structured_output
vision
reasoning
streaming
json_mode
15. Provider adapters

Create an adapter interface:

interface ProviderAdapter {
  id: string

  listModels(): Promise<Model[]>

  generate(
    request: InferenceRequest
  ): Promise<InferenceResponse>

  stream(
    request: InferenceRequest
  ): AsyncIterable<InferenceEvent>

  healthCheck(): Promise<HealthStatus>
}

Then:

OpenAIAdapter
AnthropicAdapter
GoogleAdapter
OpenRouterAdapter
GroqAdapter
TogetherAdapter
FireworksAdapter
BedrockAdapter
VertexAdapter
AzureAdapter
VLLMAdapter
SGLangAdapter
OllamaAdapter

For OpenAI-compatible providers:

OpenAICompatibleAdapter

should cover the majority.

16. But don't make everything OpenAI-compatible internally

This is another important distinction.

Use:

External compatibility:
OpenAI API

Internal abstraction:
Canonical inference protocol

Provider:
Native provider API

NOT:

OpenAI request
     ↓
Anthropic API

directly.

Anthropic, Gemini, OpenAI, etc. have different semantics.

Your adapter should explicitly translate:

Canonical IR
     ↓
Provider request

and:

Provider response
     ↓
Canonical event
17. Parameter normalization

Coding agents send parameters you may not support.

Example:

{
  "temperature": 0.2,
  "top_p": 0.95,
  "max_tokens": 8192,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "seed": 123,
  "reasoning_effort": "high"
}

You need a parameter policy:

type ParameterSupport =
  | "native"
  | "translated"
  | "ignored"
  | "rejected"

For every parameter:

temperature → native
top_p       → native
seed        → native
reasoning_effort → translated
foo          → rejected

Never silently send unsupported parameters to every provider.

18. Request sanitization

Before sending:

Validate request
↓
Normalize roles
↓
Normalize content
↓
Normalize tools
↓
Normalize tool choice
↓
Normalize response format
↓
Filter unsupported parameters
↓
Check context length
↓
Select provider
↓
Send
19. Context management is part of inference

Coding agents can send enormous conversations.

Implement:

token counting
context budgeting
automatic truncation
tool-result compaction
message prioritization
system-message preservation
recent-message preservation

A useful budget:

model context
│
├── system prompt       10%
├── repository context  30%
├── conversation        40%
├── tools                10%
└── output reserve       10%

Obviously make this configurable.

20. Tool definitions consume context

This is frequently overlooked.

If Cline sends 50 tools:

tool schemas
+
system prompt
+
repo context
+
conversation
+
tool outputs

can easily exceed the model context.

Your gateway should expose:

{
  "context": {
    "used": 142000,
    "available": 58000,
    "limit": 200000
  }
}

internally.

21. Retry architecture

Never blindly retry everything.

Classify errors:

AUTHENTICATION
INVALID_REQUEST
RATE_LIMIT
CONTEXT_TOO_LONG
MODEL_UNAVAILABLE
PROVIDER_OVERLOAD
TIMEOUT
NETWORK
SERVER_ERROR
CONTENT_FILTER
TOOL_CALL_ERROR
UNKNOWN

Retry:

429
502
503
504
network reset
connection timeout

Potentially retry.

Don't automatically retry:

400 invalid request
401 unauthorized
403 forbidden
context too long
invalid tool schema
22. Exponential backoff

Something like:

attempt 1: 250ms
attempt 2: 500ms
attempt 3: 1s
attempt 4: 2s
attempt 5: 4s

with jitter:

delay = base * 2^attempt + random(0, jitter)

But enforce:

max retry duration
max attempts
request deadline
23. Don't retry after irreversible tool execution

This is extremely important for agents.

Imagine:

model → tool_call: delete_file()

Your provider times out after the model generated the call.

You don't know whether the model response was actually delivered.

If you retry naively:

same request
↓
same tool call
↓
delete_file AGAIN

You can create duplicate side effects.

You need idempotency.

24. Idempotency keys

Every request:

Idempotency-Key: req_abc123

Internally:

tenant
+
conversation
+
request
+
attempt

Store state:

REQUESTED
RUNNING
STREAMING
COMPLETED
FAILED
UNKNOWN

This allows safe recovery.

25. Separate inference retry from agent retry

This distinction is critical.

Inference retry
provider failed
→ same inference
Agent retry
tool failed
→ model receives tool error
→ model decides what to do

Never turn every provider failure into a new agent turn.

26. Provider fallback

Build:

Primary:
Claude X

Fallback:
GPT X

Fallback:
Gemini X

Fallback:
Qwen X

But don't blindly fallback.

Capability-aware routing:

request requires:
  tools
  vision
  200k context
  structured output

        ↓

filter models

        ↓

available models

        ↓

rank

        ↓

route

OpenRouter's model routing/fallback approach is a useful reference here; its router can filter models based on requested features such as tool calling and structured outputs.

27. Model routing score

You can calculate:

score =
    capability_match * 0.30
  + reliability       * 0.25
  + latency           * 0.15
  + quality           * 0.15
  + availability      * 0.10
  + cost              * 0.05

Then select:

best eligible model

rather than:

cheapest model
28. Health-aware routing

Track:

success rate
p50 latency
p95 latency
p99 latency
429 rate
5xx rate
tool-call success
stream disconnect rate
context errors

For every provider/model.

Example:

Claude:
success 99.8%
p95 4.2s
tool success 99.9%

Model X:
success 94.2%
p95 11.8s
tool success 82%

Automatically reduce Model X's routing weight.

29. Circuit breaker

For each provider/model:

CLOSED
   ↓ failures
OPEN
   ↓ cooldown
HALF_OPEN
   ↓ successful probe
CLOSED

This prevents a broken provider from destroying your entire gateway.

30. Streaming reliability

Your gateway needs to survive:

client disconnect
provider disconnect
proxy timeout
partial stream
invalid SSE
duplicate chunks
out-of-order events
heartbeat timeout

Implement:

heartbeat
idle timeout
absolute timeout
stream cancellation
upstream cancellation
buffer limits
backpressure
31. SSE implementation

For Chat Completions:

Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

Emit:

data: {...}

data: {...}

data: [DONE]

Correctly.

Do not buffer the entire response in your HTTP framework.

32. Preserve UTF-8 boundaries

Never split raw bytes arbitrarily.

Bad:

Buffer → slice(0, 100)

You can split Unicode characters.

Use:

provider bytes
↓
UTF-8 decoder
↓
text chunks

before constructing events.

33. Client disconnect propagation

If:

Cline
  ↓
your gateway
  ↓
provider

and Cline closes the connection:

cancel provider request

immediately.

Otherwise you're paying for generations nobody is consuming.

34. Tool execution belongs outside inference

Your inference server should generally not execute arbitrary coding tools.

Instead:

Inference server
      ↓
returns tool call
      ↓
Cline/OpenCode executes tool
      ↓
tool result
      ↓
inference server

This matches the architecture of coding agents, where tools operate alongside the model. OpenCode, for example, has built-in tools and supports custom tools/MCP separately from the model provider layer.

35. MCP compatibility

If you want broad compatibility, don't implement MCP inside the model protocol.

Treat MCP as another tool source:

MCP server
    ↓
tool registry
    ↓
canonical ToolDefinition
    ↓
model

Then:

model calls MCP tool
       ↓
client/MCP runtime executes
       ↓
tool result
       ↓
model
36. Structured outputs

Support:

{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "result",
      "schema": {}
    }
  }
}

Internally:

type StructuredOutput = {
  mode: "json" | "json_schema"
  schema?: JSONSchema
  strict?: boolean
}

Validate output:

generation
↓
JSON parser
↓
schema validator
↓
success

If invalid:

do NOT blindly return invalid JSON

Instead follow a configurable repair policy.

37. JSON repair

For models that don't reliably produce JSON:

response
↓
parse
↓
invalid
↓
repair/second pass
↓
validate

But label this capability as:

structured_output = emulated

not native.

38. Reasoning support

Modern coding models may expose reasoning differently.

Your internal protocol should support:

type Reasoning = {
  text?: string
  encrypted?: string
  tokens?: number
}

But don't assume every provider exposes raw reasoning.

Support:

reasoning effort
reasoning budget
reasoning summary
reasoning tokens

as separate concepts.

39. Don't leak provider-specific reasoning formats

Normalize:

provider reasoning
       ↓
canonical reasoning event
       ↓
client-specific serializer

rather than putting:

anthropic_thinking
gemini_thought
openai_reasoning

throughout your application.

40. Usage accounting

Track:

input_tokens
output_tokens
cached_input_tokens
reasoning_tokens
total_tokens

Also:

provider_cost
customer_cost
margin

Every request should have:

request_id
tenant_id
user_id
model
provider
started_at
completed_at
latency
status
41. Billing must be based on canonical usage

Don't assume:

provider response usage

is always complete.

Some streams don't provide usage unless explicitly requested.

Therefore support:

provider-reported usage
+
fallback tokenizer estimate

and mark:

usage_source:
  provider
  estimated
42. Model pricing registry

Store:

input $ / 1M tokens
output $ / 1M tokens
cached input
reasoning
image
audio

Then calculate:

cost =
 input_tokens  * input_price
+
 output_tokens * output_price
43. Authentication

Support:

Authorization: Bearer xxx

Potentially:

x-api-key
x-org-id
x-project-id

Internally:

API key
 ↓
tenant
 ↓
user
 ↓
permissions
 ↓
quota

Never pass your internal master provider keys to clients.

44. API key isolation

Use:

client API key
      ↓
gateway
      ↓
provider credential

Never:

client
 ↓
provider key

Store credentials in:

AWS Secrets Manager
GCP Secret Manager
Vault
KMS-encrypted DB

not plain configuration files.

45. Rate limiting

Use multiple levels:

per API key
per user
per organization
per model
per provider
per IP

Token-based rate limits are better than request-only limits:

100 requests/minute

isn't enough.

Also:

1M input tokens/min
200k output tokens/min
46. Concurrency limits

Implement:

global concurrency
tenant concurrency
provider concurrency
model concurrency

Example:

tenant:
  max 20 concurrent requests

provider:
  max 100

model:
  max 50

Use queues rather than allowing uncontrolled fan-out.

47. Timeouts

You need at least:

DNS timeout
TCP timeout
TLS timeout
connect timeout
first-token timeout
idle-stream timeout
total request timeout

Especially:

TTFT = time to first token

A provider that accepts a connection but never generates a token should not hang forever.

48. Error normalization

Return a consistent error:

{
  "error": {
    "type": "rate_limit_error",
    "code": "provider_rate_limit",
    "message": "Provider rate limit exceeded",
    "param": null,
    "request_id": "req_123",
    "retryable": true
  }
}

Internally include:

provider
provider_request_id
HTTP status
retry-after
attempt
route

Never expose secrets or internal stack traces.

49. Provider error mapping

Map:

OpenAI 429
Anthropic 529
Gemini RESOURCE_EXHAUSTED
Groq 429
network timeout

into:

RATE_LIMIT
OVERLOADED
TIMEOUT

The client should not need to know which provider you selected.

50. Request tracing

Every request gets:

request_id
trace_id
span_id
provider_request_id

Example:

req_123
 ├── route:model-selection
 ├── route:provider-a
 ├── attempt:1
 ├── attempt:2
 └── serialization

Use OpenTelemetry.

51. Observability dashboard

Track:

Availability
success %
error %
timeouts %
Latency
TTFT p50/p95/p99
total latency
Model quality
tool-call success
structured-output success
agent task completion
Economics
tokens
cost
revenue
margin
Routing
provider distribution
fallback rate
circuit breaker state
52. Logging

Log:

request_id
model
provider
duration
status
token usage
error type
tool count

Avoid logging raw prompts by default.

If you offer prompt logging:

explicit opt-in
encryption
retention policy
PII redaction
53. Prompt caching

If providers support caching, make it a routing feature.

Canonical:

cacheControl?: {
  type: "ephemeral"
}

Adapter converts it appropriately.

Don't expose:

anthropic_cache_control

through your entire system.

54. Prefix caching

Coding agents often repeatedly send:

system prompt
repository instructions
tool schemas

If your inference backend supports prefix caching, exploit it.

Typical flow:

stable prefix
   ↓
cached KV/prefix
   ↓
new user/tool content
   ↓
generation

This can dramatically improve agent economics/latency.

55. Request deduplication

Agents sometimes accidentally issue duplicate requests.

Hash:

tenant
model
messages
tools
parameters

and use:

idempotency key

rather than blindly executing duplicates.

56. Model aliases

Never force clients to know infrastructure.

Expose:

coding-fast
coding-best
coding-cheap
reasoning
vision

Then:

coding-best
  ↓
router
  ↓
eligible models

This lets you change providers without breaking Cline/OpenCode configurations.

57. /v1/models matters more than people think

Return proper OpenAI-compatible data:

{
  "object": "list",
  "data": [
    {
      "id": "coding-best",
      "object": "model",
      "created": 1750000000,
      "owned_by": "your-platform"
    }
  ]
}

If the client queries models, don't make it fail.

58. Model IDs

Use stable IDs:

company/qwen-coder
company/claude-coding
company/gpt-coding

Don't expose temporary provider deployment IDs unless necessary.

Internally:

public model ID
       ↓
provider deployment
       ↓
actual model
59. Compatibility profiles

I strongly recommend:

profile: openai-chat
profile: openai-responses
profile: anthropic
profile: gemini
profile: openrouter

Each profile defines:

roles
tools
stream events
reasoning
structured outputs
usage
errors
60. Client compatibility matrix

Before claiming compatibility, test:

Client	Basic	Streaming	Tools	Parallel tools	Structured	Vision
Cline	✓	✓	✓	✓	✓	✓
OpenCode	✓	✓	✓	✓	✓	✓
Roo	✓	✓	✓	✓	✓	✓
Continue	✓	✓	✓	✓	✓	✓
Aider	✓	✓	✓	✓	—	—

Treat this as a test matrix, not a marketing assumption.

61. The golden coding-agent test

Create a repository:

test-project/
├── package.json
├── src/
│   ├── index.ts
│   ├── auth.ts
│   └── users.ts
└── tests/
    └── auth.test.ts

Then run the exact same prompt:

Inspect the repository, find the authentication bug, fix it, run the tests, and explain what changed.

The expected sequence should be approximately:

model
 ↓
list/read files
 ↓
grep/search
 ↓
read relevant file
 ↓
reason
 ↓
edit
 ↓
run tests
 ↓
inspect failure
 ↓
edit again
 ↓
run tests
 ↓
final answer

If your gateway survives this repeatedly, you're getting somewhere.

62. The REALLY important reliability test

Run:

1000 agent sessions

with:

5–20 tool calls/session

Measure:

tool-call completion rate
stream completion rate
duplicate tool calls
malformed tool calls
timeouts
fallbacks
context errors

Your target should be something like:

request success       > 99.9%
stream success        > 99.9%
tool-call transport   > 99.99%

The exact target depends on your infrastructure.

63. Fault injection

Don't only test healthy providers.

Simulate:

429
500
502
503
504
timeout
connection reset
half-closed stream
invalid JSON
truncated SSE
slow first token
slow generation
provider unavailable
wrong model
context overflow

Your gateway should recover predictably.

64. Chaos test

For example:

provider A:
  20% random 503

provider B:
  10% timeout

provider C:
  5% malformed stream

Your routing system should continue serving requests.

65. Don't promise “100% working”

There is one important correction to the premise.

You can build:

100% protocol compatibility

You cannot honestly guarantee:

100% model behavior compatibility

because models themselves differ.

For example:

Model A:
excellent tool calling

Model B:
technically supports tools but occasionally emits invalid arguments

Model C:
doesn't support tools

Model D:
supports tools but not parallel tools

Your job is to make those differences explicit and machine-detectable.

66. Your compatibility contract

I would define:

Compatibility =
  API compatibility
+ streaming compatibility
+ tool protocol compatibility
+ model capability discovery
+ error compatibility
+ retry semantics
+ context semantics
+ cancellation
+ usage reporting
+ client-specific quirks

That's what makes an inference service "OpenRouter-like."

67. Recommended repository structure

Something like:

inference/
│
├── apps/
│   ├── api/
│   ├── worker/
│   └── router/
│
├── packages/
│   ├── protocol/
│   │   ├── messages/
│   │   ├── tools/
│   │   ├── events/
│   │   ├── errors/
│   │   └── schemas/
│   │
│   ├── providers/
│   │   ├── openai/
│   │   ├── anthropic/
│   │   ├── gemini/
│   │   ├── openrouter/
│   │   └── compatible/
│   │
│   ├── router/
│   ├── capabilities/
│   ├── tokenizer/
│   ├── streaming/
│   ├── retry/
│   ├── rate-limit/
│   ├── billing/
│   ├── observability/
│   └── security/
│
├── tests/
│   ├── protocol/
│   ├── providers/
│   ├── streaming/
│   ├── tools/
│   ├── compatibility/
│   ├── chaos/
│   └── agent-e2e/
│
└── infra/
    ├── postgres/
    ├── redis/
    └── kubernetes/
68. Recommended request lifecycle

This should be your main pipeline:

HTTP request
    │
    ▼
Authentication
    │
    ▼
Tenant / quota check
    │
    ▼
Request validation
    │
    ▼
Request ID
    │
    ▼
Canonical normalization
    │
    ▼
Token estimation
    │
    ▼
Capability requirements
    │
    ▼
Model resolution
    │
    ▼
Provider selection
    │
    ▼
Circuit-breaker check
    │
    ▼
Provider adapter
    │
    ▼
Streaming / generation
    │
    ├───────────────┐
    │               │
    ▼               ▼
tool call        text
    │
    ▼
canonical events
    │
    ▼
serializer
    │
    ▼
OpenAI/SSE response
    │
    ▼
usage + metrics
69. Recommended database architecture

Use PostgreSQL for durable state:

tenants
users
api_keys

providers
provider_credentials
models
model_capabilities
model_routes

requests
request_attempts

usage
billing

health_metrics
routing_events

Redis:

rate limits
distributed locks
short-lived request state
circuit breakers
provider health
idempotency

Object storage:

long-running traces
optional prompt/response archives
evaluation datasets
70. Deployment

For production:

                    Load Balancer
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        API instance             API instance
             │                       │
             └───────────┬───────────┘
                         │
                       Redis
                         │
                     Postgres
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        provider workers      routing workers

Inference requests themselves can remain stateless.

71. Don't make the gateway depend on a single inference SDK

You can use an SDK internally, but don't let it define your architecture.

Otherwise:

SDK abstraction
      ↓
your abstraction
      ↓
provider abstraction

becomes redundant and difficult to debug.

Your own canonical protocol should remain the source of truth.

72. The most important test suite

I would make these mandatory CI tests:

[Protocol]
□ OpenAI request validates
□ OpenAI response validates
□ /models works
□ SSE works

[Tools]
□ one tool
□ two tools
□ parallel tools
□ streamed tools
□ tool results
□ tool errors
□ malformed tool arguments

[Streaming]
□ text deltas
□ UTF-8
□ tool deltas
□ usage
□ final event
□ disconnect
□ cancellation

[Context]
□ normal context
□ exact limit
□ over limit
□ huge tool schemas
□ huge tool output

[Failures]
□ 429
□ 500
□ 502
□ 503
□ timeout
□ connection reset

[Routing]
□ primary
□ fallback
□ circuit breaker
□ capability filtering

[Security]
□ invalid key
□ revoked key
□ quota
□ prompt logging disabled
□ secrets never exposed

[Billing]
□ input tokens
□ output tokens
□ cached tokens
□ failed request accounting
73. Then run actual client tests

This is mandatory.

Install the clients and configure:

Base URL:
https://your-inference.com/v1

API key:
your-key

Then test:

OpenCode

OpenCode supports custom OpenAI-compatible providers and custom base URLs, so this is one of the most important compatibility targets.

Cline

Test:

read
write
edit
terminal
search
multi-step agent
Roo

Same.

Continue

Test:

chat
edit
autocomplete if applicable
74. OpenCode-specific compatibility

Pay particular attention to:

/v1/models
/v1/chat/completions
/v1/responses
streaming
tools
tool results
context limits
reasoning
custom baseURL

OpenCode's provider documentation explicitly supports custom base URLs and custom OpenAI-compatible providers.

It also has a server/client architecture with an HTTP API, so if you want to integrate at that layer, test the HTTP behavior separately from model inference.

75. A practical MVP

Don't build everything simultaneously.

Phase 1
□ API auth
□ /v1/models
□ /v1/chat/completions
□ streaming
□ tools
□ tool results
□ OpenAI-compatible provider
□ PostgreSQL
□ Redis
□ logging

Get:

Cline → your gateway → provider

working.

Phase 2
□ Anthropic
□ Gemini
□ OpenRouter
□ model registry
□ capability registry
□ retries
□ fallbacks
□ circuit breakers
□ usage
□ billing
Phase 3
□ /v1/responses
□ reasoning
□ structured outputs
□ vision
□ prompt caching
□ provider routing
□ health scoring
□ automatic fallback
Phase 4
□ 1000-session E2E tests
□ chaos testing
□ client compatibility suite
□ load testing
□ security testing
□ automatic model capability probing
76. The "OpenRouter-level" architecture

If you're specifically trying to build something resembling OpenRouter rather than merely an API proxy, I'd use this:

                    ┌───────────────────┐
                    │     CLIENTS       │
                    │ Cline/OpenCode... │
                    └─────────┬─────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ API COMPATIBILITY    │
                   │ OpenAI Chat          │
                   │ OpenAI Responses     │
                   │ SSE                  │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ CANONICAL IR         │
                   │ Messages             │
                   │ Tools                │
                   │ Reasoning            │
                   │ Structured output    │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ CAPABILITY ENGINE    │
                   │ tools?               │
                   │ vision?              │
                   │ context?             │
                   │ reasoning?           │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ ROUTER               │
                   │ quality              │
                   │ latency              │
                   │ availability         │
                   │ cost                 │
                   └──────────┬───────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
           Provider A     Provider B     Provider C
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                   ┌──────────────────────┐
                   │ EVENT NORMALIZER     │
                   │ text                 │
                   │ tools                │
                   │ reasoning            │
                   │ usage                │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ RELIABILITY ENGINE   │
                   │ retry                │
                   │ fallback             │
                   │ circuit breaker      │
                   │ cancellation         │
                   └──────────┬───────────┘
                              │
                              ▼
                         CLIENT STREAM

That is the architecture I'd recommend.

77. Final "100% compatible" checklist

If I were signing off your inference engine, I would require every item below:

API
 OpenAI /v1/chat/completions
 OpenAI /v1/responses
 /v1/models
 SSE
 non-streaming
 authentication
 request IDs
 proper HTTP errors
Messages
 system
 developer
 user
 assistant
 tool
 multimodal content
 tool results
 multiple tool calls
Tools
 tool definitions
 JSON schema
 tool choice
 auto
 forced tool
 parallel calls
 streamed calls
 stable IDs
 malformed argument handling
Streaming
 text deltas
 tool deltas
 reasoning deltas
 usage
 finish reason
 [DONE]
 cancellation
 disconnect handling
 UTF-8 correctness
 backpressure
Models
 model registry
 aliases
 capabilities
 context limits
 output limits
 supported parameters
 health status
Providers
 OpenAI
 Anthropic
 Gemini
 OpenRouter
 generic OpenAI-compatible
 local/vLLM
 provider-specific adapters
Reliability
 retries
 exponential backoff
 jitter
 deadlines
 circuit breaker
 health-aware routing
 fallback
 idempotency
 cancellation
 duplicate prevention
Agent reliability
 tool-call preservation
 tool-result preservation
 multi-turn conversations
 context management
 large tool schemas
 large tool outputs
 agent E2E tests
 Cline test
 OpenCode test
 Roo test
 Continue test
Production
 Redis
 PostgreSQL
 secrets manager
 rate limits
 quotas
 concurrency control
 billing
 usage tracking
 OpenTelemetry
 metrics
 alerting
 audit logs
 security tests
Certification
 1,000+ agent sessions
 provider failure injection
 stream interruption tests
 malformed provider responses
 load test
 long-context test
 tool-call stress test
 fallback test
 client compatibility test
The single most important design rule

If you remember only one thing:

             DON'T BUILD THIS

Cline
  ↓
OpenAI API
  ↓
provider

Build:

Cline / OpenCode / Roo
          ↓
    Compatibility API
          ↓
    Canonical IR
          ↓
 Capability + Routing Engine
          ↓
   Provider Adapter
          ↓
 OpenAI / Anthropic / Gemini / ...
          ↓
 Canonical Event Stream
          ↓
 Client-specific serializer

That separation is what lets you add providers without continually breaking agent compatibility.

And don't optimize for "the request returned 200." Optimize for:

The coding agent can perform a 20-step tool-using task, across streaming, failures, retries, context pressure, provider failover, and cancellation, without getting confused or corrupting the conversation.

That's the real definition of a reliable inference engine for Cline/OpenCode-class agents. OpenCode's own architecture—provider abstraction, model capability/limit metadata, custom OpenAI-compatible providers, tools, and HTTP server/client APIs—is a useful reference point for the interfaces you need to reproduce.