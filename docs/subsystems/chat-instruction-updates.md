# Chat instruction updates in translation

A Chat client's leading system/developer messages form the initial system
prompt. Message-level function tools in that run join the initial tool list.
`parseChat` in [`chat.go`](../../internal/gateway/chat.go) keeps later updates
in place. Consecutive updates merge: text parts in order, then function tool
additions. Each original message keeps a text part, including empty text, so
the fallback preserves the original system-prompt separators.

`parse` returns this positioned IR. `Message.ChatSystem` distinguishes Chat
updates from Anthropic clients' system notices. Anthropic clients keep their
existing translation and passthrough behavior.

## Route selection

| Selected route | Chat update handling |
| --- | --- |
| Anthropic's own API host and a supported resolved model | Native system messages and inline tool definitions |
| Other Anthropic host or unsupported model | Whole-request fold |
| Chat, Responses, Gemini, Vertex and other backends | Whole-request fold |
| Claude subscription bridge | Explicit fold at its entry |
| Native placement fails, or the upstream rejects the capability | Whole-request fold |

The native predicate uses the selected endpoint host and the final upstream
model after alias resolution. Supported families are Fable and Mythos 5+, Opus
4.8 and 5+, and Sonnet and Haiku 5.5+. Future versions follow those floors
provisionally. Capability refusals disable native updates for that endpoint
and resolved model while the gateway runs.

A Chat request with later system/developer messages or message-level tools
leaves raw passthrough and takes translation on its provider's own API when
available. This includes Chat providers with native message-level tool support,
such as Moonshot Kimi K3. Per-provider native Chat passthrough is a follow-up.
Remote-magpie routes preserve the Chat update shape for downstream routing.
Other passthrough requests remain unchanged.

`Request.hoisted` in [`chat_system.go`](../../internal/gateway/chat_system.go)
returns a copy with all update texts joined into the initial system prompt,
including empty entries. It appends added functions and replaces earlier
definitions of the same name. The positioned request stays unchanged.
Fold routes select this view before tool-schema repairs and format instructions.
Requests without later system/developer messages and without message-level
`tools` retain their previous upstream bytes.

Message-level tool additions now reach the subscription bridge's tool list.
The previous parser lost those additions from Pi's flags-on pair.

## Native placement, cache marks and failures

[`anthropic.go`](../../internal/gateway/anthropic.go) renders ordinary messages,
removes empty content, merges adjacent user messages, and runs `pairToolBlocks`
before inserting Chat updates. Updates wait until just before the next rendered
assistant message or the end. An update must follow a user turn. If no legal
place exists, the whole request folds. Instructions keep their original order.

[`chat_native.go`](../../internal/gateway/chat_native.go) emits text followed
by `tool_addition` blocks containing `tool_definition`. Only a native request
with tool additions asks for `inline-tools-2026-09-15`; existing betas remain.
Text-only updates add no beta. No placeholder tool is inserted.

No update block gets a cache mark. A trailing update puts the tail mark on the
last eligible block before it. A non-trailing update also keeps a guard mark
before the most recent non-trailing update section. The system/tools prefix,
tail, and guard use at most three marks before other request decorators. Tests
count the final outgoing marks against Anthropic's limit of four.

A capability rejection returns to `forwardTranslated`, which rebuilds the
whole request through the fold. It never deletes the only definition of an
added tool. Unknown variants and explicit unsupported system/tool-addition
errors can be remembered. Placement errors, schema errors, rate limits, and
authentication failures are surfaced without capability memory. Native sends
skip `Provider.Retry` mending and the generic translated error retries. This
also surfaces unrelated sampling and format failures instead of interpreting
ambiguous update errors as permission to change the request.

Native sends bypass destructive generic shape decorators. If concurrent
capability learning or final beta filtering makes the native send unsuitable,
forwarding returns to the IR owner before transmission and rebuilds the fold.
Safe independent decorators still run. Anthropic-origin requests retain their
existing generic shape-refusal handling.

## Limits and verification

Pi stops anchoring tool additions after a removal or redeclaration. Its Chat
wire sends the current tools at the top level and has no removal operation.
The gateway forwards that current list. Later tool-set changes can invalidate
the tool prefix. The gateway does not infer removals from session history.

Live cache evidence covers sessions that start with at least one non-deferred
tool. Zero-tool starts remain unverified. A trailing update can cost another
cache write when the next turn follows it. Unit tests verify the large-update
prefix and persistent guard; they do not prove billed cache reuse.

`TestChatSystem*`, `TestChatNative*`, `TestChatUpdateLeavesPassthrough`, and
`TestChatUpdatesReachVertex` cover Pi's wire shapes, byte identity, route and
model selection, tool pairing, whole-fold fallback, refusal memory, and cache
marks. The protocol reference is Anthropic's
[mid-conversation system messages documentation](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages).
