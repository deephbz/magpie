package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"regexp"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

const inlineToolsBeta = "inline-tools-2026-09-15"

type nativeChatKey struct{}

var errChatUpdatesFold = errors.New("Chat updates require an IR fold")

func (s *Server) chatUpdatesFit(p provider.Provider, model string) bool {
	return s.fits(p.ID, chatUpdateRefusedKey(p, model), provider.Anthropic) &&
		s.fits(p.ID, shapeTurnSystem, provider.Anthropic) &&
		s.fits(p.ID, shapeDeferred, provider.Anthropic) &&
		s.fits(p.ID, betaKey(inlineToolsBeta), provider.Anthropic)
}

func nativeChatRequest(ctx context.Context) bool {
	native, _ := ctx.Value(nativeChatKey{}).(bool)
	return native
}

func chatUpdateRefusedKey(p provider.Provider, model string) string {
	return "shape:chat-updates\x00" + p.Base(provider.Anthropic) + "\x00" + model
}

func hasSystemMessage(body []byte) bool {
	for _, m := range gjson.GetBytes(body, "messages").Array() {
		if m.Get("role").String() == "system" {
			return true
		}
	}
	return false
}

func hasToolAddition(body []byte) bool {
	for _, m := range gjson.GetBytes(body, "messages").Array() {
		for _, b := range m.Get("content").Array() {
			if b.Get("type").String() == "tool_addition" {
				return true
			}
		}
	}
	return false
}

var chatCapabilityRefused = regexp.MustCompile("(?i)(?:unknown variant [`'\\\"](?:system|tool_addition|tool_definition)[`'\\\"]|(?:system (?:role|messages?)|role [`'\\\"]?system[`'\\\"]?|tool_addition|tool_definition)[^.;\\n]{0,60}(?:not supported|unsupported))")

// Only a capability rejection permits rebuilding from the IR. Placement,
// tool schemas, credentials and quota errors do not change shape memory.
func refusesChatUpdates(status int, body []byte) bool {
	if status != http.StatusBadRequest && status != http.StatusUnprocessableEntity {
		return false
	}
	message := gjson.GetBytes(body, "error.message").String()
	if message == "" {
		message = string(body)
	}
	lower := strings.ToLower(message)
	for _, word := range []string{"schema", "position", "placement", "immediately", "must follow", "must precede", "first message", "last message"} {
		if strings.Contains(lower, word) {
			return false
		}
	}
	if chatCapabilityRefused.MatchString(message) || strings.Contains(lower, "does not support mid-conversation system messages") {
		return true
	}
	if turnRoleRefused.MatchString(message) && strings.Contains(message, "Input should be 'user' or 'assistant'") {
		return true
	}
	return unexpectedBetas.MatchString(message) && strings.Contains(message, inlineToolsBeta)
}

var inlineClaudeFamily = regexp.MustCompile(`(?:^|[^a-z0-9])(?:claude-)?(fable|mythos|opus|sonnet|haiku)-[0-9]`)

// Future versions follow the documented family floor. A capability refusal
// disables native updates for that endpoint and resolved model.
func nativeChatModel(model string) bool {
	family := inlineClaudeFamily.FindStringSubmatch(strings.ToLower(model))
	major, minor, ok := claudeVersionOf(model)
	if !ok || family == nil {
		return false
	}
	switch family[1] {
	case "fable", "mythos":
		return major >= 5
	case "opus":
		return major >= 5 || major == 4 && minor == 8
	case "sonnet", "haiku":
		return major > 5 || major == 5 && minor >= 5
	}
	return false
}

func nativeChatRoute(model, host string) bool {
	return host == "api.anthropic.com" && nativeChatModel(model)
}

func anthropicTool(t Tool) map[string]any {
	schema := t.Schema
	if len(schema) == 0 {
		schema = json.RawMessage(`{"type":"object","properties":{}}`)
	} else {
		schema = objectSchema(schema)
	}
	return map[string]any{"name": t.Name, "description": t.Description, "input_schema": schema}
}

func chatSystemBlocks(m Message) []aBlock {
	var blocks []aBlock
	for _, p := range m.Parts {
		if p.Kind == Text && strings.TrimSpace(p.Text) != "" {
			blocks = append(blocks, aBlock{Type: "text", Text: p.Text})
		}
	}
	for _, t := range m.Tools {
		tool, _ := json.Marshal(map[string]any{"type": "tool_definition", "definition": anthropicTool(t)})
		blocks = append(blocks, aBlock{Type: "tool_addition", Tool: tool})
	}
	return blocks
}

// insertChatUpdates runs after tool pairing and empty-message removal. The
// held instructions precede the next rendered assistant turn, or the end.
func insertChatUpdates(msgs, trailing []aMessage) ([]aMessage, bool) {
	out := make([]aMessage, 0, len(msgs))
	insert := func(updates []aMessage) bool {
		if len(updates) == 0 {
			return true
		}
		if len(out) == 0 || out[len(out)-1].Role != "user" {
			return false
		}
		m := aMessage{Role: "system"}
		for _, update := range updates {
			m.Content = append(m.Content, update.Content...)
		}
		out = append(out, m)
		return true
	}
	for _, m := range msgs {
		if !insert(m.before) {
			return nil, false
		}
		m.before = nil
		out = append(out, m)
	}
	if !insert(trailing) {
		return nil, false
	}
	return out, true
}

// markChatUpdates never marks an update. The last pre-update block stays
// discoverable after a large update moves beyond the cache's lookback.
func markChatUpdates(msgs []aMessage) {
	markBefore := func(end int) {
		for i := end - 1; i >= 0; i-- {
			if msgs[i].Role == "system" {
				continue
			}
			for j := len(msgs[i].Content) - 1; j >= 0; j-- {
				b := &msgs[i].Content[j]
				if b.Type != "thinking" && b.Type != "redacted_thinking" {
					b.CacheControl = ephemeral
					return
				}
			}
		}
	}
	markBefore(len(msgs))
	for i := len(msgs) - 2; i >= 0; i-- {
		if msgs[i].Role == "system" {
			markBefore(i)
			break
		}
	}
}
