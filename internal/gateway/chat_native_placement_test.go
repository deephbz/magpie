package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

func TestChatNativeModelBoundary(t *testing.T) {
	for _, c := range []struct {
		model string
		want  bool
	}{
		{"claude-fable-5", true}, {"claude-fable-5-1", true}, {"claude-mythos-5-1", true},
		{"claude-opus-4-8", true}, {"claude-opus-4.8", true}, {"claude-opus-5", true}, {"claude-opus-6-0", true},
		{"claude-sonnet-5-5", true}, {"claude-haiku-5-5", true}, {"claude-sonnet-6", true},
		{"claude-opus-4-7", false}, {"claude-opus-4-9", false}, {"claude-sonnet-5", false}, {"claude-haiku-5-4", false}, {"claude-fable-4-9", false}, {"unknown", false},
	} {
		t.Run(c.model, func(t *testing.T) {
			r, err := parse(provider.Chat, []byte(`{"messages":[{"role":"user","content":"hi"},{"role":"system","content":"late"}]}`))
			if err != nil {
				t.Fatal(err)
			}
			body, err := build(provider.Anthropic, r, c.model, "api.anthropic.com", false)
			if err != nil {
				t.Fatal(err)
			}
			if got := hasSystemMessage(body); got != c.want {
				t.Fatalf("native=%v want=%v body=%s", got, c.want, body)
			}
		})
	}
}

func TestChatNativePlacementAfterToolPairing(t *testing.T) {
	r, err := parse(provider.Chat, []byte(`{"messages":[
 {"role":"user","content":"run"},
 {"role":"assistant","tool_calls":[{"id":"call1","type":"function","function":{"name":"read","arguments":"{}"}}]},
 {"role":"system","content":"new instructions"},
 {"role":"tool","tool_call_id":"call1","content":"real result"},
 {"role":"assistant","content":"done"},
 {"role":"user","content":"continue"}
 ]}`))
	if err != nil {
		t.Fatal(err)
	}
	body, err := build(provider.Anthropic, r, "claude-opus-5-5", "api.anthropic.com", false)
	if err != nil {
		t.Fatal(err)
	}
	msgs := gjson.GetBytes(body, "messages").Array()
	if len(msgs) != 6 || msgs[2].Get("role").String() != "user" || msgs[2].Get("content.0.type").String() != "tool_result" || msgs[2].Get("content.0.content").String() != "real result" || msgs[3].Get("role").String() != "system" || msgs[4].Get("role").String() != "assistant" || bytes.Contains(body, []byte(interruptedTool)) {
		t.Fatalf("update interrupted the tool exchange: %s", body)
	}
}

func TestChatNativeUnplaceableFoldsWholeRequest(t *testing.T) {
	for _, end := range []string{`{"role":"system","content":"third"}`, `{"role":"system","content":"third"},{"role":"user","content":""}`} {
		input := `{"messages":[{"role":"system","content":"first"},{"role":"user","content":"hi"},{"role":"system","content":"second"},{"role":"assistant","content":"ok"},` + end + `]}`
		r, err := parse(provider.Chat, []byte(input))
		if err != nil {
			t.Fatal(err)
		}
		body, err := build(provider.Anthropic, r, "claude-opus-5-5", "api.anthropic.com", false)
		if err != nil {
			t.Fatal(err)
		}
		if hasSystemMessage(body) || gjson.GetBytes(body, "system.0.text").String() != "first\n\nsecond\n\nthird" {
			t.Fatalf("unplaceable update did not fold the whole ordered request: %s", body)
		}
	}
}

func TestChatNativeLargeUpdateGuardPersists(t *testing.T) {
	for _, size := range []int{16, 32} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			var tools []map[string]any
			for i := 0; i < size; i++ {
				tools = append(tools, map[string]any{"type": "function", "function": map[string]any{"name": fmt.Sprintf("added_%d", i), "parameters": map[string]any{"type": "object"}}})
			}
			initial := []map[string]any{{"role": "system", "content": "stable rules"}, {"role": "user", "content": "start"}}
			update := map[string]any{"role": "system", "tools": tools}
			var previous []byte
			for stage := 0; stage < 4; stage++ {
				msgs := append([]map[string]any(nil), initial...)
				if stage > 0 {
					msgs = append(msgs, update)
				}
				if stage > 1 {
					msgs = append(msgs, map[string]any{"role": "assistant", "content": "added"}, map[string]any{"role": "user", "content": "use it"}, map[string]any{"role": "assistant", "content": "used"}, map[string]any{"role": "user", "content": "next"})
				}
				if stage > 2 {
					msgs = append(msgs, map[string]any{"role": "assistant", "content": "done"}, map[string]any{"role": "user", "content": "again"})
				}
				input, _ := json.Marshal(map[string]any{"messages": msgs, "tools": []map[string]any{{"type": "function", "function": map[string]any{"name": "initial", "parameters": map[string]any{"type": "object"}}}}})
				r, err := parse(provider.Chat, input)
				if err != nil {
					t.Fatal(err)
				}
				body, err := build(provider.Anthropic, r, "claude-opus-5-5", "api.anthropic.com", false)
				if err != nil {
					t.Fatal(err)
				}
				rendered := gjson.GetBytes(body, "messages").Array()
				prefix := []byte(rendered[0].Raw)
				if stage > 0 && !bytes.Equal(prefix, previous) {
					t.Fatalf("stage %d rewrote the pre-update cache prefix:\n%s\n%s", stage, previous, prefix)
				}
				previous = prefix
				if !rendered[0].Get("content.0.cache_control").Exists() {
					t.Fatalf("stage %d lost the persistent pre-update mark: %s", stage, body)
				}
				if stage > 1 && !rendered[len(rendered)-1].Get("content.0.cache_control").Exists() {
					t.Fatalf("stage %d lost its tail mark: %s", stage, body)
				}
				for _, m := range rendered {
					if m.Get("role").String() == "system" && strings.Contains(m.Raw, "cache_control") {
						t.Fatalf("update was marked: %s", body)
					}
				}
				if bytes.Count(body, []byte(`"cache_control"`)) > 4 {
					t.Fatalf("too many marks: %s", body)
				}
			}
		})
	}
}

func TestChatNativeTailAndLatestGuard(t *testing.T) {
	input := `{"messages":[{"role":"system","content":"rules"},{"role":"user","content":"one"},{"role":"system","content":"update one"},{"role":"assistant","content":"done one"},{"role":"user","content":"two"},{"role":"system","content":"update two"},{"role":"assistant","content":"done two"},{"role":"user","content":"three"},{"role":"system","content":"update three"}]}`
	r, err := parse(provider.Chat, []byte(input))
	if err != nil {
		t.Fatal(err)
	}
	body, err := build(provider.Anthropic, r, "claude-opus-5-5", "api.anthropic.com", false)
	if err != nil {
		t.Fatal(err)
	}
	msgs := gjson.GetBytes(body, "messages").Array()
	var marked []string
	for _, m := range msgs {
		for _, b := range m.Get("content").Array() {
			if b.Get("cache_control").Exists() {
				marked = append(marked, b.Get("text").String())
			}
		}
	}
	if strings.Join(marked, ",") != "two,three" {
		t.Fatalf("wanted latest non-trailing guard and pre-trailing tail, got %v: %s", marked, body)
	}
}
