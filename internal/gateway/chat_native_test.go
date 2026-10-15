package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

func TestChatSystemPairIsOneUpdate(t *testing.T) {
	r, err := parse(provider.Chat, []byte(chatSystemUpdate))
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Messages) != 4 || !r.Messages[2].ChatSystem || len(r.Messages[2].Tools) != 1 {
		t.Fatalf("Pi's pair did not become one positioned update: %+v", r.Messages)
	}
	if got := r.hoisted().System; got != "initial\n\nleading\n\n\n\nupdated" {
		t.Fatalf("merged pair lost empty entry: %q", got)
	}
}

func TestChatNativeAnthropicKeepsThePrefix(t *testing.T) {
	data, err := os.ReadFile("testdata/pi-flags-on-add.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSpace(data), []byte("\n"))
	var bodies []map[string]json.RawMessage
	for _, line := range lines[:3] {
		r, err := parse(provider.Chat, line)
		if err != nil {
			t.Fatal(err)
		}
		body, err := build(provider.Anthropic, r, "claude-opus-5-5", "api.anthropic.com", false)
		if err != nil {
			t.Fatal(err)
		}
		var out map[string]json.RawMessage
		if err = json.Unmarshal(body, &out); err != nil {
			t.Fatal(err)
		}
		bodies = append(bodies, out)
	}
	for i := 1; i < len(bodies); i++ {
		if !bytes.Equal(bodies[0]["system"], bodies[i]["system"]) || !bytes.Equal(bodies[0]["tools"], bodies[i]["tools"]) {
			t.Fatalf("request %d rewrote the initial prefix: %s %s", i+1, bodies[i]["system"], bodies[i]["tools"])
		}
		var msgs []struct {
			Role    string                       `json:"role"`
			Content []map[string]json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(bodies[i]["messages"], &msgs); err != nil {
			t.Fatal(err)
		}
		found := 0
		for j, m := range msgs {
			if m.Role != "system" {
				continue
			}
			found++
			if j == 0 || msgs[j-1].Role != "user" || len(m.Content) != 2 {
				t.Fatalf("invalid update placement or blocks: %s", bodies[i]["messages"])
			}
			if string(m.Content[0]["type"]) != `"text"` || string(m.Content[1]["type"]) != `"tool_addition"` {
				t.Fatalf("update content=%s", bodies[i]["messages"])
			}
			for _, b := range m.Content {
				if b["cache_control"] != nil {
					t.Fatalf("cache mark on update: %s", bodies[i]["messages"])
				}
			}
		}
		if found != 1 {
			t.Fatalf("request %d has %d system updates: %s", i+1, found, bodies[i]["messages"])
		}
	}
}

func TestChatNativeBetaAndResolvedModel(t *testing.T) {
	fresh(t)
	for _, c := range []struct {
		name, wire, host, body string
		native, beta           bool
	}{
		{"alias native", "claude-opus-5-5", "api.anthropic.com", chatSystemUpdate, true, true},
		{"alias sonnet5", "claude-sonnet-5", "api.anthropic.com", chatSystemUpdate, false, false},
		{"other host", "claude-opus-5-5", "relay.example", chatSystemUpdate, false, false},
		{"text only", "claude-opus-5-5", "api.anthropic.com", `{"messages":[{"role":"user","content":"hi"},{"role":"system","content":"late"}]}`, true, false},
		{"ordinary", "claude-opus-5-5", "api.anthropic.com", `{"messages":[{"role":"user","content":"hi"}]}`, false, false},
		{"guard with decorator", "claude-opus-5-5", "api.anthropic.com", `{"messages":[{"role":"system","content":"rules"},{"role":"user","content":"hi"},{"role":"system","content":"update"},{"role":"assistant","content":"ok"},{"role":"user","content":"next"}]}`, true, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := New()
			p := provider.Provider{ID: "native-test", Key: "synthetic-key", Chat: "https://other.example/v1", Anthropic: "https://" + c.host}
			if c.name == "guard with decorator" {
				p.ExtraBody = provider.ExtraBody{"cache_control": json.RawMessage(`{"type":"ephemeral"}`)}
			}
			calls := 0
			s.client = &http.Client{Transport: countTransport(func(w *http.Request) (*http.Response, error) {
				calls++
				b, _ := io.ReadAll(w.Body)
				if got := bytes.Contains(b, []byte(`"role":"system"`)); got != c.native {
					t.Errorf("native=%v want=%v body=%s", got, c.native, b)
				}
				if got := strings.Contains(w.Header.Get("anthropic-beta"), "inline-tools-2026-09-15"); got != c.beta {
					t.Errorf("inline beta=%v want=%v headers=%v", got, c.beta, w.Header)
				}
				if !strings.Contains(w.Header.Get("anthropic-beta"), "synthetic-existing-beta") {
					t.Error("existing beta lost")
				}
				if c.name == "guard with decorator" && bytes.Count(b, []byte(`"cache_control"`)) != 4 {
					t.Errorf("final decorator/guard marks missing: %s", b)
				}
				if bytes.Count(b, []byte(`"cache_control"`)) > 4 {
					t.Errorf("too many final cache marks: %s", b)
				}
				return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
			})}
			r, err := parse(provider.Chat, []byte(c.body))
			if err != nil {
				t.Fatal(err)
			}
			ctx := context.WithValue(context.Background(), wiresKey{}, func() map[string]string { return map[string]string{"native-test/alias": c.wire} })
			res, _, err := s.forwardTranslated(ctx, p, provider.Anthropic, r, "alias", http.Header{"Anthropic-Beta": []string{"synthetic-existing-beta"}})
			if err != nil {
				t.Fatal(err)
			}
			res.Body.Close()
			if calls != 1 {
				t.Fatalf("calls=%d", calls)
			}
		})
	}
}

func TestChatNativeRefusalKeepsTheAddedTool(t *testing.T) {
	fresh(t)
	for _, c := range []struct {
		name, failure string
		status        int
		fold          bool
	}{
		{"capability", "messages[3].content: unknown variant `tool_addition`", 400, true},
		{"system capability", "messages.3.role: Input should be 'user' or 'assistant'", 400, true},
		{"placement", "messages.3.role: system message must immediately follow a user turn", 400, false},
		{"schema", "messages.3.content.1.tool.definition.input_schema: invalid schema", 400, false},
		{"unsupported schema", "messages.3.content.1.tool_definition.input_schema: unsupported schema keyword", 400, false},
		{"unsupported position", "system messages are not supported at this position", 400, false},
		{"model capability", "This model does not support mid-conversation system messages", 400, true},
		{"rate limit", "system messages are not supported", 429, false},
		{"auth", "system messages are not supported", 401, false},
		{"sampling", "temperature is deprecated for this model", 400, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := New()
			p := provider.Provider{ID: "native-test", Key: "synthetic-key", Anthropic: "https://api.anthropic.com"}
			calls := 0
			s.client = &http.Client{Transport: countTransport(func(w *http.Request) (*http.Response, error) {
				calls++
				b, _ := io.ReadAll(w.Body)
				status := c.status
				response, _ := json.Marshal(map[string]any{"error": map[string]any{"message": c.failure}})
				if c.fold && calls > 1 {
					status = 200
					response = []byte(`{}`)
					var out struct {
						Tools []struct {
							Name string `json:"name"`
						} `json:"tools"`
						System []aBlock `json:"system"`
					}
					if err := json.Unmarshal(b, &out); err != nil {
						t.Fatal(err)
					}
					if len(out.Tools) != 2 || out.Tools[1].Name != "lookup" || len(out.System) != 1 || out.System[0].Text != "initial\n\nleading\n\n\n\nupdated" || bytes.Contains(b, []byte(`"role":"system"`)) {
						t.Errorf("fallback lost meaning: %s", b)
					}
					if strings.Contains(w.Header.Get("anthropic-beta"), "inline-tools-2026-09-15") {
						t.Error("fold kept inline beta")
					}
				} else if !bytes.Contains(b, []byte(`"tool_addition"`)) {
					t.Errorf("native request was stripped: %s", b)
				}
				return &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader(response))}, nil
			})}
			for i := 0; i < 2; i++ {
				r, err := parse(provider.Chat, []byte(chatSystemUpdate))
				if err != nil {
					t.Fatal(err)
				}
				if c.name == "sampling" {
					temp := 0.5
					r.Temp = &temp
				}
				res, _, err := s.forwardTranslated(context.Background(), p, provider.Anthropic, r, "claude-opus-5-5", nil)
				if err != nil {
					t.Fatal(err)
				}
				res.Body.Close()
				want := c.status
				if c.fold {
					want = 200
				}
				if res.StatusCode != want {
					t.Errorf("status=%d want=%d", res.StatusCode, want)
				}
			}
			want := 2
			if c.fold {
				want = 3
			}
			if calls != want {
				t.Fatalf("calls=%d want=%d (capabilities alone should be remembered)", calls, want)
			}
			if !s.fits(p.ID, shapeTurnSystem, provider.Anthropic) || !s.fits(p.ID, shapeDeferred, provider.Anthropic) {
				t.Fatal("Chat request changed Anthropic-origin shape memory")
			}
		})
	}
}

func TestChatLeadingSystemAddsTools(t *testing.T) {
	r, err := parseChat([]byte(`{"tools":[{"type":"function","function":{"name":"read","description":"old"}}],"messages":[{"role":"system","content":"rules","tools":[{"type":"function","function":{"name":"read","description":"new"}},{"type":"function","function":{"name":"lookup"}}]},{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Tools) != 2 || r.Tools[0].Description != "new" || r.Tools[1].Name != "lookup" {
		t.Fatalf("leading tools lost: %+v", r.Tools)
	}
}
