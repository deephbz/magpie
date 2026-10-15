package gateway

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

// The rpm lookup occurs in forwardOnce, after native construction and before
// shape decorators. Inject the other request's learned refusal at that boundary.
type astraRefusalContext struct {
	context.Context
	once  sync.Once
	learn func()
}

func (c *astraRefusalContext) Value(key any) any {
	if _, ok := key.(rpmKey); ok {
		c.once.Do(c.learn)
	}
	return c.Context.Value(key)
}

func TestAstraNativeConcurrentRefusalKeepsDefinition(t *testing.T) {
	fresh(t)
	for _, key := range []string{shapeDeferred, shapeTurnSystem, betaKey(inlineToolsBeta)} {
		t.Run(key, func(t *testing.T) {
			s := New()
			p := provider.Provider{ID: "astra-native", Key: "synthetic-key", Anthropic: "https://api.anthropic.com", MaxRPM: 100}
			ctx := &astraRefusalContext{Context: context.Background(), learn: func() { s.markUnfit(p.ID, key, provider.Anthropic) }}
			s.client = &http.Client{Transport: countTransport(func(w *http.Request) (*http.Response, error) {
				b, _ := io.ReadAll(w.Body)
				if gjson.GetBytes(b, "tools.1.name").String() != "lookup" || hasSystemMessage(b) {
					t.Errorf("concurrent %s did not rebuild the complete fold: %s", key, b)
				}
				if strings.Contains(w.Header.Get("Anthropic-Beta"), inlineToolsBeta) {
					t.Error("fold retained the inline beta")
				}
				return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
			})}
			r, err := parse(provider.Chat, []byte(chatSystemUpdate))
			if err != nil {
				t.Fatal(err)
			}
			res, _, err := s.forwardTranslated(ctx, p, provider.Anthropic, r, "claude-opus-5-5", nil)
			if err != nil {
				t.Fatal(err)
			}
			res.Body.Close()
		})
	}
}

func TestAstraNativeRefusalMemoryScope(t *testing.T) {
	fresh(t)
	s := New()
	p := provider.Provider{ID: "astra-scope", Key: "synthetic-key", Anthropic: "https://api.anthropic.com"}
	calls := 0
	wantNative := false
	s.client = &http.Client{Transport: countTransport(func(w *http.Request) (*http.Response, error) {
		calls++
		b, _ := io.ReadAll(w.Body)
		status, body := 200, `{}`
		if calls == 1 {
			status = 400
			body = "{\"error\":{\"message\":\"messages[3].content: unknown variant `tool_addition`\"}}"
		} else if hasSystemMessage(b) != wantNative {
			t.Errorf("call %d native=%v want=%v", calls, hasSystemMessage(b), wantNative)
		}
		return &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	send := func(p provider.Provider, model string) {
		r, err := parse(provider.Chat, []byte(chatSystemUpdate))
		if err != nil {
			t.Fatal(err)
		}
		res, _, err := s.forwardTranslated(context.Background(), p, provider.Anthropic, r, model, nil)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		if res.StatusCode != 200 {
			t.Fatalf("status=%d", res.StatusCode)
		}
	}
	send(p, "claude-opus-5-5") // learns only this endpoint/model
	send(p, "claude-opus-5-5") // folds without another refusal
	wantNative = true
	send(p, "claude-opus-4-8")
	p.Anthropic = "https://api.anthropic.com/other"
	send(p, "claude-opus-5-5")
	p.Anthropic = "https://api.anthropic.com"
	p.ID = "astra-other"
	send(p, "claude-opus-5-5")
	if calls != 6 {
		t.Fatalf("calls=%d want=6", calls)
	}
}

func TestChatAddedToolsBeforeRouting(t *testing.T) {
	r, err := parse(provider.Chat, []byte(`{"messages":[{"role":"user","content":"hi"},{"role":"system","tools":[{"type":"function","function":{"name":"Bash","parameters":{"type":"object"}}}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	ag := provider.Provider{ID: "antigravity", Account: &provider.Account{Agent: "antigravity", User: "synthetic"}}
	if searchesFor(ag, provider.CodeAssist, "gemini-3.8-flash", r) {
		t.Error("Google search was selected beside an added function tool")
	}
	z := zenFreeTools(r)
	if z.stubs["bash"] || z.back["bash"] != "Bash" {
		t.Error("added Bash tool became a stub instead of a callable renamed tool")
	}
}

func TestAstraUnplaceableFoldFormatOrder(t *testing.T) {
	fresh(t)
	const input = `{"response_format":{"type":"json_object"},"messages":[{"role":"system","content":"rules"},{"role":"user","content":"hi"},{"role":"assistant","content":"done"},{"role":"system","content":"late"}]}`
	const want = "rules\n\nlate\n\nRespond with a single JSON object and nothing else."
	check := func(t *testing.T, body []byte, expected string) {
		t.Helper()
		if hasSystemMessage(body) {
			t.Fatalf("expected whole fold: %s", body)
		}
		if got := gjson.GetBytes(body, "system.0.text").String(); got != expected {
			t.Fatalf("format order got=%q want=%q", got, expected)
		}
	}
	for _, c := range []struct {
		name            string
		refused, schema bool
	}{
		{"builder", false, false},
		{"format refusal memory", true, false},
		{"schema refusal memory", true, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			r, err := parse(provider.Chat, []byte(input))
			if err != nil {
				t.Fatal(err)
			}
			expected := want
			if c.schema {
				r.Format = &Format{Type: "json_schema", Schema: []byte(`{"type":"object"}`)}
				expected = "rules\n\nlate\n\nRespond with a single JSON value and nothing else, matching this JSON schema:\n{\"type\":\"object\"}."
			}
			if !c.refused {
				body, err := build(provider.Anthropic, r, "claude-opus-5-5", "api.anthropic.com", false)
				if err != nil {
					t.Fatal(err)
				}
				check(t, body, expected)
				return
			}
			s := New()
			p := provider.Provider{ID: "format-order-native", Key: "synthetic-key", Anthropic: "https://api.anthropic.com"}
			s.markUnfit(p.ID, formatRefused("claude-opus-5-5"), provider.Anthropic)
			s.client = &http.Client{Transport: countTransport(func(w *http.Request) (*http.Response, error) {
				b, _ := io.ReadAll(w.Body)
				check(t, b, expected)
				return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
			})}
			res, _, err := s.forwardTranslated(context.Background(), p, provider.Anthropic, r, "claude-opus-5-5", nil)
			if err != nil {
				t.Fatal(err)
			}
			res.Body.Close()
		})
	}
}

func TestAstraRootRepairFoldedControl(t *testing.T) {
	fresh(t)
	s := New()
	p := provider.Provider{ID: "astra-root-control", Key: "synthetic-key", Chat: "https://api.x.ai/v1"}
	s.client = &http.Client{Transport: countTransport(func(w *http.Request) (*http.Response, error) {
		b, _ := io.ReadAll(w.Body)
		if gjson.GetBytes(b, "tools.0.function.parameters.anyOf").Exists() || gjson.GetBytes(b, "tools.0.function.parameters.type").String() != "object" {
			t.Errorf("control repair failed: %s", b)
		}
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
	})}
	r, err := parse(provider.Chat, []byte(astraUnionUpdate))
	if err != nil {
		t.Fatal(err)
	}
	res, _, err := s.forwardTranslated(context.Background(), p, provider.Chat, r.hoisted(), "grok-4", nil)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

const astraUnionUpdate = `{"messages":[{"role":"user","content":"hi"},{"role":"system","tools":[{"type":"function","function":{"name":"lookup","parameters":{"anyOf":[{"type":"object","properties":{"q":{"type":"string"}}},{"type":"object","properties":{"id":{"type":"integer"}}}]}}}]}]}`

func TestAstraFoldedToolsReceiveRootRepair(t *testing.T) {
	fresh(t)
	for _, proto := range []provider.Protocol{provider.Chat, provider.Responses} {
		t.Run(string(proto), func(t *testing.T) {
			s := New()
			p := provider.Provider{ID: "astra-root", Key: "synthetic-key", Chat: "https://api.x.ai/v1", Responses: "https://api.x.ai/v1"}
			s.client = &http.Client{Transport: countTransport(func(w *http.Request) (*http.Response, error) {
				b, _ := io.ReadAll(w.Body)
				path := "tools.0.parameters"
				if proto == provider.Chat {
					path = "tools.0.function.parameters"
				}
				if gjson.GetBytes(b, path+".anyOf").Exists() || gjson.GetBytes(b, path+".type").String() != "object" {
					t.Errorf("added tool bypassed root repair: %s", b)
				}
				return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
			})}
			r, err := parse(provider.Chat, []byte(astraUnionUpdate))
			if err != nil {
				t.Fatal(err)
			}
			res, _, err := s.forwardTranslated(context.Background(), p, proto, r, "grok-4", nil)
			if err != nil {
				t.Fatal(err)
			}
			res.Body.Close()
		})
	}
}

func TestAstraFoldedToolsRootRefusalRetries(t *testing.T) {
	fresh(t)
	s := New()
	p := provider.Provider{ID: "astra-root-retry", Key: "synthetic-key", Chat: "https://chat.example/v1"}
	calls := 0
	s.client = &http.Client{Transport: countTransport(func(w *http.Request) (*http.Response, error) {
		calls++
		b, _ := io.ReadAll(w.Body)
		status, body := 200, `{}`
		if gjson.GetBytes(b, "tools.0.function.parameters.anyOf").Exists() {
			status = 400
			body = `{"error":{"message":"tool parameter root must be an object type"}}`
		}
		return &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	r, err := parse(provider.Chat, []byte(astraUnionUpdate))
	if err != nil {
		t.Fatal(err)
	}
	res, _, err := s.forwardTranslated(context.Background(), p, provider.Chat, r, "m", nil)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != 200 || calls != 2 {
		t.Fatalf("root repair did not reach added tool: status=%d calls=%d", res.StatusCode, calls)
	}
}
