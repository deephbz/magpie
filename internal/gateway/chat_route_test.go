package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

func TestChatUpdateLeavesPassthrough(t *testing.T) {
	fresh(t)
	const ordinary = `{"model":"m","stream":true,"messages":[{"role":"system","content":"rules"},{"role":"user","content":"hi"}]}`
	for _, c := range []struct {
		name, body string
		updated    bool
	}{
		{"ordinary", ordinary, false}, {"remote magpie", chatSystemUpdate, false}, {"Pi pair", chatSystemUpdate, true},
		{"leading tools", `{"model":"m","messages":[{"role":"system","content":"rules","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]},{"role":"user","content":"hi"}]}`, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := New()
			p := provider.Provider{ID: "chat-route", Key: "synthetic-key", Chat: "https://chat.example/v1", Models: []string{"m"}}
			if c.name == "remote magpie" {
				p.Preset = provider.RemoteMagpiePreset
			}
			calls := 0
			s.client = &http.Client{Transport: countTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				b, _ := io.ReadAll(r.Body)
				if c.name == "remote magpie" {
					var got, want any
					// The existing Chat relay normalizes developer to system.
					expected := strings.ReplaceAll(c.body, `"role":"developer"`, `"role":"system"`)
					if json.Unmarshal(b, &got) != nil || json.Unmarshal([]byte(expected), &want) != nil || !reflect.DeepEqual(got, want) {
						t.Errorf("remote magpie lost the original Chat shape: %s", b)
					}
				} else if !c.updated {
					if string(b) != c.body {
						t.Errorf("ordinary passthrough bytes changed: %s", b)
					}
				} else {
					found := false
					for _, tool := range gjson.GetBytes(b, "tools").Array() {
						if tool.Get("function.name").String() == "lookup" {
							found = true
						}
					}
					if !found {
						t.Errorf("Chat upstream lost added tool: %s", b)
					}
					for _, m := range gjson.GetBytes(b, "messages").Array() {
						if m.Get("tools").Exists() {
							t.Errorf("message tools were not folded: %s", b)
						}
					}
				}
				return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader("data: {\"id\":\"r\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))}, nil
			})}
			w := httptest.NewRecorder()
			status, msg := s.attempt(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(c.body)), provider.Chat, p, "m", []byte(c.body), &Call{})
			if status != 200 || calls != 1 {
				t.Fatalf("status=%d calls=%d message=%s", status, calls, msg)
			}
		})
	}
}

func TestChatFoldPrecedesFormatInstruction(t *testing.T) {
	fresh(t)
	s := New()
	p := provider.Provider{ID: "format-order", Key: "synthetic-key", Chat: "https://chat.example/v1"}
	s.markUnfit(p.ID, formatRefused("m"), provider.Chat)
	r, err := parse(provider.Chat, []byte(chatSystemUpdate))
	if err != nil {
		t.Fatal(err)
	}
	r.Format = &Format{Type: "json_object"}
	want := r.hoisted().inSystem().System
	s.client = &http.Client{Transport: countTransport(func(w *http.Request) (*http.Response, error) {
		b, _ := io.ReadAll(w.Body)
		if got := gjson.GetBytes(b, "messages.0.content").String(); got != want {
			t.Errorf("format instruction order=%q want=%q", got, want)
		}
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
	})}
	res, _, err := s.forwardTranslated(context.Background(), p, provider.Chat, r, "m", nil)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

func TestChatUpdatesReachVertex(t *testing.T) {
	r, err := parse(provider.Chat, []byte(chatSystemUpdate))
	if err != nil {
		t.Fatal(err)
	}
	body, err := buildVertex(r, "gemini-3.1-pro-preview")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(body, []byte(`"name":"lookup"`)) || !bytes.Contains(body, []byte("updated")) {
		t.Fatalf("Vertex lost the fold: %s", body)
	}
}
