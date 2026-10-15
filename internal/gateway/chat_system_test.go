package gateway

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// Pi sends an addition without content, then a separate instruction update.
const chatSystemUpdate = `{"model":"m","messages":[
 {"role":"system","content":"initial"},
 {"role":"developer","content":[{"type":"text","text":"leading"}]},
 {"role":"user","content":"hello"},
 {"role":"assistant","content":"ready"},
 {"role":"system","tools":[{"type":"function","function":{"name":"lookup","description":"look up","parameters":{"type":"object"},"strict":true}}]},
 {"role":"developer","content":"updated"},
 {"role":"user","content":"use lookup"}
],"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}]}`

func TestChatSystemUpdateKeepsItsPlace(t *testing.T) {
	r, err := parseChat([]byte(chatSystemUpdate))
	if err != nil {
		t.Fatal(err)
	}
	if r.System != "initial\n\nleading" || len(r.Messages) != 4 {
		t.Fatalf("later instructions lost their place: system=%q messages=%+v", r.System, r.Messages)
	}
	if r.Messages[2].Role != "system" || !r.Messages[2].ChatSystem || len(r.Messages[2].Parts) != 2 || r.Messages[2].Parts[0].Text != "" || r.Messages[2].Parts[1].Text != "updated" {
		t.Fatalf("system updates = %+v", r.Messages[2:4])
	}
	if len(r.Tools) != 1 || r.Tools[0].Name != "read" {
		t.Fatalf("initial tools changed: %+v", r.Tools)
	}
}

func TestChatSystemUpdateReachesUpstreams(t *testing.T) {
	r, err := parse(provider.Chat, []byte(chatSystemUpdate))
	if err != nil {
		t.Fatal(err)
	}
	r = r.hoisted()
	if r.System != "initial\n\nleading\n\n\n\nupdated" || len(r.Messages) != 3 {
		t.Fatalf("fallback did not hoist the updates: system=%q messages=%+v", r.System, r.Messages)
	}
	if len(r.Tools) != 2 || r.Tools[1].Name != "lookup" || !r.Tools[1].Strict {
		t.Fatalf("added function lost: %+v", r.Tools)
	}
	for name, body := range systemTestOutputs(t, r) {
		if !bytes.Contains(body, []byte("updated")) {
			t.Errorf("%s lost the instructions: %s", name, body)
		}
		if name == "bridge" {
			tools := bridgeTools(r)
			if len(tools) != 2 || tools[1].Name != "lookup" {
				t.Errorf("MCP tools lost the addition: %+v", tools)
			}
			continue
		}
		var out struct {
			Tools []json.RawMessage `json:"tools"`
		}
		if err := json.Unmarshal(body, &out); err != nil {
			t.Fatal(err)
		}
		if len(out.Tools) != 2 || !bytes.Contains(out.Tools[1], []byte(`"name":"lookup"`)) || !bytes.Contains(out.Tools[1], []byte(`"type":"object"`)) {
			t.Errorf("%s lost the function definition: %s", name, body)
		}
	}
}

var systemIdentityInputs = map[provider.Protocol]string{
	provider.Chat:      `{"model":"m","messages":[{"role":"system","content":"rules"},{"role":"developer","content":"more rules"},{"role":"user","content":"read"},{"role":"assistant","tool_calls":[{"id":"call1","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call1","content":"done"}],"tools":[{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{}},"strict":true}}],"temperature":0.5,"max_tokens":256}`,
	provider.Responses: `{"model":"m","instructions":"rules","input":[{"role":"user","content":"read"},{"type":"function_call","call_id":"call1","name":"read","arguments":"{}"},{"type":"function_call_output","call_id":"call1","output":"done"}],"tools":[{"type":"function","name":"read","description":"Read a file","parameters":{"type":"object","properties":{}},"strict":true}],"temperature":0.5,"max_output_tokens":256}`,
	provider.Anthropic: `{"model":"m","system":"rules","messages":[{"role":"user","content":"read"},{"role":"assistant","content":[{"type":"tool_use","id":"call1","name":"read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call1","content":"done"}]},{"role":"system","content":"native notice"}],"tools":[{"name":"read","description":"Read a file","input_schema":{"type":"object","properties":{}}}],"temperature":0.5,"max_tokens":256}`,
}

func systemTestOutputs(t *testing.T, r *Request) map[string][]byte {
	t.Helper()
	prompt, err := renderClaudePrompt(r)
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(prompt)
	if err != nil {
		t.Fatal(err)
	}
	return map[string][]byte{
		"anthropic": anthropicBody(r, "claude-sonnet-4-5", false),
		"chat":      buildChat(r, "gpt-4.1", "api.openai.com", false),
		"responses": buildResponses(r, "gpt-4.1", "api.openai.com", false),
		"bridge":    b,
	}
}

func systemIdentityCases() map[string]struct {
	proto provider.Protocol
	body  string
} {
	out := map[string]struct {
		proto provider.Protocol
		body  string
	}{}
	for client, input := range systemIdentityInputs {
		out[string(client)] = struct {
			proto provider.Protocol
			body  string
		}{client, input}
	}
	out["anthropic-no-notice"] = struct {
		proto provider.Protocol
		body  string
	}{provider.Anthropic,
		strings.Replace(systemIdentityInputs[provider.Anthropic], `,{"role":"system","content":"native notice"}`, "", 1)}
	return out
}

func TestChatSystemFoldEdges(t *testing.T) {
	for _, c := range []struct{ name, messages, want string }{
		{"no leading system", `{"role":"user","content":"hi"},{"role":"system","content":"late"}`, "late"},
		{"empty leading system", `{"role":"system","content":""},{"role":"user","content":"hi"},{"role":"developer","content":"late"}`, "\n\nlate"},
		{"empty later system", `{"role":"system","content":"first"},{"role":"user","content":"hi"},{"role":"system"}`, "first\n\n"},
		{"several updates", `{"role":"system","content":"first"},{"role":"user","content":"hi"},{"role":"system","content":"second"},{"role":"assistant","content":"ok"},{"role":"developer","content":[{"type":"text","text":"third"},{"type":"text","text":" part"}]}`, "first\n\nsecond\n\nthird part"},
	} {
		t.Run(c.name, func(t *testing.T) {
			r, err := parseChat([]byte(`{"messages":[` + c.messages + `]}`))
			if err != nil {
				t.Fatal(err)
			}
			before, _ := json.Marshal(r)
			out := r.hoisted()
			if out.System != c.want {
				t.Fatalf("system = %q, want %q", out.System, c.want)
			}
			after, _ := json.Marshal(r)
			if !bytes.Equal(before, after) {
				t.Fatal("fold changed the positioned request")
			}
			if !reflect.DeepEqual(out, out.hoisted()) {
				t.Fatal("fold changed on a second application")
			}
		})
	}
}

func TestChatSystemToolsReplaceByName(t *testing.T) {
	r, err := parseChat([]byte(`{"tools":[{"type":"function","function":{"name":"read","description":"old"}}],"messages":[
	 {"role":"user","content":"hi"},
	 {"role":"system","tools":[{"type":"web_search"},{"type":"function","function":{"name":"read","description":"new","parameters":{"type":"object"},"strict":true}},{"type":"function","function":{"name":"lookup","description":"first"}}]},
	 {"role":"system","tools":[{"type":"function","function":{"name":"lookup","description":"last"}}]}
	]}`))
	if err != nil {
		t.Fatal(err)
	}
	out := r.hoisted()
	if len(out.Tools) != 2 || out.Tools[0].Description != "new" || !out.Tools[0].Strict || out.Tools[1].Description != "last" || out.WebSearch {
		t.Fatalf("later definitions did not replace functions: %+v", out.Tools)
	}
	if r.Tools[0].Description != "old" || r.Messages[1].Tools[1].Description != "first" {
		t.Fatal("fold changed the source tools")
	}
}

// Reduced from Pi 1.2.0 RPC sessions against a loopback stub. Prompts,
// ordinary message text, descriptions and call IDs are synthetic.
func TestChatSystemPiWire(t *testing.T) {
	for _, name := range []string{"pi-flags-on-add", "pi-flags-on-remove", "pi-flags-off-add"} {
		data, err := os.ReadFile("testdata/" + name + ".jsonl")
		if err != nil {
			t.Fatal(err)
		}
		for i, body := range bytes.Split(bytes.TrimSpace(data), []byte("\n")) {
			r, err := parseChat(body)
			if err != nil {
				t.Fatal(err)
			}
			updates := 0
			for _, m := range r.Messages {
				if m.ChatSystem {
					updates++
				}
			}
			wantUpdates := 0
			if i > 0 && name == "pi-flags-on-add" {
				wantUpdates = 1
			}
			if i > 0 && name == "pi-flags-on-remove" {
				wantUpdates = 1
			}
			if updates != wantUpdates {
				t.Fatalf("%s request %d: updates=%d want=%d", name, i+1, updates, wantUpdates)
			}
			folded, err := parse(provider.Chat, body)
			if err != nil {
				t.Fatal(err)
			}
			folded = folded.hoisted()
			var names []string
			for _, tool := range folded.Tools {
				names = append(names, tool.Name)
			}
			want := []string{"wire_baseline"}
			if i > 0 {
				want = []string{"wire_baseline", "wire_new"}
				if name == "pi-flags-on-remove" {
					want = []string{"wire_new"}
				}
			}
			if !reflect.DeepEqual(names, want) {
				t.Fatalf("%s request %d: tools=%v want=%v", name, i+1, names, want)
			}
			if i > 0 && !strings.Contains(folded.System, "WIRE_SYSTEM_UPDATE") {
				t.Fatalf("%s request %d lost its system update", name, i+1)
			}
		}
	}
}

// These exact bodies were recorded from c7830731 before the parser change.
func TestChatSystemUnchangedBodies(t *testing.T) {
	data, err := os.ReadFile("testdata/chat_system_unchanged.json")
	if err != nil {
		t.Fatal(err)
	}
	var want map[string]map[string]string
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	for name, c := range systemIdentityCases() {
		r, err := parse(c.proto, []byte(c.body))
		if err != nil {
			t.Fatal(err)
		}
		for upstream, body := range systemTestOutputs(t, r) {
			if string(body) != want[name][upstream] {
				t.Errorf("%s -> %s differs from c7830731:\n%s", name, upstream, body)
			}
		}
	}
}
