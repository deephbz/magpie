package gateway

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/testenv"
)

// A turn sent fast on a model with fast mode, in Anthropic's words (speed
// "fast") or OpenAI's (service_tier "priority", withFast for a Chat
// agent such as Pi), runs Claude Code in fast mode; one at the usual speed
// or on a model without it runs it as before.
func TestClaudeSubscriptionFastRun(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("a shell script stands in for Claude Code")
	}
	dir := t.TempDir()
	script := `#!/bin/sh
for a in "$@"; do echo "$a"; done > ` + dir + `/args.$$
while read -r line; do
  echo '{"type":"stream_event","event":{"type":"message_start","message":{"id":"m","model":"claude-opus-5-5","usage":{"input_tokens":1}}}}'
  echo '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}}'
  echo '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}}'
  echo '{"type":"stream_event","event":{"type":"message_stop"}}'
  echo '{"type":"result","subtype":"success","is_error":false,"result":""}'
done
`
	testenv.Program(t, filepath.Join(dir, "claude"), script)
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))

	s := New()
	p := provider.Provider{ID: "claude", Account: &provider.Account{Agent: "claude", User: "u"}}
	for _, c := range []struct {
		proto provider.Protocol
		model string
		body  string
	}{
		{provider.Anthropic, "claude-opus-5-5", `{"model":"claude-opus-5-5","max_tokens":100,"speed":"fast","messages":[{"role":"user","content":"one"}]}`},
		{provider.Chat, "claude-opus-5-5", `{"model":"claude-opus-5-5","service_tier":"priority","messages":[{"role":"user","content":"two"}]}`},
		{provider.Anthropic, "claude-opus-5-5", `{"model":"claude-opus-5-5","max_tokens":100,"messages":[{"role":"user","content":"three"}]}`},
		{provider.Anthropic, "claude-sonnet-5", `{"model":"claude-sonnet-5","max_tokens":100,"speed":"fast","messages":[{"role":"user","content":"four"}]}`},
	} {
		rec := httptest.NewRecorder()
		var u Usage
		if code, msg := s.serveClaudeSubscription(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(c.body)), c.proto, p, c.model, []byte(c.body), &u); code != 200 {
			t.Fatalf("%d %s", code, msg)
		}
	}
	files, _ := filepath.Glob(filepath.Join(dir, "args.*"))
	fast := 0
	for _, f := range files {
		b, _ := os.ReadFile(f)
		args := strings.Split(strings.TrimSpace(string(b)), "\n")
		if i := slices.Index(args, "--settings"); i >= 0 && args[i+1] == `{"fastMode":true}` {
			fast++
		}
	}
	if len(files) != 4 || fast != 2 {
		t.Fatalf("%d of %d runs in fast mode, want 2 of 4", fast, len(files))
	}
}

// A fast turn never goes on in a run started at the usual speed, nor the
// other way: Claude Code's fast mode is the run's own.
func TestClaudeTurnKeyHoldsFast(t *testing.T) {
	msgs := []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}
	slow := &Request{Model: "claude-opus-5-5"}
	fast := &Request{Model: "claude-opus-5-5", Fast: true}
	if turnKey("o", slow, msgs) == turnKey("o", fast, msgs) {
		t.Fatal("turnKey: a fast turn matches a run at the usual speed")
	}
	if newLooseTurn("o", slow, msgs, true).key == newLooseTurn("o", fast, msgs, true).key {
		t.Fatal("looseTurn: a fast turn matches a run at the usual speed")
	}
}
