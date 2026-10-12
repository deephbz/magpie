package gateway

import (
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// markedBody is a request from a client that places its own marks: its
// system prompt's shared part and the rest each marked for systemTTL, the user's
// messages said in turn (each answered "ok"), the last one's last block
// marked for tail, and extra blocks after it in the same message.
func markedBody(systemTTL, tail string, said []string, extra ...string) string {
	mark := func(ttl string) string {
		if ttl == "" {
			return ""
		}
		return `,"cache_control":{"type":"ephemeral","ttl":"` + ttl + `"}`
	}
	var msgs []string
	for i, m := range said {
		if i > 0 {
			msgs = append(msgs, `{"role":"assistant","content":"ok"}`)
		}
		blocks := []string{fmt.Sprintf(`{"type":"text","text":%q`, m)}
		for _, e := range extra {
			if i == len(said)-1 {
				blocks = append(blocks, fmt.Sprintf(`{"type":"text","text":%q`, e))
			}
		}
		if i == len(said)-1 {
			blocks[len(blocks)-1] += mark(tail)
		}
		msgs = append(msgs, `{"role":"user","content":[`+strings.Join(blocks, "},")+`}]}`)
	}
	return `{"model":"claude-sonnet-5","max_tokens":100,"system":[{"type":"text","text":"GLOBAL rules\n"` + mark(systemTTL) +
		`},{"type":"text","text":"PROJECT rules"` + mark(systemTTL) + `}],"tools":[{"name":"read","input_schema":{"type":"object"}}],"messages":[` + strings.Join(msgs, ",") + `]}`
}

// marksOf is the TTL of each marked block of a run's first message, in order.
func marksOf(blocks []map[string]any) []string {
	var out []string
	for _, b := range blocks {
		if cc, ok := b["cache_control"].(map[string]string); ok {
			out = append(out, cc["ttl"])
		}
	}
	return out
}

// A client that names its marks' TTLs has its system prompt's marks kept
// where it put them, a block ending at each, in the message a run started
// anew is told; its request's end is marked by Claude Code on each request
// instead (setSafeguards), so the history's mark goes in the room left of
// Anthropic's four, never longer than a mark before it. Marks that name no
// TTL leave the prompt as before.
func TestClaudeClientCacheMarksKept(t *testing.T) {
	req, err := parseAnthropic([]byte(markedBody("1h", "5m", []string{"hi"})))
	if err != nil {
		t.Fatal(err)
	}
	if !req.clientCaches() || cacheTail(req) != "5m" || cacheFloor(req) != "1h" {
		t.Fatalf("client caches %t, tail %q, floor %q", req.clientCaches(), cacheTail(req), cacheFloor(req))
	}
	blocks, _ := renderClaudePrompt(req)
	if got := marksOf(blocks); strings.Join(got, ",") != "1h,1h" {
		t.Fatalf("marks: %v", blocks)
	}
	if blocks[0]["text"] != "<external_system_instructions>\nGLOBAL rules\n" || blocks[1]["text"] != "PROJECT rules" {
		t.Fatalf("blocks: %q | %q", blocks[0]["text"], blocks[1]["text"])
	}
	// another project's request shares the first block
	other, _ := parseAnthropic([]byte(strings.Replace(markedBody("1h", "5m", []string{"yo"}), "PROJECT", "ELSEWHERE", 1)))
	if theirs, _ := renderClaudePrompt(other); theirs[0]["text"] != blocks[0]["text"] {
		t.Fatalf("shared part differs: %q", theirs[0]["text"])
	}

	said := make([]string, 12) // 23 messages before the turn to answer
	for i := range said {
		said[i] = fmt.Sprintf("q%d", i)
	}
	long, _ := parseAnthropic([]byte(markedBody("1h", "5m", said)))
	if got := marksOf(rendered(long)); strings.Join(got, ",") != "1h,1h,1h" {
		t.Fatalf("with history: %v", got)
	}
	// a 5m system prompt: the history's mark is 5m, and so the request's end
	short, _ := parseAnthropic([]byte(markedBody("5m", "1h", said)))
	if got := marksOf(rendered(short)); strings.Join(got, ",") != "5m,5m,5m" || cacheFloor(short) != "5m" || shorterTTL(cacheTail(short), cacheFloor(short)) != "5m" {
		t.Fatalf("5m system: %v, floor %q", got, cacheFloor(short))
	}
	// no mark on the end: four in the message at most
	none, _ := parseAnthropic([]byte(markedBody("1h", "", said)))
	if got := marksOf(rendered(none)); len(got) != 3 || cacheTail(none) != "" {
		t.Fatalf("no tail: %v", got)
	}
	five := &Request{System: "abcd", SystemCache: []CacheMark{{1, "1h"}, {2, "1h"}, {3, "1h"}, {4, "1h"}}, Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi", Cache: "5m"}}}}}
	if got := instructionMarks(five, 0); len(got) != 3 || got[0].End != 2 {
		t.Fatalf("over Anthropic's four: %v", got)
	}

	body := strings.ReplaceAll(strings.ReplaceAll(markedBody("1h", "5m", []string{"hi"}), `,"ttl":"1h"`, ""), `,"ttl":"5m"`, "")
	if req, _ := parseAnthropic([]byte(body)); req.clientCaches() || strings.Join(marksOf(rendered(req)), ",") != "1h" {
		t.Fatalf("unnamed marks: %v", rendered(req))
	}
	if extraBody(nil, "5m") != `{"cache_control":{"ttl":"5m","type":"ephemeral"}}` || extraBody(nil, "") != `{}` {
		t.Fatalf("extra body: %s", extraBody(nil, "5m"))
	}
}

// rendered is the message a run started anew for req is told.
func rendered(req *Request) []map[string]any {
	blocks, _ := renderClaudePrompt(req)
	return blocks
}

// sendBody sends one request to the harness's Claude subscription.
func (h *resumeHarness) sendBody(body string) {
	h.t.Helper()
	rec := httptest.NewRecorder()
	var u Usage
	if code, msg := h.s.serveClaudeSubscription(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)), provider.Anthropic, h.p, "claude-sonnet-5", []byte(body), &u); code != 200 {
		h.t.Fatalf("%d %s", code, msg)
	}
}

// sessionTurn is a harness Claude Code's turn as it saves it: the user
// message it was told, after its last reply, the attachment it adds after
// that, and its reply's message, whose id it says on its stream (n counts
// its turns, $$ is its process).
const sessionTurn = `
  n=$((n+1))
  echo "env DISABLE_PROMPT_CACHING=$DISABLE_PROMPT_CACHING" >> $log
  if [ -n "$file" ]; then
    parent=; if [ $n -gt 1 ]; then parent=r$$-$((n-1)); fi
    echo '{"type":"user","uuid":"u'$$'-'$n'","parentUuid":"'$parent'","message":{"role":"user","content":[]}}' >> "$file"
    echo '{"type":"attachment","uuid":"a'$$'-'$n'","parentUuid":"u'$$'-'$n'"}' >> "$file"
    echo '{"type":"assistant","uuid":"r'$$'-'$n'","parentUuid":"a'$$'-'$n'"}' >> "$file"
  fi
  echo '{"type":"assistant","uuid":"r'$$'-'$n'","message":{"id":"m","model":"claude-sonnet-5","content":[]}}'
`

// forkStart has a harness Claude Code given --fork-session save its
// session as a new one, as Claude Code does.
const forkStart = `case " $* " in *" --fork-session "*) sid=f$$; file="$CLAUDE_CONFIG_DIR/projects/$proj/$sid.jsonl" ;; esac`

// newestRun is the log of the run started since before.
func (h *resumeHarness) newestRun(before map[string]string) (string, string) {
	for pid, log := range h.runs() {
		if _, ok := before[pid]; !ok {
			return pid, log
		}
	}
	return "", ""
}

// A client that names its marks' TTLs has its run's Claude Code start
// with its own caching off, and told to mark each request's end for the
// client's TTL, the one its request now names (its user picked 1h).
func TestClaudeClientCacheRun(t *testing.T) {
	h := newTurnHarness(t, "ok", forkStart, sessionTurn)
	h.sendBody(markedBody("1h", "5m", []string{"hi"}))
	pid, log := h.newestRun(nil)
	if !strings.Contains(log, "env DISABLE_PROMPT_CACHING=1") || !strings.Contains(log, `\"ttl\":\"5m\"`) {
		t.Fatalf("first run:\n%s", log)
	}
	h.sendBody(markedBody("1h", "1h", []string{"hi", "more"}))
	if runs := h.runs(); len(runs) != 1 || !strings.Contains(runs[pid], `\"ttl\":\"1h\"`) {
		t.Fatalf("the next turn, its end for an hour:\n%v", runs)
	}
	// a request without named marks: Claude Code's own caching, as before
	before := h.runs()
	h.ask("plain", "hello")
	if _, plain := h.newestRun(before); !strings.Contains(plain, "env DISABLE_PROMPT_CACHING=\n") || strings.Contains(plain, "\ncontrol ") {
		t.Fatalf("a plain run:\n%s", plain)
	}
}

// A client's compaction fork is the conversation as it stood at a turn, its ask
// added to the turn's last message, sent while the conversation goes on
// in its run: its Claude Code starts from the run's session as it stood
// before it was told that turn, and is told the turn again as the run was,
// then the ask, so its request repeats the run's and reads from the cache
// what the run's wrote. Claude Code would join the ask to the turn's
// message, a line break added, were it resumed after it. The conversation
// goes on in its run.
func TestClaudeForkStartsFromTheRunsSession(t *testing.T) {
	h := newTurnHarness(t, "ok", forkStart, sessionTurn)
	h.sendBody(markedBody("1h", "5m", []string{"hi"}))
	main, _ := h.newestRun(nil)
	h.sendBody(markedBody("1h", "5m", []string{"hi", "more"}))
	h.sendBody(markedBody("1h", "5m", []string{"hi", "more", "again"}))

	before := h.runs()
	h.sendBody(markedBody("1h", "5m", []string{"hi", "more"}, "HANDOFF: plan the compaction"))
	pid, log := h.newestRun(before)
	if pid == "" || !strings.Contains(log, "--resume s"+main+" --fork-session --resume-session-at r"+main+"-1") {
		t.Fatalf("the fork:\n%s", log)
	}
	told := log[strings.Index(log, "\ntold "):]
	if !strings.Contains(told, `{"text":"more","type":"text"},{"text":"HANDOFF`) || strings.Contains(told, `"hi`) || strings.Contains(told, "GLOBAL") {
		t.Fatalf("the fork was told other than its turn and ask:\n%s", told)
	}
	if !containsStr(h.sessions(), "s"+main) || !containsStr(h.sessions(), "f"+pid) {
		t.Fatalf("sessions: %v", h.sessions())
	}

	h.sendBody(markedBody("1h", "5m", []string{"hi", "more", "again", "on"}))
	if runs := h.runs(); len(runs) != 2 || strings.Count(runs[main], "\ntold ") != 4 {
		t.Fatalf("the conversation went on elsewhere:\n%v", runs)
	}
	// the run's first turn was told in a message no fork repeats, and a
	// request whose start no run answered is told the conversation anew
	for _, said := range [][]string{{"hi"}, {"other"}} {
		before = h.runs()
		h.sendBody(markedBody("1h", "5m", said, "HANDOFF"))
		if _, log := h.newestRun(before); strings.Contains(log, "--resume") {
			t.Fatalf("%s forked:\n%s", said, log)
		}
	}
}
