package gateway

import (
	"bufio"
	"bytes"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// A request may go on from the conversation as it stood when a run was
// asked an earlier one, while that run has gone on with the conversation
// since: a client's compaction fork is the conversation as it stood at a turn,
// with its own ask added to the turn's last message, sent while the
// conversation goes on in its run. Told anew in one message, it is a
// prefix the cache has never seen, the whole conversation written to it
// again. Its Claude Code is started from the run's session as it stood
// when that request was told it (--resume --fork-session
// --resume-session-at), so its request repeats the run's to there and
// reads from the cache what the run's wrote, and is told only what the
// request adds.

// runPoint is where a run answered a request: the request's conversation
// (turnKey) and its reply's first message in the run's session.
type runPoint struct{ key, reply string }

// pointsMost is how many of its requests a run keeps where it answered.
const pointsMost = 64

// replied keeps where the run answered the request it is answering, at
// its reply's first message in the session (readOutput).
func (r *subscriptionRun) replied(uuid string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.answering == "" || uuid == "" {
		return
	}
	r.points = append(r.points, runPoint{r.answering, uuid})
	if len(r.points) > pointsMost {
		r.points = r.points[len(r.points)-pointsMost:]
	}
	r.answering = ""
}

// fork is the session a request starts from when a run of the account
// answered its conversation as it stood before its last messages, or
// part of the last: the longest such, with what the request adds since.
func (b *subscriptionBridge) fork(req *Request, owner string) *savedSession {
	type at struct {
		run   *subscriptionRun
		reply string
	}
	points := map[string]at{}
	b.mu.Lock()
	for _, run := range b.runs {
		run.mu.Lock()
		if run.owner == owner && run.sessionID != "" && !run.closed {
			for _, p := range run.points {
				points[p.key] = at{run, p.reply}
			}
		}
		run.mu.Unlock()
	}
	b.mu.Unlock()
	if len(points) == 0 {
		return nil
	}
	msgs := req.Messages
	first := len(msgs) // the request's own messages, after the last reply
	for first > 0 && msgs[first-1].Role != "assistant" {
		first--
	}
	for i := len(msgs) - 1; i >= first; i-- {
		for k := len(msgs[i].Parts); k >= 0; k-- {
			if i == len(msgs)-1 && k == len(msgs[i].Parts) || i == first && k == 0 {
				continue // the request itself; a conversation that ends with a reply
			}
			head := append(msgs[:i:i], Message{Role: msgs[i].Role, Parts: msgs[i].Parts[:k]})
			found, ok := points[turnKey(owner, req, head)]
			if !ok {
				continue
			}
			found.run.mu.Lock()
			sid, sessions, floor := found.run.sessionID, found.run.sessions, found.run.floor
			found.run.mu.Unlock()
			told, ok := sessionTold(sessions, sid, found.reply)
			if !ok {
				return nil
			}
			since := append([]Message{{Role: msgs[i].Role, Parts: msgs[i].Parts[k:]}}, msgs[i+1:]...)
			if k == len(msgs[i].Parts) {
				since = msgs[i+1:]
			}
			from, again := told.uuid, []Message(nil)
			if !told.results {
				// Claude Code joins the message it is told to the one
				// before it, a line break added to that one's text, which
				// is then not the run's: the fork starts before it and is
				// told that turn again, as resume told it, then since
				from = told.parent
				at := *req
				at.Messages = head
				if _, again = nextTurn(&at, owner); again == nil || from == "" {
					return nil
				}
			}
			log.Printf("claude: a request going on from where a run answered an earlier one starts from that run's session, told the %d messages since", len(again)+len(since))
			// gone: the session is its run's, never this one's to remove
			return &savedSession{session: sid, sessions: sessions, owner: owner, since: since, again: again, floor: floor, forkAt: from, gone: true}
		}
	}
	return nil
}

// toldEntry is the user message of a run's saved session its request was
// told, the one before its reply: its id, the reply before it ("" for the
// session's first), and whether it carries tool results, which Claude
// Code told the run itself.
type toldEntry struct {
	uuid, parent string
	results      bool
}

// sessionTold is the user message before reply in a run's saved session,
// past the attachments Claude Code adds after it. Not ok when it isn't
// there, or is the session's first: a run started anew was told the
// conversation in a message no fork repeats (renderClaudePrompt).
func sessionTold(dirs []string, sid, reply string) (toldEntry, bool) {
	if sid == "" || strings.ContainsAny(sid, `/\`) || strings.Contains(sid, "..") {
		return toldEntry{}, false
	}
	type entry struct {
		Type    string `json:"type"`
		UUID    string `json:"uuid"`
		Parent  string `json:"parentUuid"`
		Message struct {
			Content json.RawMessage `json:"content"`
		} `json:"message"`
	}
	for _, dir := range dirs {
		f, err := os.Open(filepath.Join(dir, sid+".jsonl"))
		if err != nil {
			continue
		}
		entries := map[string]entry{}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<28)
		for sc.Scan() {
			var e entry
			if json.Unmarshal(sc.Bytes(), &e) == nil && e.UUID != "" {
				entries[e.UUID] = e
			}
		}
		_ = f.Close()
		e, ok := entries[reply]
		if !ok {
			continue
		}
		for e, ok = entries[e.Parent]; ok && e.Type != "user"; e, ok = entries[e.Parent] {
		}
		if !ok {
			return toldEntry{}, false
		}
		told := toldEntry{uuid: e.UUID, parent: e.Parent, results: bytes.Contains(e.Message.Content, []byte(`"tool_result"`))}
		// a reply before it: the session's first message is no turn's
		for p, ok := entries[e.Parent]; ; p, ok = entries[p.Parent] {
			if !ok {
				return toldEntry{}, false
			}
			if p.Type == "assistant" {
				return told, true
			}
		}
	}
	return toldEntry{}, false
}
