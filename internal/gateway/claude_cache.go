package gateway

// A client that places Anthropic's prompt-cache marks itself, naming each
// one's TTL (a Pi extension can: its system prompt's part shared by every
// project and the rest marked for an hour, the conversation's end for a
// TTL its user picks), has them kept on the Claude subscription too. Marks
// that name no TTL stay the gateway's to place, as before (cacheTTL). Its
// run is started with Claude Code's own caching off
// (DISABLE_PROMPT_CACHING), which marks two blocks of its system prompt
// and the request's end, three of Anthropic's four: the client's system
// prompt marks are kept in the run's first message, where they stay for
// the run's every request, and the mark on its request's end is the one
// Claude Code is told to ask for on each request (setSafeguards), which
// moves with the conversation. A mark the client put earlier in its
// conversation (a pin held at a turn) is in a message the run already has:
// what it holds is read from where the run's request wrote it, and a
// request of the client's own from there (its compaction fork) is started
// from the run's session as it stood (fork).

// clientCaches says the client placed prompt-cache marks of its own.
func (r *Request) clientCaches() bool {
	if len(r.SystemCache) > 0 || r.CacheTail != "" {
		return true
	}
	for _, m := range r.Messages {
		for _, p := range m.Parts {
			if p.Cache != "" {
				return true
			}
		}
	}
	return false
}

// cacheTail is the TTL of the client's mark on its request's end: the
// one it asked for on the whole request, else the one on the last block
// of its last message when that is its own, not a reply; "" for none.
func cacheTail(req *Request) string {
	if req.CacheTail != "" {
		return req.CacheTail
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		m := req.Messages[i]
		if len(m.Parts) == 0 {
			continue // Pi's message that carries only its effort
		}
		if m.Role == "assistant" {
			return ""
		}
		return m.Parts[len(m.Parts)-1].Cache
	}
	return ""
}

// cacheMarksMost is how many blocks Anthropic takes marked in a request.
const cacheMarksMost = 4

// instructionMarks are the client's system prompt marks a run started
// anew keeps (renderClaudePrompt), at their ends in system, the client's
// prompt without its billing header, which cut bytes were taken off the
// start of: within Anthropic's four, the request's end's mark among them,
// the ones nearest the conversation kept, and none longer than a mark
// before it.
func instructionMarks(req *Request, cut int) []CacheMark {
	room := cacheMarksMost
	if cacheTail(req) != "" {
		room--
	}
	var marks []CacheMark
	for _, m := range req.SystemCache {
		if end := m.End - cut; end > 0 && end <= len(req.System)-cut && (len(marks) == 0 || end > marks[len(marks)-1].End) {
			marks = append(marks, CacheMark{End: end, TTL: m.TTL})
		}
	}
	if len(marks) > room {
		marks = marks[len(marks)-room:]
	}
	floor := "1h"
	for i := range marks {
		marks[i].TTL = shorterTTL(marks[i].TTL, floor)
		floor = marks[i].TTL
	}
	return marks
}

// cacheFloor is the shortest TTL of the marks a run started anew for req
// keeps in its conversation: its system prompt's, and the history's
// (renderClaudePrompt), which takes the shortest of those; "" for none.
func cacheFloor(req *Request) string {
	system := withoutBillingHeader(req.System)
	floor := ""
	for _, m := range instructionMarks(req, len(req.System)-len(system)) {
		floor = shorterTTL(m.TTL, floor)
	}
	if floor == "" && historyEnd(req.Messages) >= historyChunk {
		floor = "1h"
	}
	return floor
}

// shorterTTL is the shorter of two TTLs, "" being none: either when the
// other is none.
func shorterTTL(a, b string) string {
	switch {
	case a == "":
		return b
	case b == "":
		return a
	case a == "5m" || b == "5m":
		return "5m"
	}
	return "1h"
}

// cacheMark is a prompt-cache mark for ttl.
func cacheMark(ttl string) map[string]string {
	return map[string]string{"type": "ephemeral", "ttl": ttl}
}
