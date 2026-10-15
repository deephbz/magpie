package gateway

import (
	"slices"
	"strings"

	"github.com/tidwall/gjson"
)

// hoisted gives backends without in-place Chat updates the joined system
// prompt and full tool list. It leaves the positioned request unchanged.
func (r *Request) hoisted() *Request {
	if !slices.ContainsFunc(r.Messages, func(m Message) bool { return m.ChatSystem }) {
		return r
	}
	out := *r
	out.Messages = make([]Message, 0, len(r.Messages))
	out.Tools = slices.Clone(r.Tools)
	var sys []string
	if r.System != "" || r.chatSystemSet {
		sys = append(sys, r.System)
	}
	for _, m := range r.Messages {
		if !m.ChatSystem {
			out.Messages = append(out.Messages, m)
			continue
		}
		for _, p := range m.Parts {
			if p.Kind == Text {
				sys = append(sys, p.Text)
			}
		}
		out.Tools = addTools(out.Tools, m.Tools)
	}
	out.System = strings.Join(sys, "\n\n")
	return &out
}

func addTools(to, added []Tool) []Tool {
	for _, tool := range added {
		if i := slices.IndexFunc(to, func(t Tool) bool { return t.Name == tool.Name }); i >= 0 {
			to[i] = tool
		} else {
			to = append(to, tool)
		}
	}
	return to
}

// chatNeedsTranslation recognizes the Chat extension before passthrough.
// Even a Chat upstream receives its tools at the top level through the fold.
func chatNeedsTranslation(body []byte) bool {
	leading := true
	for _, m := range gjson.GetBytes(body, "messages").Array() {
		if m.Get("tools").Exists() {
			return true
		}
		role := m.Get("role").String()
		if role == "system" || role == "developer" {
			if !leading {
				return true
			}
		} else {
			leading = false
		}
	}
	return false
}

func (r *Request) hasChatUpdates() bool {
	return slices.ContainsFunc(r.Messages, func(m Message) bool { return m.ChatSystem })
}
