package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/joern1811/llm/pkg/history"
	"github.com/joern1811/llm/pkg/llm"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func boolPtr(b bool) *bool {
	return &b
}

// Provider implements the Provider interface for Ollama
type Provider struct {
	client       *api.Client
	model        string
	systemPrompt string
}

// NewProvider creates a new Ollama provider
func NewProvider(model string, systemPrompt string) (*Provider, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}
	return &Provider{
		client:       client,
		model:        model,
		systemPrompt: systemPrompt,
	}, nil
}

func (p *Provider) CreateMessage(
	ctx context.Context,
	prompt string,
	messages []llm.Message,
	tools []llm.Tool,
) (llm.Message, error) {
	// Convert generic messages to Ollama format
	ollamaMessages := make([]api.Message, 0, len(messages)+1)

	// Add system prompt if it exists
	if p.systemPrompt != "" {
		ollamaMessages = append(ollamaMessages, api.Message{
			Role:    "system",
			Content: p.systemPrompt,
		})
	}

	// Add existing messages
	for _, msg := range messages {
		// Handle tool responses
		if msg.IsToolResponse() {
			var content string

			// Handle HistoryMessage format
			if historyMsg, ok := msg.(*history.HistoryMessage); ok {
				for _, block := range historyMsg.Content {
					if block.Type == "tool_result" {
						content = block.Text
						break
					}
				}
			}

			// If no content found yet, try standard content extraction
			if content == "" {
				content = msg.GetContent()
			}

			if content == "" {
				continue
			}

			ollamaMsg := api.Message{
				Role:    "tool",
				Content: content,
			}
			ollamaMessages = append(ollamaMessages, ollamaMsg)
			continue
		}

		// Skip completely empty messages (no content and no tool calls)
		if msg.GetContent() == "" && len(msg.GetToolCalls()) == 0 {
			continue
		}

		ollamaMsg := api.Message{
			Role:    msg.GetRole(),
			Content: msg.GetContent(),
		}

		// Add tool calls for assistant messages
		if msg.GetRole() == "assistant" {
			for _, call := range msg.GetToolCalls() {
				if call.GetName() != "" {
					args := call.GetArguments()
					ollamaMsg.ToolCalls = append(
						ollamaMsg.ToolCalls,
						api.ToolCall{
							Function: api.ToolCallFunction{
								Name:      call.GetName(),
								Arguments: args,
							},
						},
					)
				}
			}
		}

		ollamaMessages = append(ollamaMessages, ollamaMsg)
	}

	// Add the new prompt if not empty
	if prompt != "" {
		ollamaMessages = append(ollamaMessages, api.Message{
			Role:    "user",
			Content: prompt,
		})
	}

	// Convert tools to Ollama format
	ollamaTools := make([]api.Tool, len(tools))
	for i, tool := range tools {
		ollamaTools[i] = api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters: struct {
					Type       string   `json:"type"`
					Defs       any      `json:"$defs,omitempty"`
					Items      any      `json:"items,omitempty"`
					Required   []string `json:"required"`
					Properties map[string]struct {
						Type        api.PropertyType `json:"type"`
						Items       any              `json:"items,omitempty"`
						Description string           `json:"description"`
						Enum        []any            `json:"enum,omitempty"`
					} `json:"properties"`
				}{
					Type:       tool.InputSchema.Type,
					Required:   tool.InputSchema.Required,
					Properties: convertProperties(tool.InputSchema.Properties),
				},
			},
		}
	}

	var response api.Message

	err := p.client.Chat(ctx, &api.ChatRequest{
		Model:    p.model,
		Messages: ollamaMessages,
		Tools:    ollamaTools,
		Stream:   boolPtr(false),
	}, func(r api.ChatResponse) error {
		if r.Done {
			response = r.Message
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return &OllamaMessage{Message: response}, nil
}

func (p *Provider) SupportsTools() bool {
	// Check if model supports function calling
	resp, err := p.client.Show(context.Background(), &api.ShowRequest{
		Model: p.model,
	})
	if err != nil {
		return false
	}
	for _, capability := range resp.Capabilities {
		if capability == model.CapabilityTools {
			return true
		}
	}
	return false
}

func (p *Provider) Name() string {
	return "ollama"
}

func (p *Provider) CreateToolResponse(
	toolCallID string,
	content interface{},
) (llm.Message, error) {
	contentStr := ""
	switch v := content.(type) {
	case string:
		contentStr = v
	default:
		bytes, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("error marshaling tool response: %w", err)
		}
		contentStr = string(bytes)
	}

	// Create message with explicit tool role
	msg := &OllamaMessage{
		Message: api.Message{
			Role:    "tool", // Explicitly set role to "tool"
			Content: contentStr,
			// No need to set ToolCalls for a tool response
		},
		ToolCallID: toolCallID,
	}

	return msg, nil
}

// Helper function to convert properties to Ollama's format
func convertProperties(props map[string]interface{}) map[string]struct {
	Type        api.PropertyType `json:"type"`
	Items       any              `json:"items,omitempty"`
	Description string           `json:"description"`
	Enum        []any            `json:"enum,omitempty"`
} {
	result := make(map[string]struct {
		Type        api.PropertyType `json:"type"`
		Items       any              `json:"items,omitempty"`
		Description string           `json:"description"`
		Enum        []any            `json:"enum,omitempty"`
	})

	for name, prop := range props {
		if propMap, ok := prop.(map[string]interface{}); ok {
			prop := struct {
				Type        api.PropertyType `json:"type"`
				Items       any              `json:"items,omitempty"`
				Description string           `json:"description"`
				Enum        []any            `json:"enum,omitempty"`
			}{
				Type:        []string{getString(propMap, "type")},
				Description: getString(propMap, "description"),
			}

			// Handle enum if present
			if enumRaw, ok := propMap["enum"].([]interface{}); ok {
				for _, e := range enumRaw {
					if str, ok := e.(string); ok {
						prop.Enum = append(prop.Enum, str)
					}
				}
			}

			result[name] = prop
		}
	}

	return result
}

// Helper function to safely get string values from map
func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
