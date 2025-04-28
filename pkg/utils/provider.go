package utils

import (
	"context"
	"fmt"
	"github.com/joern1811/llm/pkg/llm"
	"github.com/joern1811/llm/pkg/llm/anthropic"
	"github.com/joern1811/llm/pkg/llm/google"
	"github.com/joern1811/llm/pkg/llm/ollama"
	"github.com/joern1811/llm/pkg/llm/openai"
	"os"
	"strings"
)

// Add new function to create provider
func CreateProvider(ctx context.Context, modelString, baseURL, apiKey, systemPrompt string) (llm.Provider, error) {
	parts := strings.SplitN(modelString, ":", 2)
	if len(parts) < 2 {
		return nil, fmt.Errorf(
			"invalid model format. Expected provider:model, got %s",
			modelString,
		)
	}

	provider := parts[0]
	model := parts[1]

	switch provider {
	case "anthropic":
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		}
		if apiKey == "" {
			return nil, fmt.Errorf(
				"anthropic API key not provided. Use --anthropic-api-key flag or ANTHROPIC_API_KEY environment variable",
			)
		}
		if baseURL == "" {
			baseURL = os.Getenv("ANTHROPIC_API_BASE_URL")
		}
		return anthropic.NewProvider(apiKey, baseURL, model, systemPrompt), nil

	case "ollama":
		return ollama.NewProvider(model, systemPrompt)

	case "openai":
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if apiKey == "" {
			return nil, fmt.Errorf(
				"OpenAI API key not provided. Use --openai-api-key flag or OPENAI_API_KEY environment variable",
			)
		}
		if baseURL == "" {
			baseURL = os.Getenv("OPENAI_API_BASE_URL")
		}
		return openai.NewProvider(apiKey, baseURL, model, systemPrompt), nil

	case "google":
		if apiKey == "" {
			apiKey = os.Getenv("GOOGLE_API_KEY")
		}
		if apiKey == "" {
			// The project structure is provider-specific, but Google calls this GEMINI_API_KEY in e.g., AI Studio. Support both.
			apiKey = os.Getenv("GEMINI_API_KEY")
		}
		return google.NewProvider(ctx, apiKey, model, systemPrompt)

	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}
