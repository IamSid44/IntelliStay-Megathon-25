# LM Studio Setup for Local LLM Integration

The dashboard uses a locally hosted LLM (via LM Studio) to turn SHAP and DiCE outputs into customer personas, retention actions and behavioral-nudge message templates. This step is optional: if the server is unreachable, the dashboard falls back to rule-based insights.

## Prerequisites

1. Install LM Studio from [lmstudio.ai](https://lmstudio.ai/).
2. Download the model `qwen/qwen3-4b-thinking-2507`.

## Setup Steps

### 1. Start LM Studio

Open the LM Studio application.

### 2. Load the Model

1. Go to the **Models** tab.
2. Search for `qwen/qwen3-4b-thinking-2507` and download it if it is not already present.
3. Load the model.

### 3. Start the Local Server

1. Go to the **Local Server** tab.
2. Click **Start Server**.
3. Verify it is running on `http://localhost:1234`.

### 4. Test the Connection

```bash
curl http://localhost:1234/v1/models
```

A JSON response listing the loaded model confirms the server is reachable. The dashboard's "AI Strategist" indicator will then show **Online**.
