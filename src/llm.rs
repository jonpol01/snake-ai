use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

/// Request from the browser: includes the LLM endpoint + chat payload.
#[derive(Deserialize)]
pub struct LLMProxyRequest {
    pub endpoint: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

/// Proxy a chat completion request to an external LLM endpoint.
/// Avoids CORS issues since the browser only talks to our server.
pub async fn llm_chat(
    Json(req): Json<LLMProxyRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let url = format!(
        "{}/v1/chat/completions",
        req.endpoint.trim_end_matches('/')
    );

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("HTTP client error: {e}")})),
            );
        }
    };

    let payload = ChatCompletionRequest {
        messages: req.messages,
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        model: req.model,
        stop: req.stop,
    };

    match client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
    {
        Ok(res) => {
            let status = res.status().as_u16();
            match res.text().await {
                Ok(body) => {
                    let parsed: serde_json::Value =
                        serde_json::from_str(&body).unwrap_or(serde_json::json!({"raw": body}));
                    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
                    (code, Json(parsed))
                }
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"error": format!("Failed to read response: {e}")})),
                ),
            }
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": format!("LLM request failed: {e}")})),
        ),
    }
}

/// Proxy a models list request to check if the LLM endpoint is reachable
/// and retrieve available models.
pub async fn llm_models(
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let endpoint = body
        .get("endpoint")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if endpoint.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "endpoint is required"})),
        );
    }

    let url = format!("{}/v1/models", endpoint.trim_end_matches('/'));

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("HTTP client error: {e}")})),
            );
        }
    };

    match client.get(&url).send().await {
        Ok(res) => {
            let status = res.status().as_u16();
            match res.text().await {
                Ok(body) => {
                    let parsed: serde_json::Value =
                        serde_json::from_str(&body).unwrap_or(serde_json::json!({"raw": body}));
                    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
                    (code, Json(parsed))
                }
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"error": format!("Failed to read response: {e}")})),
                ),
            }
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": format!("LLM not reachable: {e}")})),
        ),
    }
}
