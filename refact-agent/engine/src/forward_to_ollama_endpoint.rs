use crate::scratchpad_abstract::{ContextFile, GenAnyLenTextCause, GenAnyLenTextType};
use crate::custom_error::CrappyErrAnyStr;
use crate::restream::EventSource;
use crate::fetch_embedding::{fetch_embedding_struct, fetch_embedding_openai_compatible};
use crate::global_context::GlobalContext;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaCompletionRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaEmbeddingRequest {
    pub model: String,
    pub prompt: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaEmbeddingResponse {
    pub embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaChatResponse {
    pub model: String,
    pub message: OllamaMessage,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaCompletionResponse {
    pub model: String,
    pub response: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaStreamResponse {
    pub model: String,
    pub message: Option<OllamaMessage>,
    pub response: Option<String>,
    pub done: bool,
}

pub async fn forward_to_ollama_endpoint(
    global_context: Arc<GlobalContext>,
    url: &str,
    api_key: &str,
    model: &str,
    messages: Vec<(&str, &str)>,
    context_files: &[ContextFile],
    temperature: f32,
    stop_tokens: &[String],
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    info!("forward_to_ollama_endpoint called with model {}", model);
    
    let mut ollama_messages = Vec::new();
    
    for (role, content) in messages {
        ollama_messages.push(OllamaMessage {
            role: role.to_string(),
            content: content.to_string(),
        });
    }
    
    // Context files are usually added as a system message
    if !context_files.is_empty() {
        let context_content = context_files
            .iter()
            .map(|cf| format!("File: {}\n\n{}", cf.path, cf.content))
            .collect::<Vec<_>>()
            .join("\n\n");
        
        ollama_messages.insert(
            0,
            OllamaMessage {
                role: "system".to_string(),
                content: format!("Context files:\n\n{}", context_content),
            },
        );
    }
    
    let request = OllamaChatRequest {
        model: model.to_string(),
        messages: ollama_messages,
        stream: false,
        temperature: Some(temperature),
        stop: Some(stop_tokens.to_vec()),
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await?;
    
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        error!(
            "Ollama API error: status={}, error={}",
            status, error_text
        );
        return Err(Box::new(CrappyErrAnyStr(format!(
            "Ollama API error: {}: {}",
            status, error_text
        ))));
    }
    
    let response_json: OllamaChatResponse = response.json().await?;
    
    Ok(response_json.message.content)
}

pub async fn forward_to_ollama_endpoint_streaming(
    global_context: Arc<GlobalContext>,
    url: &str,
    api_key: &str,
    model: &str,
    messages: Vec<(&str, &str)>,
    context_files: &[ContextFile],
    temperature: f32,
    stop_tokens: &[String],
    text_cause: GenAnyLenTextCause,
    text_type: GenAnyLenTextType,
) -> Result<EventSource, Box<dyn std::error::Error + Send + Sync>> {
    info!("forward_to_ollama_endpoint_streaming called with model {}", model);
    
    let mut ollama_messages = Vec::new();
    
    for (role, content) in messages {
        ollama_messages.push(OllamaMessage {
            role: role.to_string(),
            content: content.to_string(),
        });
    }
    
    // Context files are usually added as a system message
    if !context_files.is_empty() {
        let context_content = context_files
            .iter()
            .map(|cf| format!("File: {}\n\n{}", cf.path, cf.content))
            .collect::<Vec<_>>()
            .join("\n\n");
        
        ollama_messages.insert(
            0,
            OllamaMessage {
                role: "system".to_string(),
                content: format!("Context files:\n\n{}", context_content),
            },
        );
    }
    
    let request = OllamaChatRequest {
        model: model.to_string(),
        messages: ollama_messages,
        stream: true,
        temperature: Some(temperature),
        stop: Some(stop_tokens.to_vec()),
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await?;
    
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        error!(
            "Ollama API error: status={}, error={}",
            status, error_text
        );
        return Err(Box::new(CrappyErrAnyStr(format!(
            "Ollama API error: {}: {}",
            status, error_text
        ))));
    }
    
    let event_source = EventSource::new(
        response,
        Box::new(move |line: &str| {
            if line.is_empty() {
                return None;
            }
            
            match serde_json::from_str::<OllamaStreamResponse>(line) {
                Ok(response) => {
                    if response.done {
                        return Some("[DONE]".to_string());
                    }
                    
                    if let Some(message) = response.message {
                        return Some(message.content);
                    } else if let Some(response_text) = response.response {
                        return Some(response_text);
                    }
                    
                    None
                }
                Err(e) => {
                    warn!("Error parsing Ollama response: {}", e);
                    None
                }
            }
        }),
        text_cause,
        text_type,
    );
    
    Ok(event_source)
}

pub async fn forward_to_ollama_completion_endpoint(
    global_context: Arc<GlobalContext>,
    url: &str,
    api_key: &str,
    model: &str,
    prompt: &str,
    temperature: f32,
    stop_tokens: &[String],
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    info!("forward_to_ollama_completion_endpoint called with model {}", model);
    
    let request = OllamaCompletionRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        stream: false,
        temperature: Some(temperature),
        stop: Some(stop_tokens.to_vec()),
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await?;
    
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        error!(
            "Ollama API error: status={}, error={}",
            status, error_text
        );
        return Err(Box::new(CrappyErrAnyStr(format!(
            "Ollama API error: {}: {}",
            status, error_text
        ))));
    }
    
    let response_json: OllamaCompletionResponse = response.json().await?;
    
    Ok(response_json.response)
}

pub async fn forward_to_ollama_completion_endpoint_streaming(
    global_context: Arc<GlobalContext>,
    url: &str,
    api_key: &str,
    model: &str,
    prompt: &str,
    temperature: f32,
    stop_tokens: &[String],
    text_cause: GenAnyLenTextCause,
    text_type: GenAnyLenTextType,
) -> Result<EventSource, Box<dyn std::error::Error + Send + Sync>> {
    info!("forward_to_ollama_completion_endpoint_streaming called with model {}", model);
    
    let request = OllamaCompletionRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        stream: true,
        temperature: Some(temperature),
        stop: Some(stop_tokens.to_vec()),
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await?;
    
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        error!(
            "Ollama API error: status={}, error={}",
            status, error_text
        );
        return Err(Box::new(CrappyErrAnyStr(format!(
            "Ollama API error: {}: {}",
            status, error_text
        ))));
    }
    
    let event_source = EventSource::new(
        response,
        Box::new(move |line: &str| {
            if line.is_empty() {
                return None;
            }
            
            match serde_json::from_str::<OllamaStreamResponse>(line) {
                Ok(response) => {
                    if response.done {
                        return Some("[DONE]".to_string());
                    }
                    
                    if let Some(response_text) = response.response {
                        return Some(response_text);
                    }
                    
                    None
                }
                Err(e) => {
                    warn!("Error parsing Ollama response: {}", e);
                    None
                }
            }
        }),
        text_cause,
        text_type,
    );
    
    Ok(event_source)
}

pub async fn fetch_embedding_ollama(
    global_context: Arc<GlobalContext>, 
    url: &str,
    api_key: &str,
    model: &str,
    text: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    info!("fetch_embedding_ollama called with model {}", model);
    
    let request = OllamaEmbeddingRequest {
        model: model.to_string(),
        prompt: text.to_string(),
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await?;
    
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        error!(
            "Ollama API error: status={}, error={}",
            status, error_text
        );
        return Err(Box::new(CrappyErrAnyStr(format!(
            "Ollama API error: {}: {}",
            status, error_text
        ))));
    }
    
    let response_json: OllamaEmbeddingResponse = response.json().await?;
    
    Ok(response_json.embedding)
}