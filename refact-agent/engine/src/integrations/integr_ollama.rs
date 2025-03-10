use crate::integrations::integr_abstract::*;
use crate::integrations::utils::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaIntegration {
    pub api_key: String,
    pub api_base: String,
    pub model: String,
}

impl Integration for OllamaIntegration {
    fn get_name(&self) -> &'static str {
        "ollama"
    }

    fn get_icon(&self) -> &'static str {
        "ollama.png"
    }

    fn get_description(&self) -> &'static str {
        "Local Ollama Integration"
    }

    fn is_available(&self) -> bool {
        !self.api_base.is_empty() && !self.model.is_empty()
    }

    fn validate(&self) -> Result<(), String> {
        if self.api_base.is_empty() {
            return Err("API base URL cannot be empty".to_string());
        }
        if self.model.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }
        Ok(())
    }
}