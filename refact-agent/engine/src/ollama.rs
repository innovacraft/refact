pub const KNOWN_MODELS_OLLAMA: &str = r####"
{
    "code_completion_models": {
        "codellama": {
            "n_ctx": 8192,
            "supports_scratchpads": {
                "FIM-PSM": {
                    "fim_prefix": "<PRE>",
                    "fim_suffix": "<SUF>",
                    "fim_middle": "<MID>",
                    "eot": "</s>",
                    "context_format": "llama"
                }
            },
            "default_scratchpad": "FIM-PSM"
        }
    },
    "code_chat_models": {
        "codellama": {
            "n_ctx": 8192,
            "supports_tools": true,
            "supports_multimodality": false,
            "supports_agent": true,
            "supports_scratchpads": {
                "CHAT-GENERIC": {
                    "token_bos": "",
                    "token_esc": "",
                    "keyword_system": "### System:\n",
                    "keyword_user": "### User:\n",
                    "keyword_assistant": "### Assistant:\n",
                    "eot": "</s>",
                    "stop_list": ["</s>", "### User:", "### Assistant:", "### System:"]
                }
            }
        },
        "qwen2.5:3b-instruct": {
            "n_ctx": 2000,
            "supports_tools": true,
            "supports_multimodality": false,
            "supports_agent": true,
            "supports_scratchpads": {
                "PASSTHROUGH": {}
            },
            "similar_models": []
        }
    }
}
"####;