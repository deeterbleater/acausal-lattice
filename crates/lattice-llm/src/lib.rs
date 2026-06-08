use lattice_core::Achronon;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use reqwest::Client;

#[derive(Debug, Deserialize, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: i32,
    messages: Vec<AnthropicMessage>,
    system: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: String,
}

/// A simplified template for Achronons that can be easily parsed from LLM JSON output.
#[derive(Debug, Deserialize)]
struct AchrononTemplate {
    id: u32,
    antecedents: Vec<u32>,
    orthogonals: Vec<u32>,
    transformation_id: String,
    content: String,
    affected_subspace: Option<usize>,
}

pub struct AnthropicClient {
    client: Client,
    api_key: String,
    model: String,
    system_prompt: String,
}

impl AnthropicClient {
    pub fn new(api_key: String, system_prompt: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: "claude-3-5-sonnet-20240620".into(),
            system_prompt,
        }
    }

    pub async fn generate_achronons(&self, lattice_state: &str) -> Result<Vec<Achronon>> {
        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 1024,
            system: self.system_prompt.clone(),
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: format!("Current Lattice State:\n\n{}", lattice_state),
            }],
        };

        let res = self.client.post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !res.status().is_success() {
            let err_text = res.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error: {}", err_text));
        }

        let response: AnthropicResponse = res.json().await?;
        let text = response.content.get(0)
            .context("Empty response from Anthropic")?
            .text.clone();

        // Clean potential markdown blocks if Claude includes them despite instructions
        let clean_json = if text.contains("```json") {
            text.split("```json").nth(1).unwrap().split("```").next().unwrap()
        } else if text.contains("```") {
            text.split("```").nth(1).unwrap().split("```").next().unwrap()
        } else {
            &text
        };

        let templates: Vec<AchrononTemplate> = serde_json::from_str(clean_json)
            .context(format!("Failed to parse Achronon JSON: {}", clean_json))?;

        let achronons = templates.into_iter().map(|t| Achronon {
            id: t.id,
            antecedents: t.antecedents.into_iter().collect(),
            orthogonals: t.orthogonals.into_iter().collect(),
            transformation_id: t.transformation_id,
            content: t.content,
            affected_subspace: t.affected_subspace,
        }).collect();

        Ok(achronons)
    }
}
