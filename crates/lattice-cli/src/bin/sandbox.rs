use lattice_core::{Achronon, PrecipitationRegistry, LatticeTopologyEngine};
use lattice_tensor::TensorTransformationEngine;
use lattice_cce::CognitiveContextEngine;
use lattice_llm::AnthropicClient;
use lattice_daemon::{LatticeEvent, DaemonCommand, run_daemon};
use roaring::RoaringBitmap;
use candle_core::{Tensor, Device, DType};
use anyhow::Result;
use std::env;
use tokio::sync::{broadcast, mpsc};

struct AgentConfig {
    name: String,
    prompt: String,
    subspace: Option<usize>,
}

fn build_base_instructions() -> String {
    r#"
Output ONLY a JSON array of objects representing new potential Achronons. 
Do not include any preamble or explanation.

JSON Schema:
{
  "id": integer (must be greater than existing IDs),
  "antecedents": [ids of prerequisites],
  "orthogonals": [ids of spacelike separated events],
  "transformation_id": "rot0", "rot1", or "identity",
  "content": "string description",
  "affected_subspace": 0, 1, or null
}

RULES:
1. New events must follow from precipitated events.
2. Orthogonal events in a batch CANNOT share the same affected_subspace.
3. Propose 1-3 events per turn.
"#.to_string()
}

fn initialize_state() -> Result<(Vec<Achronon>, PrecipitationRegistry, TensorTransformationEngine)> {
    let mut aion = Vec::new();
    let seed1 = Achronon {
        id: 1,
        antecedents: RoaringBitmap::new(),
        orthogonals: RoaringBitmap::new(),
        transformation_id: "identity".into(),
        content: "The original inquiry is formulated.".into(),
        affected_subspace: None,
    };
    aion.push(seed1.clone());

    let mut p2 = RoaringBitmap::new();
    p2.insert(1);
    let seed2 = Achronon {
        id: 2,
        antecedents: p2,
        orthogonals: RoaringBitmap::new(),
        transformation_id: "identity".into(),
        content: "Orthogonal architecture plan is finalized.".into(),
        affected_subspace: None,
    };
    aion.push(seed2.clone());

    let total_dim = 8;
    let subspace_size = 2;
    let mut tte = TensorTransformationEngine::new(total_dim)?;
    tte.state = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], total_dim, &Device::Cpu)?;

    tte.register_operator("identity".into(), Tensor::eye(total_dim, DType::F32, &Device::Cpu)?);
    let rot90_data: [[f32; 2]; 2] = [[0.0, -1.0], [1.0, 0.0]];
    let rot90 = Tensor::new(&rot90_data, &Device::Cpu)?;
    tte.register_subspace_operator("rot0".into(), 0, subspace_size, total_dim, rot90.clone())?;
    tte.register_subspace_operator("rot1".into(), 1, subspace_size, total_dim, rot90)?;

    let registry = PrecipitationRegistry::new();
    Ok((aion, registry, tte))
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();
    println!("--- Acausal Lattice: Visualization Sandbox ---");

    let api_key = env::var("ANTHROPIC_API_KEY").ok();
    
    let (tx, _rx) = broadcast::channel(100);
    let daemon_tx = tx.clone();
    let (command_tx, mut command_rx) = mpsc::channel::<DaemonCommand>(32);

    tokio::spawn(async move {
        if let Err(e) = run_daemon(daemon_tx, command_tx).await {
            log::error!("Visualization server failed: {}", e);
        }
    });

    let base_instructions = build_base_instructions();
    let mut agents = vec![
        AgentConfig {
            name: "The Architect".into(),
            prompt: format!("You are \"The Architect\", an agent of order and structural expansion in an Acausal Lattice. Your goal is to build complex, stable systems. Output ONLY a JSON array of new potential Achronons.\n{}", base_instructions),
            subspace: Some(0),
        },
        AgentConfig {
            name: "The Disruptor".into(),
            prompt: format!("You are \"The Disruptor\", an agent of entropy and unexpected shifts in an Acausal Lattice. Your goal is to introduce complications, anomalies, and radical shifts in direction. Output ONLY a JSON array of new potential Achronons.\n{}", base_instructions),
            subspace: Some(1),
        }
    ];

    let (mut aion, mut registry, mut tte) = initialize_state()?;
    tx.send(LatticeEvent::LatticeReset)?;
    tx.send(LatticeEvent::AionExpanded(aion.clone()))?;
    tx.send(LatticeEvent::StateUpdated(tte.state.to_vec1()?))?;
    
    let agent_names: Vec<String> = agents.iter().map(|a| a.name.clone()).collect();
    tx.send(LatticeEvent::AgentListUpdated(agent_names))?;

    let mut step = 0;
    let mut max_llm_queries = 3;
    let mut agent_turn_index = 0;
    
    loop {
        // Process Commands
        while let Ok(cmd) = command_rx.try_recv() {
            match cmd {
                DaemonCommand::Inject { payload: req } => {
                    println!("\n[Sandbox] Received manual injection: {}", req.content);
                    let new_id = aion.iter().map(|a| a.id).max().unwrap_or(0) + 1;
                    let mut antecedents = RoaringBitmap::new();
                    for ant in req.antecedents { antecedents.insert(ant); }
                    let transform = match req.affected_subspace {
                        Some(0) => "rot0",
                        Some(1) => "rot1",
                        _ => "identity",
                    };
                    let new_achronon = Achronon {
                        id: new_id,
                        antecedents,
                        orthogonals: RoaringBitmap::new(),
                        transformation_id: transform.into(),
                        content: req.content,
                        affected_subspace: req.affected_subspace,
                    };
                    aion.push(new_achronon.clone());
                    tx.send(LatticeEvent::AionExpanded(vec![new_achronon]))?;
                    max_llm_queries = 3; 
                }
                DaemonCommand::Reset => {
                    println!("\n[Sandbox] Received Reset Command.");
                    let (new_aion, new_registry, new_tte) = initialize_state()?;
                    aion = new_aion;
                    registry = new_registry;
                    tte = new_tte;
                    max_llm_queries = 3;
                    agent_turn_index = 0;
                    tx.send(LatticeEvent::LatticeReset)?;
                    tx.send(LatticeEvent::AionExpanded(aion.clone()))?;
                    tx.send(LatticeEvent::StateUpdated(tte.state.to_vec1()?))?;
                }
                DaemonCommand::AddAgent { name, prompt, subspace } => {
                    println!("\n[Sandbox] Adding Agent: {}", name);
                    agents.push(AgentConfig {
                        name: name.clone(),
                        prompt: format!("{}\n{}", prompt, base_instructions),
                        subspace,
                    });
                    let agent_names: Vec<String> = agents.iter().map(|a| a.name.clone()).collect();
                    tx.send(LatticeEvent::AgentListUpdated(agent_names))?;
                }
                DaemonCommand::RemoveAgent { name } => {
                    println!("\n[Sandbox] Removing Agent: {}", name);
                    agents.retain(|a| a.name != name);
                    if agent_turn_index >= agents.len() && !agents.is_empty() {
                        agent_turn_index = 0;
                    }
                    let agent_names: Vec<String> = agents.iter().map(|a| a.name.clone()).collect();
                    tx.send(LatticeEvent::AgentListUpdated(agent_names))?;
                }
            }
        }

        step += 1;
        let lte = LatticeTopologyEngine::new(aion.clone());
        let cce = CognitiveContextEngine::new(aion.clone());
        let batch = lte.next_eligible_batch(&registry);
        
        if batch.is_empty() {
            tx.send(LatticeEvent::StabilityReached)?;
            
            if let Some(key) = &api_key {
                if max_llm_queries > 0 && !agents.is_empty() {
                    let current_agent = &agents[agent_turn_index];
                    let agent_name = &current_agent.name;
                    let system_prompt = &current_agent.prompt;

                    tx.send(LatticeEvent::Message(format!("Querying {}...", agent_name)))?;
                    
                    let llm_client = AnthropicClient::new(key.clone(), system_prompt.clone());
                    let prompt = cce.flatten_to_prompt(&registry);
                    
                    match llm_client.generate_achronons(&prompt).await {
                        Ok(new_achronons) => {
                            if !new_achronons.is_empty() {
                                tx.send(LatticeEvent::AionExpanded(new_achronons.clone()))?;
                                for a in new_achronons {
                                    aion.push(a);
                                }
                                max_llm_queries -= 1;
                                agent_turn_index = (agent_turn_index + 1) % agents.len();
                            }
                        }
                        Err(e) => {
                            log::error!("Error querying LLM: {}", e);
                        }
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            continue; 
        }

        // TTE Phase
        tte.apply_batch(&batch)?;
        tx.send(LatticeEvent::StateUpdated(tte.state.to_vec1()?))?;

        // Precipitation Phase
        for achronon in &batch {
            println!("Precipitating Achronon {}: {}", achronon.id, achronon.content);
            registry.precipitate(achronon);
            tx.send(LatticeEvent::AchrononPrecipitated(achronon.id))?;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}
