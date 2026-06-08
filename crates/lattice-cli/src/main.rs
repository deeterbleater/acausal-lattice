use lattice_core::{Achronon, PrecipitationRegistry, LatticeTopologyEngine};
use lattice_tensor::TensorTransformationEngine;
use lattice_cce::CognitiveContextEngine;
use lattice_llm::AnthropicClient;
use lattice_daemon::{LatticeEvent, InjectRequest, run_daemon};
use roaring::RoaringBitmap;
use candle_core::{Tensor, Device, DType};
use anyhow::Result;
use std::env;
use tokio::sync::{broadcast, mpsc};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();
    println!("--- Acausal Lattice: Visualization Sandbox ---");

    let api_key = env::var("ANTHROPIC_API_KEY").ok();
    
    // Set up broadcasting for the visualizer
    let (tx, _rx) = broadcast::channel(100);
    let daemon_tx = tx.clone();

    // Set up MPSC for receiving user injections from the web sandbox
    let (inject_tx, mut inject_rx) = mpsc::channel::<InjectRequest>(32);

    // Spawn the visualization server
    tokio::spawn(async move {
        if let Err(e) = run_daemon(daemon_tx, inject_tx).await {
            log::error!("Visualization server failed: {}", e);
        }
    });

    // Agent Personas
    let architect_prompt = "You are \"The Architect\", an agent of order and structural expansion in an Acausal Lattice. Your goal is to build complex, stable systems. Output ONLY a JSON array of new potential Achronons.";
    let disruptor_prompt = "You are \"The Disruptor\", an agent of entropy and unexpected shifts in an Acausal Lattice. Your goal is to introduce complications, anomalies, and radical shifts in direction. Output ONLY a JSON array of new potential Achronons.";

    let base_system_instructions = r#"
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
"#;

    let agent_architect = format!("{}\n{}", architect_prompt, base_system_instructions);
    let agent_disruptor = format!("{}\n{}", disruptor_prompt, base_system_instructions);

    // 1. Initialize Aion
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

    tx.send(LatticeEvent::AionExpanded(vec![seed1, seed2]))?;

    // 2. Initialize Engines
    let total_dim = 8;
    let subspace_size = 2;
    let mut tte = TensorTransformationEngine::new(total_dim)?;
    tte.state = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], total_dim, &Device::Cpu)?;

    tte.register_operator("identity".into(), Tensor::eye(total_dim, DType::F32, &Device::Cpu)?);
    let rot90_data: [[f32; 2]; 2] = [[0.0, -1.0], [1.0, 0.0]];
    let rot90 = Tensor::new(&rot90_data, &Device::Cpu)?;
    tte.register_subspace_operator("rot0".into(), 0, subspace_size, total_dim, rot90.clone())?;
    tte.register_subspace_operator("rot1".into(), 1, subspace_size, total_dim, rot90)?;

    let mut registry = PrecipitationRegistry::new();

    // Broadcast initial state
    tx.send(LatticeEvent::StateUpdated(tte.state.to_vec1()?))?;

    // 3. Loop
    let mut step = 0;
    let mut max_llm_queries = 3;
    
    loop {
        // Process user injections non-blockingly
        while let Ok(req) = inject_rx.try_recv() {
            println!("\n[Sandbox] Received manual injection: {}", req.content);
            let new_id = aion.iter().map(|a| a.id).max().unwrap_or(0) + 1;
            let mut antecedents = RoaringBitmap::new();
            for ant in req.antecedents {
                antecedents.insert(ant);
            }
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
            // We got a user event, so we reset the LLM query budget to let it react to the user
            max_llm_queries = 3; 
        }

        step += 1;
        println!("\n[Step {}] Selecting eligible Achronons...", step);

        let lte = LatticeTopologyEngine::new(aion.clone());
        let cce = CognitiveContextEngine::new(aion.clone());

        let batch = lte.next_eligible_batch(&registry);
        
        if batch.is_empty() {
            // Only announce stability if we actually tried to run an empty batch
            // The sleep at the bottom of the loop ensures we don't spam this.
            tx.send(LatticeEvent::StabilityReached)?;
            
            if let Some(key) = &api_key {
                if max_llm_queries > 0 {
                    let agent_name = if max_llm_queries % 2 == 0 { "The Architect" } else { "The Disruptor" };
                    let system_prompt = if max_llm_queries % 2 == 0 { &agent_architect } else { &agent_disruptor };

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
                            }
                        }
                        Err(e) => {
                            log::error!("Error querying LLM: {}", e);
                        }
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            continue; // Wait for user injection or next tick
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

    println!("Simulation complete. Server still running at http://127.0.0.1:3000");
    // Keep the process alive for the visualizer
    loop { tokio::time::sleep(std::time::Duration::from_secs(60)).await; }
}
