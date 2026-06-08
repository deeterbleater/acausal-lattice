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

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();
    println!("--- Acausal Lattice Demo: The Rashomon Narrative ---");

    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set in .env");
    
    // Set up broadcasting for the visualizer
    let (tx, _rx) = broadcast::channel(100);
    let daemon_tx = tx.clone();
    let (inject_tx, _inject_rx) = mpsc::channel::<DaemonCommand>(32);

    // Spawn the visualization server
    tokio::spawn(async move {
        if let Err(e) = run_daemon(daemon_tx, inject_tx).await {
            log::error!("Visualization server failed: {}", e);
        }
    });

    // Agent Personas
    let smuggler_prompt = "You are Agent Smuggler. You are acting out the role of a rugged smuggler on a desert planet who just received a mysterious encrypted transmission. You are trying to decode it and find its source. Output ONLY a JSON array of new potential Achronons. ALWAYS set `affected_subspace` to 0.";
    let spy_prompt = "You are Agent Spy. You are acting out the role of a high-tech corporate spy on a city-planet who intercepted the same mysterious encrypted transmission. You are trying to steal the data core it points to. Output ONLY a JSON array of new potential Achronons. ALWAYS set `affected_subspace` to 1.";

    let base_system_instructions = r#"
Output ONLY a JSON array of objects representing new potential Achronons. 
Do not include any preamble or explanation.

JSON Schema:
{
  "id": integer (must be greater than existing IDs),
  "antecedents": [ids of prerequisites],
  "orthogonals": [ids of spacelike separated events],
  "transformation_id": "rot0" (for Smuggler) or "rot1" (for Spy),
  "content": "string description",
  "affected_subspace": 0 (for Smuggler) or 1 (for Spy)
}

RULES:
1. New events must logically follow from your previous events or the seed event.
2. Propose exactly 1 event per turn to advance your storyline.
"#;

    let agent_smuggler = format!("{}\n{}", smuggler_prompt, base_system_instructions);
    let agent_spy = format!("{}\n{}", spy_prompt, base_system_instructions);

    // 1. Initialize Aion
    let mut aion = Vec::new();

    // The Shared Seed
    let seed1 = Achronon {
        id: 1,
        antecedents: RoaringBitmap::new(),
        orthogonals: RoaringBitmap::new(),
        transformation_id: "identity".into(),
        content: "SEED: A mysterious encrypted transmission is broadcast across the galaxy, pointing to a hidden data core.".into(),
        affected_subspace: None,
    };
    aion.push(seed1.clone());

    // The Convergence Event (planted ahead of time)
    // This requires BOTH storylines to progress at least a few steps before it can precipitate.
    // We will dynamically update its antecedents as the LLMs generate events, but we'll 
    // seed it now so it's visible in the "Potentiality" pool.
    let mut p_convergence = RoaringBitmap::new();
    // We'll require event ID 5 (from Smuggler) and ID 6 (from Spy). 
    // We assume the LLMs will generate IDs sequentially.
    p_convergence.insert(5);
    p_convergence.insert(6);
    let convergence = Achronon {
        id: 100, // High ID so it stays at the end
        antecedents: p_convergence,
        orthogonals: RoaringBitmap::new(),
        transformation_id: "identity".into(),
        content: "CONVERGENCE: The Smuggler and the Spy arrive at the hidden orbital station simultaneously. A Mexican standoff ensues over the data core.".into(),
        affected_subspace: None, // Affects the whole lattice
    };
    aion.push(convergence.clone());

    tx.send(LatticeEvent::AionExpanded(vec![seed1, convergence]))?;

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
    tx.send(LatticeEvent::StateUpdated(tte.state.to_vec1()?))?;

    // 3. Loop
    let mut step = 0;
    // We will run 3 iterations for each agent to build their parallel storylines
    let mut agent_turns = vec!["Smuggler", "Spy", "Smuggler", "Spy", "Smuggler", "Spy"];
    
    loop {
        step += 1;
        println!("\n[Step {}] Selecting eligible Achronons...", step);

        let lte = LatticeTopologyEngine::new(aion.clone());
        let cce = CognitiveContextEngine::new(aion.clone());

        let batch = lte.next_eligible_batch(&registry);
        
        if batch.is_empty() {
            println!("Lattice has reached stability.");
            tx.send(LatticeEvent::StabilityReached)?;
            
            if let Some(agent_name) = agent_turns.pop() {
                let system_prompt = if agent_name == "Smuggler" { &agent_smuggler } else { &agent_spy };

                tx.send(LatticeEvent::Message(format!("Querying {}...", agent_name)))?;
                
                let llm_client = AnthropicClient::new(api_key.clone(), system_prompt.clone());
                let prompt = cce.flatten_to_prompt(&registry);
                
                match llm_client.generate_achronons(&prompt).await {
                    Ok(mut new_achronons) => {
                        if !new_achronons.is_empty() {
                            // Enforce subspace and transformation rules for the demo
                            for a in &mut new_achronons {
                                if agent_name == "Smuggler" {
                                    a.affected_subspace = Some(0);
                                    a.transformation_id = "rot0".into();
                                } else {
                                    a.affected_subspace = Some(1);
                                    a.transformation_id = "rot1".into();
                                }
                                a.content = format!("({}) {}", agent_name, a.content);
                            }

                            tx.send(LatticeEvent::AionExpanded(new_achronons.clone()))?;
                            for a in new_achronons {
                                aion.push(a);
                            }
                        }
                    }
                    Err(e) => log::error!("Error querying LLM: {}", e),
                }
                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                continue;
            } else {
                // If we've run out of turns and the batch is empty, 
                // it means the convergence event either precipitated or we are done.
                println!("Narrative complete.");
                break; 
            }
        }

        println!("Batch eligibility confirmed for IDs: {:?}", batch.iter().map(|a| a.id).collect::<Vec<_>>());

        // TTE Phase
        tte.apply_batch(&batch)?;
        tx.send(LatticeEvent::StateUpdated(tte.state.to_vec1()?))?;

        // Precipitation Phase
        for achronon in &batch {
            println!("Precipitating Achronon {}: {}", achronon.id, achronon.content);
            registry.precipitate(achronon);
            tx.send(LatticeEvent::AchrononPrecipitated(achronon.id))?;
            tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        }

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    println!("Simulation complete. Server still running at http://127.0.0.1:3000");
    loop { tokio::time::sleep(std::time::Duration::from_secs(60)).await; }
}
