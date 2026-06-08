use lattice_core::{Achronon, PrecipitationRegistry, LatticeTopologyEngine};
use lattice_tensor::TensorTransformationEngine;
use lattice_cce::CognitiveContextEngine;
use lattice_llm::AnthropicClient;
use roaring::RoaringBitmap;
use candle_core::{Tensor, Device, DType};
use anyhow::Result;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();
    println!("--- Acausal Lattice: Multi-Agent Dynamic Prototype ---");

    let api_key = env::var("ANTHROPIC_API_KEY").ok();
    if api_key.is_none() {
        println!("Warning: ANTHROPIC_API_KEY not set in .env. System will reach stability and stop.");
    }

    // Agent Personas
    let architect_prompt = r#"
You are "The Architect", an agent of order and structural expansion in an Acausal Lattice.
Your goal is to build complex, stable systems.
Output ONLY a JSON array of new potential Achronons.
"#;

    let disruptor_prompt = r#"
You are "The Disruptor", an agent of entropy and unexpected shifts in an Acausal Lattice.
Your goal is to introduce complications, anomalies, and radical shifts in direction.
Output ONLY a JSON array of new potential Achronons.
"#;

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

    // 1. Initialize Aion (The potentiality web)
    let mut aion = Vec::new();

    // Initial seed events
    aion.push(Achronon {
        id: 1,
        antecedents: RoaringBitmap::new(),
        orthogonals: RoaringBitmap::new(),
        transformation_id: "identity".into(),
        content: "The original inquiry is formulated.".into(),
        affected_subspace: None,
    });

    let mut p2 = RoaringBitmap::new();
    p2.insert(1);
    aion.push(Achronon {
        id: 2,
        antecedents: p2,
        orthogonals: RoaringBitmap::new(),
        transformation_id: "identity".into(),
        content: "Orthogonal architecture plan is finalized.".into(),
        affected_subspace: None,
    });

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

    // 3. The Continuous Precipitation Loop
    let mut step = 0;
    let mut max_llm_queries = 3;
    
    loop {
        step += 1;
        println!("\n[Step {}] Selecting eligible Achronons...", step);

        // Engines are re-initialized with the potentially expanded aion
        let lte = LatticeTopologyEngine::new(aion.clone());
        let cce = CognitiveContextEngine::new(aion.clone());

        let batch = lte.next_eligible_batch(&registry);
        
        if batch.is_empty() {
            println!("Lattice has reached stability.");
            
            if let Some(key) = &api_key {
                if max_llm_queries > 0 {
                    let agent_name = if max_llm_queries % 2 == 0 { "The Architect" } else { "The Disruptor" };
                    let system_prompt = if max_llm_queries % 2 == 0 { &agent_architect } else { &agent_disruptor };

                    println!("\n[CCE] Stability detected. Querying {} for new potentialities...", agent_name);
                    let llm_client = AnthropicClient::new(key.clone(), system_prompt.clone());
                    let prompt = cce.flatten_to_prompt(&registry);
                    
                    match llm_client.generate_achronons(&prompt).await {
                        Ok(new_achronons) => {
                            if new_achronons.is_empty() {
                                println!("{} returned no new Achronons. Stability is absolute.", agent_name);
                                break;
                            }
                            println!("{} proposed {} new Achronons.", agent_name, new_achronons.len());
                            for a in new_achronons {
                                println!("  - [{}] ({}) {}", a.id, agent_name, a.content);
                                aion.push(a);
                            }
                            max_llm_queries -= 1;
                            continue; // Re-evaluate eligibility with new aion
                        }
                        Err(e) => {
                            println!("Error querying LLM: {}. Stopping.", e);
                            break;
                        }
                    }
                } else {
                    println!("Reached maximum LLM query limit.");
                    break;
                }
            } else {
                break;
            }
        }

        println!("Batch eligibility confirmed for IDs: {:?}", batch.iter().map(|a| a.id).collect::<Vec<_>>());

        // TTE Phase
        println!("TTE: Applying tensor transformations...");
        tte.apply_batch(&batch)?;

        // Precipitation Phase
        for achronon in &batch {
            println!("Precipitating Achronon {}: {}", achronon.id, achronon.content);
            registry.precipitate(achronon);
        }

        // CCE Phase
        println!("\nCCE Output:");
        println!("{}", cce.flatten_to_prompt(&registry));
        println!("Current State Vector: {}", tte.state);
    }

    println!("\nFinal State Vector: \n{}", tte.state);
    println!("Final Precipitation Registry: {:?}", registry.bits);
    Ok(())
}
