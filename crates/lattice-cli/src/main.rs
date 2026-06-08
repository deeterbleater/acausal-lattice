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
    env_logger::init();
    println!("--- Acausal Lattice: LLM-Dynamic Prototype ---");

    let api_key = env::var("ANTHROPIC_API_KEY").ok();
    if api_key.is_none() {
        println!("Warning: ANTHROPIC_API_KEY not set. System will reach stability and stop.");
    }

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
                    println!("\n[CCE] Stability detected. Querying Claude for new potentialities...");
                    let llm_client = AnthropicClient::new(key.clone());
                    let prompt = cce.flatten_to_prompt(&registry);
                    
                    match llm_client.generate_achronons(&prompt).await {
                        Ok(new_achronons) => {
                            if new_achronons.is_empty() {
                                println!("Claude returned no new Achronons. Stability is absolute.");
                                break;
                            }
                            println!("Claude proposed {} new Achronons.", new_achronons.len());
                            for a in new_achronons {
                                println!("  - [{}] {}", a.id, a.content);
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
