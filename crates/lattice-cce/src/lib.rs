use lattice_core::{Achronon, PrecipitationRegistry};
use anyhow::Result;

pub struct CognitiveContextEngine {
    /// The global set of all potential Achronons.
    pub aion: Vec<Achronon>,
}

impl CognitiveContextEngine {
    pub fn new(aion: Vec<Achronon>) -> Self {
        Self { aion }
    }

    /// Flattens the current state of the lattice into a prompt for an LLM.
    /// It identifies precipitated events and organizes them by structural dependency
    /// rather than chronological time.
    pub fn flatten_to_prompt(&self, registry: &PrecipitationRegistry) -> String {
        let mut prompt = String::new();
        prompt.push_str("### Acausal Lattice State\n\n");
        prompt.push_str("The following events have precipitated from the Aion into reality:\n\n");

        // Find precipitated events
        let precipitated: Vec<&Achronon> = self.aion
            .iter()
            .filter(|a| registry.bits.contains(a.id))
            .collect();

        if precipitated.is_empty() {
            prompt.push_str("- No events have precipitated yet.\n");
        } else {
            for achronon in precipitated {
                prompt.push_str(&format!("- [{}] {}\n", achronon.id, achronon.content));
            }
        }

        prompt.push_str("\n### Current Potentiality\n\n");
        prompt.push_str("The following events are eligible for precipitation:\n\n");

        let eligible: Vec<&Achronon> = self.aion
            .iter()
            .filter(|a| !registry.bits.contains(a.id) && registry.is_eligible(a))
            .collect();

        if eligible.is_empty() {
            prompt.push_str("- None (System has reached a terminal state or is blocked).\n");
        } else {
            for achronon in eligible {
                prompt.push_str(&format!("- [{}] (Eligible) {}\n", achronon.id, achronon.content));
            }
        }

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringBitmap;

    #[test]
    fn test_flattening() {
        let a1 = Achronon {
            id: 1,
            antecedents: RoaringBitmap::new(),
            orthogonals: RoaringBitmap::new(),
            transformation_id: "init".into(),
            content: "The void stirs.".into(),
        };

        let mut p2 = RoaringBitmap::new();
        p2.insert(1);
        let a2 = Achronon {
            id: 2,
            antecedents: p2,
            orthogonals: RoaringBitmap::new(),
            transformation_id: "light".into(),
            content: "Light separates from darkness.".into(),
        };

        let cce = CognitiveContextEngine::new(vec![a1.clone(), a2.clone()]);
        let mut registry = PrecipitationRegistry::new();

        let prompt_v0 = cce.flatten_to_prompt(&registry);
        assert!(prompt_v0.contains("No events have precipitated yet"));
        assert!(prompt_v0.contains("[1] (Eligible) The void stirs."));

        registry.precipitate(&a1);
        let prompt_v1 = cce.flatten_to_prompt(&registry);
        assert!(prompt_v1.contains("- [1] The void stirs."));
        assert!(prompt_v1.contains("[2] (Eligible) Light separates from darkness."));
    }
}
