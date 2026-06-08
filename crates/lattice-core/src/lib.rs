use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// The atomic unit of the system.
/// Defined as a triple: Antecedents, Orthogonals, and Transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achronon {
    pub id: u32,
    /// Prerequisites that must be realized before this Achronon can precipitate.
    pub antecedents: RoaringBitmap,
    /// Events that are spacelike separated and have no causal bearing.
    /// Used for commutativity and batching.
    pub orthogonals: RoaringBitmap,
    /// The transformation operator (tensor) identifier or metadata.
    pub transformation_id: String,
    /// The semantic payload of the Achronon (e.g., text for the CCE).
    pub content: String,
}

/// The state of the system, tracking which Achronons have collapsed into reality.
#[derive(Debug, Clone, Default)]
pub struct PrecipitationRegistry {
    pub bits: RoaringBitmap,
}

impl PrecipitationRegistry {
    pub fn new() -> Self {
        Self {
            bits: RoaringBitmap::new(),
        }
    }

    /// Checks if an Achronon's antecedents have all precipitated.
    pub fn is_eligible(&self, achronon: &Achronon) -> bool {
        // Eligibility: P_a \subseteq R
        // This is equivalent to (R & P_a) == P_a
        self.bits.is_superset(&achronon.antecedents)
    }

    /// Mark an Achronon as precipitated.
    pub fn precipitate(&mut self, achronon: &Achronon) {
        self.bits.insert(achronon.id);
    }
}

/// The Lattice Topology Engine (LTE)
/// Handles the selection of eligible Achronons for the next "batch" of reality.
pub struct LatticeTopologyEngine {
    pub aion: Vec<Achronon>,
}

impl LatticeTopologyEngine {
    pub fn new(aion: Vec<Achronon>) -> Self {
        Self { aion }
    }

    /// Returns a list of Achronons that are eligible to precipitate
    /// given the current registry, excluding those that have already precipitated.
    pub fn next_eligible_batch(&self, registry: &PrecipitationRegistry) -> Vec<Achronon> {
        self.aion
            .iter()
            .filter(|a| !registry.bits.contains(a.id) && registry.is_eligible(a))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eligibility() {
        let mut registry = PrecipitationRegistry::new();
        
        let a1 = Achronon {
            id: 1,
            antecedents: RoaringBitmap::new(), // Root event
            orthogonals: RoaringBitmap::new(),
            transformation_id: "init".into(),
            content: "Initial state established.".into(),
        };

        let mut p2 = RoaringBitmap::new();
        p2.insert(1);
        let a2 = Achronon {
            id: 2,
            antecedents: p2,
            orthogonals: RoaringBitmap::new(),
            transformation_id: "step2".into(),
            content: "Step 2 realized.".into(),
        };

        assert!(registry.is_eligible(&a1));
        assert!(!registry.is_eligible(&a2));

        registry.precipitate(&a1);
        assert!(registry.is_eligible(&a2));
    }
}
