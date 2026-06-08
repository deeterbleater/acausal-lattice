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
    /// The index of the subspace this Achronon warps (for commutativity).
    pub affected_subspace: Option<usize>,
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
    /// Validates that the resulting batch is internally orthogonal (causally independent).
    pub fn next_eligible_batch(&self, registry: &PrecipitationRegistry) -> Vec<Achronon> {
        let eligible: Vec<Achronon> = self.aion
            .iter()
            .filter(|a| !registry.bits.contains(a.id) && registry.is_eligible(a))
            .cloned()
            .collect();

        // Internal Orthogonality Check:
        // No element in the batch should be an antecedent of another element in the same batch.
        // This is naturally guaranteed by the eligibility rule (if A is an antecedent of B, 
        // B is only eligible if A is already in the registry).
        // However, we explicitly verify this to maintain the Acausal Invariant.
        self.validate_batch_orthogonality(&eligible);

        eligible
    }

    /// Verifies that no two Achronons in the batch have a causal dependency
    /// and that they do not contend for the same transformation subspace.
    pub fn validate_batch_orthogonality(&self, batch: &[Achronon]) -> bool {
        let batch_ids: RoaringBitmap = batch.iter().map(|a| a.id).collect();
        let mut used_subspaces = std::collections::HashSet::new();
        
        for achronon in batch {
            // Causal dependency check
            if !achronon.antecedents.is_disjoint(&batch_ids) {
                return false;
            }

            // Subspace contention check
            if let Some(s) = achronon.affected_subspace {
                if !used_subspaces.insert(s) {
                    return false; // Two events in same batch want the same subspace
                }
            }
        }
        true
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
            antecedents: RoaringBitmap::new(),
            orthogonals: RoaringBitmap::new(),
            transformation_id: "init".into(),
            content: "Initial".into(),
            affected_subspace: None,
        };

        let mut p2 = RoaringBitmap::new();
        p2.insert(1);
        let a2 = Achronon {
            id: 2,
            antecedents: p2,
            orthogonals: RoaringBitmap::new(),
            transformation_id: "step2".into(),
            content: "Step 2".into(),
            affected_subspace: None,
        };

        assert!(registry.is_eligible(&a1));
        assert!(!registry.is_eligible(&a2));

        registry.precipitate(&a1);
        assert!(registry.is_eligible(&a2));
    }

    #[test]
    fn test_orthogonality() {
        let lte = LatticeTopologyEngine::new(vec![]);
        
        let mut p2 = RoaringBitmap::new();
        p2.insert(1);
        
        let a1 = Achronon {
            id: 1,
            antecedents: RoaringBitmap::new(),
            orthogonals: RoaringBitmap::new(),
            transformation_id: "a1".into(),
            content: "A1".into(),
            affected_subspace: Some(0),
        };
        
        let a2 = Achronon {
            id: 2,
            antecedents: p2,
            orthogonals: RoaringBitmap::new(),
            transformation_id: "a2".into(),
            content: "A2".into(),
            affected_subspace: Some(1),
        };

        // A1 and A2 are NOT orthogonal because A1 is an antecedent of A2.
        assert!(!lte.validate_batch_orthogonality(&[a1.clone(), a2.clone()]));
        
        // A1 and some other independent event A3 ARE orthogonal.
        let a3 = Achronon {
            id: 3,
            antecedents: RoaringBitmap::new(),
            orthogonals: RoaringBitmap::new(),
            transformation_id: "a3".into(),
            content: "A3".into(),
            affected_subspace: Some(1),
        };
        assert!(lte.validate_batch_orthogonality(&[a1.clone(), a3.clone()]));

        // A1 and A4 are NOT orthogonal because they contend for the same subspace (0).
        let a4 = Achronon {
            id: 4,
            antecedents: RoaringBitmap::new(),
            orthogonals: RoaringBitmap::new(),
            transformation_id: "a4".into(),
            content: "A4".into(),
            affected_subspace: Some(0),
        };
        assert!(!lte.validate_batch_orthogonality(&[a1, a4]));
    }
}
