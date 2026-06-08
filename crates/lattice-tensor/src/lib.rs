use ndarray::{Array1, Array2};
use lattice_core::Achronon;
use anyhow::{Result, Context};
use std::collections::HashMap;

pub struct TensorTransformationEngine {
    /// The current latent state space vector.
    pub state: Array1<f32>,
    /// A map of transformation IDs to their corresponding matrices.
    pub operators: HashMap<String, Array2<f32>>,
}

impl TensorTransformationEngine {
    pub fn new(dimension: usize) -> Self {
        Self {
            state: Array1::zeros(dimension),
            operators: HashMap::new(),
        }
    }

    pub fn register_operator(&mut self, id: String, matrix: Array2<f32>) {
        self.operators.insert(id, matrix);
    }

    /// Applies the transformation of a single Achronon to the state.
    pub fn apply_achronon(&mut self, achronon: &Achronon) -> Result<()> {
        let op = self.operators.get(&achronon.transformation_id)
            .with_context(|| format!("Operator {} not found", achronon.transformation_id))?;
        
        // State update: s' = T_a * s
        // For simplicity, we assume state is a column vector and we do matrix-vector multiplication.
        self.state = op.dot(&self.state);
        Ok(())
    }

    /// Applies a batch of orthogonal Achronons.
    /// According to the paper, these commute, so order doesn't matter.
    pub fn apply_batch(&mut self, batch: &[Achronon]) -> Result<()> {
        for achronon in batch {
            self.apply_achronon(achronon)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use roaring::RoaringBitmap;

    #[test]
    fn test_transformation() {
        let mut tte = TensorTransformationEngine::new(2);
        tte.state = array![1.0, 0.0];

        // Rotation matrix (90 degrees)
        let rot = array![[0.0, -1.0], [1.0, 0.0]];
        tte.register_operator("rot90".into(), rot);

        let achronon = Achronon {
            id: 1,
            antecedents: RoaringBitmap::new(),
            orthogonals: RoaringBitmap::new(),
            transformation_id: "rot90".into(),
            content: "Rotating...".into(),
        };

        tte.apply_achronon(&achronon).unwrap();
        assert_eq!(tte.state, array![0.0, 1.0]);
    }
}
