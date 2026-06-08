use candle_core::{Tensor, Device, DType};
use lattice_core::Achronon;
use anyhow::{Result, Context};
use std::collections::HashMap;

pub struct TensorTransformationEngine {
    /// The current latent state space vector.
    /// Expected shape: [dimension]
    pub state: Tensor,
    /// A map of transformation IDs to their corresponding matrices.
    /// Expected shape: [dimension, dimension]
    pub operators: HashMap<String, Tensor>,
    /// The device where tensors reside (CPU/CUDA).
    pub device: Device,
}

impl TensorTransformationEngine {
    pub fn new(dimension: usize) -> Result<Self> {
        let device = Device::Cpu;
        let state = Tensor::zeros(dimension, DType::F32, &device)?;
        Ok(Self {
            state,
            operators: HashMap::new(),
            device,
        })
    }

    /// Registers an operator for a specific subspace.
    /// subspace_index: which subspace to affect.
    /// subspace_size: the dimension of the subspace block.
    /// total_dimension: the total dimension of the state vector.
    /// block_matrix: the transformation matrix for that subspace (subspace_size x subspace_size).
    pub fn register_subspace_operator(
        &mut self,
        id: String,
        subspace_index: usize,
        subspace_size: usize,
        total_dimension: usize,
        block_matrix: Tensor,
    ) -> Result<()> {
        // Construct the full total_dimension x total_dimension matrix.
        // It starts as an identity matrix.
        let mut full_matrix_data = vec![0.0f32; total_dimension * total_dimension];
        for i in 0..total_dimension {
            full_matrix_data[i * total_dimension + i] = 1.0;
        }

        let block_data = block_matrix.to_vec2::<f32>()?;
        let start_offset = subspace_index * subspace_size;

        for r in 0..subspace_size {
            for c in 0..subspace_size {
                let full_r = start_offset + r;
                let full_c = start_offset + c;
                full_matrix_data[full_r * total_dimension + full_c] = block_data[r][c];
            }
        }

        let full_tensor = Tensor::from_vec(full_matrix_data, (total_dimension, total_dimension), &self.device)?;
        self.operators.insert(id, full_tensor);
        Ok(())
    }

    pub fn register_operator(&mut self, id: String, matrix: Tensor) {
        self.operators.insert(id, matrix);
    }

    /// Applies the transformation of a single Achronon to the state.
    pub fn apply_achronon(&mut self, achronon: &Achronon) -> Result<()> {
        let op = self.operators.get(&achronon.transformation_id)
            .with_context(|| format!("Operator {} not found", achronon.transformation_id))?;
        
        // State update: s' = T_a * s
        // In candle, matmul expects [M, K] and [K, N].
        // Our operator is [dim, dim], state is [dim].
        // We can treat state as [dim, 1], matmul, then flatten.
        let state_col = self.state.reshape((self.state.dims()[0], 1))?;
        let new_state_col = op.matmul(&state_col)?;
        self.state = new_state_col.flatten_all()?;
        
        Ok(())
    }

    /// Applies a batch of orthogonal Achronons using Tensor Fusion.
    /// According to the paper, these commute, so order doesn't matter.
    /// T_batch = T_n * T_{n-1} * ... * T_1
    pub fn apply_batch(&mut self, batch: &[Achronon]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let mut fused_op: Option<Tensor> = None;

        for achronon in batch {
            let op = self.operators.get(&achronon.transformation_id)
                .with_context(|| format!("Operator {} not found", achronon.transformation_id))?;
            
            match fused_op {
                None => fused_op = Some(op.clone()),
                Some(current) => {
                    // Tensor Fusion: T_new = T_next * T_current
                    fused_op = Some(op.matmul(&current)?);
                }
            }
        }

        if let Some(final_op) = fused_op {
            let state_col = self.state.reshape((self.state.dims()[0], 1))?;
            let new_state_col = final_op.matmul(&state_col)?;
            self.state = new_state_col.flatten_all()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringBitmap;

    #[test]
    fn test_transformation() -> Result<()> {
        let mut tte = TensorTransformationEngine::new(2)?;
        tte.state = Tensor::from_slice(&[1.0f32, 0.0f32], 2, &Device::Cpu)?;

        // Rotation matrix (90 degrees): [[0, -1], [1, 0]]
        let rot_data: [[f32; 2]; 2] = [[0.0, -1.0], [1.0, 0.0]];
        let rot = Tensor::new(&rot_data, &Device::Cpu)?;
        tte.register_operator("rot90".into(), rot);

        let achronon = Achronon {
            id: 1,
            antecedents: RoaringBitmap::new(),
            orthogonals: RoaringBitmap::new(),
            transformation_id: "rot90".into(),
            content: "Rotating...".into(),
            affected_subspace: None,
        };

        tte.apply_achronon(&achronon)?;
        
        let result: Vec<f32> = tte.state.to_vec1()?;
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_tensor_fusion() -> Result<()> {
        let mut tte = TensorTransformationEngine::new(2)?;
        tte.state = Tensor::from_slice(&[1.0f32, 0.0f32], 2, &Device::Cpu)?;

        // rot90: [[0, -1], [1, 0]]
        let rot_data: [[f32; 2]; 2] = [[0.0, -1.0], [1.0, 0.0]];
        let rot = Tensor::new(&rot_data, &Device::Cpu)?;
        tte.register_operator("rot90".into(), rot.clone());

        // Batch of two rot90s should result in 180 deg rotation: [[-1, 0], [0, -1]]
        // Expected result for [1, 0] is [-1, 0]
        let a1 = Achronon {
            id: 1,
            antecedents: RoaringBitmap::new(),
            orthogonals: RoaringBitmap::new(),
            transformation_id: "rot90".into(),
            content: "Rotate 90".into(),
            affected_subspace: None,
        };
        let a2 = Achronon {
            id: 2,
            antecedents: RoaringBitmap::new(),
            orthogonals: RoaringBitmap::new(),
            transformation_id: "rot90".into(),
            content: "Rotate 90 more".into(),
            affected_subspace: None,
        };

        tte.apply_batch(&[a1, a2])?;
        
        let result: Vec<f32> = tte.state.to_vec1()?;
        assert!((result[0] - (-1.0)).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_subspace_commutativity() -> Result<()> {
        let mut tte = TensorTransformationEngine::new(4)?;
        
        // rot90 in subspace 0 (dims 0, 1)
        let rot_data: [[f32; 2]; 2] = [[0.0, -1.0], [1.0, 0.0]];
        let rot = Tensor::new(&rot_data, &Device::Cpu)?;
        tte.register_subspace_operator("rot0".into(), 0, 2, 4, rot.clone())?;
        
        // rot90 in subspace 1 (dims 2, 3)
        tte.register_subspace_operator("rot1".into(), 1, 2, 4, rot)?;

        let t0 = tte.operators.get("rot0").unwrap();
        let t1 = tte.operators.get("rot1").unwrap();

        // T0 * T1 should equal T1 * T0
        let t0t1 = t0.matmul(t1)?;
        let t1t0 = t1.matmul(t0)?;

        let diff = (t0t1 - t1t0)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }
}
