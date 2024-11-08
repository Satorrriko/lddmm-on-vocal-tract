# Project Structure

The project consists of three main parts:

## 1. Notebooks
Contains the provided notebooks with all necessary data files for running the three notebooks. All code is ready to run directly. Compared to the original data, I have added all required data files for running these three notebooks.

## 2. LDDMM Transformation for Vocal Tract
Based on the three vocal tract models in the provided TractProjection, I implemented LDDMM transformation to morph the vocal tract models. For each vocal tract model:
- Applied mesh decimation to adjust the resolution to 1000 faces
- Performed LDDMM transformation to morph it into another vocal tract model
- The transformation results can be viewed in morphing_result_3d_grid.gif

## 3. VCV Classification
Implemented the preprocessing stage of VCV classification task based on the USC-TIMIT dataset, including:
- Segmentation of VCV segments
- Video clip extraction
- Extraction of phi values during the transformation
- Dataset creation using VCV phi values as input and VCV categories as output
A clear VCV transformation result can be seen in out_translation.gif

Source: Y. Yue et al., "Towards Speech Classification from Acoustic and Vocal Tract data in Real-time MRI," in Interspeech 2024, ISCA, Sep. 2024, pp. 1345–1349. doi: 10.21437/Interspeech.2024-840.

# Running Instructions
To run pylddmm:
1. Install all dependencies in requirements.txt
2. Clone repository: `git clone https://github.com/SteffenCzolbe/pyLDDMM/tree/master`
3. Additional required libraries can be found in the ipynb files

# LDDMM Transformation for 3D Meshes

## Introduction
While pyLDDMM primarily supports 2D image transformation, this implementation extends the Large Deformation Diffeomorphic Metric Mapping (LDDMM) framework to 3D meshes. LDDMM generates smooth, invertible transformations between shapes while maintaining their topological properties.

## Mathematical Framework (3D)

### 1. Diffeomorphic Flow
The transformation is modeled as a flow of diffeomorphisms φ_t, governed by a time-dependent velocity field v_t:

```
∂φ_t/∂t = v_t ∘ φ_t
φ_0 = Id (Identity mapping)
```

### 2. Energy Functional
The LDDMM optimization minimizes the energy functional:

```
E(v) = ∫_0^1 ||v_t||_V^2 dt + 1/σ^2 ||φ_1(I_0) - I_1||_L^2
```

where:
- `||v_t||_V` is the norm in the Reproducing Kernel Hilbert Space (RKHS)
- `σ` is the noise parameter
- `I_0` and `I_1` are the source and target shapes respectively

### 3. Kernel Definition
The velocity field is defined through a Gaussian kernel:

```
K(x,y) = exp(-||x-y||^2 / (2σ^2))
```

## Algorithm Flow

1. **Initialization**
   - Input: Source mesh vertices X_0 and target mesh vertices X_1
   - Initialize momentum vectors p_0

2. **Forward Integration**
   - Compute velocity field: v_t = K * p_t
   - Update positions via φ_t using Euler integration:
     ```
     φ_(t+dt) = φ_t + dt * v_t
     ```

3. **Backward Integration**
   - Compute gradient of matching term
   - Update momentum vectors using adjoint equations
   - Apply gradient descent step

4. **Convergence**
   - Iterate until the energy functional converges or maximum iterations reached

## Implementation Features

1. **Gaussian Kernel Operation**
   - Efficient computation of pairwise distances
   - GPU acceleration for large meshes

2. **Geodesic Path**
   - Smooth interpolation between source and target
   - Preservation of mesh topology

3. **Momentum-based Optimization**
   - Memory-efficient implementation
   - Adaptive step size control