Offline ONNX Model Directory Hierarchy (`app/src/main/assets/models/` or `filesDir/models/`)
============================================================================================

Local Model Subdirectories:
  models/
  ├── face_swap/
  │   ├── det_10g.onnx         (Required Core — SCRFD-10G_KPS Face & 5-Point Keypoint Detector, ~16.9 MB)
  │   ├── w600k_r50.onnx       (Required for Full Identity — ArcFace 512-D Embedding Extractor, ~166 MB)
  │   └── inswapper_128.onnx   (Required Core — InSwapper-128 Generator + [512, 512] emap matrix, ~529 MB)
  ├── head_swap/
  │   └── (Android GHOST 2.0 Full Head Replacement pose & cranial warp configuration)
  ├── segmentation/
  │   └── segformer_B5_ce.onnx (Optional — GHOST 2.0 19-class Head/Hair/Neck Parser, ~325 MB;
  │                             falls back to built-in Chromatic-Geodesic Matte if absent)
  ├── matting/
  │   └── modnet.onnx          (Optional — Portrait Hair-Strand Alpha Matting, ~25 MB;
  │                             falls back to built-in Joint Bilateral Guided Filter if absent)
  ├── inpainting/
  │   └── lama_fp32.onnx       (Optional — Background-Gap Disocclusion Inpainting, ~205 MB;
  │                             falls back to built-in Multi-Scale Boundary Marching Inpainter if absent)
  └── enhancement/
      └── gfpgan_1.4.onnx      (Optional — Face & Hair Detail Restoration, ~340 MB;
                                falls back to built-in Tiled Unsharp Detail Enhancer if absent)
