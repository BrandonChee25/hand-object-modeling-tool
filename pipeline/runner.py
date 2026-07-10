"""Pipeline runner — chains all stages in order."""

from __future__ import annotations
from pathlib import Path

from pipeline.data import PipelineData
from pipeline.stages.s1_preprocessing import PreprocessingStage
from pipeline.stages.s2_hand_recon import HandReconstructionStage
from pipeline.stages.s3_object_seg import ObjectSegmentationStage
from pipeline.stages.s4_object_mesh import ObjectMeshGenerationStage
from pipeline.stages.s5_object_pose import ObjectPoseEstimationStage
from pipeline.stages.s6_alignment import AlignmentStage
from pipeline.stages.s7_export import ExportStage


class PipelineRunner:
    """Instantiates and runs all pipeline stages from a single config dict."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.s1 = PreprocessingStage(cfg)
        self.s2 = HandReconstructionStage(cfg)
        self.s3 = ObjectSegmentationStage(cfg)
        self.s4 = ObjectMeshGenerationStage(cfg)
        self.s5 = ObjectPoseEstimationStage(cfg)
        self.s6 = AlignmentStage(cfg)
        self.s7 = ExportStage(cfg)

    def run(self, frames_dir: Path, output_dir: Path) -> Path:
        data = PipelineData(output_dir=output_dir)

        print("[1/7] Preprocessing frames...")
        data = self.s1.run(frames_dir, data)

        print(f"[2/7] Hand reconstruction ({len(data.frames)} frames)...")
        data = self.s2.run(data)

        print("[3/7] Object segmentation...")
        data = self.s3.run(data)

        print("[4/7] Blind object mesh generation (SAM-3D)...")
        data = self.s4.run(data)

        # Free memory from earlier stages before Stage 5's DINOv2 model loads.
        import torch, gc
        import models.wilor as _wilor_mod
        import models.moge_wrapper, models.sam2_wrapper, models.sam3d_wrapper
        try:
            import models.sam3_seg_wrapper  # noqa: F401
        except ImportError:
            pass

        # Clear module-level caches
        _wilor_mod._pipeline_cache.clear()

        # Move any loaded models to CPU then delete
        for stage in [self.s1, self.s2, self.s3, self.s4]:
            if stage is None:
                continue
            for val in vars(stage).values():
                for attr in ('_model', '_predictor', '_auto_gen', '_video_predictor'):
                    m = getattr(val, attr, None)
                    if m is not None and hasattr(m, 'cpu'):
                        m.cpu()
                        setattr(val, attr, None)

        self.s1 = None
        self.s2 = None
        self.s3 = None
        self.s4 = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print("[5/7] Object pose estimation...")
        data = self.s5.run(data)

        print("[6/7] Hand-object metric alignment...")
        data = self.s6.run(data)

        print("[7/7] Exporting meshes...")
        out = self.s7.run(data, output_dir)

        print(f"Done. Output written to: {out}")
        return out
