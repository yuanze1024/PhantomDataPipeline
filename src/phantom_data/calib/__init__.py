"""Canvas calibration for the Phantom-Data bbox annotations.

Self-contained, streaming-only tooling that answers one question: which image size
were the integer pixel bboxes in ``koala36M_multi_ref_merged_filtered.parquet``
measured against? Nothing here imports the eager ``PhantomIndex`` / ``build.plan``
readers, and nothing here writes to disk except the explicit ``--out`` path.
"""
