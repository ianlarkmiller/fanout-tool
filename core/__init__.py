"""Core library for the query fan-out tool.

Pure, importable functions (no CLI, no file I/O, no env reads, no stdout). API keys and in-memory
data are passed as arguments; functions return data/strings. Lifted from the tested scripts in the
job-search repo (fanout.py, model_fanout.py, unified_fanout.py, unified_brief.py, cluster_patterns.py).
"""
