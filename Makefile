.PHONY: test test-student bench demo profile

test:
	python -m pytest -q -m "not student"

test-student:
	python -m pytest -q

bench:
	python scripts/bench_gemm.py

demo:
	python scripts/run_mlp_demo.py

profile:
	python scripts/profile_gemm.py
