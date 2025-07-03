test:
	ulimit -n 1000
	PYTHONPATH=src pytest -v
