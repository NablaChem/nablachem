test:
	ulimit -n 1000
	PYTHONPATH=src pytest -v
spacedb:
	python maintenance/space_db.py	
