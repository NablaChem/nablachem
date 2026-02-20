test:
	ulimit -n 1000
	PYTHONPATH=src pytest -v --testmon
spacedb:
	python maintenance/space_db.py	
