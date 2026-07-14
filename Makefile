test:
	ulimit -n 1000
	PYTHONPATH=src pytest -v --testmon
testall:
	ulimit -n 1000
	PYTHONPATH=src pytest -v --testmon -m ""
spacedb:
	python maintenance/space_db.py
format:
	black src
