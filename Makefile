
test:
	pytest tests/*

cov:
	pytest --cov-report html --cov-report term-missing --cov=lantern tests/
