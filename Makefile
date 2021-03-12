
test:
	pytest -W ignore::DeprecationWarning tests/*

cov:
	pytest -W ignore::DeprecationWarning --cov-report html --cov-report term-missing --cov=lantern tests/
