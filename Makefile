flake:
	flake8 ./

isort:
	isort ./

mypy:
	mypy ./

lint:
	make isort && make black && make flake  && make mypy
