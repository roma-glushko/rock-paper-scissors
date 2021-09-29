flake:
	flake8 ./rock_paper_scissors ./*.py

isort:
	isort ./

black:
	black .

mypy:
	mypy ./

lint:
	make isort && make black && make flake  && make mypy
