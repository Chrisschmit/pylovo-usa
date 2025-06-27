.PHONY: fix-lint

fix-lint: ## Fix linting and formatting issues
	autopep8 --in-place --recursive --aggressive src runme plotting
	isort src runme plotting