# Run the development environment
dev:
	python initialize.py

# Run the production environment
prod:
	docker-compose up

# Run tests
test:
	python -m unittest discover -s tests -p "test_*.py"