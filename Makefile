.PHONY: up down logs test lint

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	cd backend && pytest

lint:
	cd backend && ruff check .
	cd frontend && npm run lint
