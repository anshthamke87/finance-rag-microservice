install:
	pip install -r requirements.txt

run-api:
	uvicorn app.main:app --reload --port 8000
