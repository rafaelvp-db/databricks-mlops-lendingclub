env:
	rm -rf .venv && python -m venv .venv && source .venv/bin/activate && pip install -r unit-requirements.txt && pip install -e .

deploy:
	dbx deploy

unit:
	pytest tests/unit

test:
	dbx execute lendingclub-rvp-test --cluster-id "0807-225846-motto493" --task main

train:
	dbx execute lendingclub-rvp-train --cluster-id "0807-225846-motto493" --task main

clean:
	rm -rf *.egg-info && rm -rf .pytest_cache