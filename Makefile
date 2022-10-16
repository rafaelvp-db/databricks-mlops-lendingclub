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

abtest:
	dbx execute lendingclub-rvp-abtest --cluster-id "0807-225846-motto493" --task main

eval:
	dbx execute lendingclub-rvp-eval --cluster-id "0807-225846-motto493" --task main

score:
	dbx execute lendingclub-rvp-score --cluster-id "0807-225846-motto493" --task main

clean:
	rm -rf *.egg-info && rm -rf .pytest_cache