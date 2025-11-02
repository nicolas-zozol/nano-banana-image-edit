.PHONY: test test-slow test-ai

test:
	pytest -q

test-slow:
	pytest -q -m "slow"

test-ai:
	RUN_AI_TESTS=1 pytest -q -m "ai"

