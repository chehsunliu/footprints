%equirements.txt: %equirements.in
	@echo "> Compile $<"
	@pip-compile -o $@ $<

.PHONY: sync-deps
sync-deps: compile-deps
	@echo "> Synchronize dependencies"
	@pip-sync ./dev-requirements.txt ./requirements.txt

.PHONY: compile-deps
compile-deps: requirements.txt dev-requirements.txt

dev-requirements.txt: requirements.txt
