%equirements.txt: %equirements.in
	@echo "> Compile $<"
	@pip-compile -o $@ $<

.PHONY: compile-deps
compile-deps: requirements.txt dev-requirements.txt
dev-requirements.txt: requirements.txt
