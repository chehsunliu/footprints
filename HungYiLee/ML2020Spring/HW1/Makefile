%.pdf: %.tex
	@echo "> Compile $<"
	@xelatex -output-directory=`dirname $@` $<

.PHONY: docs
docs: ./docs/hw1.pdf

.PHONY: clean
clean: clean-docs

.PHONY: clean-docs
clean-docs:
	@echo "> Clean document temporary files"
	@rm -rf ./docs/*.aux ./docs/*.log
