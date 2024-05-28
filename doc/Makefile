.PHONY: clean html

html:
	$(MAKE) -C ../logo/ xarpes_small.svg
	mkdir -p logo
	cp ../logo/xarpes.svg logo
	sphinx-build -M html . .

clean:
	rm -rf doctrees html
