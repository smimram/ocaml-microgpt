all:
	$(MAKE) -C src $@

test:
	@dune runtest
