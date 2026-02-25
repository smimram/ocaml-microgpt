all:
	$(MAKE) -C src $@

test:
	@dune runtest

doc:
	@dune build @doc
