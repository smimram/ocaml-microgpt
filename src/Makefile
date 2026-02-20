all: build

build:
	@dune build

test: input.txt
	@dune exec ./microgpt.exe

input.txt:
	wget https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt -O $@
