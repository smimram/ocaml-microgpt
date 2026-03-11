MicroGPT in OCaml
=================

This is a re-implementation of Andrej Karpathy's microgpt, a stripped down version of GPT-2, in OCaml in order to understand it in more details. You can read about it in

- [the original blog post](https://karpathy.github.io/2026/02/12/microgpt/)
- [the original python code](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [the makemore code](https://github.com/karpathy/makemore)

Although I have tried to remain concise, I have favored idiomatic code over short one.

## Running microGPT

In order to run the code, simply type

```bash
dune exec src/microgpt.exe
```

This will download the training file, train the LLM, and generate names:

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.0274
--- inference (new, hallucinated names) ---
sample 01: eill
sample 02: rulte
sample 03: ahann
sample 04: borire
sample 05: jatei
sample 06: sa
sample 07: cari
sample 08: alyli
sample 09: kamele
sample 10: amelan
sample 11: hemen
sample 12: dyana
sample 13: mamen
sample 14: arare
sample 15: bahen
sample 16: avan
sample 17: jakane
sample 18: amelen
sample 19: manin
sample 20: telian
```

## Speed

Compared to the original python code, the OCaml version runs roughly ... times faster:

```
...
```

### License

License is [MIT](LICENSE) as for the original python files.
