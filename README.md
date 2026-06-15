# Jamna
### Java Neural Nets

> A mini **Transformer / LLM engine**, built from scratch in **pure Java** — no libraries,
> not even a math one. Jamna *learns* to sort numbers the way a language model learns to
> write: by predicting the next token. She was never given a sorting algorithm — she
> discovers it from data.

![build](https://github.com/<your-username>/jamna.nn/actions/workflows/ci.yml/badge.svg)
![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Java 11+](https://img.shields.io/badge/Java-11%2B-orange.svg)

![training run in Eclipse](docs/screenshots/eclipse-training.png)
![Jamna project in Eclipse](docs/screenshots/eclipse-structure.png)
> Trained live in Eclipse to near-zero loss — **98.7% exact-match sort accuracy** — sorting on the spot.

## Meet Jamna

```
$ java -cp target/classes com.jamna.nn.Jamna chat
==================================================
  Hello, it's me -- Jamna.   (Java Neural Nets)
==================================================
you > 5 2 8 1 2 0
Jamna (thinking with my neural net)...  [0, 1, 2, 2, 5, 8]   -- nailed it!
you > 42 7 13 1 99 4 8
Jamna: that's outside what my tiny brain trained on (six single digits 0-9),
       but I'll happily sort it the classic way:  [1, 4, 7, 8, 13, 42, 99]
you > quit
Jamna: Leaving already? I'll keep my weights warm for you. Come back soon.
```

## What is this?

`Jamna` (`com.jamna.nn`) is a complete **decoder-only Transformer** (the same family as GPT)
in **one heavily-commented Java file** — token + position embeddings, a hand-written
reverse-mode autograd engine, multi-head self-attention, LayerNorm, feed-forward layers,
residuals, an Adam optimizer, and a numerically-stable softmax. No frameworks.

It trains on the toy task of **sorting six digits (0–9)** by next-token prediction, reaching
~99% accuracy — and it's the seed of a growing **JVM test-bench for small models**. In the
spirit of Karpathy's `llm.c`, but in Java, and fully your own.

## Build & run (Maven)

```bash
mvn -q exec:java                                   # train, report accuracy, then chat
mvn -q test                                        # accuracy regression test
java -cp target/classes com.jamna.nn.Jamna         # same as exec:java
java -cp target/classes com.jamna.nn.Jamna chat    # load jamna.bin and chat (no training)
```

A trained `jamna.bin` ships with the repo, so `... Jamna chat` works right away.

**In Eclipse:** File → Import → *Existing Maven Projects* → pick this folder → run `Jamna.main`.

## How it's built (everything from scratch)

| Concept | Where in `Jamna.java` |
|---|---|
| Reverse-mode autograd (backprop) | `class T` + `backprop()` |
| Matrix multiply / dot product | `T.matmul` |
| Softmax (numerically stable) | `T.softmaxRows` |
| Scaled dot-product attention | `Head.forward` |
| Multi-head attention | `MultiHead` |
| LayerNorm, feed-forward, residuals | `LayerNorm`, `FeedForward`, `Block.forward` |
| Adam optimizer | `adamStep` |

## Interactive explainer

A full visual, interactive walkthrough — every concept derived with real trained numbers —
is published to **GitHub Pages** on each push to `main`:
**https://&lt;your-username&gt;.github.io/jamna.nn/** (or open `docs/index.html` locally).

## Project layout

```
src/main/java/com/jamna/nn/Jamna.java       the whole engine (one commented file)
src/test/java/com/jamna/nn/JamnaTest.java   accuracy regression test (JUnit)
docs/                                       interactive explainer (served by GitHub Pages)
jamna.bin                                   a pre-trained model
pom.xml                                     Maven build
```

## Roadmap

A from-scratch JVM test-bench for small models: more tasks beyond sorting, pluggable
architectures, metrics and benchmarks — grown over time.

## Contributing

`main` is stable; do work on `dev` and merge via pull request. GitHub Actions runs the
tests on every push to `main`/`dev`, and publishes `docs/` to GitHub Pages on `main`.

## License

[MIT](LICENSE) © 2026 Vishnu Vardhan
