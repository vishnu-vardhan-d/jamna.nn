# Jamna · Java Neural Nets

A mini **Transformer / LLM engine** built from scratch in **pure Java** — no libraries,
not even a math one. Jamna learns to sort numbers the way a language model learns to
write: by predicting the next token.

## ▶ Live interactive explainer

### https://vishnu-vardhan-d.github.io/jamna.nn/

A click-through walkthrough of exactly how Jamna works — embeddings, attention, softmax,
and the weights adjusting as she trains — all with real, trained numbers.

## Run it

```bash
mvn -q exec:java                                    # train, then chat with Jamna
java -cp target/classes com.jamna.nn.Jamna chat     # or load the trained model and chat
```

## License

MIT © 2026 Vishnu Vardhan
