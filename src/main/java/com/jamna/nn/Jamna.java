package com.jamna.nn;

import java.io.*;
import java.util.*;

/**
 * 
 * @author Vishnu Vardhan
 * Jamna — a Mini LLM Engine from Scratch in Java.
 * =================================================
 *
 * A complete decoder-only Transformer (the same family as GPT), written in ONE
 * pure-Java file with NO external libraries — not even a math library. To make the
 * engine easy to watch end-to-end, the demo task is sorting: give it six digits and
 * it returns them in order.
 *
 *     numbers: 5 2 8 1 2 0   ->   model says: [0, 1, 2, 2, 5, 8]
 *
 * It is never given a sorting algorithm. It LEARNS to sort the way a language model
 * learns to write — by repeatedly predicting the next token. We feed it one sequence
 * made of the [unsorted 6 digits] + [sorted 6 digits] and train it to predict each
 * token of the sorted half from everything before it. Fresh random examples every
 * step mean it cannot memorise; it has to discover the rule.
 *
 * CALL HIERARCHY (who calls whom)
 * -------------------------------
 *   main
 *    |-- buildModel ........... allocates all weights (embeddings, blocks, head)
 *    |-- train .................. the learning loop
 *    |     |-- makeExample ...... invents one random (unsorted -> sorted) example
 *    |     |-- forward .......... runs the network once, returns next-token scores
 *    |     |     |-- embed ...... look up token + position vectors
 *    |     |     |-- Block.forward (x N_LAYER)
 *    |     |     |     |-- LayerNorm.forward -> T.layernorm
 *    |     |     |     |-- MultiHead.forward
 *    |     |     |     |     |-- Head.forward  (scaled dot-product attention)
 *    |     |     |     |     |     |-- T.matmul, T.transpose, T.scale,
 *    |     |     |     |     |     |-- T.addConst (causal mask), T.softmaxRows
 *    |     |     |     |     |-- T.concatCols, Linear.forward
 *    |     |     |     |-- FeedForward.forward -> Linear.forward, T.relu
 *    |     |     |     |-- T.add (residual connections)
 *    |     |     |-- LayerNorm.forward, Linear.forward (final scores)
 *    |     |-- crossEntropy ..... measures the error (only on the sorted half)
 *    |     |-- backprop ......... chain rule: fills every weight's gradient
 *    |     |-- adamStep ......... nudges every weight to reduce the error
 *    |-- accuracy / sortWithModel  evaluate by actually sorting fresh sequences
 *    |-- save / load ........... store / restore the trained weights
 *
 * Build & run:
 *     javac Jamna.java
 *     java Jamna          // train, report accuracy, then sort interactively
 *     java Jamna chat     // load saved weights (jamna.bin) and sort
 *     java Jamna 4000     // train for a custom number of steps
 *     java Jamna more 800 // continue training a saved model for 800 more steps
 */
public class Jamna {

    // ----------------------------- task + hyper-parameters -----------------------------
    static int N       = 6;    // how many numbers per sequence
    static int DIGITS  = 10;   // numbers are digits 0..9  (this is also the vocab size)
    static int N_EMBD  = 48;   // length of each token's vector ("embedding dimension")
    static int N_HEAD  = 4;    // number of attention heads
    static int N_LAYER = 2;    // number of stacked transformer blocks
    static int BLOCK;          // context length = 2*N (set in main/load)
    static int ITERS   = 2500; // training steps
    static int BATCH   = 24;   // examples averaged per optimizer step
    static double LR   = 3e-3; // Adam learning rate
    static final Random RNG = new Random(1234);

    /** Every trainable weight in the model is registered here so the optimizer can find it. */
    static final List<T> PARAMS = new ArrayList<>();

    // =====================================================================================
    //  AUTOGRAD ENGINE
    //  A T is a (rows x cols) matrix that also REMEMBERS how it was produced, so we can
    //  later run the chain rule automatically (this is "backpropagation"). Each math op
    //  below returns a new T and stores a small `backward` function that knows how to push
    //  gradients from the result back to its inputs.
    // =====================================================================================
    static final class T {
        final int r, c;            // shape
        final double[] data;       // the values, stored row-major: index = row*c + col
        final double[] grad;       // d(loss)/d(value), filled in during backprop
        Runnable backward;         // pushes this matrix's grad into its inputs' grads
        final List<T> parents = new ArrayList<>(); // the T(s) this one was computed from

        T(int r, int c) { this.r = r; this.c = c; data = new double[r*c]; grad = new double[r*c]; }

        /** Create a trainable weight matrix, filled with small random numbers, and register it. */
        static T param(int r, int c, double std) {
            T t = new T(r, c);
            for (int i = 0; i < t.data.length; i++) t.data[i] = RNG.nextGaussian() * std;
            PARAMS.add(t);
            return t;
        }

        /**
         * Matrix multiply: result = this(r x k) . other(k x c2). This is the workhorse of
         * the whole network — every "dot product" between vectors happens here.
         * Backward uses the standard rule: dThis += dOut . other^T ; dOther += this^T . dOut.
         */
        T matmul(T o) {
            int k = c, c2 = o.c;
            T out = new T(r, c2);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c2; j++) {
                    double sum = 0;
                    for (int p = 0; p < k; p++) sum += data[i*k+p] * o.data[p*c2+j];
                    out.data[i*c2+j] = sum;
                }
            out.parents.add(this); out.parents.add(o);
            final T self = this;
            out.backward = () -> {
                for (int i = 0; i < r; i++)
                    for (int j = 0; j < c2; j++) {
                        double g = out.grad[i*c2+j];
                        if (g == 0) continue;
                        for (int p = 0; p < k; p++) {
                            self.grad[i*k+p] += g * o.data[p*c2+j];
                            o.grad[p*c2+j]   += g * self.data[i*k+p];
                        }
                    }
            };
            return out;
        }

        /** Add a (1 x c) bias vector to EVERY row. Used by linear layers. */
        T addRow(T b) {
            T out = new T(r, c);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++) out.data[i*c+j] = data[i*c+j] + b.data[j];
            out.parents.add(this); out.parents.add(b);
            final T self = this;
            out.backward = () -> {
                for (int i = 0; i < r; i++)
                    for (int j = 0; j < c; j++) {
                        double g = out.grad[i*c+j];
                        self.grad[i*c+j] += g;
                        b.grad[j] += g;
                    }
            };
            return out;
        }

        /** Element-wise add of two equally-shaped matrices. This is how a RESIDUAL (skip) connection adds a layer's output back onto its input. */
        T add(T o) {
            T out = new T(r, c);
            for (int i = 0; i < data.length; i++) out.data[i] = data[i] + o.data[i];
            out.parents.add(this); out.parents.add(o);
            final T self = this;
            out.backward = () -> { for (int i = 0; i < data.length; i++) { self.grad[i] += out.grad[i]; o.grad[i] += out.grad[i]; } };
            return out;
        }

        /** Multiply every value by a constant. Used to SCALE attention scores by 1/sqrt(head size). */
        T scale(double s) {
            T out = new T(r, c);
            for (int i = 0; i < data.length; i++) out.data[i] = data[i] * s;
            out.parents.add(this);
            final T self = this;
            out.backward = () -> { for (int i = 0; i < data.length; i++) self.grad[i] += s * out.grad[i]; };
            return out;
        }

        /** ReLU: keep positives, zero out negatives. The nonlinearity inside the feed-forward network. */
        T relu() {
            T out = new T(r, c);
            for (int i = 0; i < data.length; i++) out.data[i] = data[i] > 0 ? data[i] : 0;
            out.parents.add(this);
            final T self = this;
            out.backward = () -> { for (int i = 0; i < data.length; i++) if (data[i] > 0) self.grad[i] += out.grad[i]; };
            return out;
        }

        /** Transpose (swap rows and columns). Needed to compute query . key^T inside attention. */
        T transpose() {
            T out = new T(c, r);
            for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) out.data[j*r+i] = data[i*c+j];
            out.parents.add(this);
            final T self = this;
            out.backward = () -> { for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) self.grad[i*c+j] += out.grad[j*r+i]; };
            return out;
        }

        /** Add a CONSTANT matrix (no gradient of its own). Used to apply the causal mask that stops a position from peeking at future positions. */
        T addConst(double[] m) {
            T out = new T(r, c);
            for (int i = 0; i < data.length; i++) out.data[i] = data[i] + m[i];
            out.parents.add(this);
            final T self = this;
            out.backward = () -> { for (int i = 0; i < data.length; i++) self.grad[i] += out.grad[i]; };
            return out;
        }

        /**
         * SOFTMAX applied to each row: turns a row of scores into positive numbers that add
         * up to 1 (a probability distribution). Numerically stable because it subtracts each
         * row's maximum first, so exp() can never overflow on large scores.
         */
        T softmaxRows() {
            T out = new T(r, c);
            for (int i = 0; i < r; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < c; j++) max = Math.max(max, data[i*c+j]);  // the stability trick
                double sum = 0;
                for (int j = 0; j < c; j++) { double e = Math.exp(data[i*c+j] - max); out.data[i*c+j] = e; sum += e; }
                for (int j = 0; j < c; j++) out.data[i*c+j] /= sum;
            }
            out.parents.add(this);
            final T self = this;
            out.backward = () -> {
                for (int i = 0; i < r; i++) {
                    double dot = 0;
                    for (int j = 0; j < c; j++) dot += out.grad[i*c+j] * out.data[i*c+j];
                    for (int j = 0; j < c; j++) {
                        double y = out.data[i*c+j];
                        self.grad[i*c+j] += y * (out.grad[i*c+j] - dot);
                    }
                }
            };
            return out;
        }

        /**
         * LAYER NORM on each row: rescale a row so it has mean 0 and spread 1, then apply a
         * learnable scale (gamma) and shift (beta). Keeps the numbers well-behaved between
         * layers so training stays stable. The backward pass is the standard closed form.
         */
        T layernorm(T gamma, T beta) {
            T out = new T(r, c);
            double[] mean = new double[r], inv = new double[r];
            double eps = 1e-5;
            for (int i = 0; i < r; i++) {
                double m = 0; for (int j = 0; j < c; j++) m += data[i*c+j]; m /= c;
                double v = 0; for (int j = 0; j < c; j++) { double d = data[i*c+j]-m; v += d*d; } v /= c;
                double iv = 1.0/Math.sqrt(v+eps);
                mean[i] = m; inv[i] = iv;
                for (int j = 0; j < c; j++) out.data[i*c+j] = (data[i*c+j]-m)*iv*gamma.data[j] + beta.data[j];
            }
            out.parents.add(this); out.parents.add(gamma); out.parents.add(beta);
            final T self = this;
            out.backward = () -> {
                for (int i = 0; i < r; i++) {
                    double iv = inv[i], m = mean[i];
                    double[] xhat = new double[c];
                    for (int j = 0; j < c; j++) xhat[j] = (self.data[i*c+j]-m)*iv;
                    double sdy = 0, sdyx = 0;
                    for (int j = 0; j < c; j++) {
                        double dy = out.grad[i*c+j]*gamma.data[j];
                        sdy += dy; sdyx += dy*xhat[j];
                        gamma.grad[j] += out.grad[i*c+j]*xhat[j];
                        beta.grad[j]  += out.grad[i*c+j];
                    }
                    for (int j = 0; j < c; j++) {
                        double dy = out.grad[i*c+j]*gamma.data[j];
                        self.grad[i*c+j] += iv/c * (c*dy - sdy - xhat[j]*sdyx);
                    }
                }
            };
            return out;
        }

        /** Glue several (r x ci) matrices side-by-side into one (r x sum ci) matrix. Used to join the outputs of the attention heads. */
        static T concatCols(List<T> parts) {
            int rr = parts.get(0).r, cc = 0;
            for (T p : parts) cc += p.c;
            T out = new T(rr, cc);
            int off = 0;
            for (T p : parts) {
                for (int i = 0; i < rr; i++) for (int j = 0; j < p.c; j++) out.data[i*cc + off + j] = p.data[i*p.c+j];
                off += p.c;
            }
            out.parents.addAll(parts);
            final int CC = cc;
            out.backward = () -> {
                int o = 0;
                for (T p : parts) {
                    for (int i = 0; i < rr; i++) for (int j = 0; j < p.c; j++) p.grad[i*p.c+j] += out.grad[i*CC + o + j];
                    o += p.c;
                }
            };
            return out;
        }
    }

    /** Look up rows of a table by id: turns a list of token ids into the matrix of their vectors. Backward scatters the gradient back to the rows that were used. */
    static T embed(T W, int[] ids) {
        int len = ids.length, d = W.c;
        T out = new T(len, d);
        for (int i = 0; i < len; i++) System.arraycopy(W.data, ids[i]*d, out.data, i*d, d);
        out.parents.add(W);
        out.backward = () -> { for (int i = 0; i < len; i++) for (int j = 0; j < d; j++) W.grad[ids[i]*d+j] += out.grad[i*d+j]; };
        return out;
    }

    /**
     * CROSS-ENTROPY loss: how wrong the predicted next-token probabilities are, averaged
     * only over the positions we care about (the sorted-output half — `active` marks them).
     * Its backward conveniently simplifies to (predicted probability - 1 at the correct digit).
     */
    static T crossEntropy(T logits, int[] targets, boolean[] active) {
        int len = logits.r, V = logits.c;
        int cnt = 0; for (boolean a : active) if (a) cnt++;
        final int count = cnt;
        T loss = new T(1, 1);
        double total = 0;
        double[][] probs = new double[len][V];
        for (int i = 0; i < len; i++) {
            if (!active[i]) continue;
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < V; j++) max = Math.max(max, logits.data[i*V+j]);
            double sum = 0;
            for (int j = 0; j < V; j++) { double e = Math.exp(logits.data[i*V+j]-max); probs[i][j] = e; sum += e; }
            for (int j = 0; j < V; j++) probs[i][j] /= sum;
            total += -Math.log(Math.max(probs[i][targets[i]], 1e-12));
        }
        loss.data[0] = total / count;
        loss.parents.add(logits);
        loss.backward = () -> {
            double g = loss.grad[0] / count;
            for (int i = 0; i < len; i++) if (active[i])
                for (int j = 0; j < V; j++) {
                    double d = probs[i][j] - (j == targets[i] ? 1.0 : 0.0);
                    logits.grad[i*V+j] += g * d;
                }
        };
        return loss;
    }

    /** Run backpropagation from a loss whose grad has already been seeded: visit nodes in reverse build order and call each one's backward. */
    static void backprop(T loss) {
        List<T> order = new ArrayList<>();
        Set<T> seen = new HashSet<>();
        topo(loss, seen, order);
        for (int i = order.size()-1; i >= 0; i--)
            if (order.get(i).backward != null) order.get(i).backward.run();
    }
    /** Depth-first topological sort: lists every T that fed into `n`, inputs before outputs. */
    static void topo(T n, Set<T> seen, List<T> order) {
        if (seen.contains(n)) return;
        seen.add(n);
        for (T p : n.parents) topo(p, seen, order);
        order.add(n);
    }

    // =====================================================================================
    //  LAYERS  (small reusable building blocks made from the T operations above)
    // =====================================================================================

    /** A linear (fully-connected) layer: y = x . W (+ optional bias b). */
    static final class Linear {
        T W, b;
        Linear(int in, int out, boolean bias) { W = T.param(in, out, 0.02); b = bias ? T.param(1, out, 0.0) : null; }
        T forward(T x) { T y = x.matmul(W); return b == null ? y : y.addRow(b); }
    }

    /** A LayerNorm layer holding its learnable scale (g) and shift (be). */
    static final class LayerNorm {
        T g, be;
        LayerNorm(int d) { g = T.param(1, d, 0.0); be = T.param(1, d, 0.0); for (int i = 0; i < d; i++) g.data[i] = 1.0; }
        T forward(T x) { return x.layernorm(g, be); }
    }

    /**
     * One ATTENTION HEAD (scaled dot-product attention, causal).
     * Each position builds a Query, every position a Key and Value. The match between a
     * query and the keys (their dot product, scaled) decides how much of each value to pull
     * in. The causal mask blocks looking ahead; softmax makes the weights sum to 1.
     */
    static final class Head {
        Linear key, query, value; int hs;
        Head(int headSize) {
            hs = headSize;
            key   = new Linear(N_EMBD, hs, false);
            query = new Linear(N_EMBD, hs, false);
            value = new Linear(N_EMBD, hs, false);
        }
        T forward(T x, double[] mask) {
            T k = key.forward(x);
            T q = query.forward(x);
            T v = value.forward(x);
            T scores = q.matmul(k.transpose()).scale(1.0/Math.sqrt(hs)); // how well each query matches each key
            scores = scores.addConst(mask);                             // forbid looking at the future
            T weights = scores.softmaxRows();                           // attention weights (sum to 1 per row)
            return weights.matmul(v);                                   // weighted blend of the values
        }
    }

    /** MULTI-HEAD attention: run several heads in parallel, concatenate, then mix with one more linear layer. */
    static final class MultiHead {
        Head[] heads; Linear proj;
        MultiHead() {
            heads = new Head[N_HEAD];
            int hs = N_EMBD / N_HEAD;
            for (int i = 0; i < N_HEAD; i++) heads[i] = new Head(hs);
            proj = new Linear(N_EMBD, N_EMBD, true);
        }
        T forward(T x, double[] mask) {
            List<T> outs = new ArrayList<>();
            for (Head h : heads) outs.add(h.forward(x, mask));
            return proj.forward(T.concatCols(outs));
        }
    }

    /** Position-wise FEED-FORWARD network: expand to 4x width, apply ReLU, project back. Lets each position "think" on its own. */
    static final class FeedForward {
        Linear fc1, fc2;
        FeedForward() { fc1 = new Linear(N_EMBD, 4*N_EMBD, true); fc2 = new Linear(4*N_EMBD, N_EMBD, true); }
        T forward(T x) { return fc2.forward(fc1.forward(x).relu()); }
    }

    /**
     * One TRANSFORMER BLOCK (pre-norm style): normalise, attend, add back (residual);
     * then normalise, feed-forward, add back. Stacking these is what makes the model deep.
     */
    static final class Block {
        LayerNorm ln1, ln2; MultiHead attn; FeedForward ff;
        Block() { ln1 = new LayerNorm(N_EMBD); ln2 = new LayerNorm(N_EMBD); attn = new MultiHead(); ff = new FeedForward(); }
        T forward(T x, double[] mask) {
            x = x.add(attn.forward(ln1.forward(x), mask)); // attention sublayer + residual
            x = x.add(ff.forward(ln2.forward(x)));         // feed-forward sublayer + residual
            return x;
        }
    }

    // =====================================================================================
    //  THE MODEL
    // =====================================================================================
    static T tokEmb;       // vector for each possible digit
    static T posEmb;       // vector for each position in the sequence
    static Block[] blocks; // the stack of transformer blocks
    static LayerNorm lnf;   // a final normalisation
    static Linear lmHead;  // projects the final vectors to one score per digit

    /** Allocate every weight in the model. The order here defines the order weights are saved/loaded in. */
    static void buildModel() {
        tokEmb = T.param(DIGITS, N_EMBD, 0.02);
        posEmb = T.param(BLOCK, N_EMBD, 0.02);
        blocks = new Block[N_LAYER];
        for (int i = 0; i < N_LAYER; i++) blocks[i] = new Block();
        lnf = new LayerNorm(N_EMBD);
        lmHead = new Linear(N_EMBD, DIGITS, false);
    }

    /** Build the causal mask for a length-Tn sequence: 0 on/below the diagonal, a big negative number above it (so softmax sends those to ~0). */
    static double[] causalMask(int Tn) {
        double[] m = new double[Tn*Tn];
        for (int i = 0; i < Tn; i++)
            for (int j = 0; j < Tn; j++) m[i*Tn+j] = (j > i) ? -1e9 : 0.0;
        return m;
    }

    /**
     * THE FORWARD PASS. Turn a list of token ids into next-token scores.
     * idx (length Tn)  ->  logits (Tn x DIGITS): for each position, a score per possible digit.
     */
    static T forward(int[] idx) {
        int Tn = idx.length;
        int[] pos = new int[Tn];
        for (int i = 0; i < Tn; i++) pos[i] = i;
        T x = embed(tokEmb, idx).add(embed(posEmb, pos)); // start: token meaning + position
        double[] mask = causalMask(Tn);
        for (Block b : blocks) x = b.forward(x, mask);     // think, block by block
        x = lnf.forward(x);
        return lmHead.forward(x);                          // final scores
    }

    // =====================================================================================
    //  ADAM OPTIMIZER  (the rule that updates each weight using its gradient)
    // =====================================================================================
    static double[][] mM, vM;  // running averages Adam keeps per weight
    static int adamT = 0;      // step counter (for bias correction)
    /** Allocate Adam's per-weight memory. Call once before training. */
    static void initAdam() {
        mM = new double[PARAMS.size()][]; vM = new double[PARAMS.size()][];
        for (int i = 0; i < PARAMS.size(); i++) { mM[i] = new double[PARAMS.get(i).data.length]; vM[i] = new double[PARAMS.get(i).data.length]; }
    }
    /** Reset every weight's gradient to zero (gradients accumulate across a batch, so we clear them each step). */
    static void zeroGrad() { for (T p : PARAMS) Arrays.fill(p.grad, 0.0); }
    /** One Adam update: move each weight a little in the direction that reduces the loss, scaled by a smoothed estimate of its gradient. */
    static void adamStep() {
        adamT++;
        double b1 = 0.9, b2 = 0.999, eps = 1e-8;
        double bc1 = 1 - Math.pow(b1, adamT), bc2 = 1 - Math.pow(b2, adamT);
        for (int i = 0; i < PARAMS.size(); i++) {
            T p = PARAMS.get(i);
            for (int j = 0; j < p.data.length; j++) {
                double g = p.grad[j];
                mM[i][j] = b1*mM[i][j] + (1-b1)*g;       // smoothed gradient
                vM[i][j] = b2*vM[i][j] + (1-b2)*g*g;     // smoothed squared gradient
                double mh = mM[i][j]/bc1, vh = vM[i][j]/bc2;
                p.data[j] -= LR * mh / (Math.sqrt(vh) + eps);
            }
        }
    }

    // =====================================================================================
    //  DATA  (one fresh random example per call — the model can never memorise a fixed set)
    // =====================================================================================
    static int[] curX, curY; static boolean[] curActive;
    /**
     * Build one training example. Sequence = [N random digits] + [those digits sorted].
     * The model predicts position t+1 from positions 0..t; we only grade predictions whose
     * answer lies in the sorted half (curActive[t] == true).
     */
    static void makeExample() {
        int[] in = new int[N];
        for (int i = 0; i < N; i++) in[i] = RNG.nextInt(DIGITS);
        int[] sorted = in.clone(); Arrays.sort(sorted);
        int[] seq = new int[2*N];
        for (int i = 0; i < N; i++) { seq[i] = in[i]; seq[N+i] = sorted[i]; }
        int L = 2*N - 1;
        curX = new int[L]; curY = new int[L]; curActive = new boolean[L];
        for (int t = 0; t < L; t++) { curX[t] = seq[t]; curY[t] = seq[t+1]; curActive[t] = (t >= N-1); }
    }

    // =====================================================================================
    //  TRAINING LOOP
    // =====================================================================================
    /** Train the model: for many steps, average the loss over a batch of fresh examples, backprop, and take one Adam step. */
    static void train() {
        initAdam();
        System.out.println("training " + countParams() + " parameters to sort " + N + " digits (0-" + (DIGITS-1) + ")...");
        for (int step = 1; step <= ITERS; step++) {
            zeroGrad();
            double lossSum = 0;
            for (int b = 0; b < BATCH; b++) {
                makeExample();
                T logits = forward(curX);
                T loss = crossEntropy(logits, curY, curActive);
                loss.grad[0] = 1.0 / BATCH;   // average the gradient over the batch
                backprop(loss);
                lossSum += loss.data[0];
            }
            adamStep();
            if (step % 100 == 0 || step == 1)
                System.out.printf("step %4d / %d   loss %.4f%n", step, ITERS, lossSum/BATCH);
        }
        System.out.println("training done.");
    }
    /** Count the total number of trainable weights (handy to print). */
    static int countParams() { int n = 0; for (T p : PARAMS) n += p.data.length; return n; }

    // =====================================================================================
    //  USING THE MODEL TO ACTUALLY SORT
    // =====================================================================================
    /** Sort by generation: feed the N inputs, then repeatedly take the model's most likely next digit, N times. Those N digits are its answer. */
    static int[] sortWithModel(int[] in) {
        int[] ctx = in.clone();
        for (int i = 0; i < N; i++) {
            T logits = forward(ctx);
            int last = ctx.length - 1;
            int best = 0; double bestv = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < DIGITS; j++) { double v = logits.data[last*DIGITS+j]; if (v > bestv) { bestv = v; best = j; } }
            int[] grown = Arrays.copyOf(ctx, ctx.length+1); grown[ctx.length] = best; ctx = grown;
        }
        return Arrays.copyOfRange(ctx, N, 2*N);
    }
    /** Measure exact-match accuracy on `trials` fresh random sequences. */
    static double accuracy(int trials) {
        int ok = 0;
        for (int t = 0; t < trials; t++) {
            int[] in = new int[N]; for (int i = 0; i < N; i++) in[i] = RNG.nextInt(DIGITS);
            int[] truth = in.clone(); Arrays.sort(truth);
            if (Arrays.equals(sortWithModel(in), truth)) ok++;
        }
        return (double) ok / trials;
    }

    // =====================================================================================
    //  SAVE / LOAD WEIGHTS  (so you can train once and reuse)
    // =====================================================================================
    /** Write the config and every weight to a file. */
    static void save(String path) throws IOException {
        try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            o.writeInt(N); o.writeInt(DIGITS); o.writeInt(N_EMBD); o.writeInt(N_HEAD); o.writeInt(N_LAYER);
            o.writeInt(PARAMS.size());
            for (T p : PARAMS) { o.writeInt(p.data.length); for (double d : p.data) o.writeDouble(d); }
        }
    }
    /** Rebuild the model from a saved file and load its weights. Returns false if the file is missing. */
    static boolean load(String path) throws IOException {
        File f = new File(path); if (!f.exists()) return false;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            N = in.readInt(); DIGITS = in.readInt(); N_EMBD = in.readInt(); N_HEAD = in.readInt(); N_LAYER = in.readInt();
            BLOCK = 2*N; PARAMS.clear(); buildModel();
            int np = in.readInt(); if (np != PARAMS.size()) throw new IOException("param count mismatch");
            for (T p : PARAMS) { int len = in.readInt(); for (int j = 0; j < len; j++) p.data[j] = in.readDouble(); }
        }
        return true;
    }

    // =====================================================================================
    //  MAIN
    // =====================================================================================
    public static void main(String[] args) throws IOException {
        boolean chatOnly = false, resume = false;
        for (String a : args) {
            if (a.equalsIgnoreCase("chat")) chatOnly = true;
            else if (a.equalsIgnoreCase("more")) resume = true;       // continue training from jamna.bin
            else if (a.matches("\\d+")) ITERS = Integer.parseInt(a);
        }

        if (chatOnly) {
            if (!load("jamna.bin")) { System.out.println("no jamna.bin found - run `java Jamna` to train first."); return; }
            System.out.println("loaded jamna.bin");
        } else {
            if (resume && load("jamna.bin")) System.out.println("resuming from jamna.bin");
            else { BLOCK = 2*N; buildModel(); }
            train();
            save("jamna.bin"); System.out.println("saved weights to jamna.bin");
            System.out.printf("final accuracy on 300 fresh random sequences: %.1f%%%n", 100.0*accuracy(300));
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        System.out.println();
        System.out.println("==================================================");
        System.out.println("  Hello, it's me -- Jamna.   (Java Neural Nets)");
        System.out.println("==================================================");
        System.out.println("I'm a neural net you built from scratch in Java. Nobody taught me the");
        System.out.println("rules of sorting -- I learned them by watching numbers. Give me six");
        System.out.println("digits (0-9) and watch my little brain work; or toss me any numbers");
        System.out.println("and I'll sort them anyway. Type 'quit' when you've had enough of me.");
        System.out.println();
        while (true) {
            System.out.print("you > ");
            String line = br.readLine();
            if (line == null) break;
            line = line.trim();
            if (line.equalsIgnoreCase("quit") || line.equalsIgnoreCase("exit")) {
                System.out.println("Jamna: Leaving already? I'll keep my weights warm for you. Come back soon.");
                break;
            }
            if (line.isEmpty()) { System.out.println("Jamna: ...I'm listening. Toss me some numbers, like  5 2 8 1 2 0"); continue; }
            String[] parts = line.split("\\s+");
            List<Integer> nums = new ArrayList<>();
            boolean parseOk = true;
            for (String s : parts) { if (s.isEmpty()) continue; try { nums.add(Integer.parseInt(s)); } catch (Exception e) { parseOk = false; } }
            if (!parseOk || nums.isEmpty()) { System.out.println("Jamna: Those need to be whole numbers separated by spaces -- e.g.  5 2 8 1 2 0"); continue; }
            int[] truth = new int[nums.size()];
            for (int i = 0; i < nums.size(); i++) truth[i] = nums.get(i);
            int[] sorted = truth.clone(); Arrays.sort(sorted);
            boolean inDomain = (nums.size() == N);
            if (inDomain) for (int v : nums) if (v < 0 || v >= DIGITS) inDomain = false;
            if (inDomain) {
                int[] out = sortWithModel(truth);
                boolean ok = Arrays.equals(out, sorted);
                System.out.println("Jamna (thinking with my neural net)...  " + Arrays.toString(out)
                    + (ok ? "   -- nailed it!" : "   -- hmm, my brain slipped; it should be " + Arrays.toString(sorted)));
            } else {
                System.out.println("Jamna: that's outside what my tiny brain trained on (six single digits 0-9),");
                System.out.println("       but I'll happily sort it the classic way:  " + Arrays.toString(sorted));
            }
        }
    }
}
