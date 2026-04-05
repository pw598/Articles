
---
layout: post
title:  "Positional Encodings in Transformer Language Models: A Conceptual and Visual Guide"
date:   2026-04-05 00:00:00 +0000
categories: Transformers PositionalEncodings
---

# OUTLINE

**1. Introduction**
- Why position matters: the permutation-invariance problem in self-attention
- The role of positional encoding: augmenting, not replacing, semantic content
- Scope of this article: sinusoidal encodings as the conceptual foundation, with modern alternatives in context

**2. Token Embeddings and the Addition Operation**
- What token embeddings represent: vectors in semantic space
- How positional encodings are added at the input layer
- The apparent paradox: doesn't mixing position into meaning corrupt the signal?
- Why the model can disentangle the two: capacity, gradient descent, and the residual stream

**3. Sinusoidal Positional Encodings**
- The encoding formula and its intuition
- The multi-frequency clock: each position as a unique fingerprint across oscillators
- Visualizing the raw sinusoids: frequency bands and their periods
- Why different frequencies serve different purposes (local vs. global structure)

**4. The Geometry of Positional Encodings**
- Positional encodings as a trajectory through high-dimensional space
- PCA projections: the helix/spiral structure in 2D and 3D
- Euclidean distance as a proxy for positional dissimilarity
- Explained variance: how much structure lives in the top components

**5. The Linear Transformation Property**
- The central identity: PE(k+m) = M(m) · PE(k)
- Derivation from the angle addition formulas for sin and cos
- The structure of M(m): block-diagonal rotations, one per frequency band
- Why linearity matters: attention operates through dot products and linear projections, so the model can learn to compute relative offsets from absolute encodings
- Composability: M(a) · M(b) = M(a+b) and what it implies
- Visualizing M(m) across offsets: near-identity at small m, increasingly rotated at large m
- The circular orbit view: PE(k) = M(1)^k · PE(0)

**6. Shift-Invariance and the Toeplitz Structure**
- Cosine similarity between positions depends on offset, not absolute location
- The pairwise similarity matrix and its near-Toeplitz character
- Verifying shift-invariance: similarity profiles from different anchor positions overlay almost exactly
- Residual from the Toeplitz ideal: how close is it in practice?
- Connection to the linear transformation property: the Toeplitz structure is a consequence of M(m) depending only on m

**7. Impact on Token Embeddings**
- Visualizing semantic clusters before and after PE addition
- Displacement magnitude vs. inter-cluster distance: the PE nudge is small
- Within-cluster spread and nearest-neighbour purity: semantic structure survives
- The key insight: the model learns to treat position as a separable signal, not noise

**8. How Attention Heads Exploit Positional Structure**
- Attention as a learned filter over positional and semantic signals
- Four archetypes found in trained models:
  - Local/diagonal heads: attending to immediate neighbours
  - Fixed-offset heads: exploiting M(m) to attend at consistent relative positions
  - Global/content heads: suppressing PE, attending by semantic similarity
  - Periodic/boundary heads: responding to syntactic structure at regular intervals
- Attention entropy as a measure of focus vs. diffusion
- The PE strength ablation: how positional signal shapes attention patterns

**9. Learned Positional Embeddings**
- The alternative: a trainable embedding matrix indexed by position
- What is gained: flexibility, data-driven adaptation
- What is lost: the shift-invariance guarantee, the linear transformation property, generalisation to unseen lengths
- Empirical comparison: when do learned embeddings outperform sinusoidal?

**10. Modern Alternatives and the Evolution of Positional Encoding**
- Relative position encodings: encoding offset directly rather than deriving it
- RoPE (Rotary Position Embedding): applying M(m) inside the attention dot product rather than at the input — the linear transformation property promoted to a first-class design principle
- ALiBi: position as a bias on attention logits rather than a modification to embeddings
- Comparison across methods: expressiveness, length generalisation, computational cost
- Why the field has converged toward relative and rotary schemes

**11. Open Questions and Practical Considerations**
- Length generalisation: why all methods struggle beyond training-time sequence lengths
- Position encoding in encoder-only vs. decoder-only vs. encoder-decoder architectures
- The interaction between positional encoding and other architectural choices (depth, attention span, MLP ratio)
- Directions in current research: learned dynamic encodings, position-free approaches

**12. Conclusion**
- The through-line: from the simple addition of sinusoids to the rotary embeddings in modern LLMs, the core problem has always been giving attention a consistent, learnable language for relative position
- Summary of key insights: the clock metaphor, the linear transformation, the Toeplitz structure, the semantic-positional disentanglement










# Positional Encodings in Transformer Language Models
*A conceptual and visual guide*

---

## 1. Introduction

### The permutation-invariance problem

A transformer processes a sequence of tokens — words, subwords, or characters — by passing them through a mechanism called self-attention. At its core, self-attention computes a weighted sum of all tokens in the sequence for each position, where the weights are determined by how relevant each token is to every other. This is a powerful and general operation, but it has a structural consequence that is easy to overlook: it treats the input as a *set*, not a sequence.

To see why this matters, consider the following two sentences:

> *The dog bit the man.*
> *The man bit the dog.*

Both sentences contain exactly the same tokens. A self-attention layer with no positional information would produce identical outputs for corresponding tokens across these two sentences, because the computation for any given token is determined entirely by the contents of the set of tokens — not by their order. The meaning of the sentences, however, is entirely different, and that difference lives precisely in the ordering of the words.

This is what is meant when attention is described as *permutation-invariant*. If you shuffled the input tokens in any order, the self-attention mechanism would produce the same output up to the same permutation. No information about the original ordering would survive. For any task where sequence order matters — which is essentially every interesting natural language task — this is a fundamental problem.

The feedforward layers that follow attention do not help here. They operate independently on each position's representation and cannot reconstruct ordering information that was never present. The problem cannot be patched downstream; it must be addressed at the point where tokens enter the model.

### What positional encoding does

The standard solution is to inject positional information directly into the token representations before they enter the attention layers. This is done by constructing a positional encoding — a vector of the same dimensionality as the token embedding — and adding it to each token's embedding before the first layer of the network. The result is that each token enters the model carrying both *what it is* (its semantic content) and *where it is* (its position in the sequence).

The addition operation is disarmingly simple. If **e**_t is the token embedding for the token at position *t*, and **p**_t is the positional encoding for position *t*, then the input to the transformer is simply:

**x**_t = **e**_t + **p**_t

This vector — the sum of two signals occupying the same space — is what every subsequent layer operates on. The model must learn to use both components simultaneously: attending to meaning when that is what matters, attending to position when that is what matters, and frequently attending to both at once.

### The apparent paradox

Adding two vectors together and asking a model to separate them might seem to create an impossible task. If the semantic vector for the word "bank" and the positional vector for position 12 are simply summed, how can the model ever recover which part of the resulting vector encodes meaning and which encodes location? And more practically: if the same word "bank" appears at position 12 in one sentence and position 47 in another, it will enter the model as two different vectors. How can the model learn a consistent representation of what "bank" means when the input it receives is different every time?

This concern is reasonable, but it dissolves under closer examination, for several reasons.

First, the model's weight matrices — the learned parameters that govern how representations are projected, mixed, and transformed — are not fixed at the input. They are applied to the summed vector, and through training, they learn to extract whichever component of that vector is useful for the task at hand. A weight matrix that projects primarily onto the subspace of semantic variation will effectively read the token identity; a weight matrix that projects onto the subspace of positional variation will effectively read the position. The model has more than enough capacity to learn both simultaneously, and gradient descent drives it to do exactly this whenever both signals are useful.

Second, the two signals are not equally mixed. The positional encoding is small relative to the spread of token embeddings in high-dimensional space. When visualised geometrically, adding positional encodings moves each token's embedding by a modest distance — enough to distinguish positions, but not enough to push it out of its semantic neighbourhood. Words that are similar in meaning remain nearby after positional encoding is added; the positional nudge is a perturbation, not a transformation.

Third, and most importantly, the geometry of sinusoidal positional encodings has a specific mathematical structure that makes relative positions recoverable through linear operations — the same kind of operations that attention and feedforward layers perform. This is not an accident. It was one of the design criteria for the original sinusoidal encoding scheme, and it is the reason that the model does not need to untangle an arbitrary mixture of signals: the positional signal has a clean internal structure that learned weight matrices can exploit predictably.

### Scope of this article

The remainder of this article develops these ideas in depth. We begin with the mechanics of token embeddings and the addition operation, then move through the mathematical structure of sinusoidal encodings — paying particular attention to the linear transformation property that makes relative position naturally learnable — and illustrate each concept with visualisations built from first principles. We then examine how attention heads in trained models exploit positional structure in qualitatively different ways, before broadening the view to consider learned positional embeddings, relative position encodings, and the rotary and bias-based approaches that have become standard in contemporary large language models.

Throughout, the emphasis is on building geometric and algebraic intuition rather than on formal proof. The goal is to arrive at a clear picture of why positional encoding works as well as it does, and why the field has evolved in the directions it has.










## 2. Token Embeddings and the Addition Operation

### What token embeddings represent

Before a transformer can process text, the discrete symbols that make up a sequence — words, subword pieces, or individual characters depending on the tokenisation scheme — must be converted into continuous vectors that the network can compute with. This conversion is handled by an embedding layer: a lookup table that maps each token in the vocabulary to a fixed-length real-valued vector, typically of several hundred to several thousand dimensions.

These vectors are not hand-crafted. They are parameters of the model, initialised randomly and adjusted throughout training by gradient descent alongside every other parameter. What emerges from this training process, however, is not arbitrary. Tokens that appear in similar contexts — that tend to be surrounded by the same other words across the training corpus — end up with similar embedding vectors. Tokens that appear in different contexts end up far apart. The result is a high-dimensional space in which geometric relationships encode semantic relationships: synonyms cluster together, antonyms occupy opposing directions, and structured analogies (king is to queen as man is to woman) manifest as parallel vector translations.

This learned structure is not a side effect; it is precisely what makes the embedding useful to the rest of the network. Downstream attention layers and feedforward layers can operate on these vectors algebraically, and the geometry they exploit — directions, distances, dot products — corresponds to genuine semantic content. An embedding layer that had learned nothing would give the network nothing to work with.

It is worth pausing on the dimensionality. A typical embedding dimension might be 512, 768, or 4096, depending on the model. These are genuinely high-dimensional spaces, and human intuition about geometry breaks down in them in important ways. Vectors that seem close in a projected 2D visualisation may be far apart along dimensions not shown; vectors that seem far apart may share structure in directions invisible in the projection. This caveat applies to all the visualisations in this article, and it is one reason that geometric intuitions should always be checked against the underlying algebra.

### The embedding as the starting point for a richer representation

It is tempting to think of the token embedding as *the* representation of a token — the thing the model knows about a word. This is accurate at the input layer, but it becomes increasingly inaccurate as the token's representation passes through successive transformer layers. At each layer, self-attention aggregates information from across the sequence, and the feedforward network transforms the result. By the time a representation reaches the final layer, it is no longer a pure token embedding: it is a contextualised representation that reflects not just what the token is, but what role it plays in this particular sequence, what tokens surround it, and what the model has inferred about the broader discourse.

The vocabulary embedding is therefore better thought of as an initialisation — a starting point that carries the token's identity into the network, from which richer representations are built layer by layer. This distinction matters for understanding positional encoding, because positional information does not need to survive all the way through the network in its original form. It only needs to be present at the right points for attention to exploit it, and the residual connections that run through modern transformers ensure that positional information injected at the input remains accessible at every subsequent layer without needing to be explicitly preserved by each intermediate computation.

### Adding the positional encoding

The positional encoding **p**_t is a vector of the same dimension as the token embedding **e**_t. The two are added elementwise to produce the input **x**_t = **e**_t + **p**_t, which is what the first transformer layer receives.

This operation is applied to every position independently and simultaneously. The positional encodings for all positions are computed in advance — for sinusoidal encodings, they are derived from a fixed mathematical formula requiring no training — and the appropriate encoding is added to each token's embedding before any attention is computed.

The simplicity of the addition is one of its virtues. It introduces no new parameters (for sinusoidal encodings), no new architectural complexity, and no change to the dimensions of any tensor flowing through the network. The positional signal is folded into the existing representation stream and the rest of the architecture proceeds unchanged.

### The disentanglement question

The natural concern, raised in the introduction, is whether adding two signals together makes them irrecoverable. If **x**_t = **e**_t + **p**_t, can the network ever separately access **e**_t and **p**_t? And does it need to?

The answer to the second question is: not exactly. The network does not need to perform explicit source separation — it does not need to subtract **p**_t from **x**_t to recover the pure token embedding. What it needs is to be able to compute useful functions of **e**_t and **p**_t, either separately or in combination, depending on what the task requires. And this is a much easier requirement, because the weight matrices of the attention and feedforward layers can learn to project **x**_t onto directions that are more aligned with **e**_t or more aligned with **p**_t depending on what is needed.

To see why this works, consider the query and key matrices **W**_Q and **W**_K in a self-attention layer. The attention score between positions *i* and *j* is proportional to:

(**x**_i **W**_Q) · (**x**_j **W**_K)ᵀ

Expanding **x**_i = **e**_i + **p**_i and **x**_j = **e**_j + **p**_j, this dot product decomposes into four terms:

(**e**_i **W**_Q) · (**e**_j **W**_K)ᵀ   +   (**e**_i **W**_Q) · (**p**_j **W**_K)ᵀ
(**p**_i **W**_Q) · (**e**_j **W**_K)ᵀ   +   (**p**_i **W**_Q) · (**p**_j **W**_K)ᵀ

Each term captures a different kind of interaction: content-to-content, content-to-position, position-to-content, and position-to-position. An attention head is not constrained to use all four equally. By learning appropriate weight matrices, a head can emphasise whichever terms are most useful. A head that learns **W**_Q and **W**_K to project primarily onto the semantic subspace will be driven mostly by the first term — content similarity — and will largely ignore position. A head that projects onto the positional subspace will be driven mostly by the fourth term — positional similarity — and will attend based on where tokens are rather than what they are. Most heads in practice learn something in between, with different mixtures depending on what the layer needs to compute.

This four-way decomposition is not a theoretical curiosity; it is the mechanism by which transformers simultaneously do syntactic and semantic processing. Heads in early layers of trained models often show strong positional biases — they attend locally, or to fixed offsets — while heads in later layers often show stronger content biases, attending to semantically relevant tokens regardless of distance. Both are possible within the same architecture because the weight matrices can tune the mixture freely.

### Why the model learns to separate the signals

The deeper question is not whether the model *can* disentangle position and content, but whether gradient descent will actually drive it to do so. The answer is yes, and the reason is straightforward: natural language tasks require both signals.

Consider a language model trained to predict the next token. To do this well, the model must know both what the recent tokens were (content) and where in the sentence it currently is (position) — because grammatical constraints, argument structure, and discourse patterns all depend on position. If the model's weight matrices learned to ignore positional information entirely, prediction accuracy would suffer at every point where position matters. If they learned to ignore content entirely, the model would be unable to condition on what has been said. Gradient descent, minimising prediction error over millions of sentences, finds weight configurations that use both signals in the ways that are most useful — which means learning to read content from some directions in representation space and position from others.

This is not a guarantee of perfect disentanglement in any formal sense. The representations learned by transformers are distributed and entangled in ways that resist clean decomposition. But it is a guarantee that the model will develop the capacity to use both signals effectively, which is all that is actually required.

### The scale of the positional perturbation

One further reassurance about the addition operation comes from looking at the relative magnitudes of **e**_t and **p**_t. For sinusoidal encodings, the encoding values are bounded between −1 and 1 across all dimensions. The typical norm of a sinusoidal encoding vector in *d* dimensions is approximately √(*d*/2) — for *d* = 512, this is about 16.

Token embeddings, after training, tend to occupy a much larger region of the embedding space, with norms that can be substantially larger depending on the model and initialisation scheme. The ratio of positional to semantic scale therefore varies, but it is common for the positional encoding to constitute a relatively modest perturbation — enough to distinguish positions clearly, but not so large as to overwhelm the semantic signal.

A geometric way to picture this: if you imagine semantic clusters as dense clouds of points in embedding space — all instances of "bank" near each other, all instances of "river" in a different region — then adding positional encoding moves each point within its cloud by a small, position-dependent amount. The clouds smear slightly, but they do not merge. A nearest-neighbour classifier operating on the perturbed embeddings could still recover the token identity with high accuracy, because the between-cluster distances remain much larger than the within-cluster displacements introduced by positional encoding. This is not just a plausible story; we verify it directly in section 7 using controlled synthetic embeddings.

The upshot is that the addition operation is better thought of as *annotation* than *contamination*. Each token enters the model carrying a rich semantic representation, with a small positional annotation appended in the same vector space. The rest of the network's job is to read both annotations and use them appropriately — a task it is well-equipped to perform.










## 3. Sinusoidal Positional Encodings

### The encoding formula

The original transformer paper, "Attention Is All You Need" (Vaswani et al., 2017), proposed a specific deterministic scheme for constructing positional encodings using sinusoidal functions. For a sequence position *k* and an encoding dimension *i*, the positional encoding is defined as:

PE(k, 2i)   = sin(k / 10000^(2i/d))
PE(k, 2i+1) = cos(k / 10000^(2i/d))

where *d* is the total embedding dimension and *i* ranges from 0 to *d*/2 − 1. Each pair of consecutive dimensions (2i, 2i+1) is assigned a frequency determined by the term 1 / 10000^(2i/d), and that frequency is used to compute a sine value for the even dimension and a cosine value for the odd dimension. The result is a *d*-dimensional vector for each position *k*, constructed entirely from trigonometric functions at a geometric progression of frequencies.

A few features of this formula are worth noting immediately. The encodings are *deterministic* — given a position and a dimension index, the value is fixed and requires no training. They are *bounded* — all values lie in [−1, 1]. And they are *continuous* — nearby positions produce nearby encoding vectors, because small changes in *k* produce small changes in the sine and cosine values at any given frequency. None of these properties are accidental; each was chosen to give the resulting vectors useful geometric structure.

The base value 10000 sets the range of frequencies. At dimension 0 (the highest frequency), the period of the sinusoid is 2π, meaning the encoding completes a full cycle roughly every six positions. At the highest dimension index *d*/2 − 1 (the lowest frequency), the period is 2π × 10000, meaning the encoding barely moves across sequences of typical length. The full set of dimensions therefore spans many orders of magnitude in frequency, from rapidly oscillating to nearly static across the sequence.

### The multi-frequency clock

The most intuitive way to understand sinusoidal positional encodings is through the metaphor of a clock with many hands, each running at a different speed.

An ordinary clock encodes time using two hands: an hour hand that completes one revolution in 12 hours and a minute hand that completes one revolution in 60 minutes. The pair of angles (hour hand, minute hand) jointly identifies the time of day with a resolution of one minute. The two-hand system works because the hands run at different speeds — the minute hand disambiguates within each hour, and the hour hand disambiguates across hours.

Sinusoidal positional encoding is the continuous, high-dimensional generalisation of this idea. Each frequency band corresponds to one clock hand: the high-frequency bands (small *i*) are fast hands that complete many revolutions across a typical sequence, while the low-frequency bands (large *i*) are slow hands that barely move. The position *k* is encoded as the simultaneous angle of all the hands at time *k*.

Because the hands run at a geometric progression of speeds spanning four orders of magnitude, the system can uniquely identify any position within an extremely long sequence. Ambiguity would only arise if two different positions produced identical angles simultaneously in all frequency bands — but the geometric spacing of frequencies makes this essentially impossible within any sequence length that a transformer would encounter in practice.

The sin/cos pairing within each frequency band is also deliberate. A single sinusoid sin(ωk) would be ambiguous — the same value occurs at multiple phases within each period. By pairing sin(ωk) with cos(ωk), each band represents the position as a *point on the unit circle* at angle ωk, which is unambiguous within one period. The pair (sin(ωk), cos(ωk)) is equivalent to the 2D unit vector at angle ωk, and as *k* advances, this vector rotates continuously around the circle. This is precisely the geometric structure that will prove central in section 5.

### Visualising the raw sinusoids

If we plot the value of PE(k, i) as a function of position *k* for several dimension indices *i*, we see a family of sinusoidal waves at different frequencies. The lowest-index dimensions (smallest *i*) oscillate rapidly, completing many cycles across a typical sequence. The highest-index dimensions (largest *i*) change slowly, sometimes barely visibly across the full sequence length.

This is the visualisation most commonly encountered in introductions to positional encodings: a grid of colour intensities, position on one axis, dimension on the other, with the sinusoidal pattern creating a characteristic striped texture that is finer at the left (high-frequency dimensions) and coarser at the right (low-frequency dimensions). The visualisation is accurate, but it conveys the encoding as a pattern of numbers rather than as a geometric object, which limits the intuition it can build.

More revealing is the per-frequency view: plotting each (sin(ωk), cos(ωk)) pair as a trajectory on the unit circle as *k* advances. Every frequency band traces a circle at its own speed, completing more or fewer revolutions depending on its frequency. The encoding for position *k* is then the tuple of circle positions — one per frequency band — at step *k*. No two positions produce the same tuple, and nearby positions produce tuples that differ by a small rotation at each frequency — the fast bands have moved noticeably, while the slow bands have barely moved.

This circular orbit view is one of the visualisations we produced in section 8 of the accompanying figures, where the trajectory of each frequency band under repeated application of the single-step rotation matrix is shown to be a perfect circle. The different radii of the circles reflect the different norms of the encoding vectors in each frequency band, not a property of the rotation itself (which is always by a unit rotation).

### Why different frequencies serve different purposes

The geometric progression of frequencies is not merely a way to pack many distinct vectors into the same dimensionality. Each frequency range carries structurally different positional information that is useful at different scales of language processing.

The high-frequency dimensions encode fine-grained, local position. Within a window of a few tokens, the high-frequency sinusoids vary substantially, giving the model a precise signal about exactly where a token is relative to its immediate neighbours. This is the information relevant to local syntactic patterns: whether a token is immediately before or after another, whether it is the first or last element in a small group.

The low-frequency dimensions encode coarse, global position. They change slowly across the sequence, providing a signal about whereabouts in the document a token appears — early, middle, or late — without resolving the exact position. This is the information relevant to discourse-level structure: whether the model is in an introductory passage or a conclusion, near the beginning of a long argument or near its resolution.

The middle-frequency dimensions occupy the intermediate range, encoding positional information at the scale of phrases, clauses, and sentences. A transformer with attention heads that project onto different frequency ranges can therefore simultaneously process local syntax, intermediate phrase structure, and global discourse position, using the same positional encoding vectors at every layer.

This multi-scale structure is what distinguishes sinusoidal encodings from a simpler scheme — say, encoding position as a single integer or as a linear ramp. A linear ramp would provide only one frequency, useful at one scale. Sinusoidal encodings provide the full spectrum simultaneously, and because the representation is fixed and known in advance, the model can learn projection matrices that selectively amplify whichever frequency range is relevant to any particular computation.

### The encoding as a point on a high-dimensional torus

There is a more formal geometric picture that unifies the multi-frequency clock and the unit circle interpretations. Each pair of dimensions (2i, 2i+1) defines a plane in the *d*-dimensional embedding space, and within that plane, the encoding for any position *k* lies on the unit circle at angle ω_i × k. The full positional encoding vector lies on the product of *d*/2 such circles — a geometric object known as a torus, or more precisely a (*d*/2)-dimensional torus embedded in *d*-dimensional space.

As the position *k* advances from 0 to its maximum, the encoding traces a path on this torus, moving along each circle at the rate of its corresponding frequency. The path never exactly revisits the same point (for the frequency ratios used in the formula, at least within practical sequence lengths), which is equivalent to saying that every position gets a unique encoding.

This torus picture makes the dimensionality of the encoding feel natural rather than arbitrary. The *d*-dimensional embedding space is tiled by frequency bands in pairs, and each pair contributes one circular degree of freedom to the encoding. The total positional vocabulary is the set of points reachable on the torus, which is continuous and infinite — there is no upper bound on sequence length imposed by the encoding itself, though in practice the low-frequency bands provide diminishing positional resolution at very long sequences.

The torus also clarifies what "distance" means in positional encoding space. Two positions that are close together will be close on the torus — each circular coordinate will have moved only slightly — while two positions that are far apart will have diverged on the fast-moving circles even if they happen to be close on the slow-moving ones. The overall distance is a combination of these per-circle distances, weighted by frequency, giving nearby positions high similarity even when integrated across all dimensions.

### A note on what sinusoidal encodings do not capture

The sinusoidal scheme encodes absolute position: the encoding for position *k* is a function of *k* alone, with no reference to the length of the sequence or the content of the tokens. This has practical consequences. A token at position 50 receives the same encoding regardless of whether it appears in a 60-token sentence (near the end) or a 500-token document (near the beginning). The model can learn to interpret absolute position in context — it sees the full sequence during attention — but the encoding itself carries no relative or normalised positional information.

This limitation motivates the relative and rotary encoding schemes discussed in section 10. But it does not prevent sinusoidal encodings from working well in practice: for the sequence lengths typical of early transformer applications, absolute position is a sufficient and stable signal. And crucially, as we will see in the next section, the mathematical structure of sinusoidal encodings allows the model to derive relative positional information from absolute encodings through learned linear projections — partially recovering the benefits of a relative encoding without any additional architectural machinery.










## 4. The Geometry of Positional Encodings

### Encodings as vectors in high-dimensional space

The formula from section 3 produces, for each position *k*, a *d*-dimensional vector. It is natural to ask what these vectors look like as a collection — not dimension by dimension, but as geometric objects in the full embedding space. The answer turns out to be both visually striking and mathematically meaningful, and it provides the most direct geometric intuition for why the encoding scheme works.

The first thing to notice is that the collection of positional encodings for positions 0, 1, 2, ..., *N* is not a cloud of scattered points. It is a *curve* — a smooth, continuous trajectory through the embedding space, parameterised by position. Adjacent positions produce adjacent points on this curve; the curve never doubles back on itself or produces sharp discontinuities. This continuity is a direct consequence of the smoothness of the sine and cosine functions: small changes in *k* produce small changes in the encoding vector.

The second thing to notice is the shape of this curve. When projected from *d* dimensions down to two or three principal components using PCA, the positional encodings trace a path that resembles a helix or a spiral. This is not an artefact of the projection: it reflects genuine structure in the high-dimensional space. The helical shape arises because the different frequency bands rotate at different rates, and the projection onto the first few principal components captures the dominant combination of these rotations. The result is a curve that winds through space with a characteristic rhythmic geometry, advancing steadily in some directions while oscillating in others.

This helix is the multi-frequency clock made geometric. Each revolution of the helix corresponds to one cycle of a dominant frequency component; the pitch of the helix reflects how quickly the lower-frequency components advance. A model that has internalised the structure of this curve — through learned weight matrices that respect its geometry — can read position from an encoding vector by identifying where on the curve the vector lies.

### Principal component analysis of positional encodings

To make this concrete, it is useful to examine what PCA reveals about the structure of a typical positional encoding matrix. If we construct the *N* × *d* matrix whose rows are the positional encoding vectors for positions 0 through *N* − 1, and decompose it using singular value decomposition, several things become apparent.

First, the explained variance is heavily concentrated in a small number of components. For a 64-dimensional encoding over 100 positions, the first two principal components typically capture a substantial fraction of the total variance — often more than half — and the first three capture more still. This concentration tells us that the positional encodings, despite living in a high-dimensional space, are approximately low-dimensional: most of their structure can be captured in a handful of directions.

Second, the leading principal components correspond to the lowest-frequency dimensions of the encoding. The first PC is dominated by contributions from the slow-varying dimensions (large *i* in the formula), because these dimensions show the most consistent directional trend across positions. The fast-varying dimensions contribute primarily to higher PCs, where their oscillatory behaviour is captured as alternating structure. This frequency-to-PC correspondence is not guaranteed by the PCA algorithm — it emerges from the data — and it confirms that the global trend in the data is the slow drift of the low-frequency bands, while the local oscillatory texture comes from the high-frequency bands.

Third, the two- and three-dimensional projections are faithful to the structure of the full-dimensional object. The 2D projection traces a smooth closed or near-closed curve; the 3D projection reveals the helical winding that is partially hidden in two dimensions. The labels along the curve confirm that the ordering is preserved: position 0 appears at one end, position *N* − 1 at the other, and the intermediate positions lie between them in sequence. There is no scrambling or folding.

### Euclidean distance as a proxy for positional dissimilarity

A natural measure of how different two positional encodings are is the Euclidean distance between their vectors: ‖PE(i) − PE(j)‖. If this distance behaves sensibly — increasing monotonically as positions grow further apart, with no anomalous close pairs — it provides a useful one-number summary of positional distinctiveness.

For sinusoidal encodings, the distance from position 0 to position *k* grows as a smooth, generally increasing function of *k*, at least for moderate values. At short distances, the fast-varying high-frequency dimensions dominate: nearby positions differ primarily in their high-frequency components, which change substantially over a few steps. At long distances, the slow-varying low-frequency dimensions begin to contribute meaningfully: the encodings have diverged even on the long-period bands.

This distance profile has an important practical implication. Two tokens that are close together in the sequence will have positional encodings that are only slightly different from each other — the distance is small. Two tokens that are far apart will have encodings that differ along many dimensions simultaneously — the distance is large. An attention mechanism that has learned to use this distance signal (through appropriate choice of query and key projections) will naturally attend more to nearby tokens than to distant ones, purely on the basis of positional proximity, without any explicit distance computation.

It is worth noting that the distance function is not perfectly monotonic over all ranges. At very long distances, the interplay of multiple frequencies can produce slight non-monotonicities, and at very short distances the function is nearly linear. But over the range of sequence lengths for which sinusoidal encodings are typically used, the monotonic trend is strong enough to provide a reliable positional signal.

### Pairwise similarity and the near-Toeplitz structure

Rather than measuring distance from a fixed reference point, we can ask a more symmetric question: how similar is every pair of positional encodings to every other pair? The answer is best visualised as an *N* × *N* matrix of pairwise similarities, where the entry at row *i*, column *j* is the cosine similarity between PE(*i*) and PE(*j*).

The resulting matrix has a distinctive structure. Along the main diagonal, where *i* = *j*, the similarity is 1 — every encoding is identical to itself. Moving away from the diagonal, similarity decays smoothly, with the decay rate depending on how far the positions are apart. The matrix is symmetric, since cosine similarity is symmetric, and it has a warm band near the diagonal flanked by cooler regions at the corners.

The most important structural feature of this matrix, however, is not its overall appearance but its *shift-invariance*. The similarity between positions *i* and *j* depends almost entirely on the offset |*i* − *j*| and not on the absolute values of *i* and *j* individually. This means that the matrix is approximately *Toeplitz*: each diagonal is nearly constant, with the value determined by the offset corresponding to that diagonal.

A perfectly Toeplitz similarity matrix would mean that the positional relationship between positions 10 and 15 is geometrically identical to the relationship between positions 40 and 45. The attention mechanism, which computes interactions between all pairs of positions, would therefore compute the same positional contribution for any pair of tokens separated by an offset of 5, regardless of where in the sequence those tokens appear. The model can learn to attend based on relative offset rather than absolute position, because the positional signal is structured to make relative offsets geometrically consistent.

This property — that the geometry of positional encodings is approximately shift-invariant — is not a coincidence. It is a consequence of the deeper algebraic property that we will examine in section 5: the fact that the encoding at position *k* + *m* can be obtained from the encoding at position *k* by applying a fixed linear transformation that depends only on the offset *m*. The Toeplitz structure of the similarity matrix is the observable consequence of this underlying algebraic regularity.

### Verifying shift-invariance

The shift-invariance of the similarity structure can be checked directly by plotting cosine similarity as a function of relative offset |*i* − *j*| for multiple anchor positions *i*. If the property holds, the curves for different anchor positions should overlay closely — the similarity from position 10 to its neighbours should look the same as the similarity from position 50 to its neighbours, once the curves are centred at their respective anchors.

In practice, the overlay is nearly exact for sinusoidal encodings. The standard deviation of cosine similarity values along each diagonal of the pairwise matrix is extremely small — close to the numerical precision of floating-point arithmetic — confirming that the Toeplitz approximation is not merely approximate but essentially exact. The residual from the Toeplitz ideal (the difference between the actual similarity matrix and the ideal matrix in which each diagonal is exactly constant) is uniformly close to zero across all positions and offsets.

This exactness is, again, not coincidental: it follows analytically from the sinusoidal formula, and it is the property that gives sinusoidal encodings their robustness compared to learned positional embeddings, which do not have any guaranteed geometric regularity.

### What the geometry tells us about learnability

The geometric picture developed in this section has a direct implication for how easy it is for a transformer to learn to use positional information. A model that receives positional encodings as part of its input is not being asked to extract structure from a random or arbitrary pattern — it is being asked to read a smooth, regular, shift-invariant geometric structure that is consistent across the entire sequence and across all training examples.

This regularity means that a weight matrix learned to compute some positional function on early positions in the training data will generalise, with no additional adjustment, to later positions — because the geometry is the same. It means that a model trained on short sequences will have encountered the same positional geometry as longer sequences, just over a narrower range — reducing (though not eliminating) the difficulty of length generalisation. And it means that the model's positional computations are stable across training: there is no noise or variability in the positional signal itself, only in the token content that is added to it.

The smoothness and regularity of the encoding geometry are, in a sense, the visual and geometric statement of a property that is more cleanly expressed algebraically. That algebraic statement — the linear transformation property — is the subject of the next section.










## 5. The Linear Transformation Property

### The central identity

The most important mathematical property of sinusoidal positional encodings is one that is easy to state but whose implications take time to fully appreciate. It is this: for any position *k* and any offset *m*, there exists a matrix **M**(*m*) — depending only on *m*, not on *k* — such that:

**PE**(*k* + *m*) = **M**(*m*) · **PE**(*k*)

In words: the encoding at position *k* + *m* can be obtained from the encoding at position *k* by applying a fixed linear transformation. The transformation depends only on how far you want to move — the offset *m* — and is completely independent of where you start.

This is a strong statement. It says that the relationship between any two positional encodings separated by an offset *m* is always the same linear relationship, regardless of absolute position. Moving five steps forward from position 3 involves exactly the same transformation as moving five steps forward from position 73. The geometry of relative displacements is uniform across the entire sequence.

Why does this matter? Because the operations that transformer models perform — projecting vectors with weight matrices, computing dot products, passing vectors through linear layers — are all linear operations. A property that holds under linear transformation is therefore a property that the model can exploit directly, through its learned parameters, without any special architectural support. The linear transformation property means that relative position is not something the model needs to discover indirectly or approximate from absolute encodings; it is directly available as a linear function of those encodings, and learned weight matrices can extract it precisely.

### Derivation from the angle addition formulas

The existence of **M**(*m*) follows directly from the angle addition identities for sine and cosine, which state:

sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
cos(α + β) = cos(α)cos(β) − sin(α)sin(β)

Consider a single frequency band *l*, contributing dimensions 2*l* and 2*l*+1 to the encoding. The encoding values at position *k* for this band are:

PE(*k*, 2*l*)   = sin(ω_l · *k*)
PE(*k*, 2*l*+1) = cos(ω_l · *k*)

where ω_l = 1 / 10000^(2*l*/*d*) is the angular frequency for band *l*. At position *k* + *m*, the values are:

PE(*k*+*m*, 2*l*)   = sin(ω_l · (*k*+*m*)) = sin(ω_l·*k* + ω_l·*m*)
PE(*k*+*m*, 2*l*+1) = cos(ω_l · (*k*+*m*)) = cos(ω_l·*k* + ω_l·*m*)

Applying the angle addition formulas:

sin(ω_l·*k* + ω_l·*m*) = sin(ω_l·*k*)·cos(ω_l·*m*) + cos(ω_l·*k*)·sin(ω_l·*m*)
cos(ω_l·*k* + ω_l·*m*) = cos(ω_l·*k*)·cos(ω_l·*m*) − sin(ω_l·*k*)·sin(ω_l·*m*)

The terms cos(ω_l·*m*) and sin(ω_l·*m*) do not depend on *k* — they are constants determined entirely by the offset *m* and the frequency ω_l. Writing these constants as *c_l* = cos(ω_l·*m*) and *s_l* = sin(ω_l·*m*), the equations become:

PE(*k*+*m*, 2*l*)   =  *c_l* · PE(*k*, 2*l*)  +  *s_l* · PE(*k*, 2*l*+1)
PE(*k*+*m*, 2*l*+1) = −*s_l* · PE(*k*, 2*l*)  +  *c_l* · PE(*k*, 2*l*+1)

This is a 2×2 matrix equation. Written in matrix form for the two dimensions of band *l*:

⎡ PE(*k*+*m*, 2*l*)   ⎤   ⎡  *c_l*   *s_l* ⎤   ⎡ PE(*k*, 2*l*)   ⎤
⎢                     ⎥ = ⎢               ⎥ · ⎢               ⎥
⎣ PE(*k*+*m*, 2*l*+1) ⎦   ⎣ −*s_l*   *c_l* ⎦   ⎣ PE(*k*, 2*l*+1) ⎦

The 2×2 matrix on the right is a *rotation matrix* — specifically, the standard rotation matrix for angle θ_l = ω_l · *m* in two dimensions. It rotates the two-dimensional vector (sin(ω_l·*k*), cos(ω_l·*k*)) — the unit circle position for band *l* at position *k* — by the angle ω_l·*m*, advancing it to the unit circle position for band *l* at position *k* + *m*.

This derivation holds for every frequency band *l* independently. Each band's two dimensions transform by their own rotation matrix, and the rotation angle for band *l* at offset *m* is ω_l · *m*.

### The block-diagonal structure of M(m)

Because the dimensions of different frequency bands do not interact with each other — the encoding for band *l* at position *k*+*m* depends only on the encoding for band *l* at position *k*, not on any other band — the full *d*-dimensional transformation decomposes into *d*/2 independent 2×2 rotations. The matrix **M**(*m*) is therefore block-diagonal, with one 2×2 rotation block per frequency band:

       ⎡  R(θ_0)    0      0    ···  0    ⎤
       ⎢    0     R(θ_1)   0    ···  0    ⎥
M(m) = ⎢    0       0    R(θ_2) ···  0    ⎥
       ⎢    ⋮       ⋮      ⋮    ⋱   ⋮    ⎥
       ⎣    0       0      0    ··· R(θ_{d/2−1}) ⎦

where R(θ_l) is the 2×2 rotation matrix for angle θ_l = ω_l · *m*, and the zeros indicate that off-diagonal blocks are all zero.

This structure has several immediate consequences. First, **M**(*m*) is an orthogonal matrix — its columns are orthonormal, and its transpose equals its inverse. Applied to a vector, it preserves length: ‖**M**(*m*) · **PE**(*k*)‖ = ‖**PE**(*k*)‖ for all *k* and *m*. This means that the linear transformation does not distort the scale of positional encodings as you move along the sequence; the vectors maintain constant norm.

Second, the block-diagonal structure means that **M**(*m*) is sparse — it has only *d* non-zero entries (2 per 2×2 block) rather than the *d*² entries a general matrix would have. This sparsity is computationally convenient but also conceptually important: it says that the transformation decomposes completely into independent per-band rotations, with no cross-band mixing. The frequency bands are not just a useful decomposition for understanding the encoding; they are the natural coordinate system in which the transformation is diagonal.

Third, the rotation angles θ_l = ω_l · *m* increase with frequency: high-frequency bands (small *l*, large ω_l) rotate by a large angle for any given offset *m*, while low-frequency bands (large *l*, small ω_l) rotate by a small angle. A small offset *m* rotates the high-frequency bands substantially while barely moving the low-frequency bands; a large offset *m* rotates all bands substantially. The pattern of rotation angles across bands is therefore a fingerprint of the offset, and this fingerprint is unique: no two distinct offsets within practical sequence lengths produce identical rotation angle patterns across all bands.

### Visualising M(m) across offsets

When **M**(*m*) is plotted as a colour matrix — each entry represented by a colour intensity, red for positive, blue for negative — the block-diagonal structure is immediately visible as a sequence of 2×2 coloured squares along the main diagonal, with everything else near zero. At *m* = 1, the rotation blocks are close to the identity matrix: the angles θ_l = ω_l are small for most frequency bands, so *c_l* ≈ 1 and *s_l* ≈ 0. As *m* increases, the high-frequency blocks rotate further from the identity — their off-diagonal elements grow — while the low-frequency blocks remain close to the identity for longer. At very large *m*, even the low-frequency blocks show substantial rotation.

The progression of **M**(*m*) across offsets *m* = 1, 4, 8, 16 therefore tells a story about how position changes propagate across frequency bands. At short offsets, the change is concentrated in the high-frequency dimensions; at long offsets, it has spread across all dimensions. This multi-scale response is precisely what makes the encoding informative at all positional scales simultaneously, as discussed in section 3.

### Composability: M(a + b) = M(a) · M(b)

The block-diagonal rotation structure immediately implies a further property: the transformation matrices compose correctly under addition of offsets. That is:

**M**(*a* + *b*) = **M**(*a*) · **M**(*b*)

for any offsets *a* and *b*.

The proof is straightforward. For each frequency band *l*, the rotation block for offset *a* + *b* is R(ω_l(*a*+*b*)) = R(ω_l·*a* + ω_l·*b*). By the standard composition rule for rotation matrices — R(α + β) = R(α) · R(β) — this equals R(ω_l·*a*) · R(ω_l·*b*). Since the full matrix is block-diagonal, composing block by block gives **M**(*a*) · **M**(*b*). ∎

This composability property means that the transformation matrices form a *group* under multiplication. Moving *a* steps then *b* more steps is the same as moving *a* + *b* steps in one go. The positional encodings, as a set with the transformation structure, are closed and consistent: there are no gaps or inconsistencies in the positional arithmetic.

Composability has a practical implication for attention across layers. When a transformer has multiple layers, and position information propagates through the residual stream from one layer to the next, the positional relationships that can be computed at layer *n* compose with those computed at layer *n*+1. A deep network can, in principle, chain relative position computations across layers, with each layer adding another level of positional reasoning on top of the last. The group structure of the transformation matrices ensures that this chaining is algebraically consistent.

### Why linearity matters for attention

To appreciate why the linear transformation property is specifically useful for self-attention, it helps to trace the path of positional information through an attention computation.

In a self-attention layer, the query vector for position *i* is **q**_i = **x**_i **W**_Q, and the key vector for position *j* is **k**_j = **x**_j **W**_K. The attention score is the dot product **q**_i · **k**_j. As established in section 2, this dot product has a component involving only the positional encodings:

**PE**(*i*) **W**_Q · (**PE**(*j*) **W**_K)ᵀ

Now suppose the weight matrix **W**_Q is chosen to project **PE**(*i*) onto a vector proportional to **PE**(*i*) itself (i.e., **W**_Q acts as an identity-like projection in the positional subspace), and **W**_K is chosen to project **PE**(*j*) similarly. The dot product then becomes approximately **PE**(*i*) · **PE**(*j*), which from section 4 we know is approximately a function of |*i* − *j*| alone — a relative position signal.

But the linear transformation property allows something stronger. Suppose instead that **W**_K is chosen to apply the transformation **M**(*m*) before projecting. Then the dot product **PE**(*i*) **W**_Q · (**M**(*m*) **PE**(*j*) **W**_K)ᵀ would be maximised when **PE**(*j*) is the encoding that, after applying **M**(*m*), aligns most closely with **PE**(*i*) — that is, when **PE**(*j*) = **M**(−*m*) **PE**(*i*), meaning *j* = *i* − *m*. A head with such weight matrices would attend precisely to the token *m* positions back, regardless of the absolute position of either token.

This is exactly the fixed-offset head archetype we constructed in section 8. The point is that this head is not contrived: it is a natural and learnable solution for any task that requires attending to a specific relative offset, and the linear transformation property of the sinusoidal encoding is what makes it possible. If the encoding did not have this property — if there were no matrix **M**(*m*) such that **PE**(*k*+*m*) = **M**(*m*) **PE**(*k*) — then no choice of query and key weight matrices could produce a head that attends at a consistent relative offset across all positions.

### The orbit of PE(0) under repeated application of M(1)

There is a particularly clean way to see the linear transformation property in action: by starting from the encoding at position 0 and applying the single-step transformation **M**(1) repeatedly. Since **PE**(*k*+1) = **M**(1) · **PE**(*k*), it follows by induction that:

**PE**(*k*) = **M**(1)^k · **PE**(0)

The entire sequence of positional encodings is the orbit of the initial vector **PE**(0) under powers of **M**(1). This is a complete and exact description: to generate all positional encodings, you need only the initial vector and the single rotation matrix.

The orbit of any vector under repeated application of a rotation matrix traces a circle in the rotation plane. Since **M**(1) is block-diagonal with one rotation per frequency band, the orbit in the full *d*-dimensional space is a product of circles — one per frequency band — which is, as noted in section 3, a torus. Each frequency band's circle has a different radius (determined by the norm of the initial encoding in that band's 2D subspace) and a different angular step size (determined by ω_l). The encoding at position *k* is the point reached on this torus after *k* steps.

Visualising this directly — plotting the trajectory of each 2D frequency block under repeated application of **M**(1) — shows perfect circles for each band, with the fast-rotating high-frequency bands completing many revolutions while the slow-rotating low-frequency bands have barely moved. This is one of the most visually convincing demonstrations of the structure of sinusoidal encodings, because it reduces the entire encoding scheme to a single geometric operation: rotation.

### Relation to learned embeddings and the inductive bias argument

The linear transformation property is a free gift that sinusoidal encodings provide to the model. It is not the result of learning; it is built into the encoding by construction. A model using sinusoidal encodings therefore has access to a consistent, position-independent representation of relative displacement from the very first step of training.

Learned positional embeddings, by contrast, have no such guarantee. Each position gets its own learned vector, initialised randomly and updated by gradient descent. There is no mathematical relationship between the learned vector for position 5 and the learned vector for position 10 — any such relationship must be discovered from data. In principle, with enough data and a sufficiently expressive model, learned embeddings could converge on a representation that mimics the sinusoidal structure; in practice, they learn something useful but typically less regular.

This is the sense in which sinusoidal encodings encode an *inductive bias* about the structure of sequential position: the bias that relative position is shift-invariant and derivable by linear transformation from absolute position. This is a mild but useful assumption for most sequence modelling tasks, and embedding it in the architecture rather than requiring it to be learned reduces the effective complexity of what the model must discover from data.

The inductive bias argument also points toward why the field eventually moved beyond both schemes. As we will discuss in section 10, the ideal solution — if you could have it for free — would be to make relative position directly computable within the attention mechanism, without requiring the model to extract it from a mixed signal. This is precisely what RoPE achieves, and understanding why sinusoidal encodings already point in this direction — through the **M**(*m*) structure — is the essential conceptual bridge to understanding RoPE as a natural evolution rather than a departure.










## 6. Shift-Invariance and the Toeplitz Structure

### From algebra to observable geometry

Section 5 established that the encoding at position *k* + *m* is obtained from the encoding at position *k* by applying a fixed rotation matrix **M**(*m*). This is an algebraic statement about individual encoding vectors. The present section examines what this property implies for the *relationships* between all pairs of encoding vectors — the pairwise similarity structure of the entire encoding matrix — and shows that the algebraic regularity of individual vectors produces a striking and practically important regularity in the collective geometry.

The central observation is this: if the transition from any position *k* to position *k* + *m* is always the same linear transformation, then the geometric relationship between any two positions separated by offset *m* must be the same, regardless of where in the sequence those positions lie. The relationship between positions 3 and 8 (offset 5) must be geometrically identical to the relationship between positions 43 and 48 (also offset 5). The geometry is shift-invariant: it depends on relative displacement, not on absolute location.

This shift-invariance is not an approximate or statistical regularity. It is an exact consequence of the algebraic property, and it manifests with near-machine-precision exactness in computed similarity matrices. Understanding why it holds, and what it implies for the model, is the purpose of this section.

### Cosine similarity under the rotation

To see how shift-invariance follows from the linear transformation property, consider the cosine similarity between **PE**(*k*) and **PE**(*k* + *m*) for arbitrary *k*:

cos(**PE**(*k*), **PE**(*k*+*m*)) = **PE**(*k*) · **PE**(*k*+*m*) / (‖**PE**(*k*)‖ · ‖**PE**(*k*+*m*)‖)

Substituting **PE**(*k*+*m*) = **M**(*m*) · **PE**(*k*):

= **PE**(*k*) · (**M**(*m*) · **PE**(*k*)) / (‖**PE**(*k*)‖ · ‖**M**(*m*) · **PE**(*k*)‖)

Since **M**(*m*) is orthogonal, it preserves vector norms: ‖**M**(*m*) · **PE**(*k*)‖ = ‖**PE**(*k*)‖. The denominator simplifies to ‖**PE**(*k*)‖². The expression becomes:

= **PE**(*k*)ᵀ **M**(*m*) **PE**(*k*) / ‖**PE**(*k*)‖²

This is a quadratic form in **PE**(*k*) with the matrix **M**(*m*). In general, a quadratic form **v**ᵀ **A** **v** depends on the direction of **v** in the eigenvector basis of **A**. However, for sinusoidal encodings, the block-diagonal structure of **M**(*m*) combined with the specific geometry of the encoding vectors makes this quadratic form nearly constant across all positions *k*.

To see why, expand the quadratic form in the block-diagonal basis. Each 2×2 block contributes a term of the form [sin(ω_l *k*), cos(ω_l *k*)] · R(θ_l) · [sin(ω_l *k*), cos(ω_l *k*)]ᵀ, where R(θ_l) is the rotation by angle θ_l = ω_l · *m*. Computing this:

[sin(ω_l *k*), cos(ω_l *k*)] · R(θ_l) · [sin(ω_l *k*), cos(ω_l *k*)]ᵀ
= cos(θ_l)(sin²(ω_l *k*) + cos²(ω_l *k*)) + sin(θ_l)(sin(ω_l *k*)cos(ω_l *k*) − cos(ω_l *k*)sin(ω_l *k*))
= cos(θ_l) · 1 + sin(θ_l) · 0
= cos(θ_l)
= cos(ω_l · *m*)

Each frequency band contributes exactly cos(ω_l · *m*) to the dot product, independent of *k*. Summing across all *d*/2 bands and normalising, the cosine similarity between **PE**(*k*) and **PE**(*k*+*m*) is:

cos(**PE**(*k*), **PE**(*k*+*m*)) = (2/*d*) · Σ_l cos(ω_l · *m*)

This expression depends only on *m* and the set of frequencies {ω_l} — not on *k* at all. The shift-invariance is therefore not approximate but exact: for sinusoidal encodings, the cosine similarity between any two positions separated by offset *m* is precisely the same for all starting positions *k*. This is the algebraic proof of what the visualisation in section 4 showed empirically: the standard deviation along each diagonal of the pairwise similarity matrix is zero to machine precision.

### The Toeplitz matrix defined

A matrix **S** is called *Toeplitz* if each of its diagonals is constant — that is, if **S**[*i*, *j*] depends only on *i* − *j*. A symmetric Toeplitz matrix additionally satisfies **S**[*i*, *j*] = **S**[*j*, *i*], which means it depends only on |*i* − *j*|.

The pairwise cosine similarity matrix of sinusoidal positional encodings is exactly a symmetric Toeplitz matrix, by the result derived above. The entry at row *i*, column *j* equals (2/*d*) · Σ_l cos(ω_l · |*i* − *j*|), which is a function of |*i* − *j*| alone. Every diagonal of the similarity matrix shares a single value, determined by the offset corresponding to that diagonal.

Toeplitz matrices arise naturally in signal processing — specifically in the analysis of stationary processes, where the correlation between two time points depends only on their lag and not on when in the signal they occur. The sinusoidal positional encoding similarity matrix being Toeplitz is the spatial analogue of stationarity: the "correlation" between two positions depends only on their relative displacement, and the encoding is stationary in the positional sense. This is a powerful and desirable property for a sequence model, because it means the model encounters the same positional relationships everywhere in the sequence, without privileged regions where the geometry is different.

### What the diagonal values look like

The values along each diagonal of the Toeplitz similarity matrix — that is, the cosine similarity as a function of offset *m* — have a specific character worth examining.

At offset *m* = 0, the similarity is 1: every vector is identical to itself. At small offsets, the similarity is high but less than 1: nearby positions are similar but distinct. As the offset increases, the similarity generally decreases, because the fast-rotating high-frequency components have moved substantially while the slow-rotating low-frequency components have moved little. The overall similarity is a weighted average of cosines at different frequencies, and as *m* grows, more and more of these cosines have passed through their first maximum and begun to oscillate.

The decay is not monotonic — the function (2/*d*) · Σ_l cos(ω_l · *m*) oscillates as *m* increases, because it is a sum of cosines at different frequencies — but the envelope of these oscillations decays toward zero as *m* grows. This means that at very long offsets, the cosine similarity between positions is close to zero on average, with oscillations around that baseline. Positions that are far apart are neither systematically similar nor systematically dissimilar; they are approximately orthogonal, with residual structure from the low-frequency bands.

This profile — high similarity nearby, decaying oscillations at a distance, near-zero at very long range — is well-matched to the structure of natural language. Tokens that are close together in a sentence tend to be strongly related; tokens that are far apart may or may not be related, and a prior of near-orthogonality at long range is a reasonable default from which the model can depart when content signals indicate a long-range dependency.

### Toeplitz structure and the attention mechanism

The Toeplitz structure of the positional similarity matrix has a direct consequence for how attention weights are distributed when a head is using primarily positional information.

Recall from section 2 that the attention score between positions *i* and *j* includes a positional component **PE**(*i*)**W**_Q · (**PE**(*j*)**W**_K)ᵀ. If the weight matrices **W**_Q and **W**_K project primarily onto the positional subspace, this component dominates and the attention weights are approximately:

softmax(**PE**(*i*)·**PE**(*j*) · **W**_Q **W**_K^T for all *j*)

The inner term **PE**(*i*)·**PE**(*j*) is, by the Toeplitz property, a function only of |*i* − *j*|. The attention weights from position *i* over all keys are therefore a softmax over a function of relative distance, producing a pattern that is the same shape regardless of where in the sequence position *i* sits. This is precisely the profile seen in the local attention head archetype from section 8: a smooth decay from the query position in both directions, identical in shape at every position.

This positional attention profile can be combined with content signals in any proportion. Heads that are dominated by the positional component produce smooth, position-only attention patterns; heads that are dominated by the content component produce patterns driven by semantic similarity regardless of distance; heads that mix the two produce patterns that attend to content-relevant tokens but bias toward those that are also positionally nearby. All three archetypes are observed in trained transformers, and the Toeplitz structure of the positional component is what makes the positional contribution stable and learnable rather than idiosyncratic.

### The residual from a perfect Toeplitz matrix

In practice, the pairwise cosine similarity of sinusoidal positional encodings is exactly Toeplitz, as shown by the derivation above. But it is instructive to consider what would happen if it were only approximately Toeplitz — as would be the case for a random or learned positional embedding scheme.

If the similarity matrix had significant off-diagonal variation within each diagonal — that is, if the similarity between positions 3 and 8 were meaningfully different from the similarity between positions 43 and 48, even though both pairs share the same offset — then the model's ability to generalise positional patterns across the sequence would be compromised. A head that learned to detect offset-5 relationships at positions 3–8 might not generalise to offset-5 relationships at positions 43–48, because the geometric signal it has learned to detect is not the same.

This is precisely the difficulty faced by learned positional embeddings at sequence lengths beyond those seen during training. The learned vectors for positions 0 through *N* (the training length) have whatever geometric structure gradient descent has imposed on them, but positions beyond *N* have no learned vectors at all. Even within the training range, there is no guarantee that the similarity structure is Toeplitz, so the model may develop head patterns that are position-specific rather than offset-general, which again limits generalisation.

Sinusoidal encodings avoid this failure mode exactly, by construction. The residual from the ideal Toeplitz matrix is zero, to within floating-point precision, for all positions and offsets. The model never encounters positional relationships that are geometrically inconsistent with what it has learned from other parts of the sequence.

### Shift-invariance as a symmetry

It is worth stepping back to appreciate what shift-invariance means at a higher level. In physics, symmetries of a system impose constraints on its behaviour: a system that is symmetric under translation in space must have the same laws at every location. The Toeplitz property of sinusoidal positional encodings is a discrete analogue of translational symmetry in the positional dimension: the geometric laws governing the relationship between two positions depend only on their separation, not on their absolute location.

For a model trained to process language, this symmetry is deeply appropriate. Natural language does not have privileged absolute positions in the way that, say, a musical score has a measure structure. A noun phrase at position 10 in a sentence plays the same grammatical and semantic role as a noun phrase at position 40, and the model should encode this equivalence rather than treating the two as fundamentally different by virtue of their location. Shift-invariance in the positional encoding is the architectural expression of this linguistic symmetry.

This reasoning also clarifies the sense in which learned positional embeddings break the symmetry. By assigning independent learned vectors to each absolute position, learned embeddings in principle allow the model to treat position 10 as fundamentally different from position 40. Whether the model actually does so depends on the data and the training process, but the architectural choice makes it possible — and at positions beyond the training range, there are no learned vectors at all, so the symmetry fails completely. Sinusoidal encodings maintain the symmetry unconditionally, at all positions, making them more robust in settings where position-independent generalisation is required.

### Connection forward to relative encodings

The Toeplitz structure of sinusoidal positional similarity is the observable expression, in the similarity domain, of what relative position encodings try to achieve by design. Relative encoding schemes, discussed in section 10, modify the attention mechanism to directly incorporate a bias or transformation that depends only on the relative offset between query and key positions. The result, when correctly implemented, is an attention weight that is exactly shift-invariant by construction, because the relative position term is computed fresh for each pair rather than derived from pre-added absolute encodings.

The key insight connecting the two approaches is that sinusoidal encodings already encode relative position implicitly — through the linear transformation property and its Toeplitz consequence — while relative and rotary encoding schemes make this implicit structure explicit and primary. The mathematical content is largely the same; what differs is the level of the architecture at which the relative position signal is introduced and how directly the model can access it.

Understanding the Toeplitz structure of sinusoidal encodings therefore provides the clearest possible preparation for understanding why relative and rotary encodings are natural improvements rather than radical departures. They are the logical endpoint of a design philosophy whose foundations are already visible in the shift-invariance of the simplest sinusoidal scheme.










## 7. Impact on Token Embeddings

### From theory to representation space

Sections 5 and 6 established that sinusoidal positional encodings have a clean algebraic structure — the linear transformation property and its Toeplitz consequence — that makes relative position recoverable through learned linear operations. This is a result about the positional encodings themselves, in isolation. The present section asks a different question: what actually happens to token representations when positional encodings are added to them? How much does the addition change the geometry of the embedding space, and does the semantic structure that the model needs to read survive the perturbation?

These questions matter because the algebraic properties of positional encodings, however elegant, are of no practical use if the addition corrupts the semantic signal to the point where the model cannot recover it. The argument in section 2 that the perturbation is "small relative to the inter-cluster distances" was qualitative. Here we make it quantitative, by working through the geometry carefully and examining what the embedding space looks like before and after positional encoding is added.

### The scale argument made precise

The norm of a sinusoidal positional encoding vector in *d* dimensions is given exactly by:

‖**PE**(*k*)‖² = Σ_{i=0}^{d/2−1} [sin²(ω_i · *k*) + cos²(ω_i · *k*)] = Σ_{i=0}^{d/2−1} 1 = d/2

so ‖**PE**(*k*)‖ = √(*d*/2) for all *k*. This is a constant — the norm of the positional encoding is the same at every position, which is a direct consequence of the orthogonality of **M**(*m*). For *d* = 64, this gives ‖**PE**(*k*)‖ ≈ 5.66; for *d* = 512, it gives ‖**PE**(*k*)‖ ≈ 16.

Token embeddings, after training on large corpora, typically have norms that are larger than this and more variable — driven by the training dynamics, the embedding initialisation scheme, and the implicit regularisation imposed by the loss function. In models with embedding normalisation (such as those that apply layer normalisation before the first attention layer), the effective scale of token embeddings is controlled, but in many standard architectures the raw embedding norms can range substantially above √(*d*/2).

The key ratio is between the magnitude of the positional perturbation and the distance between semantically distinct token embeddings. If the inter-cluster distance — the typical Euclidean distance between embeddings of words from different semantic categories — is much larger than √(*d*/2), then the positional encoding adds a perturbation that is small relative to the semantic separability of the space. The clusters smear but do not merge.

In practice, this condition is reliably satisfied for well-trained models. The training process drives semantically distinct tokens to occupy well-separated regions of the embedding space, and the margin of separation is determined by how useful the distinction is for the downstream task — which for language modelling is very useful indeed. The positional encoding, with its fixed norm of √(*d*/2), adds a bounded perturbation that the training dynamics implicitly learn to accommodate.

### Visualising the perturbation geometrically

To make this concrete, consider a simplified vocabulary of 25 words organised into five semantic clusters — animals, verbs, colours, places, and weather terms — with each cluster represented by a centroid in 64-dimensional space and individual words scattered around their centroid with moderate noise. This is the synthetic embedding setup used in the visualisations accompanying this article.

When these embeddings are projected to two dimensions via PCA, the five clusters appear as clearly separated clouds. Each word appears as a single point, because all instances of "cat" share the same base embedding regardless of what position they appear at.

After adding the sinusoidal positional encoding for each of ten randomly sampled positions, each word's single point becomes a small cloud of ten points, one per position. The clouds are displaced from the original word locations by the positional encoding vectors, which are different for each of the ten sampled positions. The result, when projected to the same two-dimensional PCA space, is that each semantic cluster — previously a set of tight points — becomes a set of small smeared clouds.

The critical observation is that the smearing is small relative to the between-cluster separation. The inter-cluster distances in the projected space are large compared to the within-word displacement clouds, so a nearest-neighbour classifier can still correctly identify which cluster any given post-addition point belongs to. The semantic structure of the embedding space is perturbed but not destroyed.

### Quantifying the preservation of semantic structure

Two quantitative measures capture the degree to which semantic structure survives the addition of positional encodings.

The first is the ratio of within-cluster spread to between-cluster spread, before and after adding positional encodings. Before addition, the within-cluster spread is determined entirely by the intra-cluster noise in the base embeddings — the variability among individual word embeddings within the same semantic category. After addition, the within-cluster spread increases because different positional encodings displace the same word in different directions. The between-cluster spread, however, is determined by the distances between cluster centroids, which are not affected by the positional encoding (since the average positional encoding across many positions is close to zero, the cluster centroid in the perturbed space stays near the original centroid). The ratio therefore decreases modestly after positional encoding, but remains large.

The second measure is nearest-neighbour cluster purity: for each occurrence of a word at a given position, what fraction of its five nearest neighbours in the full-dimensional space (after adding positional encodings) belong to the same semantic cluster? If positional encoding were destructive — if it pushed words from different clusters into each other's vicinities — the purity would drop sharply. In practice, the purity remains high, because the positional displacement is not large enough to bridge the inter-cluster gaps. The mean nearest-neighbour purity after adding positional encodings is only slightly lower than before, confirming that the semantic neighbourhood structure of the embedding space is substantially preserved.

### Why the displacement is bounded but not negligible

It would be misleading to conclude from the above that positional encodings have no effect worth mentioning. They are small relative to inter-cluster distances, but they are not negligible relative to intra-cluster distances. Within a semantic cluster, the displacement introduced by positional encoding is often comparable to or larger than the spread of the cluster itself. The encoding for the word "cat" at position 3 may be noticeably different from the encoding for "cat" at position 47, even though both lie closer to each other than to any word in a different cluster.

This within-cluster displacement is, of course, the whole point. It is precisely what gives the model positional information: the same word at different positions enters the network as a different vector, and the model can in principle read off the position from the direction of the perturbation. The encoding is designed to be small enough not to destroy semantic structure but large enough to provide unambiguous positional information. The fixed norm √(*d*/2) of the positional encoding achieves this balance across all positions and all embedding dimensions simultaneously.

There is also an asymmetry between high-frequency and low-frequency dimensions that matters here. In the high-frequency dimensions (small *i*), the positional encoding oscillates rapidly: nearby positions have very different values in these dimensions. In the low-frequency dimensions (large *i*), the positional encoding changes slowly. The displacement of a word's embedding due to positional encoding therefore has components that vary rapidly (from the high-frequency dimensions) and components that vary slowly (from the low-frequency dimensions), corresponding to the fine-grained and coarse-grained positional information discussed in section 3. The displacement is not a random perturbation but a structured one, with the structure aligned to the multi-frequency clock.

### The model's perspective: annotation rather than noise

From the model's perspective — meaning from the perspective of the learned weight matrices that operate on the summed embeddings — the addition of positional encodings is best understood as an *annotation*. Each token arrives at the first layer carrying a semantic vector that identifies what it is, augmented by a positional vector that identifies where it is. The two annotations are mixed in the same vector space, but they occupy partially orthogonal directions within that space, and the model's weight matrices can learn to project preferentially onto one or the other.

The word "partially" is important here. Semantic and positional information are not perfectly orthogonal in the embedding space of a typical model. Token embeddings are not constrained to lie in any particular subspace, and positional encodings are not designed to be orthogonal to all possible token embeddings. In the general case, the two signals will have some overlap, and projecting onto the "positional" direction will inadvertently include some semantic information, and vice versa. The model learns to manage this overlap through training, developing weight matrices that extract the combination of signals most useful for each computation.

What the preceding analysis establishes is that this management task is tractable. The semantic signal is strong enough — the inter-cluster distances are large enough relative to the positional perturbation — that reading token identity from a mixed embedding is not intrinsically difficult. A model with even modest capacity can learn to do it accurately, and empirically, well-trained transformers represent word meanings consistently across positions, as evidenced by the coherent semantic structure of their contextualised embeddings.

### Length of exposure and positional distribution

One subtle practical consideration is the distribution of positions at which a given token appears during training. If a particular word tends to appear only in early positions in the training corpus — say, discourse markers like "meanwhile" that typically appear sentence-initially — then the model will have seen that word's embedding combined with only a restricted range of positional encodings. Its learned representation for that word may therefore be less cleanly disentangled from early-position encodings than for words that appear uniformly across positions.

This is a genuine limitation, but it is mitigated by two factors. First, most common words in natural language appear at a wide range of positions across a sufficiently large corpus, so the coverage problem is mainly acute for rare or positionally restricted vocabulary items. Second, the sinusoidal encoding's fixed mathematical structure means that the model can, in principle, generalise its learned positional processing to positions it has not seen for a given word, because the encoding for any new position is a well-defined linear transformation of encodings the model has already encountered. The linear transformation property benefits generalisation not just across the sequence but across the vocabulary.

### Connection to contextualised representations

The analysis in this section has focused on the first layer of the transformer — the point at which positional encodings are added to token embeddings and the mixed signal enters the network. As the representation passes through successive layers, however, the picture changes substantially.

At each layer, self-attention aggregates information from across the sequence, and the feedforward network transforms the aggregated representation. The residual connections that bypass each layer ensure that the original positional signal remains accessible at every depth, but the dominant content of a deep representation is no longer the sum of a token embedding and a positional encoding. It is a complex, nonlinearly transformed mixture of information drawn from across the sequence, shaped by the model's learned understanding of language structure.

By the final layers of a large transformer, the representation for a given token position encodes not just what word appears there and where it appears, but what role it plays in the sentence, what entities it refers to, what discourse context surrounds it, and what the model predicts will follow. The positional encoding's contribution to this final representation is indirect — it influenced the attention patterns at every layer, shaping what information was aggregated and how — but it is no longer directly recoverable as a separable component.

This is the right outcome. The positional encoding was never intended to persist as a separable signal through all layers; it was intended to seed the positional computation at the input and allow attention to build on it layer by layer. The fact that it dissolves into richer contextualised representations as the model goes deeper is a sign that the positional information has been successfully absorbed into the model's understanding of the sequence, not that it has been lost.

### What the visualisations reveal

The accompanying visualisations for this section show three complementary views of the embedding impact. The side-by-side scatter plots of semantic clusters before and after positional encoding addition — projected to two dimensions via PCA — give the most immediate geometric intuition: clusters that were tight points become small clouds, but the clouds remain well-separated. The displacement arrows for representative words at multiple positions show the direction and magnitude of the positional perturbation in the projected space, making visible both the bounded magnitude and the varying direction across positions. And the quantitative summaries — within-cluster spread, nearest-neighbour purity — provide the numerical grounding for the geometric observations.

Together, these views support a consistent conclusion: the addition of positional encodings to token embeddings is a well-designed perturbation. It is large enough to encode position unambiguously, structured enough to maintain the algebraic properties that make relative position learnable, and small enough to preserve the semantic geometry that the model needs to read word identity. The balance is not accidental — it is the result of a dimensionality and scaling choice that places the positional encoding in the right part of the representation space to do its job without undoing the work of the token embedding layer.










## 8. How Attention Heads Exploit Positional Structure

### The diversity of attention behaviour

One of the most striking findings from the interpretability literature on trained transformers is that individual attention heads within the same model behave in qualitatively different ways. Some heads attend narrowly to the immediately adjacent token; others attend to the token at a specific syntactic distance; others distribute attention broadly across the sequence with little positional bias; still others show structured periodic patterns that seem to track clause or phrase boundaries. This diversity is not noise or disorder — it reflects a genuine division of computational labour, with different heads specialising in different aspects of the positional and semantic structure of the input.

Understanding this diversity requires understanding how the positional encoding interacts with the attention mechanism at the level of individual heads. The algebraic machinery of sections 5 and 6 provides the tools: the linear transformation property and the Toeplitz structure of the positional similarity matrix determine what positional computations are naturally learnable by attention heads, and the four-term decomposition of the attention score from section 2 determines how content and position can be mixed. Together, these tools allow us to characterise the space of possible head behaviours and identify the archetypes that trained models most commonly develop.

### The four-term decomposition revisited

Recall from section 2 that the attention score between positions *i* (query) and *j* (key) decomposes into four terms when the input is written as **x** = **e** + **p**:

(**e**_i **W**_Q) · (**e**_j **W**_K)ᵀ    [content → content]
(**e**_i **W**_Q) · (**p**_j **W**_K)ᵀ    [content → position]
(**p**_i **W**_Q) · (**e**_j **W**_K)ᵀ    [position → content]
(**p**_i **W**_Q) · (**p**_j **W**_K)ᵀ    [position → position]

Each head's weight matrices **W**_Q and **W**_K determine which of these four terms dominates. A head that projects primarily onto directions of high variance in the token embedding space will be dominated by the content terms; a head that projects onto the positional subspace will be dominated by the position terms. Most heads in practice project onto some mixture, but the mixture is not arbitrary — it is shaped by the training signal and the task demands at each layer.

The position → position term is the most directly connected to the material of sections 5 and 6. When **W**_Q and **W**_K both project primarily onto the positional subspace, the attention score between *i* and *j* is approximately **PE**(*i*)**W**_Q · (**PE**(*j*)**W**_K)ᵀ. If additionally **W**_K encodes a rotation by **M**(*m*) in the positional subspace, this score is maximised when *j* = *i* − *m*, producing a fixed-offset attention pattern. The linearity of the transformation is what makes this learnable: the model needs only to learn a linear projection that encodes the desired rotation, rather than discovering a nonlinear relationship between absolute positions.

### Head archetype 1 — local and diagonal

The most commonly observed attention head type in trained transformers is the local or diagonal head: a head whose attention weights are concentrated near the diagonal of the attention matrix, with each token attending primarily to its immediate neighbours. These heads implement a kind of local averaging or context-gathering operation, pooling information from the few tokens surrounding each position before the representation is passed to the feedforward layer.

Local heads arise naturally from the positional encoding structure. The Toeplitz property established in section 6 means that the positional similarity between adjacent tokens — the cosine similarity between **PE**(*k*) and **PE**(*k*±1) — is consistently high, while the similarity to distant positions is lower. A head that learns to project primarily onto the high-frequency positional dimensions will amplify this local-similarity structure, because the high-frequency components change substantially from position to position and produce the sharpest distinctions between nearby and distant positions. The result, after softmax normalisation, is an attention distribution concentrated on the few positions with the highest positional similarity to the query.

The sharpness of this concentration depends on how strongly the weight matrices project onto the high-frequency positional subspace. A head with a mild positional bias produces a broad, locally-weighted distribution; a head with a strong positional bias produces something close to a delta function at the adjacent position. Both are useful at different points in the network: broad local aggregation is most useful in early layers, where the model is building up basic contextual representations, while sharp local attention can implement specific syntactic operations at later layers.

### Head archetype 2 — fixed offset

The fixed-offset head attends to tokens at a consistent relative distance from the query, producing an attention pattern where the brightest entries lie on a diagonal that is parallel to but displaced from the main diagonal. A head offset by *m* = 3, for example, would show each query token attending primarily to the token three positions earlier, regardless of where in the sequence the query sits.

As established in section 5, this behaviour is directly enabled by the linear transformation property. The weight matrix **W**_K can encode the rotation **M**(*m*) in the positional subspace, so that the key vector for position *j* points in the direction that **PE**(*i*) would point if it were the encoding for position *i* + *m*. The dot product **PE**(*i*)**W**_Q · **PE**(*j*)**W**_K is then maximised precisely when *j* = *i* − *m*, giving the desired offset attention.

Fixed-offset heads are particularly useful for syntactic processing tasks where the relevant token tends to appear at a predictable distance. Agreement heads — heads that look for subject-verb agreement, for example — often show approximately fixed-offset behaviour, because the subject and verb in many sentence constructions are separated by a relatively predictable number of tokens. Heads that implement n-gram-like processing, attending to the previous one or two tokens to build local syntactic structure, are also common and correspond to small fixed offsets.

It is worth emphasising that fixed-offset heads do not require the model to "know" the absolute position of the query token. The offset is computed from the rotational structure of the positional encoding, which is position-independent by design. A head learned to attend three positions back will do so correctly at position 5 and at position 95, without any special-casing or position-dependent logic in the weight matrices.

### Head archetype 3 — global and content-driven

At the opposite extreme from local and fixed-offset heads are heads that show little positional bias, distributing attention according to content relevance regardless of distance. These global heads typically attend to a small number of tokens anywhere in the sequence — often tokens that are semantically or syntactically related to the query token, such as the head of a dependency arc, the antecedent of a pronoun, or a verb related to its argument.

Global heads arise when the learned weight matrices project primarily onto directions of high variance in the token embedding space rather than the positional subspace. In the four-term decomposition, the content → content term dominates, and the positional terms are suppressed. The attention scores then reflect the similarity of token content at positions *i* and *j*, with minimal contribution from their positional relationship.

From an information-theoretic perspective, global heads tend to have high attention entropy — their distributions are broad and diffuse, covering many positions with moderate weight rather than concentrating on a few nearby positions. This diffuseness reflects the fact that semantically relevant tokens can appear anywhere in the sequence, so a content-driven head must maintain attention probability across the full range of positions. The entropy of a global head is typically much closer to the log(sequence length) upper bound than the low entropy of a local or fixed-offset head.

Global heads become more prominent in later layers of deep transformers, where the representations carry rich contextual information and the model is performing more abstract linguistic computation. Early layers tend to be dominated by local processing; later layers increasingly perform long-range operations on the contextualised representations that earlier layers have built.

### Head archetype 4 — periodic and boundary-sensitive

A fourth archetype, less universally observed than the first three but present in many trained models, is the periodic or boundary-sensitive head: a head whose attention pattern shows a structured, roughly regular spacing of bright entries, suggesting sensitivity to the periodic structure of certain linguistic units such as clauses, sentences, or paragraphs.

Periodic heads arise from projection onto the mid-frequency positional dimensions — dimensions whose period is comparable to the typical length of the linguistic unit being tracked. A head projecting onto dimensions with a period of approximately six tokens will show attention peaks spaced roughly six tokens apart, corresponding in many English sentences to the average clause length. This is not the head "knowing" about clauses explicitly; it is the head detecting the periodic positional signal that co-occurs with clause boundaries in the training data.

The existence of periodic heads provides an interesting example of the model exploiting the multi-frequency structure of the positional encoding in a non-obvious way. Rather than using only the high-frequency dimensions (for local structure) or the low-frequency dimensions (for global position), periodic heads tap into intermediate frequencies that carry information at the scale of syntactic constituents. The full spectrum of frequencies available in the sinusoidal encoding — spanning many orders of magnitude — allows heads at different layers to specialise in positional patterns at different granularities.

### Attention entropy as a diagnostic

Attention entropy — the Shannon entropy of the attention weight distribution across keys for a given query — is a useful single-number summary of a head's behaviour. A head concentrating all its attention on one position has entropy close to zero; a head distributing attention uniformly across all positions has entropy equal to log(*N*), the maximum possible value for a sequence of length *N*. In practice, most heads fall between these extremes, with local and fixed-offset heads toward the low end and global heads toward the high end.

Examining attention entropy across heads and layers reveals consistent patterns in trained transformers. Early layers have more low-entropy heads, reflecting the dominance of local processing at the beginning of the network. Later layers tend to have higher entropy, reflecting the shift toward global, content-driven attention. Within each layer, there is typically a mix of entropies, with some heads doing local work and others doing global work in parallel — a computational division of labour that allows a single layer to aggregate information at multiple scales simultaneously.

Entropy is also sensitive to the PE strength: as the weight matrices project more strongly onto the positional subspace, attention concentrates on nearby positions and entropy decreases. The ablation experiment in the accompanying visualisations — varying the strength of positional encoding added to token embeddings — shows entropy decreasing systematically as positional encoding strength increases for a locally-tuned head, confirming that the local attention behaviour is driven by the positional signal rather than by token content.

### The PE strength ablation

The ablation experiment that varies the proportion of positional encoding mixed into the token embedding directly reveals the contribution of positional structure to attention patterns. At one extreme (no positional encoding, α = 0), the attention pattern for a locally-tuned head is driven entirely by token content similarity — and because the token embeddings carry no positional information, the pattern shows no preference for nearby positions. Attention is distributed according to semantic similarity, which has no local bias. At the other extreme (full positional encoding, α = 1), the diagonal structure of local attention appears clearly, with each token attending primarily to its immediate neighbours as expected.

The intermediate values of α reveal the progressive emergence of positional structure. At α = 0.33, faint diagonal structure begins to appear, with the local positional bias starting to compete with the content similarity. At α = 0.67, the diagonal is clearly visible, and the content-driven off-diagonal attention has been substantially suppressed. The transition from content-dominated to position-dominated attention is smooth and monotonic for this head type, reflecting the linear mixing of the two signals in the input vector.

This ablation provides direct empirical support for the analytical framework developed in sections 2 and 5. The attention pattern is a weighted combination of positional and content signals, and the weights can be varied continuously. The model learns the mixture that is most useful for the task, but the architecture permits any mixture, and the two components are cleanly separable by varying α. Real trained models are operating somewhere in this space, with the weight matrices implicitly setting an effective α that depends on what the head needs to compute.

### Composing heads across layers

The head archetypes described above do not operate in isolation — they compose across layers, with each layer's output feeding into the next layer's input through the residual connection and layer normalisation. The full computational behaviour of a transformer emerges from this composition, and understanding how positional information flows through the stack requires thinking about how heads at different layers build on each other.

A common pattern in well-trained models is a hierarchical processing scheme: early layers use local heads to build up representations that incorporate immediate context; middle layers use fixed-offset and locally-broader heads to process syntactic structure at the phrase level; later layers use global heads to aggregate information across the full sequence for semantic and discourse-level processing. The positional encoding, introduced at the input, seeds all of this processing — but by the time the representation has passed through several layers of global attention, the positional component has been thoroughly mixed into contextualised representations that reflect the full structure of the sequence.

This hierarchical view also explains why positional encodings need to be present at the input rather than at some intermediate layer. If positional encodings were added only at layer 3, the first two layers would have no access to positional information and could not build the local contextual representations that later layers depend on. Injecting the positional signal at the input ensures that it is available from the first computation and can propagate through the residual stream to every subsequent layer that needs it.

### The limits of the archetype framework

The four head archetypes — local, fixed-offset, global, and periodic — are idealised descriptions of a continuous space of possible behaviours. In practice, trained attention heads rarely conform perfectly to any single archetype. Many heads show mixed behaviour: partly local and partly content-driven, or attending to a range of offsets rather than a single fixed distance. The archetypes are useful for building intuition and for understanding the design space of what positional encodings make learnable, but they should not be taken as a complete taxonomy of the attention behaviour observed in real models.

What the archetype analysis does establish, cleanly and rigorously, is that the linear transformation property of sinusoidal encodings makes all four archetypes naturally learnable with standard weight matrices and standard attention mechanisms. No special architectural support is needed for local attention, fixed-offset attention, content-driven attention, or periodic attention: each is a natural consequence of particular choices of projection directions, and each of those choices is accessible through gradient descent given the right task signals. The richness of attention behaviour observed in trained transformers is not in spite of the simple sinusoidal encoding but, in a real sense, because of the mathematical structure it provides.










## 9. Learned Positional Embeddings

### A different design philosophy

Sinusoidal positional encodings are a fixed, parameter-free scheme: the encoding for any position is determined by a mathematical formula, requires no training, and is the same regardless of what model it is used in or what data it is trained on. This determinism is a feature — the geometric properties established in sections 5 and 6 hold unconditionally — but it is also a constraint. The encoding is set in advance, and the model must learn to use it as given. It cannot reshape the positional representation to better suit the task or the data.

Learned positional embeddings take a different approach. Rather than a formula, they introduce a second embedding matrix — distinct from the token embedding matrix — indexed by position rather than vocabulary item. The embedding for position *k* is the *k*-th row of this matrix, a learned vector of the same dimension as the token embedding. This vector is initialised randomly and updated by gradient descent during training, just like every other parameter in the model. The resulting positional representation is whatever the data and the training objective drive it to be.

This design was adopted early and widely: BERT, GPT-2, and many of their successors use learned positional embeddings rather than sinusoidal ones. The choice was motivated partly by the empirical observation that the two approaches perform similarly on standard benchmarks — suggesting that the mathematical elegance of the sinusoidal scheme was not conferring a measurable advantage — and partly by the general principle that allowing the model to learn its own representations tends to produce better results than imposing a fixed structure, at least when there is enough data.

### What learned embeddings can and cannot do

The primary advantage of learned positional embeddings is flexibility. The model is free to learn whatever geometry is most useful for representing position in the context of its particular task. If absolute position is important — for example, in generation tasks where the model needs to know whether it is at the beginning, middle, or end of a response — the learned embeddings can specialise to encode absolute location clearly. If certain positions are more important than others — the first token, the last token, boundaries between sentences — the learned embeddings can assign those positions distinctive vectors with larger norms or more distinctive directions. None of this flexibility is available with sinusoidal encodings, which treat all positions with the same mathematical structure regardless of their linguistic significance.

The primary disadvantage is the absence of any guaranteed geometric structure. Because each position's embedding is learned independently, there is no mathematical relationship between the embedding for position 10 and the embedding for position 11 — any such relationship must emerge from the training data. Whether it does emerge depends on how consistently the training signal rewards positional generalisation. In practice, the learned embeddings for adjacent positions do tend to be similar — because tokens at adjacent positions tend to appear in similar contexts — but this similarity is softer and less regular than the exact shift-invariance of sinusoidal encodings. The near-Toeplitz structure of the pairwise similarity matrix, which holds exactly for sinusoidal encodings, is only approximate and variable for learned embeddings.

This distinction is consequential in at least two respects. First, for tasks where precise relative position matters — where the model needs to reliably detect that token *j* is exactly *m* positions before token *i* — learned embeddings provide a noisier signal than sinusoidal ones, because the geometric relationship between positions *k* and *k*+*m* is not guaranteed to be the same as between positions *k*' and *k*'+*m*. Second, and more importantly, learned embeddings do not generalise to positions beyond the training length. If a model is trained on sequences of up to length 512, it has learned embeddings for positions 0 through 511. There is no embedding for position 512 or beyond — the matrix simply does not have that row. At inference time, the model cannot process sequences longer than what it was trained on, full stop.

### The length generalisation failure

The inability to generalise to longer sequences is the most significant practical limitation of learned positional embeddings, and it has driven much of the research into alternative positional encoding schemes over the past several years.

With sinusoidal encodings, the encoding for position 513 is simply computable by the same formula as positions 0 through 512. Whether the model can *use* this encoding effectively is a separate question — and one we will return to — but the encoding itself is well-defined. With learned embeddings, there is no such fallback: position 513 has no representation, and the model must either truncate the input, extrapolate from nearby embeddings in some ad hoc way, or fail.

Various workarounds have been proposed. One approach is to simply cap sequences at the training length, discarding tokens beyond the limit. This is computationally wasteful and linguistically lossy for long documents. Another approach is to interpolate or extrapolate the learned embeddings — for example, fitting a smooth function to the learned vectors for positions 0 through 511 and evaluating that function at position 512. This can work to a modest degree but is unprincipled and tends to degrade performance at positions far beyond the training range. A third approach is to fine-tune the model on longer sequences with freshly initialised embeddings for the new positions. This restores full-length performance but requires additional training and data.

None of these solutions is as clean as having a positional encoding scheme that generalises automatically. The length generalisation failure of learned embeddings is not a minor engineering inconvenience; it is a fundamental consequence of the design choice to represent position as a lookup table rather than a mathematical function. As sequence lengths relevant to practical applications have grown — from the 512 tokens of BERT to the 128,000 or more tokens of contemporary large language models — this limitation has become increasingly pressing, and it has accelerated the move toward encoding schemes that can handle arbitrary sequence lengths by construction.

### Empirical comparison with sinusoidal encodings

Given the theoretical advantages of sinusoidal encodings — exact shift-invariance, guaranteed relative-position learnability, arbitrary-length extrapolation — it is natural to ask why learned embeddings were adopted so widely in the first place. The answer is largely empirical: on the benchmarks that were used to evaluate early transformer models, the two approaches performed comparably, and in some settings learned embeddings performed slightly better.

Several explanations have been offered for this rough parity. First, for tasks where absolute position is informative — document classification, question answering over fixed-format passages, generation with strong positional priors — the flexibility of learned embeddings is genuinely useful, and this usefulness can offset the lack of mathematical structure. Second, the training data for large language models is vast enough that the model can discover approximate shift-invariance from data even without it being built into the encoding; learned embeddings trained on billions of tokens develop a degree of positional regularity that, while not exact, is sufficient for most tasks. Third, many standard benchmarks use sequences well within the training length, so the generalisation failure of learned embeddings is never triggered.

The comparison becomes less favourable to learned embeddings as task requirements become more demanding. At long sequence lengths, at positions beyond the training range, and in settings where precise relative position is critical, sinusoidal encodings — and more importantly, their successors, the relative and rotary schemes discussed in section 10 — consistently outperform learned position tables.

### The parameter cost

One often-overlooked dimension of the comparison is parameter count. A learned positional embedding matrix for maximum sequence length *N* and embedding dimension *d* has *N* × *d* parameters. For *N* = 512 and *d* = 768 (the settings of BERT-base), this is approximately 400,000 parameters — a small fraction of the model's roughly 110 million total parameters, and therefore not a significant concern. For contemporary large language models with *d* = 8192 and context lengths of *N* = 128,000, however, a learned position table would require over a billion parameters, comparable to the parameter count of a substantial language model in its own right. At this scale, the parameter cost of learned embeddings is prohibitive, and fixed encoding schemes become necessary on practical grounds alone.

This scaling argument is part of why the field has converged on encoding schemes — particularly RoPE and its variants — that introduce no position-specific parameters at all, applying positional transformations through fixed computations rather than learned lookup tables. The zero parameter overhead of these schemes is not merely convenient; it is essential for the context lengths that state-of-the-art models now routinely handle.

### What learned embeddings reveal about the task

Despite their limitations, learned positional embeddings are useful as diagnostic tools. Because they are unconstrained by any mathematical structure, the geometry they develop reflects purely what the training data and objective reward. Examining the similarity structure of a trained positional embedding matrix — plotting the pairwise cosine similarities between learned vectors for all pairs of positions — can reveal what kinds of positional structure the task actually requires.

In models trained on standard language modelling tasks, the learned similarity matrix tends to be banded near the diagonal — reflecting that adjacent positions are treated similarly — with some longer-range structure corresponding to common sentence and paragraph lengths. Models trained on tasks with strong absolute position signals — such as code generation, where indentation level and line position are semantically meaningful — tend to develop more position-specific structure, with distinct embedding directions for early, middle, and late positions.

These task-specific patterns are not achievable with sinusoidal encodings, which impose the same mathematical structure regardless of the task. Whether the flexibility to develop task-specific positional structure is worth the cost in geometric regularity and length generalisation is a tradeoff that depends on the application. For general-purpose language models handling long and variable-length sequences, the answer has increasingly been no. For specialised models operating on short, fixed-format inputs where absolute position carries consistent semantic content, learned embeddings remain a reasonable choice.

### The inductive bias tradeoff

The deepest way to understand the difference between sinusoidal and learned positional encodings is as a tradeoff in inductive bias. Sinusoidal encodings encode a strong prior about the structure of sequential position: that positional relationships are shift-invariant, that relative position is derivable by linear transformation, and that different frequency scales carry information relevant to different scales of linguistic structure. These priors are well-matched to natural language, which is genuinely shift-invariant at the level of positional relationships — the syntactic role of a word does not change depending on the absolute position at which it appears.

Learned embeddings encode essentially no prior beyond the fact that position is a discrete index in the range [0, *N*−1]. All structure must be learned from data. This is a weaker prior, which means the model has more freedom but also more to learn, and the learned structure is only as good as the training data and the training length permit.

In machine learning more broadly, stronger and better-matched inductive biases tend to produce better data efficiency and better generalisation, particularly in the regime of limited data or at distribution shift. Sinusoidal encodings are more data-efficient in the sense that the shift-invariance property holds from the first step of training rather than needing to be discovered. They generalise better to new lengths because the mathematical structure is length-independent. The cost is that they may be suboptimal for tasks where the correct prior is different from shift-invariance — but for the vast majority of language tasks, shift-invariance is the right prior, and the sinusoidal encoding is well-suited.

The field's movement toward relative and rotary encodings can be understood as extending the inductive bias further: rather than providing a fixed encoding that makes relative position *learnable*, these schemes make relative position *directly computable* in the attention mechanism, with no learning required. The inductive bias becomes stronger still, and the generalisation properties improve correspondingly. It is to these schemes that we now turn.










## 10. Modern Alternatives and the Evolution of Positional Encoding

### The design space, reconsidered

Sections 3 through 9 have examined sinusoidal positional encodings and learned positional embeddings in depth. Both schemes share a structural feature that, in retrospect, can be seen as a limitation: they inject positional information at the input layer, mixing it with token embeddings before any attention is computed. The attention mechanism then has to extract positional signals from this mixture — a task made tractable, as we have seen, by the linear transformation property and the Toeplitz structure, but still indirect. The positional information enters the model in a form where it must be disentangled from semantic content rather than being directly available to the attention computation.

The alternative design philosophy asks: what if positional information were introduced not at the input, but directly inside the attention mechanism, at the point where positional relationships are actually used? This would eliminate the disentanglement problem entirely. Rather than adding a positional annotation to each token embedding and asking learned weight matrices to extract it, the attention scores could incorporate positional terms explicitly and directly, computed from the query and key positions rather than from their embeddings.

This is the design philosophy behind the modern positional encoding schemes — relative position encodings, Rotary Position Embedding (RoPE), and Attention with Linear Biases (ALiBi) — that have largely replaced both sinusoidal and learned embeddings in contemporary large language models. Each realises this philosophy differently, with different tradeoffs in expressiveness, computational cost, and length generalisation. Understanding each scheme, and how it relates to the sinusoidal foundation we have built, is the purpose of this section.

### Relative position encodings

The first systematic departure from input-layer positional encoding was the family of *relative position encoding* schemes, introduced in parallel by several research groups around 2018–2020. The core idea is to replace the absolute position indices *i* and *j* in the attention score with their relative offset *i* − *j*, modifying the attention computation so that what enters the score calculation is information about how far apart the query and key are, rather than where they are absolutely.

The most influential formulation, due to Shaw et al. (2018) and later refined by Raffel et al. in the T5 model, adds a learned scalar bias *b*(*i* − *j*) to the attention logit between positions *i* and *j*. This bias is drawn from a small learned table indexed by the relative offset, clipped at some maximum distance beyond which all offsets share the same bias value. The attention score becomes:

score(*i*, *j*) = (**x**_i **W**_Q) · (**x**_j **W**_K)ᵀ + *b*(*i* − *j*)

The bias term is purely relative: it depends only on *i* − *j* and not on *i* or *j* individually. The attention weight is therefore guaranteed to be shift-invariant by construction, not approximately or as a consequence of algebraic structure, but explicitly and exactly. The Toeplitz property that sinusoidal encodings achieve through the **M**(*m*) rotation is here achieved by design.

The learned bias table is small — typically a few dozen entries, with clipping applied beyond the maximum tracked offset — so the parameter cost is negligible compared to the rest of the model. And because the bias is added directly to the attention logit rather than through the embedding, it sidesteps the disentanglement problem: the query and key projections operate on token content, and the positional term is added separately and cleanly.

The limitation of this approach is that the bias is a scalar — it modifies the attention score by a single number per offset, rather than by a vector-valued transformation. This means the positional contribution is the same regardless of what kind of attention head is doing the computation: every head at every layer gets the same positional bias for the same offset. The richer positional structure that sinusoidal encodings enable — where different heads can project onto different frequency components and achieve different positional sensitivities — is partially sacrificed in favour of simplicity and explicit shift-invariance.

### Rotary Position Embedding (RoPE)

RoPE, introduced by Su et al. in 2021 and subsequently adopted in LLaMA, PaLM, Falcon, Mistral, and most other large open-weight language models, can be understood as taking the linear transformation property of sinusoidal encodings and promoting it from a consequence of the input encoding to a first-class design principle of the attention mechanism itself.

Recall from section 5 that the key property of sinusoidal encodings is PE(*k* + *m*) = **M**(*m*) · PE(*k*), where **M**(*m*) is a block-diagonal rotation matrix. This means that attending from position *i* to position *j* involves, in the positional component of the attention score, a dot product that implicitly involves the rotation **M**(*j* − *i*). The sinusoidal scheme makes this rotation available implicitly, through the input encodings; RoPE makes it explicit, by rotating the query and key vectors directly inside the attention computation.

Concretely, RoPE modifies the query and key vectors before the dot product is computed. For query position *i* and key position *j*, the rotated vectors are:

**q̃**_i = **R**(θ, *i*) · **q**_i
**k̃**_j = **R**(θ, *j*) · **k**_j

where **R**(θ, *k*) is the same block-diagonal rotation matrix structure as **M**(*k*) — a rotation by angle ω_l · *k* in each frequency band *l*. The attention score is then the dot product:

score(*i*, *j*) = **q̃**_i · **k̃**_j = (**R**(θ, *i*) **q**_i) · (**R**(θ, *j*) **k**_j)

Using the orthogonality of rotation matrices — specifically that **R**(θ, *i*)ᵀ **R**(θ, *j*) = **R**(θ, *j* − *i*) — this simplifies to:

score(*i*, *j*) = **q**_i · (**R**(θ, *j* − *i*) **k**_j)

The dot product between the query at position *i* and the key at position *j* is therefore equal to the dot product between the unrotated query and the key rotated by the *relative* offset *j* − *i*. The attention score is explicitly a function of the relative position, not the absolute positions, with no disentanglement required. The Toeplitz property holds exactly, by construction, for every head at every layer.

This is the payoff for all the mathematics of section 5. The block-diagonal rotation structure of **M**(*m*), which was a derived consequence of the sinusoidal formula, becomes the primary computational element in RoPE. The encoding is no longer added to the token embeddings before they enter the network; instead, the rotation is applied inside the attention head, to the projected query and key vectors, at the point of the dot product. Token content and positional rotation are applied to different quantities — the embedding and the projected query/key, respectively — so there is no mixing at the input and no disentanglement problem.

RoPE also preserves the full richness of the frequency decomposition. Because the rotation is applied per frequency band in the query and key projection dimensions, different attention heads can attend to different frequency components simply by having their projections concentrate on different dimensional ranges. The multi-scale positional structure of sinusoidal encodings — the spectrum of frequencies from local to global — is fully available to RoPE-equipped attention heads, and the head archetypes of section 8 remain achievable.

### Length generalisation with RoPE

One of the practical challenges with all fixed-frequency rotary schemes, including RoPE in its original form, is that length generalisation — the ability to process sequences longer than those seen during training — remains imperfect. While the rotation formula is defined for any position, the model's weight matrices are trained to expect query-key dot products within a certain range of relative offsets. At very long sequences, the relative offsets between query and key positions can substantially exceed the maximum offset seen during training, and the model's learned behaviour may degrade in this out-of-distribution regime.

Several extensions to RoPE have been developed to address this. *Position interpolation* (Chen et al., 2023) rescales the position indices so that the full training-length range maps onto a longer target range, effectively squeezing more positions into the same angular range that the model is accustomed to. *YaRN* (Peng et al., 2023) applies a more sophisticated frequency-dependent rescaling, preserving the high-frequency components that carry local positional information while compressing the low-frequency components that encode global position — reflecting the observation that local processing (which depends on high frequencies) is more critical to model performance than global position at very long ranges.

These extensions demonstrate that length generalisation is an active and unsolved research problem even for the best current positional encoding schemes. RoPE provides a strong foundation — far better than learned embeddings, and comparable to or better than sinusoidal encodings in most settings — but it does not fully eliminate the challenge of extrapolation to lengths much longer than the training distribution. The model must still be exposed to sufficiently long sequences during training, or fine-tuned to extend its effective context, to perform reliably at very long ranges.

### Attention with Linear Biases (ALiBi)

ALiBi, introduced by Press et al. (2021), takes the simplest possible approach to relative position encoding: rather than rotating query and key vectors or adding a learned bias, it adds a fixed negative penalty to the attention score that increases linearly with the distance between query and key:

score(*i*, *j*) = (**x**_i **W**_Q) · (**x**_j **W**_K)ᵀ − *m* · (*i* − *j*)

where *m* is a head-specific slope — a fixed scalar, not a learned parameter — that controls how steeply the attention score decays with distance. Different heads receive different slopes, with slopes decreasing geometrically across heads so that some heads maintain nearly flat attention across long distances while others decay steeply.

The linear penalty has an immediate and intuitive interpretation: it encodes a prior that nearby tokens are more relevant than distant ones, with the strength of that prior varying across heads. Heads with steep slopes implement strongly local attention; heads with shallow slopes can attend globally. The model does not need to learn this preference from data — it is built into the bias structure — but the content-driven component of the attention score can override the distance penalty when a distant token is genuinely highly relevant.

ALiBi's most distinctive property is its length generalisation behaviour. Because the penalty is a simple linear function of distance, it extends automatically to any sequence length without modification. A model trained on sequences of length 1024 will, when presented with sequences of length 4096, simply apply the same linear penalty to longer-range interactions. Empirically, ALiBi generalises to sequences roughly four to eight times longer than training length with modest performance degradation — substantially better than sinusoidal or learned embeddings, and competitive with RoPE extensions at a fraction of the implementation complexity.

The tradeoff is expressiveness. The linear bias encodes a strong and simple prior (attention decays with distance) that may not be optimal for all tasks. Tasks that require attending to specific relative offsets — the fixed-offset head behaviour of section 8 — are not naturally supported by a linear distance penalty; such a head would need to fight against the penalty to focus on its target offset. RoPE, by contrast, directly enables arbitrary relative-position attention patterns through the rotation mechanism, at the cost of somewhat more complex implementation and less clean length extrapolation.

### Comparing the approaches

The four schemes — sinusoidal, learned, RoPE, and ALiBi — occupy different positions in a multidimensional design space. Comparing them across the dimensions that matter most in practice:

*Parameter cost.* Sinusoidal and RoPE and ALiBi introduce no position-specific parameters. Learned embeddings require *N* × *d* parameters, which becomes prohibitive at large context lengths.

*Shift-invariance.* Sinusoidal encodings achieve exact shift-invariance through the **M**(*m*) rotation property. Learned embeddings achieve it approximately, if at all. Relative encodings, RoPE, and ALiBi achieve it exactly by construction.

*Length generalisation.* Sinusoidal encodings provide well-defined encodings at any length, but the model may not use them correctly beyond training length. Learned embeddings fail completely beyond training length. RoPE generalises reasonably with extensions but benefits from exposure to long sequences. ALiBi generalises most naturally of all, to significantly longer sequences than training length.

*Expressiveness.* Sinusoidal encodings support the full range of head archetypes through projection onto different frequency bands. RoPE preserves this expressiveness. ALiBi sacrifices some of it in favour of the simple distance-decay prior. Learned embeddings are maximally flexible within the training range.

*Implementation simplicity.* ALiBi is the simplest: a fixed scalar subtracted from the attention logit. Sinusoidal and learned embeddings require adding a vector at the input. RoPE requires rotating query and key vectors inside each attention head, which is slightly more complex but straightforward with modern autodifferentiation frameworks.

No single scheme dominates on all dimensions, which is why the field has not converged on one universal solution. Contemporary large models typically use RoPE or a variant thereof, reflecting the judgment that its combination of expressiveness, exact shift-invariance, zero parameter cost, and reasonable length generalisation makes it the best overall choice for general-purpose language modelling. But ALiBi remains competitive for applications where long-context generalisation is the primary concern, and relative bias schemes like T5's are still in widespread use in encoder-only models where context lengths are more controlled.

### RoPE as the completion of the sinusoidal idea

It is worth pausing to appreciate how directly RoPE emerges from the mathematical structure of sinusoidal encodings. The block-diagonal rotation matrix **M**(*m*) that appeared in section 5 as a derived consequence of the angle addition formulas is the central computational element of RoPE. The frequency decomposition into per-band oscillators that gave sinusoidal encodings their multi-scale character is exactly the frequency decomposition that RoPE uses to rotate query and key vectors. The Toeplitz property that section 6 showed to hold exactly for sinusoidal encodings holds exactly for RoPE by a simpler and more direct argument.

In this sense, RoPE is not a replacement for sinusoidal encodings but their completion. Sinusoidal encodings were a way of injecting the rotation structure at the input layer and relying on the model to extract it; RoPE applies the same rotation structure directly in the attention computation, where it does its work most naturally. The mathematical content is the same — block-diagonal rotations at a geometric progression of frequencies — but the point of application has moved from the embedding layer to the attention layer, eliminating the mixing problem and making the shift-invariance exact and unconditional.

A reader who has followed the development from section 3 through section 9 should find RoPE's design feel not like a new idea but like the natural answer to the question: if the linear transformation property is the right mathematical structure for positional encoding, where in the architecture should it be applied? The answer — inside the attention dot product, where positional relationships are actually computed — is clear in retrospect. The sinusoidal encoding scheme was a very good first approximation to this answer, developed before the question was articulated in its clearest form.

### The ongoing frontier

Despite the progress represented by RoPE and ALiBi, positional encoding remains an active area of research. The fundamental challenge of length generalisation — enabling models to process sequences arbitrarily longer than their training distribution — has not been fully solved. Current extensions of RoPE achieve useful but imperfect generalisation, and the theoretical understanding of why and when extrapolation succeeds or fails is still incomplete.

Beyond length generalisation, researchers are exploring positional encoding schemes for non-sequential modalities: two-dimensional position for images and video, three-dimensional position for point clouds and molecular structures, and graph-structured position for relational data. The sinusoidal and rotary schemes generalise naturally to higher dimensions — the block-diagonal rotation structure extends straightforwardly to 2D and 3D positional encodings — but the right inductive biases for these modalities are less well understood than for language.

There is also a deeper question about whether position should be encoded explicitly at all. Some recent architectures experiment with reducing or eliminating explicit positional encodings, relying instead on content-based attention patterns and architectural inductive biases to implicitly track position. These approaches remain experimental and have not yet matched the performance of RoPE-equipped models on language tasks, but they raise fundamental questions about the nature of positional information and whether it is best treated as a separate signal or as an emergent property of the model's computation.

These open questions ensure that positional encoding will remain a productive area of investigation for the foreseeable future. The mathematical foundations laid by sinusoidal encodings — the multi-frequency clock, the linear transformation property, the Toeplitz similarity structure — will continue to provide the conceptual framework within which new approaches are understood and evaluated, even as the architectural forms in which these ideas are expressed continue to evolve.










## 11. Open Questions and Practical Considerations

### The length generalisation problem, restated

Length generalisation — the ability of a model to process sequences longer than those encountered during training — has been a recurring theme across the preceding sections. It is worth gathering the threads and stating the problem precisely, because the partial solutions offered by different encoding schemes address different aspects of a genuinely multifaceted challenge.

The problem has two separable components. The first is *representational*: does the positional encoding scheme produce a well-defined representation for positions beyond the training range? Learned embeddings fail on this component entirely — there is no vector for position *N*+1 if the model was trained on sequences of length *N*. Sinusoidal encodings and RoPE pass this test: the formula is defined at any position, and the rotation can be applied to any query or key regardless of its absolute position. ALiBi passes trivially, since its linear penalty extends to any distance by definition.

The second component is *behavioural*: even given a well-defined positional representation at positions beyond the training range, will the model use it correctly? This is where all current schemes face challenges. A model's weight matrices are trained on attention score distributions corresponding to relative offsets up to the training length. At longer sequences, relative offsets can substantially exceed this range, and the model encounters query-key interactions that are out-of-distribution — not because the positional encoding is undefined, but because the model has never learned how to respond to such large offsets. The attention patterns produced may be arbitrary or degenerate, even if the positional encoding itself is mathematically correct.

This second component is the harder problem, and it is not solved by any current encoding scheme alone. Addressing it requires either training on long sequences — expensive but effective — or carefully designed fine-tuning procedures that extend the model's effective context without full retraining. The YaRN and position interpolation extensions to RoPE described in section 10 are engineering solutions to this problem, not theoretical resolutions of it.

### Encoder-only, decoder-only, and encoder-decoder architectures

Positional encoding interacts differently with different transformer architectures, and the right scheme for one setting is not necessarily the right scheme for another.

In *encoder-only* models such as BERT and RoBERTa, the attention is bidirectional — every token attends to every other token — and the sequence length is typically fixed and moderate (512 tokens in the original BERT). In this setting, the absolute position of a token within the fixed-length context carries consistent meaning, and the length generalisation problem is less acute because inputs are typically padded or truncated to the training length. Learned positional embeddings have historically been the dominant choice here, and relative encodings (such as T5-style biases) have also been widely used. RoPE is increasingly adopted in newer encoder-only models, but the advantages over learned embeddings are less dramatic than in the decoder setting.

In *decoder-only* models such as GPT and its successors, the attention is causal — each token attends only to earlier tokens — and the sequence length is variable and potentially very long. This setting makes length generalisation critical, because users routinely present inputs near or beyond the training context length. RoPE has become the near-universal choice for contemporary decoder-only models, reflecting its combination of zero parameter cost, exact shift-invariance, and reasonable length generalisation. The causal mask interacts cleanly with RoPE's rotation structure, since the relative offset between any query and any earlier key is always positive, and the rotation is applied symmetrically.

In *encoder-decoder* models such as T5 and its variants, the encoder processes the input bidirectionally and the decoder generates output autoregressively, with cross-attention between the two. The positional encoding requirements of the encoder and decoder can differ — the encoder benefits from bidirectional position processing, while the decoder needs causal position handling — and the cross-attention adds a third case where the positional relationship between encoder and decoder tokens is relevant. T5's relative bias scheme handles this naturally by using different bias tables for self-attention and cross-attention, but the interaction is more complex than in single-stack architectures.

### The interaction with other architectural choices

Positional encoding does not operate in isolation. Its effectiveness depends on and interacts with several other architectural choices, and these interactions are often underappreciated.

*Layer normalisation placement* affects how strongly the positional signal propagates through the network. In the pre-norm architecture (layer normalisation applied before the attention and feedforward sublayers, as in GPT-3 and most contemporary models), the residual stream retains the original input scale, including the positional encoding component, and the positional signal can be accessed at any depth through the residual connection. In the post-norm architecture (as in the original transformer paper), the normalisation at the output of each layer tends to suppress the positional component, potentially reducing its effectiveness at depth.

*Attention span and local attention* interact with positional encoding in an obvious way: if attention is restricted to a local window of *w* tokens, the relative offsets relevant to positional encoding are bounded by *w*, and the long-range structure of the encoding is irrelevant. Local attention models can use simpler positional encodings without loss, and the length generalisation problem is reduced to the scale of the window rather than the full sequence. Sliding window attention models such as Longformer and BigBird use this observation explicitly, combining local and global attention with appropriate positional treatments for each.

*Embedding dimension* and *number of frequency bands* determine the resolution of the positional encoding. With *d* embedding dimensions, the sinusoidal or RoPE encoding has *d*/2 frequency bands. A larger *d* provides more frequency bands and finer positional resolution, but also means that the positional encoding occupies a larger fraction of the embedding capacity. In very high-dimensional models, the positional encoding is a modest perturbation relative to the semantic content; in lower-dimensional models, it may be proportionally more significant.

*Depth* determines how many layers can build on the positional signal. Shallow models have fewer opportunities to compose positional information across layers, and the simple local and global patterns of early layers may not fully develop into the sophisticated positional reasoning that deep models can perform. The benefit of sinusoidal or rotary encodings over simpler positional schemes may therefore be more pronounced in deep models than in shallow ones.

### Directions in current research

Beyond the length generalisation challenge and the multi-modal extensions mentioned in section 10, several active research directions are reshaping the field's understanding of positional encoding.

*Dynamic and input-dependent position encodings* learn to assign positions not according to a fixed index but according to the content and structure of the input. In code, for example, the semantically relevant "position" of a token might be its depth in the abstract syntax tree rather than its linear index in the token sequence. Models that can learn to assign content-driven positional representations may handle structured inputs more naturally than models that rely on fixed linear position indices.

*Position-free and implicit position encodings* explore whether explicit positional encoding is necessary at all. Recent work has shown that certain architectural choices — including specific initialisation schemes, attention biases learned from data, and causal masking structures — can implicitly encode position without any explicit positional signal. These approaches challenge the assumption, implicit throughout this article, that position must be explicitly injected. If correct, they would suggest that the field's extensive focus on the design of positional encoding schemes has been addressing a symptom — the need to inject position — rather than the underlying cause — the permutation invariance of attention.

*Positional encoding for long documents and retrieval-augmented generation* presents specific challenges beyond raw sequence length. When a model processes a retrieved document alongside a query, the relevant positional relationships may not be the linear positions within the concatenated sequence but rather the structural relationships between document segments, query components, and generated responses. Research into positional encoding schemes that respect document structure — encoding position relative to segment boundaries rather than sequence start — is an active and practically important area.

*Theoretical understanding of extrapolation* remains limited. Current explanations of why RoPE extensions like YaRN work as well as they do are largely empirical, and the conditions under which any positional encoding scheme will successfully generalise to longer sequences are not well characterised theoretically. Developing a principled theory of length generalisation — analogous to the well-understood theory of in-distribution generalisation — is an important open problem whose solution would likely lead to better encoding designs.

### Practical guidance for practitioners

For practitioners choosing a positional encoding scheme for a new model or adapting an existing one, the following considerations summarise the tradeoffs discussed throughout this article.

For most general-purpose language modelling applications, RoPE or a well-tested variant is the current best choice. It combines zero parameter overhead, exact shift-invariance, full positional expressiveness, and compatibility with standard causal attention, and it has been validated at scales from small research models to the largest open-weight language models. The implementation overhead relative to learned embeddings is modest and well-supported by standard deep learning frameworks.

For applications where context length substantially exceeds the training length, ALiBi or a RoPE extension with position interpolation is worth considering. ALiBi's linear distance penalty generalises most naturally, at the cost of some expressiveness; RoPE with interpolation or YaRN preserves more expressiveness at the cost of requiring careful calibration of the rescaling parameters.

For short, fixed-format inputs where absolute position carries consistent task-specific meaning — structured forms, fixed-length code snippets, standardised document formats — learned positional embeddings remain a defensible choice. The length generalisation failure is not triggered within the training range, the flexibility advantage is real, and the parameter cost is negligible at short sequence lengths.

For research contexts where understanding and interpretability are prioritised over raw performance, sinusoidal encodings remain valuable precisely because their mathematical structure is well-understood and allows principled analysis of the kind developed in this article. The geometric and algebraic properties of sinusoidal encodings make them the right tool for studying how transformers use positional information, even if RoPE is the right tool for production systems.











## 12. Conclusion

### The through-line

This article has traced a single conceptual thread from its beginning — the permutation-invariance problem of self-attention — through its mathematical elaboration — the sinusoidal encoding, its geometric properties, the linear transformation structure, the Toeplitz similarity matrix — through its empirical consequences — the impact on token embeddings, the diversity of attention head behaviour — and finally to its contemporary resolution — the relative, rotary, and bias-based encoding schemes that dominate modern practice. The thread is not a sequence of unrelated results but a progressive clarification of a single underlying idea: that sequential position is fundamentally a relative concept, that the right mathematical language for relative position is rotation, and that the right place to apply that rotation in the architecture is directly inside the attention computation.

The sinusoidal encoding of the original transformer paper was a remarkably good first answer to the positional encoding problem. Its designers recognised that the encoding should be smooth and continuous, that it should span multiple scales, and that it should have properties making relative position learnable. The linear transformation property — that PE(*k*+*m*) = **M**(*m*) · PE(*k*) — encodes all of these desiderata in a single algebraic identity, and the entire geometric analysis of sections 4 through 6 is, in a sense, an unpacking of the implications of that identity.

RoPE is the completion of this idea. By moving the block-diagonal rotation from the input layer to the inside of the attention dot product, it makes the relative-position computation direct, exact, and unconditional. The **M**(*m*) matrix that appeared as a derived consequence of the angle addition formulas in section 5 becomes the primary computational element of the positional encoding scheme. The mathematical content is conserved; the architectural placement is optimised.

### Key insights summarised

Several insights developed across the preceding sections are worth restating in compact form.

The permutation-invariance of self-attention is not a bug but a deliberate feature of the architecture: it is what allows transformers to attend to any token regardless of position. Positional encoding does not eliminate this permutation-invariance; it augments the input so that the model *can* distinguish positions when that is useful, while remaining free to ignore position when content alone is sufficient.

The addition of positional encodings to token embeddings is annotation, not contamination. The positional perturbation is bounded by √(*d*/2) while inter-cluster distances in semantic space are much larger, so the semantic geometry of the embedding space is substantially preserved. The four-term decomposition of the attention score shows that learned weight matrices can access positional and semantic information selectively, in any mixture, by projecting onto the appropriate subspaces.

The linear transformation property — PE(*k*+*m*) = **M**(*m*) · PE(*k*) — is the mathematical heart of the sinusoidal encoding scheme. It follows from the angle addition formulas for sine and cosine and implies that the pairwise similarity matrix of positional encodings is exactly Toeplitz: the geometric relationship between any two positions depends only on their relative offset, not on their absolute locations. This shift-invariance is exact, not approximate, and it is the property that makes relative position learnable by standard linear weight matrices.

Attention heads exploit positional structure in qualitatively different ways. Local heads project onto high-frequency positional dimensions to attend to immediate neighbours. Fixed-offset heads use the **M**(*m*) rotation to attend precisely *m* positions back, regardless of absolute position. Global heads suppress the positional signal and attend by content similarity. Periodic heads project onto mid-frequency bands to track clause or phrase structure. The diversity of head behaviour is not noise but a genuine division of computational labour, enabled by the multi-frequency structure of the encoding.

Learned positional embeddings are more flexible than sinusoidal encodings within the training range, but they sacrifice exact shift-invariance, geometric regularity, and — most consequentially — the ability to generalise to sequences longer than the training length. As context lengths required by practical applications have grown dramatically, this failure mode has made learned embeddings increasingly impractical for general-purpose language models.

RoPE applies the block-diagonal rotation structure of sinusoidal encodings directly inside the attention computation, achieving exact shift-invariance without any learned parameters or input-layer mixing. ALiBi achieves shift-invariance more simply but less expressively, using a fixed linear distance penalty. Both schemes generalise to longer sequences more gracefully than learned embeddings, with RoPE preserving the full expressiveness of the multi-frequency approach and ALiBi providing the most natural extrapolation behaviour.

### What the mathematics reveals about the architecture

One of the rewards of the detailed mathematical analysis conducted in this article is that it reveals the transformer architecture as more coherent and principled than it might appear from a surface reading. The sinusoidal encoding is not an arbitrary engineering choice; it is a natural solution to the constraint that relative position must be recoverable through linear operations. The attention mechanism's dot-product structure is not an arbitrary design; it is exactly the right structure to exploit the rotation-matrix representation of relative position. The multi-head architecture is not arbitrary redundancy; it is what allows simultaneous processing of positional information at multiple frequency scales.

Seen through the lens of the linear transformation property, the transformer is an architecture that has been — partly by design and partly through the pressure of training on large data — optimised to work with rotational representations of sequential position. RoPE makes this optimisation explicit. The block-diagonal rotation that was always present in the sinusoidal encoding, implicit in the mathematics of the angle addition formulas, has been placed at the centre of the architecture where it can operate most naturally and directly.

This coherence suggests that the design principles underlying sinusoidal positional encodings are not merely historically interesting — they are genuinely correct, and the field's movement toward rotary embeddings is the recognition of a truth that was present in the original design, waiting to be fully articulated.

### Looking forward

The history of positional encoding in transformers is a history of progressive clarification: from the observation that attention is permutation-invariant, to the sinusoidal encoding as a first solution, to the recognition that the linear transformation property is the right mathematical structure, to RoPE as the direct architectural expression of that structure. Each step has been driven by a combination of theoretical insight and empirical pressure — the need to handle longer sequences, reduce parameter costs, and improve generalisation — and each step has made the positional encoding more principled, more efficient, and more effective.

The open questions identified in section 11 — length generalisation, multi-modal position, position-free architectures — suggest that this history of clarification is not finished. The framework of rotational representations and shift-invariant similarity structures that has proven so powerful for linear sequences may need to be extended, or in some cases replaced, as transformer architectures are applied to richer and more varied sequential and non-sequential structures. The mathematical foundations will remain relevant as a point of comparison and a source of design principles, even as the specific architectural forms evolve.

What will not change is the underlying insight: that to process a sequence, a model must represent not just what tokens are present but where they stand in relation to each other. The manner of representing that relation — sinusoidal, rotary, biased, or something not yet devised — is an engineering and mathematical question that will continue to be refined. The necessity of representing it at all is a logical consequence of the nature of sequential information, and it will remain central to the design of any architecture that aspires to understand language.




