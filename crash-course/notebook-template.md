# Lab notebook (fill in as you run `code/10`, `code/11`, `code/12`)

Rules: write the *observed* line from the script's actual output, in your own words, with
the number. If you cannot explain the mechanism in one sentence, re-read that section of
`12-depth-layer.md` before moving on. These entries are what you cite in the interview.

## 10 · Training dynamics

### A. bf16 drops small updates
- Expected:
- Observed:
- Mechanism:
- In prod:

### B. Weight decay × normalization = effective learning rate
- Expected:
- Observed:
- Mechanism:
- In prod:

### C. Adam after a quiet period (β2)
- Expected:
- Observed:
- Mechanism:
- In prod:

### D. Critical batch size grows during training
- Expected:
- Observed:
- Mechanism:
- In prod:

### E. Double descent
- Expected:
- Observed:
- Mechanism:
- In prod:

## 11 · Logged data

### 1. Position bias is confounded with relevance
- Expected:
- Observed:
- Mechanism:
- In prod:

### 2. Offline evaluation favours the incumbent; IPS needs exploration
- Expected:
- Observed:
- Mechanism:
- In prod:

### 3. Negative downsampling shifts the prior
- Expected:
- Observed:
- Mechanism:
- In prod:

### 4. Delayed feedback
- Expected:
- Observed:
- Mechanism:
- In prod:

### 5. Selective labels
- Expected:
- Observed:
- Mechanism:
- In prod:

## 12 · Systems

### 1. Embedding spaces across retrains
- Expected:
- Observed:
- Mechanism:
- In prod:

### 2. MLP vs dot product
- Expected:
- Observed:
- Mechanism:
- In prod:

### 3. ANN recall on the tail
- Expected:
- Observed:
- Mechanism:
- In prod:

### 4. Train/serve skew with identical marginals
- Expected:
- Observed:
- Mechanism:
- In prod:

### 5. int8 outlier channels
- Expected:
- Observed:
- Mechanism:
- In prod:

### 6. Peeking, clustered variance, CUPED
- Expected:
- Observed:
- Mechanism:
- In prod:

## Earlier scripts worth an entry too

- `05`: sampled-negative evaluation inflated Recall@10 from __ to __.
- `06`: logQ correction moved head recall from __ to __ and coverage from __ to __.
- `03`: forgetting `zero_grad` doubled the gradient; double softmax bounded the loss at __.
- `08`: a leaky feature took ROC-AUC to __; the cost-optimal threshold was __, not 0.5.
