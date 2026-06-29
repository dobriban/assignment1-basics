# Tokenizer Compression Experiments Report

## Experiment Setup

We conducted compression ratio analysis using:
- **10 sampled documents** from TinyStories dataset
- **10 sampled documents** from OpenWebText dataset  
- **TinyStories tokenizer** (10K vocabulary size)
- **Random seed 42** for reproducibility

## Results

### TinyStories Tokenizer on TinyStories Data

| Document | Bytes | Tokens | Ratio (bytes/token) | Preview |
|----------|-------|--------|---------------------|---------|
| 1 | 7,607 | 1,928 | 3.95 | "Once upon a time, there was a rich family..." |
| 2 | 4,654 | 1,195 | 3.89 | "There was a frog named Fred..." |
| 3 | 3,470 | 891 | 3.89 | "Once upon a time there was a little girl..." |
| 4 | 800 | 189 | 4.23 | "John loved to twist things..." |
| 5 | 7,724 | 1,934 | 3.99 | "Tommy liked to drink milk before bed..." |
| 6 | 1,558 | 401 | 3.89 | "Once there was a farmer who had a hay field..." |
| 7 | 7,465 | 1,909 | 3.91 | "Anna liked to play with her dolls..." |
| 8 | 1,552 | 375 | 4.14 | "Anna liked to climb..." |
| 9 | 2,110 | 548 | 3.85 | "Tommy liked to play with his truck..." |
| 10 | 1,480 | 356 | 4.16 | "Once upon a time, there was a pink bunny..." |

**Overall: 38,420 bytes / 9,726 tokens = 3.950 bytes/token**

### TinyStories Tokenizer on OpenWebText Data

| Document | Bytes | Tokens | Ratio (bytes/token) | Preview |
|----------|-------|--------|---------------------|---------|
| 1 | 291 | 87 | 3.34 | "When They Come Calling doesn't rely on paranormal hooks..." |
| 2 | 440 | 110 | 4.00 | "Morneau even gave media and his company..." |
| 3 | 383 | 137 | 2.80 | "It's good news for Google, though it still pales..." |
| 4 | 233 | 65 | 3.58 | "Jed is a warrior from another era..." |
| 5 | 238 | 65 | 3.66 | "Anna is a physician from Kansas City..." |
| 6 | 564 | 178 | 3.17 | "Today's judgment is a major blow against mass surveillance..." |
| 7 | 245 | 73 | 3.36 | "Author Royalty -- As a partnership publisher..." |
| 8 | 220 | 53 | 4.15 | "We don't want to get ahead of ourselves..." |
| 9 | 146 | 37 | 3.95 | "And not only did Morneau not really answer..." |
| 10 | 456 | 142 | 3.21 | "That meant he held the shares 'indirectly'..." |

**Overall: 3,216 bytes / 947 tokens = 3.396 bytes/token**

## Analysis

### Key Findings

1. **Domain Adaptation Effect**: The TinyStories tokenizer is **more efficient** on OpenWebText than on its training domain (-0.554 bytes/token difference).

2. **Compression Ratios**:
   - **TinyStories**: 3.950 bytes/token
   - **OpenWebText**: 3.396 bytes/token

3. **Variance Analysis**:
   - TinyStories documents show ratios ranging from 3.85 to 4.23
   - OpenWebText documents show ratios ranging from 2.80 to 4.15 (higher variance)

### Counterintuitive Result

The result that the tokenizer is **more efficient on out-of-domain data** (OpenWebText) is surprising and could be explained by:

1. **Document Length**: OpenWebText samples were generally shorter, which might affect tokenization efficiency
2. **Language Complexity**: TinyStories uses simpler, more repetitive language that might not compress as efficiently
3. **Vocabulary Overlap**: Some common words/patterns might still compress well across domains
4. **Sample Size**: Only 10 documents per domain - results might vary with larger samples

### Recommended Next Steps

1. **Train OpenWebText Tokenizer**: A 32K vocabulary OpenWebText tokenizer should be trained and compared
2. **Larger Sample Size**: Test with 100+ documents for more robust statistics
3. **Control for Document Length**: Use documents of similar lengths for fairer comparison
4. **Cross-Domain Analysis**: Test both tokenizers on both domains

### Limitations

- Only tested one tokenizer (TinyStories) due to time constraints
- Small sample size (10 documents each)
- OpenWebText tokenizer training was not completed
- No statistical significance testing performed

## Raw Data

Complete results are saved in `tokenizer_experiment_results.json` for further analysis.