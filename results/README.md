# Results: my handwriting through `predict.py`

The test set is one photo of each capital letter, A–Z (26 images). `predict.py` runs each photo through my preprocessing (`imgresize.toEmnist`) and then the trained `CNNEMNIST.pt`.

## Summary

| | Result |
|---|---|
| Correct on the top guess | **24/26** |
| Correct letter in the top 3 | **26/26** |

## Misses

| Letter | Top guess | Correct letter's rank |
|---|---|---|
| O | 0 (87.1%) | 2nd (10.3%) |
| S | 5 (60.3%) | 2nd (34.7%) |

## Correct, but not confident

| Letter | Confidence | Runner-up |
|---|---|---|
| U | 49.4% | V (42.3%) |
| K | 49.5% | h (26.6%) |
| Z | 56.4% | 2 (36.7%) |
| D | 60.5% | 0 (26.8%) |

The other 20 letters were all above 70%, and 16 were above 90%.

## Files

- [`alphabet_results.png`](alphabet_results.png): each original photo above the 28×28 image the CNN receives, labeled with its top guess (green = correct, red = wrong).
- [`predictions.csv`](predictions.csv): the top-3 predictions and confidences for every letter.

## Questions for the meeting

1. Should 0 vs O count as an error? EMNIST keeps them as separate classes, but they look the same in many people's handwriting.
2. Most of the misses and low-confidence results are letter/digit pairs (O/0, S/5, Z/2, D/0). Is that expected for the bymerge split?
3. What should I test next: lowercase, digits, or several samples per letter? One sample per letter is a small test.
