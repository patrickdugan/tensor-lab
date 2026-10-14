# Candidate catalogue

Every entry is an inequality `P(a,B) >= sum P(blocks)` coming from a shared-leg degeneration of `C(a,B)` found by the bounded search (`search.py`).  Slack is measured on the extremal profile `P*` (`notes/02_extremal_profile_and_theta.md`): only a negative slack could improve the growth exponent.  An entry is *dominated* when another entry with the same `a` and a source `B' <= B` has blocks that are componentwise at least as large (then it is a trivial consequence: padding, a dropped block or a shrunk block).

## Search `search_a4_b10_k4.json`

Bounds: a <= 4, B <= 10, at most 4 sectors per leg.  3279354 sector structures, 2875257 with feasible sector weights, 1227697 with every block identified, **1723 distinct inequalities**, 373 non-dominated, **0 with negative slack**; 396 s.

| class | all | non-dominated |
|---|---|---|
| known family: alternating C(a,h_i) blocks, tight on the extremal profile | 27 | 27 |
| contains a rectangle block (output leg on the shared X leg) | 190 | 37 |
| known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | 257 | 71 |
| restriction-type (a block uses a proper subset of the shared X leg) | 988 | 155 |
| trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | 261 | 83 |

## Search `search_a5_b16_k3.json`

Bounds: a <= 5, B <= 16, at most 3 sectors per leg.  1842160 sector structures, 1665744 with feasible sector weights, 582132 with every block identified, **5381 distinct inequalities**, 956 non-dominated, **0 with negative slack**; 248 s.

| class | all | non-dominated |
|---|---|---|
| known family: alternating C(a,h_i) blocks, tight on the extremal profile | 155 | 155 |
| contains a rectangle block (output leg on the shared X leg) | 561 | 85 |
| known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | 1994 | 333 |
| restriction-type (a block uses a proper subset of the shared X leg) | 1735 | 196 |
| trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | 936 | 187 |

## Ranked non-dominated candidates

Classes in rank order; within a class, smallest slack first, then larger `a`.  No entry has negative slack, so none can improve the exponent.

| rank | inequality | slack on P* | class | block legs |
|---|---|---|---|---|
| 1 | `P(5,16) >= P(4,5) + P(4,5) + P(4,5)` | 0 | known family: alternating C(a,h_i) blocks, tight on the extremal profile | identity, swap_YZ, identity |
| 2 | `P(4,12) >= P(3,4) + P(3,4) + P(3,4)` | 0 | known family: alternating C(a,h_i) blocks, tight on the extremal profile | identity, swap_YZ, identity |
| 3 | `P(4,13) >= P(3,4) + P(3,4) + P(4,4)` | 0 | known family: alternating C(a,h_i) blocks, tight on the extremal profile | identity, swap_YZ, identity |
| 4 | `P(4,14) >= P(3,4) + P(3,4) + P(4,5)` | 0 | known family: alternating C(a,h_i) blocks, tight on the extremal profile | identity, swap_YZ, identity |
| 5 | `P(4,14) >= P(3,4) + P(4,4) + P(4,4)` | 0 | known family: alternating C(a,h_i) blocks, tight on the extremal profile | identity, swap_YZ, identity |
| 6 | `P(4,15) >= P(3,4) + P(3,4) + P(4,6)` | 0 | known family: alternating C(a,h_i) blocks, tight on the extremal profile | identity, swap_YZ, identity |
| 7 | `P(4,15) >= P(3,4) + P(4,4) + P(4,5)` | 0 | known family: alternating C(a,h_i) blocks, tight on the extremal profile | identity, swap_YZ, identity |
| 8 | `P(4,15) >= P(4,4) + P(4,4) + P(4,4)` | 0 | known family: alternating C(a,h_i) blocks, tight on the extremal profile | identity, swap_YZ, identity |
| 9 | `P(3,5) >= P(2,2) + P(2,3)` | 4/3 | contains a rectangle block (output leg on the shared X leg) | cycle_XYZ, identity |
| 10 | `P(3,6) >= P(2,2) + P(3,3)` | 4/3 | contains a rectangle block (output leg on the shared X leg) | cycle_XYZ, identity |
| 11 | `P(3,7) >= P(2,2) + P(3,4)` | 4/3 | contains a rectangle block (output leg on the shared X leg) | cycle_XYZ, identity |
| 12 | `P(3,8) >= P(2,2) + P(3,5)` | 4/3 | contains a rectangle block (output leg on the shared X leg) | cycle_XYZ, identity |
| 13 | `P(3,9) >= P(2,2) + P(3,6)` | 4/3 | contains a rectangle block (output leg on the shared X leg) | cycle_XYZ, identity |
| 14 | `P(3,10) >= P(2,2) + P(3,7)` | 4/3 | contains a rectangle block (output leg on the shared X leg) | cycle_XYZ, identity |
| 15 | `P(3,11) >= P(2,2) + P(3,8)` | 4/3 | contains a rectangle block (output leg on the shared X leg) | cycle_XYZ, identity |
| 16 | `P(3,12) >= P(2,2) + P(3,9)` | 4/3 | contains a rectangle block (output leg on the shared X leg) | cycle_XYZ, identity |
| 17 | `P(5,15) >= P(3,5) + P(4,5) + P(4,5)` | 7/243 | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | identity, swap_YZ, identity |
| 18 | `P(5,16) >= P(3,5) + P(4,5) + P(5,5)` | 7/243 | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | identity, swap_YZ, identity |
| 19 | `P(4,11) >= P(2,4) + P(3,4) + P(3,4)` | 4/81 | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | identity, swap_YZ, identity |
| 20 | `P(4,12) >= P(2,4) + P(3,4) + P(4,4)` | 4/81 | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | identity, swap_YZ, identity |
| 21 | `P(4,13) >= P(2,4) + P(3,4) + P(4,5)` | 4/81 | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | identity, swap_YZ, identity |
| 22 | `P(4,13) >= P(2,4) + P(4,4) + P(4,4)` | 4/81 | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | identity, swap_YZ, identity |
| 23 | `P(4,14) >= P(2,4) + P(3,4) + P(4,6)` | 4/81 | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | identity, swap_YZ, identity |
| 24 | `P(4,14) >= P(2,4) + P(4,4) + P(4,5)` | 4/81 | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | identity, swap_YZ, identity |
| 25 | `P(2,2) >= P(1,1) + P(1,2)` | 1/3 | restriction-type (a block uses a proper subset of the shared X leg) | identity, identity |
| 26 | `P(2,3) >= P(1,1) + P(2,2)` | 1/3 | restriction-type (a block uses a proper subset of the shared X leg) | identity, identity |
| 27 | `P(2,4) >= P(1,1) + P(2,3)` | 1/3 | restriction-type (a block uses a proper subset of the shared X leg) | identity, identity |
| 28 | `P(2,5) >= P(1,1) + P(1,2) + P(1,2) + P(1,2)` | 1/3 | restriction-type (a block uses a proper subset of the shared X leg) | identity, identity, swap_YZ, identity |
| 29 | `P(2,5) >= P(1,1) + P(2,4)` | 1/3 | restriction-type (a block uses a proper subset of the shared X leg) | identity, identity |
| 30 | `P(2,6) >= P(1,1) + P(1,2) + P(1,2) + P(2,2)` | 1/3 | restriction-type (a block uses a proper subset of the shared X leg) | identity, identity, swap_YZ, identity |
| 31 | `P(2,6) >= P(1,1) + P(2,5)` | 1/3 | restriction-type (a block uses a proper subset of the shared X leg) | identity, identity |
| 32 | `P(2,7) >= P(1,1) + P(1,2) + P(1,2) + P(2,3)` | 1/3 | restriction-type (a block uses a proper subset of the shared X leg) | identity, identity, swap_YZ, identity |
| 33 | `P(1,2) >= P(1,1) + P(1,1)` | 0 | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | identity, identity |
| 34 | `P(1,3) >= P(1,1) + P(1,1) + P(1,1)` | 0 | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | identity, identity, identity |
| 35 | `P(1,3) >= P(1,1) + P(1,2)` | 0 | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | identity, identity |
| 36 | `P(1,4) >= P(1,1) + P(1,1) + P(1,1) + P(1,1)` | 0 | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | identity, identity, identity, identity |
| 37 | `P(1,4) >= P(1,1) + P(1,1) + P(1,2)` | 0 | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | identity, identity, identity |
| 38 | `P(1,4) >= P(1,1) + P(1,3)` | 0 | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | identity, identity |
| 39 | `P(1,4) >= P(1,2) + P(1,2)` | 0 | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | identity, identity |
| 40 | `P(1,5) >= P(1,1) + P(1,1) + P(1,1) + P(1,2)` | 0 | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | identity, identity, identity, identity |

### Certified search results (independent verifier)

- `P(1,4) >= P(1,1) + P(1,1) + P(1,1) + P(1,1)`: `certificates\search\search_a1_B4_1x1_1x1_1x1_1x1.json` PASS
- `P(1,5) >= P(1,1) + P(1,1) + P(1,1) + P(1,2)`: `certificates\search\search_a1_B5_1x1_1x1_1x1_1x2.json` PASS
- `P(1,6) >= P(1,1) + P(1,1) + P(1,1) + P(1,3)`: `certificates\search\search_a1_B6_1x1_1x1_1x1_1x3.json` PASS
- `P(1,6) >= P(1,1) + P(1,1) + P(1,2) + P(1,2)`: `certificates\search\search_a1_B6_1x1_1x1_1x2_1x2.json` PASS
- `P(1,7) >= P(1,1) + P(1,1) + P(1,1) + P(1,4)`: `certificates\search\search_a1_B7_1x1_1x1_1x1_1x4.json` PASS
- `P(1,7) >= P(1,1) + P(1,1) + P(1,2) + P(1,3)`: `certificates\search\search_a1_B7_1x1_1x1_1x2_1x3.json` PASS
- `P(1,7) >= P(1,1) + P(1,2) + P(1,2) + P(1,2)`: `certificates\search\search_a1_B7_1x1_1x2_1x2_1x2.json` PASS
- `P(1,8) >= P(1,1) + P(1,1) + P(1,1) + P(1,5)`: `certificates\search\search_a1_B8_1x1_1x1_1x1_1x5.json` PASS
- `P(1,8) >= P(1,1) + P(1,1) + P(1,2) + P(1,4)`: `certificates\search\search_a1_B8_1x1_1x1_1x2_1x4.json` PASS
- `P(1,8) >= P(1,1) + P(1,1) + P(1,3) + P(1,3)`: `certificates\search\search_a1_B8_1x1_1x1_1x3_1x3.json` PASS
- `P(1,8) >= P(1,1) + P(1,2) + P(1,2) + P(1,3)`: `certificates\search\search_a1_B8_1x1_1x2_1x2_1x3.json` PASS
- `P(1,8) >= P(1,2) + P(1,2) + P(1,2) + P(1,2)`: `certificates\search\search_a1_B8_1x2_1x2_1x2_1x2.json` PASS

## Rediscovery of the paper's construction (`search_a5_b16_k3.json`)

| a | h | B = 3h+a-1 | found by the search | blocks / legs | certificate |
|---|---|---|---|---|---|
| 2 | 1 | 4 | yes | identity, swap_YZ, identity | `certificates/search/search_a2_B4_1x2_1x2_1x2.json` PASS |
| 2 | 2 | 7 | yes | identity, swap_YZ, identity | `certificates/search/search_a2_B7_2x2_2x2_2x2.json` PASS |
| 2 | 3 | 10 | yes | identity, swap_YZ, identity | `certificates/search/search_a2_B10_2x3_2x3_2x3.json` PASS |
| 2 | 4 | 13 | yes | identity, swap_YZ, identity | `certificates/search/search_a2_B13_2x4_2x4_2x4.json` PASS |
| 2 | 5 | 16 | yes | identity, swap_YZ, identity | `certificates/search/search_a2_B16_2x5_2x5_2x5.json` PASS |
| 3 | 1 | 5 | yes | identity, swap_YZ, identity | `certificates/search/search_a3_B5_1x3_1x3_1x3.json` PASS |
| 3 | 2 | 8 | yes | identity, swap_YZ, identity | `certificates/search/search_a3_B8_2x3_2x3_2x3.json` PASS |
| 3 | 3 | 11 | yes | identity, swap_YZ, identity | `certificates/search/search_a3_B11_3x3_3x3_3x3.json` PASS |
| 3 | 4 | 14 | yes | identity, swap_YZ, identity | `certificates/search/search_a3_B14_3x4_3x4_3x4.json` PASS |
| 4 | 1 | 6 | yes | identity, swap_YZ, identity | `certificates/search/search_a4_B6_1x4_1x4_1x4.json` PASS |
| 4 | 2 | 9 | yes | identity, swap_YZ, identity | `certificates/search/search_a4_B9_2x4_2x4_2x4.json` PASS |
| 4 | 3 | 12 | yes | identity, swap_YZ, identity | `certificates/search/search_a4_B12_3x4_3x4_3x4.json` PASS |
| 4 | 4 | 15 | yes | identity, swap_YZ, identity | `certificates/search/search_a4_B15_4x4_4x4_4x4.json` PASS |
| 5 | 1 | 7 | yes | identity, swap_YZ, identity | `certificates/search/search_a5_B7_1x5_1x5_1x5.json` PASS |
| 5 | 2 | 10 | yes | identity, swap_YZ, identity | `certificates/search/search_a5_B10_2x5_2x5_2x5.json` PASS |
| 5 | 3 | 13 | yes | identity, swap_YZ, identity | `certificates/search/search_a5_B13_3x5_3x5_3x5.json` PASS |
| 5 | 4 | 16 | yes | identity, swap_YZ, identity | `certificates/search/search_a5_B16_4x5_4x5_4x5.json` PASS |

All instances inside the bounds rediscovered: **True**.

## Certified representatives of each class

| inequality | class | certificate | verifier |
|---|---|---|---|
| `P(5,16) >= P(4,5) + P(4,5) + P(4,5)` | known family: alternating C(a,h_i) blocks, tight on the extremal profile | `certificates/search/search_a5_B16_4x5_4x5_4x5.json` | PASS |
| `P(4,12) >= P(3,4) + P(3,4) + P(3,4)` | known family: alternating C(a,h_i) blocks, tight on the extremal profile | `certificates/search/search_a4_B12_3x4_3x4_3x4.json` | PASS |
| `P(4,13) >= P(3,4) + P(3,4) + P(4,4)` | known family: alternating C(a,h_i) blocks, tight on the extremal profile | `certificates/search/search_a4_B13_3x4_3x4_4x4.json` | PASS |
| `P(3,5) >= P(2,2) + P(2,3)` | contains a rectangle block (output leg on the shared X leg) | `certificates/search/search_a3_B5_2x2_2x3.json` | PASS |
| `P(3,6) >= P(2,2) + P(3,3)` | contains a rectangle block (output leg on the shared X leg) | `certificates/search/search_a3_B6_2x2_3x3.json` | PASS |
| `P(3,7) >= P(2,2) + P(3,4)` | contains a rectangle block (output leg on the shared X leg) | `certificates/search/search_a3_B7_2x2_3x4.json` | PASS |
| `P(5,15) >= P(3,5) + P(4,5) + P(4,5)` | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | `certificates/search/search_a5_B15_3x5_4x5_4x5.json` | PASS |
| `P(5,16) >= P(3,5) + P(4,5) + P(5,5)` | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | `certificates/search/search_a5_B16_3x5_4x5_5x5.json` | PASS |
| `P(4,11) >= P(2,4) + P(3,4) + P(3,4)` | known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted) | `certificates/search/search_a4_B11_2x4_3x4_3x4.json` | PASS |
| `P(2,2) >= P(1,1) + P(1,2)` | restriction-type (a block uses a proper subset of the shared X leg) | `certificates/search/search_a2_B2_1x1_1x2.json` | PASS |
| `P(2,3) >= P(1,1) + P(2,2)` | restriction-type (a block uses a proper subset of the shared X leg) | `certificates/search/search_a2_B3_1x1_2x2.json` | PASS |
| `P(2,4) >= P(1,1) + P(2,3)` | restriction-type (a block uses a proper subset of the shared X leg) | `certificates/search/search_a2_B4_1x1_2x3.json` | PASS |
| `P(1,2) >= P(1,1) + P(1,1)` | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | `certificates/search/search_a1_B2_1x1_1x1.json` | PASS |
| `P(1,3) >= P(1,1) + P(1,1) + P(1,1)` | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | `certificates/search/search_a1_B3_1x1_1x1_1x1.json` | PASS |
| `P(1,3) >= P(1,1) + P(1,2)` | trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive) | `certificates/search/search_a1_B3_1x1_1x2.json` | PASS |


## Structured families

| family | status | evidence |
|---|---|---|
| three-sector `P(a,3h+a-1) >= 3P(a,h)` (paper Lemma 4.2) | reproduced, tight on P* | 25 certificates, Lean `Finite` (4 cases) |
| odd-sector `P(a,(2e+1)h+e(a-1)) >= (2e+1)P(a,h)` | valid, **implied** by (C)+(T) | Lean `Obstruction.odd_sector_implied` |
| any input-on-X convolution blocks in one `C(a,B)` | cannot violate P* | Lean `Extremal.counting_obstruction` |
| rectangle staircase `P(a,(n-1)(p+q-1)+p) >= n P(p,q)` | valid, slack > 0 on all 17666 instances with B <= 40 | `sectors.staircase_certificate` |
| double staircase (gaps filled) | **rejected**: not realizable (68 of 17666 hypothetical instances would beat P*) | Farkas certificates, `logs/realizability.json` E3 |
| two interleaved rectangle classes | **rejected**: not realizable | Farkas certificate, E2 |
| dense rectangles `C(3,2n) => n x C(2,2)^sigma` (e.g. `C(3,16)`, n=8) | **rejected for every degeneration**: needs `B >= 3n-1` (Koszul), staircase optimal | `certificates/dense_p2_n8_B16.json`, `notes/05` |
| dense rectangles from two copies `C(3,2n) (+) C(3,2n)` | valid restriction, too weak (factor 2) | `dense.two_copy_restriction` |
| dense rectangles, squares from two copies `T_2^(x)2 <= 2 C(3,4)^(x)2` | valid **degeneration** (one copy impossible): cost sqrt(2) per factor | `certificates/amortize_n2_square.json`, `notes/06` |
| dense rectangles, squares from two copies `T_n^(x)2 <| 2 C(3,2n)^(x)2`, every n <= 16 (incl. `C(3,16)`) | valid **degeneration** (parity-twisted pair): cost sqrt(2) per factor; still far from the cost 1 the exponent needs | `certificates/amortize_two_copy_square_n8.json`, `notes/06` |
| dense rectangles, squares from three copies `T_n^(x)2 <= 3 C(3,2n)^(x)2`, every n | valid **restriction** (no limit): cost sqrt(3) per factor | `certificates/amortize_three_copy_square_n8.json`, `notes/06` |
| first-order (tangential, non-vanishing factors) two-copy amortisation beyond the C(3,4) square | **rejected** (exact forced zeros, e.g. 744/1024 for `C(3,16)`, N=2) | `logs/first_order_obstruction.json` |
| any exact block packing in `C(3,B)` | cannot beat P* (Koszul LP) | `logs/koszul_lp.json` |

## Rejected packings (general monomial weights, exact LP)

| a | B | P*(a,B) | packings with value > P* | tested | realizable | all sampled obstructions verified |
|---|---|---|---|---|---|---|
| 3 | 15 | 224/9 | 31 | 31 | 0 | True |
| 3 | 16 | 238/9 | 33 | 33 | 0 | True |
| 3 | 18 | 266/9 | 172 | 150 | 0 | True |
| 4 | 12 | 70/3 | 2 | 2 | 0 | True |
| 4 | 14 | 2170/81 | 13 | 13 | 0 | True |
| 4 | 16 | 2450/81 | 72 | 72 | 0 | True |
| 5 | 12 | 6370/243 | 4 | 4 | 0 | True |
| 5 | 14 | 7280/243 | 15 | 15 | 0 | True |

