# Unified PPO Throughput

Five warmed 1M-decision-step trials per row. All runners use current shared-policy 1v1 self-play, lookup actions, kickoff resets, goal reward 10, touch reward 0.1, goal/no-touch-10-second terminals, frameskip 8, 3x256 policy/critic networks, 100,000 collection transitions, 25,000 minibatches, entropy 0.01, and learning rate 2e-4. Rates are physics ticks per second; total includes PPO updates.

## One Epoch

| Method | Decision actors | Collection ticks/s | Total ticks/s |
| --- | ---: | ---: | ---: |
| RLGymPPO Python | 64 | 150,116 +/- 2,171 | 112,678 +/- 1,393 |
| RLGymPPO C++ | 768 | 468,047 +/- 11,046 | 375,614 +/- 9,853 |
| CARL/JARL | 1,024 | 1,737,281 +/- 34,789 | 1,505,551 +/- 25,932 |
| CARL/JARL | 2,048 | 2,825,148 +/- 91,142 | 2,255,980 +/- 59,983 |
| CARL/JARL | 4,096 | 5,276,434 +/- 109,103 | 3,713,999 +/- 55,275 |
| CARL/JARL | 8,192 | 8,554,713 +/- 72,822 | 5,073,206 +/- 29,150 |
| CARL/JARL | 16,384 | 13,154,913 +/- 131,165 | 6,398,822 +/- 37,637 |

## Four Epochs

| Method | Decision actors | Collection ticks/s | Total ticks/s |
| --- | ---: | ---: | ---: |
| RLGymPPO Python | 64 | 148,497 +/- 2,940 | 102,671 +/- 1,714 |
| RLGymPPO C++ | 768 | 463,242 +/- 15,115 | 282,109 +/- 8,282 |
| CARL/JARL | 1,024 | 1,758,148 +/- 27,363 | 1,096,414 +/- 8,462 |
| CARL/JARL | 2,048 | 2,882,179 +/- 64,828 | 1,445,961 +/- 18,404 |
| CARL/JARL | 4,096 | 5,123,973 +/- 76,823 | 1,928,446 +/- 16,776 |
| CARL/JARL | 8,192 | 8,296,552 +/- 27,213 | 2,258,364 +/- 26,487 |
| CARL/JARL | 16,384 | 13,083,825 +/- 150,279 | 2,505,046 +/- 31,008 |

## Eight Epochs

| Method | Decision actors | Collection ticks/s | Total ticks/s |
| --- | ---: | ---: | ---: |
| RLGymPPO Python | 64 | 139,210 +/- 4,866 | 86,880 +/- 2,289 |
| RLGymPPO C++ | 768 | 458,270 +/- 9,571 | 212,377 +/- 3,533 |
| CARL/JARL | 1,024 | 1,708,000 +/- 35,539 | 782,865 +/- 12,176 |
| CARL/JARL | 2,048 | 2,890,048 +/- 37,146 | 973,398 +/- 8,292 |
| CARL/JARL | 4,096 | 5,134,322 +/- 120,745 | 1,195,552 +/- 6,362 |
| CARL/JARL | 8,192 | 8,341,513 +/- 126,597 | 1,296,777 +/- 12,883 |
| CARL/JARL | 16,384 | 13,040,980 +/- 207,565 | 1,384,278 +/- 10,692 |
