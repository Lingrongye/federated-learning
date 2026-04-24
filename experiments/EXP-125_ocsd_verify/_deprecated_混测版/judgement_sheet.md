# Judgement Sheet — 50 over-conf wrong samples

每张打标 A/B/C:
- **A** = 风格极端 (油画化/抽象化, shortcut 嫌疑)
- **B** = 画得不像 (本质难样本, 标注合理但画风不具备 class 特征)
- **C** = 标注歧义/错误 (图和 label 不符)

| # | 文件 | 真 | 预测 | conf | A/B/C | 备注 |
|---|---|---|---|:-:|:-:|---|
| 1 | 001_person_asPred_elephant_conf1.00_pic_123.jpg | person | elephant | 0.997 |   |   |
| 2 | 002_giraffe_asPred_horse_conf1.00_pic_273.jpg | giraffe | horse | 0.995 |   |   |
| 3 | 003_elephant_asPred_dog_conf0.99_pic_174.jpg | elephant | dog | 0.991 |   |   |
| 4 | 004_person_asPred_horse_conf0.99_pic_465.jpg | person | horse | 0.988 |   |   |
| 5 | 005_dog_asPred_person_conf0.99_pic_254.jpg | dog | person | 0.987 |   |   |
| 6 | 006_person_asPred_dog_conf0.99_pic_009.jpg | person | dog | 0.986 |   |   |
| 7 | 007_horse_asPred_giraffe_conf0.99_pic_054.jpg | horse | giraffe | 0.986 |   |   |
| 8 | 008_guitar_asPred_person_conf0.99_pic_152.jpg | guitar | person | 0.985 |   |   |
| 9 | 009_dog_asPred_elephant_conf0.98_pic_105.jpg | dog | elephant | 0.978 |   |   |
| 10 | 010_giraffe_asPred_house_conf0.98_pic_252.jpg | giraffe | house | 0.978 |   |   |
| 11 | 011_guitar_asPred_horse_conf0.98_pic_002.jpg | guitar | horse | 0.977 |   |   |
| 12 | 012_dog_asPred_elephant_conf0.98_pic_306.jpg | dog | elephant | 0.977 |   |   |
| 13 | 013_person_asPred_dog_conf0.97_pic_001.jpg | person | dog | 0.971 |   |   |
| 14 | 014_dog_asPred_horse_conf0.97_pic_002.jpg | dog | horse | 0.970 |   |   |
| 15 | 015_guitar_asPred_house_conf0.96_pic_089.jpg | guitar | house | 0.965 |   |   |
| 16 | 016_horse_asPred_elephant_conf0.96_pic_226.jpg | horse | elephant | 0.961 |   |   |
| 17 | 017_horse_asPred_dog_conf0.95_pic_083.jpg | horse | dog | 0.955 |   |   |
| 18 | 018_elephant_asPred_dog_conf0.95_pic_142.jpg | elephant | dog | 0.954 |   |   |
| 19 | 019_elephant_asPred_dog_conf0.95_pic_155.jpg | elephant | dog | 0.953 |   |   |
| 20 | 020_elephant_asPred_house_conf0.95_pic_180.jpg | elephant | house | 0.951 |   |   |
| 21 | 021_person_asPred_dog_conf0.95_pic_312.jpg | person | dog | 0.950 |   |   |
| 22 | 022_guitar_asPred_dog_conf0.95_pic_116.jpg | guitar | dog | 0.949 |   |   |
| 23 | 023_person_asPred_dog_conf0.94_pic_155.jpg | person | dog | 0.942 |   |   |
| 24 | 024_dog_asPred_elephant_conf0.94_pic_375.jpg | dog | elephant | 0.939 |   |   |
| 25 | 025_person_asPred_dog_conf0.94_pic_330.jpg | person | dog | 0.937 |   |   |
| 26 | 026_giraffe_asPred_person_conf0.92_pic_286.jpg | giraffe | person | 0.924 |   |   |
| 27 | 027_dog_asPred_guitar_conf0.92_pic_334.jpg | dog | guitar | 0.916 |   |   |
| 28 | 028_guitar_asPred_dog_conf0.91_pic_130.jpg | guitar | dog | 0.909 |   |   |
| 29 | 029_dog_asPred_person_conf0.91_pic_016.jpg | dog | person | 0.908 |   |   |
| 30 | 030_person_asPred_elephant_conf0.90_pic_122.jpg | person | elephant | 0.903 |   |   |
| 31 | 031_dog_asPred_horse_conf0.90_pic_383.jpg | dog | horse | 0.902 |   |   |
| 32 | 032_horse_asPred_person_conf0.90_pic_165.jpg | horse | person | 0.898 |   |   |
| 33 | 033_dog_asPred_elephant_conf0.90_pic_132.jpg | dog | elephant | 0.895 |   |   |
| 34 | 034_horse_asPred_elephant_conf0.89_pic_208.jpg | horse | elephant | 0.894 |   |   |
| 35 | 035_person_asPred_house_conf0.88_pic_135.jpg | person | house | 0.883 |   |   |
| 36 | 036_dog_asPred_elephant_conf0.88_pic_178.jpg | dog | elephant | 0.879 |   |   |
| 37 | 037_person_asPred_dog_conf0.88_pic_219.jpg | person | dog | 0.879 |   |   |
| 38 | 038_person_asPred_giraffe_conf0.87_pic_265.jpg | person | giraffe | 0.868 |   |   |
| 39 | 039_house_asPred_person_conf0.87_pic_219.jpg | house | person | 0.868 |   |   |
| 40 | 040_guitar_asPred_giraffe_conf0.86_pic_084.jpg | guitar | giraffe | 0.863 |   |   |
| 41 | 041_dog_asPred_horse_conf0.86_pic_202.jpg | dog | horse | 0.862 |   |   |
| 42 | 042_dog_asPred_guitar_conf0.85_pic_176.jpg | dog | guitar | 0.854 |   |   |
| 43 | 043_person_asPred_horse_conf0.85_pic_229.jpg | person | horse | 0.850 |   |   |
| 44 | 044_horse_asPred_dog_conf0.85_pic_065.jpg | horse | dog | 0.847 |   |   |
| 45 | 045_person_asPred_dog_conf0.84_pic_484.jpg | person | dog | 0.842 |   |   |
| 46 | 046_person_asPred_horse_conf0.84_pic_175.jpg | person | horse | 0.841 |   |   |
| 47 | 047_person_asPred_elephant_conf0.84_pic_034.jpg | person | elephant | 0.835 |   |   |
| 48 | 048_horse_asPred_elephant_conf0.83_pic_008.jpg | horse | elephant | 0.832 |   |   |
| 49 | 049_house_asPred_person_conf0.81_pic_129.jpg | house | person | 0.814 |   |   |
| 50 | 050_guitar_asPred_person_conf0.81_pic_003.jpg | guitar | person | 0.807 |   |   |
