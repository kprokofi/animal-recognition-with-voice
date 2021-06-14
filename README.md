# animal-recognition-with-voice
HSE project. Idea to create animal detection system with voice guidance

# Requirements
python==3.7

# Result metrics 
Pascal VOC AP evaluation
31: toucan: AP 0.9574, precision 0.2416, recall 0.9789
7: water_ouzel: AP 0.9536, precision 0.4218, recall 0.9780
44: redshank: AP 0.9506, precision 0.3932, recall 0.9787
83: three-toed_sloth: AP 0.9174, precision 0.2229, recall 0.9750
11: tailed_frog: AP 0.8963, precision 0.1876, recall 0.9405
17: African_crocodile: AP 0.8824, precision 0.1549, recall 0.9540
0: goldfish: AP 0.8808, precision 0.2061, recall 0.9072
9: European_fire_salamander: AP 0.8668, precision 0.7000, recall 0.8750
41: American_egret: AP 0.8539, precision 0.3143, recall 0.9167
30: jacamar: AP 0.8486, precision 0.3951, recall 0.8889
8: kite: AP 0.8458, precision 0.1894, recall 0.8776
61: jaguar: AP 0.8356, precision 0.4100, recall 0.9318
4: bulbul: AP 0.8249, precision 0.1502, recall 0.8636
63: tiger: AP 0.8243, precision 0.2222, recall 0.9412
10: axolotl: AP 0.8185, precision 0.2500, recall 0.8824
32: drake: AP 0.8087, precision 0.3529, recall 0.8372
58: Persian_cat: AP 0.8070, precision 0.2686, recall 0.9432
96: eel: AP 0.8002, precision 0.2476, recall 0.8636
94: indri: AP 0.7993, precision 0.1882, recall 0.8750
26: prairie_chicken: AP 0.7961, precision 0.3182, recall 0.8370
3: robin: AP 0.7944, precision 0.2390, recall 0.8444
28: macaw: AP 0.7895, precision 0.3564, recall 0.8182
99: puffer: AP 0.7706, precision 0.4375, recall 0.8000
86: chimpanzee: AP 0.7645, precision 0.3556, recall 0.7805
40: little_blue_heron: AP 0.7566, precision 0.1317, recall 0.8571
43: European_gallinule: AP 0.7503, precision 0.5424, recall 0.7805
6: chickadee: AP 0.7450, precision 0.3838, recall 0.8444
29: hornbill: AP 0.7387, precision 0.4337, recall 0.7826
39: flamingo: AP 0.7346, precision 0.5537, recall 0.7444
27: partridge: AP 0.7223, precision 0.5000, recall 0.7561
22: water_snake: AP 0.7150, precision 0.3333, recall 0.8140
65: brown_bear: AP 0.7100, precision 0.1635, recall 0.8293
42: crane: AP 0.6983, precision 0.3857, recall 0.7297
19: thunder_snake: AP 0.6908, precision 0.2547, recall 0.7941
47: killer_whale: AP 0.6897, precision 0.0785, recall 0.8293
14: common_iguana: AP 0.6868, precision 0.3699, recall 0.7941
16: Komodo_dragon: AP 0.6818, precision 0.2566, recall 0.7838
84: orangutan: AP 0.6251, precision 0.1236, recall 0.8462
72: porcupine: AP 0.6178, precision 0.2180, recall 0.7436
75: bison: AP 0.6174, precision 0.0678, recall 0.7667
34: goose: AP 0.6173, precision 0.2160, recall 0.7292
38: jellyfish: AP 0.6168, precision 0.1602, recall 0.7857
54: hyena: AP 0.6128, precision 0.2059, recall 0.8537
51: Newfoundland: AP 0.6114, precision 0.0434, recall 0.8049
90: marmoset: AP 0.6093, precision 0.2061, recall 0.7500
73: hippopotamus: AP 0.6078, precision 0.1133, recall 0.8718
48: Japanese_spaniel: AP 0.6058, precision 0.2589, recall 0.7250
45: pelican: AP 0.5936, precision 0.2542, recall 0.6818
70: tiger_beetle: AP 0.5911, precision 0.1081, recall 0.7273
1: great_white_shark: AP 0.5848, precision 0.4194, recall 0.6047
97: coho: AP 0.5843, precision 0.2381, recall 0.7500
46: king_penguin: AP 0.5773, precision 0.1263, recall 0.7705
62: lion: AP 0.5663, precision 0.4630, recall 0.6944
66: American_black_bear: AP 0.5610, precision 0.1538, recall 0.7500
98: sturgeon: AP 0.5587, precision 0.0982, recall 0.7755
85: gorilla: AP 0.5555, precision 0.1959, recall 0.7073
50: pug: AP 0.5507, precision 0.0730, recall 0.7021
33: red-breasted_merganser: AP 0.5406, precision 0.3529, recall 0.5714
52: Cardigan: AP 0.5350, precision 0.3919, recall 0.6591
71: hamster: AP 0.5348, precision 0.3333, recall 0.5714
93: Madagascar_cat: AP 0.5344, precision 0.1475, recall 0.7442
92: howler_monkey: AP 0.5330, precision 0.1954, recall 0.6939
69: meerkat: AP 0.5313, precision 0.1857, recall 0.7429
68: mongoose: AP 0.5236, precision 0.4237, recall 0.7143
55: kit_fox: AP 0.5195, precision 0.2972, recall 0.8182
64: cheetah: AP 0.5142, precision 0.1649, recall 0.7949
13: mud_turtle: AP 0.5063, precision 0.2095, recall 0.6875
81: black-footed_ferret: AP 0.5034, precision 0.1135, recall 0.8140
59: leopard: AP 0.4903, precision 0.2424, recall 0.6316
74: ox: AP 0.4841, precision 0.0929, recall 0.6176
79: weasel: AP 0.4821, precision 0.1605, recall 0.6047
15: Gila_monster: AP 0.4692, precision 0.2424, recall 0.6667
60: snow_leopard: AP 0.4615, precision 0.2857, recall 0.6829
77: impala: AP 0.4595, precision 0.2639, recall 0.5758
5: jay: AP 0.4550, precision 0.5405, recall 0.5000
24: diamondback: AP 0.4337, precision 0.2566, recall 0.7632
37: koala: AP 0.4210, precision 0.1709, recall 0.6250
53: white_wolf: AP 0.4198, precision 0.2468, recall 0.5429
76: ram: AP 0.4173, precision 0.0884, recall 0.5238
78: llama: AP 0.4078, precision 0.0682, recall 0.5854
67: ice_bear: AP 0.3905, precision 0.3134, recall 0.6364
57: tabby: AP 0.3879, precision 0.2500, recall 0.6829
95: African_elephant: AP 0.3853, precision 0.1316, recall 0.7447
36: wallaby: AP 0.3833, precision 0.1667, recall 0.5333
12: leatherback_turtle: AP 0.3825, precision 0.2472, recall 0.6667
88: colobus: AP 0.3751, precision 0.1031, recall 0.5952
35: echidna: AP 0.3738, precision 0.1981, recall 0.6000
25: trilobite: AP 0.3606, precision 0.2644, recall 0.6571
91: capuchin: AP 0.3591, precision 0.1605, recall 0.6667
89: proboscis_monkey: AP 0.3486, precision 0.2125, recall 0.4857
18: triceratops: AP 0.3427, precision 0.1556, recall 0.4516
20: ringneck_snake: AP 0.3333, precision 0.1073, recall 0.6111
21: hognose_snake: AP 0.2665, precision 0.2022, recall 0.4737
56: Arctic_fox: AP 0.2591, precision 0.1195, recall 0.6750
2: cock: AP 0.2361, precision 0.1102, recall 0.5000
87: siamang: AP 0.2317, precision 0.2277, recall 0.5750
49: Afghan_hound: AP 0.2289, precision 0.2154, recall 0.3415
23: boa_constrictor: AP 0.1518, precision 0.1477, recall 0.3611
80: mink: AP 0.1302, precision 0.1096, recall 0.4103
82: otter: AP 0.0959, precision 0.1705, recall 0.3191
mAP@IoU=0.50 result: 58.671837
mPrec@IoU=0.50 result: 24.441985
mRec@IoU=0.50 result: 72.986519
