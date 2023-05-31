#!/usr/bin/python3

import sys
import os

import easyocr

tree = os.walk(sys.argv[1])

reader = easyocr.Reader(["en"])

texts = []
data = dict()

for it in tree:
	files = it[2]

	for file in files:
		result = "".join(reader.readtext(os.path.join(it[0], file), detail = 0))
		if (result not in data):
			texts.append(result)
			data[result] = []

		data[result].append(os.path.join(it[0], file))

texts = reversed(sorted(texts, key = len))
for text in texts:
	if (len(text) < 4):
		break

	print(f"{text}:")
	for file in data[text]:
		print(file)

	print()
