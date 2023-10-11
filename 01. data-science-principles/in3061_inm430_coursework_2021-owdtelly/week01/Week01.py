# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:01:43 2023

@author: Chris Henry
"""

import csv
import numpy as np

file = open("TB_burden_countries_2014-09-29.csv", "r")

reader = csv.reader(file)

number_of_rows = 0

for row in reader:
    number_of_rows += 1

with open("TB_burden_countries_2014-09-29.csv", "r") as file:
    data = []
    reader = csv.reader(file)
    headers = next(reader)

    print(number_of_rows)

    for row in reader:
        reader_dict = {}

        for i, value in enumerate(row):
            reader_dict[headers[i]] = value

        data.append(reader_dict)


total_cases = 0
filtered_cases = 0
for item in data:
    try:
        # sum_cases = sum(float(item["e_prev_num_lo"]))
        total_cases += float(item["e_prev_num_lo"])
    except (TypeError, ValueError):
        print("Not a number")

    try:
        if item["year"] == "2011" or item["year"] == "1990":
            filtered_cases += float(item["e_prev_num_lo"])
    except (TypeError, ValueError):
        print("Not a number")

average = total_cases / (number_of_rows - 1)

filtered_average = filtered_cases / (number_of_rows - 1)


array = np.arange(5, 16)

even_array = np.arange(0, 24, 2)

print(array)

print(even_array)

print("hello")
