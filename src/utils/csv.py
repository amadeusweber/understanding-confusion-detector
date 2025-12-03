import csv
def csv_col_dict(csv_data):
    result = dict()
    for row in csv.DictReader(csv_data):
        for key, value in row.items():
            result.setdefault(key, []).append(value)

    return result