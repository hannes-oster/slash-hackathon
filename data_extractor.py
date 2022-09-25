from tqdm import tqdm
from multiprocessing import Pool
import pandas
import requests
import csv
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
from csv import reader
import datetime

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}

x = "https://www.myshiptracking.com/vessels/EUROPE-mmsi-205408000-imo-9235268"

base_url = "https://www.myshiptracking.com/vessels/"
# open file in read mode

dataset_path = Path("ships.csv")
def get_data(row):
    row = row[1]
    row = [str(i) for i in row]
    # print(list(row))
    name = row[0].replace(" ", "-")
    url = base_url + name + "-" + "mmsi" + "-" + row[1] + "-" + "imo" + "-" + row[2]
    
    # print(url)
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.findAll('table', class_='table table-sm table-borderless my-0 w-50 w-sm-75')
    # print(table)
    table = str(table).split("\n")
    data = [None] * 5
    for j in range(1, len(table)):
        if "Longitude" in table[j-1]:
            data[0] = table[j]
        elif "Latitude" in table[j-1]:
            data[1] = table[j]
        elif "Speed" in table[j-1]:
            data[2] = table[j]
        elif "Position Received" in table[j-1]:
            data[3] = table[j]
        elif "Course" in table[j-1]:
            data[4] = table[j]
    # print(data)
    if any([k is None for k in data]):
        # print("aborting...")
        return
    for j in range(5):
        data[j] = data[j].replace("<td>", "").replace("</td>", "").replace("°", "")
    data[3] = data[3][data[3].find("title=") + 7 : data[3].rfind("\"")]
    if "<i" in data[2]:
        data[2] = "-1"
    data.append(row[1])
    ports = soup.findAll('h3', class_='text-truncate m-1')
    # print("inital ports:", ports)
    source_port = str(ports[0]).replace('<h3 class="text-truncate m-1">', '')
    source_port = source_port.replace('</h3>', '')
    if "-in-" in source_port:
        source_port = source_port[source_port.find("-in-") + 4 : source_port.rfind("-id-")]
    data.append(source_port)
    dest_port = str(ports[1]).replace('<h3 class="text-truncate m-1">', '')
    dest_port = dest_port.replace('</h3>', '')
    if "-in-" in dest_port:
        dest_port = dest_port[dest_port.find("-in-") + 4 : dest_port.rfind("-id-")]
    data.append(dest_port)
    # print(data)
    # with open('ship_data.csv', 'a') as f:
    #     f.write(f"{','.join(data)}")
    return data
    

# with dataset_path.open() as read_obj:
#     # pass the file object to reader() to get the reader object
#     csv_reader = reader(read_obj)
#     # Iterate over each row in the csv using reader object
#     for row in csv_reader:
        # row variable is a list that represents a row in csv

data = pandas.read_csv(dataset_path)
pool = Pool(16)
results = list(tqdm(pool.imap(get_data, list(data.iterrows())), total=len(data)))
# print(results)

timestamp = datetime.datetime.now().timestamp()

results = [",".join(data) for data in results if data is not None and all(["," not in item for item in data])]
with Path(f"ship_positions/ship_data-{timestamp}.csv").open("w") as data_file:
    data_file.write("lon,lat,speed,timestamp,course,mmsi,source_port,dest_port\n")
    data_file.write("\n".join(results))
