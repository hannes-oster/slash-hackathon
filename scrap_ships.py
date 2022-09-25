from pathlib import Path
from multiprocessing import Pool
from bs4 import BeautifulSoup
import requests
import csv


def get_urls():
    counter = 0
    length = 0
    lengthDiff = 25 #difference between upper and lower bound of length of the searched ships
    # lengthDiff = 25 #difference between upper and lower bound of length of the searched ships
    maxLength = 425
    GT = 0
    GTDiff = 250 
    # GTDiff = 100 
    maxGT = 10000
    #open csv and write
    with open('ships', 'w') as shipTable:
        writer = csv.writer(shipTable)

        #write header
        header = ['name', 'MMSI', 'IMO', 'country']
        writer.writerow(header)
        
        while(length + lengthDiff <= maxLength):
            while(GT <= maxGT):
                for page in range(1,201):
                    side_url = 'https://www.vesselfinder.com/vessels?page='+ str(page) 
                    side_url += '&minLength=' + str(length) + '&maxLength=' +str(length+ lengthDiff)
                    side_url += '&minGT=' + str(GT) + '&maxGT=' + str(GT+GTDiff) + '&type=6'
                    # lines = scrapeShipPage(side_url)
                    # lines = scrapeShipPage(side_url)
                    yield side_url
                    # for line in lines:
                    #     writer.writerow(line)
                    #     counter += 1
                    #     print(counter)
                GT += GTDiff
            GT = 0
            length +=  lengthDiff



def scrapeShipPage(side_url):
    lines = []
    
    html_text = requests.get(side_url, headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}).text

    #get content
    soup = BeautifulSoup(html_text, 'lxml') 
        
    ships = soup.find_all('tr')
    ships = soup.find_all('td', class_ = 'v2')

    
    #extract ship info
    for ship in ships:
        link = ship.find('a', class_ = 'ship-link')['href']
        link = link.replace('/vessels/', '')
        country = ship.find('div', class_ = 'flag-icon-med flag-icon')['title']

        line = '-'
        positions = [pos for pos, char in enumerate(link) if char == line]
        mMSI = link[positions[-1]+1:]
        iMO = link[positions[-3]+1:positions[-2]]
        name = link[:positions[-4]].replace('-', ' ')

        lines.append([name, mMSI, iMO, country])
        
    return lines

from tqdm import tqdm
pool = Pool(16)
urls = list(get_urls())
results = list(tqdm(pool.imap(scrapeShipPage, urls), total=len(urls)))

real_results = []
for lines in results:
    real_results += lines

real_results = [",".join(line) for line in real_results]

with Path("ships.csv").open("w") as data_file:
    data_file.write("name,MMSI,IMO,country\n")
    data_file.write("\n".join(real_results))



