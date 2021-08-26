import urllib.request as urlReq
from bs4 import BeautifulSoup
import requests

def download_file(url):
    local_filename = "Dataset/" + url.split('/')[-1]
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=None): 
                f.write(chunk)
    return local_filename

x = urlReq.urlopen("https://www.vgmusic.com/music/console/nintendo/n64/")
content = x.read()
soup = BeautifulSoup(content, 'html.parser')
Asoup = soup.find_all('a')
iterador = 0

for ancor in Asoup:
    try:
        link = ancor['href']
        if ".mid" in link:
            url = "https://www.vgmusic.com/music/console/nintendo/n64/" + link
            download_file(url)
            iterador+=1
            print(iterador)

    except:
        pass