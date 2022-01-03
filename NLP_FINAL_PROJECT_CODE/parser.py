from lxml import html
import requests
from requests.sessions import extract_cookies_to_jar

#new link https://ucsd.libguides.com/congress_twitter/reps
page = requests.get('https://ucsd.libguides.com/congress_twitter/senators')
tree = html.fromstring(page.content)


#This will create a list of peoplo nand info
people = tree.xpath('//a[@href]/text()')
person_info = tree.xpath('//td[@class="ck_border"]/text()')


people = people[16:len(people)-10]

#get links
link = list(tree.iterlinks())
i = 0
j=0
for links in link[40:len(link)-8]:
    print(people[i]+ " "+ person_info[j]+","+person_info[j+1] + " " + links[2] )
    i+=1
    j+=2



