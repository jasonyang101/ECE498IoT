import urllib
import urllib2

url = "https://courses.engr.illinois.edu/ece498icc/sp2019/lab1_string.php"
data = urllib.urlencode({"netid": "jyang223", "name":"Jason Yang"})
req = urllib2.Request(url,data)
response = urllib2.urlopen(req)
page = response.read()
out = page[0:len(page):498]
print "".join(out[0:400])
