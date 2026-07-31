import requests

doi = "10.1371/journal.pone.0121283"
url = f"https://api.unpaywall.org/v2/{doi}?email=test@example.com"
response = requests.get(url)

print("STATUS CODE:", response.status_code)
print("RAW RESPONSE:", response.text[:500])
