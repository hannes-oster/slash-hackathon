import time
import random
import json
from pathlib import Path
# importing the module
import tweepy
import config
  
# personal details
# consumer_key ="DExqSRI3BiiW1V63kZZuinDRx"
# consumer_secret ="ES2MClwFY666zxCXx1DdjVK55MgFrs7FlsLXGk4NwkXTrjGtma"
# access_token ="1573954655268339712-czOvUsT7vkYIfTZiin0yyuZ4WEvzar"
# access_token_secret ="BhHCeDTdT5g4Fbb6uMTdTXp5iMIRLM5bCs1ZLlNqv1PtJ"
client = tweepy.Client(
    consumer_key = config.settings["consumer_key"],
    consumer_secret= config.settings["consumer_secret"],
    access_token= config.settings["access_token"],
    access_token_secret= config.settings["access_token_secret"])

# authentication of consumer key and secret
# auth = tweepy.OAuthHandler(config.settings["consumer_key"], config.settings["consumer_secret"])
  
# # authentication of access token and secret
# auth.set_access_token(config.settings["access_token"], config.settings["access_token_secret"])
# api = tweepy.API(auth)

# # update the status
# api.update_status(status ="Hello Everyone! We will be posing about suspicious alleged S2S transfers happening on sea.")

data_path = Path("incidents/suspicious_incidents_2022-09-25 08:44.json")
with data_path.open() as data_file:
    data = json.load(data_file)

exclamations = ["Oops.", "Yaowser.", "Ouuff!", "Oh my..."]

while True:
    entry = data[random.randint(0, len(data) - 1)]
    rand_exclamation = exclamations[random.randint(0, len(exclamations) - 1)]
    tweet_text = f"{rand_exclamation} Suspicious activity at \n\nlat: {entry['lat_deliverer']}\nlon: {entry['lon_deliverer']}\n\nA ship from {entry['source_deliverer']} had long open sea contact with a ship from {entry['source_receiver']}. This might might indicate something."
    print("sending text:", tweet_text)
    client.create_tweet(
        text=tweet_text
        )
    sleep_time = random.randint(600, 1200)
    print("sleeping for", sleep_time, "seconds...")
    time.sleep(sleep_time)
    