from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import json
app = Flask(__name__)

from model import model

db = model()

@app.route('/')
def index():
   print("Receiving data from DB")
   data = db.read_latest_suspicious_data()["records"]
   print("Recevied ",len(data),"Records")

   markers = []

   for record in data:
      print(record)
      if record["lat_deliverer"]=="---" or record["lon_deliverer"] == "---":
         continue
      markers.append({
         'lat': float(record["lat_deliverer"]),
         'lon': float(record['lon_deliverer']),
         'popup': f"""MMSI {record['source_deliverer']}: {record['deliverer_mmsi']}; MMSI {record['source_receiver']}: {record['receiver_mmsi']};"""
      })
   return render_template('homepage.html',markers=markers )

@app.route('/home')
def home():
   return index()

@app.route('/about')
def about():
   return render_template('about.html' )

@app.route('/community')
def community():
   return render_template('homepage.html')

@app.route('/data')
def data():
   return render_template('data.html' )

@app.route('/contact')
def contact():
   return render_template('homepage.html' )

@app.route('/write_api')
def write_api():
   with open("hist_a.json","r") as data:
      db.write_suspicious_data(json.load(data))
   return render_template('homepage.html' )

@app.route('/read_api')
def read_api():
   return render_template('homepage.html' )


@app.route('/tweet')
def tweet():
   import tweepy
   import config
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

   # update the status
   # api.update_status(status ="Hello Everyone! We will be posing about suspicious alleged S2S transfers happening on sea.")
   client.create_tweet(text="Hello Everyone! We will be posing about suspicious alleged S2S transfers happening on sea.")
   return render_template('homepage.html' )







@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('homepage.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
   app.run()