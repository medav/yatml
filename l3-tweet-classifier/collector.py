import twitter
import re
import html
import sys
import json

# This is adapted from work by Thomas Moll 
# See https://github.com/QuantumFractal

with open('settings.json') as settings_file:   
    settings = json.load(settings_file)

regex = re.compile(r"(#[A-Za-z]*)|(http(s|):\/\/[A-Za-z./0-9]*)|(@[A-Za-z]*)")

api = twitter.Api(**settings['twitter_keys'])

def collect_tweets(handle, amount=400):
    
    print("Trying to collect {} tweets from @{}...".format(amount ,handle))
    tweets = collect(handle, api, amount)
    dump_tweets_to_file(handle, tweets)

    print("Collected {} tweets from @{}.".format(len(tweets), handle))


def get_tweets(handle, api, last=None, count=200):
    try:
        raw_tweets = api.GetUserTimeline(screen_name=handle,
                                         include_rts=False,
                                         count=count,
                                         max_id=last)
        return raw_tweets
    except err:
        print("Twitter Error! {}".format(err))


def dump_tweets_to_file(handle, tweets):
    with open('tweets/' + handle+'.txt', 'w', encoding='utf-8') as f:
        for tweet in tweets:
            unescaped = html.unescape(regex.sub("", tweet.text))
            unescaped = unescaped.encode('ascii', 'ignore').decode("utf-8")
            f.write(unescaped+'\n')


def collect(handle, api, maximum):
    try:
        tweets = get_tweets(handle, api, count=maximum)
        while len(tweets) < maximum:
            tweets += get_tweets(handle, api, last=get_last_id(tweets))
        return tweets
    except err:
        print("I tried my best!")
        return tweets


def get_last_id(tweets):
    if len(tweets) > 1:
        return tweets[-1].id

for handle in settings['handles']:
    collect_tweets(handle)