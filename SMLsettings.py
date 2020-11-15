import json


def getSettings():
    infile = open('settings.json', 'r')
    s = json.load(infile)
    return s


def setSettings(settings):
    with open('settings.json', 'w') as outfile:
        json.dump(settings, outfile, indent=2)
