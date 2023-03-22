import sys
from sys import stdin
import json
from json import JSONEncoder
import face_recognition
import numpy as np
from collections import namedtuple
import datetime


class Compare:
    def __init__(self, isMatch, distance):
        self.isMatch = isMatch
        self.distance = distance


class Result:
    matches = None
    resemblances = None

    def toJSON(self):
        # , indent=2  ... if we want it beautified
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    def __init__(self, matches, resemblances):
        self.matches = matches
        self.resemblances = resemblances

    def startCall():
        pass

    def endCall():
        pass


def logit(log):
    print(log.replace("response:", "").replace("request:", ""), flush=True)
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'debug'):
            with open("/tmp/compare.log", "a") as logFile:
                logFile.write(str(datetime.datetime.now()) + ": " + log)


def logOnly(log):
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'debug'):
            with open("/tmp/compare.log", "a") as logFile:
                logFile.write(str(datetime.datetime.now()) + ": " + log)


def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)


def compare(profile, photocrop):
    resultCompare = face_recognition.compare_faces(
        profile, np.asarray(photocrop))
    resultDistance = face_recognition.face_distance(
        profile, np.asarray(photocrop))
    resInt = []
    for b in resultCompare:
        if (b):
            resInt.append(1)
        else:
            resInt.append(0)
    return Compare(resInt, resultDistance)


def runCompare(jsondata):
    try:
        x = json2obj(jsondata)
    except:
        logit("not valid json")
        return ""
    res = Result([], [])
    for i in range(len(x.picture)):
        a = compare(x.profile, x.picture[i])
        res.matches.append(a.isMatch)
        res.resemblances.append(a.distance.tolist())
    return res.toJSON()


jsonData = ""
for line in stdin:
    if(line == "exit\n"):
        logit("exiting...")
        break

    jsonData = jsonData + line.replace("\n", "")
    # logit(jsonData)
    if (line == "\n"):
        result = runCompare(jsonData)
        logOnly("request:"+jsonData.replace(" ", "") + "\n")
        logit("response:"+result+"\n")
        jsonData = ""
