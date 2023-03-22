import sys
from sys import stdin
import json
from json import JSONEncoder
import face_recognition
import numpy as np
from collections import namedtuple
import os
import datetime

class Result:
    profile = None
    location = None
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False) #, indent=2  ... if we want it beautified
    def __init__(self, profile, location):
        self.profile = profile
        self.location = location
    def startCall():
        pass
    def endCall():
        pass

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

for line in stdin:
    filepath=""
    resp=""
    if(line == "exit\n"):
        resp= "exiting..."
        break

    filepath = filepath + line.replace("\n","")
    if (os.path.isfile(filepath)):
        
            picture =  face_recognition.load_image_file(filepath)
            encode = np.array(face_recognition.face_encodings(picture))
            locations = np.array(face_recognition.face_locations(picture))
            faceLocations = []
            for x in locations:
                tup= (x[3], x[0], x[1], x[2])
                faceLocations.append(tup)
            locs = np.array(faceLocations)
            res=Result(encode.tolist(), locs.tolist())
            resp= res.toJSON()+"\n"
        
    elif (os.path.isdir(filepath)):
        try:
            profiles_images = []        
            for file in os.listdir(filepath):
                fullfile = os.path.join(filepath, file)
                if (os.path.isfile(fullfile)):
                    profiles_images.append(face_recognition.load_image_file(fullfile))
            
            encodes = []
            for profile in profiles_images:
                encodes.extend(np.array(face_recognition.face_encodings(profile)).tolist())
            res=Result(encodes)
            resp = res.toJSON()+"\n"
        except:
            resp = "path must contain only pictures"
    else:
        resp = 'Not a file or Folder'

    print(resp, flush=True)
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'debug'):
            with open("/tmp/profile.log", "a") as logFile:
                logFile.write(str(datetime.datetime.now()) + ": " + line+ ": "+resp)
