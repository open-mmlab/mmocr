import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder")
parser.add_argument("--output_folder")
args = parser.parse_args()

directory = args.input_folder
output_folder = args.output_folder
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('txt'):
            with open(directory+"/"+filename) as file:
                fileText = file.read()
                polygons = fileText.split('\n')
                numberOfPoints = 7
            allPoints = []
            for polygon in polygons:
                if polygon !="":
                    points = []
                    x1,y1,x2,y2,x3,y3,x4,y4 = [int(x) for x in polygon.split(',')]
                    initialPairs = [x1, y1, x2, y2], [x3, y3, x4, y4]
                    for pair in initialPairs:
                        distanceX = abs(pair[2]-pair[0])
                        stepX = distanceX/(numberOfPoints-1)
                        distanceY = abs(pair[3]-pair[1])
                        stepY = distanceY/(numberOfPoints-1)
                        for i in range(numberOfPoints):
                            points.append(pair[0])
                            points.append(pair[1])
                            if pair[0]<pair[2]:
                                pair[0]+=stepX
                            else:
                                pair[0]-=stepX
                            if pair[1]<pair[3]:
                                pair[1]+=stepY
                            else:
                                pair[1]-=stepY
                        points[-2],points[-1] = pair[2],pair[3]
                    allPoints.append(points)
            text=""
            for points in allPoints:
                line = ','.join([str(round(x)) for x in points])
                text += line + '\n'
            try:
                os.mkdir(output_folder)
            except:
                pass
            with open(output_folder+filename,'w') as file:
                file.write(text)
